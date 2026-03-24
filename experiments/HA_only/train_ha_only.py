"""
HA-only 训练脚本：只使用 seq_id_a / seq_id_c 两路 HA embedding，不加载 b/d。

仅修改本文件，不改动 experiment_tools / architectures 等公共模块。
"""
import sys
from pathlib import Path
import json
import pickle
from datetime import datetime

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Make local modules importable
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src" / "fluprofiler"))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

from experiment_tools import (  # noqa: E402
    find_repo_root,
    make_run_dirs,
    setup_optimizer,
    setup_tensorboard_and_logging,
    log_metrics_to_tensorboard,
    log_epoch_to_file,
    GpuEmbeddingCache,
    convert_Pass2tensor,
    generate_matrix_on_device,
)
from data.loaders import load_embedding  # noqa: E402
from models.architectures import fluProfiler_HA, fluProfiler_Config  # noqa: E402
from evaluation.metrics import EarlyStopping, print_exams  # noqa: E402


# =============================================================================
# 用户参数（请在此修改；上方 _REPO_ROOT 用于默认路径与 sys.path）
# =============================================================================
# CSV：若 SEASON_PATH 为相对路径，则读取 DATA_ROOT / SEASON_PATH / train.csv；
# 若 SEASON_PATH 为绝对路径，则直接在该目录下找 train.csv（不再拼 DATA_ROOT）。
DATA_ROOT = str(_REPO_ROOT / "data" / "HA_only") + "/"
SEASON_PATH = "split/"

# 预计算 embedding 目录（matrix_*.pt，可改为绝对路径）
EMBEDDING_ROOT = str(_REPO_ROOT / "data" / "reverse_test" / "embedding")

# 训练
EPOCHS = 250
PATIENCE = 10
BATCH_SIZE = 64
GPU_CACHE_GB = 24
LR = 0.00008
DEVICE = "cuda:1"  # 无 GPU 可改为 "cpu"
USE_LR_SCHEDULE = True

# 数据
SAMPLE_LIMIT = None  # 调试时改为整数，例如 500
USE_ARTIFICIAL_DATA = False  # True 时需存在 artificial_data.csv
ADD_SPECIAL_TOKEN = True

# 训练集聚合（在 train/valid 划分、合并 artificial 之后，sample_limit 之前）
# 若四列完全相同则合并为一行：label = 组内均值，其余列取该组第一行；仅一行则不变。
DEDUPE_TRAIN = True
TRAIN_GROUP_COLS_FOR_LABEL_MEAN = ["seq_id_a", "seq_id_c", "serumPassCat", "virusPassCat"]

# 运行目录命名（runs/<exp_id>/...）
RUN_TAG = "HA_only_cached"
EXP_ID = None  # None 则自动为 HA_only/<season>/HA；也可设为例如 "HA_only/my_exp/HA"

MODEL_CLASS = fluProfiler_HA


def _ensure_trailing_slash(path: str) -> str:
    """仅用于「已是绝对路径」的目录字符串规范化。"""
    path = str(Path(path).expanduser().resolve())
    return path if path.endswith("/") else path + "/"


def _normalize_season_path(season: str) -> str:
    """
    SEASON_PATH 为相对路径（如 split/）时不要 Path.resolve()，否则会相对当前工作目录解析，路径会错。
    绝对路径则规范为带尾部 / 的字符串。
    """
    s = str(season).strip()
    p = Path(s).expanduser()
    if p.is_absolute():
        return _ensure_trailing_slash(s)
    s = s.replace("\\", "/").strip("/")
    return (s + "/") if s else ""


def _split_csv_dir(data_root: str, season: str) -> Path:
    """
    返回存放 train.csv / test.csv 的目录。
    - season 为绝对路径：只用该目录；
    - season 为相对路径：data_root / season。
    """
    season_norm = _normalize_season_path(season).rstrip("/")
    sp = Path(season_norm).expanduser()
    if sp.is_absolute():
        return sp.resolve()
    dr = Path(str(data_root).rstrip("/")).expanduser().resolve()
    if not season_norm:
        return dr
    return (dr / season_norm).resolve()


def _aggregate_train_mean_label_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    label_col: str = "label",
    name: str = "train",
) -> pd.DataFrame:
    """group_cols 完全相同的行合并：label 取均值，其余列取组内第一行。"""
    n0 = len(df)
    if n0 == 0:
        return df
    missing = set(group_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"[dedupe] CSV 缺少列 {sorted(missing)}，当前列: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(f"[dedupe] CSV 缺少标签列 {label_col!r}")

    other_cols = [c for c in df.columns if c not in group_cols and c != label_col]
    agg: dict = {label_col: "mean"}
    for c in other_cols:
        agg[c] = "first"

    out = df.groupby(list(group_cols), as_index=False, sort=False).agg(agg)
    # 列顺序与原始 CSV 一致（仅保留仍存在的列）
    col_order = [c for c in df.columns if c in out.columns]
    out = out[col_order]
    n1 = len(out)
    if n0 != n1:
        print(
            f"[dedupe] {name}: {n0} -> {n1} 行（{n0 - n1} 条按组合并，{label_col} 为组内均值）"
        )
    return out.reset_index(drop=True)


class HAOnlyFluProfilerDataset(Dataset):
    """只暴露 matrix_a / matrix_c 的 key，与 fluProfiler_HA.forward 一致。"""

    def __init__(self, dataframe: pd.DataFrame, add_special_token: bool = True):
        self.emb_file_name_a = ("matrix_" + dataframe["seq_id_a"]).tolist()
        self.emb_file_name_c = ("matrix_" + dataframe["seq_id_c"]).tolist()
        if add_special_token:
            self.strainPassCats = convert_Pass2tensor(
                (
                    "<cls>"
                    + dataframe["serumPassCat"]
                    + "<eos>"
                    + dataframe["virusPassCat"]
                    + "<eos>"
                ).tolist()
            )
        else:
            self.strainPassCats = convert_Pass2tensor(
                (dataframe["serumPassCat"] + dataframe["virusPassCat"]).tolist()
            )
        self.labels = torch.tensor(dataframe["label"].tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.emb_file_name_a[idx],
            self.emb_file_name_c[idx],
            self.strainPassCats[idx],
            self.labels[idx],
        )


def load_ha_only_dataloaders_and_emb_keys(
    data_path: str,
    season_path: str,
    batch_size: int,
    sample_limit=None,
    use_artificial: bool = False,
    add_special_token: bool = True,
    dedupe_train: bool = False,
    train_group_cols_for_label_mean: list[str] | None = None,
):
    """
    与 experiment_tools.load_data_and_dataloaders 相同的 train/valid 划分逻辑，
    但 Dataset 与后续 embedding 列表仅含 seq_id_a / seq_id_c。

    dedupe_train：对 train_data_final 按 train_group_cols_for_label_mean 分组，
    相同组内 label 取均值、其余列取首行。
    """
    csv_dir = _split_csv_dir(data_path, season_path)
    train_data = pd.read_csv(csv_dir / "train.csv")
    test_data = pd.read_csv(csv_dir / "test.csv")
    train_data, valid_data = train_test_split(train_data, test_size=1 / 9, random_state=42)

    if use_artificial:
        try:
            artificial_data = pd.read_csv(csv_dir / "artificial_data.csv")
            train_data_final = pd.concat([train_data, artificial_data])
        except FileNotFoundError:
            train_data_final = train_data
    else:
        train_data_final = train_data

    if dedupe_train:
        gcols = train_group_cols_for_label_mean
        if not gcols:
            raise ValueError(
                "dedupe_train=True 时需要非空的 train_group_cols_for_label_mean"
            )
        train_data_final = _aggregate_train_mean_label_by_group(
            train_data_final,
            gcols,
            label_col="label",
            name="train（含 artificial 合并后）",
        )

    if sample_limit:
        train_data_final = train_data_final.iloc[:sample_limit]
        valid_data = valid_data.iloc[:sample_limit]
        test_data = test_data.iloc[:sample_limit]

    train_dataset = HAOnlyFluProfilerDataset(train_data_final, add_special_token=add_special_token)
    valid_dataset = HAOnlyFluProfilerDataset(valid_data, add_special_token=add_special_token)
    test_dataset = HAOnlyFluProfilerDataset(test_data, add_special_token=add_special_token)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
    sequence_names = pd.concat(
        [embedding_df["seq_id_a"], embedding_df["seq_id_c"]]
    ).unique().tolist()
    embedding_files = ["matrix_" + item + ".pt" for item in sequence_names]

    return {
        "train_dataloader": train_dataloader,
        "valid_dataloader": valid_dataloader,
        "test_dataloader": test_dataloader,
        "embedding_files": embedding_files,
    }


def train_step_ha_cached(model, batch, device, cache: GpuEmbeddingCache):
    emb_file_name_a, emb_file_name_c, strainPassCats, labels = batch
    matrices_a_list = [cache.get(key) for key in emb_file_name_a]
    matrices_c_list = [cache.get(key) for key in emb_file_name_c]
    matrixs_a, masks_a = generate_matrix_on_device(matrices_a_list, device=device)
    matrixs_c, masks_c = generate_matrix_on_device(matrices_c_list, device=device)
    strainPassCats = strainPassCats.to(device)
    labels = labels.to(device)
    loss, logits, output = model(
        matrices_a=matrixs_a,
        matrices_c=matrixs_c,
        matrix_attention_masks_a=masks_a,
        matrix_attention_masks_c=masks_c,
        strainPassCats=strainPassCats,
        labels=labels,
    )
    return loss


def evaluate_step_ha_cached(
    model, dataloader, device, cache: GpuEmbeddingCache, return_predictions=False
):
    model.eval()
    prediction_ls = []
    reference_ls = []
    loss_ls = []

    with torch.no_grad():
        for batch in dataloader:
            emb_file_name_a, emb_file_name_c, strainPassCats, labels = batch
            matrices_a_list = [cache.get(key) for key in emb_file_name_a]
            matrices_c_list = [cache.get(key) for key in emb_file_name_c]
            matrixs_a, masks_a = generate_matrix_on_device(matrices_a_list, device=device)
            matrixs_c, masks_c = generate_matrix_on_device(matrices_c_list, device=device)
            strainPassCats = strainPassCats.to(device)
            labels = labels.to(device)
            loss, logits, output = model(
                matrices_a=matrixs_a,
                matrices_c=matrixs_c,
                matrix_attention_masks_a=masks_a,
                matrix_attention_masks_c=masks_c,
                strainPassCats=strainPassCats,
                labels=labels,
            )
            loss_ls.append(loss.item())
            prediction_ls.extend(output.view(-1).tolist())
            reference_ls.extend(labels.tolist())

    mae, mse, pearson, spearman, r2 = print_exams(reference_ls, prediction_ls)
    pearson_value = float(getattr(pearson, "statistic", pearson[0]))
    spearman_value = float(getattr(spearman, "statistic", spearman[0]))
    metrics = {
        "loss": float(np.mean(loss_ls)) if loss_ls else 0.0,
        "mae": float(mae),
        "mse": float(mse),
        "pearson": pearson_value,
        "spearman": spearman_value,
        "r2": float(r2),
    }
    if return_predictions:
        metrics["predictions"] = prediction_ls
        metrics["references"] = reference_ls
    return metrics


# ===== 路径与运行 ID（由上方「用户参数」推导）=====
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = find_repo_root(_SCRIPT_PATH)
root_path = str(_REPO_ROOT) + "/"

data_path = _ensure_trailing_slash(DATA_ROOT)
season_path = _normalize_season_path(SEASON_PATH)
embedding_root = str(Path(EMBEDDING_ROOT).expanduser().resolve())

epochs = EPOCHS
patience = PATIENCE
sample_limit = SAMPLE_LIMIT
use_artificial = USE_ARTIFICIAL_DATA
use_lr_schedule = USE_LR_SCHEDULE
add_special_token_cfg = ADD_SPECIAL_TOKEN
dedupe_train_cfg = DEDUPE_TRAIN
train_group_cols_cfg = TRAIN_GROUP_COLS_FOR_LABEL_MEAN
batch_size = BATCH_SIZE
gpu_cache_GB = GPU_CACHE_GB
lr = LR
device = torch.device(DEVICE)
tag = RUN_TAG

# 用于 runs 子目录名：取 CSV 目录最后一段，避免把绝对路径整段写进 exp_id
_csv_dir_for_slug = _split_csv_dir(data_path, season_path)
_season = _csv_dir_for_slug.name or "split"
# 若目录名像 test_2024SH，与旧逻辑一致取赛季后缀
if "test_" in _season:
    _season = _season.split("test_")[-1]
exp_id = EXP_ID if EXP_ID else f"HA_only/{_season}/HA"
run_paths = make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)


# ===== logging =====
logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=tag)
writer = logging_info["writer"]
log_path = logging_info["log_path"]
model_save_path = str(run_paths["checkpoints"]) + "/"


# ===== data (HA a/c only) =====
_bundle = load_ha_only_dataloaders_and_emb_keys(
    data_path=data_path,
    season_path=season_path,
    batch_size=batch_size,
    sample_limit=sample_limit,
    use_artificial=use_artificial,
    add_special_token=add_special_token_cfg,
    dedupe_train=dedupe_train_cfg,
    train_group_cols_for_label_mean=train_group_cols_cfg,
)
train_dataloader = _bundle["train_dataloader"]
valid_dataloader = _bundle["valid_dataloader"]
test_dataloader = _bundle["test_dataloader"]
emb_dict = load_embedding(embedding_root, files=_bundle["embedding_files"])

max_cache_bytes = gpu_cache_GB * 1024**3
gpu_cache = GpuEmbeddingCache(cpu_store=emb_dict, device=device, max_bytes=max_cache_bytes)


# ===== model =====
with open(root_path + "configs/config_dict.json", "r") as f:
    config_dict = json.load(f)
with open(root_path + "configs/args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)

fluProfiler_args.output_mode = "regression"

fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = MODEL_CLASS(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)
# fluProfiler_HA.forward 可能引用 self.loss_reduction，若未在 __init__ 中设置则补默认
if not hasattr(model, "loss_reduction"):
    model.loss_reduction = getattr(fluProfiler_config, "loss_reduction", None) or "meanmean"

run_config = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_path": data_path,
    "season_path": season_path,
    "embedding_root": embedding_root,
    "ha_only_streams": ["seq_id_a", "seq_id_c"],
    "epochs": epochs,
    "patience": patience,
    "batch_size": batch_size,
    "sample_limit": sample_limit,
    "dedupe_train": dedupe_train_cfg,
    "train_group_cols_for_label_mean": train_group_cols_cfg,
    "use_lr_schedule": use_lr_schedule,
    "lr": lr,
    "model_class": MODEL_CLASS.__name__,
    "tag": tag,
    "device": str(device),
}
with open(log_path, "a", encoding="utf-8") as f:
    f.write("===== RUN CONFIG START =====\n")
    json.dump(run_config, f, indent=2, ensure_ascii=False)
    f.write("\n===== RUN CONFIG END =====\n\n")

model_structure_path = run_paths["run_root"] / "model_structure.txt"
total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
model_structure_path.write_text(
    "\n".join(
        [
            f"model_class: {model.__class__.__name__}",
            f"device: {device}",
            f"total_params: {total_params}",
            f"trainable_params: {trainable_params}",
            "",
            str(model),
        ]
    ),
    encoding="utf-8",
)
print(f"Model structure saved to: {model_structure_path}")


# ===== optimizer =====
optimizer = setup_optimizer(model, fluProfiler_args, lr=lr)
scheduler = None
if use_lr_schedule:
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print(f"[lr_schedule] enabled, T_max={epochs}, eta_min=1e-6")

num_training_steps = len(train_dataloader) * epochs
progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=patience, save_dir=model_save_path)

for epoch in range(epochs):
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        loss = train_step_ha_cached(model, batch, device, gpu_cache)
        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    train_loss = np.mean(loss_ls)
    print("train loss:", train_loss)

    cache_gb = gpu_cache.current_bytes / (1024**3)
    allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
    print(
        f"[cache] hits={gpu_cache.hits}, misses={gpu_cache.misses}, "
        f"copied_GB={gpu_cache.copied_bytes / (1024 ** 3):.2f}, "
        f"current_GB={cache_gb:.2f}"
    )
    print(
        f"[memory] allocated_GB={allocated_gb:.2f}, "
        f"reserved_GB={reserved_gb:.2f}, "
        f"cache_ratio={cache_gb/allocated_gb:.3f} (cache/total)"
    )

    valid_metrics = evaluate_step_ha_cached(model, valid_dataloader, device, gpu_cache)
    test_metrics = evaluate_step_ha_cached(model, test_dataloader, device, gpu_cache)

    log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

    if scheduler is not None:
        scheduler.step()
        print(f"[lr_schedule] current_lr={optimizer.param_groups[0]['lr']:.8f}")

    early_stopping(valid_metrics["mse"], model)
    if early_stopping.early_stop:
        print("Early stopping")
        writer.close()
        break

    log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics)
else:
    writer.close()
