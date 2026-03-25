"""
HA-only v2 training entrypoint.

This script keeps legacy pipeline behavior (data/cache/logging) and swaps model
implementation via `model_impl` in JSON config.
"""

import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src" / "fluprofiler"))
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

from experiment_tools import (  # noqa: E402
    GpuEmbeddingCache,
    convert_Pass2tensor,
    find_repo_root,
    generate_matrix_on_device,
    log_epoch_to_file,
    log_metrics_to_tensorboard,
    make_run_dirs,
    setup_optimizer,
    setup_tensorboard_and_logging,
)
from data.loaders import load_embedding  # noqa: E402
from evaluation.metrics import EarlyStopping, print_exams  # noqa: E402
from models.architectures import fluProfiler_Config, fluProfiler_HA  # noqa: E402
from fluprofiler.models_v2 import BatchInput, fluProfiler_HA_only_v2  # noqa: E402


def _ensure_trailing_slash(path: str) -> str:
    path = str(Path(path).expanduser().resolve())
    return path if path.endswith("/") else path + "/"


def _normalize_season_path(season: str) -> str:
    s = str(season).strip()
    p = Path(s).expanduser()
    if p.is_absolute():
        return _ensure_trailing_slash(s)
    s = s.replace("\\", "/").strip("/")
    return (s + "/") if s else ""


def _split_csv_dir(data_root: str, season: str) -> Path:
    season_norm = _normalize_season_path(season).rstrip("/")
    sp = Path(season_norm).expanduser()
    if sp.is_absolute():
        return sp.resolve()
    dr = Path(str(data_root).rstrip("/")).expanduser().resolve()
    if not season_norm:
        return dr
    return (dr / season_norm).resolve()


class HAOnlyDataset(Dataset):
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
        self.labels = torch.tensor(dataframe["label"].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.emb_file_name_a[idx],
            self.emb_file_name_c[idx],
            self.strainPassCats[idx],
            self.labels[idx],
        )


def load_dataloaders_and_embedding_keys(
    data_path: str,
    season_path: str,
    batch_size: int,
    sample_limit=None,
    use_artificial: bool = False,
    add_special_token: bool = True,
):
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

    if sample_limit:
        train_data_final = train_data_final.iloc[:sample_limit]
        valid_data = valid_data.iloc[:sample_limit]
        test_data = test_data.iloc[:sample_limit]

    train_dataset = HAOnlyDataset(train_data_final, add_special_token=add_special_token)
    valid_dataset = HAOnlyDataset(valid_data, add_special_token=add_special_token)
    test_dataset = HAOnlyDataset(test_data, add_special_token=add_special_token)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
    sequence_names = pd.concat([embedding_df["seq_id_a"], embedding_df["seq_id_c"]]).unique().tolist()
    embedding_files = ["matrix_" + item + ".pt" for item in sequence_names]
    return train_dataloader, valid_dataloader, test_dataloader, embedding_files


def _batch_to_tensors(batch, device, cache):
    emb_file_name_a, emb_file_name_c, strainPassCats, labels = batch
    matrices_a_list = [cache.get(key) for key in emb_file_name_a]
    matrices_c_list = [cache.get(key) for key in emb_file_name_c]
    matrixs_a, masks_a = generate_matrix_on_device(matrices_a_list, device=device)
    matrixs_c, masks_c = generate_matrix_on_device(matrices_c_list, device=device)
    return matrixs_a, masks_a, matrixs_c, masks_c, strainPassCats.to(device), labels.to(device)


def train_step(model, batch, device, cache, model_impl: str):
    matrixs_a, masks_a, matrixs_c, masks_c, strain_pass, labels = _batch_to_tensors(batch, device, cache)
    if model_impl == "v2":
        out = model(
            BatchInput(
                matrices={"serum_HA": matrixs_a, "virus_HA": matrixs_c},
                matrix_masks={"serum_HA": masks_a, "virus_HA": masks_c},
                passage_tokens=strain_pass,
                labels=labels.view(-1),
            )
        )
        return out.loss
    loss, _, _ = model(
        matrices_a=matrixs_a,
        matrices_c=matrixs_c,
        matrix_attention_masks_a=masks_a,
        matrix_attention_masks_c=masks_c,
        strainPassCats=strain_pass,
        labels=labels,
    )
    return loss


def evaluate_step(model, dataloader, device, cache, model_impl: str):
    model.eval()
    prediction_ls, reference_ls, loss_ls = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            matrixs_a, masks_a, matrixs_c, masks_c, strain_pass, labels = _batch_to_tensors(batch, device, cache)
            if model_impl == "v2":
                out = model(
                    BatchInput(
                        matrices={"serum_HA": matrixs_a, "virus_HA": matrixs_c},
                        matrix_masks={"serum_HA": masks_a, "virus_HA": masks_c},
                        passage_tokens=strain_pass,
                        labels=labels.view(-1),
                    )
                )
                loss = out.loss
                preds = out.pred
            else:
                loss, _, preds = model(
                    matrices_a=matrixs_a,
                    matrices_c=matrixs_c,
                    matrix_attention_masks_a=masks_a,
                    matrix_attention_masks_c=masks_c,
                    strainPassCats=strain_pass,
                    labels=labels,
                )
            loss_ls.append(loss.item())
            prediction_ls.extend(preds.view(-1).tolist())
            reference_ls.extend(labels.view(-1).tolist())

    mae, mse, pearson, spearman, r2 = print_exams(reference_ls, prediction_ls)
    pearson_value = float(getattr(pearson, "statistic", pearson[0]))
    spearman_value = float(getattr(spearman, "statistic", spearman[0]))
    return {
        "loss": float(np.mean(loss_ls)) if loss_ls else 0.0,
        "mae": float(mae),
        "mse": float(mse),
        "pearson": pearson_value,
        "spearman": spearman_value,
        "r2": float(r2),
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="HA-only v2 trainer (supports CLI overrides)")
    parser.add_argument("cfg_path", type=str, nargs="?", default=str(_THIS_FILE.parent / "config_v2_ha_only.json"))

    # Common overrides (all optional; if provided, they overwrite JSON config)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--gpu-cache-gb", type=float, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-lr-schedule", action="store_true", default=False)
    parser.add_argument("--use-lr-schedule", action="store_true", default=False)
    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Apply CLI overrides
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.learning_rate is not None:
        cfg["train"]["learning_rate"] = float(args.learning_rate)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.patience is not None:
        cfg["train"]["patience"] = int(args.patience)
    if args.sample_limit is not None:
        cfg["data"]["sample_limit"] = None if args.sample_limit < 0 else int(args.sample_limit)
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if args.gpu_cache_gb is not None:
        cfg["runtime"]["gpu_cache_gb"] = float(args.gpu_cache_gb)
    if args.no_lr_schedule:
        cfg["train"]["use_lr_schedule"] = False
    if args.use_lr_schedule:
        cfg["train"]["use_lr_schedule"] = True

    set_seed(int(cfg["train"].get("seed", 42)))

    data_path = _ensure_trailing_slash(cfg["data"]["data_root"])
    season_path = _normalize_season_path(cfg["data"]["season_path"])
    embedding_root = str(Path(cfg["data"]["embedding_root"]).expanduser().resolve())
    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["learning_rate"])
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    sample_limit = cfg["data"].get("sample_limit")
    use_artificial = bool(cfg["data"].get("use_artificial_data", False))
    add_special_token = bool(cfg["data"].get("add_special_token", True))
    use_lr_schedule = bool(cfg["train"].get("use_lr_schedule", True))
    model_impl = cfg["experiment"].get("model_impl", "v2").lower()

    device = torch.device(cfg["runtime"]["device"])
    gpu_cache_gb = float(cfg["runtime"]["gpu_cache_gb"])

    run_tag = cfg["experiment"].get("run_tag", "HA_only_v2")
    exp_id = cfg["experiment"].get("exp_id")
    if not exp_id:
        csv_dir_for_slug = _split_csv_dir(data_path, season_path)
        _season = csv_dir_for_slug.name or "split"
        exp_id = f"HA_only/{_season}/HA_{model_impl}"

    repo_root = find_repo_root(_THIS_FILE.resolve())
    run_paths = make_run_dirs(repo_root, exp_id=exp_id, tag=run_tag)
    logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=run_tag)
    writer = logging_info["writer"]
    log_path = logging_info["log_path"]

    train_dl, valid_dl, test_dl, embedding_files = load_dataloaders_and_embedding_keys(
        data_path=data_path,
        season_path=season_path,
        batch_size=batch_size,
        sample_limit=sample_limit,
        use_artificial=use_artificial,
        add_special_token=add_special_token,
    )
    emb_dict = load_embedding(embedding_root, files=embedding_files)
    gpu_cache = GpuEmbeddingCache(
        cpu_store=emb_dict,
        device=device,
        max_bytes=int(gpu_cache_gb * 1024**3),
    )

    config_dict = json.loads((repo_root / "configs" / "config_dict.json").read_text(encoding="utf-8"))
    flu_config = fluProfiler_Config.from_dict(config_dict)
    with open(repo_root / "configs" / "args.pkl", "rb") as f:
        flu_args = pickle.load(f)
    flu_args.output_mode = "regression"

    if model_impl == "v2":
        model = fluProfiler_HA_only_v2(config=flu_config, args=flu_args)
    else:
        model = fluProfiler_HA(config=flu_config, args=flu_args)
        if not hasattr(model, "loss_reduction"):
            model.loss_reduction = getattr(flu_config, "loss_reduction", None) or "meanmean"
    model.to(device)

    optimizer = setup_optimizer(model, flu_args, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) if use_lr_schedule else None
    progress_bar = tqdm(range(len(train_dl) * epochs))
    early_stopping = EarlyStopping(patience=patience, save_dir=str(run_paths["checkpoints"]) + "/")

    run_config = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cfg_path": str(cfg_path),
        "model_impl": model_impl,
        "data_path": data_path,
        "season_path": season_path,
        "embedding_root": embedding_root,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "device": str(device),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("===== RUN CONFIG START =====\n")
        json.dump(run_config, f, indent=2, ensure_ascii=False)
        f.write("\n===== RUN CONFIG END =====\n\n")

    for epoch in range(epochs):
        model.train()
        loss_ls = []
        for batch in train_dl:
            loss = train_step(model, batch, device, gpu_cache, model_impl=model_impl)
            loss.backward()
            loss_ls.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        train_loss = float(np.mean(loss_ls))

        valid_metrics = evaluate_step(model, valid_dl, device, gpu_cache, model_impl=model_impl)
        test_metrics = evaluate_step(model, test_dl, device, gpu_cache, model_impl=model_impl)
        log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

        if scheduler is not None:
            scheduler.step()

        early_stopping(valid_metrics["mse"], model)
        log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    writer.close()


if __name__ == "__main__":
    main()
