import sys
sys.path.append('../../../')

from fluProfiler_models import fluProfiler_v0_1, fluProfiler_Config
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from utilities import load_embedding, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime
import pickle
import os
import time
import subprocess
import hashlib
from pathlib import Path
from utilities import print_exams


# =========================
# NEW: 最小 RunLogger（单文件版）
# =========================
def _sha1_json(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

def _git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        return {"commit": commit, "state": "dirty" if dirty else "clean"}
    except Exception:
        return {"commit": "nogit", "state": "unknown"}

def _json_default(o):
    # 兜底：把 torch.device / Path / argparse Namespace 等转成 str
    try:
        import torch
        if isinstance(o, torch.device):
            return str(o)
    except Exception:
        pass
    try:
        from pathlib import Path
        if isinstance(o, Path):
            return str(o)
    except Exception:
        pass
    # 其他任何不能序列化的对象，统一转字符串（最小侵入策略）
    return str(o)

def _dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self._tb = None

    @classmethod
    def create(cls, run_root, name, tag, config_dict):
        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        gs = _git_state()
        base = f"{ts}__{name}"
        if tag:
            base += f"__{tag}"
        base += f"__{gs['commit'][:8]}"

        run_dir = run_root / base
        i = 1
        while run_dir.exists():
            i += 1
            run_dir = run_root / f"{base}__{i}"

        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        (run_dir / "tb").mkdir(exist_ok=True)

        _dump_json(run_dir / "config_resolved.json", config_dict)
        _dump_json(run_dir / "git.json", gs)

        logger = cls(run_dir)
        logger.write_text("command.txt", " ".join(sys.argv))
        return logger

    def write_text(self, filename, text):
        (self.run_dir / filename).write_text(text, encoding="utf-8")

    def save_json(self, filename, obj):
        _dump_json(self.run_dir / filename, obj)

    def log(self, record: dict, epoch: int = None):
        rec = dict(record)
        rec["time"] = time.time()
        if epoch is not None:
            rec["epoch"] = epoch
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def tb(self):
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
        return self._tb

    def add_scalar(self, tag, value, step):
        self.tb().add_scalar(tag, float(value), int(step))

    def flush_tb(self):
        if self._tb is not None:
            self._tb.flush()

    def close_tb(self):
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
            self._tb = None


# =========================
# 原脚本：Dataset / 工具函数
# =========================
class fluProfiler_Dataset(Dataset):
    def __init__(self, DataFrame):
        self.emb_file_name_a = ('matrix_' + DataFrame['seq_id_a']).tolist()
        self.emb_file_name_b = ('matrix_' + DataFrame['seq_id_b']).tolist()
        self.emb_file_name_c = ('matrix_' + DataFrame['seq_id_c']).tolist()
        self.emb_file_name_d = ('matrix_' + DataFrame['seq_id_d']).tolist()

        self.strainPassCats = convert_Pass2tensor(('<cls>' + DataFrame['serumPassCat'] + '<eos>' + DataFrame['virusPassCat'] + '<eos>').tolist())
        self.labels = torch.tensor(DataFrame['label'].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.emb_file_name_a[idx], self.emb_file_name_b[idx], self.emb_file_name_c[idx], self.emb_file_name_d[idx], \
               self.strainPassCats[idx], self.labels[idx]

def convert_Pass2tensor(pass_cats):
    result = [
        item.replace('<cls>', '0').replace('<eos>', '1').replace('<EGG>', '2').replace('<CELL>', '3').replace('<BOTH>', '4').replace('<NONE>', '5')
        for item in pass_cats
    ]
    result = torch.tensor([[int(number) for number in [char for char in item]] for item in result], dtype=torch.long)
    return result

def generate_matrix(matrix_list):
    seq_len = [mat.shape[0] for mat in matrix_list]
    max_len = max(seq_len)
    mask_list = []
    for i in range(len(matrix_list)):
        matrix_list[i] = F.pad(matrix_list[i], (0, 0, 0, max_len - seq_len[i]))
        mask = torch.concat((torch.ones(1, seq_len[i]), torch.zeros(1, max_len - seq_len[i])), axis=1)
        mask_list.append(mask)
    matrix = torch.stack(matrix_list)
    mask = torch.stack(mask_list).view(len(matrix_list), max_len)
    return matrix, mask


# =========================
# NEW: 你只需要改这几项即可跑不同工况
# =========================
data_path = '../../../../data/reverse_test/'
season_path = 'processed/test_2024NH/'        # 换这里即可对应不同窗口/holdout生成的train/test
run_root = './runs'                           # 所有运行统一放这里
run_name = 'v0_1'                             # 模型版本名
run_tag = 'test_2024NH'                       # 你自己标记：窗口/holdout/结构 ablation 等
device = torch.device('cuda:6')

batch_size = 8
epochs = 200
val_frac = 1/9
split_seed = 42


# =========================
# 读取数据（保持原逻辑）
# =========================
train_data = pd.read_csv(data_path + season_path + 'train.csv')
test_data = pd.read_csv(data_path + season_path + 'test.csv')
train_data, valid_data = train_test_split(train_data, test_size=val_frac, random_state=split_seed)
train_data_final = train_data  # 你原来预留 artificial_data

train_data_final = train_data_final.iloc[:200]
valid_data = valid_data.iloc[:200]
test_data = test_data.iloc[:200]

# =========================
# NEW: 创建 run_dir，并把“训练时实际用到的数据”落盘
# =========================
# 记录 split 信息（最小：来源路径+seed+val_frac；你后面可以加 report_id 列表等）
split_info = {
    "data_path": data_path,
    "season_path": season_path,
    "train_csv": str(Path(data_path) / season_path / "train.csv"),
    "test_csv": str(Path(data_path) / season_path / "test.csv"),
    "val_frac": val_frac,
    "split_seed": split_seed,
}
split_info["split_hash"] = _sha1_json(split_info)

# 先加载 config/args（用于写入 run 元信息）
with open('./config_dict.json', 'r') as f:
    config_dict = json.load(f)
with open("./args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)

# 组装一个可 JSON 化的 run_config（不要塞巨大对象）
run_config = {
    "run_name": run_name,
    "run_tag": run_tag,
    "device": str(device),
    "batch_size": batch_size,
    "epochs": epochs,
    "lr": 0.00008,
    "data": split_info,
    "model_config_dict_path": "./config_dict.json",
    "model_args_pkl_path": "./args.pkl",
    "fluProfiler_args_dict": vars(fluProfiler_args) if hasattr(fluProfiler_args, "__dict__") else str(fluProfiler_args),
    "config_dict": config_dict,  # 这个一般不大，直接存
}

logger = RunLogger.create(run_root=run_root, name=run_name, tag=run_tag, config_dict=run_config)
model_save_path = str(logger.run_dir / "checkpoints") + "/"  # 给 EarlyStopping 用（保持你原接口）

# 把 split.json 和 “训练时数据”保存到 run_dir，确保复现
logger.save_json("split.json", split_info)
train_data_final.to_csv(logger.run_dir / "artifacts" / "train_used.csv", index=False)
valid_data.to_csv(logger.run_dir / "artifacts" / "valid_used.csv", index=False)
test_data.to_csv(logger.run_dir / "artifacts" / "test_used.csv", index=False)


# =========================
# DataLoader（保持原逻辑）
# =========================
train_dataset = fluProfiler_Dataset(train_data_final)
valid_dataset = fluProfiler_Dataset(valid_data)
test_dataset = fluProfiler_Dataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# =========================
# embedding load（保持原逻辑）
# =========================
embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
sequence_names = pd.concat([embedding_df['seq_id_a'], embedding_df['seq_id_b'],
                            embedding_df['seq_id_c'], embedding_df['seq_id_d']]).unique().tolist()
sequence_names = ['matrix_' + item + '.pt' for item in sequence_names]
emb_dict = load_embedding(data_path + "/embedding", files=sequence_names)


# =========================
# 模型与优化器（保持原逻辑）
# =========================
fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = fluProfiler_v0_1(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

no_decay = ["bias", "layernorm.weight", "layer_norm.weight", "layer.norm.weight"]
optimizer_grouped_parameters = [{
    "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
    "weight_decay": fluProfiler_args.weight_decay},
    {
    "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
    "weight_decay": 0.0
}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=0.00008,
    betas=[fluProfiler_args.beta1 if fluProfiler_args.beta1 > 0 else 0.9,
           fluProfiler_args.beta2 if fluProfiler_args.beta2 > 0 else 0.98],
    eps=fluProfiler_args.adam_epsilon
)

num_training_steps = len(train_dataloader) * epochs
progress_bar = tqdm(range(num_training_steps))

early_stopping = EarlyStopping(patience=20, save_dir=model_save_path)

best_valid_mse = float("inf")


# =========================
# 训练循环（极少改动：加 logger 记录 + 保存 best 预测表）
# =========================
for epoch in range(epochs):
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        matrixs_a, masks_a = generate_matrix([emb_dict[key] for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([emb_dict[key] for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([emb_dict[key] for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([emb_dict[key] for key in emb_file_name_d])

        matrixs_a, matrixs_b, matrixs_c, matrixs_d = matrixs_a.to(device), matrixs_b.to(device), matrixs_c.to(device), matrixs_d.to(device)
        masks_a = masks_a.to(device)
        masks_b = masks_b.to(device)
        masks_c = masks_c.to(device)
        masks_d = masks_d.to(device)

        strainPassCats = strainPassCats.to(device)
        labels = labels.to(device)

        loss, logits, output = model(
            matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
            matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
            matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
            strainPassCats=strainPassCats, labels=labels
        )

        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    train_loss = float(np.mean(loss_ls))
    print('train loss :', train_loss)
    logger.add_scalar("train/loss", train_loss, epoch)

    # ---------- valid ----------
    prediction_ls_valid = []
    reference_ls_valid = []
    loss_ls_valid = []
    model.eval()
    for batch in valid_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        matrixs_a, masks_a = generate_matrix([emb_dict[key] for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([emb_dict[key] for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([emb_dict[key] for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([emb_dict[key] for key in emb_file_name_d])

        matrixs_a, matrixs_b, matrixs_c, matrixs_d = matrixs_a.to(device), matrixs_b.to(device), matrixs_c.to(device), matrixs_d.to(device)
        masks_a = masks_a.to(device)
        masks_b = masks_b.to(device)
        masks_c = masks_c.to(device)
        masks_d = masks_d.to(device)

        strainPassCats = strainPassCats.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            loss, logits, output = model(
                matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                strainPassCats=strainPassCats, labels=labels
            )

        loss_ls_valid.append(loss.item())
        prediction_ls_valid.extend(output.view(-1).tolist())
        reference_ls_valid.extend(labels.tolist())

    valid_mae, valid_mse, valid_pearson, valid_spearman, valid_R2 = print_exams(reference_ls_valid, prediction_ls_valid)

    logger.add_scalar("val/mae", valid_mae, epoch)
    logger.add_scalar("val/mse", valid_mse, epoch)

    # early stop（保持原逻辑）
    early_stopping(valid_mse, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # ---------- test ----------
    prediction_ls_test = []
    reference_ls_test = []
    loss_ls_test = []
    model.eval()
    for batch in test_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        matrixs_a, masks_a = generate_matrix([emb_dict[key] for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([emb_dict[key] for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([emb_dict[key] for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([emb_dict[key] for key in emb_file_name_d])

        matrixs_a, matrixs_b, matrixs_c, matrixs_d = matrixs_a.to(device), matrixs_b.to(device), matrixs_c.to(device), matrixs_d.to(device)
        masks_a = masks_a.to(device)
        masks_b = masks_b.to(device)
        masks_c = masks_c.to(device)
        masks_d = masks_d.to(device)

        strainPassCats = strainPassCats.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            loss, logits, output = model(
                matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                strainPassCats=strainPassCats, labels=labels
            )

        loss_ls_test.append(loss.item())
        prediction_ls_test.extend(output.view(-1).tolist())
        reference_ls_test.extend(labels.tolist())

    test_mae, test_mse, test_pearson, test_spearman, test_R2 = print_exams(reference_ls_test, prediction_ls_test)

    logger.add_scalar("test/mae", test_mae, epoch)
    logger.add_scalar("test/mse", test_mse, epoch)
    logger.flush_tb()

    # ---------- NEW: 统一写 metrics.jsonl（机器可读）
    logger.log({
        "train_loss": train_loss,
        "valid_mae": float(valid_mae),
        "valid_mse": float(valid_mse),
        "valid_pearson": float(valid_pearson.statistic),
        "valid_spearman": float(valid_spearman.statistic),
        "valid_r2": float(valid_R2),
        "test_mae": float(test_mae),
        "test_mse": float(test_mse),
        "test_pearson": float(test_pearson.statistic),
        "test_spearman": float(test_spearman.statistic),
        "test_r2": float(test_R2),
    }, epoch=epoch)

    # ---------- NEW: 保留你原来的 log.txt（人类可读），但写到 run_dir
    with open(str(logger.run_dir / 'log.txt'), 'a') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"[{current_time}] Epoch {epoch + 1}/{epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"valid MAE: {valid_mae:.5f}, valid MSE: {valid_mse:.5f}, "
            f"valid Pearson: {valid_pearson.statistic:.5f}, "
            f"valid Spearman: {valid_spearman.statistic:.5f}, "
            f"valid R2: {valid_R2:.5f}, "
            f"test MAE: {test_mae:.5f}, test MSE: {test_mse:.5f}, "
            f"test Pearson: {test_pearson.statistic:.5f}, "
            f"test Spearman: {test_spearman.statistic:.5f}, "
            f"test R2: {test_R2:.5f}\n"
        )

    # ---------- NEW: 保存 best epoch 的逐样本预测（只保存 best，避免文件爆炸）
    if valid_mse < best_valid_mse:
        best_valid_mse = float(valid_mse)

        valid_out = valid_data.reset_index(drop=True).copy()
        valid_out["pred"] = prediction_ls_valid
        valid_out.to_csv(logger.run_dir / "artifacts" / "valid_preds_best.csv", index=False)

        test_out = test_data.reset_index(drop=True).copy()
        test_out["pred"] = prediction_ls_test
        test_out.to_csv(logger.run_dir / "artifacts" / "test_preds_best.csv", index=False)

        logger.save_json("best.json", {"best_epoch": epoch, "best_valid_mse": best_valid_mse})

# 训练结束
logger.close_tb()
print("Run saved to:", str(logger.run_dir))