import sys
sys.path.append('../../../../src/fluprofiler')
from models.architectures import fluProfiler_v0_1, fluProfiler_Config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
import json
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from data.loaders import load_embedding
from evaluation.metrics import print_exams, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime
import pickle

class fluProfiler_Dataset(Dataset):
    def __init__(self, DataFrame):
        self.emb_file_name_a = ('matrix_' + DataFrame['seq_id_a']).tolist()
        self.emb_file_name_b = ('matrix_' + DataFrame['seq_id_b']).tolist()
        self.emb_file_name_c = ('matrix_' + DataFrame['seq_id_c']).tolist()
        self.emb_file_name_d = ('matrix_' + DataFrame['seq_id_d']).tolist()

        self.strainPassCats = convert_Pass2tensor(('<cls>' + DataFrame['serumPassCat'] + '<eos>' + DataFrame['virusPassCat'] + '<eos>').tolist())

        self.labels = torch.tensor(DataFrame['label'].tolist())
    
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
    result = torch.tensor([[int(number) for number in [char for char in item]] for item in result])
    return result

def generate_matrix(matrix_list):
    seq_len = [mat.shape[0] for mat in matrix_list]
    max_len = max(seq_len)
    mask_list = []
    for i in range(len(matrix_list)): 
        matrix_list[i] = F.pad(matrix_list[i], (0, 0, 0, max_len - seq_len[i]))
        mask = torch.concat((torch.ones(1,seq_len[i]),torch.zeros(1,max_len-seq_len[i])),axis=1)
        mask_list.append(mask)
    matrix = torch.stack(matrix_list)
    mask = torch.stack(mask_list).view(len(matrix_list),max_len)
    return matrix, mask

def _find_repo_root(start: Path) -> Path:
    """Find fluProfiler repo root by walking parents."""
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "configs").exists():
            return p
        if (p / ".git").exists():
            return p
    # Fallback: assume 3 levels up (typical experiments/<...>/run.py)
    return start.parents[2] if len(start.parents) >= 3 else start

def _default_exp_id(script_path: Path, repo_root: Path) -> str:
    """Infer exp_id like 'reverse_tests/2024NH/model4' from script location."""
    try:
        rel = script_path.resolve().relative_to(repo_root)
        parts = list(rel.parts)
        if "experiments" in parts:
            i = parts.index("experiments")
            exp_parts = parts[i + 1 : -1]  # drop filename
            if exp_parts:
                return "/".join(exp_parts)
    except Exception:
        pass
    return "v0_1"

def _make_run_dirs(repo_root: Path, exp_id: str, tag: str = "v0_1") -> dict:
    """Create run directory tree under <repo_root>/runs/<exp_id>/<run_id>/..."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}__{tag}__pid{os.getpid()}"
    run_root = (repo_root / "runs" / exp_id / run_id).resolve()

    paths = {
        "run_root": run_root,
        "checkpoints": run_root / "checkpoints",
        "metrics": run_root / "metrics",
        "preds": run_root / "preds",
        "plots": run_root / "plots",
        "meta": run_root / "run_meta.json",
    }

    # Create directories (fail fast if collision)
    for k, p in paths.items():
        if k == "meta":
            continue
        p.mkdir(parents=True, exist_ok=False)

    return paths
 
 
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _find_repo_root(_SCRIPT_PATH)
## read complete information
root_path = str(_REPO_ROOT) + "/"
data_path = root_path + 'data/reverse_test/'
season_path = 'processed/test_2024NH/'

exp_id = os.environ.get("FLUPROFILER_EXP_ID") or _default_exp_id(_SCRIPT_PATH, _REPO_ROOT)
tag = os.environ.get("FLUPROFILER_TAG") or "v0_1"
run_paths = _make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)

# 创建与 checkpoints 同级的 tensorboard 目录
tensorboard_log_dir = run_paths['run_root'] / "tensorboard"
tensorboard_log_dir.mkdir(parents=True, exist_ok=False)  # 与原目录创建风格一致
writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
print(f"TensorBoard logs saved to: {tensorboard_log_dir}")

model_save_path = str(run_paths["checkpoints"]) + "/"  # EarlyStopping expects a dir
log_path = run_paths['run_root'] / "log.txt"

run_paths["meta"].write_text(
    json.dumps(
        {
            "exp_id": exp_id,
            "season_path": season_path,
            "device": "cuda:6",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


train_data = pd.read_csv(data_path + season_path + 'train.csv')
test_data = pd.read_csv(data_path + season_path + 'test.csv')
train_data, valid_data = train_test_split(train_data, test_size=1/9, random_state=42)     

# Artificial_data = pd.read_csv(data_path + season_path + 'artificial_data.csv')
# train_data_final = pd.concat([train_data, Artificial_data])
train_data_final = train_data

train_data_final = train_data_final.iloc[:100]
valid_data = valid_data.iloc[:100]
test_data = test_data.iloc[:100]

train_dataset = fluProfiler_Dataset(train_data_final)
valid_dataset = fluProfiler_Dataset(valid_data)
test_dataset = fluProfiler_Dataset(test_data)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load embedding
device = torch.device('cuda:6')
embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
sequence_names = pd.concat([embedding_df['seq_id_a'], embedding_df['seq_id_b'], 
                            embedding_df['seq_id_c'], embedding_df['seq_id_d']]).unique().tolist()
sequence_names = ['matrix_' + item + '.pt' for item in sequence_names]
emb_dict = load_embedding(data_path + "/embedding", files=sequence_names)

with open(root_path + 'configs/config_dict.json', 'r') as f:
    config_dict = json.load(f)
with open(root_path + "configs/args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)


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
}
]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=0.00008,
                  betas=[fluProfiler_args.beta1 if fluProfiler_args.beta1 > 0 else 0.9,
                         fluProfiler_args.beta2 if fluProfiler_args.beta2 > 0 else 0.98],
                  eps=fluProfiler_args.adam_epsilon)

epochs = 200
num_training_steps = len(train_dataloader) * epochs

progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=20, save_dir=model_save_path)

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

        loss, logits, output = model(matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                                     matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                                     matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                                     strainPassCats=strainPassCats, labels=labels)

        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    train_loss = np.mean(loss_ls)
    print('train loss :', train_loss)

    prediction_ls_valid = []
    reference_ls_valid = []
    logits_ls = []
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
            loss, logits, output = model(matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c,
                                         matrices_d=matrixs_d, matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                                         matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d, strainPassCats=strainPassCats,
                                         labels=labels)

        loss_ls_valid.append(loss.item())
        logits_ls.append(logits.tolist())
        prediction_ls_valid.extend(output.view(-1).tolist())
        reference_ls_valid.extend(labels.tolist())

    valid_mae, valid_mse, valid_pearson, valid_spearman, valid_R2 = print_exams(reference_ls_valid, prediction_ls_valid)

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', np.mean(loss_ls_valid), epoch)
    writer.add_scalar('MAE/valid', valid_mae, epoch)
    writer.add_scalar('MSE/valid', valid_mse, epoch)
    writer.add_scalar('Pearson/valid', valid_pearson.statistic, epoch)
    writer.add_scalar('Spearman/valid', valid_spearman.statistic, epoch)
    writer.add_scalar('R2/valid', valid_R2, epoch)

    early_stopping(valid_mse, model)
    if early_stopping.early_stop:
        print("Early stopping")
        writer.close()
        break

    prediction_ls_test = []
    reference_ls_test = []
    logits_ls = []
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
            loss, logits, output = model(matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c,
                                         matrices_d=matrixs_d, matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                                         matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d, strainPassCats=strainPassCats,
                                         labels=labels)

        loss_ls_test.append(loss.item())
        logits_ls.append(logits.tolist())
        prediction_ls_test.extend(output.view(-1).tolist())
        reference_ls_test.extend(labels.tolist())

    test_mae, test_mse, test_pearson, test_spearman, test_R2 = print_exams(reference_ls_test, prediction_ls_test)

    writer.add_scalar('MAE/test', test_mae, epoch)
    writer.add_scalar('MSE/test', test_mse, epoch)
    writer.add_scalar('Pearson/test', test_pearson.statistic, epoch)
    writer.add_scalar('Spearman/test', test_spearman.statistic, epoch)
    writer.add_scalar('R2/test', test_R2, epoch)

    # 将epoch信息写入log.txt
    with open(log_path, 'a') as f:
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

else:  # 仅当未触发 break 时执行（正常完成所有 epoch）
    writer.close()