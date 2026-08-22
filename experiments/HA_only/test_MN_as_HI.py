from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===== 路径配置（按需修改）=====
REPO_ROOT = Path('/home/chenyh/workspace/fluProfiler')
CHECKPOINT_PATH = REPO_ROOT / 'runs/HA_only/split/HA/20260322_101738__HA_only_cached__pid908736/checkpoints/2026-03-22_21-21-28.pth'
CSV_PATH = REPO_ROOT / 'data/HA_only/split/test.csv'
EMBEDDING_ROOT = REPO_ROOT / 'data/reverse_test/embedding'
#OUTPUT_PRED_CSV = Path("./test_predictions.csv")
OUTPUT_PRED_CSV = REPO_ROOT / "runs/MN_as_HI_test/MN_as_HI_predictions.csv"
OUTPUT_PRED_CSV.parent.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
GPU_CACHE_GB = 28
DEVICE = 'cuda:2'  # 没有 GPU 可改成 'cpu'
ADD_SPECIAL_TOKEN = True

# 本地模块路径
sys.path.append(str(REPO_ROOT / 'src' / 'fluprofiler'))
sys.path.append(str(REPO_ROOT / 'experiments' / 'reverse_tests'))

from experiment_tools import GpuEmbeddingCache, convert_Pass2tensor, generate_matrix_on_device
from data.loaders import load_embedding
from evaluation.metrics import print_exams

print('checkpoint:', CHECKPOINT_PATH)
print('csv:', CSV_PATH)
print('embedding_root:', EMBEDDING_ROOT)

class HAOnlyInferenceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, add_special_token: bool = True):
        self.df = dataframe.reset_index(drop=True)
        self.emb_file_name_a = ('matrix_' + self.df['seq_id_a']).tolist()
        self.emb_file_name_c = ('matrix_' + self.df['seq_id_c']).tolist()
        if add_special_token:
            self.strainPassCats = convert_Pass2tensor(
                (
                    '<cls>'
                    + self.df['serumPassCat']
                    + '<eos>'
                    + self.df['virusPassCat']
                    + '<eos>'
                ).tolist()
            )
        else:
            self.strainPassCats = convert_Pass2tensor(
                (self.df['serumPassCat'] + self.df['virusPassCat']).tolist()
            )
        self.labels = torch.tensor(self.df['label'].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.emb_file_name_a[idx],
            self.emb_file_name_c[idx],
            self.strainPassCats[idx],
            self.labels[idx],
        )


from tqdm.auto import tqdm

def run_inference(model, dataloader, device, cache, show_progress=True):
    model.eval()
    preds, refs = [], []
    iterator = tqdm(dataloader, total=len(dataloader), desc="Inference") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            emb_file_name_a, emb_file_name_c, strainPassCats, labels = batch
            matrices_a_list = [cache.get(key) for key in emb_file_name_a]
            matrices_c_list = [cache.get(key) for key in emb_file_name_c]
            matrixs_a, masks_a = generate_matrix_on_device(matrices_a_list, device=device)
            matrixs_c, masks_c = generate_matrix_on_device(matrices_c_list, device=device)
            strainPassCats = strainPassCats.to(device)
            labels = labels.to(device)

            loss, _, batch_preds = model(
                matrices_a=matrixs_a,
                matrices_c=matrixs_c,
                matrix_attention_masks_a=masks_a,
                matrix_attention_masks_c=masks_c,
                strainPassCats=strainPassCats,
                labels=labels,
            )

            preds.extend(batch_preds.view(-1).detach().cpu().tolist())
            refs.extend(labels.view(-1).detach().cpu().tolist())

    return refs, preds

device = torch.device(DEVICE)

# 1) 读测试集
inference_df = pd.read_csv(CSV_PATH)
print('test rows:', len(inference_df))

if 'label' not in inference_df.columns:
    if 'HI_Dist' in inference_df.columns:
        inference_df['label'] = inference_df['HI_Dist']
    else:
        raise ValueError("test.csv must contain label or HI_Dist")

if 'HI_Dist' in inference_df.columns:
    inference_df['label'] = pd.to_numeric(inference_df['label'], errors='coerce')
    hi = pd.to_numeric(inference_df['HI_Dist'], errors='coerce')
    inference_df['label'] = inference_df['label'].fillna(hi)

core_cols = ['seq_id_a', 'seq_id_c', 'serumPassCat', 'virusPassCat', 'label']
before = len(inference_df)
inference_df = inference_df.dropna(subset=core_cols).copy()
print('dropped rows with core NA:', before - len(inference_df))
print('usable rows:', len(inference_df))


# 2) 仅加载推理所需 embedding
sequence_names = pd.concat([inference_df['seq_id_a'], inference_df['seq_id_c']]).unique().tolist()
embedding_files = ['matrix_' + item + '.pt' for item in sequence_names]
#emb_dict = load_embedding("/home/chenyh/workspace/fluProfiler_back3/workplace/dms_experiment/20260323meeting/embedding", files=embedding_files)
emb_dict = load_embedding(str(EMBEDDING_ROOT), files=embedding_files)

max_cache_bytes = int(GPU_CACHE_GB * 1024**3)
gpu_cache = GpuEmbeddingCache(cpu_store=emb_dict, device=device, max_bytes=max_cache_bytes)

# 3) DataLoader
dataset = HAOnlyInferenceDataset(inference_df, add_special_token=ADD_SPECIAL_TOKEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print('serum count', len(inference_df['seq_id_a'].unique()))
print('virus count', len(inference_df['seq_id_c'].unique()))

# 4) 加载模型（该 checkpoint 为 torch.save(model, ...)，直接 torch.load）
model = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.to(device)
model.eval()

# 5) 推理
refs, preds = run_inference(model, dataloader, device, gpu_cache)

# 6) 指标
refs = np.array(refs, dtype=float)
preds = np.array(preds, dtype=float)

finite_mask = np.isfinite(refs) & np.isfinite(preds)
print("total predictions:", len(preds))
print("finite predictions:", finite_mask.sum())
print("non-finite predictions:", (~finite_mask).sum())

if (~finite_mask).sum() > 0:
    bad_df = inference_df.iloc[np.where(~finite_mask)[0]].copy()
    bad_df["true_label"] = refs[~finite_mask]
    bad_df["prediction"] = preds[~finite_mask]
    bad_df.to_csv(OUTPUT_PRED_CSV.parent / "MN_as_HI_nonfinite_predictions.csv", index=False)
    print("saved non-finite rows:", OUTPUT_PRED_CSV.parent / "MN_as_HI_nonfinite_predictions.csv")

refs = refs[finite_mask].tolist()
preds = preds[finite_mask].tolist()
mae, mse, pearson, spearman, r2 = print_exams(refs, preds, print_result=True)

# 7) 保存预测
# 保存过滤 non-finite 后的预测结果
out_df = inference_df.iloc[np.where(finite_mask)[0]].copy()
out_df["true_label"] = refs
out_df["prediction"] = preds
out_df.to_csv(OUTPUT_PRED_CSV, index=False)
print("saved:", OUTPUT_PRED_CSV.resolve())


# 8) 打印缓存统计
print(f"[cache] hits={gpu_cache.hits}, misses={gpu_cache.misses}, copied_GB={gpu_cache.copied_bytes/(1024**3):.2f}")