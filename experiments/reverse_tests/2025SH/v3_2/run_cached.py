import sys
sys.path.append('../../../../src/fluprofiler')
sys.path.append('../../')
from experiment_tools import (
    find_repo_root, default_exp_id, make_run_dirs,
    load_data_and_dataloaders, setup_optimizer, setup_tensorboard_and_logging,
    train_step_cached, evaluate_step_cached, log_metrics_to_tensorboard, log_epoch_to_file,
    GpuEmbeddingCache,
)
from models.architectures import fluProfiler_v3_0, fluProfiler_v3_1, fluProfiler_v3_2, fluProfiler_Config
from tqdm import tqdm
import os
from pathlib import Path
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from datetime import datetime
from evaluation.metrics import EarlyStopping
import pickle

# 设置路径和实验ID
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = find_repo_root(_SCRIPT_PATH)
root_path = str(_REPO_ROOT) + "/"
data_path = root_path + 'data/reverse_test/'
season_path = 'processed/test_2025SH/'
epochs = 250
patience = 20
sample_limit = None  # sampling data for testing
use_lr_schedule = True  # True 时启用 CosineAnnealingLR 学习率衰减
MODEL_CLASS = fluProfiler_v3_2  # 使用的模型类
tag = os.environ.get("FLUPROFILER_TAG") or "v3_2_cached"
batch_size = 32
gpu_cache_GB =26
lr = 0.00008
device = torch.device('cuda:4')

exp_id = os.environ.get("FLUPROFILER_EXP_ID") or default_exp_id(_SCRIPT_PATH, _REPO_ROOT)
run_paths = make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)

# 设置 TensorBoard 和日志
logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=tag)
writer = logging_info['writer']
log_path = logging_info['log_path']
model_save_path = str(run_paths["checkpoints"]) + "/"

# 加载数据和创建 DataLoader
data_loaders = load_data_and_dataloaders(
    data_path=data_path,
    season_path=season_path,
    batch_size=batch_size,
    sample_limit=sample_limit,
    use_artificial=False
)
train_dataloader = data_loaders['train_dataloader']
valid_dataloader = data_loaders['valid_dataloader']
test_dataloader = data_loaders['test_dataloader']
emb_dict = data_loaders['emb_dict']

# 创建 GPU 端 embedding 缓存
# 调整为 20GB，避免 OOM（给模型参数、激活等预留空间）
max_cache_bytes = gpu_cache_GB * 1024 ** 3
gpu_cache = GpuEmbeddingCache(cpu_store=emb_dict, device=device, max_bytes=max_cache_bytes)

# 加载模型配置和创建模型
with open(root_path + 'configs/config_dict.json', 'r') as f:
    config_dict = json.load(f)
with open(root_path + "configs/args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)

fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = MODEL_CLASS(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

run_config = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "data_path": data_path,
    "season_path": season_path,
    "data_path": data_path,
    "season_path": season_path,
    "epochs": epochs,
    "patience": patience,
    "batch_size": batch_size,
    "sample_limit": sample_limit,
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

# 将模型结构保存到当前 run 的输出目录
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


# 设置优化器
optimizer = setup_optimizer(model, fluProfiler_args, lr=lr)

# 学习率调度（可选，use_lr_schedule=True 时启用）
scheduler = None
if use_lr_schedule:
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print(f"[lr_schedule] CosineAnnealingLR 已启用, T_max={epochs}, eta_min=1e-6")

# 训练设置
num_training_steps = len(train_dataloader) * epochs
progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=patience, save_dir=model_save_path)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        loss = train_step_cached(model, batch, emb_dict, device, gpu_cache)
        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    train_loss = np.mean(loss_ls)
    print('train loss :', train_loss)

    # 显存监控信息
    cache_gb = gpu_cache.current_bytes / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"[cache] hits={gpu_cache.hits}, misses={gpu_cache.misses}, "
          f"copied_GB={gpu_cache.copied_bytes / (1024 ** 3):.2f}, "
          f"current_GB={cache_gb:.2f}")
    print(f"[memory] allocated_GB={allocated_gb:.2f}, "
          f"reserved_GB={reserved_gb:.2f}, "
          f"cache_ratio={cache_gb/allocated_gb:.3f} (cache/total)")

    # 验证阶段
    valid_metrics = evaluate_step_cached(model, valid_dataloader, emb_dict, device, gpu_cache)

    # 测试阶段
    test_metrics = evaluate_step_cached(model, test_dataloader, emb_dict, device, gpu_cache)

    # 记录到 TensorBoard
    log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

    # 学习率调度（逐 epoch 衰减）
    if scheduler is not None:
        scheduler.step()
        print(f"[lr_schedule] current_lr={optimizer.param_groups[0]['lr']:.8f}")

    # 早停检查
    early_stopping(valid_metrics['mse'], model)
    if early_stopping.early_stop:
        print("Early stopping")
        writer.close()
        break

    # 记录到日志文件
    log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics)

else:  # 仅当未触发 break 时执行（正常完成所有 epoch）
    writer.close()

