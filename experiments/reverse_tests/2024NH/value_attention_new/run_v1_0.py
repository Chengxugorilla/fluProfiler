#!/usr/bin/env python3
"""
fluProfiler v1.0 改进版训练脚本

主要改进：
1. 使用 fluProfiler_v1_0 模型（跨句子注意力、自适应池化）
2. 添加梯度裁剪和更好的正则化
3. 学习率调度器
4. 更稳定的训练过程
"""

import sys
import os
from pathlib import Path
sys.path.append('../../../../src/fluprofiler')
sys.path.append('../../')
from experiment_tools import (
    find_repo_root, default_exp_id, make_run_dirs, generate_matrix,
    load_data_and_dataloaders, setup_optimizer, setup_tensorboard_and_logging,
    train_step, evaluate_step, log_metrics_to_tensorboard, log_epoch_to_file
)
from models.architectures import fluProfiler_v1_0, fluProfiler_Config
from tqdm import tqdm
import json
import torch
import numpy as np
from evaluation.metrics import EarlyStopping
import pickle

# 设置路径和实验ID
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = find_repo_root(_SCRIPT_PATH)
root_path = str(_REPO_ROOT) + "/"
data_path = root_path + 'data/reverse_test/'
season_path = 'processed/test_2024NH/'

exp_id = os.environ.get("FLUPROFILER_EXP_ID") or default_exp_id(_SCRIPT_PATH, _REPO_ROOT)
tag = os.environ.get("FLUPROFILER_TAG") or "v1_0"
run_paths = make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)

# 设置设备
device = torch.device('cuda:5')

# 设置 TensorBoard 和日志
logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=tag)
writer = logging_info['writer']
log_path = logging_info['log_path']
model_save_path = str(run_paths["checkpoints"]) + "/"

# 加载数据和创建 DataLoader
data_loaders = load_data_and_dataloaders(
    data_path=data_path,
    season_path=season_path,
    batch_size=16,  # 增大batch_size以提高稳定性
    sample_limit=100,
    use_artificial=False
)
train_dataloader = data_loaders['train_dataloader']
valid_dataloader = data_loaders['valid_dataloader']
test_dataloader = data_loaders['test_dataloader']
emb_dict = data_loaders['emb_dict']

# 加载模型配置和创建模型
with open(root_path + 'configs/config_dict.json', 'r') as f:
    config_dict = json.load(f)
with open(root_path + "configs/args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)

fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = fluProfiler_v1_0(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

# 设置优化器（增强版）
optimizer = setup_optimizer(model, fluProfiler_args, lr=0.0001)  # 稍微降低学习率

# 学习率调度器
from transformers import get_linear_schedule_with_warmup
epochs = 50  # 增加训练轮数
num_training_steps = len(train_dataloader) * epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=15, save_dir=model_save_path)  # 增加耐心值

print("开始训练 fluProfiler v1.0...")
print(f"训练轮数: {epochs}")
print(f"批次大小: 16")
print(f"学习率: 0.0001 (带warmup和衰减)")
print(f"设备: {device}")

for epoch in range(epochs):
    # 训练阶段
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        loss = train_step(model, batch, emb_dict, device, generate_matrix)

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        # 学习率调度
        scheduler.step()

        progress_bar.update(1)

    train_loss = np.mean(loss_ls)
    current_lr = scheduler.get_last_lr()[0]
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, LR: {current_lr:.6f}')

    # 验证阶段
    valid_metrics = evaluate_step(model, valid_dataloader, emb_dict, device, generate_matrix)

    # 测试阶段
    test_metrics = evaluate_step(model, test_dataloader, emb_dict, device, generate_matrix)

    # 记录到 TensorBoard
    log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

    # 记录学习率
    writer.add_scalar('Learning_Rate', current_lr, epoch)

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

print("训练完成！")
print(f"模型保存路径: {run_paths['checkpoints']}")
print(f"TensorBoard路径: {run_paths['tensorboard']}")
print(f"日志文件: {log_path}")