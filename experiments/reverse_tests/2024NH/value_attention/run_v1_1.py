#!/usr/bin/env python3
"""
fluProfiler v1.1 均衡优化版训练脚本

主要改进：
1. 使用 fluProfiler_v1_1 模型（轻量级改进，参数增加<10%）
2. 更强的正则化（dropout, weight decay）
3. Huber Loss（对异常值更鲁棒）
4. 标签平滑（减少过拟合）
5. 学习率调度器
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
from models.architectures import fluProfiler_v1_1, fluProfiler_Config
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import numpy as np
from evaluation.metrics import EarlyStopping
import pickle

# Huber Loss实现
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        error = pred - target
        is_small_error = torch.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * torch.abs(error) - 0.5 * self.delta ** 2
        return torch.where(is_small_error, squared_loss, linear_loss).mean()

# 标签平滑
def smooth_labels(labels, smoothing=0.1):
    """对回归任务的标签进行平滑"""
    # 对于回归任务，我们使用高斯噪声进行平滑
    noise = torch.randn_like(labels) * smoothing * labels.std()
    return labels + noise

# 设置路径和实验ID
_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = find_repo_root(_SCRIPT_PATH)
root_path = str(_REPO_ROOT) + "/"
data_path = root_path + 'data/reverse_test/'
season_path = 'processed/test_2024NH/'

exp_id = os.environ.get("FLUPROFILER_EXP_ID") or default_exp_id(_SCRIPT_PATH, _REPO_ROOT)
tag = os.environ.get("FLUPROFILER_TAG") or "v1_1"
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
    batch_size=8,  # 保持batch_size=16
    sample_limit=None,  # 使用全部数据
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
model = fluProfiler_v1_1(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

# 设置优化器（增强版：更强的weight decay）
optimizer = setup_optimizer(model, fluProfiler_args, lr=0.0001)
# 增加weight decay
for param_group in optimizer.param_groups:
    param_group['weight_decay'] = 5e-4  # 从默认值增加到5e-4

# 学习率调度器
from transformers import get_linear_schedule_with_warmup
epochs = 50
num_training_steps = len(train_dataloader) * epochs
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# Huber Loss
huber_loss = HuberLoss(delta=1.0)

progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=15, save_dir=model_save_path)

print("开始训练 fluProfiler v1.1...")
print(f"训练轮数: {epochs}")
print(f"批次大小: 16")
print(f"学习率: 0.0001 (带warmup和衰减)")
print(f"Weight Decay: 5e-4")
print(f"Loss: Huber Loss (delta=1.0)")
print(f"标签平滑: 0.1")
print(f"设备: {device}")

# 修改train_step以使用Huber Loss和标签平滑
def train_step_v1_1(model, batch, emb_dict, device, generate_matrix_fn, label_smoothing=0.1):
    """改进的训练步骤，使用Huber Loss和标签平滑"""
    emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch
    
    # 生成矩阵和掩码
    matrixs_a, masks_a = generate_matrix_fn([emb_dict[key] for key in emb_file_name_a])
    matrixs_b, masks_b = generate_matrix_fn([emb_dict[key] for key in emb_file_name_b])
    matrixs_c, masks_c = generate_matrix_fn([emb_dict[key] for key in emb_file_name_c])
    matrixs_d, masks_d = generate_matrix_fn([emb_dict[key] for key in emb_file_name_d])
    
    # 移动到设备
    matrixs_a, matrixs_b, matrixs_c, matrixs_d = matrixs_a.to(device), matrixs_b.to(device), matrixs_c.to(device), matrixs_d.to(device)
    masks_a, masks_b, masks_c, masks_d = masks_a.to(device), masks_b.to(device), masks_c.to(device), masks_d.to(device)
    strainPassCats = strainPassCats.to(device)
    labels = labels.to(device)
    
    # 标签平滑
    if label_smoothing > 0:
        labels_smooth = smooth_labels(labels, smoothing=label_smoothing)
    else:
        labels_smooth = labels
    
    # 前向传播
    loss, logits, output = model(
        matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
        matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
        matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
        strainPassCats=strainPassCats, labels=labels_smooth
    )
    
    # 使用Huber Loss替代MSE
    huber_loss_value = huber_loss(output.view(-1), labels.view(-1))
    
    # 组合损失：原始损失 + Huber Loss
    total_loss = 0.7 * loss + 0.3 * huber_loss_value
    
    return total_loss

for epoch in range(epochs):
    # 训练阶段
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        loss = train_step_v1_1(model, batch, emb_dict, device, generate_matrix, label_smoothing=0.1)

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