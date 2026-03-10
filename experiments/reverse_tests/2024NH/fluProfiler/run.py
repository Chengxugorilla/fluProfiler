import sys
sys.path.append('../../../../src/fluprofiler')
sys.path.append('../../')
from experiment_tools import (
    find_repo_root, default_exp_id, make_run_dirs, generate_matrix,
    load_data_and_dataloaders, setup_optimizer, setup_tensorboard_and_logging,
    train_step, evaluate_step, log_metrics_to_tensorboard, log_epoch_to_file
)
from models.architectures import fluProfiler, fluProfiler_Config
from tqdm import tqdm
import os
from pathlib import Path
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
tag = os.environ.get("FLUPROFILER_TAG") or "v0_1_1"
run_paths = make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)

# 设置设备
device = torch.device('cuda:6')

# 设置 TensorBoard 和日志
logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=tag)
writer = logging_info['writer']
log_path = logging_info['log_path']
model_save_path = str(run_paths["checkpoints"]) + "/"

# 加载数据和创建 DataLoader
data_loaders = load_data_and_dataloaders(
    data_path=data_path,
    season_path=season_path,
    batch_size=8,
    sample_limit=None,
    use_artificial=False,
    add_special_token=True
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
model = fluProfiler(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

# 设置优化器
optimizer = setup_optimizer(model, fluProfiler_args, lr=0.00008)

# 训练设置
epochs = 20
num_training_steps = len(train_dataloader) * epochs
progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=20, save_dir=model_save_path)

for epoch in range(epochs):
    # 训练阶段
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        loss = train_step(model, batch, emb_dict, device, generate_matrix)
        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    train_loss = np.mean(loss_ls)
    print('train loss :', train_loss)

    # 验证阶段
    valid_metrics = evaluate_step(model, valid_dataloader, emb_dict, device, generate_matrix)

    # 测试阶段
    test_metrics = evaluate_step(model, test_dataloader, emb_dict, device, generate_matrix)

    # 记录到 TensorBoard
    log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

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