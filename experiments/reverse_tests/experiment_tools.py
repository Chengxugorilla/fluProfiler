import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import os
import pandas as pd
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

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

def find_repo_root(start: Path) -> Path:
    """Find fluProfiler repo root by walking parents."""
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "configs").exists():
            return p
        if (p / ".git").exists():
            return p
    # Fallback: assume 3 levels up (typical experiments/<...>/run.py)
    return start.parents[2] if len(start.parents) >= 3 else start

def default_exp_id(script_path: Path, repo_root: Path) -> str:
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

def make_run_dirs(repo_root: Path, exp_id: str, tag: str = "v0_1") -> dict:
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


def load_data_and_dataloaders(data_path, season_path, batch_size=8, sample_limit=None, use_artificial=False):
    """
    加载数据并创建 DataLoader
    
    Args:
        data_path: 数据根目录路径
        season_path: 季节数据子目录路径
        batch_size: 批次大小
        sample_limit: 限制样本数量（用于调试）
        use_artificial: 是否使用人工数据
    
    Returns:
        dict: 包含 train_dataloader, valid_dataloader, test_dataloader, emb_dict
    """
    # 导入 load_embedding，需要根据实际路径调整
    try:
        from data.loaders import load_embedding
    except ImportError:
        # 如果导入失败，尝试其他路径
        import sys
        sys.path.append('../../../../src/fluprofiler')
        from data.loaders import load_embedding
    
    train_data = pd.read_csv(data_path + season_path + 'train.csv')
    test_data = pd.read_csv(data_path + season_path + 'test.csv')
    train_data, valid_data = train_test_split(train_data, test_size=1/9, random_state=42)
    
    # 可选的人工数据合并
    if use_artificial:
        try:
            artificial_data = pd.read_csv(data_path + season_path + 'artificial_data.csv')
            train_data_final = pd.concat([train_data, artificial_data])
        except FileNotFoundError:
            train_data_final = train_data
    else:
        train_data_final = train_data
    
    # 限制样本数量用于调试
    if sample_limit:
        train_data_final = train_data_final.iloc[:sample_limit]
        valid_data = valid_data.iloc[:sample_limit]
        test_data = test_data.iloc[:sample_limit]
    
    # 创建数据集
    train_dataset = fluProfiler_Dataset(train_data_final)
    valid_dataset = fluProfiler_Dataset(valid_data)
    test_dataset = fluProfiler_Dataset(test_data)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 加载 embedding
    embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
    sequence_names = pd.concat([embedding_df['seq_id_a'], embedding_df['seq_id_b'],
                               embedding_df['seq_id_c'], embedding_df['seq_id_d']]).unique().tolist()
    sequence_names = ['matrix_' + item + '.pt' for item in sequence_names]
    emb_dict = load_embedding(data_path + "/embedding", files=sequence_names)
    
    return {
        'train_dataloader': train_dataloader,
        'valid_dataloader': valid_dataloader,
        'test_dataloader': test_dataloader,
        'emb_dict': emb_dict
    }


def setup_optimizer(model, args, lr=0.00008):
    """
    设置优化器
    
    Args:
        model: PyTorch 模型
        args: 模型参数对象（包含 weight_decay, beta1, beta2, adam_epsilon）
        lr: 学习率
    
    Returns:
        optimizer: AdamW 优化器
    """
    no_decay = ["bias", "layernorm.weight", "layer_norm.weight", "layer.norm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
        "weight_decay": args.weight_decay},
        {
        "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
        "weight_decay": 0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters,
                     lr=lr,
                     betas=[args.beta1 if args.beta1 > 0 else 0.9,
                           args.beta2 if args.beta2 > 0 else 0.98],
                     eps=args.adam_epsilon)
    return optimizer


def setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=None):
    """
    设置 TensorBoard 和日志
    
    Args:
        run_paths: 运行目录路径字典
        exp_id: 实验ID
        season_path: 季节路径
        device: 设备（字符串，如 'cuda:6'）
        tag: 标签（可选）
    
    Returns:
        dict: 包含 writer, log_path, tensorboard_dir
    """
    # 创建 tensorboard 目录
    tensorboard_log_dir = run_paths['run_root'] / "tensorboard"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    print(f"TensorBoard logs saved to: {tensorboard_log_dir}")
    
    log_path = run_paths['run_root'] / "log.txt"
    
    # 写入运行元数据
    run_paths["meta"].write_text(
        json.dumps(
            {
                "exp_id": exp_id,
                "season_path": season_path,
                "device": str(device),
                "tag": tag,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    
    return {
        'writer': writer,
        'log_path': log_path,
        'tensorboard_dir': tensorboard_log_dir
    }


def train_step(model, batch, emb_dict, device, generate_matrix_fn):
    """
    执行一个训练步骤
    
    Args:
        model: PyTorch 模型
        batch: 批次数据
        emb_dict: embedding 字典
        device: 设备
        generate_matrix_fn: 生成矩阵的函数
    
    Returns:
        loss: 损失值（标量）
    """
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
    
    # 前向传播
    loss, logits, output = model(matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                                matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                                matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                                strainPassCats=strainPassCats, labels=labels)
    
    return loss


def evaluate_step(model, dataloader, emb_dict, device, generate_matrix_fn):
    """
    评估模型
    
    Args:
        model: PyTorch 模型
        dataloader: 数据加载器
        emb_dict: embedding 字典
        device: 设备
        generate_matrix_fn: 生成矩阵的函数
    
    Returns:
        dict: 包含所有评估指标
    """
    try:
        from evaluation.metrics import print_exams
    except ImportError:
        import sys
        sys.path.append('../../../../src/fluprofiler')
        from evaluation.metrics import print_exams
    
    model.eval()
    prediction_ls = []
    reference_ls = []
    loss_ls = []
    
    with torch.no_grad():
        for batch in dataloader:
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
            
            # 前向传播
            loss, logits, output = model(matrices_a=matrixs_a, matrices_b=matrixs_b, matrices_c=matrixs_c, matrices_d=matrixs_d,
                                        matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                                        matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                                        strainPassCats=strainPassCats, labels=labels)
            
            loss_ls.append(loss.item())
            prediction_ls.extend(output.view(-1).tolist())
            reference_ls.extend(labels.tolist())
    
    # 计算指标
    mae, mse, pearson, spearman, r2 = print_exams(reference_ls, prediction_ls)
    
    return {
        'mae': mae,
        'mse': mse,
        'pearson': pearson.statistic,
        'spearman': spearman.statistic,
        'r2': r2,
        'loss': np.mean(loss_ls)
    }


def log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics=None):
    """
    记录指标到 TensorBoard
    
    Args:
        writer: SummaryWriter 对象
        epoch: 当前 epoch
        train_loss: 训练损失
        valid_metrics: 验证指标字典
        test_metrics: 测试指标字典（可选）
    """
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', valid_metrics['loss'], epoch)
    writer.add_scalar('MAE/valid', valid_metrics['mae'], epoch)
    writer.add_scalar('MSE/valid', valid_metrics['mse'], epoch)
    writer.add_scalar('Pearson/valid', valid_metrics['pearson'], epoch)
    writer.add_scalar('Spearman/valid', valid_metrics['spearman'], epoch)
    writer.add_scalar('R2/valid', valid_metrics['r2'], epoch)
    
    if test_metrics:
        writer.add_scalar('MAE/test', test_metrics['mae'], epoch)
        writer.add_scalar('MSE/test', test_metrics['mse'], epoch)
        writer.add_scalar('Pearson/test', test_metrics['pearson'], epoch)
        writer.add_scalar('Spearman/test', test_metrics['spearman'], epoch)
        writer.add_scalar('R2/test', test_metrics['r2'], epoch)


def log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics):
    """
    将 epoch 信息写入日志文件
    
    Args:
        log_path: 日志文件路径
        epoch: 当前 epoch（0-indexed）
        epochs: 总 epoch 数
        train_loss: 训练损失
        valid_metrics: 验证指标字典
        test_metrics: 测试指标字典
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(
            f"[{current_time}] Epoch {epoch + 1}/{epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"valid MAE: {valid_metrics['mae']:.5f}, valid MSE: {valid_metrics['mse']:.5f}, "
            f"valid Pearson: {valid_metrics['pearson']:.5f}, "
            f"valid Spearman: {valid_metrics['spearman']:.5f}, "
            f"valid R2: {valid_metrics['r2']:.5f}, "
            f"test MAE: {test_metrics['mae']:.5f}, test MSE: {test_metrics['mse']:.5f}, "
            f"test Pearson: {test_metrics['pearson']:.5f}, "
            f"test Spearman: {test_metrics['spearman']:.5f}, "
            f"test R2: {test_metrics['r2']:.5f}\n"
        )
 