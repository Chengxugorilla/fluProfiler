import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import os
import pandas as pd
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

class fluProfiler_Dataset(Dataset):
    def __init__(self, DataFrame, add_special_token=True):
        self.emb_file_name_a = ('matrix_' + DataFrame['seq_id_a']).tolist()
        self.emb_file_name_b = ('matrix_' + DataFrame['seq_id_b']).tolist()
        self.emb_file_name_c = ('matrix_' + DataFrame['seq_id_c']).tolist()
        self.emb_file_name_d = ('matrix_' + DataFrame['seq_id_d']).tolist()

        if add_special_token:
            self.strainPassCats = convert_Pass2tensor(('<cls>' + DataFrame['serumPassCat'] + '<eos>' + DataFrame['virusPassCat'] + '<eos>').tolist())
        else:
            self.strainPassCats = convert_Pass2tensor((DataFrame['serumPassCat'] + DataFrame['virusPassCat']).tolist())

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


class GpuEmbeddingCache:
    """
    简单的基于 LRU 的 GPU 端 embedding 缓存。
    - cpu_store: 原始 CPU 端的 emb_dict（key -> Tensor on CPU）
    - device: 目标 GPU 设备
    - max_bytes: 缓存可使用的最大字节数（例如 28 * 1024**3）

    只缓存“纯数据”张量，不参与 autograd。
    """
    def __init__(self, cpu_store: dict, device: torch.device, max_bytes: int):
        self.cpu_store = cpu_store
        self.device = device
        self.max_bytes = int(max_bytes)

        self.gpu_store: dict = {}
        self.lru = OrderedDict()  # key -> None，占位即可
        self.current_bytes = 0

        # 统计信息
        self.hits = 0
        self.misses = 0
        self.copied_bytes = 0

    @staticmethod
    def _nbytes(t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def _touch(self, key: str):
        # 更新 LRU：最近访问的放到末尾
        if key in self.lru:
            self.lru.move_to_end(key)
        else:
            self.lru[key] = None

    def _evict_if_needed(self, need_bytes: int):
        # 如果一个 embedding 比整个缓存预算还大，就不要缓存，直接用一次
        if need_bytes > self.max_bytes:
            return
        while self.current_bytes + need_bytes > self.max_bytes and self.lru:
            old_key, _ = self.lru.popitem(last=False)
            old_tensor = self.gpu_store.pop(old_key, None)
            if old_tensor is not None:
                self.current_bytes -= self._nbytes(old_tensor)
                del old_tensor

    def get(self, key: str) -> torch.Tensor:
        """
        返回在 GPU 上的 embedding：
        - 命中：直接返回缓存中的 GPU Tensor
        - 未命中：从 CPU 端加载并迁移到 GPU，按需触发淘汰
        """
        if key in self.gpu_store:
            self.hits += 1
            tensor = self.gpu_store[key]
            self._touch(key)
            return tensor

        # miss：从 CPU 端迁移到 GPU
        self.misses += 1
        cpu_tensor = self.cpu_store[key]
        gpu_tensor = cpu_tensor.to(self.device, non_blocking=True).detach()

        need_bytes = self._nbytes(gpu_tensor)
        self.copied_bytes += need_bytes

        # 如果太大超过缓存预算，就不入缓存，直接返回一次性使用
        if need_bytes > self.max_bytes:
            return gpu_tensor

        self._evict_if_needed(need_bytes)
        self.gpu_store[key] = gpu_tensor
        self.current_bytes += need_bytes
        self._touch(key)
        return gpu_tensor


def generate_matrix_on_device(matrix_list, device=None):
    """
    在目标 device 上组装 batch：
    - 输入: 若干形状为 (L_i, E) 的 Tensor，已经在 device 上
    - 输出: 
        matrix: (B, max_len, E)
        mask:   (B, max_len) ，float32，1 为有效位，0 为 padding
    """
    if len(matrix_list) == 0:
        raise ValueError("matrix_list is empty.")

    first = matrix_list[0]
    if device is None:
        device = first.device

    lengths = torch.tensor([mat.shape[0] for mat in matrix_list], device=device, dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(matrix_list)
    embed_dim = first.shape[1]

    # 预分配 batch buffer
    matrix = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=first.dtype)
    for i, (mat, L) in enumerate(zip(matrix_list, lengths)):
        matrix[i, :int(L.item())] = mat

    # 向量化生成 mask
    arange = torch.arange(max_len, device=device).unsqueeze(0)      # (1, max_len)
    mask = (arange < lengths.unsqueeze(1)).to(dtype=torch.float32) # (B, max_len)

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


def load_data_and_dataloaders(data_path, season_path, batch_size=8, sample_limit=None, use_artificial=False, add_special_token=True, test_only=False):
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
    
    if test_only:
        test_data = pd.read_csv(data_path + season_path + 'test.csv')
        test_dataset = fluProfiler_Dataset(test_data, add_special_token=add_special_token)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        embedding_df = test_data
        sequence_names = pd.concat([embedding_df['seq_id_a'], embedding_df['seq_id_b'],
                               embedding_df['seq_id_c'], embedding_df['seq_id_d']]).unique().tolist()
        sequence_names = ['matrix_' + item + '.pt' for item in sequence_names]
        emb_dict = load_embedding(data_path + "/embedding", files=sequence_names)

        return {
            'test_dataloader': test_dataloader,
            'emb_dict': emb_dict
        }

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
    train_dataset = fluProfiler_Dataset(train_data_final, add_special_token=add_special_token)
    valid_dataset = fluProfiler_Dataset(valid_data, add_special_token=add_special_token)
    test_dataset = fluProfiler_Dataset(test_data, add_special_token=add_special_token)
    
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


def train_step_cached(model, batch, emb_dict, device, cache: GpuEmbeddingCache):
    """
    使用 GPU 端 embedding 缓存的训练步骤。
    - cache: GpuEmbeddingCache 实例，内部持有 CPU 端 emb_dict
    """
    emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

    # 直接从缓存中获取 GPU 上的单条序列 embedding
    matrices_a_list = [cache.get(key) for key in emb_file_name_a]
    matrices_b_list = [cache.get(key) for key in emb_file_name_b]
    matrices_c_list = [cache.get(key) for key in emb_file_name_c]
    matrices_d_list = [cache.get(key) for key in emb_file_name_d]

    # 在 GPU 上组装 batch
    matrixs_a, masks_a = generate_matrix_on_device(matrices_a_list, device=device)
    matrixs_b, masks_b = generate_matrix_on_device(matrices_b_list, device=device)
    matrixs_c, masks_c = generate_matrix_on_device(matrices_c_list, device=device)
    matrixs_d, masks_d = generate_matrix_on_device(matrices_d_list, device=device)

    strainPassCats = strainPassCats.to(device)
    labels = labels.to(device)

    loss, logits, output = model(
        matrices_a=matrixs_a, matrices_b=matrixs_b,
        matrices_c=matrixs_c, matrices_d=matrixs_d,
        matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
        matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
        strainPassCats=strainPassCats, labels=labels
    )

    return loss


def evaluate_step(model, dataloader, emb_dict, device, generate_matrix_fn, return_predictions=False):
    """
    评估模型
    
    Args:
        model: PyTorch 模型
        dataloader: 数据加载器
        emb_dict: embedding 字典
        device: 设备
        generate_matrix_fn: 生成矩阵的函数
    
    Returns:
        dict: 包含 loss、mae、mse、pearson、spearman、r2 等评估指标
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
    
    # 计算指标。scipy 在新版本里会返回 PearsonRResult / SignificanceResult，
    # 这里统一取相关系数本身，避免后续写 TensorBoard 时类型不兼容。
    mae, mse, pearson, spearman, r2 = print_exams(reference_ls, prediction_ls)
    pearson_value = float(getattr(pearson, "statistic", pearson[0]))
    spearman_value = float(getattr(spearman, "statistic", spearman[0]))
    metrics = {
        'loss': float(np.mean(loss_ls)) if loss_ls else 0.0,
        'mae': float(mae),
        'mse': float(mse),
        'pearson': pearson_value,
        'spearman': spearman_value,
        'r2': float(r2),
    }

    if return_predictions:
        metrics['predictions'] = prediction_ls
        metrics['references'] = reference_ls

    return metrics


def evaluate_step_cached(model, dataloader, emb_dict, device, cache: GpuEmbeddingCache, return_predictions=False):
    """
    使用 GPU 端 embedding 缓存的评估步骤。
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

            matrices_a_list = [cache.get(key) for key in emb_file_name_a]
            matrices_b_list = [cache.get(key) for key in emb_file_name_b]
            matrices_c_list = [cache.get(key) for key in emb_file_name_c]
            matrices_d_list = [cache.get(key) for key in emb_file_name_d]

            matrixs_a, masks_a = generate_matrix_on_device(matrices_a_list, device=device)
            matrixs_b, masks_b = generate_matrix_on_device(matrices_b_list, device=device)
            matrixs_c, masks_c = generate_matrix_on_device(matrices_c_list, device=device)
            matrixs_d, masks_d = generate_matrix_on_device(matrices_d_list, device=device)

            strainPassCats = strainPassCats.to(device)
            labels = labels.to(device)

            loss, logits, output = model(
                matrices_a=matrixs_a, matrices_b=matrixs_b,
                matrices_c=matrixs_c, matrices_d=matrixs_d,
                matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                strainPassCats=strainPassCats, labels=labels
            )

            loss_ls.append(loss.item())
            prediction_ls.extend(output.view(-1).tolist())
            reference_ls.extend(labels.tolist())

    mae, mse, pearson, spearman, r2 = print_exams(reference_ls, prediction_ls)
    pearson_value = float(getattr(pearson, "statistic", pearson[0]))
    spearman_value = float(getattr(spearman, "statistic", spearman[0]))
    metrics = {
        'loss': float(np.mean(loss_ls)) if loss_ls else 0.0,
        'mae': float(mae),
        'mse': float(mse),
        'pearson': pearson_value,
        'spearman': spearman_value,
        'r2': float(r2),
    }

    if return_predictions:
        metrics['predictions'] = prediction_ls
        metrics['references'] = reference_ls

    return metrics


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
 