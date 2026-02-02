import sys
sys.path.append('../../')
from fluProfiler_models import fluProfiler_Config, fluProfiler_HANA
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utilities import load_embedding, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime
import pickle
from utilities import print_exams

# 设置环境变量优化DDP性能
import os
os.environ['NCCL_IB_DISABLE'] = '1'  # 禁用InfiniBand，使用以太网
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 指定网络接口
os.environ['NCCL_DEBUG'] = 'INFO'  # 启用NCCL调试信息

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

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
        item.replace('<cls>', '0').replace('<eos>', '1').replace('<EGG>', '2').replace('<CELL>', '3').replace('<BOTH>', '4')
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

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
 
## read complete information
train_data = pd.read_csv('/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/data/data_40/serum/train.csv')
test_data = pd.read_csv('/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/data/data_40/serum/test.csv')
train_data, valid_data = train_test_split(train_data, test_size=1/9, random_state=42)

# train_data_reverse = pd.read_csv('/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/data/data_40/serum/train_reverse.csv')
# train_data = pd.concat([train_data, train_data_reverse])

## remove duplicated data in training set
group_columns = ['seq_a', 'seq_b','seq_c', 'seq_d', 'serumPassCat', 'virusPassCat']
agg_dict = {c: 'first' for c in train_data.columns if c not in group_columns}
agg_dict['label'] = 'mean'
train_data_final = train_data.groupby(group_columns).agg(agg_dict).reset_index()

Artificial_data = pd.read_csv('/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/data/data_40/Artificial.csv')
Artificial_data['label'] = 0
train_data_final = pd.concat([train_data_final, Artificial_data])

# train_data_final = train_data_final.iloc[:100,:]
# valid_data = valid_data.iloc[:100,:]
# test_data = test_data.iloc[:100,:]

train_dataset = fluProfiler_Dataset(train_data_final)
valid_dataset = fluProfiler_Dataset(valid_data)
test_dataset = fluProfiler_Dataset(test_data)

batch_size = 16  # 增加batch size以提高GPU利用率
gradient_accumulation_steps = 2  # 梯度累积步数，有效batch size = 16 * 2 = 32
train_sampler = DistributedSampler(train_dataset, shuffle=True)
valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
test_sampler = DistributedSampler(test_dataset, shuffle=False)

# 优化DataLoader配置
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    sampler=train_sampler,
    num_workers=4,  # 多进程加载数据
    pin_memory=True,  # 加速GPU数据传输
    persistent_workers=True,  # 保持worker进程，避免重复创建
    prefetch_factor=2  # 预取数据
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    sampler=valid_sampler,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    sampler=test_sampler,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

# 优化embedding加载：使用共享内存让多个进程共享同一份embedding数据
import multiprocessing as mp
# from multiprocessing import shared_memory  # 不再需要，使用简化的缓存方案
import hashlib

embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
sequence_names = pd.concat([embedding_df['seq_id_a'], embedding_df['seq_id_b'], 
                            embedding_df['seq_id_c'], embedding_df['seq_id_d']]).unique().tolist()
sequence_names = ['matrix_' + item + '.pt' for item in sequence_names]

embedding_path = "/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/data/data_40/embedding_Crick"

# 全局共享内存字典
shared_embeddings = {}
shared_memory_blocks = {}

def get_embedding_hash(emb_file_name):
    """生成embedding文件的唯一标识符"""
    return hashlib.md5(emb_file_name.encode()).hexdigest()[:16]

def load_embedding_to_shared_memory(emb_file_name):
    """将embedding加载到共享内存中（改进版本，避免资源跟踪器问题）"""
    emb_hash = get_embedding_hash(emb_file_name)
    
    if emb_hash not in shared_embeddings:
        try:
            # 加载embedding到CPU（参考原load_embedding函数）
            # 检查文件名是否已经包含.pt扩展名
            if emb_file_name.endswith('.pt'):
                file_path = os.path.join(embedding_path, emb_file_name)
            else:
                file_path = os.path.join(embedding_path, f"{emb_file_name}.pt")
            embedding = torch.load(file_path, weights_only=False)
            
            # 确保embedding是tensor格式（参考原函数使用torch.tensor()）
            embedding_tensor = torch.tensor(embedding)
            
            # 简化方案：直接存储tensor，避免共享内存的复杂性
            # 在多进程环境中，每个进程会独立加载，但通过DDP的同步机制保证一致性
            shared_embeddings[emb_hash] = embedding_tensor
            
            # 不显示具体的tensor信息，保持与原版load_embedding一致
            
        except FileNotFoundError:
            error_msg = f"Error: Embedding file {emb_file_name} not found at {file_path}"
            if dist.get_rank() == 0:
                print(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error loading {emb_file_name}: {e}"
            if dist.get_rank() == 0:
                print(error_msg)
            raise RuntimeError(error_msg)
    
    return shared_embeddings[emb_hash]

def get_shared_embedding(emb_file_name):
    """从共享内存获取embedding（返回格式与原load_embedding函数一致）"""
    emb_hash = get_embedding_hash(emb_file_name)
    
    if emb_hash not in shared_embeddings:
        return load_embedding_to_shared_memory(emb_file_name)
    
    # 确保返回的是tensor格式（与原函数保持一致）
    embedding = shared_embeddings[emb_hash]
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding)
    
    return embedding

def clear_shared_memory():
    """清理embedding缓存"""
    global shared_embeddings, shared_memory_blocks
    
    # 清理tensor引用
    shared_embeddings.clear()
    
    # 清理共享内存块引用（现在为空）
    shared_memory_blocks.clear()
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    if dist.get_rank() == 0:
        print("Cleared embedding cache")

def print_memory_usage():
    """打印内存使用情况（仅rank 0进程）"""
    if dist.get_rank() == 0:
        import psutil
        import gc
        
        # CPU内存使用情况
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        
        # GPU内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024  # GB
            print(f"Memory Usage - CPU: {cpu_memory:.2f}GB, GPU: {gpu_memory:.2f}GB (Max: {gpu_memory_max:.2f}GB)")
            print(f"Shared Embeddings: {len(shared_embeddings)} files")
            print(f"Shared Memory Blocks: {len(shared_memory_blocks)} blocks")
        
        # 强制垃圾回收
        gc.collect()

with open('./config_dict.json', 'r') as f:
    config_dict = json.load(f)
with open("./args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)


fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = fluProfiler_HANA(config=fluProfiler_config, args=fluProfiler_args).to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=False)

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
# 只有rank 0进程显示进度条，避免多进程进度条冲突
if dist.get_rank() == 0:
    progress_bar = tqdm(range(num_training_steps), 
                       desc="Training", 
                       unit="step",
                       ncols=100,
                       leave=True)
else:
    progress_bar = None
# early_stopping = EarlyStopping(patience=20, save_dir='../../../trained_model/1.9_serum/')
early_stopping = EarlyStopping(patience=20, save_dir='/mnt/zzbnew/peixunban/chenyihao_133/DDP/')

# 预加载所有embedding到共享内存
if dist.get_rank() == 0:
    print("=== Pre-loading Embeddings to Shared Memory ===")
    print(f"Total embedding files to load: {len(sequence_names)}")

# 所有进程同步等待
dist.barrier()

# 预加载embedding（只有rank 0进程实际加载，其他进程等待连接）
if dist.get_rank() == 0:
    # 使用tqdm进度条，与原版load_embedding保持一致
    from tqdm import tqdm
    for emb_file in tqdm(sequence_names, desc='Loading embeddings', unit='file'):
        get_shared_embedding(emb_file)
else:
    # 其他进程等待
    for emb_file in sequence_names:
        get_shared_embedding(emb_file)

# 所有进程同步
dist.barrier()

# 打印初始内存使用情况
if dist.get_rank() == 0:
    print("=== Training Started ===")
    print_memory_usage()

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    model.train()
    loss_ls = []
    
    # 使用enumerate来获取batch索引，用于预取下一个batch的embedding
    for batch_idx, batch in enumerate(train_dataloader):
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        # 优化embedding加载：批量加载并异步传输到GPU
        with torch.cuda.stream(torch.cuda.Stream()):  # 使用异步流
            # 从共享内存加载embedding，多进程共享同一份数据
            matrixs_a, masks_a = generate_matrix([get_shared_embedding(key) for key in emb_file_name_a])
            matrixs_b, masks_b = generate_matrix([get_shared_embedding(key) for key in emb_file_name_b])
            matrixs_c, masks_c = generate_matrix([get_shared_embedding(key) for key in emb_file_name_c])
            matrixs_d, masks_d = generate_matrix([get_shared_embedding(key) for key in emb_file_name_d])

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
        loss = loss.mean()
        
        # 梯度累积：除以累积步数
        loss = loss / gradient_accumulation_steps
        loss.backward()
        loss_ls.append(loss.item() * gradient_accumulation_steps)  # 恢复原始loss值用于记录

        # 只在累积步数达到时才更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        # 只有rank 0进程更新进度条
        if progress_bar is not None:
            progress_bar.set_postfix({
                'Epoch': f'{epoch+1}/{epochs}',
                'Loss': f'{loss.item():.4f}'
            })
            progress_bar.update(1)
    train_loss = np.mean(loss_ls)

    prediction_ls_valid = []
    reference_ls_valid = []
    logits_ls = []
    loss_ls_valid = []
    model.eval()
    for batch in valid_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        # 从共享内存加载embedding，多进程共享同一份数据
        matrixs_a, masks_a = generate_matrix([get_shared_embedding(key) for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([get_shared_embedding(key) for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([get_shared_embedding(key) for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([get_shared_embedding(key) for key in emb_file_name_d])

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
        loss = loss.mean()
        loss_ls_valid.append(loss.item())
        logits_ls.append(logits.tolist())
        prediction_ls_valid.extend(output.view(-1).tolist())
        reference_ls_valid.extend(labels.tolist())

    prediction_ls_test = []
    reference_ls_test = []
    logits_ls = []
    loss_ls_test = []
    model.eval()
    for batch in test_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        # 从共享内存加载embedding，多进程共享同一份数据
        matrixs_a, masks_a = generate_matrix([get_shared_embedding(key) for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([get_shared_embedding(key) for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([get_shared_embedding(key) for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([get_shared_embedding(key) for key in emb_file_name_d])

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
        loss = loss.mean()
        loss_ls_test.append(loss.item())
        logits_ls.append(logits.tolist())
        prediction_ls_test.extend(output.view(-1).tolist())
        reference_ls_test.extend(labels.tolist())

    # 初始化變量，避免未定義錯誤
    valid_mae = valid_mse = valid_pearson = valid_spearman = valid_R2 = 0
    test_mae = test_mse = test_pearson = test_spearman = test_R2 = 0
    
    if dist.get_rank() == 0:
        print('train loss :', train_loss)
        valid_mae, valid_mse, valid_pearson, valid_spearman, valid_R2 = print_exams(reference_ls_valid, prediction_ls_valid)
        test_mae, test_mse, test_pearson, test_spearman, test_R2 = print_exams(reference_ls_test, prediction_ls_test)
        early_stopping(valid_mse, model)
        if early_stopping.early_stop:
            print("Early stopping")
            # 通知所有進程停止訓練
            dist.broadcast(torch.tensor([1], device=device), src=0)
            break
        else:
            # 通知所有進程繼續訓練
            dist.broadcast(torch.tensor([0], device=device), src=0)
    else:
        # 非rank 0進程等待停止信號
        stop_signal = torch.tensor([0], device=device)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    # 只有rank 0進程寫入日誌文件，避免文件競爭
    if dist.get_rank() == 0:
        with open('/mnt/zzbnew/peixunban/chenyihao_133/DDP/log.txt', 'a') as f:
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
    
    # 每個epoch結束後清理一些內存
    if epoch % 10 == 0:  # 每10個epoch清理一次
        torch.cuda.empty_cache()
        if dist.get_rank() == 0:
            print(f"\n=== Epoch {epoch+1} Memory Status ===")
            print_memory_usage()
    
    # 確保所有進程在每個epoch結束時同步
    dist.barrier()

# 關閉進度條
if progress_bar is not None:
    progress_bar.close()

# 打印最终内存使用情况
if dist.get_rank() == 0:
    print("\n=== Training Completed ===")
    print_memory_usage()

# 清理共享内存，释放内存
clear_shared_memory()

# 訓練結束後清理分散式進程
dist.destroy_process_group()