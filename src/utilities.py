from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import torch
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def assign_season(date):
    # if less than 31st January, then NH season
    if date[5:] <= "01-31":
        return date[:4] + "NH"
    # if between 1st February and 31st August, then SH
    elif (date[5:] >= "02-01") and (date[5:] <= "08-31"):
        return date[:4] + "SH"
    # if between 1st September and 31st December, then next year's NH
    elif (date[5:] >= "09-01") and (date[5:] <= "12-31"):
        season_year = int(date[:4])
        return str(season_year+1) + "NH"
    else:
        return None

# def DF_standardize_date(data):
#     '''
#     turn character date
#     '''
#     # turn character date to datetime date
#     data['serumDate'] = pd.to_datetime(data['serumDate'])
#     data['serumDate'] = data['serumDate'].dt.strftime('%Y-%m-%d')
#     data['virusDate'] = pd.to_datetime(data['virusDate'])
#     data['virusDate'] = data['virusDate'].dt.strftime('%Y-%m-%d')
#     # remove record without date
#     data = data.dropna(subset=['virusDate'])
#     return data
def DF_standardize_date(data):
    '''
    turn character date
    '''
    # turn character date to datetime date
    data['serumDate'] = pd.to_datetime(
        data['serumDate'], format='%Y-%m-%d', errors='coerce')
    data['virusDate'] = pd.to_datetime(
        data['virusDate'], format='%Y-%m-%d', errors='coerce')

    try:
        data['serumDate'] = data['serumDate'].dt.strftime('%Y-%m-%d')
    except:
        date = data['serumDate'].dt.strftime('%Y-%m')
        date.replace(day=1)
        data['serumDate'] = date

    try:
        data['virusDate'] = data['virusDate'].dt.strftime('%Y-%m-%d')
    except:
        date = data['virusDate'].dt.strftime('%Y-%m')
        date.replace(day=1)
        data['virusDate'] = date

    # remove record without date
    data = data.dropna(subset=['virusDate'])

    return data

def print_exams(Prediction, Observation, print_result=True):
    P = Prediction
    O = Observation
    MAE, MSE, Pearson, Spearman, R2 = mean_absolute_error(O, P), mean_squared_error(O, P), pearsonr(O, P), spearmanr(O, P), r2_score(O, P)
    if print_result:
        print(f"MAE: {MAE:.5f}\nMSE: {MSE:.5f}\npearson correlation: {Pearson[0]:.5f}\nspearman correlation: {Spearman[0]:.5f}\nR2_score: {R2:.5f}")

    return MAE, MSE, Pearson, Spearman, R2

def data_split(data, type, year, split_ratio=[0.8, 0.1, 0.1]):
    if type == 'seasonal':
        start_year = year[0]
        middel_year = year[1]
        end_year = year[2]

        train_season = [
            str(year)+s for year in range(start_year, middel_year) for s in ['NH', 'SH']]
        test_season = [str(year)+s for year in range(middel_year, end_year)
                       for s in ['NH', 'SH']]
        train_data = data.loc[data['season'].isin(train_season)]
        train_idx = train_data.index.tolist()
        valid_idx = train_data.groupby('season').filter(lambda x: len(x) >= 100).groupby('season') \
            .apply(lambda x: x.sample(frac=0.1)).index.get_level_values(1).tolist()
        test_data = data.loc[data['season'].isin(test_season)]
        test_idx = test_data.index.tolist()
    elif type == 'random':
        if sum(split_ratio) != 1:
            raise ValueError('Invalid split ratio')
        index_range = list(range(data.shape[0]))
        train_idx, test_idx = train_test_split(
            index_range, test_size=split_ratio[2], random_state=523)
        train_idx, valid_idx = train_test_split(
            train_idx, test_size=split_ratio[1], random_state=523)
    else:
        raise ValueError('Invalid type')
    return train_idx, valid_idx, test_idx

def expand_sequences(isolates):
    sequence_list = [list(seq) for seq in isolates.sequence]
    sequences_expand = pd.DataFrame(sequence_list, index=isolates.index)

    return sequences_expand

def write_unique_fasta(seqs_list, filename):
    '''
    Args:
        seqs_list: list of sequences of which each element is a string of amino acids
        filename: the path to write the fasta file which removes the duplicate sequences. 
        The name of the sequence in fasta file is the index of the sequence similar to the returned index_list
    Returns:
        index_list: list of indices of the unique sequences
    '''
    seq_unique = list(set(seqs_list))
    index_list = []
    for i in range(len(seqs_list)):
        index_list.append(seq_unique.index(seqs_list[i]))
    write_fasta(seq_unique, filename)
    return index_list

def write_fasta(seqs_list, filename):
    '''
    write the sequences to a fasta file with the name of the sequence as the index of the seqs_list
    '''
    with open(filename, 'w') as f:
        for i in range(len(seqs_list)):
            f.write(f">{i}\n{seqs_list[i]}\n")

def load_embedding(path, files=None, map_location=None):
    if files is None:
        files = os.listdir(path)

    IDs = []
    embeddings = []
    for pt_file in tqdm(files, desc='Loading tensor', unit='file'):
        file_path = os.path.join(path, pt_file)
        embedding = torch.load(file_path, weights_only=False)
        IDs.append(pt_file.split('.')[0])
        embeddings.append(torch.tensor(embedding))

    if map_location is not None:
        embeddings = [emb.to(map_location) for emb in embeddings]

    emb_dict = dict(zip(IDs, embeddings))
    return emb_dict

# def load_embedding(path, files=None, device=None):
#     """
#     从给定目录加载 .pt 文件（内部内容为 numpy.ndarray），
#     转成 torch.Tensor 后返回 {ID: tensor} 字典。

#     参数
#     ----
#     path : str
#         存放 .pt 文件的目录
#     files : list[str] 或 None
#         指定要加载的文件名列表（只写文件名，不含路径）。
#         如果为 None，则自动扫描 path 下所有 .pt 文件。
#     device : str 或 torch.device 或 None
#         想把 tensor 放到的设备，如 'cpu'、'cuda:0'。为 None 则不搬运。
#     """

#     # 自动发现 .pt 文件
#     if files is None:
#         files = [f for f in os.listdir(path) if f.endswith('.pt')]

#     emb_dict = {}

#     for pt_file in tqdm(files, desc='Loading embeddings', unit='file'):
#         file_path = os.path.join(path, pt_file)

#         arr = torch.load(file_path, weights_only=False)  # 这里读出来是 numpy.ndarray

#         if not isinstance(arr, np.ndarray):
#             raise TypeError(
#                 f"File {pt_file} does not contain a numpy.ndarray, "
#                 f"got {type(arr)} instead."
#             )

#         # 零拷贝从 numpy 转 tensor（共享内存，比 torch.tensor(arr) 高效）
#         tensor = torch.from_numpy(arr)

#         if device is not None:
#             tensor = tensor.to(device)

#         # 去掉后缀作为 ID，例如 matrix_H1_1.pt -> matrix_H1_1
#         emb_id = pt_file.rsplit('.', 1)[0]
#         emb_dict[emb_id] = tensor

#     return emb_dict

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, trace_func=print, save_dir=None):
        """
        初始化早停类

        参数:
            patience (int): 在早停之前允许性能没有改善的epochs数量
            verbose (bool): 如果为True，则打印一条消息，每次更新时
            delta (float): 最小变化阈值以认定为改善
            path (str): 保存模型的文件路径
            trace_func (function): 用于输出消息的函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        调用早停逻辑检查

        参数:
            val_loss (float): 当前验证集的MSE loss
            model (torch.nn.Module): 需要保存的PyTorch模型
        """
        if self.best_score > val_loss:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_score = val_loss
        else:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """
        保存模型当验证集召回率提升时

        参数:
            val_recall (float): 当前验证集的召回率
            model (torch.nn.Module): 需要保存的PyTorch模型
        """
        if self.verbose:
            self.trace_func(
                f'Validation MSE decrease ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')

        current_time = datetime.now()

        # 格式化当前时间为字符串，例如：2024-06-09_12-34-56
        checkpoint_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(model, self.save_dir + checkpoint_name + '.pth')

def convert_Pass2tensor(pass_cats):
    result = [
        item.replace('<cls>', '0').replace('<eos>', '1').replace(
            '<EGG>', '2').replace('<CELL>', '3').replace('<BOTH>', '4')
        for item in pass_cats
    ]
    result = torch.tensor(
        [[int(number) for number in [char for char in item]] for item in result])
    return result
def generate_matrix(matrix_list):
    seq_len = [mat.shape[0] for mat in matrix_list]
    max_len = max(seq_len)
    mask_list = []
    for i in range(len(matrix_list)): 
        matrix_list[i] = F.pad(matrix_list[i], (0, 0, 0, max_len - seq_len[i]))
        mask = torch.concat((torch.ones(1, seq_len[i]), torch.zeros(1, max_len-seq_len[i])), axis=1)
        mask_list.append(mask)
    matrix = torch.stack(matrix_list)
    mask = torch.stack(mask_list).view(len(matrix_list), max_len)
    return matrix, mask

def split_data_by_strain(dataframe, identity_cols, frac):
    strains = dataframe[identity_cols].drop_duplicates().reset_index(drop=True)
    test_strains = strains.sample(frac=frac, random_state=42).reset_index(drop=True)
    test_strains_set = set(zip(*test_strains[identity_cols].values.T))

    mask = dataframe[identity_cols].apply(
        lambda row: tuple(row) in test_strains_set, axis=1)
    test_data = dataframe[mask]
    train_data = dataframe[~mask]

    return train_data, test_data

def assign_seq_id(data):
    unique_HA = pd.concat([data['seq_a'], data['seq_c']]).unique().tolist()
    unique_NA = pd.concat([data['seq_b'], data['seq_d']]).unique().tolist()
    map_dict_HA = {seq: 'HA_' + str(i) for i, seq in enumerate(unique_HA)}
    map_dict_NA = {seq: 'NA_' + str(i) for i, seq in enumerate(unique_NA)}
    
    data['seq_id_a'] = data['seq_a'].map(map_dict_HA)
    data['seq_id_c'] = data['seq_c'].map(map_dict_HA)
    data['seq_id_b'] = data['seq_b'].map(map_dict_NA)
    data['seq_id_d'] = data['seq_d'].map(map_dict_NA)

    return data

