import numpy as np
import pickle
import pandas as pd
import random
import sys
sys.path.append('../../')
import Adaboost_utilities
import ast
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def fasta_to_series(fasta_file):
    # 使用 BioPython 读取 FASTA 文件
    records = list(SeqIO.parse(fasta_file, "fasta"))
    
    # 创建一个字典，将序列 ID 映射到序列（包括 gap 字符）
    sequence_dict = {record.id: str(record.seq) for record in records}
    
    # 将字典转换为 pandas.Series
    sequence_series = pd.Series(sequence_dict)
    
    return sequence_series

def pair_representation(serumHA, virusHA, type, mutate_matrix):
    serumChar = [char for char in serumHA]
    virusChar = [char for char in virusHA]

    if len(serumChar) != len(virusChar):
        return [np.nan for _ in range(len(serumChar))]
    
    return diff_calculation(serumChar, virusChar, type=type, mutate_matrix=mutate_matrix)

def split_data_by_strain(dataframe, identity_cols, frac):
    strains = dataframe[identity_cols].drop_duplicates().reset_index(drop=True)
    test_strains = strains.sample(frac=frac, random_state=42).reset_index(drop=True)
    test_strains_set = set(zip(*test_strains[identity_cols].values.T))

    mask = dataframe[identity_cols].apply(lambda row: tuple(row) in test_strains_set, axis=1)
    test_data = dataframe[mask]
    train_data = dataframe[~mask]

    return train_data, test_data

def diff_calculation(SChar, VChar, type, mutate_matrix):
    difference = []
    for i in range(len(SChar)):
        if type == 'one-hot':
            diff = 0 if SChar[i] == VChar[i] else 1
        if type == 'mut-mat':
            if any(item not in mutate_matrix.columns for item in SChar[i] + VChar[i]):
                diff = 0
            else:
                mut_score = mutate_matrix.loc[VChar[i], VChar[i]]
                ori_score = mutate_matrix.loc[SChar[i], SChar[i]]
                cross_score = mutate_matrix.loc[SChar[i], VChar[i]]
                diff = mut_score + ori_score + cross_score
        difference.append(diff)
    return difference

def run_one_subtype(full_df, subtype, save_path=None, diff_col=None):
    sub_df = full_df[full_df["Type"] == subtype].copy()
   
    group_cols = ["serumName", "virusName", "serumHA", "virusHA", "serumPassCat", "virusPassCat"]
    agg = {c: "first" for c in sub_df.columns if c not in group_cols}
    agg["label"] = "mean"
    sub_df = sub_df.groupby(group_cols, as_index=False).agg(agg)

    meta = sub_df[meta_features].fillna('None').astype('str')

    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe = ohe.fit(meta)
    meta = ohe.transform(meta).toarray()
    seq_diff = np.array(sub_df[diff_col].tolist())[:,16:345]
    data_array = np.hstack((seq_diff, meta))

    nan_mask = pd.isna(data_array).any(axis=1)
    data_array = data_array[~nan_mask]

    X_train, X_valid = train_test_split(data_array, test_size=1/9, random_state=42)
    label_train, label_valid = train_test_split(sub_df['label'].loc[~nan_mask], test_size=1/9, random_state=42)

    print(X_train.shape)
    print(X_valid.shape)

    model_name    = 'AdaBoost'   # the type of model to be used
    model = getattr(Adaboost_utilities, f"model_{model_name}")

    results = model(X_train=np.vstack((X_train, X_valid)), y_train=pd.concat([label_train, label_valid]).tolist())
    
    results['Encoder'] = ohe
    with open(save_path, 'wb') as file:
        pickle.dump(results, file)
    return results

## meta feature for model training
meta_features = ['virusName',   # virus avidity (based on both name and passage)
                'serumName',   # antiserum potency (based on both name and passage)
                'virusPassCat',   # virus passage category
                'serumPassCat']   # serum passage category

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
diff_columns = 'seq_diff_ohe'
Crick_train = pd.read_csv('/home/chenyh/workspace/fluProfiler/data/reverse_test/processed/test_2024SH/train.csv', index_col=False)
diff_columns = 'seq_diff_ohe'
Crick_train = pd.read_csv('/home/chenyh/workspace/fluProfiler/data/reverse_test/processed/test_2024SH/train.csv', index_col=False)

# 如果 seq_diff_ohe 列不存在，则从 serumHA 和 virusHA 计算
if diff_columns not in Crick_train.columns:
    print(f"计算 {diff_columns} 列...")
    Crick_train[diff_columns] = Crick_train.apply(
        lambda row: pair_representation(row['serumHA'], row['virusHA'], type='one-hot', mutate_matrix=None),
        axis=1
    )
else:
    # 如果列存在，尝试解析（可能是字符串格式的列表）
    try:
        Crick_train[diff_columns] = Crick_train[diff_columns].apply(ast.literal_eval)
    except:
        pass  # 如果已经是列表格式，则跳过

run_one_subtype(full_df=Crick_train, subtype='H1N1',save_path='./Adaboost_H1N1_titer.pkl', diff_col=diff_columns)
run_one_subtype(full_df=Crick_train, subtype='H3N2',save_path='./Adaboost_H3N2_titer.pkl', diff_col=diff_columns)