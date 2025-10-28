import torch
import numpy as np
import pandas as pd
import json
from utilities import DF_standardize_date
from utilities import assign_season
from Bio import SeqIO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MEGA_Data:
    def __init__(self, padding_len, embedding_type, HA_embeddings, NA_embeddings, HA_IDs, NA_IDs, meta_embedder, meta_tokenizer):
        self.padding_len = padding_len
        self.embedding_type = embedding_type
        self.HA_embeddings = HA_embeddings
        self.NA_embeddings = NA_embeddings
        self.HA_IDs = HA_IDs
        self.NA_IDs = NA_IDs
        self.meta_embedder = meta_embedder
        self.meta_tokenizer = meta_tokenizer
    
    def generate_input(self, batch_ids):
        result = []
        batchSeqLenth_list = []
        for i in range(len(batch_ids[0])):
            ids = (batch_ids[0][i], batch_ids[1][i], batch_ids[2][i], batch_ids[3][i],
                   batch_ids[4][i], batch_ids[5][i], batch_ids[6][i])
            seq_length, integrated_emb = self.embedding_integrate(ids)
            batchSeqLenth_list.append(seq_length)
            result.append(integrated_emb)
            
        batch_input = torch.stack((result), dim=0)
        batch_mask = torch.vstack([self.generate_mask(seq_len) for seq_len in batchSeqLenth_list])
        
        return(batch_input, batch_mask)
    
    def generate_mask(self, seq_length):
        return(torch.hstack((torch.ones(1, seq_length), torch.zeros(1, self.padding_len-seq_length))))
    
    def embedding_integrate(self, info_tuple):
        if self.embedding_type == 'lucaone':
            serumHA_emb = self.HA_embeddings[info_tuple[0]]
            serumNA_emb = self.NA_embeddings[info_tuple[1]]
            virusHA_emb = self.HA_embeddings[info_tuple[2]]
            virusNA_emb = self.NA_embeddings[info_tuple[3]]
    
        elif self.embedding_type == 'esm2':
            serumHA_idx = self.HA_IDs.index(str(info_tuple[0]))
            serumNA_idx = self.NA_IDs.index(str(info_tuple[1]))
            virusHA_idx = self.HA_IDs.index(str(info_tuple[2]))
            virusNA_idx = self.NA_IDs.index(str(info_tuple[3]))
            serumHA_emb = self.HA_embeddings[serumHA_idx]
            serumNA_emb = self.NA_embeddings[serumNA_idx]
            virusHA_emb = self.HA_embeddings[virusHA_idx]
            virusNA_emb = self.NA_embeddings[virusNA_idx]

        meta_string = '<eos>' + info_tuple[4] + '<eos>' + info_tuple[5] + '<eos>' + info_tuple[6] + '<eos>'
        meta_emb = self.meta_embedder(torch.tensor(self.meta_tokenizer(meta_string, add_special_tokens=False)['input_ids'])).detach().numpy()

        inte_embedding_array = np.vstack((serumHA_emb,serumNA_emb,virusHA_emb,virusNA_emb,meta_emb))
        inte_embedding_tensor = torch.from_numpy(inte_embedding_array)
        seq_length = inte_embedding_tensor.shape[0]
        inte_embedding_tensor = torch.nn.functional.pad(inte_embedding_tensor, (0, 0, 0, self.padding_len-inte_embedding_tensor.shape[0]))

        return((seq_length, inte_embedding_tensor))
    
class ferret_tokenizer:
    def __init__(self, ferrets, json_path=None):
        self.ferrets = ferrets
        self.json_path = json_path
        if self.json_path is None:
            self.ferret_map_dict = self.generate_map_dict()
        else:
            self.ferret_map_dict = self.load_map_dict(self.json_path)

    def __call__(self, ferret_name):
        if ferret_name in self.ferret_map_dict:
            return(self.ferret_map_dict[ferret_name])
        else:
            return('<unkFerret>')
    
    def show_vocal(self):
        return(self.ferret_map_dict)
    
    def generate_map_dict(self):
        ferrets = self.ferrets.sort_values().unique()
        ferrets = ferrets[ferrets != 'NA'].tolist()
        ferrets.append('NA')
        IDs = ["<F{}>".format(i) for i in range(1, len(ferrets) + 1)]
        IDs.append('<unkFerret>')
        ferret_map_dict = dict(zip(ferrets, IDs))
        return(ferret_map_dict)
    
    def load_map_dict(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            map_dict = json.load(f)
        data_ferrets = self.ferrets.sort_values().unique().tolist()
        self.check_dict(data_ferrets, map_dict.keys())
        return(map_dict)
    
    def check_dict(self, data_ferrets, dict_ferrets):
        data_ferrets.append('NA')
        result_1 = all(item in dict_ferrets for item in data_ferrets)
        result_2 = all(item in data_ferrets for item in dict_ferrets)
        print('check all ferrets in ferrets dict:', result_1)
        print('check all dict ferrets in ferrets:', result_2)
        
    def write_map_dict(self, json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.ferret_map_dict, f, ensure_ascii=False, indent=4)
    
    
def ferret_standardize(ferret_name):
    ferret_name = ferret_name.replace("NIBSC","NIB").replace("Nib","NIB")
    ferret_name = ferret_name.fillna('NA')
    ferret_name = ferret_name.str.replace('\*[0-9]+','', regex=True) # remove '*' in ferret names
    return(ferret_name)

def data_col_standardize(dataframe, ferret_dict_path=None):
    if ferret_dict_path is None:
        # 1. Standardize and tokenize ferret
        dataframe['ferret'] = ferret_standardize(dataframe['ferret'])
        Ferret_tokenizer = ferret_tokenizer(ferrets=dataframe['ferret'], 
                                            json_path=ferret_dict_path)
        dataframe['ferret'] = [Ferret_tokenizer(ferret) for ferret in dataframe['ferret']]

    # 2. tokenize passage category
    dataframe['serumPassCat'] = dataframe['serumPassCat'].replace('EGG','<EGG>').replace('CELL','<CELL>').replace('BOTH', '<BOTH>')
    dataframe['virusPassCat'] = dataframe['virusPassCat'].replace('EGG','<EGG>').replace('CELL','<CELL>').replace('BOTH', '<BOTH>')
    
    # 3. standardize date and obtain flu season
    dataframe = DF_standardize_date(dataframe)
    dataframe = dataframe.assign(season=dataframe['virusDate'].apply(assign_season))
    # data.loc[:, 'season'] = data['virusDate'].apply(assign_season)
    return dataframe

def data_remove_duplicate(dataframe, group_columns=['serumHA', 'serumNA', 'virusHA', 'virusNA', 'serumPassCat', 'virusPassCat', 'ferret'],
                          new_columns=['serumType', 'season', 'serumPassCat', 'virusPassCat', 'ferret', 'serumHA', 
                                       'serumNA', 'virusHA', 'virusNA', 'HI_Dist']):
    dataframe = dataframe.groupby(group_columns) \
        .agg({'serumType': 'first', 'season': lambda x: x.mode().iloc[0], 'season': 'first', 'HI_Dist': 'mean'}).reset_index()
    dataframe = dataframe[new_columns]
    return dataframe

def fasta_to_dict(fasta_file):
    sequence_name = []
    sequence = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_name.append(record.id)
        sequence.append(str(record.seq))
    return(dict(zip(sequence_name, sequence)))

def generate_sequence_idx(seq_list, fasta_dict):
    sequence_idx = []
    for i in range(len(seq_list)):
        match_idx = [key for key, value in fasta_dict.items() if seq_list[i] == value]
        if len(match_idx) == 1:
            sequence_idx.append(match_idx[0])
        else:
            print(seq_list[i])
            raise ValueError("Sequence error")
    return(sequence_idx)

def check_fasta_match(fasta_dict, seq_list, seq_index_list):
    for i in range(len(seq_list)):
        sequence_in_list = seq_list[i]
        idx_in_fasta = str(seq_index_list[i])
        sequence_in_fasta = str(fasta_dict[idx_in_fasta])
        if sequence_in_list != sequence_in_fasta:
            print(i)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class PropertyDataset(Dataset):
    def __init__(self, dataset_seq, dataset_label) -> None:
        super().__init__()
        self.dataset_seq = dataset_seq
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.dataset_seq)
    
    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index]

class PropertyDataset(Dataset):
    def __init__(self, dataset_seq, dataset_label) -> None:
        super().__init__()
        self.dataset_seq = dataset_seq
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.dataset_seq)
    
    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index]

def generate_dataloader(dataframe, type, train_idx, valid_idx, test_idx, select_columns, batch_num=16):
    if(type == 'embedding'):
        sequences = [tuple(dataframe[select_columns].iloc[i, :]) for i in range(len(dataframe))]
        label = dataframe['HI_Dist'].to_list()
    elif(type == 'sequence'):
        sequences = dataframe[select_columns].agg("<eos>".join, axis=1)
        label = dataframe['HI_Dist'].to_list()

        train_seq_ids = [sequences[i] for i in train_idx]
        train_label = [label[i] for i in train_idx]
        valid_seq_ids = [sequences[i] for i in valid_idx]
        valid_label = [label[i] for i in valid_idx]
        test_seq_ids = [sequences[i] for i in test_idx]
        test_label = [label[i] for i in test_idx]
        
        # dataloader
        train_data = PropertyDataset(train_seq_ids, train_label)
        valid_data = PropertyDataset(valid_seq_ids, valid_label)
        test_data = PropertyDataset(test_seq_ids, test_label)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_num)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_num)
        test_dataloader = DataLoader(test_data, batch_size=batch_num)
        
    return train_dataloader, valid_dataloader, test_dataloader

def data_add_idx(dataframe, HA_fasta_path, NA_fasta_path):
    fasta_dict_HA = fasta_to_dict(HA_fasta_path)
    fasta_dict_NA = fasta_to_dict(NA_fasta_path)

    # generate sequence idx and add to data_unique
    serum_HA_idx = generate_sequence_idx(dataframe['serumHA'].to_list(), fasta_dict_HA)
    serum_NA_idx = generate_sequence_idx(dataframe['serumNA'].to_list(), fasta_dict_NA)
    virus_HA_idx = generate_sequence_idx(dataframe['virusHA'].to_list(), fasta_dict_HA)
    virus_NA_idx = generate_sequence_idx(dataframe['virusNA'].to_list(), fasta_dict_NA)
    dataframe['serumHA_idx'] = serum_HA_idx
    dataframe['serumNA_idx'] = serum_NA_idx
    dataframe['virusHA_idx'] = virus_HA_idx
    dataframe['virusNA_idx'] = virus_NA_idx
    dataframe = dataframe[['serumType', 'season', 'serumHA_idx', 'serumNA_idx', 'virusHA_idx', 'virusNA_idx',
                           'serumPassCat', 'virusPassCat', 'ferret','serumHA', 'serumNA', 'virusHA', 'virusNA', 'HI_Dist']]
    return dataframe