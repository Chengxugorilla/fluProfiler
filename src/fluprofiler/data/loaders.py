"""
Data loading utilities for fluProfiler.

Functions for loading embeddings, sequences, and other data formats.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader


def DF_standardize_date(data):
    """
    Standardize date formats in DataFrame.

    Args:
        data: DataFrame with date columns

    Returns:
        DataFrame with standardized dates
    """
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


def assign_season(date):
    """
    Assign flu season based on date.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        Season string (e.g., '2023NH', '2023SH')
    """
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


def load_embedding(path, files=None, map_location=None):
    """
    Load embeddings from .pt files.

    Args:
        path: Directory containing .pt files
        files: List of specific files to load (optional)
        map_location: Device to map tensors to

    Returns:
        Dictionary mapping IDs to embeddings
    """
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


def write_unique_fasta(seqs_list, filename):
    """
    Write unique sequences to FASTA file.

    Args:
        seqs_list: List of sequences
        filename: Output FASTA filename

    Returns:
        List of indices mapping original sequences to unique ones
    """
    seq_unique = list(set(seqs_list))
    index_list = []
    for i in range(len(seqs_list)):
        index_list.append(seq_unique.index(seqs_list[i]))
    write_fasta(seq_unique, filename)
    return index_list


def write_fasta(seqs_list, filename):
    """
    Write sequences to FASTA file.

    Args:
        seqs_list: List of sequences
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for i in range(len(seqs_list)):
            f.write(f">{i}\n{seqs_list[i]}\n")


def fasta_to_dict(fasta_file):
    """
    Convert FASTA file to dictionary.

    Args:
        fasta_file: Path to FASTA file

    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequence_name = []
    sequence = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_name.append(record.id)
        sequence.append(str(record.seq))
    return(dict(zip(sequence_name, sequence)))


def generate_sequence_idx(seq_list, fasta_dict):
    """
    Generate sequence indices based on FASTA dictionary.

    Args:
        seq_list: List of sequences
        fasta_dict: FASTA dictionary from fasta_to_dict()

    Returns:
        List of sequence indices
    """
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
    """
    Check if sequences match FASTA dictionary entries.

    Args:
        fasta_dict: FASTA dictionary
        seq_list: List of sequences
        seq_index_list: List of sequence indices
    """
    for i in range(len(seq_list)):
        sequence_in_list = seq_list[i]
        idx_in_fasta = str(seq_index_list[i])
        sequence_in_fasta = str(fasta_dict[idx_in_fasta])
        if sequence_in_list != sequence_in_fasta:
            print(i)


def expand_sequences(isolates):
    """
    Expand sequences from single string to list of amino acids.

    Args:
        isolates: DataFrame with sequence column

    Returns:
        DataFrame with expanded sequences
    """
    sequence_list = [list(seq) for seq in isolates.sequence]
    sequences_expand = pd.DataFrame(sequence_list, index=isolates.index)

    return sequences_expand


class ferret_tokenizer:
    """
    Tokenizer for ferret names.
    """
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


class PropertyDataset(Dataset):
    """
    Dataset for property prediction tasks.
    """
    def __init__(self, dataset_seq, dataset_label) -> None:
        super().__init__()
        self.dataset_seq = dataset_seq
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.dataset_seq)

    def __getitem__(self, index):
        return self.dataset_seq[index], self.dataset_label[index]


def generate_dataloader(dataframe, type, train_idx, valid_idx, test_idx, select_columns, batch_num=16):
    """
    Generate DataLoader objects for train/validation/test sets.

    Args:
        dataframe: Input DataFrame
        type: Data type ('embedding' or 'sequence')
        train_idx, valid_idx, test_idx: Indices for splits
        select_columns: Columns to select
        batch_num: Batch size

    Returns:
        train_dataloader, valid_dataloader, test_dataloader
    """
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
    """
    Add sequence indices to dataframe based on FASTA files.

    Args:
        dataframe: Input DataFrame
        HA_fasta_path: Path to HA FASTA file
        NA_fasta_path: Path to NA FASTA file

    Returns:
        DataFrame with sequence indices added
    """
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