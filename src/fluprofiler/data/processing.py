"""
Data processing utilities for fluProfiler.

Functions for data cleaning, normalization, and preprocessing.
"""

import pandas as pd
from datetime import datetime
from .loaders import DF_standardize_date, assign_season


def data_col_standardize(dataframe, ferret_dict_path=None):
    """
    Standardize column data types and formats.

    Args:
        dataframe: Input DataFrame
        ferret_dict_path: Path to ferret mapping dictionary (optional)

    Returns:
        Standardized DataFrame
    """
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


def ferret_standardize(ferret_name):
    """Standardize ferret names."""
    ferret_name = ferret_name.replace("NIBSC","NIB").replace("Nib","NIB")
    ferret_name = ferret_name.fillna('NA')
    ferret_name = ferret_name.str.replace('\*[0-9]+','', regex=True) # remove '*' in ferret names
    return ferret_name


def data_remove_duplicate(dataframe, group_columns=['serumHA', 'serumNA', 'virusHA', 'virusNA', 'serumPassCat', 'virusPassCat', 'ferret'],
                          new_columns=['serumType', 'season', 'serumPassCat', 'virusPassCat', 'ferret', 'serumHA',
                                       'serumNA', 'virusHA', 'virusNA', 'HI_Dist']):
    """
    Remove duplicates and aggregate data.

    Args:
        dataframe: Input DataFrame
        group_columns: Columns to group by for duplicate removal
        new_columns: Final columns to keep

    Returns:
        Deduplicated DataFrame
    """
    dataframe = dataframe.groupby(group_columns) \
        .agg({'serumType': 'first', 'season': lambda x: x.mode().iloc[0], 'season': 'first', 'HI_Dist': 'mean'}).reset_index()
    dataframe = dataframe[new_columns]
    return dataframe


def assign_seq_id(data):
    """
    Assign sequence IDs to unique sequences.

    Args:
        data: DataFrame with sequence columns

    Returns:
        DataFrame with sequence IDs added
    """
    unique_HA = pd.concat([data['seq_a'], data['seq_c']]).unique().tolist()
    unique_NA = pd.concat([data['seq_b'], data['seq_d']]).unique().tolist()
    map_dict_HA = {seq: 'HA_' + str(i) for i, seq in enumerate(unique_HA)}
    map_dict_NA = {seq: 'NA_' + str(i) for i, seq in enumerate(unique_NA)}

    data['seq_id_a'] = data['seq_a'].map(map_dict_HA)
    data['seq_id_c'] = data['seq_c'].map(map_dict_HA)
    data['seq_id_b'] = data['seq_b'].map(map_dict_NA)
    data['seq_id_d'] = data['seq_d'].map(map_dict_NA)

    return data


def split_data_by_strain(dataframe, identity_cols, frac):
    """
    Split data by strain for train/test sets.

    Args:
        dataframe: Input DataFrame
        identity_cols: Columns identifying unique strains
        frac: Fraction for test set

    Returns:
        train_data, test_data: Split DataFrames
    """
    strains = dataframe[identity_cols].drop_duplicates().reset_index(drop=True)
    test_strains = strains.sample(frac=frac, random_state=42).reset_index(drop=True)
    test_strains_set = set(zip(*test_strains[identity_cols].values.T))

    mask = dataframe[identity_cols].apply(
        lambda row: tuple(row) in test_strains_set, axis=1)
    test_data = dataframe[mask]
    train_data = dataframe[~mask]

    return train_data, test_data