import numpy as np
import pickle
import pandas as pd
import random
import Adaboost_utilities
import ast
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


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
Crick_train = pd.read_csv('../../../data/data_40/serum/train.csv', index_col=False)
Crick_train[diff_columns] = Crick_train[diff_columns].apply(ast.literal_eval)
run_one_subtype(full_df=Crick_train, subtype='H1N1',save_path='../../../trained_model/Adaboost/Adaboost_H1N1_serum_new.pkl', diff_col=diff_columns)
run_one_subtype(full_df=Crick_train, subtype='H3N2',save_path='../../../trained_model/Adaboost/Adaboost_H3N2_serum_new.pkl', diff_col=diff_columns)