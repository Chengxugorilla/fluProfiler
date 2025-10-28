import sys
sys.path.append('../../')
from tqdm import tqdm
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from utilities import EarlyStopping, print_exams
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from bio_tokenizer import BioTokenizer
from transformers import MegaConfig, MegaForSequenceClassification
from MEGA_utilities import count_parameters
from tqdm import tqdm

class HANADataset(Dataset):
    def __init__(self, DataFrame):
        self.sequence = (DataFrame['seq_a'] + '<eos>' + DataFrame['seq_b'] + '<eos>' + DataFrame['seq_c'] + '<eos>' + DataFrame['seq_d'] + \
                         '<eos>' + DataFrame['serumPassCat'] + '<eos>' + DataFrame['virusPassCat']).tolist()
        self.labels = torch.tensor(DataFrame['label'].tolist())
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequence[idx], self.labels[idx]

save_dir = '/mnt/zzbnew/peixunban/chenyihao/fluProfiler_source/trained_model/1.3_H1N1_only_model/'
## read complete information
train_data = pd.read_csv('../../../data/MEGA_data/train_data.csv')
test_data = pd.read_csv('../../../data/MEGA_data/test_data.csv')
train_data, valid_data = train_test_split(train_data, test_size=1/9, random_state=42)

## remove duplicated row and mean HI_Dist
HANA_data_filt1 = train_data.groupby(['seq_a', 'seq_b', 'seq_c', 'seq_d', 'serumPassCat', 'virusPassCat']) \
        .agg({'serumName': 'first', 'virusName': 'first', 'Type': 'first', 'label': 'mean'}) \
        .reset_index()
## remove PassCat = 'BOTH'
HANA_data_filt2 = HANA_data_filt1[(HANA_data_filt1['serumPassCat'] != 'BOTH') &
                              (HANA_data_filt1['virusPassCat'] != 'BOTH')].reset_index(drop=True)
## replace PassCat to special token
HANA_data_filt3 = HANA_data_filt2.replace({'serumPassCat': {'EGG': '<EGG>', 'CELL': '<CELL>'},
                                       'virusPassCat': {'EGG': '<EGG>', 'CELL': '<CELL>'}})

train_data_final = HANA_data_filt3.copy()

train_data_final = train_data_final[train_data_final['Type'] == 'H1N1']
valid_data = valid_data[valid_data['Type'] == 'H1N1']
test_data = test_data[test_data['Type'] == 'H1N1']

with open(save_dir + 'log.txt', 'a') as f:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(
        f"[trian data size: {len(train_data_final)}, valid data size: {len(valid_data)}, test data size: {len(test_data)}]\n")

train_dataset = HANADataset(train_data_final)
valid_dataset = HANADataset(valid_data)
test_dataset = HANADataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# get tokenizer and model
tokenizer = BioTokenizer(vocab_file='./vocab_AA.txt')

# update the num_vocab and num_label
config = MegaConfig()
config.num_labels=1
config.vocab_size=28
config.max_positions=4000
config.num_attention_heads=4
config.num_hidden_layers=5

device = torch.device("cuda:0")
model = MegaForSequenceClassification(config)
model = model.to(device)

# 计数：DP时用 .module
print("Number of parameters: %e" % count_parameters(model))

# optimizer
optimizer = AdamW(model.parameters(), lr=5e-4)

# scheduler
num_epochs = 160
num_training_steps = num_epochs * len(train_loader)


progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=20, save_dir=save_dir)
# Training loop
for epoch in range(num_epochs):
    model.train()
    loss_ls = []
    for batch_seq, batch_label in train_loader:
        batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt").to(device)
        batch_label = batch_label.to(device)

        outputs = model(**batch_input, labels=batch_label)

        loss = outputs.loss
        loss_ls.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    train_loss = sum(loss_ls) / len(loss_ls)
    print('train loss :', train_loss)

    prediction_ls_valid = []
    reference_ls_valid = []
    logits_ls_valid = []
    loss_ls_valid = []
    model.eval()
    with torch.no_grad():
        for batch_seq, batch_label in valid_loader:
            batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt").to(device)
            batch_label = batch_label.to(device)

            outputs = model(**batch_input, labels=batch_label)
            logits = outputs.logits
            loss = outputs.loss

            logits_ls_valid.append(logits)
            loss_ls_valid.append(loss.item())
            prediction_ls_valid.extend(logits.squeeze().tolist())
            reference_ls_valid += batch_label.tolist()
    valid_mae, valid_mse, valid_pearson, valid_spearman, valid_R2 = print_exams(reference_ls_valid, prediction_ls_valid, print_result=False)

    early_stopping(valid_mse, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    prediction_ls_test = []
    reference_ls_test = []
    logits_ls_test = []
    loss_ls_test = []
    model.eval()
    with torch.no_grad():
        for batch_seq, batch_label in test_loader:
            batch_input = tokenizer(batch_seq, padding='longest', return_tensors="pt").to(device)
            batch_label = batch_label.to(device)

            outputs = model(**batch_input, labels=batch_label)
            logits = outputs.logits
            loss = outputs.loss

            logits_ls_test.append(logits)
            loss_ls_test.append(loss.item())
            prediction_ls_test.extend(logits.squeeze().tolist())
            reference_ls_test.extend(batch_label.tolist())
    
    test_mae, test_mse, test_pearson, test_spearman, test_R2 = print_exams(reference_ls_test, prediction_ls_test, print_result=False)


    ## 将epoch信息写入log.txt
    with open(save_dir + 'log.txt', 'a') as f:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"[{current_time}] Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.5f}, valid MAE: {valid_mae:.5f}, valid MSE: {valid_mse:.5f}, "
            f"valid Pearson: {valid_pearson.statistic:.5f}, valid Spearman: {valid_spearman.statistic:.5f}, valid R2: {valid_R2:.5f}, test MAE: {test_mae:.5f}, test MSE: {test_mse:.5f}, "
            f"test Pearson: {test_pearson.statistic:.5f}, test Spearman: {test_spearman.statistic:.5f}, test R2: {test_R2:.5f}\n")