import sys
sys.path.append('../../../')
from fluProfiler_models import fluProfiler_v0_1, fluProfiler_Config, fluProfiler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
from utilities import load_embedding, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datetime import datetime
import pickle
from utilities import print_exams

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

 
## read complete information
data_path = '../../../../data/reverse_test/'
season_path = 'processed/test_2024NH/'
model_save_path = './model_v0_1/'
train_data = pd.read_csv(data_path + season_path + 'train.csv')
test_data = pd.read_csv(data_path + season_path + 'test.csv')
train_data, valid_data = train_test_split(train_data, test_size=1/9, random_state=42)     

# Artificial_data = pd.read_csv(data_path + season_path + 'artificial_data.csv')
# train_data_final = pd.concat([train_data, Artificial_data])
train_data_final = train_data

train_dataset = fluProfiler_Dataset(train_data_final)
valid_dataset = fluProfiler_Dataset(valid_data)
test_dataset = fluProfiler_Dataset(test_data)

batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# load embedding
device = torch.device('cuda:6')
embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
sequence_names = pd.concat([embedding_df['seq_id_a'], embedding_df['seq_id_b'], 
                            embedding_df['seq_id_c'], embedding_df['seq_id_d']]).unique().tolist()
sequence_names = ['matrix_' + item + '.pt' for item in sequence_names]
emb_dict = load_embedding(data_path + "/embedding", files=sequence_names)

with open('./config_dict.json', 'r') as f:
    config_dict = json.load(f)
with open("./args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)


fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = fluProfiler_v0_1(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

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

progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=20, save_dir=model_save_path)

for epoch in range(epochs):
    model.train()
    loss_ls = []
    for batch in train_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        matrixs_a, masks_a = generate_matrix([emb_dict[key] for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([emb_dict[key] for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([emb_dict[key] for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([emb_dict[key] for key in emb_file_name_d])

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

        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    train_loss = np.mean(loss_ls)
    print('train loss :', train_loss)

    prediction_ls_valid = []
    reference_ls_valid = []
    logits_ls = []
    loss_ls_valid = []
    model.eval()
    for batch in valid_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        matrixs_a, masks_a = generate_matrix([emb_dict[key] for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([emb_dict[key] for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([emb_dict[key] for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([emb_dict[key] for key in emb_file_name_d])

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

        loss_ls_valid.append(loss.item())
        logits_ls.append(logits.tolist())
        prediction_ls_valid.extend(output.view(-1).tolist())
        reference_ls_valid.extend(labels.tolist())

    valid_mae, valid_mse, valid_pearson, valid_spearman, valid_R2 = print_exams(reference_ls_valid, prediction_ls_valid)

    early_stopping(valid_mse, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    prediction_ls_test = []
    reference_ls_test = []
    logits_ls = []
    loss_ls_test = []
    model.eval()
    for batch in test_dataloader:
        emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

        matrixs_a, masks_a = generate_matrix([emb_dict[key] for key in emb_file_name_a])
        matrixs_b, masks_b = generate_matrix([emb_dict[key] for key in emb_file_name_b])
        matrixs_c, masks_c = generate_matrix([emb_dict[key] for key in emb_file_name_c])
        matrixs_d, masks_d = generate_matrix([emb_dict[key] for key in emb_file_name_d])

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

        loss_ls_test.append(loss.item())
        logits_ls.append(logits.tolist())
        prediction_ls_test.extend(output.view(-1).tolist())
        reference_ls_test.extend(labels.tolist())

    test_mae, test_mse, test_pearson, test_spearman, test_R2 = print_exams(reference_ls_test, prediction_ls_test)

    # 将epoch信息写入log.txt
    with open(model_save_path + 'log.txt', 'a') as f:
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
        