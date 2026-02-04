from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# 依赖你现有工程（确保 repo_root/src 已加入 sys.path）
from fluProfiler_models import fluProfiler_v0_1, fluProfiler_Config
from utilities import load_embedding, EarlyStopping, print_exams
from fluprofiler.utils.run_logger import RunLogger


class fluProfiler_Dataset(Dataset):
    def __init__(self, DataFrame: pd.DataFrame):
        self.emb_file_name_a = ('matrix_' + DataFrame['seq_id_a']).tolist()
        self.emb_file_name_b = ('matrix_' + DataFrame['seq_id_b']).tolist()
        self.emb_file_name_c = ('matrix_' + DataFrame['seq_id_c']).tolist()
        self.emb_file_name_d = ('matrix_' + DataFrame['seq_id_d']).tolist()

        self.strainPassCats = convert_Pass2tensor(
            ('<cls>' + DataFrame['serumPassCat'] + '<eos>' + DataFrame['virusPassCat'] + '<eos>').tolist()
        )
        self.labels = torch.tensor(DataFrame['label'].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.emb_file_name_a[idx], self.emb_file_name_b[idx],
                self.emb_file_name_c[idx], self.emb_file_name_d[idx],
                self.strainPassCats[idx], self.labels[idx])


def convert_Pass2tensor(pass_cats):
    result = [
        item.replace('<cls>', '0').replace('<eos>', '1')
            .replace('<EGG>', '2').replace('<CELL>', '3')
            .replace('<BOTH>', '4').replace('<NONE>', '5')
        for item in pass_cats
    ]
    return torch.tensor([[int(ch) for ch in item] for item in result], dtype=torch.long)


def generate_matrix(matrix_list):
    seq_len = [mat.shape[0] for mat in matrix_list]
    max_len = max(seq_len)
    mask_list = []
    for i in range(len(matrix_list)):
        matrix_list[i] = F.pad(matrix_list[i], (0, 0, 0, max_len - seq_len[i]))
        mask = torch.concat((torch.ones(1, seq_len[i]), torch.zeros(1, max_len - seq_len[i])), axis=1)
        mask_list.append(mask)
    matrix = torch.stack(matrix_list)
    mask = torch.stack(mask_list).view(len(matrix_list), max_len)
    return matrix, mask


def run_v0_1_from_csv(
    train_csv: str,
    test_csv: str,
    embedding_dir: str,
    config_json: str,
    args_pkl: str,
    run_root: str = "runs",
    run_name: str = "v0_1",
    run_tag: str = "",
    device: str = "cuda:0",
    batch_size: int = 8,
    epochs: int = 200,
    lr: float = 0.00008,
    val_frac: float = 1/9,
    split_seed: int = 42,
    patience: int = 20,
) -> str:
    device_t = torch.device(device)

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    train_data, valid_data = train_test_split(train_data, test_size=val_frac, random_state=split_seed)
    train_data_final = train_data

    with open(config_json, "r") as f:
        config_dict = json.load(f)
    with open(args_pkl, "rb") as f:
        fluProfiler_args = pickle.load(f)

    run_config: Dict[str, Any] = {
        "run_name": run_name,
        "run_tag": run_tag,
        "device": str(device_t),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "patience": patience,
        "data": {"train_csv": train_csv, "test_csv": test_csv, "val_frac": val_frac, "split_seed": split_seed},
        "embedding_dir": embedding_dir,
        "config_json": config_json,
        "args_pkl": args_pkl,
        "config_dict": config_dict,
        "args_dict": vars(fluProfiler_args) if hasattr(fluProfiler_args, "__dict__") else str(fluProfiler_args),
    }

    logger = RunLogger.create(run_root=run_root, name=run_name, tag=run_tag, config=run_config, repo_dir=Path(run_root).resolve().parent)
    (logger.run_dir / "artifacts").mkdir(exist_ok=True)
    train_data_final.to_csv(logger.run_dir / "artifacts" / "train_used.csv", index=False)
    valid_data.to_csv(logger.run_dir / "artifacts" / "valid_used.csv", index=False)
    test_data.to_csv(logger.run_dir / "artifacts" / "test_used.csv", index=False)

    train_data_final = train_data_final.iloc[:200]
    valid_data = valid_data.iloc[:200]
    test_data = test_data.iloc[:200]

    train_dataset = fluProfiler_Dataset(train_data_final)
    valid_dataset = fluProfiler_Dataset(valid_data)
    test_dataset = fluProfiler_Dataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # embedding（保持你原来的全量加载）
    embedding_df = pd.concat([train_data_final, valid_data, test_data], axis=0)
    seqs = pd.concat([
        embedding_df['seq_id_a'], embedding_df['seq_id_b'],
        embedding_df['seq_id_c'], embedding_df['seq_id_d']
    ]).unique().tolist()
    sequence_names = ['matrix_' + item + '.pt' for item in seqs]
    emb_dict = load_embedding(embedding_dir, files=sequence_names)

    fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
    model = fluProfiler_v0_1(config=fluProfiler_config, args=fluProfiler_args).to(device_t)

    no_decay = ["bias", "layernorm.weight", "layer_norm.weight", "layer.norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": fluProfiler_args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        betas=[fluProfiler_args.beta1 if fluProfiler_args.beta1 > 0 else 0.9,
               fluProfiler_args.beta2 if fluProfiler_args.beta2 > 0 else 0.98],
        eps=fluProfiler_args.adam_epsilon
    )

    num_training_steps = len(train_dataloader) * epochs
    progress_bar = tqdm(range(num_training_steps))

    ckpt_dir = logger.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=patience, save_dir=str(ckpt_dir) + "/")


    best_valid_mse = float("inf")

    for epoch in range(epochs):
        # --- train ---
        model.train()
        loss_ls = []
        for batch in train_dataloader:
            emb_a, emb_b, emb_c, emb_d, passcats, labels = batch

            mats_a, masks_a = generate_matrix([emb_dict[k] for k in emb_a])
            mats_b, masks_b = generate_matrix([emb_dict[k] for k in emb_b])
            mats_c, masks_c = generate_matrix([emb_dict[k] for k in emb_c])
            mats_d, masks_d = generate_matrix([emb_dict[k] for k in emb_d])

            mats_a, mats_b, mats_c, mats_d = mats_a.to(device_t), mats_b.to(device_t), mats_c.to(device_t), mats_d.to(device_t)
            masks_a, masks_b, masks_c, masks_d = masks_a.to(device_t), masks_b.to(device_t), masks_c.to(device_t), masks_d.to(device_t)
            passcats = passcats.to(device_t)
            labels = labels.to(device_t)

            loss, logits, output = model(
                matrices_a=mats_a, matrices_b=mats_b, matrices_c=mats_c, matrices_d=mats_d,
                matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                strainPassCats=passcats, labels=labels
            )
            loss.backward()
            loss_ls.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        train_loss = float(np.mean(loss_ls))
        logger.add_scalar("train/loss", train_loss, epoch)

        # --- valid ---
        model.eval()
        pred_v, ref_v, loss_v = [], [], []
        for batch in valid_dataloader:
            emb_a, emb_b, emb_c, emb_d, passcats, labels = batch

            mats_a, masks_a = generate_matrix([emb_dict[k] for k in emb_a])
            mats_b, masks_b = generate_matrix([emb_dict[k] for k in emb_b])
            mats_c, masks_c = generate_matrix([emb_dict[k] for k in emb_c])
            mats_d, masks_d = generate_matrix([emb_dict[k] for k in emb_d])

            mats_a, mats_b, mats_c, mats_d = mats_a.to(device_t), mats_b.to(device_t), mats_c.to(device_t), mats_d.to(device_t)
            masks_a, masks_b, masks_c, masks_d = masks_a.to(device_t), masks_b.to(device_t), masks_c.to(device_t), masks_d.to(device_t)
            passcats = passcats.to(device_t)
            labels = labels.to(device_t)

            with torch.no_grad():
                loss, logits, output = model(
                    matrices_a=mats_a, matrices_b=mats_b, matrices_c=mats_c, matrices_d=mats_d,
                    matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                    matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                    strainPassCats=passcats, labels=labels
                )
            loss_v.append(loss.item())
            pred_v.extend(output.view(-1).tolist())
            ref_v.extend(labels.tolist())

        valid_mae, valid_mse, valid_pearson, valid_spearman, valid_R2 = print_exams(ref_v, pred_v)
        logger.add_scalar("val/mae", valid_mae, epoch)
        logger.add_scalar("val/mse", valid_mse, epoch)

        early_stopping(valid_mse, model)
        if early_stopping.early_stop:
            with open(logger.run_dir / "log.txt", "a") as f:
                f.write(f"[{datetime.now()}] Early stopping at epoch {epoch}\n")
            break

        # --- test ---
        model.eval()
        pred_t, ref_t, loss_t = [], [], []
        for batch in test_dataloader:
            emb_a, emb_b, emb_c, emb_d, passcats, labels = batch

            mats_a, masks_a = generate_matrix([emb_dict[k] for k in emb_a])
            mats_b, masks_b = generate_matrix([emb_dict[k] for k in emb_b])
            mats_c, masks_c = generate_matrix([emb_dict[k] for k in emb_c])
            mats_d, masks_d = generate_matrix([emb_dict[k] for k in emb_d])

            mats_a, mats_b, mats_c, mats_d = mats_a.to(device_t), mats_b.to(device_t), mats_c.to(device_t), mats_d.to(device_t)
            masks_a, masks_b, masks_c, masks_d = masks_a.to(device_t), masks_b.to(device_t), masks_c.to(device_t), masks_d.to(device_t)
            passcats = passcats.to(device_t)
            labels = labels.to(device_t)

            with torch.no_grad():
                loss, logits, output = model(
                    matrices_a=mats_a, matrices_b=mats_b, matrices_c=mats_c, matrices_d=mats_d,
                    matrix_attention_masks_a=masks_a, matrix_attention_masks_b=masks_b,
                    matrix_attention_masks_c=masks_c, matrix_attention_masks_d=masks_d,
                    strainPassCats=passcats, labels=labels
                )
            loss_t.append(loss.item())
            pred_t.extend(output.view(-1).tolist())
            ref_t.extend(labels.tolist())

        test_mae, test_mse, test_pearson, test_spearman, test_R2 = print_exams(ref_t, pred_t)
        logger.add_scalar("test/mae", test_mae, epoch)
        logger.add_scalar("test/mse", test_mse, epoch)
        logger.flush_tb()

        logger.log({
            "train_loss": train_loss,
            "valid_mae": float(valid_mae),
            "valid_mse": float(valid_mse),
            "test_mae": float(test_mae),
            "test_mse": float(test_mse),
            "valid_pearson": float(valid_pearson.statistic),
            "valid_spearman": float(valid_spearman.statistic),
            "valid_r2": float(valid_R2),
            "test_pearson": float(test_pearson.statistic),
            "test_spearman": float(test_spearman.statistic),
            "test_r2": float(test_R2),
        }, epoch=epoch)

        # best epoch artifacts（只保存 best）
        if valid_mse < best_valid_mse:
            best_valid_mse = float(valid_mse)

            valid_out = valid_data.reset_index(drop=True).copy()
            valid_out["pred"] = pred_v
            valid_out.to_csv(logger.run_dir / "artifacts" / "valid_preds_best.csv", index=False)

            test_out = test_data.reset_index(drop=True).copy()
            test_out["pred"] = pred_t
            test_out.to_csv(logger.run_dir / "artifacts" / "test_preds_best.csv", index=False)

            logger.save_json("best.json", {"best_epoch": epoch, "best_valid_mse": best_valid_mse})

        with open(logger.run_dir / "log.txt", "a") as f:
            f.write(f"[{datetime.now()}] epoch={epoch} train_loss={train_loss:.4f} valid_mse={valid_mse:.6f} test_mse={test_mse:.6f}\n")

    logger.close_tb()
    return str(logger.run_dir)
