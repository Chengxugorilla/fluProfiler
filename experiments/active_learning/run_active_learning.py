#!/usr/bin/env python3
import argparse
import csv
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active learning experiment with one required argument."
    )
    parser.add_argument(
        "--sample-virus-size",
        type=int,
        required=True,
        help="Number of representative virus clusters to sample.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string. Supports "6", "cuda:6", "cuda", "cpu". Default: auto ("cuda" if available else "cpu").',
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=60,
        help="Training epochs for active/random retraining.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Batch size for active/random training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size for test evaluation.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=128,
        help="Batch size when extracting candidate virus vectors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Default: runs/holdout_tests/titer/2025-08-19_17-43-32.pth",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Default: runs/active_learning/seasonal",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        normalized = device_arg.strip()
        if normalized.isdigit():
            normalized = f"cuda:{normalized}"
        return torch.device(normalized)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bootstrap_imports(project_root: Path):
    src_path = project_root / "src"
    fluprofiler_path = src_path / "fluprofiler"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(fluprofiler_path) not in sys.path:
        sys.path.insert(0, str(fluprofiler_path))

    # Backward-compatible alias for legacy checkpoints serialized with fluProfiler_models
    if "fluProfiler_models" in sys.modules:
        del sys.modules["fluProfiler_models"]
    importlib.invalidate_caches()
    old_mod = importlib.import_module("fluProfiler_models")
    sys.modules["fluProfiler_models"] = old_mod

    required_symbols = ["fluProfiler_HANA", "MaskedMSELoss", "attention_mask"]
    missing_symbols = [name for name in required_symbols if not hasattr(old_mod, name)]
    if missing_symbols:
        raise RuntimeError(f"fluProfiler_models missing symbols: {missing_symbols}")

    from utilities import load_embedding, print_exams  # noqa: WPS433

    return load_embedding, print_exams


class FluProfilerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.emb_file_name_a = ("matrix_" + dataframe["seq_id_a"]).tolist()
        self.emb_file_name_b = ("matrix_" + dataframe["seq_id_b"]).tolist()
        self.emb_file_name_c = ("matrix_" + dataframe["seq_id_c"]).tolist()
        self.emb_file_name_d = ("matrix_" + dataframe["seq_id_d"]).tolist()

        self.strain_pass_cats = convert_pass_to_tensor(
            (
                "<cls>"
                + dataframe["serumPassCat"]
                + "<eos>"
                + dataframe["virusPassCat"]
                + "<eos>"
            ).tolist()
        )
        self.labels = torch.tensor(dataframe["label"].tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.emb_file_name_a[idx],
            self.emb_file_name_b[idx],
            self.emb_file_name_c[idx],
            self.emb_file_name_d[idx],
            self.strain_pass_cats[idx],
            self.labels[idx],
        )


def convert_pass_to_tensor(pass_cats):
    result = [
        item.replace("<cls>", "0")
        .replace("<eos>", "1")
        .replace("<EGG>", "2")
        .replace("<CELL>", "3")
        .replace("<BOTH>", "4")
        for item in pass_cats
    ]
    return torch.tensor([[int(ch) for ch in item] for item in result], dtype=torch.long)


def generate_matrix(matrix_list):
    seq_len = [mat.shape[0] for mat in matrix_list]
    max_len = max(seq_len)
    mask_list = []

    padded = []
    for i, mat in enumerate(matrix_list):
        padded_mat = F.pad(mat, (0, 0, 0, max_len - seq_len[i]))
        padded.append(padded_mat)
        mask = torch.cat(
            (
                torch.ones(1, seq_len[i], dtype=torch.float32, device=mat.device),
                torch.zeros(1, max_len - seq_len[i], dtype=torch.float32, device=mat.device),
            ),
            dim=1,
        )
        mask_list.append(mask)

    matrix = torch.stack(padded)
    mask = torch.stack(mask_list).view(len(matrix_list), max_len)
    return matrix, mask


def group_and_filt(df):
    group_columns = ["seq_a", "seq_b", "seq_c", "seq_d", "serumPassCat", "virusPassCat"]
    agg_dict = {c: "first" for c in df.columns if c not in group_columns}
    agg_dict["label"] = "mean"
    return df.groupby(group_columns).agg(agg_dict).reset_index()


class Emb2Vec(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.matrix_dropout = model.matrix_dropout
        self.matrix_pooler = model.matrix_pooler
        self.linear = model.linear[0][0]
        self.act = model.linear[0][1] if len(model.linear[0]) > 1 else None

    def forward(self, embedding):
        embedding = self.matrix_dropout(embedding)
        seq_vector = self.matrix_pooler(embedding)
        seq_vector = self.linear(seq_vector)
        if self.act is not None:
            seq_vector = self.act(seq_vector)
        return seq_vector


def run_emb2vec_on_dict(emb_dict, emb2vec, device):
    emb2vec = emb2vec.to(device)
    emb2vec.eval()

    vec_dict = {}
    for key, emb in tqdm(emb_dict.items(), desc="Pooling embeddings"):
        emb = emb.to(device)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)

        with torch.no_grad():
            vec = emb2vec(emb)[0].detach().cpu()
        vec_dict[key] = vec

    return vec_dict


def select_representative_viruses(virus_emb: np.ndarray, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(virus_emb)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    selected_indices = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        cluster_points = virus_emb[idx]
        center = centers[i]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_idx = idx[np.argmin(dists)]
        selected_indices.append(best_idx)

    return selected_indices


def log_metrics(epoch_idx, test_mae, test_mse, test_pearson, test_spearman, r2, log_path: Path):
    new_file = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["epoch", "mae", "mse", "pearson", "spearman", "r2"])
        writer.writerow([
            epoch_idx,
            test_mae,
            test_mse,
            test_pearson[0],
            test_spearman[0],
            r2,
        ])


def build_batch_tensors(batch, emb_dict, device, emb_in_device=True):
    emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strain_pass_cats, labels = batch

    def lookup(keys):
        return [emb_dict[key.replace("_active", "")] if key.endswith("_active") else emb_dict[key] for key in keys]

    matrixs_a, masks_a = generate_matrix(lookup(emb_file_name_a))
    matrixs_b, masks_b = generate_matrix(lookup(emb_file_name_b))
    matrixs_c, masks_c = generate_matrix(lookup(emb_file_name_c))
    matrixs_d, masks_d = generate_matrix(lookup(emb_file_name_d))

    if not emb_in_device:
        matrixs_a = matrixs_a.to(device)
        matrixs_b = matrixs_b.to(device)
        matrixs_c = matrixs_c.to(device)
        matrixs_d = matrixs_d.to(device)

    masks_a = masks_a.to(device)
    masks_b = masks_b.to(device)
    masks_c = masks_c.to(device)
    masks_d = masks_d.to(device)
    strain_pass_cats = strain_pass_cats.to(device)
    labels = labels.to(device)

    return {
        "matrices_a": matrixs_a,
        "matrices_b": matrixs_b,
        "matrices_c": matrixs_c,
        "matrices_d": matrixs_d,
        "matrix_attention_masks_a": masks_a,
        "matrix_attention_masks_b": masks_b,
        "matrix_attention_masks_c": masks_c,
        "matrix_attention_masks_d": masks_d,
        "strainPassCats": strain_pass_cats,
        "labels": labels,
    }


def multidata_trainer(
    model,
    train_dataloader,
    test_dataloader,
    emb_dict,
    lr_rate,
    epochs,
    save_path,
    device,
    print_exams,
    emb_in_device=True,
):
    save_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    no_decay = ["bias", "layernorm.weight", "layer_norm.weight", "layer.norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr_rate, betas=(0.9, 0.99), eps=1e-8)

    test_mae_ls, test_mse_ls, test_pearson_ls, test_spearman_ls, test_r2_ls = [], [], [], [], []
    minimal_mse = float("inf")

    for epoch_idx in range(epochs):
        model.train()
        train_loss_ls = []

        for batch in train_dataloader:
            optimizer.zero_grad()
            model_inputs = build_batch_tensors(batch, emb_dict, device, emb_in_device=emb_in_device)
            loss, logits, output = model(**model_inputs)
            loss.backward()
            optimizer.step()
            train_loss_ls.append(loss.item())

        test_prediction_ls = []
        test_reference_ls = []

        model.eval()
        for batch in test_dataloader:
            with torch.no_grad():
                model_inputs = build_batch_tensors(batch, emb_dict, device, emb_in_device=emb_in_device)
                labels = model_inputs["labels"]
                loss, logits, output = model(**model_inputs)

            test_prediction_ls.extend(output.view(-1).detach().cpu().tolist())
            test_reference_ls.extend(labels.detach().cpu().tolist())

        test_mae, test_mse, test_pearson, test_spearman, r2 = print_exams(
            test_reference_ls,
            test_prediction_ls,
            print_result=False,
        )

        test_mae_ls.append(test_mae)
        test_mse_ls.append(test_mse)
        test_pearson_ls.append(test_pearson)
        test_spearman_ls.append(test_spearman)
        test_r2_ls.append(r2)

        if test_mse < minimal_mse:
            minimal_mse = test_mse
            torch.save(model, save_path / "model.pth")

        log_metrics(
            epoch_idx,
            test_mae,
            test_mse,
            test_pearson,
            test_spearman,
            r2,
            save_path / "log.csv",
        )

        print(
            f"[{save_path.name}] epoch={epoch_idx + 1}/{epochs} "
            f"train_loss={np.mean(train_loss_ls):.6f} "
            f"test_mae={test_mae:.6f} "
            f"test_mse={test_mse:.6f} "
            f"pearson={test_pearson[0]:.6f} "
            f"spearman={test_spearman[0]:.6f} "
            f"r2={r2:.6f}"
        )

    return [test_mae_ls, test_mse_ls, test_pearson_ls, test_spearman_ls, test_r2_ls]


def main():
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[2]
    load_embedding, print_exams = bootstrap_imports(project_root)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else project_root / "runs" / "holdout_tests" / "titer" / "2025-08-19_17-43-32.pth"
    )
    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else project_root / "runs" / "active_learning" / "seasonal"
    )

    all_csv = project_root / "data" / "active_learning" / "processed" / "All.csv"
    embedding_dir = project_root / "data" / "reverse_test" / "embedding"

    print("Loading data...")
    all_data = pd.read_csv(all_csv)

    active_final = all_data[
        (all_data["sheet"].str.split("-").str[0].astype(int) >= 41)
        & (all_data["sheet"].str.split("-").str[0].astype(int) <= 42)
    ]
    active_final = group_and_filt(active_final)

    active_candidate, active_test = train_test_split(
        active_final,
        test_size=0.1,
        random_state=args.seed,
    )

    print(f"Active_final: {len(active_final)}")
    print(f"Active_candidate: {len(active_candidate)}")
    print(f"Active_test: {len(active_test)}")

    active_candidate_dataloader = DataLoader(
        FluProfilerDataset(active_candidate),
        batch_size=args.candidate_batch_size,
        shuffle=False,
    )
    active_test_dataloader = DataLoader(
        FluProfilerDataset(active_test),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    emb_files = (
        "matrix_"
        + pd.concat(
            [
                active_final["seq_id_a"],
                active_final["seq_id_b"],
                active_final["seq_id_c"],
                active_final["seq_id_d"],
            ]
        ).drop_duplicates()
        + ".pt"
    )
    emb_dict = load_embedding(
        path=str(embedding_dir),
        files=emb_files,
        map_location=device,
    )

    print("Loading checkpoint for feature extraction...")
    model = torch.load(checkpoint_path, weights_only=False, map_location=device)
    embedding_pooler = Emb2Vec(model)
    vector_dict = run_emb2vec_on_dict(emb_dict, embedding_pooler, device=device)

    print("Building candidate virus vectors...")
    virus_tensor_list = []
    for row in active_candidate.itertuples():
        ha_vector = vector_dict["matrix_" + row.seq_id_c]
        na_vector = vector_dict["matrix_" + row.seq_id_d]
        ha_na_vector = torch.cat([ha_vector, na_vector], dim=0)
        virus_tensor_list.append(ha_na_vector)

    virus_tensor = torch.stack(virus_tensor_list)
    virus_np = virus_tensor.numpy()

    unique_virus_emb, selected_indices = np.unique(virus_np, axis=0, return_index=True)
    virus_df = active_candidate.copy().reset_index(drop=True).iloc[selected_indices]

    print(f"Selecting representative viruses with sample_virus_size={args.sample_virus_size}...")
    active_selected_virus = select_representative_viruses(
        unique_virus_emb,
        n_clusters=args.sample_virus_size,
    )
    active_selected = (
        virus_df.iloc[active_selected_virus][["seq_id_c", "seq_id_d"]]
        .merge(active_candidate, on=["seq_id_c", "seq_id_d"], how="left")
    )

    random_selected_virus = virus_df.sample(len(active_selected_virus), random_state=args.seed)
    random_selected = (
        random_selected_virus[["seq_id_c", "seq_id_d"]]
        .merge(active_candidate, on=["seq_id_c", "seq_id_d"], how="left")
    )

    if len(active_selected) < len(random_selected):
        random_selected = random_selected.sample(len(active_selected), random_state=args.seed)
    else:
        active_selected = active_selected.sample(len(random_selected), random_state=args.seed)

    save_dir = output_root / str(args.sample_virus_size)
    save_dir.mkdir(parents=True, exist_ok=True)

    active_selected.to_csv(save_dir / "active_selected.csv", index=False)
    random_selected.to_csv(save_dir / "random_selected.csv", index=False)

    print(f"Active selection size: {len(active_selected)}")
    print(f"Random selection size: {len(random_selected)}")
    print(f"Saving outputs to: {save_dir}")

    active_dataloader = DataLoader(
        FluProfilerDataset(active_selected),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    random_dataloader = DataLoader(
        FluProfilerDataset(random_selected),
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    print("Running active training...")
    model_active = torch.load(checkpoint_path, weights_only=False, map_location=device)
    multidata_trainer(
        model=model_active,
        train_dataloader=active_dataloader,
        test_dataloader=active_test_dataloader,
        emb_dict=emb_dict,
        lr_rate=args.lr,
        epochs=args.epoch,
        save_path=save_dir / "active",
        device=device,
        print_exams=print_exams,
        emb_in_device=True,
    )

    print("Running random training...")
    model_random = torch.load(checkpoint_path, weights_only=False, map_location=device)
    multidata_trainer(
        model=model_random,
        train_dataloader=random_dataloader,
        test_dataloader=active_test_dataloader,
        emb_dict=emb_dict,
        lr_rate=args.lr,
        epochs=args.epoch,
        save_path=save_dir / "random",
        device=device,
        print_exams=print_exams,
        emb_in_device=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()