#!/usr/bin/env python3
"""Train Pair-Site Attention to directly predict antigenic distance labels."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_THIS_FILE.parent))

import train_minus_homo as base  # noqa: E402
from fluprofiler.features.na_glycan_features import na_head_glycan_mismatch  # noqa: E402
from fluprofiler.models.pair_site_attention_minus_model import (  # noqa: E402
    PairSiteAttentionMinusConfig,
    PairSiteAttentionScoreModel,
)
from fluprofiler.models.serum_gate_model import (  # noqa: E402
    SerumGateBatch,
    label_bin_weights,
    pairwise_ranking_loss,
    weighted_masked_mean,
)
from train_pair_site_minus_homo import (  # noqa: E402
    prepare_pair_site_embeddings,
    validate_pair_site_alignment,
)


class DistanceOnlyTaskDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        vocabs: base.SerumGateVocabs,
        task_cols: list[str] | None = None,
        max_queries_per_task: int | None = 128,
        label_col: str = "label",
    ):
        if max_queries_per_task is not None and max_queries_per_task <= 0:
            raise ValueError("max_queries_per_task must be positive or None")
        self.frame = dataframe.reset_index(drop=True).copy()
        self.vocabs = vocabs
        self.task_cols = task_cols or base.DEFAULT_TASK_COLS
        self.max_queries_per_task = max_queries_per_task
        self.label_col = label_col
        self.frame["_task_key"] = base.make_task_keys(self.frame, self.task_cols)
        self.task_chunks: list[tuple[str, np.ndarray]] = []
        for task_key, group in self.frame.groupby("_task_key", sort=False):
            indices = group.index.to_numpy()
            if max_queries_per_task is None:
                self.task_chunks.append((task_key, indices))
            else:
                for start in range(0, len(indices), max_queries_per_task):
                    self.task_chunks.append(
                        (task_key, indices[start : start + max_queries_per_task])
                    )

    def __len__(self) -> int:
        return len(self.task_chunks)

    def _passage_id(self, value: Any) -> int:
        return self.vocabs.passage_to_id.get(base.normalize_passage(value), 0)

    def _subtype_id(self, row: pd.Series) -> int:
        return self.vocabs.subtype_to_id.get(
            base.normalize_subtype(row[base.subtype_column(self.frame)]),
            0,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task_key, row_indices = self.task_chunks[idx]
        task_frame = self.frame.loc[row_indices].reset_index(drop=True)
        first = task_frame.iloc[0]
        serum_passage = self._passage_id(first["serumPassCat"])
        query_passage = torch.tensor(
            [self._passage_id(value) for value in task_frame["virusPassCat"].tolist()],
            dtype=torch.long,
        )
        passage_pair = serum_passage * len(self.vocabs.passage_to_id) + query_passage
        labels = torch.tensor(
            task_frame[self.label_col].to_numpy(dtype=float),
            dtype=torch.float32,
        )
        s_nagly = torch.tensor(
            [
                na_head_glycan_mismatch(str(row["seq_b"]), str(row["seq_d"]))
                for _, row in task_frame.iterrows()
            ],
            dtype=torch.float32,
        )
        return {
            "task_key": task_key,
            "support_size": 0,
            "reference_ha_key": f"matrix_{first['seq_id_a']}",
            "query_ha_keys": [
                f"matrix_{seq_id}" for seq_id in task_frame["seq_id_c"].tolist()
            ],
            "reference_na_key": f"matrix_{first['seq_id_b']}",
            "query_na_keys": [
                f"matrix_{seq_id}" for seq_id in task_frame["seq_id_d"].tolist()
            ],
            "reference_ha_aligned": str(first.get("serumHA", "")),
            "query_ha_aligned": [
                str(value)
                for value in task_frame.get("virusHA", pd.Series([], dtype=str)).tolist()
            ],
            "serum_passage": torch.tensor(serum_passage, dtype=torch.long),
            "query_passage": query_passage,
            "passage_pair": passage_pair.long(),
            "subtype": torch.tensor(self._subtype_id(first), dtype=torch.long),
            "s_nagly": s_nagly,
            "labels": labels,
            "row_indices": torch.tensor(row_indices, dtype=torch.long),
            "query_meta": task_frame.drop(columns=["_task_key"]).reset_index(drop=True),
        }


def build_distance_loader(
    frame: pd.DataFrame,
    vocabs: base.SerumGateVocabs,
    embeddings: dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
    max_queries_per_task: int | None,
    gpu_cache: base.GpuEmbeddingCache | None,
    device: torch.device | None,
    task_cols: list[str],
    label_col: str,
) -> DataLoader:
    if batch_size != 1:
        raise ValueError("Use --batch-size 1 for variable-size PairSite tasks")
    dataset = DistanceOnlyTaskDataset(
        frame,
        vocabs,
        task_cols=task_cols,
        max_queries_per_task=max_queries_per_task,
        label_col=label_col,
    )
    if gpu_cache is not None:
        if device is None:
            raise ValueError("device is required when gpu_cache is enabled")
        collate_fn = lambda items: base.collate_serum_gate_tasks_with_gpu_cache(
            items,
            gpu_cache,
            device,
            align_ha_embeddings=True,
            include_na_embeddings=False,
        )
    else:
        collate_fn = lambda items: base.collate_serum_gate_tasks(
            items,
            embeddings,
            align_ha_embeddings=True,
            include_na_embeddings=False,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def distance_loss(
    out: dict[str, torch.Tensor],
    batch: SerumGateBatch,
    loss_name: str,
    config: PairSiteAttentionMinusConfig,
) -> torch.Tensor:
    labels = batch.labels.float().to(out["mean"].device)
    query_mask = batch.query_mask.float().to(out["mean"].device)
    log_var = out["log_var"].clamp(-12.0, 12.0)
    if loss_name == "nll":
        per_value = 0.5 * (log_var + (labels - out["mean"]).pow(2) * torch.exp(-log_var))
    else:
        per_value = F.smooth_l1_loss(out["mean"], labels, beta=0.5, reduction="none")
    loss_weights = label_bin_weights(
        labels,
        config.label_weight_thresholds,
        config.label_weight_values,
    ).to(out["mean"].device)
    loss = weighted_masked_mean(per_value, loss_weights, query_mask)
    rank_loss = pairwise_ranking_loss(
        out["mean"],
        labels,
        query_mask,
        margin=config.rank_loss_margin,
        min_label_delta=config.rank_loss_min_label_delta,
    )
    out["rank_loss"] = rank_loss
    return loss + float(config.rank_loss_weight) * rank_loss


def evaluate_model(
    model: PairSiteAttentionScoreModel,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    config: PairSiteAttentionMinusConfig,
    progress_bar=None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch, items in loader:
            item = items[0]
            batch = base.move_batch(batch, device)
            out = model(batch)
            loss = distance_loss(out, batch, loss_name, config)
            losses.append(float(loss.item()))
            mean = out["mean"].detach().cpu().view(-1).numpy()
            log_var = out["log_var"].detach().cpu().view(-1).numpy()
            labels = item["labels"].detach().cpu().view(-1).numpy()
            meta = item["query_meta"]
            for idx in range(len(labels)):
                row = meta.iloc[idx].to_dict()
                row.update(
                    {
                        "task_key": item["task_key"],
                        "label": float(labels[idx]),
                        "mean": float(mean[idx]),
                        "log_var": float(log_var[idx]),
                    }
                )
                rows.append(row)
            if progress_bar is not None:
                progress_bar.update(1)
    pred_frame = pd.DataFrame(rows)
    metrics = base.serum_regression_metrics(pred_frame)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, pred_frame


def train_one_epoch(
    model: PairSiteAttentionScoreModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    config: PairSiteAttentionMinusConfig,
    progress_bar=None,
) -> float:
    model.train()
    losses: list[float] = []
    for batch, _items in loader:
        optimizer.zero_grad()
        batch = base.move_batch(batch, device)
        out = model(batch)
        loss = distance_loss(out, batch, loss_name, config)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if progress_bar is not None:
            progress_bar.update(1)
    return float(np.mean(losses)) if losses else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pair-Site Attention direct antigenic distance model."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", dest="type_filter", default="")
    parser.add_argument("--serum-task-cols", type=str, default=",".join(base.DEFAULT_TASK_COLS))
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument(
        "--refit-train-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-queries-per-task", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--early-stopping-patience", type=int, default=-1)
    parser.add_argument("--site-proj-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--site-attention-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--passage-dim", type=int, default=8)
    parser.add_argument("--subtype-dim", type=int, default=8)
    parser.add_argument("--predictor-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--use-subtype-feature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--label-weight-thresholds", type=str, default="2,4,6")
    parser.add_argument(
        "--label-weight-threshold-mode",
        choices=("fixed", "quantile"),
        default="fixed",
    )
    parser.add_argument("--label-weight-quantiles", type=str, default="0.35,0.75,0.95")
    parser.add_argument("--label-weight-values", type=str, default="1,1.3,1.8,2.5")
    parser.add_argument(
        "--within-serum-rank-loss-weight",
        dest="rank_loss_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--within-serum-rank-margin",
        dest="rank_loss_margin",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--within-serum-rank-min-label-delta",
        dest="rank_loss_min_label_delta",
        type=float,
        default=0.0,
    )
    parser.add_argument("--loss", choices=("huber", "nll"), default="nll")
    parser.add_argument("--best-metric", type=str, default="pooled_mse")
    parser.add_argument(
        "--skip-test-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu-cache-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.lr_min < 0:
        parser.error("--lr-min must be non-negative")
    if args.early_stopping_patience < -1:
        parser.error("--early-stopping-patience must be -1 or greater")
    if args.type_filter:
        args.use_subtype_feature = False
    if not args.use_subtype_feature:
        args.subtype_dim = 0
    if args.max_queries_per_task <= 0:
        args.max_queries_per_task = None
    return args


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    task_cols = base.parse_column_list(args.serum_task_cols)
    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = base.load_fixed_split_frames(
        args.data_dir,
        sample_limit=sample_limit,
        refit_train_valid=args.refit_train_valid,
        type_filter=args.type_filter,
        label_columns=(args.label_col,),
    )
    base.validate_aligned_ha_columns(frames)
    embedding_files = base.required_embedding_files(frames, include_na_embeddings=False)
    embedding_dir = base.validate_embedding_files(args.embedding_dir, embedding_files)
    embeddings = base.load_embeddings(
        embedding_dir,
        embedding_files,
        show_progress=args.progress,
    )
    embeddings = prepare_pair_site_embeddings(frames, embeddings)
    max_site_length = validate_pair_site_alignment(frames, embeddings)
    hidden_size = base.infer_hidden_size(embeddings)
    vocabs = base.build_serum_gate_vocabs(
        frames,
        use_subtype_feature=args.use_subtype_feature,
    )
    paths = base.prepare_output_dir(args.output_dir)
    base.set_seed(args.seed)

    device = torch.device(args.device)
    gpu_cache = None
    if args.gpu_cache_gb > 0:
        if device.type != "cuda":
            raise ValueError("--gpu-cache-gb requires a CUDA device")
        gpu_cache = base.GpuEmbeddingCache(
            cpu_store=embeddings,
            device=device,
            max_bytes=int(float(args.gpu_cache_gb) * 1024**3),
        )
    loaders = {
        name: build_distance_loader(
            frame,
            vocabs,
            embeddings,
            args.batch_size,
            shuffle=(name == "train"),
            max_queries_per_task=args.max_queries_per_task,
            gpu_cache=gpu_cache,
            device=device,
            task_cols=task_cols,
            label_col=args.label_col,
        )
        for name, frame in frames.items()
    }

    label_weight_values = base.parse_float_list(args.label_weight_values)
    label_weight_thresholds = base.resolve_label_weight_thresholds(
        frames,
        mode=args.label_weight_threshold_mode,
        thresholds_arg=args.label_weight_thresholds,
        quantiles_arg=args.label_weight_quantiles,
        label_col=args.label_col,
    )
    label_weight_quantiles = base.parse_float_list(args.label_weight_quantiles)
    if len(label_weight_values) != len(label_weight_thresholds) + 1:
        raise ValueError(
            "--label-weight-values must have exactly one more entry than "
            "--label-weight-thresholds"
        )

    model_config = PairSiteAttentionMinusConfig(
        hidden_size=hidden_size,
        max_site_length=max_site_length,
        site_proj_dim=args.site_proj_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        site_attention_dim=args.site_attention_dim,
        dropout=args.dropout,
        passage_vocab_size=len(vocabs.passage_to_id),
        passage_pair_vocab_size=len(vocabs.passage_to_id) ** 2,
        subtype_vocab_size=len(vocabs.subtype_to_id),
        passage_dim=args.passage_dim,
        subtype_dim=args.subtype_dim,
        predictor_hidden_dim=args.predictor_hidden_dim,
        label_weight_thresholds=tuple(label_weight_thresholds),
        label_weight_values=tuple(label_weight_values),
        rank_loss_weight=args.rank_loss_weight,
        rank_loss_margin=args.rank_loss_margin,
        rank_loss_min_label_delta=args.rank_loss_min_label_delta,
    )
    model = PairSiteAttentionScoreModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_min,
        )

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "PairSiteAttn-DistanceOnly",
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "embedding_dir": str(embedding_dir),
        "output_dir": str(paths["root"]),
        "type_filter": args.type_filter,
        "task_cols": task_cols,
        "refit_train_valid": bool(args.refit_train_valid),
        "task_overlap_counts": base.task_overlap_counts(frames, task_cols),
        "row_counts": {name: len(frame) for name, frame in frames.items()},
        "task_counts": {
            name: len(base.make_task_keys(frame, task_cols).drop_duplicates())
            for name, frame in frames.items()
        },
        "episode_counts": {
            name: len(
                DistanceOnlyTaskDataset(
                    frame,
                    vocabs,
                    task_cols=task_cols,
                    max_queries_per_task=args.max_queries_per_task,
                    label_col=args.label_col,
                )
            )
            for name, frame in frames.items()
        },
        "max_queries_per_task": args.max_queries_per_task,
        "gpu_cache_gb": args.gpu_cache_gb,
        "embedding_token_policy": "strip_boundary_special_tokens_when_present",
        "label_col": args.label_col,
        "best_metric": args.best_metric,
        "label_weight_threshold_mode": args.label_weight_threshold_mode,
        "label_weight_quantiles": label_weight_quantiles,
        "label_weight_thresholds": label_weight_thresholds,
        "label_weight_values": label_weight_values,
        "model_config": model_config.__dict__,
        "passage_to_id": vocabs.passage_to_id,
        "subtype_to_id": vocabs.subtype_to_id,
        "use_subtype_feature": bool(args.use_subtype_feature),
        "loss": args.loss,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "lr_min": args.lr_min,
        "early_stopping_patience": args.early_stopping_patience,
        "skip_test_eval": bool(args.skip_test_eval),
        "seed": args.seed,
    }
    paths["config"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["log"].write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    best_valid = float("inf")
    best_result: dict[str, Any] = {}
    epochs_without_improvement = 0
    eval_steps = 0 if args.skip_test_eval else len(loaders["test"])
    total_steps = args.epochs * (
        len(loaders["train"]) + len(loaders["valid"]) + eval_steps
    )
    with tqdm(
        total=total_steps,
        desc="training",
        disable=not args.progress,
        dynamic_ncols=True,
        file=sys.stdout,
    ) as progress_bar:
        for epoch in range(args.epochs):
            progress_bar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", split="train")
            train_loss = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                args.loss,
                model_config,
                progress_bar=progress_bar,
            )
            if args.refit_train_valid:
                valid_metrics = base.serum_regression_metrics(pd.DataFrame())
                valid_metrics["loss"] = 0.0
                valid_preds = pd.DataFrame()
            else:
                progress_bar.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    split="valid",
                )
                valid_metrics, valid_preds = evaluate_model(
                    model,
                    loaders["valid"],
                    device,
                    args.loss,
                    model_config,
                    progress_bar=progress_bar,
                )
            if args.skip_test_eval:
                test_metrics = base.serum_regression_metrics(pd.DataFrame())
                test_metrics["loss"] = 0.0
                test_preds = pd.DataFrame()
            else:
                progress_bar.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    split="test",
                )
                test_metrics, test_preds = evaluate_model(
                    model,
                    loaders["test"],
                    device,
                    args.loss,
                    model_config,
                    progress_bar=progress_bar,
                )
            epoch_result = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "valid": valid_metrics,
                "test": test_metrics,
            }
            base.append_metrics_csv(paths["metrics"], epoch_result)
            if args.best_metric not in valid_metrics:
                available = ", ".join(sorted(valid_metrics))
                raise ValueError(
                    f"--best-metric {args.best_metric!r} not found in validation "
                    f"metrics: {available}"
                )
            is_refit_final_epoch = args.refit_train_valid and epoch + 1 == args.epochs
            is_best_valid_epoch = (
                not args.refit_train_valid
                and valid_metrics[args.best_metric] < best_valid
            )
            if is_refit_final_epoch or is_best_valid_epoch:
                best_valid = valid_metrics[args.best_metric]
                best_result = epoch_result
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config.__dict__,
                        "passage_to_id": vocabs.passage_to_id,
                        "subtype_to_id": vocabs.subtype_to_id,
                        "task_cols": task_cols,
                        "label_col": args.label_col,
                        "best_metric": args.best_metric,
                        "epoch": epoch + 1,
                        "valid_metrics": valid_metrics,
                        "refit_train_valid": bool(args.refit_train_valid),
                    },
                    paths["checkpoints"] / "best_model.pth",
                )
                if not valid_preds.empty:
                    base.write_rounded_predictions(
                        valid_preds,
                        paths["root"] / "predictions_valid.csv",
                    )
                if not test_preds.empty:
                    base.write_rounded_predictions(
                        test_preds,
                        paths["root"] / "predictions_test.csv",
                    )
            elif not args.refit_train_valid:
                epochs_without_improvement += 1
            if scheduler is not None:
                scheduler.step()
            if (
                args.early_stopping_patience >= 0
                and not args.refit_train_valid
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                with paths["log"].open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"no valid_{args.best_metric} improvement for "
                        f"{epochs_without_improvement} epoch(s).\n"
                    )
                break
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "best": best_result,
    }


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
