#!/usr/bin/env python3
"""Train Pair-Site Attention SerumGate-Minus with three-target supervision."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_THIS_FILE.parent))

import train_minus_homo as base  # noqa: E402
from fluprofiler.models.pair_site_attention_minus_model import (  # noqa: E402
    PairSiteAttentionMinusConfig,
    PairSiteAttentionMinusModel,
)


PAIR_ALIGNMENT_COLUMNS = {
    "seq_id_a",
    "seq_id_c",
    "seq_a",
    "seq_c",
    "serumHA",
    "virusHA",
}


def prepare_pair_site_embeddings(
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return residue-only embeddings, removing one special token at each end."""

    sequence_lengths: dict[str, int] = {}
    for split_name, frame in frames.items():
        required = {"seq_id_a", "seq_id_c", "seq_a", "seq_c"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{split_name} split is missing sequence columns: {', '.join(missing)}"
            )
        for row_index, row in frame.iterrows():
            for id_column, sequence_column in (
                ("seq_id_a", "seq_a"),
                ("seq_id_c", "seq_c"),
            ):
                seq_id = str(row[id_column])
                sequence_length = len(str(row[sequence_column]))
                previous = sequence_lengths.setdefault(seq_id, sequence_length)
                if previous != sequence_length:
                    raise ValueError(
                        f"{split_name} row {row_index}: {seq_id} has inconsistent "
                        "sequence lengths"
                    )

    prepared = dict(embeddings)
    for seq_id, sequence_length in sequence_lengths.items():
        embedding_key = f"matrix_{seq_id}"
        if embedding_key not in embeddings:
            raise ValueError(f"Missing embedding for {seq_id}: {embedding_key}")
        matrix = embeddings[embedding_key]
        if matrix.ndim != 2:
            raise ValueError(
                f"{seq_id}: expected a 2D embedding, got shape {tuple(matrix.shape)}"
            )
        row_count = int(matrix.shape[0])
        if row_count == sequence_length:
            prepared[embedding_key] = matrix
        elif row_count == sequence_length + 2:
            prepared[embedding_key] = matrix[1:-1]
        else:
            raise ValueError(
                f"{seq_id}: expected {sequence_length} residue rows or "
                f"{sequence_length + 2} rows with boundary special tokens, "
                f"got {row_count}"
            )
    return prepared


def validate_pair_site_alignment(
    frames: dict[str, pd.DataFrame],
    embeddings: dict[str, torch.Tensor],
) -> int:
    """Validate post-embedding alignment and return its maximum site length."""

    max_site_length = 0
    reference_alignments: dict[str, str] = {}
    for split_name, frame in frames.items():
        missing = sorted(PAIR_ALIGNMENT_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{split_name} split is missing pair-site columns: {', '.join(missing)}"
            )
        for row_index, row in frame.iterrows():
            serum_aligned = str(row["serumHA"])
            virus_aligned = str(row["virusHA"])
            if len(serum_aligned) != len(virus_aligned):
                raise ValueError(
                    f"{split_name} row {row_index}: alignment lengths differ "
                    f"({len(serum_aligned)} != {len(virus_aligned)})"
                )

            for side, id_column, sequence_column, aligned in (
                ("serum", "seq_id_a", "seq_a", serum_aligned),
                ("virus", "seq_id_c", "seq_c", virus_aligned),
            ):
                seq_id = str(row[id_column])
                raw_sequence = str(row[sequence_column])
                ungapped = aligned.replace("-", "")
                if ungapped != raw_sequence:
                    raise ValueError(
                        f"{split_name} row {row_index}: {side} alignment does not "
                        f"match {seq_id}"
                    )
                embedding_key = f"matrix_{seq_id}"
                if embedding_key not in embeddings:
                    raise ValueError(f"Missing embedding for {seq_id}: {embedding_key}")
                matrix = embeddings[embedding_key]
                if matrix.ndim != 2:
                    raise ValueError(
                        f"{seq_id}: expected a 2D embedding, got shape {tuple(matrix.shape)}"
                    )
                expected_rows = len(ungapped)
                actual_rows = int(matrix.shape[0])
                if actual_rows != expected_rows:
                    raise ValueError(
                        f"{seq_id}: expected {expected_rows} embedding rows, "
                        f"got {actual_rows}"
                    )

            reference_id = str(row["seq_id_a"])
            previous = reference_alignments.setdefault(reference_id, serum_aligned)
            if previous != serum_aligned:
                raise ValueError(
                    f"{reference_id}: multiple serumHA alignments are not supported"
                )
            max_site_length = max(max_site_length, len(serum_aligned))

    if max_site_length <= 0:
        raise ValueError("No aligned HA sites found")
    return max_site_length


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Pair-Site Attention SerumGate-Minus-Homo."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--type",
        dest="type_filter",
        default="",
        help="Optional Type/virusType value to keep, e.g. H1N1 or H3N2.",
    )
    parser.add_argument(
        "--serum-task-cols",
        type=str,
        default=",".join(base.DEFAULT_TASK_COLS),
    )
    parser.add_argument("--query-label-col", type=str, default="label")
    parser.add_argument("--homo-label-col", type=str, default="homo_label")
    parser.add_argument("--distance-label-col", type=str, default="diff_label")
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
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine"),
        default="none",
    )
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
    parser.add_argument(
        "--label-weight-quantiles",
        type=str,
        default="0.35,0.75,0.95",
    )
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
    parser.add_argument("--distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--homo-loss-weight", type=float, default=1.0)
    parser.add_argument("--query-loss-weight", type=float, default=1.0)
    parser.add_argument("--best-metric", type=str, default="diff_mse")
    parser.add_argument(
        "--score-log-var-mode",
        choices=("sum", "query"),
        default="sum",
    )
    parser.add_argument(
        "--skip-test-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu-cache-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args(argv)
    if args.lr_min < 0:
        parser.error("--lr-min must be non-negative")
    if args.early_stopping_patience < -1:
        parser.error("--early-stopping-patience must be -1 or greater")
    if min(
        args.distance_loss_weight,
        args.homo_loss_weight,
        args.query_loss_weight,
    ) < 0:
        parser.error("loss weights must be non-negative")
    if (
        args.distance_loss_weight
        + args.homo_loss_weight
        + args.query_loss_weight
        <= 0
    ):
        parser.error("at least one loss weight must be positive")
    if args.type_filter:
        args.use_subtype_feature = False
    if not args.use_subtype_feature:
        args.subtype_dim = 0
    if args.max_queries_per_task <= 0:
        args.max_queries_per_task = None
    return args


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    task_cols = base.parse_column_list(args.serum_task_cols)
    label_columns = (
        args.query_label_col,
        args.homo_label_col,
        args.distance_label_col,
    )
    sample_limit = None if args.sample_limit < 0 else int(args.sample_limit)
    frames = base.load_fixed_split_frames(
        args.data_dir,
        sample_limit=sample_limit,
        refit_train_valid=args.refit_train_valid,
        type_filter=args.type_filter,
        label_columns=label_columns,
    )
    base.validate_aligned_ha_columns(frames)
    embedding_files = base.required_embedding_files(
        frames,
        include_na_embeddings=False,
    )
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
        name: base.build_loader(
            frame,
            vocabs,
            embeddings,
            args.batch_size,
            shuffle=(name == "train"),
            max_queries_per_task=args.max_queries_per_task,
            gpu_cache=gpu_cache,
            device=device,
            task_cols=task_cols,
            query_label_col=args.query_label_col,
            homo_label_col=args.homo_label_col,
            distance_label_col=args.distance_label_col,
            align_ha_embeddings=True,
            include_na_embeddings=False,
        )
        for name, frame in frames.items()
    }

    label_weight_values = base.parse_float_list(args.label_weight_values)
    label_weight_thresholds = base.resolve_label_weight_thresholds(
        frames,
        mode=args.label_weight_threshold_mode,
        thresholds_arg=args.label_weight_thresholds,
        quantiles_arg=args.label_weight_quantiles,
        label_col=args.distance_label_col,
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
        score_log_var_mode=args.score_log_var_mode,
    )
    model = PairSiteAttentionMinusModel(model_config).to(device)
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
        "model": "SerumGate-Minus-PairSiteAttn-Homo",
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
                base.SerumGateTaskDataset(
                    frame,
                    vocabs,
                    task_cols,
                    max_queries_per_task=args.max_queries_per_task,
                    query_label_col=args.query_label_col,
                    homo_label_col=args.homo_label_col,
                    distance_label_col=args.distance_label_col,
                )
            )
            for name, frame in frames.items()
        },
        "max_queries_per_task": args.max_queries_per_task,
        "gpu_cache_gb": args.gpu_cache_gb,
        "embedding_token_policy": "strip_boundary_special_tokens_when_present",
        "query_label_col": args.query_label_col,
        "homo_label_col": args.homo_label_col,
        "distance_label_col": args.distance_label_col,
        "distance_loss_weight": args.distance_loss_weight,
        "homo_loss_weight": args.homo_loss_weight,
        "query_loss_weight": args.query_loss_weight,
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
            train_loss = base.train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                args.loss,
                args.distance_loss_weight,
                args.homo_loss_weight,
                args.query_loss_weight,
                progress_bar=progress_bar,
            )
            if args.refit_train_valid:
                valid_metrics = base.serum_homo_regression_metrics(pd.DataFrame())
                valid_metrics["loss"] = 0.0
                valid_preds = pd.DataFrame()
            else:
                progress_bar.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    split="valid",
                )
                valid_metrics, valid_preds = base.evaluate_model(
                    model,
                    loaders["valid"],
                    device,
                    args.loss,
                    args.distance_loss_weight,
                    args.homo_loss_weight,
                    args.query_loss_weight,
                    progress_bar=progress_bar,
                )
            if args.skip_test_eval:
                test_metrics = base.serum_homo_regression_metrics(pd.DataFrame())
                test_metrics["loss"] = 0.0
                test_preds = pd.DataFrame()
            else:
                progress_bar.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}",
                    split="test",
                )
                test_metrics, test_preds = base.evaluate_model(
                    model,
                    loaders["test"],
                    device,
                    args.loss,
                    args.distance_loss_weight,
                    args.homo_loss_weight,
                    args.query_loss_weight,
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
                        "query_label_col": args.query_label_col,
                        "homo_label_col": args.homo_label_col,
                        "distance_label_col": args.distance_label_col,
                        "distance_loss_weight": args.distance_loss_weight,
                        "homo_loss_weight": args.homo_loss_weight,
                        "query_loss_weight": args.query_loss_weight,
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
