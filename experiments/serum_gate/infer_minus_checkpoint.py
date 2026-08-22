#!/usr/bin/env python3
"""
Run inference with a saved SerumGate-Minus checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
sys.path.append(str(_REPO_ROOT / "src"))
sys.path.append(str(_THIS_FILE.parent))
sys.path.append(str(_REPO_ROOT / "experiments" / "reverse_tests"))

from fluprofiler.models.serum_gate_minus_model import SerumGateMinusModel  # noqa: E402
from train_zero_shot_minus import (  # noqa: E402
    GpuEmbeddingCache,
    SerumGateVocabs,
    build_loader,
    load_embeddings,
    move_batch,
    normalize_subtype,
    parse_column_list,
    required_embedding_files,
    serum_regression_metrics,
    subtype_column,
    validate_embedding_files,
    write_rounded_predictions,
)


DEFAULT_CHECKPOINT = Path(
    "/home/chenyh/workspace/fluProfiler/results/H1H3_HA1/"
    "SerumGate-Minus-latent8/serum/subtype/H1N1/checkpoints/best_model.pth"
)
DEFAULT_EMBEDDING_DIR = _REPO_ROOT / "data" / "embedding" / "files"
FALLBACK_DATA_DIR = _REPO_ROOT / "data" / "dataset" / "H1H3_HA1" / "splited" / "20260710_211620" / "serum"
SPLIT_NAMES = ("train", "valid", "test")
REQUIRED_INFERENCE_COLUMNS = {
    "seq_id_a",
    "seq_id_b",
    "seq_id_c",
    "seq_id_d",
    "seq_b",
    "seq_d",
    "serumPassCat",
    "virusPassCat",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with a SerumGate-Minus checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-csv", type=Path, help="Single CSV to score.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing train.csv/valid.csv/test.csv.")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--embedding-dir", type=Path, help="Directory containing matrix_<seq_id>.pt files.")
    parser.add_argument("--output-csv", type=Path, help="Prediction CSV path.")
    parser.add_argument("--metrics-json", type=Path, help="Optional metrics JSON path when labels are present.")
    parser.add_argument(
        "--type",
        dest="type_filter",
        default="auto",
        help='Type/virusType to keep. Default "auto" uses checkpoint run_config.json; pass "" to disable.',
    )
    parser.add_argument(
        "--serum-task-cols",
        type=str,
        default="",
        help="Comma-separated task key columns. Default uses checkpoint task_cols.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-queries-per-task", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gpu-cache-gb", type=float, default=0.0)
    parser.add_argument("--round-decimals", type=int, default=4)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.input_csv is not None and args.data_dir is not None:
        parser.error("Use either --input-csv or --data-dir, not both.")
    if args.batch_size != 1:
        parser.error("--batch-size must be 1 for variable-size SerumGate tasks.")
    if args.max_queries_per_task <= 0:
        args.max_queries_per_task = None
    return args


def checkpoint_root(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_run_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_root(checkpoint_path) / "run_config.json"
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_type_filter(value: str, run_config: dict[str, Any]) -> str:
    if value != "auto":
        return value
    return str(run_config.get("type_filter") or "")


def resolve_embedding_dir(args: argparse.Namespace, run_config: dict[str, Any]) -> Path:
    if args.embedding_dir is not None:
        return args.embedding_dir.expanduser().resolve()
    config_value = run_config.get("embedding_dir")
    if config_value:
        config_path = Path(config_value).expanduser()
        if config_path.is_dir():
            return config_path.resolve()
    return DEFAULT_EMBEDDING_DIR.resolve()


def resolve_input_path(args: argparse.Namespace, run_config: dict[str, Any]) -> Path:
    if args.input_csv is not None:
        return args.input_csv.expanduser().resolve()
    data_dir = args.data_dir
    if data_dir is None:
        config_value = run_config.get("data_dir")
        if config_value and Path(config_value).expanduser().is_dir():
            data_dir = Path(config_value)
        elif FALLBACK_DATA_DIR.is_dir():
            data_dir = FALLBACK_DATA_DIR
    if data_dir is None:
        raise FileNotFoundError("Provide --input-csv or --data-dir; no usable data_dir was found.")
    return (data_dir.expanduser().resolve() / f"{args.split}.csv")


def default_output_csv(args: argparse.Namespace, input_path: Path) -> Path:
    if args.output_csv is not None:
        return args.output_csv.expanduser().resolve()
    if args.input_csv is not None:
        stem = input_path.stem
    else:
        stem = args.split
    return checkpoint_root(args.checkpoint) / f"inference_{stem}.csv"


def prepare_inference_frame(
    frame: pd.DataFrame,
    source: str | Path,
    type_filter: str = "",
) -> tuple[pd.DataFrame, bool]:
    missing = sorted(REQUIRED_INFERENCE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required column(s): {', '.join(missing)}")
    if "Type" not in frame.columns and "virusType" not in frame.columns:
        raise ValueError(f"{source} is missing required column(s): Type or virusType")

    prepared = frame.reset_index(drop=True).copy()
    if type_filter:
        target = normalize_subtype(type_filter)
        if target != "unknown":
            st_col = subtype_column(prepared)
            values = prepared[st_col].map(normalize_subtype)
            mask = values.astype(str).str.casefold() == target.casefold()
            observed = sorted(value for value in values.unique().tolist() if value != "unknown")
            prepared = prepared.loc[mask].reset_index(drop=True)
            if prepared.empty:
                available = ", ".join(observed) if observed else "none"
                raise ValueError(f"No rows found for --type {target!r}; available Type values: {available}")

    has_labels = "label" in prepared.columns
    if has_labels:
        prepared.loc[:, "label"] = pd.to_numeric(prepared["label"], errors="raise")
    else:
        prepared.loc[:, "label"] = 0.0
    return prepared, has_labels


def load_inference_frame(args: argparse.Namespace, run_config: dict[str, Any]) -> tuple[pd.DataFrame, bool, Path, str]:
    input_path = resolve_input_path(args, run_config)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    type_filter = resolve_type_filter(args.type_filter, run_config)
    frame, has_labels = prepare_inference_frame(frame, source=input_path, type_filter=type_filter)
    if args.sample_limit >= 0:
        frame = frame.iloc[: args.sample_limit].copy().reset_index(drop=True)
    return frame, has_labels, input_path, type_filter


def task_columns(args: argparse.Namespace, checkpoint: dict[str, Any]) -> list[str]:
    if args.serum_task_cols:
        return parse_column_list(args.serum_task_cols)
    return list(checkpoint.get("task_cols") or ["seq_id_a", "seq_id_b", "serumPassCat"])


def predict_model(
    model: SerumGateMinusModel,
    loader,
    device: torch.device,
    show_progress: bool = True,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(loader),
        desc="inference",
        disable=not show_progress,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    with progress:
        with torch.no_grad():
            for batch, items in loader:
                item = items[0]
                batch = replace(move_batch(batch, device), labels=None)
                out = model(batch)
                mean = out["mean"].detach().cpu().view(-1).numpy()
                log_var = out["log_var"].detach().cpu().view(-1).numpy()
                self_score = out["self_score"].detach().cpu().view(-1).numpy()
                query_score = out["query_score"].detach().cpu().view(-1).numpy()
                self_log_var = out["self_log_var"].detach().cpu().view(-1).numpy()
                query_log_var = out["query_log_var"].detach().cpu().view(-1).numpy()
                row_indices = item["row_indices"].detach().cpu().view(-1).numpy()
                meta = item["query_meta"]
                for idx in range(len(mean)):
                    row = meta.iloc[idx].to_dict()
                    row.update(
                        {
                            "_infer_row_order": int(row_indices[idx]),
                            "task_key": item["task_key"],
                            "mean": float(mean[idx]),
                            "log_var": float(log_var[idx]),
                            "std": float(math.exp(0.5 * float(log_var[idx]))),
                            "self_score": float(self_score[idx]),
                            "query_score": float(query_score[idx]),
                            "self_log_var": float(self_log_var[idx]),
                            "query_log_var": float(query_log_var[idx]),
                        }
                    )
                    rows.append(row)
                progress.update(1)
    predictions = pd.DataFrame(rows)
    if "_infer_row_order" in predictions.columns:
        predictions = (
            predictions.sort_values("_infer_row_order", kind="stable")
            .drop(columns=["_infer_row_order"])
            .reset_index(drop=True)
        )
    return predictions


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    run_config = load_run_config(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    frame, has_labels, input_path, type_filter = load_inference_frame(args, run_config)
    embedding_dir = resolve_embedding_dir(args, run_config)
    model_config = dict(checkpoint["model_config"])
    include_na_embeddings = model_config.get("na_branch", "none") != "none"
    embedding_files = required_embedding_files({"input": frame}, include_na_embeddings=include_na_embeddings)
    embedding_dir = validate_embedding_files(embedding_dir, embedding_files)
    embeddings = load_embeddings(embedding_dir, embedding_files, show_progress=args.progress)

    device = torch.device(args.device)
    gpu_cache = None
    if args.gpu_cache_gb > 0:
        if device.type != "cuda":
            raise ValueError("--gpu-cache-gb requires a CUDA device")
        gpu_cache = GpuEmbeddingCache(
            cpu_store=embeddings,
            device=device,
            max_bytes=int(float(args.gpu_cache_gb) * 1024**3),
        )

    loader = build_loader(
        frame,
        SerumGateVocabs(
            passage_to_id=checkpoint["passage_to_id"],
            subtype_to_id=checkpoint["subtype_to_id"],
        ),
        embeddings,
        args.batch_size,
        shuffle=False,
        max_queries_per_task=args.max_queries_per_task,
        gpu_cache=gpu_cache,
        device=device,
        task_cols=task_columns(args, checkpoint),
        align_ha_embeddings=(model_config.get("ha_pair_mode", "independent") == "delta"),
        include_na_embeddings=include_na_embeddings,
    )
    model = SerumGateMinusModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict_model(model.to(device), loader, device, show_progress=args.progress)
    metrics = serum_regression_metrics(predictions) if has_labels else {}
    if not has_labels and "label" in predictions.columns:
        predictions = predictions.drop(columns=["label"])

    output_csv = default_output_csv(args, input_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.round_decimals >= 0:
        write_rounded_predictions(predictions, output_csv, decimals=args.round_decimals)
    else:
        predictions.to_csv(output_csv, index=False)

    metrics_path = args.metrics_json.expanduser().resolve() if args.metrics_json is not None else None
    if metrics and metrics_path is None:
        metrics_path = output_csv.with_suffix(".metrics.json")
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "input_csv": str(input_path),
                    "type_filter": type_filter,
                    "rows": int(len(predictions)),
                    "metrics": metrics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return {
        "checkpoint": str(checkpoint_path),
        "input_csv": str(input_path),
        "output_csv": str(output_csv),
        "metrics_json": str(metrics_path) if metrics_path is not None else None,
        "rows": int(len(predictions)),
        "labels_available": bool(has_labels),
        "type_filter": type_filter,
    }


def main() -> None:
    result = run_inference(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
