"""Run paired gradient-times-input attribution and write token summaries."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fluprofiler.models.serum_gate_minus_all_attention_model import SerumGateMinusAllAttentionModel  # noqa: E402
from fluprofiler.models.serum_gate_model import SerumGateBatch  # noqa: E402

DEDUP_COLUMNS = ["seq_a", "seq_c", "serumPassCat", "virusPassCat"]


def clean(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def normalize_category(value: Any) -> str:
    text = clean(value).casefold()
    if text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    return text or "unknown"


def collect_samples(csv_path: Path, subtype: str = "H3N2") -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {path}")
    frame = pd.read_csv(path)
    type_column = "serumType" if "serumType" in frame else "Type" if "Type" in frame else None
    if type_column is None:
        raise ValueError("Dataset must contain serumType or Type")
    required = {type_column, "seq_id_a", "seq_id_c", *DEDUP_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")
    rows = frame[frame[type_column].map(clean).str.casefold().eq(subtype.strip().casefold())].copy()
    rows = rows[
        rows["seq_a"].map(clean).str.len().eq(329)
        & rows["seq_c"].map(clean).str.len().eq(329)
    ]
    rows = rows.drop_duplicates(DEDUP_COLUMNS, keep="first").reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No eligible paired samples found for subtype {subtype!r}")
    if rows["seq_id_a"].map(clean).eq("").any() or rows["seq_id_c"].map(clean).eq("").any():
        raise ValueError("Filtered rows must contain seq_id_a and seq_id_c")
    rows.insert(0, "sample_index", np.arange(len(rows), dtype=np.int64))
    return rows


def load_model(path: Path, device: str | torch.device) -> tuple[SerumGateMinusAllAttentionModel, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_config", "model_state_dict", "passage_to_id"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint must contain: {', '.join(sorted(required))}")
    model = SerumGateMinusAllAttentionModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(target).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def category_ids(values: Iterable[Any], vocabulary: dict[str, int]) -> list[int]:
    normalized = {normalize_category(key): int(value) for key, value in vocabulary.items()}
    unknown = normalized.get("unknown", 0)
    return [normalized.get(normalize_category(value), unknown) for value in values]


def load_matrix(path: Path, shape: tuple[int, int]) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding: {path}")
    matrix = torch.as_tensor(torch.load(path, map_location="cpu", weights_only=False)).float()
    if tuple(matrix.shape) != shape:
        raise ValueError(f"{path.name} must have shape {shape}, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{path.name} contains non-finite values")
    return matrix


def attribute(
    model: SerumGateMinusAllAttentionModel,
    checkpoint: dict[str, Any],
    rows: pd.DataFrame,
    embedding_dir: Path,
    device: str | torch.device,
    batch_size: int,
    token_count: int,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    folder = Path(embedding_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {folder}")
    target = torch.device(device)
    shape = (token_count, int(model.config.hidden_size))
    vocab = dict(checkpoint["passage_to_id"])
    subtype_vocab = dict(checkpoint.get("subtype_to_id", {"constant": 0}))
    subtype_id = int(subtype_vocab.get("constant", subtype_vocab.get("h3n2", 0)))
    output: dict[str, list[np.ndarray]] = {key: [] for key in ("reference", "query", "mean", "self_score", "query_score")}

    for start in range(0, len(rows), batch_size):
        part = rows.iloc[start : start + batch_size]
        reference = torch.stack([load_matrix(folder / f"matrix_{clean(x)}.pt", shape) for x in part["seq_id_a"]]).to(target)
        query = torch.stack([load_matrix(folder / f"matrix_{clean(x)}.pt", shape) for x in part["seq_id_c"]]).to(target)
        reference.requires_grad_(True)
        query.requires_grad_(True)
        size = len(part)
        serum_passage = torch.tensor(category_ids(part["serumPassCat"], vocab), dtype=torch.long, device=target)
        query_passage = torch.tensor(category_ids(part["virusPassCat"], vocab), dtype=torch.long, device=target).view(size, 1)
        passage_pair = serum_passage.view(size, 1) * int(model.config.passage_vocab_size) + query_passage
        mask = torch.ones(size, token_count, dtype=torch.float32, device=target)
        batch = SerumGateBatch(
            reference_ha=reference,
            query_ha=query[:, None],
            reference_ha_mask=mask,
            query_ha_mask=mask[:, None],
            serum_passage=serum_passage,
            query_passage=query_passage,
            passage_pair=passage_pair,
            subtype=torch.full((size,), subtype_id, dtype=torch.long, device=target),
            s_nagly=torch.zeros(size, 1, device=target),
            query_mask=torch.ones(size, 1, device=target),
        )
        prediction = model(batch)
        grad_reference, grad_query = torch.autograd.grad(prediction["mean"].sum(), (reference, query))
        output["reference"].append((grad_reference * reference).sum(-1).detach().cpu().numpy().astype(np.float64))
        output["query"].append((grad_query * query).sum(-1).detach().cpu().numpy().astype(np.float64))
        for key in ("mean", "self_score", "query_score"):
            output[key].append(prediction[key].detach().cpu().reshape(-1).numpy().astype(np.float64))

    result = {key: np.concatenate(parts, axis=0) for key, parts in output.items()}
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Attribution output contains non-finite values")
    return result


def summarize(reference: np.ndarray, query: np.ndarray) -> pd.DataFrame:
    reference = np.asarray(reference, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if reference.shape != query.shape or reference.ndim != 2 or not len(reference):
        raise ValueError("Attribution arrays must have the same nonempty 2D shape")
    frames = []
    for side, values in (("reference", reference), ("query", query), ("combined", reference + query)):
        indices = np.arange(values.shape[1])
        quantiles = np.quantile(values, [0.05, 0.25, 0.75, 0.95], axis=0)
        frames.append(pd.DataFrame({
            "side": side,
            "token_index": indices,
            "token_label": ["BOS" if i == 0 else "EOS" if i == len(indices) - 1 else f"AA_{i}" for i in indices],
            "aa_position": pd.array([None if i in (0, len(indices) - 1) else i for i in indices], dtype="Int64"),
            "count": len(values),
            "mean": values.mean(0),
            "median": np.median(values, 0),
            "std": values.std(0),
            "min": values.min(0),
            "max": values.max(0),
            "q05": quantiles[0], "q25": quantiles[1], "q75": quantiles[2], "q95": quantiles[3],
            "mean_abs": np.abs(values).mean(0),
            "median_abs": np.median(np.abs(values), 0),
            "positive_fraction": (values > 0).mean(0),
        }))
    return pd.concat(frames, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    rows = collect_samples(args.data_csv, args.type_filter)
    if args.expected_count and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} samples, found {len(rows)}")
    model, checkpoint = load_model(args.checkpoint, args.device)
    values = attribute(model, checkpoint, rows, args.embedding_dir, args.device, args.batch_size, args.token_count)
    site_summary = summarize(values["reference"], values["query"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        columns = [name for name in (
            "sample_index", "seq_id_a", "seq_id_c", "seq_a", "seq_c", "serumPassCat",
            "virusPassCat", "serumName", "virusName", "label", "serumType", "Type",
        ) if name in rows]
        samples = rows[columns].copy()
        for key in ("mean", "self_score", "query_score"):
            samples[key] = values[key]
        samples["reference_attribution_sum"] = values["reference"].sum(1)
        samples["query_attribution_sum"] = values["query"].sum(1)
        samples = pd.concat([
            samples,
            pd.DataFrame(values["reference"], columns=[f"reference_token_{i}" for i in range(args.token_count)]),
            pd.DataFrame(values["query"], columns=[f"query_token_{i}" for i in range(args.token_count)]),
        ], axis=1)
        samples.to_csv(temporary / "attribution_by_sample.csv", index=False, float_format="%.9g")
        site_summary.to_csv(temporary / "site_summary.csv", index=False, float_format="%.9g")
        np.savez_compressed(temporary / "attribution_arrays.npz", **values)
        config = {
            "method": "gradient_x_input",
            "target": "mean = self_score - query_score",
            "hidden_reduction": "sum over embedding dimension",
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "data_csv": str(Path(args.data_csv).expanduser().resolve()),
            "embedding_dir": str(Path(args.embedding_dir).expanduser().resolve()),
            "subtype_column_priority": ["serumType", "Type"],
            "type_filter": args.type_filter,
            "sequence_length": 329,
            "embedding_shape": [args.token_count, int(model.config.hidden_size)],
            "deduplication_columns": DEDUP_COLUMNS,
            "sample_count": len(rows),
            "device": str(args.device),
            "batch_size": args.batch_size,
            "token_mapping": {"0": "BOS", "1-329": "aligned positions", "330": "EOS"},
        }
        with (temporary / "analysis_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"sample_count": len(rows), "token_count": args.token_count, "output_dir": str(destination)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--data-csv", type=Path, required=True)
    result.add_argument("--embedding-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--type", dest="type_filter", default="H3N2")
    result.add_argument("--device", default="cpu")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--token-count", type=int, default=331)
    result.add_argument("--expected-count", type=int, default=0)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
