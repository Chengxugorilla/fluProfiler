"""Analyze token-level pooling attention from a SerumGate all-attention model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_SRC_DIR = str(_REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fluprofiler.models.serum_gate_minus_all_attention_model import (  # noqa: E402
    SerumGateMinusAllAttentionModel,
)


def collect_unique_virus_sequences(csv_path: Path, subtype: str) -> pd.DataFrame:
    """Collect one deterministic record per virus-side HA sequence ID."""
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV does not exist: {csv_path}")

    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    subtype_column = "Type" if "Type" in columns else "virusType" if "virusType" in columns else None
    if subtype_column is None:
        raise ValueError("Dataset must contain Type or virusType")
    required = {subtype_column, "seq_id_c", "seq_c", "virusHA"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing)}")

    metadata_columns = [name for name in ("virusName", "virusDate", "virusPassCat") if name in columns]
    usecols = [subtype_column, "seq_id_c", "seq_c", "virusHA", *metadata_columns]
    target = str(subtype).strip().casefold()
    records: dict[str, dict[str, object]] = {}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=10_000):
        chunk = chunk[chunk[subtype_column].astype(str).str.strip().str.casefold() == target]
        for row in chunk.to_dict(orient="records"):
            seq_id = str(row["seq_id_c"]).strip()
            seq_c = str(row["seq_c"]).strip()
            sequence = str(row["virusHA"]).strip()
            if not seq_id or seq_id.casefold() == "nan" or not seq_c or seq_c.casefold() == "nan":
                raise ValueError("Filtered rows must contain seq_id_c and seq_c")
            if len(seq_c) != 329:
                continue
            if not sequence or sequence.casefold() == "nan":
                raise ValueError("Filtered rows must contain virusHA")
            if seq_id in records:
                if records[seq_id]["virusHA"] != sequence:
                    raise ValueError(f"{seq_id} maps to multiple virusHA sequences")
                records[seq_id]["occurrence_count"] = int(records[seq_id]["occurrence_count"]) + 1
                continue
            record: dict[str, object] = {
                "seq_id_c": seq_id,
                "seq_c": seq_c,
                "virusHA": sequence,
                "occurrence_count": 1,
            }
            for column in metadata_columns:
                record[column] = row[column]
            records[seq_id] = record

    if not records:
        raise ValueError(f"No virus HA sequences found for subtype {subtype!r}")
    return pd.DataFrame(records.values()).reset_index(drop=True)


def validate_attention(
    attention: np.ndarray,
    expected_rows: int,
    token_count: int,
) -> dict[str, float | int]:
    """Validate a token-attention probability matrix."""
    values = np.asarray(attention)
    expected_shape = (int(expected_rows), int(token_count))
    if values.shape != expected_shape:
        raise ValueError(f"Attention shape must be {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Attention contains non-finite values")
    if (values < 0).any():
        raise ValueError("Attention contains negative values")
    row_sum_error = np.abs(values.sum(axis=1) - 1.0)
    max_row_sum_error = float(row_sum_error.max(initial=0.0))
    if max_row_sum_error > 1e-6:
        raise ValueError(f"Attention rows must sum to one; maximum error is {max_row_sum_error:.3g}")
    return {
        "row_count": int(values.shape[0]),
        "token_count": int(values.shape[1]),
        "minimum_attention": float(values.min()),
        "maximum_attention": float(values.max()),
        "max_row_sum_error": max_row_sum_error,
    }


def summarize_attention(
    attention: np.ndarray,
    token_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize equally weighted per-sequence token attention."""
    values = np.asarray(attention, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != token_count or values.shape[0] == 0:
        raise ValueError(f"attention must have nonzero shape [sequences, {token_count}]")

    token_index = np.arange(token_count)
    token_labels = ["BOS", *[f"AA_{idx}" for idx in range(1, token_count - 1)], "EOS"]
    token_types = ["special", *(["amino_acid"] * (token_count - 2)), "special"]
    aa_positions: list[int | None] = [None, *range(1, token_count - 1), None]
    top1_counts = np.bincount(np.argmax(values, axis=1), minlength=token_count)
    top_n = min(5, token_count)
    top_indices = np.argsort(values, axis=1, kind="stable")[:, -top_n:]
    top5_counts = np.bincount(top_indices.ravel(), minlength=token_count)
    quantiles = np.quantile(values, [0.05, 0.25, 0.75, 0.95], axis=0)

    summary = pd.DataFrame(
        {
            "token_index": token_index,
            "token_label": token_labels,
            "token_type": token_types,
            "aa_position": pd.array(aa_positions, dtype="Int64"),
            "count": values.shape[0],
            "mean": values.mean(axis=0),
            "median": np.median(values, axis=0),
            "std": values.std(axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "q05": quantiles[0],
            "q25": quantiles[1],
            "q75": quantiles[2],
            "q95": quantiles[3],
            "top1_count": top1_counts,
            "top5_count": top5_counts,
        }
    )
    ranked = summary.sort_values(["mean", "token_index"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, token_count + 1))
    return summary, ranked


def load_model(
    checkpoint_path: Path,
    device: str | torch.device,
) -> tuple[SerumGateMinusAllAttentionModel, dict[str, Any]]:
    """Strictly load an all-attention checkpoint on the requested device."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if resolved_device.index is not None and resolved_device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {resolved_device}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_config and model_state_dict")
    model = SerumGateMinusAllAttentionModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def extract_attention(
    model: SerumGateMinusAllAttentionModel,
    records: pd.DataFrame,
    embedding_dir: Path,
    device: str | torch.device,
    batch_size: int,
    token_count: int,
) -> np.ndarray:
    """Extract final pooling attention for each unique virus HA embedding."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    embedding_dir = Path(embedding_dir).expanduser().resolve()
    if not embedding_dir.is_dir():
        raise FileNotFoundError(f"Embedding directory does not exist: {embedding_dir}")
    resolved_device = torch.device(device)
    expected_shape = (int(token_count), int(model.config.hidden_size))
    attention_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            tensors: list[torch.Tensor] = []
            for seq_id in records.iloc[start : start + batch_size]["seq_id_c"]:
                embedding_path = embedding_dir / f"matrix_{seq_id}.pt"
                if not embedding_path.is_file():
                    raise FileNotFoundError(f"Missing embedding: {embedding_path}")
                value = torch.load(embedding_path, map_location="cpu", weights_only=False)
                tensor = torch.as_tensor(value).float()
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"{embedding_path.name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
                    )
                tensors.append(tensor)
            batch = torch.stack(tensors, dim=0).to(resolved_device)
            mask = torch.ones(batch.shape[:2], dtype=torch.bool, device=resolved_device)
            _, weights = model.score_model.ha_encoder.encode_with_attention(batch, mask)
            attention_batches.append(weights.detach().cpu().numpy().astype(np.float64, copy=False))

    return np.concatenate(attention_batches, axis=0)


def _write_plots(
    attention: np.ndarray,
    summary: pd.DataFrame,
    ranked: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> None:
    token_index = summary["token_index"].to_numpy(dtype=np.float64)
    mean = summary["mean"].to_numpy(dtype=np.float64)
    q25 = summary["q25"].to_numpy(dtype=np.float64)
    q75 = summary["q75"].to_numpy(dtype=np.float64)

    width, height = 1600, 600
    left, right, top_margin, bottom = 90, 40, 60, 75
    plot_width = width - left - right
    plot_height = height - top_margin - bottom
    y_max = max(float(q75.max()), float(mean.max()), 1e-12) * 1.05

    def profile_point(index: float, value: float) -> tuple[float, float]:
        x = left + index / max(1.0, token_index[-1]) * plot_width
        y = top_margin + (1.0 - value / y_max) * plot_height
        return x, y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), "Mean H3N2 virus-HA token attention", fill="black")
    draw.line(
        [(left, top_margin), (left, top_margin + plot_height), (left + plot_width, top_margin + plot_height)],
        fill="black",
        width=2,
    )
    band = [profile_point(idx, value) for idx, value in zip(token_index, q25)]
    band += [profile_point(idx, value) for idx, value in zip(token_index[::-1], q75[::-1])]
    draw.polygon(band, fill=(210, 225, 245))
    draw.line(
        [profile_point(idx, value) for idx, value in zip(token_index, mean)],
        fill=(25, 85, 160),
        width=3,
    )
    for idx in (0, len(mean) - 1):
        x, y = profile_point(float(idx), float(mean[idx]))
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
    for tick in list(range(0, len(mean), 50)) + [len(mean) - 1]:
        x, _ = profile_point(float(tick), 0.0)
        draw.line([(x, top_margin + plot_height), (x, top_margin + plot_height + 6)], fill="black")
        draw.text((x - 10, top_margin + plot_height + 10), str(tick), fill="black")
    for fraction in np.linspace(0.0, 1.0, 5):
        y = top_margin + (1.0 - fraction) * plot_height
        draw.line([(left - 6, y), (left, y)], fill="black")
        draw.text((5, y - 7), f"{fraction * y_max:.4g}", fill="black")
    draw.text((width // 2 - 35, height - 28), "Token index", fill="black")
    draw.text((8, 42), "Attention", fill="black")
    image.save(output_dir / "mean_attention_profile.png")

    top_frame = ranked.head(max(1, min(int(top_k), len(ranked)))).iloc[::-1]
    bar_width = 1200
    bar_height = max(500, 100 + len(top_frame) * 30)
    image = Image.new("RGB", (bar_width, bar_height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((140, 20), f"Top {len(top_frame)} tokens by mean attention", fill="black")
    maximum = max(float(top_frame["mean"].max()), 1e-12)
    chart_left, chart_right = 140, bar_width - 50
    for row_index, (_, row) in enumerate(top_frame.iterrows()):
        y = 55 + row_index * 30
        length = float(row["mean"]) / maximum * (chart_right - chart_left)
        draw.text((20, y + 4), str(row["token_label"]), fill="black")
        draw.rectangle((chart_left, y, chart_left + length, y + 20), fill=(40, 120, 180))
        draw.text((chart_left + length + 6, y + 4), f"{float(row['mean']):.5g}", fill="black")
    image.save(output_dir / "top_tokens.png")

    centers = attention @ np.arange(attention.shape[1], dtype=np.float64)
    order = np.argsort(centers, kind="stable")
    ordered = attention[order]
    maximum = max(float(ordered.max()), 1e-12)
    normalized = np.clip(ordered / maximum, 0.0, 1.0)
    anchors = np.asarray(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
        dtype=np.float64,
    )
    scaled = normalized * (len(anchors) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - fraction) + anchors[upper] * fraction
    heatmap = Image.fromarray(rgb.astype(np.uint8)).resize((1500, 900), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (1700, 1020), "white")
    canvas.paste(heatmap, (130, 55))
    draw = ImageDraw.Draw(canvas)
    draw.text((130, 20), "H3N2 token attention heatmap", fill="black")
    draw.text((760, 975), "Token index (0-330)", fill="black")
    draw.text((10, 55), "Unique virus HA", fill="black")
    draw.text((130, 958), "Sequences sorted by attention center; color scaled to global maximum", fill="black")
    canvas.save(output_dir / "attention_heatmap.png")


def run_analysis(args: argparse.Namespace) -> dict[str, object]:
    """Run the complete unique-virus token-attention analysis."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if args.expected_count < 0:
        raise ValueError("expected_count must be non-negative")
    if args.token_count < 3:
        raise ValueError("token_count must include BOS, at least one residue, and EOS")

    records = collect_unique_virus_sequences(args.data_csv, args.type_filter)
    if args.expected_count and len(records) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} unique sequences, found {len(records)}")

    model, checkpoint = load_model(args.checkpoint, args.device)
    attention = extract_attention(
        model=model,
        records=records,
        embedding_dir=args.embedding_dir,
        device=args.device,
        batch_size=args.batch_size,
        token_count=args.token_count,
    )
    validation = validate_attention(attention, len(records), args.token_count)
    summary, ranked = summarize_attention(attention, args.token_count)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        token_columns = [f"token_{idx}" for idx in range(args.token_count)]
        attention_frame = pd.concat(
            [
                records.reset_index(drop=True),
                pd.DataFrame(attention, columns=token_columns),
            ],
            axis=1,
        )
        attention_frame.to_csv(
            temporary_dir / "attention_by_sequence.csv",
            index=False,
            float_format="%.10g",
        )
        summary.to_csv(
            temporary_dir / "token_summary.csv",
            index=False,
            float_format="%.10g",
        )
        ranked.to_csv(
            temporary_dir / "top_tokens.csv",
            index=False,
            float_format="%.10g",
        )
        _write_plots(attention, summary, ranked, temporary_dir, args.top_k)

        config_payload = {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "model_config": checkpoint["model_config"],
            "data_csv": str(Path(args.data_csv).expanduser().resolve()),
            "embedding_dir": str(Path(args.embedding_dir).expanduser().resolve()),
            "output_dir": str(output_dir),
            "type_filter": args.type_filter,
            "device": str(torch.device(args.device)),
            "batch_size": int(args.batch_size),
            "sequence_count": int(len(records)),
            "token_count": int(args.token_count),
            "token_mapping": {
                "0": "BOS",
                f"1-{args.token_count - 2}": "HA amino-acid positions",
                str(args.token_count - 1): "EOS",
            },
            "validation": validation,
        }
        (temporary_dir / "analysis_config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    return {
        "output_dir": str(output_dir),
        "sequence_count": int(len(records)),
        "token_count": int(args.token_count),
        "validation": validation,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--type", dest="type_filter", default="H3N2")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-count", type=int, default=331)
    parser.add_argument("--expected-count", type=int, default=0, help="Optional exact count check; 0 disables it")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


def main() -> None:
    result = run_analysis(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
