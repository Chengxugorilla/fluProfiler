#!/usr/bin/env python3

from pathlib import Path
import argparse
import inspect
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


def configure_pickle_import_paths() -> None:
    for path in (ROOT / "src", ROOT / "src" / "fluprofiler"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


configure_pickle_import_paths()

from fluprofiler.data.loaders import load_embedding  # noqa: E402
from fluprofiler.models_v2 import BatchInput, ModelOutput  # noqa: E402


def passage_tokens(serum: str, virus: str) -> torch.Tensor:
    s = f"<cls>{serum}{virus}<eos>"
    for k, v in {"<cls>": "0", "<eos>": "1", "<EGG>": "2", "<CELL>": "3", "<BOTH>": "4", "<NONE>": "5"}.items():
        s = s.replace(k, v)
    return torch.tensor([int(ch) for ch in s], dtype=torch.long)


def predict_one_pair(model, matrices: dict[str, torch.Tensor], matrix_masks: dict[str, torch.Tensor], passage_tokens: torch.Tensor) -> float:
    params = inspect.signature(model.forward).parameters
    if "matrices_a" not in params:
        out = model(
            BatchInput(
                matrices=matrices,
                matrix_masks=matrix_masks,
                passage_tokens=passage_tokens,
                labels=None,
            )
        )
    else:
        ma = matrices["serum_HA"]
        mc = matrices["virus_HA"]
        mb = matrices.get("serum_NA", ma)
        md = matrices.get("virus_NA", mc)
        mk_a = matrix_masks["serum_HA"]
        mk_c = matrix_masks["virus_HA"]
        mk_b = matrix_masks.get("serum_NA", mk_a)
        mk_d = matrix_masks.get("virus_NA", mk_c)
        kwargs = {
            "matrices_a": ma,
            "matrix_attention_masks_a": mk_a,
            "matrices_b": mb,
            "matrix_attention_masks_b": mk_b,
            "matrices_c": mc,
            "matrix_attention_masks_c": mk_c,
            "matrices_d": md,
            "matrix_attention_masks_d": mk_d,
            "strainPassCats": passage_tokens,
            "labels": None,
        }
        out = model(**{key: value for key, value in kwargs.items() if key in params})

    pred = out.pred if hasattr(out, "pred") else out[-1]
    return float(torch.as_tensor(pred).view(-1)[0].cpu())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--embedding-dir", required=True)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    df = pd.read_csv(a.test_csv)
    for col in ("seq_id_a", "seq_id_c", "serumPassCat", "virusPassCat"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    has_b = "seq_id_b" in df.columns
    has_d = "seq_id_d" in df.columns

    ids = sorted(
        {str(v) for v in df["seq_id_a"].tolist()}
        | {str(v) for v in df["seq_id_c"].tolist()}
        | ({str(v) for v in df["seq_id_b"].tolist()} if has_b else set())
        | ({str(v) for v in df["seq_id_d"].tolist()} if has_d else set())
    )
    emb = load_embedding(a.embedding_dir, files=[f"matrix_{i}.pt" for i in ids])
    model = torch.load(a.model_path, map_location=a.device, weights_only=False)
    model.to(a.device).eval()

    preds = []
    with torch.no_grad():
        for _, r in df.iterrows():
            ma = emb[f"matrix_{r.seq_id_a}"].to(a.device).unsqueeze(0)
            mc = emb[f"matrix_{r.seq_id_c}"].to(a.device).unsqueeze(0)
            mb = emb[f"matrix_{r.seq_id_b}"].to(a.device).unsqueeze(0) if has_b else ma
            md = emb[f"matrix_{r.seq_id_d}"].to(a.device).unsqueeze(0) if has_d else mc
            mk_a = torch.ones(1, ma.shape[1], device=a.device, dtype=ma.dtype)
            mk_b = torch.ones(1, mb.shape[1], device=a.device, dtype=mb.dtype)
            mk_c = torch.ones(1, mc.shape[1], device=a.device, dtype=mc.dtype)
            mk_d = torch.ones(1, md.shape[1], device=a.device, dtype=md.dtype)
            pred = predict_one_pair(
                model=model,
                matrices={
                    "serum_HA": ma,
                    "serum_NA": mb,
                    "virus_HA": mc,
                    "virus_NA": md,
                },
                matrix_masks={
                    "serum_HA": mk_a,
                    "serum_NA": mk_b,
                    "virus_HA": mk_c,
                    "virus_NA": mk_d,
                },
                passage_tokens=passage_tokens(str(r.serumPassCat), str(r.virusPassCat)).unsqueeze(0).to(a.device),
            )
            preds.append(pred)

    df["prediction"] = preds
    out = Path(a.output_csv).expanduser().resolve() if a.output_csv else Path(a.test_csv).with_name(f"{Path(a.test_csv).stem}_with_prediction.csv")
    df.to_csv(out, index=False)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
