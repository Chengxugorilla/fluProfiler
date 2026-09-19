# Nextflu Serum Temporary Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a sibling `serum_temp` fixed-split dataset whose `label` field contains the source `diff_label` values for legacy Nextflu compatibility.

**Architecture:** Stream each source CSV through Python's structured CSV reader into a staging directory, replacing only the `label` field with the exact `diff_label` field text. Validate every generated row and atomically rename the staging directory only after all files are complete.

**Tech Stack:** Python 3 standard library (`csv`, `filecmp`, `math`, `pathlib`, `shutil`)

## Global Constraints

- Do not modify the source `serum` directory or the Nextflu training code.
- Preserve every source row, its order, and all columns.
- Keep `diff_label` and all non-`label` values unchanged.
- Copy `manifest.json` byte-for-byte.
- Do not commit the generated temporary dataset to Git.

---

### Task 1: Generate And Verify The Temporary Dataset

**Files:**
- Read: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum/train.csv`
- Read: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum/valid.csv`
- Read: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum/test.csv`
- Read: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum/manifest.json`
- Create: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp/train.csv`
- Create: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp/valid.csv`
- Create: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp/test.csv`
- Create: `data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp/manifest.json`

**Interfaces:**
- Consumes: source split CSVs with `label` and numeric, non-null `diff_label` fields.
- Produces: a complete `serum_temp` directory accepted by `load_split_frames(data_dir: Path)` in `experiments/benchmark_pairwise/train_nextflu_fixed_split.py`.

- [ ] **Step 1: Verify the target does not already exist and the source is complete**

```bash
python -c '
from pathlib import Path

root = Path("data/dataset/H1H3_HA1_homo/splited/20260711_154610")
source = root / "serum"
target = root / "serum_temp"
stage = root / "serum_temp.building"
required = [source / name for name in ("train.csv", "valid.csv", "test.csv", "manifest.json")]
assert all(path.is_file() for path in required), required
assert not target.exists(), target
assert not stage.exists(), stage
print("preconditions passed")
'
```

Expected: `preconditions passed`.

- [ ] **Step 2: Stream the remapped CSVs into a staging directory**

```bash
python -c '
import csv
import math
import shutil
from pathlib import Path

root = Path("data/dataset/H1H3_HA1_homo/splited/20260711_154610")
source = root / "serum"
target = root / "serum_temp"
stage = root / "serum_temp.building"
assert not target.exists(), target
assert not stage.exists(), stage
stage.mkdir()

try:
    for split in ("train", "valid", "test"):
        source_path = source / f"{split}.csv"
        output_path = stage / f"{split}.csv"
        count = 0
        with source_path.open("r", encoding="utf-8", newline="") as source_file:
            reader = csv.DictReader(source_file)
            assert reader.fieldnames is not None
            missing = {"label", "diff_label"} - set(reader.fieldnames)
            if missing:
                raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")
            with output_path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
                writer.writeheader()
                for line_number, row in enumerate(reader, start=2):
                    diff_label = row["diff_label"]
                    normalized_label = diff_label.strip()
                    try:
                        is_valid = bool(normalized_label) and math.isfinite(float(normalized_label))
                    except ValueError:
                        is_valid = False
                    if not is_valid:
                        raise ValueError(f"{source_path}:{line_number} has invalid diff_label")
                    row["label"] = diff_label
                    writer.writerow(row)
                    count += 1
        print(f"{split}: wrote {count} rows")

    shutil.copy2(source / "manifest.json", stage / "manifest.json")
    stage.rename(target)
except BaseException:
    shutil.rmtree(stage, ignore_errors=True)
    raise
'
```

Expected row counts: `train: wrote 37065 rows`, `valid: wrote 4717 rows`, and `test: wrote 5579 rows`.

- [ ] **Step 3: Compare every output row with its source row**

```bash
python -c '
import csv
import filecmp
from itertools import zip_longest
from pathlib import Path

root = Path("data/dataset/H1H3_HA1_homo/splited/20260711_154610")
source = root / "serum"
target = root / "serum_temp"

for split in ("train", "valid", "test"):
    with (source / f"{split}.csv").open("r", encoding="utf-8", newline="") as source_file, (target / f"{split}.csv").open("r", encoding="utf-8", newline="") as target_file:
        source_reader = csv.DictReader(source_file)
        target_reader = csv.DictReader(target_file)
        assert source_reader.fieldnames == target_reader.fieldnames
        count = 0
        for line_number, pair in enumerate(zip_longest(source_reader, target_reader), start=2):
            source_row, target_row = pair
            assert source_row is not None and target_row is not None, (split, line_number)
            assert target_row["label"] == source_row["diff_label"], (split, line_number)
            for column in source_reader.fieldnames or []:
                if column != "label":
                    assert target_row[column] == source_row[column], (split, line_number, column)
            count += 1
    print(f"{split}: verified {count} rows")

assert filecmp.cmp(source / "manifest.json", target / "manifest.json", shallow=False)
print("manifest: identical")
'
```

Expected row counts match Step 2, followed by `manifest: identical`.

- [ ] **Step 4: Verify the legacy Nextflu loader accepts the result**

```bash
python -c '
import importlib.util
from pathlib import Path

script = Path("experiments/benchmark_pairwise/train_nextflu_fixed_split.py")
spec = importlib.util.spec_from_file_location("train_nextflu_fixed_split", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data_dir = Path("data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp")
frames = module.load_split_frames(data_dir)
assert {name: len(frame) for name, frame in frames.items()} == {"train": 37065, "valid": 4717, "test": 5579}
assert all(frame["label"].equals(frame["diff_label"]) for frame in frames.values())
print("Nextflu loader validation passed")
'
```

Expected: `Nextflu loader validation passed`.

- [ ] **Step 5: Leave the generated data uncommitted**

Run:

```bash
git status --short -- data/dataset/H1H3_HA1_homo/splited/20260711_154610/serum_temp
```

Expected: the temporary directory is untracked or ignored; do not add it to Git.
