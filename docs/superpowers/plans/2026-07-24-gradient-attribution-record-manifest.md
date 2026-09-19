# 梯度归因 Record 对照表实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为 H3N2 Gradient × Input 归因 array 创建一张逐 record 对照表，使每个 `sample_index` 可稳定反查完整原始 meta 信息与模型预测值。

**架构：** 新增一个独立的 Python 脚本，读取归因结果目录中的 `analysis_config.json`、`attribution_arrays.npz` 和 `attribution_by_sample.csv`，再严格复现 `run_gradient_x_input.py::collect_samples` 的筛选和去重顺序。脚本生成 `attribution_records.csv`，并在写出前完成数组维度、meta 一致性、键唯一性和重读一致性验证。

**技术栈：** Python 3.10、Pandas、NumPy、JSON、Pytest。

## 全局约束

- 仅处理 `analysis_config.json["data_csv"]` 指定的源表；不得用 `train.csv` 替代。
- 结果表一行对应一个 array 第一维索引；第 `n` 行必须对应 `reference[n, :]` 与 `query[n, :]`。
- 严格执行 H3N2 筛选、329 长度过滤、`seq_a/seq_c/serumPassCat/virusPassCat` 首次出现去重及连续 `sample_index` 规则。
- 输出表保留全部源表字段，新增 `sample_index`、`source_row_index`、`record_key`，并附加 `mean`、`self_score`、`query_score`。
- 不重新计算归因，不修改 `.npz`，不展开 token 归因列，不修改 notebook。
- 所有验证失败必须抛出带具体原因的 `ValueError`，不得写出部分 CSV。

---

### Task 1：先定义源表重建与对照表的可测试接口

**文件：**
- 新建：`/home/chenyh/workspace/fluProfiler/tests/test_build_gradient_attribution_record_manifest.py`
- 新建：`/home/chenyh/workspace/fluProfiler/scripts/build_gradient_attribution_record_manifest.py`

**接口：**
- 输入：`reconstruct_records(source: pd.DataFrame, subtype: str) -> pd.DataFrame`。
- 输出：保留原始字段、`source_row_index`、连续 `sample_index` 与唯一 `record_key` 的 DataFrame。
- 输入：`build_manifest(result_dir: Path, output_path: Path | None = None) -> pd.DataFrame`。
- 输出：经过 array 与 meta 校验的完整对照表，并只在全部验证通过后写出 CSV。

- [ ] **Step 1：写失败测试，固定筛选、首次去重与顺序契约**

在 `tests/test_build_gradient_attribution_record_manifest.py` 写入：

```python
import pandas as pd

from scripts.build_gradient_attribution_record_manifest import reconstruct_records


def test_reconstruct_records_preserves_first_eligible_source_order():
    source = pd.DataFrame({
        "serumType": ["H1N1", " H3N2 ", "H3N2", "H3N2", "H3N2"],
        "seq_a": ["A" * 329, "A" * 329, "A" * 329, "A" * 328, "C" * 329],
        "seq_c": ["B" * 329, "B" * 329, "B" * 329, "B" * 329, "D" * 329],
        "serumPassCat": ["<EGG>"] * 5,
        "virusPassCat": ["<CELL>"] * 5,
        "seq_id_a": ["h1", "first", "duplicate", "short", "second"],
        "seq_id_c": ["c1", "c2", "c3", "c4", "c5"],
    })

    records = reconstruct_records(source, subtype="H3N2")

    assert records["source_row_index"].tolist() == [1, 4]
    assert records["sample_index"].tolist() == [0, 1]
    assert records["seq_id_a"].tolist() == ["first", "second"]
    assert records["record_key"].is_unique
```

- [ ] **Step 2：运行测试，确认因接口尚不存在而失败**

运行：

```bash
conda run -n fluProfiler python -m pytest tests/test_build_gradient_attribution_record_manifest.py::test_reconstruct_records_preserves_first_eligible_source_order -q
```

预期：测试因无法导入 `reconstruct_records` 而失败。

- [ ] **Step 3：实现最小的源表重建函数**

在 `scripts/build_gradient_attribution_record_manifest.py` 中实现：

```python
DEDUP_COLUMNS = ["seq_a", "seq_c", "serumPassCat", "virusPassCat"]


def clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def reconstruct_records(source: pd.DataFrame, subtype: str) -> pd.DataFrame:
    type_column = "serumType" if "serumType" in source else "Type"
    required = {type_column, "seq_id_a", "seq_id_c", *DEDUP_COLUMNS}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"源表缺少字段：{', '.join(missing)}")
    rows = source.copy()
    rows.insert(0, "source_row_index", rows.index.to_numpy(dtype="int64"))
    rows = rows[rows[type_column].map(clean).str.casefold().eq(subtype.strip().casefold())]
    rows = rows[rows["seq_a"].map(clean).str.len().eq(329) & rows["seq_c"].map(clean).str.len().eq(329)]
    rows = rows.drop_duplicates(DEDUP_COLUMNS, keep="first").reset_index(drop=True)
    rows.insert(0, "sample_index", range(len(rows)))
    rows.insert(2, "record_key", rows[DEDUP_COLUMNS].apply(lambda row: json.dumps([clean(value) for value in row], ensure_ascii=False, separators=(",", ":")), axis=1))
    if not rows["record_key"].is_unique:
        raise ValueError("去重后 record_key 不唯一")
    return rows
```

- [ ] **Step 4：重新运行单元测试，确认通过**

运行同一条 pytest 命令。

预期：`1 passed`。

### Task 2：实现归因文件对齐、输出冲突处理与真实数据验证

**文件：**
- 修改：`/home/chenyh/workspace/fluProfiler/scripts/build_gradient_attribution_record_manifest.py`
- 修改：`/home/chenyh/workspace/fluProfiler/tests/test_build_gradient_attribution_record_manifest.py`

**接口：**
- `build_manifest` 从 `result_dir/analysis_config.json` 获得源表路径与样本数。
- `build_manifest` 默认写入 `result_dir/attribution_records.csv`，返回写入前的 DataFrame。

- [ ] **Step 1：写失败测试，固定 array/meta 对齐与输出 schema**

向测试文件添加一个使用 `tmp_path` 的最小结果目录：写入 2 行源 CSV、`analysis_config.json`、shape 为 `(2, 331)` 的 NPZ 和按 `sample_index` 排列的 `attribution_by_sample.csv`。测试应调用 `build_manifest` 并断言：

```python
manifest = build_manifest(result_dir)

assert manifest["sample_index"].tolist() == [0, 1]
assert manifest["source_row_index"].tolist() == [0, 1]
assert manifest[["mean", "self_score", "query_score"]].to_dict("list") == {
    "mean": [1.5, -0.5],
    "self_score": [2.0, 0.0],
    "query_score": [0.5, 0.5],
}
assert (result_dir / "attribution_records.csv").is_file()
assert pd.read_csv(result_dir / "attribution_records.csv")["sample_index"].tolist() == [0, 1]
```

- [ ] **Step 2：运行测试，确认因 `build_manifest` 尚不存在而失败**

运行：

```bash
conda run -n fluProfiler python -m pytest tests/test_build_gradient_attribution_record_manifest.py -q
```

预期：测试因无法导入 `build_manifest` 而失败。

- [ ] **Step 3：实现 `build_manifest` 与验证逻辑**

实现步骤：

1. 读取 `analysis_config.json`，验证 `data_csv` 存在，取得 `sample_count` 与 `type_filter`；若配置没有 `type_filter`，使用结果目录对应的 `H3N2`。
2. 用 `reconstruct_records` 重建记录顺序；读取 `attribution_arrays.npz`，验证 `reference` 和 `query` 均为二维、第一维等于重建记录数及配置样本数。
3. 读取 `attribution_by_sample.csv`，验证 `sample_index` 是连续 `0..N-1`；按 `sample_index` 逐列比较 `seq_id_a`、`seq_id_c`、`seq_a`、`seq_c`、`serumPassCat`、`virusPassCat`、`serumName`、`virusName`、`label`。
4. 对源表中同名的 `mean`、`self_score`、`query_score` 先改名为 `source_mean`、`source_self_score`、`source_query_score`，然后按 `sample_index` 合并归因文件中的预测列。
5. 使用 `np.allclose(..., rtol=1e-7, atol=1e-7)` 将合并后的预测列与 NPZ 的相应一维数组比较。
6. 写入临时 CSV；重新读取并验证行数和完整 `sample_index` 序列后，再原子替换为 `attribution_records.csv`。

- [ ] **Step 4：运行全部新增测试，确认通过**

运行：

```bash
conda run -n fluProfiler python -m pytest tests/test_build_gradient_attribution_record_manifest.py -q
```

预期：`2 passed`。

### Task 3：对真实 H3N2 归因结果生成对照表并作端到端验证

**文件：**
- 新建：`/home/chenyh/workspace/fluProfiler/results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input/attribution_records.csv`
- 测试：`/home/chenyh/workspace/fluProfiler/tests/test_build_gradient_attribution_record_manifest.py`

**接口：**
- CLI：`python scripts/build_gradient_attribution_record_manifest.py --result-dir <gradient_x_input 目录>`。
- 输出：24,740 行的 `attribution_records.csv`。

- [ ] **Step 1：运行真实数据 CLI**

运行：

```bash
conda run -n fluProfiler python scripts/build_gradient_attribution_record_manifest.py \
  --result-dir results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input
```

预期：命令输出记录数、输出路径及通过的关键验证项；生成 `attribution_records.csv`。

- [ ] **Step 2：独立重读输出并验证 array 对齐不变量**

运行：

```bash
conda run -n fluProfiler python -c "from pathlib import Path; import numpy as np; import pandas as pd; p=Path('results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input'); d=pd.read_csv(p/'attribution_records.csv'); a=np.load(p/'attribution_arrays.npz'); assert len(d)==a['reference'].shape[0]==24740; assert d.sample_index.tolist()==list(range(len(d))); assert d.source_row_index.is_unique and d.record_key.is_unique; assert np.allclose(d['mean'],a['mean']); assert np.allclose(d['self_score'],a['self_score']); assert np.allclose(d['query_score'],a['query_score']); print({'rows':len(d),'columns':len(d.columns),'first_sample_index':int(d.sample_index.iloc[0]),'last_sample_index':int(d.sample_index.iloc[-1])})"
```

预期：输出 24,740 行、首尾索引 0 和 24,739，且全部断言通过。

- [ ] **Step 3：运行相关回归测试**

运行：

```bash
conda run -n fluProfiler python -m pytest tests/test_build_gradient_attribution_record_manifest.py tests/test_run_gradient_x_input.py -q
```

预期：全部通过。

- [ ] **Step 4：提交实现与测试，不提交用户原有改动**

运行：

```bash
git add scripts/build_gradient_attribution_record_manifest.py tests/test_build_gradient_attribution_record_manifest.py results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input/attribution_records.csv
git commit --only scripts/build_gradient_attribution_record_manifest.py tests/test_build_gradient_attribution_record_manifest.py results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input/attribution_records.csv -m "feat: add attribution record manifest"
```
