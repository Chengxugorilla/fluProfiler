# 梯度归因 Record 对照表设计

## 目标

创建一张可审计的 record 级对照表，使 H3N2 Gradient × Input 归因数组的每一行都能对应回原始数据记录及其完整 meta 信息。

## 范围

本功能将创建：

`results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input/attribution_records.csv`

本功能不重新计算归因值、不修改 `attribution_arrays.npz`、不展开逐 token 归因列，也不修改现有分析 notebook。

## 权威输入

- 归因结果目录：`results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-latent8/titer/seed_0/subtype/H3N2/gradient_x_input`。
- 归因数组文件：`attribution_arrays.npz`，其中 `reference` 与 `query` 的 shape 均为 `(24740, 331)`。
- 归因配置：`analysis_config.json`；其中的 `data_csv` 是原始数据表的权威路径。
- 原始数据表：`/home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/whole/whole.csv`。
- 归因运行时已输出的预测与部分 meta：`attribution_by_sample.csv`。

`train.csv` 不是本对照表的有效来源，因为本次归因运行使用的是 `whole/whole.csv`。

## 复现 array 顺序

对照表必须严格复现导出脚本中 `collect_samples` 的过程：

1. 以文件中的原始顺序读取源表，并将零基行号保存为 `source_row_index`。
2. 筛选 `serumType == H3N2`；比较前去除首尾空格且不区分大小写。只有源表不存在 `serumType` 时才使用 `Type`。
3. 保留 `seq_a` 和 `seq_c` 去除首尾空格后长度均为 329 的记录。
4. 以 `seq_a`、`seq_c`、`serumPassCat`、`virusPassCat` 去重，并保留首次出现的合格原始记录。
5. 按上述结果重置顺序，赋值 `sample_index = 0, ..., N-1`。

最终第 `n` 行对应 array 的第一维：`reference[n, :]` 与 `query[n, :]`。

## 对照表字段

CSV 每行对应一条归因 record，保留所选源表记录的全部原始字段；在最前方加入以下对齐与溯源字段：

1. `sample_index`：零基、唯一、连续的归因 array 第一维索引。
2. `source_row_index`：该记录在未过滤 `whole/whole.csv` 中的零基行号。
3. `record_key`：由 `seq_a`、`seq_c`、`serumPassCat`、`virusPassCat` 确定性序列化生成；经指定去重后必须唯一。

在表末附加从 `attribution_by_sample.csv` 按 `sample_index` 对齐的模型输出：

- `mean`
- `self_score`
- `query_score`

若原始源表已有与这些字段同名的列，则保留原始列但将其改名为 `source_` 前缀；对齐与溯源字段使用未加前缀的名称。

## 验证条件

写出文件前必须断言：

- 重建后的记录数等于 `analysis_config.json["sample_count"]`，并等于 `attribution_arrays.npz["reference"].shape[0]`。
- `sample_index` 完全等于 `0..N-1`，且唯一、单调递增。
- `source_row_index` 唯一，并可反查到相同的源表记录。
- `record_key` 唯一。
- 对每一行，`seq_id_a`、`seq_id_c`、`seq_a`、`seq_c`、`serumPassCat`、`virusPassCat`、`serumName`、`virusName`、`label` 均与同一 `sample_index` 的 `attribution_by_sample.csv` 一致。
- 对照表中的 `mean`、`self_score`、`query_score` 与归因 meta 文件中的相应值在数值容差内一致。
- 写出的 CSV 被重新读取后，行数与 `sample_index` 序列保持不变。

## 设计理由

`Fig.3.ipynb` 通过保持 DataLoader 顺序（`shuffle=False`）及按批次文件编号顺序拼接 attention 来关联 meta 与 attention 值。新的对照表保留“行顺序对应”的思想，但通过 `sample_index`、`source_row_index` 与 `record_key` 明确记录并独立验证这一对应关系。
