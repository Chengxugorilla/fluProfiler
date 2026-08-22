# fluProfiler-v2 项目架构设计

日期：2026-06-10

## 背景

`fluProfiler` v1 是服务于预印本阶段的研究代码库。它已经留下了很多有价值的东西：HA-only 和 HA+NA 训练路径、论文对齐的 titer/strain/serum split、active learning 实验、运行日志，以及早期 `models_v2` 中围绕 `BatchInput` 和 `ModelOutput` 建立的模型输入输出约定。

现在的问题不是缺少想法，而是想法太多。如果 v2 一开始就同时铺开 HI/MN 多任务学习、metadata 校准、NA inhibitor gate、retrieval/MSA-like context，很容易再次变成一堆并行脚本。更稳的路线是：先把数据、split、batch、模型输出、运行产物和评估诊断这些“契约”固定住，再把科学模块一个个接进去。

因此，v2 的推荐方向是：

> 小核心、强契约、模型用 recipe 扩展。

也就是说，fluProfiler-v2 先做一个干净、可复现、可比较的核心框架。后续的 metadata adapter、NA gate、HI/MN multitask、retrieval 都作为扩展模块接入，而不是各自开一套训练流程。

## 目标

1. 新建独立仓库 `fluProfiler-v2`，不在 v1 上继续打补丁。
2. 保留和 v1 的可比较性，优先复现 HA-only 和 HA+NA baseline。
3. 把 serum-holdout generalization 作为第一阶段核心科学问题。
4. 显式区分 biological antigenic signal 与 assay、serum、batch、passage、NA inhibitor 等实验条件影响。
5. 所有实验都通过配置驱动，能复现、能审计、能追踪。
6. 后续增加新模块时，不再复制出一堆新的训练脚本。

## 非目标

1. 第一阶段不实现 retrieval 或 MSA-like context。
2. 不把 MN label 直接当成 HI label 混在一起训练。
3. 不把 institution 当作普通生物学序列特征直接喂给 baseline。
4. 不在 v2 仓库第一阶段建设 web app 或服务端。
5. 不迁移 v1 中所有 notebook、历史实验和 `src/deprecated` 代码。

## 推荐仓库结构

```text
fluProfiler-v2/
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- data/
|   |   |-- hi.yaml
|   |   |-- hi_mn.yaml
|   |   `-- metadata_fields.yaml
|   |-- split/
|   |   |-- random.yaml
|   |   |-- virus_holdout.yaml
|   |   |-- serum_holdout.yaml
|   |   |-- season_holdout.yaml
|   |   `-- institution_holdout.yaml
|   |-- model/
|   |   |-- baseline.yaml
|   |   |-- serum_reference.yaml
|   |   |-- metadata_adapter.yaml
|   |   `-- na_gate.yaml
|   `-- experiment/
|       |-- e0_baseline_hi_serum.yaml
|       |-- e1_serum_reference_hi.yaml
|       `-- e2_weighted_huber_hi.yaml
|-- src/fluprofiler/
|   |-- core/
|   |   |-- schema.py
|   |   |-- types.py
|   |   |-- config.py
|   |   `-- registry.py
|   |-- data/
|   |   |-- preprocess.py
|   |   |-- splits.py
|   |   |-- dataset.py
|   |   |-- collate.py
|   |   `-- embeddings.py
|   |-- models/
|   |   |-- encoders.py
|   |   |-- pairing.py
|   |   |-- heads.py
|   |   `-- recipes.py
|   |-- losses/
|   |   |-- weighted_huber.py
|   |   |-- censored.py
|   |   `-- multitask.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   |-- bias.py
|   |   |-- calibration.py
|   |   `-- benchmark.py
|   |-- training/
|   |   |-- trainer.py
|   |   |-- callbacks.py
|   |   `-- checkpoint.py
|   |-- extensions/
|   |   |-- metadata_adapter.py
|   |   |-- na_gate.py
|   |   |-- multitask_hi_mn.py
|   |   `-- retrieval.py
|   `-- utils/
|       |-- io.py
|       |-- seed.py
|       `-- logging.py
|-- scripts/
|   |-- prepare_data.py
|   |-- make_splits.py
|   |-- train.py
|   |-- evaluate.py
|   `-- predict.py
|-- tests/
|-- docs/
`-- outputs/
```

`outputs/` 必须加入 `.gitignore`。原始数据、embedding、checkpoint、生成的预测结果默认不提交到 git。只有用于单元测试的小型 synthetic fixture 可以进入仓库。

## 架构原则

### 1. 先定契约，再写模块

v2 最核心的公共契约包括：

- `Schema`：必需字段、可选 metadata 字段、label 规则、缺失值规则。
- `SplitManifest`：数据源 checksum、split 策略、随机种子、分组字段、泄漏检查、样本数。
- `BatchInput`：张量、mask、label、metadata、row id。
- `ModelOutput`：预测值、可选 loss、可选 uncertainty、可解释中间结果。
- `RunArtifact`：保存后的 config、git 状态、metrics、predictions、label-bin diagnostics、checkpoint。

任何新模块都应该消费和返回这些契约，而不是在脚本里重新定义自己的输入输出格式。

### 2. 只有一条主数据流

```text
raw HI/MN tables + sequence maps
-> standardized pair table
-> schema validation
-> split generation with manifest
-> dataset and collator
-> model recipe
-> predictions
-> metrics, label-bin bias, and calibration diagnostics
```

split 系统应该能作为独立脚本运行，但实现放在 `src/fluprofiler/data/` 中，由 `scripts/make_splits.py` 调用。不建议再建一个根目录级别的 `data_pipeline/` 包，否则后面很容易出现两套数据逻辑。

### 3. 用 recipe 组装模型

模型不应该写成很多相似但略有差异的大类。更合理的方式是把 encoder、pair interaction、head、loss、adapter 作为组件，然后通过 recipe 组装。

第一阶段 recipe：

- `BaselineRecipe`：HA 和可选 NA encoder、pair interaction、passage embedding、MLP regression head。
- `SerumReferenceRecipe`：virus encoder、serum-reference encoder、pair interaction、weighted loss、可选 calibration head。
- `MetadataAdapterRecipe`：biological prediction 加 metadata bias adapter。

后续 recipe：

- `NAGateRecipe`：HA+NA 模型中加入 metadata-conditioned NA contribution。
- `HIMNMultitaskRecipe`：共享 antigenic latent representation，并分别接 HI 和 MN head。
- `RetrievalRecipe`：time-aware homologous context 和 antigenic anchor 特征融合。

### 4. Institution 作为 batch context

`institution` 不应该作为 baseline 的普通生物学特征。它应该进入 metadata adapter，用来估计 assay 或 batch bias：

```text
final_prediction = biological_prediction + metadata_bias
```

需要支持三种模式：

- `none`
- `categorical_embedding`
- `batch_effect_adapter`

baseline 之后的默认实验模式建议使用 `batch_effect_adapter`。

### 5. HI 和 MN 相关，但不是同一个 label

HI 和 MN 可以共享 schema、split、dataset、collator 和训练框架，但默认不能把 label 混成同一个目标。

HI distance：

```text
log2(homologous_HI_titer) - log2(heterologous_HI_titer)
```

MN distance：

```text
log2(homologous_MN_titer) - log2(heterologous_MN_titer)
```

如果 MN 数据没有 homologous MN titer，这条样本应该保留为 `normalized_reactivity`，并进入 MN-specific head 或单独实验。不要静默地把它塞进 HI distance。

## 数据 Schema

标准化后的 pair table 分三类字段。

必需 identity 和 label 字段：

- `row_id`
- `virus_id`
- `serum_id`
- `assay_type`
- `label`
- `label_type`

序列和表示字段：

- `virus_HA_id`
- `virus_NA_id`
- `serum_reference_virus_id`
- `serum_reference_HA_id`
- `serum_reference_NA_id`
- 当 embedding 没有预计算时，可以保留可选 raw sequence 字段。

metadata 字段：

- `virus_subtype`
- `season`
- `collection_date`
- `institution`
- `serum_passage`
- `virus_passage`
- `RBC_type`
- `NA_inhibitor`
- `NA_inhibitor_name`

可选 metadata 缺失时统一归一化为 `unknown`。必需字段缺失时，schema validation 必须失败。

## Split 协议

v2 第一版需要支持：

- `random`：row-level random split。
- `virus_holdout`：按 `virus_id` 分组 holdout。
- `serum_holdout`：按 `serum_id` 分组 holdout。
- `season_holdout`：按完整 season holdout。
- `institution_holdout`：按完整 institution holdout。

每个 split 必须写出：

```text
train.csv
valid.csv
test.csv
manifest.json
```

每个 manifest 至少包含：

- split strategy 和 seed
- source path 和 checksum
- group column
- row counts
- group counts
- leakage checks
- duplicate aggregation rules
- timestamp

特殊约束：

- `serum_holdout`：test 中的 `serum_id` 不能出现在 train。
- `season_holdout`：后续启用 retrieval 时，只能检索目标 season 之前已经可获得的序列。
- `institution_holdout`：test 中未见过的 institution 在 metadata adapter 中映射到 `unknown`。

## 模型组件

### Encoders

第一阶段 encoder 保持简单，并优先兼容预计算 embedding：

- `SequenceEncoder`：对 HA 或 NA embedding 做 pooling。
- `SerumReferenceEncoder`：编码 serum reference virus 的 HA 和可选 NA。
- `MetadataEncoder`：为后续 adapter 编码 categorical metadata。

模型需要通过 config 支持 HA-only 和 HA+NA，而不是通过两套训练脚本支持。

### Pair Interaction

支持的 interaction modes：

- `concat`
- `abs_diff`
- `product`
- `concat_abs_diff`
- `distance`

推荐第一版 baseline 使用 `concat_abs_diff`。它既保留 v1-like MLP 的灵活性，又更适合 serum-reference 对比。

### Prediction Heads

第一阶段 head：

- `RegressionHead`
- `DistanceHead`
- `CalibrationHead`

`CalibrationHead` 从简单 affine transform 开始：

```text
y_calibrated = a * y_pred + b
```

后续可以扩展成 assay-specific 或 label-bin-specific calibration。

## Losses

baseline 默认可以使用 MSE 或 Huber。serum-generalization 实验优先加入 weighted Huber：

```text
weight(y) = 1 / frequency(bin(y))
```

或者：

```text
weight(y) = 1 + lambda * abs(y - median(y))
```

第一版实现应同时支持这两种权重策略，并通过 config 选择。

## Evaluation

每次运行必须输出：

- MAE
- MSE
- RMSE
- Pearson
- Spearman
- R2
- label-bin MAE
- label-bin RMSE
- label-bin prediction bias

label-bin diagnostics 输出：

```text
label_bin
n
true_mean
pred_mean
bias
mae
rmse
```

这是诊断 serum-holdout regression-to-the-mean 的关键表：

- true large label 是否仍被预测偏小。
- true small label 是否仍被预测偏大。

## Run Artifacts

每次训练运行写出：

```text
outputs/{experiment_name}/{run_id}/
|-- config.yaml
|-- git.json
|-- metrics.json
|-- metrics_by_split.json
|-- predictions.csv
|-- label_bin_metrics.csv
|-- checkpoint.pt
`-- manifest.json
```

一次 run 应该能通过 `config.yaml`、split manifest 和 source data checksum 被追溯和复现。

## 阶段规划

### Phase 1：可靠核心

实现：

1. 仓库骨架和 packaging。
2. Schema validation。
3. Split generation。
4. Dataset 和 collator。
5. HA-only 和 HA+NA v1-like baseline。
6. Serum-reference model。
7. Weighted Huber loss。
8. Label-bin bias evaluation。
9. schema、split、loss、model forward 的单元测试。

第一阶段最重要的科学问题：

> serum-reference encoding + tail-aware loss 是否能缓解 serum-holdout 中的预测收缩问题？

### Phase 2：Metadata 与 NA effects

实现：

1. Metadata encoder。
2. Institution batch-effect adapter。
3. NA inhibitor usage statistics。
4. Metadata-conditioned NA gate。
5. H3N2 HI with/without NA inhibitor 的 benchmark。

### Phase 3：HI/MN Multitask Learning

实现：

1. HI 和 MN 的 assay-aware dataset handling。
2. Shared antigenic latent representation。
3. HI 和 MN heads。
4. matched virus-serum pairs 的 consistency loss。
5. Assay-specific calibration。

### Phase 4：Time-Aware Retrieval

实现：

1. Sequence index。
2. Time-aware retrieval。
3. Homologous neighborhood features。
4. Antigenic anchor features。
5. Retrieval fusion。

在 leakage tests 完成前，retrieval 默认关闭。

## 测试策略

最低测试集：

- `test_schema.py`：缺失必需字段会失败；可选 metadata 会归一化为 `unknown`。
- `test_splits.py`：random、virus、serum、season、institution splits 都满足泄漏检查。
- `test_batch_contract.py`：dataset 和 collator 能产生合法 `BatchInput`。
- `test_model_forward.py`：baseline 和 serum-reference recipes 输出 shape 为 `[B]` 的预测。
- `test_losses.py`：weighted Huber 能处理 label-bin weights 和 empty bins。
- `test_bias_metrics.py`：label-bin diagnostics 正确计算 true mean、pred mean、bias、MAE、RMSE。

测试数据优先使用小型 synthetic fixtures，不复制真实研究数据。

## 从 v1 迁移的原则

迁移概念，不迁移目录。

应该保留：

- `src/fluprofiler/models_v2/io.py` 中 `BatchInput` 和 `ModelOutput` 的思想。
- `experiments/tools/build_splits.py` 中 split manifest 的纪律。
- `experiments/HA_only` 和 `experiments/HANA` 中 HA-only、HA+NA 与 v1 可比较的实验目标。

不应该迁移：

- v1 的 notebooks。
- `src/deprecated` 中的历史实验代码。
- 各种为单次实验写死路径的训练脚本。

v1 的 run logs 和 paper-aligned splits 应作为 benchmark reference，而不是 v2 的内部结构模板。

## 已确认决策

1. v2 是新仓库，名称为 `fluProfiler-v2`。
2. 第一阶段聚焦可复现 baseline 和 serum generalization。
3. 核心包路径使用 `src/fluprofiler/`。
4. 模型变体通过 recipe 和 config 组合。
5. Metadata effects 作为 biological prediction 上的 additive adapter。
6. MN 支持先从 schema 层开始，完整 multitask learning 放到 Phase 3。
7. Retrieval 是 Phase 4 扩展，并且必须 time-aware。

