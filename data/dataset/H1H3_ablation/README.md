# Crick H1/H3 消融数据集构建与训练流程

以下流程从已有的 `processed/source.csv` 开始。每次只处理一个亚型；以下以 H3N2 为例。

```bash
SUBTYPE=H3N2
```

## 1. 导出当前亚型的唯一 Full-HA 序列

合并 `seq_a` 和 `seq_c`，去重后写入 `main/HA1_sequences/`。H3N2 输出以 `H3_` 开头，H1N1 输出以 `H1_` 开头。

```bash
python src/fluprofiler/dataset/full_ha_to_ha1.py \
  --dataset-dir data/dataset/H1H3_ablation \
  --subtype "${SUBTYPE}" \
  --step export
```

## 2. 对 Full-HA 序列进行 MAFFT 比对

```bash
python src/fluprofiler/dataset/full_ha_to_ha1.py \
  --dataset-dir data/dataset/H1H3_ablation \
  --subtype "${SUBTYPE}" \
  --step align
```

## 3. 按对齐坐标截取 HA1

坐标从 1 开始，且包含起止位点。H3N2 当前使用 17–345；H1N1 应填写其单独确认的范围。

```bash
HA1_START=17
HA1_END=345

python src/fluprofiler/dataset/full_ha_to_ha1.py \
  --dataset-dir data/dataset/H1H3_ablation \
  --subtype "${SUBTYPE}" \
  --ha1-start "${HA1_START}" \
  --ha1-end "${HA1_END}" \
  --step truncate
```

## 4. 构建 Full-HA 与 HA1 训练视图

两张视图按相同去重键生成，拥有相同样本、标签和 `row_id`，仅序列及其 ID 不同。

```bash
python src/fluprofiler/dataset/build_ha_views.py \
  --dataset-dir data/dataset/H1H3_ablation \
  --subtype "${SUBTYPE}" \
  --group-cols seq_a,seq_c,serumPassCat,virusPassCat,serumName,virusName
```

## 5. 生成血清与季节切分

`manifest.csv` 是 HA 与 HA1 共用的切分索引，保证两张视图的 train、valid、test 行一一对应。

```bash
python src/fluprofiler/dataset/split_ablation_views.py \
  --dataset-dir data/dataset/H1H3_ablation \
  --subtype "${SUBTYPE}" \
  --split-modes serum,season \
  --seeds 88,99 \
  --test-seasons 33,34 \
  --output-dir "data/dataset/H1H3_ablation/main/splits/${SUBTYPE}"
```

## 6. 导出 LucaVirus 输入 FASTA

该步骤从两个视图各自的序列 ID 导出 FASTA；输入 LucaVirus 前会移除对齐 gap（`-`）。

```bash
python src/fluprofiler/dataset/build_view_embedding_fastas.py \
  --dataset-dir data/dataset/H1H3_ablation \
  --subtype "${SUBTYPE}"
```

## 7. 使用 LucaVirus 生成 embedding

将 `VIEW` 依次设为 `HA` 和 `HA1` 分别运行。`--gpu_id` 是 `CUDA_VISIBLE_DEVICES` 内的逻辑编号。

```bash
cd /mnt/zzbnew/peixunban/chenyihao/LucaVirus/src/embedding

export CUDA_VISIBLE_DEVICES=0,1,2,3
VIEW=HA1

nohup conda run --no-capture-output -n lucavirus python get_embedding.py \
  --llm_dir ../../.. \
  --llm_type lucavirus \
  --llm_version v1.0 \
  --llm_task_level token_level,span_level,seq_level \
  --llm_time_str 20240815023346 \
  --llm_step 3800000 \
  --trunc_type right \
  --seq_type prot \
  --input_file "/mnt/zzbnew/peixunban/chenyihao/fluProfiler/data/dataset/H1H3_ablation/main/views/${SUBTYPE}/${VIEW}_lucavirus.fasta" \
  --save_path "/mnt/zzbnew/peixunban/chenyihao/fluProfiler/data/dataset/H1H3_ablation/main/embedding/${SUBTYPE}/${VIEW}" \
  --embedding_type matrix \
  --matrix_add_special_token \
  --embedding_complete \
  --embedding_complete_seg_overlap \
  --gpu_id 1 \
  > "/mnt/zzbnew/peixunban/chenyihao/fluProfiler/data/dataset/H1H3_ablation/main/embedding/${SUBTYPE}/${VIEW}_embedding.log" 2>&1 &
```

## 8. 训练

以下以 H3N2 为例。每个视图分别运行一次；一次会在 GPU 0–3 上并行启动两个季节和两个血清切分。`--refit-train-valid` 会将 train 与 valid 合并后训练，并仅评估 test。

```bash
cd /mnt/zzbnew/peixunban/chenyihao/fluProfiler

VIEW=HA
SUBTYPE=H3N2

mkdir -p "results/H1H3_ablation/${SUBTYPE}/${VIEW}/launch_logs"

run_season () {
  local season="$1"
  local gpu="$2"

  CUDA_VISIBLE_DEVICES="$gpu" nohup /home/chenyihao/miniconda3/bin/conda run --no-capture-output -n fluProfiler python \
    experiments/serum_gate/train_serum_mutation_set.py \
    --data-dir "data/dataset/H1H3_ablation/main/splits/${SUBTYPE}/season/${season}/${VIEW}" \
    --embedding-dir "data/dataset/H1H3_ablation/main/embedding/${SUBTYPE}/${VIEW}" \
    --ha-distance-matrix ha1_distance_no_bias_567.npy \
    --output-dir "results/H1H3_ablation/${SUBTYPE}/${VIEW}/season/${season}" \
    --type "${SUBTYPE}" \
    --refit-train-valid \
    --batch-size 1 \
    --max-queries-per-task 32 \
    --epochs 5 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --lr-scheduler cosine \
    --lr-min 1e-6 \
    --site-dim 64 \
    --background-dim 64 \
    --mutation-dim 128 \
    --position-dim 32 \
    --amino-acid-dim 16 \
    --presence-dim 4 \
    --theta-dim 128 \
    --mutation-attention-heads 4 \
    --mutation-attention-layers 1 \
    --mutation-ffn-dim 256 \
    --attention-dropout 0.1 \
    --attention-alpha-init 0.05 \
    --attention-tau-init 8.0 \
    --predictor-hidden-dim 256 \
    --predictor-dropout 0.1 \
    --zero-init-film \
    --direct-background \
    --no-use-background-to-mutation \
    --task-bias-loss-weight 0.1 \
    --seed 42 \
    --device cuda:0 \
    --gpu-cache-gb 24 \
    > "results/H1H3_ablation/${SUBTYPE}/${VIEW}/launch_logs/season_${season}.log" 2>&1 &
}

run_season 33 0
run_season 34 1
```

```bash
cd /mnt/zzbnew/peixunban/chenyihao/fluProfiler

VIEW=HA
SUBTYPE=H3N2

mkdir -p "results/H1H3_ablation/${SUBTYPE}/${VIEW}/launch_logs"

run_serum () {
  local seed="$1"
  local gpu="$2"

  CUDA_VISIBLE_DEVICES="$gpu" nohup /home/chenyihao/miniconda3/bin/conda run --no-capture-output -n fluProfiler python \
    experiments/serum_gate/train_serum_mutation_set.py \
    --data-dir "data/dataset/H1H3_ablation/main/splits/${SUBTYPE}/serum/seed_${seed}/${VIEW}" \
    --embedding-dir "data/dataset/H1H3_ablation/main/embedding/${SUBTYPE}/${VIEW}" \
    --ha-distance-matrix ha1_distance_no_bias_567.npy \
    --output-dir "results/H1H3_ablation/${SUBTYPE}/${VIEW}/serum/seed_${seed}" \
    --type "${SUBTYPE}" \
    --refit-train-valid \
    --batch-size 1 \
    --max-queries-per-task 32 \
    --epochs 50 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --lr-scheduler cosine \
    --lr-min 1e-6 \
    --site-dim 64 \
    --background-dim 64 \
    --mutation-dim 128 \
    --position-dim 32 \
    --amino-acid-dim 16 \
    --presence-dim 4 \
    --theta-dim 128 \
    --mutation-attention-heads 4 \
    --mutation-attention-layers 1 \
    --mutation-ffn-dim 256 \
    --attention-dropout 0.1 \
    --attention-alpha-init 0.05 \
    --attention-tau-init 8.0 \
    --predictor-hidden-dim 256 \
    --predictor-dropout 0.1 \
    --zero-init-film \
    --direct-background \
    --no-use-background-to-mutation \
    --task-bias-loss-weight 0.1 \
    --seed 42 \
    --device cuda:0 \
    --gpu-cache-gb 24 \
    > "results/H1H3_ablation/${SUBTYPE}/${VIEW}/launch_logs/serum_seed_${seed}.log" 2>&1 &
}

run_serum 88 0
run_serum 99 1
```

待 HA 的四个训练完成后，将上面代码块中的 `VIEW=HA` 改为 `VIEW=HA1`，再运行一次。
