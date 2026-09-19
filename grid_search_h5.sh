#!/bin/bash

# 切换到工作目录
cd /home/chenyh/workspace/fluProfiler

# 固定路径变量
DATA_DIR="/home/chenyh/workspace/fluProfiler_H5/data/dataset/H5/splited/20260727_142640_HA1_Global_Aligned/serum/seed_0"
EMBED_DIR="/home/chenyh/workspace/fluProfiler_H5/data/embedding/files_HA1_global_aligned"
DIST_MAT="/home/chenyh/workspace/fluProfiler/ha1_distance_no_bias_567.npy"
BASE_OUT_DIR="/home/chenyh/workspace/fluProfiler_H5/results/H5_grid_search"

# 定义要搜索的参数组合
HIDDEN_DIMS=(128 256)
WEIGHT_DECAYS=(0.01 0.05)
# 对比实验：全1默认权重 vs H5真实分布阶梯加权
WEIGHT_VALS_LIST=("1,1,1,1" "1,1.3,1.8,2.5")

echo "开始执行 H5 参数网格搜索 (含 H5 定制 Thresholds)..."

for DIM in "${HIDDEN_DIMS[@]}"; do
    for WD in "${WEIGHT_DECAYS[@]}"; do
        for W_VALS in "${WEIGHT_VALS_LIST[@]}"; do
            
            # 联动调整 mutation-dim
            if [ "$DIM" -eq 128 ]; then MUT_DIM=64; else MUT_DIM=128; fi
            
            # 区分实验名称
            if [ "$W_VALS" == "1,1,1,1" ]; then W_NAME="flat"; else W_NAME="weighted"; fi
            
            EXP_NAME="dim${DIM}_wd${WD}_w-${W_NAME}"
            OUT_DIR="${BASE_OUT_DIR}/${EXP_NAME}/seed_0"
            
            echo "=================================================="
            echo "正在运行实验: ${EXP_NAME}"
            echo "输出目录: ${OUT_DIR}"
            echo "=================================================="
            
            CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n fluProfiler python \
              experiments/serum_gate/train_serum_mutation_set.py \
              --data-dir "$DATA_DIR" \
              --embedding-dir "$EMBED_DIR" \
              --ha-distance-matrix "$DIST_MAT" \
              --output-dir "$OUT_DIR" \
              --type H5 \
              --serum-task-cols seq_id_a,serumPassCat,serumName \
              --refit-train-valid \
              --batch-size 1 \
              --max-queries-per-task 32 \
              --no-shuffle-queries-within-task-each-epoch \
              --epochs 50 \
              --save-epoch 50 \
              --learning-rate 1e-4 \
              --weight-decay "$WD" \
              --lr-scheduler cosine \
              --lr-min 1e-6 \
              --early-stopping-patience 10 \
              --site-dim 64 \
              --site-bottleneck-dim 0 \
              --background-dim 64 \
              --direct-background \
              --mutation-dim "$MUT_DIM" \
              --position-dim 32 \
              --amino-acid-dim 16 \
              --presence-dim 4 \
              --theta-dim 128 \
              --passage-dim 8 \
              --subtype-dim 0 \
              --no-use-subtype-feature \
              --use-passage-pair-feature \
              --mutation-attention-heads 4 \
              --mutation-attention-layers 1 \
              --no-bypass-mutation-transformer \
              --no-use-background-to-mutation \
              --mutation-ffn-dim 256 \
              --attention-dropout 0.1 \
              --attention-alpha-init 0.05 \
              --attention-tau-init 8.0 \
              --predictor-hidden-dim "$DIM" \
              --predictor-dropout 0.1 \
              --zero-init-film \
              --use-film-beta \
              --use-pool-mutation-count \
              --use-attention-pool \
              --use-predictor-mutation-count \
              --label-weight-thresholds 2,3.5,5 \
              --label-weight-values "$W_VALS" \
              --within-serum-rank-loss-weight 0.0 \
              --task-bias-loss-weight 0.1 \
              --no-full-task-bias-loss \
              --no-use-output-identity-bias \
              --device cuda:0 \
              --gpu-cache-gb 24 \
              --seed 42 \
              --no-progress
              
            echo "实验 ${EXP_NAME} 已完成！"
            sleep 2
        done
    done
done

echo "所有网格搜索任务全部运行完毕！"