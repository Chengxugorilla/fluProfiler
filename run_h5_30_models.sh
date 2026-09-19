#!/bin/bash

# 切换到工作目录
cd /home/chenyh/workspace/fluProfiler

# 固定不变的特征与矩阵路径
EMBED_DIR="/home/chenyh/workspace/fluProfiler_H5/data/embedding/files_HA1_global_aligned"
DIST_MAT="/home/chenyh/workspace/fluProfiler/ha1_distance_no_bias_567.npy"

# 根目录设定
BASE_DATA_DIR="/home/chenyh/workspace/fluProfiler_H5/data/dataset/H5/splited/20260727_142640_HA1_Global_Aligned"
BASE_OUT_DIR="/home/chenyh/workspace/fluProfiler_H5/results/H5_final_cv_30"

# 定义三种划分方式
SPLITS=("serum" "titer" "strain")

echo "开始执行 H5 最优参数 30 组全量交叉验证..."

# 外层循环：遍历 serum, titer, strain
for SPLIT in "${SPLITS[@]}"; do
    
    # 内层循环：遍历 seed_0 到 seed_9
    for SEED in {0..9}; do
        
        # 动态拼接每组任务的数据和输出目录
        DATA_DIR="${BASE_DATA_DIR}/${SPLIT}/seed_${SEED}"
        OUT_DIR="${BASE_OUT_DIR}/${SPLIT}/seed_${SEED}"
        
        echo "=================================================="
        echo "正在运行划分: ${SPLIT} | Seed: ${SEED}"
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
          --weight-decay 0.01 \
          --lr-scheduler cosine \
          --lr-min 1e-6 \
          --early-stopping-patience 10 \
          --site-dim 64 \
          --site-bottleneck-dim 0 \
          --background-dim 64 \
          --direct-background \
          --mutation-dim 128 \
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
          --predictor-hidden-dim 256 \
          --predictor-dropout 0.1 \
          --zero-init-film \
          --use-film-beta \
          --use-pool-mutation-count \
          --use-attention-pool \
          --use-predictor-mutation-count \
          --label-weight-thresholds 2,3.5,5 \
          --label-weight-values 1,1.3,1.8,2.5 \
          --within-serum-rank-loss-weight 0.0 \
          --task-bias-loss-weight 0.1 \
          --no-full-task-bias-loss \
          --no-use-output-identity-bias \
          --device cuda:0 \
          --gpu-cache-gb 24 \
          --seed 42 \
          --no-progress
          
        echo "${SPLIT} - Seed ${SEED} 训练完成！"
        sleep 2 # 给显卡清理缓存留一点喘息时间
    done
done

echo "🎉 30 组交叉验证任务全部运行完毕！"