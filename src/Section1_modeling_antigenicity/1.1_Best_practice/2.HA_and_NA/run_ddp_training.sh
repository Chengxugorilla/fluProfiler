#!/bin/bash

# DDP训练启动脚本
echo "Starting DDP training with all available GPUs..."

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Found $GPU_COUNT GPUs"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际GPU数量调整
export NCCL_DEBUG=INFO

# 运行DDP训练
python run_HANA.py

echo "DDP training completed!"
