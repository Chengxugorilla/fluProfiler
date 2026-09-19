#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/chenyh/workspace/fluProfiler"
data_dir="${repo_root}/data/dataset/H1H3_HA1_v1.0/splited/20260717_164256/titer/seed_0"
embedding_dir="${repo_root}/data/embedding/files"
results_root="${repo_root}/results/H1H3_HA1_v1.0/20260717_164256/SerumGate-Minus-AllMean-NoNA-latent8/titer/seed_0/subtype"
python_bin="${FLUPROFILER_PYTHON_BIN:-python}"
device="${FLUPROFILER_DEVICE:-cuda:0}"
gpu_cache_gb="${FLUPROFILER_GPU_CACHE_GB:-24}"
action="${1:-all}"

if [[ "${action}" != "train" && "${action}" != "test" && "${action}" != "all" ]]; then
    echo "Usage: $0 [train|test|all]" >&2
    exit 2
fi

for subtype in H1N1 H3N2; do
    output_dir="${results_root}/${subtype}"

    if [[ "${action}" == "train" || "${action}" == "all" ]]; then
        "${python_bin}" "${repo_root}/experiments/serum_gate/train_zero_shot_minus_all_mean.py" \
            --data-dir "${data_dir}" \
            --embedding-dir "${embedding_dir}" \
            --output-dir "${output_dir}" \
            --type "${subtype}" \
            --refit-train-valid \
            --batch-size 1 \
            --max-queries-per-task 32 \
            --epochs 200 \
            --learning-rate 1e-4 \
            --weight-decay 0.01 \
            --lr-scheduler cosine \
            --lr-min 1e-6 \
            --latent-dim 8 \
            --theta-dim 128 \
            --predictor-arch conditioned_mlp \
            --predictor-hidden-dim 256 \
            --ha-pooling mean \
            --ha-pair-mode independent \
            --na-branch none \
            --loss nll \
            --score-log-var-mode sum \
            --device "${device}" \
            --gpu-cache-gb "${gpu_cache_gb}" \
            --seed 42 \
            --progress
    fi

    if [[ "${action}" == "test" || "${action}" == "all" ]]; then
        checkpoint="${output_dir}/checkpoints/best_model.pth"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "Missing checkpoint: ${checkpoint}" >&2
            exit 1
        fi
        "${python_bin}" "${repo_root}/experiments/serum_gate/infer_minus_all_mean_checkpoint.py" \
            --checkpoint "${checkpoint}" \
            --data-dir "${data_dir}" \
            --split test \
            --embedding-dir "${embedding_dir}" \
            --output-csv "${output_dir}/predictions_test_reloaded.csv" \
            --metrics-json "${output_dir}/test_metrics_reloaded.json" \
            --type "${subtype}" \
            --batch-size 1 \
            --max-queries-per-task 32 \
            --device "${device}" \
            --gpu-cache-gb "${gpu_cache_gb}" \
            --progress
    fi
done
