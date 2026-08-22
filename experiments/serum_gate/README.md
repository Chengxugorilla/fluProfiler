cd /home/chenyh/workspace/fluProfiler

python experiments/serum_gate/train_zero_shot_minus_whole.py \
  --data-dir data/dataset/H1H3_HA1_Crick-CNIC/splited/20260722_124218/whole \
  --embedding-dir data/embedding/files \
  --output-dir results/H1H3_HA1_Crick-CNIC/20260722_124218/SerumGate-Minus-latent8/whole/subtype/H3N2 \
  --type H3N2 \
  --serum-task-cols seq_id_a,serumPassCat,serumName \
  --epochs 200 \
  --latent-dim 8 \
  --ha-attention-dim 64 \
  --ha-attention-heads 4 \
  --ha-attention-dropout 0.2 \
  --max-queries-per-task 32 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --lr-scheduler cosine \
  --lr-min 1e-6 \
  --gpu-cache-gb 24 \
  --device cuda:7 \
  --seed 42