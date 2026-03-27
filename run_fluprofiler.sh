task=ha_only          # ha_only | hana
impl=v2              # v2 | legacy
config=experiments/HA_only/config_v2_ha_only.json
batch_size=64
learning_rate=8e-5
epochs=250
device=cuda:1
gpu_cache_gb=20
sample_limit=128       # -1 表示不截样本；例如 128 用于快速测试

cd ./src
python -m fluprofiler.cli.dispatch \
  --task "$task" \
  --impl "$impl" \
  --config $config \
  --batch-size $batch_size \
  --learning-rate $learning_rate \
  --epochs $epochs \
  --device $device \
  --gpu-cache-gb $gpu_cache_gb \
  --sample-limit $sample_limit