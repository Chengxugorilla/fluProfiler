config=experiments/HA_only/config_v2_ha_only.json
batch_size=64
learning_rate=8e-5
epochs=250
device=cuda:1

cd ./src
python -m fluprofiler.cli.dispatch \
  ha_only_v2 \
  --config $config \
  --batch-size $batch_size \
  --learning-rate $learning_rate \
  --epochs $epochs \
  --device $device