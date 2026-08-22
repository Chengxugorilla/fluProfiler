# 1. 准备原始数据
	在这里，原始数据是指R包处理后，包含序列、label（对于预测同源滴度模型有homo_label, hete_label, diff_label）的data4model(机构-亚型).csv文件。
	在 fluProfiler/data/dataset/ 目录下，创建数据集目录（如H1H3）作为这个数据集的名字，并在该目录下新建 /raw 目录，将各个亚型原始数据文件（也可以是单一亚型）放如其中。

# 2.将raw处理成processed（去重）

raw2processed
```bash
python src/fluprofiler/dataset/raw2processed.py \
--dataset-dir /home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1_homo \
--registry-csv /home/chenyh/workspace/fluProfiler/data/embedding/registry/sequences.csv \
--embedding-files-dir /home/chenyh/workspace/fluProfiler/data/embedding/files \
--group-cols seq_a,seq_c,serumPassCat,virusPassCat
```

若将Name也纳入切分（Nextflu，Adaboost切分方式）
```bash
python src/fluprofiler/dataset/raw2processed.py \
--dataset-dir /home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1_Crick-CNIC \
--registry-csv /home/chenyh/workspace/fluProfiler/data/embedding/registry/sequences.csv \
--embedding-files-dir /home/chenyh/workspace/fluProfiler/data/embedding/files \
--group-cols seq_a,seq_c,serumPassCat,virusPassCat,serumName,virusName
```

### 3.processed2splited，对数据进行切分
```bash
python src/fluprofiler/dataset/processed2splited.py \
  --dataset-dir /home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1_Crick-CNIC \
  --seed 0,1,2,3,4,5,6,7,8,9 \
  --test-ratio 0.1 \
  --valid-ratio 0.1 \
  --valid-split group \
  --strain-col seq_c,virusPassCat,virusName \
  --serum-col seq_a,serumPassCat,serumName \
  --split-modes titer,strain,serum,season \
  --season-col sheet \
  --test-seasons 39,40,41,42,43,44
```

```bash
python src/fluprofiler/dataset/processed2splited.py \
  --dataset-dir /home/chenyh/workspace/fluProfiler/data/dataset/H1H3_HA1_homo \
  --seed 42 \
  --test-ratio 0.1 \
  --valid-ratio 0.1 \
  --valid-split group \
  --split-modes titer,strain,serum,season \
  --season-col sheet \
  --test-seasons 39,40,41,42,42,44
```

