# Project Scripts

Run commands from the project root:

```bash
cd /Users/zhiyujia/Documents/UNC/CHASE/Biofame
```

Convert wECGdb 12-lead data:

```bash
bash scripts/convert_wecg_12lead.sh
```

Smoke test one epoch:

```bash
DEVICE=mps bash scripts/smoke_wecg_12lead.sh
```

By default the smoke test uses `ml-famae/data_wecg_probe`. To smoke-test the full converted dataset explicitly:

```bash
DATA_DIR=./data_wecg_12lead DEVICE=cuda BATCH_SIZE=64 bash scripts/smoke_wecg_12lead.sh
```

Full pretraining, local Mac:

```bash
DEVICE=mps BATCH_SIZE=32 EPOCHS=50 bash scripts/train_wecg_12lead.sh
```

Full pretraining, RunPod/CUDA:

```bash
DEVICE=cuda BATCH_SIZE=64 NUM_WORKERS=4 EPOCHS=300 bash scripts/train_wecg_12lead.sh
```

Convert and train wECGdb chest V2+V6:

```bash
bash scripts/convert_wecgdb_v2_v6.sh
DEVICE=cuda BATCH_SIZE=64 EPOCHS=300 bash scripts/train_wecg_chest_v2_v6.sh
```

Convert and train wECGdb wrist 2-lead:

```bash
bash scripts/convert_wecgdb_wrist_2lead.sh
DEVICE=cuda BATCH_SIZE=64 EPOCHS=300 bash scripts/train_wecg_wrist_2lead.sh
```

Outputs are saved by `main.py` under:

```text
ml-famae/model_ckpt/${TB_NAME}_bioFAME/ckpt.pt
ml-famae/TB_logs/${TB_NAME}_bioFAME/
```
