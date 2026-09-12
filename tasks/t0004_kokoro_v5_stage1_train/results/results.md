# t0004 — Stage 1 Results

**Machine**: LLM-T1-NC80 (2× H100 NVL, single-GPU via plain `python3 train_first.py`)
**Data**: 1557 clips (v5 phoneme-corrected corpus)
**Config**: `code/config_david_v5_stage1.yml`, 10 epochs

## Val loss per epoch

| Epoch | Val loss |
|-------|----------|
| 1 | 1.170 |
| 2 | 0.873 |
| 3 | 0.805 |
| 4 | 0.787 |
| 5 | 0.814 |
| 6 | 0.788 |
| 7 | 0.807 |
| 8 | 0.757 |
| 9 | 0.792 |
| **10** | **0.740** ← best |

Best checkpoint: **epoch 10, val=0.740** (`results/checkpoints/epoch_1st_00007.pth`).

Note: t0004 launched `train_first.py` as plain `python3` (not `accelerate launch`), so it ran on
GPU0 only (1 of 2). Silently half-speed but trained correctly. See t0005 background for details.

## Files Produced

- `results/checkpoints/epoch_1st_00003.pth.dvc` — epoch 3 checkpoint
- `results/checkpoints/epoch_1st_00007.pth.dvc` — epoch 10 checkpoint (best; handed off to t0005)
