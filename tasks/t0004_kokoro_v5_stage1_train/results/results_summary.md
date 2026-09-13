# Results Summary: Kokoro v5 Stage 1 Fine-tune

## Summary

Ran StyleTTS2 Stage 1 on v5 phoneme-corrected corpus (1557 clips) for 10 epochs on LLM-T1-NC80
(single GPU — train_first.py does not support DDP). Best checkpoint: **epoch 10, val_loss=0.740**.
Checkpoint handed off to t0005 for Stage 2. Note: ran on GPU0 only due to plain python3 invocation
(not accelerate launch) — silently half-speed but trained correctly.

## Key Metrics

| Metric | Value |
| --- | --- |
| Best val loss | 0.740 |
| Best epoch | 10 |
| Epochs completed | 10 |
| Machine | LLM-T1-NC80 (1× H100 NVL, GPU0 only) |
| Training clips | 1557 (v5 phoneme-corrected) |

## Training Curve

| Epoch | Val loss |
| --- | --- |
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

## Outcome

Best checkpoint `epoch_1st_00007.pth` (val=0.740) used as Stage 2 starting point in t0005.
