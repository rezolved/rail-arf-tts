# Results Summary: Kokoro v4 Stage 2 Fine-tune

## Summary

Trained StyleTTS2 Stage 2 on LLM-T1-NC80 (2× H100 NVL) for 8 epochs on 1557 ElevenLabs David v4
clips. Best checkpoint: **epoch 6, val_loss=0.751**. GAN diverged at epoch 7 (val 0.751 → 1.147),
cutting the run short. Root cause (discovered in t0003): 21% of the training manifest contained raw
grapheme text instead of phonemes. The epoch 6 checkpoint was used as source for voicepack extraction
in t0002 but produced a broken duration predictor (10× duration explosion).

## Key Metrics

| Metric | Value |
| --- | --- |
| Best val loss | 0.751 |
| Best epoch | 6 |
| Epochs completed | 8 (diverged at 7) |
| Machine | LLM-T1-NC80 (2× H100 NVL) |
| Training clips | 1557 |
| Val clips | 96 |

## Training Curve

| Epoch | Val loss |
| --- | --- |
| 1 | 0.820 |
| 2 | 0.801 |
| 3 | 0.770 |
| 4 | 0.815 |
| 5 | 0.763 |
| **6** | **0.751** ← best |
| 7 | 1.147 |
| 8 | 1.129 |

## Outcome

Checkpoint not usable for production. Root cause identified in t0003 (corrupted training data).
v5 re-run required (executed in t0004/t0005/t0006).
