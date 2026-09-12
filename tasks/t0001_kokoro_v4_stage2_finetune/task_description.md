# t0001 — Kokoro v4 Stage 2 Fine-tune

## Objective

Train StyleTTS2 Stage 2 on LLM-T1-NC80 (2× H100 NVL) using the ElevenLabs David v4 dataset,
producing a Kokoro-82M checkpoint for voicepack extraction in t0002.

## Approach

1. Set up VM environment, upload 1557 v4 training clips and Stage 1 checkpoint.
2. Launch `train_second.py` via `accelerate launch --num_processes 2` (DDP).
3. Monitor val loss per epoch; download top checkpoints via `sync_and_monitor.sh`.
4. Best checkpoint → DVC-tracked artifact.

## Results

| Epoch | Val loss |
|-------|----------|
| 1 | 0.820 |
| 2 | 0.801 |
| 3 | 0.770 |
| 4 | 0.815 |
| 5 | 0.763 |
| **6** | **0.751** ← best |
| 7 | 1.147 |
| 8 | 1.129 |

Best: **epoch 6, val=0.751**. GAN diverged at epoch 7. Root cause (discovered in t0003): 21% of
the training manifest contained raw grapheme text instead of phonemes — corrupted data caused the
divergence. This checkpoint was still used as the source for voicepack extraction in t0002.

## Files

- `plan.md` — original planning document
- `research.md` — research and background
- `code/` — setup, upload, and monitoring scripts
- `logs/` — training logs
- `results/checkpoints/epoch_2nd_00002.pth.dvc` — epoch 2 checkpoint
- `results/checkpoints/epoch_2nd_00004.pth.dvc` — epoch 4 checkpoint
- `results/checkpoints/epoch_2nd_00005.pth.dvc` — epoch 5 checkpoint (best pre-divergence)
- `results/checkpoints/first_stage.pth.dvc` — Stage 1 checkpoint used
- `results/audio_check/` — sample audio from epoch 2
- `results/audio_check_epoch3/` — sample audio from epoch 3
