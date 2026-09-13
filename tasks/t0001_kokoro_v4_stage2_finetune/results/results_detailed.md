# Results Detailed: Kokoro v4 Stage 2 Fine-tune

## Methodology

- **Machine**: LLM-T1-NC80, 2× H100 NVL, Azure ML
- **Training harness**: `semidark/StyleTTS2` via `accelerate launch --num_processes 2` (DDP)
- **Dataset**: 1557 train clips, 96 val clips (ElevenLabs David v4)
- **Config**: `code/config_david_v4.yml` — `batch_size=2`, `lr=3e-5`, `lambda_slm=0.0`,
  `train_LM=false`, `joint_epoch=3`, `load_only_params=true`
- **Stage 1 starting point**: `best_v4/first_stage.pth` (epoch 10, val=0.594)

## Training Results

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

GAN discriminators activated at `joint_epoch=3` (epoch 4). Best checkpoint: epoch 6 (val=0.751).
Divergence occurred at epoch 7 (val 0.751 → 1.147 in a single epoch). Run terminated at epoch 8
with no recovery.

## Known Fixes Applied

1. `lambda_slm == 0` guard around `slmadv()` — prevents WavLM shape mismatch crash
2. `lr=3e-5`, `ft_lr=3e-5`, `bert_lr=1e-6` — NaN fix (1e-4 caused collapse at epoch 4)
3. `train_LM: false` — prevents LM Loss explosion
4. `batch_size=2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — OOM fix at GAN phase

## Root Cause (Identified Retrospectively in t0003)

21% of the training manifest (329/1557 lines) contained raw English orthography instead of IPA
phonemes. The `prepare_v4_data.py` phonemizer silently fell back to raw text on out-of-vocabulary
words (Rezolve=273 occurrences, Ai=46, brainpowa=4, etc.). StyleTTS2's `TextCleaner` accepts ASCII
letters, so no crash or warning occurred. This injected noise into the duration predictor, causing
both the epoch-7 divergence and t0002's 10× duration explosion.

## Files

- `results/checkpoints/epoch_2nd_00002.pth.dvc` — epoch 2 checkpoint
- `results/checkpoints/epoch_2nd_00004.pth.dvc` — epoch 4 checkpoint
- `results/checkpoints/epoch_2nd_00005.pth.dvc` — epoch 5 checkpoint (used in t0002)
- `results/checkpoints/first_stage.pth.dvc` — Stage 1 checkpoint used
- `results/audio_check/` — sample audio from epoch 2
- `results/audio_check_epoch3/` — sample audio from epoch 3
- `plan/plan.md` — execution plan
- `research/research_internet.md` — background and known issues

## Analysis

The epoch-7 divergence (val 0.751 → 1.147) is consistent with GAN discriminator destabilization.
The same pattern appeared in all failed v5 runs (t0005) until the dataset scale was reduced (t0006).
However, in t0001's case the primary cause was corrupted training data, not dataset scale alone —
v3 ran successfully on 266 clips with the same `joint_epoch=3` configuration.

## Next Steps

Executed in subsequent tasks:
- t0002: voicepack/decoder packaging (validated pipeline, isolated defect to training data)
- t0003: v5 phoneme manifest regeneration (fixed corrupted data)
- t0004: v5 Stage 1 re-run
- t0005: v5 Stage 2 re-run
- t0006: v5 Stage 2 subset (250 clips, multispeaker=false)
