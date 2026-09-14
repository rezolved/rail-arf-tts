# ✅ Kokoro v4 Stage 2 fine-tune

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0001_kokoro_v4_stage2_finetune` |
| **Status** | ✅ completed |
| **Started** | 2026-09-10T08:00:00Z |
| **Completed** | 2026-09-11T19:00:00Z |
| **Duration** | 35h 0m |
| **Task types** | `tts-finetuning-eval` |
| **Expected assets** | 1 model |
| **Step progress** | 5/5 |
| **Task folder** | [`t0001_kokoro_v4_stage2_finetune/`](../../../tasks/t0001_kokoro_v4_stage2_finetune/) |
| **Detailed results** | [`results_detailed.md`](../../../tasks/t0001_kokoro_v4_stage2_finetune/results/results_detailed.md) |

<details>
<summary><strong>Task Description</strong></summary>

*Source:
[`task_description.md`](../../../tasks/t0001_kokoro_v4_stage2_finetune/task_description.md)*

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

Best: **epoch 6, val=0.751**. GAN diverged at epoch 7. Root cause (discovered in t0003): 21%
of the training manifest contained raw grapheme text instead of phonemes — corrupted data
caused the divergence. This checkpoint was still used as the source for voicepack extraction
in t0002.

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

</details>

## Remote Machines

| Provider | GPU | Count | RAM | Duration | Cost |
|----------|-----|-------|-----|----------|------|
| azure_ml | — | 2 | — GB | — | — |

## Metrics

| Metric | Value |
|--------|-------|
| [`task_id`](../../metrics-results/task_id.md) | **t0001_kokoro_v4_stage2_finetune** |
| [`metrics`](../../metrics-results/metrics.md) | **[{'name': 'best_val_loss', 'value': 0.751, 'unit': None, 'variant': None}, {'name': 'best_epoch', 'value': 6, 'unit': None, 'variant': None}, {'name': 'epochs_completed', 'value': 8, 'unit': None, 'variant': None}, {'name': 'divergence_epoch', 'value': 7, 'unit': None, 'variant': None}, {'name': 'train_clips', 'value': 1557, 'unit': None, 'variant': None}, {'name': 'val_clips', 'value': 96, 'unit': None, 'variant': None}]** |

## Research

* [`research_internet.md`](../../../tasks/t0001_kokoro_v4_stage2_finetune/research/research_internet.md)

<details>
<summary><strong>Results Summary</strong></summary>

*Source:
[`results_summary.md`](../../../tasks/t0001_kokoro_v4_stage2_finetune/results/results_summary.md)*

# Results Summary: Kokoro v4 Stage 2 Fine-tune

## Summary

Trained StyleTTS2 Stage 2 on LLM-T1-NC80 (2× H100 NVL) for 8 epochs on 1557 ElevenLabs David
v4 clips. Best checkpoint: **epoch 6, val_loss=0.751**. GAN diverged at epoch 7 (val 0.751 →
1.147), cutting the run short. Root cause (discovered in t0003): 21% of the training manifest
contained raw grapheme text instead of phonemes. The epoch 6 checkpoint was used as source for
voicepack extraction in t0002 but produced a broken duration predictor (10× duration
explosion).

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

Checkpoint not usable for production. Root cause identified in t0003 (corrupted training
data). v5 re-run required (executed in t0004/t0005/t0006).

</details>

<details>
<summary><strong>Detailed Results</strong></summary>

*Source:
[`results_detailed.md`](../../../tasks/t0001_kokoro_v4_stage2_finetune/results/results_detailed.md)*

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

GAN discriminators activated at `joint_epoch=3` (epoch 4). Best checkpoint: epoch 6
(val=0.751). Divergence occurred at epoch 7 (val 0.751 → 1.147 in a single epoch). Run
terminated at epoch 8 with no recovery.

## Known Fixes Applied

1. `lambda_slm == 0` guard around `slmadv()` — prevents WavLM shape mismatch crash
2. `lr=3e-5`, `ft_lr=3e-5`, `bert_lr=1e-6` — NaN fix (1e-4 caused collapse at epoch 4)
3. `train_LM: false` — prevents LM Loss explosion
4. `batch_size=2` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — OOM fix at GAN phase

## Root Cause (Identified Retrospectively in t0003)

21% of the training manifest (329/1557 lines) contained raw English orthography instead of IPA
phonemes. The `prepare_v4_data.py` phonemizer silently fell back to raw text on
out-of-vocabulary words (Rezolve=273 occurrences, Ai=46, brainpowa=4, etc.). StyleTTS2's
`TextCleaner` accepts ASCII letters, so no crash or warning occurred. This injected noise into
the duration predictor, causing both the epoch-7 divergence and t0002's 10× duration
explosion.

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

The epoch-7 divergence (val 0.751 → 1.147) is consistent with GAN discriminator
destabilization. The same pattern appeared in all failed v5 runs (t0005) until the dataset
scale was reduced (t0006). However, in t0001's case the primary cause was corrupted training
data, not dataset scale alone — v3 ran successfully on 266 clips with the same `joint_epoch=3`
configuration.

## Next Steps

Executed in subsequent tasks:
- t0002: voicepack/decoder packaging (validated pipeline, isolated defect to training data)
- t0003: v5 phoneme manifest regeneration (fixed corrupted data)
- t0004: v5 Stage 1 re-run
- t0005: v5 Stage 2 re-run
- t0006: v5 Stage 2 subset (250 clips, multispeaker=false)

</details>
