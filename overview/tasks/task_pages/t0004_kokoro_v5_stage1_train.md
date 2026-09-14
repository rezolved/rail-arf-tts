# ✅ Kokoro v5 Stage 1 fine-tune

[Back to all tasks](../README.md)

## Overview

| Field | Value |
|---|---|
| **ID** | `t0004_kokoro_v5_stage1_train` |
| **Status** | ✅ completed |
| **Started** | 2026-09-12T14:00:00Z |
| **Completed** | 2026-09-12T23:00:00Z |
| **Duration** | 9h 0m |
| **Dependencies** | [`t0003_kokoro_v5_phoneme_data`](../../../overview/tasks/task_pages/t0003_kokoro_v5_phoneme_data.md) |
| **Task types** | `tts-finetuning-eval` |
| **Expected assets** | 1 model |
| **Step progress** | 5/5 |
| **Task folder** | [`t0004_kokoro_v5_stage1_train/`](../../../tasks/t0004_kokoro_v5_stage1_train/) |
| **Detailed results** | [`results_detailed.md`](../../../tasks/t0004_kokoro_v5_stage1_train/results/results_detailed.md) |

<details>
<summary><strong>Task Description</strong></summary>

*Source:
[`task_description.md`](../../../tasks/t0004_kokoro_v5_stage1_train/task_description.md)*

# t0004 — Kokoro v5 Stage 1 fine-tune

## Objective

Run StyleTTS2 Stage 1 on t0003's phoneme-corrected corpus (`data/v5/`), producing a
`first_stage.pth` to hand off to Stage 2. Stage 1 must be redone from scratch — t0001's Stage
1 was trained on the same corrupted grapheme/phoneme mixture as its Stage 2, so its checkpoint
cannot be reused (see t0003's Stage 1 duration probe: it explodes exactly like v4's).

## Background

t0001's Stage 2 run diverged; t0003 traced the likely cause to 21% of the training manifest
holding raw text instead of phonemes. t0003 rebuilt clean manifests but did not train
anything. This task is the actual run, plus the operational tooling t0001 lacked from the
start: a sync script that also prunes old checkpoints (both locally and on the ephemeral VM
disk) and shows epoch progress, so a long run doesn't need constant manual log-tailing.

## Approach

1. Provision `/mnt/kikiri-tts` on LLM-T1-NC80: clone the `semidark/StyleTTS2` and
   `semidark/kokoro` submodules, build a venv, upload the v4 wavs and v5 manifests.
2. Launch `train_first.py` against `config_david_v5.yml` (10 epochs, matching v3's schedule).
3. Run `code/sync_and_monitor.sh` locally: polls the VM every 30s, rsyncs the log and any new
   `epoch_1st_*.pth`, prints an epoch progress bar, and prunes to the top-2 checkpoints by
   `val_loss` — on both the local copy and the VM (the VM disk is ephemeral, but there's no
   reason to let 10 epochs of checkpoints pile up mid-run either).
4. Stop and report if the log shows a crash; otherwise stop when the configured final epoch is
   reached.

## Environment setup notes (for whoever runs this next)

Getting `train_first.py` to launch on a fresh VM took five sequential fixes, none related to
the data or config — pure environment gaps in the from-scratch `/mnt/kikiri-tts` setup:

1. `pandas` was missing from the initial pip install list (`meldataset.py` imports it
   directly).
2. `tensorboard` was missing (`torch.utils.tensorboard.SummaryWriter`).
3. The pip package `monotonic_align` only ships `maximum_path` — `utils.py` also imports
   `mask_from_lens`, which the real StyleTTS2 repo defines itself but this pip package
   doesn't. Patched it directly into the installed package
   (`venv/lib/python3.10/site-packages/monotonic_align/__init__.py`).
4. A stray `/mnt/hf_home_cache` default (not from any visible env var or dotfile — never fully
   traced) is not writable; launch with `HF_HOME`/`HF_HUB_CACHE` pointed at
   `/mnt/kikiri-tts/hf_cache` explicitly.
5. `transformers==5.17.0` refuses `torch.load` on torch < 2.6 for non-safetensors checkpoints
   (`microsoft/wavlm-base-plus`, loaded unconditionally by `WavLMLoss.__init__` even though
   `lambda_slm: 0.0` means the loss is never used). Downgraded to `transformers==4.46.3`,
   which predates the check.

## Verification criteria

- `train_first.py` completes 10 epochs without a traceback.
- **Stop-early signal, carried over from t0003**: this is Stage 1, so Dur/CE loss aren't
  logged yet — the real check is at Stage 2's first step (Dur ~0.8 = healthy, 8-17 = the data
  hypothesis is wrong). Recorded here for continuity into the next task.
- Exactly 2 checkpoints survive locally and on the VM at any point after epoch 3.

</details>

## Remote Machines

| Provider | GPU | Count | RAM | Duration | Cost |
|----------|-----|-------|-----|----------|------|
| azure_ml | — | 2 | — GB | — | — |

## Metrics

| Metric | Value |
|--------|-------|
| [`task_id`](../../metrics-results/task_id.md) | **t0004_kokoro_v5_stage1_train** |
| [`metrics`](../../metrics-results/metrics.md) | **[{'name': 'best_val_loss', 'value': 0.74, 'unit': None, 'variant': None}, {'name': 'best_epoch', 'value': 10, 'unit': None, 'variant': None}, {'name': 'epochs_completed', 'value': 10, 'unit': None, 'variant': None}, {'name': 'train_clips', 'value': 1557, 'unit': None, 'variant': None}]** |

<details>
<summary><strong>Results Summary</strong></summary>

*Source:
[`results_summary.md`](../../../tasks/t0004_kokoro_v5_stage1_train/results/results_summary.md)*

# Results Summary: Kokoro v5 Stage 1 Fine-tune

## Summary

Ran StyleTTS2 Stage 1 on v5 phoneme-corrected corpus (1557 clips) for 10 epochs on LLM-T1-NC80
(single GPU — train_first.py does not support DDP). Best checkpoint: **epoch 10,
val_loss=0.740**. Checkpoint handed off to t0005 for Stage 2. Note: ran on GPU0 only due to
plain python3 invocation (not accelerate launch) — silently half-speed but trained correctly.

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

</details>

<details>
<summary><strong>Detailed Results</strong></summary>

*Source:
[`results_detailed.md`](../../../tasks/t0004_kokoro_v5_stage1_train/results/results_detailed.md)*

# t0004 — Stage 1 Results

**Machine**: LLM-T1-NC80 (2× H100 NVL, single-GPU via plain `python3 train_first.py`)
**Data**: 1557 clips (v5 phoneme-corrected corpus) **Config**:
`code/config_david_v5_stage1.yml`, 10 epochs

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

Note: t0004 launched `train_first.py` as plain `python3` (not `accelerate launch`), so it ran
on GPU0 only (1 of 2). Silently half-speed but trained correctly. See t0005 background for
details.

## Files Produced

- `results/checkpoints/epoch_1st_00003.pth.dvc` — epoch 3 checkpoint
- `results/checkpoints/epoch_1st_00007.pth.dvc` — epoch 10 checkpoint (best; handed off to
  t0005)

</details>
