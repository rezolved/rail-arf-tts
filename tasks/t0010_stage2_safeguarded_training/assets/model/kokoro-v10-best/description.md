---
spec_version: "2"
model_id: "kokoro-v10-best"
documented_by_task: "t0010_stage2_safeguarded_training"
date_documented: "2026-09-16"
---
# Kokoro-82M Stage 2 v10 Best Checkpoint

## Metadata

- **Name**: Kokoro-82M Stage 2 v10 Best Checkpoint
- **Version**: 1.0.0
- **Framework**: pytorch
- **Base model**: kokoro-82m (first_stage_v3.pth, from t0008_tts_eval_harness_baselines)
- **Training task**: t0010_stage2_safeguarded_training
- **Date created**: 2026-09-16

## Overview

This model is the primary output of t0010, the first controlled Kokoro-82M Stage 2 fine-tuning
experiment at Rezolve that implements both root-cause fixes identified in t0009: (1) DP-aware
checkpoint loader with `module.`-prefix stripping and parameter-count assertion at startup, and (2)
`joint_epoch=8` (delayed GAN activation, up from 6 in the prior v6c run that crashed at epoch 9).

The training used the t0009 safeguard library — JSONL step logger, per-epoch CheckpointManager with
SHA-256 manifest, and health gates for dur_loss, acoustic_norm, val_spike, and consecutive_skip — to
instrument training and detect divergence early. No health gate events were recorded across 17
completed epochs, confirming that the two root-cause fixes resolved the crash observed in t0006/v6c.

This checkpoint is intended as a drop-in replacement for ElevenLabs David voice in Rezolve's voice
commerce filler synthesis pipeline. The project success criteria are GE2E cosine speaker similarity
≥ 0.85 vs ElevenLabs David reference and TTFB ≤ 300 ms. Harness evaluation was deferred due to
ephemeral disk saturation on LLM-T1-NC80; see `intervention/eval_deferred_disk_full.md` for the
resolution path.

## Architecture

Kokoro-82M is an 82-million-parameter StyleTTS2 speech synthesis model. Stage 2 fine-tuning adds a
GAN phase beginning at `joint_epoch=8` (epoch 9 in 1-indexed terms). The key architectural
components packaged in this checkpoint are:

- **bert**: Phoneme-level BERT encoder (25 parameter tensors)
- **bert_encoder**: Lightweight adapter on top of BERT (2 parameter tensors)
- **predictor**: Duration and F0 predictor (122 parameter tensors)
- **text_encoder**: Text encoder mapping phoneme sequences to style-conditioned hidden states (24
  parameter tensors)
- **decoder**: StyleTTS2 mel-spectrogram decoder with adversarial training (678 parameter tensors)

The model is packaged as a 5-module state dict (317 MB) produced by
`tasks.t0008_tts_eval_harness_baselines.code.extract_decoder.extract()`. Weight-norm layers are
converted from the `.parametrizations.weight.original0/1` training format to the production
`.weight_g/.weight_v` format. DDP `module.` prefixes are stripped from all keys.

The full raw checkpoint (`epoch_2nd_00016.pth`, 2 GB) is separately DVC-tracked at
`data/run_v10/epoch_2nd_00016.pth.dvc`.

## Training

**Dataset**: 1557 Rezolve filler synthesis audio clips from `data/data_list_v5_train_250.txt`
(train) and 96 held-out val clips from `data/val_list.txt`.

**Hardware**: LLM-T1-NC80 — Azure ML `Standard_NC80adis_H100_v5`, 2 × H100 NVL SXM5 (80 GB VRAM
each). Training used a single GPU (no DataParallel during Stage 2).

**Config** (v10 relative to v6c):

| Parameter | v6c | v10 |
| --- | --- | --- |
| `joint_epoch` | 6 | **8** |
| `epochs` | 10 | **20** |

All other hyperparameters unchanged: `lr=1e-4`, `lambda_gen=1.0`, `batch_size=8`,
`multispeaker=true`, `train_LM=false`.

**Training run**: started 2026-09-15T15:18:01Z, stopped 2026-09-15T16:00:55Z (ephemeral disk full
after epoch 17). 17 epochs completed of the planned 20.

**Loss trajectory**:

- Epochs 1-8 (pre-GAN): val_loss ~2.07 (stable)
- Epoch 9: val_loss 1.079 (GAN kicked in at joint_epoch=8, as designed)
- Epoch 10-16: 0.969 → 0.797 (epoch 16 is the best val_loss)
- Epoch 17 (this checkpoint): val_loss 0.853

**Health gates**: 0 events across all 17 epochs. Parameter-count assertion passed at startup (Stage
1 weights confirmed loaded, ≥80% match).

## Evaluation

Harness evaluation (speaker_sim, TTFB, RTF) was deferred because the VM ephemeral disk reached 100%
capacity after epoch 17, preventing synthesis of audio clips needed for the GE2E scoring pass.

- **Best val_loss**: 0.797 at epoch 16
- **This checkpoint val_loss**: 0.853 (epoch 17, latest completed epoch, selected per user
  instruction)
- **speaker_sim**: null (eval deferred)
- **ttfb_ms**: null (eval deferred)
- **rtf**: null (eval deferred)

Baseline for comparison: v3_bundle speaker_sim = 0.631 (t0008), ElevenLabs David target = 0.85.

See `intervention/eval_deferred_disk_full.md` for the procedure to run evaluation manually.

## Usage Notes

Load with the `tts_eval_harness` library (t0008):

```python
from tasks.t0008_tts_eval_harness_baselines.code.adapters import load_kokoro_model_with_checkpoint
model = load_kokoro_model_with_checkpoint(
    checkpoint_path="tasks/t0010_stage2_safeguarded_training/assets/model/kokoro-v10-best/files/kokoro-v10-best.pth",
    config_path="tasks/t0010_stage2_safeguarded_training/assets/model/kokoro-v10-best/files/config_david_v10.yml",
    device="cuda",
)
```

**Known limitations**:

- Only 17 of 20 planned epochs completed; the model may benefit from continued training.
- Harness evaluation not yet run; speaker_sim vs ElevenLabs David reference is unknown.
- `multispeaker=true` — requires a speaker embedding at inference time; use the David voice
  embedding from `data/david_v3_best_voicepack.pt`.
- Extraction uses CPU-safe `map_location="cpu"` — no GPU required for loading.

## Main Ideas

* `joint_epoch=8` successfully delayed GAN activation to epoch 9, eliminating the gradient explosion
  crash seen in v6c (which had `joint_epoch=6`). Val_loss fell from 2.07 to 0.797 across 17 epochs
  with 0 health gate events — a clean training run.
* The DP-aware checkpoint loader (module.-prefix stripping + parameter-count assertion) is critical
  for correct Stage 1 weight initialization. Without it, training loads 0 parameters and trains from
  random weights (the root cause of all prior failures).
* Training stopped at epoch 17 due to disk saturation, not divergence. The best val_loss is at epoch
  16 (0.797); this checkpoint (epoch 17, val_loss 0.853) was selected per user instruction as the
  latest epoch. If speaker_sim peaks earlier (e.g., epoch 14-16), the backup checkpoint
  `epoch_2nd_00014.pth` (val_loss 0.818) is available.

## Summary

Kokoro-v10-best is a StyleTTS2 Stage 2 fine-tune of the 82M-parameter Kokoro model, trained on 1557
Rezolve filler clips with `joint_epoch=8` and a DP-safe checkpoint loader. It is the first
successful Rezolve Stage 2 run — 17 epochs completed without divergence, val_loss declining from
2.07 to a minimum of 0.797. The model is packaged as a 5-module state dict (317 MB) ready for direct
loading via the tts_eval_harness adapter.

Harness evaluation (speaker_sim vs ElevenLabs David centroid, TTFB, RTF) was not completed due to
ephemeral disk saturation on LLM-T1-NC80. The checkpoint's val_loss trajectory is promising — the
GAN phase activated cleanly at epoch 9 and continued improving for 8 more epochs. Whether this
translates to speaker similarity ≥ 0.85 (the project target) requires the deferred eval run.

The primary follow-up is running `eval_all_checkpoints.py` after freeing VM disk space, and then
deciding whether to continue training for epochs 18-20 based on the speaker_sim trajectory.
