---
spec_version: "2"
model_id: "kokoro-v11-best"
documented_by_task: "t0014_v11_decoder_fix_retrain"
date_documented: "2026-09-16"
---
# Kokoro-82M Stage 2 v11 Best Checkpoint (decoder-init fix)

## Metadata

- **Name**: Kokoro-82M Stage 2 v11 Best Checkpoint (decoder-init fix)
- **Version**: 1.0.0
- **Framework**: pytorch
- **Base model**: kokoro-82m (Stage 1) + decoder initialized from `yl4579/StyleTTS2-LibriTTS`'s
  `epochs_2nd_00020.pth`
- **Training task**: t0014_v11_decoder_fix_retrain
- **Date created**: 2026-09-16

## Overview

This model fixes the root cause diagnosed by t0013 that left `kokoro-v10-best`'s HiFi-GAN decoder
producing clipped/saturated noise instead of speech: `train_second_v10.py`'s Stage-2 loader passed
`first_stage_path=first_stage_v3.pth`, an ISTFTNet-shaped checkpoint, into a HiFi-GAN-shaped decoder
config, and the loader's zero-match guard only catches a *fully* failed load, not a partial
architecture-mismatched one -- so the decoder's real vocoder layers (`ups`, `resblocks`, `alphas`,
`conv_post`, `noise_convs`) trained from a "Frankenstein" mixed init that t0013 proved is actively
worse than pure random initialization.

This task's fix repoints `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth`
-- a checkpoint whose decoder is genuinely HiFi-GAN-shaped and field-matches
`config_david_v11.yml`'s decoder block exactly, confirmed by a tensor-level pre-flight check
(`results/checkpoint_forensics_v11.md`) before any GPU spend. Training then ran the full 50-epoch
schedule (`joint_epoch=30`, `diff_epoch=10`) on t0012's larger, LUFS-normalized 1,531-clip corpus
(vs. v10's 250-clip subset), with 0 `HealthGate` firings across all 50 epochs. Unlike v10, this
checkpoint's completion claim is gated on an actual audible-speech check
(`code/audio_quality_check.py`'s `check_audio_quality()`), not on `val_loss` alone -- the exact gap
that let v10 ship broken with a healthy-looking loss curve. The gate **passed**:
`is_likely_noise=False` (`clip_fraction=0.0048`, `spectral_flatness=0.0058`, both far outside v10's
confirmed-broken range of `clip_fraction=0.750-0.807`).

## Architecture

Kokoro-82M is an 82-million-parameter StyleTTS2 speech synthesis model. Stage 2 fine-tuning adds a
GAN phase beginning at `joint_epoch=30` (up from v10's `joint_epoch=8`, anchored on StyleTTS2's own
official `Configs/config_ft.yml` fine-tuning recipe at comparable corpus scale instead of a
from-scratch-sized budget). The key components packaged in this checkpoint (mirroring
`kokoro-v10-best`'s exact packaging convention) are:

- **bert**: Phoneme-level BERT encoder (25 parameter tensors)
- **bert_encoder**: Lightweight adapter on top of BERT (2 parameter tensors)
- **predictor**: Duration and F0 predictor (122 parameter tensors)
- **text_encoder**: Text encoder mapping phoneme sequences to style-conditioned hidden states (24
  parameter tensors)
- **decoder**: StyleTTS2 HiFi-GAN mel-spectrogram decoder with adversarial training (678 parameter
  tensors) -- the module this task's fix targets

The model is packaged as a 5-module state dict (331 MB) produced by
`tasks.t0008_tts_eval_harness_baselines.code.extract_decoder.extract()`, the same function and
5-module convention `kokoro-v10-best` used. Weight-norm layers are converted from the
`.parametrizations.weight.original0/1` training format to the production `.weight_g/.weight_v`
format; DDP `module.` prefixes are stripped. The full raw checkpoint (`epoch_00048.pth`, 2 GB, 13
modules including `predictor_encoder`, `style_encoder`, `diffusion`, `text_aligner`,
`pitch_extractor`, `mpd`, `msd`, `wd`) is the file this task's own audible-speech gate validation
actually ran inference against (see Usage Notes -- **important**: this 5-module extract is *not* the
file the gate result below was measured on).

## Training

**Dataset**: 1,531 Rezolve filler synthesis clips from t0012's LUFS-normalized, re-audited corpus
(`train_list_v5_normalized_clean.txt`), plus the unchanged 96-clip held-out `val_96` set
(`data/val_list.txt`) -- confirmed 0 overlap with train data (`results/val96_leak_check.txt`).

**Hardware**: LLM-T1-NC80 -- Azure ML 2xH100 SXM5. Training used a single GPU for Stage 2.

**Config** (v11 relative to v10):

| Parameter | v10 | v11 |
| --- | --- | --- |
| `first_stage_path` | `first_stage_v3.pth` (ISTFTNet-shaped, mismatched) | `epochs_2nd_00020.pth` (HiFi-GAN-shaped, matched) |
| `train_data` | 250-clip subset | 1,531-clip LUFS-normalized corpus |
| `epochs` / `epochs_2nd` | 20 | **50** |
| `diff_epoch` | 6 | **10** |
| `joint_epoch` | 8 | **30** |

All other hyperparameters unchanged: `lr=1e-4`, `ft_lr=1e-4`, `bert_lr=1e-6`, `lambda_gen=1.0`,
`batch_size=8`, `multispeaker=true`, `train_LM=false`.

**Training run**: completed all 50 planned epochs cleanly (tmux log ends with `DONE`), with 0
`HealthGate` firings (`grep -c gate_fired data/run_v11/metrics.jsonl` = 0). Checkpoints saved every
2 epochs; the last saved checkpoint is epoch 48 (only even epochs are saved).

**Loss trajectory** (post-`joint_epoch=30`, i.e. GAN phase active): val_loss runs 0.339-0.367 across
epochs 30-48 with no clear regression at the final saved epoch -- epoch 38 has the numerically
lowest val_loss (0.3394) but epoch 48 (the last saved epoch, val_loss 0.3469) was selected per
plan.md's rule ("latest epoch unless a clear regression is visible"), same precedent as
`kokoro-v10-best`'s epoch selection.

## Evaluation

**Audible-speech gate (this task's sole, pre-registered pass/fail criterion) -- PASS**:
`code/audio_quality_check.py`'s `check_audio_quality()` on `results/audio_samples/ft/v11_best.wav`
(synthesized from the full raw `epoch_00048.pth`, not the packaged 5-module file, via the
StyleTTS2-native `code/infer_styletts2.py` harness) returned `is_likely_noise=False`:
`clip_fraction=0.0048`, `spectral_flatness=0.0058`, `silence_fraction=0.0`. This is a qualitative
reversal of v10's confirmed failure (`clip_fraction=0.750-0.807`, `is_likely_noise=True`). Full
detail and side-by-side comparison table: `results/v11_gate_verdict.md`.

- **speaker_sim**: 0.444 (GE2E cosine vs. 11labs_david centroid) -- higher than both
  confirmed-broken v10 checkpoints (0.311-0.351), but per this task's own findings `speaker_sim` is
  a weak signal for vocoder-level failure (both broken v10 checkpoints still scored non-trivially
  positive), so it is reported as corroborating evidence only, not the pass/fail criterion.
- **rtf**: 3.18 (single-clip, unbatched CPU inference -- corroborating only, not a benchmark claim).
- **ttfb_ms**: not measured -- offline batch harness, no streaming endpoint in scope.
- **Known anomaly**: synthesis output duration (73.95s for a 10-word test sentence) is 15-30x longer
  than the control (4.92s) and both v10 checkpoints (2.50-3.42s) on comparable/identical inputs. The
  audio is not noise (gate passes cleanly), but this points to a likely duration-predictor
  calibration issue distinct from the decoder/vocoder defect this task fixed. See
  `results/v11_gate_verdict.md`'s "Caveat" section and `results/results_summary.md` for the
  recommended follow-up.

Baseline for comparison: v3_bundle speaker_sim = 0.631 (t0008), ElevenLabs David target = 0.85 --
neither v10 nor v11 has yet closed this gap; v11's contribution is fixing the decoder so audible
speech exists at all, not closing the full speaker-similarity gap.

## Usage Notes

**Important -- do not load `files/kokoro-v11-best.pth` via Kokoro's `KModel`/`KPipeline`.** Per this
task's own research (`plan/plan.md` Approach section), Kokoro's `KModel`/`KPipeline` cannot
correctly load a `hifigan`-decoder StyleTTS2 checkpoint -- this is a pre-existing architectural
limitation of this fork's Kokoro inference code, not something this task introduces or fixes. This
5-module `files/kokoro-v11-best.pth` is packaged to mirror `kokoro-v10-best`'s exact convention for
downstream consistency, but it was **not** the file this task's audible-speech gate validated --
that gate ran against the full 13-module `epoch_00048.pth` through `code/infer_styletts2.py`'s
StyleTTS2-native harness (`models.py`'s `build_model`, dispatching on `model_params.decoder.type`),
which is the proven-working inference path for this checkpoint family. To reproduce the
gate-validated inference:

```bash
.venv-styletts2/bin/python code/infer_styletts2.py \
    --checkpoint-path <path-to-epoch_00048.pth-or-full-raw-checkpoint> \
    --config-path files/config_david_v11.yml \
    --text "your text here" \
    --reference-audio <a reference wav >= ~1s, StyleEncoder floor> \
    --output-wav out.wav
```

**Known limitations**:

- Do not use for direct `KModel`/`KPipeline` loading (see above) -- use the StyleTTS2-native
  harness.
- Anomalous output duration (see Evaluation) -- investigate before production use; may indicate a
  duration-predictor calibration issue on the larger/normalized corpus.
- `speaker_sim` (0.444) is well below the project's 0.85 target; this task fixed audibility, not
  speaker similarity.
- `multispeaker=true` -- requires a speaker/style embedding at inference time (`compute_style()` on
  a reference clip of at least ~1s, per `StyleEncoder`'s architectural minimum found by t0013).
- Extraction and packaging use CPU-safe `map_location="cpu"` -- no GPU required for loading the
  5-module file.

## Main Ideas

* The decoder-init bug was an **architecture-mismatch** problem (ISTFTNet-shaped source checkpoint
  loaded into a HiFi-GAN-shaped config), not a missing-`ignore_modules`-entry problem -- the fix is
  repointing `first_stage_path` at a real HiFi-GAN checkpoint, not excluding `decoder` from the load
  and accepting random init.
* A pre-flight tensor-level check (module-name-aware, not just shape-count) **before** any GPU spend
  is essential: the same silent-partial-match failure mode that broke v10 would otherwise repeat
  undetected, since `load_checkpoint()`'s pass condition only raises on a fully-zero match.
* **`val_loss` alone cannot certify a working vocoder** -- v10 had a healthy-looking `val_loss`
  curve across all 17 epochs despite producing pure clipped noise. This task's audible-speech gate
  (`is_likely_noise`) is the discriminator that actually worked, and should be the standard for all
  future Stage-2 completion claims, not just this task's.
* Fixing the decoder does not automatically fix everything: this checkpoint has a separate,
  previously-unobserved duration-predictor anomaly (73.95s output for a 10-word sentence) that
  warrants its own follow-up investigation.

## Summary

Kokoro-v11-best is a StyleTTS2 Stage 2 fine-tune of the 82M-parameter Kokoro model that fixes the
decoder-initialization bug diagnosed in t0013: `first_stage_path` now points at a genuinely
HiFi-GAN-shaped pretrained checkpoint (`yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth`) instead
of an architecturally mismatched one. Trained for the full 50 epochs (`joint_epoch=30`,
`diff_epoch=10`) on t0012's 1,531-clip normalized corpus with 0 health-gate firings, the resulting
checkpoint (epoch 48, val_loss 0.347) **passes** this task's mandatory audible-speech gate
(`is_likely_noise=False`, `clip_fraction=0.0048` vs. v10's confirmed-broken 0.750-0.807) -- the
first Rezolve Stage-2 checkpoint in this fork's history confirmed, by objective audio measurement
rather than loss curves alone, to produce real, non-noise speech.

The model is not yet production-ready: `speaker_sim` (0.444) remains well below the project's 0.85
target, and a newly-discovered duration-predictor anomaly (output ~15-30x longer than expected for a
given input) needs investigation before this checkpoint should be treated as a drop-in ElevenLabs
David replacement. The primary follow-ups are (1) diagnosing the duration anomaly, and (2) continued
fine-tuning or corpus expansion aimed at closing the speaker-similarity gap now that the vocoder
itself is confirmed to work.
