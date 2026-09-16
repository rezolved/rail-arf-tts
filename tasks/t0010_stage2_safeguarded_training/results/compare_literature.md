---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
date_compared: "2026-09-16"
---
# Comparison with Published Results

## Summary

This task ran the first controlled Kokoro-82M Stage 2 fine-tuning implementing two root-cause fixes
from t0009 forensics: the DP-aware checkpoint loader and `joint_epoch=8`. Kokoro-82M follows the
StyleTTS2 architecture [Li2024], which uses an adversarial GAN stage activated mid-training to
refine acoustic quality. The primary training-side metric available is val_loss per epoch; speaker
similarity (GE2E cosine), TTFB, and RTF evaluations are deferred due to VM disk full at teardown.
Compared to the prior controlled run (v6c, which had both a DP loader bug and `joint_epoch=6`), v10
achieves a best val_loss of **0.797** versus v6c's **0.849** — a **−6% improvement** — with zero
health gate events across 17 epochs. The gap to the project speaker similarity target (**speaker_sim
≥ 0.85**) cannot be measured until harness eval runs.

## Comparison Table

| Method / Paper | Metric | Published Value | Our Value | Delta | Notes |
| --- | --- | --- | --- | --- | --- |
| Kokoro-v6c (t0009 forensics) | val_loss (best epoch) | 0.849 | **0.797** | −0.052 | v6c had DP loader bug + joint_epoch=6; v10 fixes both; lower is better |
| ElevenLabs David — target system (t0008) | speaker_sim (GE2E cosine, fillers) | 0.832 | — | — | v10 harness eval deferred; project target ≥ 0.85 |
| Kokoro v3_bundle — prior best (t0008) | speaker_sim (GE2E cosine, fillers) | 0.631 | — | — | v10 harness eval deferred; v3_bundle is current best Kokoro result |

## Methodology Differences

**v10 vs v6c (val_loss comparison):**

- **Joint epoch**: v6c used `joint_epoch=6`; v10 uses `joint_epoch=8`, giving the decoder two
  additional pre-GAN epochs before adversarial gradients are introduced.
- **DP-aware checkpoint loader**: v6c's loader used `strict=False` without stripping `module.`
  prefixes, silently loading zero parameters from the DataParallel checkpoint. v10 strips `module.`
  keys and raises `RuntimeError` on zero-parameter match.
- **Epoch budget**: v6c planned 10 epochs (but defaulted to 200 via an `epochs_2nd` key that Kokoro
  ignores); v10 explicitly sets `epochs: 20`. v10 ran 17 epochs before disk fill; v6c ran 6.
- **Safeguard library**: both runs used the t0009 safeguard library (JSONL logger, health gates,
  per-epoch checkpoints), but v6c's `CheckpointManager` was called without `joint_epoch` — a bug
  that was fixed in v10's `train_second_v10.py` at line 403.

**v10 vs ElevenLabs David / v3_bundle (speaker_sim comparison):**

- **Evaluation not yet run**: v10 harness eval is deferred due to VM disk full at teardown. The
  ElevenLabs David and v3_bundle numbers come from t0008 baseline measurements using the
  `tts_eval_harness` library with 96 held-out val clips.
- **Architecture difference**: ElevenLabs David is a black-box commercial TTS system. Kokoro-82M is
  an open-weight 82M-parameter model following the StyleTTS2 architecture [Li2024], fine-tuned from
  a pretrained Stage 1 checkpoint (`first_stage_v3.pth`) on a 1557-clip v5 corpus.
- **GE2E scoring**: speaker similarity is computed as cosine distance using the GE2E encoder from
  resemblyzer, comparing synthesized fillers against 1358 ElevenLabs David reference clips. Both v10
  and the t0008 baselines use the same scorer.

## Analysis

**GAN activation at joint_epoch=8 produced the expected val_loss improvement.** Val_loss dropped
**49%** from **2.067** (epoch 8, last pre-GAN) to **1.079** (epoch 9, first GAN epoch). This matches
the adversarial training dynamics described for StyleTTS2 [Li2024]: the GAN discriminators reshape
the acoustic distribution, causing an immediate large reduction in validation reconstruction loss
when activated. In contrast, v6c's early GAN activation at `joint_epoch=6` caused instability within
3 epochs — consistent with the StyleTTS2 recommendation for sufficient pre-GAN convergence.

**Val_loss improvement over v6c is real but its link to speaker_sim is indirect.** Val_loss is a
training proxy metric; it measures reconstruction quality but not perceptual speaker similarity. The
−0.052 improvement (**−6%**) in best val_loss (0.849 → 0.797) indicates that v10 found a better
acoustic optimum than the DP-bug-corrupted v6c run. However, without speaker_sim curves across
epochs, it is not yet known whether epoch 16 (val_loss 0.797) also maximizes speaker_sim or whether
a different epoch would be preferred by the GE2E metric.

### Prior Task Comparison

| Metric | v6c — t0009 (prior) | v10 — t0010 (this task) | Delta |
| --- | --- | --- | --- |
| Best val_loss | 0.849 (epoch 6) | **0.797** (epoch 16) | −0.052 (−6%) |
| GAN activation epoch | 6 | 8 | +2 |
| Health gate events | 2 | **0** | −2 |
| Epochs completed | 6 of 10 | 17 of 20 | +11 |
| speaker_sim | null (no eval run) | null (deferred) | — |

The **−6% val_loss improvement** and **zero health gate events** over 17 epochs confirm that the two
t0009 fixes (DP-aware loader + `joint_epoch=8`) produced a materially healthier training run. The
v6c result was compromised from epoch 1 by silent random initialization; v10's parameter-count
assertion passed at startup, confirming correct Stage 1 weight loading.

The gap to the Kokoro v3_bundle speaker_sim baseline (**0.631**, t0008) remains unknown for v10
until harness eval runs. The project target of **speaker_sim ≥ 0.85** requires a delta of at least
**+0.219** GE2E cosine units from the v3_bundle baseline. Val_loss descent alone cannot confirm that
this target has been reached.

## Limitations

1. **No registered paper corpus.** The literature search step was skipped for this task; no
   research_papers.md was generated. The StyleTTS2 paper [Li2024] is the architecture reference but
   no specific published val_loss or GE2E numbers from the paper are available to compare against
   directly.

2. **speaker_sim, TTFB, and RTF are null.** The t0008 harness could not run because both the VM
   ephemeral disk and root disk were 100% full at teardown. The ElevenLabs David (0.832) and
   v3_bundle (0.631) speaker_sim numbers come from t0008 baseline measurements, not from v10. The
   comparison table uses "—" for our v10 values on those rows.

3. **Val_loss is not a direct proxy for speaker_sim.** The −6% improvement in val_loss over v6c is
   informative but does not guarantee a proportional improvement in speaker similarity. The two
   metrics may peak at different checkpoints.

4. **Intra-project baselines only.** The primary numeric comparison in the table (v6c vs v10
   val_loss) is an intra-project comparison, not a literature comparison. Without registered papers
   and without speaker_sim data, no head-to-head comparison against published TTS systems is
   possible at this time.

5. **Only 17 of 20 planned epochs completed.** The training run ended at disk full, so val_loss may
   not have reached its true minimum. The epoch 17 rebound (0.797 → 0.853) is ambiguous without
   speaker_sim data to confirm whether epoch 16 is the generalization optimum.
