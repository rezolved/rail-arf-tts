# t0006 — Kokoro v5 Stage 2: 250-clip subset, multispeaker: false

## Hypothesis

t0005 run 6 passed the joint_epoch boundary (first time!) thanks to the `istftnet.py exp()` clamp,
but val_loss diverged 0.848 → 2.07 across 8 post-GAN epochs. Analysis of v3 (val 0.506, the
working benchmark) revealed two key differences from t0005:

1. **Data scale**: v3 used 266 clips (31 steps/epoch). t0005 used 1557 clips (194 steps/epoch).
   Larger dataset → more diverse discriminator batches → stronger GAN gradient at joint_epoch.
2. **multispeaker: false** in v3 vs **true** in t0005. Single-speaker makes the discriminator's
   task easier at the joint_epoch transition and reduces gradient magnitude.

This task tests both in combination: 250 clips (seed=42 random sample from v5's 1557, matching v3's
scale) plus `multispeaker: false`, with all t0005's proven fixes retained. One deliberate change
at a time — lr and lambda_gen are NOT changed (keeping 3e-5 and 1.0) so the data+multispeaker
effect is isolated.

## Changes from t0005 run 6

| Parameter | t0005 run 6 | t0006 | Reason |
|-----------|-------------|-------|--------|
| `train_data` | 1557 clips | 250 clips (train_list_250.txt) | Match v3's data scale |
| `multispeaker` | true | false | Match v3's recipe; single-speaker is safe for David-only |
| `log_dir` | kokoro-david-v5 | kokoro-david-v6 | Separate logs |
| All other params | — | unchanged | Isolate the variables |

## Retained from t0005

All seven fixes from t0005 run 6:
1. `lambda_slm == 0` guard around `slmadv()` (crash #1 fix)
2. LR reverted to v4-safe values: `lr/ft_lr: 3e-5`, `bert_lr: 1e-6` (crash #2 fix)
3. `train_LM: false` guard (crash #3 fix)
4. `clip_grad_norm_` on msd/mpd/decoder/style_encoder (crash #3 fix)
5. `set_detect_anomaly(False)` (crash #4 fix)
6. Finite-grad skip guard + 50-consecutive-skip abort (crash #5 fix)
7. `torch.exp(torch.clamp(..., max=15.0))` in `istftnet.py:523/541` (crash #6 root-cause fix)

## Expected outcome

If the hypothesis is correct: GAN activates at epoch 4 without divergence (matching v3's
0.549 at epoch 4), val_loss trends toward v3's 0.506 benchmark.

Health gate at epoch 4 (first GAN epoch): `acoustic_norm` must stay < 20 and val_loss must not
exceed epoch 3's value by more than 0.05. If it does → data scale is not the cause, and we need
to reduce `lambda_gen`.

## Results Summary (4 runs)

| Run | Config | lambda_gen | joint_epoch | Best val | acoustic_norm | Audio |
|-----|--------|-----------|-------------|----------|---------------|-------|
| run01 v6 | v6 | 1.0 | 3 | ~0.89 | 10→27 | diverged ep7 |
| run02 v6b | v6b | 0.2 | 3 | 0.895 (ep4) | 10→61 | diverged ep7 |
| run03 v6c | v6c | 0.2 | 3 | 0.849 (ep5) | 0.37→6.07 | noisy |
| run04 v6d | v6d | 0.05 | 6 | **0.846 (ep6)** | 8.51 | **noisy** |

v6b used wrong first_stage.pth (multispeaker:true mismatch). v6c/v6d used v3's first_stage.pth
(multispeaker:false), which correctly drops the baseline acoustic_norm from 10 to 0.36.

**Target not reached**: best val=0.846 vs v3 gold standard 0.506. Audio remains noisy across all
runs. The GAN activates stably (lambda_gen=0.05 prevented divergence) but 10 epochs is
insufficient for the model to converge to clean synthesis quality.

## Recommendation for Next Task

Three options, in priority order:

1. **Check v3 original config** (5 min, free) — v3 reached val=0.569 at epoch 1, far better than
   our pre-GAN baseline of 1.51. If v3 used `joint_epoch: 0` (GAN active from ep1), this explains
   the gap. Understanding v3's exact settings before launching more compute is the cheapest step.

2. **Continue from v6d ep6** — resume training from `epoch_2nd_00006.pth` (val=0.846, stable) with
   corrected `epochs_2nd: 20`. The model is stable; it may simply need more GAN epochs to converge.
   Fix required: set `epochs_2nd: 20` (was accidentally left at 10 in v6d config, so training
   stopped at epoch 10 instead of 15 as intended).

3. **v6e: weaker GAN + longer warmup** — `lambda_gen: 0.01`, `joint_epoch: 8`, `epochs: 25`.
   Fresh start with 7 pre-GAN warmup epochs and near-zero GAN weight.

**Recommended sequence**: do option 1 first, then decide between 2 and 3 based on v3's config.

## Files

- `code/config_david_v6_stage2.yml` — run01 config
- `code/config_david_v6b_stage2.yml` — run02 config
- `code/config_david_v6c_stage2.yml` — run03 config (v3 first_stage.pth, lambda_gen=0.2)
- `code/config_david_v6d_stage2.yml` — run04 config (lambda_gen=0.05, joint_epoch=6)
- `code/infer_v6c.py` — inference script for v6c (en-gb G2P, strips DataParallel prefix)
- `code/infer_v6d.py` — inference script for v6d (same, points to v6d ep6 checkpoint)
- `code/train_list_250.txt` — 250-clip subset list (local copy; VM: `/mnt/kikiri-tts/data/v5/train_list_250.txt`)
- `logs/` — training logs, synced from VM
- `results/audio_v6c_ep5/` — v6c synthesis audio (noisy)
- `results/audio_v6d_ep6/` — v6d synthesis audio (noisy)
- `results/checkpoints/` — top-2 checkpoints by val_loss (synced)
- `reference/v3/` — DVC-tracked v3 gold standard assets (Stage 1 checkpoint, best voicepack, decoder, reference audio)
