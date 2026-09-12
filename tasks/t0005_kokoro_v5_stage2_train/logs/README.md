# Crash log archive — Stage 2 v5, launches 1-6

Launches 1-5 all died at the same point: the transition from epoch 3 to epoch 4
(`epoch >= joint_epoch`, `joint_epoch: 3`), where MSD/MPD discriminators activate and
`decoder`/`style_encoder` get their first-ever optimizer update. Each fix addressed a real,
distinct bug. Launch 6 finally passed that boundary — and then diverged instead of crashing.
See `task_description.md` for the full narrative.

| # | File | Symptom | Fix applied |
|---|------|---------|-------------|
| 1 | `crash_01_unboundlocalerror_ref.log` | `UnboundLocalError: ref` in `slmadv()` call | Restored `lambda_slm==0` guard around `slmadv()` |
| 2 | `crash_02_powbackward_nan.log` | `PowBackward0 returned nan` in `d_loss.backward()` | Reverted LR to v4-safe (`lr/ft_lr: 3e-5`, `bert_lr: 1e-6`) |
| 3 | `crash_03_nan_loss_mel.log` | `NaN detected in loss_mel` (script's own guard, clean exit) | Restored `train_LM` guard; added `clip_grad_norm_` on decoder/style_encoder/msd/mpd |
| 4 | `crash_04_convolutionbackward_nan.log` | `ConvolutionBackward0 returned nan` in `d_loss.backward()` | Found `set_detect_anomaly(True)` aborts backward before clipping ever runs — disabled it, added finite-grad skip guards |
| 5 | `crash_05_silent_skip_freeze.log` | No crash: 409 consecutive `[skip]`s, `val_loss: nan` checkpoints, GPUs busy for hours learning nothing | Added `_MAX_CONSECUTIVE_SKIPS = 50` abort. My own design flaw — a skip guard cannot escape a state it stops the optimizer from fixing |
| 6 | `crash_06_divergence_after_clamp.log` | No crash, all counters clean, but `acoustic_norm` 10 → 671 and val_loss 0.848 → 2.07 after GAN activates | **Root cause found**: unbounded `torch.exp()` in `istftnet.py:523/541` → 1e22 amplitude → fp32 overflow → NaN. Clamped at `max=15.0`. Stops the crash; does NOT stop the divergence |

## Root cause (launches 1-5)

`Modules/istftnet.py:523` and `:541`:

```python
spec = torch.exp(torch.clamp(x[:, : self.post_n_fft // 2 + 1, :], max=15.0))
```

The first GAN gradient at `joint_epoch` shifts `conv_post` by tens in log space; unbounded
`exp()` turns that into ~1e22 amplitude; the mel power spectrogram squares it to 1e44, over
fp32's 3.4e38 → `inf`; `SpectralConvergengeLoss` then computes `inf - inf` → NaN. Measured, not
inferred — see crash_06 for the full chain and the hypotheses ruled out by measurement.

## Full run logs

* `run06_divergence_full.log` — complete log of launch 6 (289 lines), preserved because the
  launch command `rm -f`s the remote log and the sync overwrites the local copy.
* `stage2_v5.log` — live sync target, overwritten by each new run. Not a durable record.

## Status

Launch 6's best checkpoint is **epoch 2, val_loss 0.848** — from before the GAN phase, saved
locally as `results/checkpoints/epoch_2nd_00002.pth` and uploaded back to the VM as
`/mnt/kikiri-tts/StyleTTS2/epoch2_base.pth`. v3's benchmark is 0.506, so this is not yet
competitive. Config for all 6 launches: `code/config_david_v5_stage2.yml` (edited in place
between launches; current file reflects launch 6).
