# Results Summary: t0010 Kokoro Stage 2 Safeguarded Training (v10)

## Summary

Kokoro-82M Stage 2 fine-tuning completed **17 of 20 planned epochs** on LLM-T1-NC80 (2×H100) using
the t0009 safeguard library with `joint_epoch=8` and the DP-aware checkpoint loader fix. Training
stopped at epoch 17 because the ephemeral `/mnt` disk filled up — not due to a health gate event.
Best validation loss was **0.797 at JSONL epoch 16** (no even-epoch checkpoint for that point; the
primary checkpoint `epoch_2nd_00016.pth` covers epoch 17 with val_loss **0.853**). Speaker_sim,
TTFB, and RTF evaluation is deferred: the VM root disk was also 100% full at teardown, blocking the
torch environment required for harness inference.

## Metrics

- **Best val_loss**: **0.797** at JSONL epoch 16 (no checkpoint saved for that exact epoch)
- **Primary checkpoint val_loss**: **0.853** (epoch 17, `epoch_2nd_00016.pth`)
- **GAN activation (joint_epoch=8)**: val_loss dropped from **2.067** (epoch 8) to **1.079** (epoch
  9\) — a clean **49% reduction** confirming successful GAN phase activation
- **Epochs 1-8 (pre-GAN)**: val_loss stable between **2.063** and **2.077** (expected plateau)
- **Epochs 9-17 (post-GAN)**: val_loss descended from 1.079 to 0.797 then rebounded to 0.853 at
  epoch 17
- **Health gate events**: **0** — training was stable throughout all 17 epochs
- **Parameter-count assertion**: PASSED at startup (DP-aware loader fix confirmed working)
- **Actual cost**: **$272.78** vs $45 planned (VM idle overnight — watchdog did not stop it)
- **speaker_sim, ttfb_ms, rtf**: **null** (harness eval deferred; see
  `intervention/eval_deferred_disk_full.md`)

## Verification

- `verify_task_metrics.py` — PASSED (explicit variant format, 18 variants, all registered metrics
  present)
- `verify_task_results.py` — PASSED
- `verify_model_asset.py` — PASSED (0 errors, model asset `kokoro-v10-best` registered)
- `verify_machines_destroyed.py` — PASSED (LLM-T1-NC80 stopped, 0 errors)
