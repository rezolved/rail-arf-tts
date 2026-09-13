# t0006 Results Detailed

See `task_description.md` for full hypothesis, methodology, per-run configs, and recommendations.

## Run Details

### run01 (v6): lambda_gen=1.0, joint_epoch=3

Baseline test. GAN activated at epoch 3, val_loss diverged epoch 7 (~0.89 → spike). acoustic_norm
climbed 10 → 27. Same divergence pattern as t0005 despite smaller dataset.

### run02 (v6b): lambda_gen=0.2, joint_epoch=3

Used wrong `first_stage.pth` (multispeaker:true, from t0004). acoustic_norm 10 → 61. Diverged
epoch 7, val=0.895 at epoch 4. Result invalidated by checkpoint mismatch.

### run03 (v6c): lambda_gen=0.2, joint_epoch=3, v3 first_stage.pth

Switched to v3's `first_stage.pth` (multispeaker:false). acoustic_norm dropped from 10 to 0.37 at
epoch 1 — confirms checkpoint mismatch was causing high baseline norm. Best val=0.849 at epoch 5.
Audio noisy throughout.

### run04 (v6d): lambda_gen=0.05, joint_epoch=6, v3 first_stage.pth

Weakest GAN weight tested. GAN activated at epoch 6, acoustic_norm=8.51 at joint_epoch. Best
val=0.846 at epoch 6. Training stopped at epoch 10 due to `epochs_2nd: 10` (intended: 15 — config
bug). Audio noisy but stable — no divergence.

## Assets

- `results/checkpoints/` — top-2 checkpoints by val_loss (DVC-tracked)
- `results/audio_v6c_ep5/` — v6c epoch 5 synthesis samples (DVC-tracked)
- `results/audio_v6d_ep6/` — v6d epoch 6 synthesis samples (DVC-tracked)
- `data/reference/` — v3 gold standard assets (Stage 1 checkpoint, voicepack, decoder, reference audio)
