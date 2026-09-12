# t0005 — Stage 2 Results

**Machine**: LLM-T1-NC80 (2× H100 NVL via `torch.nn.DataParallel`, both GPUs used)
**Data**: 1557 clips (v5 phoneme-corrected corpus)
**Stage 1 checkpoint**: t0004 epoch 10 (val=0.740)
**Config**: `code/config_david_v5_stage2.yml`

## Summary

6 runs attempted. All converged through GAN activation at joint_epoch, but all diverged within
4 post-GAN epochs. Root cause: 1557-clip dataset is too large for the discriminator at joint_epoch
transition — GAN gradient overwhelms the generator. Fixed in t0006 (250-clip subset).

## Run history

| Run | Key change | Best val | GAN result |
|-----|-----------|----------|-----------|
| run01 | baseline | crashed pre-GAN | OOM / WavLM crash |
| run02 | `lambda_slm: 0.0` guard | crashed pre-GAN | guard fixed crash |
| run03 | `train_LM: false` | 0.918 (ep1) | diverged ep2 |
| run04 | `clip_grad_norm_` | 0.918 (ep1) | diverged ep2 |
| run05 | `istftnet.py` clamp fix | 0.905 (ep1) | diverged ep5 |
| **run06** | all fixes combined | **0.848 (ep3)** | diverged ep4→2.070 |

Best checkpoint: **run06 epoch 3, val=0.848** (`results/checkpoints/epoch_2nd_00003.pth`).

All 7 fixes from t0005 run06 were carried forward into t0006.

## Key fixes accumulated in this task

1. `lambda_slm == 0` guard around `slmadv()` (WavLM crash)
2. `train_LM: false` guard (BERT update crash at GAN phase)
3. `clip_grad_norm_` on msd/mpd/decoder/style_encoder
4. `set_detect_anomaly(False)`
5. Finite-grad skip guard + 50-consecutive-skip abort
6. `torch.exp(torch.clamp(..., max=15.0))` in `istftnet.py:523/541` (root-cause NaN fix)
7. `lambda_slm: 0.0` in config

## Files Produced

- `results/checkpoints/epoch_2nd_00002.pth.dvc` — epoch 2 checkpoint
- `results/checkpoints/epoch_2nd_00003.pth.dvc` — epoch 3 checkpoint (best)
- `logs/stage2_v5.log` — full run06 training log
- `logs/crash_04_convolutionbackward_nan.log` — run05 crash log (NaN root cause)
- `code/train_second_patch.diff` — all 7 patches applied to train_second.py
