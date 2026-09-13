# Results Summary: Kokoro v5 Stage 2 Fine-tune

## Summary

6 Stage 2 training runs on LLM-T1-NC80 (DataParallel, both GPUs). All converged through GAN
activation but all diverged within 4 post-GAN epochs. Best checkpoint: **run06 epoch 3,
val_loss=0.848**. Root cause: 1557-clip dataset is too large for the discriminator at joint_epoch
transition — GAN gradient overwhelms the generator. 7 crash fixes accumulated across runs. All fixes
carried forward to t0006 (250-clip subset).

## Key Metrics

| Metric | Value |
| --- | --- |
| Best val loss | 0.848 |
| Best run | run06 |
| Best epoch | 3 |
| Runs attempted | 6 |
| Machine | LLM-T1-NC80 (2× H100 NVL, DataParallel) |
| Training clips | 1557 (v5 phoneme-corrected) |

## Run History

| Run | Key change | Best val | GAN result |
| --- | --- | --- | --- |
| run01 | baseline | crashed pre-GAN | OOM / WavLM crash |
| run02 | `lambda_slm: 0.0` guard | crashed pre-GAN | guard fixed crash |
| run03 | `train_LM: false` | 0.918 (ep1) | diverged ep2 |
| run04 | `clip_grad_norm_` | 0.918 (ep1) | diverged ep2 |
| run05 | `istftnet.py` clamp fix | 0.905 (ep1) | diverged ep5 |
| **run06** | all fixes combined | **0.848 (ep3)** | diverged ep4→2.070 |
