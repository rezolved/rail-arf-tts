# t0006 Results Summary

## Objective

Test 250-clip data scale + `multispeaker: false` in combination to isolate Stage 2 divergence cause
from t0005.

## Key Results

| Run | Config | lambda_gen | joint_epoch | Best val | Audio |
|-----|--------|-----------|-------------|----------|-------|
| run01 | v6 | 1.0 | 3 | ~0.89 | diverged ep7 |
| run02 | v6b | 0.2 | 3 | 0.895 (ep4) | diverged ep7 |
| run03 | v6c | 0.2 | 3 | 0.849 (ep5) | noisy |
| run04 | v6d | 0.05 | 6 | **0.846 (ep6)** | noisy |

**Best checkpoint**: v6d epoch 6, val_loss = 0.846. Target not reached (v3 gold: 0.506).

## Conclusion

GAN activates stably with `lambda_gen=0.05` but 10 epochs insufficient for convergence. v6b used
wrong `first_stage.pth` (multispeaker:true mismatch). v6c/v6d used v3's `first_stage.pth`
(multispeaker:false), reducing baseline acoustic_norm from 10 to 0.36.

## Next Steps

1. Check v3 original config — v3 reached val=0.569 at epoch 1, possibly `joint_epoch: 0`.
2. Continue from v6d ep6 with `epochs_2nd: 20` (was accidentally left at 10).
3. v6e: `lambda_gen: 0.01`, `joint_epoch: 8`, `epochs: 25`.
