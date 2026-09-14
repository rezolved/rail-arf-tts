# Checkpoint Audit

## Known Inconsistencies Resolved

### 1. t0004 `epoch_1st_00007.pth` vs 'epoch 10' label

The file is named `epoch_1st_00007.pth` → 0-based epoch 7 = 1-based epoch 8. The README labels this
'epoch 10' — incorrect. The filename is authoritative.

### 2. t0005 best checkpoint `epoch_2nd_00003.pth` vs README 'epoch 2'

The file is 0-based epoch 3 (the 4th epoch). README says 'epoch 2' (0-based epoch 1 by human count
or 1-based). Off by one. Filename is authoritative: it is the 4th stage-2 epoch.

### 3. t0006 `epochs_2nd: 10` bug

The training script reads `config.epochs_2nd` (not `config.epochs`). v6d config has
`epochs: 15, epochs_2nd: 10`. The script stopped at epoch 10 (0-based 9). The extra `epochs_2nd` key
overrode the intended 15-epoch run.

### 4. Top-2 val_loss pruning risk

t0005 and t0006 prune checkpoints to keep only the top-2 by val_loss. For v6c (joint_epoch=6, 10
epochs), all saved checkpoints are pre-GAN. The pre-GAN epochs have low val_loss (easy
reconstruction) and are most likely to be kept. Post-joint_epoch checkpoints (where divergence first
appears) would be pruned first. This means the last healthy checkpoint before divergence is the most
at risk of deletion.

## Checkpoint Map

| run_id | file | epoch (0b) | epoch (1b) | val_loss | pre_joint_epoch | notes |
| --- | --- | --- | --- | --- | --- | --- |
| t0004_run01 | epoch_1st_00007.pth | 7 | 8 | ? | ? | Stage 1 checkpoint. 0-based epoch 7. README incorrectly labels it 'epoch 10' — t... |
| t0005_run06 | epoch_2nd_00003.pth | 3 | 4 | ? | False | t0005 best checkpoint. 0-based epoch 3 (4th epoch). README says 'epoch 2' — off ... |
| t0006_run03_v6c | epoch_2nd_00003.pth | 3 | 4 | 0.884 | True | t0006 v6c best. 0-based epoch 3. val=0.884 from committed log. joint_epoch=6 → a... |
| t0006_run03_v6c | epoch_2nd_00005.pth | 5 | 6 | 0.849 | True | v6c best by val_loss. 0-based epoch 5. val=0.849. Still pre-joint_epoch (5 < 6).... |
| t0006_run04_v6d | epoch_2nd_00003.pth | 3 | 4 | 0.846 | True | v6d best. 0-based epoch 3. val=0.846. Training stopped at epoch 10 (0-based) due... |

## SHA-256 Hashes

VM not accessible. SHA-256 hashes cannot be computed from local files. The `first_stage_v3.pth`
checkpoint used in v6c/v6d is DVC-tracked in t0006.

## Recommendation

Switch to per-epoch checkpoint saves with the `CheckpointManager` (library asset). Never prune the
last checkpoint with `is_pre_joint_epoch=True` — that is the last recovery point before the GAN
activates.
