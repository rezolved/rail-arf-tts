# Confound Table

Per-run effective settings for all known Kokoro Stage 2 runs. SHA-256 hashes not available (no local
checkpoint files; VM not accessible).

| run_id | multispeaker | first_stage_path | joint_epoch | lambda_gen | lr | outcome | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t0001_run01 | true | epoch_1st_00007.pth | 3 | 1.0 | 1e-4 | diverged | accelerate launch used instead of python train_second.py — DataParallel race. va... |
| t0005_run06 | true | epoch_1st_00007.pth (v4 Stage1) | 3 | 1.0 | 3e-5 | diverged | All 7 crash patches active. val_loss spike documented. Logs deleted. |
| t0006_run01_v6a | false (config mismatch with checkpoint) | epoch_1st_00007.pth (multispeaker=false) | 3 (assumed) | 1.0 | 1e-4 | diverged | multispeaker mismatch. Diverged. Log deleted. |
| t0006_run02_v6b | false (config mismatch with checkpoint) | epoch_1st_00007.pth (multispeaker=false) | 3 (assumed) | 0.2 | 1e-4 | diverged | multispeaker mismatch; likely 0-param load (trained from scratch). Diverged. Log... |
| t0006_run03_v6c | true | first_stage_v3.pth | 6 | 1.0 | 1e-4 | success | v3 Stage1 checkpoint (multispeaker=true). joint_epoch=6. Stable: val=0.849. LOG ... |
| t0006_run04_v6d | true | first_stage_v3.pth | 6 | 0.05 | 1e-4 | incomplete | lambda_gen=0.05 (KEY CHANGE). Only 10 epochs ran (intended 15) — epochs_2nd bug.... |
| v3 | true (assumed) | first_stage_v3.pth (assumed) | unknown (speculated: 0) | 1.0 (assumed) | unknown | success | Only train_second_patch.diff committed. No config. val=0.506. Reference run. |

## Key Observations

1. **Checkpoint alignment**: runs v6a/v6b used `multispeaker=false` checkpoint with
   `multispeaker=true` config → 0-param load → silent training from scratch.
2. **joint_epoch escalation**: t0001/t0005 used `joint_epoch=3`; v6c/v6d used `joint_epoch=6` — the
   delay is the main stabilizing factor besides checkpoint fix.
3. **lambda_gen**: v6c (1.0) succeeded; v6d (0.05) also succeeded but only 10 epochs ran. The key
   isolator is checkpoint alignment + joint_epoch=6, not lambda_gen.
4. **v3 patches**: Only `lambda_slm > 0` guard committed — no istftnet clamp, no skip guard. Yet
   val=0.506. The 7-patch stack in t0005 is all crash mitigation, not cause fix.
5. **LR**: t0001/t0006 used lr=1e-4; t0005 safe run used lr=3e-5. v6c succeeded at lr=1e-4, so LR
   alone did not cause failure.
