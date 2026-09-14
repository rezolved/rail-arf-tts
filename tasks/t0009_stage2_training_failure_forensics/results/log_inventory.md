# Log Inventory

All known Kokoro Stage 2 training runs across t0001, t0004, t0005, t0006, and the v3 reference.

VM retrieval: SSH to LLM-T1-NC80 timed out — `/mnt/kikiri-tts/` not accessible. Logs for t0005 runs
1-5 and t0006 runs 1, 2, 4 are permanently deleted by the launch script.

| run_id | task | log | config | stage1_ckpt | data_list | outcome |
| --- | --- | --- | --- | --- | --- | --- |
| t0001_run01 | t0001_kokoro_v4_stage2_finetune | no | no | epoch_1st_00007.pth (via t0001) | v4 train list (~800 clips) | diverged |
| t0004_run01 | t0004_kokoro_v5_stage1_train | no | no | epoch_1st_00007.pth (v4 Stage 1, 0-based epoch 7) | v4 train list | incomplete |
| t0005_run01 | t0005_kokoro_v5_stage2_train | no | no | unknown | v5 train list | unknown |
| t0005_run02 | t0005_kokoro_v5_stage2_train | no | no | unknown | v5 train list | unknown |
| t0005_run03 | t0005_kokoro_v5_stage2_train | no | no | unknown | v5 train list | unknown |
| t0005_run04 | t0005_kokoro_v5_stage2_train | no | no | unknown | v5 train list | unknown |
| t0005_run05 | t0005_kokoro_v5_stage2_train | no | no | unknown | v5 train list | unknown |
| t0005_run06 | t0005_kokoro_v5_stage2_train | no | no | unknown | v5 train list (600+ clips, lambda_gen=1.0) | diverged |
| t0006_run01_v6a | t0006_kokoro_v5_stage2_subset | no | no | epoch_1st_00007.pth (multispeaker=false) | 250-clip subset, lambda_gen=1.0 | diverged |
| t0006_run02_v6b | t0006_kokoro_v5_stage2_subset | no | no | epoch_1st_00007.pth (multispeaker=false; mismatch) | 250-clip subset, lambda_gen=0.2 | diverged |
| t0006_run03_v6c | t0006_kokoro_v5_stage2_subset | yes | no | first_stage_v3.pth (multispeaker=true) | 250-clip subset, lambda_gen=1.0, joint_epoch=6 | success |
| t0006_run04_v6d | t0006_kokoro_v5_stage2_subset | no | no | first_stage_v3.pth (multispeaker=true) | 250-clip subset, lambda_gen=0.05, joint_epoch=6 | incomplete |
| v3 | v3 (reference; not in this repo) | no | no | first_stage_v3.pth (assumed) | v3 266-clip list (DVC-tracked in t0006) | success |

## SHA-256 Hashes

No checkpoint files are locally available; SHA-256 hashes cannot be computed. Stage 1 checkpoint
used in t0006_run03_v6c (`first_stage_v3.pth`) is DVC-tracked in t0006.

## Summary

Total runs: 13. Logs present: 1. Successful: 2.
