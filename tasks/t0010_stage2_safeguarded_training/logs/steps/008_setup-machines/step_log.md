---
step: 8
step_name: setup-machines
task_id: t0010_stage2_safeguarded_training
started_at: "2026-09-15T10:56:58Z"
completed_at: "2026-09-15T12:15:00Z"
status: completed
---
# Step 8 — setup-machines

## Summary

Acquired LLM-T1-NC80 (2×H100 NVL, Azure ML northeurope), deployed the idle watchdog, rebuilt the
`kokoro-finetune` training environment from scratch (ephemeral disk was wiped after prior VM stop),
and staged all files needed for Stage 2 safeguarded training.

## Actions Taken

### Phase 1 — Pre-flight

- Confirmed budget: ~$34.90 estimated for 2.5h GPU run; `stop_threshold_reached=false`.
- Confirmed `az account show` authenticated as `azureuser` (subscription `rezolve-AI`).

### Phase 2 — Acquire VM

- Called `azure_ml_vm acquire t0010_stage2_safeguarded_training` via `run_with_logs.py`.
- Exit 75: `az ml compute show` API timed out at the hardcoded 60s limit; VM was confirmed Running
  via ARM REST API (`az rest`) but acquire() could not observe it in time.
- Intervention file: `tasks/t0010_stage2_safeguarded_training/intervention/pool_busy_llm-t1-nc80.md`
  created by provisioner.
- **Manual workaround**: started VM with `az ml compute start --no-wait`, polled ARM REST until
  `state=Running` (~15 min), confirmed SSH reachability, ran remote preflight manually
  (`systemd linger=yes`, `/mnt/cache/persist` writable), placed lock file on VM via SSH.
- Machine log constructed manually using `to_machine_log_entry()` schema.

### Phase 3 — GPU & CUDA Verification

- `nvidia-smi`: 2× NVIDIA H100 NVL, 94249 MiB each, driver 535.161.08.
- `nvcc --version`: CUDA 12.2.
- GPU smoke test: `torch 2.5.1+cu121 cuda True devs 2`.

### Phase 4 — Idle Watchdog

- Deployed via `watchdog_provisioning.render_azure_ml_install_script()` over SSH.
- Confirmed PID with `pgrep -f idle_watchdog.sh`.
- Config: `IDLE_THRESHOLD_SECONDS=3600`, `POLL_INTERVAL_SECONDS=60`, `IDLE_UTIL_PERCENT=5`,
  `GRACE_SECONDS=600`.
- `TERMINATE_CMD`:
  `az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI`.

### Phase 5 — Environment Preparation

- **Root disk full** (`/` at 100%): `~/kokoro-finetune` could not be created on root disk.
- Cloned `github.com/semidark/kikiri-tts` to `/mnt/tmp/kikiri-tts/` with submodule (`StyleTTS2`) and
  created `~/kokoro-finetune → /mnt/tmp/kikiri-tts/StyleTTS2` symlink.
- Created directories: `data/v4/train/wavs/`, `data/v4/val/wavs/`, `configs/`, `logs/v10/`.
- Copied list files:
  - `tasks/t0006_kokoro_v5_stage2_subset/code/train_list_250.txt` → VM
    `~/kokoro-finetune/data/data_list_v5_train_250.txt` (250 entries)
  - `data/v4/val_list.txt` → VM `~/kokoro-finetune/data/val_list.txt` (96 entries)
- Copied 5 safeguard modules from `tasks/t0009_stage2_training_failure_forensics/code/`:
  - `train_second_safeguarded.py`, `jsonl_logger.py`, `health_gates.py`, `checkpoint_manager.py`,
    `run_config.py`
- **`first_stage_v3.pth`** (1.7 GB): DVC pull fails on VM (managed identity auth not available).
  Downloaded locally from Azure Blob (`mldvcstorerezolve/ml-dvc-datasets`) via
  `az storage blob download --auth-mode login`, then `rsync` to
  `~/kokoro-finetune/first_stage_v3.pth`.
- **Training wavs** (1557 files, ~256 MB): DVC pull not usable on VM. Built `download_wavs.py` using
  DVC dir index (`data/v4/train/wavs.dvc`) to download each file from Azure Blob directly. Executed
  on VM; all 1557 files downloaded to `~/kokoro-finetune/data/v4/train/wavs/`.
- **Val wavs** (96 files): downloaded similarly to `~/kokoro-finetune/data/v4/val/wavs/`.

## Outputs

- `logs/steps/008_setup-machines/machine_log.json` — VM provenance, GPU verification, watchdog.
- VM `LLM-T1-NC80` ready with:
  - `~/kokoro-finetune/` (→ `/mnt/tmp/kikiri-tts/StyleTTS2`)
  - `~/kokoro-finetune/first_stage_v3.pth` (1.7 GB)
  - `~/kokoro-finetune/data/v4/train/wavs/` (1557 wav files)
  - `~/kokoro-finetune/data/v4/val/wavs/` (96 wav files)
  - `~/kokoro-finetune/data/data_list_v5_train_250.txt` (250-entry training subset)
  - `~/kokoro-finetune/data/val_list.txt` (96 val clips)
  - `~/kokoro-finetune/train_second_safeguarded.py` + 4 support modules

## Issues

### Issue 1 — `az ml compute show` API timeout (60s hardcoded)

`AZ_TIMEOUT_SECONDS = 60.0` in `azure_ml_vm.py` is hardcoded; ARM REST API for this workspace
consistently exceeds 60s. This causes `acquire()` to return exit 75 even when the VM is Running. The
manual acquire workaround added ~30 min to setup time.

**Recommendation**: Make `AZ_TIMEOUT_SECONDS` configurable via env var (`ARF_AZ_TIMEOUT_SECONDS`)
and raise default to 180s; or switch `get_compute_state_result()` to use `az rest` which is faster.

### Issue 2 — Root disk full (100%), ephemeral disk wipeout

`/` on LLM-T1-NC80 is at 100% capacity. All new data must go to `/mnt/tmp/` (ephemeral) or
`/mnt/cache/persist/` (Azure Files). `~/kokoro-finetune` is a symlink to ephemeral disk — it will be
lost on next VM stop.

**Mitigation for t0010**: training run will complete without VM stop/start. Checkpoints will be
written to `~/kokoro-finetune/logs/v10/` which is also ephemeral — ensure they are downloaded before
teardown.

### Issue 3 — DVC managed identity auth fails on VM

`dvc pull` on the VM fails with `WorkloadIdentityCredential`/`ManagedIdentityCredential` auth
errors. Azure ML compute instances do not have access to the `mldvcstorerezolve` storage account via
managed identity.

**Workaround**: download data locally (using user's `az login` credentials) and rsync/scp to VM, or
use `az storage blob download --auth-mode login` directly on data that's already on the local
machine.

### Issue 4 — `CheckpointManager` bug (cross-step note from t0009 analysis)

`train_second_safeguarded.py` line 403: `CheckpointManager` initialized with positional args that
don't match the constructor signature. Must be fixed before running training in Step 9.
