# Plan: Kokoro-82M v4 Stage 2 Fine-tune

## Objective

Train StyleTTS2 Stage 2 on LLM-T1-NC80 (2× H100 NVL) using DDP, producing a Kokoro-82M checkpoint
fine-tuned on the ElevenLabs David voice dataset. Deliver the best checkpoint as a DVC-tracked
artifact for evaluation in t0004.

## Approach

1. Set up the training environment on the VM from scratch (VM is wiped — ephemeral disk).
2. Upload training data and Stage 1 checkpoint via rsync.
3. Configure accelerate for 2-GPU DDP.
4. Run Stage 2 training with OOM guards and watchdog.
5. Track best checkpoint in DVC and push.

## Cost Estimation

- LLM-T1-NC80: $13.96/h × ~10 h (20 epochs × ~30 min/epoch on 2× H100) ≈ **$140**.
- Stage 2 epochs are faster with 2 GPUs (vs. ~50 min/epoch on 1 GPU) — estimated 20 epochs in 10 h.

## Step by Step

### Step 1 — VM environment setup

SSH into LLM-T1-NC80 and run `code/setup_vm.sh`. This installs system deps, clones kikiri-tts,
creates a Python venv, and installs Python deps including accelerate.

```bash
ssh LLM-T1-NC80
bash /mnt/setup_vm.sh
```

Expected: `Setup complete. GPUs: 2` at end of script.

### Step 2 — Upload training data

From local machine, rsync data and Stage 1 checkpoint to VM:

```bash
bash code/upload_data.sh
```

Uploads:
- `data/v4/train/` (1557 clips) → `/mnt/kikiri-tts/data/v4/train/`
- `data/v4/val/` (96 clips) → `/mnt/kikiri-tts/data/v4/val/`
- `best_v4/first_stage.pth` → `/mnt/kikiri-tts/StyleTTS2/logs/kokoro-david-v4/first_stage.pth`

### Step 3 — Upload patched trainer and config

```bash
bash code/upload_code.sh
```

Uploads:
- `code/train_second_patched.py` → `/mnt/kikiri-tts/StyleTTS2/train_second.py`
- `code/config_david_v4.yml` → `/mnt/kikiri-tts/configs/config_david_v4.yml`
- `code/accelerate_config.yaml` → `/mnt/kikiri-tts/accelerate_config.yaml`

### Step 4 — Deploy idle watchdog

Prevents billing runaway if training crashes or orchestrator misses teardown:

```bash
ssh LLM-T1-NC80 "bash /mnt/kikiri-tts/deploy_watchdog.sh"
```

Watchdog: polls nvidia-smi every 60 s; if GPU util ≤ 5% for 60 min → `az ml compute stop`.

### Step 5 — Launch Stage 2 training

```bash
ssh LLM-T1-NC80 "bash /mnt/kikiri-tts/run_stage2.sh"
```

Runs `accelerate launch --num_processes 2 train_second.py` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Output logged to `/mnt/kikiri-tts/logs/stage2_v4.log`.

Monitor:
```bash
ssh LLM-T1-NC80 "tail -f /mnt/kikiri-tts/logs/stage2_v4.log"
```

### Step 6 — Track best checkpoint

When training completes, identify best epoch by val loss:

```bash
ssh LLM-T1-NC80 "grep 'Validation loss' /mnt/kikiri-tts/logs/stage2_v4.log | sort -t: -k2 -n | head -3"
```

Download best checkpoint:

```bash
bash code/download_best.sh <epoch_number>
```

Add to DVC:

```bash
dvc add tasks/t0001_kokoro_v4_stage2_finetune/results/best_checkpoint.pth
dvc push
```

### Step 7 — Teardown VM

```bash
az ml compute stop --name LLM-T1-NC80 \
  --workspace-name brainpowa-northeurope --resource-group rezolve-AI \
  --subscription caa7bcdb-c3e1-4687-a73b-7b621b7e4b23
```

## Remote Machines

| Machine | Role | Cost |
| --- | --- | --- |
| LLM-T1-NC80 | 2× H100 NVL training | $13.96/h |

SSH alias: `LLM-T1-NC80` (configured in `~/.ssh/config`; public IP refreshes on each start).

## Assets Needed

- `data/v4/train/train_list.txt` + audio (1557 clips) — from `rail-benchmarks/kokoro-finetune/data/v4/`.
- `data/v4/val/val_list.txt` + audio (96 clips) — same source.
- `best_v4/first_stage.pth` (Stage 1 checkpoint) — from `rail-benchmarks/kokoro-finetune/best_v4/`.

## Expected Assets

- `results/best_checkpoint.pth` — best Stage 2 checkpoint (DVC-tracked).
- `results/training_log.txt` — full training log with per-epoch val losses.
- `results/metrics.json` — best val loss, epoch number, training duration.
- `results/results_summary.md` — training curve summary, best epoch, ready for t0004.

## Time Estimation

| Step | Time |
| --- | --- |
| VM setup + data upload | 30 min |
| Stage 2 training (20 epochs, 2× H100) | ~10 h |
| Checkpoint download + DVC push | 20 min |
| Total | ~11 h |

## Risks and Fallbacks

| Risk | Mitigation |
| --- | --- |
| OOM at GAN phase (epoch 4+) | `expandable_segments:True` + `batch_size: 2`; if persists: `max_split_size_mb: 512` |
| NaN collapse | LR already reduced to 3e-5; `train_LM: false` |
| WavLM crash | `lambda_slm: 0.0` + patched train_second guards slmadv call |
| VM disconnects mid-training | nohup + log to file; watchdog prevents idle billing |
| IP changes after restart | `azure_ml_vm.py` refreshes SSH HostName automatically |

## Verification Criteria

- Training log contains ≥ 20 `Validation loss:` lines (one per epoch).
- Best val loss < 1.539 (prior best from corrupted run).
- `results/best_checkpoint.pth` exists and is DVC-tracked.
- `results/metrics.json` contains `best_val_loss`, `best_epoch`, `training_duration_seconds`.
