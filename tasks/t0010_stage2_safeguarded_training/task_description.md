# Kokoro Stage 2: Safeguarded Training with joint_epoch=8

## Motivation

t0009 identified two root causes for every Stage 2 divergence across 19+ training launches:

1. **Silent zero-param load** — the upstream `load_checkpoint` uses `strict=False` and silently
   loads zero parameters when a DataParallel-saved checkpoint (`module.`-prefixed keys) is passed to
   a non-wrapped model. v6a and v6b almost certainly trained from random initialization as a result.

2. **joint_epoch=3 too early** — activating GAN discriminator losses at epoch 3, before the decoder
   has converged, causes gradient explosion. v6c used `joint_epoch=6` and was the only run to reach
   a usable checkpoint (val_loss=0.849 at epoch 6).

t0009 shipped `train_second_safeguarded.py` and four support modules (JSONL logger, health gates,
checkpoint manager, run config capture) to fix both causes and make the next failure diagnosable
from committed logs.

t0008 established the evaluation baseline: the best existing checkpoint (v3_bundle) scores
speaker_sim=0.631 vs the ElevenLabs target of 0.85 (gap=0.20). TTFB is already within spec (185 ms).
Improving speaker similarity through additional fine-tuning is the remaining gap.

This task runs the first controlled Stage 2 experiment with both fixes applied. **One variable
changes from v6c**: `joint_epoch` 6 → 8. All other settings stay at v6c values (250-clip data,
`first_stage_v3.pth`, `multispeaker=true`, `lr=1e-4`, `lambda_gen=1.0`). Each saved checkpoint is
evaluated with the t0008 harness so speaker_sim drives early stopping rather than val_loss alone.

## Key Questions

1. Does the DP-aware loader confirm that Stage 1 weights are actually loaded (parameter-count
   assertion at startup)?
2. Does `joint_epoch=8` produce a stable post-GAN phase, i.e. does divergence not appear within 4
   epochs of GAN activation (vs 3 epochs in v6c)?
3. What is the best speaker_sim achieved, and at which epoch? Does the run reach or approach 0.85?
4. Does the JSONL log survive and does the health gate fire at the right point (or not at all if the
   run stays stable)?

## Scope

### 1. Environment setup

SSH to LLM-T1-NC80, confirm the `kokoro-finetune/` repo is present, install any missing deps, and
copy `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` plus its four
imports (`jsonl_logger.py`, `health_gates.py`, `checkpoint_manager.py`, `run_config.py`) to the VM.
Deploy `arf/scripts/utils/idle_watchdog.sh` with 60 min idle threshold as per `CLAUDE.md`.

### 2. Config

Create `configs/config_david_v10.yml` based on v6c with exactly one change:

| Parameter | v6c | v10 |
| --- | --- | --- |
| `joint_epoch` | 6 | **8** |
| `epochs_2nd` | 10 (bug, defaulted) | **20** |
| `multispeaker` | true | true |
| `first_stage_path` | first_stage_v3.pth | same |
| `lr` | 1e-4 | 1e-4 |
| `lambda_gen` | 1.0 | 1.0 |
| `train_lm` | false | false |
| `train_list` | data_list_v5_train_250.txt | same |
| `val_list` | val_list.txt | same |
| `batch_size` | 8 | 8 |

Also add the JSONL log path and per-epoch checkpoint directory to the config. Save the resolved
config YAML and git SHA to `data/run_v10/launch_info.json` via `capture_run_config`.

### 3. Training run

Launch `train_second_safeguarded.py -p configs/config_david_v10.yml` wrapped in `run_with_logs.py`
so stdout/stderr are captured. The script will:

- Assert parameter-count at startup: raise `RuntimeError` if fewer than 80% of expected parameters
  match (catches silent DP mismatch before any training).
- Write one JSONL record per step to `logs/v10/metrics.jsonl` (all losses, grad norms, skip count,
  LR, epoch).
- Save a checkpoint after every epoch to `checkpoints/v10/epoch_NNN.pth` with SHA-256 manifest.
  Never prune the last pre-`joint_epoch` checkpoint. Apply top-N pruning only to post-GAN epochs.
- Check health gates after each epoch: dur_loss (first step < 2.0), acoustic_norm (< 20), val_spike
  (≤ 0.05 per epoch post-joint), consecutive skips (≤ 50). On gate fire: write the triggering epoch
  and the last healthy checkpoint path, then stop.

Monitor progress on the VM via `tail -f logs/v10/metrics.jsonl`. If training diverges (health gate
fires), collect the JSONL log, commit it, and note the epoch + gate trigger in results.

### 4. Checkpoint evaluation

After each epoch checkpoint is saved, run the t0008 `tts_eval_harness` on it:

- Speaker_sim (GE2E cosine vs ElevenLabs David reference centroid) on the 100-filler prompt set.
- Duration ratio and WER as sanity metrics.
- Report speaker_sim trajectory per epoch (curve, not just the final value).

Stop early if speaker_sim starts declining after a peak (early stopping by harness, not val_loss).

The evaluation can be batched at the end (evaluate all saved checkpoints after training completes)
rather than after each epoch, if live evaluation adds too much latency.

### 5. Results

Commit the JSONL log, per-epoch checkpoint manifest, and harness results to git. The JSONL log is a
`.jsonl` file — not `.log` — so it is NOT excluded by `.gitignore`.

Report:

- speaker_sim curve per epoch (chart + table)
- Best epoch by speaker_sim and corresponding TTFB/WER
- Health gate events (if any), with epoch and trigger signal
- Whether the parameter-count assertion passed at startup
- Comparison to t0008 baselines: v3_bundle (0.631), ElevenLabs target (0.85)
- If gate fired: recommended diagnostic steps for the next run

## Compute and Budget

- LLM-T1-NC80 (2×H100 NVL): 20 epochs × ~6 min/epoch ≈ 2 h training + ~30 min eval ≈ ~$35.
- Setup and teardown through `setup-remote-machine` skill.
- Write `results/costs.json` with `total_cost_usd`.

Planned total: ≤ $45.

## Expected Outputs

- **Model asset**: best checkpoint by speaker_sim, packaged with `extract_decoder_generic.py` from
  t0002.
- `results/metrics.json` with speaker_sim, ttfb_ms, rtf variants per evaluated epoch.
- `results_detailed.md` tables: per-epoch speaker_sim, val_loss, health gate events, config delta
  from v6c.
- `results/images/speaker_sim_curve.png` — speaker_sim vs epoch, dashed line at 0.85.
- `results/images/loss_timeline.png` — val_loss, dur_loss, acoustic_norm vs epoch from JSONL.
- `data/run_v10/metrics.jsonl` — full step-level JSONL log (committed to git).
- `data/run_v10/checkpoint_manifest.json` — SHA-256 hashes of all saved checkpoints.
- `data/run_v10/launch_info.json` — resolved config + git SHA + hostname.

## Dependencies

- `t0009_stage2_training_failure_forensics` — provides `train_second_safeguarded.py` and the four
  safeguard modules.
- `t0008_tts_eval_harness_baselines` — provides `tts_eval_harness` and the ElevenLabs reference
  centroid for speaker_sim scoring.
