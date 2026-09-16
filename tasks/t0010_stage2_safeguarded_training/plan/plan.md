---
spec_version: "2"
task_id: "t0010_stage2_safeguarded_training"
date_completed: "2026-09-15"
status: "complete"
---
# t0010 — Kokoro Stage 2: Safeguarded Training with joint_epoch=8

## Objective

Run the first controlled Kokoro-82M StyleTTS2 Stage 2 fine-tuning experiment that implements both
root-cause fixes identified in t0009: (1) DP-aware checkpoint loader with parameter-count assertion
at startup, and (2) `joint_epoch=8` (up from 6 in v6c). All other hyperparameters are identical to
the only successful prior run (v6c). The training uses the t0009 safeguard library (JSONL step
logger, health gates, per-epoch checkpoint manager, run-config capture). After training, every saved
epoch checkpoint is evaluated with the t0008 `tts_eval_harness` (speaker_sim, TTFB, RTF). The best
checkpoint by speaker_sim is packaged as a model asset.

**Success criteria:**

* Training completes 20 epochs without health-gate-triggered divergence (or divergence is correctly
  detected and reported with the triggering epoch and gate signal).
* Parameter-count assertion passes at startup (Stage 1 weights confirmed loaded).
* Per-epoch speaker_sim curve produced for all saved checkpoints.
* Best checkpoint achieves speaker_sim > 0.631 (v3_bundle baseline from t0008); approaching 0.85
  (ElevenLabs target) is the stretch goal.
* Model asset registered for the best epoch checkpoint.

* * *

## Task Requirement Checklist

**Operative task text** (from `task.json` `short_description` + `task_description.md`):

> First controlled Stage 2 run implementing the two t0009 root-cause fixes: DP-aware checkpoint
> loader and joint_epoch=8. Uses the t0009 safeguard library (JSONL logger, health gates, per-epoch
> checkpoints). Evaluates each epoch checkpoint with the t0008 harness. One variable changed from
> v6c.

Full requirements extracted from `task_description.md`:

* **REQ-1** SSH to LLM-T1-NC80, confirm `kokoro-finetune/` repo present, install missing deps, copy
  training script + four support modules to VM, deploy idle watchdog with 60 min idle threshold.
  **Satisfied by**: Step 1. **Evidence**: `data/run_v10/launch_info.json` hostname field + watchdog
  PID in step log.

* **REQ-2** Create `configs/config_david_v10.yml` on the VM based on v6c with exactly two changes:
  `joint_epoch: 6 → 8` and `epochs_2nd: 10 → 20` (also set `epochs: 20`). Add JSONL log path and
  per-epoch checkpoint dir. Save resolved config + git SHA to `data/run_v10/launch_info.json` via
  `capture_run_config`. **Satisfied by**: Step 2. **Evidence**: `data/run_v10/launch_info.json`.

* **REQ-3** Fix the `CheckpointManager` bug: line 403 of the copied script must pass
  `joint_epoch=joint_epoch` to `CheckpointManager(...)`. **Satisfied by**: Step 3 (fix applied when
  copying script). **Evidence**: `code/train_second_v10.py` line 403 shows
  `joint_epoch=joint_epoch`.

* **REQ-4** Launch `train_second_v10.py` wrapped in `run_with_logs.py`. The script must assert
  parameter-count at startup (≥ 80% Stage 1 param match), write per-step JSONL log, save per-epoch
  checkpoints with SHA-256 manifest, and check health gates after each epoch. **Satisfied by**: Step
  4\. **Evidence**: `data/run_v10/metrics.jsonl` (training log),
  `data/run_v10/checkpoint_manifest.json`.

* **REQ-5** Monitor health gates: dur_loss gate (first step < 2.0 from epoch 2+), acoustic_norm (<
  20), val_spike (≤ 0.05 post joint_epoch=8), consecutive_skip (≤ 50). On gate fire: collect log,
  commit it, note epoch + trigger in results. **Satisfied by**: Steps 4 + 9. **Evidence**: health
  gate status in `data/run_v10/metrics.jsonl` + results report.

* **REQ-6** After training, evaluate all saved epoch checkpoints with the t0008 `tts_eval_harness`:
  speaker_sim (GE2E cosine vs ElevenLabs David centroid) on the 100-filler prompt set, TTFB, RTF.
  Five-module extraction (`extract_decoder.extract()`) must be called per checkpoint before eval.
  **Satisfied by**: Steps 5–7. **Evidence**: `data/run_v10/eval_results/epoch_NNN_eval.json` per
  epoch.

* **REQ-7** Report per-epoch speaker_sim trajectory (curve + table). Apply early stopping if
  speaker_sim declines after peak. **Satisfied by**: Steps 7–8. **Evidence**:
  `results/images/speaker_sim_curve.png`, per-epoch table in results.

* **REQ-8** Commit JSONL log, per-epoch checkpoint manifest, and harness results to git. The JSONL
  file must not be excluded by `.gitignore`. **Satisfied by**: Step 9 (commit). **Evidence**: git
  history includes `data/run_v10/metrics.jsonl`.

* **REQ-9** Report: speaker_sim curve per epoch; best epoch by speaker_sim + TTFB/WER; health gate
  events; parameter-count assertion outcome; comparison to t0008 baselines (v3_bundle=0.631,
  ElevenLabs=0.85 target). **Satisfied by**: Step 9. **Evidence**: `results/metrics.json`,
  `results/images/`, per-epoch table.

* **REQ-10** Package model asset: best checkpoint by speaker_sim in
  `tasks/t0010_stage2_safeguarded_training/assets/model/kokoro-v10-best/`. Run
  `verify_model_asset.py`. **Satisfied by**: Step 10. **Evidence**: verificator passes with 0
  errors.

* **REQ-11** Write `data/run_v10/checkpoint_manifest.json` with SHA-256 hashes for all saved
  checkpoints. **Satisfied by**: Step 4 (CheckpointManager writes it automatically). **Evidence**:
  file present and non-empty after training.

* **REQ-12** Write `results/images/loss_timeline.png` — val_loss, dur_loss, acoustic_norm vs epoch
  from JSONL log. **Satisfied by**: Step 9. **Evidence**: PNG present in `results/images/`.

* * *

## Approach

### Technical approach

The approach has three phases: (A) environment setup and script preparation on the VM, (B) training
run, (C) batch checkpoint evaluation.

**Phase A — Script preparation.** Research (task t0009) identified that
`train_second_safeguarded.py` must be copied (not imported) because it is not a registered library.
Before copying, apply a one-line fix to the `CheckpointManager` call at line 403:

```python
# Bug (original):
_ckpt_mgr = CheckpointManager(log_dir=log_dir, run_id=_run_id)

# Fix (required — CheckpointManager.__init__ signature requires joint_epoch):
_ckpt_mgr = CheckpointManager(log_dir=log_dir, run_id=_run_id, joint_epoch=joint_epoch)
```

`joint_epoch` is already in local scope from the config — this is a one-word addition. Without this
fix, training crashes with `TypeError` before epoch 1.

Also confirmed from research: `diff_epoch` is present in the v6c config reconstruction but not
explicitly set — the training script reads `loss_params.diff_epoch` from the config, so v10 must
include this field. Set it to the same effective value (6 in v6c; we keep it at 6 for v10 since we
are only changing `joint_epoch`).

**Phase B — Config v10.** The v10 config is v6c with these changes:

| Parameter | v6c value | v10 value | Reason |
| --- | --- | --- | --- |
| `joint_epoch` | 6 | **8** | Main experimental variable |
| `epochs` | 10 | **20** | Extend training (Kokoro reads `epochs`, not `epochs_2nd`) |
| `epochs_2nd` | 10 | **20** | Also set for completeness |

All other v6c parameters unchanged: `lambda_gen=1.0`, `lr=1e-4`, `multispeaker=true`,
`first_stage_path=first_stage_v3.pth`, `train_LM=false`, `lambda_slm=0.0`, `batch_size=8`.

**Health gate thresholds** (from t0009 `HealthGate` implementation, calibrated on v6c):
`DUR_LOSS_STEP1_MAX=2.0`, `ACOUSTIC_NORM_MAX=20.0`, `VAL_SPIKE_MAX=0.05`, `CONSECUTIVE_SKIP_MAX=50`.
The val_spike gate fires only from epoch > `joint_epoch` (= epoch 9+ for v10), so the first 8 epochs
are not subject to that gate.

**Phase C — Batch evaluation.** After training completes (or health gate fires), evaluate all saved
checkpoints in one batch. Each checkpoint requires 5-module extraction before eval: modules `bert`,
`bert_encoder`, `predictor`, `text_encoder`, `decoder` must be extracted into a packaged `.pth`
using `extract_decoder.extract()` from the `tts_eval_harness` library. Running only `decoder`
extraction causes duration explosions (~10× audio length, null speaker_sim). The resemblyzer venv
(separate from main pyproject to avoid webrtcvad conflict) must be pre-built on the VM.

**Library imports (not copied — registered libraries):**
* `from tasks.t0009_stage2_training_failure_forensics.code.jsonl_logger import StepLogger`
* `from tasks.t0009_stage2_training_failure_forensics.code.checkpoint_manager import CheckpointManager`
* `from tasks.t0009_stage2_training_failure_forensics.code.health_gates import HealthGate, GateResult`
* `from tasks.t0009_stage2_training_failure_forensics.code.run_config import capture_run_config`
* `from tasks.t0009_stage2_training_failure_forensics.code.constants import DUR_LOSS_STEP1_MAX, ...`
* `from tasks.t0008_tts_eval_harness_baselines.code.extract_decoder import extract`
* `from tasks.t0008_tts_eval_harness_baselines.code.run_eval import main as run_eval`
* `from tasks.t0008_tts_eval_harness_baselines.code.score_speaker_sim import ...` (invoked as
  subprocess)

**Code to copy (not a library — copy into task):**
* `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` →
  `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` (apply CheckpointManager fix)

**Task type**: `tts-finetuning-eval` — matches this task exactly. The type instruction emphasizes:
smoke-gate synthesis check before full eval, 50-warmup-clip discard, GE2E speaker_sim vs ElevenLabs
David reference centroid, and no training on val_96.

### Alternatives considered

**Alternative: Live per-epoch evaluation (eval after each epoch during training).** Rejected because
resemblyzer venv setup takes ~10 minutes; initializing it 20 times interleaved with training adds
~200 min of idle risk. The research summary recommends batching: train to completion (or gate fire),
then evaluate all checkpoints in one pass. This is the chosen approach.

**Alternative: Reuse t0009's training script as-is from the library path.** Rejected because
`train_second_safeguarded.py` is not registered as a library (it's 1027 lines of training code, not
a reusable utility). Cross-task imports are only allowed via the library mechanism. The script must
be copied and the CheckpointManager bug fixed in the copy.

* * *

## Cost Estimation

| Item | Estimate | Basis |
| --- | --- | --- |
| LLM-T1-NC80 H100 training (20 epochs × ~6 min/epoch) | ~$35 | ~120 min × $17.50/hr |
| LLM-T1-NC80 H100 batch eval (20 checkpoints × ~1.5 min/ckpt) | ~$5 | ~30 min × $17.50/hr |
| VM setup/teardown overhead | ~$2 | ~7 min × $17.50/hr |
| **Total estimated** | **~$42** | — |
| Project budget remaining | $5,000 − prior spend | See `ctx/costs.json` |
| Per-task default limit | $100 | `project/budget.json` |

The $42 estimate is within both the task's documented cap of $45 and the $100 per-task default
limit. No API call costs — training and eval are fully local on the VM.

* * *

## Step by Step

### Milestone 1: Environment and Script Setup

1. [CRITICAL] **Copy and fix the training script.** On the local workstation (in the worktree): copy
   `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` to
   `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py`. Apply the one-line
   `CheckpointManager` fix at line 403: change `CheckpointManager(log_dir=log_dir, run_id=_run_id)`
   to `CheckpointManager(log_dir=log_dir, run_id=_run_id, joint_epoch=joint_epoch)`. Run
   `uv run ruff check --fix tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py && uv run ruff format tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py && uv run mypy -p tasks.t0010_stage2_safeguarded_training.code`
   and confirm 0 errors. Expected output: ruff clean, mypy 0 errors. Satisfies **REQ-3**.

2. **Create the eval orchestration script.** Write
   `tasks/t0010_stage2_safeguarded_training/code/eval_all_checkpoints.py`. This script accepts
   `--checkpoint-dir`, `--output-dir`, `--config`, `--filler-list`, `--ref-dir` args; iterates over
   `epoch_2nd_NNNNN.pth` files in epoch order; for each checkpoint calls `extract()` from
   `tasks.t0008_tts_eval_harness_baselines.code.extract_decoder` to produce a packaged `.pth`, then
   calls `run_eval.py` CLI (via subprocess wrapped in `run_with_logs`) to produce per-clip JSON,
   then calls `score_speaker_sim.py` (via subprocess in resemblyzer venv) to add speaker_sim;
   aggregates per-epoch results into `--output-dir/epoch_NNN_eval.json`; returns early if
   speaker_sim has peaked and declined for 3 consecutive epochs. Inputs: checkpoint `.pth` files.
   Outputs: `data/run_v10/eval_results/epoch_NNN_eval.json` per epoch. Run `uv run ruff check --fix`
   and `ruff format` after writing. Satisfies **REQ-6**, **REQ-7**.

3. **Create the results aggregation and chart script.** Write
   `tasks/t0010_stage2_safeguarded_training/code/aggregate_results.py`. This script reads
   `data/run_v10/metrics.jsonl` (per-step training log from `StepLogger`) and the per-epoch
   `eval_results/epoch_NNN_eval.json` files, then produces: (a) per-epoch table CSV at
   `data/run_v10/per_epoch_summary.csv` (columns: epoch, val_loss, dur_loss_step1, acoustic_norm,
   speaker_sim_mean, ttfb_p50_ms, rtf_mean); (b) `results/images/speaker_sim_curve.png` with dashed
   reference lines at 0.631 (v3_bundle) and 0.85 (ElevenLabs target); (c)
   `results/images/loss_timeline.png` — val_loss, dur_loss, acoustic_norm vs epoch. Writes
   `results/metrics.json` in **explicit variant format** (one variant per epoch + `best` variant for
   peak speaker_sim epoch) with keys `speaker_sim`, `ttfb_ms`, `rtf`. Run `uv run ruff check --fix`
   and `ruff format` after writing. Satisfies **REQ-9**, **REQ-12**.

**Validation gate for Step 3 chart output**: before generating the full chart, confirm that
`data/run_v10/per_epoch_summary.csv` has at least 1 row and the `speaker_sim_mean` column contains
non-NaN values. Baseline: v3_bundle speaker_sim = 0.631. If all speaker_sim values are 0.0 or NaN,
STOP and investigate — do not produce charts from null data.

### Milestone 2: VM Setup

4. **Start LLM-T1-NC80 and verify GPU.** Via `setup-remote-machine` skill: start the VM, SSH in,
   confirm `nvidia-smi` shows 2 × H100 NVL, confirm `kokoro-finetune/` repo is present at
   `~/kokoro-finetune/`. If not present: clone from the known repo URL (see `CLAUDE.md` section on
   GPU machine). Expected: `nvidia-smi` shows 2 GPUs, no error. Satisfies **REQ-1** (partial).

5. **Deploy idle watchdog.** Copy `arf/scripts/utils/idle_watchdog.sh` to VM, then start it:

```bash
export TERMINATE_CMD="az ml compute stop --name LLM-T1-NC80 \
  --workspace-name brainpowa-northeurope --resource-group rezolve-AI"
export IDLE_THRESHOLD_SECONDS=3600
export POLL_INTERVAL_SECONDS=60
export IDLE_UTIL_PERCENT=5
export GRACE_SECONDS=600
export WATCHDOG_LOG=/var/log/arf_idle_watchdog.log
nohup bash ~/idle_watchdog.sh >> "$WATCHDOG_LOG" 2>&1 &
```

Verify: `ps aux | grep idle_watchdog` shows the process running. Log PID in step log. Satisfies
**REQ-1** (watchdog).

6. **Copy training code and create v10 config on VM.** SCP the fixed training script
   `code/train_second_v10.py` to the VM at `~/kokoro-finetune/`. Create
   `~/kokoro-finetune/configs/config_david_v10.yml` based on v6c (content below). Run
   `capture_run_config` to write `data/run_v10/launch_info.json`:

```yaml
# config_david_v10.yml — v6c + joint_epoch=8, epochs=20
log_dir: logs/v10
loss_params:
  joint_epoch: 8       # CHANGED from 6 (main experimental variable)
  diff_epoch: 6        # same as v6c
  lambda_gen: 1.0
  lambda_slm: 0.0
  lambda_dur: 1.0
  lambda_ce: 20.0
  lambda_F0: 1.0
  lambda_norm: 1.0
  lambda_mel: 5.0
optimizer_params:
  lr: 0.0001
  ft_lr: 0.0001
  bert_lr: 0.000001
epochs: 20             # CHANGED from 10 (Kokoro reads `epochs`, not `epochs_2nd`)
epochs_2nd: 20         # also updated for completeness
batch_size: 8
model_params:
  multispeaker: true
first_stage_path: first_stage_v3.pth
load_only_params: true
second_stage_load_pretrained: false
train_LM: false
data_params:
  train_data: data/data_list_v5_train_250.txt
  val_data: data/val_list.txt
  root_path: ""
  min_length: 50
  OOD_data: ""
  num_workers: 2
```

Commit `data/run_v10/launch_info.json` to the task branch. Satisfies **REQ-2**.

### Milestone 3: Training Run

7. [CRITICAL] **Launch training.** On VM inside `~/kokoro-finetune/`, run:

```bash
uv run python -m arf.scripts.utils.run_with_logs \
  --task-id t0010_stage2_safeguarded_training -- \
  python train_second_v10.py -p configs/config_david_v10.yml --run-id v10
```

**Validation gate (pre-training):** Before launching the full 20-epoch run, confirm the
parameter-count assertion fires successfully by checking the first 50 lines of stdout — look for a
log line like `"Stage 1 params matched: NNN/NNN (≥ 80%)"`. If the assertion raises `RuntimeError`
(0-param match — DP key mismatch), STOP: this means `first_stage_v3.pth` has the `module.`-prefix
issue. In that case: strip the prefix manually and retry. Do NOT proceed past the startup assertion
failure. Baseline for comparison: any prior failed run (v6a, v6b) loaded 0 parameters silently and
trained for epochs before diverging.

Monitor via `tail -f ~/kokoro-finetune/logs/v10/metrics.jsonl` in a separate SSH session.

Expected output after 20 epochs: `logs/v10/metrics.jsonl` with ~20 × steps_per_epoch records,
`logs/v10/epoch_2nd_NNNNN.pth` checkpoints (up to 20), `logs/v10/checkpoint_manifest.json`.

If health gate fires: collect the JSONL log and manifest immediately, download to task folder at
`data/run_v10/`, commit them, and record the triggering epoch + gate signal. Do NOT re-launch
without an intervention file.

Satisfies **REQ-4**, **REQ-5**.

### Milestone 4: Checkpoint Evaluation

8. **Build resemblyzer venv on VM.** In `~/kokoro-finetune/`:

```bash
python -m venv ~/resemblyzer-venv
source ~/resemblyzer-venv/bin/activate
pip install resemblyzer webrtcvad
deactivate
```

Verify: `~/resemblyzer-venv/bin/python -c "import resemblyzer; print('ok')"` prints `ok`. If already
present from t0008, skip creation. Satisfies **REQ-6** (venv prerequisite).

9. [CRITICAL] **Run batch checkpoint evaluation.** Download all saved epoch checkpoints from VM to
   `data/run_v10/checkpoints/epoch_NNN.pth` (or leave on VM and eval in place). Run:

```bash
uv run python -m arf.scripts.utils.run_with_logs \
  --task-id t0010_stage2_safeguarded_training -- \
  python tasks/t0010_stage2_safeguarded_training/code/eval_all_checkpoints.py \
    --checkpoint-dir logs/v10 \
    --output-dir data/run_v10/eval_results \
    --config configs/config_david_v10.yml \
    --filler-list data/fillers_100.txt \
    --ref-dir data/11labs_david
```

**Validation gate (post-eval):** After the first checkpoint is evaluated, check that speaker_sim >
0.40 (well below even the v3_bundle baseline of 0.631 — any reasonable checkpoint should be above
random). If speaker_sim is 0.0 or NaN for the first checkpoint, STOP: 5-module extraction may have
failed. Inspect the packaged `.pth` for module keys before proceeding.

After all checkpoints are evaluated, download `data/run_v10/eval_results/` and
`data/run_v10/metrics.jsonl` and `data/run_v10/checkpoint_manifest.json` to the local worktree.
Satisfies **REQ-6**, **REQ-7**.

### Milestone 5: Results, Asset, and Metrics

10. **Aggregate results and produce charts.** On the local workstation, run:

```bash
uv run python -m arf.scripts.utils.run_with_logs \
  --task-id t0010_stage2_safeguarded_training -- \
  python tasks/t0010_stage2_safeguarded_training/code/aggregate_results.py \
    --jsonl data/run_v10/metrics.jsonl \
    --eval-dir data/run_v10/eval_results \
    --output-dir results
```

Expected outputs: `results/metrics.json` (explicit variant format, one variant per epoch + `best`),
`results/images/speaker_sim_curve.png`, `results/images/loss_timeline.png`,
`data/run_v10/per_epoch_summary.csv`. Satisfies **REQ-9**, **REQ-12**.

**Metrics to write** in `results/metrics.json` explicit variant format per epoch:
* `speaker_sim` — mean GE2E cosine on 100-filler set (registered metric key)
* `ttfb_ms` — p50 TTFB on 100-filler set (registered metric key)
* `rtf` — mean RTF on 100-filler set (registered metric key)

11. **Package the model asset.** Identify the best epoch by `speaker_sim_mean` from
    `data/run_v10/per_epoch_summary.csv`. Create the model asset folder:

```
tasks/t0010_stage2_safeguarded_training/assets/model/kokoro-v10-best/
├── details.json       (spec_version: "2", all required fields)
├── description.md     (all 7 mandatory sections)
└── files/
    ├── kokoro-v10-best.pth    (packaged checkpoint — 5-module extracted)
    └── config_david_v10.yml   (training config)
```

Key `details.json` fields:
* `model_id`: `"kokoro-v10-best"`
* `framework`: `"pytorch"`
* `base_model`: `"kokoro-82m"`
* `architecture`: `"StyleTTS2 Stage 2 fine-tune of Kokoro-82M with joint_epoch=8, DP-safe loader"`
* `training_task_id`: `"t0010_stage2_safeguarded_training"`
* `hyperparameters`: include `joint_epoch: 8`, `epochs: 20`, `lr: 1e-4`, `lambda_gen: 1.0`,
  `batch_size: 8`, `multispeaker: true`
* `training_metrics`: include `best_speaker_sim`, `best_epoch`, `best_val_loss`, `ttfb_p50_ms`
* The checkpoint file must be DVC-tracked: run `dvc add` on the `.pth` file and commit the `.dvc`
  pointer.

Run
`uv run python -m arf.scripts.verificators.verify_model_asset --task-id t0010_stage2_safeguarded_training`
and fix all errors before proceeding. Satisfies **REQ-10**.

12. **Commit all data files and verify.** Run `dvc push` for the checkpoint `.pth` file and any
    other large binary. Confirm `data/run_v10/metrics.jsonl` is committed to git (not DVC — it is a
    text JSONL file, not binary). Confirm `data/run_v10/checkpoint_manifest.json` is committed. Run
    `git status` to verify no untracked required files remain. Satisfies **REQ-8**, **REQ-11**.

* * *

## Remote Machines

LLM-T1-NC80 (Azure ML, `Standard_NC80adis_H100_v5`, 2 × H100 NVL, 80 GB VRAM each). Required for
training and inference. Provisioned via `setup-remote-machine` skill. The VM pool config is at
`project/azure_vm.json`. SSH alias: `LLM-T1-NC80`.

**Idle watchdog mandatory** (`arf/scripts/utils/idle_watchdog.sh`): 60-min idle threshold, 5% GPU
util = idle. `TERMINATE_CMD`:
`az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI`.
See `CLAUDE.md` for full deployment instructions.

**Estimated runtime**: ~2 h training + ~0.5 h eval = ~2.5 h total on VM.

* * *

## Assets Needed

* **`t0009_training_safeguards` library** — provides `StepLogger`, `CheckpointManager`,
  `HealthGate`, `capture_run_config`, `constants`. Import path:
  `tasks.t0009_stage2_training_failure_forensics.code.*`. Registered library at
  `tasks/t0009_stage2_training_failure_forensics/assets/library/t0009_training_safeguards/`.

* **`tts_eval_harness` library** — provides `extract_decoder.extract()`, `run_eval.py` CLI,
  `score_speaker_sim.py` CLI, `adapters.py` (`load_kokoro_model_with_checkpoint`, `SynthResult`,
  `save_wav`), `constants.py` (`CHECKPOINT_MODULES`, `SUCCESS_SPEAKER_SIM=0.85`). Import path:
  `tasks.t0008_tts_eval_harness_baselines.code.*`. Registered library at
  `tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/`.

* **`train_second_safeguarded.py`** (source for copy) —
  `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py`. Not a library;
  must be copied and fixed.

* **v6c config** (reference only) —
  `tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml`.

* **`first_stage_v3.pth`** — Stage 1 checkpoint on the VM at `~/kokoro-finetune/first_stage_v3.pth`.
  Required for Stage 2 initialization with `load_only_params: true`.

* **Training data** — `data/data_list_v5_train_250.txt` (250-clip subset, on VM in
  `~/kokoro-finetune/data/`). Val data: `data/val_list.txt` (96 clips, = val_96).

* **ElevenLabs David reference audio** — `data/11labs_david/` (1358 WAVs, used as GE2E speaker_sim
  reference centroid). On VM from t0008 eval run.

* **100-filler prompt list** — `data/fillers_100.txt` (or the `data/v4/fillers/` prompt set used by
  t0008 harness). Confirm exact path from t0008 harness constants.

* * *

## Expected Assets

* **Model asset** (`model`, 1): `kokoro-v10-best` — best Kokoro Stage 2 v10 checkpoint by
  speaker_sim. Packaged with 5-module extraction, stored in
  `tasks/t0010_stage2_safeguarded_training/assets/model/kokoro-v10-best/files/kokoro-v10-best.pth`
  (DVC-tracked). Matches `task.json` `expected_assets: {"model": 1}`.

* * *

## Time Estimation

| Phase | Wall-clock estimate |
| --- | --- |
| Step 1-3: Write code locally (script copy, eval script, aggregator) | ~45 min |
| Step 4-6: VM setup, watchdog, config deploy | ~20 min |
| Step 7: Training run (20 epochs) | ~120 min |
| Step 8: Resemblyzer venv build | ~10 min |
| Step 9: Batch checkpoint evaluation (20 ckpts) | ~30 min |
| Step 10: Aggregate results + charts | ~10 min |
| Step 11-12: Model asset, DVC push, verification | ~20 min |
| **Total** | **~255 min (~4.3 h)** |

* * *

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `CheckpointManager` bug not fixed causes `TypeError` crash at epoch 1 | High (if forgotten) | Blocking | Step 1 explicitly fixes line 403 and runs mypy to catch the type error. |
| `first_stage_v3.pth` has `module.`-prefix keys (DP-saved) — parameter count assertion fails | Medium | Blocking | Startup assertion catches this immediately. Strip prefix with `{k.removeprefix('module.'): v for k, v in sd.items()}` and retry. Create intervention file if unavailable. |
| `joint_epoch=8` still causes GAN-phase divergence (gradient explosion at epoch 9) | Medium | Blocking | Health gate fires, log is committed, intervention file created. Next step: try `joint_epoch=10` in t0011. |
| `diff_epoch` not in v10 config causes `KeyError` at training start | Low | Blocking | Step 6 explicitly adds `diff_epoch: 6` to config YAML. |
| resemblyzer venv not present on VM — `score_speaker_sim.py` fails silently | Low | High | Step 8 builds venv before eval; validation gate checks first eval result. |
| 5-module extraction incomplete — only decoder extracted — duration explosions | Medium | High | Eval script imports `CHECKPOINT_MODULES` from `tts_eval_harness.constants`; validation gate checks speaker_sim > 0.40. |
| `data/fillers_100.txt` path differs from t0008 — eval script fails to find prompts | Low | Medium | Check t0008 harness constants file for exact filler list path before eval. |
| Training diverges before epoch 8 (pre-GAN instability) | Low | High | Health gate dur_loss and acoustic_norm fire immediately. Collect log, commit, write intervention file. |
| VM cost overrun — run takes >3 h due to eval overhead | Low | Medium | Idle watchdog terminates VM at 60 min idle. Monitor TTFB/RTF per epoch to confirm reasonable pace. |
| `epochs_2nd` key bug — Kokoro reads `epochs` not `epochs_2nd` | Known | Blocking if not fixed | Config v10 sets both `epochs: 20` and `epochs_2nd: 20`. |

* * *

## Verification Criteria

* **Plan verificator passes**:
  `uv run python -m arf.scripts.verificators.verify_plan t0010_stage2_safeguarded_training` exits 0
  with 0 errors.

* **CheckpointManager fix present**:
  `grep "joint_epoch=joint_epoch" tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py`
  returns a match at line 403.

* **Training completed or gate fired**: `data/run_v10/metrics.jsonl` exists, is non-empty, and
  contains at least 1 epoch record. Confirmed via:
  `python -c "import json; lines=open('tasks/t0010_stage2_safeguarded_training/data/run_v10/metrics.jsonl').readlines(); assert len(lines) > 0"`.

* **Checkpoint manifest present**: `data/run_v10/checkpoint_manifest.json` exists and contains at
  least 1 entry. Confirmed via:
  `python -c "import json; m=json.load(open('tasks/t0010_stage2_safeguarded_training/data/run_v10/checkpoint_manifest.json')); assert len(m) > 0"`.

* **Eval results present**: at least 1 file matching `data/run_v10/eval_results/epoch_*_eval.json`
  exists. Confirmed via:
  `ls tasks/t0010_stage2_safeguarded_training/data/run_v10/eval_results/epoch_*_eval.json`.

* **Metrics JSON valid and uses explicit variant format**:
  `uv run python -m arf.scripts.verificators.verify_task_metrics t0010_stage2_safeguarded_training`
  exits 0. Check that `results/metrics.json` contains `"variants"` key:
  `python -c "import json; d=json.load(open('tasks/t0010_stage2_safeguarded_training/results/metrics.json')); assert 'variants' in d"`.

* **Charts produced**:
  `ls tasks/t0010_stage2_safeguarded_training/results/images/speaker_sim_curve.png tasks/t0010_stage2_safeguarded_training/results/images/loss_timeline.png`
  exits 0 for both files.

* **Model asset verificator passes**:
  `uv run python -m arf.scripts.verificators.verify_model_asset --task-id t0010_stage2_safeguarded_training`
  exits 0 with 0 errors. Satisfies REQ-10.

* **All three registered metrics present in best variant**:
  `python -c "import json; d=json.load(open('tasks/t0010_stage2_safeguarded_training/results/metrics.json')); best=[v for v in d['variants'] if v['variant_id']=='best'][0]; assert all(k in best['metrics'] for k in ['speaker_sim','ttfb_ms','rtf'])"`
  exits 0.

* **REQ coverage**: all REQ-1 through REQ-12 items have corresponding outputs. Verify via
  `results/results_detailed.md` `## Task Requirement Coverage` section (written in the results
  step).

* * *

## Rejection Criteria

These conditions declare the task's speaker_sim measurements null and require an intervention file
before the task can be considered complete:

* **Successful synthesis rate < 80%**: if fewer than 80% of filler prompts complete synthesis
  without duration explosion (clip > 30 s or duration_ratio > 5×), the speaker_sim values are
  unreliable. In this case: report the failure rate, commit the raw results, write
  `intervention/eval_null_low_success_rate.md`, and do not report speaker_sim as a valid
  measurement.

* **All speaker_sim values NaN or 0.0**: if `score_speaker_sim.py` failed silently (resemblyzer venv
  issue or 5-module extraction failure), all cosine scores will be NaN. Do not report NaN as a
  metric — investigate and fix the venv or extraction step first.

* **Parameter-count assertion failed and training proceeded anyway**: if the startup assertion was
  bypassed or never ran (log does not contain the assertion pass/fail line), the training weights
  may be random. Declare training results null and create
  `intervention/param_count_assertion_skipped.md`.

* **Health gate fired before epoch 1 completes**: if `dur_loss_step1 ≥ 2.0` triggers at the very
  first epoch, the model diverged immediately. Do not report any speaker_sim values as meaningful —
  write `intervention/training_immediate_divergence.md` with the gate trigger value.
