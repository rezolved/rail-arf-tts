# Research Summary — t0010_stage2_safeguarded_training

## Key Findings (top 10 insights directly actionable for this task)

1. **Critical bug in `train_second_safeguarded.py` line 403**: `CheckpointManager` is called without
   the required `joint_epoch` argument — will crash with `TypeError` before epoch 1. Fix: add
   `joint_epoch=joint_epoch` to the call. `joint_epoch` is already in scope.

2. **Two libraries are registered and ready**: `t0009_training_safeguards` (StepLogger,
   CheckpointManager, HealthGate, capture_run_config) and `tts_eval_harness` (extract_decoder,
   run_eval, score_speaker_sim). Both import via the `tasks.tXXXX.code.*` import path.

3. **`train_second_safeguarded.py` must be copied, not imported**: it is not a registered library
   (1027 lines). Copy to `tasks/t0010_stage2_safeguarded_training/code/` and apply the one-line
   `CheckpointManager` fix.

4. **Config v10 = v6c with two changes**: `joint_epoch: 6 → 8` and `epochs_2nd: 10 → 20`. All other
   parameters identical: `multispeaker: true`, `first_stage_v3.pth`, `lr: 1e-4`, `lambda_gen: 1.0`,
   `train_LM: false`, `batch_size: 8`. Base config archived at
   `tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml`.

5. **Five-module extraction is mandatory before evaluation**: loading only `decoder` from a raw
   StyleTTS2 `.pth` causes duration explosions (~10× longer audio). `extract_decoder.extract()` must
   be called per checkpoint. Modules: `bert`, `bert_encoder`, `predictor`, `text_encoder`,
   `decoder`.

6. **Health gate val_spike will not fire until epoch 9+** (since `epoch > joint_epoch = 8`). The
   dur_loss gate fires when `dur_loss ≥ 2.0` at step 1 of any epoch ≥ 2 (1-based). Thresholds:
   `DUR_LOSS_STEP1_MAX=2.0`, `ACOUSTIC_NORM_MAX=20.0`, `VAL_SPIKE_MAX=0.05`,
   `CONSECUTIVE_SKIP_MAX=50`.

7. **resemblyzer runs in a separate venv** (`score_speaker_sim.py` is a subprocess, not an import).
   The webrtcvad dep conflicts with the main pyproject. Build the venv on the VM before the
   evaluation phase.

8. **Baseline gap**: best Kokoro (`v3_bundle`) scores `speaker_sim=0.631` (fillers) vs ElevenLabs
   target of `0.85`. TTFB is already within spec (`185 ms vs 300 ms limit`). The task must report
   per-epoch speaker_sim even if 0.85 is not reached.

9. **Batch evaluation after training completes** is preferred over per-epoch live evaluation. The
   resemblyzer venv setup takes ~10 minutes; running it serially during a 2-hour training adds
   unnecessary risk of interrupting training.

10. **Deploy idle watchdog before launching training**: `arf/scripts/utils/idle_watchdog.sh` with
    60-min idle threshold. Prevents runaway H100 billing if orchestrator disconnects.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Copy-and-fix training script

Copy `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` to
`tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py`, apply the one-line
`CheckpointManager(joint_epoch=joint_epoch)` fix, and point it at `config_david_v10.yml`. The script
already integrates all four safeguard components and the DP-aware loader; no other changes are
needed for the training phase.

### Approach 2: Batch checkpoint evaluation pipeline

After training completes, iterate over all saved epoch checkpoints: (1) call
`extract_decoder.extract()` on each raw `.pth`, (2) run `run_eval.py` with the packaged checkpoint
to get synthesis + TTFB/RTF metrics, (3) run `score_speaker_sim.py` in the resemblyzer venv to add
`speaker_sim`. Aggregate per-clip JSONs into a per-epoch curve. This batches the expensive venv
startup once rather than 20 times.

### Approach 3: Per-epoch results table driven by JSONL log

Parse `data/run_v10/metrics.jsonl` (written by `StepLogger`) for val_loss and dur_loss per epoch,
merge with the per-epoch speaker_sim scores from the harness. Produce the results table and
`speaker_sim_curve.png` / `loss_timeline.png` charts from the merged data. The JSONL log is the
authoritative source for training-side metrics; the harness JSON is the source for eval-side
metrics.

## Reusable Code / Assets

* `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` — full training
  script; copy into task, fix CheckpointManager call **(copy into task)**
* `tasks/t0009_stage2_training_failure_forensics/code/jsonl_logger.py` — `StepLogger` class
  **(import via library `t0009_training_safeguards`)**
* `tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py` — `CheckpointManager`
  class **(import via library)**
* `tasks/t0009_stage2_training_failure_forensics/code/health_gates.py` — `HealthGate`, `GateResult`
  **(import via library)**
* `tasks/t0009_stage2_training_failure_forensics/code/run_config.py` — `capture_run_config`
  **(import via library)**
* `tasks/t0009_stage2_training_failure_forensics/code/constants.py` — gate thresholds, column names
  **(import via library)**
* `tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml` — base config for
  v10 **(reference only)**
* `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py` — `extract()` function **(import
  via library `tts_eval_harness`)**
* `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py` — GE2E scoring CLI **(import
  via library, invoked as subprocess)**
* `tasks/t0008_tts_eval_harness_baselines/code/run_eval.py` — synthesis + timing CLI **(import via
  library)**
* `tasks/t0008_tts_eval_harness_baselines/code/adapters.py` — `load_kokoro_model_with_checkpoint`,
  `SynthResult`, `save_wav` **(import via library)**
* `tasks/t0008_tts_eval_harness_baselines/code/constants.py` — `CHECKPOINT_MODULES`,
  `SUCCESS_SPEAKER_SIM=0.85` **(import via library)**

## Key Papers (top 5, with finding most relevant to this task)

(not generated — step skipped)

## Risks Flagged in Research

* **`CheckpointManager` missing `joint_epoch`** — crashes at startup; must fix before training.
* **`epochs_2nd` bug**: Kokoro Stage 2 reads `epochs` not `epochs_2nd`; without a matching
  `epochs: 20` key in config, training may default to 200 epochs and run indefinitely.
* **Gate thresholds calibrated on single log (v6c)**: may fire too early or too late for v10 if
  training dynamics differ materially (different batch size, data, or LR not applicable here since
  all match v6c).
* **`diff_epoch` not specified in task description**: confirm its value in config (v6c had 6); if
  left unset, training will not activate diffusion losses at the expected epoch.
* **5-module extraction needed per epoch**: forgetting this step before harness eval causes duration
  explosion and null speaker_sim results.
* **resemblyzer venv must be pre-built on VM**: if not present, `score_speaker_sim.py` crashes
  silently.

## Full Detail Available In

* `tasks/t0010_stage2_safeguarded_training/research/research_papers.md` — (not generated — step
  skipped)
* `tasks/t0010_stage2_safeguarded_training/research/research_internet.md` — (not generated — step
  skipped)
* `tasks/t0010_stage2_safeguarded_training/research/research_code.md` — 9 tasks reviewed, 5 cited, 2
  libraries found (both relevant)
