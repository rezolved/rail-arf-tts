---
spec_version: "1"
task_id: "t0010_stage2_safeguarded_training"
research_stage: "code"
tasks_reviewed: 9
tasks_cited: 5
libraries_found: 2
libraries_relevant: 2
date_completed: "2026-09-15"
status: "complete"
---
# Research Code — t0010 Stage 2 Safeguarded Training

## Task Objective

Run the first controlled Kokoro-82M Stage 2 fine-tuning experiment that implements both t0009
root-cause fixes: the DP-aware checkpoint loader and `joint_epoch=8`. Uses the t0009 safeguard
library (JSONL logger, health gates, per-epoch checkpoints, run config capture) to prevent the class
of silent failures that made 19+ prior training runs undiagnosable. Evaluates each saved epoch
checkpoint with the t0008 `tts_eval_harness` library (speaker_sim, TTFB, RTF) to track progress
toward the GE2E cosine target of ≥ 0.85 vs ElevenLabs David. One variable changes from the only
successful prior run (v6c): `joint_epoch` 6 → 8, with `epochs_2nd` increased from 10 to 20.

The task is implementation-focused: create `configs/config_david_v10.yml`, run
`train_second_safeguarded.py`, and feed every saved checkpoint through the harness to produce a
per-epoch speaker_sim curve.

## Library Landscape

Two libraries are registered in the project, both directly relevant.

### `t0009_training_safeguards` (version 0.1.0)

Created by `t0009_stage2_training_failure_forensics`. Provides four components:

* **`StepLogger`** (`code/jsonl_logger.py`) — appends one JSONL record per training step or
  validation pass to `<log_dir>/metrics.jsonl`. Constructor:
  `StepLogger(log_dir: Path, run_id: str)`.
* **`CheckpointManager`** (`code/checkpoint_manager.py`) — saves every epoch checkpoint with SHA-256
  manifest. Constructor: `CheckpointManager(log_dir: Path, run_id: str, joint_epoch: int)`. Returns
  saved path from `save()`.
* **`HealthGate`** (`code/health_gates.py`) — monitors four training signals and calls `sys.exit(3)`
  when a threshold is exceeded. Constructor:
  `HealthGate(joint_epoch: int, last_healthy_ckpt: str | None, logger: StepLogger | None)`.
* **`capture_run_config`** (`code/run_config.py`) — records config YAML + git SHA at startup.

Import path for all components (this is a registered library; import directly):

```python
from tasks.t0009_stage2_training_failure_forensics.code.jsonl_logger import StepLogger
from tasks.t0009_stage2_training_failure_forensics.code.checkpoint_manager import CheckpointManager
from tasks.t0009_stage2_training_failure_forensics.code.health_gates import HealthGate
from tasks.t0009_stage2_training_failure_forensics.code.run_config import capture_run_config
```

Health gate thresholds (from `code/constants.py`): `DUR_LOSS_STEP1_MAX = 2.0`,
`ACOUSTIC_NORM_MAX = 20.0`, `VAL_SPIKE_MAX = 0.05`, `CONSECUTIVE_SKIP_MAX = 50`.

### `tts_eval_harness` (version 0.1.0)

Created by `t0008_tts_eval_harness_baselines`. Provides synthesis adapters, GE2E speaker similarity
scoring, WER, and a CLI evaluation pipeline. Key entry points:

* **`extract`** (`code/extract_decoder.py`) — converts a raw StyleTTS2 `.pth` to the five-module
  packaged format required by KModel.
* **`load_kokoro_model_with_checkpoint`** (`code/adapters.py`) — loads KModel with optional
  five-module checkpoint.
* **`score_speaker_sim.py`** — CLI runner for GE2E scoring in a separate resemblyzer venv.
* **`run_eval.py`** — full synthesis + timing CLI.

Import path for library components:

```python
from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
    load_kokoro_model_with_checkpoint,
    save_wav,
)
from tasks.t0008_tts_eval_harness_baselines.code.extract_decoder import extract
from tasks.t0008_tts_eval_harness_baselines.code.constants import CHECKPOINT_MODULES, SUCCESS_SPEAKER_SIM
```

The resemblyzer dependency is isolated to a separate venv; `score_speaker_sim.py` is always invoked
as a subprocess rather than imported directly.

## Key Findings

### The Training Script Is the Primary Reusable Artifact

`tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py` (1027 lines) is the
complete, tested training script for t0010 [t0009]. It integrates all four safeguard components and
implements the DP-aware `load_checkpoint` fix. The script is nearly identical to what t0010 needs;
the only required changes are:

1. Point to the new `config_david_v10.yml` via the `-p` flag.
2. Pass `joint_epoch=8` to `CheckpointManager` (see critical bug below).

The script should be copied into `tasks/t0010_stage2_safeguarded_training/code/` since it is not a
registered library, and any task-specific modifications must live in the task's own `code/`
directory per the cross-task import rule.

### Critical Bug: `CheckpointManager` Missing Required `joint_epoch` Argument

`CheckpointManager.__init__` has signature `(log_dir: Path, run_id: str, joint_epoch: int)` — the
`joint_epoch` parameter is required with no default value [t0009]. However,
`train_second_safeguarded.py` line 403 calls it as
`CheckpointManager(log_dir=log_dir, run_id=_run_id)`, omitting `joint_epoch`. This will crash with
`TypeError` at training startup before any epoch runs.

The fix must be applied when copying the script into t0010's `code/` directory:

```python
# Before (crashes):
_ckpt_mgr = CheckpointManager(log_dir=log_dir, run_id=_run_id)

# After (correct — joint_epoch from config, same value as used for HealthGate):
_ckpt_mgr = CheckpointManager(log_dir=log_dir, run_id=_run_id, joint_epoch=joint_epoch)
```

The `joint_epoch` variable is already in scope in `main()` (set from `loss_params.joint_epoch`), so
this is a one-line fix. The library's `description.md` documents `joint_epoch` as optional, but the
source code requires it — trust the code.

### DP-Aware Checkpoint Loader Is Proven and Ready

The `load_checkpoint` function in `train_second_safeguarded.py` (lines 86-117) strips `module.`
prefixes from all keys and raises `RuntimeError` if zero parameters match [t0009]. This directly
addresses Root Cause 1 from the forensics analysis: all prior Stage 2 failures after t0001 used the
upstream `strict=False` loader, which silently loaded zero parameters from DataParallel checkpoints,
causing runs to train from random initialization.

The fix is embedded in the script itself; no additional import is needed.

### Health Gate Thresholds Are Calibrated on v6c

All gate thresholds in `code/constants.py` are calibrated from the only surviving training log
(t0006 v6c run) [t0009]. The val_spike gate (`VAL_SPIKE_MAX = 0.05`) is specifically tuned to fire
at epoch 8 of v6c (when val_loss jumped +0.114 from 0.883 to 0.997) while not firing at epochs 1-7.
The offline replay test (`test_replay.py`) confirms this behavior.

For t0010 with `joint_epoch=8` (vs 6 in v6c), the val_spike gate will not activate until epoch 9+
(since it fires only when `epoch > joint_epoch` in the 1-based epoch counting). This is the correct
behavior: gate only activates after GAN losses turn on.

### StepLogger Field Names Are Defined in `constants.py`

Column names for the JSONL records are defined as string constants in `code/constants.py` [t0009]:
`COL_DUR_LOSS`, `COL_VAL_LOSS`, `COL_ACOUSTIC_NORM`, etc. The training script uses these field names
when calling `_step_logger.log({...})`. When reading the JSONL file for analysis, parse using these
same column names.

### Evaluation Harness Requires Two-Step Workflow

Evaluating each epoch checkpoint requires two sequential steps [t0008]:

1. **Extraction**: `extract_decoder.py` converts the raw StyleTTS2 checkpoint (with `module.`
   prefixes and parametrization keys) to the five-module packaged format. The five modules are
   `bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`. Missing any of these causes
   duration explosions (~10× longer audio).

2. **Synthesis + scoring**: `run_eval.py` synthesizes audio (saves WAV files), then
   `score_speaker_sim.py` scores them in the resemblyzer venv. The venv must be pre-built on the VM.
   The two-step split exists because resemblyzer depends on `webrtcvad`, which conflicts with the
   main pyproject.toml deps.

The recommended evaluation approach (from task_description.md) is batch evaluation after training
completes rather than per-epoch live evaluation, to avoid serial latency during the training run.

### t0008 Baseline Numbers Are the Comparison Targets

From `t0008_tts_eval_harness_baselines` results [t0008]:

* **ElevenLabs David** (target): `speaker_sim = 0.832` (fillers), `TTFB_p50 = 132 ms`
* **Best Kokoro so far** (`kokoro_v3_bundle`): `speaker_sim = 0.631` (fillers), `TTFB_p50 = 185 ms`
* **t0006 v6d epoch-6** checkpoint: `speaker_sim = 0.601` (fillers)
* **Project target**: `speaker_sim ≥ 0.85`, `TTFB ≤ 300 ms`

The gap is 0.631 → 0.85 = 0.22 GE2E cosine units. t0010 must report whether `joint_epoch=8` narrows
this gap, and at which epoch the peak occurs.

### Config Delta from v6c: Two Changes

The v6c config (archived at
`tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml`) is the only
successful run's config [t0009]. t0010 makes exactly two changes:

| Parameter | v6c | v10 |
| --- | --- | --- |
| `joint_epoch` | 6 | **8** |
| `epochs_2nd` | 10 (bug — defaulted to this) | **20** |

All other parameters stay the same: `multispeaker: true`, `first_stage_path: first_stage_v3.pth`,
`lr: 1e-4`, `ft_lr: 1e-4`, `bert_lr: 1e-6`, `lambda_gen: 1.0`, `train_LM: false`, `batch_size: 8`,
`lambda_slm: 0.0`.

The `epochs_2nd` fix is also necessary: in v6c, the config used `epochs_2nd: 10` which was
effectively dead (Kokoro reads `epochs` not `epochs_2nd` in Stage 2), so it defaulted to 200.
Setting `epochs_2nd: 20` explicitly in v10 sets a controlled termination point.

### `diff_epoch` vs `joint_epoch` Interaction

In `train_second_safeguarded.py`, both `diff_epoch` and `joint_epoch` are read from `loss_params`
[t0009]. `diff_epoch` controls when diffusion losses activate; `joint_epoch` controls GAN losses.
For the v6c config (and therefore v10), both are 6 in v6c; in v10 we set `joint_epoch=8` but need to
decide whether `diff_epoch` also changes or stays at 6. The task_description.md does not mention
changing `diff_epoch`, so it should remain at the v6c value (6) unless the plan specifies otherwise.

## Reusable Code and Assets

### 1. `train_second_safeguarded.py` — **copy into task**

* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/train_second_safeguarded.py`
* **What it does**: Full Kokoro Stage 2 training loop with four safeguard integrations: DP-aware
  loader, JSONL step logging, per-epoch checkpoint manager, health gates. 1027 lines.
* **Reuse method**: **copy into task** — non-library code, must live in
  `tasks/t0010_stage2_safeguarded_training/code/`
* **Key functions**: `load_checkpoint(model, optimizer, path, load_only_params, ignore_modules)`,
  `main(config_path, run_id)` (click command)
* **Adaptation needed**:
  1. Fix `CheckpointManager` call (line 403): add `joint_epoch=joint_epoch` argument.
  2. The script already reads config from `--config-path` CLI flag — no other changes needed for v10
     other than pointing to the new config.

### 2. `extract_decoder.py` — **import via library**

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py`
* **What it does**: Converts raw StyleTTS2 checkpoint → five-module packaged format (strips
  `module.` prefixes, converts parametrization keys). 96 lines.
* **Reuse method**: **import via library** — registered as `tts_eval_harness`
* **Key function**: `extract(*, ckpt_path: Path, out_path: Path) -> dict[str, int]`
* **Usage for t0010**: call once per epoch checkpoint to produce the packaged `.pth` needed by
  `load_kokoro_model_with_checkpoint`

### 3. `StepLogger`, `CheckpointManager`, `HealthGate`, `capture_run_config` — **import via library**

* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/` (multiple files)
* **Reuse method**: **import via library** — registered as `t0009_training_safeguards`
* **Usage in copied script**: already integrated; no additional import changes needed beyond the
  `CheckpointManager` fix above

### 4. `score_speaker_sim.py` — **import via library** (invoked as subprocess)

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`
* **What it does**: Computes GE2E cosine speaker_sim and WER from synthesized WAV files using
  resemblyzer in a separate venv. Takes `--per-clip-in`, `--corpus-dir`, `--per-clip-out` args.
* **Reuse method**: **import via library** — registered as `tts_eval_harness`, but invoked via
  subprocess (not direct import) because resemblyzer requires a separate venv.
* **Usage for t0010**: after batch synthesis of all epoch checkpoints, run once per epoch to score
  speaker_sim; then parse output JSON to produce per-epoch curve.

### 5. `run_eval.py` — **import via library** (CLI invocation)

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/run_eval.py`
* **What it does**: synthesis + timing harness for multiple TTS systems. Takes `--systems`,
  `--prompt-set`, `--out-dir` etc. CLI flags. 465 lines.
* **Reuse method**: **import via library** — invoke as
  `python -m tasks.t0008_tts_eval_harness_baselines.code.run_eval` with appropriate flags.
* **Usage for t0010**: one invocation per epoch checkpoint with `--systems kokoro_v10_epochN` (or
  equivalent custom adapter). May require adding a new adapter for the v10 checkpoint format.

### 6. v6c config (`t0006_run03_v6c.yml`) — reference only (not copied)

* **Source**: `tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml`
* **What it does**: base config for v10, with `joint_epoch=6` changed to 8 and `epochs_2nd: 20`.
* **Reuse method**: read as reference; write new `config_david_v10.yml` based on it.

## Architecture Overview

The implementation involves three components interacting in sequence:

```
Training (VM):
  train_second_safeguarded.py  ──────────────────────────────────────────────────
    │  load_checkpoint() [DP-aware]                                              │
    │  capture_run_config() → <log_dir>/<run_id>/launch_info.json + config.yml  │
    │  StepLogger → <log_dir>/metrics.jsonl (per-step)                          │
    │  CheckpointManager → <log_dir>/<run_id>/epoch_NNNNN.pth + manifest.json  │
    │  HealthGate → sys.exit(3) on threshold breach                             │
    └──────────────────────────────────────────────────────────────────────────

Extraction (local or VM):
  extract_decoder.py → five-module packaged .pth per epoch

Evaluation (VM, resemblyzer venv):
  run_eval.py → per_clip_metrics.json  (ttfb, rtf, synth audio WAVs)
  score_speaker_sim.py → per_clip_metrics_scored.json  (speaker_sim per clip)
  report.py → metrics.json (variant format), tables.json, charts
```

The CheckpointManager saves to `<log_dir>/<run_id>/epoch_NNNNN.pth`, not the flat
`epoch_2nd_NNNNN.pth` that the training script also writes (line 965-966). Both exist after each
epoch. The harness consumes the flat checkpoint; the `CheckpointManager` copy is for safe per-epoch
retention with SHA-256 verification.

## Lessons Learned

### From t0009 (forensics):

* The upstream `load_checkpoint` with `strict=False` is a silent catastrophic failure. Never use it
  without the DP-aware wrapper. The fix (strip `module.`, raise on 0-param match) has been in t0001
  since the beginning but was dropped from every subsequent task.
* `joint_epoch=3` is too early. v6c at `joint_epoch=6` was the only successful run across 13
  attempts. Even at 6, divergence appeared 3 epochs after GAN activation. Setting 8 gives 2 more
  epochs of decoder convergence before GAN gradients arrive.
* Health gate thresholds need per-run calibration. The current thresholds are calibrated on a single
  log (v6c). If t0010 uses different batch size or data, thresholds may need adjustment.
* Only 1 of 13 training logs survived because of the launch script's cleanup behavior. JSONL logging
  is mandatory for post-hoc diagnosis.

### From t0008 (eval harness):

* Five-module packaging is required. Loading only `decoder` into KModel causes duration explosions
  (audio 10× too long). Always use `extract_decoder.py` before evaluation.
* resemblyzer must run in a separate venv. The `webrtcvad` dependency conflicts with the main
  project deps; the two-venv workflow is the proven pattern.
* Warmup synthesis must be discarded (at least 1 run). JIT compilation and CUDA cache effects
  inflate TTFB significantly if the first synthesis is included in timing.
* Duration ratio > 5.0 flags a failed synthesis (explosion). If > 10% of clips explode, the system
  result should be nulled rather than averaged.

### From t0005 and t0006 (training history):

* The `epochs_2nd` key is effectively unused in Stage 2 training — Kokoro reads `epochs` (not
  `epochs_2nd`) for the Stage 2 loop. Setting it in the config has no effect unless the script
  explicitly reads it.
* `train_LM: false` (frozen BERT) is necessary for stable Stage 2 convergence. With
  `train_LM: true`, LM Loss grows from ~23 to 400+ within a few epochs.

## Recommendations for This Task

1. **Copy `train_second_safeguarded.py` into `code/`** and immediately apply the one-line
   `CheckpointManager` fix (add `joint_epoch=joint_epoch`). This is the highest-priority adaptation
   required; without it, training crashes before epoch 1.

2. **Build `config_david_v10.yml` from the archived v6c config** at
   `tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml`. Change exactly
   two values: `joint_epoch: 8` and `epochs_2nd: 20`. All other parameters stay identical. Confirm
   `multispeaker: true`, `first_stage_path: first_stage_v3.pth`, `train_LM: false`.

3. **Verify `diff_epoch` value** in the new config. The task_description.md does not specify
   changing it from v6c (6), so keep it at 6 unless the plan says otherwise.

4. **Import safeguard library components directly** — do not copy `jsonl_logger.py`,
   `checkpoint_manager.py`, `health_gates.py`, or `run_config.py` into t0010's `code/`. They are
   registered library code and must be imported via
   `from tasks.t0009_stage2_training_failure_forensics.code.<module> import <class>`.

5. **Use `extract_decoder.py` before every harness evaluation** — the raw training checkpoint cannot
   be loaded by KModel directly. The extraction step takes ~30s per checkpoint and produces a
   packaged `.pth` in
   `CHECKPOINT_MODULES = ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder']` format.

6. **Run evaluation in batch after training completes**, not per-epoch during training. The
   resemblyzer venv setup takes ~10 minutes; running it 20 times serially during a 2-hour training
   run adds unacceptable overhead and risk of interrupting training.

7. **The `speaker_sim ≥ 0.85` target is ambitious**: best existing Kokoro is 0.631. Plan for partial
   results: report speaker_sim per epoch even if 0.85 is not reached; the curve is the scientific
   output.

8. **Deploy the idle watchdog** before launching training as per `CLAUDE.md` instructions. The
   watchdog (`arf/scripts/utils/idle_watchdog.sh`) runs on the VM and shuts it down after 60 min of
   ≤ 5% GPU utilization. This prevents runaway billing if the orchestrator disconnects.

## Task Index

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Provides the `tts_eval_harness` library that t0010 uses for per-epoch speaker_sim
  evaluation. Established baseline: best Kokoro (`v3_bundle`) scores 0.631 vs ElevenLabs 0.832.
  `extract_decoder.py` and `score_speaker_sim.py` are the two evaluation-side scripts that t0010
  invokes directly.

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Primary dependency. Provides `train_second_safeguarded.py` (the training script to
  copy) and the `t0009_training_safeguards` library (StepLogger, CheckpointManager, HealthGate,
  capture_run_config). Root-cause analysis identified two fixes that t0010 implements: DP-aware
  loader and `joint_epoch=8`. Also archives the v6c config that is the baseline for v10.

### [t0006]

* **Task ID**: `t0006_kokoro_v5_stage2_subset`
* **Name**: Kokoro v5 Stage 2: 250-clip subset, multispeaker: false
* **Status**: completed
* **Relevance**: The v6c run in this task is the only successful prior Stage 2 run. Its config
  (`joint_epoch=6`, `multispeaker=true`, `first_stage_v3.pth`) is the direct predecessor of the v10
  config. The v6c training log is the sole source for health-gate threshold calibration.

### [t0005]

* **Task ID**: `t0005_kokoro_v5_stage2_train`
* **Name**: Kokoro v5 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: Provides historical context for Stage 2 failure patterns (6 training attempts, all
  diverged). The `train_second_patched.py` in this task dropped the DP-aware loader that t0001 had,
  explaining the divergence. Also provides `first_stage_v3.pth` via the v3 reference path that v10
  uses as its Stage 1 checkpoint.

### [t0002]

* **Task ID**: `t0002_kokoro_v4_voicepack_decoder_package`
* **Name**: Package Kokoro v4 voicepack + decoder
* **Status**: completed
* **Relevance**: First task to establish the five-module checkpoint packaging pattern
  (`extract_decoder_generic.py`, 38 lines). The t0008 `extract_decoder.py` is a direct descendant
  with a clean function API. Understanding the five-module format requirement (bert + bert_encoder +
  predictor + text_encoder + decoder) was first documented here.
