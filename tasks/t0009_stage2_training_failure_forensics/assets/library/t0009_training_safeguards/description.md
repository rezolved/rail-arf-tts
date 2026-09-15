---
spec_version: "2"
library_id: "t0009_training_safeguards"
documented_by_task: "t0009_stage2_training_failure_forensics"
date_documented: "2026-09-14"
---
# Kokoro Stage 2 Training Safeguards

## Metadata

- **Name**: Kokoro Stage 2 Training Safeguards
- **Version**: 0.1.0
- **Task**: `t0009_stage2_training_failure_forensics`
- **Dependencies**: None beyond project stdlib and torch
- **Modules**: `code/jsonl_logger.py`, `code/checkpoint_manager.py`, `code/health_gates.py`,
  `code/run_config.py`, `code/constants.py`, `code/paths.py`

## Overview

This library was built during the Stage 2 forensics task to prevent the class of failures that
caused 19+ Kokoro-82M Stage 2 training runs to diverge or crash between tasks t0001 and t0006.
Forensics identified two root causes: (1) the upstream `load_checkpoint` silently loads zero
parameters when a DataParallel checkpoint is passed, causing runs to train from random
initialization; (2) `joint_epoch` set too early activates GAN losses before the decoder is stable.

The library provides four components that work together: `StepLogger` writes every training step and
validation pass to a JSONL file, making offline diagnosis possible without TensorBoard.
`CheckpointManager` saves a checkpoint after every epoch with a SHA-256 manifest, eliminating the
top-2 pruning risk that could delete the last pre-divergence checkpoint. `HealthGate` monitors four
signals (dur_loss at epoch start, acoustic_norm, val_loss spike, consecutive NaN skip count) and
aborts training with `sys.exit(3)` when a threshold is exceeded, before divergence makes the run
worthless. `capture_run_config` records the config YAML and git SHA at startup so every run is
reproducible and traceable.

The components are designed to be dropped into an existing training script with minimal changes.
`train_second_safeguarded.py` in the same task demonstrates a complete integration derived from
`t0005_kokoro_v5_stage2_train/code/train_second_patched.py`.

## API Reference

### `StepLogger` (`code/jsonl_logger.py`)

```python
class StepLogger:
    def __init__(self, log_dir: str, run_id: str) -> None: ...
    def log(self, record: dict[str, object]) -> None: ...
```

Appends a JSON record to `<log_dir>/metrics.jsonl`. Each call adds a UTC timestamp. The `record`
dict should include `epoch`, `step`, and any loss scalars. Field names are freeform; the class does
not validate them.

### `CheckpointManager` (`code/checkpoint_manager.py`)

```python
class CheckpointManager:
    def __init__(self, log_dir: str, run_id: str) -> None: ...
    def save(
        self,
        model: dict,
        optimizer: object,
        epoch: int,
        step: int,
        val_loss: float,
    ) -> Path: ...
    def last_pre_joint_checkpoint(self, joint_epoch: int) -> Path | None: ...
```

Saves `<log_dir>/<run_id>/epoch_{N:05d}.pth` for every epoch. Writes/updates
`<log_dir>/<run_id>/checkpoints.json` manifest with epoch, val_loss, file size, and SHA-256 hash.
Returns the saved path. `last_pre_joint_checkpoint(joint_epoch)` returns the path of the last epoch
saved before GAN activation, or None if none exists.

### `HealthGate` (`code/health_gates.py`)

```python
@dataclass(frozen=True, slots=True)
class GateResult:
    fired: bool
    gate_name: str
    value: float | None
    threshold: float | None
    last_healthy_ckpt: str | None
    message: str

class HealthGate:
    def __init__(
        self,
        joint_epoch: int,
        last_healthy_ckpt: str | None = None,
        logger: StepLogger | None = None,
    ) -> None: ...
    def check(self, epoch: int, step: int, metrics: dict[str, float | None]) -> GateResult: ...
    def update_last_checkpoint(self, ckpt_path: str) -> None: ...
    def record_skip(self) -> GateResult | None: ...
    def reset_skips(self) -> None: ...
```

`check()` evaluates all applicable gates for the given epoch/step:

- **Gate 1 (dur_loss_step1)**: fires when `dur_loss >= 2.0` at step 1 of epoch >= 2. Indicates the
  model is not converging.
- **Gate 2 (acoustic_norm)**: fires when `acoustic_norm >= 20.0` at a val record (step == 0).
  Indicates the style encoder is diverging.
- **Gate 3 (val_spike)**: fires when `val_loss` increases by more than 0.05 in a single epoch after
  `joint_epoch`. Indicates post-GAN divergence.

`record_skip()` increments the consecutive-skip counter; fires when count reaches 50. Call
`reset_skips()` after any successful step. Call `update_last_checkpoint()` each time a checkpoint is
saved so the fired `GateResult` carries the recovery path.

All gate thresholds are in `code/constants.py`.

### `capture_run_config` (`code/run_config.py`)

```python
def capture_run_config(
    config_path: str,
    log_dir: str,
    run_id: str,
    extra_info: dict[str, object] | None = None,
) -> None: ...
```

Copies the config YAML to `<log_dir>/<run_id>/config.yml` and writes
`<log_dir>/<run_id>/launch_info.json` with `run_id`, `config_path`, `git_sha`, `git_branch`,
`timestamp_utc`, and any `extra_info`. Never overwrites an existing run directory.

## Usage Examples

```python
import sys
from tasks.t0009_stage2_training_failure_forensics.code.jsonl_logger import StepLogger
from tasks.t0009_stage2_training_failure_forensics.code.checkpoint_manager import CheckpointManager
from tasks.t0009_stage2_training_failure_forensics.code.health_gates import HealthGate
from tasks.t0009_stage2_training_failure_forensics.code.run_config import capture_run_config

# At training startup
capture_run_config(config_path="Configs/config_david_v6c.yml", log_dir="logs/run01", run_id="v6c_run01")

step_logger = StepLogger(log_dir="logs/run01", run_id="v6c_run01")
ckpt_mgr = CheckpointManager(log_dir="logs/run01", run_id="v6c_run01")
gate = HealthGate(joint_epoch=6, logger=step_logger)

for epoch in range(start_epoch, epochs):
    for i, batch in enumerate(train_dataloader):
        # ... compute losses ...

        if nan_detected:
            result = gate.record_skip()
            if result and result.fired:
                sys.exit(3)
            continue

        gate.reset_skips()

        # Gate: dur_loss at first step
        if i == 0:
            result = gate.check(epoch=epoch + 1, step=1, metrics={"dur_loss": float(loss_dur)})
            if result.fired:
                sys.exit(3)

        # Log step
        step_logger.log({"epoch": epoch + 1, "step": i + 1, "dur_loss": float(loss_dur)})

    # After validation
    ckpt_path = ckpt_mgr.save(model=model, optimizer=optimizer, epoch=epoch, step=iters, val_loss=val_loss)
    gate.update_last_checkpoint(str(ckpt_path))

    # Gate: acoustic_norm and val_spike
    result = gate.check(epoch=epoch + 1, step=0, metrics={"val_loss": val_loss, "acoustic_norm": acoustic_norm})
    if result.fired:
        sys.exit(3)
```

## Dependencies

No external dependencies beyond `torch` and the Python standard library. `torch` is required only by
`CheckpointManager.save()` which calls `torch.save`. The other three components (`StepLogger`,
`HealthGate`, `capture_run_config`) use only stdlib (`json`, `pathlib`, `hashlib`, `datetime`,
`subprocess`).

## Testing

Run the offline log-replay test:

```bash
uv run python -m arf.scripts.utils.run_with_logs --task-id t0009_stage2_training_failure_forensics \
  -- python -m tasks.t0009_stage2_training_failure_forensics.code.test_replay
```

Expected output: `PASS: 1 gate check(s) passed, 0 early false positives.`

Run the v5 phoneme gate smoke test:

```bash
uv run python -m arf.scripts.utils.run_with_logs --task-id t0009_stage2_training_failure_forensics \
  -- python -m tasks.t0009_stage2_training_failure_forensics.code.test_gates_v5
```

Expected output: `gates ok`

The replay test parses the only available log (t0006_run03_v6c `stage2.log`) and verifies that the
health gates fire at epoch 8 (val spike +0.114 > 0.05) while not firing before epoch 7 (where the
best checkpoint at epoch 6 was already saved). This validates the gate logic against real training
data.

## Main Ideas

* **Fail loudly on checkpoint mismatch.** The upstream loader uses `strict=False` and silently loads
  zero parameters when key names do not match. This library's `load_checkpoint` (in
  `train_second_safeguarded.py`) strips `module.` prefixes and raises `RuntimeError` on zero-param
  match, making the mismatch visible immediately at startup.
* **Per-epoch retention over top-2 pruning.** Top-2 pruning preferentially keeps the lowest val_loss
  checkpoints (pre-GAN epochs), deleting the most recent pre-divergence checkpoint precisely when it
  is most needed. `CheckpointManager` keeps every epoch.
* **Gates are abort conditions, not logs.** When a gate fires, the correct response is `sys.exit(3)`
  — not a warning. Early exit preserves the last healthy checkpoint and avoids continuing to consume
  GPU budget on a diverging run.

## Summary

The Kokoro Stage 2 Training Safeguards library addresses the root causes of 19+ failed training
runs. The two proven causes — silent zero-param checkpoint loading and premature GAN activation —
are fixed at the script level in `train_second_safeguarded.py`. The four library components
(StepLogger, CheckpointManager, HealthGate, capture_run_config) provide the monitoring
infrastructure needed to detect future divergence before it wastes the remainder of a training run.

The library integrates with the existing Kokoro training pipeline with minimal changes: three
imports, one startup call to `capture_run_config`, and three gate checks per epoch (at step 1, after
each NaN skip, and after validation). No changes to the model architecture, optimizer, or data
pipeline are required.

Current limitations: gate thresholds (dur_loss >= 2.0, acoustic_norm >= 20.0, val_spike > 0.05) are
calibrated from a single successful run (v6c). If future runs use a different dataset, batch size,
or learning rate, thresholds may need to be adjusted. The constants are centralized in
`code/constants.py` for easy tuning. The late divergence in v6c (epoch 9+) is not fully explained by
these safeguards; investigating SLM adversarial loss stability and learning rate schedule effects is
the recommended next step.
