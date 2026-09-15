---
spec_version: "2"
task_id: "t0009_stage2_training_failure_forensics"
date_completed: "2026-09-14"
status: "complete"
---
## Objective

Systematically audit every Kokoro StyleTTS2 Stage 2 training run across four tasks (t0001, t0004,
t0005, t0006; 19+ launches, ~\$440 GPU spend) to identify the root cause(s) of persistent failure
and divergence. No new GPU training runs in this task. Deliverables are: (a) an **answer asset**
stating ranked root causes with evidence and a recommended next-run configuration, and (b) a
**library asset** containing a JSONL metrics logger, health-gate monitor, per-epoch checkpoint
retention policy, and an offline replay test that fires on every known failure and passes on v3's
successful run. Done when the answer asset explains the v3-vs-failure gap, the library passes its
offline replay test, and all seven result tables are written.

## Task Requirement Checklist

> **Task name**: Stage 2 training failure forensics and safeguards
> 
> **Short description**: Audit every Kokoro training run's logs, data, pipeline and checkpoints to
> explain why Stage 2 breaks, and ship checkpoint and log safeguards.
> 
> **Full scope** (from `task_description.md`): Collect surviving logs and configs; build per-run
> loss timelines; produce a confound table; audit training data; diff the patch stack against
> upstream and v3; audit checkpoint saving and loading; ship a safeguard library; write an answer
> asset.

- **REQ-1** — Collect every surviving training log from git, the VM (`/mnt/kikiri-tts/`), and
  `rail-benchmarks`. Produce a log inventory table (run_id, task, log present y/n, config present,
  Stage 1 checkpoint SHA-256, data list, outcome). Store logs in `data/logs/<run_id>/`.
- **REQ-2** — Parse every available loss field per step into a tidy table. Plot each run aligned on
  `joint_epoch`. Mark the first anomalous step and the leading indicator per run.
- **REQ-3** — Produce a confound table listing effective settings per run: data list and size,
  `multispeaker`, `first_stage_path` + SHA-256, `joint_epoch`, `lambda_gen`, `lambda_slm`,
  `lr`/`ft_lr`/`bert_lr`, `batch_size`, GPU/parallelism mode, `train_LM`, active patches.
  Reconstruct t0005's per-launch configs from log headers and git history.
- **REQ-4** — Data audit on v5 train/val lists and v3's 266-clip list: sample rate, channels,
  duration histogram, peak/clipping, silence, LUFS, frames-per-phoneme outliers, OOV tokens,
  train/val overlap, and whether v3's val set equals val_96.
- **REQ-5** — Pipeline audit: diff `train_second_patched.py` (t0005) against upstream
  `train_second.py` and v3's `train_second_patch.diff`. Classify each of the 7 patches as "fixes
  cause" or "hides symptom". Compare mel extraction and 24 kHz setup against Kokoro decoder
  expectations.
- **REQ-6** — Checkpoint audit: map every checkpoint file to its epoch, step, `val_loss`, config,
  and launch. Resolve the 4 known inconsistencies (t0004 epoch label, t0005 best epoch discrepancy,
  t0006 `epochs_2nd` bug, top-2 pruning vs quality). Verify checkpoint selection code and whether a
  pre-divergence checkpoint survives.
- **REQ-7** — Ship a tested safeguard library:
  - Per-epoch checkpoint saves + retention policy (never prune last pre-`joint_epoch` checkpoint).
  - JSONL metrics logger (one record per step: all losses, grad norms, skip count, LR, epoch).
  - Health gates: Dur Loss step-1 < 2.0; `acoustic_norm` after any epoch < 20; val spike ≤ 0.05
    post-`joint_epoch`; consecutive skip count ≤ 50.
  - `test_replay.py`: offline gates replay over committed logs — fires on failures, passes on v3.
- **REQ-8** — Answer asset: root causes ranked with evidence and confidence; what is still unknown;
  one recommended next-run configuration (one variable changed at a time from the closest known-good
  run).

## Approach

### Technical approach

All analysis is local/CPU-only. The single committed Stage 2 log
(`tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log`) is the primary log replay
input. VM access (≤ 30 min, \~\$7) is used only to retrieve any surviving logs and checkpoint
SHA-256 hashes from `/mnt/kikiri-tts/`.

**Forensic baseline**: v3's committed `train_second_patch.diff`
(`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff`) is the only record
of what v3 actually ran. Diffing it against t0005's full patch stack directly answers REQ-5.

**Safeguard library base**: Copy `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py`
(995 lines, all 7 crash patches) into `code/train_second_safeguarded.py`. Add JSONL logger, health
gates, and per-epoch checkpoint retention on top — minimal adaptation, no rewrite. Replace the
upstream `load_checkpoint` with the key-mismatch-raising version from
`tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py` lines 49–80.

**Data audit**: Copy `rejection_reason`, `load_kokoro_vocab`, `slug` from
`tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` (per cross-task copy rule — no library
infrastructure exists yet). Run them against v5 and v3 manifests to verify zero rejected lines;
compute audio stats with `librosa`/`soundfile` (already in the venv).

**Checkpoint audit**: The `load_val_loss` pattern from
`tasks/t0004_kokoro_v5_stage1_train/code/sync_and_monitor.sh` (embedded Python reads
`ckpt["val_loss"]`) is copied for the log parser. SHA-256 hashes are computed from checkpoint bytes
directly with `hashlib`.

**Offline replay test** (`test_replay.py`): Parses every collected log using the log-parser
functions, runs health gates against the step-by-step loss stream, asserts gates fire on every
known-failure run and do not fire on `run03_v6c_v3_stage1.log`. Pattern modelled on
`tasks/t0003_kokoro_v5_phoneme_data/code/test_gates.py` (58 lines, assert-based, no pytest
dependency).

### Key research findings embedded

- v3 opened at val_loss 0.57; all v5/v6 runs open at ~1.64 — a 3× gap before GAN activates. Most
  likely cause: v3 may have used `joint_epoch: 0` (GAN active from epoch 1, different loss
  definition). No v3 config is committed.
- `istftnet.py exp()` clamp (patch 7) stops the NaN crash but explicitly does NOT stop divergence.
  The GAN gradient is the cause; the clamp is a symptom fix.
- Upstream `load_checkpoint` with `strict=False` silently loads 0 params when a DataParallel-saved
  checkpoint meets a non-wrapped model. t0006 run02 (v6b) likely trained from scratch for this
  reason.
- Top-2 `val_loss` pruning may delete the only pre-divergence checkpoint. `val_loss` is not verified
  to track audio quality.
- t0006 v6d changed 6 variables simultaneously — the confound table is the only way to isolate which
  one stopped divergence.

### Alternatives considered

**Alternative: run a new training experiment to isolate variables** — rejected. The researcher
explicitly requested forensics-first; no new training in this task. Budget for GPU is capped at \$7
(log retrieval) plus an optional \$14 smoke test.

**Alternative: instrument upstream StyleTTS2 instead of copying t0005's script** — rejected. t0005's
script is the forensic baseline; copying it preserves the exact state under audit. Future tasks can
rebase on upstream once the root cause is confirmed.

### Task types

Task type: `data-analysis` (as declared in `task.json`). Planning Guidelines for `data-analysis`:
include `research-code` (done), `planning` (this step), `creative-thinking` (pending step 9), skip
`setup-machines` except for log retrieval (already skipped in step_tracker).

## Cost Estimation

| Item | Cost |
| --- | --- |
| Local analysis (log parsing, data audit, plan writing, library coding) | \$0 |
| VM access to LLM-T1-NC80 for log retrieval (≤ 30 min at \$13.96/hr for H100) | ~\$7 |
| Optional GPU smoke test (≤ 1 h) — only if offline replay cannot exercise the code path | ~\$14 |
| **Planned total** | **≤ \$25** |

Project budget remaining: \$5,000. Well within limits.

## Step by Step

### Milestone 1: Log collection and inventory (REQ-1)

1. **[CRITICAL] Collect surviving logs from git.** Search git history for any `.log` files committed
   to the repo. The only known committed log is
   `tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log`. Copy it to
   `tasks/t0009_stage2_training_failure_forensics/data/logs/t0006_run03_v6c/stage2.log`. Also copy
   the v3 reference diff:
   `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff` to
   `data/reference/v3/train_second_patch.diff`. Script: `code/collect_logs.py` (input: git log
   paths; output: files in `data/logs/`). Satisfies REQ-1 (partial — git sources only).

2. **Attempt log retrieval from VM.** SSH to `LLM-T1-NC80`. Check `/mnt/kikiri-tts/StyleTTS2/logs/`,
   `/mnt/kikiri-tts/kokoro-finetune/logs/`, `~/kokoro-finetune/logs/`. Retrieve any `.log`, `*.json`
   (TensorBoard event files), and any surviving config `.yml` files. Store each in
   `data/logs/<run_id>/`. If VM is down or `/mnt` was wiped, record in the inventory as "not
   present." Use `setup-remote-machine` skill only if the VM needs to be started first; otherwise
   SSH directly using the `LLM-T1-NC80` alias. Runtime: ≤ 30 min. Satisfies REQ-1 (VM sources).

3. **Write the log inventory table.** Script: `code/build_inventory.py` (input: `data/logs/`;
   output: `data/log_inventory.json` + markdown table written to `results/log_inventory.md`).
   Columns: `run_id`, `task`, `log_present`, `config_present`, `stage1_ckpt_path`,
   `stage1_ckpt_sha256`, `data_list`, `outcome`. Compute SHA-256 hashes of any locally available
   checkpoint files using `hashlib.sha256`. Record SHA-256 as `null` where file is unavailable.
   Satisfies REQ-1.

### Milestone 2: Per-run loss timelines (REQ-2)

4. **[CRITICAL] Write log parser.** Script: `code/parse_logs.py` (input: `data/logs/<run_id>/`;
   output: `data/timelines/<run_id>_steps.csv` with columns `run_id`, `epoch`, `step`, `loss_total`,
   `disc_loss`, `dur_loss`, `ce_loss`, `mel_loss`, `f0_loss`, `val_loss`, `acoustic_norm`,
   `grad_norm_msd`, `grad_norm_mpd`, `grad_norm_decoder`, `grad_norm_style_encoder`, `skip_count`).
   Parse the fixed-structure text log format:
   - Step lines: `Epoch [E/N], Step [S/T], Loss: X, Disc Loss: X, Dur Loss: X, CE Loss: X, ...`
   - Val lines: `Validation loss: X, Dur loss: X, F0 loss: X`
   - Norm lines: `voicepack/acoustic_norm=X prosodic_norm=X` Fields absent from a log (e.g., grad
     norms not present in t0001 logs) are written as `null`. The `joint_epoch` for each run is read
     from the config table (Step 7). Satisfies REQ-2.

5. **Plot loss timelines.** Script: `code/plot_timelines.py` (input: `data/timelines/*.csv`; output:
   `results/images/loss_timelines.png`, `results/images/acoustic_norm_grad_norm.png`). Panel 1:
   `dur_loss` per step, all runs aligned on `joint_epoch` (x-axis shifted so `joint_epoch` = 0).
   Panel 2: `acoustic_norm` and `grad_norm_decoder` per epoch, log scale. Mark the first anomalous
   step per run (defined as `dur_loss > 2.0` at step 1, or `acoustic_norm
   > 20`at any epoch, or val spike > 0.05). Use`matplotlib` (already a venv dependency). Satisfies
   > REQ-2.

### Milestone 3: Confound table (REQ-3)

6. **Reconstruct per-launch configs.** Read every config file in git history for t0001, t0004,
   t0005, t0006. For t0005 (config edited in place between launches), extract per-launch effective
   settings from log headers (log lines beginning with `Config:` or equivalent) and from the
   descriptions in `tasks/t0005_kokoro_v5_stage2_train/logs/README.md` and
   `tasks/t0006_kokoro_v5_stage2_subset/logs/README.md`. Document what cannot be recovered. Output:
   `data/configs/<run_id>.yml` for each recoverable config. Script: `code/collect_configs.py`
   (input: git paths + log headers; output: `data/configs/`). Satisfies REQ-3 (partial).

7. **Write confound table.** Script: `code/build_confound_table.py` (input: `data/configs/` +
   `data/log_inventory.json`; output: `data/confound_table.json` + `results/confound_table.md`).
   Columns per run: `run_id`, `task`, `data_list`, `n_clips`, `multispeaker`, `first_stage_path`,
   `first_stage_sha256`, `joint_epoch`, `lambda_gen`, `lambda_slm`, `lr`, `ft_lr`, `bert_lr`,
   `batch_size`, `gpu_count`, `parallelism_mode`, `train_LM`, `patches_active` (list of patch
   numbers 1-7). Cross-reference `first_stage_sha256` from Step 3 inventory to confirm which
   checkpoint each run actually loaded. Satisfies REQ-3.

### Milestone 4: Data audit (REQ-4)

8. **Copy gate functions from t0003.** Copy
   `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` to `code/prepare_v5_data.py`. Copy
   `tasks/t0003_kokoro_v5_phoneme_data/code/test_gates.py` to `code/test_gates_v5.py` (rename only,
   no change). These provide `rejection_reason`, `load_kokoro_vocab`, `slug`.

9. **[CRITICAL] Data audit script.** Script: `code/audit_data.py` (input: `data/v5/train_list.txt`,
   `data/v5/val_list.txt`, and v3 manifest from rail-benchmarks if accessible; output:
   `data/data_audit.json` + `results/data_audit_summary.md` +
   `results/images/duration_histogram.png`
   + `results/images/loudness_histogram.png`). Per clip: load audio with `soundfile`, compute sample
     rate, channels, duration, peak amplitude, LUFS (using `pyloudnorm` if present, else estimate
     from peak), leading/trailing silence (> -60 dBFS threshold), frames-per-phoneme (duration /
     n_phonemes from manifest). Aggregate: histogram of durations (v5 vs v3 side by side), LUFS
     distribution, clipping rate (peak > 0.99), silence rate, OOV check via `load_kokoro_vocab`.
     Train/val overlap check: compare audio filenames. val_96 overlap check: compare filenames
     against `data/v4/val/val_list.txt` (the 96-clip held-out set). Satisfies REQ-4.

### Milestone 5: Pipeline audit (REQ-5)

10. **Run the pipeline diff.** Script: `code/pipeline_diff.py` (input: v3 diff at
    `data/reference/v3/train_second_patch.diff`, t0005 patched script at
    `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py`; output:
    `data/pipeline_audit.md`). Apply v3's diff programmatically (using Python `subprocess` +
    `patch --dry-run`) to count which hunks are already in t0005's script and which are not. For
    each of the 7 known patches, state: (a) present in v3 diff? (b) present in t0005 script? (c)
    classification: "fixes cause", "hides symptom", or "neutral crash fix". Also compare mel
    extraction parameters (sample rate, n_mels, hop_length, win_length) in t0005's config against
    Kokoro's `config.json` defaults. Satisfies REQ-5.

### Milestone 6: Checkpoint audit (REQ-6)

11. **Checkpoint map.** Script: `code/audit_checkpoints.py` (input: `data/log_inventory.json`,
    `data/configs/`, `data/timelines/`; output: `data/checkpoint_map.json` +
    `results/checkpoint_audit.md`). For each run: list every checkpoint filename found in git or on
    VM (from inventory), derive 0-based epoch from filename, cross-reference val_loss from the
    parsed timeline (Step 4), note config's `joint_epoch`, note whether the checkpoint is before or
    after `joint_epoch`, compute SHA-256 where file is accessible. Resolve the 4 known
    inconsistencies:
    - t0004 `epoch_1st_00007.pth` vs "epoch 10" label: state 0-based epoch = 7, 1-based = 8, not 10.
    - t0005 best checkpoint: `epoch_2nd_00003.pth` = 0-based epoch 3 (4th epoch). README "epoch 2"
      is wrong.
    - t0006 `epochs_2nd: 10` bug: document that script read `epochs_2nd` not `epochs`; training
      stopped at epoch 10, not 15.
    - Top-2 pruning: document which pre-`joint_epoch` checkpoints were pruned and whether a healthy
      GAN-phase checkpoint survived. Satisfies REQ-6.

### Milestone 7: Safeguard library (REQ-7)

12. **[CRITICAL] Copy and extend training script.** Copy
    `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` to
    `code/train_second_safeguarded.py`. Apply the following additions:
    - Replace `load_checkpoint` (upstream `utils.py` version, `strict=False`) with the t0001 version
      (lines 49-80 of `tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py`): strips
      `module.` prefix, raises `RuntimeError` on 0-param match.
    - Add `code/jsonl_logger.py`: class `StepLogger` with `log(step_dict)` that appends one JSON
      record per call to `<log_dir>/metrics.jsonl`. Fields: `epoch`, `step`, `loss_total`,
      `disc_loss`, `dur_loss`, `ce_loss`, `mel_loss`, `f0_loss`, `val_loss` (null during training
      steps), `acoustic_norm` (null during training steps), `grad_norm_msd`, `grad_norm_mpd`,
      `grad_norm_decoder`, `grad_norm_style_encoder`, `skip_count`, `lr`, `timestamp_utc`.
    - Call `StepLogger.log(...)` at every training step and after each validation pass inside
      `train_second_safeguarded.py`.
    - Replace top-2 `val_loss` pruning with per-epoch retention policy in
      `code/checkpoint_manager.py`: `CheckpointManager` saves every epoch checkpoint to
      `<log_dir>/<run_id>/epoch_{N:05d}.pth`, computes SHA-256, writes a `checkpoints.json` manifest
      (fields: epoch, step, val_loss, sha256, is_pre_joint_epoch, flagged_healthy). Never deletes
      the last checkpoint with `is_pre_joint_epoch=True`.
    - Add `code/health_gates.py`: `HealthGate` class with thresholds:
      - `dur_loss_step1_max: 2.0` (fires if Dur Loss at first Step 2 step ≥ 2.0)
      - `acoustic_norm_max: 20.0` (fires after any epoch if `acoustic_norm ≥ 20`)
      - `val_spike_max: 0.05` (fires post-`joint_epoch` if `val_loss` increases by > 0.05)
      - `consecutive_skip_max: 50` (already in t0005; keep)
        `HealthGate.check(epoch, step, metrics) -> GateResult` returns
        `(fired: bool, gate_name: str, value: float, threshold: float, last_healthy_ckpt: str)`.
        When fired: log to JSONL, print message pointing to last healthy checkpoint, call
        `sys.exit(3)`.
    - `code/run_config.py`: at launch, copy the resolved config YAML and current git SHA to
      `<log_dir>/<run_id>/config.yml` and `<log_dir>/<run_id>/launch_info.json`. Never overwrite;
      use a per-run subdirectory with timestamp. Satisfies REQ-7.

13. **[CRITICAL] Offline replay test.** Script: `code/test_replay.py` (no pytest — assert-based
    `main()` block). Imports: `parse_logs.py` (Step 4), `health_gates.py` (Step 12). For each log in
    `data/logs/`:
    - Parse step records.
    - Run `HealthGate` over the stream.
    - Assert: gate fires at or before the step documented in the inventory as first anomalous step.
      Required: fires on t0006 run03 (if divergence data available), does not fire on v3's
      `run03_v6c_v3_stage1.log` (the successful run). For runs with no log (t0005 runs 1-5): skip
      with a printed note. Expected output on success:
      `PASS: N gates fired on N_fail runs, 0 false positives on N_pass runs`. Satisfies REQ-7.

14. **Run `test_replay.py` and `test_gates_v5.py`.** Wrap with `run_with_logs.py`. Both must exit 0
    before this milestone is considered complete.

### Milestone 8: Answer asset (REQ-8)

15. **[CRITICAL] Write answer asset.** File: `assets/answer/t0009_stage2_forensics_answer.md`.
    Format per `meta/asset_types/answer/specification.md`. Sections:
    - **Root causes (ranked)**: Each cause with evidence (log file + step, confound table row, code
      reference), confidence level (high/medium/speculative), and the forensic signal that confirms
      or disconfirms it.
    - **What remains unknown**: v3 config (especially `joint_epoch`), logs for t0005 runs 1-5,
      whether `val_loss` 0.506 is on the same val set as v5 runs.
    - **Recommended next-run configuration**: Single table. Base: t0006 run04 (v6d, the closest
      known-good run, `lambda_gen=0.05`, `joint_epoch=6`, 250 clips, v3 Stage 1 checkpoint). Change
      exactly one variable (the most likely remaining cause from the confound table). State which
      health gate to watch and what threshold to expect on a healthy run. Satisfies REQ-8.

## Remote Machines

VM access to `LLM-T1-NC80` (Azure ML H100, `project/azure_vm.json`) is needed only for Step 2 (log
retrieval, ≤ 30 min). Use SSH directly via the `LLM-T1-NC80` alias; no GPU work, so the VM can be
started and stopped via the pool manager without a full `setup-remote-machine` invocation. If the
VM's `/mnt` has been wiped since the last task, record what is missing and proceed with git-only
logs.

Optional: one short GPU smoke test (≤ 1 h) to confirm `train_second_safeguarded.py` writes JSONL and
triggers a health gate. Only run if the offline replay cannot exercise the code path. Budget: ~\$14.
Total budget: ≤ \$25.

## Assets Needed

| Asset | Source |
| --- | --- |
| `tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py` | Direct read from t0005 task folder (dependency) |
| `tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py` (lines 49-80) | Direct read from t0001 task folder |
| `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` | Copy into task code/ |
| `tasks/t0003_kokoro_v5_phoneme_data/code/test_gates.py` | Copy into task code/ |
| `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff` | Copy into data/reference/ |
| `tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log` | Copy into data/logs/ |
| `tasks/t0004_kokoro_v5_stage1_train/code/sync_and_monitor.sh` | Reference for checkpoint prune pattern |
| v5 train/val lists (`data/v5/train_list.txt`, `data/v5/val_list.txt`) | From t0003 results in this repo |
| v3 audio and manifests | DVC-tracked in t0006; `dvc pull` to materialise |
| VM logs and checkpoints | Retrieved via SSH in Step 2 |

## Expected Assets

| Asset type | Asset ID | Description |
| --- | --- | --- |
| `answer` | `t0009_stage2_forensics_answer` | Root-cause analysis and recommended next-run configuration |
| `library` | `t0009_training_safeguards` | JSONL logger, health gates, checkpoint manager, log parser, offline replay test |

These match `task.json` `expected_assets: {answer: 1, library: 1}`.

## Time Estimation

| Phase | Estimated wall-clock time |
| --- | --- |
| Log collection from git + VM (Steps 1-2) | 1-2 h (dominated by VM SSH session) |
| Log inventory + parsing + timelines (Steps 3-5) | 2-3 h |
| Confound table + config reconstruction (Steps 6-7) | 1-2 h |
| Data audit + plots (Steps 8-9) | 2-3 h |
| Pipeline diff (Step 10) | 1 h |
| Checkpoint audit (Step 11) | 1-2 h |
| Safeguard library implementation (Steps 12-13) | 3-4 h |
| Replay test execution + answer asset (Steps 14-15) | 1-2 h |
| **Total** | **12-19 h** |

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| VM `/mnt` wiped since last task — t0005 logs 1-5 are gone | High | Confound table has gaps; replay test cannot assert on runs 1-5 | Record as "not present" per LESSONS Lesson 10; replay test skips those runs with a note; answer asset explicitly lists them as unknown |
| v3 config never committed — `joint_epoch` unknown | High | Root-cause ranking has a gap | Document as "speculative" confidence; recommended next-run tests `joint_epoch: 0` as one variable change |
| `librosa`/`pyloudnorm` not available in venv | Low | LUFS audit is approximate | Use `soundfile` + peak-based estimate; note in data audit |
| `train_second_safeguarded.py` cannot be import-tested without GPU | Medium | Replay test covers only log-parser and gate logic, not the full training loop | Run offline replay test for gate/parser; smoke test (Step optional) for full loop |
| t0006 v6d audio shows `acoustic_norm=8.51` despite `lambda_gen=0.05` fix | Medium | Root cause still partially unknown after forensics | Confound table will isolate whether `lr=1e-4` (reverted from safe `3e-5`) is the remaining variable |
| Checkpoint SHA-256 mismatch reveals t0006 run02 trained from scratch | Medium | Changes answer asset conclusion | Document; recommend safe `load_checkpoint` in next run |

## Verification Criteria

- `uv run python -m arf.scripts.verificators.verify_plan t0009_stage2_training_failure_forensics` —
  zero errors (confirms plan structure).
- `uv run python -m arf.scripts.utils.run_with_logs --task-id t0009_stage2_training_failure_forensics -- python tasks/t0009_stage2_training_failure_forensics/code/test_replay.py`
  — exits 0, prints `PASS`.
- `uv run python -m arf.scripts.utils.run_with_logs --task-id t0009_stage2_training_failure_forensics -- python tasks/t0009_stage2_training_failure_forensics/code/test_gates_v5.py`
  — exits 0.
- `data/log_inventory.json` exists; every known run_id (t0001, t0004, t0005 r1-r6, t0006 r01-r04,
  v3) has an entry.
- `data/confound_table.json` exists; every run has `first_stage_sha256` set (to actual hash or
  explicit `null`).
- `data/timelines/t0006_run03_v6c_steps.csv` exists with > 0 rows; all val_loss values are finite.
- `results/images/loss_timelines.png` and `results/images/acoustic_norm_grad_norm.png` exist and are
  \> 0 bytes.
- `assets/answer/t0009_stage2_forensics_answer.md` exists and passes
  `uv run python -m arf.scripts.verificators.verify_answer_asset --task-id t0009_stage2_training_failure_forensics`.
- `assets/library/` contains the safeguard module files; `verify_library_asset.py` passes.
- `results/costs.json` has `total_cost_usd` set to actual spend (not the zero-cost default);
  `breakdown` lists VM time if used.
- All 7 REQ items are addressed in `results/results_detailed.md` under
  `## Task Requirement Coverage`.
