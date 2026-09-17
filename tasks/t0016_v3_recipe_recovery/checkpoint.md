---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
updated_at: "2026-09-17T13:59:30Z"
completed_steps: 8
next_step_number: 7
next_step_id: "planning"
---
# Task Objective

Reconstruct the exact Kokoro v3 Stage 2 recipe (config, data list, patches, epochs, environment)
from VM files, DVC artifacts and checkpoint forensics, without access to the original author.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0016_v3_recipe_recovery` created. Initial folder structure initialized in
`tasks/t0016_v3_recipe_recovery/`. Step 1 is a mechanical setup step with no research output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via `prestep` and again via `run_with_logs` for the log record):
both declared dependencies, `t0006_kokoro_v5_stage2_subset` and
`t0009_stage2_training_failure_forensics`, have `status: "completed"` in their `task.json`, and the
verificator reported PASSED with 0 errors and 0 warnings. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats.

### Step 3 — init-folders

Created the mandatory task folder structure via `init_task_folders` (12 directories with `.gitkeep`,
plus `__init__.py` and `code/__init__.py`), recorded in
`logs/steps/003_init-folders/folders_created.txt`. Populated the local, gitignored aggregator
context cache at `tasks/t0016_v3_recipe_recovery/ctx/` (task_types, costs, tasks, metrics,
suggestions) for reuse by downstream step-executors. No caveats.

### Step 4 — research-papers

Skipped, per `step_tracker.json`: this task is checkpoint/VM forensics, not literature-driven — no
papers in the corpus bear on Kokoro/StyleTTS2 recipe recovery from local artifacts.

### Step 5 — research-internet

Skipped, per `step_tracker.json`: `task_description.md`'s evidence sources are entirely internal (VM
home dir, DVC artifacts, prior task checkpoints/code); no external internet research is required or
listed as an evidence source.

### Step 6 — research-code

Wrote `research/research_code.md` (13 tasks cited, 2 libraries documented; verificator PASSED, 0
errors/0 warnings). Confirmed the four tasks named in `task_description.md` are directly reusable:
`t0006`'s `code/config_david_v6c_stage2.yml` is the reconstruction template; `t0015`'s
`code/predictor_tensor_forensics.py` and `code/audio_quality_check.py` are the checkpoint-diff and
audio-gate scripts to copy into this task's `code/`; `t0002`'s `extract_decoder_generic.py` confirms
the packaging recipe; `t0009`'s `checkpoint_manager.py`/`health_gates.py`/`confound_table.md` supply
the forensics conventions and the open confound ledger. Caveat for downstream: found a genuine
three-way `multispeaker` contradiction across `best/config.json` (true), `t0009`'s confound table
(true, assumed), and `t0006`'s `config_david_v6c_stage2.yml` (false, inline-commented) — none
carries a SHA-256, so only checkpoint-shape forensics in the implementation step can resolve it.

### Step 11 — creative-thinking

Skipped, per `step_tracker.json`: `task_description.md` fully specifies the evidence sources,
priority order, and reconstruction methodology; no exploratory alternative-approach analysis is
called for.

### Step 13 — compare-literature

Skipped, per `step_tracker.json`: this task reconstructs an internal training recipe from artifacts;
it does not produce results comparable to published external baselines.

* * *

## Cross-Step Decisions

* * *

## Next Step Notes

`research/research_code.md` is complete and verified (13 tasks cited, 2 libraries documented, 0
errors/0 warnings). Proceed to step 7, `planning`: design the bounded VM inspection (90-minute / $21
cap on `LLM-T1-NC80`), checkpoint/sample forensics, config reconstruction, audio packaging, and
answer-asset plan under the task's $30 total budget. Reuse
`t0015/code/predictor_tensor_forensics.py` and `t0015/code/audio_quality_check.py` (copy into this
task's `code/`), `t0006`'s `config_david_v6c_stage2.yml` as the reconstruction template, and
`t0002`'s packaging pointer. The plan must explicitly schedule resolution of the `multispeaker`
true/false contradiction via checkpoint-shape forensics (module presence/shape check), not source
preference.
