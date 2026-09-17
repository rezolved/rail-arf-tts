---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
updated_at: "2026-09-17T14:39:00Z"
completed_steps: 10
next_step_number: 9
next_step_id: "implementation"
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

### Step 7 — planning

Wrote `plan/plan.md` (17 `REQ-*` items, 6 milestones, 20 numbered steps; verificator PASSED, 0
errors/0 warnings). The plan schedules: bounded VM inspection with a hard 90-minute wall-clock timer
and unconditional teardown step regardless of progress; checkpoint forensics generalizing `t0015`'s
`predictor_tensor_forensics.py` to all five v3 bundle modules to compute weight-norm deltas and
resolve the `multispeaker` three-way contradiction via module presence/shape (not source
preference); per-epoch sample gating with `t0015`'s `audio_quality_check.py`; the annotated
`data/config_david_v3_reconstructed.yml` with confirmed/inferred/unknown provenance on every line;
the mandatory human-listenable audio set (`v3_shipped`, `v3_per_epoch`, `elevenlabs_reference`,
`listening_guide.md`); the `v3_module_weight_delta.png` chart; and the `v3-recipe` answer asset.
Cost itemized at ~$21 VM + $0 local/API against the task's $30 cap. Caveat for downstream: while
planning, a live `dvc pull` against the v3 reference checkpoint failed with a transient Azure
`DefaultAzureCredential` auth error (cross-checked against `t0015`'s command logs as a known,
retry-resolvable failure) — the plan's Step 1 and Step 13 both build in bounded retry-with-backoff
for `dvc pull`/`dvc push` rather than treating one failure as a hard blocker.

### Step 8 — setup-machines

Acquired `LLM-T1-NC80` (2xH100 NVL, the project's sole Azure ML pool entry) via
`/setup-remote-machine` through Phase 5. GPU/CUDA verified (`2x NVIDIA H100 NVL`, CUDA 12.2), idle
watchdog installed and confirmed alive (PID 6143, 3600s idle timeout), and environment
sanity-checked (`~/kokoro-finetune/` present, `/mnt/cache/persist` resolves to the real Azure Files
share, SSH lingering enabled). Result recorded in `logs/steps/008_setup-machines/machine_log.json`.
Caveat for downstream: the first `acquire` attempt hit a one-time boot-timing race (VM's own Azure
`Start` operation hadn't finished within the tool's 480s SSH-readiness window), burning ~9.5 minutes
of the 90-minute VM cap and writing `intervention/pool_busy_llm-t1-nc80.md`; a retry succeeded in
~16 seconds with no wasted cost. The VM is left **running** with the watchdog active and locked for
this task — the `implementation` step must complete its read-only inventory and hand off to
`teardown` within the remaining budget (~80 of the original 90 minutes left).

* * *

## Cross-Step Decisions

* Planning (step 7) fixed the VM inspection budget at a hard 90-minute wall-clock cap with an
  unconditional teardown step (Step 7 of the plan) — this overrides "keep investigating" instincts
  if the cap is hit before all VM evidence sources are covered.

* * *

## Next Step Notes

`LLM-T1-NC80` is acquired, verified, watchdog-protected, and running now. Proceed to step 9,
`implementation`: execute `plan/plan.md` Milestone 1 Steps 4-7 (the read-only SSH inventory of
`~/kokoro-finetune/` and `/mnt/cache/persist`, per `task_description.md`'s Evidence sources), then
the checkpoint forensics, config reconstruction, audio packaging, and the `v3-recipe` answer asset.
The VM has already burned ~9.5 minutes of its 90-minute cap on the resolved boot-timing race (see
Step 8 above) — budget the remaining wall-clock time accordingly and do not exceed the $21 VM
sub-cap. `dvc pull` may still need a retry if it hits the transient `DefaultAzureCredential` auth
failure documented under Step 7. Tear the VM down via the `teardown` step immediately once the VM
portion of Milestone 1 is done — do not hold it open through the CPU-only forensics work.
