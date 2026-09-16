---
spec_version: "1"
task_id: "t0013_v10_synthesis_quality_forensics"
updated_at: "2026-09-16T12:48:00Z"
completed_steps: 10
next_step_number: 9
next_step_id: "implementation"
---
# Task Objective

Determine whether kokoro-v10-best produces noise instead of speech due to a bug in an ad hoc
inference reproduction, or a real checkpoint defect, and act accordingly.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0013_v10_synthesis_quality_forensics` created. Initial folder structure initialized in
`tasks/t0013_v10_synthesis_quality_forensics/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py`, which passed with no errors or warnings; the aggregator confirms
`t0010_stage2_safeguarded_training` (the task that produced the kokoro-v10-best checkpoints under
investigation) has `status: "completed"`. Result recorded in
`logs/steps/002_check-deps/deps_report.json`. No caveats — the checkpoint under investigation is
confirmed available for step 6 (`research-code`) and step 9 (`implementation`).

### Step 4 — research-papers

Skipped: this is an empirical debugging/forensics task (checkpoint loading, weight inspection, audio
inference), not a literature question. No published-paper evidence bears on the root cause.

### Step 5 — research-internet

Skipped: `task_description.md` already specifies the exact reproduction recipe (torch pin, StyleTTS2
demo notebook, dependency list) discovered in the prior ad hoc session, so no new external research
is needed to execute it.

### Step 8 — setup-machines

Skipped: `task_description.md` specifies a CPU-only venv reproduction (torch==2.5.1 CPU, espeak-ng,
local StyleTTS2 inference) with no GPU training or large-scale inference involved.

### Step 10 — teardown

Skipped: no remote machine was provisioned (`setup-machines` not included), so there is nothing to
tear down.

### Step 13 — compare-literature

Skipped: this task produces internal forensic evidence (key-load diagnostics, weight-norm
comparisons, audio checks) about one project's own checkpoints, not quantitative results comparable
to a published baseline.

### Step 3 — init-folders

Created the mandatory task folder structure (`plan/`, `research/`, `results/`, `results/images/`,
`corrections/`, `intervention/`, `code/`, `logs/commands/`, `logs/searches/`, `logs/sessions/`,
`logs/steps/`, `assets/`) via `init_task_folders`, logged to
`logs/steps/003_init-folders/folders_created.txt`. Populated the local aggregator cache under
`tasks/t0013_v10_synthesis_quality_forensics/ctx/` (task_types, costs, tasks, metrics, suggestions)
for reuse by downstream subagents; `ctx/` is gitignored and not committed. No caveats.

### Step 6 — research-code

Reviewed 12 completed tasks (deep-diving into t0001, t0002, t0005, t0006, t0008, t0009, t0010) and 2
registered libraries, wrote `research/research_code.md` (verificator: PASSED, no errors/warnings),
then spawned `/research-summarize` to produce `research/research_summary.md`. Central new finding:
`config_david_v10.yml` is the project's only Stage 2 config with
`model_params.decoder.type: hifigan` (all others use `istftnet`), and `train_second_v10.py`'s
checkpoint loader does not exclude `decoder` when loading the ISTFTNet-shaped `first_stage_v3.pth`,
making a silent architecture mismatch the leading noise-output hypothesis.

### Step 7 — planning

Spawned a dedicated subagent to execute `/planning`, which wrote `plan/plan.md` (verificator:
PASSED, no errors/warnings) sequencing a cheap, venv-free checkpoint-tensor falsifier (direct
`net["decoder"]` key-name / NaN-Inf / weight-norm inspection against a known-good `istftnet`
control) strictly before the full instrumented StyleTTS2-native inference harness build. Traced
`first_stage_v3.pth` to a concrete DVC-tracked path
(`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth`) and located a
reusable speaker-sim scoring pattern
(`tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`) and the 11labs David reference
corpus. Caveat: `ctx/task_types.json` shows `has_external_costs: true` for this task's types
(`tts-benchmark-run`, `code-reproduction`), not `false` — did not block planning since the budget
gate only fires at `create-branch`, but implementation should be aware the type declares external
costs even though this task's actual compute is CPU-only local work with $0 real cost.

* * *

## Cross-Step Decisions

* Leading root-cause hypothesis for v10 noise (from `research-code`, step 6):
  `config_david_v10.yml`'s `decoder.type: hifigan` vs. Stage 1's ISTFTNet-shaped weights in
  `first_stage_v3.pth`, silently partially loaded by `train_second_v10.py:load_checkpoint()`'s
  zero-match-only failure guard. Later steps (planning, implementation) should test this first via a
  cheap checkpoint-tensor check before building the full inference harness.
* `plan/plan.md` (step 7) sequences work into milestones: Milestone A (venv/checkpoint pull setup),
  Milestone B (cheap checkpoint-tensor falsifier — run first, before any inference code), a hard
  control-validation gate, then Milestones C-E (instrumented StyleTTS2-native harness, control test,
  v10 diagnosis). Implementation must follow this order and must not skip Milestone B.

* * *

## Next Step Notes

Step 7 (`planning`) completed; `plan/plan.md` is in place with all 11 mandatory sections plus a
`## Rejection Criteria` section, verificator PASSED with no errors or warnings. Step 8
(`setup-machines`) is already marked `skipped` in `step_tracker.json` (CPU-only local task). Proceed
to step 9, `implementation`: follow `plan/plan.md`'s Step by Step section exactly, starting with
Milestone A (build the CPU StyleTTS2/kokoro-finetune venv per the task description's pinned
dependency recipe, `dvc pull` the checkpoints and `first_stage_v3.pth` control file) then Milestone
B (the cheap checkpoint-tensor falsifier) before writing any inference code. Read `plan/plan.md` in
full — it is self-contained and names every script, file path, and expected output. If Milestone B
alone resolves the root-cause question, the plan's Rejection Criteria section describes when to skip
straight to the diagnosis write-up rather than building the full harness.
