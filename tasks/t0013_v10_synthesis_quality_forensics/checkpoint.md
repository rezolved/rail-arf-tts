---
spec_version: "1"
task_id: "t0013_v10_synthesis_quality_forensics"
updated_at: "2026-09-16T12:35:00Z"
completed_steps: 9
next_step_number: 7
next_step_id: "planning"
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

* * *

## Cross-Step Decisions

* Leading root-cause hypothesis for v10 noise (from `research-code`, step 6):
  `config_david_v10.yml`'s `decoder.type: hifigan` vs. Stage 1's ISTFTNet-shaped weights in
  `first_stage_v3.pth`, silently partially loaded by `train_second_v10.py:load_checkpoint()`'s
  zero-match-only failure guard. Later steps (planning, implementation) should test this first via a
  cheap checkpoint-tensor check before building the full inference harness.

* * *

## Next Step Notes

Step 6 (`research-code`) completed; `research/research_code.md` (12 tasks reviewed, 7 cited, 2
libraries) and `research/research_summary.md` are in place. Central new finding:
`config_david_v10.yml` is the only Stage 2 config in the project with
`model_params.decoder.type: hifigan` — every other config uses `istftnet` — and
`train_second_v10.py:load_checkpoint()` does not exclude `decoder` from the modules loaded from the
ISTFTNet-shaped `first_stage_v3.pth`, with a loader guard that only raises on zero matched keys (not
partial mismatch). This makes a silent decoder-architecture mismatch (HiFi-GAN built fresh, then
only partially overwritten by ISTFTNet-shaped Stage 1 weights, leaving the vocoder proper
effectively randomly initialized after only 17 epochs) the leading hypothesis for v10 producing
noise. Proceed to step 7, `planning`: design the instrumented-harness build (per-module
missing/unexpected key logging, StyleTTS2-native `models.py` path — not `kokoro.KModel`), the cheap
checkpoint-tensor falsifier (check `net["decoder"]` key names for `ups.*`/`resblocks.*` HiFi-GAN
naming plus NaN/Inf and weight-norm per module) recommended to run before any inference code, the
control test against a known-good checkpoint, and the v10 diagnosis write-up. Read
`research/research_summary.md` for the full top-10 findings list before planning.
