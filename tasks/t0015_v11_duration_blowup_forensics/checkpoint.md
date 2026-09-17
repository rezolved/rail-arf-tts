---
spec_version: "1"
task_id: "t0015_v11_duration_blowup_forensics"
updated_at: "2026-09-17T08:10:00Z"
completed_steps: 9
next_step_number: 7
next_step_id: "planning"
---
# Task Objective

Root-cause why kokoro-v11-best's synthesis runs 15-20x too long and sounds like a droning babble to
a human, despite passing the clipping/flatness noise gate, then close that gate's blind spot.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0015_v11_duration_blowup_forensics` created. Initial folder structure initialized in
`tasks/t0015_v11_duration_blowup_forensics/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` (via prestep and again via `run_with_logs`) against both
dependencies: `t0013_v10_synthesis_quality_forensics` and `t0014_v11_decoder_fix_retrain`. Both have
`status: "completed"` in their `task.json`, so the check passed with 0 errors and 0 warnings. Result
recorded in `logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory folder structure (`plan/`, `research/`, `results/`,
`results/images/`, `corrections/`, `intervention/`, `code/`,
`logs/commands|searches|sessions|steps/`, `assets/`), each with `.gitkeep`; `expected_assets` is
`{}` so no asset-type subdirectory was added. Populated the aggregator cache at
`tasks/t0015_v11_duration_blowup_forensics/ctx/` (task_types.json, costs.json, tasks.json,
metrics.json, suggestions.json) for downstream subagents to reuse; `ctx/` is gitignored and not
committed.

### Step 4 — research-papers

Skipped: this is an empirical debugging/forensics task about this project's own checkpoints and
code, and no published-paper evidence bears on the specific duration-blowup root cause. StyleTTS2
background is already present in the corpus from prior dependency tasks.

### Step 5 — research-internet

Skipped: `task_description.md` already specifies the exact reproduction recipe, the parameters to
sweep, and the ASR library (faster-whisper) already vendored in t0008's environment, so no new
external research is needed.

### Step 8 — setup-machines

Skipped: `task_description.md` states no GPU is required for the diagnostic, localization, and
inference-parameter-sweep work; GPU-based predictor retraining is explicitly scoped out as a
follow-up task.

### Step 10 — teardown

Skipped: no remote machine was provisioned (`setup-machines` was skipped), so there is nothing to
tear down.

### Step 13 — compare-literature

Skipped: this task produces internal forensic evidence about this project's own checkpoint and code,
not quantitative results comparable to a published baseline.

### Step 6 — research-code

Wrote `research/research_code.md` (verificator: 0 errors, 0 warnings) documenting the reusable
inference/diagnostic code and hard evidence on the predictor-gradient question. Key finding: t0014's
`train_second_v11.py` shows `predictor`/`predictor_encoder` DID receive unconditional
`optimizer.step()` calls for all 50 epochs (lines 733-734), refuting the "rode along frozen"
hypothesis — but `data/run_v11/metrics.jsonl` shows `dur_loss` plateaued at 0.53-0.62 the entire run
versus t0009's reference run converging to 0.034, pointing instead at a predictor-calibration
failure. Also identified the exact instrumentation point in `infer_styletts2.py`'s `synthesize()`
(t0014, lines 282-371) for logging `pred_dur`/`pred_aln_trg`, plus reusable diagnostic scripts
(`inspect_checkpoint.py`, `random_decoder_probe.py`, `build_reference_concat.py`) and varied-text
prompt sets from t0008, all labeled "copy into task" (no library registration). Since this was the
only research step executed, also ran `/research-summarize` to produce
`research/research_summary.md` for downstream subagents.

* * *

## Cross-Step Decisions

* The v11 duration blowup is NOT caused by frozen predictor/predictor_encoder modules during t0014's
  finetune — gradients were applied every epoch. The leading hypothesis going into planning is a
  duration-predictor calibration failure evidenced by the stalled `dur_loss` (0.53-0.62 vs. a 0.034
  reference), which should steer the planning step's localization approach toward `pred_dur`/frame
  ratio instrumentation rather than a plumbing/freeze-bug search.

* * *

## Next Step Notes

Step 6 (`research-code`) completed: `research/research_code.md` and `research/research_summary.md`
are ready, verificator passed. Proceed to step 7 (`planning`). The plan should center the
localization step on `pred_dur`/`pred_aln_trg` instrumentation (exact insertion point identified in
`infer_styletts2.py`'s `synthesize()`, t0014 code, lines 282-371) and should treat the stalled
`dur_loss` (0.53-0.62 plateau vs. t0009's 0.034 reference) as the primary root-cause lead, not a
frozen-module bug. Reuse the diagnostic scripts and varied-text prompt sets cataloged in
`research/research_code.md`'s Reusable Code and Assets section (all "copy into task").
