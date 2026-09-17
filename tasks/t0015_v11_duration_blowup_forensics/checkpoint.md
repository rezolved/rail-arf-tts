---
spec_version: "1"
task_id: "t0015_v11_duration_blowup_forensics"
updated_at: "2026-09-17T08:20:00Z"
completed_steps: 10
next_step_number: 9
next_step_id: "implementation"
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

### Step 7 — planning

Wrote `plan/plan.md` (11 mandatory sections plus an added `## Rejection Criteria` section);
verificator passed with 0 errors and 0 warnings on the first attempt. The plan defines 14 numbered,
milestone-grouped steps (env setup, instrument `synthesize()` for `pred_dur`/frame-count logging, a
10-text varied characterization sweep, localization analysis, tensor-level forensics, an
alpha/beta/diffusion_steps/embedding_scale parameter sweep with pre-registered pass criteria, a
conditional cheap-fix-or-documented-negative-result branch, gate hardening with two new signals
(duration-sanity, silence-gap), an ASR round-trip evaluation via faster-whisper, a three-way
v10/v11/corrected regression check, `results/metrics.json` in explicit multi-variant format, and the
canonical `results/duration_blowup_diagnosis.md`), with 3 steps marked `[CRITICAL]`. Cost estimate
is $0.00 (CPU-only, no paid APIs, no GPU); estimated implementation time ~3.5-5.5 hours. The plan
corrects a `task_description.md` assumption during research: the ElevenLabs David reference corpus
actually lives at `tasks/t0008_tts_eval_harness_baselines/data/11labs_david`, not a top-level
`data/11labs_david`, and notes that DVC-tracked checkpoint/reference data has not yet been pulled in
this worktree — `dvc pull` is an explicit early step.

* * *

## Cross-Step Decisions

* The v11 duration blowup is NOT caused by frozen predictor/predictor_encoder modules during t0014's
  finetune — gradients were applied every epoch. The leading hypothesis going into planning is a
  duration-predictor calibration failure evidenced by the stalled `dur_loss` (0.53-0.62 vs. a 0.034
  reference), which should steer the planning step's localization approach toward `pred_dur`/frame
  ratio instrumentation rather than a plumbing/freeze-bug search.

* No GPU is used anywhere in the plan: all 14 implementation steps run CPU-only. Estimated cost is
  $0.00. Any escalation to targeted `predictor`/`predictor_encoder` fine-tuning is explicitly
  deferred to a follow-up task, per `task_description.md`'s scope and the `setup-machines` skip
  rationale from step 8.

* The ElevenLabs David reference corpus path used throughout downstream steps is
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david` (confirmed during planning research),
  not the top-level `data/11labs_david` implied by `CLAUDE.md`'s benchmark description —
  implementation must use the t0008 path and run `dvc pull` early since the checkpoint and reference
  audio are DVC-tracked and not yet present in this worktree.

* * *

## Next Step Notes

Step 7 (`planning`) completed: `plan/plan.md` is ready and the plan verificator passed with 0 errors
and 0 warnings. Step 8 (`setup-machines`) is already marked `skipped` in `step_tracker.json`, so the
next pending step is step 9 (`implementation`). The implementation subagent should follow
`plan/plan.md`'s 14 numbered steps in order, starting with environment setup and `dvc pull`, then
instrumenting `infer_styletts2.py`'s `synthesize()` for `pred_dur`/`pred_aln_trg`/frame-count
logging per the exact insertion point from `research/research_code.md` (t0014 lines 282-371). Steps
1 (CPU inference environment build), 4 (`pred_dur`/frame-count instrumentation), 12 (three-way gate
regression), and 14 (`duration_blowup_diagnosis.md`) are marked `[CRITICAL]` in the plan — if any of
those become blocked, the implementation agent must write an intervention file rather than silently
substitute a different approach. The plan's Step by Step ends at `results/metrics.json` and
`results/duration_blowup_diagnosis.md`; results_summary.md/results_detailed.md/suggestions/
compare-literature remain orchestrator-owned steps, not part of implementation.
