---
spec_version: "3"
task_id: "t0015_v11_duration_blowup_forensics"
step_number: 12
step_name: "results"
status: "completed"
started_at: "2026-09-17T10:41:03Z"
completed_at: "2026-09-17T11:20:00Z"
---
## Summary

Wrote `results/results_summary.md` and `results/results_detailed.md` (spec_version "2") summarizing
Step 9's forensic findings and Step 11's creative-thinking findings, added the two missing
`results/costs.json` and `results/remote_machines_used.json` files, generated two new charts in
`results/images/` from already-produced JSON data, and re-verified `verify_task_metrics` and
`verify_task_results` both pass clean.

## Actions Taken

1. Read `arf/skills/execute-task/SKILL.md`'s Part B step-executor protocol and the `results` step's
   detailed requirements in Phase 5, plus `arf/specifications/task_results_specification.md` and
   `arf/specifications/logs_specification.md` (the two specs listed for this step in the Per-Step
   Spec Table), and `.claude/rules/task-documents.md`.
2. Re-read `task.json`, `plan/plan.md`'s Task Requirement Checklist and Rejection Criteria, and
   every results file already produced by Step 9 (`duration_blowup_diagnosis.md`,
   `duration_characterization.json`, `param_sweep.json`, `predictor_tensor_forensics.md`,
   `asr_roundtrip_evaluation.md`, `gate_regression.json`, `metrics.json`, `speaker_sim_scores.json`,
   `localization_summary.json`) plus Step 11's `logs/steps/011_creative-thinking/step_log.md`, to
   ground every claim in this step's markdown in existing evidence rather than restating hypotheses.
3. Found `results/costs.json` and `results/remote_machines_used.json` did not exist (Step 9 did not
   produce them, since they are results-step-owned per the spec) and wrote them:
   `{"total_cost_usd": 0, "breakdown": {}}` and `[]`, matching the plan's $0.00 CPU-only cost
   estimation.
4. Found `results/images/` contained no charts (only `.gitkeep`). Since this task's `task_types`
   include `code-reproduction` (an `EXPERIMENT_TASK_TYPES` entry per
   `arf/scripts/verificators/verify_task_results.py`) and the task has rich existing quantitative
   data, generated two charts via a `run_with_logs`-wrapped matplotlib script reading directly from
   `results/duration_characterization.json` and `results/param_sweep.json`: a horizontal bar chart
   of `duration_ratio` per characterization text (colored by short/long category) and a bar chart of
   the 13-combination parameter sweep against the pass threshold.
5. Wrote `results/results_summary.md` (Summary, Metrics with 6 bullets of specific numbers,
   Verification) and `results/results_detailed.md` (`spec_version: "2"`; Summary, Methodology,
   Metrics Tables, Comparison vs Baselines, Visualizations, Analysis, Examples with 11 concrete
   input/output instances in fenced code blocks, Limitations, Verification, Files Created, and Task
   Requirement Coverage as the final section covering all 11 `REQ-*` items with exact evidence
   paths). The Analysis section documents the plan's own contradicted assumption: the
   inference-parameter sweep was framed as the cheapest rung but did not come close to working (best
   `duration_ratio=5.54` vs. the `<=3.0` bar).
6. Ran `uv run flowmark --inplace --nobackup` on both new markdown files.
7. Re-ran
   `uv run python -u -m arf.scripts.verificators.verify_task_metrics t0015_v11_duration_blowup_forensics`
   (PASSED, no errors/warnings) and
   `uv run python -u -m arf.scripts.verificators.verify_task_results t0015_v11_duration_blowup_forensics`
   (PASSED, no errors/warnings), both `run_with_logs`-wrapped.
8. Updated `checkpoint.md`'s Step History, Cross-Step Decisions, Next Step Notes, and frontmatter.

## Outputs

- `results/results_summary.md`
- `results/results_detailed.md`
- `results/costs.json`
- `results/remote_machines_used.json`
- `results/images/duration_ratio_by_text.png`
- `results/images/param_sweep_duration_ratio.png`
- `logs/steps/012_results/step_log.md` (this file)

## Issues

`results/costs.json` and `results/remote_machines_used.json` were missing after Step 9 (the
implementation step correctly did not produce them, since they are results-step-owned deliverables
per `arf/specifications/task_results_specification.md`); both were written this step with the
correct zero-cost/no-remote-machines content, matching the plan's Cost Estimation and Remote
Machines sections. No other issues encountered.
