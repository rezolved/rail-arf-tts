---
spec_version: "3"
task_id: "t0010_stage2_safeguarded_training"
step_number: 7
step_name: "planning"
status: "completed"
started_at: "2026-09-15T10:48:35Z"
completed_at: "2026-09-15T11:05:00Z"
---
# Step 7 — Planning

## Summary

Produced `plan/plan.md` (spec_version "2", 12 REQ items, 12 numbered steps across 5 milestones) for
the safeguarded Kokoro Stage 2 training run with joint_epoch=8. The plan explicitly addresses the
critical CheckpointManager bug (REQ-3: missing `joint_epoch` arg at line 403 of the copied training
script), the batch evaluation strategy, three registered metrics (speaker_sim, ttfb_ms, rtf) in
explicit variant format, and pre-registered rejection criteria. Verificator passed with 0 errors and
0 warnings.

## Actions Taken

1. Read all required planning specs: `plan_specification.md`, `project_budget_specification.md`,
   `logs_specification.md`, `planning/SKILL.md`, `tts-finetuning-eval` task type instruction.
2. Read dependency summaries: `t0008_tts_eval_harness_baselines/results/results_summary.md` and
   `t0009_stage2_training_failure_forensics/results/results_summary.md`.
3. Read `tasks/t0010_stage2_safeguarded_training/research/research_summary.md` and
   `task_description.md` to extract all 12 concrete requirements.
4. Verified the CheckpointManager bug by inspecting
   `tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py` line 44 —
   `joint_epoch` is a required positional argument in `__init__`.
5. Verified v6c config at
   `tasks/t0009_stage2_training_failure_forensics/data/configs/t0006_run03_v6c.yml` to confirm
   `diff_epoch` is not in the archived config (must be added to v10 config explicitly).
6. Reviewed registered metrics cache (`ctx/metrics.json`) — 3 applicable metrics: `speaker_sim`,
   `ttfb_ms`, `rtf`.
7. Wrote `tasks/t0010_stage2_safeguarded_training/plan/plan.md` with all 11 mandatory sections plus
   `## Rejection Criteria`.
8. Ran `uv run flowmark --inplace --nobackup tasks/t0010_stage2_safeguarded_training/plan/plan.md`.
9. Ran `uv run python -u -m arf.scripts.verificators.verify_plan t0010_stage2_safeguarded_training`
   — PASSED, 0 errors, 0 warnings.

## Outputs

* `tasks/t0010_stage2_safeguarded_training/plan/plan.md` — verified plan, 12 REQ items, 12 steps

## Issues

No issues encountered. Verificator passed on first run after converting bold `**N.**` step numbers
to standard `N.` numbered list format required by `PL-E006`.
