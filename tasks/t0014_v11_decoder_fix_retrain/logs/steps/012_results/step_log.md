---
spec_version: "3"
task_id: "t0014_v11_decoder_fix_retrain"
step_number: 12
step_name: "results"
status: "completed"
started_at: "2026-09-16T23:24:02Z"
completed_at: "2026-09-16T23:40:00Z"
---
## Summary

Finalized the `results` step. `results/results_summary.md`, `results/results_detailed.md`,
`results/metrics.json`, `results/costs.json`, and `results/remote_machines_used.json` already
existed as by-products of step 9 (`implementation`) writing Milestone C/D deliverables; this step
re-read them against the full `task_results_specification.md` requirements, corrected the one stale
fact (REQ-8/cost section still described the VM as "still running" with an interim ~$86.46 figure,
predating step 10's teardown), and added the two required charts for this experiment-type task.

## Actions Taken

1. Read `arf/skills/execute-task/SKILL.md`'s Phase 5 "Step: `results`" section and
   `arf/specifications/task_results_specification.md` / `logs_specification.md` in full, then
   re-read `task.json` and `plan/plan.md`'s Task Requirement Checklist before touching any file.
2. Reviewed the existing `results/results_summary.md` and `results/results_detailed.md` (written
   during step 9) against the spec's mandatory sections, word counts, `## Examples` requirement
   (confirmed `tts-finetuning-eval`'s `description.json` sets `requires_result_examples: true`, so
   this is an experiment-type task), and the `## Task Requirement Coverage` final-section rule — all
   already satisfied except REQ-8 and the `## Cost` section, which still reflected step 9's interim
   (pre-teardown) cost figure.
3. Cross-checked `results/metrics.json` (`rtf=3.1811602714837277`,
   `speaker_sim=0.44419100880622864`) against every number quoted in both markdown files, and
   confirmed both metric keys are registered
   (`uv run -m arf.scripts.aggregators.aggregate_metrics --format ids` lists `rtf`, `speaker_sim`,
   `ttfb_ms`; `ttfb_ms` is correctly and explicitly omitted with a stated reason).
4. Updated `results_summary.md`'s `## Cost` section and `results_detailed.md`'s `## Methodology`
   (added total runtime/timestamps) and REQ-8 row in `## Task Requirement Coverage` to reflect step
   10's final, authoritative figures ($89.85 over 6.436 hours, `destroyed_at`
   `2026-09-16T23:11:25Z`) instead of the superseded interim estimate.
5. Generated two charts into `results/images/` (previously empty) per the "Charts" requirement for
   experiment tasks: `val_loss_by_epoch.png` (per-epoch `val_loss` from
   `data/run_v11/checkpoints.json`, joint-epoch transition and best-epoch marked) and
   `audible_gate_comparison.png` (log-scale `clip_fraction` bar chart comparing the control, both
   confirmed-broken v10 checkpoints, and v11 against the `is_likely_noise` gate threshold).
   Consulted the `dataviz` skill for color/mark guidance before drawing them. Embedded both under a
   new `## Visualizations` section in `results_detailed.md` with descriptions of what each shows.
6. Ran the verificators: `verify_task_results` (PASSED, 0 errors/0 warnings) and
   `verify_task_metrics` (PASSED, 0 errors/0 warnings), each wrapped in `run_with_logs`.

## Outputs

* `results/results_summary.md` — `## Cost` section updated to the final, post-teardown figures.
* `results/results_detailed.md` — `## Methodology` runtime/timestamps added, new `## Visualizations`
  section, REQ-8 updated to `Done`, `## Files Created` updated.
* `results/images/val_loss_by_epoch.png`, `results/images/audible_gate_comparison.png` — new charts.
* `results/metrics.json`, `results/costs.json`, `results/remote_machines_used.json` — confirmed
  already final from steps 9/10; no edits needed.

## Issues

No issues encountered. `results/costs.json` and `results/remote_machines_used.json` needed no edits
— step 10 (`teardown`) had already finalized them with the authoritative $89.85/6.436-hour figures;
only the markdown prose in `results_summary.md`/`results_detailed.md` still referenced the
pre-teardown interim estimate and has now been corrected.
