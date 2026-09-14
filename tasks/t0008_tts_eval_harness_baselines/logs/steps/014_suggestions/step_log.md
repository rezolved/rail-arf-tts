---
spec_version: "3"
task_id: "t0008_tts_eval_harness_baselines"
step_number: 14
step_name: "suggestions"
status: "completed"
started_at: "2026-09-14T18:03:46Z"
completed_at: "2026-09-14T18:25:00Z"
---
## Summary

Generated 5 follow-up task suggestions by synthesizing t0008 results: no Kokoro system reaches the
0.85 GE2E cosine target (best is v3_bundle at 0.631), all healthy Kokoro systems meet TTFB ≤300ms on
fillers, and t0006_v6d shows a concerning speaker_sim drop from fillers to val96. The suggestions
prioritize continued training, root-cause investigation of the val96 degradation, and corpus
expansion as the highest-priority paths toward the project success criterion.

## Actions Taken

1. Read `project/description.md`, `task.json`, `task_description.md`, `plan/plan.md`,
   `results/results_summary.md`, `results/results_detailed.md`, `results/metrics.json`,
   `logs/steps/009_implementation/step_log.md`, and `logs/steps/012_results/step_log.md` to gather
   full task context.
2. Read `arf/specifications/suggestions_specification.md` for file format requirements.
3. Ran `aggregate_task_types` to get available task type slugs.
4. Ran `aggregate_suggestions --uncovered` — returned 0 existing suggestions; no deduplication
   needed.
5. Ran `aggregate_tasks --detail short` — confirmed t0009 covers training forensics (explosion root
   causes), so that was excluded from candidates.
6. Ran `aggregate_categories` — confirmed no categories exist in this project (empty list); all
   suggestions use `categories: []`.
7. Generated 5 candidate suggestions covering the main research gaps: training continuation, val96
   degradation investigation, corpus expansion, WER metric registration, and prompt corpus
   augmentation.
8. Wrote `results/suggestions.json` with spec_version "1", 5 suggestions (S-0008-01 to S-0008-05).
9. Ran `verify_suggestions t0008_tts_eval_harness_baselines` — PASSED (0 errors, 0 warnings).

## Outputs

- `tasks/t0008_tts_eval_harness_baselines/results/suggestions.json` — 5 suggestions, verificator
  PASSED

## Issues

No issues encountered. The project has no registered categories, so all suggestions use `[]` for the
categories field — this is valid per the specification.
