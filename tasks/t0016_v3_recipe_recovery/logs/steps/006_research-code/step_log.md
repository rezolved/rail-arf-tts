---
spec_version: "3"
task_id: "t0016_v3_recipe_recovery"
step_number: 6
step_name: "research-code"
status: "completed"
started_at: "2026-09-17T13:53:36Z"
completed_at: "2026-09-17T13:59:00Z"
---
## Summary

Reviewed prior task code, results, and answer/library aggregators to inform the v3 recipe
reconstruction, then wrote `research/research_code.md` covering the four tasks `task_description.md`
names explicitly plus every other completed task the aggregators surfaced as relevant.

## Actions Taken

1. Spawned a dedicated subagent (per Critical Rule 9) to execute the `/research-code` skill for
   `t0016_v3_recipe_recovery`, per `arf/skills/research-code/SKILL.md`.
2. The subagent ran `aggregate_libraries`, `aggregate_answers`, and
   `aggregate_tasks --status completed`, found 2 libraries and 15 completed tasks, then read actual
   source files (not just descriptions) for `t0006` (`config_david_v6c_stage2.yml`,
   `train_second_patch.diff`, `best/config.json`), `t0002` (`extract_decoder_generic.py`), `t0009`
   (`confound_table.md`, `checkpoint_manager.py`, `health_gates.py`, `build_confound_table.py`,
   `build_inventory.py`, `collect_configs.py`), `t0015` (`predictor_tensor_forensics.py`,
   `audio_quality_check.py`), and results summaries for `t0001`, `t0003`, `t0005`, `t0008`, `t0010`,
   `t0013`, `t0014`.
3. The subagent wrote `tasks/t0016_v3_recipe_recovery/research/research_code.md` (484 lines, 13
   tasks cited, 2 libraries documented) and ran the verificator, which passed on the first attempt.
4. Re-ran `verify_research_code` via `run_with_logs` from the orchestrator for the log record;
   confirmed PASSED with 0 errors and 0 warnings.

## Outputs

* `tasks/t0016_v3_recipe_recovery/research/research_code.md`
* `tasks/t0016_v3_recipe_recovery/logs/commands/` — command log for the `run_with_logs`-wrapped
  verificator run.

## Issues

No issues encountered. The subagent flagged a genuine three-way `multispeaker` contradiction across
`best/config.json` (true), `t0009`'s confound table (true, assumed), and `t0006`'s
`config_david_v6c_stage2.yml` (false, with an inline comment claiming the Stage 1 checkpoint is
`multispeaker: false`) — none of the three sources carries a SHA-256, so this is documented in Key
Findings/Recommendations as resolvable only by checkpoint-shape forensics, feeding directly into the
implementation step.
