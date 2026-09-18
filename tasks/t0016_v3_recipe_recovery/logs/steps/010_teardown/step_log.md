---
spec_version: "3"
task_id: "t0016_v3_recipe_recovery"
step_number: 10
step_name: "teardown"
status: "completed"
started_at: "2026-09-17T15:57:29Z"
completed_at: "2026-09-17T16:04:32Z"
---
## Summary

`LLM-T1-NC80` was already deallocated and this task's lock already cleared by a teardown call the
implementation step (step 9) made itself, so this step verified and reconciled that state rather
than issuing a fresh stop. The implementation step's own `azure_ml_vm teardown` call omitted the
billing anchor flags, so it recorded a bogus `duration_hours: 0.0` / `total_cost_usd: 0.0`; this
step recomputed the real figures (~13.47 min, ~$3.13) from Azure's own activity log against the
correct billing window and corrected `machine_log.json` and the results files.

## Actions Taken

1. Spawned a dedicated subagent (per Critical Rule 9) to execute the Teardown Protocol from
   `arf/skills/setup-remote-machine/SKILL.md` in verify/reconcile mode: confirm the VM state
   read-only, recompute cost attribution against the true billing window without re-invoking
   `azure_ml_vm teardown` (which would have restarted the VM and computed cost through the current
   wall-clock time instead of the already-recorded `destroyed_at`).
2. The subagent's `az ml compute show` check found `LLM-T1-NC80` in state `"Running"` — on its face
   the runbook's "still up, treat as emergency" case. Before re-triggering anything, it investigated
   further: an SSH lock listing showed a live
   `~/.arf-locks/t0018_zero_shot_cloning_calibration.lock` (acquired `2026-09-17T15:26:48Z`), and
   Azure's activity log showed the VM was stopped by t0016's own teardown at `14:48:57Z` and later
   restarted by `t0018_zero_shot_cloning_calibration` (a later task sharing this project's sole pool
   VM), not left running by a failed t0016 teardown. This is the correct read of the evidence:
   t0016's own teardown genuinely succeeded, and the VM being `Running` now is t0018 legitimately
   using the shared pool afterward. Re-running teardown with `--vm-name` as the runbook's literal
   contingency wording suggests would have pulled the VM out from under t0018's active work — the
   subagent correctly did not do this, and this step-executor independently reviewed the underlying
   evidence (activity-log timestamps and the competing lock file) and agrees with that conclusion.
3. Ran a read-only reconciliation script (via `run_with_logs`, not a task deliverable — kept out of
   the task folder) calling this project's own `get_power_events` and `compute_cost_attribution`
   helpers (`arf/scripts/utils/azure_ml_vm.py`) for the real billing window
   `2026-09-17T14:34:28.703377Z` &#8594; `2026-09-17T14:47:56.755481Z` (the window
   `machine_log.json` already recorded at acquire time and the `destroyed_at` the original teardown
   call already produced). Result: `status: "verified"`, `method: "activity_log"`, 1 fully-billable
   segment, `total_duration_hours = 0.22445891777777777` (approximately 13.47 min),
   `total_cost_usd = 3.133446492177778` (approximately $3.13 of the $21 VM sub-cap) — matching the
   ballpark the implementation step's own inventory already estimated.
4. Wrote the reconciled `destroyed_at`, `total_duration_hours`, `total_cost_usd`,
   `billing_ended_at`, `cost_attribution_method`/`status`/`notes`, and `billable_segments` into
   `logs/steps/008_setup-machines/machine_log.json`, and corrected its `provider` field from the
   non-spec `"azure-ml"` to the spec-enum `"azure_ml"` (matching the `t0014_v11_decoder_fix_retrain`
   precedent; `arf/scripts/verificators/verify_machines_destroyed.py`'s `_KNOWN_PROVIDERS` does not
   actually recognize the hyphenated form despite a code comment claiming it is aliased).
5. Wrote `results/remote_machines_used.json` and `results/costs.json` (neither previously existed)
   with the reconciled duration/cost figures, following
   `arf/specifications/remote_machines_specification.md`'s Cost Integration schema
   (`azure-ml-2xh100` breakdown key).
6. Independently re-ran
   `uv run python -m arf.scripts.verificators.verify_machines_destroyed t0016_v3_recipe_recovery`
   myself (not just trusting the subagent's report) — confirmed **PASSED**, 0 errors, 2 warnings
   (`RM-W007` legacy `spec_version`, `RM-W001` Azure ML API unreachable from this sandbox for a live
   cross-check — both consistent with the `t0014` precedent and non-blocking).

## Outputs

- `tasks/t0016_v3_recipe_recovery/logs/steps/008_setup-machines/machine_log.json` (updated:
  `destroyed_at`, `total_duration_hours`, `total_cost_usd`, billing/cost-attribution fields,
  corrected `provider`)
- `tasks/t0016_v3_recipe_recovery/results/remote_machines_used.json` (new)
- `tasks/t0016_v3_recipe_recovery/results/costs.json` (new)
- `tasks/t0016_v3_recipe_recovery/logs/commands/047_*` through `050_*` — `az ml compute show`, the
  SSH lock-file check, the cost-reconciliation script run, and the `verify_machines_destroyed` run
- `tasks/t0016_v3_recipe_recovery/logs/steps/010_teardown/step_log.md` (this file)

## Issues

`LLM-T1-NC80` reads as `Running` at the time of this step because a later task
(`t0018_zero_shot_cloning_calibration`) legitimately re-acquired the shared pool VM after t0016
released it — not because t0016's own teardown failed. This is documented above and in the evidence
logs rather than treated as this task's problem; no action was taken against t0018's live lock.
Separately, the implementation step's own `azure_ml_vm teardown` call (step 9) omitted
`--billing-started-at`/`--billing-anchor`, producing an incorrect `$0.00` cost figure in its own
stdout; this is a real gap in that call site (missing flags), reconciled here from the authoritative
timestamps already on record rather than by re-running teardown.
