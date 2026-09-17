---
spec_version: "3"
task_id: "t0014_v11_decoder_fix_retrain"
step_number: 10
step_name: "teardown"
status: "completed"
started_at: "2026-09-16T23:09:18Z"
completed_at: "2026-09-16T23:14:00Z"
---
## Summary

Tore down `LLM-T1-NC80` (2xH100 NVL, Azure ML), which had been live and billing since
`2026-09-16T16:45:14Z` (roughly 6.4 hours) after step 9's training run finished all 50 epochs and
passed the mandatory audible-speech gate. Confirmed no job was still running, destroyed the
instance, and finalized cost tracking with the actual measured duration and cost rather than the
interim estimate step 9 recorded.

## Actions Taken

1. Verified independently over SSH that the `v11train` tmux session no longer exists on
   `LLM-T1-NC80` (training already finished cleanly per step 9's `DONE` marker) before authorizing
   any teardown work.
2. Spawned a dedicated subagent (per Rule 9) to execute the Teardown Protocol from
   `arf/skills/setup-remote-machine/SKILL.md`: re-confirmed the job was finished, checked whether
   any further downloads were needed (none — the `kokoro-v11-best` model asset and all evidence
   files were already pulled down and verified in step 9), and ran
   `azure_ml_vm teardown t0014_v11_decoder_fix_retrain --vm-name LLM-T1-NC80 --acquired-at 2026-09-16T16:45:14.004092Z`
   wrapped in `run_with_logs`.
3. The subagent also proactively checked VM disk usage before stop (per this step's "watching disk
   usage proactively" instruction) and removed this task's own throwaway scratch venv at
   `/mnt/tmp/t0014_venv` (5.8 GB, fully regenerable, built there in step 9 due to a CIFS-mount write
   stall) — left the pre-existing unrelated pool clutter on `/` untouched since it predates and is
   out of scope for this task.
4. Updated `logs/steps/008_setup-machines/machine_log.json` with the final `destroyed_at`
   (`2026-09-16T23:11:25.308664Z`), `total_duration_hours` (6.436h), and `total_cost_usd` ($89.85);
   also corrected the `provider` field from the non-spec `"azure-ml"` to the spec-enum `"azure_ml"`
   value (`arf/specifications/remote_machines_specification.md` `RM-E007` / `_KNOWN_PROVIDERS`).
5. Rewrote `results/remote_machines_used.json` and `results/costs.json` to replace step 9's interim
   figures (~6.19h / $86.46) with the final measured totals (6.44h / $89.85), removing the "INTERIM"
   note.
6. Independently re-ran
   `uv run python -m arf.scripts.verificators.verify_machines_destroyed t0014_v11_decoder_fix_retrain`
   myself (not just trusting the subagent's report) — confirmed **PASSED**, 0 errors, 2 warnings
   (`RM-W007` legacy `spec_version`, consistent with sibling tasks; `RM-W001` Azure ML API
   unreachable from this sandbox for a live cross-check, expected/non-blocking).
7. Independently re-checked `machine_log.json` has a non-null `destroyed_at`, and that
   `remote_machines_used.json` / `costs.json` cross-check consistently (`instance_id` ==
   `machine_id`, `total_cost_usd` within tolerance of `cost_usd`).

## Outputs

* `tasks/t0014_v11_decoder_fix_retrain/logs/steps/008_setup-machines/machine_log.json` (updated:
  `destroyed_at`, `total_duration_hours`, `total_cost_usd`, corrected `provider`)
* `tasks/t0014_v11_decoder_fix_retrain/results/remote_machines_used.json` (final duration/cost)
* `tasks/t0014_v11_decoder_fix_retrain/results/costs.json` (final total, $89.85)
* `tasks/t0014_v11_decoder_fix_retrain/logs/steps/010_teardown/step_log.md` (this file)
* `tasks/t0014_v11_decoder_fix_retrain/logs/commands/*` — command logs from the teardown subagent's
  `run_with_logs`-wrapped SSH/`azure_ml_vm` calls

## Issues

No issues encountered. `LLM-T1-NC80` is confirmed stopped (`deallocated: true`,
`other_locks_present: false`); no other task held a competing lock on this VM.
