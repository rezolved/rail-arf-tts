---
spec_version: "3"
task_id: "t0010_stage2_safeguarded_training"
step_number: 10
step_name: "teardown"
status: "completed"
started_at: "2026-09-16T06:57:58Z"
completed_at: "2026-09-16T07:20:00Z"
---
# Step 10 — teardown

## Summary

LLM-T1-NC80 stopped at 2026-09-16T07:10:15Z after 19.54 hours ($272.78). Harness eval was attempted
but could not run: root disk at 100% (no torch env available), and Azure ML refuses `stop` when disk
is full; cleared 2.9 GB `~/.cache/whisper` to unblock the stop API. Machine lock released via
`azure_ml_vm teardown` (lock cleared via SSH before VM was stopped). Results files written;
`verify_machines_destroyed` passes with 0 errors, 4 warnings.

## Actions Taken

1. Ran `prestep t0010_stage2_safeguarded_training teardown`.
2. Checked VM: root disk still 100% full despite context claiming 24 GB freed (context was stale;
   `/dev/root` 0 bytes free, `/mnt` ephemeral has 24 GB free).
3. Attempted harness eval: Python torch import fails (`ncclCommResume` undefined symbol in default
   env); no working torch environment found on root or `/mnt`; resemblyzer venv at
   `~/resemblyzer-venv` not present. Eval cannot proceed.
4. Ran `azure_ml_vm teardown t0010_stage2_safeguarded_training --vm-name LLM-T1-NC80` — lock
   cleared, `destroyed_at` = 2026-09-16T07:10:15Z, duration 19.54 h, cost $272.78.
5. VM still showed `Running` after teardown (Azure ML `stop` failed with 400; error: "StopCompute is
   not allowed when ComputeInstance disk is full").
6. Cleared `~/.cache/whisper` on VM (2.9 GB) via SSH to free root disk. Root disk at 98%.
7. Issued `az ml compute stop` — VM entered `Updating` then `Succeeded` (SSH no longer reachable; VM
   confirmed stopped).
8. Updated `machine_log.json`: set `destroyed_at`, `total_duration_hours`, `total_cost_usd`, fixed
   `provider` to `azure_ml` and `spec_version` to `6`, added Azure ML sentinel fields (`offer_id`,
   `search_criteria`, `image`).
9. Updated `intervention/eval_deferred_disk_full.md` with teardown outcome.
10. Wrote `results/costs.json` ($272.78 total; breakdown: azure-ml-2xh100).
11. Wrote `results/remote_machines_used.json` (LLM-T1-NC80, 2xH100, 19.54 h, $272.78).
12. Ran `verify_machines_destroyed`: 0 errors, 4 warnings (RM-W001 API unreachable, RM-W002 cost
    > 50% over plan estimate, RM-W003 >12h runtime, RM-W005 failed_attempts[0] missing offer_id).

## Outputs

- `tasks/t0010_stage2_safeguarded_training/logs/steps/008_setup-machines/machine_log.json` — updated
  with `destroyed_at`, `total_duration_hours`, `total_cost_usd`, provider fix
- `tasks/t0010_stage2_safeguarded_training/results/costs.json` — total $272.78
- `tasks/t0010_stage2_safeguarded_training/results/remote_machines_used.json` — LLM-T1-NC80 entry
- `tasks/t0010_stage2_safeguarded_training/intervention/eval_deferred_disk_full.md` — updated
  status: VM stopped, eval not completed

## Issues

- **Harness eval not completed**: Root disk was 100% full and no working torch Python environment
  was available on the VM. `speaker_sim`, `ttfb_ms`, and `rtf` remain null for all epoch variants.
  These metrics must be obtained by running the eval harness locally or in a follow-up task once the
  Kokoro environment is set up.
- **Azure ML stop blocked by full disk**: Had to clear `~/.cache/whisper` (2.9 GB) before the Azure
  ML stop API would accept the request.
- **Cost over plan estimate**: VM ran 19.54 h vs plan estimate of ~7 h; training completed at ~7 h
  but step 9 started late (next day) and harness eval deferred; stop was also delayed by the disk
  issue. Total cost $272.78 vs plan estimate $100.
