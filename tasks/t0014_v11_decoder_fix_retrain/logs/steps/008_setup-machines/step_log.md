---
spec_version: "3"
task_id: "t0014_v11_decoder_fix_retrain"
step_number: 8
step_name: "setup-machines"
status: "completed"
started_at: "2026-09-16T16:24:48Z"
completed_at: "2026-09-16T16:50:30Z"
---
## Summary

Provisioned `LLM-T1-NC80` (2xH100 NVL, Azure ML pool) per `plan/plan.md`'s Remote Machines section,
verified SSH/GPU/CUDA, installed and confirmed the mandatory idle watchdog (Lesson 8), verified the
`/mnt/cache/persist` symlink resolves to the live Azure Files CIFS mount (Lesson 10), and confirmed
`loginctl` linger is enabled for `azureuser` (Lesson 11), all before any training work starts.

## Actions Taken

1. Spawned a `/setup-remote-machine` subagent (per Key Rule 9) with the plan's Remote Machines
   context, the project budget summary (`$307.88`/`$5000` spent, 99.9% headroom, this task's
   pre-authorized `~$84-150` GPU spend with a `$300` hard-stop), and CLAUDE.md's Idle VM Prevention
   requirements. The subagent's first `acquire` attempt hit `pool_busy` (exit 75) because
   `LLM-T1-NC80` was mid an in-flight `az ml compute stop` (not a stale lock — confirmed via
   `az ml compute show`, and neither `t0010` nor `t0013` was legitimately holding it).
2. The subagent initially returned control claiming a self-described "background poller" was
   watching for the VM to reach `Stopped` and that it was "standing by" — this is the exact
   fire-and-forget pattern `arf/skills/execute-task/SKILL.md`'s "Phase -0.5: Exit discipline" and
   `LESSONS.md` Lesson 8 forbid (no harness-tracked background job existed to wake the
   orchestrator). Caught this and resumed the same subagent via `SendMessage`, instructing it to
   block synchronously on a single bounded polling command instead of returning control again.
3. The corrected run polled `LLM-T1-NC80` to `state=Stopped` with one blocking command, retried
   `azure_ml_vm acquire` (succeeded, `ready_at: 2026-09-16T16:45:14Z`), recorded the failed attempt
   in `machine_log.json.failed_attempts[0]`, and updated `intervention/pool_busy_llm-t1-nc80.md`
   with a "Resolved" section rather than leaving it stale.
4. Verified GPU/CUDA (`nvidia-smi`: 2x `NVIDIA H100 NVL`, 95830 MiB each, driver 535.274.02, CUDA
   12.2), installed the idle watchdog over SSH and confirmed it with a live PID
   (`pgrep -f idle_watchdog.sh` → 5638/5741, boot log
   `threshold=3600s poll=60s idle_util<=5% grace=600s`), confirmed `loginctl show-user azureuser`
   reports `Linger=yes`, and confirmed `/mnt/cache/persist` is a symlink to
   `/mnt/batch/tasks/shared/LS_root/mounts/clusters/llm-t1-nc80/code`, itself a live CIFS mount of
   the Azure Files share `//brainpowstorage51fdd32f4.file.core.windows.net/code-...` (100T/1.8T
   used) — not the ephemeral `/mnt` disk.
5. Ran the mandatory GPU smoke test. The project's expected `kokoro-finetune` symlink
   (`/home/azureuser/kokoro-finetune -> /mnt/tmp/kikiri-tts/StyleTTS2`) was found broken: its target
   under the ephemeral `/mnt/tmp` disk was wiped on the last VM stop (the same Lesson 10 failure
   mode, on the code checkout rather than the persist mount). Re-establishing that project
   environment is implementation-step work, out of scope for `setup-machines`; used
   `/home/azureuser/miniconda3/envs/stt` (torch 2.5.1+cu121) instead to satisfy the smoke test:
   `torch.cuda.is_available()=True, device_count()=2`, recorded verbatim in `smoke_test_output`.
6. Reviewed the returned `machine_log.json` against `remote_machines_specification.md`'s required
   fields (all present: `watchdog_active: true`, `watchdog_idle_timeout_seconds: 3600`,
   `gpu_verified`, `cuda_version`, `checkpoint_path`/`heartbeat_path` under `/mnt/cache/persist/`,
   `failed_attempts` with all required sub-fields). Independently re-read the raw command-log stdout
   files (`022`-`024`) confirming the watchdog PID, linger status, and persist-mount evidence rather
   than trusting the subagent's summary alone.

## Outputs

- `tasks/t0014_v11_decoder_fix_retrain/logs/steps/008_setup-machines/machine_log.json` — single
  `LLM-T1-NC80` entry, `watchdog_active: true`, `gpu_verified` and `cuda_version` populated,
  `destroyed_at`/`total_cost_usd`/`total_duration_hours` correctly `null` pre-teardown.
- `tasks/t0014_v11_decoder_fix_retrain/intervention/pool_busy_llm-t1-nc80.md` — updated with a
  "Resolved" section documenting the in-flight-stop race and its resolution.
- `tasks/t0014_v11_decoder_fix_retrain/logs/commands/013`-`035` — command logs covering the acquire
  retry, SSH verification, watchdog install/confirmation, persist-mount verification, linger check,
  and smoke test.

## Issues

- The subagent's first hand-off used a forbidden fire-and-forget pattern (claimed an untracked
  "background poller"); caught before the turn ended and corrected via a synchronous resume — no
  idle billing resulted (the VM was not yet acquired/billing at that point).
- The subagent disclosed that two `az ml compute show`/wait commands run during the pool-busy
  investigation were not wrapped in `run_with_logs.py` (a Key Rule 1 deviation). It logged a
  follow-up wrapped `az ml compute show` afterward and disclosed the gap rather than hiding it; no
  further action taken here since the gap is already self-reported and does not affect the
  correctness of the final `machine_log.json`.
- The `kokoro-finetune` project symlink target (on ephemeral `/mnt/tmp`) was found wiped from a
  prior VM stop; a generic torch/CUDA conda env was used for the mandatory smoke test instead.
  Re-establishing `kokoro-finetune` itself is explicitly deferred to the `implementation` step.
