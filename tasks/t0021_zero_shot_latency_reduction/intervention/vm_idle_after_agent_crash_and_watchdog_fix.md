# VM went idle after an agent crash mid-implementation; watchdog re-armed with a fixed TERMINATE_CMD

## What happened

During the `implementation` step (step 9), the subagent driving the acceleration-variant sweep was
terminated mid-turn by an API rate-limit error (session limit) right after the CosyVoice2
`baseline_new_ref` smoke gate passed. No process was left actively driving or monitoring the remote
work on `LLM-T1-NC80` afterward. The VM's idle watchdog (armed in step 8, PID 6807) detected the
idle GPU and attempted to self-terminate via `TERMINATE_CMD="az ml compute stop ..."`, but that
command failed on the VM itself because the `ml` Azure CLI extension is broken there (see "Root
cause" below), so the watchdog kept retrying and logging warnings instead of actually stopping the
VM.

The coordinator detected this directly (not via any automated notification), worked around the
broken CLI by calling the ARM REST API directly from the **local** machine
(`az rest --method post .../computes/LLM-T1-NC80/stop?api-version=2024-04-01`), and confirmed via a
REST GET that the compute instance reached `state: "Stopped"` (`lastOperation` `Stop`/`Succeeded` at
`2026-09-18T13:53:29Z`). Billing stopped within about a minute of the watchdog's first failed retry
— no meaningful idle-billing cost was incurred beyond that window.

## Root cause

On the VM, `az ml ...` commands trigger Azure CLI's dynamic-extension-install path, which fails:

```text
File ".../azure/cli/core/extension/operations.py", line 436, in update_extension
    shutil.copytree(backup_dir, extension_path)
FileExistsError: [Errno 17] File exists: '/opt/az/extensions/ml'
No module named 'rpds.rpds'
```

The `ml` extension (v2.40.1) has a broken native dependency (`rpds-py`, used transitively by
`jsonschema`). Any `az ml ...` invocation attempts to re-install/upgrade the extension to recover,
which fails on a stale extension directory, printing the `rpds.rpds` import error. This affects
**every** `az ml compute ...` call on the VM, including the watchdog's own `TERMINATE_CMD`. Plain
`az account show` and `az rest` (core CLI, no extension) work fine — `az rest` was confirmed to
return a clean ARM error (`ERROR: Not Found`) against a bogus URL, with no Python traceback, proving
it does not touch the broken `ml` extension code path at all.

Separately, the earlier `azure-cli-ml` (legacy v1) extension was already removed from the VM by the
coordinator before this fix (it conflicted with `ml` v2). That removal did not resolve the `rpds`
issue, since the `ml` extension itself is what is broken, not the conflict between the two.

## Resolution

1. Restarted the VM for this task via
   `arf.scripts.utils.azure_ml_vm acquire t0021_zero_shot_latency_reduction --vm-name LLM-T1-NC80`,
   polled synchronously (bounded bash loops with real sleeps, no reliance on any external
   notification). First attempt hit the tool's `VM_START_TIMEOUT_SECONDS` (8 minutes) — SSH did not
   come up before the deadline (this VM has taken 484-504s to become SSH-reachable across both this
   run and the original step 8 provisioning, right at the edge of that budget) — and `acquire`'s own
   rollback logic (`_try_acquire_one`) correctly stopped the VM again rather than leaving a
   billing-but-abandoned box. The second `acquire` attempt succeeded
   (`total_provisioning_seconds: 504.15`, `started_vm: true`).

2. Confirmed `nvidia-smi -L` still lists both H100 NVL GPUs and the VM's persistent storage
   (`/mnt/cache/persist`, OS disk state) survived the stop/start cycle — this is a persistent Azure
   ML Compute Instance, not ephemeral.

3. Re-copied `arf/scripts/utils/idle_watchdog.sh` to `/tmp/idle_watchdog.sh` on the VM (the previous
   copy did not survive the stop/start — ephemeral `/tmp`) and restarted it with a **fixed**
   `TERMINATE_CMD` that uses `az rest` instead of `az ml compute stop`, avoiding the broken `ml`
   extension entirely:

   ```bash
   TERMINATE_CMD='az rest --method post --url "https://management.azure.com/subscriptions/caa7bcdb-c3e1-4687-a73b-7b621b7e4b23/resourceGroups/rezolve-AI/providers/Microsoft.MachineLearningServices/workspaces/brainpowa-northeurope/computes/LLM-T1-NC80/stop?api-version=2024-04-01"'
   ```

   Same `IDLE_THRESHOLD_SECONDS=3600`, `POLL_INTERVAL_SECONDS=60`, `IDLE_UTIL_PERCENT=5`,
   `GRACE_SECONDS=600` as step 8. New watchdog PID confirmed: **5864**. Verified via `ps aux` that
   exactly one watchdog process is running.

4. Validated the fix without actually triggering a real stop: ran `az rest --method post` against a
   deliberately wrong path on the VM and got a clean ARM `ERROR: Not Found` with no Python
   traceback, confirming `az rest` does not invoke the broken extension code.

## Cost impact

Idle window: VM was stopped from approximately `13:53:29Z` (watchdog-triggered stop, via the
coordinator's manual REST workaround) until `14:19:58Z` (this step's successful re-acquire) — during
this window the VM was **not billing** (Azure ML compute instances only bill while `Running`). The
only wasted cost was the two `acquire` attempts' own wall-clock (the first one's ~8-minute timeout
while the VM was mid-boot, which the tool's own `_wasted_cost` accounting attributes if
`started_vm=true` and the attempt fails — see `azure_ml_vm.py`'s `AcquireAttempt`). Cost tracking
(`results/cost_tracking.json`) resumes from the last recorded checkpoint (`$22.15` at
`cosyvoice2 baseline_new_ref smoke gate PASSED`, 1.59h elapsed) plus this restart overhead; the
running total remains well within the task's `$100` hard cap.

## Standing fix for future steps/tasks

Any future watchdog deployment on `LLM-T1-NC80` (or any VM sharing this account's broken `ml`
extension) should default to the `az rest` form of `TERMINATE_CMD` shown above, not
`az ml compute stop`, until the `ml` extension's `rpds-py` dependency is repaired at the image level
(an infrastructure fix outside this task's scope — `arf/` and VM image maintenance are not task work
per `CLAUDE.md` Key Rule 0).
