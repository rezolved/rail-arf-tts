---
spec_version: "3"
task_id: "t0021_zero_shot_latency_reduction"
step_number: 8
step_name: "setup-machines"
status: "completed"
started_at: "2026-09-18T11:27:21Z"
completed_at: "2026-09-18T12:15:18Z"
---
## Summary

Provisioned `LLM-T1-NC80` (Azure ML 2xH100 NVL) via the `/setup-remote-machine` skill, armed and
confirmed the idle watchdog (PID 6807) before any build started, verified `/mnt/cache/persist`
resolves to the real Azure Files mount, confirmed both `.venv-cosyvoice2` and `.venv-chatterbox`
from t0018 are intact and reusable, and exported CosyVoice2's `load_jit`/`load_trt` artifacts
successfully. The `.venv-cosyvoice2-vllm` install hit its pre-authorized 20-minute cutoff and is
documented as a null variant per the plan's own fallback.

## Actions Taken

1. Ran the `/setup-remote-machine` skill's Phases 1-4 (pre-flight, VM acquisition, GPU/CUDA
   verification, environment preparation) via a dedicated subagent. Acquired `LLM-T1-NC80` from the
   Azure ML pool (priority-1, sole pool entry, no sibling lock, VM was stopped and started by this
   acquire). Verified 2x NVIDIA H100 NVL (93.6 GB each), CUDA 12.2, and confirmed
   `readlink -f /mnt/cache/persist` resolves to the real Azure Files SMB share (100 TB, 2.0 TB
   used), not ephemeral `/mnt` (Lesson 10). Confirmed `loginctl show-user azureuser` reports
   `Linger=yes` (Lesson 11 precondition for detached remote work surviving SSH disconnection).
2. Deployed and started `arf/scripts/utils/idle_watchdog.sh` with
   `TERMINATE_CMD="az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI"`,
   `IDLE_THRESHOLD_SECONDS=3600`, `IDLE_UTIL_PERCENT=5`, `GRACE_SECONDS=600`,
   `POLL_INTERVAL_SECONDS=60`. Confirmed the watchdog process via `ps aux | grep idle_watchdog` (PID
   6807\) before any build work started, and re-confirmed it was still alive (PID 6807, running
   2193s) at the end of this step. Recorded `watchdog_active: true`,
   `watchdog_idle_timeout_seconds: 3600`, `watchdog_pid: 6807` in `machine_log.json`.
3. Confirmed `.venv-cosyvoice2` (`torch==2.3.1+cu121`, `torch.cuda.is_available()==True`,
   `device_count()==2`) and `.venv-chatterbox` (`torch==2.6.0+cu124`, same GPU checks, plus a
   successful `from chatterbox.tts import ChatterboxTTS` import, ~3m53s on first import,
   `chatterbox-tts==0.1.7` confirmed via `pip show`) both reused as-is from t0018's persistent
   `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/venvs/` — no recreation needed.
4. Ran CosyVoice2's own `export_trt.py` (`load_jit=True, load_trt=True, fp16=True`) inside
   `.venv-cosyvoice2`, in the background on the VM (concurrently with step 5 below), writing
   artifacts to `/mnt/cache/persist/t0021_zero_shot_latency_reduction/pretrained/cosyvoice2/`
   (`flow.encoder.fp32.zip`, 192369155 bytes; `flow.encoder.fp16.zip`, 116706819 bytes) — never
   under ephemeral `/mnt`. Completed successfully in ~301s ("load_trt/load_jit export completed
   successfully"), despite several benign TensorRT tactic-skip warnings during engine compilation.
5. Attempted the new, separate `.venv-cosyvoice2-vllm` venv install (`vllm==0.11.0`,
   `transformers==4.57.1`, `numpy==1.26.4`) under a `timeout 1100` (~18.3 min, within the plan's
   20-minute cap) cutoff. All package downloads completed with no errors, but the install was still
   in its `Installing collected packages` unpacking phase (had reached `torch`, had not yet reached
   `vllm` itself) when the timeout fired. This was originally executed by the provisioning subagent
   as a detached remote background job and, after the subagent ended its own turn without
   registering any wakeup for it (a fire-and-forget gap this step-executor caught and corrected per
   `LESSONS.md` Lesson 8 and the `execute-task` SKILL.md Phase -0.5 exit-discipline rule), this
   step-executor directly polled the remote PID via SSH (bounded, foreground `kill -0`/`sleep`
   loops, not an unmonitored background hand-off) until both the export and the install attempt
   reached a terminal state. Wrote `intervention/cosyvoice2_vllm_install_timeout.md` documenting the
   exact timing, the confirmed partial venv state (`torch` present, `vllm` absent,
   `ModuleNotFoundError` on import), and marked the `vllm_backend` CosyVoice2 acceleration variant
   null per the plan's own pre-authorized fallback ("record the exact error ... mark the
   cosyvoice2_vllm variant null with the reason, and continue with the other 5 CosyVoice2
   variants").
6. Verified GPU via `nvidia-smi --query-gpu=name,memory.total,driver_version` (2x NVIDIA H100 NVL,
   95830 MiB, driver 535.274.02) and captured the full smoke-test output (persist-mount check,
   Linger check, both venvs' torch/CUDA checks, the JIT/TRT export result) into `machine_log.json`'s
   `smoke_test_output` field.
7. Wrote `logs/steps/008_setup-machines/machine_log.json` per
   `arf/specifications/remote_machines_specification.md` with all required fields for the `azure_ml`
   provider (`spec_version`, `provider`, `instance_id`, `vm_name`, `selected_offer`,
   `selection_rationale`, `hourly_cost_usd`, `started_vm`, `ssh_host`/`ssh_port`, `gpu_verified`,
   `cuda_version`, `created_at`/`ready_at`, `watchdog_active`/`watchdog_idle_timeout_seconds`/
   `watchdog_pid`, `search_started_at`/`total_provisioning_seconds`, `failed_attempts: []`,
   `smoke_test_output`). `destroyed_at`/`total_duration_hours`/`total_cost_usd` remain `null` — the
   VM stays up for the `implementation` step that follows; teardown happens in its own later step.

## Outputs

* `tasks/t0021_zero_shot_latency_reduction/logs/steps/008_setup-machines/machine_log.json`
* `tasks/t0021_zero_shot_latency_reduction/intervention/cosyvoice2_vllm_install_timeout.md`
* Remote (`LLM-T1-NC80`, under `/mnt/cache/persist/t0021_zero_shot_latency_reduction/`):
  `pretrained/cosyvoice2/flow.encoder.fp32.zip`, `pretrained/cosyvoice2/flow.encoder.fp16.zip`,
  `logs/export_trt.log`, `logs/install_cosyvoice2_vllm.log`, `venvs/.venv-cosyvoice2-vllm/` (partial
  — `torch` installed, `vllm` not yet installed).
* Idle watchdog running on `LLM-T1-NC80` as PID 6807, confirmed alive both at arming time and at the
  end of this step.

## Issues

The `.venv-cosyvoice2-vllm` install did not finish within its pre-authorized 20-minute cutoff (see
`intervention/cosyvoice2_vllm_install_timeout.md`); the `vllm_backend` CosyVoice2 acceleration
variant is null for this task per the plan's own pre-registered fallback, and the other 5 CosyVoice2
variants plus all Chatterbox variants are unaffected. Separately, the provisioning subagent spawned
two long-running remote jobs (this vLLM install and the TensorRT export) and then ended its own turn
claiming it would "wait for background job notifications," without any actual mechanism registered
to resume it or the parent step — a fire-and-forget gap matching `LESSONS.md` Lesson 8. This
step-executor caught the gap (the VM remained watchdog-protected throughout, so no idle-billing risk
materialized) and drove the remaining monitoring to completion directly via bounded, synchronous SSH
polling rather than leaving the step `in_progress` on an unmonitored assumption.
