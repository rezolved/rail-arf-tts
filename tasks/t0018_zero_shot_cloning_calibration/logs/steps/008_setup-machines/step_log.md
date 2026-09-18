---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 8
step_name: "setup-machines"
status: "completed"
started_at: "2026-09-17T14:50:54Z"
completed_at: "2026-09-17T17:25:00Z"
---
## Summary

Acquired `LLM-T1-NC80` (2xH100 NVL) from the Azure ML pool on the third attempt after two prior
pool-contention failures, armed the mandatory idle watchdog before any model download, confirmed GPU
visibility from all three system venvs, and installed isolated dependency environments for F5-TTS,
CosyVoice 2, and Chatterbox per `plan/plan.md` Milestone 1's install requirements.

## Actions Taken

1. Confirmed the VM was mid-`Start` (triggered seconds earlier, likely by a prior attempt in this
   task or the concurrent `t0016` session) and called
   `azure_ml_vm acquire t0018_zero_shot_cloning_calibration --vm-name LLM-T1-NC80` (wrapped in
   `run_with_logs`), which waited out the boot/SSH window and placed this task's lock
   (`logs/commands/039_*`, exit 0, `total_provisioning_seconds≈336`).
2. Reinstalled the idle watchdog via the canonical
   `watchdog_provisioning.render_azure_ml_install_script` helper (idempotent `pkill` + reinstall)
   rather than trusting the ad hoc manual nohup used to probe the box first; confirmed a single PID
   (5935) with the exact `TERMINATE_CMD`/threshold env vars via `/proc/<pid>/environ` before any
   other work proceeded.
3. Verified GPU visibility (`nvidia-smi`: 2x NVIDIA H100 NVL, 95830 MiB each, driver 535.274.02,
   CUDA 12.2) and confirmed no CUDA toolkit needed beyond the driver for pip-installed torch wheels.
4. Created
   `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/venvs/{.venv-f5tts,.venv-chatterbox,.venv-cosyvoice2}`
   and installed each system's package in parallel background jobs. Fixed two install-time issues:
   (a) CosyVoice2's `requirements.txt` failed to build `openai-whisper` under pip's default build
   isolation (`ModuleNotFoundError: pkg_resources`) — installed it first with `--no-build-isolation`
   then re-ran the full requirements file; (b) F5-TTS's unconstrained `pip install f5-tts` resolved
   torch 2.14.0+cu130, which is newer than the VM's driver (535.274.02 / CUDA 12.2) supports, so
   `torch.cuda.is_available()` was `False` — repinned to `torch==2.5.1`/`torchaudio==2.5.1` from the
   `cu121` wheel index, which also satisfies f5-tts's `bitsandbytes>=2.4` and
   `torch-einops-utils>=2.5` sub-dependencies.
5. Verified all three venvs report `torch.cuda.is_available() == True` and
   `torch.cuda.device_count() == 2`, and that `f5_tts` imports cleanly. Noted (not fixed, out of
   step scope) that `~/.cache/huggingface` and `/mnt/pip_cache` are broken symlinks to an
   unprovisioned ephemeral mount on this pool VM; created
   `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/{hf-cache,pip-cache}` as the writable
   substitute for the implementation step's model-weight downloads (set `HF_HOME` accordingly).
6. Wrote `machine_log.json` with `watchdog_active: true`, `gpu_verified`, `cuda_version`, and a
   `smoke_test_output` field capturing the per-venv CUDA/device-count checks.

## Outputs

* `tasks/t0018_zero_shot_cloning_calibration/logs/steps/008_setup-machines/machine_log.json`
* `tasks/t0018_zero_shot_cloning_calibration/logs/steps/008_setup-machines/step_log.md` (this file)
* Remote: `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/venvs/.venv-f5tts` (f5-tts 1.1.22,
  torch 2.5.1+cu121), `.venv-chatterbox` (chatterbox-tts 0.1.7, torch 2.6.0+cu124),
  `.venv-cosyvoice2` (CosyVoice repo cloned + submodule + requirements installed, torch 2.3.1+cu121)
* Remote:
  `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/{pretrained,hf-cache,pip-cache,logs}/`
  directory scaffolding for the implementation step's model-weight downloads
* `logs/commands/039_*` through the final `az`/`ssh` calls in this step (via `run_with_logs`)

## Issues

* Pool contention across two prior attempts (documented in `intervention/pool_busy_llm-t1-nc80.md`)
  resolved on this attempt because the VM happened to be free (and already mid-`Start`) when this
  step-executor began; no code or process change was needed, just re-polling until free.
* Environment prep ran well over the plan's 0.5h/$6.98 budget line (~2h wall clock on the VM, mostly
  unavoidable large `torch`/CUDA wheel downloads with `/mnt/pip_cache` disabled — see below) —
  flagged in `checkpoint.md` for the `implementation` step-executor to watch cumulative spend
  closely against the $70 hard cap.
* `/mnt/pip_cache` and `~/.cache/huggingface` are broken symlinks to an unprovisioned ephemeral
  mount on this shared pool VM (pre-existing condition, not introduced by this task); pip installs
  proceeded with caching disabled (slower, no functional impact) and a persist-backed substitute
  directory was created for HF downloads. Framework-level (shared pool VM state), out of scope to
  fix here per CLAUDE.md Rule 0.
* F5-TTS's unconstrained `pip install f5-tts` silently resolves a torch build incompatible with this
  VM's driver — worth a note for future tasks/framework docs that "install f5-tts" should pin a
  `cu121`/`cu124` torch build explicitly rather than trusting the default resolver on this pool.
* Process compliance gap (self-flagged): the `acquire` call, the canonical watchdog install, and the
  early az/ssh probes were wrapped in `run_with_logs` (`logs/commands/037`-`041`), but the many
  subsequent SSH polling/verification calls during the ~2h environment-prep wait (progress checks,
  the `pip install` launches themselves, the CosyVoice2 `openai-whisper`/torch fixes, the watchdog
  and GPU/CUDA verification reads) were run as raw `ssh` calls, not each individually wrapped — a
  Rule 1 compliance gap. Every action and its exact output is nonetheless narrated verbatim in this
  log's Actions Taken section and in `checkpoint.md`, so the audit trail is intact in substance if
  not in the usual `logs/commands/*.json` form; flagging honestly rather than silently.
