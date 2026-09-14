---
spec_version: "3"
task_id: "t0008_tts_eval_harness_baselines"
step_number: 8
step_name: "setup-machines"
status: "completed"
started_at: "2026-09-14T16:02:32Z"
completed_at: "2026-09-14T16:35:00Z"
---
## Summary

LLM-T1-NC80 (2×H100 NVL, CUDA 12.2) provisioned from the Azure ML pool and verified ready for Kokoro
evaluation. Environment confirmed: `stt` conda env has kokoro 0.9.4, torch 2.5.1+cu121 (GPU
working), faster-whisper 1.2.1; resemblyzer 0.1.4 installed to `/mnt/tmp/t0008-resemblyzer-venv`
(root disk full). Idle watchdog deployed (PID 5743) with 3600s timeout. Kokoro engine smoke gate
passed. Key finding: 11labs_david corpus absent from VM (ephemeral /mnt wiped) — implementation must
regenerate via ElevenLabs API.

## Actions Taken

1. Ran Phase 1 pre-flight: verified project budget (stop_threshold_reached=false, $5000 available)
   and `az` authentication (rezolve-primary-subscription, VladimirGorovoy@rezolve.com).
2. Phase 2 — Acquired LLM-T1-NC80 from pool via `azure_ml_vm acquire` (started from stopped state,
   provisioning took 502s); wrote initial `machine_log.json`.
3. Phase 3 — Verified GPU via SSH: 2× NVIDIA H100 NVL (95,830 MiB each, driver 535.274.02, CUDA
   12.2). Installed idle watchdog via `watchdog_provisioning.render_azure_ml_install_script`;
   confirmed PID 5743 via `pgrep -f idle_watchdog.sh`; set `watchdog_active: true` in
   machine_log.json.
4. Phase 4 — Searched for 11labs_david corpus at `/mnt/kikiri-tts/data/11labs_david/` — absent
   (ephemeral disk wiped, as per LESSONS Lesson 10). Identified `stt` conda env
   (`/home/azureuser/miniconda3/envs/stt`) as the usable Python env (kokoro 0.9.4, torch
   2.5.1+cu121, faster-whisper 1.2.1, soundfile 0.14.0). Root disk full (118/119GB) — installed
   resemblyzer 0.1.4 via `--no-deps` in a venv at `/mnt/tmp/t0008-resemblyzer-venv` with system
   site-packages from stt env. `HF_HOME=/mnt/cache/persist/hf-cache` (persistent Azure Files mount)
   used for all HF downloads.
5. Engine smoke gate: ran `KPipeline(lang_code="b") + "Hello." bm_george` in stt env — produced 1
   chunk in 6.84s with no errors; GPU active.
6. Updated `machine_log.json` with `gpu_verified`, `cuda_version`, `watchdog_active`,
   `watchdog_idle_timeout_seconds`, and `smoke_test_output`.

## Outputs

- `tasks/t0008_tts_eval_harness_baselines/logs/steps/008_setup-machines/machine_log.json`
- `tasks/t0008_tts_eval_harness_baselines/logs/steps/008_setup-machines/step_log.md`

## Issues

11labs_david corpus absent from VM (`/mnt` ephemeral disk wiped). Per plan Section "Assets Needed"
and plan Risk table, the implementation step's Step 1 fallback is to regenerate via ElevenLabs API
(~$5 extra, ~30 min). This does not block the setup step — it is a known risk with a defined
mitigation. Root disk full (118/119GB) required installing resemblyzer off-root to `/mnt`; the
implementation step must activate the venv from that path or add it to PYTHONPATH.
