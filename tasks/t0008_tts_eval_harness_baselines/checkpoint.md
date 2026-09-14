---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
updated_at: "2026-09-14T18:25:00Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
---
# Task Objective

Build a reusable speaker_sim/TTFB/RTF harness and score ElevenLabs David, base Kokoro, v3 and the
best t0005/t0006 checkpoints.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0008_tts_eval_harness_baselines` created. Initial folder structure initialized in
`tasks/t0008_tts_eval_harness_baselines/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Dependency `t0006_kokoro_v5_stage2_subset` verified as completed. Output written to
`logs/steps/002_check-deps/deps_report.json`.

### Step 11 — creative-thinking

Skipped — structured evaluation against fixed success criteria; no open-ended analysis needed at
this stage.

### Step 13 — compare-literature

Skipped — no published speaker_sim/TTFB baselines for Kokoro-82M David voice fine-tuning exist in
the corpus to compare against.

### Step 3 — init-folders

Created all mandatory task subdirectories including `assets/library` for the expected library asset;
aggregator cache populated in `tasks/t0008_tts_eval_harness_baselines/ctx/` (5 files: task_types,
costs, tasks, metrics, suggestions). Key output: `logs/steps/003_init-folders/folders_created.txt`.

### Step 4 — research-papers

Paper corpus is empty (zero papers, zero categories) — all prior tasks were implementation/training
tasks without paper downloads. Wrote `research/research_papers.md` with `status: "partial"`, all 7
mandatory sections present, and an empty Paper Index; verificator passes with zero errors.

### Step 5 — research-internet

Conducted 11 searches, found 23 sources, discovered 4 papers (GE2E/Wan2018, StyleTTS2/Li2023,
VERSA2024, AnalyzeSim2025). Key findings: resemblyzer same-speaker cosine ≈ 0.85+ validates the
success criterion; Kokoro GPU RTF ≈ 0.03 on A100, first-chunk 28–97 ms; ElevenLabs p50 TTFB 264–335
ms; WER thresholds < 5% clean, > 20% hard failure. Output: `research/research_internet.md`.

### Step 6 — research-code

Reviewed 7 completed tasks, 0 registered libraries. Key outputs: five-module checkpoint packaging
via `tasks/t0002.../code/extract_decoder_generic.py` (38 lines, copy into task); mandatory synthesis
entry point `tasks/t0003.../code/build_pipeline.py` (import directly; `lang_code="b"` + brand
lexicon); t0005 best = `epoch_2nd_00003.pth` (val 0.848); t0006 v6d best = `epoch_2nd_00006.pth`
(val 0.846); all checkpoints DVC-only. Research summary written to `research/research_summary.md`.

### Step 7 — planning

Produced `plan/plan.md`: 21 REQ items, 17-step plan across 5 milestones, cost estimate ~$21.50
(ElevenLabs ~$0.30, H100 VM ~$21), 9 risks, 6 verification criteria, and a Rejection Criteria
section. Key decisions: centroid-half split 679/679 seed=42 for speaker_sim; filler corpus copy from
VM as Step 1; five-module extraction via adapted `extract_decoder.py`; all Kokoro synthesis via
`build_pipeline` import from t0003; variant metrics.json with 16+ variants (8 systems × 2 prompt
sets). Verificator passes with 0 errors.

### Step 8 — setup-machines

LLM-T1-NC80 acquired (started from stopped state, provisioning 502s); 2× H100 NVL (95,830 MiB each),
CUDA 12.2, driver 535.274.02. Environment: `stt` conda env at `/home/azureuser/miniconda3/envs/stt`
has kokoro 0.9.4, faster-whisper 1.2.1, torch 2.5.1+cu121 (CUDA working). resemblyzer 0.1.4
installed to `/mnt/tmp/t0008-resemblyzer-venv` (root disk full at 118/119GB).
`HF_HOME=/mnt/cache/persist/hf-cache` (persistent). Idle watchdog deployed (PID confirmed). Kokoro
engine smoke gate passed (1 chunk, 6.84s). Key finding: 11labs_david corpus absent (ephemeral `/mnt`
wiped) — implementation step must regenerate via ElevenLabs API. Machine log:
`logs/steps/008_setup-machines/machine_log.json`.

### Step 9 — implementation

Built `tts_eval_harness` library (v0.1.0). Ran synthesis for all 8 TTS systems on val96 and filler
prompt sets on LLM-T1-NC80 (2×H100 NVL). Scored all 1568 per-clip records for speaker_sim (GE2E GPU
via resemblyzer), WER (faster-whisper base.en CPU), and duration ratio. Generated `metrics.json` (16
explicit-format variants), `tables.json`, 3 PNG charts, and both results files. DVC-tracked
11labs_david corpus, synth_audio, and packaged checkpoints. VM torn down at 17:42Z. Library
verificator: PASSED. Unit tests: 11/11. Task results verificator: PASSED (0 errors). Gap to 0.85
target: ~0.20 GE2E cosine units.

### Step 10 — teardown

VM was already deallocated during implementation at 2026-09-14T17:42:17Z. Updated `machine_log.json`
with `destroyed_at`, `total_duration_hours` (2.0), and `total_cost_usd` (27.92). Fixed provider
field to "azure_ml" enum value. `verify_machines_destroyed`: PASSED (0 errors).

### Step 12 — results

Verified existing `results_summary.md` and `results_detailed.md` against spec. Added mandatory
`## Examples` section (10+ concrete per-clip instances: random, best, worst, boundary, contrastive
across systems). `verify_task_results`: PASSED (0 errors, 0 warnings). `verify_task_metrics`: PASSED
(0 errors, 0 warnings).

### Step 14 — suggestions

Generated 5 follow-up suggestions (S-0008-01 through S-0008-05) covering: continued Stage 2 training
from v6d epoch 6, val96 speaker_sim collapse investigation, full-corpus training experiment, WER
metric registration, and filler corpus augmentation. No duplicates against existing suggestions (0
in project) or existing tasks. `verify_suggestions`: PASSED (0 errors, 0 warnings). Key output:
`results/suggestions.json`.

* * *

## Cross-Step Decisions

* **Speaker_sim reference design**: centroid from half-A (679 clips, seed=42); ElevenLabs scored
  against half-B (679 clips) to avoid self-comparison; all other systems scored against half-A
  centroid.
* **Checkpoint packaging**: t0005/t0006 raw `.pth` files packaged via adapted `extract_decoder.py`
  (five modules) before any synthesis.
* **GPU required for TTFB**: Kokoro TTFB/RTF must be measured on LLM-T1-NC80 H100 (project success
  criterion specifies "local inference, H100"); CPU TTFB may be reported in addition.
* **resemblyzer in optional extra only**: do NOT add to main `pyproject.toml` dependencies — use
  `[speaker-sim]` extra (webrtcvad/pkg_resources breakage).
* **VM environment (setup-machines)**: use `/home/azureuser/miniconda3/envs/stt` conda env (Python
  3.11, torch 2.5.1+cu121, kokoro 0.9.4, faster-whisper 1.2.1). resemblyzer venv at
  `/mnt/tmp/t0008-resemblyzer-venv` (root disk full). `HF_HOME=/mnt/cache/persist/hf-cache` for all
  remote runs. 11labs_david corpus absent from VM — must regenerate via ElevenLabs API in
  implementation step (plan Step 1 fallback path).
* **Implementation results**: ElevenLabs David speaker_sim=0.832 (fillers)/0.792 (val96),
  TTFB_p50=132ms/153ms; v3_bundle=0.631/0.588; t0006_v6d=0.601/0.482, TTFB=132ms; t0005=not viable
  (duration explosions). No Kokoro system meets the 0.85 target; 7/8 Kokoro meet TTFB ≤300ms.

* * *

## Next Step Notes

Step 15 (reporting): Run all verificators (verify_task_file, verify_task_dependencies,
verify_suggestions, verify_task_metrics, verify_task_results, verify_task_folder, verify_logs,
verify_library_asset, verify_machines_destroyed), capture session transcripts via
capture_task_sessions, set task.json status to "completed" with end_time, then commit and run
poststep. Key assets to verify: library asset tts_eval_harness v0.1.0, suggestions.json (5 items),
metrics.json (16 variants), 3 charts in results/images/. DVC push should already be complete from
implementation step; confirm before finalizing.
