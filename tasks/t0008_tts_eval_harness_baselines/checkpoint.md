---
spec_version: "1"
task_id: "t0008_tts_eval_harness_baselines"
updated_at: "2026-09-14T16:01:00Z"
completed_steps: 9
next_step_number: 8
next_step_id: "setup-machines"
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

* * *

## Next Step Notes

Step 8 is setup-machines. The setup executor should: (1) start LLM-T1-NC80 via the
`setup-remote-machine` skill; (2) verify the `kokoro-finetune` conda environment or install `kokoro`
\+ `resemblyzer` + `faster-whisper`; (3) confirm `/mnt/kikiri-tts/data/11labs_david/` exists (if
absent, the implementation step must regenerate via ElevenLabs API); (4) capture `nvidia-smi` and
torch/CUDA versions for `results/metadata.json`; (5) deploy the idle watchdog per CLAUDE.md. Budget
context: ~$21 of the $30 task budget is for VM time (~1.5 h at $13.96/h).
