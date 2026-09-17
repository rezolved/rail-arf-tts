---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
updated_at: "2026-09-17T14:12:04Z"
completed_steps: 5
next_step_number: 6
next_step_id: "research-code"
---
# Task Objective

Benchmark three open zero-shot voice-cloning TTS models (F5-TTS, CosyVoice 2, Chatterbox) on David
reference audio with the t0008 harness to calibrate the reachable speaker_sim/TTFB envelope.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0018_zero_shot_cloning_calibration` created. Initial folder structure initialized in
`tasks/t0018_zero_shot_cloning_calibration/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` for `t0018_zero_shot_cloning_calibration`: the single declared
dependency, `t0008_tts_eval_harness_baselines`, has `status: "completed"` in its `task.json`, so the
check passed with 0 errors and 0 warnings. Result written to
`logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory task folder structure (`plan/`, `research/`,
`results/`, `results/images/`, `corrections/`, `intervention/`, `code/`, `logs/commands/`,
`logs/searches/`, `logs/sessions/`, `logs/steps/`, `assets/answer/`) with `.gitkeep` files; recorded
in `logs/steps/003_init-folders/folders_created.txt`. Populated the local aggregator cache at
`tasks/t0018_zero_shot_cloning_calibration/ctx/` (task_types, costs, tasks, metrics, suggestions) —
gitignored and not committed.

### Step 4 — research-papers

Reviewed the full existing paper corpus (`aggregate_papers`) for material on F5-TTS, CosyVoice 2,
Chatterbox, and GE2E speaker-similarity evaluation. The corpus currently holds only 4 papers, all
added by an unrelated prior task (`t0014_v11_decoder_fix_retrain`): HiFi-GAN, iSTFTNet, a
multi-generator vocoder study, and StyleTTS 2 — none cover the three target zero-shot cloning
systems directly, though StyleTTS 2 underlies the Kokoro baseline this task re-scores. Wrote
`research/research_papers.md` (`status: "complete"`, 4/4 papers cited) documenting this gap
explicitly; verificator passed with 0 errors, 1 expected warning (`RP-W003`, project has no category
taxonomy yet).

### Step 5 — research-internet

Wrote `research/research_internet.md` (19 web searches, 4 targeted fetches, 18 sources cited, 7
papers discovered, `status: "complete"`) covering install/checkpoints, licensing, and streaming/TTFB
behavior for F5-TTS, CosyVoice 2, and Chatterbox. Verificator passed with 0 errors, 0 warnings. Key
new findings: F5-TTS default weights are CC-BY-NC-4.0 (non-commercial risk for Key Question 6); only
CosyVoice 2 has genuine chunked streaming (F5-TTS and base Chatterbox are whole-utterance TTFB);
F5-TTS reference clips over ~20 s risk mid-word truncation, a direct risk to this task's
`ref_concat` (~30 s) condition — flag for validation before the full implementation run; Chatterbox
itself has no research paper (confirmed by the maintainer). Of 7 discovered papers, the first batch
of 3 `/add-paper` subagents (max-3-concurrent cap) completed and their assets are committed in this
step: `Chen2024-F5TTS` (`10.48550_arXiv.2410.06885`), `Du2024-CosyVoice2`
(`10.48550_arXiv.2412.10117`), and `Wan2018-GE2E` (`10.1109_ICASSP.2018.8462665`). 4 more remain
queued, not yet dispatched: `Du2024-CosyVoice1`, `Casanova2022-YourTTS`, `Zhang2025-ECAPA` (low
priority), `ChatterboxFlash2026` (low priority).

* * *

## Cross-Step Decisions

* The paper corpus has zero coverage of F5-TTS, CosyVoice 2, or Chatterbox specifically —
  `research-internet` (step 5) independently sourced install/inference/checkpoint details for all
  three since there is no corpus paper to fall back on.
* F5-TTS's default checkpoints are CC-BY-NC-4.0 and remain non-commercial even after fine-tuning —
  must be flagged explicitly in `results/suggestions.json` if F5-TTS leads on `speaker_sim`/TTFB
  (Key Question 6, production viability).
* F5-TTS official docs warn reference clips over ~20 s risk mid-word truncation, with a 30 s
  prompt+generation cap — this task's `ref_concat` condition (~30 s) must be validated against this
  limit before the full implementation run (smoke-gate stage).
* Published SIM-o/SS/MOS numbers from F5-TTS, CosyVoice 2, and the Podonos Chatterbox AB study use
  different metrics (WavLM-based or MOS/preference-%) than this project's GE2E-cosine `speaker_sim`
  and must never be merged/compared directly — extends the same rule already established for CMOS-S
  in `research_papers.md`.
* **Paper-addition tracking (for `compare-literature`/`reporting` to check)**: 3 of 7 discovered
  papers landed in the corpus and are committed in this step — `Chen2024-F5TTS`
  (`10.48550_arXiv.2410.06885`), `Du2024-CosyVoice2` (`10.48550_arXiv.2412.10117`), `Wan2018-GE2E`
  (`10.1109_ICASSP.2018.8462665`) — confirmed via `aggregate_papers` (now 7 total papers in the
  corpus, up from 4). 4 papers remain queued and undispatched: `Du2024-CosyVoice1`,
  `Casanova2022-YourTTS`, `Zhang2025-ECAPA` (low priority), `ChatterboxFlash2026` (low priority). A
  later step (ideally before `compare-literature`) must dispatch the remaining 4 and verify all 7
  discovered papers landed in the corpus before `reporting` finalizes.

* * *

## Next Step Notes

Proceed to step 6 (`research-code`): review the t0008 harness code, t0014 reference-concat approach,
and t0015 audio quality gate for reuse, per `step_tracker.json`. The first batch of 3 `/add-paper`
subagents already completed and landed in the corpus (see Cross-Step Decisions). Before
`compare-literature` (step 13) or `reporting` (step 15), a step-executor must: (1) dispatch the 4
still-queued `/add-paper` subagents (`Du2024-CosyVoice1`, `Casanova2022-YourTTS`, `Zhang2025-ECAPA`,
`ChatterboxFlash2026`), (2) confirm via `aggregate_papers` that all 7 discovered papers from
`research/research_internet.md` landed in the corpus, and (3) if any subagent failed, attempt the
paper addition inline per the `execute-task` paper-addition protocol.
`research/research_internet.md` and `research/research_papers.md` should both be read before
`planning` (step 7) — key recommendations from internet research (license risk flag, TTFB
reporting-per-capability, ref_concat duration validation) should carry into the plan.
