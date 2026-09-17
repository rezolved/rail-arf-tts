---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
updated_at: "2026-09-17T14:49:01Z"
completed_steps: 7
next_step_number: 8
next_step_id: "setup-machines"
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

### Step 6 — research-code

Reviewed the `tts_eval_harness` library (t0008) plus t0013/t0014/t0015's audible-speech gate lineage
and `build_reference_concat.py`, and t0010's GPU cost-overrun precedent; wrote
`research/research_code.md` (5 tasks cited, `status: "complete"`, verificator passed 0
errors/warnings). In the same step, dispatched and confirmed all 4 remaining queued `/add-paper`
subagents from step 5 — the paper corpus now holds all 7 discovered papers (11 total, up from 4).
After all research steps completed, a `/research-summarize` subagent wrote
`research/research_summary.md` for downstream planning/implementation subagents to read instead of
the full research files.

### Step 7 — planning

Wrote `plan/plan.md` (spec_version 2, all 11 mandatory sections, 18 `REQ-*` items) covering the
3-system × 2-condition zero-shot cloning benchmark plus the two paired baselines; `verify_plan`
passed 0 errors/0 warnings. Cost estimate ≈$42-45 GPU wall-clock against the $70 hard cap, with a
dedicated Step 4 F5-TTS `ref_concat` smoke-gate check (confirmed the standard F5-TTS pipeline
auto-crops reference audio to ~15 s by default) run before the full 196-prompt job. Caveat: the
planning subagent initially wrote its output to the main repo instead of the task worktree
(spawn-prompt omission); the step-executor recovered the file into the worktree and cleaned the
stray main-repo files before verifying — no content was lost.

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
* **Paper-addition tracking — RESOLVED in step 6**: all 7 papers discovered in
  `research/research_internet.md` are now landed in the corpus (11 papers total, up from 4 before
  this task started); no papers remain queued. Confirmed via
  `aggregate_papers --format json --detail short`. Final DOI/commit map:
  * `Chen2024-F5TTS` — `10.48550_arXiv.2410.06885` (landed step 5)
  * `Du2024-CosyVoice2` — `10.48550_arXiv.2412.10117` (landed step 5)
  * `Wan2018-GE2E` — `10.1109_ICASSP.2018.8462665` (landed step 5)
  * `Du2024-CosyVoice1` — `10.48550_arXiv.2407.05407` (landed step 6, commit `0268535`)
  * `Casanova2022-YourTTS` — `10.48550_arXiv.2112.02418` (landed step 6, commit `a305347`)
  * `Zhang2025-ECAPA` (dispatch label) — actually **`Kunesova2025`**, "An Exploration of ECAPA-TDNN
    and x-vector Speaker Representations in Zero-shot Multi-speaker TTS" (Kunešová, Hanzlíček,
    Matoušek, TSD 2025) — `10.48550_arXiv.2506.20190` (landed step 6, commit `4bb35d8`). The
    research-internet snippet's authorship guess was wrong; the `/add-paper` subagent re-verified
    against the actual PDF and corrected the citation key.
  * `ChatterboxFlash2026` (dispatch label) — actually "Chatterbox-Flash: Prior-Calibrated Block
    Diffusion for Streaming Zero-Shot TTS" (Seo, Park, Nam, 2026) — `10.48550_arXiv.2605.30748`
    (landed step 6, commit `a5f7160`). Same situation: snippet authorship was wrong, re-verified and
    corrected on download.
  * No paper addition failed; the inline-fallback path in the paper-addition protocol was not
    needed. `compare-literature` and `reporting` can treat paper coverage as complete — no further
    dispatch action required.
* Two independent `/add-paper` subagents flagged that `arf/skills/add-paper/SKILL.md`'s documented
  verificator module path (`arf.scripts.verificators.verify_paper_asset`) does not exist in this
  repo; the real module is `meta.asset_types.paper.verificator`. This is a framework documentation
  bug (out of scope for this task per CLAUDE.md Rule 0) — worth a future `self-improvement` pass.
* **F5-TTS `ref_concat` risk — resolved as a planned smoke-gate check, not a blocker**: confirmed
  during planning (step 7) that F5-TTS's standard pipeline (`preprocess_ref_audio_text`) auto-crops
  reference audio over ~15 s by default. The plan's Step 4 requires a dedicated F5-TTS ×
  `ref_concat` synthesis check before the full 196-prompt run, recording whether the effective
  reference consumed was ~30 s or ~15 s; this is a documented caveat on that one variant, not
  grounds for nulling it, unless synthesis fails outright even after a manual 15 s-trim retry.
* **`results/metrics.json` unregistered-key ambiguity — resolved**: the task text's
  `efficiency_inference_time_per_item_seconds`/`efficiency_inference_cost_per_item_usd` keys are not
  registered in `meta/metrics/` (only `rtf`, `speaker_sim`, `ttfb_ms` are). The plan routes these
  two fields (plus WER, duration ratio, gate-failure count) into `results/tables.json` instead,
  following t0008's own `REGISTERED_METRIC_KEYS` convention, and flags that `suggestions.json`
  (orchestrator step) should recommend registering the two efficiency metrics via `/add-metric` for
  future tasks.
* **Planning-subagent worktree miss (process note for future spawns)**: the step 7 planning subagent
  was spawned without an explicit worktree `cd` instruction and defaulted to the main repo checkout
  on `main`, writing `plan/plan.md` and regenerated `ctx/` files there. The step-executor caught
  this via `git status`/`git branch` before trusting the subagent's report, copied the plan into the
  worktree, and deleted the stray main-repo files. Future step-executors in this task should include
  the explicit worktree path in subagent spawn prompts to avoid repeating this.

* * *

## Next Step Notes

Proceed to step 8 (`setup-machines`) per `step_tracker.json`. `plan/plan.md` is complete and
verified (0 errors/0 warnings); read it in full before provisioning — it names `LLM-T1-NC80` (H100),
gives the exact watchdog `TERMINATE_CMD`, and requires the idle watchdog PID to be confirmed before
the first model download (Lesson 8). Plan Step 1 lists the exact `dvc pull` targets needed first
(`11labs_david.dvc`, `synth_audio.dvc`, and the v3 decoder/voicepack `.dvc` files under
`t0006_kokoro_v5_stage2_subset`). Budget: base estimate ≈$42-45 GPU wall-clock against the
user-authorized $70 hard cap — monitor cumulative spend at each milestone boundary per the plan's
Risks & Fallbacks table. When spawning the `setup-machines` subagent (and every subsequent
skill-invocation subagent for this task), explicitly state the worktree path
(`/home/azureuser/rail-metarepo/real-repos/rail-arf-tts-worktrees/t0018_zero_shot_cloning_calibration`)
and branch (`task/t0018_zero_shot_cloning_calibration`) in the spawn prompt — step 7 showed a fresh
subagent otherwise defaults to the main repo checkout.
