---
spec_version: "2"
task_id: "t0018_zero_shot_cloning_calibration"
date_completed: "2026-09-17"
status: "complete"
---
# Plan: Zero-Shot Voice-Cloning Calibration

## Objective

Benchmark three open-weight zero-shot voice-cloning TTS models — F5-TTS (`SWivid/F5-TTS`, base
English checkpoint), CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`), and Chatterbox
(`ResembleAI/chatterbox`) — on David reference audio using the existing `tts_eval_harness` library
from `t0008_tts_eval_harness_baselines`, to empirically calibrate the reachable `speaker_sim` (GE2E
cosine similarity vs ElevenLabs David) and TTFB/RTF envelope on this voice, without any fine-tuning.
Each system is run under two reference-audio conditions (`ref_single` ~10 s and `ref_concat` ~30 s,
both drawn only from the ElevenLabs half-A split so half-B stays a clean scoring set), for 3 systems
× 2 conditions = 6 cloning variants, plus two paired baselines re-measured in the same GPU session:
`elevenlabs_david` (self-consistency ceiling, scored vs half-B) and `kokoro_v3_bundle` (current best
fine-tune). This is a measurement task, not a model-selection or fine-tuning task — no training, no
tuning of reference selection on `val_96`.

**Done looks like**: one answer asset (`assets/answer/zero-shot-speaker-sim-ceiling/`) stating the
highest reachable `speaker_sim` on David and whether the project's `≥0.85` success criterion should
be restated; `results/metrics.json` and `results/per_clip_metrics.json` covering all 8 variants ×
196 prompts (~1568 rows); four required charts; a human-listenable, DVC-tracked audio set with
`results/listening_guide.md`; and a documented, non-silent resolution for every system that fails
installation or its smoke gate. Total GPU spend stays within the **$70 hard cap** authorized for
this task (budgeted ≈$45).

## Task Requirement Checklist

Operative task text (from `task.json` `short_description` and the resolved long description at
`tasks/t0018_zero_shot_cloning_calibration/task_description.md`):

> Benchmark three open zero-shot voice-cloning TTS models (F5-TTS, CosyVoice 2, Chatterbox) on David
> reference audio with the t0008 harness to calibrate the reachable speaker_sim/TTFB envelope.

* **REQ-1**: Benchmark F5-TTS, CosyVoice 2, and Chatterbox on David reference audio via the t0008
  `tts_eval_harness`. Satisfied by Steps 4-6, 9-11. Evidence: per-system entries in
  `results/per_clip_metrics.json` and `results/metrics.json`.
* **REQ-2**: Two reference-audio conditions per cloning system — `ref_single` (one fixed ~10 s
  half-A clip, longest clean half-A clip, filename recorded) and `ref_concat` (a fixed ~30 s
  concatenation of half-A clips, filenames and order recorded) — both drawn only from
  `build_reference_split(seed=42)`'s half-A. Satisfied by Step 3. Evidence:
  `data/references/manifest.json`, `results/audio_samples/references/`.
* **REQ-3**: Re-run `elevenlabs_david` (scored vs half-B, the t0008 protocol) and `kokoro_v3_bundle`
  in the same GPU session as paired baselines (Lesson 1: cross-session latency numbers are not
  pairable). Satisfied by Steps 7-8. Evidence: `elevenlabs_david` and `kokoro_v3_bundle` rows in
  `results/per_clip_metrics.json` with `ttfb_ms`/`rtf` measured in this task's own engine session.
* **REQ-4**: Same protocol as t0008 — 96 val96 texts + 100 filler texts (196 total); per system:
  smoke gate (one synthesis succeeds), 50 discarded warmup requests, then all 196 prompts measured
  in one engine session; record per clip TTFB (labelled streaming vs whole-utterance), RTF, audio
  duration, duration ratio vs the ElevenLabs reference clip, WER, and GE2E cosine vs the half-A
  centroid (see the ambiguity note below). Satisfied by Steps 2, 4-8. Evidence:
  `results/per_clip_metrics.json` row count and per-variant `n_clips`/`n_successful` in
  `results/tables.json`.
* **REQ-5**: Run the hardened audible-speech gate
  (`tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`) on every synthesized
  clip; report the failure count per system; exclude gate-failing clips from `speaker_sim` means
  only if stated in the table caption, with the unfiltered mean shown alongside. Satisfied by Step
  12\. Evidence: `results/gate_failures.json`, caption text in `results/tables.json` and
  `results_detailed.md`.
* **REQ-6**: Rejection rule — any system with `successful_prompts / total_prompts < 0.8` on a prompt
  set has null metrics for that set (Lesson 3). Satisfied by Step 13 and the `## Rejection Criteria`
  section below. Evidence: null markers in `results/metrics.json` for any variant that trips the
  threshold.
* **REQ-7**: If a named system cannot be installed or fails its smoke gate within 45 minutes of
  effort, mark it null with the exact error (Lesson 2), do not substitute silently, and continue
  with the other two. A substitute (e.g. XTTS-v2, Fish Speech) may be added only if time/budget
  remain after all three named systems are done, and must be named in the results as an addition.
  Satisfied by Step 4 (per-system time-box) and the `## Risks & Fallbacks` table. Evidence:
  intervention file (if triggered) plus a note in `results_detailed.md`.
* **REQ-8**: Mandatory human-listenable audio, DVC-tracked (`dvc add`, `dvc push` before the PR):
  `results/audio_samples/harness/<system>_<condition>/` (all 196 clips per variant, nothing
  discarded, gate-failing clips included); `results/audio_samples/comparison_set/` (3 fixed gate
  texts + 7 val96 prompts at seed 42, same 10 texts per system, ElevenLabs original + v3 bundle
  included, named `<text_id>__<system>_<condition>.wav`); `results/audio_samples/references/` (the
  exact `ref_single`/`ref_concat` clips fed to the cloning models); `results/listening_guide.md`
  (one row per comparison text, one column per system/condition plus ElevenLabs and v3, clickable
  relative links with per-clip `speaker_sim`/WER/gate verdict, and a "what to listen for" line per
  row). Satisfied by Steps 3, 14-16.
* **REQ-9**: Answer the six Key Questions (envelope vs ElevenLabs self-consistency ceiling; does any
  system beat `kokoro_v3_bundle` fillers 0.631; TTFB p50 ≤300 ms / RTF pass, with whole-utterance
  latency labelled as such for non-streaming systems; effect of `ref_single` vs `ref_concat`; WER
  acceptability on brand names/product terms; whether the project's success criterion should be
  restated and whether Kokoro Stage 2 remains the main line). Satisfied by Step 17 (the answer
  asset). Evidence: `full_answer.md` sections citing the specific numbers.
* **REQ-10**: Produce one answer asset, `assets/answer/zero-shot-speaker-sim-ceiling/`, for the
  question "What speaker_sim and latency envelope is reachable on the David voice by zero-shot
  cloning, and what does that imply for the Kokoro fine-tuning line?" — short answer states the
  numbers, full answer carries tables and the success-criterion recommendation. Satisfied by Step
  17\.
* **REQ-11**: `results/per_clip_metrics.json` with ≥100 rows per system (8 variants × 196 prompts ≈
  1568 rows). Satisfied by Steps 4-8. Evidence: row-count check in `## Verification Criteria`.
* **REQ-12**: `results/metrics.json` in explicit-variant format: each system/condition × prompt-set
  variant with `speaker_sim`, `ttfb_ms`, `rtf` (the three project-registered metrics), plus WER,
  duration ratio, gate-failure count, `efficiency_inference_time_per_item_seconds`, and
  `efficiency_inference_cost_per_item_usd` (machine hourly price × wall-clock / clips). Satisfied by
  Step 13. **Ambiguity flag**: the last two keys are not registered in `meta/metrics/`
  (`uv run python -u -m arf.scripts.aggregators.aggregate_metrics` lists only `rtf`, `speaker_sim`,
  `ttfb_ms`) and `arf/specifications/metrics_specification.md` states unregistered keys in
  `metrics.json` are verification errors. Resolution: follow t0008's own `REGISTERED_METRIC_KEYS`
  convention (`code/report.py`) — write only `speaker_sim`, `ttfb_ms`, `rtf` into each variant's
  `metrics` object in `results/metrics.json`, and write the two efficiency fields (plus WER,
  duration ratio, gate-failure count) into `results/tables.json` per variant. This plan cannot
  register new project metrics itself (Key Rule 3 forbids editing files outside the task folder, and
  `meta/metrics/` is outside it) — `results/suggestions.json` (orchestrator-managed, not an
  implementation step) should recommend registering `efficiency_inference_time_per_item_seconds` and
  `efficiency_inference_cost_per_item_usd` via the `/add-metric` skill for future tasks.
* **REQ-13**: Four charts in `results/images/`, embedded in `results_detailed.md`:
  `speaker_sim_by_system.png` (grouped bars, fillers vs val96, horizontal lines at the ElevenLabs
  ceiling 0.832/0.792 and the v3 score 0.631/0.588), `ttfb_vs_speaker_sim.png` (scatter, one point
  per variant, x = TTFB p50 ms with a vertical line at 300, y = fillers speaker_sim),
  `ref_condition_effect.png` (paired bars per system), `wer_by_system.png`. Satisfied by Step 14.
* **REQ-14**: Tables: per variant × prompt-set (speaker_sim mean±std, TTFB p50/p95/p99, RTF
  mean±std, WER, explosions, gate failures, n successful); environment table (model commit/version,
  torch, CUDA — Lesson 4). Satisfied by Steps 5, 13.
* **REQ-15**: `results/suggestions.json` — orchestrator-managed step, **not** part of this plan's
  Step by Step (per `arf/specifications/plan_specification.md`, implementation stops at metrics/
  charts). This plan records here, for the orchestrator's suggestions step to use, that the
  suggestion should include a proposed restated success criterion for `project/description.md` and,
  if any system meets both `speaker_sim` and TTFB bars, a production-integration feasibility task.
* **REQ-16**: Compute on `LLM-T1-NC80` (H100), budget ≈3.2 h × $13.96/h ≈ $45, **hard cap $70**
  (matches the user-authorized cap for this task). Watchdog armed and PID confirmed before the first
  model download (Lesson 8). Satisfied by Steps 1, 18-19; tracked in `## Cost Estimation` and
  `## Risks & Fallbacks`.
* **REQ-17**: Forbidden actions — no fine-tuning of any system; no tuning of reference selection on
  `val_96`; no `resemblyzer` in main `pyproject.toml` dependencies (keep `[speaker-sim]` extra); no
  silent substitution of a named system without a null result and recorded reason. Enforced
  throughout Steps 2-13; verified in `## Verification Criteria`.
* **REQ-18**: Dependency `t0008_tts_eval_harness_baselines` supplies the `tts_eval_harness` library
  (`code/harness.py`, `code/adapters.py`, `code/scoring.py`, `code/report.py`), the prompt sets
  (`data/val96_prompts.json`, `data/filler_prompts_100.json`), the half-A/half-B split protocol
  (`build_reference_split(seed=42)`), and the paired baseline numbers (ElevenLabs speaker_sim=0.832
  fillers/0.792 val96, TTFB p50=132 ms fillers/153 ms val96; `kokoro_v3_bundle` speaker_sim=0.631
  fillers/0.588 val96, TTFB p50=185 ms fillers/282 ms val96). Everything else is public (model
  weights, PyPI packages). Referenced throughout; no dependency-specific step needed beyond
  `dvc pull` in Step 1.

**Additional ambiguity flag (not a task.json requirement, but affects correctness)**: the task text
says systems are scored "GE2E cosine vs the half-B centroid," but `t0008`'s actual
`build_reference_split(seed=42)` (in `tasks/t0008_tts_eval_harness_baselines/code/harness.py`)
builds the **centroid from half-A** and returns `half_b_paths` as raw file paths, not a centroid. In
t0008's usage, every synthesized system (including `kokoro_v3_bundle`) is scored with
`compute_speaker_sim(synth_wavs, ref_embeddings)` where `ref_embeddings` is the **half-A centroid**;
half-B is used only once, as the raw audio input when checking ElevenLabs's own self-consistency
(half-B clips scored against the half-A centroid). This plan follows the actual t0008 code path —
all new zero-shot systems are scored against the half-A centroid — not the half-B centroid the task
prose literally names, so the reference clips fed to the cloning models (drawn from half-A) and the
clips used to build the scoring centroid (also half-A) are the same split, exactly as t0008 already
validated. This is called out explicitly per `plan_specification.md`'s rule to never silently merge
or discard an ambiguity.

## Approach

**Grounding (no `research/` folder exists for this task — research steps were not run; the findings
below come from this planning pass reading `task_description.md`, the t0008 dependency's code, and
targeted verification of the one technical claim flagged by the task owner as needing validation
before committing to the full run: F5-TTS's reference-audio duration limit).**

**Reuse, do not reimplement, the t0008 harness.** `t0008_tts_eval_harness_baselines` is a completed,
immutable task folder (Key Rule 5); this task's code imports from it (e.g.
`from tasks.t0008_tts_eval_harness_baselines.code.adapters import SynthResult, save_wav`) rather
than copying its logic. Reused building blocks: `harness.build_reference_split`,
`harness.get_prompts_by_set`, `adapters.SynthResult`/`save_wav`, `scoring.build_centroid` /
`compute_speaker_sim` / `compute_duration_ratio` / `compute_wer`, and
`report.compute_variant_metrics` as a pattern to extend (new code adds a `condition` dimension it
does not have). This task writes its own `code/adapters_zeroshot.py`, `code/run_eval_zeroshot.py`,
and `code/report_zeroshot.py` in `tasks/t0018_zero_shot_cloning_calibration/code/` — never editing
t0008's files (Key Rule 5).

**F5-TTS reference-audio limit — the fact the task owner asked to validate before committing to the
full run.** F5-TTS's standard inference pipeline (`preprocess_ref_audio_text` in
`src/f5_tts/infer/utils_infer.py`, used by both `infer_cli.py` and the Gradio demo, confirmed via
the F5-TTS GitHub repository) **auto-crops reference audio longer than ~15 s** (`clip_short=True` by
default) before synthesis. Community reports (SWivid/F5-TTS issue #55) show that reference audio fed
around the model **without** this preprocessing, or manually forced past ~20-30 s, produces
truncated output with missing words and degraded quality — but the standard, documented pipeline
does not error or crash on a long clip; it silently uses only the first ~15 s. This directly affects
`ref_concat` (~30 s, REQ-2): for F5-TTS specifically, the effective reference the model consumes is
only about half of the intended clip, silently. This is a **documentation/labelling correctness
issue, not an installation blocker** — but if undetected it would make the
`ref_condition_effect.png` chart (REQ-13) and the Q4 comparison across systems misleading, since
CosyVoice2 and Chatterbox (no comparable hard cap found in their docs or Hugging Face model cards)
are expected to consume the full ~30 s clip. **Mitigation, before the full 196-prompt run for
F5-TTS**: Step 4's smoke gate includes a dedicated F5-TTS × `ref_concat` check (one synthesis with
the real ~30 s `ref_concat` clip, not a placeholder), inspecting the library's own log output for a
clipping/cropping message and measuring the actual reference duration consumed. The result (full ~30
s vs auto-cropped ~15 s) is recorded in `results/tables.json`'s environment/variant metadata and
called out explicitly in `results/listening_guide.md` and the answer's Q4 discussion — not silently
absorbed into a chart that implies all three systems saw the same 30 s of audio. If synthesis
instead fails outright (exception, not silent cropping) on the unmodified ~30 s clip, the fallback
is to retry with the clip manually pre-trimmed to 15 s for F5-TTS only (documenting the deviation),
and only if that also fails does F5-TTS × `ref_concat` become a null variant (REQ-6/REQ-7 style),
while `ref_single` (~10 s, well under any known F5-TTS limit) proceeds normally.

**Streaming vs whole-utterance TTFB.** CosyVoice2-0.5B documents text-in/audio-out streaming with
latency as low as 150 ms (FunAudioLLM/CosyVoice2-0.5B model card) — TTFB is measured from the first
streamed audio chunk, same as the harness's Kokoro/ElevenLabs adapters. F5-TTS (flow-matching,
non-autoregressive, generates the full mel/waveform in one pass) and the official
`resemble-ai/chatterbox` package (single `generate()` call returning a complete waveform; a
community fork `chatterbox-streaming` exists but is **not** what the task names —
`ResembleAI/chatterbox` is the named system) are both measured as whole-utterance latency and
labelled as such in every table and chart, per the task's own instruction ("TTFB ... whole-utterance
latency for non-streaming, labelled as such").

**Isolated venvs, not shared `pyproject.toml`.** F5-TTS, CosyVoice2, and Chatterbox each pin their
own `torch`/`torchaudio` versions and have known cross-framework dependency conflicts when installed
together. Rather than adding them to the shared `pyproject.toml` (which Key Rule 3 permits but which
would pollute every future task's environment with three large, single-purpose audio packages), each
system gets its own isolated venv on the GPU VM — `.venv-f5tts`, `.venv-cosyvoice2`,
`.venv-chatterbox` — following the existing project convention for single-purpose ML venvs (see
`tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`'s own usage comment:
`.venv-styletts2/bin/python code/build_reference_concat.py`). Each venv's Python interpreter invokes
this task's own `code/run_eval_zeroshot.py --system <name> ...` directly (not through `uv run`,
since these venvs are outside the main `uv` project). No change to the repo's `pyproject.toml` or
`uv.lock` is needed or made.

**Alternatives considered**:

* *Add all three systems' dependencies to the main `pyproject.toml`.* Rejected: high risk of
  `torch`/`torchaudio` version conflicts breaking the main project's Kokoro/StyleTTS2 environment
  for every future task, for packages only this one measurement task needs.
* *Skip the F5-TTS `ref_concat` pre-check and just run the full 196-prompt job.* Rejected per the
  task owner's explicit instruction: if F5-TTS silently mishandles the 30 s reference, discovering
  that after a full ~0.8 h ($11) run wastes budget and produces a mislabelled `ref_condition_effect`
  chart (Lesson 2's exact failure shape — a smoke gate is cheaper than discovering the problem in
  post-mortem).
* *Substitute a different long-reference-tolerant model for F5-TTS if the 30 s clip is a problem.*
  Rejected: the task's Forbidden section explicitly bars silent substitution of a named system; the
  auto-crop behavior is not an installation/smoke-gate failure (synthesis succeeds), so REQ-7's
  substitution allowance does not apply here — this is a documented condition-level caveat, not a
  system failure.

**Task types**: `task_types` in `task.json` is already `["tts-benchmark-run", "answer-question"]` —
both apply and are confirmed correct; no change needed. The `tts-benchmark-run` Planning Guidelines
(`meta/task_types/tts-benchmark-run/instruction.md`) drove the smoke-gate → 50-warmup →
≥100-measured-requests structure and the `[speaker-sim]` extra rule (REQ-17). The `answer-question`
Planning Guidelines (`meta/task_types/answer-question/instruction.md`) drove treating the six Key
Questions as the evidence plan's stopping criterion (Step 17) and the one-answer-asset-per-task
structure (REQ-10), with `answer_methods` including `"code-experiment"` (the harness runs) as the
primary evidence channel plus `"internet"` for the F5-TTS reference-limit finding above.

## Cost Estimation

All compute is GPU wall-clock time on `LLM-T1-NC80` (Azure ML, 2×H100 SXM5, `$13.96`/hour per
`project/azure_vm.json`). No paid API calls: the ElevenLabs baseline is re-scored from t0008's
already-synthesized audio (`tasks/t0008_tts_eval_harness_baselines/data/synth_audio.dvc`, CPU-only
re-scoring, $0) rather than calling the ElevenLabs API; `resemblyzer` (GE2E) and `faster-whisper`
(WER) both run on CPU with no external API. `project/budget.json`'s `available_services` includes
`elevenlabs_api`, but this task makes zero calls to it.

| Item | Estimate | Notes |
| --- | --- | --- |
| Setup + model downloads (3 venvs, 3 model weight sets to `/mnt/cache/persist/pretrained/`) | 0.5 h ≈ $6.98 | Per Lesson 10, all downloads go to the persistent Azure Files mount |
| F5-TTS × 2 conditions × 196 prompts + smoke gate + warmup | 0.4 h ≈ $5.58 | Non-streaming; includes the dedicated `ref_concat` pre-check |
| CosyVoice2 × 2 conditions × 196 prompts + smoke gate + warmup | 0.4 h ≈ $5.58 | Streaming |
| Chatterbox × 2 conditions × 196 prompts + smoke gate + warmup | 0.4 h ≈ $5.58 | Non-streaming |
| (subtotal: 6 cloning variants) | 1.2 h ≈ $16.75 |  |
| `elevenlabs_david` re-scoring (CPU, from stored audio) + `kokoro_v3_bundle` re-synthesis (GPU) + teardown | 0.3 h ≈ $4.19 |  |
| CPU-side scoring/reporting/chart generation while GPU still up (avoids a second billable session) | included above | Runs during the same session, no extra GPU-hours beyond the table above |
| **Total GPU** | **≈3.0-3.2 h ≈ $42-$45** | Matches `task_description.md`'s own $45 estimate |
| **Hard cap (user-authorized)** | **$70** | ≈$25-28 buffer (≈56-65%) over the base estimate |
| Anthropic/OpenAI/ElevenLabs API | $0 | No API calls made by this task |
| DVC/Azure Blob storage | $0 (not separately billed to this task) | Existing `azureblob` remote, already provisioned |

Budget comparison: `project/budget.json` sets `per_task_default_limit: 100.0` against a
`total_budget` of `5000.0`; this task's $70 hard cap is stricter than the default per-task limit and
is the binding constraint. The base estimate leaves real headroom, but three separate model/venv
installs (a common source of dependency-resolution time overruns — see `## Risks & Fallbacks`) are
the most likely way to approach the cap.

## Step by Step

### Milestone 0: Setup and reference audio (Steps 1-3)

1. **[CRITICAL] Pull dependency data and provision the GPU machine.** Run
   `dvc pull tasks/t0008_tts_eval_harness_baselines/data/11labs_david.dvc` (needed for
   `build_reference_split` and reference-clip selection) and
   `dvc pull tasks/t0008_tts_eval_harness_baselines/data/synth_audio.dvc` (t0008's stored ElevenLabs
   audio, for the $0 re-scoring path; if this pull fails, fall back to quoting t0008's stored
   numbers with the session caveat, per `task_description.md`). Also pull
   `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/*.dvc` (the v3 decoder + voicepack
   needed to re-synthesize `kokoro_v3_bundle`). Then follow `/setup-remote-machine` to acquire
   `LLM-T1-NC80`, run `arf/scripts/utils/remote_preflight.sh` (verifies the `/mnt/cache/persist`
   symlink per Lesson 10 and enables `loginctl enable-linger` per Lesson 11), and deploy
   `arf/scripts/utils/idle_watchdog.sh` with `TERMINATE_CMD` set to
   `az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI`,
   confirming the watchdog PID **before** the first model download (Lesson 8, REQ-16). Expected
   output: `dvc pull` reports the pulled files present under
   `tasks/t0008_tts_eval_harness_baselines/data/` and
   `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/`; `ps aux | grep idle_watchdog`
   shows a live PID on the VM. Satisfies REQ-16, REQ-18.

2. **Create `tasks/t0018_zero_shot_cloning_calibration/code/paths.py` and `code/constants.py`.**
   `paths.py` centralizes: `TASK_ROOT`, `DATA_DIR`, `DATA_REFERENCES_DIR = DATA_DIR / "references"`,
   `RESULTS_DIR`, `RESULTS_IMAGES_DIR`,
   `RESULTS_AUDIO_HARNESS_DIR = RESULTS_DIR / "audio_samples" / "harness"`,
   `RESULTS_AUDIO_COMPARISON_DIR = RESULTS_DIR / "audio_samples" / "comparison_set"`,
   `RESULTS_AUDIO_REFERENCES_DIR = RESULTS_DIR / "audio_samples" / "references"`, plus imported path
   constants re-exported from `tasks.t0008_tts_eval_harness_baselines.code.paths`
   (`DATA_11LABS_DAVID_DIR`, `VAL_LIST`, `SYNTH_AUDIO_DIR`, `V3_DECODER`, `V3_VOICEPACK`).
   `constants.py` defines, as `typing.Final` (per `arf/styleguide/python_styleguide.md`):
   `SYSTEM_F5_TTS = "f5_tts"`, `SYSTEM_COSYVOICE2 = "cosyvoice2"`,
   `SYSTEM_CHATTERBOX = "chatterbox"`, `CONDITION_REF_SINGLE = "ref_single"`,
   `CONDITION_REF_CONCAT = "ref_concat"`,
   `CLONING_SYSTEMS = [SYSTEM_F5_TTS, SYSTEM_COSYVOICE2, SYSTEM_CHATTERBOX]`,
   `REFERENCE_CONDITIONS = [CONDITION_REF_SINGLE, CONDITION_REF_CONCAT]`,
   `SMOKE_GATE_TIMEBOX_MINUTES = 45` (REQ-7), `SUCCESS_RATE_THRESHOLD = 0.8` (Lesson 3, REQ-6),
   `GATE_TEXT_NAMES = ("lining_up_suggestions_17", "lining_up_suggestions_10", "putting_them_head_to_head_15")`
   (the same 3 fixed gate texts reused from `tasks/t0014_v11_decoder_fix_retrain` and
   `tasks/t0015_v11_duration_blowup_forensics` for cross-task comparability — their transcript text
   is looked up from `data/filler_prompts_100.json`/`val96_prompts.json` by matching the slugified
   filename stem, the same derivation `harness._load_val96_from_manifest` uses),
   `COMPARISON_SET_SEED = 42`, `COMPARISON_SET_VAL96_COUNT = 7`, `VM_HOURLY_COST_USD = 13.96`.
   Expected output: both files import cleanly
   (`uv run python -c "from tasks.t0018_zero_shot_cloning_calibration.code import constants"`).
   Satisfies REQ-1, REQ-2 (naming groundwork for later steps).

3. **[CRITICAL] Build the two reference-audio conditions.** Create `code/build_references.py`,
   adapted from `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`'s concatenation
   approach (0.2 s silence gaps between clips, `soundfile` read/write) but sourcing clips from
   **half-A of the David corpus** via
   `tasks.t0008_tts_eval_harness_baselines.code.harness.build_reference_split(corpus_dir=DATA_11LABS_DAVID_DIR, seed=42)`
   (import `RANDOM_SEED=42` from t0008's `constants.py`) — this call also returns the half-A
   embeddings needed for scoring (see the ambiguity note above). From the half-A file list: (a)
   `ref_single` = the single longest clip under ~10 s (compute each half-A clip's duration with
   `soundfile.info`, pick the longest with `duration <= 10.0`; if none qualifies, pick the shortest
   clip `> 10.0` and log a warning), written to `data/references/ref_single.wav` and copied to
   `results/audio_samples/references/ref_single.wav`; (b) `ref_concat` = concatenate half-A clips
   with 0.2 s silence gaps, in filename-sorted order, stopping once total duration first exceeds
   30.0 s (do not truncate the last clip), written to `data/references/ref_concat.wav` and copied to
   `results/audio_samples/references/ref_concat.wav`. Write `data/references/manifest.json`
   recording: `ref_single_filename`, `ref_single_duration_s`, `ref_concat_filenames` (ordered list),
   `ref_concat_duration_s`, `seed=42`. Expected output: `data/references/manifest.json` exists with
   both durations in the expected ranges (`ref_single_duration_s` in `[8, 12]`,
   `ref_concat_duration_s` in `[28, 34]`). Satisfies REQ-2, REQ-8 (references audio deliverable).

### Milestone 1: Install, smoke-gate, and the F5-TTS ref_concat pre-check (Step 4)

4. **[CRITICAL] Install each cloning system into its own venv and run the smoke gate, including the
   dedicated F5-TTS `ref_concat` pre-check.** On `LLM-T1-NC80`, under `/mnt/cache/persist/` (Lesson
   10): create `.venv-f5tts` (`pip install f5-tts` or clone `SWivid/F5-TTS` and `pip install -e .`),
   `.venv-cosyvoice2` (clone `FunAudioLLM/CosyVoice2-0.5B` per its Hugging Face model card, install
   its `requirements.txt`), `.venv-chatterbox` (`pip install chatterbox-tts`, the official
   `resemble-ai/chatterbox` package — **not** the `chatterbox-streaming` fork, since the named
   system `ResembleAI/chatterbox` is the non-streaming official package). Download each model's
   weights to `/mnt/cache/persist/pretrained/<system>/`. For **each** system, time-box installation
   \+ smoke gate at 45 minutes (`SMOKE_GATE_TIMEBOX_MINUTES`, REQ-7): run one synthesis with
   `ref_single` and confirm non-empty audio output. **For F5-TTS specifically, also run one
   synthesis with the actual `ref_concat` clip (~30 s, from Step 3) before any full-scale run for
   that system** — inspect the process log for a clipping/cropping message from
   `preprocess_ref_audio_text` (F5-TTS's standard pipeline auto-crops reference audio to ~15 s by
   default) and record whether the full ~30 s or only ~15 s was effectively consumed; check the
   resulting clip with
   `tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check.check_audio_quality` (import
   directly, do not copy) to confirm it is not garbled/truncated speech. If this check fails
   outright (exception or `is_likely_noise=True`), retry once with the reference clip manually
   pre-trimmed to 15 s for F5-TTS only, documenting the deviation in `results/tables.json`'s notes
   field; if that retry also fails, F5-TTS × `ref_concat` becomes a null variant (REQ-6 style) and
   only F5-TTS × `ref_single` proceeds to the full run. If a system fails to install or fails its
   `ref_single` smoke gate within the 45-minute time-box, write
   `intervention/<system>_smoke_gate_failed.md` with the exact error, mark that system null in
   `results/tables.json`, and continue with the remaining systems (REQ-7) — do not substitute
   silently. Expected output: a short `results/smoke_gate_log.md` recording, per system: install
   time, smoke-gate pass/fail, and (F5-TTS only) the effective `ref_concat` duration consumed.
   Satisfies REQ-1, REQ-2, REQ-7.

### Milestone 2: Full synthesis runs (Steps 5-8)

5. **[CRITICAL] Write `code/adapters_zeroshot.py`.** Following the exact `SynthResult` contract from
   `tasks.t0008_tts_eval_harness_baselines.code.adapters` (import `SynthResult`, `save_wav`,
   `_resample_to_16k`-equivalent behavior), implement
   `f5_tts_synth(text, *, ref_wav_path, ref_text, model) -> SynthResult`,
   `cosyvoice2_synth(text, *, ref_wav_path, prompt_text, model) -> SynthResult` (measures TTFB from
   the first streamed chunk, matching `_kokoro_synth_via_pipeline`'s chunk-timing pattern), and
   `chatterbox_synth(text, *, ref_wav_path, model) -> SynthResult` (whole-utterance timing, matching
   `elevenlabs_david`'s single-call timing pattern but without HTTP). Also implement
   `load_f5tts_model()`, `load_cosyvoice2_model()`, `load_chatterbox_model()` loaders (one per venv,
   called only from that venv's Python process). Also capture, at synthesis time (Lesson 4):
   `model_commit_or_version`, `torch.__version__`, `torch.version.cuda`, written to
   `results/environment.json` keyed by system. Expected output:
   `uv run pytest tasks/t0018_zero_shot_cloning_calibration/code/test_zeroshot.py -v -k adapters`
   passes (Step 9 writes this test file). Satisfies REQ-1, REQ-4, REQ-14 (environment table).

6. **[CRITICAL] Write `code/run_eval_zeroshot.py`.** CLI entry point analogous to t0008's
   `code/run_eval.py`:
   `--system {f5_tts,cosyvoice2,chatterbox,elevenlabs_david,kokoro_v3_bundle} --condition {ref_single,ref_concat} --prompt-set both --n-warmup 50 --limit N`.
   For each (system, condition) pair: load the model once, run 50 discarded warmup requests using
   the warmup text `"Checking the latest press release."` (same text t0008 uses, for consistency),
   then synthesize all 196 measured prompts (from
   `tasks.t0008_tts_eval_harness_baselines.code.harness.get_prompts_by_set("both")`), saving each
   clip to `results/audio_samples/harness/<system>_<condition>/<prompt_set>/<index>.wav` (REQ-8) and
   appending a per-clip record (schema matching t0008's: `system`, `condition`, `prompt_set`,
   `text`, `ttfb_ms`, `rtf`, `speaker_sim: null`, `duration_ratio: null`, `wer: null`, `audio_path`,
   `ref_duration_s`, `synth_duration_s`, `is_streaming: bool`) to
   `results/per_clip_metrics_<system>.json`. On a per-clip synthesis exception, record a null-metric
   row (matching t0008's failure-row shape) rather than aborting the whole run — this is what makes
   the Step 13 `successful_prompts / total_prompts` count possible. Expected output: for a
   `--limit 10` smoke run, ≥8/10 successful rows per (system, condition, prompt_set) before
   committing to the full 196-prompt run (validation gate — trivial baseline: a working synth call
   should never fail more than 20% of the time on well-formed short prompts; if the small run is
   below this, stop and read 5 individual failures before scaling up). Satisfies REQ-1, REQ-2,
   REQ-4, REQ-11.

7. **Run the 6 cloning variants.** Execute, from each system's own venv,
   `<venv>/bin/python -m tasks.t0018_zero_shot_cloning_calibration.code.run_eval_zeroshot --system <system> --condition <condition> --prompt-set both --n-warmup 50`
   for all 3 systems × 2 conditions (skip any variant marked null in Step 4). Each invocation
   wrapped in
   `arf/scripts/utils/run_with_logs.py --task-id t0018_zero_shot_cloning_calibration -- <cmd>` (Key
   Rule 1). Expected output: 6 files `results/per_clip_metrics_<system>.json` (or fewer if a variant
   is null); each covers 2 conditions × 196 prompts = 392 rows per system (196 per condition), minus
   any per-clip failures. Satisfies REQ-1, REQ-2, REQ-4, REQ-11.

8. **[CRITICAL] Re-run the two paired baselines in the same session.** (a) `elevenlabs_david`: if
   `tasks/t0008_tts_eval_harness_baselines/data/synth_audio.dvc` was pulled successfully in Step 1,
   re-score the existing WAVs at
   `tasks/t0008_tts_eval_harness_baselines/data/synth_audio/elevenlabs_david/{val96,fillers}/*.wav`
   with `scoring.compute_speaker_sim` against this task's half-A centroid (Step 3) — this is
   CPU-only, $0, no ElevenLabs API call; if the pull failed, skip this and instead quote t0008's
   stored numbers (speaker_sim=0.832 fillers/0.792 val96) in `results/tables.json` with an explicit
   `"source": "t0008 stored, not re-measured this session"` field and omit `ttfb_ms`/`rtf` for this
   variant rather than falsely pairing a different session's latency with the new systems' latency
   (Lesson 1). (b) `kokoro_v3_bundle`: re-synthesize fresh on the GPU this session using
   `tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline.build_pipeline` +
   `tasks.t0008_tts_eval_harness_baselines.code.adapters.load_kokoro_model_with_checkpoint(V3_DECODER)`
   + `adapters.kokoro_v3_bundle(text, pipeline=pipeline, voicepack_path=V3_VOICEPACK)`, same
     50-warmup
   + 196-prompt protocol as Step 6, saved to `results/per_clip_metrics_kokoro_v3_bundle.json`.
     Expected output: both baseline files exist; `kokoro_v3_bundle`'s new TTFB p50 is within a
     plausible range of t0008's 185 ms (fillers) — if it diverges by more than 2x, treat as a red
     flag and inspect before trusting downstream deltas. Satisfies REQ-3, REQ-18.

### Milestone 3: Scoring and the audible-speech gate (Steps 9-12)

9. **Write `code/test_zeroshot.py`.** Task-specific tests (per CLAUDE.md: task tests live in
   `tasks/$TASK_ID/code/test_*.py`, never a top-level `tests/`): (a) `build_references.py`'s
   `ref_concat` duration lands in `[28, 34]` s on a synthetic fixture corpus; (b) the gate-text
   lookup resolves all 3 `GATE_TEXT_NAMES` to non-empty text; (c) the comparison-set sampler with
   `COMPARISON_SET_SEED=42` is deterministic across two calls; (d) `report_zeroshot.py`'s
   variant-metrics builder (Step 13) produces the 3-dimension (`system`, `condition`, `prompt_set`)
   `variant_id` format on a small synthetic records list. Run
   `uv run pytest tasks/t0018_zero_shot_cloning_calibration/code/test_zeroshot.py -v`; expected
   output: all tests pass, 0 failures.

10. **Merge per-system files and compute GE2E speaker similarity.** Concatenate all
    `results/per_clip_metrics_<system>.json` files (from Steps 7-8) plus t0008's re-scored
    ElevenLabs rows into `results/per_clip_metrics.json`. For every row with a non-null
    `audio_path`, compute `speaker_sim` via
    `tasks.t0008_tts_eval_harness_baselines.code.scoring.compute_speaker_sim([audio_path], half_a_centroid)`
    (the half-A centroid built in Step 3), updating the row in place. Expected output:
    `results/per_clip_metrics.json` has ≥1568 rows (8 variants × 196, minus any null variants/failed
    clips) with `speaker_sim` populated for all successful clips. Satisfies REQ-4, REQ-11.

11. **Compute duration ratio and WER.** For every row with a non-null `audio_path` and a matched
    reference (val96 rows only have `ref_duration_s`; filler rows use the ElevenLabs reference
    clip's own duration if available, else are skipped for duration ratio per t0008's convention):
    `scoring.compute_duration_ratio` and `scoring.compute_wer` (the latter gated by
    `DURATION_RATIO_LOW`/`DURATION_RATIO_HIGH` from t0008's `constants.py`, exactly as t0008 does).
    Update `results/per_clip_metrics.json` in place. Expected output: `wer` and `duration_ratio`
    populated for all non-gated clips; spot-check 5 individual WER values against their transcripts
    to confirm the STT model and normalization are behaving sanely before trusting the aggregate
    (validation gate — WER should not be uniformly ~1.0, which would indicate a broken transcription
    pipeline rather than genuinely bad TTS). Satisfies REQ-4.

12. **Run the audible-speech gate on every clip.** Create `code/run_gate_check.py`: for every row in
    `results/per_clip_metrics.json` with a non-null `audio_path`, call
    `tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check.check_audio_quality(Path(audio_path), text=row["text"])`
    (imported directly — Key Rule 5 forbids copying/modifying t0015's module). Combine
    `is_likely_noise`, `duration_sanity_pass is False`, and
    `longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S` into a per-clip
    `hardened_gate_pass` boolean (the same combination `t0015`'s own `run_gate_regression.py` uses).
    Write `results/gate_failures.json`: per-variant failure count and the list of failing
    `audio_path`s. Expected output: `results/gate_failures.json` exists with a failure count for
    every one of the (up to) 8 variants, including 0 for variants with no failures. Satisfies REQ-5.

### Milestone 4: Reporting (Steps 13-17)

13. **Write `code/report_zeroshot.py` and produce `results/metrics.json` / `results/tables.json`.**
    Extend t0008's `report.compute_variant_metrics(records, system, prompt_set)` pattern to a third
    dimension: `compute_variant_metrics_zeroshot(records, system, condition, prompt_set)`, filtering
    on all three keys (cloning systems have a `condition`; the two baselines do not — use
    `condition=None` for those two, and format their `variant_id` as `<system>_<prompt_set>` to
    match t0008's existing convention so cross-task comparison stays possible). Apply the Rejection
    Criteria (below): any variant with `n_successful / n_clips < 0.8` gets `null` for every metric
    in that variant, with a `"rejected_reason": "successful_requests/total_requests < 0.8"` field.
    Write `results/metrics.json` in the **explicit variant format**
    (`arf/specifications/metrics_specification.md` / `task_results_specification.md`), each
    variant's `metrics` object containing **only** the 3 registered keys `speaker_sim`, `ttfb_ms`,
    `rtf` (per the REQ-12 ambiguity resolution above). Write `results/tables.json` with the full
    per-variant row (registered + non-registered: `wer_mean`, `duration_ratio_median`,
    `duration_explosion_fraction`, gate-failure count from Step 12,
    `efficiency_inference_time_per_item_seconds` = wall-clock synthesis time for that variant's
    session ÷ `n_clips`, `efficiency_inference_cost_per_item_usd` =
    `VM_HOURLY_COST_USD × wall_clock_hours / n_clips`, plus the environment fields from
    `results/environment.json` (Step 5) and the F5-TTS `ref_concat` effective-duration caveat from
    Step 4). Expected output: `results/metrics.json` has one `variants` entry per non-null (system,
    condition, prompt_set) triple plus the 2 baselines × 2 prompt sets; every entry's `metrics` has
    exactly the 3 registered keys. Satisfies REQ-6, REQ-12, REQ-14.

14. **Generate the four required charts.** In `code/report_zeroshot.py`, adapt t0008's
    `report.plot_speaker_sim_boxplot`/`plot_ttfb_cdf` plotting patterns (same
    `matplotlib.use("Agg")` setup) to produce: `results/images/speaker_sim_by_system.png` (grouped
    bars, x=system, grouped by prompt_set fillers/val96, horizontal reference lines at 0.832/0.792
    [ElevenLabs] and 0.631/0.588 [kokoro_v3_bundle]); `results/images/ttfb_vs_speaker_sim.png`
    (scatter, one point per variant — cloning variants colored by system with condition as marker
    shape, baselines highlighted — x=TTFB p50 ms with a vertical line at 300, y=fillers
    speaker_sim); `results/images/ref_condition_effect.png` (paired bars per cloning system,
    `ref_single` vs `ref_concat`, fillers speaker_sim); `results/images/wer_by_system.png` (bar
    chart, mean WER per system/condition). Expected output: all 4 PNGs exist under `results/images/`
    and are non-zero-byte. Satisfies REQ-13.

15. **Build the comparison audio set.** Create `code/build_comparison_set.py`: resolve the 10 fixed
    comparison texts (3 `GATE_TEXT_NAMES` + 7 val96 prompts sampled with
    `random.Random(COMPARISON_SET_SEED).sample(val96_texts, COMPARISON_SET_VAL96_COUNT)`, sampling
    from the val96 prompt list in the same order `harness.load_val96_prompts()` returns it, for
    reproducibility). For each of the 10 texts, copy the corresponding clip from
    `results/audio_samples/harness/<system>_<condition>/<prompt_set>/<index>.wav` for every non-null
    variant, plus the matching ElevenLabs original
    (`tasks/t0008_tts_eval_harness_baselines/data/synth_audio/elevenlabs_david/...` or
    `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` if the text has a natural reference
    clip) and the `kokoro_v3_bundle` output, into
    `results/audio_samples/comparison_set/<text_id>__<system>_<condition>.wav` (and
    `<text_id>__elevenlabs_david.wav`, `<text_id>__kokoro_v3_bundle.wav`). Expected output:
    `results/audio_samples/comparison_set/` contains 10 texts × up to 8 variants = up to 80 files
    (fewer if any variant is null). Satisfies REQ-8.

16. **Write `results/listening_guide.md`.** One row per comparison text (10 rows), one column per
    system/condition (up to 6) plus ElevenLabs and `kokoro_v3_bundle`, each cell a clickable
    relative markdown link to the file from Step 15 with the per-clip `speaker_sim`, `wer`, and gate
    verdict (`hardened_gate_pass` from Step 12) in the cell text, plus one "what to listen for" line
    per row (timbre match, accent drift, brand-name pronunciation, artifacts — written from the
    actual per-row data, e.g. flag a row where WER is high specifically on a brand-name word).
    Follow `arf/styleguide/markdown_styleguide.md` (100-char lines, `*` bullets). Run
    `uv run flowmark --inplace --nobackup tasks/t0018_zero_shot_cloning_calibration/results/listening_guide.md`.
    Expected output: the file renders as a markdown table on GitHub with working relative links.
    Satisfies REQ-8.

17. **[CRITICAL] Write the answer asset.** Create
    `assets/answer/zero-shot-speaker-sim-ceiling/details.json` (`spec_version: "2"`, `question`:
    "What speaker_sim and latency envelope is reachable on the David voice by zero-shot cloning, and
    what does that imply for the Kokoro fine-tuning line?", `short_title`: "Zero-shot cloning's
    speaker_sim/latency ceiling on David", `short_answer_path: "short_answer.md"`,
    `full_answer_path: "full_answer.md"`, `categories: []` (no categories are defined in
    `meta/categories/` for this project — matches t0008's own empty-categories precedent),
    `answer_methods: ["code-experiment", "internet"]`,
    `source_task_ids: ["t0008_tts_eval_harness_baselines"]`, `source_urls`: the
    F5-TTS/CosyVoice2/Chatterbox repo/model-card URLs actually used for installation, `confidence`:
    `"high"` if all 3 systems completed both conditions with ≥0.8 success rate, `"medium"` if any
    system is null or any condition was rejected,
    `created_by_task: "t0018_zero_shot_cloning_calibration"`, `date_created`: the actual completion
    date). Write `short_answer.md` (`## Question`, `## Answer` — 2-5 sentences, state the highest
    reached speaker_sim number and system/condition, compare to the 0.832/0.792 ElevenLabs ceiling
    and the 0.631/0.588 `kokoro_v3_bundle` number, state the restated-criterion recommendation
    directly, no hedging, no `[[tNNNN]]`-style inline citations — `## Sources`). Write
    `full_answer.md` as a mini-paper answering all six Key Questions (REQ-9) with tables from
    `results/tables.json`, explicitly discussing the F5-TTS `ref_concat` caveat from Step 4, and a
    `## Sources` section with markdown reference-link definitions per
    `meta/asset_types/answer/specification.md`. Expected output:
    `uv run python -u -m meta.asset_types.answer.verificator --task-id t0018_zero_shot_cloning_calibration zero-shot-speaker-sim-ceiling`
    passes with 0 errors. Satisfies REQ-9, REQ-10.

### Milestone 5: DVC push and teardown (Steps 18-19)

18. **DVC-track and push all audio.** Run
    `dvc add results/audio_samples/harness results/audio_samples/comparison_set results/audio_samples/references`
    (three `.dvc` pointer files committed to git; the raw WAVs are gitignored per CLAUDE.md's DVC
    workflow), then `dvc push` **before** the task's PR is opened. Expected output: `dvc status`
    reports nothing to push after the push completes; `git status` shows only `.dvc` files and
    `.gitignore` changes under `results/audio_samples/`, no raw `.wav` files staged. Satisfies
    REQ-8, REQ-17 (data-handling rule).

19. **[CRITICAL] Teardown.** Confirm all measurement work is complete and no `in_progress` step
    remains for this task, then run the orchestrator's teardown path
    (`azure_ml_vm.teardown(task_id="t0018_zero_shot_cloning_calibration", deallocate=True)`), which
    clears the task lock and calls `stop_compute(vm="LLM-T1-NC80")`. Verify with
    `az ml compute show --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI --query "provisioningState"`
    that the VM is stopped, and confirm the idle watchdog process is no longer needed (VM stopped).
    Expected output: compute state is `Stopped`/deallocated; total logged GPU wall-clock time ×
    $13.96 is under the $70 hard cap. Satisfies REQ-16.

## Remote Machines

**Required.** `gpu_class: H100`, `provider: azure_ml` (the only provider serving this class per
`project/azure_vm.json`'s single-VM pool, so no ambiguity requiring an explicit choice between
providers — `azure_ml` is still named explicitly per the routing rule). VM: `LLM-T1-NC80` (2×H100
SXM5 NVL, $13.96/hour). Estimated runtime: ≈3.0-3.2 hours total (setup, 6 cloning variants, 2
baselines, scoring/reporting done while the VM is still up, teardown). VRAM requirement: F5-TTS,
CosyVoice2-0.5B, and Chatterbox are each well under 24 GB VRAM per published requirements
(Chatterbox ~8-16 GB, CosyVoice2 0.5B params, F5-TTS comparable) — a single H100 (80 GB) comfortably
runs them sequentially; no multi-GPU parallelism is needed or planned. Watchdog and
persistent-storage symlink verification are mandatory before any model download (Lessons 8, 10, 11;
Step 1).

## Assets Needed

* **Dependency task `t0008_tts_eval_harness_baselines`**: the `tts_eval_harness` library
  (`assets/library/tts_eval_harness/`) — `code/harness.py`, `code/adapters.py`, `code/scoring.py`,
  `code/report.py` and their `paths.py`/`constants.py`; the prompt-set manifests
  `data/val96_prompts.json` and `data/filler_prompts_100.json`; the ElevenLabs reference corpus
  `data/11labs_david/` (DVC); the stored synthesized audio `data/synth_audio/` (DVC, for $0
  ElevenLabs re-scoring); the paired baseline numbers quoted throughout this plan.
* **`t0006_kokoro_v5_stage2_subset`** (transitive, via t0008's `paths.py`): the v3 decoder +
  voicepack (`data/reference/v3/best/*.dvc`) needed to re-synthesize `kokoro_v3_bundle`.
* **`t0015_v11_duration_blowup_forensics`**: `code/audio_quality_check.py` (imported, not copied)
  for the audible-speech gate.
* **`t0014_v11_decoder_fix_retrain`**: `code/build_reference_concat.py` as the pattern (not the code
  itself, since it sources different clips) for building `ref_concat`.
* **External, public**: F5-TTS weights (`SWivid/F5-TTS` base English checkpoint, Hugging Face),
  CosyVoice2 weights (`FunAudioLLM/CosyVoice2-0.5B`, Hugging Face), Chatterbox weights
  (`ResembleAI/chatterbox`, Hugging Face), and their respective PyPI/source installations. No
  authentication/paid access required for any of the three.

## Expected Assets

* **`answer`** (1, matching `task.json` `expected_assets.answer: 1`):
  `assets/answer/zero-shot-speaker-sim-ceiling/` — answers "What speaker_sim and latency envelope is
  reachable on the David voice by zero-shot cloning, and what does that imply for the Kokoro
  fine-tuning line?" with `short_answer.md` and `full_answer.md` per
  `meta/asset_types/answer/specification.md`.
* **Non-asset-registry results** (required by `task_description.md` but not a registered asset
  type): `results/per_clip_metrics.json`, `results/metrics.json`, `results/tables.json`,
  `results/gate_failures.json`, `results/environment.json`, `results/smoke_gate_log.md`,
  `results/listening_guide.md`, `results/images/*.png` (4 charts), and the DVC-tracked
  `results/audio_samples/{harness,comparison_set,references}/`.
* **`results/suggestions.json`** is produced by the orchestrator's suggestions step, not by this
  plan's Step by Step (see REQ-15's note); this plan records the content the orchestrator should
  include (restated success criterion; production-integration feasibility task if a system meets
  both bars).

## Time Estimation

* Research: already skipped for this task (no `research/` folder; grounding done during this
  planning pass via targeted verification of the F5-TTS reference-limit claim).
* Planning (this document): complete.
* Implementation, Milestone 0 (setup, reference audio): ≈0.5-0.75 h wall clock (overlaps with GPU
  setup time already counted in Cost Estimation).
* Implementation, Milestone 1 (install + smoke gates, 3 systems): ≈1-2 h wall clock (dependency
  installation is not fully parallelizable across venvs on one VM without risking contention; GPU
  billing during this milestone is the 0.5 h "Setup + model downloads" line item).
* Implementation, Milestones 2-3 (synthesis + scoring, 6 variants + 2 baselines): ≈2 h wall clock,
  matching the ≈1.5 h of GPU-billed synthesis time in Cost Estimation plus CPU-only scoring
  overlapping the same session.
* Implementation, Milestone 4 (reporting: metrics, charts, comparison set, listening guide, answer
  asset): ≈1-1.5 h wall clock, CPU-only, can continue after GPU teardown (Step 19) since none of it
  needs the GPU.
* Implementation, Milestone 5 (DVC push, teardown): ≈15-30 min.
* Validation/verification (Phase 4 of this skill plus post-implementation verificators): ≈30 min.
* **Total wall clock**: ≈5-7 hours end to end; **GPU-billed portion**: ≈3.0-3.2 hours (≈$42-45).

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| F5-TTS silently auto-crops the 30 s `ref_concat` clip to ~15 s, making the ref-condition-effect comparison across systems misleading if undetected | High (confirmed behavior of the standard pipeline) | Medium — wrong conclusion in Q4/chart, not a budget or crash risk | Step 4's dedicated F5-TTS × `ref_concat` smoke-gate check, before the full run, logs the effective duration consumed and this is carried through to every downstream table/chart caption (Approach section) |
| One or more of F5-TTS/CosyVoice2/Chatterbox has undocumented dependency conflicts (common with bleeding-edge `torch`/`torchaudio`/`transformers` pins across 3 separate research repos) that exceed the 45-minute smoke-gate time-box | Medium | Medium — burns Setup budget, risks the $70 cap if all 3 hit this | Isolated venvs per system (Approach) limit blast radius to one system; 45-min hard time-box (REQ-7) with a null result + intervention file rather than open-ended debugging; if 2 of 3 systems fail to install, the task still produces a valid (if narrower) answer from the surviving system(s) plus the two baselines |
| Total GPU wall-clock exceeds the $45 base estimate and approaches or exceeds the $70 hard cap (e.g. from repeated install retries, or CosyVoice2's LLM component being slower than the flow-matching-only systems) | Medium | High — hard cap violation is a stop condition | Cost Estimation's ≈$25-28 buffer; monitor cumulative wall-clock against $70 at each milestone boundary (end of Milestone 1, end of Milestone 2) and stop/escalate via intervention file rather than continuing past the cap; scoring/reporting (Milestone 3-4) is CPU-only and can run after teardown if GPU time is running tight, decoupling report-writing time from the GPU cost clock |
| A system's smoke gate passes but the full 196-prompt run has a much lower success rate (e.g. rate-limiting-like backpressure, GPU OOM under sustained load, or a text/prompt edge case not exercised by the single smoke prompt) | Low-Medium | Medium — could trigger the REQ-6 null-variant rule after spending the GPU time on that variant | Step 6's `--limit 10` validation gate before the full run catches this early for each variant; per-clip exception handling (Step 6) means one bad prompt does not abort the whole variant |
| Idle GPU billing from a missed wakeup or fire-and-forget handoff between the GPU-bound steps (4-8) and the CPU-only steps (9-17), given the multi-hour wall-clock span | Low (framework-mitigated) | High if it occurs (Lesson 8's $130 incident) | Watchdog armed before first download with `IDLE_THRESHOLD_SECONDS=3600` (Step 1, Lesson 8); the plan explicitly separates GPU-bound work (Milestones 0-2, ends with Step 8) from CPU-only work (Milestones 3-4) so teardown (Step 19) can happen as soon as Step 8 completes rather than waiting on reporting |
| The `elevenlabs_david` `synth_audio.dvc` pull fails or the data was garbage-collected from the DVC remote | Low | Low — REQ-3 has an explicit documented fallback | `task_description.md` and Step 8(a) both specify the fallback: quote t0008's stored numbers with an explicit "not re-measured this session" caveat and omit paired latency for that baseline rather than fabricating a re-measurement |

## Verification Criteria

* Run `uv run python -u -m arf.scripts.verificators.verify_plan t0018_zero_shot_cloning_calibration`
  — expect 0 errors (this plan's own structural check).
* Run `uv run pytest tasks/t0018_zero_shot_cloning_calibration/code/test_zeroshot.py -v` — expect
  all tests to pass, 0 failures (Step 9).
* Run
  `python3 -c "import json; d=json.load(open('tasks/t0018_zero_shot_cloning_calibration/results/per_clip_metrics.json')); print(len(d)); assert len(d) >= 100"`
  — expect the printed count to be at least 100, and for a manual per-system breakdown
  (`{r['system'] for r in d}`) to show ≥100 rows for every system that was not marked null in Step 4
  (REQ-11).
* Run
  `python3 -c "import json; d=json.load(open('tasks/t0018_zero_shot_cloning_calibration/results/metrics.json')); assert 'variants' in d; assert all(set(v['metrics'].keys()) <= {'speaker_sim','speaker_sim_std','ttfb_ms','ttfb_ms_p50','ttfb_ms_p95','ttfb_ms_p99','rtf','rtf_std'} for v in d['variants'])"`
  — expect no assertion error, confirming `results/metrics.json` contains only registered metric
  keys in each variant's `metrics` object (REQ-12 ambiguity resolution).
* Run
  `uv run python -u -m meta.asset_types.answer.verificator --task-id t0018_zero_shot_cloning_calibration zero-shot-speaker-sim-ceiling`
  — expect 0 errors (REQ-9, REQ-10).
* Run
  `uv run python -u -m arf.scripts.verificators.verify_task_results t0018_zero_shot_cloning_calibration`
  — expect 0 errors.
* Confirm `results/images/speaker_sim_by_system.png`, `ttfb_vs_speaker_sim.png`,
  `ref_condition_effect.png`, and `wer_by_system.png` all exist and are non-zero-byte via
  `ls -la tasks/t0018_zero_shot_cloning_calibration/results/images/*.png` — expect all four expected
  filenames present with nonzero size (REQ-13).
* Confirm DVC tracking with
  `dvc status tasks/t0018_zero_shot_cloning_calibration/results/audio_samples/harness.dvc tasks/t0018_zero_shot_cloning_calibration/results/audio_samples/comparison_set.dvc tasks/t0018_zero_shot_cloning_calibration/results/audio_samples/references.dvc`
  — expect "Data and pipelines are up to date" after `dvc push` (REQ-8, REQ-17).
* **Requirement coverage check**:
  `grep -c "REQ-" tasks/t0018_zero_shot_cloning_calibration/results/results_detailed.md` (produced
  by the orchestrator's reporting step, not this plan) should be ≥18, confirming every `REQ-1`
  through `REQ-18` item from this plan's checklist is traceable in the final results document.

## Rejection Criteria

Pre-registered before any measurement run, per Lesson 3 and this plan's REQ-6:

* **Default rule**: for any (system, condition, prompt_set) variant, if
  `successful_prompts / total_prompts < 0.8`, that variant's `speaker_sim`, `ttfb_ms`, and `rtf` are
  reported as `null` in `results/metrics.json` regardless of what the successful subset's numbers
  look like, with `"rejected_reason": "successful_requests/total_requests < 0.8"` recorded in
  `results/tables.json`. This threshold cannot be loosened after the run to rescue a borderline
  result.
* **Smoke-gate failure (REQ-7)**: a system that fails installation or its `ref_single` smoke gate
  within the 45-minute time-box (Step 4) is null for **all** of its variants, with the exact error
  recorded in an `intervention/<system>_smoke_gate_failed.md` file — not silently omitted from the
  results tables.
* **F5-TTS `ref_concat` special case**: this variant is null only if synthesis fails outright even
  after the 15 s manual-trim retry (Step 4) — the auto-crop-to-15s behavior alone, if synthesis
  succeeds and passes the audible-speech gate, is **not** grounds for rejection; it is a documented
  caveat on that one variant's comparability, not a null result.
* **Gate failures do not by themselves null a variant**: per `task_description.md`, gate-failing
  clips are excluded from the `speaker_sim` mean only when explicitly stated in the table caption,
  with the unfiltered mean shown alongside — a high gate-failure count is evidence to report, not a
  rejection trigger on its own (unless it also drags `successful_prompts / total_prompts` below 0.8,
  in which case the default rule above applies).
