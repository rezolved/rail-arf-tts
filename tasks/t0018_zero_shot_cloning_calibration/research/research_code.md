---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
research_stage: "code"
tasks_reviewed: 15
tasks_cited: 5
libraries_found: 2
libraries_relevant: 1
date_completed: "2026-09-17"
status: "complete"
---
## Task Objective

t0018 benchmarks three open zero-shot voice-cloning TTS models — F5-TTS (`SWivid/F5-TTS`), CosyVoice
2 (`FunAudioLLM/CosyVoice2-0.5B`), and Chatterbox (`ResembleAI/chatterbox`) — on David reference
audio, using the existing t0008 `tts_eval_harness` protocol (val_96 + 100 filler prompts, smoke
gate, 50 discarded warmup requests, GE2E cosine `speaker_sim`, `ttfb_ms`, `rtf`, WER, duration
ratio). Each system is run under two reference-audio conditions (`ref_single`, a ~10 s half-A clip;
`ref_concat`, a ~30 s half-A concatenation) drawn only from `build_reference_split(seed=42)`'s
half-A, so the half-B centroid used for scoring is never contaminated. The task also re-scores the
`elevenlabs_david` self-consistency ceiling and the `kokoro_v3_bundle` fine-tune baseline in the
same session for paired comparison, runs the t0015 hardened audible-speech gate on every clip, and
produces a mandatory human-listening deliverable (`results/listening_guide.md` plus indexed audio
under `results/audio_samples/`). This is a measurement task only — no fine-tuning, no tuning on
val_96 — and its output recalibrates the project's `speaker_sim >= 0.85` success criterion.

## Library Landscape

The library aggregator (`aggregate_libraries --format json --detail short`) returns exactly 2
registered libraries, no corrections or replacements applied to either:

* **`tts_eval_harness`** (v0.1.0, created by `t0008_tts_eval_harness_baselines`, categories
  `evaluation`/`tts`/`benchmark`) — **relevant, primary dependency**. Provides the entire
  measurement protocol t0018 must reuse verbatim for comparability: `run_eval.py` synthesis+timing
  CLI, `score_speaker_sim.py` GE2E scoring CLI, `report.py` aggregation/chart CLI,
  `harness.build_reference_split()`,
  `scoring.compute_speaker_sim/compute_wer/compute_duration_ratio`, and the `adapters.SynthResult`
  dataclass convention every new system adapter must match. Import path:
  `from tasks.t0008_tts_eval_harness_baselines.code.<module> import <name>` (e.g.
  `from tasks.t0008_tts_eval_harness_baselines.code.harness import build_reference_split`). This is
  a direct `dependencies` entry in `task.json`.
* **`t0009_training_safeguards`** (v0.1.0, created by `t0009_stage2_training_failure_forensics`, no
  categories) — **not relevant**. JSONL step logger, per-epoch checkpoint manager, health gates, and
  run-config capture for Kokoro StyleTTS2 Stage 2 *training*. t0018 runs no training (Forbidden
  section of `task_description.md` explicitly bars fine-tuning), so nothing in this library applies.

The answer aggregator (`aggregate_answers --format json --detail short`) returns exactly 1 answer
asset, `t0009-stage2-forensics-answer` ("Why does Kokoro-82M Stage 2 fine-tuning diverge...", high
confidence, created by t0009). It is **not relevant** to t0018 — it diagnoses a training-checkpoint
loading/`joint_epoch` bug, and t0018 does not train anything or load StyleTTS2 checkpoints of that
kind.

## Key Findings

### The `tts_eval_harness` adapter pattern is the mandatory shape for the three new systems [t0008]

Every system in the harness is a pure function `(text: str, **kwargs) -> SynthResult`
(`tasks/t0008_tts_eval_harness_baselines/code/adapters.py`, 330 lines), where `SynthResult` (frozen
dataclass, `adapters.py:36-43`) carries `audio_array_16khz`, `audio_array_native`,
`native_sample_rate`, `ttfb_s`, `rtf`, and `audio_duration_s`. Models are loaded once by the caller
and passed in as kwargs — adapters never load weights themselves, which matters for t0018 because
F5-TTS/CosyVoice2/Chatterbox model loads are each multi-second to multi-minute and must not repeat
per prompt.
`harness.run_system_eval(system_name, adapter_fn, prompts, ref_durations, centroid, half_b_paths, ...)`
(`harness.py`) drives one system across all prompts, handling warmup discard and per-clip metric
assembly. Six new adapter functions (F5-TTS ×2 ref conditions, CosyVoice2 ×2, Chatterbox ×2 — or one
adapter per system parameterized by a `reference_wav` kwarg) must be written to this exact contract
so `run_eval.py`'s CLI and `report.py`'s aggregation keep working unmodified.

### TTFB semantics differ by streaming capability, and the harness already has both precedents [t0008, t0013]

t0008's `elevenlabs_david` adapter measures TTFB as wall time to the first non-empty streamed HTTP
chunk (`adapters.py:108-124`); its Kokoro adapters measure TTFB as wall time to the first non-empty
chunk yielded by `KPipeline`'s generator (`_kokoro_synth_via_pipeline`, `adapters.py:165-212`) —
both are genuine streaming measurements. By contrast, t0013's StyleTTS2 inference harness
(`infer_styletts2.py`, 423 lines) is a single blocking call with no chunked/streaming output; that
task's `results_summary.md` explicitly records
`ttfb_ms: not measured (offline batch harness, no streaming endpoint in scope)`. This is the exact
precedent `task_description.md` asks t0018 to follow: F5-TTS (flow-matching, non-autoregressive,
whole-clip output) should report whole-utterance latency labelled as such; CosyVoice 2 supports
chunked streaming output and should get a genuine first-chunk TTFB measurement analogous to the
Kokoro adapters; Chatterbox's streaming capability must be checked against its actual API before
deciding which pattern applies.

### The audible-speech gate has been hardened twice by human-listening escalation, and t0018 must use the latest version [t0013, t0014, t0015]

`audio_quality_check.py` originated in t0013
(`tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py`, 6298 bytes) with three
signals — `silence_fraction` (20 ms-frame RMS below -40 dBFS), `spectral_flatness` (Wiener entropy,
catches white-noise-like output), and `clip_fraction` (fraction of samples with `|x| > 0.99`,
catches rail-to-rail saturation) — combined into `is_likely_noise`. t0014 copied it forward
unchanged (`tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py`) and used it as a hard
completion gate for `kokoro-v11-best`. t0015 found a **third failure mode** that gate could not see
— 73.95 s of locally speech-shaped-but-linguistically-empty audio for a ~10-word sentence, i.e.
droning babble that never pauses — and hardened the module
(`tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`, 233 lines) with two new
signals: `duration_sanity_pass` (synthesized duration vs. `3x` a generous words-per-second floor of
1.5 wps) and `longest_nonsilent_run_s` (max contiguous non-silent run, threshold 12.0 s), while
deliberately leaving `is_likely_noise` unchanged for backward compatibility.
`check_audio_quality(wav_path, text=None) -> AudioQualityResult` is the public function;
`task_description.md` names this exact t0015 path as the gate to run on every synthesized clip in
t0018 — using the t0013 or t0014 copies would silently drop the duration/silence-gap signals that
are precisely the failure mode most likely in unfamiliar zero-shot models (F5-TTS and CosyVoice2 are
also duration/flow-matching-driven, the same architecture family that produced v11's blowup).

### Reference-clip concatenation for style conditioning has one prior implementation, but it is purpose-built and needs adaptation [t0014]

`build_reference_concat.py` (`tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`,
52 lines) concatenates three hardcoded named clips (`lining_up_suggestions_17.wav`,
`lining_up_suggestions_10.wav`, `putting_them_head_to_head_15.wav`) with 0.2 s silence gaps into a
5.48 s WAV, specifically because StyleTTS2's `StyleEncoder` needs more than the ~1 s of a single
`11labs_david` clip. `task_description.md` tells t0018 to "reuse `build_reference_concat.py`'s
approach" for its `ref_concat` condition — but the concrete parameters differ: t0018 needs a **~30
s** concatenation (not 5.48 s), built from **half-A-only** clips per
`build_reference_split(seed=42)` (t0014's three named clips must first be checked against the
half-A/half-B split — if any fall in half-B they cannot be reused), and the clip filenames/order
must be recorded per `task_description.md`'s Reference-audio conditions section. The reusable part
is the pattern (load N wavs, insert silence gaps, concatenate, write one WAV, log which filenames
were used), not the specific clip list or duration.

### GPU VM cost discipline on `LLM-T1-NC80` has one severe failure and one clean recovery in this project's history [t0010, t0014]

t0010 planned $45-100 for its Stage 2 training run and actually spent **$272.78 over 19.54 hours**,
mostly idle overnight, caused by a combination of a watchdog gap and a full-root-disk teardown
delay. t0014, running on the same VM pool for a heavier full-corpus retrain, monitored disk usage
proactively throughout (reclaimed from 98%/3 GB free to 87%/16 GB free before training, stable at
87% for the whole run) and the same failure mode did not recur; final cost was $89.85 over 6.436
hours, in line with plan. t0018's own budget (~$45 planned, $70 hard cap, ≈3.2 h) is tight relative
to t0010's overrun and involves loading three separate heavyweight model stacks (F5-TTS, CosyVoice2,
Chatterbox) plus their pretrained weight downloads to `/mnt/cache/persist/pretrained/` — disk-usage
monitoring and watchdog-PID confirmation before the first download (as `task_description.md`'s
Compute and budget section already requires) are directly actionable lessons from this pair of
tasks, not generic advice.

### The reporting/aggregation layer computes variant-format metrics but t0018 needs new chart types not yet implemented [t0008]

`report.py` (435 lines) has three functions worth reusing directly: `compute_variant_metrics()`
(aggregates a per-clip list into mean/std/percentiles for one system+prompt-set combination),
`build_metrics_json()` (assembles the variant-format `metrics.json` t0018's Expected Outputs also
require), and `build_tables_json()`. Its three existing chart functions —
`plot_speaker_sim_boxplot`, `plot_ttfb_cdf`, `plot_speaker_sim_wer_scatter` — do **not** cover the
four charts `task_description.md` requires (`speaker_sim_by_system.png` grouped bars,
`ttfb_vs_speaker_sim.png` scatter with a 300 ms vertical line, `ref_condition_effect.png` paired
bars per system, `wer_by_system.png`), so new plotting functions must be written; only the
percentile helper (`_percentile`, `report.py:27-31`) and the variant-aggregation pipeline are
directly reusable as-is.

## Reusable Code and Assets

* **Source**: `tts_eval_harness` library (registered asset, created by t0008). **What it does**:
  full synthesis-timing-scoring-reporting pipeline for TTS systems. **Reuse method**: **import via
  library** —
  `from tasks.t0008_tts_eval_harness_baselines.code.harness import build_reference_split, get_prompts_by_set`,
  `from tasks.t0008_tts_eval_harness_baselines.code.scoring import build_centroid, compute_speaker_sim, compute_wer, compute_duration_ratio`,
  `from tasks.t0008_tts_eval_harness_baselines.code.adapters import SynthResult, save_wav, elevenlabs_david, kokoro_v3_bundle`.
  **Function signatures**:
  `build_reference_split(corpus_dir: Path, seed: int = 42) -> tuple[np.ndarray, list[Path]]`;
  `compute_speaker_sim(synth_wavs: list[Path], ref_embeddings: np.ndarray) -> SpeakerSimResult`;
  `compute_wer(synth_wavs: list[Path], texts: list[str], duration_ratios: list[float|None]) -> WerResult`;
  `compute_duration_ratio(synth_wavs: list[Path], ref_durations_s: list[float]) -> DurationRatioResult`.
  **Adaptation needed**: none for the scoring/split/prompt-loading functions; the `elevenlabs_david`
  and `kokoro_v3_bundle` adapters are reused as-is to re-score the two paired baselines. New adapter
  functions matching the `SynthResult` contract must be written for F5-TTS, CosyVoice2, and
  Chatterbox — these do not exist anywhere in the codebase and are new work, not reuse. **Line
  count**: harness.py 193, scoring.py 364, adapters.py 330 (only the non-Kokoro-specific portions
  apply).

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/report.py` (`compute_variant_metrics`,
  `build_metrics_json`, `build_tables_json`, `_percentile`). **What it does**: aggregates per-clip
  JSON into the variant-format `metrics.json`/`tables.json` t0018's Expected Outputs section also
  requires. **Reuse method**: **copy into task** (not a library module — only
  `harness.py`/`adapters.py`/
  `scoring.py`/`report.py`/`extract_decoder.py`/`constants.py`/`paths.py`/`run_eval.py`/
  `prepare_prompts.py`/`score_speaker_sim.py` are all bundled under the single `tts_eval_harness`
  library id, so `report.py`'s functions ARE importable via the library — but t0018 needs to add new
  chart functions alongside them, which means either extending the library in place, per this repo's
  provenance rule that `arf/` and libraries may be modified freely, or copying the aggregation
  functions and writing new plotting code in `code/`. Given `report.py` is part of the registered
  `tts_eval_harness` library, prefer importing `compute_variant_metrics`/`build_metrics_json`/
  `build_tables_json` and writing the four new chart functions as new code in t0018's own `code/`
  module that calls into the imported aggregation helpers. **Function signatures**:
  `compute_variant_metrics(...)` at `report.py:33`, `build_metrics_json(...)` at `report.py:122`,
  `build_tables_json(...)` at `report.py:152`. **Adaptation needed**: none for aggregation; new
  plotting functions needed for `speaker_sim_by_system.png`, `ttfb_vs_speaker_sim.png`,
  `ref_condition_effect.png`, `wer_by_system.png`. **Line count**: 435 total, ~140 lines of directly
  reusable aggregation logic (lines 1-183).

* **Source**: `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`. **What it
  does**: `check_audio_quality(wav_path: Path, *, text: str | None = None) -> AudioQualityResult`
  computing `rms`, `peak`, `silence_fraction`, `spectral_flatness`, `clip_fraction`,
  `longest_nonsilent_run_s`, `duration_sanity_pass`, `is_likely_noise` — the hardened, most-current
  version of the project's audible-speech gate. **Reuse method**: **copy into task** — this file
  lives in a task `code/` directory, not a registered library, so per the Cross-Task Code Reuse Rule
  it must be copied, not imported. **Function signatures**:
  `check_audio_quality(wav_path: Path, *, text: str | None = None) -> AudioQualityResult`;
  `estimate_naive_duration_bound_s(text: str) -> float`. **Adaptation needed**: none to the logic
  itself — call it unchanged on every synthesized clip (across all 8 variants × 196 prompts) and
  report the failure count per system, per `task_description.md`'s "Audible-speech gate"
  requirement. Pass `text=` so `duration_sanity_pass` is populated (it is `None` without it). **Line
  count**: 233 lines.

* **Source**: `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`. **What it
  does**: loads N WAV files, inserts 0.2 s silence gaps, concatenates, writes one output WAV, prints
  the clip list used. **Reuse method**: **copy into task** — task `code/` file, not a library.
  **Function signatures**: `build_reference_concat(out_path: Path) -> Path` (currently hardcodes
  `REFERENCE_CLIP_NAMES`, a module-level tuple of 3 filenames, and reads from
  `ELEVENLABS_DAVID_DIR`). **Adaptation needed**: replace `REFERENCE_CLIP_NAMES` (currently 3 fixed
  clips, ~5.5 s total) with a parameter or a new constant selecting enough half-A clips to reach ~30
  s, restricted to `build_reference_split(seed=42)`'s half-A set; keep the
  silence-gap-and-concatenate mechanism. **Line count**: 52 lines.

* **Source**: `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py` (423 lines) and
  its t0014/t0015 copies. **What it does**: StyleTTS2-native inference recipe with instrumented
  per-module missing/unexpected key-count logging on checkpoint load. **Reuse method**: **not
  directly reusable** — this is a StyleTTS2/Kokoro-specific inference harness (raw checkpoint
  loading, `compute_style`, `pred_dur`/alignment plumbing) with no relevance to
  F5-TTS/CosyVoice2/Chatterbox, which each ship their own inference APIs. Cited here only as the
  precedent for *instrumenting* a new inference wrapper (log what loaded, log raw duration
  predictions, don't trust a silent `strict=False` success) — the same discipline t0018's new
  adapters should apply when integrating three unfamiliar model APIs for the first time.

## Lessons Learned

* **Adapter-per-system with an already-loaded model is the correct integration shape** [t0008] — it
  keeps model loading (expensive, once per system) separate from per-prompt timing (cheap, must be
  precise), and every downstream tool (`run_eval.py`, `report.py`) already assumes this shape.
  Breaking it (e.g. loading a model inside the adapter) would silently inflate TTFB/RTF for the
  first prompt of each system.
* **A "the harness must be proven correct on a known-good control before trusting a defect finding"
  discipline caught a real training bug and would have caught a harness bug just as fast** [t0013] —
  t0013 ran its new inference harness against an external known-good StyleTTS2 checkpoint first,
  which is what let it conclude with evidence that v10's noise was a training defect, not a
  reproduction bug. For t0018, the equivalent is the mandatory smoke gate (one synthesis succeeds)
  before the 50-warmup-then-196-prompt run for each of F5-TTS/CosyVoice2/Chatterbox — if smoke
  fails, the task's own protocol says mark that system null with the exact error rather than
  debugging silently into the budget.
* **A single automated audio-quality heuristic is never final** [t0013 → t0014 → t0015] — three
  tasks in a row found a new failure mode a human caught that the existing gate could not
  (clipping/flatness, then duration/silence-gaps). t0018 should treat `is_likely_noise`-style gates
  as necessary but not sufficient, and its mandatory human-listening deliverable
  (`results/listening_guide.md` + full audio indexing) is the direct mitigation the project has
  already converged on rather than trusting metrics alone.
* **`speaker_sim` alone does not distinguish "broken" from "merely mediocre" audio** [t0013] — t0013
  found confirmed-garbage v10 audio still scored 0.31-0.35 GE2E cosine, not a near-zero outlier.
  t0018 must report the gate-failure count and the unfiltered-vs-gate-filtered `speaker_sim` mean
  side by side (as `task_description.md` already requires), not just the filtered number, since a
  high `speaker_sim` cannot by itself prove a zero-shot system's output is genuine speech.
* **GPU idle billing is the single largest realized risk in this project's task history on
  `LLM-T1-NC80`** [t0010, t0014] — $272.78 lost to a watchdog gap and full-disk teardown delay in
  one task, $0 lost when the same class of risk was actively monitored in a later task on the same
  pool. t0018's $70 hard cap (vs. ~$45 planned) leaves little room for a repeat.

## Recommendations for This Task

1. **Import, don't reimplement, the scoring and reference-split machinery.** Use
   `tts_eval_harness`'s `build_reference_split`, `compute_speaker_sim`, `compute_wer`,
   `compute_duration_ratio`, and prompt loaders unchanged — these are exactly what t0008 built for
   this purpose and re-scoring `elevenlabs_david`/`kokoro_v3_bundle` for paired comparison depends
   on identical scoring code being used for old and new systems alike.
2. **Copy `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`** (not t0013's or
   t0014's earlier copies) into t0018's `code/`, and call
   `check_audio_quality(wav, text=prompt_text)` on every one of the ~1568 synthesized clips, per the
   task's mandatory gate requirement. Report both `is_likely_noise` and the hardened
   duration/silence-gap failures separately, matching t0015's own three-way-regression discipline.
3. **Adapt, don't reuse verbatim, `build_reference_concat.py`.** Change the target duration from
   ~5.5 s to ~30 s and restrict the source clip pool to `build_reference_split(seed=42)`'s half-A
   set; record the exact filenames and order used, as `task_description.md` requires, in both
   `results/references/` and the results tables.
4. **Write six new adapter functions matching the `SynthResult` contract** (F5-TTS × 2 ref
   conditions, CosyVoice2 × 2, Chatterbox × 2), following the pattern in `adapters.py`'s
   `_kokoro_synth_via_pipeline`/`elevenlabs_david`: measure TTFB as first-chunk wall time only for
   systems with genuine streaming output (check CosyVoice2's actual streaming API surface and
   Chatterbox's before assuming); for F5-TTS (non-autoregressive, whole-clip), report
   whole-utterance latency and label it explicitly as such in every table, matching t0013's
   disclosed precedent for `ttfb_ms: not measured`.
5. **Budget disk and watchdog discipline as first-class steps**, not an afterthought — confirm the
   watchdog PID before the first model weight download (three separate model stacks to
   `/mnt/cache/persist/pretrained/`), and monitor `/mnt` disk usage proactively through the session,
   directly following t0014's successful mitigation of t0010's $272.78 overrun on the same VM pool.
6. **Extend `report.py`'s four missing chart types as new functions**, reusing its
   `compute_variant_metrics`/`build_metrics_json`/`build_tables_json`/`_percentile` aggregation
   layer unchanged (import via the `tts_eval_harness` library) rather than re-deriving
   percentile/aggregation logic.
7. **Gap requiring new implementation**: none of the three target models (F5-TTS, CosyVoice2,
   Chatterbox) have any prior integration anywhere in this codebase — their installation, weight
   download, and Python environment (likely a dedicated venv per model given each ships its own
   heavy, potentially conflicting dependency set, mirroring the existing `resemblyzer` isolation
   precedent in `tts_eval_harness`) is new work with no reusable prior art beyond the general
   "isolate heavy/conflicting deps in a separate venv, invoke via subprocess or explicit PYTHONPATH"
   pattern already used for `score_speaker_sim.py`.

## Task Index

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Direct dependency; source of the `tts_eval_harness` library t0018 must reuse for
  synthesis timing, GE2E scoring, WER, duration ratio, the half-A/half-B reference split, and
  variant-format metrics/report aggregation.

### [t0010]

* **Task ID**: `t0010_stage2_safeguarded_training`
* **Name**: Kokoro Stage 2: safeguarded training with joint_epoch=8
* **Status**: completed
* **Relevance**: Cautionary precedent on the same `LLM-T1-NC80` VM pool — $272.78 spent over 19.54 h
  against a $45-100 plan, due to a watchdog gap and full-disk teardown delay, directly informing
  t0018's disk/watchdog discipline under its own tight $70 cap.

### [t0013]

* **Task ID**: `t0013_v10_synthesis_quality_forensics`
* **Name**: v10 checkpoint synthesis quality forensics
* **Status**: completed
* **Relevance**: Originated `audio_quality_check.py` (later hardened by t0015) and established the
  project's precedent for labelling non-streaming whole-utterance latency as `ttfb_ms: not measured`
  and for treating `speaker_sim` as insufficient on its own to certify audio quality.

### [t0014]

* **Task ID**: `t0014_v11_decoder_fix_retrain`
* **Name**: Kokoro Stage 2 v11: decoder-init fix, full normalized corpus retrain
* **Status**: completed
* **Relevance**: Source of `build_reference_concat.py`, the concatenation-with-silence-gaps pattern
  `task_description.md` names for t0018's `ref_concat` condition; also the successful
  counter-example to t0010's disk/watchdog cost overrun on the same VM pool.

### [t0015]

* **Task ID**: `t0015_v11_duration_blowup_forensics`
* **Name**: v11 duration-blowup forensics and audible-speech gate hardening
* **Status**: completed
* **Relevance**: Source of the current, hardened `audio_quality_check.py` that `task_description.md`
  requires t0018 to run on every synthesized clip; demonstrates that automated audio-quality gates
  need repeated human-listening-driven hardening, directly motivating t0018's own mandatory
  human-listening deliverable.
