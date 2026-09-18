---
spec_version: "2"
task_id: "t0021_zero_shot_latency_reduction"
date_completed: "2026-09-18"
status: "complete"
---
# Plan: Zero-Shot Latency Reduction (CosyVoice2 / Chatterbox TTFB Floor)

## Objective

Find the lowest time-to-first-byte (TTFB) that CosyVoice2 and Chatterbox can reach on `LLM-T1-NC80`
(Azure ML 2xH100 SXM5) without losing the `speaker_sim` (GE2E cosine similarity) they showed in
`t0018_zero_shot_cloning_calibration`, and determine whether either system can reach the project's
300 ms TTFB target. t0018 measured both systems at 1.3-2.9 s p50 TTFB against the 300 ms target,
while both already run faster than real time (RTF 0.40-0.54), meaning the delay is in per-request
start-up work (reference encoding, LM prefill/decode, first flow-matching chunk), not raw synthesis
throughput. This task instruments each system's first-chunk latency stage by stage, then tests
concrete acceleration levers (reference-embedding caching, `load_jit`/`load_trt`/ `fp16` for
CosyVoice2, a vLLM-adapted LM backend, sentence-level chunking and `torch.compile` for Chatterbox)
one at a time and in combination, always re-scoring `speaker_sim`/WER/gate-failure rate in the same
GPU session as the paired t0018-equivalent baseline so speed and quality stay comparable.

**Binding correction, folded into this plan (see "Owner Correction" section below for the full
text):** the project owner discovered on 2026-09-18 that `data/11labs_david/` (used by t0008 and
therefore by t0018's `ref_single`/`ref_concat` reference clips and speaker-sim centroid) is the
WRONG ElevenLabs "David" voice ("David - British Radio Host", `voice_id=5gLuKtB16QIQv1vuSas1`).
Production and the Kokoro training corpus `data/v4` use a different voice ("David - narrator and
newsreader", `voice_id=rWV5HleMkWb5oluMwkA7`). t0018 is completed and immutable and is NOT corrected
retroactively. From this planning step onward, every reference clip and every speaker-similarity
centroid this task builds MUST come from `data/v4/val/wavs` (the val_96 held-out set, which IS the
correct production voice), never from `data/11labs_david`. t0018's own `speaker_sim` numbers remain
on record as measured against the wrong voice; this task's numbers are the first correct-voice
measurement and are not directly comparable to t0018's headline numbers without the explicit
"radiohost (wrong voice) control" column defined below.

**Success criteria for this plan / the resulting implementation:**

* A per-stage latency breakdown (`results/latency_breakdown.json`) exists for both systems' baseline
  setting, decomposing first-chunk TTFB into named stages.
* At least 6 named acceleration variants per system are attempted (one is the re-run baseline), each
  scored for `ttfb_ms` (p50/p95/p99), `speaker_sim` (against the corrected val_96 centroid), `rtf`,
  and WER, in the same GPU session as its paired baseline.
* The answer asset `assets/answer/zero-shot-ttfb-floor/` states, per system: the lowest reachable
  TTFB p50, the setting that achieves it, the `speaker_sim` cost (if any) of reaching it, and
  whether 300 ms is reachable (yes / no / not with this architecture).
* All owner-correction requirements (below) are satisfied and traceable in the outputs.
* `uv run python -m arf.scripts.verificators.verify_plan t0021_zero_shot_latency_reduction` and,
  later, the implementation-stage verificators, pass with 0 errors.

## Owner Correction

**Full text of the correction, as recorded in `checkpoint.md`'s Cross-Step Decisions section
(binding for this task from step 7/planning onward):**

> The project owner found on 2026-09-18 that the ElevenLabs account has two voices named "David".
> Production (`brainpowa-voice-gateway` FluxCD) and the Kokoro training corpus `data/v4` use
> `voice_id=rWV5HleMkWb5oluMwkA7` ("David - narrator and newsreader", `model_id=eleven_flash_v2_5`,
> `output_format=pcm_24000`). t0008's harness resolved the voice by name and picked
> `5gLuKtB16QIQv1vuSas1` ("David - British Radio Host") instead, so `data/11labs_david` is the WRONG
> David, and t0018's `ref_single`/`ref_concat` clips (built from `data/11labs_david` half-A) cloned
> the wrong voice. t0018 is completed and immutable — do not modify it; this correction applies
> going forward starting at this planning step.

The root cause is concretely located: `get_elevenlabs_voice_id()` in
`tasks/t0008_tts_eval_harness_baselines/code/run_eval.py` (lines ~110-137) resolves the ElevenLabs
voice by fuzzy name match against `ELEVENLABS_DAVID_VOICE_NAME`, which is how the wrong voice
(`5gLuKtB16QIQv1vuSas1`) was picked. This task must never call that function, and must never resolve
an ElevenLabs voice by name anywhere in its own code.

This task's implementation is required to satisfy all seven binding sub-requirements (REQ-11 through
REQ-17 in the checklist below); each is mapped to a concrete Step by Step action.
`intervention/owner_correction_wrong_david_voice.md` (created in Step 1) is the single canonical
place documenting this correction and is cited from `results/results_detailed.md` (an
orchestrator-managed file outside this plan's Step by Step, but the intervention file itself is this
plan's output).

## Task Requirement Checklist

Operative task text, quoted verbatim from `task.json` and `task_description.md`:

> **short_description**: Profile where CosyVoice2 and Chatterbox spend their 1.3-2.9 s TTFB and test
> streaming, chunking, vLLM/TensorRT backends and precision to find the best reachable TTFB at
> unchanged speaker_sim.

> **Key Questions** (task_description.md): (1) Where does the time go? Per-stage breakdown of
> first-chunk latency: reference-audio encoding, text frontend, LLM prefill and first decode tokens,
> flow-matching/token-to-wav for the first chunk, vocoder — warm cache, 50 discarded warmups. (2)
> How much of that is per-request work cacheable per voice? (3) What does each documented
> acceleration path buy, one at a time: CosyVoice2 `load_jit`/`load_trt`/`fp16`/vLLM LLM
> backend/smaller streaming chunk size/cached reference embedding; Chatterbox sentence-level
> chunking/`torch.compile`/fp16/bf16/cached voice conditioning/any streaming API at the pinned
> version. (4) Which combination gives the lowest TTFB p50/p95, and what is `speaker_sim`, WER and
> gate-failure rate at that setting vs t0018's baseline setting? (5) Does either system reach TTFB
> p50 ≤ 300 ms? If not, is the gap architectural or engineering? (6) Cheap closures: CosyVoice2
> `ref_concat` retry at 29.5 s (S-0018-02); F5-TTS retry with `py-spy dump` on hang, 45-minute cap,
> null with stack trace if it hangs again (S-0018-01).

> **Protocol**: smoke gate, 50 warmup requests, val96 + 100 fillers + 3 gate texts, per-clip
> TTFB/RTF/duration/WER/GE2E cosine vs the half-B centroid, hardened gate on every clip. Use the
> corrected harness from t0019 if merged by task start, otherwise t0008's, stated explicitly. Every
> acceleration variant is a separate metrics variant; the t0018 baseline setting for each system is
> re-run in the same session as the paired control (Lesson 1). Rejection:
> `successful_prompts / total_prompts < 0.8` nulls a variant (Lesson 3). Capture engine, CUDA,
> TensorRT and vLLM versions per variant (Lesson 4).

> **Forbidden**: No fine-tuning. No change to reference selection tuned on val96. No variant
> reported without its paired same-session baseline. No substitution of a model version without
> recording it in the environment table.

Plus the seven binding owner-correction items quoted in full in the "Owner Correction" section
above.

| ID | Requirement | Satisfied by step(s) | Evidence |
| --- | --- | --- | --- |
| REQ-1 | Per-stage first-chunk latency breakdown per system (ref encoding, text frontend, LM prefill/decode, flow-matching, vocoder), warm cache, 50 discarded warmups | Steps 6, 8, 10 | `results/latency_breakdown.json`, `results/images/latency_breakdown_stacked.png` |
| REQ-2 | Identify and cache per-voice work (reference-audio embedding/conditioning) as a zero-cost lever | Steps 6, 9, 11 | `results/latency_breakdown.json` ref-cache variant deltas; `results/metrics.json` |
| REQ-3 | Test each documented acceleration lever one at a time / cumulatively for both systems | Steps 9, 11 | `results/metrics.json` variants; `results/environment.json` |
| REQ-4 | Best combination TTFB p50/p95 per system + paired `speaker_sim`/WER/gate-failure rate vs paired baseline | Steps 9, 11, 14 | `results/metrics.json`, `results/tables.json` |
| REQ-5 | Determine whether either system reaches 300 ms p50; if not, characterize gap as architectural vs engineering | Step 16 (answer asset) | `assets/answer/zero-shot-ttfb-floor/` |
| REQ-6 | Cheap closures: CosyVoice2 `ref_concat` @ 29.5 s both prompt sets (S-0018-02); F5-TTS retry with `py-spy`, 45-min cap (S-0018-01) | Steps 12, 13 | `results/per_clip_metrics.json`, `intervention/f5_tts_retry_*.md` (only if it hangs again) |
| REQ-7 | Full protocol compliance (smoke gate, 50 warmups, val96+100 fillers+3 gate texts, per-clip metrics, hardened gate, corrected-harness statement, paired same-session baseline, 0.8 rejection threshold, per-variant environment capture) | Steps 5, 7, 8-13, 14 | `results/smoke_gate_log.md`, `results/per_clip_metrics.json`, `results/environment.json`, `results/gate_failures.json` |
| REQ-8 | Produce all Expected Outputs: answer asset, `latency_breakdown.json`, `metrics.json`, `per_clip_metrics.json`, `costs.json`, `remote_machines_used.json`, 3 named charts, tables, `audio_samples/` (DVC) + `listening_guide.md` | Steps 6-17 | listed files under `results/` and `assets/answer/` |
| REQ-9 | Stay within budget (~5.5 h ≈ $77, hard cap $100); watchdog armed + PID confirmed before first build; artifacts on `/mnt/cache/persist/` | Step 2 (setup), tracked throughout via Step 18 | `results/cost_tracking.json`, `machine_log.json` |
| REQ-10 | Forbidden-list compliance: no fine-tuning; no change to reference-clip *selection methodology*; no variant without paired baseline; no silent model-version substitution | All steps (constraint, not a single step) | code review of `code/`, `results/environment.json` |
| REQ-11 | (Owner correction #1) Build `ref_single`/`ref_concat` from `data/v4/val/wavs`, NOT `data/11labs_david`; record exact source filenames | Step 3 | `data/references/manifest.json` |
| REQ-12 | (Owner correction #2) Build the scoring centroid from the val_96 half NOT used for references; keep a second "radiohost (wrong voice) control" column vs the OLD `data/11labs_david` centroid | Step 3, Step 10 | `data/references/manifest.json`, `results/per_clip_metrics.json` (`speaker_sim_radiohost_control` field), `results/tables.json` |
| REQ-13 | (Owner correction #3) Any ElevenLabs API call pins `voice_id=rWV5HleMkWb5oluMwkA7`, `model_id=eleven_flash_v2_5`, `output_format=pcm_24000`, `stability=0.5`, `similarity_boost=0.75`; never resolve by name | N/A — this task makes no new ElevenLabs API calls (see Approach); documented as a standing constraint in Step 1's intervention file and re-affirmed in code review | `intervention/owner_correction_wrong_david_voice.md` |
| REQ-14 | (Owner correction #4) t0018's baseline setting re-run with NEW references, same session as every variant | Steps 8, 9, 11 | `results/per_clip_metrics.json` (`acceleration_variant="baseline_new_ref"` rows), `results/environment.json` timestamps |
| REQ-15 | (Owner correction #5) `intervention/` file documenting the correction; cited in plan.md; note t0018's `speaker_sim` was measured against the wrong voice | Step 1 (this plan.md's "Owner Correction" section is the citation) | `intervention/owner_correction_wrong_david_voice.md` |
| REQ-16 | (Owner correction #6) `results/audio_samples/` 3-way per comparison text (new-ref output, old t0018 wrong-voice-ref output, val_96 original), indexed in `results/listening_guide.md` | Steps 15, 16 | `results/audio_samples/comparison_set/`, `results/listening_guide.md` |
| REQ-17 | (Owner correction #7) Note that the automated gate is necessary, not sufficient; owner will listen manually | Risks & Fallbacks, Verification Criteria (below); also stated in `results/listening_guide.md` header | this plan.md; `results/listening_guide.md` |

**Ambiguity called out explicitly:** REQ-10's "no change to reference selection tuned on val96"
(from the Forbidden list) predates the owner correction and originally meant "don't cherry-pick
which val96 clips go into `ref_single`/`ref_concat` to inflate `speaker_sim`." The owner correction
requires *changing the corpus* references are drawn from (`data/11labs_david` → `data/v4/val/wavs`),
which is a different axis (correcting a wrong-voice bug, not re-tuning a selection heuristic). The
two are not in conflict: this plan uses the exact same construction *method* t0018 used
(filename-sorted concatenation with 0.2 s silence gaps, seed=42, first-N-clips-until-target-duration
logic, no cherry-picking or manual curation of which clips are "best"), only pointed at the correct
source corpus. This is stated explicitly in Step 3 and in `data/references/manifest.json`.

## Approach

**Grounded in research (`research/research_summary.md`):**

* Adopt CosyVoice2's additive latency model `L_TTS = M·d_lm + M·d_fm + M·d_voc` [Du2024] as the
  schema for `results/latency_breakdown.json`, extended with a reference-encoding and a
  text-frontend stage so it covers Key Question 1's full stage list (5 stages total:
  `ref_encoding_ms`, `text_frontend_ms`, `lm_prefill_decode_ms`, `flow_matching_ms`, `vocoder_ms`).
* Published vocoder speeds are 150x-3700x real time unoptimized [Kong2020, Kaneko2022], so the
  LM/decoder stage (`d_lm`) is the most likely dominant TTFB term — this plan prioritizes LM/decoder
  acceleration (vLLM, `fp16`, cached reference embedding) over vocoder-stage work, per Approach 2 in
  research_summary.md.
* Streaming removes the *quality* penalty of chunking, not the *latency* penalty — the LM must still
  generate enough tokens before flow-matching can run, which is exactly why t0018 measured 2.86 s
  p50 TTFB for CosyVoice2 even with `stream=True`. This plan treats "streaming" and "fewer tokens
  before first chunk" as two separate levers, not one.
* RTF is not a proxy for TTFB (research_summary.md finding 5) — every chart and table in this task
  reports TTFB and RTF as separate columns, never substituting one for the other.
* Reference-audio speaker embedding is a deterministic function of the fixed reference clip —
  caching it per voice is zero-cost and zero-risk [Wan2018] and is tested first, before any
  serving-stack change, per Approach 1 (instrument first, then land cheap/free wins before heavier
  rework).
* CosyVoice2 only exposes an `fp16` boolean flag (see `load_cosyvoice2_model` in
  `tasks/t0018_zero_shot_cloning_calibration/code/adapters_zeroshot.py`), not a separate bf16
  toggle. Research recommends "bf16-everywhere" based on H100-confirmed Chatterbox gains
  [Chatterbox-TTS-Server-GH], but that evidence does not extend to CosyVoice2's flow-matching or
  vocoder stages, and the `[Lian2026]` (dots.tts) bf16/int8-split citation was retracted during
  research-internet (step 5) as unsupported. This plan therefore: (a) uses CosyVoice2's own
  `fp16=True` flag exactly as named (not bf16, since the repo doesn't expose that choice), and (b)
  for Chatterbox, attempts bf16 first (H100-confirmed) and falls back to fp16 only if bf16 raises an
  unsupported-dtype error, recording which one was actually used in `results/environment.json`.
* Cross-system/cross-encoder speaker-similarity numbers are not comparable [Kunešová2025] — every
  variant is re-scored with the same GE2E scorer (`resemblyzer.VoiceEncoder`) used for the t0018
  baseline, never against a paper's own published number.
* Whether acceleration shifts `speaker_sim` at fixed content is unmeasured anywhere in the
  literature (research_summary.md finding/risk) — this task's paired per-variant `speaker_sim`
  measurement is the first evidence on this question and must be reported even if the answer is "no
  measurable shift," not silently assumed.

**Alternatives considered:**

* *Rebuild the ElevenLabs filler-corpus baseline properly (re-synthesize `data/11labs_david` with
  the correct `voice_id`) instead of switching to `data/v4/val/wavs`.* Rejected: the owner
  correction explicitly names `data/v4/val/wavs` as the reference source going forward, and
  re-synthesizing 1358+ clips via the ElevenLabs API would cost real API spend and time this task's
  ~$77 budget does not allocate for. `data/v4/val/wavs` is already the correct voice, already local
  (DVC-tracked), and large enough (96 clips) to support a disjoint reference/centroid split.
* *Test every acceleration lever in full factorial combination (2^6 combinations per system) instead
  of a fixed cumulative-stack ordering.* Rejected on cost grounds: the $77 budget affords ~6
  variants/system at ~0.25 h each. A cumulative stack (each variant adds exactly one lever to the
  previous winning combination) answers "what does each lever buy, one at a time" (Key Question 3)
  while still producing one "best combination" data point (Key Question 4), which a small fixed set
  of isolated single-lever tests would not.
* *Instrument only wall-clock TTFB without per-stage breakdown.* Rejected: Key Question 1 explicitly
  requires the per-stage decomposition, and without it Key Question 5's "architectural vs.
  engineering gap" cannot be answered — a TTFB-only number can't say whether the remaining gap to
  300 ms is dominated by an inherently-serial LM decode step or by a fixable inefficiency in, say,
  reference encoding.

**Task types:** `tts-benchmark-run`, `experiment-run`, `answer-question` (already set in
`task.json`, no change needed). `tts-benchmark-run`'s Planning Guidelines drove the
warmup/smoke-gate protocol (Steps 5, 7-13); `experiment-run`'s guidelines drove the explicit
multi-variant `metrics.json` format, per-variant environment capture, and the requirement to save
every raw per-clip output (Steps 9, 11, 14); `answer-question`'s guidelines drove the single
canonical question and answer asset in Step 16.

**Libraries to import (as-is, per `research/research_code.md`):**

* `tasks.t0008_tts_eval_harness_baselines.code.adapters`: `SynthResult`, `save_wav`,
  `_resample_to_16k`.
* `tasks.t0008_tts_eval_harness_baselines.code.harness`: `get_prompts_by_set`, `PromptItem`,
  `load_val96_prompts`.
* `tasks.t0008_tts_eval_harness_baselines.code.scoring`: `compute_speaker_sim`, `compute_wer`,
  `compute_duration_ratio`.
* `tasks.t0008_tts_eval_harness_baselines.code.constants`: `RANDOM_SEED`,
  `DURATION_RATIO_LOW`/`DURATION_RATIO_HIGH`.
* `tasks.t0008_tts_eval_harness_baselines.code.report`: `REGISTERED_METRIC_KEYS`, `_percentile`.

**Code to copy (not import — cross-task `code/` imports outside registered libraries are disallowed
per current spec) from `tasks/t0018_zero_shot_cloning_calibration/code/` into this task's own
`code/`, then adapt:** `adapters_zeroshot.py`, `run_eval_zeroshot.py`, `constants.py`, `paths.py`,
`build_references.py` (renamed `build_references_val96.py` here), `run_gate_check.py`,
`track_cost.py`, `report_zeroshot.py`, `merge_and_score.py`, `build_comparison_set.py`,
`build_listening_guide.py`, `build_smoke_gate_log.py`, `transcribe_references.py`. Also copy
`tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` (233 lines) directly into
this task's `code/` (per `research/research_code.md`: t0018's cross-task import of this file is no
longer valid under the current spec, so t0021 must have its own copy).

**What is genuinely new (not present in any prior task's code):** per-stage timing instrumentation
inside the copied `adapters_zeroshot.py` (none of t0018's three adapters records intermediate
timestamps); the `--acceleration-variant` CLI dimension; the val_96-sourced reference/centroid
builder; the dual-centroid scoring (new val_96 centroid + old-voice control column); the 3-way
audio-sample layout; and the three new charts.

## Cost Estimation

All costs are GPU-hour compute on `LLM-T1-NC80` (Azure ML 2xH100 SXM5, $13.96/hr) plus zero external
API cost (no ElevenLabs calls are planned — see Approach and REQ-13's resolution above; no new LLM
API calls beyond this planning agent's own token usage, which is not billed against the project's
`$5000` compute/API budget line in the way GPU time is).

| Item | Est. hours | Est. cost (USD) |
| --- | --- | --- |
| Reference/centroid construction from `data/v4/val/wavs` (CPU-only, resemblyzer on CPU) | 0 (done before VM provisioning, on the local/dev machine, no GPU billing) | $0 |
| VM setup: isolated venvs, `load_jit`/`load_trt` export, vLLM backend install, watchdog | 1.0 | $13.96 |
| Per-stage instrumentation + smoke gate (both systems) | 0.5 | $6.98 |
| Acceleration variant sweep: 6 variants × 2 systems × ~0.25 h (196 prompts each, `ref_single`) | 3.0 | $41.88 |
| S-0018-02 closure cell: CosyVoice2 `ref_concat` @ 29.5 s, baseline setting, both prompt sets | 0.3 | $4.19 |
| S-0018-01 closure: F5-TTS retry with `py-spy`, hard 45-min cap | 0.75 | $10.47 |
| Teardown (watchdog stop, `stop_compute`) | ~0.05 | $0.70 |
| **Total** | **~5.6 h** | **~$78.18** |

Compared against `project/budget.json`: total project budget is `$5000`, `$459.11` already spent
(9.18%), `$4540.89` remaining, warn/stop thresholds (80%/100%) nowhere close. This task's own
`per_task_default_limit` is `$100.0`; the ~$78 estimate leaves ~~$22 of headroom against the task's
own hard cap (`BUDGET_HARD_CAP_USD = 100.0` in `code/constants.py`, see Step 1), consistent with
`task_description.md`'s stated "~~$77" estimate and "$100" hard cap. If the vLLM backend install or
TensorRT export overruns the 1.0 h setup budget, Step 2 defines an explicit cutoff (see Risks &
Fallbacks).

## Step by Step

### Milestone 1 — Correction paperwork, reference/centroid rebuild (no GPU, local/dev machine)

1. **[CRITICAL] Write the intervention file documenting the owner correction.** Create
   `intervention/owner_correction_wrong_david_voice.md`, following the structure of
   `tasks/t0018_zero_shot_cloning_calibration/intervention/f5_tts_smoke_gate_failed.md` (What
   happened / Why / Resolution / Cost impact sections). Content: quote the correction text verbatim
   (same text as this plan's "Owner Correction" section); state the root cause
   (`get_elevenlabs_voice_id()` in `tasks/t0008_tts_eval_harness_baselines/code/run_eval.py` fuzzy
   name-matched to the wrong voice); state explicitly that t0018's `speaker_sim` values
   (`0.8420862078666687`-`0.8627852474649748` for CosyVoice2, `0.7964`-`0.8112` for Chatterbox, from
   `tasks/t0018_zero_shot_cloning_calibration/results/metrics.json`) were measured against
   `voice_id=5gLuKtB16QIQv1vuSas1` ("David - British Radio Host"), not the production voice
   `voice_id=rWV5HleMkWb5oluMwkA7` ("David - narrator and newsreader"); state that t0018 is not
   modified. Satisfies REQ-15.

2. **[CRITICAL] Build `ref_single`/`ref_concat` and the corrected centroid from
   `data/v4/val/wavs`.** First run `dvc pull data/v4/val/wavs` (per `CLAUDE.md`'s DVC workflow) to
   materialize the 96 val_96 clips locally (`data/v4/val/wavs.dvc` confirms 96 files, ~19.9 MB
   total, avg ~4.3 s/clip — short filler-style phrases, similar in character to the ElevenLabs
   filler corpus, not long sentences; verify this assumption by running `soundfile.info()` over all
   96 files as a preflight check before building anything, and document the actual duration
   distribution found). Create `code/build_references_val96.py`, adapted from
   `tasks/t0018_zero_shot_cloning_calibration/code/build_references.py`:
   * Read all 96 filenames from `data/v4/val/wavs/*.wav` (or from `data/v4/val_list.txt`'s
     `wav_path` column — this file IS git-tracked, unlike the audio itself).
   * `rng = random.Random(RANDOM_SEED)` (import `RANDOM_SEED = 42` from
     `tasks.t0008_tts_eval_harness_baselines.code.constants`, same seed t0018 used, for consistency
     of method — this is the "same construction method, correct source corpus" resolution of the
     REQ-10 ambiguity noted in the checklist above). Shuffle and split into two **disjoint 48/48
     halves**: `ref_source_half` (first 48) and `centroid_half` (remaining 48). This is a deliberate
     change from t0018's design (which built both the centroid and the references from the *same*
     half-A) — the owner correction (item 2) explicitly requires the centroid to come from "the half
     not used for references," so this task uses two disjoint halves, not one shared half.
   * Build `ref_single.wav` (~10 s target, `REF_SINGLE_MAX_DURATION_S = 10.0`) and `ref_concat.wav`
     (29.5 s target, `REF_CONCAT_TARGET_DURATION_S = 29.5` — the S-0018-02 fix, changed from t0018's
     `30.0`) by concatenating filename-sorted clips from `ref_source_half` with 0.2 s silence gaps
     (`REF_CONCAT_SILENCE_GAP_S = 0.2`), stopping once the running total first exceeds the target —
     same method as t0018's `_build_concat()`, copied verbatim into this file. `ref_concat` excludes
     any filenames already consumed by `ref_single` (same non-overlap rule t0018 used). If a
     preflight check on the real per-clip durations shows a single clip near 10 s exists (unlike
     t0018's corpus), prefer a single natural clip for `ref_single`, matching t0018's own
     preflight-deviation pattern — document whichever path is taken in
     `data/references/manifest.json`'s `ref_single_is_concat` field, never silently.
   * Build the corrected centroid from `centroid_half` (48 clips) using `resemblyzer.VoiceEncoder`
     on CPU (`device="cpu"`), following the exact working pattern in t0018's
     `_build_half_a_centroid()` (call `encoder.embed_utterance()` directly per clip, no minimum-
     duration pre-filter — t0018 discovered the officially "reusable" `build_reference_split()` in
     `tasks.t0008_tts_eval_harness_baselines.code.harness` hard-skips clips under 1.6 s and can
     raise `RuntimeError` on a small/short corpus; first *try* `build_reference_split()`/
     `build_centroid()` from that library since val_96's ~4.3 s average clips are unlikely to hit
     the 1.6 s floor, and only fall back to the copied inline approach if it does). Save to
     `data/references/val96_centroid.npy`.
   * Copy `tasks/t0018_zero_shot_cloning_calibration/data/references/half_a_centroid.npy` verbatim
     (`shutil.copy2`, read-only, t0018 is immutable) to
     `data/references/old_wrongvoice_centroid.npy` — this is the fixed "radiohost (wrong voice)
     control" centroid for REQ-12's second scoring column; it is never rebuilt from
     `data/11labs_david` in this task.
   * Write `data/references/manifest.json` recording: `ref_single_filenames`,
     `ref_single_duration_s`, `ref_single_is_concat`, `ref_concat_filenames`,
     `ref_concat_duration_s`, `centroid_half_filenames` (all 48, for REQ-11's "record exact source
     filenames" requirement), `seed=42`, `source_corpus="data/v4/val/wavs"`,
     `old_control_centroid_source="tasks/t0018_zero_shot_cloning_calibration/data/references/half_a_centroid.npy"`.
     Satisfies REQ-1 (indirectly, by producing valid inputs), REQ-11, REQ-12.

3. **Transcribe the new reference clips.** Copy
   `tasks/t0018_zero_shot_cloning_calibration/code/transcribe_references.py` to
   `code/transcribe_references.py`, point it at the new `data/references/ref_single.wav` and
   `data/references/ref_concat.wav`, run it (Whisper-based, CPU or GPU) to produce the `prompt_text`
   transcript CosyVoice2's `inference_zero_shot()` requires (it needs a text transcript of what is
   spoken in the reference clip, not the phoneme string in `val_list.txt`). Append
   `ref_single_transcript`/`ref_concat_transcript` to `data/references/manifest.json`. Expected
   output: two non-empty transcript strings roughly matching the concatenated clips' known phrases
   (e.g. "checking for the latest press release...").

### Milestone 2 — Remote machine setup (GPU billing starts here)

4. **[CRITICAL] Provision `LLM-T1-NC80` and arm the idle watchdog before any build.** Follow
   `arf/skills/setup-remote-machine/SKILL.md`. SSH via the `LLM-T1-NC80` alias. Confirm
   `/mnt/cache/persist` resolves to the real Azure Files mount (Lesson 10 —
   `readlink -f /mnt/cache/persist` must NOT point at ephemeral `/mnt`; if it does, symlink it per
   the fix in `LESSONS.md` Lesson 10). Deploy and start `arf/scripts/utils/idle_watchdog.sh` per
   `CLAUDE.md`'s "Deploying the watchdog" section
   (`TERMINATE_CMD="az ml compute stop --name LLM-T1-NC80 --workspace-name brainpowa-northeurope --resource-group rezolve-AI"`,
   `IDLE_THRESHOLD_SECONDS=3600`, `IDLE_UTIL_PERCENT=5`, `GRACE_SECONDS=600`). Confirm the watchdog
   PID with `ps aux | grep idle_watchdog` and record `watchdog_active: true` + `watchdog_pid: <pid>`
   in `machine_log.json` (Lesson 8). Reuse t0018's already-installed `.venv-cosyvoice2` and
   `.venv-chatterbox` if the VM's persistent storage still has them (check
   `/mnt/cache/persist/**/.venv-cosyvoice2` and `.venv-chatterbox` first); otherwise recreate them
   exactly as t0018 did (`CosyVoice2-0.5B` in a `torch==2.3.1+cu121` venv, `chatterbox==0.1.7` in a
   `torch==2.6.0+cu124` venv). Satisfies part of REQ-9.

5. **[CRITICAL] Smoke gate before any warmup or measurement (Lesson 2).** For each system, run one
   synthesis call end to end
   (`.venv-cosyvoice2/bin/python -m tasks.t0021_zero_shot_latency_reduction.code.run_eval_zeroshot --system cosyvoice2 --conditions ref_single --prompt-set fillers --limit 1 --cosyvoice-model-dir <path>`,
   and the Chatterbox equivalent). If either smoke gate fails, STOP: write an intervention file
   (following the `f5_tts_smoke_gate_failed.md` pattern), mark that system null for the whole task,
   and do not proceed to warmup/measurement for it — do not spend GPU time debugging past the smoke
   gate without a documented decision. Record `smoke_gate_status: "pass"/"fail"` and the reason in
   `results/smoke_gate_log.md` (build via a copy of `code/build_smoke_gate_log.py`). Satisfies part
   of REQ-7.

6. **Set up the acceleration-lever environments.** Install what each variant needs, in the isolated
   venv that variant runs in (never mix into the main project venv):
   * CosyVoice2 `load_jit`/`load_trt`: follow the CosyVoice2 repo's own export scripts (already
     available in the pinned CosyVoice2 checkout used by `.venv-cosyvoice2`) to produce the JIT and
     TensorRT artifacts; save exported artifacts under
     `/mnt/cache/persist/t0021_zero_shot_latency_reduction/` (Lesson 10 — never under ephemeral
     `/mnt`).
   * CosyVoice2 vLLM backend: create a **new**, separate isolated venv `.venv-cosyvoice2-vllm` and
     install `swulling/CosyVoice2-0.5B-vllm` there (per research: this pins its own `torch` version,
     which is expected to conflict with `.venv-cosyvoice2`'s `torch==2.3.1+cu121` — do not attempt
     to share a venv). If the install fails or produces an incompatible `torch`/CUDA combination
     within a 20-minute cutoff, record the exact error in an intervention file, mark the
     `cosyvoice2_vllm` variant null with the reason, and continue with the other 5 CosyVoice2
     variants — do not let this block the rest of Milestone 3.
   * Chatterbox `torch.compile`: no separate install needed (built into the pinned
     `torch==2.6.0+cu124`); confirm `torch.compile` actually reduces call-to-call latency on a
     throwaway smoke call before counting it as a real variant (compile overhead on the *first* call
     is expected and must be excluded from measured latency — treat the compile call itself as an
     extra warmup, not a measured request). Record every installed package version (`pip freeze`
     output for each venv, or targeted `importlib.metadata.version()` calls) into
     `results/environment.json` alongside the existing `torch_version`/`cuda_version`/`gpu_name`
     fields t0018's `_write_environment_record()` already captures — add `tensorrt_version`,
     `vllm_version`, `cosyvoice2_git_sha` fields (all `null` when not applicable to a given
     variant). Satisfies REQ-3 (setup half), REQ-7 (Lesson 4 capture), REQ-9.

### Milestone 3 — Per-stage instrumentation and acceleration variant sweep

7. **[CRITICAL] Add per-stage timing checkpoints inside the copied adapters.** Copy
   `tasks/t0018_zero_shot_cloning_calibration/code/adapters_zeroshot.py` into
   `code/adapters_zeroshot.py`. Inside `cosyvoice2_synth()`, split the single
   `model.inference_zero_shot(text, prompt_text, str(ref_wav_path), stream=True)` call into its
   constituent internal calls by inspecting the installed `cosyvoice` package source on the VM
   (`python -c "import cosyvoice, inspect; print(inspect.getsourcefile(cosyvoice.cli.model))"`, then
   reading `cosyvoice/cli/model.py` and `cosyvoice/cli/frontend.py` directly) to find the natural
   call boundaries between: reference-audio encoding (`frontend.frontend_zero_shot()` — speaker
   embedding + prompt tokens extraction), the LLM prefill/decode loop, the first flow-matching
   chunk, and the vocoder (HiFi-GAN/HiFT) call. Wrap each with `time.perf_counter()` before/after
   and accumulate into a `StageTiming` dataclass (`@dataclass(frozen=True, slots=True)` per
   `arf/styleguide/python_styleguide.md`) with fields `ref_encoding_ms`, `text_frontend_ms`,
   `lm_prefill_decode_ms`, `flow_matching_first_chunk_ms`, `vocoder_first_chunk_ms`. If two adjacent
   stages turn out not to be separable at the pinned CosyVoice2 version without invasive
   monkey-patching (e.g., reference encoding and text frontend happen inside one opaque call), merge
   them into one combined field and say so explicitly in a code comment and in
   `results/latency_breakdown.json`'s notes field — never silently attribute a merged duration to
   just one of the two stage names. Apply the same approach to `chatterbox_synth()`, inspecting
   `chatterbox.tts.ChatterboxTTS.generate()`'s internals (voice conditioning / T3 decoder / S3Gen
   vocoder) for its natural call boundaries; Chatterbox has no flow-matching stage, so its schema
   uses `ref_conditioning_ms`, `text_frontend_ms`, `lm_decode_ms` (T3), `vocoder_ms` (S3Gen)
   instead. Return `StageTiming` alongside the existing `SynthResult` from each `*_synth()` function
   (add a `stage_timing: StageTiming | None` field to a local subclass/wrapper, not to t0008's own
   `SynthResult` — that class belongs to another task's code). Also fix the documented
   `frontend_zero_shot` bug noted in t0018's `adapters_zeroshot.py` docstring (passing a file path,
   not a pre-loaded tensor, to `prompt_wav`) if it resurfaces. Expected output: for a single smoke
   call per system, a `StageTiming` object whose stage sums approximately equal the whole-call wall
   time (within measurement noise), printed to the log for a sanity check before any measured run.
   Satisfies REQ-1.

8. **Add the `--acceleration-variant` CLI flag and per-variant kwargs mapping.** Copy
   `tasks/t0018_zero_shot_cloning_calibration/code/run_eval_zeroshot.py` to
   `code/run_eval_zeroshot.py`. Add
   `p.add_argument("--acceleration-variant", required=True, choices=[...])`. Define, in
   `code/constants.py` (copied+extended from t0018's), the fixed cumulative-stack variant lists:
   ```
   COSYVOICE2_VARIANTS = [
       "baseline_new_ref",       # load_jit=False, load_trt=False, fp16=False, no ref-cache
       "ref_cache",              # + cached speaker embedding/prompt tokens (computed once)
       "fp16",                   # + fp16=True
       "load_jit",               # + load_jit=True
       "load_trt",               # + load_trt=True  (full local-optimization stack)
       "vllm_backend",           # separate stack: ref_cache + vLLM Qwen2.5-0.5B LM backend
   ]
   CHATTERBOX_VARIANTS = [
       "baseline_new_ref",       # whole-utterance, no cache, fp32, no compile, no chunking
       "ref_cache",              # + cached voice-conditioning embedding
       "precision_bf16_or_fp16", # + bf16 (fallback fp16 if bf16 unsupported at this torch/GPU combo)
       "torch_compile",          # + torch.compile on the T3 decoder
       "sentence_chunking",      # + pre-split text into 2+ chunks, first chunk streamed/returned
                                  #   first (full cumulative stack)
       "streaming_api",          # separate: whatever native streaming API resemble-ai/chatterbox
                                  #   exposes at the pinned 0.1.7 version, ref_cache applied; if no
                                  #   such API exists at this version, mark null with that reason
                                  #   recorded, do not substitute a different lever silently
   ]
   ```
   Each variant name maps to a fixed set of loader/synth kwargs implemented in
   `code/adapters_zeroshot.py`'s loader functions (extend `load_cosyvoice2_model()` to accept
   `load_jit`, `load_trt`, `fp16`, `use_ref_cache` kwargs it already half-supports; extend
   `load_chatterbox_model()` similarly). `baseline_new_ref` is the exact re-run of t0018's setting
   (`load_jit=False, load_trt=False, fp16=False` / whole-utterance no cache), just pointed at the
   new `data/references/` clips — this is the REQ-14 paired control. Expected output: running with
   `--acceleration-variant baseline_new_ref` reproduces t0018's *method* exactly (same flags), only
   the reference audio differs. Satisfies REQ-2, REQ-3, REQ-14.

9. **[CRITICAL] Run the acceleration variant sweep, one system-session per system.** For each
   system, in ONE continuous VM session (model process not restarted between variants except when
   loading a genuinely different backend, e.g. `vllm_backend`), run all 6 variants in the fixed
   order above, each with: 50 discarded warmup requests (`WARMUP_TEXT`, per Lesson 1) then the full
   196-prompt measured set (`--prompt-set both`, `ref_single` condition only — `val96` = 96 prompts
   from `data/v4/val_list.txt`'s text field via `get_prompts_by_set`, `fillers` = 100 sampled filler
   prompts from t0008's existing `filler_prompts_100.json`, unaffected by the voice correction since
   these are TEXT prompts, not reference audio). Command example:
   ```
   .venv-cosyvoice2/bin/python -m tasks.t0021_zero_shot_latency_reduction.code.run_eval_zeroshot \
       --system cosyvoice2 --acceleration-variant fp16 --conditions ref_single \
       --prompt-set both --n-warmup 50 --cosyvoice-model-dir <path>
   ```
   **Validation gate before the full 196-prompt run for each new variant:** first run with
   `--limit 10`. Trivial baseline: TTFB must be lower than or within measurement noise of the
   *previous* variant in the cumulative stack (a "faster" lever that measures slower than the
   un-accelerated baseline indicates a bug, e.g. `fp16=True` silently falling back to fp32, or the
   ref-cache not actually skipping re-encoding). If the 10-prompt TTFB is *higher* than the
   `baseline_new_ref` variant's 10-prompt TTFB, STOP: read 5 individual per-clip records from
   `results/per_clip_metrics_<variant_slug>.json`, confirm the input text and reference audio path
   are correct, confirm the acceleration flag was actually honored (log the loaded model's actual
   dtype/JIT status), and confirm the timing instrumentation from Step 7 is wired to the right calls
   — do not proceed to the full 196-prompt run until this is resolved or explicitly documented as a
   genuine (not bug-driven) regression. Save per-clip records to
   `results/per_clip_metrics_<system>_<variant>_ref_single.json` and per-stage timing sums to
   `results/latency_breakdown_<system>_<variant>.json` (aggregated into the final
   `results/latency_breakdown.json` in Step 14). After each variant, call
   `code/track_cost.py --milestone "<system> <variant> done"` (copied from t0018, update
   `VM_HOURLY_COST_USD=13.96`, `BUDGET_HARD_CAP_USD=100.0`, new `VM_BILLING_ANCHOR_ISO`) to keep a
   running cost checkpoint. If cumulative cost approaches the $100 hard cap before all 12 cells (6
   variants x 2 systems) complete, stop after the current variant, document which cells were skipped
   and why in an intervention file, and proceed to scoring/reporting with whatever data exists — do
   not silently truncate the variant list without a record. Rejection rule (Lesson 3,
   pre-registered, see Rejection Criteria section below): any variant with
   `successful_prompts / total_prompts < 0.8` is null regardless of its measured numbers. Satisfies
   REQ-1, REQ-2, REQ-3, REQ-4, REQ-7, REQ-9, REQ-14.

10. **Score every clip: dual speaker-sim centroids, WER, duration ratio, hardened gate.** Copy
    `tasks/t0018_zero_shot_cloning_calibration/code/merge_and_score.py` to
    `code/merge_and_score.py`. Merge all `per_clip_metrics_<system>_<variant>_<condition>.json`
    files (written in Steps 9, 12, 13) into `results/per_clip_metrics.json`. For every row with a
    non-null `audio_path`, call `compute_speaker_sim()` (from
    `tasks.t0008_tts_eval_harness_baselines.code.scoring`) **twice**: once against
    `data/references/val96_centroid.npy` (the corrected centroid — write this value into the row's
    `speaker_sim` field, the one that also becomes the registered `speaker_sim` metric in
    `results/metrics.json`), and once against `data/references/old_wrongvoice_centroid.npy` (write
    this into a new, non-registered `speaker_sim_radiohost_control` field on the same row — this
    field is per-clip data, not a `metrics.json` metric key, because `meta/metrics/` only registers
    `speaker_sim`/`ttfb_ms`/`rtf` and `results/metrics.json` may not contain unregistered keys per
    `arf/specifications/metrics_specification.md`; the control column lives in
    `per_clip_metrics.json` and `results/tables.json` instead). Compute `duration_ratio` (val96 rows
    only, against `ref_duration_s`) and WER (gated by `DURATION_RATIO_LOW`/`_HIGH`), same as t0018.
    Copy `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` into
    `code/audio_quality_check.py` (233 lines, per research_code.md — do NOT import it cross-task).
    Copy `tasks/t0018_zero_shot_cloning_calibration/code/run_gate_check.py` to
    `code/run_gate_check.py`, changing its import to
    `from tasks.t0021_zero_shot_latency_reduction.code.audio_quality_check import (...)` (the local
    copy, not t0015's). Run it over every row, writing `hardened_gate_pass` per clip and
    `results/gate_failures.json` per variant. Expected output: every row in
    `results/per_clip_metrics.json` has non-null `speaker_sim`, `speaker_sim_radiohost_control`,
    `hardened_gate_pass` (for rows with existing audio). Satisfies REQ-4, REQ-6, REQ-7, REQ-12.

### Milestone 4 — Closure items (S-0018-01, S-0018-02)

11. **CosyVoice2 `ref_concat` closure (S-0018-02).** With `data/references/ref_concat.wav` now built
    at the 29.5 s target (Step 2 already fixed this via `REF_CONCAT_TARGET_DURATION_S = 29.5`), run
    `--system cosyvoice2 --acceleration-variant baseline_new_ref --conditions ref_concat --prompt-set both --n-warmup 50`
    (baseline setting only, not swept across all 6 variants — this is the one-cell closure the
    budget allocates 0.3 h for, not a second full variant matrix). Expected output:
    `successful_prompts / total_prompts >= 0.8` (unlike t0018's 0/196 hard failure, since the
    reference audio no longer exceeds CosyVoice2's internal 30 s assertion in
    `cosyvoice/cli/frontend.py`). If it still fails for a different reason, write an intervention
    file with the new failure signature. Satisfies REQ-6.

12. **F5-TTS retry (S-0018-01), 45-minute hard cap.** In a fresh shell session (not reusing any
    process/venv state from a prior hung attempt), first try `pip install py-spy` in `.venv-f5tts`,
    then attempt the same smoke-gate call t0018 tried (`load_f5tts_model()` → one `f5_tts_synth()`
    call) against the NEW `data/references/ref_single.wav` and its transcript from Step 3. Try the
    two candidate fixes research flagged before assuming a repeat hang: (a) clear/refresh the HF
    cache (`rm -rf` the F5-TTS-specific cache dir and re-download), (b) reinstall `.venv-f5tts` from
    scratch (t0018's own intervention note recommended this as "the first thing to try"). If it
    hangs again past 5 minutes with zero incremental log output (t0018's own signature for a genuine
    hang, vs. slow-but-progressing downloads), run `py-spy dump --pid <pid>` to capture the stack
    trace, kill the process, and write `intervention/f5_tts_retry_still_hangs.md` with the stack
    trace and a null verdict for F5-TTS — do not retry a third time past the 45-minute cap. If it
    succeeds, run ONE measured pass (`ref_single` and `ref_concat`, both prompt sets, 50 warmups)
    using the NEW references — no acceleration variants for F5-TTS (out of this task's scope; task
    title names only CosyVoice2 and Chatterbox for the latency-reduction study). Satisfies REQ-6,
    REQ-14 (if it succeeds, its baseline is also paired against the new references, consistent with
    the rest of the task).

### Milestone 5 — Metrics, charts, audio samples, answer asset

13. **[CRITICAL] Write `results/metrics.json` in the explicit multi-variant format.** Copy
    `tasks/t0018_zero_shot_cloning_calibration/code/report_zeroshot.py` and `build_final_reports.py`
    to `code/`, extending `compute_variant_metrics_zeroshot()` with a 4th dimension,
    `acceleration_variant` (in addition to `system`/`condition`/`prompt_set`). `variant_id` pattern:
    `<system>_<acceleration_variant>_<condition>_<prompt_set>`, e.g.
    `cosyvoice2_fp16_ref_single_val96`. Each variant's `metrics` object contains exactly the three
    registered keys `speaker_sim`, `ttfb_ms`, `rtf` (per
    `arf/specifications/metrics_specification.md` — no unregistered keys). Apply the Lesson 3
    rejection rule inside this function: if `n_successful / n_clips < 0.8`, set all three metric
    values to `null` for that variant (same pattern as t0018's `cosyvoice2_ref_concat_*` null
    variants). Write `results/tables.json` with a human-readable comparison table per system
    including BOTH `speaker_sim` (corrected) and `speaker_sim_radiohost_control` (old wrong-voice)
    columns side by side, explicitly labelled, so a reader can see both numbers without recomputing
    anything. Expected output:
    `uv run python -u -m arf.scripts.verificators.verify_task_metrics t0021_zero_shot_latency_reduction`
    passes with 0 errors once run at implementation time. Satisfies REQ-4, REQ-8.

14. **Write `results/latency_breakdown.json` and produce the three required charts.** Aggregate the
    per-variant `StageTiming` sums from Step 9 into one file keyed by
    `<system>_<acceleration_variant>`, each value a dict of the 4-5 stage fields from Step 7 plus a
    `total_ms` sanity-check field. In `code/report_zeroshot.py`, add three new plotting functions
    (import `matplotlib`, follow the `dataviz` skill's color/accessibility guidance):
    * `plot_latency_breakdown_stacked()` → `results/images/latency_breakdown_stacked.png`: one
      stacked bar per system/variant, one segment per stage, error bars or Q1-Q3 shading per Key
      Question 1's requirement.
    * `plot_ttfb_vs_speaker_sim_variants()` → `results/images/ttfb_vs_speaker_sim_variants.png`:
      scatter, x-axis TTFB p50 with a vertical reference line at 300 ms, y-axis fillers
      `speaker_sim`, one point per variant, t0018's baseline numbers marked distinctly (different
      marker/color, labelled "t0018 baseline (wrong-voice reference, for continuity only)").
    * `plot_ttfb_p50_p95_by_variant()` → `results/images/ttfb_p50_p95_by_variant.png`: grouped bar
      or dot-and-whisker, p50/p95 per variant per system, with the 300 ms target line. Expected
      output: all three PNGs exist, non-zero-byte, and `results/tables.json` references them by
      relative path. Satisfies REQ-1, REQ-4, REQ-5, REQ-8.

15. **Build the 3-way comparison audio set (owner correction #6).** Copy
    `tasks/t0018_zero_shot_cloning_calibration/code/build_comparison_set.py` to
    `code/build_comparison_set.py`, reusing its `select_comparison_texts()` logic (3 fixed
    `GATE_TEXT_NAMES` + 7 seeded val96 samples via `random.Random(COMPARISON_SET_SEED).sample(...)`,
    same seed/count constants as t0018 for continuity). For each of the 10 comparison texts, copy
    THREE files into `results/audio_samples/comparison_set/`:
    * `<text_id>__<system>_<best_variant>__new_ref.wav` — this task's own best-setting output (from
      `results/audio_samples/harness/...` written during Step 9).
    * `<text_id>__<system>__t0018_old_ref.wav` — copied directly (read-only) from
      `tasks/t0018_zero_shot_cloning_calibration/results/audio_samples/comparison_set/` (t0018's own
      already-generated wrong-voice-reference output for the matching text; t0018 is not modified,
      only read from).
    * `<text_id>__val96_original.wav` — the actual production-voice ground-truth clip. For the 7
      val96-sampled texts, this is a direct lookup in `data/v4/val/wavs/` by matching the prompt
      text to its slugified filename prefix (filenames follow the pattern `<text_slug>_<6hex>.wav`,
      confirmed from `data/v4/val_list.txt`). For the 3 fixed gate texts (filler-corpus phrases like
      "lining up suggestions 17"), attempt the same slug-prefix match against `data/v4/val/wavs/`;
      if no match exists (t0018's own listening guide documented that these 3 gate texts are not
      among the 100 sampled filler prompts either — the same gap may recur here), write `-` / omit
      the file and say so explicitly in `results/listening_guide.md`, following the exact
      "documented deviation, not silent" pattern t0018's `build_listening_guide.py` already used for
      this identical situation. Satisfies REQ-16.

16. **Write `results/listening_guide.md` (owner correction #6) and the answer asset.** Copy
    `tasks/t0018_zero_shot_cloning_calibration/code/build_listening_guide.py` to
    `code/build_listening_guide.py`, adapting the table to 3 columns per text (new-ref / t0018
    old-ref / val96 original) instead of one column per system/condition, each cell a clickable
    relative link plus `speaker_sim`/`speaker_sim_radiohost_control`/`wer`/gate-verdict where
    applicable. Add a header note stating, verbatim in spirit: "t0015's `audio_quality_check.py`
    gate (imported here as `hardened_gate_pass`) is necessary, not sufficient — it is known to pass
    clips a human listener would describe as 'voice plus strong noise.' A PASS here does not certify
    audio quality; the owner will listen to this guide's clips directly before any
    production-readiness conclusion is drawn." (REQ-17). Then create the answer asset per
    `meta/asset_types/answer/specification.md`: `assets/answer/zero-shot-ttfb-floor/details.json`
    (`answer_id="zero-shot-ttfb-floor"`, `question` = "What is the lowest reachable TTFB for
    CosyVoice2 and Chatterbox on our hardware without losing speaker_sim, and does either reach 300
    ms?", `answer_methods=["code-experiment"]`,
    `source_task_ids=["t0018_zero_shot_cloning_calibration", "t0021_zero_shot_latency_reduction"]`,
    `confidence` set honestly based on how many of the 12 variant cells actually produced non-null
    data), `short_answer.md` (2-5 sentences, direct, stating the best TTFB per system, the setting,
    the `speaker_sim` delta at that setting, and a yes/no/not-with-this-architecture verdict on 300
    ms — no `[tNNNN]`/`[AuthorYear]` brackets in the Answer section), `full_answer.md` (mini-paper:
    methodology, per-stage findings, the architectural-vs-engineering discussion for Key Question 5,
    explicit mention that these numbers are the first correct-voice measurement following the owner
    correction and are not directly comparable to t0018's headline numbers without the
    radiohost-control column, limitations, `## Sources` with markdown reference-link definitions for
    `[Du2024]`, `[Kong2020]`, `[Seo2026]`, etc., matching the citation keys already used in
    `research/research_papers.md`). Satisfies REQ-5, REQ-8.

## Remote Machines

**Required.** `LLM-T1-NC80` (Azure ML 2xH100 SXM5 VM, `$13.96`/hr, connect via SSH alias
`LLM-T1-NC80`, pool config `project/azure_vm.json`). Both CosyVoice2 and Chatterbox need a GPU for
inference; the TensorRT export (`load_trt`) and vLLM backend install also require the H100's CUDA
toolchain. Estimated total GPU time ~5.6 hours across setup, instrumentation, the 12-cell variant
sweep, the two closure items, and teardown (see Cost Estimation). Reference/centroid construction
(Milestone 1) is CPU-only and does NOT require the GPU VM — do that first, locally, before
provisioning. The idle watchdog (`arf/scripts/utils/idle_watchdog.sh`) must be armed with a
confirmed PID before the first `load_jit`/`load_trt`/vLLM build starts (Lesson 8), and all build
artifacts/weights must live under `/mnt/cache/persist/` (Lesson 10), never bare `/mnt`.

## Assets Needed

* `data/v4/val/wavs/` (96 clips, DVC-tracked at `data/v4/val/wavs.dvc`) — the correct-voice
  reference/centroid source corpus, per the owner correction. Pull via `dvc pull data/v4/val/wavs`.
* `data/v4/val_list.txt` (git-tracked manifest, 96 lines, `wav_path|phonemes|speaker_id`).
* From `t0018_zero_shot_cloning_calibration` (dependency, completed): the CosyVoice2/Chatterbox
  loading code pattern (`code/adapters_zeroshot.py`), the harness runner pattern
  (`code/run_eval_zeroshot.py`), the reference-construction method (`code/build_references.py`),
  `results/audio_samples/comparison_set/` (read-only, for the 3-way comparison in Step 15), and
  `data/references/half_a_centroid.npy` (copied verbatim as the "radiohost (wrong voice) control"
  centroid).
* From `t0008_tts_eval_harness_baselines` (registered library `tts_eval_harness`): `SynthResult`,
  `save_wav`, `get_prompts_by_set`, `PromptItem`, `load_val96_prompts`, `compute_speaker_sim`,
  `compute_wer`, `compute_duration_ratio`, `REGISTERED_METRIC_KEYS`, `RANDOM_SEED`.
* From `t0015_v11_duration_blowup_forensics`: `code/audio_quality_check.py` (233 lines, copied, not
  imported).
* Isolated venvs already installed on `LLM-T1-NC80` from t0018's own setup: `.venv-cosyvoice2`
  (`torch==2.3.1+cu121`), `.venv-chatterbox` (`torch==2.6.0+cu124`); a NEW `.venv-cosyvoice2-vllm`
  and `.venv-f5tts` (retry) created in this task.
* Community reference (unbenchmarked, used only as a starting point, not a citation of authoritative
  numbers): `swulling/CosyVoice2-0.5B-vllm`.

## Expected Assets

* `assets/answer/zero-shot-ttfb-floor/` (answer asset, matching `task.json`'s
  `expected_assets: {"answer": 1}`): `details.json`, `short_answer.md`, `full_answer.md`, per
  `meta/asset_types/answer/specification.md` v2 (with `short_answer_path`/`full_answer_path` set).
  States the lowest reachable TTFB per system, the winning setting, the `speaker_sim` cost, and the
  300 ms verdict.

No other new asset types (`dataset`, `library`, `model`, `predictions`, `paper`,
`latency_benchmark_run`) are produced by this task — this task consumes and re-scores existing data
rather than registering new corpora or models. `latency_benchmark_run` (a registered asset type in
`meta/asset_types/latency_benchmark_run/`) was considered and not used: it targets a different
harness shape (`vorontsov-latency-harness`, request-count/duration-window runs against an HTTP
endpoint) than this task's fixed-corpus TTS synthesis benchmark, which already has its own
established asset shape from t0008/t0018 (`results/per_clip_metrics.json` + `results/metrics.json`,
not a dedicated per-run asset folder).

## Time Estimation

* Research (already done, steps 4-6): complete, not part of this estimate.
* Milestone 1 (correction paperwork + reference/centroid rebuild, local/CPU): ~30-45 minutes
  wall-clock, $0 GPU cost.
* Milestone 2 (VM setup, watchdog, venvs): ~1 hour GPU wall-clock.
* Milestone 3 (instrumentation + 12-cell variant sweep): ~3.5 hours GPU wall-clock (0.5 h
  instrumentation/smoke gate + 3 h sweep).
* Milestone 4 (S-0018-02 + S-0018-01 closures): ~1.05 hours GPU wall-clock (0.3 h + 0.75 h cap).
* Milestone 5 (metrics/charts/audio samples/answer asset): ~30-45 minutes, mostly CPU-bound
  post-processing on already-downloaded audio, negligible additional GPU billing beyond teardown.
* Teardown: ~5 minutes.
* **Total wall-clock, GPU-billed portion: ~5.6 hours** (matches Cost Estimation).

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| vLLM/TensorRT install conflicts with CosyVoice2's pinned `torch==2.3.1+cu121` (research flagged this as unverified) | High | Loses the `vllm_backend`/`load_trt` variants, the two most novel acceleration levers | Install vLLM in a fully separate venv (`.venv-cosyvoice2-vllm`), never touching `.venv-cosyvoice2`; cap the install attempt at 20 minutes; on failure, mark that one variant null with the exact error in an intervention file and continue with the other 4-5 CosyVoice2 variants — do not let one variant's failure block the whole system's sweep. |
| val_96's actual per-clip durations don't support both a ~10 s `ref_single` and a ~29.5 s `ref_concat` from a 48-clip half (e.g. if clips are shorter than expected and 48 clips can't reach 29.5 s cumulatively) | Low-medium | Blocks Milestone 1's reference construction, which every later step depends on | Preflight `soundfile.info()` check across all 96 clips before building anything (Step 2); with ~19.9 MB / 96 clips ≈ 207 KB/clip ≈ 4.3 s/clip average at 24kHz/16-bit, 48 clips sum to ~206 s of audio, far more than 29.5 s needed — low actual risk, but if the real distribution is pathological (e.g. many near-zero-length clips), fall back to drawing from the full 96-clip pool for `ref_source_half` construction only (not for the centroid half) and document the deviation, matching t0018's own "documented deviation, not silent" precedent. |
| F5-TTS hangs a third time with the identical signature (S-0018-01) | Medium (it hung 3/3 times in t0018 even after `HF_HUB_OFFLINE=1`) | Loses F5-TTS as a ceiling-comparison data point, but this is explicitly budgeted for | Hard 45-minute cap (already budgeted in Cost Estimation); `py-spy dump` for a stack trace; write `intervention/f5_tts_retry_still_hangs.md` with the null verdict and move on — F5-TTS is a "cheap closure," not a critical path item for this task's core CosyVoice2/Chatterbox latency question. |
| The automated `hardened_gate_pass` check (t0015's `audio_quality_check.py`) reports PASS on clips that are actually degraded ("voice plus strong noise" per REQ-17) | Medium (known limitation, not hypothetical — explicitly flagged by the owner) | A variant could be reported as "clean" when a human would reject it, misleading the answer asset's confidence | Never state or imply "gate PASS = production-quality" anywhere in `results/` or the answer asset; `results/listening_guide.md`'s header explicitly states the gate is necessary-not-sufficient (Step 16); the answer asset's `confidence` field and full_answer.md limitations section state that automated gate results have not been human-verified at plan-writing time. |
| Cumulative GPU spend approaches the $100 hard cap before all 12 variant cells + 2 closures complete | Medium (setup/build steps historically run over estimate — t0018's setup alone cost ~$26-28 against a $70 total) | Incomplete variant matrix, weaker "best combination" answer | `code/track_cost.py` checkpoint after every variant (Step 9); explicit stop-and-document rule if approaching $100; report whatever subset of variants completed rather than silently truncating without a record, and let the answer asset's confidence reflect the actual coverage achieved. |
| A variant's acceleration flag is silently not honored (e.g. `fp16=True` passed but the model still runs fp32 due to a library default) | Medium | Reports a "precision" variant that measured nothing different from baseline, producing a false null-effect finding | Step 9's validation gate: log the model's actual runtime dtype/JIT/TensorRT status for every variant before the full run, and require the 10-prompt smoke TTFB to move in the expected direction (or be explicitly explained if it doesn't) before scaling to 196 prompts. |
| Acceleration changes `speaker_sim` at fixed content (the literature has zero prior data on this) and the delta is large enough to disqualify an otherwise-fast variant | Medium (explicitly flagged as unknown/novel in research) | A "fastest" variant could actually fail the "unchanged speaker_sim" requirement in the task's own title | Every variant is scored for `speaker_sim` in the same pass as `ttfb_ms` (Step 10); Key Question 4 and the answer asset both report the paired (speed, similarity) pair per variant, never speed alone — a fast-but-degraded variant is reported as exactly that, not as a win. |

## Verification Criteria

* `uv run python -u -m arf.scripts.verificators.verify_plan t0021_zero_shot_latency_reduction`
  (already run for this planning step) — must report 0 errors.
* After implementation:
  `uv run python -u -m arf.scripts.verificators.verify_task_metrics t0021_zero_shot_latency_reduction`
  — expected: PASSED, 0 errors; `results/metrics.json` contains only the registered keys
  `speaker_sim`, `ttfb_ms`, `rtf` in every variant (confirms REQ-4, REQ-7, REQ-12's constraint that
  the control column stays out of `metrics.json`).
* `uv run python -u -m arf.scripts.aggregators.aggregate_metrics --format ids` — expected output
  exactly `rtf`, `speaker_sim`, `ttfb_ms` (confirms no unregistered metric keys were introduced).
* `uv run python -u -m meta.asset_types.answer.verificator --task-id t0021_zero_shot_latency_reduction zero-shot-ttfb-floor`
  — expected: PASSED (confirms REQ-5, REQ-8's answer-asset requirement).
* File-existence check:
  `test -f tasks/t0021_zero_shot_latency_reduction/results/latency_breakdown.json && test -f tasks/t0021_zero_shot_latency_reduction/results/listening_guide.md && test -f tasks/t0021_zero_shot_latency_reduction/intervention/owner_correction_wrong_david_voice.md`
  — expected: exit code 0 for all three (confirms REQ-1, REQ-15, REQ-16 produced their required
  files).
* `python3 -c "import json; d=json.load(open('tasks/t0021_zero_shot_latency_reduction/data/references/manifest.json')); assert d['source_corpus']=='data/v4/val/wavs'; assert len(d['centroid_half_filenames'])==48"`
  — expected: no `AssertionError` (confirms REQ-11/REQ-12: references and centroid were built from
  `data/v4/val/wavs`, not `data/11labs_david`, with a disjoint 48-clip centroid half).
* `grep -c "hex" tasks/t0021_zero_shot_latency_reduction/tasks/t0008_tts_eval_harness_baselines/code/run_eval.py`
  — N/A; instead:
  `grep -q "get_elevenlabs_voice_id" tasks/t0021_zero_shot_latency_reduction/code/*.py; test $? -ne 0`
  — expected: no match found in this task's own `code/` (confirms REQ-13: this task never calls the
  by-name voice resolver).
* Requirement-coverage check: every `REQ-*` row in the Task Requirement Checklist above has a
  non-empty "Satisfied by step(s)" and "Evidence" cell; at implementation completion, a manual
  read-through of `results/results_detailed.md`'s `## Task Requirement Coverage` section (an
  orchestrator-managed file, populated after this plan's steps run) must mark every REQ-1 through
  REQ-17 as `Done` or explicitly `Partial`/`Not done` with a stated reason — no REQ silently
  omitted.
* `ls tasks/t0021_zero_shot_latency_reduction/results/images/*.png | wc -l` — expected: `>= 3`
  (confirms the three named charts from Step 14 exist).

## Rejection Criteria

Pre-registered before any measurement runs, per Lesson 3 (`LESSONS.md`) and the task's own Protocol
section — these cannot be loosened after seeing the numbers:

* **Any acceleration variant with `successful_prompts / total_prompts < 0.8` is null for that
  variant, regardless of any TTFB/speaker_sim numbers it produced.** This applies independently to
  each `(system, acceleration_variant, condition, prompt_set)` cell — a variant that succeeds on
  `val96` but fails this threshold on `fillers` is null for `fillers` only, not for `val96`.
* **A system whose smoke gate (Step 5) fails is null for the entire task**, not partially reported —
  do not report partial numbers from a system that never passed its smoke gate.
* **Any variant reported without its paired same-session baseline measurement (REQ-14) is excluded
  from the "best combination" comparison in the answer asset** — a variant run in isolation, without
  `baseline_new_ref` measured in the same VM session, cannot support a speed-vs-baseline claim
  (Lesson 1: cold-cache and warm-cache measurements are not pairable).
* **F5-TTS's third hang (if it recurs) is a null result for F5-TTS, not a task failure** — F5-TTS is
  a cheap closure item, not a critical-path deliverable of this task.
