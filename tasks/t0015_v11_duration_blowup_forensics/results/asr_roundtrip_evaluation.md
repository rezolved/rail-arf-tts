# ASR-Round-Trip Evaluation (Milestone D Step 11, REQ-9)

## What was evaluated

Whether an ASR-round-trip check (`compute_wer` from the `tts_eval_harness` library,
`tasks.t0008_tts_eval_harness_baselines.code.scoring`) is worth adding as an optional third gate
layer, per `task_description.md` Key Question 6 and `plan/plan.md` Milestone D step 11.
`compute_wer` was run once against 3 of Step 5's characterization outputs
(`code/run_asr_roundtrip_eval.py`), selected to cover both ends of the observed `duration_ratio`
range: the least-blown-up record (`let_me_look_into_how_rezolve_transforms_`, `duration_ratio=8.61`)
and the two most-blown-up records (`improving_product_discovery_for_retailer`,
`duration_ratio=17.83`; `cross_checking_that_00`, `duration_ratio=17.87`). Raw output:
`results/asr_roundtrip_raw.json`.

## Dependency friction: none observed

`faster-whisper` and `jiwer` installed cleanly into `.venv-styletts2` in under 6 seconds
(`ctranslate2`, `onnxruntime`, `av`, `rapidfuzz` as transitive deps -- ~90 MB total, all from
cached/standard PyPI wheels, no build-from-source step). The `base.en` Whisper model downloaded from
the Hugging Face Hub automatically on first `WhisperModel(...)` construction inside `compute_wer()`,
with no manual step required (a rate-limit warning was printed for unauthenticated requests, but the
download completed regardless). No new system packages, no GPU requirement, no authentication
needed.

## The duration-ratio pre-gate makes this evaluation moot for the exact failure mode this task cares about

`compute_wer`'s own `DURATION_RATIO_LOW=0.5` / `DURATION_RATIO_HIGH=2.0` gate
(`tasks/t0008_tts_eval_harness_baselines/code/constants.py`) skipped **all 3** sampled clips --
`gated_out_by_duration_ratio=true` for every record in `results/asr_roundtrip_raw.json` -- because
every characterization clip's `duration_ratio` (8.6-17.9, see
`results/duration_characterization.json`) falls far outside `[0.5, 2.0]`. `compute_wer` returned
`wer=None` for all 3 without ever calling `model.transcribe()`. This confirms the prediction in
`research/research_summary.md` point 9 and `plan/plan.md`'s Approach section: an ASR-round-trip
layer, as currently implemented in `tts_eval_harness`, would almost never actually run on the exact
class of clip this task investigates -- it needs a plausible-duration clip to transcribe in the
first place, and v11's blowup produces the opposite.

## Per-call cost (even when gated out)

`WhisperModel(...)` is constructed fresh inside every `compute_wer()` call
(`tasks/t0008_tts_eval_harness_baselines/code/scoring.py:301`, not cached across calls), so even a
fully gated-out call pays the full model-load cost: 8.79s wall time for the first call (includes the
one-time `base.en` weights download), 2.6-3.4s for subsequent calls (model already cached locally,
but still reloaded into memory each call). This is a `tts_eval_harness` implementation detail, not
something specific to this evaluation, but it means wiring this in as a per-clip, always-on third
gate layer would add several seconds of CPU-only overhead per synthesis call even when the duration
gate skips the actual transcription.

## Verdict: do not implement now; documented recommendation instead

Per `plan/plan.md`'s explicit conditional ("if the evaluation concludes it is trivially easy to wire
in as a fully optional (default-off) parameter to `check_audio_quality()`, add it in this step; if
not, do not implement it"): **not implemented**. Two independent reasons:

1. **Wrong tool for this specific failure mode.** The duration-sanity and longest-non-silent-run
   signals added in Step 10 already catch v11's blowup directly and cheaply (no model load, no
   transcription). An ASR-round-trip layer's own duration pre-gate means it would skip transcribing
   exactly the clips this task's gate needs to catch -- it is not a stronger version of the same
   check, it is a check for a *different* failure mode (mistranscribed/garbled-but-normal-duration
   speech) that this task did not find evidence of in `kokoro-v11-best`'s output.
2. **Non-trivial integration cost for the third-layer role `task_description.md` proposes.** Wiring
   `compute_wer` into `audio_quality_check.py` as an optional parameter would require: (a) a
   reference `duration_ratio` computed against a REAL reference recording, which
   `check_audio_quality()` does not currently have access to (unlike `compute_duration_ratio`, which
   is reference-paired); (b) reworking `compute_wer` to accept a pre-loaded `WhisperModel` so
   repeated calls do not each pay the multi-second model-load cost; (c) a new dependency
   (`faster-whisper`, `jiwer`) added to this task's `.venv-styletts2` only, not the main project
   environment. None of this is a "trivially easy... fully optional parameter" addition.

**Recommendation for a future task**: if a stronger, general-purpose intelligibility check is wanted
(catching mistranscription/garbling in *normal-duration* clips, which is a distinct failure mode
from anything found in this task), evaluate `compute_wer` again with (a) a cached/shared
`WhisperModel` instance and (b) a text-only (not reference-paired) duration-sanity gate matching
`audio_quality_check.py`'s own `estimate_naive_duration_bound_s`, rather than the reference-paired
`compute_duration_ratio` this evaluation used as-is.
