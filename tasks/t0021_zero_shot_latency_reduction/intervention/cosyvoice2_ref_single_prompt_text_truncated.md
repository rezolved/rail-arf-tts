# CosyVoice2 `ref_single` measurements invalidated by a truncated `prompt_text`; found via WER,

# fixed, and the affected 5 variants re-run

## What happened

During post-hoc scoring (Phase 1.5 inspection of `merge_and_score.py`'s output), the aggregate WER
came back at `mean=0.9352` -- catastrophically high compared to t0018's own baseline (`mean=0.42` on
similar filler/val96 prompts). Breaking WER down by `(system, acceleration_variant, prompt_set)`
showed Chatterbox in the expected 0.17-0.46 range, but **every CosyVoice2 variant under the
`ref_single` condition** at 0.9-2.3 -- `baseline_new_ref` itself included, not just an accelerated
variant, so this was not an acceleration-lever bug.

Manually re-transcribing several `cosyvoice2_baseline_new_ref_ref_single` output clips with
`faster-whisper` directly confirmed the audio is pure gibberish unrelated to the requested text --
e.g. the clip for `"checking for the latest press release"` actually says `"A wolf profferpate."`;
another says `"for Stih Gaspas."`; another `"is Paycase Asia, Wiester."` `speaker_sim` for these
clips was still reasonable (0.84-0.89), consistent with a cloned-voice-saying-nonsense failure mode,
not an acoustic/vocoder collapse.

## Root cause

`code/transcribe_references.py` (Milestone 1, run before any GPU work) used
`WhisperModel("small.en")` with `beam_size=5` to build `ref_single`'s `prompt_text` -- CosyVoice2's
zero-shot conditioning requires `prompt_text` to match what is actually spoken in `prompt_wav`
(`ref_single.wav`, a real 15.00 s single natural clip, per this task's own
`ref_single_is_concat: false` deviation). That transcription call returned exactly ONE segment whose
reported timestamp span correctly covered the full 15.08 s clip, but whose **decoded text was
silently truncated to only the clip's first sentence**:
`"I'm not able to compare us with other companies."` -- the clip actually continues for ~14 more
seconds (confirmed by direct listening/re-transcription:
`"...Resolve AI to live as purpose-built AI for commerce, unifying discovery, conversation, and data intelligence to help shoppers find products, get personalized guidance, and complete purchases, all within a seamless, all-"`,
cut off by the clip's own 15 s boundary). This is a `small.en`-model-specific early-stop decoding
quirk on this particular audio, not a VAD/silence-detection bug (the segment's own timestamps were
correct) and not a `ref_concat` problem (`ref_concat`'s transcript, built from the same script, was
already multi-sentence and reasonably complete before this fix -- its WER, 0-0.8 on the closure run,
was never anomalous).

A `prompt_text` that only describes 1/15th of the actual reference audio confuses CosyVoice2's
zero-shot LM conditioning badly enough to produce unrelated output text for every subsequent
`ref_single` synthesis call, regardless of which acceleration variant is active -- explaining why
`baseline_new_ref` was affected identically to `fp16`/`load_jit`/`load_trt`/`ref_cache`.

## Fix

Switched `transcribe_references.py`'s `WHISPER_MODEL_SIZE` from `"small.en"` (with `beam_size=5`) to
`"base.en"` (plain defaults, matching `t0008`'s own `WHISPER_MODEL_SIZE` used throughout this
project's WER scoring) -- verified directly to transcribe the full clip correctly across multiple
segments. Re-ran `transcribe_references.py`; `data/references/manifest.json`'s
`ref_single_transcript` now reads the full multi-sentence text.

## Re-measurement

All 5 CosyVoice2 `ref_single` variants (`baseline_new_ref`, `ref_cache`, `fp16`, `load_jit`,
`load_trt`) were re-run in full (50 warmups + 196 measured prompts each) against the corrected
`prompt_text`, in a fresh VM re-acquisition (the VM had been idle-stopped by the watchdog in the
interim, as designed -- no incident, just normal protection kicking in during the long CPU-only
scoring wait). `results/per_clip_metrics.json`, `results/environment.json`, and the per-variant
`latency_breakdown_*_ref_single.json` files were regenerated and re-merged/re-scored.
`cosyvoice2_baseline_new_ref_ref_concat` (the S-0018-02 closure item, a different `prompt_text`
entirely) was NOT re-run -- it was never affected by this bug.

## Why this was not caught earlier

Milestone 1's own `transcribe_references.py` run only asserted `len(ref_single_text) > 0`
(non-empty), which a truncated-but-non-empty transcript trivially satisfies. The per-stage
`StageTiming` smoke gates (Step 5) only checked that timing fields summed correctly and that a
synthesis call completed without raising -- they never checked whether the OUTPUT TEXT matched the
INPUT TEXT, because no ASR/WER step runs during the smoke gate (WER is a scoring-time,
`merge_and_score.py`-only computation, run only after the full sweep). This is exactly the failure
mode Phase 1.5 of the implementation skill warns about ("a model with access to context that
performs at or below a context-free heuristic is broken, not bad") -- but the smoke gate's own scope
(one clip, no ASR) could not have caught it; only the full aggregate WER did, after the fact.
**Recommendation for future tasks' plans:** add a cheap ASR sanity check on the reference clip's OWN
constructed `prompt_text` (transcribe the reference audio a second, independent way, or at minimum
print its full length/segment count) as part of Milestone 1's reference-construction step, before
any GPU synthesis begins -- this would have caught the truncation for $0 additional cost.

## Cost impact

The original (invalidated) 5-variant CosyVoice2 `ref_single` sweep cost approximately 5 x ~11
minutes x $13.96/hr $\approx$ $12.80 of GPU time that produced unusable data. The re-run cost a
similar amount again. Combined, this bug cost roughly $25-26 of this task's ~$100 budget -- a real,
avoidable cost, recorded honestly here rather than smoothed over. See `results/cost_tracking.json`
for the exact running ledger.
