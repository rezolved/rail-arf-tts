# Can Zero-Shot Cloning Reach 300 ms TTFB?

## Motivation

t0018 showed CosyVoice2 (`speaker_sim` 0.84-0.86) and Chatterbox (0.80-0.81) clone David's voice
better than any Kokoro fine-tune, but their time to first byte is 1.3-2.9 s against a 300 ms target.
The product's text is fully dynamic, so pre-generation is not an option; latency is the only thing
standing between these systems and production. Both models generate faster than real time (RTF
0.40-0.54 on H100), which means the delay is in start-up and first-chunk work, not raw throughput,
and start-up work is exactly what engineering usually can cut.

This task runs in parallel with the Kokoro training line (t0017, t0019, t0020). It answers one
question: **what is the lowest TTFB each system can reach on our hardware without losing speaker
similarity, and does either reach 300 ms?**

## What t0018 already established (do not re-measure blindly)

* CosyVoice2 was already run with `inference_zero_shot(stream=True)` and still measured 2.86 s p50
  TTFB on val96 (`code/adapters_zeroshot.py`). Streaming alone is not the answer; the first chunk
  itself is slow.
* Chatterbox was run whole-utterance, no streaming, 1.34-1.6 s p50.
* CosyVoice2's `ref_concat` failed only because the reference was 30.57 s against a 30 s hard limit
  (S-0018-02). F5-TTS hung on load three times (S-0018-01).
* Reference clips, prompt sets, adapters and the harness wiring live in t0018's `code/` and
  `data/references/`; reuse them.

## Key Questions

1. Where does the time go? For each system, a per-stage breakdown of the first-chunk latency:
   reference-audio encoding (speaker embedding, prompt tokens), text frontend, LLM prefill and first
   decode tokens, flow-matching / token-to-wav for the first chunk, vocoder. Measured with
   timestamps inside the adapter, warm cache, 50 warm-up requests discarded (Lesson 1).
2. How much of that is per-request work that can be cached per voice? The reference-audio encoding
   is identical for every request with the same David clip; caching it is free latency.
3. What does each documented acceleration path buy, one at a time, on the same prompts:
   * CosyVoice2: `load_jit`, `load_trt` (the flow-matching TensorRT export the repo ships), `fp16`,
     the vLLM LLM backend the repo supports, smaller first-chunk size in streaming mode, and the
     cached reference embedding from Q2.
   * Chatterbox: sentence-level chunking so the first sentence returns before the rest is generated,
     `torch.compile` / fp16 / bf16, cached voice conditioning, and any streaming API the package
     exposes at the pinned version.
4. Which combination gives the lowest TTFB p50 and p95 for each system, and what is `speaker_sim`,
   WER and gate-failure rate at that setting versus t0018's baseline setting? Speed that costs
   similarity is not a win; report both.
5. Does either system reach TTFB p50 ≤ 300 ms? If not, how far away is it, and is the remaining gap
   architectural (an autoregressive LLM must produce N tokens before any audio) or engineering?
6. Cheap closures folded in, each bounded: re-run CosyVoice2 `ref_concat` with a 29.5 s reference
   (S-0018-02, one cell, both prompt sets); retry F5-TTS on a fresh session with `py-spy dump` on a
   hang (S-0018-01, 45 minutes cap, null with the stack trace if it hangs again).

## Protocol

Same harness protocol as t0018: smoke gate, 50 warm-up requests, val96 + 100 fillers + the three
gate texts, per-clip TTFB / RTF / duration / WER / GE2E cosine vs the half-B centroid, hardened gate
on every clip. Use the corrected harness from t0019 if it has merged by the time this task starts;
otherwise t0008's, stated explicitly. Every acceleration variant is a separate metrics variant; the
t0018 baseline setting for each system is re-run in the same session as the paired control (Lesson
1). Rejection: `successful_prompts / total_prompts < 0.8` nulls a variant (Lesson 3). Capture
engine, CUDA, TensorRT and vLLM versions per variant (Lesson 4).

## Expected Outputs

* `assets/answer/zero-shot-ttfb-floor/` — the answer: best reachable TTFB per system, the setting
  that achieves it, the similarity cost, and whether 300 ms is reachable (yes / no / not with this
  architecture).
* `results/latency_breakdown.json` — per-stage timings per system per variant.
* `results/metrics.json` (variants: system × acceleration setting × prompt set),
  `per_clip_metrics.json`, `costs.json`, `remote_machines_used.json`.
* Charts in `results/images/`, embedded: `latency_breakdown_stacked.png` (stacked bars per
  system/variant, one segment per stage; Q1-Q3), `ttfb_vs_speaker_sim_variants.png` (scatter, x:
  TTFB p50 with a line at 300 ms, y: fillers `speaker_sim`, one point per variant, t0018 baselines
  marked; Q4-Q5), `ttfb_p50_p95_by_variant.png`.
* Tables: per variant (TTFB p50/p95/p99, RTF, `speaker_sim` mean ± std, WER, gate failures, n
  successful); environment table per variant.
* `results/audio_samples/` (DVC): the ten comparison texts from t0018 through the best setting of
  each system, next to t0018's baseline output and the ElevenLabs original, plus all harness clips;
  `results/listening_guide.md` so the owner can hear whether the fast setting still sounds like
  David.
* `results/suggestions.json` — including, if a system lands near 300 ms, a production-integration
  feasibility task, and if not, whether a distilled or smaller model is the next lever.

## Compute and budget

* `LLM-T1-NC80`. Setup incl. TensorRT/vLLM builds about 1 h; profiling about 0.5 h; about 6
  acceleration variants × 2 systems at about 0.25 h each (196 prompts, faster than t0018 as caches
  warm) about 3 h; S-0018-02 cell 0.3 h; F5-TTS retry 0.75 h cap; teardown. **About 5.5 h × $13.96 ≈
  $77. Hard cap: $100.** Watchdog armed and PID confirmed before the first build (Lesson 8); build
  artifacts and weights on `/mnt/cache/persist/` (Lesson 10).

## Forbidden

* No fine-tuning. No change to reference selection tuned on val96.
* No variant reported without its paired same-session baseline.
* No substitution of a model version without recording it in the environment table.

## Dependencies

* `t0018_zero_shot_cloning_calibration` — adapters, references, prompt sets, baseline numbers and
  the two open suggestions this task closes.
