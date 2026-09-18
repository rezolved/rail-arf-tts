---
spec_version: "3"
task_id: "t0021_zero_shot_latency_reduction"
step_number: 11
step_name: "creative-thinking"
status: "completed"
started_at: "2026-09-18T22:50:54Z"
completed_at: "2026-09-18T23:20:00Z"
---
## Summary

Out-of-the-box review of the 10-variant acceleration sweep (steps 8-10) before the `results`
step-executor locks in headline numbers and the `assets/answer/zero-shot-ttfb-floor/` answer asset's
"architectural, not engineering-closeable" conclusion. Four angles were pressure-tested against the
actual per-stage `latency_breakdown*.json` and `tables.json` numbers, looking specifically for
levers outside the tested cumulative-stack matrix (distillation, smaller models, pipelining,
product-level bypasses) per this step's "alternative levers beyond the planned acceleration matrix"
brief. This step produced no code changes — it is a written critique and idea layer for the
`results`/`suggestions` steps to draw on, per SKILL.md Phase 5's "optional, out-of-the-box analysis"
description.

## Actions Taken

1. Re-read `checkpoint.md` in full (steps 1-10), with particular attention to the owner-correction
   binding requirements and the implementation step's two mid-sweep bug fixes.
2. Read `research/research_summary.md` in full, focusing on the `L_TTS = M*d_lm + M*d_fm + M*d_voc`
   additive latency model, the `vllm-omni` RFC's per-stage TTFA decomposition, and Approach 3's
   (untested) CosyVoice2 streaming-chunk-size recommendation.
3. Read `results/tables.json` in full (all 22 variant rows, including the 6 null rows for
   `cosyvoice2_vllm_backend`, `chatterbox_streaming_api`, and `f5_tts`) and `results/metrics.json`,
   cross-checking every `latency_breakdown` sub-stage (`ref_encoding_ms`/`ref_conditioning_ms`,
   `text_frontend_ms`, `lm_prefill_decode_ms`/`lm_decode_ms`,
   `flow_matching_vocoder_ms`/`vocoder_ms`, `total_ms`) against each variant's
   `ttfb_ms_p50`/`ttfb_ms_p95`/`speaker_sim`.
4. Read `assets/answer/zero-shot-ttfb-floor/{short_answer.md,full_answer.md}` in full to identify
   which findings the implementation step already reported, so this step could avoid restating them
   and instead add genuinely new angles.
5. Read `tasks/t0021_zero_shot_latency_reduction/plan/plan.md`'s Step by Step (Milestone 3) to
   confirm exactly which acceleration variants were pre-registered per system, and cross-checked
   that list against `results/tables.json`'s actual rows to find any research-recommended lever that
   was never scheduled at all (not merely one that failed at runtime).
6. Traced `data/filler_prompts_100.json`'s provenance via
   `t0008_tts_eval_harness_baselines/code/ prepare_prompts.py` (`build_filler_prompts`, sampling
   from `data/11labs_david/`) to check whether the "fillers" prompt set represents a small,
   cacheable, closed-set production filler catalog or an open-ended sample, before proposing a
   caching-based lever.
7. Wrote the four-part creative-thinking analysis below and cross-checked its claims against
   `LESSONS.md` Lesson 1 (cold/warm-cache pairing) where a within-session variance anomaly is
   discussed.

## Outputs

* This step log (`logs/steps/011_creative-thinking/step_log.md`) — the analysis is written inline
  here (see `## Analysis` below), matching the established convention from
  `t0018_zero_shot_cloning_calibration`'s `011_creative-thinking/step_log.md` (this task's own
  dependency): `logs_specification.md` and `step_registry.py`'s `creative-thinking` entry
  (`required_files`: `step_log.md`, min 100 words, no other required file) specify no dedicated
  output path, and `SKILL.md` Phase 5 describes the step only as "Out-of-the-box analysis... Update
  `checkpoint.md`."
* `checkpoint.md` — Step History entry, Next Step Notes rewritten for the `results` step-executor.

## Issues

No issues encountered. This step involved no GPU/remote-machine work (confirmed already torn down in
step 10, `destroyed_at` non-null in `machine_log.json`) and no code execution beyond reading
existing task files.

## Analysis

### 1. The measured "TTFB floor" may be a non-pipelined instrumentation artifact

The answer asset's central claim is that the LM decode/prefill stage (815-1,004 ms across every
variant of both systems) is an architectural cost the tested levers cannot touch. That claim is well
supported by the data for the levers actually tested — but a closer read of the stage-timing numbers
raises a question the answer asset does not ask: is `lm_prefill_decode_ms` timing the entire
utterance's token generation, sequentially followed by flow-matching, or only the tokens needed
before the *first* flow-matching chunk can start?

* For CosyVoice2 `load_trt` (the best variant), `lm_prefill_decode_ms` (904.4 ms) plus
  `flow_matching_vocoder_ms` (151.1 ms) sums to 1,055.5 ms, matching `total_ms` (1,056.9 ms) almost
  exactly. This is consistent with a **sequential** pipeline: run the LM to completion, then run
  flow-matching on the full token sequence, then emit audio. If that is what the copied adapters
  actually do (they were built in t0018 with no per-stage timestamps at all, per
  `research/research_code.md`'s caveat, so this task's own `StageTiming` checkpoints are the first
  instrumentation of this boundary), then `ttfb_ms` here is closer to "time to first byte of a
  fully-decoded utterance" than "time to first byte of a properly interleaved streaming pipeline."
* Research point 4 in `research_summary.md` is directly relevant and is *prior negative evidence*
  against a naive fix: "Streaming removes the quality penalty of chunking, not the latency — the LM
  must still generate enough tokens before flow-matching runs (explains t0018's 2.86s p50 with
  `stream=True`)." A bare `stream=True` flag made TTFB *worse* in t0018, not better. But this task's
  own plan (Milestone 3) never scheduled a CosyVoice2 chunk-size-tuned streaming variant at all —
  Approach 3 in `research_summary.md` explicitly recommends "evaluate smaller streaming chunk sizes"
  for CosyVoice2, citing [Du2024]'s Table 8 finding that this is near-zero quality cost on typical
  text — and that recommendation was never scheduled as a matrix cell for CosyVoice2 (only
  Chatterbox got a `sentence_chunking` variant; CosyVoice2's 6 variants are `baseline_new_ref`,
  `ref_cache`, `fp16`, `load_jit`, `load_trt`, `vllm_backend`, with zero streaming/chunk-size
  cells). This is a genuine gap between what research recommended and what was measured, distinct
  from `vllm_backend`'s and `streaming_api`'s already-documented "attempted but failed" gaps — this
  one was never attempted.
* The concrete, testable hypothesis for a follow-up task: measure CosyVoice2 with `stream=True`
  **and** a small, explicitly tuned `token_hop_len` (CosyVoice2's own streaming chunk-size
  parameter, distinct from the un-tuned default that likely produced t0018's 2.86 s regression),
  instrumenting TTFB as "time to first flow-matching chunk's audio," not "time to the full decoded
  utterance." If the `vllm-omni` RFC's 142.3 ms AR-decode-to-first-chunk number (research finding
  #3, admittedly a different, unbenchmarked stack) is even directionally right, a properly chunked
  CosyVoice2 pipeline could plausibly beat the 904 ms full-decode floor by a wide margin, because
  most of that 904 ms is spent generating tokens for the *back half* of the utterance — tokens a
  correctly streaming pipeline would emit audio for concurrently with (not after) the front half's
  flow-matching pass. This does not contradict the answer asset's finding for the levers tested
  (precision/backend swaps genuinely cannot touch a sequential LM stage), but it means
  "architectural" may currently be conflating "this specific inference call shape" with "this
  model's true minimum latency," and the honest label for the untested streaming-chunk lever is
  "unknown," not "ruled out."

### 2. Why precision/JIT levers left LM decode flat or worse — and what that predicts for `vllm_backend`

Every precision or compilation lever tested moved the LM-decode number by less than 10%, and in
three of five cases (CosyVoice2 `fp16`/`load_jit`/`load_trt`, Chatterbox `precision_bf16_or_fp16`)
it got *worse*, not better, than the `ref_cache` baseline. The answer asset notes this without
offering a mechanism beyond "likely attention-kernel dtype-conversion overhead." A more complete
explanation, grounded in how autoregressive decoding actually behaves, is worth stating explicitly
because it predicts which untested lever should work:

* At batch size 1 (a single filler utterance, no concurrent requests — confirmed architecturally
  batch-1 for the flow-matching stage per research finding #3, and effectively batch-1 here for LM
  decode too since this task never tested request batching), autoregressive token-by-token decoding
  is typically **memory-bandwidth-bound**, not compute-bound: each step reads the full set of model
  weights (and growing KV-cache) from GPU memory to produce one token, and the arithmetic itself is
  a tiny fraction of that per-step cost. Halving the arithmetic precision (fp16/bf16) reduces FLOP
  cost, which is not the bottleneck at batch 1, while adding a real (if small) per-op
  dtype-dispatch/cast cost — which is exactly the "flat or worse" pattern measured for all four
  precision/JIT-adjacent variants on both systems.
* This mechanism predicts that `vllm_backend` (the one lever specifically built to attack this
  stage, and the one that never ran due to the 20-minute install cutoff) would help via a
  fundamentally different mechanism than precision: continuous batching, paged KV-cache, and
  fused/persistent decode kernels reduce *memory traffic per token*, not FLOP count — the actual
  bottleneck this data implies. That reframes the missing `vllm_backend` measurement from "one more
  variant that happened not to finish" to "the one variant in the entire matrix mechanistically
  aimed at the stage that dominates 55-85% of TTFB," which should be the highest-priority single
  follow-up action, above any further precision/compile tuning on the already-tested stages.
* A second, not-yet-considered lever in the same family: **speculative decoding** (a small draft
  model proposing multiple tokens per step, verified in parallel by the full LM) is specifically
  designed to reduce wall-clock time for memory-bandwidth-bound, batch-1 autoregressive decoding —
  closer to this task's actual bottleneck than either a serving-framework swap or a precision flag,
  and distinct from both (it changes the decoding algorithm's arithmetic-intensity profile without
  changing the model's weights or requiring `vllm_backend`'s full serving stack). It was not in the
  plan's Forbidden-list sense of "fine-tuning" (a draft model can be a separate, small,
  off-the-shelf or distilled model used only at inference time) and was not part of the tested
  matrix (precision/backend/chunking only, no decoding-algorithm variants) — a concrete,
  mechanism-grounded candidate for a follow-up task's matrix that neither this task's plan nor its
  research summary raised.

### 3. If the LM stage were hidden, the non-LM stages already clear 300 ms — quantified

A different way to read the same per-stage numbers: strip out `lm_prefill_decode_ms`/`lm_decode_ms`
entirely (as a proxy for a hypothetical fully-pipelined system where LM decode happens concurrently
with flow-matching/vocoding rather than before it, per angle 1 above) and ask what TTFB the
remaining stages alone would produce:

* CosyVoice2 `load_trt`: `ref_encoding_ms` (0.17) + `text_frontend_ms` (1.2) +
  `flow_matching_vocoder_ms` (151.1) ≈ **152.5 ms** — well under the 300 ms target, with more than
  100 ms of headroom to spare.
* Chatterbox `precision_bf16_or_fp16`: `ref_conditioning_ms` (0.001) + `text_frontend_ms` (0.39) +
  `vocoder_ms` (334.4) ≈ **334.8 ms** — 12% over the target, the closest any Chatterbox
  configuration gets.
* This is not a claim that either system already meets 300 ms (`ttfb_ms` as actually measured is
  826-1,161 ms and 837-1,681 ms respectively, and the LM stage cannot simply be deleted). It is a
  falsifiable, data-grounded statement about where the *ceiling* on a pipelining-based fix would
  land: CosyVoice2's non-LM stages are already comfortably inside the target, so if angle 1's
  hypothesis (interleaved streaming) is validated in a follow-up task, CosyVoice2 is architecturally
  much closer to 300 ms than the current "2.7-2.8x the target" framing suggests — the entire
  remaining gap would then be "can enough LM tokens for the first chunk be produced fast enough," a
  narrower and more tractable question than "can 900+ ms of LM decode be removed."
* This also explains, retroactively, why `load_trt`'s 62% cut to `flow_matching_vocoder_ms` (397 to
  151 ms) produced the single biggest TTFB win in the whole matrix even though it never touched the
  LM stage: it was optimizing the one stage that was already close to being the *sole* remaining
  cost in a hypothetical pipelined system, not just the sole remaining cost in the current
  sequential one.

### 4. Tail latency (p95), not just p50, should gate "best variant" selection

The answer asset selects `load_trt` (CosyVoice2) and `torch_compile` (Chatterbox) as the best
variants using `ttfb_ms` (p50/mean). The same `tables.json` rows carry `ttfb_ms_p95` values that
tell a different, product-relevant story the answer asset does not discuss:

* Chatterbox `baseline_new_ref` (val96): p50 1,537.4 ms, **p95 3,524.5 ms** — a 2.3x tail.
  `ref_cache` (val96) is worse on both ends: p50 1,681.0 ms, **p95 4,003.3 ms**, despite reducing
  `ref_conditioning_ms` to near-zero — the theoretically "free" caching win was outweighed by
  something else getting worse in that run (`vocoder_ms` rose from 536.7 to 784.4 ms in the same
  variant's breakdown, a 46% regression in a stage caching should not touch at all). Per
  `LESSONS.md` Lesson 1 (cold/warm-cache measurements are not silently pairable), this smells like
  within-session GPU-state variance rather than a genuine architectural cost of caching, but it was
  not flagged or re-measured, and it means `ref_cache` — plausible as a "free," zero-risk win on
  paper — was in practice the single worst-tail-latency Chatterbox variant measured on val96.
* `torch_compile` (Chatterbox), the answer asset's own pick for the `fillers` prompt set, is the
  **second-worst** p50 on val96 (already documented in the answer's Limitations) and also has a wide
  val96 tail (p95 3,289.1 ms) — worse than `precision_bf16_or_fp16`'s val96 p95 (3,590.7 ms is
  actually worse — precision is the wider-tailed one on val96, torch_compile the better-tailed one
  among the two, so the two limitations partially offset, but neither dominates the other on both
  ends). For a live voice-commerce filler pipeline, a caller occasionally waiting 3.3-4.0 seconds
  for a "just a moment" filler (a >10x overshoot of the 300 ms target, on top of an already-missed
  p50) is arguably a worse user-facing failure mode than a merely-elevated median, and no variant in
  this matrix was selected or reported with tail behavior as a criterion.
* Recommendation for `results`/`suggestions`: report p95 alongside p50 for every headline number
  (the data already exists in `tables.json`, so this is a reporting change, not a new measurement),
  and flag `ref_cache`'s Chatterbox val96 tail regression as an unresolved anomaly worth a
  repeat-run (multiple seeds/sessions) before treating any single-session `ref_cache` number as a
  reliable estimate — directly following `LESSONS.md` Lesson 1's caution about pairing measurements
  across runs without controlling for session-level variance.

## Recommendations Carried Forward

* `results/suggestions.json` (step 14) should propose, as distinct follow-up task ideas: (a) a
  properly chunk-size-tuned CosyVoice2 streaming variant, instrumented to measure
  time-to-first-chunk rather than time-to-full-decode, to test whether angle 1's "instrumentation
  artifact" hypothesis holds; (b) a retry of `cosyvoice2_vllm_backend` with a longer install budget,
  now explicitly motivated by the batch-1 memory-bandwidth mechanism in angle 2 rather than just
  "the lever that timed out"; (c) speculative decoding as a decoding-algorithm-level lever for the
  LM stage, orthogonal to both precision and serving-backend changes; (d) a repeat-run (multiple
  sessions/seeds) of the Chatterbox `ref_cache` variant specifically to resolve the vocoder-stage
  regression anomaly in angle 4 before it is relied on as a "free win"; (e) reporting `ttfb_ms_p95`
  alongside `ttfb_ms` in all headline tables going forward, per angle 4.
* A separate, product-architecture angle worth one line in `suggestions.json`, distinct from the
  above engineering-only items: `data/filler_prompts_100.json` is built by sampling from the
  1,358-clip `data/11labs_david` corpus
  (`t0008_tts_eval_harness_baselines/code/prepare_prompts.py`), i.e., production filler content is
  drawn from a large but *finite, pre-existing* catalog, not generated fresh per call. If Rezolve's
  actual voice-commerce filler inventory is similarly bounded, offline precomputation and caching of
  filler audio (near-zero live TTFB by construction) may be a more effective near-term path to a
  sub-300 ms filler-synthesis experience than any runtime acceleration lever, reserving live
  zero-shot synthesis for the smaller, more latency-tolerant slice of genuinely novel/dynamic
  content (e.g., inserting a live order number or product name). This task measured the
  live-synthesis engineering ceiling correctly and thoroughly; whether that ceiling actually needs
  to be reached for the production pipeline as a whole is a product-scoping question this task's own
  data cannot answer, and is worth surfacing rather than assuming.
* None of the above changes any file this task has already produced (`results/tables.json`,
  `results/metrics.json`, `assets/answer/zero-shot-ttfb-floor/`) — this step is a critique-and-idea
  layer for the `results`/`suggestions`/`reporting` steps (which have not yet run) to draw on, not a
  correction to any committed measurement.
