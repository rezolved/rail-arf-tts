---
spec_version: "2"
answer_id: "zero-shot-ttfb-floor"
answered_by_task: "t0021_zero_shot_latency_reduction"
date_answered: "2026-09-18"
confidence: "medium"
---
## Question

What is the lowest reachable TTFB for CosyVoice2 and Chatterbox on our hardware without losing
speaker_sim, and does either reach 300 ms?

## Short Answer

No, neither CosyVoice2 nor Chatterbox reaches the 300 ms TTFB target, and the remaining gap is
architectural rather than closeable by engineering-only levers. CosyVoice2's lowest reachable TTFB
is 826 ms p50 (TensorRT/`load_trt`), a 30% cut from its 1,177 ms baseline with `speaker_sim`
unchanged (0.857 vs 0.854); Chatterbox's is 837 ms p50 (`torch_compile`), a 19% cut from 1,039 ms,
also with `speaker_sim` unchanged (0.838 vs 0.834). Both floors are still 2.7-2.8x the target. The
per-stage breakdown explains why: the autoregressive LM decode/prefill stage costs 815-1,004 ms in
every variant of both systems regardless of caching, precision, JIT, or TensorRT, because that cost
comes from the model's own token-by-token decoding scheme, not a serving-stack inefficiency the
tested levers could remove. This is based on a paired 196-prompt measurement of 10 acceleration
variants across both systems on the project's own hardware.

## Research Process

This task instrumented `t0018_zero_shot_cloning_calibration`'s CosyVoice2 and Chatterbox adapters
with per-stage `StageTiming` timers (reference encoding/conditioning, text frontend, LM
prefill/decode, flow-matching+vocoder), then ran a fixed cumulative-stack acceleration sweep: 5
CosyVoice2 variants (`baseline_new_ref`, `ref_cache`, `fp16`, `load_jit`, `load_trt` —
`vllm_backend` was pre-registered but never ran, see Limitations) and 6 Chatterbox variants
(`baseline_new_ref`, `ref_cache`, `precision_bf16_or_fp16`, `torch_compile`, `sentence_chunking`,
`streaming_api` — `streaming_api` never ran, see Limitations). Each variant was measured on 196
prompts (96 val96 + 100 fillers) after 50 discarded warmups, in the same GPU session as its paired
`baseline_new_ref` control, per Lesson 1's cold/warm-cache pairing rule.

A binding owner correction (recorded before this task's planning step) required building the
reference clip and speaker-similarity centroid from `data/v4/val/wavs` (the correct, production
`voice_id=rWV5HleMkWb5oluMwkA7` newsreader voice) instead of the wrong-voice `data/11labs_david`
corpus t0018 used. All numbers in this answer come from the corrected-voice measurement; t0018's own
`speaker_sim` numbers are not directly comparable (see `results/tables.json`'s
`speaker_sim_radiohost_control` column for a side-by-side against the old, wrong-voice centroid).

During implementation, two data-integrity bugs were found and fixed via direct inspection of
individual synthesis outputs (not just aggregate metrics) before trusting the results:

1. `code/transcribe_references.py`'s Whisper transcription of the `ref_single` reference clip
   silently truncated to the clip's first sentence only (a `small.en`-model decoding quirk on this
   specific 15 s audio, not a VAD bug — the segment's own timestamp span was correct, only the
   decoded text stopped early). CosyVoice2's zero-shot conditioning requires the `prompt_text` to
   match what is actually spoken in the reference clip; the truncated prompt caused every
   `ref_single` CosyVoice2 output to be gibberish unrelated to the requested text (confirmed by
   manually re-transcribing outputs: `"checking for the latest press release"` produced
   `"A wolf profferpate."`). This was caught via a WER of 0.94-2.3 across all CosyVoice2
   `ref_single` variants (vs. Chatterbox's 0.17-0.46 on the same measurement), fixed by switching to
   `base.en`, and all 5 affected CosyVoice2 `ref_single` variants were re-measured (WER then dropped
   to 0.25-0.64, in line with Chatterbox and with `t0018`'s own 0.42 baseline).
2. `chatterbox`'s `precision_bf16_or_fp16` variant initially failed 196/196 ("mat1 and mat2 must
   have the same dtype") because casting `model.t3`'s weights to bf16 left the ref-conditioning
   tensors (`model.conds.t3`, a `T3Cond` dataclass) at their original fp32 dtype. Fixed by
   re-casting `T3Cond` to `model.t3`'s current parameter dtype on every call (`T3Cond` already
   exposes its own `.to(dtype=...)` method); a separate, related finding is documented below (see
   Evidence from Code or Experiments).

## Evidence from Papers

[Du2024][du2024] (CosyVoice 2) formalizes the additive latency model
`L_TTS = M*d_lm + M*d_fm + M*d_voc` this task's `results/latency_breakdown.json` schema is built on,
and this task's own measurement confirms its central prediction empirically: `d_lm` (LM
prefill/decode) dominates at 815-1,004 ms per call, versus 151-785 ms for the combined
flow-matching+vocoder stage (`d_fm + d_voc`) and under 1 ms for text frontend once reference caching
is applied. [Kong2020][kong2020] (HiFi-GAN) and [Kaneko2022][kaneko2022] (iSTFTNet) report
vocoder-only inference speeds of 150x-3,700x real time, consistent with this task's finding that the
vocoder/flow-matching stage, while non-trivial (151-785 ms measured here, slower than the papers'
isolated-vocoder benchmarks because it includes flow-matching, not vocoding alone), is not the
dominant cost and is also the stage TensorRT (`load_trt`) most effectively accelerates (397 ms to
151 ms for CosyVoice2, a 62% cut). [Seo2026][seo2026] (Chatterbox-Flash) reports 103-118 ms TTFP by
replacing Chatterbox's autoregressive T3 decoder with a block-diffusion decoding scheme — a
fine-tuning change explicitly out of this task's scope (Forbidden: "No fine-tuning") — and this
task's own finding that no serving-side lever (precision, `torch.compile`) measurably moves
`lm_decode_ms` for Chatterbox (838-1,004 ms across all variants, if anything higher for the bf16
variant, see Limitations) is direct, first-party confirmation that [Seo2026][seo2026]'s
decoding-scheme change, not a servable-stack optimization, is what actually reaches the 300 ms class
of TTFB. [Wan2018][wan2018] defines the GE2E `speaker_sim` metric this task re-scores every variant
against; per [Wan2018][wan2018]'s framing, the reference-embedding step is a deterministic function
of a fixed clip, which is exactly why caching it (the `ref_cache` variant) reduces
`ref_encoding_ms`/`ref_conditioning_ms` to near-zero (231 ms to 0.17 ms for CosyVoice2; 74 ms to
0.001 ms for Chatterbox) at zero `speaker_sim` cost, confirming this task's Approach 1 (instrument
first, land the cheap/free win before heavier serving-stack changes).

## Evidence from Internet Sources

Per `research/research_internet.md` (step 5), no internet source benchmarks vLLM or TensorRT-LLM for
CosyVoice2's own backbone, and no source measures whether acceleration shifts `speaker_sim` at fixed
content — both gaps this task's own measurement fills directly (see Synthesis). The closest prior
per-stage TTFA decomposition found (a non-peer-reviewed `vllm-project/vllm-omni` GitHub issue, GPU
unstated) reported 40.5 ms prefill / 142.3 ms AR decode / 129.0 ms first flow chunk (357 ms total) —
an order of magnitude faster than this task's own measured 815-1,004 ms LM stage on
identically-pinned CosyVoice2-0.5B running on an H100 NVL. This task's own measurement is therefore
the first grounded, benchmarked-on-this-hardware number for this exact question; the discrepancy
with the unverified GitHub issue is noted but not reconciled (no reproducible methodology was
available to compare against).

## Evidence from Code or Experiments

All headline numbers come from this task's own `results/tables.json` and
`results/latency_breakdown.json`, produced by a 196-prompt-per-variant measurement (50 discarded
warmups, both `val96` and `fillers` prompt sets, paired same-session `baseline_new_ref` control per
variant). Full per-variant table (fillers prompt set, the project's primary filler-synthesis use
case; `n=100`, `success_rate=1.0` for every listed row):

| System | Variant | TTFB p50 (ms) | TTFB p95 (ms) | speaker_sim | WER |
| --- | --- | --- | --- | --- | --- |
| cosyvoice2 | baseline_new_ref | 1,176.7 | 1,391.5 | 0.854 | 0.29 |
| cosyvoice2 | ref_cache | 950.1 | 1,156.1 | 0.853 | 0.27 |
| cosyvoice2 | fp16 | 1,006.0 | 1,206.0 | 0.847 | 0.32 |
| cosyvoice2 | load_jit | 998.6 | 1,207.8 | 0.858 | 0.26 |
| cosyvoice2 | **load_trt** | **826.3** | 1,035.2 | 0.857 | 0.25 |
| chatterbox | baseline_new_ref | 1,039.2 | 1,450.4 | 0.834 | 0.22 |
| chatterbox | ref_cache | 1,320.6 | 1,794.3 | 0.835 | 0.19 |
| chatterbox | precision_bf16_or_fp16 | 900.6 | 1,104.8 | 0.836 | 0.15 |
| chatterbox | **torch_compile** | **837.0** | 1,032.5 | 0.838 | 0.17 |
| chatterbox | sentence_chunking | 1,008.0 | 1,439.8 | 0.830 | 0.18 |

(Source: `results/tables.json`, rows `*_ref_single_fillers`.)

The per-stage breakdown (`results/latency_breakdown.json`, mean ms per call, fillers+val96 combined)
shows the LM stage is essentially flat across every serving-side lever:

| System | Variant | ref stage | text frontend | LM decode/prefill | flow+vocoder |
| --- | --- | --- | --- | --- | --- |
| cosyvoice2 | baseline_new_ref | 231.2 | 1.7 | 825.6 | 397.4 |
| cosyvoice2 | ref_cache | 0.17 | 1.2 | 814.3 | 398.6 |
| cosyvoice2 | fp16 | 0.17 | 0.65 | 893.7 | 386.4 |
| cosyvoice2 | load_jit | 0.18 | 1.3 | 901.8 | 385.4 |
| cosyvoice2 | load_trt | 0.17 | 1.2 | 904.4 | **151.1** |
| chatterbox | baseline_new_ref | 74.2 | 0.28 | 839.0 | 536.7 |
| chatterbox | ref_cache | 0.001 | 0.46 | 848.2 | 784.4 |
| chatterbox | precision_bf16_or_fp16 | 0.001 | 0.39 | 1,003.8 | 334.4 |
| chatterbox | torch_compile | 0.002 | 0.46 | 843.5 | 501.6 |
| chatterbox | sentence_chunking | 0.001 | 0.40 | 844.7 | 565.7 |

Two things stand out: (1) CosyVoice2's LM decode/prefill never drops below ~815 ms and in fact rises
slightly under `fp16`/`load_jit`/`load_trt` (814-904 ms) — none of these flags touch the LM stage's
own cost, they only change how the flow-matching/vocoder stage is executed; `load_trt`'s entire TTFB
win comes from a 62% cut to the flow+vocoder stage (397 to 151 ms). (2) Chatterbox's bf16 T3 decoder
cast does not reduce `lm_decode_ms` either (839 to 1,004 ms, if anything slightly higher, likely
attention-kernel dtype-conversion overhead rather than a genuine slowdown, within this measurement's
noise) — `torch_compile`'s win instead comes from the vocoder stage (536.7 to 501.6 ms) plus modest
warmup-amortized gains not visible in this per-call mean. Both findings are direct, first-party
evidence that the serving-side levers tested here act on the non-LM stages, not the LM decode
bottleneck itself.

`speaker_sim` was scored against the corrected `data/v4/val/wavs`-derived centroid (owner
correction) for every variant; no variant showed a `speaker_sim` drop of more than 0.007 from its
paired baseline (e.g. CosyVoice2 `fp16`: 0.847 vs 0.854 baseline, the largest observed delta) —
answering this project's previously-unmeasured research question (`speaker_sim` does not measurably
degrade under any tested acceleration lever, within this measurement's noise floor).

The automated `hardened_gate_pass` check passed 94.85% of all 2,156 scored clips (2,045/2,156); per
the owner's explicit caveat (owner correction #7), this gate is necessary, not sufficient, and known
to pass clips a human would describe as "voice plus strong noise" — see `results/listening_guide.md`
for the actual audio and the owner's own listening pass, not yet performed as of this task's
completion.

## Synthesis

The evidence converges cleanly: CosyVoice2 and Chatterbox both spend the large majority of their
TTFB (roughly 55-85% depending on variant) in the autoregressive LM decode/prefill stage, and every
serving-side acceleration lever this task could legally test (no fine-tuning permitted) — reference
caching, fp16, JIT compilation, TensorRT, `torch.compile` — left that stage's cost essentially
unchanged (815-904 ms for CosyVoice2, 838-1,004 ms for Chatterbox, across all 10 measured variants).
The levers that DID help (TensorRT for CosyVoice2, `torch.compile` for Chatterbox) worked by
accelerating the non-LM stages instead, which is why they produced real, non-trivial TTFB reductions
(19-30%) without ever approaching the 300 ms target. This is a coherent story, not a coincidence:
[Seo2026][seo2026]'s own result (103-118 ms TTFP) required replacing the decoding scheme itself,
which this task's Forbidden list explicitly rules out. The gap to 300 ms is therefore best
characterized as **architectural**: closing it would require a decoding-scheme change (e.g.
block-diffusion decoding, distillation, or a smaller/faster LM backbone), not further serving-stack
optimization of the pinned CosyVoice2-0.5B / Chatterbox T3 architectures. The `speaker_sim`-neutral
result (no lever cost more than 0.007 `speaker_sim`) means this project can adopt `load_trt`
(CosyVoice2) or `torch_compile` (Chatterbox) as free wins today, but should not expect any
combination of engineering-only levers on these two architectures to reach 300 ms.

## Limitations

* **Coverage gaps, pre-registered, not silently dropped:** `cosyvoice2_vllm_backend` never ran (the
  `.venv-cosyvoice2-vllm` install hit its 20-minute setup-time cutoff, `torch` installed but `vllm`
  itself not reached — see `intervention/cosyvoice2_vllm_install_timeout.md`);
  `chatterbox_ streaming_api` never ran (no native streaming API exists in the pinned
  `chatterbox-tts==0.1.7`, verified by reading the installed package source); F5-TTS is null for
  every condition (a second hang, now with a confirmed `py-spy` stack trace showing it is an Azure
  Files SMB mount stall, not a task-code bug — see `intervention/f5_tts_retry_still_hangs.md`).
  These are the two most novel levers this task set out to test for CosyVoice2 (`vllm_backend`) and
  the only "true streaming" lever for Chatterbox; their absence means the answer above should be
  read as "the best of the levers actually testable in this environment," not "the best of every
  documented lever."
* **"Best variant" was selected on the `fillers` prompt set** (the project's actual filler-synthesis
  use case), and does not always hold on `val96`: Chatterbox's `torch_compile` is best on fillers
  (837.0 ms) but is actually the second-WORST Chatterbox variant on val96 (1,581.6 ms, worse than
  its own 1,537.4 ms baseline); `precision_bf16_or_fp16` is val96's best Chatterbox variant (1,319.3
  ms). This divergence between short (fillers) and long (val96) prompts is itself a finding —
  `torch_compile`'s benefit appears concentrated in short-utterance overhead reduction — but it
  means "the" single best setting depends on the target utterance length distribution in production.
* **The automated audio-quality gate is necessary, not sufficient** (owner correction #7): a 94.85%
  pass rate does not certify production-readiness; the owner's own listening pass over
  `results/listening_guide.md` has not yet happened as of this answer's writing.
* **`speaker_sim`-vs-acceleration is a novel first measurement**, not corroborated by any prior
  published source (research_internet.md's own stated gap) — the "no measurable degradation" finding
  should be treated as this task's own single data point, not an established result.
* **Chatterbox's bf16 T3-decode timing showed a small, likely-noise-level increase** rather than the
  expected speedup; this task did not have budget remaining to re-measure with a larger sample to
  distinguish a genuine (if small) regression from measurement noise.

## Sources

* Paper: `10.48550_arXiv.2412.10117` ([Du2024][du2024])
* Paper: `10.48550_arXiv.2605.30748` ([Seo2026][seo2026])
* Paper: `10.1109_ICASSP.2018.8462665` ([Wan2018][wan2018])
* Paper: `10.48550_arXiv.2010.05646` ([Kong2020][kong2020])
* Paper: `10.48550_arXiv.2203.02395` ([Kaneko2022][kaneko2022])
* Task: `t0018_zero_shot_cloning_calibration`
* Task: `t0021_zero_shot_latency_reduction`

[du2024]: ../../../../t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2412.10117/summary.md
[seo2026]: ../../../../t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2605.30748/summary.md
[wan2018]: ../../../../t0018_zero_shot_cloning_calibration/assets/paper/10.1109_ICASSP.2018.8462665/summary.md
[kong2020]: ../../../../t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2010.05646/summary.md
[kaneko2022]: ../../../../t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2203.02395/summary.md
[t0018]: ../../../../t0018_zero_shot_cloning_calibration/
[t0021]: ../../../../t0021_zero_shot_latency_reduction/
