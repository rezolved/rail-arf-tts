---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
research_stage: "internet"
searches_conducted: 21
sources_cited: 15
papers_discovered: 8
date_completed: "2026-09-18"
status: "complete"
---
# Internet Research: Zero-Shot Cloning Latency Reduction

## Task Objective

t0021 profiles where CosyVoice2 and Chatterbox spend their 1.3-2.9 s p50 TTFB in Rezolve's
voice-commerce filler pipeline and tests streaming, chunking, vLLM/TensorRT backends, and precision
changes to find the lowest reachable TTFB without losing `speaker_sim` versus the ElevenLabs David
reference. This pass follows `research/research_papers.md` (11 papers reviewed) and targets its six
literature gaps — vLLM/TensorRT-LLM serving benchmarks for CosyVoice2's Qwen2.5-0.5B backbone, a
measured per-stage ms breakdown, precision ablations, direct Chatterbox data, F5-TTS load-hang
diagnosis, and acceleration's effect on speaker similarity — using GitHub, model cards, and
engineering sources a paper corpus alone cannot surface. 30 sources were consulted; the 15 most
actionable are kept below per the skill's size limit, prioritizing sources with measured numbers.

## Gaps Addressed

From `research_papers.md` Gaps and Limitations (six gaps):

1. **No vLLM/TensorRT-LLM benchmark for CosyVoice2's Qwen2.5-0.5B backbone or TensorRT flow-matching
   export.** **Partially resolved.** No peer-reviewed benchmark exists, but `vllm-project/vllm-omni`
   issue #6870 is an active RFC with measured concurrency/throughput numbers for a CosyVoice2/3
   serving stack [vllm-omni-6870], and a community HuggingFace checkpoint packages a vLLM-adapted LM
   backbone with no published numbers [CosyVoice2-vllm-HF].
2. **No full per-stage ms breakdown on H100-class hardware.** **Partially resolved.** Issue #6870
   gives a TTFA decomposition (median 357 ms: prefill 40.5 ms/11.3%, AR decode 142.3 ms/39.9%, first
   flow chunk 129.0 ms/36.1%) [vllm-omni-6870], but for vLLM-Omni serving, not the stock CosyVoice2
   repo t0018/t0021 use, and GPU unstated. Chatterbox has a comparable number (472 ms first-chunk,
   RTF 0.499) on RTX 4090, not H100 [chatterbox-streaming-GH]. Narrowed, not closed.
3. **No precision ablation for flow-matching/vocoder stages.** **Unresolved for this specific split;
   partially resolved for the LM stage.** Correction after full-text review of [Lian2026]: the
   dots.tts paper does **not** state a bf16-acoustic/quantized-LLM-trunk precision design — its own
   benchmark tables explicitly run in float32 with `torch.compile` disabled, and no precision
   ablation appears anywhere in the 22-page text. This citation was wrong in an earlier draft of
   this section and is corrected here; do not carry the bf16-acoustic claim forward. What remains
   supported: Chatterbox-TTS-Server measured ~40% throughput gain from bf16 on the decoder stage
   specifically, H100 explicitly listed as supported [Chatterbox-TTS-Server-GH]; an IEEE paper
   applies INT8 to LLM-based voice-cloning transformers generally, though its own full text could
   not be retrieved (IEEE Xplore blocked automated access) so its numbers are unverified
   [LlasaQuant2026]. No source isolates flow-matching/vocoder precision alone with a measured
   similarity score.
4. **No reviewed paper addresses Chatterbox directly.** **Partially resolved.** No new academic
   paper, but concrete engineering data: 472 ms first-chunk/RTF 0.499 on RTX 4090 with 50-token
   chunking [chatterbox-streaming-GH]; confirmation that `stream=True` is chunk-level, not
   token-level, because "the underlying model synthesizes a full chunk in one forward pass"; and a
   voice-conditioning cache keyed by `(path, mtime, exaggeration)` [Chatterbox-TTS-Server-GH].
5. **No paper on F5-TTS's load robustness.** **Partially resolved.** No source matches t0018's exact
   hang, but open `SWivid/F5-TTS` issues tie startup hangs to `jieba` dictionary initialization and
   stale Hugging Face cache state, with "clear cache and re-download" as the reported workaround
   [F5TTS-GH-Issues] — a plausible, unconfirmed hypothesis for S-0018-01.
6. **No paper on whether acceleration shifts speaker_sim at fixed content.** **Unresolved.** Only a
   general (non-CosyVoice2/Chatterbox) claim that INT8 quantization can change sampled token
   sequences and must be re-validated per voice/setting [Audio8-HackerNoon]. No controlled ablation
   isolating an acceleration lever's effect on speaker similarity was found anywhere.

## Search Strategy

**Sources searched**: WebSearch, GitHub (repos and issue trackers), Hugging Face model cards, arXiv,
NVIDIA developer blog, vLLM project blog/recipes, IEEE Xplore.

**Queries executed** (21 total, exact text; passes 1-8 gap-targeted, 9-14 broadening, 15-21
snowball):

1. `CosyVoice2 vLLM Qwen2.5-0.5B inference acceleration benchmark`
2. `CosyVoice2 TensorRT-LLM flow matching export latency`
3. `CosyVoice2 first chunk latency breakdown milliseconds H100`
4. `Chatterbox TTS streaming latency benchmark TTFB`
5. `flow matching vocoder fp16 bf16 int8 quantization latency TTS`
6. `Chatterbox TTS torch.compile fp16 bf16 latency github`
7. `F5-TTS hang loading startup issue github`
8. `real-time streaming TTS time-to-first-byte vLLM 2026`
9. `CosyVoice2 vllm-omni RFC performance acceleration paged KV cache`
10. `Chatterbox resemble-ai sentence chunking streaming API conditioning cache`
11. `speaker similarity degradation quantization voice cloning TTS int8`
12. `TensorRT-LLM 0.5B parameter model inference speedup H100 benchmark`
13. `CosyVoice2 github issue streaming latency voice quality not good enough`
14. `reference audio speaker embedding caching zero-shot TTS voice conditioning latency`
15. `"Towards Lightweight Voice Cloning" quantization LLaMA-based TTS Transformers authors arxiv`
16. `Llasa quantization INT8 TTS transformers IEEE 2026 paper author name`
17. `VocalNet-MDM Accelerating Streaming Speech LLM Self-Distilled Masked Diffusion authors`
18. `"Ultra-Low Latency" "End-to-End Streaming Speech Synthesis" "Block-Wise Generation" "Depth-Wise Codec Decoding" authors arxiv 2604.12438`
19. `Stream2LLM "Overlap Context Streaming and Prefill" Reduced TTFT authors arxiv 2604.16395`
20. `Qwen3-TTS Technical Report authors arxiv 2601.15621`
21. `dots.tts Technical Report authors arxiv 2606.07080`

**Date range**: no restriction; results skew 2024-2026. **Inclusion**: measured
latency/throughput/precision numbers, CosyVoice2/Chatterbox-specific engineering reports, or
transferable serving-stack techniques. **Exclusion**: pages with no quantitative content and
paywalled pages returning no extractable text (the IEEE abstract page returned HTTP 418 to automated
fetch). **Iterations**: queries 15-21 followed names/titles surfaced in queries 1, 5, 8, 9
(`swulling/CosyVoice2-0.5B-vllm`, the IEEE paper title, adjacent arXiv TTS-latency papers).

## Key Findings

### A Community RFC Supplies a Per-Stage Breakdown, With Caveats

`vllm-project/vllm-omni` issue #6870 (open RFC, non-peer-reviewed, GPU unstated) reports median TTFA
of **357 ms** at concurrency 1: prefill **40.5 ms (11.3%)**, AR decode over 24 steps **142.3 ms
(39.9%)**, first flow chunk **129.0 ms (36.1%)** [vllm-omni-6870]. Per-token costs: **6.589
ms/token** LM steady-state (3.140 ms execute + 3.449 ms gap), **≈2.65 ms/token** amortized flow.
**13.2 blocking device-to-host syncs per decode step (~1.53 ms/step, 44% of the gap)** come from the
repetition-avoidance sampler, with the LM-stage GPU idle **75%** of decode [vllm-omni-6870]. This
decomposes a different serving stack (vLLM-Omni) than the stock CosyVoice2 repo, not H100-confirmed,
but it is the closest available analogue to the missing breakdown and identifies a structural fact
new to this project: the flow-matching DiT stage asserts `token.shape[0] == 1` and cannot batch
across requests, so **4x concurrency buys only 2.2x throughput** [vllm-omni-6870].

### Chatterbox Has Concrete Community Engineering Data, No Paper

`davidbrowne17/chatterbox-streaming` reports, on RTX 4090, **472 ms** latency to first chunk and
**RTF 0.499** with token-level chunking (default 50 speech tokens) [chatterbox-streaming-GH] —
faster than t0018's whole-utterance baseline (1.34-1.6 s p50) but different hardware and code path,
so an existence proof of headroom, not a transferable number. `devnen/Chatterbox-TTS-Server` adds a
`TTS_BF16` flag ("roughly 40% throughput" on bf16-capable GPUs including H100, "numerically slightly
different... typically inaudible") and a voice-conditioning cache keyed by
`(path, mtime, exaggeration)` [Chatterbox-TTS-Server-GH]. Critically for Key Question 3, the same
docs clarify Chatterbox streaming is **chunk-level, not token-level** — a single short filler
utterance gets **no TTFB benefit** from `stream=True` alone unless pre-split into multiple chunks
[Chatterbox-TTS-Server-GH].

### The Field Is Actively Attacking the Same `d_lm`/`d_fm` Bottleneck

Several 2025-2026 preprints not yet in this project's corpus target the same terms the additive
latency model attributes TTFB to. FlashTTS cuts flow-matching to **2 function evaluations** via
multi-token prediction and mean-flow distillation, reporting **325 ms** first-packet latency
[Xie2026]. VocalNet-MDM fine-tunes a streaming speech LLM into a self-distilled masked-diffusion
decoder, structurally similar to the already-reviewed Chatterbox-Flash approach applied to a
different backbone [Cheng2026]. A block-wise, depth-wise-codec architecture avoids autoregressive LM
decoding entirely via a modified FastSpeech2 backbone on Mimi codec tokens [Su2026]. Orthogonal to
TTS, Stream2LLM (MLSys 2026) shows up to **11x TTFT improvement** for general LLM serving by
overlapping context streaming with prefill [Bachkaniwala2026] — not yet applied to speech-token LMs
anywhere found. Qwen3-TTS reports **RTF 0.34 / TTFP 131 ms** served via vLLM-Omni [Hu2026], and
MOSS-TTS-Realtime (1.7B) reports **TTFB ~180 ms** on a single A10G with a tuned `codec_chunk_frames`
[MOSS-TTS-Realtime-vLLM]. None are directly comparable to CosyVoice2/Chatterbox (different
architectures, hardware, scoring), but together they show sub-350 ms TTFP is reached within the same
architecture family t0021 studies, reinforcing the prior stage's conclusion that engineering levers
alone can likely close much, though not necessarily all, of the gap to 300 ms without fine-tuning.

## Methodology Insights

* **Profile the flow stage expecting a batch-1 ceiling**: [vllm-omni-6870] shows it cannot batch
  across requests, so `d_fm` should be stable across repeated single-request measurements even
  though this is an architectural fact, not one t0021's concurrency-1 protocol needs to work around.
* **Pre-split short filler text into 2+ chunks before invoking Chatterbox streaming** — a
  single-chunk utterance gets zero TTFB benefit otherwise [Chatterbox-TTS-Server-GH], matching
  `task_description.md`'s sentence-level chunking lever.
* **Best practice, cross-validated**: cache reference-voice conditioning per voice, keyed by content
  and encoding parameters (e.g. `(path, mtime, exaggeration)`), extending `research_papers.md`'s
  caching recommendation with a concrete key design.
* **bf16 on the LM/decoder stage is a low-risk lever** (~40% throughput, H100-listed
  [Chatterbox-TTS-Server-GH]) — treat bf16-everywhere as the safe default and int8/int4 as LM-only.
  Note: an earlier draft of this section also cited [Lian2026] (dots.tts) as evidence for a
  bf16-acoustic/quantized-LLM-trunk split; full-text review found no such claim in that paper (its
  benchmark tables run float32 with `torch.compile` disabled, no precision ablation present), so
  that citation has been removed — the bf16-decoder recommendation now rests on
  [Chatterbox-TTS-Server-GH] alone.
* **Hypothesis**: since [vllm-omni-6870] attributes 44% of the LM stage's per-token gap to blocking
  device-to-host syncs in the repetition-avoidance sampler, disabling or relaxing an equivalent knob
  (if exposed) may cut `d_lm` with zero architecture change — worth a time-boxed probe before
  TensorRT export.
* **Hypothesis**: a follow-up task (not t0021, which forbids fine-tuning) could test whether
  [Xie2026]'s 2-NFE distillation or [Cheng2026]'s masked-diffusion decoder reaches 300 ms without
  CosyVoice2/Chatterbox's own architecture change; route to `results/suggestions.json`.

## Discovered Papers

### FlashTTS: Fast Streaming TTS with MTP Acceleration and X-pred Mean Flow Distillation

* **Authors**: Xie, H., Ren, X., Guo, D., et al. (13 authors) | **Year**: 2026
* **DOI**: `10.48550/arXiv.2606.09141` | **URL**: https://arxiv.org/abs/2606.09141
* **Suggested categories**: none registered; topically `tts-latency`, `flow-matching`
* **Why download**: Attacks CosyVoice2's `d_fm` term via 2-NFE flow distillation, 325 ms FPL.

### DiFlow-TTS: Compact and Low-Latency Zero-Shot TTS with Discrete Flow Matching

* **Authors**: Nguyen, N.-S., Tran, T. V. T., Huynh-Nguyen, H.-N., Hy, T.-S., Nguyen, V.
* **Year**: 2025 | **DOI**: `10.48550/arXiv.2509.09631` | **URL**: https://arxiv.org/abs/2509.09631
* **Suggested categories**: none registered; topically `zero-shot-tts`, `flow-matching`
* **Why download**: Alternative low-latency architecture comparable to the already-reviewed F5-TTS.

### VocalNet-MDM: Accelerating Streaming Speech LLM via Self-Distilled Masked Diffusion Modeling

* **Authors**: Cheng, Z., et al. | **Year**: 2026
* **DOI**: `10.48550/arXiv.2602.08607` | **URL**: https://arxiv.org/abs/2602.08607
* **Suggested categories**: none registered; topically `streaming-tts`, `speech-llm`
* **Why download**: Second data point (after Chatterbox-Flash) on decoding-fine-tune fixes for AR.

### An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture via Block-Wise Generation and Depth-Wise Codec Decoding

* **Authors**: Su, T., Tan, T.-P., Mdhaffar, S., Estève, Y., Sini, A. | **Year**: 2026
* **DOI**: `10.48550/arXiv.2604.12438` | **URL**: https://arxiv.org/abs/2604.12438
* **Suggested categories**: none registered; topically `streaming-tts`, `low-latency`
* **Why download**: Third architecture family (non-AR, non-diffusion) for Key Question 5.

### Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token (TTFT)

* **Authors**: Bachkaniwala, R., Luo, C., So, R., Mahajan, D., Rong, K.
* **Year**: 2026 (MLSys) | **DOI**: `10.48550/arXiv.2604.16395`
* **URL**: https://arxiv.org/abs/2604.16395
* **Suggested categories**: none registered; topically `llm-serving`, `ttft`
* **Why download**: General 11x TTFT technique not yet applied to speech-token LMs; transferable.

### Qwen3-TTS Technical Report

* **Authors**: Hu, H., Zhu, X., He, T., et al. (16 authors) | **Year**: 2026
* **DOI**: `10.48550/arXiv.2601.15621` | **URL**: https://arxiv.org/abs/2601.15621
* **Suggested categories**: none registered; topically `streaming-tts`, `vllm-serving`
* **Why download**: RTF 0.34 / TTFP 131 ms on vLLM-Omni; paired data point alongside the
  already-reviewed Chatterbox-Flash.

### dots.tts Technical Report

* **Authors**: Lian, S., Li, C., Li, B., Wang, H., Zheng, D., Tian, J., Ma, Y., Zhang, C., Yu, K.
* **Year**: 2026 | **DOI**: `10.48550/arXiv.2606.07080` | **URL**: https://arxiv.org/abs/2606.07080
* **Suggested categories**: none registered; topically `tts-precision`, `flow-matching`
* **Why download**: Initially flagged for an apparent bf16-for-acoustic/quantize-LLM-trunk precision
  design; full-text review after download found no such claim (see Gap 3 correction) — a streaming
  zero-shot architecture comparable to CosyVoice2/Chatterbox is the paper's actual value for this
  task.

### Towards Lightweight Voice Cloning: Quantization of LLaMA-based TTS Transformers

* **Authors**: not resolved (IEEE Xplore inaccessible to automated fetch, HTTP 418)
* **Year**: 2026 | **DOI**: not resolved; IEEE document `11539760`
* **URL**: https://ieeexplore.ieee.org/abstract/document/11539760/
* **Suggested categories**: none registered; topically `tts-quantization`, `voice-cloning`
* **Why download**: Closest published INT8 ablation for an LLM-based voice-cloning transformer
  (Llasa family, 1B/3B/8B); requires authenticated IEEE access for full text.

## Recommendations for This Task

1. **Expect the flow-matching stage to be architecturally batch-1**, per [vllm-omni-6870] —
   single-request `d_fm` measurements should be stable, refining `research_papers.md`'s use of the
   additive latency model.
2. **Pre-split short filler text into 2+ chunks for Chatterbox streaming** — single-chunk text gets
   zero TTFB benefit otherwise [Chatterbox-TTS-Server-GH].
3. **Test bf16 on the LM/decoder stage before int8, for both systems** — the ~40% gain is
   H100-confirmed [Chatterbox-TTS-Server-GH]. Treat int8-on-the-LM-trunk-only as a weaker,
   unverified lead: [LlasaQuant2026]'s full text could not be retrieved (IEEE access blocked), so
   its numbers are unconfirmed, and [Lian2026] (dots.tts) does not in fact contain a precision
   ablation (see Gap 3 correction above) — do not cite it as int8 evidence.
4. **If F5-TTS hangs again during S-0018-01's retry, check `jieba` dictionary init and HF cache
   state** before falling back to a null result [F5TTS-GH-Issues].
5. **Route [Xie2026], [Cheng2026], [Su2026], [Bachkaniwala2026] into `results/suggestions.json`** as
   fine-tuning-or-new-stack follow-ups, per `research_papers.md`'s existing recommendation 6.
6. **Treat Gap 6 (speaker-sim under acceleration) as still fully open** in the final answer — no
   source measures this for any zero-shot TTS system, so t0021's own paired measurement is novel.

## Source Index

### [vllm-omni-6870]

* **Type**: forum
* **Title**: RFC: CosyVoice 2/3 Performance Acceleration (Issue #6870)
* **Author/Org**: vllm-project/vllm-omni (GitHub)
* **Date**: 2026 (open)
* **URL**: https://github.com/vllm-project/vllm-omni/issues/6870
* **Peer-reviewed**: no
* **Relevance**: Only source with a measured per-stage TTFA breakdown and the flow-batch-1 finding.

### [chatterbox-streaming-GH]

* **Type**: repository
* **Title**: chatterbox-streaming
* **Author/Org**: davidbrowne17 (GitHub)
* **Date**: 2026
* **URL**: https://github.com/davidbrowne17/chatterbox-streaming
* **Last updated**: 2026
* **Peer-reviewed**: no
* **Relevance**: 472 ms first-chunk / RTF 0.499 on RTX 4090; only Chatterbox TTFB number outside
  t0018.

### [Chatterbox-TTS-Server-GH]

* **Type**: repository
* **Title**: Chatterbox-TTS-Server
* **Author/Org**: devnen (GitHub)
* **Date**: 2026
* **URL**: https://github.com/devnen/Chatterbox-TTS-Server/blob/main/README.md
* **Last updated**: 2026
* **Peer-reviewed**: no
* **Relevance**: bf16 flag, conditioning cache, chunk-level-not-token-level streaming clarification.

### [CosyVoice2-vllm-HF]

* **Type**: documentation
* **Title**: swulling/CosyVoice2-0.5B-vllm (model card)
* **Author/Org**: swulling (Hugging Face)
* **Date**: 2026
* **URL**: https://huggingface.co/swulling/CosyVoice2-0.5B-vllm
* **Peer-reviewed**: no
* **Relevance**: Working community vLLM integration for CosyVoice2's LM backbone, unbenchmarked.

### [MOSS-TTS-Realtime-vLLM]

* **Type**: documentation
* **Title**: OpenMOSS-Team/MOSS-TTS-Realtime (vLLM recipe)
* **Author/Org**: vLLM project (recipes.vllm.ai)
* **Date**: 2026
* **URL**: https://recipes.vllm.ai/OpenMOSS-Team/MOSS-TTS-Realtime
* **Peer-reviewed**: no
* **Relevance**: TTFB ~180 ms for a 1.7B streaming model on vLLM-Omni; serving-config data point.

### [F5TTS-GH-Issues]

* **Type**: forum
* **Title**: SWivid/F5-TTS issue tracker (issues #689, #728, #747)
* **Author/Org**: SWivid/F5-TTS (GitHub)
* **Date**: 2024-2026
* **URL**: https://github.com/SWivid/F5-TTS/issues
* **Peer-reviewed**: no
* **Relevance**: Startup hangs tied to `jieba` init and HF cache state; candidate root cause.

### [Audio8-HackerNoon]

* **Type**: blog
* **Title**: How Audio8 TTS 0.1B Brings Voice Cloning to Smaller GPUs
* **Author/Org**: HackerNoon
* **Date**: 2026
* **URL**: https://hackernoon.com/how-audio8-tts-01b-brings-voice-cloning-to-smaller-gpus
* **Peer-reviewed**: no
* **Relevance**: Only source touching Gap 6: INT8 can change sampled tokens, re-validate per voice.

### [Xie2026]

* **Type**: paper
* **Title**: FlashTTS: Fast Streaming TTS with MTP Acceleration and X-pred Mean Flow Distillation
* **Authors**: Xie, H., Ren, X., Guo, D., et al. (13 authors)
* **Year**: 2026
* **DOI**: `10.48550/arXiv.2606.09141`
* **URL**: https://arxiv.org/abs/2606.09141
* **Peer-reviewed**: no (arXiv preprint)
* **Relevance**: 2-NFE flow-matching distillation reaching 325 ms first-packet latency.

### [Nguyen2025]

* **Type**: paper
* **Title**: DiFlow-TTS: Compact and Low-Latency Zero-Shot TTS with Discrete Flow Matching
* **Authors**: Nguyen, N.-S., Tran, T. V. T., Huynh-Nguyen, H.-N., Hy, T.-S., Nguyen, V.
* **Year**: 2025
* **DOI**: `10.48550/arXiv.2509.09631`
* **URL**: https://arxiv.org/abs/2509.09631
* **Peer-reviewed**: no (arXiv preprint)
* **Relevance**: Compact low-latency zero-shot TTS via discrete flow matching; comparison point.

### [Cheng2026]

* **Type**: paper
* **Title**: VocalNet-MDM: Accelerating Streaming Speech LLM via Self-Distilled Masked Diffusion
  Modeling
* **Authors**: Cheng, Z., et al.
* **Year**: 2026
* **DOI**: `10.48550/arXiv.2602.08607`
* **URL**: https://arxiv.org/abs/2602.08607
* **Peer-reviewed**: no (arXiv preprint)
* **Relevance**: Masked-diffusion decoding fine-tune, structurally parallel to Chatterbox-Flash.

### [Su2026]

* **Type**: paper
* **Title**: An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture via Block-Wise
  Generation and Depth-Wise Codec Decoding
* **Authors**: Su, T., Tan, T.-P., Mdhaffar, S., Estève, Y., Sini, A.
* **Year**: 2026
* **DOI**: `10.48550/arXiv.2604.12438`
* **URL**: https://arxiv.org/abs/2604.12438
* **Peer-reviewed**: no (arXiv preprint)
* **Relevance**: Non-autoregressive block-wise architecture avoiding LM decoding entirely.

### [Bachkaniwala2026]

* **Type**: paper
* **Title**: Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token
  (TTFT)
* **Authors**: Bachkaniwala, R., Luo, C., So, R., Mahajan, D., Rong, K.
* **Year**: 2026 (MLSys)
* **DOI**: `10.48550/arXiv.2604.16395`
* **URL**: https://arxiv.org/abs/2604.16395
* **Peer-reviewed**: yes (MLSys 2026)
* **Relevance**: General LLM-serving TTFT technique (up to 11x), not yet applied to speech-token
  LMs.

### [Hu2026]

* **Type**: paper
* **Title**: Qwen3-TTS Technical Report
* **Authors**: Hu, H., Zhu, X., He, T., et al. (16 authors)
* **Year**: 2026
* **DOI**: `10.48550/arXiv.2601.15621`
* **URL**: https://arxiv.org/abs/2601.15621
* **Peer-reviewed**: no (arXiv preprint)
* **Relevance**: RTF 0.34 / TTFP 131 ms for a production vLLM-Omni-served streaming TTS system.

### [Lian2026]

* **Type**: paper
* **Title**: dots.tts Technical Report
* **Authors**: Lian, S., Li, C., Li, B., Wang, H., Zheng, D., Tian, J., Ma, Y., Zhang, C., Yu, K.
* **Year**: 2026
* **DOI**: `10.48550/arXiv.2606.07080`
* **URL**: https://arxiv.org/abs/2606.07080
* **Peer-reviewed**: no (arXiv preprint)
* **Relevance**: Corrected relevance after full-text review — this paper does NOT contain a
  bf16-acoustic-stages precision design (its benchmark tables use float32 with `torch.compile`
  disabled, and no precision ablation appears in the text). Its actual relevance to this task is
  architectural: a streaming zero-shot TTS system comparable to CosyVoice2/Chatterbox, useful for
  Key Question 5's cross-architecture comparison, not for Gap 3.

### [LlasaQuant2026]

* **Type**: paper
* **Title**: Towards Lightweight Voice Cloning: Quantization of LLaMA-based TTS Transformers
* **Authors**: not resolved (page inaccessible)
* **Year**: 2026
* **DOI**: not resolved (IEEE Xplore document `11539760`)
* **URL**: https://ieeexplore.ieee.org/abstract/document/11539760/
* **Peer-reviewed**: yes (IEEE conference, per listing; full text unverified)
* **Relevance**: Closest published INT8 ablation for an LLM-based voice-cloning TTS transformer.
