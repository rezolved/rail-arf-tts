---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
research_stage: "papers"
papers_reviewed: 11
papers_cited: 10
categories_consulted: []
date_completed: "2026-09-18"
status: "complete"
---
## Task Objective

t0021 profiles where CosyVoice2 and Chatterbox spend their 1.3-2.9 s p50 time-to-first-byte (TTFB)
in Rezolve's voice-commerce filler synthesis pipeline, and tests streaming, chunking, vLLM/TensorRT
backends and precision changes to find the lowest reachable TTFB on the project's H100 hardware
without losing speaker similarity (GE2E cosine vs. the ElevenLabs David reference set). Per
`task_description.md`, both systems already run faster than real time on H100 (RTF 0.40-0.54, per
t0018), so the delay is start-up and first-chunk work rather than raw throughput — this review
targets literature on zero-shot voice-cloning architectures, streaming/chunked autoregressive
decoding, flow-matching vocoder latency, and LLM-serving acceleration as it applies to speech-token
language models, to ground the task's per-stage latency-breakdown methodology and its choice of
acceleration levers (caching, chunk size, precision, serving-stack changes).

## Category Selection Rationale

`meta/categories/` contains no registered category folders (only a `.gitkeep` placeholder), and
`uv run python -u -m arf.scripts.aggregators.aggregate_categories --format json` returns
`{"categories": []}` for this project. Both the category-filtered and unfiltered invocations of
`aggregate_papers` therefore return the identical 11-paper corpus, confirming category-based
filtering is not available in this project and every paper had to be triaged manually by title,
venue, and topic against t0021's five stated research areas (zero-shot cloning architectures,
streaming/chunked decoding, flow-matching vocoder latency, LLM-serving acceleration, and per-stage
latency-breakdown methodology). No categories were excluded because none exist; this section
documents the absence rather than a selection among existing categories, as instructed when the
aggregator returns an empty list.

## Key Findings

### An Additive Latency Model Is the Right Frame for Per-Stage Profiling

[Du2024] (CosyVoice 2) is the only reviewed paper that explicitly formalizes TTS first-package
(TTFB-equivalent) latency as a sum of per-stage compute times: `L_TTS = M·d_lm + M·d_fm + M·d_voc`,
where `d_lm`, `d_fm`, and `d_voc` are the per-token/per-chunk compute times of the language model,
the flow-matching model, and the vocoder respectively, and for a full voice-chat pipeline
`L_Chat <= N·d_llm + L_TTS`. This decomposition matches t0021's Key Question 1 almost exactly (it
omits only the reference-audio encoding and text-frontend stages, which the paper treats as
amortized/negligible because CosyVoice 2 drops the utterance-level speaker embedding from the LM
entirely, moving all speaker conditioning into the flow-matching stage). No other reviewed paper
proposes a comparably explicit additive latency model for zero-shot TTS; [Chen2024] (F5-TTS) and
[Seo2026] (Chatterbox-Flash) report only aggregate real-time-factor (RTF) and time-to-first-packet
(TTFP) numbers, not stage decompositions.

### Autoregressive Decoding Sets an Architectural Latency Floor

CosyVoice2's text-speech LM and Chatterbox's T3 decoder [Seo2026] are both autoregressive
Llama-style causal transformers that must emit enough discrete speech tokens to fill a chunk before
any audio can be produced — this is the architectural constraint t0021's Key Question 5 asks about.
By contrast, [Chen2024] shows a fully non-autoregressive (NAR) alternative is viable: F5-TTS pads
text with filler tokens to the target mel length and denoises the whole sequence with a Diffusion
Transformer (DiT) and Optimal-Transport conditional flow matching, reaching **RTF 0.15** at 16
function evaluations (NFE) and **2.53% WER** on LibriSpeech-PC test-clean (32-NFE variant: RTF
**0.31**, WER **2.42%**, SIM-o **0.66**) [Chen2024]. However, F5-TTS's NAR design generates the
entire utterance's mel spectrogram in one denoising pass over the whole padded sequence — it is not
naturally chunked, so a fast RTF does not automatically imply a low TTFB for early output (see "RTF
Is Not TTFB" below). [Li2023] (StyleTTS 2), the architecture underlying this project's own Kokoro
fine-tune, is a third design point: text encoder, duration predictor, and style-diffusion sampler
run in parallel (non-autoregressive) with only 5 diffusion steps at inference [Li2023, p. 7], which
is architecturally why a StyleTTS2-family model can meet the project's 300 ms TTFB target
(`project/description.md`) while LLM-token-based systems like CosyVoice2/Chatterbox structurally
cannot without changing the decoding scheme.

[Seo2026] provides the most direct evidence on whether the autoregressive bottleneck is strictly
architectural or partly engineering: Chatterbox-Flash fine-tunes the existing Chatterbox T3 decoder
(no architecture change, only a different training objective) into a block-diffusion decoder that
denoises multiple positions per block in parallel while remaining block-causal for streaming. In its
production streaming configuration (`D=16, alpha=0.5`) it reaches **TTFP 118 ms** and **RTF 0.107**
(~9x real time) on H100 at concurrency 1; a more aggressive setting (`D=32, alpha=0.75`) reaches
**TTFP 103 ms** and **RTF 0.076** (~13x real time) [Seo2026]. This is below the project's 300 ms
target on comparable H100 hardware, but it required fine-tuning a new decoding objective into the
model — squarely outside t0021's "no fine-tuning" constraint — so it demonstrates the gap is not
*purely* architectural (a fine-tuned decoding change can close it) but is not closeable by
engineering-only levers (caching, chunk size, precision, serving stack) applied to a stock
autoregressive checkpoint alone, at least based on what this literature shows.

### Chunk-Aware Flow Matching Cuts the Quality Cost of Streaming, Not the Latency

[Du2024]'s chunk-aware causal flow-matching model is trained with four randomly sampled attention
masks (non-causal, full-causal, chunk-M, chunk-2M) so a single checkpoint spans the full
offline-to-fully-causal latency/quality spectrum. Its own streaming ablation (chunk size 15) shows
that on typical text, fully streaming LM+flow-matching (configuration M4) reaches **1.45% CER /
0.812 SS** on SEED test-zh versus **1.45% CER / 0.806 SS** fully offline (M1) — streaming costs
essentially nothing in quality on ordinary prompts, but the gap widens sharply on adversarial "hard"
text: **6.83%** CER offline versus **8.08%** CER streaming [Du2024, Table 8]. This is consistent
with, and gives a likely mechanism for, t0018's own finding (`task_description.md`) that
`inference_zero_shot(stream=True)` alone still measured **2.86 s p50 TTFB** on val96: chunk-aware
streaming avoids a *quality* penalty but does not by itself reduce the *absolute* wall-clock cost of
producing the first chunk, because the LM must still autoregressively generate enough tokens (at the
paper's streaming ratio of **N:M = 5:15** text:speech tokens, 25 Hz token rate [Du2024]) before the
flow-matching stage can run on them. [Seo2026] corroborates this from the Chatterbox side: reducing
the number of denoising steps per block from 8 to **~6.3** via early-decoding (PMI+ED, a **~20%**
reduction) was needed on top of block-level streaming to materially cut latency — chunking/streaming
architecture alone was not sufficient; a decoding-schedule optimization was also required.

### RTF Is Not TTFB

[Chen2024] reports F5-TTS's RTF (**0.15** at 16 NFE, measured on an RTX 3090 for 10 s of generated
speech) as its headline efficiency number, and shows it beats CosyVoice v1's reported RTF of
**0.92** on the same LibriSpeech-PC benchmark [Chen2024]. But RTF is a whole-utterance throughput
metric (total generation wall-clock time divided by output audio duration), computed after the
entire clip is produced; it says nothing about how long a caller waits before the first audio byte,
especially for a NAR model that denoises the full padded sequence in one pass. This is the same
distinction t0018 already established empirically for the two systems t0021 studies: CosyVoice2 and
Chatterbox were already RTF 0.40-0.54 (faster than real time) while measuring 1.3-2.9 s p50 TTFB
(`task_description.md`) — no paper in the reviewed corpus reports TTFB/TTFP and RTF together for the
same configuration except [Seo2026], which is the only source that reports both **TTFP (103-118
ms)** and **RTF (0.076-0.107)** side by side for a streaming zero-shot system, making it the closest
published methodological precedent for the paired TTFB/RTF variant reporting t0021's protocol
requires.

### LM-Serving Acceleration Targets the Term Most Likely to Dominate TTFB

Published vocoder inference speeds are already two-to-three orders of magnitude faster than real
time: HiFi-GAN V1 (13.92M params) runs at **3,701x real-time** on a V100 GPU with **MOS 4.36** (vs.
ground truth **4.45**), and the smallest configuration V3 (0.92M params, MOS **4.05**) still reaches
**1,186.80x real-time on GPU** and **13.44x real-time on CPU** [Kong2020, Table 1, p. 6]. iSTFTNet
pushes this further by replacing HiFi-GAN's final output-side convolutional layers with a
closed-form inverse short-time Fourier transform: the `V1-C8C8I` variant reaches **245.68x real-time
on GPU** (versus the unmodified V1's **143.59x** in the same paper's re-measurement) and **MOS 4.26
± 0.17**, matching or slightly beating the unmodified V1's **4.22 ± 0.17**, at **13.26M** versus
**13.94M** parameters [Kaneko2022, Table 1, p. 3]. Because a single output chunk's vocoder work is
therefore likely sub-millisecond to low-millisecond even unoptimized, while an autoregressive LM
must run one or more sequential forward passes per chunk, the `d_lm` term in [Du2024]'s latency
model is the more probable dominant TTFB contributor — consistent with [Seo2026]'s finding that a
serving-stack optimization applied to the LM/decoder stage specifically (FlashInfer attention
kernels, paged KV-cache management, `bfloat16` precision, on a similarly-sized ~0.5B-parameter
causal decoder) was what delivered its sub-120 ms TTFP, not vocoder changes (its Stage-2
flow-matching vocoder was left unmodified from a CosyVoice2-style MeanFlow few-step sampler
[Seo2026]).

### Reference-Audio Encoding Is Cacheable, But the Encoder Is Not a Free Swap

[Wan2018] defines the GE2E speaker-embedding pipeline this project's `speaker_sim` metric is named
after: 25 ms frames / 10 ms step log-mel-filterbank energies into an LSTM d-vector encoder,
L2-normalized, with speaker similarity scored as cosine similarity to a centroid computed from
enrollment utterances (with the query utterance excluded from its own centroid to avoid a trivial
solution) [Wan2018]. Because this encoding is a deterministic function of a fixed reference clip, it
is safe to cache per voice with zero measurable per-request cost — directly answering t0021's Key
Question 2 for the reference-encoding stage specifically. However, [Kunesova2025] shows encoder
*choice* is not latency-neutral in effect on quality: holding a YourTTS-based TTS backbone fixed and
swapping only the speaker encoder, the original H/ASP encoder scored **47.31 ± 20.50** on a
normalized 0-100 MUSHRA-style subjective similarity scale, versus **42.61 ± 20.38** for ECAPA-TDNN
and **40.96 ± 20.85** for x-vector, with H/ASP significantly better than both alternatives (**p <
0.001**) and ECAPA-TDNN significantly better than x-vector (**p = 0.02-0.03** across the paper's
subjective and objective tests) [Kunesova2025]. Objective cosine-distance scoring under the
ECAPA-TDNN extractor showed the same ranking (H/ASP **0.524 ± 0.052**, ECAPA-TDNN-conditioned TTS
**0.533 ± 0.054**, x-vector-conditioned TTS **0.696 ± 0.090**; lower = more similar) [Kunesova2025].
The practical implication for t0021 is narrow but important: caching the *existing* reference
embedding for each acceleration variant is free and safe, but substituting a different or
lower-precision speaker encoder to save time is a quality-relevant change that must be validated
against `speaker_sim`, not assumed neutral.

### Cross-System and Cross-Encoder Similarity Scores Are Not Directly Comparable

[Du2024] itself flags that WavLM-based and ERes2Net-based speaker-similarity (SS) scores for
CosyVoice2 are not consistent with each other (Section 4.2), and reports its own SEED test-zh SS as
**0.748** under a WavLM-based scorer versus **0.806** under ERes2Net for the identical outputs
[Du2024]. [Chen2024] reports F5-TTS similarity as **SIM-o** using WavLM-large embeddings (e.g.,
**0.66** on LibriSpeech-PC at 16 NFE), and [Du2024a] (CosyVoice v1) reports raw ERes2Net cosine
similarity on a roughly 0-100 scale (**74.30 ± 0.15** on LibriTTS test-clean, **81.58 ± 0.16** on
AISHELL-3) — neither is on the same scale or encoder as this project's GE2E-cosine `speaker_sim`.
[Casanova2022] (YourTTS) is the one reviewed paper whose similarity metric (SECS) is defined
identically to a GE2E-style cosine similarity, reporting VCTK zero-shot SECS of **0.864** versus a
ground-truth SECS of **0.824** [Casanova2022] — making it the most directly comparable calibration
point in the corpus for interpreting this project's own ≥0.85 GE2E-cosine threshold, though it is
from a different, older (VITS-based) architecture. [Kunesova2025]'s four-extractor study
(ECAPA-TDNN, x-vector, TitaNet-large, Resemblyzer/GE2E) independently confirms these rankings are
not always consistent across extractors. Taken together, these findings mean that any published
CosyVoice2/F5-TTS/Chatterbox SS or SIM-o number cannot be used as a drop-in comparison point for
t0021's `speaker_sim` results; every acceleration variant must be re-scored with the same GE2E
scorer used for the t0018 baseline.

## Methodology Insights

* **Use the CosyVoice2 additive latency model as the profiling schema.** Structure
  `results/latency_breakdown.json` around `L_TTS = M·d_lm + M·d_fm + M·d_voc` [Du2024], adding a
  reference-encoding term and a text-frontend term (both implicit/near-zero in CosyVoice2's own
  formulation but explicitly required by t0021's Key Question 1) as two additional additive stages
  measured with the same instrumented-timestamp methodology.
* **Cache the reference-audio (speaker) embedding before touching anything else.** Per [Wan2018],
  the embedding is a pure function of the fixed David reference clip; per t0021's Key Question 2
  this is a zero-cost latency win that should be validated first and re-used as a fixed baseline
  input across every subsequent acceleration variant, not re-measured per variant.
* **Prioritize LM/decoder-stage acceleration (vLLM, TensorRT-LLM, fp16/bf16, torch.compile,
  FlashInfer-style kernels, paged KV-cache) over vocoder-stage acceleration.** Published vocoder
  RTFs are already 150x-3,700x real time even unoptimized [Kong2020, Kaneko2022], while [Seo2026]'s
  sub-120 ms TTFP on a similarly-sized (~0.5B-parameter) decoder was achieved via exactly this class
  of serving-stack optimization (FlashInfer attention, paged KV-cache, bfloat16) applied to the
  decoder stage, with the flow-matching vocoder left unmodified — the evidence points to `d_lm` as
  the highest-leverage term in the latency model.
* **Evaluate CosyVoice2 chunk-size reduction on typical fillers and hard/adversarial prompts
  separately**, because [Du2024]'s own streaming ablation (Table 8) shows near-zero quality cost on
  typical text (1.45% CER offline vs. 1.45% streaming) but a real cost on hard text (6.83% vs. 8.08%
  CER) — a single pooled `successful_prompts / total_prompts` pass/fail check could mask a
  hard-prompt-specific regression.
* **Do not compare `speaker_sim` numbers against any paper's published SS/SIM-o figures directly.**
  [Du2024] itself documents WavLM-vs-ERes2Net disagreement on its own outputs, and [Chen2024]/
  [Du2024a] use yet other encoders; only [Casanova2022]'s GE2E-equivalent SECS is on a comparable
  scale, and even that is from a different architecture. Every t0021 variant must be scored with the
  same GE2E scorer as the t0018 baseline, in the same session, as the task protocol already
  requires.
* **Treat vocoder-head replacement (iSTFTNet-style, [Kaneko2022]) and decoding-scheme fine-tuning
  (block diffusion, [Seo2026]) as out-of-scope levers to flag in `results/suggestions.json`, not
  actions to attempt in this task** — both require model retraining/fine-tuning, which
  `task_description.md`'s Forbidden section rules out for t0021.
* **Report RTF and TTFB in the same table but never treat one as a proxy for the other.**
  [Chen2024]'s F5-TTS achieves an excellent RTF (0.15) while being a non-streaming, whole-utterance
  NAR model; [Seo2026] is the only reviewed paper that reports TTFP and RTF together for a streaming
  system, and its two numbers move independently across configurations (`D=16,alpha=0.5`: TTFP 118
  ms / RTF 0.107; `D=32,alpha=0.75`: TTFP 103 ms / RTF 0.076), confirming they are separate axes.

## Gaps and Limitations

* **No paper in this corpus benchmarks vLLM or TensorRT-LLM specifically for CosyVoice2's
  Qwen2.5-0.5B backbone**, nor the flow-matching TensorRT export the CosyVoice2 repository ships
  (referenced in `task_description.md`). [Du2024] mentions the Qwen2.5-0.5B backbone choice but
  reports no vLLM/TensorRT-LLM serving benchmarks; this is a genuine literature gap t0021 must fill
  empirically.
* **No paper reports a full per-stage wall-clock latency breakdown (reference encoding, text
  frontend, LM prefill, LM decode, flow matching, vocoder) in absolute milliseconds on H100-class
  hardware** for either CosyVoice2 or Chatterbox. [Du2024]'s latency model is analytical/symbolic
  (`d_lm`, `d_fm`, `d_voc`), not populated with measured numbers; [Seo2026] reports only the
  aggregate TTFP, not its internal decomposition.
* **No paper ablates inference precision (fp16/bf16/int8) specifically for the flow-matching or
  vocoder stages of a CosyVoice2-family pipeline.** [Seo2026] uses `bfloat16` for the LM/decoder
  stage during training and production serving, but does not ablate precision's effect on latency or
  quality in isolation, and does not address the vocoder stage's precision at all.
* **No reviewed paper addresses Chatterbox (the plain autoregressive model t0021 must also
  accelerate) directly** — Chatterbox has no dedicated research paper in this corpus or, per
  [Seo2026]'s own framing, in the literature generally; [Seo2026] is a downstream derivative that
  fine-tunes the same T3 backbone into a different decoding scheme, so it is informative about the
  backbone's potential but not a direct source of plain-Chatterbox latency numbers.
* **No paper discusses F5-TTS's loading/startup robustness** (t0018 reported F5-TTS hanging on load
  three times); [Chen2024] describes only steady-state inference RTF, not model-load behavior, so
  S-0018-01's retry has no literature guidance to draw on.
* **No paper studies whether an acceleration change (chunking, caching, precision, serving stack)
  measurably shifts `speaker_sim`/SIM-o/SS for CosyVoice2 or Chatterbox at fixed decoding content.**
  [Du2024]'s own streaming ablation is the closest analogue but measures CER, not speaker
  similarity, across its streaming/offline conditions; whether streaming or precision changes affect
  speaker similarity is untested in the reviewed literature and must be established empirically by
  t0021.

## Recommendations for This Task

1. **Adopt [Du2024]'s additive latency model (`L_TTS = M·d_lm + M·d_fm + M·d_voc`) as the schema for
   `results/latency_breakdown.json`**, extended with explicit reference-encoding and text-frontend
   stages to fully answer Key Question 1.
2. **Implement and validate reference-embedding caching before any other acceleration variant**,
   since [Wan2018] confirms the embedding is a deterministic function of the fixed reference clip —
   this is the lowest-risk, highest-confidence win identified in the literature.
3. **Prioritize LM/decoder-stage serving acceleration (vLLM, TensorRT-LLM, fp16/bf16, torch.compile)
   over vocoder-stage work**, because published vocoder speeds are already 150x-3,700x real time
   [Kong2020, Kaneko2022] while [Seo2026] shows LM/decoder-stage serving optimization alone was
   sufficient to reach sub-120 ms TTFP on a comparably sized decoder.
4. **Evaluate CosyVoice2 chunk-size reduction against both the standard prompt set and the hardest
   prompts separately**, per [Du2024, Table 8]'s finding that streaming quality cost concentrates in
   adversarial text.
5. **Score every acceleration variant's `speaker_sim` with the same GE2E scorer used for the t0018
   baseline, in the same session** — do not benchmark against any paper's own published SS/SIM-o
   number, since [Du2024], [Chen2024], and [Du2024a] all use encoders incompatible with GE2E cosine,
   and only [Casanova2022]'s SECS is directly comparable in principle.
6. **Route vocoder-head replacement ([Kaneko2022]-style) and block-diffusion decoding fine-tuning
   ([Seo2026]-style) into `results/suggestions.json` as follow-up task candidates**, not in-task
   experiments, since both require retraining forbidden under this task's scope.
7. **When answering Key Question 5 (is the 300 ms gap architectural or engineering), cite
   [Seo2026]'s 103-118 ms TTFP as evidence the gap is not strictly architectural for a
   similarly-sized decoder** — but qualify that this required fine-tuning a new decoding objective,
   so within t0021's no-fine-tuning constraint, engineering-only levers should be expected to close
   much, but not necessarily all, of the distance from 1.3-2.9 s down to 300 ms.

## Benchmark Comparison

Published reference numbers for the zero-shot cloning systems and related architectures in this
corpus, on their own respective benchmarks (not directly comparable across rows due to different
speaker-similarity encoders — see "Cross-System and Cross-Encoder Similarity Scores" above):

| System | Benchmark | WER/CER | Speaker sim | RTF | TTFB/TTFP | Source |
| --- | --- | --- | --- | --- | --- | --- |
| CosyVoice 2 | LibriSpeech test-clean | 2.47% WER | 0.745 SS (WavLM) | not reported | not reported | [Du2024, main results] |
| CosyVoice 2 | SEED test-zh | 1.45% CER | 0.748 SS (0.806 ERes2Net) | not reported | not reported | [Du2024, main results] |
| CosyVoice 2-S (streaming) | LibriSpeech test-clean | 2.45% WER | 0.751 SS | not reported | not reported | [Du2024, main results] |
| CosyVoice (v1) | LibriTTS test-clean | 2.89% ± 0.18% WER | 74.30 ± 0.15 SS (ERes2Net, 0-100 scale) | not reported | not reported | [Du2024a, Table 8] |
| F5-TTS (16 NFE) | LibriSpeech-PC test-clean | 2.53% WER | 0.66 SIM-o (WavLM) | 0.15 | not reported (non-streaming) | [Chen2024] |
| F5-TTS (32 NFE) | LibriSpeech-PC test-clean | 2.42% WER | 0.66 SIM-o | 0.31 | not reported | [Chen2024] |
| Chatterbox (AR baseline) | LibriSpeech-PC | 1.99% WER | 0.707 SIM-o (WavLM-ECAPA) | not reported | not reported | [Seo2026] |
| Chatterbox-Flash (canonical) | LibriSpeech-PC | 1.67% WER | 0.717 SIM-o | not reported | not reported | [Seo2026] |
| Chatterbox-Flash (D=16, α=0.5) | production streaming | not reported | not reported | 0.107 | 118 ms | [Seo2026] |
| Chatterbox-Flash (D=32, α=0.75) | production streaming | not reported | not reported | 0.076 | 103 ms | [Seo2026] |
| YourTTS | VCTK zero-shot | not reported | 0.864 SECS (GE2E-equivalent) | not reported | not reported | [Casanova2022] |

## Gaps in Latency-Specific Published Data (Cross-Reference)

The Benchmark Comparison table above makes visible, in one place, the central limitation already
detailed in "Gaps and Limitations": only [Seo2026] reports a first-byte/first-package latency number
at all, and it is for a fine-tuned derivative, not the stock Chatterbox or CosyVoice2 checkpoints
t0021 must accelerate. Every other row's "not reported" cell for TTFB/TTFP confirms that t0021's own
measurement work — not literature reuse — must establish these numbers for CosyVoice2 and Chatterbox
under Rezolve's own hardware, prompts, and acceleration variants.

## Paper Index

### [Du2024]

* **Title**: CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
* **Authors**: Du, Z., Wang, Y., Chen, Q., et al. (19 authors)
* **Year**: 2024
* **DOI**: `10.48550/arXiv.2412.10117`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2412.10117/`
* **Categories**: none (project has no registered categories)
* **Relevance**: The system directly under test; primary source for its streaming architecture,
  additive latency model, chunk-aware flow-matching ablations, and the documented acceleration
  levers (FSQ tokenizer, pretrained-LLM backbone, streaming ratio) this task profiles.

### [Chen2024]

* **Title**: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
* **Authors**: Chen, Y., Niu, Z., Ma, Z., et al. (8 authors)
* **Year**: 2024
* **DOI**: `10.48550/arXiv.2410.06885`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2410.06885/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Non-autoregressive flow-matching alternative architecture; its RTF-vs-TTFB
  distinction and portable, training-free Sway Sampling inference technique inform how t0021 should
  interpret and report RTF alongside TTFB for the systems it accelerates.

### [Seo2026]

* **Title**: Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS
* **Authors**: Seo, D., Park, G., Nam, K.
* **Year**: 2026
* **DOI**: `10.48550/arXiv.2605.30748`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2605.30748/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Built on the same Chatterbox T3 decoder t0021 accelerates; the only reviewed paper
  reporting paired TTFP/RTF numbers for a streaming zero-shot system (103-118 ms TTFP on H100),
  providing direct evidence on whether the architectural latency floor is closeable and by what kind
  of change (decoding scheme vs. serving-stack optimization).

### [Wan2018]

* **Title**: Generalized End-to-End Loss for Speaker Verification
* **Authors**: Wan, L., Wang, Q., Papir, A., Lopez Moreno, I.
* **Year**: 2018
* **DOI**: `10.1109/ICASSP.2018.8462665`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.1109_ICASSP.2018.8462665/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Defines the GE2E cosine-similarity methodology `speaker_sim` implements;
  establishes that reference-audio speaker embeddings are a deterministic, cacheable function of the
  reference clip, directly answering t0021's Key Question 2 for the reference-encoding stage.

### [Casanova2022]

* **Title**: YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for
  everyone
* **Authors**: Casanova, E., Weber, J., Shulby, C., et al. (6 authors)
* **Year**: 2022
* **DOI**: `10.48550/arXiv.2112.02418`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2112.02418/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Its SECS metric is defined identically to this project's GE2E-cosine `speaker_sim`,
  making it the only reviewed paper whose published similarity numbers are calibration-comparable to
  t0021's own results, used to sanity-check what a GE2E-cosine ≥0.85 target means in absolute terms.

### [Kunesova2025]

* **Title**: An Exploration of ECAPA-TDNN and x-vector Speaker Representations in Zero-shot
  Multi-speaker TTS
* **Authors**: Kunešová, M., Hanzlíček, Z., Matoušek, J.
* **Year**: 2025
* **DOI**: `10.48550/arXiv.2506.20190`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2506.20190/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Shows speaker-encoder choice materially changes speaker-similarity quality (not
  just speed) in a fixed TTS pipeline, and that similarity rankings vary by scoring extractor —
  directly cautions t0021 against treating the reference-encoding stage as a free substitution
  point.

### [Du2024a]

* **Title**: CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on
  Supervised Semantic Tokens
* **Authors**: Du, Z., Chen, Q., Zhang, S., et al. (12 authors)
* **Year**: 2024
* **DOI**: `10.48550/arXiv.2407.05407`
* **Asset**: `tasks/t0018_zero_shot_cloning_calibration/assets/paper/10.48550_arXiv.2407.05407/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Predecessor architecture CosyVoice2 explicitly benchmarks against; needed to
  correctly interpret CosyVoice2's own relative-improvement claims and to confirm CosyVoice's
  original HiFi-GAN-based vocoder stage, relevant to the vocoder term of the latency model.

### [Kong2020]

* **Title**: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech
  Synthesis
* **Authors**: Kong, J., Kim, J., Bae, J.
* **Year**: 2020
* **DOI**: `10.48550/arXiv.2010.05646`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2010.05646/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Origin architecture of the vocoder family used by CosyVoice/CosyVoice2; its
  published inference speeds (150x-3,700x real time) support the finding that the vocoder stage is
  unlikely to dominate CosyVoice2/Chatterbox TTFB relative to LM decoding.

### [Kaneko2022]

* **Title**: iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time
  Fourier Transform
* **Authors**: Kaneko, T., Tanaka, K., Kameoka, H., Seki, S.
* **Year**: 2022
* **DOI**: `10.48550/arXiv.2203.02395`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2203.02395/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Demonstrates a concrete, off-the-shelf vocoder-acceleration technique (1.2-1.7x
  additional speedup, flat-or-better quality) applicable if profiling shows the vocoder stage is a
  non-negligible share of TTFB; flagged as an out-of-scope future lever since it requires
  retraining.

### [Li2023]

* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., Mesgarani, N.
* **Year**: 2023
* **DOI**: `10.48550/arXiv.2306.07691`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2306.07691/`
* **Categories**: none (project has no registered categories)
* **Relevance**: Architecture of the project's own Kokoro fine-tune, used here as the
  non-autoregressive contrast case that explains why a StyleTTS2-family model can meet the 300 ms
  TTFB target architecturally while CosyVoice2/Chatterbox's autoregressive LM stage cannot without a
  decoding-scheme change.
