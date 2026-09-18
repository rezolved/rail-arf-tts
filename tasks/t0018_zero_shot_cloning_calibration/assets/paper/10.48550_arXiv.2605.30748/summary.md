---
spec_version: "3"
paper_id: "10.48550_arXiv.2605.30748"
citation_key: "Seo2026"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
# Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS

## Metadata

* **File**: `files/seo_2026_chatterbox-flash-block-diffusion-streaming.pdf`
* **Published**: 2026 (submitted 2026-05-29, this asset is based on v3, updated 2026-08-21)
* **Authors**: Deokjin Seo (Resemble AI), Gangin Park 🇰🇷 (Seoul National University), Kihyun Nam 🇰🇷
  (KAIST)
* **Venue**: arXiv preprint (cs.SD / cs.AI / eess.AS)
* **DOI**: `10.48550/arXiv.2605.30748`

## Abstract

We present Chatterbox-Flash, a zero-shot text-to-speech model obtained by fine-tuning a pretrained
autoregressive TTS decoder into a block-diffusion decoder, enabling parallel token generation within
each block while retaining block-by-block streaming. We find that naively transferring mainstream
block-diffusion decoding to discrete speech tokens degrades quality, as a long-tail token
distribution biases parallel position selection toward a few high-frequency tokens. To mitigate this
without architectural modification, we introduce two inference-time techniques: prior-calibrated
scoring, which subtracts the block-level marginal token distribution, and an early-decoding
schedule, which adaptively terminates iteration based on calibrated confidence. On standard
zero-shot TTS benchmarks, Chatterbox-Flash attains high-fidelity synthesis comparable to strong
autoregressive and non-autoregressive baselines, while supporting streaming inference with
time-to-first-packet on par with streaming AR systems and substantially lower real-time factor. Code
and audio samples are available at https://github.com/resemble-ai/chatterbox-flash.

## Overview

Chatterbox-Flash is built by taking the open-source Chatterbox-TTS autoregressive decoder (Resemble
AI's T3 Llama-style transformer, the same Stage-1 decoder architecture underlying the Chatterbox
voice-cloning stack this task benchmarks) and fine-tuning it, without any architectural change, into
a block-diffusion decoder: the discrete speech-token sequence is partitioned into non-overlapping
blocks of size `D`, blocks are generated left-to-right (enabling streaming), and within each block
all masked positions are denoised in parallel via masked prediction. Stage 2 remains an unmodified
flow-matching vocoder (a CosyVoice2-style architecture distilled into a MeanFlow few-step sampler)
that converts speech tokens to waveform with chunk-wise streaming, so all reported quality
differences are attributable purely to the Stage-1 decoding strategy.

The paper's central empirical finding is that decoding techniques from text diffusion language
models (e.g. Fast-dLLM v2's top-confidence position selection) transfer poorly to discrete speech
codecs: codec token distributions are heavily long-tailed (dominated by silence/low-energy "dominant
tokens"), so raw model confidence preferentially unmasks these high-frequency, low-information
tokens first, corrupting the acoustic context for the rest of the block (boundary-induced context
truncation). The authors fix this with prior-calibrated scoring (PMI): subtracting a cached,
context-free marginal token distribution from the per-position confidence to produce a
pointwise-mutual-information score that ranks positions by how specifically the predicted token is
licensed by local context rather than by raw frequency. This calibrated score is then reused as a
confidence signal to drive an early-decoding schedule that adaptively terminates block denoising
once scores exceed a quantile threshold, cutting steps without quality loss.

Evaluated on LibriSpeech-PC test-clean and Seed-TTS test-en, the 0.5B-parameter Chatterbox-Flash
(trained on ~70k hours of English speech) matches or exceeds strong autoregressive baselines
(IndexTTS2, CosyVoice3, VoxCPM, Qwen3-TTS, Chatterbox itself) and non-autoregressive baselines
(F5-TTS, ZipVoice, MaskGCT, OmniVoice) on SIM-o/WER/UTMOS while being the only evaluated system with
native block-causal streaming support. A held-out human evaluation against ElevenLabs v3 found
statistically indistinguishable mean naturalness but a significant speaker-similarity advantage for
Chatterbox-Flash. The paper also reports a harder out-of-distribution stress test (EmergentTTS-Eval)
where PMI's advantage over plain top-confidence scoring becomes most visible, particularly on the
"pronunciation" category under streaming.

## Architecture, Models and Methods

* **Base architecture**: Chatterbox-TTS's Stage-1 T3 decoder (Llama-style causal transformer, 0.5B
  parameters) performing next-token prediction over discrete speech tokens extracted at 25 Hz by a
  neural audio codec, conditioned on `c = [speaker_embedding, text_tokens, prompt_speech_tokens]`;
  the speaker embedding comes from a GE2E-trained voice encoder (Wan et al. 2020, the same GE2E
  paper already in this project's corpus as `Wan2018`).
* **Block diffusion**: the T3 decoder is fine-tuned to model `p(x) = prod_b p(x^(b) | x^(<b))` over
  `B = ceil(T/D)` speech blocks; within a block, positions are masked (`[M]`) and recovered in
  parallel (masked denoising), while cross-block attention stays causal, enabling block-by- block
  streaming.
* **Hybrid attention mask**: causal over the conditioning prefix, full (bidirectional)
  speech-to-prefix and intra-block attention, and causal inter-block attention; implemented via
  PyTorch `flex_attention` or MagiAttention's Flex-Flash-Attention kernel.
* **Training objective**: token-shift denoising loss (Fast-dLLM v2 style, predicting masked position
  `i` from hidden state at `i-1`) with complementary masking (both a mask `m` and its complement
  `1-m` supervised per batch); noise level `t ~ U(epsilon, 1-epsilon)`; trained with AdamW, cosine
  LR schedule (peak `1e-5`, 10% warmup), effective batch size 440, bfloat16 precision, block size
  `D = 32` at training time, on NVIDIA H100 GPUs.
* **Prior-calibrated scoring (PMI)**: score
  `s_i^(k) = log p_i^(k)(x_hat_i^(k)) - log p_bar(x_hat_i^(k))`, where `p_bar` is an unconditional
  block prior computed once per model from a single forward pass on an all-masked sequence with
  zeroed conditioning, and cached for the model's lifetime.
* **Early-decoding schedule**: adaptive per-step quantile threshold
  `theta_k = Quantile({s_i^(k)}, q_k)` with `q_k = max(0, 1 - alpha * (k+1)/K)`, combined with a
  time-shifted (TS) unmasking-rate schedule (`tau = 0.5`); `alpha` in `{0, 0.5, 0.75, 1.0}` trades
  quality for compute (canonical settings: `alpha=0` quality-strongest, `alpha=0.5`
  efficiency-oriented).
* **Canonical inference config**: block size `D = 16`, denoising steps `K = 8`, TS schedule
  `tau = 0.5`, classifier-free guidance scale `w = 1.0`, sampling temperature `T = 0.2`, position
  (Gumbel) temperature `beta = 5`.
* **Training data**: ~70k hours / 43.8M utterances / 528k speakers, combining public corpora
  (MLS-English 10.8M utterances, Emilia-en 9.1M, Loquacious 3.9M, GLOBE 582K, LibriTTS-R 375K,
  HiFi-TTS 324K, EARS 12K, Expresso 12K) and privately collected audiobook (17.7M), podcast (726K),
  voice-conversion-augmented IVR (445K), short-form (292K), conversational (50K), and stylized
  speech (62K) data.
* **Evaluation benchmarks/metrics**: LibriSpeech-PC test-clean and Seed-TTS test-en, reporting SIM-o
  (WavLM-ECAPA-TDNN cosine speaker similarity), WER (HuBERT ASR on LibriSpeech-PC, Whisper-large-v3
  on Seed-TTS), and UTMOS naturalness; the harder EmergentTTS-Eval benchmark (model-as-judge) is
  used for prosodic/expressive/linguistic stress testing; a 10-utterance, 7-rating-per-system human
  study (NMOS/SMOS, 5-point Likert, 95% CI) benchmarks against ElevenLabs v3. Latency is measured as
  time-to-first-packet (TTFP) and real-time factor (RTF) on NVIDIA H100 GPUs at concurrency 1 over
  50 utterances, using FlashInfer attention kernels and paged KV-cache management.

## Results

* Canonical config (`D=16, K=8, alpha=0`, quality-strongest) attains **SIM-o 0.717 / WER 1.67 /
  UTMOS 4.29** on LibriSpeech-PC and **SIM-o 0.704 / WER 1.96 / UTMOS 4.09** on Seed-TTS test-en
  with PMI scoring, versus autoregressive Chatterbox's **0.707 / 1.99 / 4.29** and **0.685 / 2.20 /
  4.10** respectively -- improving both SIM-o and WER over its own AR backbone at preserved UTMOS.
* Chatterbox-Flash attains the **best UTMOS among all non-autoregressive baselines compared** on
  both LibriSpeech-PC (**4.29**) and Seed-TTS test-en (**4.09**), and beats F5-TTS, MaskGCT, and
  OmniVoice-Emilia on WER, despite training on only ~70k hours versus OmniVoice's 581k hours.
* Naive Fast-dLLM v2 decoding transferred to speech collapses quality: **WER 15.36** on
  LibriSpeech-PC and **14.49** on Seed-TTS test-en (vs. 1.67 / 1.96 for PMI), confirming that
  off-the-shelf text-diffusion decoding does not transfer to discrete speech codecs.
* Prior-calibrated early decoding (PMI+ED, `alpha=0.5`) reduces the average denoising step count
  from 8 to **~6.3 steps/block (~20% reduction)** while matching the no-early-decoding TS baseline's
  quality, and beats the TS+ED baseline at the same step budget (**1.67 vs. 1.83 WER** on
  LibriSpeech-PC, **2.04 vs. 2.20 WER** on Seed-TTS).
* On the harder EmergentTTS-Eval benchmark, moving from offline to streaming raises the TS
  baseline's overall WER by **+3.78 points (35.75 -> 39.53)** but PMI's by only **+0.89 points
  (34.79 -> 35.68)**; the gap is driven by the pronunciation category, where TS streaming WER
  reaches **85.88** versus PMI's **~70-72**.
* In production streaming configuration (`D=16, alpha=0.5`), Chatterbox-Flash achieves **TTFP 118 ms
  and RTF 0.107** on H100 (roughly **9x real-time** at concurrency 1); a more aggressive setting
  (`D=32, alpha=0.75`) reaches **TTFP 103 ms and RTF 0.076 (~13x real-time)**.
* Human evaluation against ElevenLabs v3 (10 Seed-TTS test-en utterances, 7 ratings/system) found
  naturalness statistically indistinguishable (**NMOS 3.91 +/- 0.23 vs. 4.04 +/- 0.26**, CIs
  overlap) but a significant speaker-similarity win for Chatterbox-Flash (**SMOS 4.56 +/- 0.18 vs.
  3.50 +/- 0.35**, non-overlapping CIs), and fewer catastrophic naturalness failures (**8.6% of
  ratings <= 2 vs. 12.9%** for ElevenLabs v3).
* Block size ablation shows SIM-o/UTMOS are essentially flat for `D` in `{8, 16, 24}` but WER
  degrades sharply at `D >= 24`; the model becomes unstable at `D >= 128` in appendix explorations
  of block-size scaling.

## Innovations

### Streaming Block-Diffusion TTS

To the authors' knowledge, the first zero-shot TTS model to perform block-diffusion decoding (in the
sense of Arriola et al.'s block diffusion) directly over discrete speech codec tokens under
block-causal attention over the full prefix, giving native block-by-block streaming without any
change to the underlying autoregressive backbone's architecture.

### Prior-Calibrated Scoring (PMI)

An inference-time-only correction that subtracts a cached, context-free marginal token distribution
from per-position model confidence, converting raw confidence (which is biased toward high-frequency
"dominant" tokens such as silence) into a pointwise-mutual-information score that reflects how
specifically a predicted token is licensed by local context. Requires no architectural change or
extra forward pass beyond a one-time cached prior computation.

### Early-Decoding Schedule

An adaptive termination rule that uses the calibrated PMI confidence to relax a per-step quantile
threshold over the decoding budget, cutting the average number of denoising iterations per block by
roughly 20% at negligible quality cost, and by up to ~41% at a larger but still WER-bounded quality
cost (`alpha=1.0`, WER increase capped at "+0.6" in the paper's reporting).

## Datasets

* **Training** (~70k hours, 43.8M utterances, 528k speakers): public corpora MLS-English (10.8M
  utterances), Emilia (English portion, 9.1M), Loquacious (3.9M), GLOBE (582K), LibriTTS-R (375K),
  HiFi-TTS (324K), EARS (12K), Expresso (12K); privately collected audiobook (17.7M), podcast
  (726K), IVR/voice-conversion-augmented (445K), short-form (292K), conversational (50K), and
  stylized speech (62K) data.
* **Evaluation**: LibriSpeech-PC test-clean (zero-shot voice cloning benchmark built on the
  test-clean split of LibriSpeech-PC), Seed-TTS test-en (English evaluation set from Seed-TTS), and
  EmergentTTS-Eval (model-as-judge benchmark targeting complex prosodic, expressive, and
  linguistically difficult utterances, broken into pronunciation/paralinguistic/foreign-word
  categories) for harder stress testing.
* **Human evaluation**: 10 utterances from Seed-TTS test-en, 7 ratings per utterance per system (70
  ratings total per system), 5-point Likert NMOS/SMOS scales.
* All evaluation datasets are publicly available; the privately collected training subsets
  (audiobook, podcast, IVR, conversational, stylized) are not described as publicly released.

## Main Ideas

* Chatterbox-Flash is built directly on top of the open-source Chatterbox-TTS T3 decoder that this
  task (`t0018_zero_shot_cloning_calibration`) benchmarks -- it is the closest available
  research-grade extension of the Chatterbox family and can inform how block-diffusion decoding and
  streaming might apply to Rezolve's own Chatterbox-based comparisons, even though it is not itself
  a paper about the base Chatterbox model.
* Raw model confidence is an unreliable ranking signal for parallel token unmasking over discrete
  speech codecs because codec tokens are long-tailed (dominant silence/low-energy tokens); any
  project doing parallel/non-autoregressive decoding over speech tokens should consider a PMI-style,
  context-conditioned calibration rather than raw softmax confidence.
* The paper's streaming latency methodology (TTFP measured to first emitted audio packet, RTF at
  concurrency 1, chunk-wise vocoder emission with a progressively widening chunk schedule) is a
  directly relevant reference protocol for this project's own `ttfb_ms` and `rtf` metrics.
* Speaker-similarity (SIM-o via WavLM-ECAPA-TDNN, and human SMOS) and naturalness (UTMOS, human
  NMOS) are evaluated as separate axes in this paper, reinforcing that a model can match naturalness
  while differing sharply on speaker similarity -- relevant given this project's own GE2E-cosine
  speaker-similarity target.
* The appendix's block-size-scaling failure modes (loss of prosodic variability under fully causal
  block training; confidence collapse when block size grows faster than denoising capacity) are
  useful cautionary references if any future work in this project explores diffusion-style or
  block-parallel decoding for the Kokoro fine-tune.

## Summary

Chatterbox-Flash asks whether a pretrained autoregressive zero-shot TTS decoder can be converted,
via fine-tuning alone and without architectural change, into a block-diffusion decoder that keeps
block-by-block streaming while allowing parallel token generation within each block -- the goal
being to remove the sequential, linearly-growing latency of autoregressive decoding without
sacrificing quality. The work is built directly on top of Resemble AI's open-source Chatterbox-TTS
T3 decoder, making it the closest available research artifact for grounding the Chatterbox family,
which otherwise has no dedicated research paper.

The authors' key methodological insight is that decoding techniques inherited from text-diffusion
language models do not transfer cleanly to discrete speech codecs: raw model confidence
preferentially commits high-frequency, low-information "dominant" tokens (e.g. silence), corrupting
acoustic context for later positions in the block. They fix this at inference time only, with no
architectural changes, via prior-calibrated (PMI) scoring that subtracts a cached, context-free
marginal token distribution from per-position confidence, and pair it with an adaptive
early-decoding schedule that uses the calibrated confidence to terminate block denoising early.

Empirically, the 0.5B-parameter model, fine-tuned on ~70k hours of English speech, matches or beats
strong autoregressive and non-autoregressive zero-shot TTS baselines (including its own AR
Chatterbox backbone) on SIM-o, WER, and UTMOS across LibriSpeech-PC and Seed-TTS test-en, while
being the only evaluated non-autoregressive system with native streaming support, achieving TTFP as
low as 103 ms and RTF as low as 0.076 on H100 GPUs. A human evaluation against ElevenLabs v3 found
indistinguishable naturalness but a clear speaker-similarity advantage for Chatterbox-Flash, and the
PMI calibration's benefit is shown to be largest precisely on harder, streaming, out-of-distribution
inputs (EmergentTTS-Eval).

For this project, which directly benchmarks Chatterbox as a zero-shot cloning baseline, this paper
is a useful, if only tangentially relevant, secondary source of Chatterbox-family grounding: it
confirms the underlying T3/Chatterbox-TTS architecture, GE2E speaker conditioning, and gives a
concrete streaming latency measurement protocol (TTFP/RTF) that parallels this project's own
`ttfb_ms`/`rtf` metrics. It is not a paper about Chatterbox itself, and its block-diffusion decoding
contribution is orthogonal to this project's Kokoro fine-tuning work, so its practical applicability
here is limited to background grounding and methodology cross-reference rather than a direct
baseline or technique to adopt.
