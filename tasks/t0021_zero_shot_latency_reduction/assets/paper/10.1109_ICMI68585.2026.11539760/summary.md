---
spec_version: "3"
paper_id: "10.1109_ICMI68585.2026.11539760"
citation_key: "Zain2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
## Metadata

* File: Download failed
* Published: 2026
* Authors: Ali Zain 🇺🇸
* Venue: 2026 IEEE 5th International Conference on Computing and Machine Intelligence (ICMI)
* DOI: `10.1109/ICMI68585.2026.11539760`

## Abstract

Recent advancements in Text-to-Speech (TTS) have been propelled by Large Language Models (LLMs),
leading to systems like Llasa that achieve remarkable naturalness and speaker similarity in voice
cloning. However, these models, with parameter counts in the billions, present significant
challenges for deployment on resource-constrained devices due to their substantial memory footprint
and computational demands. This paper investigates the application of post-training quantization
(PTQ) to the Llasa family of TTS models (1B, 3B, and 8B parameters). Our goal is to drastically
reduce their size while maintaining high-fidelity voice cloning performance. We apply an 8-bit
(INT8) quantization scheme and evaluate the trade-offs between model compression and synthesis
quality using a suite of objective metrics. Our experiments reveal that INT8 quantization provides a
substantial reduction in model size, making large models more accessible for resource-constrained
environments. This efficiency gain comes at the cost of a modest, predictable decrease in synthesis
quality and, in our software-based setup, an increase in inference latency. Furthermore, we note
issues of output stability, suggesting that quantization is a mitigant, not a panacea. These
findings highlight a critical trade-off between model footprint and performance, providing a
practical guide for deploying large-scale TTS models.

## Overview

This summary is based on the abstract and publicly available metadata only; the full paper could not
be downloaded. IEEE Xplore returned HTTP 418 ("I'm a Teapot") to every automated fetch attempt
against both the abstract landing page and the direct PDF link recorded in the CrossRef metadata,
and the work is closed access (not available through OpenAlex, Semantic Scholar, or any indexed
repository), so no full text could be retrieved in this environment.

From the abstract, the paper studies post-training quantization (PTQ) applied to the Llasa family of
LLaMA-based TTS models, evaluated at three parameter scales (1B, 3B, and 8B). The stated motivation
mirrors the deployment problem this project cares about: billion-parameter, LLM-based TTS models are
expensive to run and hard to fit on resource-constrained hardware, so the paper asks whether simple
INT8 quantization can shrink them without destroying voice-cloning quality. The reported takeaway is
a genuine trade-off rather than a free win: model size drops substantially, but the authors observe
a "modest, predictable" quality decrease, an *increase* in inference latency in their software-based
setup, and output-stability issues at some settings — leading them to frame quantization as a
mitigant, not a panacea, for large-scale TTS deployment.

## Architecture, Models and Methods

Full methodology not available — paper not downloaded. Based on the abstract and the venue-indexed
reference list (via CrossRef), the following can be inferred: the paper targets the Llasa family of
LLaMA-based TTS transformers (citing Ye et al., "Llasa: Scaling train-time and inference-time
compute for llama-based speech synthesis," 2025) at three model sizes — 1B, 3B, and 8B parameters —
and applies post-training quantization (PTQ) using an 8-bit (INT8) integer quantization scheme,
evaluated against a "suite of objective metrics" for synthesis quality (specific metric names are
not given in the abstract). The reference list also cites foundational quantization work (Nagel et
al., "A white paper on neural network quantization," 2021; Bengio et al. on gradient estimation
through stochastic neurons, 2013) and a model-compression paper for zero-shot settings (Zhao et al.,
"Atom: A generalized framework for model compression," 2024), suggesting the PTQ approach may build
on established weight/activation quantization and straight-through-estimator-style techniques,
though this cannot be confirmed without the full text. The paper is a short conference paper
(CrossRef records pages 1-4), presented at ICMI 2026 in Al-Ahsa, Saudi Arabia (April 9-10, 2026). No
hardware specs, training/ calibration sample sizes, or exact evaluation metric names are available
from the abstract or metadata alone.

## Results

Results not available — paper not downloaded. Abstract reports (qualitatively, without specific
numeric values): INT8 quantization provides a "substantial reduction in model size" across the 1B,
3B, and 8B Llasa variants; this comes with a "modest, predictable decrease in synthesis quality"; in
the authors' software-based inference setup, quantization produced an *increase* in inference
latency rather than a decrease; and the authors observed "issues of output stability" at some
configurations. No specific percentages, latency figures (ms), model-size figures (MB/GB), or
objective-metric scores (e.g., MOS, WER, speaker similarity) are given in the abstract itself.

## Innovations

### Post-training quantization applied specifically to LLaMA-based voice-cloning TTS

Rather than quantizing a general-purpose LLM, the paper applies INT8 PTQ specifically to the Llasa
family of TTS transformers, evaluating the effect on voice-cloning fidelity rather than only on
generic language-modeling perplexity — a narrower and more directly relevant ablation than most
generic LLM-quantization literature.

### Multi-scale evaluation (1B / 3B / 8B)

The paper evaluates the same quantization scheme across three model sizes, which (per the abstract)
allows it to characterize how the size-vs-quality trade-off shifts with scale, rather than reporting
a single-model anecdote.

### Reporting a latency regression, not just a size win

Unlike much of the quantization literature that reports faster inference as a side benefit of
smaller weights, this paper explicitly reports an *increase* in inference latency in its
software-based setup — a cautionary, non-obvious finding that quantized weight formats do not
automatically translate into faster wall-clock inference without kernel/hardware support for the
lower-precision format.

## Datasets

Not available — paper not downloaded. The abstract does not name specific evaluation datasets,
corpora, or their sizes/languages/licenses. This cannot be determined without full-text access.

## Main Ideas

* INT8 post-training quantization of an LLM-based, LLaMA-family voice-cloning TTS model (Llasa,
  1B/3B/8B) is the closest published precedent found for quantizing a similarly-styled
  autoregressive TTS transformer for latency/footprint reduction — directly relevant context for
  this project's TTFB-reduction goal, even though Kokoro-82M is architecturally different (non-LLM,
  StyleTTS2-family) from Llasa.
* The paper's headline caution — that naive INT8 PTQ can *increase* inference latency in a
  software-only (non-kernel-optimized) deployment, despite shrinking model size — is a relevant risk
  to flag before assuming any quantization pass on Kokoro will help TTFB; latency gains from
  quantization are conditional on hardware/kernel support, not automatic.
* The paper also reports output-stability issues under quantization, reinforcing the project's need
  to re-run speaker-similarity (GE2E cosine) and audio-quality regression checks after any
  quantization experiment, not just a latency benchmark, before considering a quantized variant a
  valid candidate.
* Because full text is inaccessible (IEEE paywall, HTTP 418 to automated fetch), any deeper
  technical detail needed (exact quantization recipe, calibration procedure, specific quality-metric
  deltas) would require manual/authenticated access to IEEE Xplore before this paper could inform an
  implementation plan in detail.

## Summary

This paper investigates whether simple 8-bit post-training quantization (PTQ) can make large,
LLaMA-based TTS voice-cloning transformers cheaper to deploy without unacceptably degrading
synthesis quality. The motivating problem — billion-parameter LLM-based TTS models being too
expensive and slow for resource-constrained deployment — closely parallels this project's own goal
of finding a cheaper, faster alternative to a proprietary TTS system while preserving speaker
similarity.

The authors apply an INT8 quantization scheme to the Llasa family of LLaMA-based TTS models at three
parameter scales (1B, 3B, and 8B) and measure the resulting trade-offs with a suite of objective
metrics, according to the abstract. Beyond that, the specific quantization recipe, calibration
procedure, evaluation metrics, and hardware setup are not known, because the full paper could not be
downloaded: IEEE Xplore returned HTTP 418 to every automated fetch attempt (both the landing page
and the direct CrossRef-indexed PDF link), and the work is not open access anywhere else that was
checked (OpenAlex, Semantic Scholar, general web search).

What the abstract does report is a genuine, non-trivial trade-off: model size drops substantially
under INT8 quantization, but synthesis quality suffers a "modest, predictable" decline, inference
latency *increases* in the authors' software-based setup, and some configurations show output
instability. The authors' own framing — quantization as "a mitigant, not a panacea" — signals that
this is not a case of a straightforward win being reported.

For this project, the paper is useful primarily as the closest located published precedent for
INT8-quantizing an LLM-style voice-cloning TTS transformer, and as a concrete warning that
quantization does not guarantee latency improvement without matching hardware/kernel support. It
should be treated as directional context rather than a source of transferable numeric benchmarks,
since Kokoro-82M is architecturally distinct from the Llasa family and the paper's quantitative
results are not accessible without an authenticated IEEE Xplore subscription.
