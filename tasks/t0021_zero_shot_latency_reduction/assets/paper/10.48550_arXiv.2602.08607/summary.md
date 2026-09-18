---
spec_version: "3"
paper_id: "10.48550_arXiv.2602.08607"
citation_key: "Cheng2026"
summarized_by_task: "t0021_zero_shot_latency_reduction"
date_summarized: "2026-09-18"
---
# VocalNet-MDM: Accelerating Streaming Speech LLM via Self-Distilled Masked Diffusion Modeling

## Metadata

* **File**: `files/cheng_2026_vocalnet-mdm-streaming-speech-llm.pdf`
* **Published**: 2026
* **Authors**: Ziyang Cheng 🇨🇳, Yuhao Wang 🇨🇳, Heyang Liu 🇨🇳, Ronghua Wu 🇨🇳, Qunshan Gu 🇨🇳, Yanfeng
  Wang 🇨🇳, Yu Wang 🇨🇳
* **Venue**: preprint (arXiv)
* **DOI**: `10.48550/arXiv.2602.08607`

## Abstract

Recent Speech Large Language Models (LLMs) have achieved impressive capabilities in end-to-end
speech interaction. However, the prevailing autoregressive paradigm imposes strict serial
constraints, limiting generation efficiency and introducing exposure bias. In this paper, we
investigate Masked Diffusion Modeling (MDM) as a non-autoregressive paradigm for speech LLMs and
introduce VocalNet-MDM. To adapt MDM for streaming speech interaction, we address two critical
challenges: training-inference mismatch and iterative overhead. We propose Hierarchical Block-wise
Masking to align training objectives with the progressive masked states encountered during block
diffusion decoding, and Iterative Self-Distillation to compress multi-step refinement into fewer
steps for low-latency inference. Trained on a limited scale of only 6K hours of speech data,
VocalNet-MDM achieves a 3.7x-10x decoding speedup and reduces first-chunk latency by 34% compared to
AR baselines. It maintains competitive recognition accuracy while achieving state-of-the-art text
quality and speech naturalness, demonstrating that MDM is a promising and scalable alternative for
low-latency, efficient speech LLMs.

## Overview

VocalNet-MDM targets a full speech-dialogue LLM (Thinker-Talker architecture, built on Qwen3-8B plus
a discrete-codec Talker), not a standalone TTS vocoder, but its contribution is squarely about
fixing autoregressive (AR) decoding inefficiency for streaming speech generation — the same problem
class as Chatterbox-Flash's prior-calibrated block diffusion. Where Chatterbox-Flash calibrates a
diffusion prior against AR-decoded statistics, VocalNet-MDM instead trains a masked diffusion model
(MDM) Talker with two purpose-built mechanisms: Hierarchical Block-wise Masking, which closes the
gap between how the model is trained (globally, uniformly masked sequences) and how it is actually
decoded at inference (block-causal, progressively revealed blocks), and Iterative Self-Distillation,
which distills a multi-step (K=4) teacher's late-stage, low-entropy predictions into a student that
can decode in as few as 1-2 diffusion steps per block.

The paper's central empirical claim is a speed/latency trade-off curve, not just a single operating
point: by varying diffusion steps per block (1, 2, 4, 8, 16) the same trained model can trade
text/speech quality for throughput, with Step-4 recommended as the default (matches Step-16
naturalness at much lower cost) and Step-1 recommended for latency-critical deployment. This mirrors
the kind of step-count/quality knob relevant to any diffusion- or MDM-based decoder being evaluated
for a latency-constrained production pipeline such as this project's TTFB <= 300 ms target, even
though VocalNet-MDM's own architecture (LLM Talker + external CosyVoice2 vocoder) is structurally
different from Kokoro-82M's StyleTTS2 decoder.

A second notable design decision is the explicit two-stage curriculum: first train the Talker with
LMDM under Global Bernoulli Masking to establish basic parallel-prediction ability, then fine-tune
with the distillation objective (LDistill) under Hierarchical Block-wise Masking. The ablations show
this ordering matters — applying Hierarchical Block-wise Masking from scratch (without the
distillation-stage teacher signal) actually degrades WER relative to plain Bernoulli masking, so the
two innovations are not independently useful; they depend on each other in this specific sequence.

## Architecture, Models and Methods

VocalNet-MDM adopts the Thinker-Talker paradigm from Qwen3-Omni. A Whisper-large-v3 speech encoder
(5x downsampling) feeds a Downsample Adaptor producing continuous representations r_{1:L}. The
Thinker (initialized from Qwen3-8B) autoregressively generates text tokens y_{1:N} and hidden states
h_{1:N}. The Talker (4 Transformer layers, trained from scratch, separate from the Thinker)
generates discrete speech tokens s_{1:T} via masked diffusion, conditioned on semantic anchors
derived from the Thinker's hidden states through "intra-block sparse semantic anchor alignment"
(anchors placed at the first Q=4 positions of each block of size B=16, assigned in an
order-preserving, prefix-only manner to avoid future semantic leakage). Predicted discrete speech
tokens are converted to waveform via the pretrained flow-matching model and HiFi-GAN vocoder from
CosyVoice2 (Du et al., 2024).

Decoding uses Block-Causal attention (from Arriola et al.'s Block Diffusion, 2025): tokens within a
block attend to each other fully, but attention across blocks is causal. Training masking
strategies: Global Bernoulli Masking samples a per-token mask probability gamma_g ~ U(0.3, 0.8).
Hierarchical Block-wise Masking is a two-stage process — first select a subset of blocks via
block-level ratio gamma_c ~ U(0.5, 1.0), then mask a fraction of positions within each selected
block via gamma_t ~ U(0.3, 1.0). Iterative Self-Distillation uses a frozen-teacher/student setup:
the teacher runs K=4 MDM iterations with block-wise parallel confidence-based unmasking (an
even-allocation update schedule n_j = ceil(R_j / (K-j+1)) reveals more tokens per block as
iterations proceed), recording its logits at the moment each token is revealed as distillation
targets Z_tea. The student is trained with Reverse KL at temperature tau=2.0 against these targets
(LKD), combined with the standard masked cross-entropy (LMDM) via LDistill = alpha*LKD +
(1-alpha)*LMDM, with alpha=0.7.

Training data: VoiceAssistant-400K (~430K single-turn pairs, GPT-4o generated) + UltraChat (~300K
single-turn splits from multi-round dialogues), ~730K examples total, ~6K hours of speech,
synthesized via CosyVoice2-0.5B. Training proceeds in three stages: (1) LoRA-align the Thinker to
audio for 1 epoch; (2) train the Talker with LMDM under Global Bernoulli Masking for 8 epochs; (3)
fine-tune with LDistill under Hierarchical Block-wise Masking for 4 epochs, using a frozen copy of
the stage-2 checkpoint as teacher. Optimizer: AdamW, learning rate 2e-4, batch size 32, trained on
A100 GPUs. Evaluation ran on a single L20 GPU, using English subsets of OpenAudioBench (AlpacaEval,
Llama Questions, TriviaQA, Web Questions), text quality scored 0-10 by Qwen-max, speech naturalness
via UTMOS, and WER via Whisper-large-v3 transcription.

## Results

* VocalNet-MDM achieves the best average text-quality score across 4 OpenAudioBench subsets:
  **7.07/10** (AlpacaEval **7.43**, Llama Questions **8.27**, TriviaQA **6.15**, Web Questions
  **6.42**) — identical across all diffusion-step variants since Talker step count does not affect
  the Thinker's text output.
* Relative to Baseline-AR (NTP), Step-4 gives **3.7x** higher TPS and a **73%** RTF reduction
  (0.0329 vs. 0.1222); Step-1 gives **10.5x** higher TPS (2153.43 vs. 204.64) and a **90.5%** RTF
  reduction (0.0116 vs. 0.1222).
* Relative to Baseline-AR (MTP), Step-4 gives **2.0x** higher TPS and Step-1 gives **5.7x** higher
  TPS (both baselines share TPS 374.81 / RTF 0.0667 with VocalNet-MDM Step-16).
* First-chunk latency (Table 2): VocalNet-MDM Step-1 achieves **368.67 +/- 13.82 ms**, the lowest of
  all compared systems; Step-16 achieves **427.45 +/- 16.25 ms**, versus GLM-4-Voice **1066.02 ms**,
  MiniCPM-o **1329.52 ms**, and Kimi-Audio **1371.48 ms** — over 2x lower than those three.
* Latency breakdown at first chunk (Figure 3): Speech Encoding **36.1 ms**, Thinker **99.6 ms**,
  Vocoder **225.6 ms** (constant across step counts), Talker **7.4 ms** (Step-1) to **72.2 ms**
  (Step-16) — the vocoder (flow matching + HiFi-GAN) dominates end-to-end latency, not the MDM
  Talker itself.
* WER (Whisper-transcribed): Step-4 achieves **5.34**, improving from **7.65** without Hierarchical
  Block-wise Masking during distillation fine-tuning — a **30%** relative improvement; Step-1
  achieves **6.23**, down from **9.69** (a **36%** relative improvement) under the same ablation.
* UTMOS naturalness is stable across step counts: **4.49** for Steps 4-16, **4.47** (Step 2), and
  **4.46** (Step 1) — Step-4 through Step-16 exceed all other compared open-source models.
* Ablation on distillation weight alpha: alpha=0 (no distillation) gives WER **41.27** at Step-1
  versus **6.23** at alpha=0.7, tau=2.0 — i.e., without Iterative Self-Distillation, few-step
  decoding collapses in quality; alpha=0.9 over-weights distillation and degrades WER to **15.95**
  at Step-1, showing an optimum at alpha=0.7.
* Without any distillation fine-tuning, Hierarchical Block-wise Masking alone is worse than Global
  Bernoulli Masking alone (WER **50.70** vs. **39.00** at Step-1), confirming the two innovations
  must be applied in the paper's specific two-stage order to help.

## Innovations

### Hierarchical Block-wise Masking

Standard MDM training (Global Bernoulli Masking) samples a single sequence-level masking ratio and
applies it uniformly, so all blocks in a training example look similarly (partially) masked. But at
block-diffusion inference time, past blocks are fully visible while the current block is
progressively revealed from fully masked to fully visible — a distribution the uniform-masking
training regime never explicitly exposes the model to. Hierarchical Block-wise Masking closes this
gap with a two-stage sampling process (block selection, then intra-block masking ratio) that
explicitly covers the progressive per-block masked states seen during real block-diffusion decoding.
The ablation shows this only helps when paired with the distillation fine-tuning stage — used alone
from scratch it underperforms plain Bernoulli masking, because it lacks the iterative refinement
signal needed to establish basic parallel-prediction ability first.

### Iterative Self-Distillation

Existing diffusion acceleration methods either only address throughput without cutting first-chunk
latency (KV-cache adaptation) or require complex multi-model coordination with high training cost on
long sequences (generic distillation). Iterative Self-Distillation instead uses a frozen K-step
(K=4) teacher — a self-copy of the model rather than a separate larger model — and transfers its
late-iteration, higher-confidence predictions to a single-forward-pass student via Reverse-KL
distillation at each block, using a block-wise parallel confidence-ranked update schedule (rather
than costly global-sequence confidence ranking or fully serial block updates). This lets the same
architecture decode in 1-2 steps per block at inference while approximating the quality of the
4-step teacher, directly targeting cumulative iterative-refinement latency rather than only
steady-state throughput.

## Datasets

* **VoiceAssistant-400K** (from Mini-Omni): GPT-4o-generated, ~430K single-turn query-response
  pairs, English, publicly available.
* **UltraChat** (from SLAM-Omni): split from multi-round dialogues into single-turn interactions,
  ~300K samples, English, publicly available.
* Combined training set: ~730K examples, ~6K hours of speech; speech responses synthesized via the
  open-source CosyVoice2-0.5B TTS model and its discrete codec tokens used directly for
  training/decoding targets — no original human speech data was collected for this work.
* Evaluation: English subsets of OpenAudioBench (AlpacaEval, Llama Questions, TriviaQA, Web
  Questions), all publicly available benchmark sets.

## Main Ideas

* The step-count knob (1/2/4/8/16 diffusion steps per block) demonstrates that a single trained MDM
  decoder can expose a tunable latency/quality trade-off at inference time with no retraining —
  directly relevant if this project ever evaluates diffusion-style decoders as a latency lever
  against the TTFB <= 300 ms target, since a single checkpoint could serve multiple latency SLAs.
  Kokoro-82M's StyleTTS2 decoder is not step-count-tunable in this way, so this is a comparison
  point rather than a directly transferable technique.
* The dominant latency contributor in the paper's own breakdown is the vocoder (flow matching +
  HiFi-GAN from CosyVoice2), not the sequence decoder — 225.6 ms of the ~370-427 ms first-chunk
  budget. This is a strong reminder to profile Kokoro's own vocoder/decoder latency breakdown rather
  than assuming gains come only from the acoustic-token or duration-predictor stage.
* The paper's ablation methodology — reporting WER/UTMOS at every step count, with and without each
  proposed component in isolation — is a useful template for this project's own ablations when
  comparing baseline vs. fine-tuned Kokoro checkpoints across epochs.
* Confirms (alongside Chatterbox-Flash) that "fix AR decoding via a distillation/calibration step
  built on top of an already-trained AR-style model" is an emerging, reproducible pattern across at
  least two independent groups for cutting first-chunk/TTFB-style latency in speech LLMs — this
  strengthens the case that decoding-side fixes (rather than only architecture swaps) are a viable
  lever, though neither paper's decoder architecture is Kokoro/StyleTTS2-compatible off the shelf.

## Summary

VocalNet-MDM investigates whether Masked Diffusion Modeling (MDM), a non-autoregressive paradigm
already proven for text LLMs, can replace the autoregressive (AR) Talker in a streaming
speech-dialogue LLM without the usual quality/latency penalties of naive parallel decoding. The
motivation is that AR decoding imposes strict serial constraints that grow latency linearly with
sequence length and introduces exposure bias, while prior NAR approaches suffer from weak long-range
dependency modeling, extra alignment overhead, or unresolved one-to-many generation uncertainty.

The method introduces two mechanisms built on top of a Thinker-Talker architecture (Qwen3-8B
Thinker, 4-layer from-scratch Talker, CosyVoice2 vocoder): Hierarchical Block-wise Masking, a
two-stage masking scheme that trains the model on the same progressively-revealed block states it
will see during block-diffusion inference (rather than uniform Global Bernoulli Masking); and
Iterative Self-Distillation, which uses a frozen K=4-step self-copy as teacher to distill
late-stage, high-confidence predictions into a student capable of 1-2-step decoding via Reverse-KL
loss. Training used a three-stage curriculum (Thinker audio alignment, base MDM training, then
distillation fine-tuning) on only ~6K hours of synthetic speech data.

The headline findings are a 3.7x-10.5x TPS speedup and up to 90.5% RTF reduction relative to an AR
(NTP) baseline sharing the same backbone, a 34% first-chunk latency reduction versus AR baselines
(368.67 ms at Step-1 vs. 555.86 ms for Baseline-AR NTP), and state-of-the-art text quality (7.07/10
average across four OpenAudioBench subsets) with competitive WER (5.34-6.23 for Steps 1-4) and UTMOS
naturalness (4.46-4.49) that exceeds most compared open-source speech LLMs. Ablations confirm both
mechanisms are necessary and must be applied in sequence — Hierarchical Block-wise Masking alone
(without prior distillation-stage training) underperforms plain Bernoulli masking.

For this project, VocalNet-MDM is a second, independently-developed data point (after
Chatterbox-Flash) showing that decoding-side fixes — rather than wholesale architecture changes —
can meaningfully cut first-chunk/TTFB-style latency in a speech-generation pipeline. Its
architecture (full LLM Talker + external vocoder) does not map directly onto Kokoro-82M's StyleTTS2
pipeline, so no component is directly portable, but two findings are actionable: (1) the paper's own
latency breakdown shows the vocoder, not the sequence model, dominates first-chunk latency, which
should prompt a similar breakdown for Kokoro's TTFB budget; and (2) the systematic step-count
ablation methodology (quality/latency reported at every operating point, with clean component
ablations) is a good template for this project's own Kokoro v4 checkpoint-epoch comparisons.
