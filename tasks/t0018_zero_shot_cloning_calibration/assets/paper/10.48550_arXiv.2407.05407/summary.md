---
spec_version: "3"
paper_id: "10.48550_arXiv.2407.05407"
citation_key: "Du2024a"
summarized_by_task: "t0018_zero_shot_cloning_calibration"
date_summarized: "2026-09-17"
---
# CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised

# Semantic Tokens

## Metadata

* **File**: `files/du_2024_cosyvoice-scalable-multilingual-zero-shot-tts.pdf`
* **Published**: 2024
* **Authors**: Zhihao Du 🇨🇳, Qian Chen 🇨🇳, Shiliang Zhang 🇨🇳, Kai Hu 🇨🇳, Heng Lu 🇨🇳, Yexin Yang 🇨🇳,
  Hangrui Hu 🇨🇳, Siqi Zheng 🇨🇳, Yue Gu 🇨🇳, Ziyang Ma 🇨🇳, Zhifu Gao 🇨🇳, Zhijie Yan 🇨🇳
* **Venue**: preprint (arXiv, cs.SD)
* **DOI**: `10.48550/arXiv.2407.05407`

## Abstract

Recent years have witnessed a trend that large language model (LLM) based text-to-speech (TTS)
emerges into the mainstream due to their high naturalness and zero-shot capacity. In this paradigm,
speech signals are discretized into token sequences, which are modeled by an LLM with text as
prompts and reconstructed by a token-based vocoder to waveforms. Obviously, speech tokens play a
critical role in LLM-based TTS models. Current speech tokens are learned in an unsupervised manner,
which lacks explicit semantic information and alignment to the text. In this paper, we propose to
represent speech with supervised semantic tokens, which are derived from a multilingual speech
recognition model by inserting vector quantization into the encoder. Based on the tokens, we further
propose a Codec-based synthesizer for Voice generation, CosyVoice1, which consists of an LLM for
text-to-token generation and a conditional flow matching model for token-to-speech synthesis.
Experimental results show that supervised semantic tokens significantly outperform existing
unsupervised tokens in terms of content consistency and speaker similarity for zero-shot voice
cloning. Moreover, we find that utilizing large-scale data further improves the synthesis
performance, indicating the scalable capacity of CosyVoice. To the best of our knowledge, this is
the first attempt to involve supervised speech tokens into TTS models.

## Overview

This is the original CosyVoice paper (referred to by the authors as "CosyVoice1" once CosyVoice 2
was released), from Alibaba's Speech Lab. Its central claim is that speech tokens used by LLM-based
TTS systems should be learned with a supervision signal tied to text, rather than learned
unsupervised (as HuBERT- or EnCodec-style tokenizers do). The authors build a "Supervised Semantic
Speech" (S3) tokenizer by taking a multilingual ASR model (a fine-tuned SenseVoice for the
large-scale setting, an ESPnet Conformer for the small-scale LibriTTS setting), splitting its
encoder into two stages, and inserting a single-codebook vector quantizer (4,096 codes) between
them, trained end-to-end with the ASR cross-entropy loss so the discrete tokens stay text-aligned.

On top of the S3 tokens, CosyVoice couples an autoregressive text-to-token LLM with an
optimal-transport conditional flow matching (OT-CFM) model that converts tokens into a Mel
spectrogram, followed by a HiFi-GAN vocoder to produce the waveform. This factorizes voice identity
from content: the LLM models semantic content and prosody (conditioned on a speaker
x-vector/ERes2Net embedding and BPE text encodings), while the flow-matching model captures timbre
and channel/environment characteristics from the reference (prompt) audio. Because the sequence
construction interleaves text and speech tokens directly, CosyVoice does not need a separate
phonemizer or forced aligner, unlike prior flow-matching TTS systems (VoiceBox, Matcha-TTS,
ReFlow-TTS) that predict phoneme durations explicitly.

The paper reports evaluations at two data scales: a small-scale LibriTTS-only setting (used for
architecture ablations against VALL-E, UniAudio, and SpearTTS) and a large-scale multilingual
setting (170,000+ hours across Chinese, English, Cantonese, Japanese, and Korean) used for the
headline human-parity results on LibriTTS (English) and AISHELL-3 (Chinese). The paper also
introduces "CosyVoice-instruct," an instruction-tuned variant supporting speaker-identity,
speaking-style, and fine-grained paralinguistic control (laughter, breaths, emphasis), and
demonstrates using CosyVoice as a synthetic-data generator that improves a downstream ASR model when
mixed with real Librispeech data.

## Architecture, Models and Methods

* **S3 tokenizer**: ASR encoder split into `Encoder1` -> vector quantizer (single codebook, 4,096
  entries, EMA codebook update with decay coefficient alpha) -> `Encoder2` -> ASR decoder.
  Small-scale backbone: ESPnet Conformer, quantizer inserted after the first 6 encoder layers,
  trained from scratch on LibriSpeech for 50 epochs with a 4,000-token word-piece text tokenizer.
  Large-scale backbone: SenseVoice-Large (pretrained, then fine-tuned with the inserted quantizer)
  for 210,000 training steps on 8 A800 GPUs.
* **LLM**: autoregressive sequence model over `[S, v, text_encodings, T, speech_tokens, E]`, where
  `v` is a speaker embedding, `T`/`S`/`E` are turn/start/end markers, and only the speech-token and
  end-of-sequence cross-entropy terms are optimized (teacher forcing).
* **Text encoder**: aligns BPE text encodings with the speech-token semantic space before LLM
  modeling.
* **Flow matching**: optimal-transport conditional flow matching (OT-CFM) with a cosine-scheduled
  timestep `t := 1 - cos(t*pi/2)`, classifier-free guidance with condition-dropout probability 0.2
  during training and guidance strength beta = 0.7 at inference, conditioned on speaker embedding,
  speech tokens, and a masked Mel spectrogram; ResNet1D + Transformer blocks with timestep
  embedding.
* **Vocoder**: HiFi-GAN, converting generated Mel spectrograms to waveforms.
* **Model sizes**: "tiny" (6 text-encoder layers, attention dim 512, 8 heads, 2,048 linear units; 12
  LLM layers) vs. "normal" (6 text-encoder layers, attention dim 1,024, 16 heads, 4,096 linear
  units; 14 LLM layers). Tiny models trained on LibriTTS (585 hours, 2,456 speakers) for 50 epochs
  on 4x V100-32GB GPUs at learning rate 1e-3; normal/multilingual models trained for 800,000 steps
  on 64x V100-32GB GPUs at learning rate 1e-4, with a 10,000-step warmup.
* **Training data (large-scale)**: 130,000 hours Mandarin, 30,000 hours English, 5,000 hours
  Cantonese, 4,600 hours Japanese, 2,200 hours Korean; plus 101 hours speaker-identity, 407 hours
  speaking-style, and 48 hours fine-grained-paralinguistics instruction data for CosyVoice-instruct.
* **Evaluation metrics**: word error rate (WER) / character error rate (CER) via Whisper-Large V3
  (English) and Paraformer (Chinese) for content consistency; raw cosine similarity of ERes2Net
  speaker embeddings for speaker similarity (SS); emotion-control accuracy via the public emo2vec
  speech emotion recognition model. Random-sampling decoding evaluated over 5 seeds (0, 7, 42, 123,
  1337), reporting mean +/- standard deviation; an ASR 5x re-ranking variant is also evaluated for
  offline-mode gains.

## Results

* Inserting the vector quantizer into the ASR encoder only slightly degrades recognition:
  Conformer-VQ reaches **3.18%** WER on LibriTTS "test-clean" vs. **2.89%** for the non-VQ Conformer
  baseline, and **7.56%** vs. **6.57%** on "test-other".
* On Common Voice zh-CN, the multilingual S3 tokenizer (with language ID) achieves **12.06%** CER,
  beating Whisper-Large V3's **12.55%** -- a **4.14%** relative error reduction.
* On the LibriTTS "test-clean" content-consistency/speaker-similarity comparison (Table 7), the best
  CosyVoice configuration (BPE text token + S3 speech token, large-scale data) reaches **3.17%** WER
  and **69.49** speaker similarity (SS), against VALL-E's **18.70%** WER / **53.19** SS and
  SpearTTS's **6.14%** WER / **51.71** SS.
* On English (LibriTTS test-clean, Table 8), CosyVoice achieves **2.89% +/- 0.18%** WER and **74.30
  +/- 0.15** SS versus the original human recordings' **2.66%** WER; with 5x ASR re-ranking, WER
  drops to **1.51%**.
* On Chinese (AISHELL-3, Table 9), CosyVoice reaches **3.82% +/- 0.24%** CER and **81.58 +/- 0.16**
  SS, versus ChatTTS's **3.87%** CER (SS not evaluated for ChatTTS); 5x re-ranking lowers CER to
  **1.84%**.
* CosyVoice-instruct with emotional instructions raises accuracy sharply over CosyVoice-base on hard
  emotions, e.g. "Sad" accuracy **0.98 +/- 0.02** vs. **0.45 +/- 0.05**, and "Disgusted" **0.93 +/-
  0.02** vs. **0.46 +/- 0.06**.
* As a synthetic-data generator for ASR, adding CosyVoice-synthesized audio over MLS text on top of
  the real 960-hour Librispeech set lowers WER from **2.79%/5.97%** (Librispeech alone, dev/
  test-other columns) to **1.93%/4.53%** (dev_clean/test_other) in the best combined configuration.

## Innovations

### Supervised Semantic (S3) Speech Tokens

First TTS work to derive discrete speech tokens from a *supervised* ASR encoder (via an inserted
vector-quantization layer trained with the ASR cross-entropy loss) rather than from unsupervised
representation learning (HuBERT, EnCodec-style codecs). The paper argues, and shows empirically,
that this explicit text alignment improves both content consistency and speaker similarity for
zero-shot cloning relative to unsupervised tokenizers, while barely hurting the underlying ASR
model's recognition accuracy.

### LLM + Conditional Flow Matching Without Phonemizers or Forced Aligners

Unlike prior flow-matching TTS systems (VoiceBox, Matcha-TTS, ReFlow-TTS) that require external
phoneme duration prediction, CosyVoice interleaves text and speech tokens directly in the LLM input
sequence, removing the dependency on phonemizers and forced aligners while retaining the
training/inference speed benefits of flow matching over denoising diffusion.

### Disentangled Content/Prosody vs. Timbre Modeling

The LLM is responsible for semantic content and prosody generation (conditioned on a speaker
x-vector/ERes2Net embedding), while the conditional flow-matching model is responsible for timbre
and acoustic-environment fidelity, giving a clean separation of "what is said and how" from "who it
sounds like."

### Instruction-Tuned Controllability (CosyVoice-instruct)

Fine-tunes CosyVoice-base (without speaker-embedding conditioning) on speaker-identity,
speaking-style, and fine-grained paralinguistic instruction data (laughter, breath, emphasis
markup), enabling natural-language control over delivery in addition to zero-shot voice cloning.

## Datasets

* **LibriTTS**: 585 hours, 2,456 English speakers; official train-clean-100/360 + train-other-500
  merged for training, dev-clean for model selection, test-clean for the small-scale and English
  evaluation sets.
* **LibriSpeech**: 960-hour corpus, used to train the small-scale S3 tokenizer from scratch and as
  the base corpus for the synthetic-data-augmentation ASR experiment (with an MLS-text synthesis
  variant).
* **Large-scale internal multilingual corpus**: 130,000 hours Mandarin, 30,000 hours English, 5,000
  hours Cantonese, 4,600 hours Japanese, 2,200 hours Korean; collected in-house with speech
  detection, SNR estimation, speaker diarization/separation, and pseudo-labeled via SenseVoice-
  Large and Paraformer, refined with forced-alignment models. Not publicly released.
* **Instruction-tuning data**: 101 hours speaker-identity, 407 hours speaking-style, 48 hours
  fine-grained-paralinguistics examples (internal, not publicly released).
* **AISHELL-3**: multi-speaker Mandarin TTS test set used for the Chinese content-consistency/
  speaker-similarity evaluation.
* **Common Voice** (zh-CN, en): used to evaluate the S3 tokenizer's semantic-preservation ability
  against Whisper-Large V3 and SenseVoice-Large.
* Models and code released at `https://github.com/FunAudioLLM/CosyVoice`; demos at
  `https://fun-audio-llm.github.io`.

## Main Ideas

* CosyVoice 2 (already in this project's corpus, `10.48550_arXiv.2412.10117`) explicitly positions
  itself as an improvement over this paper -- its abstract opens with "In our previous work, we
  introduced CosyVoice..." -- so any claim CosyVoice 2 makes about *relative* gains (e.g. streaming
  latency, pronunciation error reduction) should be checked against this paper's own reported
  numbers rather than taken at face value.
* This paper reports raw cosine speaker-similarity using **ERes2Net** embeddings (values like 74.30
  and 81.58, on roughly a 0-100 scale), not GE2E cosine similarity -- the two are not directly
  comparable to this task's `speaker_sim` metric (GE2E cosine vs. 11labs David references) without
  re-scoring on the same encoder, reinforcing the metric-incompatibility issue already flagged in
  `research_papers.md`.
* The core architectural idea -- supervised, text-aligned discrete speech tokens rather than
  unsupervised codec tokens -- is a candidate explanation for why CosyVoice-family models might
  achieve better content consistency than Kokoro-style non-token or other-codec zero-shot systems;
  worth referencing when explaining *why* CosyVoice 2 benchmarks differ from Kokoro's fixed-speaker
  approach.
* CosyVoice's approach of separating "prompt speech" (zero-shot reference audio + optional prompt
  text transcript) from generation is architecturally similar to what this task's zero-shot cloning
  calibration is testing; the paper's own ablations (Exp-1 through Exp-4 in Table 7) show that going
  from a single-lingual to the large-scale multilingual S3 tokenizer without enough training data
  actually *degrades* content consistency and speaker similarity -- a caution against assuming more
  languages always helps without matching data scale.

## Summary

This paper introduces CosyVoice (later retroactively called "CosyVoice1"), a scalable multilingual
zero-shot TTS system from Alibaba's Speech Lab, whose central research question is whether tying
discrete speech-token learning to an explicit supervision signal -- rather than learning tokens
unsupervised -- improves zero-shot voice cloning quality. The motivating gap is that prior LLM-based
TTS systems (VALL-E, UniAudio, SpearTTS) rely on unsupervised tokenizers (HuBERT, EnCodec) whose
tokens lack explicit semantic/text alignment.

Methodologically, the authors build a "Supervised Semantic Speech" (S3) tokenizer by inserting a
single-codebook (4,096-entry) vector quantizer into a multilingual ASR encoder (SenseVoice-Large for
the large-scale setting), trained jointly with the ASR objective so the resulting discrete tokens
stay text-aligned. On top of these tokens, CosyVoice couples an autoregressive LLM (which models
semantic content and prosody, conditioned on a speaker embedding and BPE text encodings) with an
optimal-transport conditional flow-matching model (which models timbre and acoustic environment) and
a HiFi-GAN vocoder, avoiding the phonemizer/forced-aligner dependency of prior flow-matching TTS
systems.

The headline finding is that supervised S3 tokens outperform unsupervised tokenizers on both content
consistency and speaker similarity at equal scale, and that scaling training data further improves
quality to the point of "human parity": on LibriTTS test-clean, CosyVoice reaches **2.89% +/-
0.18%** WER and **74.30 +/- 0.15** ERes2Net speaker similarity versus the original human recordings'
**2.66%** WER, and on AISHELL-3, **3.82% +/- 0.24%** CER with **81.58 +/- 0.16** speaker similarity.
An instruction-tuned variant (CosyVoice-instruct) adds controllable emotion, speaking style, and
fine-grained paralinguistics on top of the base zero-shot cloning capability.

For this project, this paper matters primarily as the architectural and empirical foundation that
CosyVoice 2 (already in this task's corpus) explicitly builds on and benchmarks against, so reading
it is necessary to correctly interpret CosyVoice 2's own relative-improvement claims. It also
supplies a second, independent speaker-similarity methodology (ERes2Net raw cosine) that is *not*
GE2E-based, reinforcing the existing project note that GE2E `speaker_sim` results cannot be directly
compared against CosyVoice-family or F5-TTS numbers without re-scoring on a common speaker-encoder.
Finally, its ablation that scaling to more languages without proportionally more data can hurt both
content consistency and speaker similarity is a relevant caution for planning this task's own
zero-shot calibration experiments.
