---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
research_stage: "internet"
searches_conducted: 19
sources_cited: 18
papers_discovered: 7
date_completed: "2026-09-17"
status: "complete"
---
## Task Objective

Benchmark three open zero-shot voice-cloning TTS models — F5-TTS, CosyVoice 2, and Chatterbox — on
David reference audio with the t0008 harness to calibrate the reachable `speaker_sim`/TTFB envelope
for the project's Kokoro fine-tuning line. `research_papers.md` found zero corpus coverage of any of
the three systems, so this pass is the primary source for their install/inference requirements,
checkpoints, streaming behavior, and any published benchmark numbers.

## Gaps Addressed

From `research_papers.md` Gaps and Limitations:

1. **No paper describes F5-TTS, CosyVoice 2, or Chatterbox** — **Resolved** for install/inference
   mechanics via official repos and model cards
   [SWivid-F5TTS-GH; FunAudioLLM-CosyVoice-GH; ResembleAI-Chatterbox-GH; CosyVoice2-HF-ModelCard].
   Papers now exist to download for F5-TTS and CosyVoice 2
   [Chen2024-F5TTS; Du2024-CosyVoice2; Du2024-CosyVoice1]; Chatterbox has **no** research paper,
   confirmed directly by its maintainer [HF-Chatterbox-NoPaper-Discussion21].
2. **No paper documents GE2E speaker-similarity evaluation methodology** — **Partially resolved**.
   The GE2E paper [Wan2018-GE2E] and YourTTS's GE2E-based Speaker Encoder Cosine Similarity (SECS)
   methodology [Casanova2022-YourTTS] are now available, but neither gives expected cosine ranges
   for a voice like David's — that stays an empirical question for this task.
3. **Metric incompatibility (CMOS-S vs. GE2E) limits cross-paper comparison** — **Extended**: a
   second incompatibility was found between GE2E-cosine `speaker_sim` and the WavLM-TDNN-based
   "SIM-o"/"SS" that F5-TTS and CosyVoice 2 actually report [Chen2024-F5TTS; Du2024-CosyVoice2].
4. **No reference-duration ablation exists** — **Unresolved** as a formal ablation, but **partially
   resolved** operationally: F5-TTS's docs give concrete duration limits bearing on this task's
   `ref_concat` condition [F5TTS-InferReadme-GH; F5TTS-Streaming-Issue1225].

## Search Strategy

**Sources searched**: WebSearch, arXiv, GitHub (repos + issue trackers), Hugging Face model cards
and discussions, Resemble AI's site/blog, Podonos, PyPI.

**Queries executed** (19 total, exact text):

1. `F5-TTS SWivid GitHub zero-shot voice cloning install checkpoint`
2. `F5-TTS paper arxiv flow matching diffusion transformer text-to-speech`
3. `CosyVoice 2 FunAudioLLM GitHub streaming inference TTFB latency`
4. `CosyVoice 2 paper arxiv scalable streaming speech synthesis`
5. `Chatterbox ResembleAI TTS GitHub zero-shot voice cloning`
6. `F5-TTS SIM-O speaker similarity WER benchmark Seed-TTS test set`
7. `CosyVoice 2 speaker similarity WER benchmark table content consistency`
8. `Chatterbox TTS benchmark speaker similarity word error rate technical report`
9. `Generalized End-to-End Loss for Speaker Verification GE2E Wan et al paper`
10. `F5-TTS streaming inference real-time factor GPU RTX H100`
11. `F5-TTS github issue streaming output chunked infer_cli reference audio duration`
12. `"Chatterbox" resemble-ai arxiv technical report paper`
13. `CosyVoice 2 first packet latency 150ms streaming mode benchmark table`
14. `resemblyzer GE2E speaker embedding cosine similarity TTS evaluation zero-shot`
15. `Chatterbox TTS pip install python inference example reference audio clone`
16. `CosyVoice2-0.5B model card VRAM GPU requirements pip install python inference example`
17. `Chatterbox Turbo latency streaming real-time factor low-latency production benchmark`
18. `CosyVoice2 license Apache 2.0 FunAudioLLM github LICENSE`
19. `F5-TTS Emilia dataset CC-BY-NC license commercial use checkpoint`

Plus 4 targeted WebFetch reads (CosyVoice 2 arXiv abstract + full HTML benchmark tables, the
official Chatterbox GitHub README, the Hugging Face discussion confirming no Chatterbox paper) —
under the 12-page WebFetch budget. Roughly 30 distinct pages/results surfaced across all queries; 18
sources were selected for citation below as the most directly actionable (official repos, model
cards, papers, and issues with concrete numbers), dropping SEO listicles and duplicate install-guide
blogs.

**Date range**: no restriction; all support material is 2024-2026 (F5-TTS Oct 2024, CosyVoice 2 Dec
2024, Chatterbox 2025, Chatterbox Turbo 2026). **Inclusion criteria**: official
repo/model-card/paper for the three systems; GitHub issues with concrete numbers rather than
opinion; papers operationalizing GE2E-based similarity. **Exclusion criteria**: SEO blog posts with
no verifiable numbers; unofficial forks unless corroborating an official claim. **Iterations**:
queries 11-19 were follow-ups — 11 followed from query 1 to find F5-TTS's duration limits; 12
followed from finding no arXiv paper for Chatterbox in query 5; 17-19 chased Turbo-specific latency
and license terms not yet surfaced.

## Key Findings

### Checkpoints and Install Paths

F5-TTS installs via `git clone` + `pip install -e .`; the default zero-shot checkpoint is
`F5TTS_v1_Base` (1.25M steps), auto-downloaded from Hugging Face on first run [SWivid-F5TTS-GH].
CosyVoice 2's checkpoint `FunAudioLLM/CosyVoice2-0.5B` is fetched from Hugging Face or via
ModelScope's `snapshot_download('iic/CosyVoice2-0.5B', ...)`, with a `cosyvoice.cli.cosyvoice`
Python API [FunAudioLLM-CosyVoice-GH; CosyVoice2-HF-ModelCard]. Chatterbox is simplest:
`pip install chatterbox-tts`, then `ChatterboxTTS.from_pretrained(device="cuda")` and
`model.generate(text, audio_prompt_path=...)` [ResembleAI-Chatterbox-GH]. All three match the
checkpoint identifiers named in `task_description.md`.

### Licensing Diverges — a New Production-Relevance Finding

F5-TTS's code is MIT, but its released `F5TTS_v1_Base`/`F5TTS_Base` checkpoints are **CC-BY-NC-4.0**
because the training corpus (Emilia) is non-commercial; this restriction survives fine-tuning per a
maintainer discussion [F5TTS-License-Discussion997]. CosyVoice 2's code and assets are Apache-2.0
[FunAudioLLM-CosyVoice-GH]. Chatterbox is MIT end-to-end [ResembleAI-Chatterbox-GH].
`research_papers.md` could not surface this; it matters directly for Key Question 6 — if F5-TTS
leads on `speaker_sim`/TTFB, its default weights cannot ship in a commercial pipeline without a
separately licensed retrain.

### Streaming/TTFB Behavior Is Not Uniform

CosyVoice 2 is natively streaming (chunk-aware causal flow matching) and its authors claim
**first-packet latency as low as 150 ms** [Du2024-CosyVoice2], but that figure states no hardware or
percentile, and an independent GitHub issue reports persistent **P99 latency outliers at concurrency
of 4 or more** on a single RTX 4090 [CosyVoice-Issue1835] — the 150 ms number should be re-measured
as p50/p95/p99 on this task's H100, not assumed. F5-TTS's official CLI/Gradio path is
**non-streaming** (whole-utterance); community-patched streaming exists only as open issues, one
reporting a **first streamed packet with RTF over 1 second** [F5TTS-Streaming-Issue1225].
Chatterbox's base 0.5B model (`ResembleAI/chatterbox`, this task's checkpoint) is also non-streaming
officially [ResembleAI-Chatterbox-GH]; a separate, faster **Chatterbox-Turbo** (350M, 1-step
decoder) targets sub-200 ms latency [ResembleAI-ChatterboxTurbo-Blog], and a community wrapper
reports **RTF 0.499** and **~472 ms first-chunk latency** on an RTX 4090 for Turbo specifically
[Chatterbox-Issue193] — not applicable to the base model this task actually benchmarks.

### Published Speaker-Similarity Numbers Use a Different Embedding Backbone

The most consequential new finding relative to `research_papers.md`. F5-TTS reports "SIM"/"SIM-o"
using a **WavLM-large-based** verification model (Seed-TTS-eval convention), not GE2E: **WER 4.17 /
SIM 0.54** on Seed-TTS test-zh (945 h training data, 800K steps), **WER 2.42** on LibriSpeech-PC
test-clean at 32 NFE [Chen2024-F5TTS]. CosyVoice 2 reports similar WavLM-family "SS": **WER 2.47%,
SS 0.745, NMOS 3.96** on LibriSpeech test-clean; on SEED, **CER 1.45%/SS 0.806** (zh), **WER
2.57%/SS 0.736** (en), **WER 6.83%/SS 0.776** (hard cases); plus a **30-50%** pronunciation-error
reduction and MOS **5.4 -> 5.53** versus CosyVoice 1 [Du2024-CosyVoice2]. Neither is numerically
comparable to this project's GE2E-cosine `speaker_sim`, for the same reason `research_papers.md`
flagged for [Li2023]'s CMOS-S. The closer published precedent is [Casanova2022-YourTTS], which
defines Speaker Encoder Cosine Similarity (SECS) using a GE2E-trained `Resemblyzer` encoder — but it
reports SECS for YourTTS/VCTK/LibriTTS speakers, not for any of this task's three systems or David's
voice.

### No Official Chatterbox Paper — Only Vendor-Run AB Tests

A Hugging Face maintainer states plainly: *"There's no research paper [for Chatterbox] - mainly
because the research team is small (3 people) and we're working full steam ahead on another model"*
[HF-Chatterbox-NoPaper-Discussion21] — first-party, non-peer-reviewed. The only quantitative
third-party benchmark found is a Podonos blind AB test (naturalness/quality preference, not GE2E
similarity): **63.75%** preferred Chatterbox over ElevenLabs, **27.5%** preferred ElevenLabs,
**8.75%** no preference, using 7-20 s reference clips zero-shot [Podonos-Chatterbox-CaseStudy]. This
is vendor-commissioned and should not be cited quantitatively alongside this task's own GE2E
numbers.

### Reference-Audio Duration Limits Directly Affect the `ref_concat` Design

F5-TTS's docs recommend reference clips **under ~12 s**, warn that clips over **20 s risk mid-word
truncation**, and cap total prompt + generated audio at **30 s** per generation
[F5TTS-InferReadme-GH]. This task's `ref_concat` targets ~30 s of concatenated half-A audio — for
F5-TTS specifically that sits at or beyond the documented truncation-risk threshold and could
collide with the 30 s generation cap. Neither CosyVoice 2 nor Chatterbox publishes a comparable hard
limit in the sources reviewed.

## Methodology Insights

* **Use the exact checkpoint identifiers found here**: `F5TTS_v1_Base` [SWivid-F5TTS-GH],
  `FunAudioLLM/CosyVoice2-0.5B` (HF or ModelScope `iic/CosyVoice2-0.5B`) [CosyVoice2-HF-ModelCard],
  `ResembleAI/chatterbox` via `pip install chatterbox-tts` [ResembleAI-Chatterbox-GH].
* **Validate the `ref_concat` clip against F5-TTS's documented duration ceiling before the full
  run.** If ~30 s collides with the 20 s truncation-risk warning or 30 s total cap
  [F5TTS-InferReadme-GH], either trim the F5-TTS-specific input or document the deviation explicitly
  in `results_detailed.md` rather than let it silently truncate.
* **Instrument TTFB per actual streaming capability, not a uniform assumption**: true chunked TTFB
  for CosyVoice 2, re-measured as p50/p95/p99 rather than assuming the 150 ms marketing figure
  [Du2024-CosyVoice2; CosyVoice-Issue1835]; whole-utterance latency, explicitly labelled, for F5-TTS
  and base Chatterbox [F5TTS-Streaming-Issue1225; ResembleAI-Chatterbox-GH].
* **Never merge GE2E-cosine `speaker_sim` with any published SIM-o/SS/MOS/AB-test-%** from
  [Chen2024-F5TTS], [Du2024-CosyVoice2], or [Podonos-Chatterbox-CaseStudy] — extends
  `research_papers.md`'s existing CMOS-S rule to two more, differently-computed metric families.
* **Best practice, cross-source agreement**: all three systems' guidance converges on reference-clip
  *cleanliness* (single-speaker, low noise) mattering more than raw duration
  [ResembleAI-Chatterbox-GH; F5TTS-InferReadme-GH].
* **Record per-system license in the environment table** (Lesson 4): F5-TTS default checkpoint
  CC-BY-NC-4.0 [F5TTS-License-Discussion997]; CosyVoice 2 Apache-2.0 [FunAudioLLM-CosyVoice-GH];
  Chatterbox MIT [ResembleAI-Chatterbox-GH] — information invisible to `research_papers.md`.
* **Hypothesis**: because CosyVoice 2 has genuine chunked streaming and F5-TTS/base-Chatterbox do
  not, CosyVoice 2 is the most likely of the three to meet the TTFB <= 300 ms bar (Key Question 3);
  the other two should be expected to report whole-utterance latency well above it regardless of
  RTF.

## Discovered Papers

### Chen2024-F5TTS — F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

* Chen, Y., Niu, Z., Ma, Z., et al. (2024; ACL 2025 camera-ready). DOI `10.48550/arXiv.2410.06885`.
  URL: https://arxiv.org/abs/2410.06885
* Suggested categories: `zero-shot-tts`, `flow-matching`
* Why download: One of the three systems this task directly benchmarks; only source of its official
  SIM/WER numbers and architecture, and the corpus has zero coverage of it.

### Du2024-CosyVoice2 — CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models

* Du, Z., Wang, Y., Chen, Q., et al. (2024). DOI `10.48550/arXiv.2412.10117`. URL:
  https://arxiv.org/abs/2412.10117
* Suggested categories: `zero-shot-tts`, `streaming-tts`
* Why download: Second target system; only source of its official SS/WER/NMOS tables and streaming
  architecture underlying its TTFB claims.

### Du2024-CosyVoice1 — CosyVoice: A Scalable Multilingual Zero-shot TTS Synthesizer

* Du, Z., Chen, Q., Zhang, S., et al. (2024). DOI `10.48550/arXiv.2407.05407`. URL:
  https://arxiv.org/abs/2407.05407
* Suggested categories: `zero-shot-tts`
* Why download: Predecessor CosyVoice 2 explicitly benchmarks itself against (e.g. the 30-50%
  pronunciation-error-reduction claim).

### Wan2018-GE2E — Generalized End-to-End Loss for Speaker Verification

* Wan, L., Wang, Q., Papir, A., Lopez Moreno, I. (2018; ICASSP 2018). DOI
  `10.1109/ICASSP.2018.8462665`. URL: https://arxiv.org/abs/1710.10467
* Suggested categories: `speaker-verification`, `evaluation-methodology`
* Why download: Foundational paper behind this project's own `speaker_sim` (GE2E cosine) metric;
  closes the GE2E-methodology gap `research_papers.md` flagged.

### Casanova2022-YourTTS — YourTTS: Zero-Shot Multi-Speaker TTS and Voice Conversion for Everyone

* Casanova, E., Weber, J., Shulby, C., et al. (2022; ICML 2022). DOI `10.48550/arXiv.2112.02418`.
  URL: https://arxiv.org/abs/2112.02418
* Suggested categories: `zero-shot-tts`, `evaluation-methodology`
* Why download: Defines GE2E-based SECS (`Resemblyzer`) — the closest published precedent to this
  project's own similarity metric, unlike the WavLM-based SIM-o/SS used by F5-TTS/CosyVoice 2.

### Zhang2025-ECAPA — ECAPA-TDNN and x-vector Speaker Representations in Zero-shot TTS (low priority)

* Authorship not independently verified beyond the search snippet; re-verify on download. (2025).
  DOI `10.48550/arXiv.2506.20190`. URL: https://arxiv.org/abs/2506.20190
* Suggested categories: `evaluation-methodology`
* Why download: Compares four external speaker-embedding extractors (incl. GE2E/Resemblyzer) for
  zero-shot TTS eval — background for judging the project's GE2E-vs-WavLM metric choice.

### ChatterboxFlash2026 — Chatterbox-Flash: Block Diffusion for Streaming Zero-Shot TTS (low priority)

* Authorship not independently verified beyond the search snippet; re-verify on download. (2026).
  DOI `10.48550/arXiv.2605.30748`. URL: https://arxiv.org/abs/2605.30748
* Suggested categories: `zero-shot-tts`, `streaming-tts`
* Why download: Third-party streaming extension of the Chatterbox family; only tangentially relevant
  (Chatterbox itself has no paper) but the closest substitute for deeper Chatterbox-family
  grounding.

## Recommendations for This Task

1. **Validate the F5-TTS `ref_concat` clip (~30 s) against the 20 s truncation-risk threshold and 30
   s generation cap before the full run** [F5TTS-InferReadme-GH] — new, system-specific risk
   information `research_papers.md` could not surface.
2. **Report TTFB per real streaming capability**: chunked, re-measured p50/p95/p99 TTFB for
   CosyVoice 2 [Du2024-CosyVoice2; CosyVoice-Issue1835]; whole-utterance latency, labelled as such,
   for F5-TTS and base Chatterbox [F5TTS-Streaming-Issue1225; ResembleAI-Chatterbox-GH].
3. **Do not merge GE2E-cosine `speaker_sim` with any published SIM-o/SS/MOS/AB-test-% number** from
   F5-TTS, CosyVoice 2, or the Podonos Chatterbox study — extends `research_papers.md`'s CMOS-S rule
   to these newly found metrics.
4. **Record per-system license and flag F5-TTS's CC-BY-NC-4.0 default checkpoint explicitly in
   `results/suggestions.json`** if it leads on `speaker_sim`/TTFB — bears directly on Key Question
   6\.
5. **Add [Wan2018-GE2E] and [Casanova2022-YourTTS] to a follow-up literature-grounding task** (per
   `research_papers.md`'s Recommendation 6) — closest available precedent for this project's own
   `speaker_sim` metric family.
6. **Treat Key Question 4 (`ref_single` vs. `ref_concat`) as answerable only empirically** — no
   formal ablation exists anywhere found, academic or vendor; this task's own data is the first
   measurement for this metric family, as `research_papers.md` already anticipated.

## Source Index

### [Chen2024-F5TTS]

* **Type**: paper
* **Title**: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
* **Authors**: Chen, Y., Niu, Z., Ma, Z., et al.
* **Year**: 2024
* **DOI**: `10.48550/arXiv.2410.06885`
* **URL**: https://arxiv.org/abs/2410.06885
* **Peer-reviewed**: yes (ACL 2025 long paper)
* **Relevance**: Official SIM/WER benchmarks and architecture for one of the three target systems.

### [Du2024-CosyVoice2]

* **Type**: paper
* **Title**: CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models
* **Authors**: Du, Z., Wang, Y., Chen, Q., et al.
* **Year**: 2024
* **DOI**: `10.48550/arXiv.2412.10117`
* **URL**: https://arxiv.org/abs/2412.10117
* **Peer-reviewed**: no (arXiv technical report)
* **Relevance**: Official SS/WER/NMOS benchmarks and streaming architecture behind its TTFB claims.

### [Du2024-CosyVoice1]

* **Type**: paper
* **Title**: CosyVoice: A Scalable Multilingual Zero-shot TTS Synthesizer based on Supervised
  Semantic Tokens
* **Authors**: Du, Z., Chen, Q., Zhang, S., et al.
* **Year**: 2024
* **DOI**: `10.48550/arXiv.2407.05407`
* **URL**: https://arxiv.org/abs/2407.05407
* **Peer-reviewed**: no (arXiv technical report)
* **Relevance**: Predecessor baseline CosyVoice 2 benchmarks itself against.

### [Wan2018-GE2E]

* **Type**: paper
* **Title**: Generalized End-to-End Loss for Speaker Verification
* **Authors**: Wan, L., Wang, Q., Papir, A., Lopez Moreno, I.
* **Year**: 2018
* **DOI**: `10.1109/ICASSP.2018.8462665`
* **URL**: https://arxiv.org/abs/1710.10467
* **Peer-reviewed**: yes (ICASSP 2018)
* **Relevance**: Foundational paper behind this project's own `speaker_sim` (GE2E cosine) metric.

### [Casanova2022-YourTTS]

* **Type**: paper
* **Title**: YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for
  Everyone
* **Authors**: Casanova, E., Weber, J., Shulby, C., et al.
* **Year**: 2022
* **DOI**: `10.48550/arXiv.2112.02418`
* **URL**: https://arxiv.org/abs/2112.02418
* **Peer-reviewed**: yes (ICML 2022)
* **Relevance**: Defines GE2E-based SECS, the closest published precedent to this project's metric.

### [Li2023]

* **Type**: paper
* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., Mesgarani, N.
* **Year**: 2023
* **DOI**: `10.48550/arXiv.2306.07691`
* **URL**: https://arxiv.org/abs/2306.07691
* **Peer-reviewed**: yes (NeurIPS 2023)
* **Relevance**: Already in the project's corpus; its CMOS-S incompatibility precedent is extended
  here by the newly found WavLM-based SIM-o/SS incompatibility.

### [SWivid-F5TTS-GH]

* **Type**: repository
* **Title**: F5-TTS official code
* **Author/Org**: SWivid (F5-TTS authors)
* **Date**: 2024-10
* **URL**: https://github.com/SWivid/F5-TTS
* **Last updated**: 2026 (actively maintained)
* **Peer-reviewed**: no
* **Relevance**: Official install path, checkpoint names, and zero-shot cloning API for F5-TTS.

### [F5TTS-InferReadme-GH]

* **Type**: documentation
* **Title**: F5-TTS Inference README
* **Author/Org**: SWivid (F5-TTS authors)
* **Date**: 2026
* **URL**: https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/infer/README.md
* **Peer-reviewed**: no
* **Relevance**: Reference-audio duration guidance affecting this task's `ref_concat` design.

### [F5TTS-Streaming-Issue1225]

* **Type**: forum
* **Title**: The first packet of the streaming output is too long and the RTF is over 1 second.
* **Author/Org**: GitHub issue, SWivid/F5-TTS
* **Date**: 2026
* **URL**: https://github.com/SWivid/F5-TTS/issues/1225
* **Peer-reviewed**: no
* **Relevance**: Evidence F5-TTS streaming is community-patched/unreliable; base TTFB is
  whole-utterance latency.

### [F5TTS-License-Discussion997]

* **Type**: forum
* **Title**: Clarification on Training Data, Licensing, and Building a Commercial Base Model
* **Author/Org**: GitHub discussion, SWivid/F5-TTS
* **Date**: 2026
* **URL**: https://github.com/SWivid/F5-TTS/discussions/997
* **Peer-reviewed**: no
* **Relevance**: Confirms default F5-TTS checkpoints are CC-BY-NC-4.0 and stay non-commercial after
  fine-tuning — production risk for Key Question 6.

### [FunAudioLLM-CosyVoice-GH]

* **Type**: repository
* **Title**: CosyVoice official code
* **Author/Org**: FunAudioLLM
* **Date**: 2024-2026
* **URL**: https://github.com/FunAudioLLM/CosyVoice
* **Last updated**: 2026 (actively maintained)
* **Peer-reviewed**: no
* **Relevance**: Official install path, Apache-2.0 license, and Python inference API for CosyVoice
  2\.

### [CosyVoice2-HF-ModelCard]

* **Type**: documentation
* **Title**: FunAudioLLM/CosyVoice2-0.5B model card
* **Author/Org**: FunAudioLLM (Hugging Face)
* **Date**: 2024-2026
* **URL**: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
* **Peer-reviewed**: no
* **Relevance**: Confirms the checkpoint identifier named in `task_description.md`.

### [CosyVoice-Issue1835]

* **Type**: forum
* **Title**: Persistent P99 latency outliers at concurrency >= 4 on single RTX 4090
* **Author/Org**: GitHub issue, FunAudioLLM/CosyVoice
* **Date**: 2026
* **URL**: https://github.com/FunAudioLLM/CosyVoice/issues/1835
* **Peer-reviewed**: no
* **Relevance**: Counter-evidence to the headline "150 ms" latency claim; motivates measuring
  p50/p95/p99 directly.

### [ResembleAI-Chatterbox-GH]

* **Type**: repository
* **Title**: Chatterbox — SoTA open-source TTS
* **Author/Org**: Resemble AI
* **Date**: 2025-2026
* **URL**: https://github.com/resemble-ai/chatterbox
* **Last updated**: 2026 (actively maintained)
* **Peer-reviewed**: no
* **Relevance**: Official install path, checkpoint variants, MIT license, and non-streaming
  `generate()` API for Chatterbox.

### [HF-Chatterbox-NoPaper-Discussion21]

* **Type**: forum
* **Title**: Is research paper available for this model?
* **Author/Org**: Resemble AI maintainer, Hugging Face discussion
* **Date**: 2025
* **URL**: https://huggingface.co/ResembleAI/chatterbox/discussions/21
* **Peer-reviewed**: no
* **Relevance**: First-party confirmation that no Chatterbox research paper exists.

### [Podonos-Chatterbox-CaseStudy]

* **Type**: blog
* **Title**: How Resemble AI Used Podonos to Benchmark Chatterbox
* **Author/Org**: Podonos
* **Date**: 2025
* **URL**: https://www.podonos.com/blog/chatterbox
* **Peer-reviewed**: no
* **Relevance**: Only quantitative third-party Chatterbox-vs-ElevenLabs comparison found; vendor-run
  and not GE2E-based, so not directly usable in this task's tables.

### [ResembleAI-ChatterboxTurbo-Blog]

* **Type**: blog
* **Title**: Chatterbox Turbo: Open Source, Ultrafast Text-to-Speech
* **Author/Org**: Resemble AI
* **Date**: 2026
* **URL**: https://www.resemble.ai/chatterbox-turbo/
* **Peer-reviewed**: no
* **Relevance**: Describes the distilled low-latency Turbo variant (not this task's checkpoint),
  context for why the base model's latency differs.

### [Chatterbox-Issue193]

* **Type**: forum
* **Title**: Ways to reduce latency
* **Author/Org**: GitHub issue, resemble-ai/chatterbox
* **Date**: 2026
* **URL**: https://github.com/resemble-ai/chatterbox/issues/193
* **Peer-reviewed**: no
* **Relevance**: Community-reported Turbo streaming numbers (RTF 0.499, ~472 ms first chunk on RTX
  4090); single-source and Turbo-specific, not the base model this task uses.
