---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
research_stage: "papers"
papers_reviewed: 4
papers_cited: 4
categories_consulted: []
date_completed: "2026-09-17"
status: "complete"
---
## Task Objective

This task benchmarks three open zero-shot voice-cloning TTS systems — F5-TTS (`SWivid/F5-TTS`, base
English checkpoint), CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`), and Chatterbox
(`ResembleAI/chatterbox`) — on David reference audio using the existing t0008 evaluation harness,
across two reference-audio conditions (`ref_single`, a ~10 s clip, and `ref_concat`, a ~30 s
concatenation), on the same 96 val96 prompts and 100 filler prompts used throughout the project. The
two ElevenLabs David / Kokoro v3 baselines are re-scored in the same session for a paired
comparison. Per `task_description.md`, this is explicitly "a measurement task, not a model-selection
task": no fine-tuning of any system and no tuning of reference selection on val_96. The purpose is
to establish an empirical ceiling for `speaker_sim` (GE2E cosine vs. the ElevenLabs half-B centroid)
and a TTFB/RTF envelope reachable on the David voice without any training, so the project can judge
whether Kokoro Stage 2's 0.2 gap to the 0.85 success criterion is a Kokoro limitation, a
fine-tuning-recipe limitation, or a ceiling of the metric itself, and potentially restate the
project's success criterion (project `description.md`) accordingly.

## Category Selection Rationale

`meta/categories/` in this project contains no defined categories (the directory holds only a
`.gitkeep`; `uv run python -u -m arf.scripts.aggregators.aggregate_categories --format json` returns
`{"categories": []}`). Phase 2 of the research-papers procedure therefore had no categories to
select from or exclude — the project has not yet adopted a category taxonomy for its paper corpus,
and every downloaded paper's `categories` field is likewise empty (confirmed via the paper
aggregator's short-detail output). Rather than fabricate a category selection that the project's
metadata does not support, this research proceeded directly to the unfiltered paper aggregator run
(`aggregate_papers --format json --detail short`, no `--categories` flag), which is also the second,
mandatory step of Phase 3 regardless of category filtering, and is sufficient here because it
already returns the entire corpus (4 papers, all added by `t0014_v11_decoder_fix_retrain`). No
category-based exclusions were possible or necessary.

## Key Findings

### Vocoder Quality Traces to the Discriminator, Not the Generator

Three of the four papers in the corpus are vocoder papers built around the same core empirical
claim, from different angles. [You2021] holds the discriminator fixed (HiFi-GAN's multi-resolution
discriminating framework: a Multi-Period Discriminator, MPD, plus a Multi-Scale Discriminator, MSD)
and swaps in six structurally different generators — HiFi-GAN, MelGAN, Parallel WaveGAN, Universal
MelGAN, VocGAN, and a novel axial-residual design — with parameter counts spanning **928,514** to
**93,077,506** (roughly two orders of magnitude). Three of four evaluation metrics (MOS on
ground-truth mels, MOS on Tacotron-2-synthesized mels, and MCD on TTS mels) show no statistically
resolvable difference among the six generators; only MCD on ground-truth reconstruction separates
them, and not in proportion to parameter count. [Kong2020]'s own ablation on HiFi-GAN corroborates
this from the opposite direction: removing MPD from the discriminator drops MOS from **4.10 to
2.28** (a **1.82**-point collapse), by far the largest of four ablated components — larger than
removing MSD (**3.74**, a 0.36-point drop), removing the Multi-Receptive-Field-Fusion generator
module (**3.92**), or removing the mel-spectrogram reconstruction loss (**3.25**)
[Kong2020, Table 2, p. 7]. [Kong2020] also shows that adding MPD to an unrelated generator (MelGAN)
improves its MOS by **+0.47** (2.88 → 3.35) [Kong2020, Table 2, p. 7], evidence the effect is
discriminator-driven rather than tied to HiFi-GAN's specific generator.

[Kaneko2022] extends this picture with a speed/quality trade-off: replacing a HiFi-GAN generator's
final output-side layers with a closed-form inverse short-time Fourier transform (iSTFT) can match
or slightly exceed baseline quality while running faster, but only up to an architecture-dependent
point. For the heaviest variant (V1, 13.94M params, baseline MOS 4.22), a moderate-depth replacement
(`V1-C8C8I`) reaches **MOS 4.26 ± 0.17** at **245.68x real-time on GPU** (1.71x the unmodified V1's
143.59x) [Kaneko2022, Table 1, p. 3], while over-replacement (`V1-C8I`) collapses quality to **MOS
3.32 ± 0.22** despite reaching 609.43x real-time. The already-minimal V3 configuration (1.46M
params, baseline MOS 3.78) tolerates almost no replacement before quality drops (mildest tested
variant: MOS 3.41) — the benefit is architecture- and capacity-dependent, not universal
[Kaneko2022, Table 1, p. 3]. Independently, [Kong2020] reports HiFi-GAN V1 running at **3,701x
real-time** on a single V100 GPU and its smallest variant V3 at **1,186.80x real-time** GPU /
**13.44x real-time CPU** [Kong2020, Table 1, p. 6] (note: this figure is measured on different
hardware/methodology than [Kaneko2022]'s "143.59x" baseline for the same V1 model, so the two
numbers are not directly comparable to each other, only within each paper).

**Cross-cutting implication for this task**: across all three papers, the vocoder/decoder stage of a
HiFi-GAN-family pipeline runs at two to three orders of magnitude faster than real-time on GPU, and
its perceptual quality is governed primarily by discriminator design rather than generator
architecture or size. None of these papers describes F5-TTS, CosyVoice 2, or Chatterbox directly
(see Gaps below), so this is background, not a direct claim about those systems — but it is a
literature-grounded reason to expect that if any of the three cloning systems misses the TTFB ≤ 300
ms bar, the bottleneck is far more likely to be the autoregressive/flow-matching backbone that
produces the acoustic representation than the final waveform-decoding step.

### Zero-Shot Cloning Trades Data Scale for Similarity

[Li2023] (StyleTTS 2) is architecturally the closest paper in the corpus to this task's subject
matter, because it is the literal basis of `kokoro_v3_bundle` — the second baseline this task
re-runs — and because it reports its own zero-shot speaker-adaptation results. StyleTTS 2 models
style as a diffusion-sampled latent variable conditioned on text alone by default, but explicitly
supports optional reference-audio conditioning for voice-transfer/cloning use cases
[Li2023, p. 1-2]. In its zero-shot evaluation on LibriTTS test-clean (3-second reference clips),
StyleTTS 2 beats Vall-E in naturalness (**CMOS +0.67**, p < 1e-3) but trails it in similarity
(**CMOS-S -0.47**, p < 1e-3), while using only **245 hours** of training data versus Vall-E's
**60,000 hours** — a **250x** data-efficiency advantage
[Li2023, Table 1 and accompanying text, p. 8]. This establishes, within the corpus, a documented
precedent that a lower-resource zero-shot/style-based approach can sacrifice speaker similarity for
naturalness and data efficiency relative to a larger-scale system — directly relevant to
interpreting why any of F5-TTS/CosyVoice 2/Chatterbox (all trained on large public multi-speaker
corpora, per `task_description.md`'s systems list) may outperform `kokoro_v3_bundle` (a
small-corpus, per-speaker StyleTTS2 fine-tune, not a true zero-shot model) on `speaker_sim`,
addressing this task's Key Question 2 directly.

Critically, StyleTTS 2's own zero-shot similarity number is a **human-judged comparative MOS
(CMOS-S)**, not a GE2E cosine embedding score — the metric this project uses throughout
(`speaker_sim` vs. the ElevenLabs half-B centroid). These two metrics are not numerically
comparable; CMOS-S measures perceived similarity relative to a reference recording as rated by
listeners, whereas GE2E cosine measures embedding-space distance between a speaker-verification
network's representations of synthesized and reference audio. [Li2023] provides no GE2E-cosine
number that could be used as a direct data point for this task's speaker_sim tables.

### The StyleTTS2/Kokoro Training-Scale Gap

[Li2023]'s own multi-speaker (VCTK) and zero-shot (LibriTTS) configurations were trained on
**43,470** and roughly **233,000** train-split utterances respectively (**~44** and **~245** hours)
[Li2023, p. 6-7] — two to three orders of magnitude larger than this project's 1,531-clip David
corpus used for Kokoro Stage 2 fine-tuning. This corroborates, from the zero-shot-adaptation
literature, why a project the size of this one's David dataset would be expected to sit well below a
genuinely zero-shot, large-corpus-trained system's similarity ceiling — reinforcing the motivation
this task states directly: to measure that ceiling empirically with systems that were never trained
on David at all, rather than continuing to extrapolate from small-corpus fine-tuning results alone.

## Methodology Insights

* **Instrument TTFB by pipeline stage, not just end-to-end**, when interpreting Key Question 3 (TTFB
  ≤ 300 ms). The vocoder-quality literature in this corpus consistently shows HiFi-GAN-family
  decoders running at hundreds to thousands of times real-time on GPU
  [Kong2020, Table 1, p. 6; Kaneko2022, Table 1, p. 3], so if any of F5-TTS, CosyVoice 2, or
  Chatterbox misses the 300 ms TTFB bar, the most literature-consistent first hypothesis is a slow
  backbone (autoregressive decoding or multi-step flow-matching sampling), not the terminal
  waveform-decoding step. This is a hypothesis to test with per-stage timing instrumentation during
  the benchmark run, not an assumption to bake into the harness.

* **Do not expect a literature-calibrated answer for reference-clip duration** (Key Question 4,
  `ref_single` ~10 s vs. `ref_concat` ~30 s). [Li2023]'s own zero-shot evaluation uses fixed
  3-second reference clips [Li2023, p. 6] and does not ablate reference duration; none of the four
  reviewed papers test how reference-clip length affects cloning fidelity. This task's own
  `ref_single`/`ref_concat` comparison should be treated and reported as a first-of-its-kind
  empirical measurement for this project rather than a replication of a published result.

* **Use [Kong2020]'s MPD ablation as a diagnostic checklist item, not a tuning target.** If any
  cloning system's audio-quality gate (`audio_quality_check.py`) flags a disproportionate share of
  clips from one system, and that system is known to use a HiFi-GAN-family vocoder internally, the
  literature's strongest single quality lever is discriminator design (MPD presence/absence cost
  1.82 MOS points, the largest of four ablated components) [Kong2020, Table 2, p. 7] — useful
  context for writing up *why* a gate-failure pattern might exist, even though this task performs no
  fine-tuning or architecture changes.

* **Report `speaker_sim` (GE2E cosine) results without attempting numeric comparison to any
  CMOS/CMOS-S figures from [Li2023].** The two metric families (embedding-cosine vs. human
  comparative-MOS) are not interchangeable, and the corpus contains no paper establishing a
  conversion or correlation between them. Use [Li2023]'s CMOS-S results (StyleTTS 2 trailing Vall-E
  by 0.47 CMOS-S despite 250x less training data) only as qualitative framing — e.g., as
  literature-external context for why a small-corpus, style-diffusion-based system might land below
  a large-corpus zero-shot system on similarity — never as a benchmark table entry.

* **Best practice, cross-paper agreement**: both [You2021] and [Kong2020] independently converge on
  discriminator design (specifically multi-period/multi-resolution discrimination) as the dominant
  driver of neural-vocoder perceptual quality, largely independent of generator parameter count
  [You2021, full study; Kong2020, Table 2, p. 7]. This is a community-converged best practice for
  vocoder-quality debugging in general, applicable as background if this task's results need to be
  explained in terms of architecture rather than data or training scale.

## Gaps and Limitations

* **No paper in the corpus describes F5-TTS, CosyVoice 2, or Chatterbox** — the three systems this
  task actually benchmarks. All four downloaded papers were added by a prior, unrelated task
  (`t0014_v11_decoder_fix_retrain`, a Kokoro decoder-architecture debugging task) and their
  relevance here is incidental (shared vocoder lineage, shared StyleTTS2 ancestry with
  `kokoro_v3_bundle`) rather than purpose-selected for this task. This is a genuine, unfilled gap:
  none of the three target systems' published architectures, training data scale, or reported
  speaker-similarity/TTFB numbers are available in this project's paper corpus. Per the
  research-papers skill's scope, downloading new papers is a separate step; if the planning stage
  wants literature grounding for F5-TTS (Chen et al.), CosyVoice 2 (Du et al.), or Chatterbox
  specifically, that requires a dedicated `add-paper`/`download_paper` step, not this research pass.

* **No paper in the corpus documents GE2E speaker-similarity evaluation methodology.** This task's
  core metric (`speaker_sim`, GE2E cosine vs. a half-B centroid) has no grounding source in the
  corpus — for example, the original GE2E loss paper (Wan et al., "Generalized End-to-End Loss for
  Speaker Verification") is not present. The corpus therefore cannot answer questions like expected
  cosine-similarity ranges for genuinely different speakers, centroid-construction sensitivity, or
  known failure modes of GE2E-based similarity scoring — all of which would be directly useful for
  interpreting whether an observed speaker_sim ceiling reflects the models or the metric.

* **Metric incompatibility limits cross-paper quantitative comparison.** [Li2023]'s only
  zero-shot-similarity numbers are human CMOS-S ratings, not GE2E cosine scores, so even the one
  paper in the corpus that reports zero-shot speaker-similarity results cannot supply a directly
  comparable number for this task's tables.

* **No reference-duration ablation exists anywhere in the corpus** (see Methodology Insights) — the
  `ref_single` vs. `ref_concat` question (Key Question 4) is unanswerable from existing literature
  in this corpus and must be resolved empirically by this task's own run.

* **Vocoder-speed figures are not necessarily representative of the three target systems'
  bottlenecks.** [Kong2020] and [Kaneko2022]'s real-time-factor numbers describe HiFi-GAN-family
  vocoders specifically; F5-TTS is flow-matching-based and CosyVoice 2 combines an LLM stage with
  flow-matching, so their end-to-end latency profiles may not be dominated by a HiFi-GAN-style
  decoder at all. The "vocoder is fast, backbone is the likely bottleneck" framing in Methodology
  Insights is a hypothesis grounded in adjacent literature, not a documented fact about these three
  specific systems.

## Recommendations for This Task

1. **Treat the corpus gap as expected, not a blocker.** Per `task_description.md`, this is
   explicitly a measurement task; proceed to planning and execution using the t0008 harness protocol
   without waiting for system-specific papers on F5-TTS/CosyVoice 2/Chatterbox to be downloaded.
   Flag, for a possible follow-up task, that a dedicated paper-download pass (F5-TTS, CosyVoice 2,
   Chatterbox technical reports, and the GE2E loss paper) would strengthen future literature
   grounding for this line of work.

2. **When analyzing TTFB against the 300 ms bar (Key Question 3), instrument and report
   backbone-vs-decoder timing breakdowns where the harness allows it**, since the corpus's vocoder
   literature consistently shows decoder-stage inference running two to three orders of magnitude
   faster than real time [Kong2020, Table 1, p. 6; Kaneko2022, Table 1, p. 3] — a missed TTFB target
   is more likely attributable to the backbone than the vocoder, and this framing should guide how
   results are written up, not just measured.

3. **Do not use [Li2023]'s CMOS-S zero-shot numbers as quantitative benchmark entries.** Report them
   only as qualitative literature context (StyleTTS 2 trailed Vall-E by 0.47 CMOS-S despite 250x
   less training data) when discussing why `kokoro_v3_bundle` might underperform genuinely
   zero-shot, large-corpus systems on `speaker_sim` — never merge CMOS-S and GE2E-cosine numbers in
   the same table.

4. **Report the `ref_single`/`ref_concat` comparison (Key Question 4) as a novel measurement**,
   explicitly noting in `results_detailed.md` that no literature precedent exists in the reviewed
   corpus for expected effects of reference-clip duration on zero-shot cloning fidelity.

5. **If any cloning system shows unusually high audio-quality gate failures traceable to
   vocoder-level artifacts, cite [Kong2020]'s MPD ablation as background** for why discriminator
   design (not generator size or vocoder architecture family) is the literature-established primary
   quality lever — useful narrative framing for the results report, not an action this task can take
   (no fine-tuning is permitted here).

6. **Flag the missing GE2E-methodology and system-specific papers explicitly in
   `results/suggestions.json`** as a candidate follow-up research task, since this gap will recur
   for any future task that needs literature grounding for the GE2E metric itself or for these three
   specific cloning systems.

## Paper Index

### [You2021]

* **Title**: GAN Vocoder: Multi-Resolution Discriminator Is All You Need
* **Authors**: You, J., Kim, D., Nam, G., Hwang, G., Chae, G.
* **Year**: 2021
* **DOI**: `10.21437/Interspeech.2021-41`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.21437_Interspeech.2021-41/`
* **Categories**: none (project has no categories defined; see Category Selection Rationale above)
* **Relevance**: Establishes that GAN-vocoder perceptual quality is driven by the multi-resolution
  discriminating framework rather than generator architecture or parameter count (928K-93M params
  produced statistically indistinguishable MOS). Background for interpreting whether any
  vocoder-attributable quality issues observed in the cloning systems trace to discriminator design
  rather than model scale.

### [Kong2020]

* **Title**: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech
  Synthesis
* **Authors**: Kong, J., Kim, J., Bae, J.
* **Year**: 2020
* **DOI**: `10.48550/arXiv.2010.05646`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2010.05646/`
* **Categories**: none (project has no categories defined; see Category Selection Rationale above)
* **Relevance**: Source of the dominant open-weight neural vocoder family; its real-time-factor
  numbers (up to 3,701x real-time on GPU) and MPD ablation (1.82 MOS-point effect, the largest of
  four ablated components) ground this task's hypothesis that any TTFB shortfall in the cloning
  systems is more likely a backbone than a vocoder bottleneck.

### [Kaneko2022]

* **Title**: iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time
  Fourier Transform
* **Authors**: Kaneko, T., Tanaka, K., Kameoka, H., Seki, S.
* **Year**: 2022
* **DOI**: `10.48550/arXiv.2203.02395`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2203.02395/`
* **Categories**: none (project has no categories defined; see Category Selection Rationale above)
* **Relevance**: Demonstrates an architecture-dependent speed/quality trade-off in HiFi-GAN-family
  decoders (e.g., V1-C8C8I: 245.68x real-time GPU at MOS 4.26 vs. baseline V1's 143.59x), further
  evidence that decoder-stage inference is fast relative to whole-utterance TTS latency, informing
  TTFB-bottleneck attribution for this task's Key Question 3.

### [Li2023]

* **Title**: StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial
  Training with Large Speech Language Models
* **Authors**: Li, Y. A., Han, C., Raghavan, V. S., Mischler, G., Mesgarani, N.
* **Year**: 2023
* **DOI**: `10.48550/arXiv.2306.07691`
* **Asset**: `tasks/t0014_v11_decoder_fix_retrain/assets/paper/10.48550_arXiv.2306.07691/`
* **Categories**: none (project has no categories defined; see Category Selection Rationale above)
* **Relevance**: The architecture underlying `kokoro_v3_bundle`, one of the two baselines re-scored
  in this task, and the only paper in the corpus reporting zero-shot speaker-adaptation results
  (CMOS-S -0.47 vs. Vall-E, using 250x less training data). Provides literature precedent for why a
  small-corpus fine-tune may trail large-corpus zero-shot systems on similarity, directly relevant
  to Key Question 2, while its use of a different similarity metric (CMOS-S, not GE2E cosine) limits
  direct numeric comparison.
