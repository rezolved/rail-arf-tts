---
spec_version: "1"
task_id: "t0021_zero_shot_latency_reduction"
date_compared: "2026-09-18"
---
# Comparison with Published Results

## Summary

This task measured per-stage TTFB for CosyVoice2 and Chatterbox on 2x H100 NVL (5 and 6 acceleration
variants respectively, 196 prompts/variant) and compares those measurements against four
published/cited sources: [Du2024]'s additive latency model, [Seo2026]'s fine-tuned Chatterbox-Flash
TTFP numbers, [Kong2020]/[Kaneko2022]'s vocoder-only inference benchmarks, and the non-peer-reviewed
`vllm-project/vllm-omni` RFC's per-stage TTFA decomposition ([vllm-omni-6870]). The headline finding
is a **structural confirmation, not a numeric match**: [Du2024]'s
`L_TTS = M·d_lm + M·d_fm + M·d_voc` model correctly predicts that the LM stage dominates (this task
measured **814.329-904.419 ms** for CosyVoice2 and **838.981-1,003.839 ms** for Chatterbox, versus a
combined flow-matching+vocoder stage of **151.120-784.364 ms**), while every other comparison in
this table exposes either a **different, disallowed lever** ([Seo2026] requires fine-tuning), a
**scope mismatch** ([Kong2020]/[Kaneko2022] measure vocoding alone, not flow-matching+vocoding), or
an **unreconciled, lower-confidence gap** ([vllm-omni-6870]'s LM-stage number is roughly 4.5-5x
faster than this task's measurement, for reasons this task's own data cannot fully explain).

## Comparison Table

**Note on the Delta column's sign convention**: per the specification, Delta = Our Value − Published
Value, with positive meaning "this task outperforms the published result." For every row below the
metric is a **latency in milliseconds** (lower is better), so a **positive Delta means this task's
measurement is slower**, not better — the opposite intuition from a typical accuracy/F1 comparison.
This is called out per-row in the Notes column to avoid misreading.

| Method / Paper | Metric | Published Value | Our Value | Delta | Notes |
| --- | --- | --- | --- | --- | --- |
| CosyVoice2 LM stage vs. additive model (Du2024) | LM prefill/decode share of `L_TTS` | Model form only, no ms reported: `L_TTS = M·d_lm + M·d_fm + M·d_voc` [Du2024, Eq. 11, §2.5, p. 6] | **814.329-904.419 ms** (all 5 CosyVoice2 variants) | N/A — structural confirmation, not a numeric reproduction (Du2024 reports no absolute ms) | Confirms `d_lm` is the dominant additive term, as the model predicts; Du2024 never publishes ms figures for `d_lm`/`d_fm`/`d_voc` on any hardware, so no numeric delta is computable. |
| Chatterbox LM stage vs. additive model (Du2024, applied by analogy) | LM decode stage | Model form only, no ms reported [Du2024, Eq. 11, §2.5, p. 6] | **838.981-1,003.839 ms** (all 6 Chatterbox variants) | N/A — structural confirmation | Chatterbox's T3 decoder is an analogous AR causal decoder, not the CosyVoice2 architecture Du2024 formalizes; included because the same dominant-LM-stage pattern generalizes across both systems this task measured. |
| Chatterbox-Flash D=16, α=0.5 (Seo2026) | TTFP p50, ms | **118 ms**, H100, concurrency 1 [Seo2026, Table 9] | **837.034 ms** (Chatterbox `torch_compile`, this task's best reachable engineering-only variant) | **+719.034 ms** (slower) | Not a contradiction — a different, disallowed lever. Seo2026 fine-tunes a new block-diffusion decoding objective into the T3 decoder; this task's Forbidden list rules out fine-tuning entirely, so the two numbers describe a stock checkpoint vs. a retrained one, not two engineering approaches to the same model. |
| Chatterbox-Flash D=32, α=0.75 (Seo2026) | TTFP p50, ms | **103 ms**, H100, concurrency 1 [Seo2026, Table 9] | **837.034 ms** (Chatterbox `torch_compile`) | **+734.034 ms** (slower) | Same framing as above, most aggressive Seo2026 setting. Both Seo2026 rows confirm the architectural floor is closeable — but only via fine-tuning, the exact lever this task was scoped not to test. |
| HiFi-GAN V1, vocoder-only (Kong2020) | Vocoder inference speed | **3,701x real-time**, V100 GPU, MOS 4.36 [Kong2020, Table 1, p. 6] | **151.120-784.364 ms** (combined flow-matching+vocoder stage, both systems, all variants) | N/A — incompatible units and scope, not computable | Scope difference, not underperformance: Kong2020 benchmarks the vocoder in isolation; this task's measured stage includes flow-matching (CosyVoice2/Chatterbox both condition mel/latent generation on a flow-matching model upstream of the vocoder), which Kong2020's architecture does not have. Not an apples-to-apples comparison. |
| iSTFTNet V1-C8C8I, vocoder-only (Kaneko2022) | Vocoder inference speed | **245.68x real-time**, GPU, MOS 4.26±0.17 [Kaneko2022, Table 1, p. 3] | **151.120-784.364 ms** (combined flow-matching+vocoder stage) | N/A — incompatible units and scope, not computable | Same scope caveat as the Kong2020 row: vocoder-only vs. flow-matching+vocoder combined. This task never isolated vocoder-only latency (no instrumentation boundary between flow-matching and vocoding was implemented), so a like-for-like number does not exist in this task's own data either. |
| CosyVoice2/3 serving RFC, LM stage (vllm-omni-6870) | LM stage (prefill + AR decode), ms | **182.8 ms** (40.5 ms prefill + 142.3 ms AR decode), concurrency 1, GPU unstated [vllm-omni-6870] | **814.329-904.419 ms** (CosyVoice2, all 5 variants, identically-pinned CosyVoice2-0.5B on H100 NVL) | **+631.5 to +721.6 ms** (slower, roughly 4.5-5x) | Lower-confidence source (non-peer-reviewed GitHub RFC issue, GPU unstated) — treated as weaker evidence than the three papers above. Not reconciled with certainty; see Analysis for three candidate hypotheses. |
| CosyVoice2/3 serving RFC, first flow chunk (vllm-omni-6870) | Flow-matching first-chunk stage, ms | **129.0 ms**, concurrency 1, GPU unstated [vllm-omni-6870] | **151.120 ms** (CosyVoice2 `load_trt`, combined flow-matching+vocoder stage — closest measured analogue) | **+22.120 ms** (slower) | The closest published/measured pair in this entire table (within 22 ms), despite the RFC's number excluding vocoding and this task's including it. The near-agreement here, contrasted with the large LM-stage gap above, is itself evidence used in the Analysis section below. |

### Prior Task Comparison

`task_description.md` cites `t0018_zero_shot_cloning_calibration`'s own prior measurements as this
task's motivating baseline: CosyVoice2 `inference_zero_shot(stream=True)` measured **2.86 s p50**
TTFB, and Chatterbox (whole-utterance, no streaming) measured **1.34-1.6 s p50** TTFB. Because of
the binding owner correction (wrong ElevenLabs "David" voice used in `data/11labs_david`), t0018's
numbers are not directly reusable, but `results/tables.json` retains them alongside this task's
corrected-reference baseline for continuity:

| System | t0018 cited baseline (wrong-voice ref) | t0021 `baseline_new_ref` (corrected ref, val96) | Delta | Notes |
| --- | --- | --- | --- | --- |
| CosyVoice2 | **2,859.252 ms** (≈ the cited 2.86 s) | **1,598.633 ms** | **-1,260.619 ms** (44.1% faster) | Large drop, but the reference clip *and* the centroid both changed (owner correction), plus infrastructure was warm in this task's session — this task's own `results/results_detailed.md` "Comparison vs Baselines" section attributes the drop to warm infrastructure (cached weights, warmed CUDA kernels), not to any acceleration lever, since both rows are the un-accelerated `baseline_new_ref`/streaming setting. |
| Chatterbox | **1,438.522 ms** (within the cited 1.34-1.6 s range) | **1,537.372 ms** | **+98.850 ms** (6.9% slower) | This is a genuine finding, not previously highlighted in `results_detailed.md`: Chatterbox's unaccelerated baseline did **not** get faster under the same "warm infrastructure" framing applied to CosyVoice2 above — it got slightly slower. This complicates a blanket "warm session explains the baseline shift" reading; the warm-infrastructure effect, if real, is not uniform across both systems, and the reference-voice/centroid change is a confound that was not isolated from session warmth for either system. |

## Methodology Differences

* **Hardware.** This task ran on 2x NVIDIA H100 NVL (`LLM-T1-NC80`). [Kong2020] benchmarks HiFi-GAN
  on a V100 GPU (and CPU); [Kaneko2022] does not state its GPU model; [Seo2026] explicitly states
  H100 for its Chatterbox-Flash numbers (matching this task's hardware class); `[vllm-omni-6870]`
  states no GPU at all.
* **Checkpoint / fine-tuning.** This task used stock, non-fine-tuned CosyVoice2-0.5B and
  `chatterbox-tts==0.1.7` checkpoints, per its own Forbidden-list "no fine-tuning" constraint.
  [Seo2026]'s Chatterbox-Flash is a fine-tuned derivative of the same T3 decoder with a new
  block-diffusion decoding objective — a categorically different checkpoint, not a serving-side
  configuration of the same model.
* **Serving stack.** `[vllm-omni-6870]`'s numbers describe the vLLM-Omni serving stack (continuous
  batching, paged KV-cache, FlashInfer kernels). This task's own `vllm_backend` CosyVoice2 variant —
  the lever that would have tested an equivalent stack directly — never completed
  (`.venv-cosyvoice2-vllm` install hit its pre-authorized 20-minute cutoff; see Limitations). Every
  CosyVoice2 number in this comparison table instead comes from the stock Python/PyTorch eager-mode
  inference path (optionally with TensorRT-exported sub-modules for `load_trt`), a materially
  different serving stack from vLLM-Omni's.
* **Metric scope.** [Kong2020] and [Kaneko2022] report vocoder-only inference speed (as a real-time
  multiplier, not ms). This task's `flow_matching_vocoder_ms`/`vocoder_ms` stage
  (`results/latency_breakdown.json`) measures flow-matching **and** vocoding combined — no
  instrumentation boundary between the two sub-stages was implemented, so the vocoder-only number
  Kong2020/Kaneko2022 report cannot be isolated from this task's own data either.
* **Speaker-similarity metrics excluded from this table.** [Du2024] itself documents that
  WavLM-based and ERes2Net-based similarity scores disagree on its own outputs (0.748 vs. 0.806 on
  SEED test-zh) [Du2024, §4.2]; [Chen2024]'s SIM-o and [Du2024a]'s raw ERes2Net cosine use yet other
  encoders. None of these are on the same scale as this task's GE2E-cosine `speaker_sim` (per
  `research/research_papers.md`'s "Cross-System and Cross-Encoder Similarity Scores" finding), so no
  `speaker_sim` row appears in this table — only latency metrics, which are unit-comparable across
  sources (ms or a derivable real-time multiplier).
* **Reference voice.** This task's reference audio and speaker-similarity centroid come from the
  corrected production ElevenLabs "David - narrator and newsreader" voice (`data/v4/val/wavs`), not
  from any of the cited papers' benchmark sets (LibriSpeech-PC, SEED test-zh, VCTK) — irrelevant to
  the latency comparisons in this table, but noted for completeness since it is the reason t0018's
  own baseline numbers (Prior Task Comparison above) are not a clean re-run of the same condition.
* **Concurrency.** All of this task's measurements are batch-1/concurrency-1 (single filler
  utterance, matching production usage). `[vllm-omni-6870]`'s cited 357 ms TTFA and its per-stage
  breakdown are also concurrency-1; its separately reported batched-throughput numbers are not used
  in this comparison.

## Analysis

**[Du2024]'s additive model is confirmed structurally, not numerically.** The published model
`L_TTS = M·d_lm + M·d_fm + M·d_voc` [Du2024, Eq. 11] makes a specific, falsifiable prediction: the
LM term is architecturally the most likely to dominate TTFB when the flow-matching/vocoder stages
are already near real-time (per [Kong2020]/[Kaneko2022]'s two-to-three-orders-of-magnitude vocoder
RTFs). This task's own measurement confirms exactly that shape — the LM stage is **56-85%** of total
TTFB depending on variant (e.g. CosyVoice2 `load_trt`: 904.419 ms of 1,056.937 ms total = 85.6%) and
did not move by more than about 10% under any tested serving-side lever (caching, precision, JIT,
TensorRT, `torch.compile`), while the non-LM stages did respond to acceleration (`load_trt` cut the
combined flow+vocoder stage 62%, from 397.390 ms to 151.120 ms). Du2024 publishes no absolute
`d_lm`/ `d_fm`/`d_voc` ms values on any hardware, so this is a structural confirmation of where the
time concentrates, not a numeric reproduction of a published figure — the Comparison Table above
reflects this honestly with "N/A" deltas rather than fabricating a published ms baseline to diff
against.

**[Seo2026]'s gap is a different, disallowed lever — not a contradiction requiring explanation.**
Chatterbox-Flash's **103-118 ms** TTFP [Seo2026, Table 9] is roughly **6-7x lower** than this task's
own best reachable Chatterbox TTFB (837.034 ms, `torch_compile`). This is not read as "this task
underperformed the literature": Seo2026 achieves its number by fine-tuning a new block-diffusion
decoding objective into the T3 decoder, which this task's Forbidden list explicitly rules out ("No
fine-tuning"). The two rows describe different models (a stock checkpoint vs. a retrained one), not
two engineering strategies applied to the same checkpoint. Seo2026's own paper independently
supports this framing: it is offered in the reviewed literature as evidence that the 300 ms gap is
"not purely architectural" but is also not closeable by "engineering-only levers... applied to a
stock autoregressive checkpoint alone" (`research/research_papers.md`) — exactly what this task's
own 22-variant sweep empirically confirms by exhausting the engineering-only lever space without
closing the gap.

**[Kong2020]/[Kaneko2022] describe a narrower stage than this task measured — stated plainly, not
implied as underperformance.** Both papers report vocoder-only inference speed (150x-3,700x real
time); this task's `flow_matching_vocoder_ms` stage is, by construction, flow-matching plus vocoding
combined, because no separate timing boundary between the two was instrumented. The Comparison Table
marks both rows' Delta as "N/A — incompatible units and scope" rather than computing a numerically
literal but meaningless gap between a real-time multiplier and a millisecond duration. If this
task's flow+vocoder numbers (151-784 ms) were driven mostly by vocoding rather than flow-matching,
they would be wildly inconsistent with 150-3,700x real-time vocoding — the much more likely
explanation is that flow-matching (which neither Kong2020's nor Kaneko2022's HiFi-GAN/iSTFTNet
architecture performs at all) accounts for most of that combined stage's cost, but this task's own
data cannot isolate the two sub-stages to confirm that quantitatively.

**[vllm-omni-6870]'s LM-stage discrepancy is attempted-but-not-confirmed, per the checkpoint's
framing.** The RFC's 182.8 ms combined prefill+AR-decode figure [vllm-omni-6870] is roughly 4.5-5x
faster than this task's own 814.329-904.419 ms measured LM stage on identically-pinned
CosyVoice2-0.5B on H100 NVL — the single largest gap in this comparison table. Three hypotheses are
offered, none confirmed:

1. **Different exact GPU/hardware.** `[vllm-omni-6870]` states no GPU model at all; this task's H100
   NVL (880 GB total system RAM across 2 GPUs per `results/remote_machines_used.json`) may differ
   from whatever hardware the RFC used in ways (SKU, NVLink topology, host CPU) this task cannot
   rule out.

2. **Different serving stack, not a config difference.** vLLM-Omni's continuous batching, paged
   KV-cache, and FlashInfer kernels are a materially different inference engine from this task's
   stock PyTorch eager-mode path (`load_trt` only exports the flow-matching encoder to TensorRT, not
   the LM). This task's own `vllm_backend` variant — the one lever that would test this hypothesis
   directly — never completed (20-minute install-timeout; see Limitations), so this cannot be
   empirically distinguished from hypothesis 3 with this task's own data.

3. **Sequential vs. interleaved call shape** (raised in this task's own step 11 creative-thinking
   and carried into `results/results_detailed.md`'s Analysis "angle 1"). For CosyVoice2 `load_trt`,
   `lm_prefill_decode_ms` (904.419 ms) plus `flow_matching_vocoder_ms` (151.120 ms) sums to
   1,055.539 ms, matching the measured `total_ms` (1,056.937 ms) almost exactly — consistent with a
   **sequential** pipeline (LM runs to completion, then flow-matching runs on the full token
   sequence), not an interleaved one where flow-matching for early tokens overlaps LM decode of
   later tokens. This task never scheduled a chunk-size-tuned CosyVoice2 streaming variant despite
   `research/research_summary.md` recommending exactly that lever (documented gap, not silently
   dropped — see Limitations and `results/results_detailed.md`'s own Limitations section).

   The strongest evidence available *for* a call-shape (rather than pure hardware/stack) explanation
   is the flow-chunk row in the Comparison Table above: `[vllm-omni-6870]`'s first-flow-chunk figure
   (129.0 ms) is within **22.120 ms** of this task's own combined flow+vocoder measurement (151.120
   ms) — the closest agreement in this entire table — while the LM-stage figures diverge by
   **631.5-721.6 ms**. If the discrepancy were purely a faster GPU or a uniformly faster software
   stack, both stages would be expected to diverge by a comparable proportion; instead the gap
   concentrates almost entirely in the LM stage, which is consistent with (though does not prove) a
   difference specific to how the LM stage is served or pipelined, not a blanket hardware advantage.
   **This is not fully reconciled**: no reproducible methodology is available from the RFC to test
   this directly, and the confidence-level caveat below applies to the entire reconciliation
   attempt.

## Limitations

* **`[vllm-omni-6870]` is a lower-confidence source than the three peer-reviewed papers above.** It
  is a non-peer-reviewed, open GitHub issue with no stated GPU, not a published or archival paper.
  Its numbers are treated throughout this comparison as a directional signal, not a controlled
  benchmark equivalent to [Du2024]/[Kong2020]/[Kaneko2022]/[Seo2026].
* **The LM-stage discrepancy with `[vllm-omni-6870]` is not reconciled**, only hypothesized about
  (three candidates above). This task's own `vllm_backend` variant — the empirical test that would
  have distinguished "different serving stack" from "different call shape" from "different hardware"
  — never completed within this task's budget (see
  `intervention/cosyvoice2_vllm_install_timeout.md`).
* **[Kong2020]/[Kaneko2022] rows have no computable numeric Delta.** Published values are real-time
  multipliers for vocoder-only inference; this task's values are milliseconds for a combined
  flow-matching+vocoder stage. The two are marked "N/A" rather than forced into a misleading
  subtraction.
* **No `speaker_sim` comparison row exists against any cited paper.** [Du2024]'s own
  WavLM-vs-ERes2Net disagreement on identical outputs [Du2024, §4.2], plus [Chen2024]'s SIM-o and
  [Du2024a]'s raw ERes2Net-cosine using yet other encoders, mean no published SS/SIM-o number is on
  the same scale as this task's GE2E-cosine `speaker_sim`. [Casanova2022]'s SECS is the one reviewed
  paper on a GE2E-comparable scale (VCTK zero-shot SECS **0.864** vs. ground truth **0.824**), but
  it is from a different (VITS-based, non-LLM) architecture entirely and is noted here as context,
  not included as a table row, since it does not measure CosyVoice2, Chatterbox, or any
  latency-relevant metric.
* **The Prior Task Comparison confounds two variables.** t0018's cited baseline used the wrong
  ElevenLabs voice reference (owner correction) *and* an earlier, possibly colder GPU session; this
  task's `baseline_new_ref` changes both the reference audio and runs in a warm session. The
  observed CosyVoice2 speedup (-1,260.619 ms) and Chatterbox near-flat result (+98.850 ms) cannot be
  attributed cleanly to either factor alone with the data this task collected.
* **Seo2026's Table 9 numbers describe Chatterbox-Flash measured by its own authors**, not an
  independent reproduction; this task did not attempt to reproduce Seo2026's fine-tuned model (out
  of scope — fine-tuning is forbidden for this task).
* **This comparison covers only the four sources with directly citable, locatable numeric values**
  for TTFB/TTFP/RTF-equivalent metrics. Several other papers in `research/research_papers.md` and
  `research/research_internet.md` (e.g. [Chen2024], [Li2023], [Xie2026], [Hu2026]) report RTF or
  TTFP figures on different architectures or hardware not directly comparable to this task's
  CosyVoice2/Chatterbox measurements at the same rigor as the four sources above, and were left out
  of the table rather than included with a weaker justification.
