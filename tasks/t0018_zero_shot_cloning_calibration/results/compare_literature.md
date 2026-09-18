---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
date_compared: "2026-09-17"
---
# Comparison with Published Results

## Summary

This task measured `speaker_sim` (GE2E cosine, `resemblyzer`, vs. an ElevenLabs-David half-A
centroid) for CosyVoice2 (`ref_single`: **0.863** val96 / **0.842** fillers) and Chatterbox
(**0.807-0.810** across its three successful variants) and compares those numbers, plus WER, against
the closest published results for the same three named systems: [Chen2024] (F5-TTS), [Du2024]
(CosyVoice2), and [Seo2026] (Chatterbox-Flash, which reports the base autoregressive Chatterbox as
its own baseline row). **The headline finding of this comparison is that no numeric row below is an
apples-to-apples measurement**: every published similarity number uses a WavLM-large, ERes2Net, or
WavLM-ECAPA-TDNN speaker-verification embedding on public English/Mandarin ASR corpora (LibriSpeech,
Seed-TTS/SEED), whereas this project's `speaker_sim` is a GE2E-trained `resemblyzer` cosine score on
the private ElevenLabs-David voice-commerce corpus. The two families of numbers happen to sit on a
comparable **0-1 cosine scale**, which makes a numeric "Delta" column computable, but that delta
reflects a different embedding model measuring a different speaker on different text, not a
model-quality gap — this table must be read as an order-of-magnitude/qualitative comparison only,
never a ranking. F5-TTS itself produced zero data in this task's own run (indefinite model-loading
hang, three attempts), so every F5-TTS row below has "—" for Our Value; only the corresponding
published SIM-o numbers from [Chen2024] are reported, as a target this task could not reach this
session, not as a result to be ranked against.

## Comparison Table

| Method / Paper | Metric | Published Value | Our Value | Delta | Notes |
| --- | --- | --- | --- | --- | --- |
| F5-TTS ([Chen2024, Table 1]) | SIM-o (WavLM-large cosine) | 0.66 | — | — | **METRIC MISMATCH + NO DATA.** LibriSpeech-PC test-clean, 32 NFE, WavLM-large speaker-verification embedding — not GE2E-cosine. In addition, F5-TTS never produced a usable clip this session (indefinite hang in model loading, 3 attempts, `intervention/f5_tts_smoke_gate_failed.md`), so there is no "Our Value" to compare regardless of metric compatibility. Do not read this row as a gap this task measured — it is a target this task could not reach a measurement for. |
| F5-TTS ([Chen2024, Table 2]) | SIM-o (WavLM-large cosine) | 0.76 | — | — | **METRIC MISMATCH + NO DATA.** Seed-TTS test-zh (Mandarin speakers), 32 NFE. Same WavLM-large embedding and same F5-TTS null-data caveat as the row above; additionally a different language/speaker population than David (English). |
| CosyVoice2 ([Du2024, Table 5]) | SS (ERes2Net cosine) | 0.745 | 0.863 (`ref_single`, val96) | +0.118 | **METRIC MISMATCH.** Published SS is LibriSpeech test-clean, ERes2Net speaker-verification embedding, public-corpus English speakers. Our value is GE2E-cosine on the private David voice via `resemblyzer` vs. an ElevenLabs half-A centroid. The **+0.118** delta is an arithmetic difference between two different embedding models scoring two entirely different speakers/corpora — it must not be read as "our CosyVoice2 run is 0.118 better than the published one." |
| CosyVoice2 ([Du2024, Table 6]) | SS (ERes2Net cosine) | 0.806 | 0.863 (`ref_single`, val96) | +0.057 | **METRIC MISMATCH.** Published SS is SEED test-zh (Mandarin speakers), ERes2Net embedding — the higher of [Du2024]'s two reported SS variants for this test set (WavLM variant is 0.748). Same order-of-magnitude-only caveat as above; language/speaker population also differs. |
| Chatterbox ([Seo2026, Table 1]) | SIM-o (WavLM-ECAPA-TDNN cosine) | 0.707 | 0.807 (`ref_single`, val96) | +0.100 | **METRIC MISMATCH.** Published row is [Seo2026]'s own re-benchmark of the base autoregressive Chatterbox (the same open-weight `ResembleAI/chatterbox` checkpoint this task uses) on LibriSpeech-PC test-clean, WavLM-ECAPA-TDNN embedding. This is the closest published number to a "same system" comparison in this table, but the embedding backbone and corpus/speaker population still differ from our GE2E/David setup, so the **+0.100** delta is qualitative context, not a validated quality gap. |
| Chatterbox ([Seo2026, Table 1]) | SIM-o (WavLM-ECAPA-TDNN cosine) | 0.685 | 0.807 (`ref_single`, val96) | +0.122 | **METRIC MISMATCH.** Same base-Chatterbox row as above, Seed-TTS test-en split instead of LibriSpeech-PC. Same caveats apply. |
| Chatterbox ([Seo2026, Table 1]) | WER (%) | 1.99 | 41.01 (`ref_single`, val96, `faster-whisper base.en`) | +39.02 | **METRIC MISMATCH, DIFFERENT DIRECTION OF "BETTER," AND DIFFERENT TEXT DOMAIN.** [Seo2026] scores WER with a Hubert-large ASR on curated LibriSpeech-PC sentences. This task scores WER with `faster-whisper base.en` on voice-commerce filler prompts that include brand names, numbers, and alphanumeric session/response IDs (e.g. `"llm sess a5158e64865142c3 resp c73cb5ba73e648b4"`, `results/results_detailed.md` Example 4) — text that is inherently far more ASR-adversarial than natural LibriSpeech sentences. Unlike the SIM/SS rows, for WER **lower is better**, so the "positive = this task outperforms" Delta convention is inverted here: **+39.02** means this task's measured WER is far worse, not better. This gap is very likely dominated by ASR-model size (`base.en` vs. Hubert-large) and text-domain difficulty, not by Chatterbox audio quality — see `## Analysis`. |
| CosyVoice2 ([Du2024, Table 5]) | WER (%) | 2.47 | — | — | **NO DATA, DOCUMENTED GAP.** This task's CosyVoice2 WER is `null` in `results/tables.json` because t0008's duration-ratio-before-WER gate skips WER scoring on CosyVoice2's frequently duration-inflated clips (`duration_ratio_median` 2.76-4.13 for CosyVoice2 vs. 0.86-1.60 for Chatterbox); see `results/results_detailed.md` Limitations. Even if it had been computed, the same ASR-model and text-domain mismatch noted in the Chatterbox WER row above would apply. |

## Methodology Differences

* **Speaker-embedding backbone**: this task's `speaker_sim` uses a GE2E-trained `resemblyzer`
  encoder (per `tasks/t0008_tts_eval_harness_baselines/code/scoring.py`, itself grounded in
  [Wan2018]'s original GE2E loss). [Chen2024] and [Seo2026] score SIM-o with a WavLM-large or
  WavLM-ECAPA-TDNN verification model; [Du2024] reports SS with either an ERes2Net or a WavLM-based
  model (and explicitly notes in its own Section 4.2 that the two do not agree with each other, e.g.
  0.806 ERes2Net vs. 0.748 WavLM for the same SEED test-zh condition — a caution this task's
  research already flagged in `research/research_internet.md`). No paper in the corpus reports a
  GE2E-cosine number directly comparable to this project's `speaker_sim`.
* **Reference speaker and corpus**: the published numbers score public multi-speaker corpora
  (LibriSpeech-PC test-clean, Seed-TTS/SEED test-en/test-zh) averaged across many held-out speakers.
  This task scores a single voice (ElevenLabs David) against itself/its own clones, using a
  purpose-built 196-prompt voice-commerce filler-text set, not a general-purpose ASR benchmark.
* **Reference-audio duration and construction**: [Chen2024] and [Du2024] use short (3-10 s), single
  natural utterances as cloning prompts. This task's `ref_single` (~10.8 s) and `ref_concat` (~30.6
  s) are both concatenations of 10 short half-A clips, not single natural utterances (see
  `results/results_detailed.md` `## Limitations`/`## Analysis` — the David corpus has no single clip
  near 10 s). No paper in the corpus ablates reference-clip duration at all
  (`research/research_papers.md`), so `ref_single` vs. `ref_concat` is a first-of-its-kind
  measurement for this project, not a literature replication.
* **WER scorer and text domain**: [Seo2026]/[Chen2024]/[Du2024] use Whisper-large-v3, HuBERT-large,
  or Paraformer ASR on curated, natural-sentence test sets. This task uses `faster-whisper base.en`
  (a materially smaller ASR model) on voice-commerce filler text containing brand names and
  alphanumeric session IDs — both factors independently inflate WER relative to the published
  numbers, unrelated to underlying TTS audio quality.
* **F5-TTS was never measured on this task's own harness.** F5-TTS's model loading hung indefinitely
  across three attempts in this session (`intervention/f5_tts_smoke_gate_failed.md`), so this
  comparison can only place [Chen2024]'s own published SIM-o numbers alongside this task's other
  (measured) systems — it cannot report an "our F5-TTS" data point at all this session.
* **CosyVoice2's `ref_concat` condition produced no data** (0/196 successful, a genuine ≤30 s
  reference-audio limit in CosyVoice2's own code, `cosyvoice/cli/frontend.py`), so all CosyVoice2
  rows above use `ref_single` only; [Du2024]'s own published numbers use CosyVoice2's native
  short-prompt conditioning, not a 30 s-class reference at all.

## Analysis

Every row in the Comparison Table is qualified with a metric-mismatch flag, per this task's own
prior findings (`research/research_papers.md`, `research/research_internet.md`, and
`results/results_detailed.md` all independently establish that GE2E-cosine `speaker_sim` is not
numerically interchangeable with WavLM/ERes2Net-based SIM-o/SS, CMOS-S, or MOS/preference-%). This
comparison does not overturn that finding; it extends it with three more data points ([Chen2024],
[Du2024], [Seo2026]) and confirms the pattern holds even for the exact three systems this task
targets, not just for adjacent literature (StyleTTS 2's CMOS-S, previously the only in-corpus
precedent).

Read qualitatively rather than numerically, this task's own measured `speaker_sim` values
(**0.807-0.863** across the six successful cloning variants) land in a similar order of magnitude to
the published SIM-o/SS numbers for the same three systems (**0.66-0.76** for F5-TTS, **0.745-0.806**
for CosyVoice2, **0.685-0.707** for base Chatterbox) — none of the published numbers are close to
this project's `speaker_sim >= 0.85` success criterion either, which is a mild piece of
cross-validating context (both metric families place these systems in a broadly similar "good but
not perfect" cloning-fidelity band), but the **positive deltas in every SIM/SS row** (**+0.057** to
**+0.122**) must not be read as "this task's runs outperform the published papers" — a raw
GE2E-cosine score on one speaker's own voice-commerce corpus is not expected to sit on the same
numeric scale as a WavLM/ERes2Net cosine averaged across many held-out LibriSpeech/SEED speakers,
and the sign and magnitude of the gap could easily reverse under a shared scorer. This is exactly
the caution [Du2024] itself raises about cross-backbone SS comparisons (its own WavLM vs. ERes2Net
SS numbers disagree by **0.058** on the identical SEED test-zh condition, Table 6) — if even the
*same paper's* two embedding backbones disagree by that much on the identical audio, a cross-paper,
cross-corpus, cross-backbone GE2E-vs-WavLM/ERes2Net comparison is far less trustworthy still.

The one genuinely negative, honestly-reportable finding here is the Chatterbox WER row: this task's
measured WER (**41.01%**, `ref_single` val96) is dramatically higher than [Seo2026]'s published
**1.99%** for the identical open-weight Chatterbox checkpoint. Per the skill's "never omit negative
results" rule, this gap is reported plainly rather than smoothed over — but
`## Methodology Differences` above gives two concrete, literature-external reasons this is very
likely a measurement-methodology artifact rather than a real audio-quality regression: (1) this
task's ASR scorer (`faster-whisper base.en`) is a much smaller, lower-accuracy model than
[Seo2026]'s Whisper-large-v3/HuBERT-large, and (2) this task's prompt text (brand names, numeric
session/response IDs) is categorically harder to transcribe than LibriSpeech's or Seed-TTS's natural
sentences. Both factors would inflate WER independent of Chatterbox's actual intelligibility, and
this task's own per-clip WER on brand-name vs. non-brand-name text (`results/results_detailed.md`,
Key Question 5) shows no elevated WER specifically on brand-name text, evidence against a
Chatterbox-specific pronunciation failure as the primary driver.

### Prior Task Comparison

`plan/plan.md` (REQ-18, and the motivation section of `task_description.md`) explicitly cites
`t0008_tts_eval_harness_baselines`'s numbers as the project's calibration baseline: ElevenLabs David
self-consistency `speaker_sim` of **0.832** (fillers) / **0.792** (val96), and `kokoro_v3_bundle` at
**0.631** (fillers) / **0.588** (val96). This task re-measured `elevenlabs_david` fresh in the same
session (`results/tables.json`, `elevenlabs_david_fillers`/`elevenlabs_david_val96` variants) and
got **0.8324875295162201** / **0.7923378584285578** — matching t0008's cited **0.832**/**0.792** to
within rounding, confirming t0008's baseline rather than contradicting it. `kokoro_v3_bundle` was
not re-measured this session (identical indefinite-hang failure signature to F5-TTS,
`intervention/kokoro_v3_bundle_not_remeasured.md`) and is carried forward unchanged from t0008 at
**0.631**/**0.588**.

The finding that *does* contradict an implicit prior-task assumption is this: t0008's own framing
(echoed in `task_description.md`'s Motivation section) treated the **0.85** GE2E-cosine success
criterion as sitting *above* the reference voice's own self-consistency ceiling (0.792-0.832),
implying it might be structurally unreachable by any system scored the same way. This task's
CosyVoice2 `ref_single` result (**0.8627852474649748** on val96) clears that ceiling and comes
within **0.0072852474649748** — well under 1 percentage point — of the literal 0.85 bar, using a
zero-shot system with no David-specific training at all. This directly contradicts the implicit
prior assumption that 0.85 was unreachable in principle under this project's own GE2E-cosine scoring
methodology; `results/results_detailed.md`'s Key Question 6 already recommends restating the success
criterion for exactly this reason, and this comparison confirms that recommendation is not an
artifact of a metric change — it uses the identical `speaker_sim` GE2E-cosine measure t0008 itself
defined.

## Limitations

* **No paper in the corpus reports GE2E-cosine SIM/SS numbers for F5-TTS, CosyVoice2, or
  Chatterbox** — every comparison row above is cross-metric by necessity, not by choice. This
  limitation was already anticipated in `research/research_papers.md` and
  `research/research_internet.md` before this comparison was written, and this file's Comparison
  Table and Analysis sections operationalize that warning rather than discover it fresh.
* **F5-TTS has zero measured data points from this task**, so both F5-TTS rows in the Comparison
  Table report only the published side; there is no way to compute even a metric-mismatched delta
  for F5-TTS specifically. The published SIM-o numbers ([Chen2024, Table 1] and [Chen2024, Table 2])
  are included only as the target this task's environment failed to reach a measurement for, per
  `intervention/f5_tts_smoke_gate_failed.md`.
* **CosyVoice2's `ref_concat` condition has zero measured data points** (0/196, a genuine ≤30 s
  reference-audio limit in CosyVoice2's own code), so all CosyVoice2 comparison rows use
  `ref_single` only; no published number in the corpus specifically addresses CosyVoice2 behavior
  with reference audio near 30 s.
* **[Seo2026] (Chatterbox-Flash) is a third-party paper, not an official Chatterbox technical
  report** — Chatterbox itself has no published research paper, confirmed directly by its maintainer
  (`research/research_internet.md`, `HF-Chatterbox-NoPaper-Discussion21`). [Seo2026]'s Table 1 base
  autoregressive Chatterbox row is used here because it is the only in-corpus paper that
  independently re-benchmarks the exact open-weight `ResembleAI/chatterbox` checkpoint this task
  uses, but it was not authored by Chatterbox's own maintainers.
* **The Podonos blind AB-test finding** (**63.75%** preferred Chatterbox over ElevenLabs,
  `research/research_internet.md`) is not included as a Comparison Table row because it is a vendor
  case-study web page, not a downloadable paper in this task's corpus with a citable table/figure —
  per the skill's citation rule, a value that cannot be traced to a specific table in a specific
  paper is omitted rather than estimated. It is noted here only as further qualitative context that
  independently-run comparisons of these systems tend to favor them over ElevenLabs on some axis,
  consistent with this task's own `speaker_sim` numbers clearing the ElevenLabs self-consistency
  ceiling.
* **Every numeric Delta in the Comparison Table is an arithmetic subtraction only.** None of them
  should be read as a validated measure of relative model quality, given the compounding differences
  in embedding backbone, speaker/corpus population, reference-audio protocol, and (for WER rows) ASR
  scorer and text domain documented in `## Methodology Differences` above. This is the single most
  important caveat governing this entire file, restated here for emphasis per this task's own
  explicit, repeated finding across `research/research_papers.md`, `research/research_internet.md`,
  and `results/results_detailed.md`.
* This comparison draws on one measurement session on one shared-pool Azure ML VM
  (`results/results_detailed.md` `## Limitations`); TTFB/RTF are not compared against literature
  here because none of the three published papers' latency numbers were measured on comparable
  hardware under this task's own protocol, and `results/results_detailed.md` already treats this
  task's own TTFB numbers as possibly hardware/session-specific.
