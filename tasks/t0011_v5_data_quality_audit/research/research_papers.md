---
spec_version: "1"
task_id: "t0011_v5_data_quality_audit"
research_stage: "papers"
papers_reviewed: 0
papers_cited: 0
categories_consulted: []
date_completed: "2026-09-15"
status: "partial"
---
# Research Papers — t0011 v5 Data Quality Audit

## Task Objective

This task audits all 1557 v5 training clips for the Kokoro-82M fine-tuning project against six
quality dimensions: peak amplitude clipping (> −1 dBFS), pathological silence (> 30% of duration),
integrated loudness distribution (LUFS), duration outliers (< 1.5 s or > 15 s), sample-rate and
channel mismatches (expected: 24 kHz mono), and OOV / vocab-invalid tokens in transcripts. The audit
produces a cleaned train manifest (`data/train_list_v5_clean.txt`) and per-clip statistics
(`data/per_clip_stats.jsonl`) for safe use in the full-corpus Stage 2 run after t0010 confirms the
safeguards work. t0009's forensics identified missing audio quality metrics as a gap — a single
clipped or near-silent clip can produce NaN gradients that health gates stop but cannot prevent.

**Why `status: "partial"`**: The project paper corpus is empty (0 papers downloaded as of this
task). No papers have been retrieved via any prior task, so no paper summaries are available for
review. The key findings and methodology insights below are drawn from the author's domain knowledge
of audio quality standards, TTS data preparation practice, and loudness normalisation standards
(ITU-R BS.1770, EBU R128), rather than from a reviewed corpus. When the project acquires relevant
papers (e.g. StyleTTS2, EBU R128, pyloudnorm), this section should be revisited.

## Category Selection Rationale

No categories exist in `meta/categories/` (the directory contains only `.gitkeep`). The category
aggregator returned an empty list. Therefore no categories could be consulted and
`categories_consulted` is empty.

If categories were present, the following would be the natural candidates for this task:

* **audio-quality** — directly relevant to clipping, silence, and loudness metric computation.
* **tts-data-preparation** — covers best practices for cleaning speech corpora before fine-tuning.
* **loudness-normalisation** — covers ITU-R BS.1770 and EBU R128 integrated loudness standards.
* **speech-corpus-curation** — covers duration filtering, sample-rate validation, and OOV token
  analysis in TTS training datasets.

All other categories (e.g. transformer-models, NLP benchmarks, conversational AI) would be excluded
— they are unrelated to low-level audio signal analysis.

## Key Findings

### Clipping Damages TTS Model Training Disproportionately

Clipping (peak amplitude ≥ 0 dBFS, or near-clipping > −1 dBFS) introduces hard waveform
nonlinearities that corrupt the mel-spectrogram features used as reconstruction targets in
StyleTTS2. Because mel extraction operates on the log-power spectrum, a clipped waveform produces a
flat, distorted upper band. Models trained on clipped targets learn to reproduce the distortion
rather than natural phonation. A small number of heavily clipped clips (even 1–5 out of 1557) can
produce gradient spikes large enough to destabilise training, since the loss on those clips is
anomalously high. The standard industry threshold for flagging clips is > −1 dBFS peak — clips above
this threshold should be removed from training, not normalised, because normalisation cannot undo
the nonlinearity already baked into the waveform.

**Best practice**: Flag and remove clips with `peak_dbfs > −1`. Do not attempt to rescue them with
gain reduction — the waveform shape is already corrupted.

**Hypothesis**: The t0009 NaN gradient events may correlate with a small subset of heavily clipped
clips in the 1557-clip v5 corpus. If the cleaned manifest eliminates those clips, NaN frequency
should drop to zero even without the gradient-clipping safeguard added in t0010.

### Loudness Normalisation Standards and LUFS Thresholds

The ITU-R BS.1770-4 standard defines *integrated loudness* (LUFS — Loudness Units relative to Full
Scale) as the primary perceptual measure for broadcast audio. EBU R128 adopted −23 LUFS as the
target for broadcast delivery with a ±1 LU tolerance. For TTS training corpora, a tighter range is
appropriate because the model must learn a consistent loudness target: ElevenLabs and similar
production TTS systems typically normalise to −14 to −20 LUFS. Extreme outliers in the training
corpus — clips at < −30 LUFS (near-inaudible) or > −6 LUFS (severely over-loud) — force the model to
reconcile incompatible loudness targets, which manifests as inconsistent output loudness and
potentially as loss spikes on outlier clips.

The `pyloudnorm` library implements BS.1770-4 gating and is the standard Python tool for computing
integrated loudness. It applies a two-stage K-weighting filter followed by block-by-block gating
(400 ms blocks, 75% overlap, absolute gate at −70 LUFS, relative gate at −10 LU below ungated
loudness). For short clips (< 3 s), the relative gating stage may produce unreliable results because
there are insufficient gating blocks; in that case, falling back to unweighted RMS is acceptable.

**Best practice**: Flag clips outside −6 to −30 LUFS. Consider soft normalisation (not hard
clipping) for clips in the mild outlier range (−6 to −10 LUFS or −24 to −30 LUFS) rather than
removal, since loudness outliers corrupt training less severely than clipped waveforms.

### Duration Outliers and Mel-Extractor Stability

StyleTTS2's mel extractor allocates fixed-size feature buffers whose dimensions depend on clip
duration. Very short clips (< 1.5 s) may produce mel frames too sparse for the style encoder to
extract meaningful statistics. Very long clips (> 15 s) exceed the practical context window and can
cause memory spikes on batch assembly. Both extremes produce anomalous training dynamics.

Published TTS training datasets (LJSpeech, VCTK, LibriTTS-R) typically restrict clip duration to
1–15 s. LJSpeech clips have a mean duration of approximately 6.6 s with a standard deviation of 2.3
s. Clips outside the 1–15 s range are excluded from training in all major TTS baselines.

**Best practice**: Remove clips with `duration_s < 1.5` or `duration_s > 15.0`. For the v5 corpus,
also report the p5/p95/p99 distribution to identify whether the tail is a small fraction (expected)
or a systematic recording issue.

### Silence Detection and Phoneme-Rate Outliers

Silence detection in raw audio distinguishes two pathological cases: (1) clips that are
*predominantly silent* (e.g. a failed recording that captured room noise instead of speech), and (2)
clips with *long leading/trailing silence* that inflates measured duration without adding linguistic
content. Both cases are detected by computing the fraction of short-time frames with RMS energy
below a threshold (typically −60 dBFS).

A silence fraction > 30% of total clip duration is a practical heuristic used in production TTS
pipeline quality checks. Clips exceeding this threshold are either failed recordings or contain
anomalously long pauses that the model will learn to reproduce, inflating the synthesised pause
rate.

The phoneme-per-second rate is a complementary proxy for silence: a clip with 5 phonemes/s is likely
normal speech; a clip with 1 phoneme/s is either very slowly spoken or contains long pauses. t0009's
analysis used transcript character counts as a proxy; this task replaces that with actual per-clip
audio metrics, which is more reliable.

**Hypothesis**: Silence fraction and phoneme rate should be correlated. Clips that fail the silence
threshold (> 30%) should show proportionally low phoneme-per-second rates. Validating this
cross-correlation confirms that both metrics are identifying the same underlying pathology.

### OOV Token Analysis for TTS Transcripts

Out-of-vocabulary (OOV) tokens in TTS transcripts cause the phonemizer to fall back to
character-by-character pronunciation, which produces incorrect pronunciation for proper nouns,
abbreviations, and non-English words. t0003 already performed a corpus-wide OOV fix; this task
checks whether any residual OOV tokens remain in the v5 train transcripts after that fix, and
whether new OOV tokens were introduced in clips added since t0003.

The practical threshold for flagging a clip as OOV-contaminated is > 20% of tokens being OOV (i.e.
falling back to character pronunciation). Below this threshold the impact on training is marginal
because the affected phoneme sequences are short relative to the clip. Above 20%, a significant
fraction of the clip's target phoneme sequence is incorrect, which forces the model to learn
inconsistent mappings between text and audio.

## Methodology Insights

* **Peak clipping detection**: Compute `peak_dbfs = 20 * log10(abs(samples).max())` per clip using
  `librosa.load` (which normalises to float32 in [−1, 1]). The flag threshold is > −1 dBFS. No
  papers in the corpus to cite, but this formula directly implements the standard definition.

* **LUFS measurement**: Use `pyloudnorm.Meter(sample_rate).integrated_loudness(audio)`. For clips
  shorter than 3 s, fall back to `20 * log10(rms)` (unweighted RMS in dBFS) and mark those records
  with a `lufs_method: "rms_fallback"` field. Flag clips outside the range [−30, −6] LUFS.

* **Silence fraction**: Use `librosa.effects.split` with `top_db=60` to detect voiced segments.
  Compute `silence_fraction = 1 - (sum of voiced segment durations) / clip_duration`. Flag if
  silence fraction > 0.30.

* **Duration and format checks**: Read duration, sample rate, and channel count from audio file
  headers using `soundfile.info` (no decoding needed, fast for 1557 files). Flag if
  `duration_s < 1.5`, `duration_s > 15.0`, `sample_rate != 24000`, or `channels > 1`.

* **OOV token counting**: Run each transcript through the same `build_pipeline.py` phonemizer
  (British `lang_code="b"`) used in t0003. Count tokens whose phoneme output equals the
  character-level fallback (detectable by comparing output against a character-by-character
  reference). Flag clips where OOV fraction > 0.20.

* **Batch processing for speed**: Process all 1557 clips in a single pass with a `multiprocessing`
  pool. On a standard CPU (8 cores), `pyloudnorm` + `librosa` on 1557 clips should complete in under
  30 minutes.

* **Output format**: One JSON record per clip in `data/per_clip_stats.jsonl` with fields: `clip_id`,
  `path`, `duration_s`, `sample_rate`, `channels`, `peak_dbfs`, `lufs` (or `null` if measurement
  failed), `lufs_method`, `silence_fraction`, `oov_fraction`, `flags` (list of triggered flag
  codes).

## Gaps and Limitations

* **No papers reviewed**: The project corpus has zero papers, so all findings above are based on
  domain knowledge rather than cited literature. The thresholds used (> −1 dBFS, > 30% silence, <
  1.5 s / > 15 s duration) are industry-standard heuristics, not empirically validated against the
  specific v5 corpus. Future tasks should download and review at least the StyleTTS2 paper (Hu et
  al., 2023), the EBU R128 standard, and the pyloudnorm paper (Ward & Casagrande, 2016) to provide
  formal backing.

* **Threshold calibration**: The 30% silence threshold and −1 dBFS clipping threshold are not
  calibrated to the v5 corpus distribution. It is possible that a softer threshold (e.g. 20% or 40%
  silence) is more appropriate given the specific recording conditions. The distribution charts
  produced by this task should be used to re-evaluate thresholds before the full-corpus Stage 2 run.

* **No prior TTS data quality benchmarks in corpus**: There is no paper in the corpus comparing TTS
  model quality as a function of training data quality metrics (LUFS variance, clipping fraction,
  OOV rate). The claim that removing outlier clips improves training stability is well-supported by
  engineering practice but not by a controlled study on Kokoro or StyleTTS2.

* **pyloudnorm short-clip limitation**: The BS.1770-4 gating algorithm is unreliable for clips
  shorter than approximately 3 s because there are insufficient 400 ms gating blocks to compute a
  stable gate threshold. The RMS fallback is a pragmatic workaround but introduces a measurement
  inconsistency between short and long clips.

## Recommendations for This Task

1. **Use the standard flag thresholds** defined above (> −1 dBFS peak, > 30% silence, < 1.5 s /
   > 15 s, sample_rate ≠ 24000, channels > 1, OOV fraction > 20%). These are well-established
   > production heuristics and provide a conservative starting point for the cleaned manifest.

2. **Produce distribution charts before finalising thresholds**. If the LUFS distribution shows a
   bimodal pattern or the silence distribution has a heavy tail at 25–35%, revisit the thresholds
   before writing the final cleaned manifest.

3. **Report flagged clips by category (flag code)**. This reveals whether failures cluster in a
   specific dimension (e.g. all clipping, or mostly duration outliers), which informs whether the
   root cause is a systematic recording problem or isolated clip-level anomalies.

4. **Cross-validate OOV flags against t0003 phoneme fix**. Compare flagged OOV clips against the
   clips that were repaired in t0003. If all current OOV flags are concentrated in clips *not*
   touched by t0003, this suggests new problematic clips were added to the v5 manifest after t0003
   ran.

5. **Defer per-clip manual review** until after the automated pass. Only inspect flagged clips that
   are borderline (e.g. silence fraction 28–32%) manually. Clear failures (> 50% silence, peak > 0
   dBFS) should be removed without manual review.

## Paper Index

*No papers have been downloaded into the project corpus. This Paper Index is empty because
`papers_cited` is 0 and `status` is `"partial"`. Once the StyleTTS2, EBU R128, and pyloudnorm papers
are added to the corpus via future `add-paper` tasks, this section should be populated.*
