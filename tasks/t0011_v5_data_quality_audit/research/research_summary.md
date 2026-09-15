# Research Summary — t0011_v5_data_quality_audit

## Key Findings (top 10 insights directly actionable for this task)

1. **DVC pull is mandatory first**: t0009 skipped all audio-level metrics because `dvc pull` was not
   run; its loudness histogram is a placeholder PNG. t0011 must `dvc pull data/v5/` and confirm clip
   count == 1557 before any metric computation.
2. **Flag thresholds are well-established**: peak > −1 dBFS (clipping), silence_fraction > 0.30,
   duration < 1.5 s or > 15 s, sample_rate ≠ 24000, channels > 1, OOV fraction > 0.20.
3. **OOV audit is read-only**: v5 manifests already contain phonemized strings from t0003. Scan the
   phonemes field for `❓` (the `OOV_MARKER` constant) — no G2P re-run needed. Constants
   `IPA_MARKERS` and `DOUBLE_PHONEMIZED_MARKERS` from t0003 enable additional corruption checks.
4. **Manifest format**: pipe-delimited `wav_path|phonemes|speaker_id`, speaker_id always `"0"`.
   Canonical paths: train `data/v5/train_list.txt`, val `data/v5/val_list.txt`, held-out
   `data/v4/val_list.txt` (all relative to repo root).
5. **Peak dBFS formula**: `20 * log10(abs(samples).max())` on float32 waveform (librosa normalises
   to [−1, 1]). Flag if > −1.
6. **LUFS via pyloudnorm**: `pyloudnorm.Meter(sample_rate).integrated_loudness(audio)`. For clips <
   3 s, fall back to RMS-based dBFS and mark `lufs_method: "rms_fallback"`. Flag outside [−30, −6]
   LUFS.
7. **Silence fraction via librosa**: `librosa.effects.split(top_db=60)` returns voiced intervals.
   `silence_fraction = 1 − sum(voiced_durations) / total_duration`. Flag if > 0.30.
8. **soundfile for headers**: `soundfile.info(path)` returns duration, samplerate, channels without
   decoding audio — fast for 1557 files.
9. **Clipped waveforms cannot be rescued**: Normalisation cannot undo the hard nonlinearity; remove
   clipped clips entirely rather than gain-reducing them.
10. **pyloudnorm short-clip caveat**: BS.1770-4 gating is unreliable for clips < 3 s (insufficient
    400 ms blocks). Use the RMS fallback and mark those records.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Single-pass per-clip audit writing JSONL

Read each clip's audio once with `soundfile.read()`, compute all metrics (peak dBFS, LUFS, silence
fraction, duration, sample rate, channels) in that pass, then append a JSON record to
`data/per_clip_stats.jsonl`. Process manifests sequentially or with `multiprocessing.Pool` — on 8
CPU cores the full 1557-clip pass should complete in < 30 min. This keeps I/O minimal and produces
the primary output file in a single script invocation.

### Approach 2: Separate OOV scan over the manifest (no audio loading)

Parse `data/v5/train_list.txt` into `(wav_path, phonemes, speaker_id)` tuples and scan the phonemes
field for `❓`, `DOUBLE_PHONEMIZED_MARKERS`, and characters outside the Kokoro 114-symbol vocab. This
needs no audio and can run independently of the audio metrics pass. Output is an `oov_fraction` per
clip merged into the JSONL records or kept in a sidecar file.

### Approach 3: Centralised paths module + generalized histogram helper

Define all output paths in `code/paths.py` (pattern from t0003 and t0009), and write a single
`plot_histogram(train_vals, val_vals, title, xlabel, out_path)` helper covering all four charts
(peak dBFS, LUFS, duration, phoneme-string length). This avoids per-chart boilerplate and matches
the project's established matplotlib pattern (`Agg` backend, `dpi=120`, `alpha=0.6`,
`tight_layout`).

## Reusable Code / Assets

* `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py` lines 36–52 — `_parse_list`:
  pipe-delimited manifest parser returning `list[dict[str, str]]`. **Copy into task.**
* `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py` lines 96–124 —
  `_phoneme_length_histogram`: matplotlib histogram pattern (Agg, dpi=120, alpha=0.6). **Copy and
  generalise into task.**
* `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py` lines 47–61 — `OOV_MARKER`, `IPA_MARKERS`,
  `DOUBLE_PHONEMIZED_MARKERS`. **Copy constants into task.**
* `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` lines 139–153 — `rejection_reason`:
  per-string corruption gate. **Copy into task; simplify by dropping the `vocab` out-of-vocab check
  if not needed.**
* `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` lines 57–66 — `_get_wav_duration` pattern
  using `soundfile.info`. **Copy and extend** to also return `samplerate` and `channels`.
* `tasks/t0009_stage2_training_failure_forensics/code/paths.py` — centralized paths pattern. **Copy
  structure** (all output paths as module-level Path constants).

## Key Papers (top 5, with finding most relevant to this task)

*No papers in project corpus (research-papers step returned `status: "partial"`, 0 papers cited).*

* **(Domain knowledge)** ITU-R BS.1770-4 / EBU R128 — −23 LUFS broadcast target; integrated loudness
  computed with K-weighting + gating; flag corpus outliers outside [−30, −6] LUFS.
* **(Domain knowledge)** LJSpeech / LibriTTS-R convention — clip duration restricted to 1–15 s in
  all major TTS baselines; LJSpeech mean ≈ 6.6 s ± 2.3 s.

## Risks Flagged in Research

* **DVC pull may fail** if Azure Blob credentials are not configured; block the pipeline on a clear
  error message if audio files are absent.
* **pyloudnorm short-clip unreliability** — clips < 3 s get an RMS fallback; mark `lufs_method`
  field accordingly so downstream consumers know which records are approximate.
* **LUFS threshold is uncalibrated** to the v5 corpus — produce distribution charts and revisit
  thresholds before finalising the cleaned manifest.
* **t0003 OOV fix may not cover all clips** — new OOV clips added after t0003 ran would not have
  been repaired; cross-check flagged OOV clips against the t0003 reject list.
* **Clipping hypothesis** (from research_papers.md): NaN gradient events in t0009 may correlate with
  a small subset of heavily clipped clips; cleaning them could reduce NaN frequency independent of
  t0010's safeguards.

## Full Detail Available In

* `tasks/t0011_v5_data_quality_audit/research/research_papers.md` — 0 papers (status: partial; no
  papers in corpus; findings from domain knowledge only)
* `tasks/t0011_v5_data_quality_audit/research/research_internet.md` — (not generated — step skipped)
* `tasks/t0011_v5_data_quality_audit/research/research_code.md` — 3 tasks cited (t0003, t0008,
  t0009); 2 libraries found; 6 reusable code items identified
