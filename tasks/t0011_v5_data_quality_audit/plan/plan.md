---
spec_version: "2"
task_id: "t0011_v5_data_quality_audit"
date_completed: "2026-09-15"
status: "complete"
---
## Objective

Audit all 1557 clips in the v5 training manifest (`data/v5/train_list.txt`) for six categories of
audio and transcript quality issues — clipping, excessive silence, LUFS outliers, duration outliers,
sample rate/channel mismatches, and OOV tokens — then produce a cleaned train manifest that strips
flagged clips. The cleaned manifest (`data/train_list_v5_clean.txt`) will be used as the input for
the full-corpus Stage 2 fine-tuning run once t0010 confirms the training safeguards work.

Success looks like:

* `data/per_clip_stats.jsonl` — one JSON record per clip with all computed metrics.
* `data/flagged_clips.txt` — clip paths that failed at least one threshold.
* `data/train_list_v5_clean.txt` — original manifest minus flagged entries.
* Four histogram PNGs in `results/images/` comparing train vs val distributions.
* `results/results_detailed.md` containing flag counts by category and distribution summaries.
* Total cost: $0 (CPU-only local compute, no paid services).

## Task Requirement Checklist

Operative task text (from `task.json` `short_description` and `task_description.md`):

> Audit all 1557 v5 training clips for clipping, silence, LUFS, duration outliers, and OOV tokens.
> Produce a cleaned train manifest for use in the full-corpus Stage 2 run after t0010 confirms the
> safeguards work.

Long-description sections:

> 1. Pull audio data: `dvc pull` both v5 train and v4 val; confirm clip count = 1557.
> 2. Per-clip audio stats: peak_dbfs, rms_lufs, silence_fraction, duration_s, sample_rate, channels.
> 3. Flag thresholds: peak_dbfs > −1, silence_fraction > 0.30, duration_s < 1.5 or > 15, sample_rate
>    ≠ 24000, channels > 1. Save `data/flagged_clips.txt` and `data/train_list_v5_clean.txt`.
> 4. Transcript audit: OOV fraction > 20% → flag.
> 5. Distribution charts: 4 histograms (peak dBFS, LUFS, duration, phoneme string length).
> 6. Summary: total clips, flag counts by category, cleaned manifest size, distribution stats.

Concrete requirements:

* **REQ-1** Run `dvc pull` for `data/v5/` (1557 clips) and `data/v4/val/` (96 clips); confirm clip
  counts match manifests before computing any metrics. (Step 1. Evidence: printed clip count in run
  log.)
* **REQ-2** Compute per-clip audio metrics: `peak_dbfs`, `rms_lufs`, `silence_fraction`,
  `duration_s`, `sample_rate`, `channels` for all 1557 train clips. (Step 2. Evidence:
  `data/per_clip_stats.jsonl` with 1557 records.)
* **REQ-3** Apply flag thresholds and save `data/flagged_clips.txt` listing all clip paths that
  failed at least one threshold; include a `flag_reasons` field. (Step 3. Evidence:
  `data/flagged_clips.txt` non-empty if any flags triggered, verified by wc -l.)
* **REQ-4** Write `data/train_list_v5_clean.txt` — original train manifest entries minus flagged
  clips. (Step 3. Evidence: file exists and line count = 1557 − flagged count.)
* **REQ-5** Run transcript OOV audit: for each clip in the train manifest scan `phonemes` field for
  `❓` (OOV_MARKER); flag clips where OOV fraction > 0.20. (Step 4. Evidence: `oov_fraction` field
  present in every `per_clip_stats.jsonl` record.)
* **REQ-6** Produce four histograms (peak dBFS, LUFS, duration, phoneme string length), each showing
  train vs val overlaid. (Step 5. Evidence: four PNG files in `results/images/`.)
* **REQ-7** Report total clips, flagged count by category, cleaned manifest size, and distribution
  stats (mean, p5, p95, p99). (Step 6 / results step. Evidence: `results/results_detailed.md`
  contains a table for each flag category and a stats table.)

## Approach

**Method chosen — single-pass audio audit writing JSONL, separate OOV scan:**

Research found that t0009 skipped all audio-level metrics because DVC data was not pulled at the
time. The v5 manifest format is pipe-delimited `wav_path|phonemes|speaker_id`; speaker_id is always
`"0"`. Canonical manifest paths are `data/v5/train_list.txt` (1557 clips) and `data/v5/val_list.txt`
(val) plus `data/v4/val/val_list.txt` (val_96 held-out — do NOT train on this).

Audio metrics are computed via:

* **Peak dBFS**: `20 * log10(abs(samples).max())` on a float32 waveform loaded with
  `soundfile.read()`. Flag if > −1 dBFS (hard clipping).
* **LUFS**: `pyloudnorm.Meter(sample_rate).integrated_loudness(audio)`. For clips < 3 s, the
  BS.1770-4 gating is unreliable (insufficient 400 ms blocks), so fall back to
  `20 * log10(rms(audio))` and set `lufs_method: "rms_fallback"`. Flag outside [−30, −6] LUFS.
* **Silence fraction**: `librosa.effects.split(top_db=60)` returns voiced intervals;
  `silence_fraction = 1 − sum(voiced_durations) / total_duration`. Flag if > 0.30.
* **Duration / sample rate / channels**: `soundfile.info(path)` without full decode for speed; flag
  duration < 1.5 s or > 15 s; flag sample_rate ≠ 24000; flag channels > 1.
* **OOV audit**: scan `phonemes` field for `❓` (OOV_MARKER constant from t0003). No audio load
  needed. Flag if `oov_count / total_phoneme_tokens > 0.20`.

The OOV scan is a pure string operation on manifest lines and runs in a separate script
(`code/audit_transcripts.py`) — no audio I/O. Results are merged into the per-clip JSONL by
`wav_path` key.

**Reuse from prior tasks:**

* `_parse_list` function from `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py`
  lines 36–52 — copy into `code/audit_audio.py`.
* `_phoneme_length_histogram` from t0009 `audit_data.py` lines 96–124 — generalise into
  `code/plot_histograms.py` covering all four charts.
* `OOV_MARKER`, `IPA_MARKERS`, `DOUBLE_PHONEMIZED_MARKERS` from
  `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py` lines 47–61 — copy into
  `code/constants.py`.
* `_get_wav_duration` pattern from `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` lines
  57–66 — extend with samplerate and channels fields.
* Paths pattern from `tasks/t0009_stage2_training_failure_forensics/code/paths.py` — copy structure
  into `code/paths.py`.

**Alternative considered — ffmpeg for LUFS:**

The task description mentions `ffmpeg -filter_complex ebur128` as an alternative to pyloudnorm.
Rejected because pyloudnorm is already in the project's Python environment (installed alongside
librosa), produces equivalent ITU-R BS.1770-4 integrated loudness, and avoids subprocess overhead.
The short-clip RMS fallback is cleaner to implement in pure Python than to parse ffmpeg JSON output.

**Alternative considered — pandas vectorised batch:**

Processing all 1557 clips in a pandas DataFrame with vectorised audio loading was considered.
Rejected because audio loading is inherently sequential (I/O bound, not compute bound); a simple
loop is simpler and the `multiprocessing.Pool` option is available for speed if needed. Given the
1557-clip count and < 30 min CPU budget, a sequential loop is sufficient.

**Registered metrics:** `ttfb_ms`, `speaker_sim`, and `rtf` are the three registered project
metrics. None applies to this data-analysis task: we do not run TTS inference, measure TTFB, or
compute GE2E similarity here. `results/metrics.json` will be written as `{}`.

**Task type:** `data-analysis` (confirmed in `task.json`). Planning guidelines followed: input
datasets confirmed from t0009, questions defined upfront as REQ-1 through REQ-7, all chart types
listed, no statistical tests required (flag counts are simple thresholds not requiring
significance).

## Cost Estimation

All computation is local CPU; no paid APIs, no remote GPU, no DVC storage write.

* DVC pull (read-only Azure Blob fetch): $0 — DVC reads from existing storage, no new writes.
* Python audio processing (soundfile, librosa, pyloudnorm): $0.
* Plotting (matplotlib): $0.
* **Total: $0**

Project budget is $5,000 USD; this task consumes $0. Well within the $100 per-task default limit.

## Step by Step

### Milestone A — Data acquisition and validation

1. **[CRITICAL] Pull v5 audio data via DVC.** From the repo root, run:

   ```bash
   uv run python -m arf.scripts.utils.run_with_logs --task-id t0011_v5_data_quality_audit -- \
     dvc pull data/v5/ data/v4/val/
   ```

   Expected output: DVC reports the number of files downloaded; no errors. After pull, verify clip
   count:

   ```bash
   wc -l data/v5/train_list.txt   # expected: 1557
   wc -l data/v5/val_list.txt     # expected: 96 or similar
   ls data/v5/wavs/ | wc -l       # expected: ≥ 1557
   ```

   If DVC fails with a credential error, create
   `tasks/t0011_v5_data_quality_audit/intervention/ dvc_pull_failed.md` and stop — audio metrics
   cannot be computed without the files. If the file count does not match the manifest line count,
   log the discrepancy and continue (missing files will produce `None` values in JSONL with `error`
   field set).

   Satisfies **REQ-1**.

### Milestone B — Per-clip audio metrics

2. **Create `code/paths.py`.** Copy the path constants pattern from
   `tasks/t0009_stage2_training_failure_forensics/code/paths.py`. Define:

   ```python
   TASK_ROOT = Path(__file__).parent.parent
   DATA_DIR = TASK_ROOT / "data"
   RESULTS_DIR = TASK_ROOT / "results"
   IMAGES_DIR = RESULTS_DIR / "images"
   V5_TRAIN_LIST = Path("data/v5/train_list.txt")
   V5_VAL_LIST = Path("data/v5/val_list.txt")
   VAL_96_LIST = Path("data/v4/val_list.txt")
   PER_CLIP_STATS_JSONL = DATA_DIR / "per_clip_stats.jsonl"
   FLAGGED_CLIPS_TXT = DATA_DIR / "flagged_clips.txt"
   CLEAN_MANIFEST_TXT = DATA_DIR / "train_list_v5_clean.txt"
   ```

   No inputs, no outputs. Used by all other scripts.

3. **Create `code/constants.py`.** Copy from `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py`
   lines 47–61:

   * `OOV_MARKER: Final[str] = "❓"`
   * `IPA_MARKERS: Final[frozenset[str]]` — set of IPA-only characters
   * `DOUBLE_PHONEMIZED_MARKERS: Final[tuple[str, ...]]` — espeak double-phonemize fragments

   Add task-specific audio thresholds:

   ```python
   PEAK_DBFS_MAX: Final[float] = -1.0      # flag if peak_dbfs > this
   SILENCE_FRACTION_MAX: Final[float] = 0.30
   DURATION_MIN_S: Final[float] = 1.5
   DURATION_MAX_S: Final[float] = 15.0
   EXPECTED_SAMPLE_RATE: Final[int] = 24000
   EXPECTED_CHANNELS: Final[int] = 1
   LUFS_MIN: Final[float] = -30.0
   LUFS_MAX: Final[float] = -6.0
   OOV_FRACTION_MAX: Final[float] = 0.20
   SHORT_CLIP_LUFS_THRESHOLD_S: Final[float] = 3.0
   ```

4. **Create `code/audit_audio.py`.** Inputs: `data/v5/train_list.txt` and `data/v5/val_list.txt`.
   Output: `data/per_clip_stats.jsonl` (one JSON record per clip, one line per record).

   Copy `_parse_list` from `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py` lines
   36–52. Implement `compute_clip_stats(wav_path: Path) -> dict[str, object]` that:

   a. Opens the file with `soundfile.read(str(path), dtype="float32", always_2d=True)` to get
   `(audio, sr)`. `audio` shape is `(samples, channels)`. b. Reads `soundfile.info(str(path))` for
   `duration`, `samplerate`, `channels` (fast header read). c. Computes
   `peak_dbfs = 20 * math.log10(abs(audio).max() + 1e-9)`. d. Converts to mono:
   `audio_mono = audio.mean(axis=1)`. e. If `info.duration >= 3.0`:
   `lufs = pyloudnorm.Meter(sr).integrated_loudness(audio_mono)`; `lufs_method = "bs1770"`. Else:
   `rms = math.sqrt((audio_mono ** 2).mean())`; `lufs = 20 * math.log10(rms + 1e-9)`;
   `lufs_method = "rms_fallback"`. f. Runs `librosa.effects.split(audio_mono, top_db=60)` to get
   voiced intervals.
   `silence_fraction = 1.0 - sum(end - start for start, end in intervals) / len(audio_mono)`. g.
   Returns dict with fields: `wav_path`, `duration_s`, `sample_rate`, `channels`, `peak_dbfs`,
   `lufs`, `lufs_method`, `silence_fraction`, `error` (None on success).

   Handle exceptions per clip: on `soundfile.SoundFileError` or any other exception, return a record
   with `error: str(exc)` and all metric fields set to `null` / `None`.

   `main()` parses `V5_TRAIN_LIST` and `V5_VAL_LIST` with `_parse_list`, iterates all train clips,
   calls `compute_clip_stats`, appends each record to `PER_CLIP_STATS_JSONL` (open in append mode,
   write one JSON line per clip). Also compute val-clip stats for histogram comparison (stored in a
   separate in-memory list, written to `data/val_clip_stats.jsonl`). Print progress every 100 clips.

   Run:

   ```bash
   uv run python -m arf.scripts.utils.run_with_logs --task-id t0011_v5_data_quality_audit -- \
     uv run python -u tasks/t0011_v5_data_quality_audit/code/audit_audio.py
   ```

   Expected: `data/per_clip_stats.jsonl` with 1557 lines; `data/val_clip_stats.jsonl`. Print summary
   of how many clips had errors (missing files etc.).

   Validation gate: after the run, `wc -l data/per_clip_stats.jsonl` must equal 1557. If < 1400
   (fewer than 90% of expected), create an intervention file and stop — the DVC pull may have been
   incomplete.

   Satisfies **REQ-2**.

### Milestone C — Transcript OOV audit

5. **Create `code/audit_transcripts.py`.** Inputs: `data/v5/train_list.txt`. Output: merges
   `oov_fraction` and `oov_count` fields into each record already in `data/per_clip_stats.jsonl`
   (rewrite in place as a new file, then rename).

   Implement `compute_oov_stats(phonemes: str) -> dict[str, object]`:

   a. Split `phonemes` into tokens by iterating characters (phoneme strings are character streams,
   not space-separated tokens — count characters as proxy tokens). b. Count occurrences of
   `OOV_MARKER` (`❓`) in the string. c. Count occurrences of any fragment in
   `DOUBLE_PHONEMIZED_MARKERS`. d. `total_len = len(phonemes)` (character count). e.
   `oov_count = phonemes.count(OOV_MARKER)`. f. `oov_fraction = oov_count / max(total_len, 1)`. g.
   `has_double_phonemize = any(m in phonemes for m in DOUBLE_PHONEMIZED_MARKERS)`.

   `main()` reads `V5_TRAIN_LIST`, builds a dict `{wav_path: oov_stats}` for all 1557 clips, then
   reads `PER_CLIP_STATS_JSONL` line by line, merges the OOV fields, writes to a temp file, and
   renames to `PER_CLIP_STATS_JSONL`.

   Run:

   ```bash
   uv run python -m arf.scripts.utils.run_with_logs --task-id t0011_v5_data_quality_audit -- \
     uv run python -u tasks/t0011_v5_data_quality_audit/code/audit_transcripts.py
   ```

   Expected: every line of `data/per_clip_stats.jsonl` now has `oov_fraction` and `oov_count`.

   Satisfies **REQ-5**.

### Milestone D — Flagging and manifest generation

6. **Create `code/build_manifest.py`.** Inputs: `data/per_clip_stats.jsonl`,
   `data/v5/train_list.txt`. Outputs: `data/flagged_clips.txt`, `data/train_list_v5_clean.txt`.

   Read `PER_CLIP_STATS_JSONL`; for each record evaluate all flag conditions:

   ```
   clipping:     peak_dbfs > PEAK_DBFS_MAX  (and peak_dbfs is not None)
   silence:      silence_fraction > SILENCE_FRACTION_MAX
   duration_low: duration_s < DURATION_MIN_S
   duration_high: duration_s > DURATION_MAX_S
   samplerate:   sample_rate != EXPECTED_SAMPLE_RATE
   channels:     channels > EXPECTED_CHANNELS
   lufs_low:     lufs < LUFS_MIN
   lufs_high:    lufs > LUFS_MAX
   oov:          oov_fraction > OOV_FRACTION_MAX
   error:        error is not None (file unreadable)
   ```

   A clip is flagged if ANY condition is true. For each flagged clip, write one line to
   `FLAGGED_CLIPS_TXT`:

   ```
   wav_path<TAB>clipping,silence,...
   ```

   Print a table of flag counts per category to stdout.

   Read `V5_TRAIN_LIST`; write every non-flagged line to `CLEAN_MANIFEST_TXT` unchanged.

   Run:

   ```bash
   uv run python -m arf.scripts.utils.run_with_logs --task-id t0011_v5_data_quality_audit -- \
     uv run python -u tasks/t0011_v5_data_quality_audit/code/build_manifest.py
   ```

   Expected: `data/flagged_clips.txt` (may be 0 lines if corpus is clean);
   `data/train_list_v5_clean.txt` with line count = 1557 − (number of flagged clips). Print
   per-category flag counts.

   Satisfies **REQ-3**, **REQ-4**.

### Milestone E — Distribution charts

7. **Create `code/plot_histograms.py`.** Inputs: `data/per_clip_stats.jsonl`,
   `data/val_clip_stats.jsonl`. Outputs: four PNG files in `results/images/`.

   Implement `plot_histogram(train_vals, val_vals, title, xlabel, out_path, flag_lines=None)`:

   * Use `matplotlib.use("Agg")`.
   * `fig, ax = plt.subplots(figsize=(10, 5))`.
   * `ax.hist(train_vals, bins=40, alpha=0.6, label=f"train (n={len(train_vals)})", color="steelblue")`.
   * `ax.hist(val_vals, bins=40, alpha=0.6, label=f"val (n={len(val_vals)})", color="orange")`.
   * If `flag_lines` provided, draw vertical red dashed lines at each threshold value.
   * `ax.set_title(title)`, `ax.set_xlabel(xlabel)`, `ax.set_ylabel("Count")`, `ax.legend()`,
     `ax.grid(True, alpha=0.3)`, `fig.tight_layout()`, `fig.savefig(out_path, dpi=120)`.
   * `plt.close(fig)`.

   Produce four charts:

   a. `results/images/peak_distribution.png` — peak_dbfs; flag line at −1 dBFS. b.
   `results/images/lufs_distribution.png` — lufs; flag lines at −30 and −6 LUFS. c.
   `results/images/duration_distribution.png` — duration_s; flag lines at 1.5 s and 15 s. d.
   `results/images/phoneme_distribution.png` — len(phonemes) from manifest; no flag line.

   Skip None values when building histogram arrays.

   Run:

   ```bash
   uv run python -m arf.scripts.utils.run_with_logs --task-id t0011_v5_data_quality_audit -- \
     uv run python -u tasks/t0011_v5_data_quality_audit/code/plot_histograms.py
   ```

   Expected: four PNG files exist in `results/images/`, each > 10 KB.

   Satisfies **REQ-6**.

### Milestone F — Distribution statistics

8. **Extend `code/build_manifest.py` or create `code/compute_stats.py`.** Inputs:
   `data/per_clip_stats.jsonl`. Outputs: `data/distribution_stats.json`.

   For each numeric metric (`peak_dbfs`, `lufs`, `duration_s`, `silence_fraction`), compute and
   print:

   * count (non-null), mean, std, min, p5, p50, p95, p99, max.

   Write these to `data/distribution_stats.json` as a dict keyed by metric name. Also write
   `data/flag_counts.json`:

   ```json
   {
     "total_clips": 1557,
     "flagged_total": N,
     "flagged_by_category": {
       "clipping": N, "silence": N, "duration_low": N, "duration_high": N,
       "samplerate": N, "channels": N, "lufs_low": N, "lufs_high": N,
       "oov": N, "error": N
     },
     "clean_manifest_clips": N
   }
   ```

   Run as part of `build_manifest.py` main function (or as a separate script). Either approach is
   acceptable, but all outputs must be produced.

   Satisfies **REQ-7** (together with `results_detailed.md`).

### Milestone G — Quality checks

9. **Run ruff and mypy on all task code.** After all scripts are written:

   ```bash
   uv run ruff check --fix tasks/t0011_v5_data_quality_audit/code/
   uv run ruff format tasks/t0011_v5_data_quality_audit/code/
   uv run mypy -p tasks.t0011_v5_data_quality_audit.code
   ```

   Fix all errors before committing.

## Remote Machines

None required. All computation runs locally on CPU. The full 1557-clip pass with `soundfile` +
`librosa` + `pyloudnorm` completes in < 30 min on any modern multi-core CPU. No GPU, no remote
provisioning, no DVC write operations.

## Assets Needed

* **Audio clips** — `data/v5/wavs/` (1557 WAV files, 24 kHz mono). Fetched via `dvc pull data/v5/`.
  DVC remote: `azure://ml-dvc-datasets/datasets/rail-arf-tts` (account: `mldvcstorerezolve`).
* **Val audio clips** — `data/v4/val/wavs/` (96 WAV files). Fetched via `dvc pull data/v4/val/`.
* **v5 train manifest** — `data/v5/train_list.txt` (pipe-delimited, format
  `wav_path|phonemes|speaker_id`). Present in repo; produced by t0003.
* **v5 val manifest** — `data/v5/val_list.txt`. Present in repo.
* **Val-96 manifest** — `data/v4/val/val_list.txt`. Present in repo.
* **Prior task code** — read-only references:
  * `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py` (`_parse_list`, histogram
    pattern)
  * `tasks/t0009_stage2_training_failure_forensics/code/paths.py` (paths pattern)
  * `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py` (`OOV_MARKER`, IPA constants)
  * `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` (`_get_wav_duration` pattern)

## Expected Assets

`task.json` `expected_assets` is `{}` — this task produces no registered asset types (no paper,
model, library, dataset, predictions, or answer assets). The outputs are task-internal data files
used by future tasks:

* `data/per_clip_stats.jsonl` — one JSON line per v5 train clip with all computed metrics.
* `data/val_clip_stats.jsonl` — same for v5 val clips (used only for histogram comparison).
* `data/flagged_clips.txt` — clip paths failing at least one quality threshold.
* `data/train_list_v5_clean.txt` — cleaned train manifest for downstream Stage 2 run.
* `data/distribution_stats.json` — per-metric percentile stats.
* `data/flag_counts.json` — structured flag-count summary.
* `results/images/peak_distribution.png`
* `results/images/lufs_distribution.png`
* `results/images/duration_distribution.png`
* `results/images/phoneme_distribution.png`
* `results/metrics.json` — `{}` (no registered project metrics apply to this analysis).
* `results/costs.json` — `{"total_cost_usd": 0, "breakdown": {}}`
* `results/remote_machines_used.json` — `[]`

## Time Estimation

* Research (already done): 0 min.
* Step 1 — DVC pull (1557 × ~100 KB WAVs ≈ 150 MB): 2–10 min depending on network.
* Steps 2–3 — `code/paths.py`, `code/constants.py`: 10 min.
* Step 4 — `audit_audio.py` script writing: 20 min.
* Step 4 run — 1557 clips × ~1 s/clip = 25–40 min on single core; ~10 min with Pool(4).
* Step 5 — `audit_transcripts.py` writing + run: 10 min (no audio I/O, fast).
* Step 6 — `build_manifest.py` writing + run: 15 min.
* Step 7 — `plot_histograms.py` writing + run: 10 min.
* Step 8 — stats extension: 5 min.
* Step 9 — ruff/mypy: 5 min.
* Total implementation: ~2 h wall clock.

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| DVC pull fails (Azure credential not configured) | Medium | Blocking | Check `dvc remote list` and `az login` before starting; create `intervention/dvc_pull_failed.md` with exact error if pull fails |
| Audio file missing despite dvc pull (partial pull) | Low | Partial data | Set `error` field in JSONL for missing files; continue processing; report missing count; flag as `error` category |
| pyloudnorm not installed in uv environment | Low | Blocking | Check `uv run python -c "import pyloudnorm"` before starting Milestone B; add to `pyproject.toml` if missing |
| librosa not installed | Low | Blocking | Check `uv run python -c "import librosa"` before starting; add to `pyproject.toml` if missing |
| LUFS threshold miscalibrated (corpus has unusual loudness) | Medium | Over-flagging | Inspect distribution chart before finalising manifest; adjust thresholds if > 30% of clips fall outside [−30, −6] |
| val_96 accidentally included in training | High impact if occurs | Verify `data/v4/val/val_list.txt` paths do NOT appear in `data/train_list_v5_clean.txt` after generation |  |

## Verification Criteria

* **DVC pull succeeded and clip count matches manifest:**

  ```bash
  wc -l data/v5/train_list.txt   # must print 1557
  ls data/v5/wavs/ | wc -l       # must be ≥ 1557
  ```

* **Per-clip JSONL has correct record count:**

  ```bash
  wc -l tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl   # must print 1557
  ```

  Satisfies REQ-2.

* **Every JSONL record has all required fields:**

  ```bash
  uv run python -c "
  import json; lines = open('tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl').readlines()
  required = {'wav_path','duration_s','sample_rate','channels','peak_dbfs','lufs','lufs_method',
              'silence_fraction','oov_fraction','oov_count','error'}
  missing = [i for i,l in enumerate(lines) if not required.issubset(json.loads(l).keys())]
  print('missing fields in records:', missing)
  "
  ```

  Expected: empty list. Satisfies REQ-2, REQ-5.

* **Flagged clips file exists and clean manifest line count is consistent:**

  ```bash
  FLAGGED=$(wc -l < tasks/t0011_v5_data_quality_audit/data/flagged_clips.txt)
  CLEAN=$(wc -l < tasks/t0011_v5_data_quality_audit/data/train_list_v5_clean.txt)
  echo "flagged=$FLAGGED clean=$CLEAN total=$((FLAGGED + CLEAN))"
  # total must equal 1557
  ```

  Satisfies REQ-3, REQ-4.

* **OOV fraction field present in all records** (verified by the field-check above). Satisfies
  REQ-5.

* **Four chart PNGs exist:**

  ```bash
  ls -lh tasks/t0011_v5_data_quality_audit/results/images/*.png
  # must list peak_distribution.png, lufs_distribution.png, duration_distribution.png,
  # phoneme_distribution.png each > 10 KB
  ```

  Satisfies REQ-6.

* **val_96 not leaked into clean manifest:**

  ```bash
  uv run python -c "
  val96 = set(l.split('|')[0] for l in open('data/v4/val/val_list.txt'))
  clean = set(l.split('|')[0] for l in open('tasks/t0011_v5_data_quality_audit/data/train_list_v5_clean.txt'))
  overlap = val96 & clean
  print('val96 leak count:', len(overlap))
  "
  # must print 0
  ```

* **Plan verificator passes:**

  ```bash
  uv run python -m arf.scripts.verificators.verify_plan t0011_v5_data_quality_audit
  # must exit 0 with zero errors
  ```

## Rejection Criteria

This is a data-analysis task, not a benchmark task, so there are no benchmark numbers to declare
null. However, the cleaned manifest must meet minimum quality:

* If fewer than 1200 clips remain in `data/train_list_v5_clean.txt` after flagging (< 77% of 1557),
  the thresholds are likely miscalibrated. Do not proceed with the manifest; instead create an
  intervention file listing the per-category flag counts and requesting human threshold review.
* If DVC pull succeeds for < 90% of clips listed in `data/v5/train_list.txt`, flag as incomplete and
  do not produce a cleaned manifest without human review.

## Data Flow

```
data/v5/train_list.txt  (1557 lines)  ─┐
data/v5/val_list.txt                   │  parse with _parse_list()
data/v5/wavs/*.wav  (DVC)              │
                                       ↓
                            audit_audio.py
                                       │
                     data/per_clip_stats.jsonl (audio metrics)
                     data/val_clip_stats.jsonl  (for charts)
                                       │
                            audit_transcripts.py
                            (merges oov_fraction)
                                       │
                     data/per_clip_stats.jsonl (+ OOV fields)
                                       │
              ┌────────────────────────┤
              ↓                        ↓
     build_manifest.py         plot_histograms.py
              │                        │
     data/flagged_clips.txt    results/images/*.png
     data/train_list_v5_clean.txt
     data/distribution_stats.json
     data/flag_counts.json
```
