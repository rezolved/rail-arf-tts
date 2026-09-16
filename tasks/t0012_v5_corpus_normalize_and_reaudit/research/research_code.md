---
spec_version: "1"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
research_stage: "code"
tasks_reviewed: 3
tasks_cited: 3
libraries_found: 2
libraries_relevant: 0
date_completed: "2026-09-16"
status: "complete"
---
## Task Objective

t0012 implements two follow-up suggestions from t0011 together: **S-0011-03** replaces the
`peak_dbfs > -0.1 dBFS` clipping heuristic with a `clipped_fraction` metric (fraction of samples
within 1 LSB of full scale, threshold > 0.1%) that stops mistaking ElevenLabs' deliberate
peak-normalization for genuine clipping, and **S-0011-01** LUFS-normalizes the full v5 corpus (1557
clips) to -14 LUFS (EBU R128) using the corrected audit as the safety gate. The task must re-check
the *normalized* audio with the same corrected clipping metric (gain changes can push a
previously-safe clip into genuine clipping), then emit `data/per_clip_stats_v2.jsonl`,
`data/flagged_clips_v2.txt`, and `data/train_list_v5_normalized_clean.txt` as the near-full-corpus
clean manifest for the eventual full-corpus Stage 2 run (tracked separately as S-0011-02). This is a
CPU-only, no-GPU, $0-budget data task with no training involved.

## Library Landscape

The library aggregator (`aggregate_libraries --format json --detail short`) returns 2 registered
libraries, both created by prior tasks:

* **`tts_eval_harness`** (v0.1.0, created by `t0008_tts_eval_harness_baselines`) — a TTS evaluation
  harness for speaker similarity (GE2E cosine), TTFB, RTF, and WER, with entry points such as
  `compute_speaker_sim`, `build_centroid`, `compute_wer` (`code/scoring.py`) and adapter functions
  for ElevenLabs/Kokoro synthesis (`code/adapters.py`). **Not relevant** — t0012 does no synthesis
  or speaker-similarity scoring; it only re-audits and gain-normalizes existing recorded audio.
* **`t0009_training_safeguards`** (v0.1.0, created by `t0009_stage2_training_failure_forensics`) — a
  JSONL step logger, per-epoch checkpoint manager, health gates, and run-config capture for Kokoro
  StyleTTS2 Stage 2 GPU training. **Not relevant** — t0012 runs no training and needs no GPU or
  checkpoint machinery.

Neither library is imported by t0012. No aggregator output showed a correction or replacement
overlay for either library. `libraries_relevant` is `0`; this is documented per the skill's
requirement to record the full landscape even when nothing is reusable, rather than skipping the
aggregator.

The answer aggregator (`aggregate_answers --format json --detail short`) returns 1 answer asset,
`t0009-stage2-forensics-answer` (root causes of Kokoro Stage 2 training divergence). It concerns GPU
training configuration (`load_checkpoint` strict-mode bug, `joint_epoch` timing) and has no bearing
on CPU-only audio normalization/auditing — not cited further.

## Key Findings

### The corrected clipping heuristic and its exact rationale are already specified in t0011

t0011's own creative-thinking step ([t0011]) diagnosed the peak_dbfs false-positive problem in
detail: the v5 corpus's peak dBFS distribution has p95 = p99 = 0.0 dBFS (heavily right-skewed toward
full scale) because ElevenLabs peak-normalizes its TTS output to 0 dBFS by design, not because of
saturation. `results/creative_thinking.md` §1 and §5 (Strategy A) propose the exact metric this task
must implement: `clipped_fraction > 0.001` (fraction of samples within 1 LSB of full scale),
predicting "0-10 clips flagged" and a clean manifest of "~1532 clips (98.4%)" if applied alone.
`results/suggestions.json` (S-0011-03) restates the same threshold value (`> 0.1%`) as the task's
mandate — t0012's threshold is not a new design decision, it is implementing a fully-specified prior
recommendation.

### LUFS-normalization approach, target value, and code skeleton already exist

`results/creative_thinking.md` §4 gives a runnable code skeleton for the exact operation t0012 must
perform: read each clip with `soundfile.read(..., dtype="float32", always_2d=True)`, mono-mix,
compute integrated loudness with `pyloudnorm.Meter(sr).integrated_loudness(mono)` (RMS-dB fallback
on exception), compute `gain = 10 ** ((TARGET_LUFS - loudness) / 20)` with `TARGET_LUFS = -14.0`,
and `np.clip(audio * gain, -1.0, 1.0)` before writing. The same section flags the DVC anti-pattern
of mutating source files in place and recommends writing into a fresh directory
(`data/v5_normalized/`) with a new manifest — which matches t0012's task description exactly (§3,
`data/v5_normalized/`). t0011's corpus-level LUFS stats (mean -14.69, std 2.34, all already within
[-30, -6]) mean the -14 LUFS target requires only modest gain per clip — no clip needs a large boost
that would be likely to introduce new clipping, though re-checking after normalization (t0012 §4) is
still mandated as a safety net.

### t0011's three-script pipeline structure is the direct template for t0012's code layout

t0011 ([t0011]) split its audit into `audit_audio.py` (per-clip waveform metrics →
`per_clip_stats.jsonl`), `audit_transcripts.py` (OOV fraction merge into the same JSONL, in-place
tmp-file-then-rename pattern), `build_manifest.py` (flag thresholds → `flagged_clips.txt`,
`train_list_v5_clean.txt`, `distribution_stats.json`, `flag_counts.json`, `results_detailed.md`),
and `plot_histograms.py` (train-vs-val overlaid PNG histograms with threshold lines via
`ax.axvline`). All four scripts share `constants.py` (named `Final` thresholds and JSONL field-name
constants) and `paths.py` (centralized `Path` constants relative to `TASK_ROOT`). This
separation-of-concerns (compute → merge → flag/report → plot) is a clean, reusable pattern for
t0012's four required stages (corrected audit, normalization, post-normalization re-check,
manifest/report).

### DVC pull is unreliable for this corpus; a fallback download path exists

t0011's Limitations section ([t0011]) records that `dvc pull` failed due to Azure credential
chaining, and the task fell back to `az storage blob download --auth-mode login` (16 parallel
workers, ~64 minutes for 1557 + 96 clips) to fetch the WAV files directly from Azure Blob Storage.
In the current worktree, `data/v4/train/wavs.dvc` and `data/v4/val/wavs.dvc` exist but the `wavs/`
directories are empty (0 files), confirming the data has not been pulled in this worktree either.
Planning should budget for the same `dvc pull` → `az storage blob download` fallback sequence rather
than assuming a plain `dvc pull` will succeed, and should not spend excessive time debugging DVC
credentials before falling back.

### Manifest path naming: the "v5" manifest still stores clips under `data/v4/...` paths

Both `train_list_v5_clean.txt` entries and the JSONL `wav_path` field point at
`data/v4/train/wavs/<name>.wav` — the v5 manifest
(`tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt`) reuses the v4 audio directory
layout; only the phoneme transcripts changed between v4 and v5, not the underlying WAV files or
their paths ([t0011], confirmed directly by reading `data/train_list_v5_clean.txt`). t0012's
normalized-audio output directory (`data/v5_normalized/`) should be a genuinely new directory, not a
rewrite of `data/v4/...`, per the task description and t0011's own recommendation against in-place
mutation.

### Threshold and dependency versions are unchanged and already available

All Python packages t0012 needs (`pyloudnorm>=0.1`, `soundfile>=0.12`, `librosa>=0.10`,
`numpy>=2.0`, `matplotlib>=3.0`) are already declared in the repo's top-level `pyproject.toml` and
were exercised successfully by t0011 at 1557-clip scale in under 60 seconds of pure compute time
([t0011]) — no new dependencies are required.

## Reusable Code and Assets

All items below are non-library task code and must be **copied into task**
(`tasks/t0012_v5_corpus_normalize_and_reaudit/code/`) per the cross-task reuse rule — none of
t0011's code is registered as a library.

* **Source**: `tasks/t0011_v5_data_quality_audit/code/constants.py` (60 lines). **What it does**:
  `Final`-typed threshold constants (`SILENCE_FRACTION_MAX = 0.30`, `DURATION_MIN_S = 1.5`,
  `DURATION_MAX_S = 15.0`, `EXPECTED_SAMPLE_RATE = 24000`, `LUFS_MIN = -30.0`, `LUFS_MAX = -6.0`,
  `OOV_FRACTION_MAX = 0.20`, `SHORT_CLIP_LUFS_THRESHOLD_S = 3.0`, `MIN_CLEAN_CLIPS = 1200`) plus
  JSONL field-name constants (`FIELD_WAV_PATH`, `FIELD_PEAK_DBFS`, `FIELD_LUFS`, etc.). **Reuse
  method**: copy into task. **Adaptation needed**: keep all thresholds except clipping — per
  task_description.md §2, replace `PEAK_DBFS_MAX: Final[float] = -0.1` with a new
  `CLIPPED_FRACTION_MAX: Final[float] = 0.001` and add
  `FIELD_CLIPPED_FRACTION = "clipped_fraction"`; add `TARGET_LUFS: Final[float] = -14.0` for the
  normalization step; add pre/post-normalization peak/LUFS field names (e.g. `FIELD_PEAK_DBFS_PRE`,
  `FIELD_PEAK_DBFS_POST`, `FIELD_LUFS_PRE`, `FIELD_LUFS_POST`). **Line count**: ~60 lines, most
  reused verbatim.

* **Source**: `tasks/t0011_v5_data_quality_audit/code/paths.py` (27 lines). **What it does**:
  centralizes `Path` constants (`TASK_ROOT`, `DATA_DIR`, `RESULTS_DIR`, `IMAGES_DIR`, manifest
  paths, output file paths) relative to `Path(__file__).parent.parent`. **Reuse method**: copy into
  task. **Adaptation needed**: update `TASK_ROOT`-relative import to t0012's own package path;
  source manifest should point at
  `tasks/t0011_v5_data_quality_audit/data/train_list_v5_clean.txt`... but per task_description.md
  §1/§2, t0012 re-audits the *original* 1557-clip v5 manifest (same `V5_TRAIN_LIST` as t0011,
  `tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt`) with the corrected metric — not
  t0011's already-flagged 1311-clip subset — since the corrected heuristic will reclassify some of
  the 224 peak-flagged clips as clean. Add new output paths: `PER_CLIP_STATS_V2_JSONL`,
  `FLAGGED_CLIPS_V2_TXT`, `CLEAN_MANIFEST_V2_TXT`,
  `V5_NORMALIZED_DIR = DATA_DIR.parent / "data" / "v5_normalized"` (or equivalent, matching the task
  description's `data/v5_normalized/` output location). **Line count**: ~27 lines.

* **Source**: `tasks/t0011_v5_data_quality_audit/code/audit_audio.py` (`compute_clip_stats`
  function, lines 76–140; `_parse_list` helper, lines 55–70). **What it does**:
  `compute_clip_stats(wav_path_str: str) -> dict[str, object]` opens a WAV with `soundfile.info()`
  (fast header) then `soundfile.read(dtype="float32", always_2d=True)` (full decode), computes
  `peak_dbfs = 20 * log10(|max sample| + 1e-9)`, integrated LUFS via
  `pyloudnorm.Meter(sr).integrated_loudness(mono)` with an RMS-dB fallback for clips shorter than
  `SHORT_CLIP_LUFS_THRESHOLD_S`, and `silence_fraction` via
  `librosa.effects.split(y=mono, top_db=60)`. Returns `None`-filled fields plus an `error` string on
  any exception rather than raising. **Reuse method**: copy into task. **Adaptation needed**: add a
  `clipped_fraction` computation — fraction of samples with `abs(sample) >= 1.0 - 1/32768` (1 LSB of
  16-bit full scale, or the equivalent for the actual bit depth read) — alongside `peak_dbfs`; keep
  `peak_dbfs` in the output for comparison/reporting even though it's no longer the flag driver.
  This function must run twice: once on the original clips (pre-normalization corrected audit) and
  once on the `data/v5_normalized/` outputs (post-normalization re-check), so factor it to accept a
  `wav_path` and be reusable for both passes. **Line count**: ~65 lines of the ~200-line file are
  directly reusable.

* **Source**: `tasks/t0011_v5_data_quality_audit/code/build_manifest.py` (`_get_flag_reasons`, lines
  54–96; `_percentile`/`_compute_metric_stats`, lines 99–139; overall `main()` structure, lines
  142–316). **What it does**: `_get_flag_reasons(record: dict[str, object]) -> list[str]` applies
  every threshold check and returns a list of reason strings (short-circuits on `error`).
  `_compute_metric_stats(values: list[float]) -> dict[str, object]` computes count/mean/std/min/
  p5/p50/p95/p99/max. `main()` loads the JSONL, tallies `category_counts`, writes
  `flagged_clips.txt` (tab-separated path + reasons), writes the clean manifest (original minus
  flagged), and writes `distribution_stats.json`, `flag_counts.json`, and a Markdown
  `results_detailed.md` table. Also includes a `MIN_CLEAN_CLIPS` intervention-file safety check
  (writes `intervention/too_few_clean_clips.md` and exits 1 if too few clips survive). **Reuse
  method**: copy into task. **Adaptation needed**: replace the `"clipping"` reason branch
  (`peak_dbfs > PEAK_DBFS_MAX`) with `clipped_fraction > CLIPPED_FRACTION_MAX`; add a second pass
  over the post-normalization JSONL that re-applies the same clipped-fraction/silence checks and
  moves any clip that becomes genuinely clipped after gain into the flagged list
  (task_description.md §4) instead of writing it to `train_list_v5_normalized_clean.txt`; extend
  `_get_flag_reasons` or add a sibling function for this second pass. Keep the `MIN_CLEAN_CLIPS`
  intervention pattern as-is — it is a useful safety net given t0012 expects a near-full-corpus
  (~1556) result. **Line count**: ~200 of 316 lines are directly reusable structure.

* **Source**: `tasks/t0011_v5_data_quality_audit/code/plot_histograms.py` (`_load_field`,
  `plot_histogram`, full file, 181 lines). **What it does**:
  `plot_histogram(train_vals, val_vals, title, xlabel, out_path, flag_lines=None)` draws overlaid
  `matplotlib` histograms (`alpha=0.6`, 40 bins) with optional red dashed threshold lines
  (`ax.axvline`), saved via `Agg` backend at 120 dpi. **Reuse method**: copy into task. **Adaptation
  needed**: task_description.md §6 asks for before-vs-after (not train-vs-val) comparisons:
  `peak_before_after.png`, `lufs_before_after.png`, `clipped_fraction_distribution.png`. Reuse
  `plot_histogram` directly but pass (pre-normalization values, post-normalization values) as the
  two series instead of (train, val), and relabel series names accordingly (the function's
  `label=f"train (n=...)"` / `f"val (n=...)"` strings need overriding — consider adding a
  `series_labels` parameter or duplicating with renamed labels). `clipped_fraction_distribution.png`
  is a single-series histogram, so either extend `plot_histogram` to accept an optional second
  series or call it with an empty second array. **Line count**: ~120 of 181 lines reusable as-is;
  ~15-20 lines of adaptation for the before/after relabeling.

* **Source**: `tasks/t0011_v5_data_quality_audit/code/audit_transcripts.py` (full file, 124 lines) —
  **not needed**. t0012's scope (task_description.md) does not include an OOV re-audit; OOV was
  already checked exhaustively by t0011 with 0 flags and the transcripts do not change under gain
  normalization. Not recommended for copying; noted here only to explain the omission.

* **Source**: `tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl` (1557 records) and
  `tasks/t0011_v5_data_quality_audit/data/train_list_v5_clean.txt` (1311 entries). **What it does**:
  pre-computed peak_dbfs/lufs/silence_fraction/duration/oov stats for every v5 train clip from
  t0011's run — not DVC-tracked data, committed directly as small text/JSONL files in the
  (immutable) t0011 task folder. **Reuse method**: read-only reference (not copied) — t0011's folder
  is immutable per repo rule 5, and this task needs its own re-audit under the corrected metric on
  the original 1557-clip manifest, but t0011's `per_clip_stats.jsonl` is useful for consistency
  checks (e.g. confirming which of the 224 previously-peak-flagged clips reclassify as clean, and
  that the OOV/LUFS/ sample-rate/channel fields need not be recomputed since gain normalization does
  not change OOV and t0011 already found 0 flags in every other category besides
  clipping/duration/silence).

## Lessons Learned

* **`dvc pull` is not reliable for this corpus** — t0011 hit Azure credential-chaining failures and
  had to fall back to `az storage blob download --auth-mode login` with 16 parallel workers, taking
  ~64 of its ~65 total minutes just to fetch 1557 + 96 WAVs ([t0011], Methodology and Limitations
  sections). Plan for this fallback path up front rather than debugging DVC credentials from
  scratch.
* **Threshold calibration should be checked against real data before committing to it** — t0011's
  original plan specified `peak_dbfs > -1.0 dBFS`, but a 20-clip preflight check showed 90% of clips
  exceeded that threshold, forcing an in-flight adjustment to -0.1 dBFS (still imperfect, which is
  exactly why t0012 exists). The corrected `clipped_fraction > 0.001` metric is pre-validated by
  t0011's own creative-thinking analysis (§5, Strategy A) with an expected 0-10 flagged clips — a
  small sanity spot-check against a handful of genuinely full-scale-run clips before finalizing is
  still worth doing.
* **Writing normalized audio into a new directory, not in place, is required** — t0011's
  creative-thinking section explicitly flags in-place gain mutation as a "DVC anti-pattern" (§4
  Trade-off) and recommends a fresh `data/v5_normalized/` directory with a new manifest, which
  matches this task's own `task_description.md` requirement. This also keeps t0011's own data folder
  immutable, satisfying repo rule 5.
* **Processing itself is fast; data transfer dominates runtime** — t0011's actual audit compute
  (soundfile + librosa + pyloudnorm across 1557 clips) took under 60 seconds; nearly all wall-clock
  time was the WAV download. t0012's normalization pass (read + gain + write, roughly double the I/O
  of a read-only audit) should still be well within the "a few minutes" budget stated in
  `task_description.md`, as long as the WAV files are already local from a prior `dvc pull`/`az`
  download.
* **A hard floor on clean-manifest size catches miscalibrated thresholds early** — t0011's
  `build_manifest.py` writes an `intervention/too_few_clean_clips.md` file and exits nonzero if
  fewer than `MIN_CLEAN_CLIPS = 1200` clips remain. t0012 should keep an equivalent floor (e.g.
  ~1400, since the corrected metric plus normalization is expected to leave ~1550+ clean clips per
  t0011's own estimate) so a threshold or normalization bug that flags too many clips is caught by
  the script itself rather than only in manual review.
* **Val-set leakage checks are cheap insurance** — t0011's Verification section explicitly checked
  that 0 val_96 paths appear in the clean train manifest. t0012 should repeat this check on
  `train_list_v5_normalized_clean.txt`, since the project's `CLAUDE.md` states val_96 must never be
  trained on.

## Recommendations for This Task

1. **Copy, don't reimplement, t0011's four-script skeleton** into
   `tasks/t0012_v5_corpus_normalize_and_reaudit/code/`: `constants.py`, `paths.py`, an audio-stats
   module (from `audit_audio.py`'s `compute_clip_stats`), a flag/report module (from
   `build_manifest.py`), and a plotting module (from `plot_histograms.py`). Do not copy
   `audit_transcripts.py` — OOV re-audit is out of scope per `task_description.md`.
2. **Implement `clipped_fraction`** exactly as specified by S-0011-03 and t0011's creative-thinking
   §5 Strategy A: fraction of samples within 1 LSB of full scale, threshold `> 0.001` (0.1%).
   Replace the `peak_dbfs`-based clipping branch in the flag-reasons function with this metric,
   while still recording `peak_dbfs` in the JSONL for the required before/after comparison plots.
3. **Run the corrected audit against the full original 1557-clip v5 manifest**
   (`tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt`), not t0011's already-reduced
   1311-clip clean list — the corrected clipping metric is expected to reclassify most of the 224
   previously peak-flagged clips as clean, so starting from the reduced set would silently drop ~200
   recoverable clips.
4. **Normalize with `pyloudnorm`, target -14.0 LUFS**, writing outputs to `data/v5_normalized/`
   using the gain-then-clip pattern from t0011's creative-thinking §4 code skeleton (adapted from a
   for-loop sketch into the task's own audio-stats/normalization module). Keep sample rate (24 kHz)
   and mono layout unchanged, matching t0011's finding that all 1557 clips are already uniformly 24
   kHz mono.
5. **Re-run the corrected clipped-fraction (and silence) check on the normalized output**, and move
   any newly-genuinely-clipped clip to the flagged list rather than the clean manifest — this is the
   explicit safety net task_description.md §4 requires and is not optional.
6. **Reuse `pyproject.toml`'s existing dependencies** (`pyloudnorm`, `soundfile`, `librosa`,
   `numpy`, `matplotlib`) — no new dependency additions are needed; t0011 already exercised all of
   them at this corpus's scale.
7. **`dvc add data/v5_normalized/` and `dvc push`** before completion per the repo's DVC rules
   (`CLAUDE.md`) — the normalized WAVs must not be committed to git directly. Budget time for the
   `dvc pull`-then-`az storage blob download` fallback sequence t0011 needed to fetch the source
   WAVs, since the worktree's `data/v4/train/wavs/` and `data/v4/val/wavs/` are currently empty
   (only `.dvc` pointer files present).
8. **No library import is needed** — `tts_eval_harness` and `t0009_training_safeguards` are both out
   of scope for this CPU-only audio-normalization task; do not import either.
9. **Repeat t0011's verification checklist**: total-count consistency (`flagged + clean = 1557`),
   val_96 leak check, JSONL required-field completeness, and a `MIN_CLEAN_CLIPS`-style intervention
   floor — all cheap, all already proven useful in t0011.

## Task Index

### [t0011]

* **Task ID**: `t0011_v5_data_quality_audit`
* **Name**: v5 training data audio quality audit and clean manifest
* **Status**: completed
* **Relevance**: Direct dependency and sole prior source of this task's data, thresholds, and both
  suggestions (S-0011-01, S-0011-03) being implemented here. Its code (`audit_audio.py`,
  `build_manifest.py`, `plot_histograms.py`, `constants.py`, `paths.py`), data
  (`per_clip_stats.jsonl`, `train_list_v5_clean.txt`), and creative-thinking analysis are the direct
  template and rationale for t0012's corrected audit and normalization pipeline.

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Created the `tts_eval_harness` library (speaker-similarity/TTFB/RTF/WER scoring).
  Reviewed via the library aggregator and found not relevant — t0012 does no TTS synthesis or
  evaluation, only audio-level re-auditing and loudness normalization of existing recordings.

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Created the `t0009_training_safeguards` library (GPU training step logging,
  checkpointing, health gates) and the sole registered answer asset (Kokoro Stage 2 divergence root
  causes). Reviewed via the library and answer aggregators and found not relevant — t0012 runs no
  GPU training.
