---
spec_version: "1"
task_id: "t0011_v5_data_quality_audit"
research_stage: "code"
tasks_reviewed: 9
tasks_cited: 3
libraries_found: 2
libraries_relevant: 1
date_completed: "2026-09-15"
status: "complete"
---
## Task Objective

Audit all 1557 v5 training clips for clipping (peak > −1 dBFS), silence (> 30% frames), LUFS
outliers, duration outliers (< 1.5 s or > 15 s), sample-rate or channel mismatches, and OOV tokens
in the phoneme transcripts. Produce `data/per_clip_stats.jsonl`, `data/flagged_clips.txt`, and
`data/train_list_v5_clean.txt` (the cleaned train manifest for the full-corpus Stage 2 run that
follows t0010). Also produce four distribution histograms comparing train and val clips.

## Library Landscape

Two libraries are currently registered in the project.

**`tts_eval_harness`** (v0.1.0, created by t0008) — reusable evaluation harness for TTS systems
measuring speaker similarity (GE2E cosine), TTFB, RTF, WER, and duration ratio. The library contains
audio loading helpers in `tasks/t0008_tts_eval_harness_baselines/code/scoring.py`:
`_get_wav_duration(path: Path) -> float | None` reads duration via `soundfile.info`. The constants
module at `tasks/t0008_tts_eval_harness_baselines/code/constants.py` defines
`KOKORO_SAMPLE_RATE = 24_000` and `MIN_CLIP_DURATION_S = 1.6`. This library is relevant for
understanding Kokoro's expected audio parameters and for duration reading, but its primary concern
is synthesis evaluation rather than training-data auditing. Import path:
`from tasks.t0008_tts_eval_harness_baselines.code.scoring import _get_wav_duration` — but because
`_get_wav_duration` is a private helper (`_` prefix) inside a non-library code directory, the
correct approach is to **copy** that pattern rather than import it.

**`t0009_training_safeguards`** (v0.1.0, created by t0009) — JSONL step logger, per-epoch checkpoint
manager, health gates, and run config capture for Kokoro StyleTTS2 Stage 2 training. Module paths
are
`tasks/t0009_stage2_training_failure_forensics/code/{jsonl_logger.py, checkpoint_manager.py, health_gates.py, run_config.py}`.
None of these modules deal with audio loading or per-clip statistics; they are exclusively
training-loop infrastructure. Not directly relevant to this data-audit task, but the `StepLogger`
class pattern (append one JSON record per item to a JSONL file) is exactly the output format
specified for `data/per_clip_stats.jsonl`. Import path for library:
`from tasks.t0009_stage2_training_failure_forensics.code.jsonl_logger import StepLogger` — however,
`StepLogger` is tightly coupled to training metrics field names. A simpler direct `json.dumps` +
file append is preferable for the per-clip audit record.

## Key Findings

### Manifest Parsing Pattern

The v5 train manifest format is consistently `wav_path|phonemes|speaker_id` (pipe-delimited,
`speaker_id = "0"`). This format is established in [t0003]'s `constants.py` (`FIELD_SEP = "|"`,
`SPEAKER_ID = "0"`) and consumed by [t0009]'s `audit_data.py` via the `_parse_list` function
(238-line file). The `_parse_list` function (lines 36–52 of `audit_data.py`) is a clean, reusable
manifest reader:

```python
def _parse_list(list_path: Path) -> list[dict[str, str]]:
    """Parse a manifest line: wav_path|phonemes|speaker_id"""
    entries: list[dict[str, str]] = []
    for raw in list_path.open():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        wav_path = parts[0] if len(parts) > 0 else ""
        phonemes = parts[1] if len(parts) > 1 else ""
        speaker_id = parts[2] if len(parts) > 2 else "0"
        entries.append({"wav_path": wav_path, "phonemes": phonemes, "speaker_id": speaker_id})
    return entries
```

The manifest lives at `data/v5/train_list.txt` (note: [t0009]'s `paths.py` line 48 sets
`V5_TRAIN_LIST = Path("data/v5/train_list.txt")`, confirming the canonical path relative to repo
root). The v5 val manifest is at `data/v5/val_list.txt` and the held-out val_96 set at
`data/v4/val_list.txt`.

### Audio Data Gap in t0009 and What t0011 Must Close

[t0009]'s `audit_data.py` explicitly skipped audio-level metrics because the v5 clips are
DVC-tracked and `dvc pull` was not run during forensics (see `audit_data.py` lines 183–188):

```python
"audio_audit": {
    "available": False,
    "reason": "Audio files DVC-tracked; dvc pull not run in this task. "
              "Peak/LUFS/silence stats not computed.",
},
```

The `_loudness_histogram` function (lines 127–148) produced a placeholder PNG with text saying audio
is unavailable. This task's entire purpose is to fill that gap — DVC-pull the clips and compute the
real metrics. The `_duration_stats` function used phoneme-string length as a proxy (not actual audio
duration). t0011 must compute actual duration in seconds via `soundfile` or `librosa`.

### OOV Token Pattern from t0003

The OOV detection logic is thoroughly established in [t0003]. The `rejection_reason` function in
`tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` (lines 139–153, ~14 lines) checks
whether a phoneme string contains the OOV marker `❓` (defined as `OOV_MARKER` in
`tasks/t0003_kokoro_v5_phoneme_data/code/constants.py`). For t0011, the transcripts are already
phonemized (stored in the manifest), so the OOV audit is simply scanning for `❓` in the phoneme
field — no G2P re-run needed. The `constants.py` also defines `IPA_MARKERS` (a frozenset of IPA
characters) and `DOUBLE_PHONEMIZED_MARKERS` (a tuple of espeak artifact strings), which can be
reused to detect additional corruption classes beyond plain OOV.

The `phonemize` function and `load_kokoro_vocab` / `install_lexicon` chain from [t0003] are only
needed if the task requires re-phonemizing from text. Since v5 manifests already contain phonemes,
t0011 can audit the phoneme strings directly without re-running G2P — a significant simplification.

### Path and Output Conventions

Both [t0009] and [t0003] use a centralized `paths.py` pattern defining all task-level paths as
module-level constants derived from `Path(__file__).parent.parent` (task root). This pattern is
clean and should be replicated in t0011's `code/paths.py`. Output locations for this task are:

* `tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl` — one JSON record per clip
* `tasks/t0011_v5_data_quality_audit/data/flagged_clips.txt` — flagged clip IDs
* `tasks/t0011_v5_data_quality_audit/data/train_list_v5_clean.txt` — cleaned manifest
* `tasks/t0011_v5_data_quality_audit/results/images/` — four PNG histograms

### Histogram Pattern from t0009

[t0009]'s `_phoneme_length_histogram` function (lines 96–124 of `audit_data.py`) shows the
matplotlib pattern used in the project: `matplotlib.use("Agg")` (non-interactive backend),
`fig, ax = plt.subplots(figsize=(10, 5))`, `ax.hist()` with `alpha=0.6` for both train and val,
`fig.tight_layout()`, `fig.savefig(out, dpi=120)`. This 28-line pattern should be copied into t0011
for all four histogram scripts (peak dBFS, LUFS, duration, phoneme length).

### JSONL Per-Record Output Pattern

[t0009]'s `StepLogger` in `jsonl_logger.py` (62 lines) demonstrates the JSONL append pattern: open
file in `"a"` mode, write `json.dumps(record) + "\n"`. For t0011's per-clip stats, a lightweight
direct implementation is preferable since the training-specific fields in `StepLogger` are
irrelevant. Each record in `per_clip_stats.jsonl` should be a dict with keys: `wav_path`,
`speaker_id`, `phonemes`, `peak_dbfs`, `rms_lufs`, `silence_fraction`, `duration_s`, `sample_rate`,
`channels`, `flags` (list of flag names), `oov_fraction`.

### Audio Metric Libraries

`librosa` and `soundfile` are already in the project's `pyproject.toml` (used by t0008 via
`soundfile` for duration reading and by resemblyzer for audio loading). For peak dBFS, `numpy` array
operations on the waveform are sufficient. For integrated LUFS, `pyloudnorm` (already referenced in
the task description and in the research_papers findings) is the correct tool:
`meter = pyloudnorm.Meter(sample_rate); lufs = meter.integrated_loudness(audio)`.

## Dataset Landscape

The v5 corpus is DVC-tracked under `data/v5/` (train) and `data/v4/val/` (val_96). As of [t0009],
`dvc pull` had not been run and actual audio was unavailable. The manifest counts confirmed by
[t0009]'s data audit: **1557 train clips**, **96 val_96 clips**. The train manifest path is
`data/v5/train_list.txt`; val_96 manifest is `data/v4/val_list.txt`. The first field in each line is
the relative wav path (e.g., `data/v4/train/wavs/filename.wav`) — these paths are relative to the
repo root and must be resolved accordingly when the audio files are accessed.

The 11labs David reference corpus (1358 clips at `data/11labs_david/`) is used by the evaluation
harness (t0008) for speaker similarity but is not needed for this data-quality audit.

## Reusable Code and Assets

### `_parse_list` — manifest line parser

* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py`, lines 36–52
* **What it does**: Parses pipe-delimited manifest lines into `list[dict[str, str]]` with keys
  `wav_path`, `phonemes`, `speaker_id`.
* **Reuse method**: **copy into task** (non-library code)
* **Function signature**: `_parse_list(list_path: Path) -> list[dict[str, str]]`
* **Adaptation needed**: None — the function is self-contained. Rename to `parse_manifest` for
  clarity (drop leading underscore since it becomes a module-level utility, not a private helper).
* **Line count**: ~17 lines

### `OOV_MARKER`, `IPA_MARKERS`, `DOUBLE_PHONEMIZED_MARKERS` — corruption detection constants

* **Source**: `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py`, lines 47–61
* **What it does**: `OOV_MARKER = "❓"` (misaki OOV token), `IPA_MARKERS` (frozenset of characters
  that only appear in IPA output), `DOUBLE_PHONEMIZED_MARKERS` (tuple of espeak artifact strings).
* **Reuse method**: **copy into task** (non-library code; t0003 is not a library)
* **Adaptation needed**: Copy the three constants into
  `tasks/t0011_v5_data_quality_audit/code/ constants.py`. Import nothing from t0003 directly.
* **Line count**: ~15 lines total for the three constants

### `rejection_reason` — per-phoneme-string OOV/corruption gate

* **Source**: `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py`, lines 139–153
* **What it does**: Checks a phoneme string against `OOV_MARKER`, `IPA_MARKERS`, and
  `DOUBLE_PHONEMIZED_MARKERS`, returning the first gate name hit or `None` if clean.
* **Function signature**: `rejection_reason(phonemes: str, vocab: frozenset[str]) -> str | None`
* **Reuse method**: **copy into task** (non-library code)
* **Adaptation needed**: For t0011 the `vocab` argument is only needed for the `out_of_vocab` gate.
  Since the manifests are already in-vocab (t0003 validated them), a simpler variant that only
  checks `OOV_MARKER` and `DOUBLE_PHONEMIZED_MARKERS` may suffice. Copy the full function and adjust
  as needed.
* **Line count**: ~15 lines

### `_phoneme_length_histogram` — matplotlib histogram pattern

* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/audit_data.py`, lines 96–124
* **What it does**: Renders a side-by-side histogram of two series (train vs val) with matplotlib
  using `Agg` backend, saves to PNG at `dpi=120`.
* **Reuse method**: **copy into task** (non-library code)
* **Adaptation needed**: Generalize into a
  `plot_histogram(train_values, val_values, title, xlabel, out_path)` utility. Four histograms
  needed: peak dBFS, LUFS, duration, phoneme length.
* **Line count**: ~28 lines; generalized version ~35 lines

### `_get_wav_duration` pattern — soundfile duration reading

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/scoring.py`, lines 57–66
* **What it does**: Returns duration in seconds via `soundfile.info(str(path)).duration`, or `None`
  on error.
* **Reuse method**: **copy into task** (non-library code; private helper in non-library module)
* **Adaptation needed**: Extend to also return `sample_rate` and `channels` from `sf.info()`. The
  `soundfile.SoundFileInfo` object exposes `.duration`, `.samplerate`, `.channels`.
* **Line count**: ~10 lines in extended form

## Lessons Learned

### t0009: Never Assume Audio Availability — Always DVC-Pull First

The primary lesson from [t0009]'s data audit failure is that DVC-tracked audio files are not
available by default: `dvc pull` must be run explicitly before any audio metric computation. The
placeholder charts produced by t0009 (`_loudness_histogram` wrote a "audio not available" PNG
instead of real data) caused an incomplete audit. t0011 must confirm `dvc pull` succeeds and clip
count matches the manifest before proceeding.

### t0003: OOV Rate Was 22% Before Lexicon Installation

[t0003] found that 22% of the corpus (359/1653 clips) had OOV tokens before lexicon installation,
overwhelmingly brand and person names ("Rezolve", "brainpowa", etc.). After the lexicon in
`tasks/t0003_kokoro_v5_phoneme_data/code/lexicon.py` was installed and the manifests regenerated,
the v5 manifests were declared clean by the rejection gate. The t0011 transcript audit is a
cross-check: if `❓` tokens appear in the v5 train manifest phoneme fields, it means the manifest was
not cleanly regenerated or new OOV words exist.

### t0008: `soundfile` Is the Standard Audio Loader

[t0008]'s harness uses `soundfile` (via `sf.info()`) for duration reading and numpy arrays for audio
data. This is the established project pattern. `librosa.load` would also work but adds resampling
overhead. For raw amplitude access, `soundfile.read()` returns a numpy array directly.

### t0009: Centralized `paths.py` Prevents Hardcoded Path Bugs

Both [t0003] and [t0009] use a single `paths.py` module defining all file paths as constants. This
prevents scattered `Path("tasks/tXXXX/.../something.json")` strings across scripts and makes
refactoring safe. t0011 should follow the same pattern.

### t0009: Histogram Pattern Works Well

The `_phoneme_length_histogram` function from [t0009] was used successfully to produce
`results/images/duration_histogram.png`. The `alpha=0.6`, `dpi=120`, `tight_layout` combination
produces clean overlapping histograms suitable for a results report.

## Recommendations for This Task

1. **Copy `_parse_list` from t0009/audit_data.py** as the manifest reader. Rename to
   `parse_manifest`. This is well-tested (used in [t0009]) and handles the pipe-delimited format
   correctly. Reuse method: copy into task.

2. **Use `soundfile` for audio loading** following the [t0008] pattern. Extend `_get_wav_duration`
   to a `load_clip_info(path: Path) -> ClipInfo` function that returns
   `(samples: np.ndarray, sample_rate: int, channels: int, duration_s: float)` in one pass. Copy
   into task.

3. **Copy OOV constants from t0003/constants.py** (`OOV_MARKER`, `IPA_MARKERS`,
   `DOUBLE_PHONEMIZED_MARKERS`) and the `rejection_reason` logic from `prepare_v5_data.py`. For
   t0011 the audit is read-only (no G2P re-run needed): scan `phonemes` field for `❓` directly.
   Reuse method: copy into task.

4. **Generalize the histogram pattern** from `t0009/audit_data.py` into a single
   `plot_histogram(train_vals, val_vals, title, xlabel, out_path)` function covering all four charts
   (peak dBFS, LUFS, duration, phoneme string length). Copy into task.

5. **Create `code/paths.py`** following the [t0003] and [t0009] convention. All output paths
   (`DATA_DIR`, `PER_CLIP_STATS_JSONL`, `FLAGGED_CLIPS_TXT`, `TRAIN_LIST_CLEAN_TXT`, `IMAGES_DIR`)
   defined there.

6. **Do not use `StepLogger` from t0009_training_safeguards** for per-clip output. The class is
   coupled to training-step field names. Use a direct `json.dumps(record) + "\n"` append loop. The
   JSONL format is correct but the implementation should be purpose-built.

7. **Verify DVC pull before processing**: add an early assertion that the wav files resolved from
   the manifest actually exist on disk, with a clear error message pointing to `dvc pull`.

## Task Index

### [t0003]

* **Task ID**: `t0003_kokoro_v5_phoneme_data`
* **Name**: Kokoro v5 phoneme data
* **Status**: completed
* **Relevance**: Defines the v5 manifest format, OOV detection constants (`OOV_MARKER`,
  `IPA_MARKERS`), the `rejection_reason` gate function, and the corpus WAV paths. The
  `build_pipeline.py` and phonemizer logic are not needed for this task (phonemes already in
  manifest), but the corruption-gate code and constants are directly reusable.

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness baselines
* **Status**: completed
* **Relevance**: Establishes `soundfile`-based audio duration reading (`_get_wav_duration`),
  confirms `KOKORO_SAMPLE_RATE = 24_000`, and provides the `scoring.py` audio loading pattern. Not
  directly involved in data auditing, but the audio utilities are the reference implementation for
  how the project reads WAV files.

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Direct dependency of t0011. Contains the manifest parser (`_parse_list`),
  histogram-rendering pattern, centralized `paths.py` convention, and JSONL-append pattern. Most
  importantly, explicitly documented that the audio-level audit was skipped due to missing DVC data
  — confirming exactly what t0011 must complete.
