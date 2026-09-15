"""Per-clip audio metrics computation for t0011_v5_data_quality_audit.

Reads the v5 train and val manifests, computes audio quality metrics for every
clip, and writes results to JSONL files:
  - tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl  (train)
  - tasks/t0011_v5_data_quality_audit/data/val_clip_stats.jsonl  (val)

Metrics per clip: peak_dbfs, lufs, lufs_method, silence_fraction,
duration_s, sample_rate, channels, error.

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0011_v5_data_quality_audit -- \
        uv run python -u tasks/t0011_v5_data_quality_audit/code/audit_audio.py
"""

import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None  # type: ignore[assignment]

from tasks.t0011_v5_data_quality_audit.code.constants import (
    EXPECTED_TRAIN_CLIPS,
    FIELD_CHANNELS,
    FIELD_DURATION_S,
    FIELD_ERROR,
    FIELD_LUFS,
    FIELD_LUFS_METHOD,
    FIELD_PEAK_DBFS,
    FIELD_SAMPLE_RATE,
    FIELD_SILENCE_FRACTION,
    FIELD_WAV_PATH,
    MIN_PULL_FRACTION,
    SHORT_CLIP_LUFS_THRESHOLD_S,
)
from tasks.t0011_v5_data_quality_audit.code.paths import (
    DATA_DIR,
    PER_CLIP_STATS_JSONL,
    V5_TRAIN_LIST,
    V5_VAL_LIST,
    VAL_CLIP_STATS_JSONL,
)


def _parse_list(list_path: Path) -> list[dict[str, str]]:
    """Parse a pipe-delimited manifest: wav_path|phonemes|speaker_id."""
    entries: list[dict[str, str]] = []
    if not list_path.exists():
        print(f"  MISSING: {list_path}")
        return entries
    with list_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if len(line) == 0:
                continue
            parts = line.split("|")
            wav_path = parts[0] if len(parts) > 0 else ""
            phonemes = parts[1] if len(parts) > 1 else ""
            speaker_id = parts[2] if len(parts) > 2 else "0"
            entries.append({"wav_path": wav_path, "phonemes": phonemes, "speaker_id": speaker_id})
    return entries


def compute_clip_stats(wav_path_str: str) -> dict[str, object]:
    """Compute audio quality metrics for a single WAV file.

    Returns a dict with fields: wav_path, duration_s, sample_rate, channels,
    peak_dbfs, lufs, lufs_method, silence_fraction, error.
    On any exception, metric fields are None and error is set.
    """
    record: dict[str, object] = {
        FIELD_WAV_PATH: wav_path_str,
        FIELD_DURATION_S: None,
        FIELD_SAMPLE_RATE: None,
        FIELD_CHANNELS: None,
        FIELD_PEAK_DBFS: None,
        FIELD_LUFS: None,
        FIELD_LUFS_METHOD: None,
        FIELD_SILENCE_FRACTION: None,
        FIELD_ERROR: None,
    }
    try:
        wav_path = Path(wav_path_str)
        if not wav_path.exists():
            record[FIELD_ERROR] = f"file not found: {wav_path_str}"
            return record

        # Fast header read for duration, samplerate, channels
        info = sf.info(str(wav_path))
        record[FIELD_DURATION_S] = float(info.duration)
        record[FIELD_SAMPLE_RATE] = int(info.samplerate)
        record[FIELD_CHANNELS] = int(info.channels)

        # Full decode for waveform metrics
        audio_2d: np.ndarray
        sr: int
        audio_2d, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        # audio_2d shape: (samples, channels)

        # Peak dBFS on all channels
        peak_abs = float(np.abs(audio_2d).max())
        record[FIELD_PEAK_DBFS] = 20.0 * math.log10(peak_abs + 1e-9)

        # Mono mix for LUFS and silence detection
        audio_mono: np.ndarray = audio_2d.mean(axis=1)

        # LUFS
        if info.duration >= SHORT_CLIP_LUFS_THRESHOLD_S and pyln is not None:
            meter = pyln.Meter(sr)
            lufs_val = meter.integrated_loudness(audio_mono)
            record[FIELD_LUFS] = float(lufs_val)
            record[FIELD_LUFS_METHOD] = "bs1770"
        else:
            rms = float(np.sqrt(np.mean(audio_mono**2)))
            record[FIELD_LUFS] = 20.0 * math.log10(rms + 1e-9)
            record[FIELD_LUFS_METHOD] = "rms_fallback"

        # Silence fraction via librosa voiced-interval detection
        intervals = librosa.effects.split(y=audio_mono, top_db=60)
        total_samples = len(audio_mono)
        voiced_samples = sum(int(end) - int(start) for start, end in intervals)
        record[FIELD_SILENCE_FRACTION] = 1.0 - voiced_samples / max(total_samples, 1)

    except Exception as exc:
        record[FIELD_ERROR] = str(exc)

    return record


def main() -> None:
    print("=== audit_audio.py: per-clip audio metrics for v5 train+val ===")

    # Parse manifests
    train_entries = _parse_list(V5_TRAIN_LIST)
    val_entries = _parse_list(V5_VAL_LIST)

    print(f"  v5 train: {len(train_entries)} clips")
    print(f"  v5 val:   {len(val_entries)} clips")

    # Validate clip count
    n_train = len(train_entries)
    if n_train < int(EXPECTED_TRAIN_CLIPS * MIN_PULL_FRACTION):
        print(
            f"ERROR: Only {n_train} train entries found; "
            f"expected >= {int(EXPECTED_TRAIN_CLIPS * MIN_PULL_FRACTION)}. "
            "DVC pull may be incomplete. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Compute train stats
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    error_count = 0
    with PER_CLIP_STATS_JSONL.open("w", encoding="utf-8") as out_f:
        for idx, entry in enumerate(train_entries):
            record = compute_clip_stats(wav_path_str=entry["wav_path"])
            if record[FIELD_ERROR] is not None:
                error_count += 1
            out_f.write(json.dumps(record) + "\n")
            if (idx + 1) % 100 == 0:
                print(f"  Train progress: {idx + 1}/{n_train} clips processed", flush=True)

    print(f"  Train: {n_train} clips processed, {error_count} errors")
    print(f"  Wrote: {PER_CLIP_STATS_JSONL}")

    # Validate final record count
    with PER_CLIP_STATS_JSONL.open(encoding="utf-8") as f:
        actual_lines = sum(1 for _ in f)
    if actual_lines < int(EXPECTED_TRAIN_CLIPS * MIN_PULL_FRACTION):
        print(
            f"ERROR: per_clip_stats.jsonl has {actual_lines} lines, "
            f"expected >= {int(EXPECTED_TRAIN_CLIPS * MIN_PULL_FRACTION)}. "
            "Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  Validation: {actual_lines} lines in JSONL (expected {EXPECTED_TRAIN_CLIPS})")

    # Compute val stats for histogram comparison
    val_errors = 0
    with VAL_CLIP_STATS_JSONL.open("w", encoding="utf-8") as out_f:
        for entry in val_entries:
            record = compute_clip_stats(wav_path_str=entry["wav_path"])
            if record[FIELD_ERROR] is not None:
                val_errors += 1
            out_f.write(json.dumps(record) + "\n")

    print(f"  Val: {len(val_entries)} clips processed, {val_errors} errors")
    print(f"  Wrote: {VAL_CLIP_STATS_JSONL}")
    print("Done.")


if __name__ == "__main__":
    main()
