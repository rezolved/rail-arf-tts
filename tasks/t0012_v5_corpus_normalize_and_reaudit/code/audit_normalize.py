"""Corrected audit + LUFS normalization for t0012_v5_corpus_normalize_and_reaudit.

For every clip in the v5 train manifest (1557 clips), in a single per-clip pass:
  1. Decode once, compute pre-normalization stats (peak_dbfs, lufs, silence_fraction,
     clipped_fraction -- REQ-2) using the corrected clipped_fraction metric instead of
     t0011's peak_dbfs > -0.1 dBFS flag (REQ-3).
  2. If the clip passes the corrected pre-normalization flags, normalize it to -14 LUFS
     (REQ-5), write it to data/v5_normalized/, and re-check clipped_fraction/silence on the
     normalized array in-memory (REQ-6, no re-read from disk).
  3. Write one JSON record per clip to data/per_clip_stats_v2.jsonl (REQ-7).

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0012_v5_corpus_normalize_and_reaudit -- \
        uv run python -u tasks/t0012_v5_corpus_normalize_and_reaudit/code/audit_normalize.py \
        [--limit N]
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from tasks.t0012_v5_corpus_normalize_and_reaudit.code.constants import (
    CLIPPED_FRACTION_MAX,
    DEFAULT_BIT_DEPTH,
    DURATION_MAX_S,
    DURATION_MIN_S,
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLE_RATE,
    EXPECTED_TRAIN_CLIPS,
    FIELD_CHANNELS,
    FIELD_DURATION_S,
    FIELD_ERROR,
    FIELD_FLAGS,
    FIELD_LUFS_METHOD,
    FIELD_NORMALIZED_PATH,
    FIELD_POST_CLIPPED_FRACTION,
    FIELD_POST_LUFS,
    FIELD_POST_PEAK_DBFS,
    FIELD_POST_SILENCE_FRACTION,
    FIELD_PRE_CLIPPED_FRACTION,
    FIELD_PRE_LUFS,
    FIELD_PRE_PEAK_DBFS,
    FIELD_PRE_SILENCE_FRACTION,
    FIELD_SAMPLE_RATE,
    FIELD_WAV_PATH,
    LUFS_MAX,
    LUFS_MIN,
    MIN_PULL_FRACTION,
    PEAK_CEILING_DBFS,
    SHORT_CLIP_LUFS_THRESHOLD_S,
    SILENCE_FRACTION_MAX,
    SUBTYPE_BITS,
    TARGET_LUFS,
)
from tasks.t0012_v5_corpus_normalize_and_reaudit.code.paths import (
    PER_CLIP_STATS_V2_JSONL,
    V5_NORMALIZED_DIR,
    V5_TRAIN_LIST,
)


@dataclass(frozen=True, slots=True)
class ClipArrayStats:
    """Metrics computed from a decoded audio array."""

    peak_dbfs: float
    lufs: float
    lufs_method: str
    silence_fraction: float
    clipped_fraction: float


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


def _bit_depth(subtype: str) -> int:
    """Map soundfile subtype to bit depth. Unknown/float subtypes default to 16 bits."""
    return SUBTYPE_BITS.get(subtype, DEFAULT_BIT_DEPTH)


def compute_array_stats(
    *,
    audio_2d: np.ndarray,
    sr: int,
    duration_s: float,
    bit_depth: int,
    meter: pyln.Meter | None,
) -> ClipArrayStats:
    """Compute peak/lufs/silence/clipped_fraction metrics from a decoded audio array."""
    peak_abs = float(np.abs(audio_2d).max())
    peak_dbfs = 20.0 * math.log10(peak_abs + 1e-9)

    audio_mono = audio_2d.mean(axis=1)

    if duration_s >= SHORT_CLIP_LUFS_THRESHOLD_S and meter is not None:
        lufs = float(meter.integrated_loudness(audio_mono))
        lufs_method = "bs1770"
    else:
        rms = float(np.sqrt(np.mean(audio_mono**2)))
        lufs = 20.0 * math.log10(rms + 1e-9)
        lufs_method = "rms_fallback"

    intervals = librosa.effects.split(y=audio_mono, top_db=60)
    total_samples = len(audio_mono)
    voiced_samples = sum(int(end) - int(start) for start, end in intervals)
    silence_fraction = 1.0 - voiced_samples / max(total_samples, 1)

    lsb = 1.0 / (2 ** (bit_depth - 1))
    clip_threshold = 1.0 - lsb
    clipped_fraction = float(np.mean(np.abs(audio_2d.flatten()) >= clip_threshold))

    return ClipArrayStats(
        peak_dbfs=peak_dbfs,
        lufs=lufs,
        lufs_method=lufs_method,
        silence_fraction=silence_fraction,
        clipped_fraction=clipped_fraction,
    )


def _pre_normalization_flags(
    *,
    stats: ClipArrayStats,
    duration_s: float,
    sample_rate: int,
    channels: int,
) -> list[str]:
    """Corrected pre-normalization flag set (REQ-3): clipped_fraction replaces peak_dbfs."""
    reasons: list[str] = []
    if stats.clipped_fraction > CLIPPED_FRACTION_MAX:
        reasons.append("clipping")
    if stats.silence_fraction > SILENCE_FRACTION_MAX:
        reasons.append("silence")
    if duration_s < DURATION_MIN_S:
        reasons.append("duration_low")
    if duration_s > DURATION_MAX_S:
        reasons.append("duration_high")
    if sample_rate != EXPECTED_SAMPLE_RATE:
        reasons.append("samplerate")
    if channels != EXPECTED_CHANNELS:
        reasons.append("channels")
    if stats.lufs < LUFS_MIN:
        reasons.append("lufs_low")
    if stats.lufs > LUFS_MAX:
        reasons.append("lufs_high")
    return reasons


def process_clip(wav_path_str: str, meter_cache: dict[int, pyln.Meter]) -> dict[str, object]:
    """Audit + normalize a single clip. Returns the per_clip_stats_v2.jsonl record."""
    record: dict[str, object] = {
        FIELD_WAV_PATH: wav_path_str,
        FIELD_DURATION_S: None,
        FIELD_SAMPLE_RATE: None,
        FIELD_CHANNELS: None,
        FIELD_PRE_PEAK_DBFS: None,
        FIELD_PRE_LUFS: None,
        FIELD_LUFS_METHOD: None,
        FIELD_PRE_SILENCE_FRACTION: None,
        FIELD_PRE_CLIPPED_FRACTION: None,
        FIELD_POST_PEAK_DBFS: None,
        FIELD_POST_LUFS: None,
        FIELD_POST_SILENCE_FRACTION: None,
        FIELD_POST_CLIPPED_FRACTION: None,
        FIELD_NORMALIZED_PATH: None,
        FIELD_FLAGS: [],
        FIELD_ERROR: None,
    }
    try:
        wav_path = Path(wav_path_str)
        if not wav_path.exists():
            record[FIELD_ERROR] = f"file not found: {wav_path_str}"
            record[FIELD_FLAGS] = ["error"]
            return record

        info = sf.info(str(wav_path))
        duration_s = float(info.duration)
        sample_rate = int(info.samplerate)
        channels = int(info.channels)
        record[FIELD_DURATION_S] = duration_s
        record[FIELD_SAMPLE_RATE] = sample_rate
        record[FIELD_CHANNELS] = channels

        audio_2d, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        bit_depth = _bit_depth(info.subtype)

        if sr not in meter_cache:
            meter_cache[sr] = pyln.Meter(sr)
        meter = meter_cache[sr]

        pre_stats = compute_array_stats(
            audio_2d=audio_2d, sr=sr, duration_s=duration_s, bit_depth=bit_depth, meter=meter
        )
        record[FIELD_PRE_PEAK_DBFS] = pre_stats.peak_dbfs
        record[FIELD_PRE_LUFS] = pre_stats.lufs
        record[FIELD_LUFS_METHOD] = pre_stats.lufs_method
        record[FIELD_PRE_SILENCE_FRACTION] = pre_stats.silence_fraction
        record[FIELD_PRE_CLIPPED_FRACTION] = pre_stats.clipped_fraction

        pre_flags = _pre_normalization_flags(
            stats=pre_stats, duration_s=duration_s, sample_rate=sample_rate, channels=channels
        )

        if len(pre_flags) == 0:
            lufs_gain = 10.0 ** ((TARGET_LUFS - pre_stats.lufs) / 20.0)
            peak_abs = float(np.abs(audio_2d).max())
            peak_ceiling_amp = 10.0 ** (PEAK_CEILING_DBFS / 20.0)
            peak_safe_gain = peak_ceiling_amp / (peak_abs + 1e-9)
            gain = min(lufs_gain, peak_safe_gain)
            normalized = np.clip(audio_2d * gain, -1.0, 1.0).astype(np.float32)

            out_path = V5_NORMALIZED_DIR / wav_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), normalized, sr, subtype=info.subtype)
            record[FIELD_NORMALIZED_PATH] = str(out_path)

            post_stats = compute_array_stats(
                audio_2d=normalized,
                sr=sr,
                duration_s=duration_s,
                bit_depth=bit_depth,
                meter=meter,
            )
            record[FIELD_POST_PEAK_DBFS] = post_stats.peak_dbfs
            record[FIELD_POST_LUFS] = post_stats.lufs
            record[FIELD_POST_SILENCE_FRACTION] = post_stats.silence_fraction
            record[FIELD_POST_CLIPPED_FRACTION] = post_stats.clipped_fraction

            post_flags: list[str] = []
            if post_stats.clipped_fraction > CLIPPED_FRACTION_MAX:
                post_flags.append("clipping_after_normalization")
            if post_stats.silence_fraction > SILENCE_FRACTION_MAX:
                post_flags.append("silence_after_normalization")
            record[FIELD_FLAGS] = post_flags
        else:
            record[FIELD_FLAGS] = pre_flags

    except Exception as exc:  # noqa: BLE001
        record[FIELD_ERROR] = str(exc)
        record[FIELD_FLAGS] = ["error"]

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit + normalize v5 train clips.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N manifest lines."
    )
    args = parser.parse_args()

    print("=== audit_normalize.py: corrected audit + LUFS normalization for v5 train ===")

    entries = _parse_list(V5_TRAIN_LIST)
    n_total = len(entries)
    print(f"  v5 train: {n_total} clips in manifest")

    if args.limit is None and n_total < int(EXPECTED_TRAIN_CLIPS * MIN_PULL_FRACTION):
        print(
            f"ERROR: Only {n_total} train entries found; "
            f"expected >= {int(EXPECTED_TRAIN_CLIPS * MIN_PULL_FRACTION)}. "
            "Data pull may be incomplete. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit is not None:
        entries = entries[: args.limit]
        print(f"  --limit {args.limit}: processing first {len(entries)} clips")

    V5_NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    PER_CLIP_STATS_V2_JSONL.parent.mkdir(parents=True, exist_ok=True)

    meter_cache: dict[int, pyln.Meter] = {}
    error_count = 0
    normalized_count = 0
    with PER_CLIP_STATS_V2_JSONL.open("w", encoding="utf-8") as out_f:
        for idx, entry in enumerate(entries):
            record = process_clip(entry["wav_path"], meter_cache)
            if record[FIELD_ERROR] is not None:
                error_count += 1
            if record[FIELD_NORMALIZED_PATH] is not None:
                normalized_count += 1
            out_f.write(json.dumps(record) + "\n")
            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx + 1}/{len(entries)} clips processed", flush=True)

    print(f"  Processed {len(entries)} clips: {error_count} errors, {normalized_count} normalized")
    print(f"  Wrote: {PER_CLIP_STATS_V2_JSONL}")
    print("Done.")


if __name__ == "__main__":
    main()
