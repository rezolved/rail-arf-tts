"""Named constants for t0012_v5_corpus_normalize_and_reaudit."""

from typing import Final

# Audio quality thresholds (kept unchanged from t0011 per task instruction, except
# PEAK_DBFS_MAX which is dropped as a flag and replaced by CLIPPED_FRACTION_MAX -- REQ-3).
SILENCE_FRACTION_MAX: Final[float] = 0.30  # flag if silence_fraction > this
DURATION_MIN_S: Final[float] = 1.5  # flag if duration_s < this
DURATION_MAX_S: Final[float] = 15.0  # flag if duration_s > this (kept for parity, 0 clips fire)
EXPECTED_SAMPLE_RATE: Final[int] = 24000  # flag if sample_rate != this
EXPECTED_CHANNELS: Final[int] = 1  # flag if channels != this
LUFS_MIN: Final[float] = -30.0  # flag if lufs < this
LUFS_MAX: Final[float] = -6.0  # flag if lufs > this
SHORT_CLIP_LUFS_THRESHOLD_S: Final[float] = 3.0  # use RMS fallback below this

# Corrected clipping metric (S-0011-03, REQ-2/REQ-3): fraction of samples within 1 LSB of
# full scale. Replaces the old peak_dbfs > -0.1 dBFS flag, which over-flagged clips that are
# merely peak-normalized (not hard-clipped).
CLIPPED_FRACTION_MAX: Final[float] = 0.001  # flag if clipped_fraction > this (0.1%)

# LUFS normalization target (S-0011-01, REQ-5).
TARGET_LUFS: Final[float] = -14.0

# Peak ceiling applied as a cap on the LUFS gain (not a post-hoc clip): a naive
# `gain = 10**((TARGET_LUFS - lufs)/20)` on this corpus pushes many already-peak-normalized-near-0-
# dBFS clips into genuine post-normalization clipping (measured empirically: 408/1531 clips, up to
# 0.6% clipped samples, when gain is applied uncapped and only np.clip(...,-1,1) guards the output).
# Capping the gain so peak never exceeds this ceiling is standard true-peak-limited loudness
# normalization practice and keeps clipped_fraction ~0 for every normalized clip; a clip whose LUFS
# gain would exceed the ceiling lands quieter than -14 LUFS rather than being flagged and dropped.
PEAK_CEILING_DBFS: Final[float] = -1.0

# Bit-depth lookup for clipped_fraction (REQ-2 ambiguity note in plan/plan.md): detect from
# soundfile's reported subtype rather than hardcoding 16-bit. FLOAT/DOUBLE and any unseen
# subtype fall back to 16-bit since float PCM has no fixed LSB -- documented here, not silent.
SUBTYPE_BITS: Final[dict[str, int]] = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
}
DEFAULT_BIT_DEPTH: Final[int] = 16

# Rejection / intervention thresholds
MIN_CLEAN_CLIPS: Final[int] = 1400  # < 1400 clean clips -> flag for human review
MIN_PULL_FRACTION: Final[float] = 0.90  # data pull < 90% of expected -> incomplete
EXPECTED_TRAIN_CLIPS: Final[int] = 1557
T0011_BASELINE_CLEAN_CLIPS: Final[int] = 1311  # t0011's clean-manifest count; the floor to beat

# Field names used in per_clip_stats_v2.jsonl records
FIELD_WAV_PATH: Final[str] = "wav_path"
FIELD_DURATION_S: Final[str] = "duration_s"
FIELD_SAMPLE_RATE: Final[str] = "sample_rate"
FIELD_CHANNELS: Final[str] = "channels"
FIELD_PRE_PEAK_DBFS: Final[str] = "pre_peak_dbfs"
FIELD_PRE_LUFS: Final[str] = "pre_lufs"
FIELD_LUFS_METHOD: Final[str] = "lufs_method"
FIELD_PRE_SILENCE_FRACTION: Final[str] = "pre_silence_fraction"
FIELD_PRE_CLIPPED_FRACTION: Final[str] = "pre_clipped_fraction"
FIELD_POST_PEAK_DBFS: Final[str] = "post_peak_dbfs"
FIELD_POST_LUFS: Final[str] = "post_lufs"
FIELD_POST_SILENCE_FRACTION: Final[str] = "post_silence_fraction"
FIELD_POST_CLIPPED_FRACTION: Final[str] = "post_clipped_fraction"
FIELD_NORMALIZED_PATH: Final[str] = "normalized_path"
FIELD_FLAGS: Final[str] = "flags"
FIELD_ERROR: Final[str] = "error"
