"""Named constants for t0011_v5_data_quality_audit."""

from typing import Final

# OOV and phoneme corruption markers (copied from t0003_kokoro_v5_phoneme_data/code/constants.py)
OOV_MARKER: Final[str] = "❓"

# Characters that only appear in IPA output, never in source orthography.
IPA_MARKERS: Final[frozenset[str]] = frozenset("ˈˌːɹɪʊəɐɒɑɔɛɜæʌʒʃθðŋɡɾʔᵊᵻ")

# Literal fragments espeak emits when fed IPA: "SECONDARY STRESS", "SMALL CAPITAL", ...
DOUBLE_PHONEMIZED_MARKERS: Final[tuple[str, ...]] = (
    "sˌɛkəndɹɪstɹˌɛs",
    "smˈɔːlkˌap",
    "smˌɔːlkˌap",
    "pɹˈaɪmɚɹistɹˈɛs",
    "pɹˈaɪməɹɪstɹˈɛs",
    "mˈɒdɪfaɪɚlˈɛtɚ",
    "lˈatɪnsmˈɔːl",
)

# Audio quality thresholds
# NOTE: ElevenLabs David corpus is peak-normalized close to 0 dBFS (typical for TTS).
# The original plan specified -1.0 dBFS, but preflight inspection of 20 clips showed
# 18/20 (90%) exceeded -1.0 dBFS — all are peak-normalized, not hard-clipped.
# True hard clipping only occurs when the waveform saturates at exactly 0 dBFS (i.e.,
# integer overflow in PCM encoding). A threshold of -0.1 dBFS catches only genuine
# saturation events while ignoring the expected TTS normalization. Adjusted per
# plan/plan.md Risks & Fallbacks: "Inspect distribution chart before finalising manifest;
# adjust thresholds if > 30% of clips fall outside."
PEAK_DBFS_MAX: Final[float] = -0.1  # flag if peak_dbfs > this (true saturation clipping)
SILENCE_FRACTION_MAX: Final[float] = 0.30  # flag if silence_fraction > this
DURATION_MIN_S: Final[float] = 1.5  # flag if duration_s < this
DURATION_MAX_S: Final[float] = 15.0  # flag if duration_s > this
EXPECTED_SAMPLE_RATE: Final[int] = 24000  # flag if sample_rate != this
EXPECTED_CHANNELS: Final[int] = 1  # flag if channels > this
LUFS_MIN: Final[float] = -30.0  # flag if lufs < this
LUFS_MAX: Final[float] = -6.0  # flag if lufs > this
OOV_FRACTION_MAX: Final[float] = 0.20  # flag if oov_fraction > this
SHORT_CLIP_LUFS_THRESHOLD_S: Final[float] = 3.0  # use RMS fallback below this

# Rejection / intervention thresholds
MIN_CLEAN_CLIPS: Final[int] = 1200  # < 1200 clean clips → flag for human review
MIN_PULL_FRACTION: Final[float] = 0.90  # DVC pull < 90% of expected → incomplete
EXPECTED_TRAIN_CLIPS: Final[int] = 1557

# Field names used in JSONL records
FIELD_WAV_PATH: Final[str] = "wav_path"
FIELD_DURATION_S: Final[str] = "duration_s"
FIELD_SAMPLE_RATE: Final[str] = "sample_rate"
FIELD_CHANNELS: Final[str] = "channels"
FIELD_PEAK_DBFS: Final[str] = "peak_dbfs"
FIELD_LUFS: Final[str] = "lufs"
FIELD_LUFS_METHOD: Final[str] = "lufs_method"
FIELD_SILENCE_FRACTION: Final[str] = "silence_fraction"
FIELD_OOV_FRACTION: Final[str] = "oov_fraction"
FIELD_OOV_COUNT: Final[str] = "oov_count"
FIELD_HAS_DOUBLE_PHONEMIZE: Final[str] = "has_double_phonemize"
FIELD_ERROR: Final[str] = "error"
