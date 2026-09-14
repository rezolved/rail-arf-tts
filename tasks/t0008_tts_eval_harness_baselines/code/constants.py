"""Named constants for t0008 TTS evaluation harness."""

from typing import Final

# ── Seed ─────────────────────────────────────────────────────────────────────
RANDOM_SEED: Final[int] = 42

# ── Corpus parameters ─────────────────────────────────────────────────────────
REFERENCE_CORPUS_SIZE: Final[int] = 1358  # expected total 11labs clips
REFERENCE_HALF_SIZE: Final[int] = 679  # each half of the centroid split
MIN_CORPUS_FOR_EVAL: Final[int] = 1000  # halt if fewer than this
FILLER_SAMPLE_SIZE: Final[int] = 100  # filler prompts to evaluate

# ── Resemblyzer ───────────────────────────────────────────────────────────────
MIN_CLIP_DURATION_S: Final[float] = 1.6  # clips shorter than this are skipped

# ── Duration gate ─────────────────────────────────────────────────────────────
DURATION_RATIO_HIGH: Final[float] = 2.0  # above → skip WER, flag for review
DURATION_RATIO_LOW: Final[float] = 0.5  # below → skip WER, flag for review
DURATION_EXPLOSION_RATIO: Final[float] = 5.0  # above → system result nulled
DURATION_EXPLOSION_FRACTION: Final[float] = 0.10  # fraction to trigger null

# ── Evaluation parameters ─────────────────────────────────────────────────────
DEFAULT_N_WARMUP: Final[int] = 1
DEFAULT_LIMIT: Final[int | None] = None  # None = all prompts
ELEVENLABS_RATE_LIMIT_SLEEP_S: Final[float] = 0.5
ELEVENLABS_MAX_RETRIES: Final[int] = 3

# ── Audio parameters ──────────────────────────────────────────────────────────
RESEMBLYZER_SAMPLE_RATE: Final[int] = 16_000  # required by resemblyzer
KOKORO_SAMPLE_RATE: Final[int] = 24_000  # Kokoro output sample rate
ELEVENLABS_SAMPLE_RATE: Final[int] = 44_100  # default ElevenLabs mp3

# ── System identifiers ────────────────────────────────────────────────────────
SYSTEM_ELEVENLABS_DAVID: Final[str] = "elevenlabs_david"
SYSTEM_KOKORO_BASE_GEORGE: Final[str] = "kokoro_base_george"
SYSTEM_KOKORO_BASE_LEWIS: Final[str] = "kokoro_base_lewis"
SYSTEM_KOKORO_BASE_V3_VOICEPACK: Final[str] = "kokoro_base_v3_voicepack"
SYSTEM_KOKORO_V3_BUNDLE: Final[str] = "kokoro_v3_bundle"
SYSTEM_KOKORO_T0006_V6D: Final[str] = "kokoro_t0006_v6d"
SYSTEM_KOKORO_T0005_BEST: Final[str] = "kokoro_t0005_best"
SYSTEM_KOKORO_FLOOR_CONTROL: Final[str] = "kokoro_floor_control"

ALL_SYSTEMS: Final[list[str]] = [
    SYSTEM_ELEVENLABS_DAVID,
    SYSTEM_KOKORO_BASE_GEORGE,
    SYSTEM_KOKORO_BASE_LEWIS,
    SYSTEM_KOKORO_BASE_V3_VOICEPACK,
    SYSTEM_KOKORO_V3_BUNDLE,
    SYSTEM_KOKORO_T0006_V6D,
    SYSTEM_KOKORO_T0005_BEST,
    SYSTEM_KOKORO_FLOOR_CONTROL,
]

KOKORO_SYSTEMS: Final[list[str]] = [
    SYSTEM_KOKORO_BASE_GEORGE,
    SYSTEM_KOKORO_BASE_LEWIS,
    SYSTEM_KOKORO_BASE_V3_VOICEPACK,
    SYSTEM_KOKORO_V3_BUNDLE,
    SYSTEM_KOKORO_T0006_V6D,
    SYSTEM_KOKORO_T0005_BEST,
    SYSTEM_KOKORO_FLOOR_CONTROL,
]

# ── Prompt-set identifiers ────────────────────────────────────────────────────
PROMPT_SET_VAL96: Final[str] = "val96"
PROMPT_SET_FILLERS: Final[str] = "fillers"
PROMPT_SET_BOTH: Final[str] = "both"
ALL_PROMPT_SETS: Final[list[str]] = [PROMPT_SET_VAL96, PROMPT_SET_FILLERS]

# ── Five-module checkpoint keys ───────────────────────────────────────────────
CHECKPOINT_MODULES: Final[list[str]] = [
    "bert",
    "bert_encoder",
    "predictor",
    "text_encoder",
    "decoder",
]

# ── Kokoro voice IDs ──────────────────────────────────────────────────────────
KOKORO_VOICE_GEORGE: Final[str] = "bm_george"
KOKORO_VOICE_LEWIS: Final[str] = "bm_lewis"
KOKORO_VOICE_FLOOR_CONTROL: Final[str] = "af_heart"  # female American

# ── ElevenLabs ────────────────────────────────────────────────────────────────
ELEVENLABS_DAVID_VOICE_NAME: Final[str] = "David"
ELEVENLABS_API_BASE: Final[str] = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL_ID: Final[str] = "eleven_turbo_v2_5"

# ── WER model ─────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE: Final[str] = "base.en"

# ── Metrics keys (must match registered project metrics) ─────────────────────
METRIC_SPEAKER_SIM: Final[str] = "speaker_sim"
METRIC_TTFB_MS: Final[str] = "ttfb_ms"
METRIC_RTF: Final[str] = "rtf"

# ── Per-clip JSON fields ──────────────────────────────────────────────────────
FIELD_SYSTEM: Final[str] = "system"
FIELD_PROMPT_SET: Final[str] = "prompt_set"
FIELD_TEXT: Final[str] = "text"
FIELD_TTFB_MS: Final[str] = "ttfb_ms"
FIELD_RTF: Final[str] = "rtf"
FIELD_SPEAKER_SIM: Final[str] = "speaker_sim"
FIELD_DURATION_RATIO: Final[str] = "duration_ratio"
FIELD_WER: Final[str] = "wer"
FIELD_AUDIO_PATH: Final[str] = "audio_path"
FIELD_REF_DURATION_S: Final[str] = "ref_duration_s"
FIELD_SYNTH_DURATION_S: Final[str] = "synth_duration_s"

# ── Success criteria ──────────────────────────────────────────────────────────
SUCCESS_SPEAKER_SIM: Final[float] = 0.85
SUCCESS_TTFB_MS: Final[float] = 300.0
