"""Named constants for t0018 zero-shot voice-cloning calibration."""

from typing import Final

# ── System identifiers (cloning systems under test) ───────────────────────────
SYSTEM_F5_TTS: Final[str] = "f5_tts"
SYSTEM_COSYVOICE2: Final[str] = "cosyvoice2"
SYSTEM_CHATTERBOX: Final[str] = "chatterbox"

CLONING_SYSTEMS: Final[list[str]] = [SYSTEM_F5_TTS, SYSTEM_COSYVOICE2, SYSTEM_CHATTERBOX]

# ── Paired baselines (re-measured this GPU session) ───────────────────────────
SYSTEM_ELEVENLABS_DAVID: Final[str] = "elevenlabs_david"
SYSTEM_KOKORO_V3_BUNDLE: Final[str] = "kokoro_v3_bundle"

BASELINE_SYSTEMS: Final[list[str]] = [SYSTEM_ELEVENLABS_DAVID, SYSTEM_KOKORO_V3_BUNDLE]

# ── Reference-audio conditions ────────────────────────────────────────────────
CONDITION_REF_SINGLE: Final[str] = "ref_single"
CONDITION_REF_CONCAT: Final[str] = "ref_concat"

REFERENCE_CONDITIONS: Final[list[str]] = [CONDITION_REF_SINGLE, CONDITION_REF_CONCAT]

# ── Smoke gate / rejection thresholds ─────────────────────────────────────────
SMOKE_GATE_TIMEBOX_MINUTES: Final[int] = 45
SUCCESS_RATE_THRESHOLD: Final[float] = 0.8

# ── Fixed gate texts (shared with t0014/t0015 for cross-task comparability) ──
GATE_TEXT_NAMES: Final[tuple[str, ...]] = (
    "lining_up_suggestions_17",
    "lining_up_suggestions_10",
    "putting_them_head_to_head_15",
)

# ── Comparison-set sampling ────────────────────────────────────────────────────
COMPARISON_SET_SEED: Final[int] = 42
COMPARISON_SET_VAL96_COUNT: Final[int] = 7

# ── Cost tracking ──────────────────────────────────────────────────────────────
VM_HOURLY_COST_USD: Final[float] = 13.96
VM_BILLING_ANCHOR_ISO: Final[str] = "2026-09-17T15:21:13Z"
BUDGET_HARD_CAP_USD: Final[float] = 70.0
BUDGET_NEW_VARIANT_STOP_USD: Final[float] = 60.0

# ── Warmup ─────────────────────────────────────────────────────────────────────
N_WARMUP: Final[int] = 50
WARMUP_TEXT: Final[str] = "Checking the latest press release."

# ── Reference-audio duration targets (Step 3) ─────────────────────────────────
REF_SINGLE_MAX_DURATION_S: Final[float] = 10.0
REF_CONCAT_TARGET_DURATION_S: Final[float] = 30.0
REF_CONCAT_SILENCE_GAP_S: Final[float] = 0.2

# ── Baseline numbers from t0008 (for chart reference lines; see report_zeroshot.py) ──
# Source: tasks/t0008_tts_eval_harness_baselines/results/metrics.json
ELEVENLABS_SPEAKER_SIM_FILLERS: Final[float] = 0.832
ELEVENLABS_SPEAKER_SIM_VAL96: Final[float] = 0.792
KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS: Final[float] = 0.631
KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96: Final[float] = 0.588
