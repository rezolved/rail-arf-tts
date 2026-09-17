"""Path constants for t0018 zero-shot voice-cloning calibration."""

from pathlib import Path

from tasks.t0008_tts_eval_harness_baselines.code.paths import (
    DATA_11LABS_DAVID_DIR,
    SYNTH_AUDIO_DIR,
    V3_DECODER,
    V3_VOICEPACK,
    VAL_LIST,
)

__all__ = [
    "DATA_11LABS_DAVID_DIR",
    "SYNTH_AUDIO_DIR",
    "V3_DECODER",
    "V3_VOICEPACK",
    "VAL_LIST",
    "TASK_ROOT",
    "DATA_DIR",
    "DATA_REFERENCES_DIR",
    "REF_SINGLE_WAV",
    "REF_CONCAT_WAV",
    "REFERENCES_MANIFEST",
    "HALF_A_CENTROID_NPY",
    "RESULTS_DIR",
    "RESULTS_IMAGES_DIR",
    "RESULTS_AUDIO_DIR",
    "RESULTS_AUDIO_HARNESS_DIR",
    "RESULTS_AUDIO_COMPARISON_DIR",
    "RESULTS_AUDIO_REFERENCES_DIR",
    "RESULTS_PER_CLIP_METRICS",
    "RESULTS_METRICS",
    "RESULTS_TABLES",
    "RESULTS_GATE_FAILURES",
    "RESULTS_ENVIRONMENT",
    "RESULTS_SMOKE_GATE_LOG",
    "RESULTS_LISTENING_GUIDE",
    "CHART_SPEAKER_SIM_BY_SYSTEM",
    "CHART_TTFB_VS_SPEAKER_SIM",
    "CHART_REF_CONDITION_EFFECT",
    "CHART_WER_BY_SYSTEM",
]

# Task root (tasks/t0018_zero_shot_cloning_calibration/)
TASK_ROOT: Path = Path(__file__).parent.parent

# ── Input data ────────────────────────────────────────────────────────────────
DATA_DIR: Path = TASK_ROOT / "data"
DATA_REFERENCES_DIR: Path = DATA_DIR / "references"
REF_SINGLE_WAV: Path = DATA_REFERENCES_DIR / "ref_single.wav"
REF_CONCAT_WAV: Path = DATA_REFERENCES_DIR / "ref_concat.wav"
REFERENCES_MANIFEST: Path = DATA_REFERENCES_DIR / "manifest.json"
HALF_A_CENTROID_NPY: Path = DATA_REFERENCES_DIR / "half_a_centroid.npy"

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_IMAGES_DIR: Path = RESULTS_DIR / "images"
RESULTS_AUDIO_DIR: Path = RESULTS_DIR / "audio_samples"
RESULTS_AUDIO_HARNESS_DIR: Path = RESULTS_AUDIO_DIR / "harness"
RESULTS_AUDIO_COMPARISON_DIR: Path = RESULTS_AUDIO_DIR / "comparison_set"
RESULTS_AUDIO_REFERENCES_DIR: Path = RESULTS_AUDIO_DIR / "references"

RESULTS_PER_CLIP_METRICS: Path = RESULTS_DIR / "per_clip_metrics.json"
RESULTS_METRICS: Path = RESULTS_DIR / "metrics.json"
RESULTS_TABLES: Path = RESULTS_DIR / "tables.json"
RESULTS_GATE_FAILURES: Path = RESULTS_DIR / "gate_failures.json"
RESULTS_ENVIRONMENT: Path = RESULTS_DIR / "environment.json"
RESULTS_SMOKE_GATE_LOG: Path = RESULTS_DIR / "smoke_gate_log.md"
RESULTS_LISTENING_GUIDE: Path = RESULTS_DIR / "listening_guide.md"

CHART_SPEAKER_SIM_BY_SYSTEM: Path = RESULTS_IMAGES_DIR / "speaker_sim_by_system.png"
CHART_TTFB_VS_SPEAKER_SIM: Path = RESULTS_IMAGES_DIR / "ttfb_vs_speaker_sim.png"
CHART_REF_CONDITION_EFFECT: Path = RESULTS_IMAGES_DIR / "ref_condition_effect.png"
CHART_WER_BY_SYSTEM: Path = RESULTS_IMAGES_DIR / "wer_by_system.png"
