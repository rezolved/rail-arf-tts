"""Path constants for t0021 zero-shot latency reduction."""

from pathlib import Path

from tasks.t0008_tts_eval_harness_baselines.code.paths import (
    DATA_FILLER_PROMPTS,
    DATA_VAL96_PROMPTS,
)

__all__ = [
    "DATA_FILLER_PROMPTS",
    "DATA_VAL96_PROMPTS",
    "T0018_HALF_A_CENTROID_NPY",
    "TASK_ROOT",
    "REPO_ROOT",
    "DATA_V4_DIR",
    "DATA_V4_VAL_WAVS_DIR",
    "DATA_V4_VAL_LIST",
    "DATA_DIR",
    "DATA_REFERENCES_DIR",
    "REF_SINGLE_WAV",
    "REF_CONCAT_WAV",
    "REFERENCES_MANIFEST",
    "VAL96_CENTROID_NPY",
    "OLD_WRONGVOICE_CENTROID_NPY",
    "T0018_TASK_ROOT",
    "T0018_COMPARISON_SET_DIR",
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
    "RESULTS_LATENCY_BREAKDOWN",
    "CHART_LATENCY_BREAKDOWN_STACKED",
    "CHART_TTFB_VS_SPEAKER_SIM_VARIANTS",
    "CHART_TTFB_P50_P95_BY_VARIANT",
]

# Task root (tasks/t0021_zero_shot_latency_reduction/)
TASK_ROOT: Path = Path(__file__).parent.parent
REPO_ROOT: Path = TASK_ROOT.parent.parent

# ── Input data: correct-voice source corpus (owner correction) ───────────────
DATA_V4_DIR: Path = REPO_ROOT / "data" / "v4"
DATA_V4_VAL_WAVS_DIR: Path = DATA_V4_DIR / "val" / "wavs"
DATA_V4_VAL_LIST: Path = DATA_V4_DIR / "val_list.txt"

# ── This task's own reference/centroid outputs ────────────────────────────────
DATA_DIR: Path = TASK_ROOT / "data"
DATA_REFERENCES_DIR: Path = DATA_DIR / "references"
REF_SINGLE_WAV: Path = DATA_REFERENCES_DIR / "ref_single.wav"
REF_CONCAT_WAV: Path = DATA_REFERENCES_DIR / "ref_concat.wav"
REFERENCES_MANIFEST: Path = DATA_REFERENCES_DIR / "manifest.json"
VAL96_CENTROID_NPY: Path = DATA_REFERENCES_DIR / "val96_centroid.npy"
OLD_WRONGVOICE_CENTROID_NPY: Path = DATA_REFERENCES_DIR / "old_wrongvoice_centroid.npy"

# ── Read-only sources in t0018 (immutable, never modified; paths only, no cross-task code
# import — t0018 is not a registered library) ─────────────────────────────────
T0018_TASK_ROOT: Path = REPO_ROOT / "tasks" / "t0018_zero_shot_cloning_calibration"
T0018_COMPARISON_SET_DIR: Path = T0018_TASK_ROOT / "results" / "audio_samples" / "comparison_set"
T0018_HALF_A_CENTROID_NPY: Path = T0018_TASK_ROOT / "data" / "references" / "half_a_centroid.npy"

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
RESULTS_LATENCY_BREAKDOWN: Path = RESULTS_DIR / "latency_breakdown.json"

CHART_LATENCY_BREAKDOWN_STACKED: Path = RESULTS_IMAGES_DIR / "latency_breakdown_stacked.png"
CHART_TTFB_VS_SPEAKER_SIM_VARIANTS: Path = RESULTS_IMAGES_DIR / "ttfb_vs_speaker_sim_variants.png"
CHART_TTFB_P50_P95_BY_VARIANT: Path = RESULTS_IMAGES_DIR / "ttfb_p50_p95_by_variant.png"
