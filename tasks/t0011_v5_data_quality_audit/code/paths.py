"""Centralized path constants for t0011_v5_data_quality_audit."""

from pathlib import Path

TASK_ROOT: Path = Path(__file__).parent.parent

# Task-level output directories
DATA_DIR: Path = TASK_ROOT / "data"
RESULTS_DIR: Path = TASK_ROOT / "results"
IMAGES_DIR: Path = RESULTS_DIR / "images"

# Manifests (committed, from t0003 — relative to repo root)
V5_TRAIN_LIST: Path = Path("tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt")
V5_VAL_LIST: Path = Path("tasks/t0003_kokoro_v5_phoneme_data/results/v5/val_list.txt")
VAL_96_LIST: Path = Path("data/v4/val_list.txt")

# Task output files
PER_CLIP_STATS_JSONL: Path = DATA_DIR / "per_clip_stats.jsonl"
VAL_CLIP_STATS_JSONL: Path = DATA_DIR / "val_clip_stats.jsonl"
FLAGGED_CLIPS_TXT: Path = DATA_DIR / "flagged_clips.txt"
CLEAN_MANIFEST_TXT: Path = DATA_DIR / "train_list_v5_clean.txt"
DISTRIBUTION_STATS_JSON: Path = DATA_DIR / "distribution_stats.json"
FLAG_COUNTS_JSON: Path = DATA_DIR / "flag_counts.json"

# Results outputs
METRICS_JSON: Path = RESULTS_DIR / "metrics.json"
