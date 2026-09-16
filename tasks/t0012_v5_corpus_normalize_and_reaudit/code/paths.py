"""Centralized path constants for t0012_v5_corpus_normalize_and_reaudit."""

from pathlib import Path

TASK_ROOT: Path = Path(__file__).parent.parent

# Task-level output directories
DATA_DIR: Path = TASK_ROOT / "data"
RESULTS_DIR: Path = TASK_ROOT / "results"
IMAGES_DIR: Path = RESULTS_DIR / "images"

# Manifests (relative to repo root)
V5_TRAIN_LIST: Path = Path("tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt")
VAL_96_LIST: Path = Path("data/v4/val_list.txt")  # read-only leak-check reference (REQ-19)

# t0011 outputs (read-only reference, t0011's folder is immutable)
T0011_FLAGGED_CLIPS_TXT: Path = Path("tasks/t0011_v5_data_quality_audit/data/flagged_clips.txt")
T0011_FLAG_COUNTS_JSON: Path = Path("tasks/t0011_v5_data_quality_audit/data/flag_counts.json")

# Normalized audio output (DVC-tracked)
V5_NORMALIZED_DIR: Path = DATA_DIR / "v5_normalized"

# Task output files
PER_CLIP_STATS_V2_JSONL: Path = DATA_DIR / "per_clip_stats_v2.jsonl"
FLAGGED_CLIPS_V2_TXT: Path = DATA_DIR / "flagged_clips_v2.txt"
CLEAN_MANIFEST_V2_TXT: Path = DATA_DIR / "train_list_v5_normalized_clean.txt"
FLAG_COUNTS_V2_JSON: Path = DATA_DIR / "flag_counts_v2.json"
ANALYSIS_V2_JSON: Path = DATA_DIR / "analysis_v2.json"

# Results outputs
METRICS_JSON: Path = RESULTS_DIR / "metrics.json"
