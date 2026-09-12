"""Centralized path constants for t0003 (v5 phoneme manifest regeneration)."""

from pathlib import Path

TASK_DIR = Path(__file__).parent.parent
RESULTS_DIR = TASK_DIR / "results"
V5_DIR = RESULTS_DIR / "v5"

# Source corpus lives in the sibling rail-benchmarks repo. Wav files there are intact;
# only the v4 *_list.txt manifests were corrupted, so v5 regenerates text only.
BENCHMARKS_ROOT = TASK_DIR.parents[2] / "rail-benchmarks" / "kokoro-finetune"
BENCHMARKS_DATA = BENCHMARKS_ROOT / "data"

V3_TRAIN_MANIFEST = BENCHMARKS_DATA / "train" / "manifest.csv"
V3_VAL_MANIFEST = BENCHMARKS_DATA / "val" / "manifest.csv"
FILLERS_FILE = BENCHMARKS_DATA / "fillers_from_logs.txt"

# Existing v4 wav directories define the train/val split; v5 reuses them verbatim.
V4_TRAIN_WAVS = BENCHMARKS_DATA / "v4" / "train" / "wavs"
V4_VAL_WAVS = BENCHMARKS_DATA / "v4" / "val" / "wavs"

TRAIN_LIST_OUT = V5_DIR / "train_list.txt"
VAL_LIST_OUT = V5_DIR / "val_list.txt"
REJECTS_OUT = V5_DIR / "rejects.txt"
