"""Centralized path constants for t0009_stage2_training_failure_forensics."""

from pathlib import Path

TASK_ROOT = Path(__file__).parent.parent
CODE_DIR = TASK_ROOT / "code"

# Data directories
DATA_DIR = TASK_ROOT / "data"
LOGS_DIR = DATA_DIR / "logs"
TIMELINES_DIR = DATA_DIR / "timelines"
CONFIGS_DIR = DATA_DIR / "configs"
REFERENCE_DIR = DATA_DIR / "reference"
V3_REFERENCE_DIR = REFERENCE_DIR / "v3"

# Specific source files (git-committed)
V3_PATCH_DIFF = V3_REFERENCE_DIR / "train_second_patch.diff"
RUN03_LOG = LOGS_DIR / "t0006_run03_v6c" / "stage2.log"

# Inventory / audit outputs
LOG_INVENTORY_JSON = DATA_DIR / "log_inventory.json"
CONFOUND_TABLE_JSON = DATA_DIR / "confound_table.json"
DATA_AUDIT_JSON = DATA_DIR / "data_audit.json"
PIPELINE_AUDIT_MD = DATA_DIR / "pipeline_audit.md"
CHECKPOINT_MAP_JSON = DATA_DIR / "checkpoint_map.json"

# Results
RESULTS_DIR = TASK_ROOT / "results"
IMAGES_DIR = RESULTS_DIR / "images"
LOG_INVENTORY_MD = RESULTS_DIR / "log_inventory.md"
CONFOUND_TABLE_MD = RESULTS_DIR / "confound_table.md"
DATA_AUDIT_SUMMARY_MD = RESULTS_DIR / "data_audit_summary.md"
CHECKPOINT_AUDIT_MD = RESULTS_DIR / "checkpoint_audit.md"
METRICS_JSON = RESULTS_DIR / "metrics.json"

# Asset directories
ASSETS_DIR = TASK_ROOT / "assets"
ANSWER_ASSET_DIR = ASSETS_DIR / "answer" / "t0009-stage2-forensics-answer"
LIBRARY_ASSET_DIR = ASSETS_DIR / "library" / "t0009_training_safeguards"

# Cross-task source files (read-only; do not commit from here)
T0001_TRAIN = Path("tasks/t0001_kokoro_v4_stage2_finetune/code/train_second_patched.py")
T0005_TRAIN = Path("tasks/t0005_kokoro_v5_stage2_train/code/train_second_patched.py")
T0003_PREPARE = Path("tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py")
T0003_TEST_GATES = Path("tasks/t0003_kokoro_v5_phoneme_data/code/test_gates.py")

# v5 data lists (produced by t0003 — present in repo)
V5_TRAIN_LIST = Path("data/v5/train_list.txt")
V5_VAL_LIST = Path("data/v5/val_list.txt")
VAL_96_LIST = Path("data/v4/val_list.txt")
