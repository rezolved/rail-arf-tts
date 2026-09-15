"""Path constants for t0010 Stage 2 safeguarded training."""

from pathlib import Path

# Task root
TASK_ROOT: Path = Path(__file__).parent.parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR: Path = TASK_ROOT / "data"
RUN_V10_DIR: Path = DATA_DIR / "run_v10"
RUN_V10_METRICS_JSONL: Path = RUN_V10_DIR / "metrics.jsonl"
RUN_V10_CHECKPOINT_MANIFEST: Path = RUN_V10_DIR / "checkpoint_manifest.json"
RUN_V10_LAUNCH_INFO: Path = RUN_V10_DIR / "launch_info.json"
RUN_V10_EVAL_RESULTS_DIR: Path = RUN_V10_DIR / "eval_results"
RUN_V10_PER_EPOCH_CSV: Path = RUN_V10_DIR / "per_epoch_summary.csv"
RUN_V10_CHECKPOINTS_DIR: Path = RUN_V10_DIR / "checkpoints"

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_METRICS: Path = RESULTS_DIR / "metrics.json"
RESULTS_IMAGES_DIR: Path = RESULTS_DIR / "images"
SPEAKER_SIM_CURVE: Path = RESULTS_IMAGES_DIR / "speaker_sim_curve.png"
LOSS_TIMELINE: Path = RESULTS_IMAGES_DIR / "loss_timeline.png"

# ── Assets ────────────────────────────────────────────────────────────────────
ASSETS_DIR: Path = TASK_ROOT / "assets"
MODEL_ASSET_DIR: Path = ASSETS_DIR / "model" / "kokoro-v10-best"
MODEL_ASSET_FILES_DIR: Path = MODEL_ASSET_DIR / "files"
