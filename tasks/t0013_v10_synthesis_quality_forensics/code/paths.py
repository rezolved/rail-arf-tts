"""Path constants for t0013 v10 checkpoint synthesis quality forensics."""

from pathlib import Path

# Task root
TASK_ROOT: Path = Path(__file__).parent.parent
REPO_ROOT: Path = TASK_ROOT.parent.parent

# ── Checkpoints under investigation (t0010) ─────────────────────────────────────
_T0010_TASK_DIR: Path = REPO_ROOT / "tasks" / "t0010_stage2_safeguarded_training"
T0010_RUN_V10_DIR: Path = _T0010_TASK_DIR / "data" / "run_v10"
V10_EPOCH16_CKPT: Path = T0010_RUN_V10_DIR / "epoch_2nd_00016.pth"  # primary ("best")
V10_EPOCH14_CKPT: Path = T0010_RUN_V10_DIR / "epoch_2nd_00014.pth"  # backup
V10_METRICS_JSONL: Path = T0010_RUN_V10_DIR / "metrics.jsonl"
V10_PER_EPOCH_CSV: Path = T0010_RUN_V10_DIR / "per_epoch_summary.csv"
V10_CONFIG_YML: Path = T0010_RUN_V10_DIR / "config_david_v10.yml"

# ── Reference checkpoints / corpora (read-only, other tasks) ───────────────────
FIRST_STAGE_V3_CKPT: Path = (
    REPO_ROOT
    / "tasks"
    / "t0006_kokoro_v5_stage2_subset"
    / "data"
    / "reference"
    / "v3"
    / "stage1"
    / "first_stage.pth"
)
ELEVENLABS_DAVID_DIR: Path = (
    REPO_ROOT / "tasks" / "t0008_tts_eval_harness_baselines" / "data" / "11labs_david"
)

# ── Inference harness (cloned + submodules, gitignored) ─────────────────────────
KIKIRI_TTS_DIR: Path = TASK_ROOT / "code" / "kikiri-tts"
STYLETTS2_DIR: Path = KIKIRI_TTS_DIR / "StyleTTS2"
STYLETTS2_VENV_PYTHON: Path = TASK_ROOT / "code" / ".venv-styletts2" / "bin" / "python"
CONTROL_CKPT: Path = STYLETTS2_DIR / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
CONTROL_REFERENCE_AUDIO_DIR: Path = STYLETTS2_DIR / "Demo" / "reference_audio"

# ── Results ──────────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_AUDIO_DIR: Path = RESULTS_DIR / "audio_samples"
RESULTS_METRICS: Path = RESULTS_DIR / "metrics.json"
CHECKPOINT_FORENSICS_MD: Path = RESULTS_DIR / "checkpoint_forensics.md"
CHECKPOINT_FORENSICS_RAW_JSON: Path = RESULTS_DIR / "checkpoint_forensics_raw.json"
CONTROL_TEST_MD: Path = RESULTS_DIR / "control_test.md"
V10_DIAGNOSIS_MD: Path = RESULTS_DIR / "v10_diagnosis.md"
