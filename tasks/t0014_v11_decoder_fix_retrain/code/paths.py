"""Path constants for t0014 v11 decoder-init fix and retrain."""

from pathlib import Path

# Task root
TASK_ROOT: Path = Path(__file__).parent.parent
REPO_ROOT: Path = TASK_ROOT.parent.parent

# ── New first_stage_path target (REQ-1, REQ-2) ──────────────────────────────────
# Downloaded to this task's local scratch/VM persist storage (Lesson 10: never bare /mnt or /) --
# also present locally at this exact path on this dev machine because t0013 already cloned
# `semidark/kikiri-tts` (which bundles this checkpoint under StyleTTS2/Models/LibriTTS/) while
# validating it as a "known-good control", so the pre-flight tensor check below can run locally
# without re-downloading 771MB. On the VM, the same file is staged at
# /mnt/cache/persist/pretrained/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth (matches
# config_david_v11.yml's first_stage_path).
_T0013_KIKIRI_STYLETTS2: Path = (
    REPO_ROOT.parent
    / "t0013_v10_synthesis_quality_forensics"
    / "tasks"
    / "t0013_v10_synthesis_quality_forensics"
    / "code"
    / "kikiri-tts"
    / "StyleTTS2"
)
LIBRITTS_CONTROL_CKPT: Path = (
    _T0013_KIKIRI_STYLETTS2 / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
)
VM_LIBRITTS_CKPT: str = (
    "/mnt/cache/persist/pretrained/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"
)

# ── Reference checkpoint for the v10-bug before/after contrast (t0013's own forensics; not
# re-pulled here -- see results/checkpoint_forensics_v11.md for the citation instead of a 1.7GB
# re-download of first_stage_v3.pth just for restating an already-documented number) ─────────────
T0013_CHECKPOINT_FORENSICS_MD: Path = (
    REPO_ROOT
    / "tasks"
    / "t0013_v10_synthesis_quality_forensics"
    / "results"
    / "checkpoint_forensics.md"
)

# ── This task's own manifests ───────────────────────────────────────────────────
TASK_DATA_DIR: Path = TASK_ROOT / "data"
TRAIN_LIST_V11: Path = TASK_DATA_DIR / "train_list_v11_normalized.txt"
VAL_LIST_96: Path = REPO_ROOT / "data" / "v4" / "val_list.txt"

# ── Results ──────────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_AUDIO_DIR: Path = RESULTS_DIR / "audio_samples"
RESULTS_METRICS: Path = RESULTS_DIR / "metrics.json"
CHECKPOINT_FORENSICS_V11_MD: Path = RESULTS_DIR / "checkpoint_forensics_v11.md"
CHECKPOINT_FORENSICS_V11_RAW_JSON: Path = RESULTS_DIR / "checkpoint_forensics_v11_raw.json"
VAL96_LEAK_CHECK_TXT: Path = RESULTS_DIR / "val96_leak_check.txt"
V11_GATE_VERDICT_MD: Path = RESULTS_DIR / "v11_gate_verdict.md"
AUDIO_QUALITY_V11_JSON: Path = RESULTS_DIR / "audio_quality_v11.json"

# ── ElevenLabs David reference corpus (speaker_sim centroid, REQ-9) ────────────
ELEVENLABS_DAVID_DIR: Path = (
    REPO_ROOT / "tasks" / "t0008_tts_eval_harness_baselines" / "data" / "11labs_david"
)
