"""Path constants for t0015 v11 duration-blowup forensics and gate hardening."""

from pathlib import Path

# ── Task root ────────────────────────────────────────────────────────────────────
TASK_ROOT: Path = Path(__file__).parent.parent
REPO_ROOT: Path = TASK_ROOT.parent.parent

# ── This task's own code/data ───────────────────────────────────────────────────
CODE_DIR: Path = TASK_ROOT / "code"
KIKIRI_TTS_DIR: Path = CODE_DIR / "kikiri-tts"
STYLETTS2_DIR: Path = KIKIRI_TTS_DIR / "StyleTTS2"
TASK_DATA_DIR: Path = TASK_ROOT / "data"
REFERENCE_CONCAT_WAV: Path = TASK_DATA_DIR / "reference_concat.wav"

# ── kokoro-v11-best checkpoint (dependency: t0014_v11_decoder_fix_retrain) ──────
V11_CHECKPOINT: Path = (
    REPO_ROOT / "tasks" / "t0014_v11_decoder_fix_retrain" / "data" / "run_v11" / "epoch_00048.pth"
)
V11_CONFIG: Path = CODE_DIR / "config_david_v11.yml"

# ── LibriTTS control checkpoint (for predictor tensor-forensics cross-check,
# REQ-3) -- bundled inside this task's own fresh kikiri-tts clone (Step 1), no
# extra download needed beyond the submodule checkout. ─────────────────────────
LIBRITTS_CONTROL_CKPT: Path = STYLETTS2_DIR / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
LIBRITTS_CONTROL_CONFIG: Path = STYLETTS2_DIR / "Models" / "LibriTTS" / "config.yml"

# ── ElevenLabs David reference corpus (dependency: t0008_tts_eval_harness_baselines) ──
ELEVENLABS_DAVID_DIR: Path = (
    REPO_ROOT / "tasks" / "t0008_tts_eval_harness_baselines" / "data" / "11labs_david"
)
FILLER_PROMPTS_100_JSON: Path = (
    REPO_ROOT / "tasks" / "t0008_tts_eval_harness_baselines" / "data" / "filler_prompts_100.json"
)
VAL96_PROMPTS_JSON: Path = (
    REPO_ROOT / "tasks" / "t0008_tts_eval_harness_baselines" / "data" / "val96_prompts.json"
)

# ── Prior-task fixtures reused for the three-way regression (dependencies:
# t0013_v10_synthesis_quality_forensics, t0014_v11_decoder_fix_retrain) ─────────
T0013_V10_AUDIO_DIR: Path = (
    REPO_ROOT / "tasks" / "t0013_v10_synthesis_quality_forensics" / "results" / "audio_samples"
)
V10_EPOCH16_PRIMARY_WAV: Path = T0013_V10_AUDIO_DIR / "v10_epoch16_primary.wav"
T0013_CONTROL_WAV: Path = T0013_V10_AUDIO_DIR / "control_epochs_2nd_00020.wav"
T0014_V11_AUDIO_DIR: Path = (
    REPO_ROOT / "tasks" / "t0014_v11_decoder_fix_retrain" / "results" / "audio_samples"
)
V11_BEST_WAV: Path = T0014_V11_AUDIO_DIR / "ft" / "v11_best.wav"

# ── Results ──────────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_AUDIO_DIR: Path = RESULTS_DIR / "audio_samples"
RESULTS_CHARACTERIZATION_AUDIO_DIR: Path = RESULTS_AUDIO_DIR / "characterization"
RESULTS_METRICS_JSON: Path = RESULTS_DIR / "metrics.json"
DURATION_CHARACTERIZATION_JSON: Path = RESULTS_DIR / "duration_characterization.json"
PREDICTOR_TENSOR_FORENSICS_MD: Path = RESULTS_DIR / "predictor_tensor_forensics.md"
PARAM_SWEEP_JSON: Path = RESULTS_DIR / "param_sweep.json"
ASR_ROUNDTRIP_EVALUATION_MD: Path = RESULTS_DIR / "asr_roundtrip_evaluation.md"
GATE_REGRESSION_JSON: Path = RESULTS_DIR / "gate_regression.json"
DURATION_BLOWUP_DIAGNOSIS_MD: Path = RESULTS_DIR / "duration_blowup_diagnosis.md"
SPEAKER_SIM_SCORES_JSON: Path = RESULTS_DIR / "speaker_sim_scores.json"
