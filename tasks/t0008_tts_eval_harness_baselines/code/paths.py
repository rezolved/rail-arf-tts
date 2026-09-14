"""Path constants for t0008 TTS evaluation harness."""

from pathlib import Path

# Task root (tasks/t0008_tts_eval_harness_baselines/)
TASK_ROOT: Path = Path(__file__).parent.parent

# ── Input data ────────────────────────────────────────────────────────────────
DATA_DIR: Path = TASK_ROOT / "data"
DATA_11LABS_DAVID_DIR: Path = DATA_DIR / "11labs_david"
DATA_PACKAGED_DIR: Path = DATA_DIR / "packaged"
DATA_VAL96_PROMPTS: Path = DATA_DIR / "val96_prompts.json"
DATA_FILLER_PROMPTS: Path = DATA_DIR / "filler_prompts_100.json"

# ── Cross-task checkpoint sources (DVC-tracked) ───────────────────────────────
REPO_ROOT: Path = TASK_ROOT.parent.parent  # repo root
T0005_CHECKPOINT_DVC: Path = (
    REPO_ROOT
    / "tasks"
    / "t0005_kokoro_v5_stage2_train"
    / "results"
    / "checkpoints"
    / "epoch_2nd_00003.pth.dvc"
)
T0005_CHECKPOINT: Path = T0005_CHECKPOINT_DVC.with_suffix("")

T0006_V6D_CHECKPOINT_DVC: Path = (
    REPO_ROOT
    / "tasks"
    / "t0006_kokoro_v5_stage2_subset"
    / "results"
    / "checkpoints"
    / "v6d"
    / "epoch_2nd_00006.pth.dvc"
)
T0006_V6D_CHECKPOINT: Path = T0006_V6D_CHECKPOINT_DVC.with_suffix("")

V3_BEST_DIR: Path = (
    REPO_ROOT / "tasks" / "t0006_kokoro_v5_stage2_subset" / "data" / "reference" / "v3" / "best"
)
V3_DECODER: Path = V3_BEST_DIR / "david_v3_best_decoder_kokoro.pth"
V3_VOICEPACK: Path = V3_BEST_DIR / "david_v3_best_voicepack.pt"

VAL_LIST: Path = (
    REPO_ROOT / "tasks" / "t0003_kokoro_v5_phoneme_data" / "results" / "v5" / "val_list.txt"
)

# ── Packaged checkpoint outputs ───────────────────────────────────────────────
T0005_PACKAGED: Path = DATA_PACKAGED_DIR / "t0005_run06_epoch3.pth"
T0006_V6D_PACKAGED: Path = DATA_PACKAGED_DIR / "t0006_v6d_epoch6.pth"

# ── Results ───────────────────────────────────────────────────────────────────
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_IMAGES_DIR: Path = RESULTS_DIR / "images"
RESULTS_METADATA: Path = RESULTS_DIR / "metadata.json"
RESULTS_PER_CLIP_METRICS: Path = RESULTS_DIR / "per_clip_metrics.json"
RESULTS_PER_CLIP_ELEVENLABS: Path = RESULTS_DIR / "per_clip_metrics_elevenlabs.json"
RESULTS_PER_CLIP_KOKORO: Path = RESULTS_DIR / "per_clip_metrics_kokoro.json"
RESULTS_METRICS: Path = RESULTS_DIR / "metrics.json"
RESULTS_TABLES: Path = RESULTS_DIR / "tables.json"

CHART_SPEAKER_SIM_BOXPLOT: Path = RESULTS_IMAGES_DIR / "speaker_sim_boxplot.png"
CHART_TTFB_CDF: Path = RESULTS_IMAGES_DIR / "ttfb_cdf.png"
CHART_SPEAKER_SIM_WER_SCATTER: Path = RESULTS_IMAGES_DIR / "speaker_sim_wer_scatter.png"

# ── Synthesized audio output ──────────────────────────────────────────────────
SYNTH_AUDIO_DIR: Path = DATA_DIR / "synth_audio"

# ── Centroid cache (for speaker_sim) ─────────────────────────────────────────
REFERENCE_CENTROID_NPY: Path = DATA_DIR / "reference_centroid.npy"
HALF_B_PATHS_JSON: Path = DATA_DIR / "reference_half_b_paths.json"
