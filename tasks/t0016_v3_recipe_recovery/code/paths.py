"""Path constants for t0016 v3 recipe recovery."""

from pathlib import Path

# -- Task root --------------------------------------------------------------------
TASK_ROOT: Path = Path(__file__).parent.parent
REPO_ROOT: Path = TASK_ROOT.parent.parent

# -- v3 reference bundle (dependency: t0006_kokoro_v5_stage2_subset) ---------------
V3_REF_DIR: Path = (
    REPO_ROOT / "tasks" / "t0006_kokoro_v5_stage2_subset" / "data" / "reference" / "v3"
)
V3_STAGE1_CKPT: Path = V3_REF_DIR / "stage1" / "first_stage.pth"
V3_BEST_DECODER_CKPT: Path = V3_REF_DIR / "best" / "david_v3_best_decoder_kokoro.pth"
V3_BEST_VOICEPACK: Path = V3_REF_DIR / "best" / "david_v3_best_voicepack.pt"
V3_BEST_VOICEPACK_FT_ENC: Path = V3_REF_DIR / "best" / "david_v3_voicepack_ft_enc.pt"
V3_BEST_CONFIG: Path = V3_REF_DIR / "best" / "config.json"
V3_AUDIO_DIR: Path = V3_REF_DIR / "audio"
V3_TRAIN_SECOND_PATCH_DIFF: Path = V3_REF_DIR / "train_second_patch.diff"

# -- t0006 config template (dependency: t0006_kokoro_v5_stage2_subset) -------------
V6C_CONFIG_TEMPLATE: Path = (
    REPO_ROOT / "tasks" / "t0006_kokoro_v5_stage2_subset" / "code" / "config_david_v6c_stage2.yml"
)

# -- t0009 confound table / answer (dependency: t0009_stage2_training_failure_forensics)
T0009_CONFOUND_TABLE: Path = (
    REPO_ROOT
    / "tasks"
    / "t0009_stage2_training_failure_forensics"
    / "results"
    / "confound_table.md"
)
T0009_FULL_ANSWER: Path = (
    REPO_ROOT
    / "tasks"
    / "t0009_stage2_training_failure_forensics"
    / "assets"
    / "answer"
    / "t0009-stage2-forensics-answer"
    / "full_answer.md"
)

# -- ElevenLabs David reference corpus (dependency: t0008_tts_eval_harness_baselines)
ELEVENLABS_DAVID_DIR: Path = (
    REPO_ROOT / "tasks" / "t0008_tts_eval_harness_baselines" / "data" / "11labs_david"
)

# -- Project-level held-out val set ------------------------------------------------
VAL96_LIST: Path = REPO_ROOT / "data" / "v4" / "val_list.txt"

# -- VM inventory -------------------------------------------------------------------
VM_INVENTORY_DIR: Path = TASK_ROOT / "data" / "vm_inventory"
VM_INVENTORY_JSON: Path = VM_INVENTORY_DIR / "inventory.json"
VM_INVENTORY_KOKORO_FINETUNE_CURRENT_DIR: Path = VM_INVENTORY_DIR / "kokoro_finetune_current"

# -- This task's own data outputs ---------------------------------------------------
DATA_DIR: Path = TASK_ROOT / "data"
RECONSTRUCTED_CONFIG_YML: Path = DATA_DIR / "config_david_v3_reconstructed.yml"
TRAIN_LIST_266_TXT: Path = DATA_DIR / "v3_train_list_266.txt"
TRAIN_LIST_266_UNRECOVERED_MD: Path = DATA_DIR / "v3_train_list_UNRECOVERED.md"

# -- Results --------------------------------------------------------------------------
RESULTS_DIR: Path = TASK_ROOT / "results"
RESULTS_IMAGES_DIR: Path = RESULTS_DIR / "images"
CHECKPOINT_FORENSICS_MD: Path = RESULTS_DIR / "v3_checkpoint_forensics.md"
MODULE_WEIGHT_DELTA_RAW_JSON: Path = RESULTS_DIR / "v3_module_weight_delta.raw.json"
MODULE_WEIGHT_DELTA_PNG: Path = RESULTS_IMAGES_DIR / "v3_module_weight_delta.png"
METRICS_JSON: Path = RESULTS_DIR / "metrics.json"
LISTENING_GUIDE_MD: Path = RESULTS_DIR / "listening_guide.md"
RESULTS_AUDIO_DIR: Path = RESULTS_DIR / "audio_samples"
RESULTS_AUDIO_V3_SHIPPED_DIR: Path = RESULTS_AUDIO_DIR / "v3_shipped"
RESULTS_AUDIO_V3_PER_EPOCH_DIR: Path = RESULTS_AUDIO_DIR / "v3_per_epoch"
RESULTS_AUDIO_ELEVENLABS_REFERENCE_DIR: Path = RESULTS_AUDIO_DIR / "elevenlabs_reference"
