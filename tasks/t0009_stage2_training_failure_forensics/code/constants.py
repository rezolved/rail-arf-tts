"""Named constants for t0009_stage2_training_failure_forensics."""

# Log parser field names (CSV column names)
COL_RUN_ID = "run_id"
COL_EPOCH = "epoch"
COL_STEP = "step"
COL_LOSS_TOTAL = "loss_total"
COL_DISC_LOSS = "disc_loss"
COL_DUR_LOSS = "dur_loss"
COL_CE_LOSS = "ce_loss"
COL_MEL_LOSS = "mel_loss"
COL_F0_LOSS = "f0_loss"
COL_VAL_LOSS = "val_loss"
COL_ACOUSTIC_NORM = "acoustic_norm"
COL_GRAD_NORM_MSD = "grad_norm_msd"
COL_GRAD_NORM_MPD = "grad_norm_mpd"
COL_GRAD_NORM_DECODER = "grad_norm_decoder"
COL_GRAD_NORM_STYLE_ENCODER = "grad_norm_style_encoder"
COL_SKIP_COUNT = "skip_count"

# All CSV columns in order
TIMELINE_COLUMNS = [
    COL_RUN_ID,
    COL_EPOCH,
    COL_STEP,
    COL_LOSS_TOTAL,
    COL_DISC_LOSS,
    COL_DUR_LOSS,
    COL_CE_LOSS,
    COL_MEL_LOSS,
    COL_F0_LOSS,
    COL_VAL_LOSS,
    COL_ACOUSTIC_NORM,
    COL_GRAD_NORM_MSD,
    COL_GRAD_NORM_MPD,
    COL_GRAD_NORM_DECODER,
    COL_GRAD_NORM_STYLE_ENCODER,
    COL_SKIP_COUNT,
]

# Health gate thresholds (from plan.md)
DUR_LOSS_STEP1_MAX: float = 2.0  # dur_loss at first step of epoch 2+ must be < 2.0
ACOUSTIC_NORM_MAX: float = 20.0  # acoustic_norm after any epoch < 20
VAL_SPIKE_MAX: float = 0.05  # val_loss increase post-joint_epoch <= 0.05
CONSECUTIVE_SKIP_MAX: int = 50  # consecutive NaN skips <= 50

# Known joint_epoch values per run (from config files and research)
JOINT_EPOCHS: dict[str, int] = {
    "t0006_run03_v6c": 6,  # config_david_v6c: joint_epoch=6
    "v3_synthetic": 0,  # v3 unknown; speculated 0 (GAN from epoch 1)
}

# Run outcome labels
OUTCOME_SUCCESS = "success"
OUTCOME_DIVERGED = "diverged"
OUTCOME_CRASHED = "crashed"
OUTCOME_UNKNOWN = "unknown"
OUTCOME_INCOMPLETE = "incomplete"

# Known run outcomes (from research_code.md and plan)
KNOWN_OUTCOMES: dict[str, str] = {
    "t0001_run01": OUTCOME_DIVERGED,  # val_loss blew up after joint_epoch
    "t0004_run01": OUTCOME_INCOMPLETE,  # only 8 epochs, not conclusive
    "t0005_run01": OUTCOME_UNKNOWN,  # no log
    "t0005_run02": OUTCOME_UNKNOWN,  # no log
    "t0005_run03": OUTCOME_UNKNOWN,  # no log
    "t0005_run04": OUTCOME_UNKNOWN,  # no log
    "t0005_run05": OUTCOME_UNKNOWN,  # no log
    "t0005_run06": OUTCOME_DIVERGED,  # val_loss spike documented in README
    "t0006_run01_v6a": OUTCOME_DIVERGED,  # lambda_gen=1.0, diverged
    "t0006_run02_v6b": OUTCOME_DIVERGED,  # multispeaker mismatch; likely trained from scratch
    # val=0.849 at epoch 6 (best); diverged after ep8 (gates SHOULD fire after best ckpt)
    "t0006_run03_v6c": OUTCOME_SUCCESS,
    "t0006_run04_v6d": OUTCOME_INCOMPLETE,  # 10 epochs only (intended 15 due to config bug)
    "v3": OUTCOME_SUCCESS,  # val=0.506, the reference run
}
