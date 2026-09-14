"""Step 6: Reconstruct per-launch configs from git history and log headers (REQ-3).

Output: data/configs/<run_id>.yml  for each recoverable config
"""

import shutil
from pathlib import Path

from tasks.t0009_stage2_training_failure_forensics.code.paths import (
    CONFIGS_DIR,
)

# Config files committed to git (cross-task read-only access)
_GIT_CONFIGS: list[tuple[str, Path]] = [
    (
        "t0005_run06",
        Path("tasks/t0005_kokoro_v5_stage2_train/code/config_david_v5_stage2.yml"),
    ),
    # t0006 configs: only run03 (v6c) has a recoverable config from code/
    # run04 (v6d) is reconstructed below based on README notes
]


def _write_reconstructed(run_id: str, content: str) -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = CONFIGS_DIR / f"{run_id}.yml"
    out.write_text(content)
    print(f"  Reconstructed: {out}")


def main() -> None:
    print("=== collect_configs.py: collect and reconstruct configs ===")
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy committed configs
    for run_id, src in _GIT_CONFIGS:
        if src.exists():
            dst = CONFIGS_DIR / f"{run_id}.yml"
            shutil.copy2(src, dst)
            print(f"  Copied: {run_id} <- {src}")
        else:
            print(f"  MISSING: {run_id} config at {src}")

    # t0006_run03_v6c config (from log header: lr=1e-4, joint_epoch=6, lambda_gen=1.0)
    _write_reconstructed(
        run_id="t0006_run03_v6c",
        content="""# Reconstructed from log header and task README (t0006_kokoro_v5_stage2_subset)
# Run: v6c — first_stage_v3.pth, lambda_gen=1.0, joint_epoch=6
loss_params:
  joint_epoch: 6
  lambda_gen: 1.0
  lambda_slm: 0.0
  lambda_dur: 1.0
  lambda_ce: 20.0
  lambda_F0: 1.0
  lambda_norm: 1.0
  lambda_mel: 5.0
optimizer_params:
  lr: 0.0001          # reverted to 1e-4 from t0005's 3e-5
  ft_lr: 0.0001
  bert_lr: 0.000001
epochs: 10
epochs_2nd: 10        # same key bug present
model_params:
  multispeaker: true
first_stage_path: first_stage_v3.pth
second_stage_load_pretrained: false
train_LM: false
""",
    )

    # t0006_run04_v6d config (from README: lambda_gen=0.05, otherwise same as v6c)
    _write_reconstructed(
        run_id="t0006_run04_v6d",
        content="""# Reconstructed from task README (t0006_kokoro_v5_stage2_subset)
# Run: v6d — first_stage_v3.pth, lambda_gen=0.05, joint_epoch=6
# ONLY 10 epochs ran (intended 15) due to epochs_2nd: 10 config bug
loss_params:
  joint_epoch: 6
  lambda_gen: 0.05    # KEY CHANGE from v6c; this is the isolating variable
  lambda_slm: 0.0
  lambda_dur: 1.0
  lambda_ce: 20.0
  lambda_F0: 1.0
  lambda_norm: 1.0
  lambda_mel: 5.0
optimizer_params:
  lr: 0.0001
  ft_lr: 0.0001
  bert_lr: 0.000001
epochs: 15
epochs_2nd: 10        # script reads epochs_2nd, not epochs — only 10 ran
model_params:
  multispeaker: true
first_stage_path: first_stage_v3.pth
second_stage_load_pretrained: false
train_LM: false
""",
    )

    # t0006_run01_v6a / run02_v6b (partially reconstructed from README)
    _write_reconstructed(
        run_id="t0006_run01_v6a",
        content="""# Reconstructed from task README (t0006_kokoro_v5_stage2_subset)
# Run: v6a — epoch_1st_00007.pth (multispeaker=false), lambda_gen=1.0
loss_params:
  joint_epoch: 3      # assumed (same as t0005 default)
  lambda_gen: 1.0
  lambda_slm: 0.0
optimizer_params:
  lr: 0.0001
  ft_lr: 0.0001
  bert_lr: 0.000001
model_params:
  multispeaker: false  # MISMATCH with stage1 checkpoint
first_stage_path: epoch_1st_00007.pth
""",
    )
    _write_reconstructed(
        run_id="t0006_run02_v6b",
        content="""# Reconstructed from task README (t0006_kokoro_v5_stage2_subset)
# Run: v6b — epoch_1st_00007.pth (multispeaker=false), lambda_gen=0.2
loss_params:
  joint_epoch: 3      # assumed
  lambda_gen: 0.2
  lambda_slm: 0.0
optimizer_params:
  lr: 0.0001
  ft_lr: 0.0001
  bert_lr: 0.000001
model_params:
  multispeaker: false  # MISMATCH
first_stage_path: epoch_1st_00007.pth
""",
    )

    print(f"\nConfigs written to: {CONFIGS_DIR}")
    configs = list(CONFIGS_DIR.glob("*.yml"))
    print(f"Total configs: {len(configs)}")
    for c in sorted(configs):
        print(f"  {c.name}")


if __name__ == "__main__":
    main()
