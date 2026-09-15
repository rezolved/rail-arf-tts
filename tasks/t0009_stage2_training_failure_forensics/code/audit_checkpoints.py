"""Step 11: Checkpoint map and audit (REQ-6).

Input:  data/log_inventory.json, data/configs/, data/timelines/
Output: data/checkpoint_map.json + results/checkpoint_audit.md
"""

import json
from dataclasses import asdict, dataclass

from tasks.t0009_stage2_training_failure_forensics.code.paths import (
    CHECKPOINT_AUDIT_MD,
    CHECKPOINT_MAP_JSON,
    RESULTS_DIR,
)


@dataclass(frozen=True, slots=True)
class CheckpointEntry:
    run_id: str
    filename: str
    epoch_0based: int
    epoch_1based: int
    val_loss: float | None
    is_pre_joint_epoch: bool | None
    sha256: str | None  # null — files not locally available
    notes: str


# Known checkpoints from plan, READMEs, and research
# Format: (run_id, filename, epoch_0based, val_loss_from_log_if_known, joint_epoch, notes)
KNOWN_CHECKPOINTS: list[tuple[str, str, int, float | None, int, str]] = [
    (
        "t0004_run01",
        "epoch_1st_00007.pth",
        7,
        None,  # Stage 1, not Stage 2 val_loss
        -1,
        "Stage 1 checkpoint. 0-based epoch 7. "
        "README incorrectly labels it 'epoch 10' — the filename is authoritative.",
    ),
    (
        "t0005_run06",
        "epoch_2nd_00003.pth",
        3,
        None,  # log deleted; val_loss unknown
        3,
        "t0005 best checkpoint. 0-based epoch 3 (4th epoch). "
        "README says 'epoch 2' — off by one; filename is authoritative. "
        "Top-2 pruning kept this; pre-joint_epoch checkpoints may have been pruned.",
    ),
    (
        "t0006_run03_v6c",
        "epoch_2nd_00003.pth",
        3,
        0.884,  # from run03 log: Validation loss at epoch 4
        6,
        "t0006 v6c best. 0-based epoch 3. val=0.884 from committed log. "
        "joint_epoch=6 → all v6c checkpoints are pre-joint_epoch (GAN not yet active). "
        "Top-2 pruning would delete epoch 0, 1, 2 or 4+ — all before GAN activated.",
    ),
    (
        "t0006_run03_v6c",
        "epoch_2nd_00005.pth",
        5,
        0.849,
        6,
        "v6c best by val_loss. 0-based epoch 5. val=0.849. Still pre-joint_epoch (5 < 6). "
        "This is the reference 'good' checkpoint but GAN never activated in this run.",
    ),
    (
        "t0006_run04_v6d",
        "epoch_2nd_00003.pth",
        3,
        0.846,
        6,
        "v6d best. 0-based epoch 3. val=0.846. "
        "Training stopped at epoch 10 (0-based) due to epochs_2nd=10 config bug "
        "(script reads epochs_2nd; config has epochs: 15, epochs_2nd: 10).",
    ),
]


def build_entries() -> list[CheckpointEntry]:
    entries: list[CheckpointEntry] = []
    for run_id, filename, epoch_0, val_loss, joint_epoch, notes in KNOWN_CHECKPOINTS:
        is_pre = (epoch_0 < joint_epoch) if joint_epoch >= 0 else None
        entries.append(
            CheckpointEntry(
                run_id=run_id,
                filename=filename,
                epoch_0based=epoch_0,
                epoch_1based=epoch_0 + 1,
                val_loss=val_loss,
                is_pre_joint_epoch=is_pre,
                sha256=None,
                notes=notes,
            )
        )
    return entries


def write_json(entries: list[CheckpointEntry]) -> None:
    CHECKPOINT_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_MAP_JSON.write_text(
        json.dumps(
            {
                "spec_version": "1",
                "task_id": "t0009_stage2_training_failure_forensics",
                "vm_note": "VM not accessible; SHA-256 hashes not computed",
                "checkpoints": [asdict(e) for e in entries],
            },
            indent=2,
        )
    )
    print(f"Wrote: {CHECKPOINT_MAP_JSON}")


def write_markdown(entries: list[CheckpointEntry]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Checkpoint Audit",
        "",
        "## Known Inconsistencies Resolved",
        "",
        "### 1. t0004 `epoch_1st_00007.pth` vs 'epoch 10' label",
        "",
        "The file is named `epoch_1st_00007.pth` → 0-based epoch 7 = 1-based epoch 8. "
        "The README labels this 'epoch 10' — incorrect. The filename is authoritative.",
        "",
        "### 2. t0005 best checkpoint `epoch_2nd_00003.pth` vs README 'epoch 2'",
        "",
        "The file is 0-based epoch 3 (the 4th epoch). README says 'epoch 2' "
        "(0-based epoch 1 by human count or 1-based). "
        "Off by one. Filename is authoritative: it is the 4th stage-2 epoch.",
        "",
        "### 3. t0006 `epochs_2nd: 10` bug",
        "",
        "The training script reads `config.epochs_2nd` (not `config.epochs`). "
        "v6d config has `epochs: 15, epochs_2nd: 10`. The script stopped at epoch 10 (0-based 9). "
        "The extra `epochs_2nd` key overrode the intended 15-epoch run.",
        "",
        "### 4. Top-2 val_loss pruning risk",
        "",
        "t0005 and t0006 prune checkpoints to keep only the top-2 by val_loss. "
        "For v6c (joint_epoch=6, 10 epochs), all saved checkpoints are pre-GAN. "
        "The pre-GAN epochs have low val_loss (easy reconstruction) and are most likely to be kept."
        " "
        "Post-joint_epoch checkpoints (where divergence first appears) would be pruned first. "
        "This means the last healthy checkpoint before divergence is the most at risk of deletion.",
        "",
        "## Checkpoint Map",
        "",
        "| run_id | file | epoch (0b) | epoch (1b) | val_loss | pre_joint_epoch | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in entries:
        val_str = f"{e.val_loss:.3f}" if e.val_loss is not None else "?"
        pre_str = str(e.is_pre_joint_epoch) if e.is_pre_joint_epoch is not None else "?"
        notes_short = e.notes[:80] + "..." if len(e.notes) > 80 else e.notes
        lines.append(
            f"| {e.run_id} | {e.filename} | {e.epoch_0based} | {e.epoch_1based} | "
            f"{val_str} | {pre_str} | {notes_short} |"
        )
    lines += [
        "",
        "## SHA-256 Hashes",
        "",
        "VM not accessible. SHA-256 hashes cannot be computed from local files.",
        "The `first_stage_v3.pth` checkpoint used in v6c/v6d is DVC-tracked in t0006.",
        "",
        "## Recommendation",
        "",
        "Switch to per-epoch checkpoint saves with the `CheckpointManager` (library asset). "
        "Never prune the last checkpoint with `is_pre_joint_epoch=True` — that is the last "
        "recovery point before the GAN activates.",
    ]
    CHECKPOINT_AUDIT_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {CHECKPOINT_AUDIT_MD}")


def main() -> None:
    print("=== audit_checkpoints.py: checkpoint map and audit ===")
    entries = build_entries()
    write_json(entries=entries)
    write_markdown(entries=entries)
    print(f"Done: {len(entries)} checkpoint entries.")


if __name__ == "__main__":
    main()
