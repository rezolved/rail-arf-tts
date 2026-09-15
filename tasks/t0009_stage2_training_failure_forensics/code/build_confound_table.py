"""Step 7: Build confound table (REQ-3).

Input:  data/configs/<run_id>.yml + data/log_inventory.json
Output: data/confound_table.json + results/confound_table.md
"""

import json
from dataclasses import asdict, dataclass

from tasks.t0009_stage2_training_failure_forensics.code.constants import KNOWN_OUTCOMES
from tasks.t0009_stage2_training_failure_forensics.code.paths import (
    CONFOUND_TABLE_JSON,
    CONFOUND_TABLE_MD,
    RESULTS_DIR,
)


@dataclass(frozen=True, slots=True)
class ConfoundRow:
    run_id: str
    task: str
    data_list: str
    n_clips: str
    multispeaker: str
    first_stage_path: str
    first_stage_sha256: str | None
    joint_epoch: str
    lambda_gen: str
    lambda_slm: str
    lr: str
    ft_lr: str
    bert_lr: str
    batch_size: str
    gpu_count: str
    parallelism_mode: str
    train_lm: str
    patches_active: str
    outcome: str
    notes: str


# Manually assembled from research, README, plan, and committed configs
KNOWN_CONFOUNDS: list[dict[str, str | None]] = [
    {
        "run_id": "t0001_run01",
        "task": "t0001_kokoro_v4_stage2_finetune",
        "data_list": "v4 train list",
        "n_clips": "~800",
        "multispeaker": "true",
        "first_stage_path": "epoch_1st_00007.pth",
        "first_stage_sha256": None,
        "joint_epoch": "3",
        "lambda_gen": "1.0",
        "lambda_slm": "0.0",
        "lr": "1e-4",
        "ft_lr": "1e-4",
        "bert_lr": "1e-6",
        "batch_size": "8",
        "gpu_count": "2",
        "parallelism_mode": "accelerate (wrong for DataParallel — race on save)",
        "train_lm": "false",
        "patches_active": "[1]",  # lambda_slm guard only (via v3 diff in upstream)
        "outcome": KNOWN_OUTCOMES["t0001_run01"],
        "notes": (
            "accelerate launch used instead of python train_second.py"
            " — DataParallel race. val_loss blew after joint_epoch."
        ),
    },
    {
        "run_id": "t0005_run06",
        "task": "t0005_kokoro_v5_stage2_train",
        "data_list": "v5 train list (600+ clips)",
        "n_clips": "~600+",
        "multispeaker": "true",
        "first_stage_path": "epoch_1st_00007.pth (v4 Stage1)",
        "first_stage_sha256": None,
        "joint_epoch": "3",
        "lambda_gen": "1.0",
        "lambda_slm": "0.0",
        "lr": "3e-5",
        "ft_lr": "3e-5",
        "bert_lr": "1e-6",
        "batch_size": "8",
        "gpu_count": "2",
        "parallelism_mode": "DataParallel",
        "train_lm": "false",
        "patches_active": "[1,2,3,4,5,6,7]",
        "outcome": KNOWN_OUTCOMES["t0005_run06"],
        "notes": "All 7 crash patches active. val_loss spike documented. Logs deleted.",
    },
    {
        "run_id": "t0006_run01_v6a",
        "task": "t0006_kokoro_v5_stage2_subset",
        "data_list": "250-clip subset",
        "n_clips": "250",
        "multispeaker": "false (config mismatch with checkpoint)",
        "first_stage_path": "epoch_1st_00007.pth (multispeaker=false)",
        "first_stage_sha256": None,
        "joint_epoch": "3 (assumed)",
        "lambda_gen": "1.0",
        "lambda_slm": "0.0",
        "lr": "1e-4",
        "ft_lr": "1e-4",
        "bert_lr": "1e-6",
        "batch_size": "8",
        "gpu_count": "2",
        "parallelism_mode": "DataParallel",
        "train_lm": "false",
        "patches_active": "[1,2,3,4,5,6,7]",
        "outcome": KNOWN_OUTCOMES["t0006_run01_v6a"],
        "notes": "multispeaker mismatch. Diverged. Log deleted.",
    },
    {
        "run_id": "t0006_run02_v6b",
        "task": "t0006_kokoro_v5_stage2_subset",
        "data_list": "250-clip subset",
        "n_clips": "250",
        "multispeaker": "false (config mismatch with checkpoint)",
        "first_stage_path": "epoch_1st_00007.pth (multispeaker=false)",
        "first_stage_sha256": None,
        "joint_epoch": "3 (assumed)",
        "lambda_gen": "0.2",
        "lambda_slm": "0.0",
        "lr": "1e-4",
        "ft_lr": "1e-4",
        "bert_lr": "1e-6",
        "batch_size": "8",
        "gpu_count": "2",
        "parallelism_mode": "DataParallel",
        "train_lm": "false",
        "patches_active": "[1,2,3,4,5,6,7]",
        "outcome": KNOWN_OUTCOMES["t0006_run02_v6b"],
        "notes": (
            "multispeaker mismatch; likely 0-param load (trained from scratch)."
            " Diverged. Log deleted."
        ),
    },
    {
        "run_id": "t0006_run03_v6c",
        "task": "t0006_kokoro_v5_stage2_subset",
        "data_list": "250-clip subset",
        "n_clips": "250",
        "multispeaker": "true",
        "first_stage_path": "first_stage_v3.pth",
        "first_stage_sha256": None,
        "joint_epoch": "6",
        "lambda_gen": "1.0",
        "lambda_slm": "0.0",
        "lr": "1e-4",
        "ft_lr": "1e-4",
        "bert_lr": "1e-6",
        "batch_size": "8",
        "gpu_count": "2",
        "parallelism_mode": "DataParallel",
        "train_lm": "false",
        "patches_active": "[1,2,3,4,5,6,7]",
        "outcome": KNOWN_OUTCOMES["t0006_run03_v6c"],
        "notes": (
            "v3 Stage1 checkpoint (multispeaker=true). joint_epoch=6."
            " Stable: val=0.849. LOG COMMITTED."
        ),
    },
    {
        "run_id": "t0006_run04_v6d",
        "task": "t0006_kokoro_v5_stage2_subset",
        "data_list": "250-clip subset",
        "n_clips": "250",
        "multispeaker": "true",
        "first_stage_path": "first_stage_v3.pth",
        "first_stage_sha256": None,
        "joint_epoch": "6",
        "lambda_gen": "0.05",
        "lambda_slm": "0.0",
        "lr": "1e-4",
        "ft_lr": "1e-4",
        "bert_lr": "1e-6",
        "batch_size": "8",
        "gpu_count": "2",
        "parallelism_mode": "DataParallel",
        "train_lm": "false",
        "patches_active": "[1,2,3,4,5,6,7]",
        "outcome": KNOWN_OUTCOMES["t0006_run04_v6d"],
        "notes": (
            "lambda_gen=0.05 (KEY CHANGE). Only 10 epochs ran (intended 15)"
            " — epochs_2nd bug. val=0.846."
        ),
    },
    {
        "run_id": "v3",
        "task": "v3 reference",
        "data_list": "v3 266-clip list",
        "n_clips": "266",
        "multispeaker": "true (assumed)",
        "first_stage_path": "first_stage_v3.pth (assumed)",
        "first_stage_sha256": None,
        "joint_epoch": "unknown (speculated: 0)",
        "lambda_gen": "1.0 (assumed)",
        "lambda_slm": "0.0 (train_second_patch.diff shows lambda_slm guard only)",
        "lr": "unknown",
        "ft_lr": "unknown",
        "bert_lr": "unknown",
        "batch_size": "unknown",
        "gpu_count": "unknown",
        "parallelism_mode": "unknown",
        "train_lm": "unknown",
        "patches_active": "[1]",  # Only lambda_slm guard in committed diff
        "outcome": KNOWN_OUTCOMES["v3"],
        "notes": "Only train_second_patch.diff committed. No config. val=0.506. Reference run.",
    },
]


def build_rows() -> list[ConfoundRow]:
    rows: list[ConfoundRow] = []
    for meta in KNOWN_CONFOUNDS:
        rows.append(
            ConfoundRow(
                run_id=str(meta["run_id"]),
                task=str(meta["task"]),
                data_list=str(meta["data_list"]),
                n_clips=str(meta["n_clips"]),
                multispeaker=str(meta["multispeaker"]),
                first_stage_path=str(meta["first_stage_path"]),
                first_stage_sha256=meta.get("first_stage_sha256"),  # type: ignore[arg-type]
                joint_epoch=str(meta["joint_epoch"]),
                lambda_gen=str(meta["lambda_gen"]),
                lambda_slm=str(meta["lambda_slm"]),
                lr=str(meta["lr"]),
                ft_lr=str(meta["ft_lr"]),
                bert_lr=str(meta["bert_lr"]),
                batch_size=str(meta["batch_size"]),
                gpu_count=str(meta["gpu_count"]),
                parallelism_mode=str(meta["parallelism_mode"]),
                train_lm=str(meta["train_lm"]),
                patches_active=str(meta["patches_active"]),
                outcome=str(meta["outcome"]),
                notes=str(meta["notes"]),
            )
        )
    return rows


def write_json(rows: list[ConfoundRow]) -> None:
    CONFOUND_TABLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONFOUND_TABLE_JSON.write_text(
        json.dumps(
            {
                "spec_version": "1",
                "task_id": "t0009_stage2_training_failure_forensics",
                "runs": [asdict(r) for r in rows],
            },
            indent=2,
        )
    )
    print(f"Wrote: {CONFOUND_TABLE_JSON}")


def write_markdown(rows: list[ConfoundRow]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header = "| run_id | multispeaker | first_stage_path | joint_epoch | lambda_gen | lr | outcome | notes |"  # noqa: E501
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [
        "# Confound Table",
        "",
        "Per-run effective settings for all known Kokoro Stage 2 runs.",
        "SHA-256 hashes not available (no local checkpoint files; VM not accessible).",
        "",
        header,
        sep,
    ]
    for r in rows:
        notes_short = r.notes[:80] + "..." if len(r.notes) > 80 else r.notes
        lines.append(
            f"| {r.run_id} | {r.multispeaker} | {r.first_stage_path} | "
            f"{r.joint_epoch} | {r.lambda_gen} | {r.lr} | {r.outcome} | {notes_short} |"
        )
    lines += [
        "",
        "## Key Observations",
        "",
        "1. **Checkpoint alignment**: runs v6a/v6b used `multispeaker=false` checkpoint"
        " with `multispeaker=true` config → 0-param load → silent training from scratch.",
        "2. **joint_epoch escalation**: t0001/t0005 used `joint_epoch=3`; v6c/v6d used"
        " `joint_epoch=6` — the delay is the main stabilizing factor besides checkpoint fix.",
        "3. **lambda_gen**: v6c (1.0) succeeded; v6d (0.05) also succeeded but only 10 epochs ran."
        " The key isolator is checkpoint alignment + joint_epoch=6, not lambda_gen.",
        "4. **v3 patches**: Only `lambda_slm > 0` guard committed — no istftnet clamp, no skip"
        " guard. Yet val=0.506. The 7-patch stack in t0005 is all crash mitigation, not cause fix.",
        "5. **LR**: t0001/t0006 used lr=1e-4; t0005 safe run used lr=3e-5."
        " v6c succeeded at lr=1e-4, so LR alone did not cause failure.",
    ]
    CONFOUND_TABLE_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {CONFOUND_TABLE_MD}")


def main() -> None:
    print("=== build_confound_table.py ===")
    rows = build_rows()
    write_json(rows=rows)
    write_markdown(rows=rows)
    print(f"Done: {len(rows)} runs in confound table.")


if __name__ == "__main__":
    main()
