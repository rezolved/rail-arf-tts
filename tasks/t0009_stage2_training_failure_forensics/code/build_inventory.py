"""Step 3: Build log inventory table (REQ-1).

Input:  data/logs/ directory
Output: data/log_inventory.json + results/log_inventory.md
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tasks.t0009_stage2_training_failure_forensics.code.constants import (
    KNOWN_OUTCOMES,
    OUTCOME_UNKNOWN,
)
from tasks.t0009_stage2_training_failure_forensics.code.paths import (
    CONFIGS_DIR,
    LOG_INVENTORY_JSON,
    LOG_INVENTORY_MD,
    LOGS_DIR,
    RESULTS_DIR,
)


@dataclass(frozen=True, slots=True)
class RunEntry:
    run_id: str
    task: str
    log_present: bool
    config_present: bool
    stage1_ckpt_path: str | None
    stage1_ckpt_sha256: str | None
    data_list: str | None
    outcome: str


# Known run metadata (best effort from research, plan, and READMEs)
KNOWN_RUNS: list[dict[str, str | None]] = [
    {
        "run_id": "t0001_run01",
        "task": "t0001_kokoro_v4_stage2_finetune",
        "stage1_ckpt_path": "epoch_1st_00007.pth (via t0001)",
        "data_list": "v4 train list (~800 clips)",
        "note": "accelerate launch --num_processes 2 (DataParallel race)",
    },
    {
        "run_id": "t0004_run01",
        "task": "t0004_kokoro_v5_stage1_train",
        "stage1_ckpt_path": "epoch_1st_00007.pth (v4 Stage 1, 0-based epoch 7)",
        "data_list": "v4 train list",
        "note": "Stage 1 only; logged in t0004",
    },
    {
        "run_id": "t0005_run01",
        "task": "t0005_kokoro_v5_stage2_train",
        "stage1_ckpt_path": None,
        "data_list": "v5 train list",
        "note": "log deleted by launch script",
    },
    {
        "run_id": "t0005_run02",
        "task": "t0005_kokoro_v5_stage2_train",
        "stage1_ckpt_path": None,
        "data_list": "v5 train list",
        "note": "log deleted by launch script",
    },
    {
        "run_id": "t0005_run03",
        "task": "t0005_kokoro_v5_stage2_train",
        "stage1_ckpt_path": None,
        "data_list": "v5 train list",
        "note": "log deleted by launch script",
    },
    {
        "run_id": "t0005_run04",
        "task": "t0005_kokoro_v5_stage2_train",
        "stage1_ckpt_path": None,
        "data_list": "v5 train list",
        "note": "log deleted by launch script",
    },
    {
        "run_id": "t0005_run05",
        "task": "t0005_kokoro_v5_stage2_train",
        "stage1_ckpt_path": None,
        "data_list": "v5 train list",
        "note": "log deleted by launch script",
    },
    {
        "run_id": "t0005_run06",
        "task": "t0005_kokoro_v5_stage2_train",
        "stage1_ckpt_path": None,
        "data_list": "v5 train list (600+ clips, lambda_gen=1.0)",
        "note": "val_loss spike documented in README; no log",
    },
    {
        "run_id": "t0006_run01_v6a",
        "task": "t0006_kokoro_v5_stage2_subset",
        "stage1_ckpt_path": "epoch_1st_00007.pth (multispeaker=false)",
        "data_list": "250-clip subset, lambda_gen=1.0",
        "note": "diverged; log deleted",
    },
    {
        "run_id": "t0006_run02_v6b",
        "task": "t0006_kokoro_v5_stage2_subset",
        "stage1_ckpt_path": "epoch_1st_00007.pth (multispeaker=false; mismatch)",
        "data_list": "250-clip subset, lambda_gen=0.2",
        "note": "likely trained from scratch due to key mismatch; log deleted",
    },
    {
        "run_id": "t0006_run03_v6c",
        "task": "t0006_kokoro_v5_stage2_subset",
        "stage1_ckpt_path": "first_stage_v3.pth (multispeaker=true)",
        "data_list": "250-clip subset, lambda_gen=1.0, joint_epoch=6",
        "note": "committed log; val=0.849; config confirmed",
    },
    {
        "run_id": "t0006_run04_v6d",
        "task": "t0006_kokoro_v5_stage2_subset",
        "stage1_ckpt_path": "first_stage_v3.pth (multispeaker=true)",
        "data_list": "250-clip subset, lambda_gen=0.05, joint_epoch=6",
        "note": "10 epochs due to epochs_2nd bug; val=0.846; log deleted",
    },
    {
        "run_id": "v3",
        "task": "v3 (reference; not in this repo)",
        "stage1_ckpt_path": "first_stage_v3.pth (assumed)",
        "data_list": "v3 266-clip list (DVC-tracked in t0006)",
        "note": "val=0.506 at best epoch; train_second_patch.diff is only record",
    },
]


def _sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _log_dir_of(run_id: str) -> Path:
    return LOGS_DIR / run_id


def build_entries() -> list[RunEntry]:
    entries: list[RunEntry] = []
    for meta in KNOWN_RUNS:
        run_id: str = str(meta["run_id"])
        task: str = str(meta["task"])
        log_dir = _log_dir_of(run_id)
        log_present = any(log_dir.glob("*.log")) if log_dir.exists() else False
        config_path = CONFIGS_DIR / f"{run_id}.yml"
        config_present = config_path.exists()
        # SHA-256: only computable if checkpoint file is locally available — not the case here
        stage1_sha256: str | None = None
        entries.append(
            RunEntry(
                run_id=run_id,
                task=task,
                log_present=log_present,
                config_present=config_present,
                stage1_ckpt_path=meta.get("stage1_ckpt_path"),  # type: ignore[arg-type]
                stage1_ckpt_sha256=stage1_sha256,
                data_list=meta.get("data_list"),  # type: ignore[arg-type]
                outcome=KNOWN_OUTCOMES.get(run_id, OUTCOME_UNKNOWN),
            )
        )
    return entries


def write_json(entries: list[RunEntry]) -> None:
    LOG_INVENTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": "1",
        "task_id": "t0009_stage2_training_failure_forensics",
        "description": "Log inventory for all known Kokoro Stage 2 training runs",
        "vm_retrieval": "SSH to LLM-T1-NC80 timed out; /mnt/kikiri-tts/ not accessible",
        "runs": [asdict(e) for e in entries],
    }
    LOG_INVENTORY_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote: {LOG_INVENTORY_JSON}")


def write_markdown(entries: list[RunEntry]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Log Inventory",
        "",
        "All known Kokoro Stage 2 training runs across t0001, t0004, t0005, t0006, and the v3 reference.",  # noqa: E501
        "",
        "VM retrieval: SSH to LLM-T1-NC80 timed out — `/mnt/kikiri-tts/` not accessible.",
        "Logs for t0005 runs 1-5 and t0006 runs 1, 2, 4 are permanently deleted by the launch script.",  # noqa: E501
        "",
        "| run_id | task | log | config | stage1_ckpt | data_list | outcome |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in entries:
        log_col = "yes" if e.log_present else "no"
        cfg_col = "yes" if e.config_present else "no"
        ckpt_col = str(e.stage1_ckpt_path) if e.stage1_ckpt_path is not None else "unknown"
        data_col = str(e.data_list) if e.data_list is not None else "unknown"
        lines.append(
            f"| {e.run_id} | {e.task} | {log_col} | {cfg_col} | {ckpt_col} | {data_col} | {e.outcome} |"  # noqa: E501
        )
    lines += [
        "",
        "## SHA-256 Hashes",
        "",
        "No checkpoint files are locally available; SHA-256 hashes cannot be computed.",
        "Stage 1 checkpoint used in t0006_run03_v6c (`first_stage_v3.pth`) is DVC-tracked in t0006.",  # noqa: E501
        "",
        "## Summary",
        "",
        f"Total runs: {len(entries)}. Logs present: {sum(1 for e in entries if e.log_present)}. "
        f"Successful: {sum(1 for e in entries if e.outcome == 'success')}.",
    ]
    LOG_INVENTORY_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {LOG_INVENTORY_MD}")


def main() -> None:
    entries = build_entries()
    write_json(entries=entries)
    write_markdown(entries=entries)
    print(
        f"\nInventory: {len(entries)} runs, {sum(1 for e in entries if e.log_present)} with logs."
    )


if __name__ == "__main__":
    main()
