"""Step 1: Collect surviving logs from git and record VM retrieval attempt.

Input:  Git-committed paths known from research
Output: Files in data/logs/<run_id>/  and  data/reference/v3/
"""

import shutil
import sys
from pathlib import Path

from tasks.t0009_stage2_training_failure_forensics.code.paths import (
    LOGS_DIR,
    V3_PATCH_DIFF,
    V3_REFERENCE_DIR,
)


def _copy_if_exists(src: Path, dst: Path, label: str) -> bool:
    if not src.exists():
        print(f"  MISSING: {label} at {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  OK: {label} -> {dst}")
    return True


def main() -> None:
    print("=== collect_logs.py: git-committed log collection ===")

    # Source paths (relative to repo root, where this is run from)
    src_log = Path("tasks/t0006_kokoro_v5_stage2_subset/logs/run03_v6c_v3_stage1.log")
    src_diff = Path("tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    V3_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Copy the one committed log
    run03_dir = LOGS_DIR / "t0006_run03_v6c"
    run03_dir.mkdir(parents=True, exist_ok=True)
    dst_log = run03_dir / "stage2.log"
    ok_log = _copy_if_exists(src=src_log, dst=dst_log, label="run03_v6c stage2.log")

    # Copy v3 patch diff
    ok_diff = _copy_if_exists(src=src_diff, dst=V3_PATCH_DIFF, label="v3 train_second_patch.diff")

    print()
    print("VM retrieval attempt:")
    print(
        "  SSH to LLM-T1-NC80 timed out (VM stopped). "
        "/mnt/kikiri-tts/ not accessible. Logs for t0005 runs 1-5 and t0001 are not recoverable."
    )
    print()
    print(f"Result: {int(ok_log)} log file copied, {int(ok_diff)} diff file copied")
    print("Logs available: t0006_run03_v6c only.")

    if not ok_log:
        print("ERROR: committed log not found — check path", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
