"""Presence checks for files that may be tracked (and pushed) via DVC instead of git."""

import subprocess
from pathlib import Path

from arf.scripts.verificators.common.paths import REPO_ROOT

_MISSING_MARKER: str = "missing:"
_DVC_STATUS_TIMEOUT_SECONDS: float = 60.0


def is_present_locally_or_on_dvc_remote(*, tracked_path: Path) -> bool:
    """Return True if ``tracked_path`` exists on disk or is confirmed pushed to the DVC remote.

    A committed ``.dvc`` pointer only proves a file was *tracked*, not that ``dvc push`` ever
    ran -- accepting the pointer's mere existence as proof of durability is exactly the gap
    that let a task's model checkpoint go un-backed-up while its git history claimed the task
    was complete. ``dvc status --cloud`` is the authoritative source: it reports the tracked
    path under "missing:" only when the remote itself does not have the object (as opposed to
    "deleted:", which means the remote has it but the local workspace doesn't -- the normal
    state for any file nobody has ``dvc pull``-ed on this machine).
    """
    if tracked_path.exists():
        return True

    dvc_pointer: Path = tracked_path.with_name(tracked_path.name + ".dvc")
    if not dvc_pointer.exists():
        return False

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["uv", "run", "dvc", "status", "--cloud", str(dvc_pointer)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=_DVC_STATUS_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    if result.returncode != 0:
        return False

    return _MISSING_MARKER not in result.stdout
