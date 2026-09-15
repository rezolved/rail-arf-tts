"""Run config capture utility (REQ-7, library component).

At launch, copies the resolved config YAML and current git SHA to:
  <log_dir>/<run_id>/config.yml
  <log_dir>/<run_id>/launch_info.json

Never overwrites an existing run directory; uses a timestamp-suffixed subdirectory
if a conflict is detected.
"""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def capture_run_config(
    config_path: Path,
    log_dir: Path,
    run_id: str,
    extra_info: dict[str, object] | None = None,
) -> Path:
    """Copy config and record launch metadata to a per-run subdirectory.

    Args:
        config_path: Path to the resolved config YAML file.
        log_dir: Root log directory.
        run_id: Run identifier (e.g. "v6e").
        extra_info: Optional additional metadata to include in launch_info.json.

    Returns:
        Path to the run subdirectory where config and launch info were written.
    """
    run_dir = log_dir / run_id
    # Avoid overwriting an existing run directory
    if run_dir.exists():
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = log_dir / f"{run_id}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy config
    dst_config = run_dir / "config.yml"
    if config_path.exists():
        dst_config.write_text(config_path.read_text())
    else:
        dst_config.write_text(f"# Config not found: {config_path}\n")

    # Write launch info
    launch_info: dict[str, object] = {
        "run_id": run_id,
        "launched_at": datetime.now(tz=UTC).isoformat(),
        "git_sha": _git_sha(),
        "config_source": str(config_path),
    }
    if extra_info is not None:
        launch_info.update(extra_info)

    (run_dir / "launch_info.json").write_text(json.dumps(launch_info, indent=2))
    print(f"[run_config] Launch info written to: {run_dir}")
    return run_dir
