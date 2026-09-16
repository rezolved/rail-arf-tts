"""Tests for ``is_present_locally_or_on_dvc_remote``.

Guards against the gap that let a task's model checkpoint's git history claim the task was
complete while the only copy of the weights sat on a since-stopped training VM: a committed
``.dvc`` pointer proves the file was tracked, not that ``dvc push`` ever ran.
"""

import subprocess
from pathlib import Path

import pytest

from arf.scripts.verificators.common.dvc_utils import is_present_locally_or_on_dvc_remote


def test_file_present_locally_short_circuits_without_calling_dvc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("subprocess.run must not be called when the file exists locally")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    tracked_path: Path = tmp_path / "model.pth"
    tracked_path.write_bytes(b"weights")

    assert is_present_locally_or_on_dvc_remote(tracked_path=tracked_path) is True


def test_no_local_file_and_no_dvc_pointer_is_not_present(tmp_path: Path) -> None:
    tracked_path: Path = tmp_path / "model.pth"

    assert is_present_locally_or_on_dvc_remote(tracked_path=tracked_path) is False


def test_dvc_pointer_reported_missing_from_remote_is_not_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_path: Path = tmp_path / "model.pth"
    (tmp_path / "model.pth.dvc").write_text("outs:\n- md5: abc\n", encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"\tmissing:            {tracked_path}\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_present_locally_or_on_dvc_remote(tracked_path=tracked_path) is False


def test_dvc_pointer_reported_deleted_locally_but_present_on_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_path: Path = tmp_path / "model.pth"
    (tmp_path / "model.pth.dvc").write_text("outs:\n- md5: abc\n", encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=f"\tdeleted:            {tracked_path}\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_present_locally_or_on_dvc_remote(tracked_path=tracked_path) is True


def test_dvc_status_command_failure_is_not_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_path: Path = tmp_path / "model.pth"
    (tmp_path / "model.pth.dvc").write_text("outs:\n- md5: abc\n", encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert is_present_locally_or_on_dvc_remote(tracked_path=tracked_path) is False
