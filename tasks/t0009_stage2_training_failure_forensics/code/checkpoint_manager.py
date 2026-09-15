"""Per-epoch checkpoint retention manager (REQ-7, library component).

Saves every epoch checkpoint, computes SHA-256, writes a JSON manifest.
NEVER deletes the last checkpoint with is_pre_joint_epoch=True.

Usage:
    mgr = CheckpointManager(
        log_dir=Path("logs/my_run"),
        run_id="v6e",
        joint_epoch=6,
    )
    # After each epoch:
    mgr.save(model=model, optimizer=optimizer, epoch=1, step=310, val_loss=1.643)
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    epoch: int  # 0-based
    step: int
    val_loss: float | None
    sha256: str
    path: str
    is_pre_joint_epoch: bool
    flagged_healthy: bool


class CheckpointManager:
    """Manages per-epoch checkpoints with safe retention policy.

    Args:
        log_dir: Root log directory for this run.
        run_id: Run identifier (used in manifest and filenames).
        joint_epoch: Epoch at which GAN losses activate (0-based).
    """

    def __init__(self, log_dir: Path, run_id: str, joint_epoch: int) -> None:
        self._run_dir = log_dir / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._joint_epoch = joint_epoch
        self._manifest_path = self._run_dir / "checkpoints.json"
        self._records: list[CheckpointRecord] = []

    def save(
        self,
        model: dict[str, object],
        optimizer: object,
        epoch: int,
        step: int,
        val_loss: float | None = None,
    ) -> str:
        """Save a checkpoint and update the manifest.

        Args:
            model: Model state dict (or Munch of sub-model dicts).
            optimizer: Optimizer state.
            epoch: Current epoch (0-based).
            step: Current global step.
            val_loss: Validation loss at this epoch, or None.

        Returns:
            Absolute path to the saved checkpoint file.
        """
        filename = f"epoch_{epoch:05d}.pth"
        ckpt_path = self._run_dir / filename
        payload = {
            "net": model,
            "optimizer": optimizer,
            "epoch": epoch,
            "iters": step,
            "val_loss": val_loss,
        }
        torch.save(payload, ckpt_path)
        sha = self._sha256(ckpt_path)
        is_pre = epoch < self._joint_epoch
        record = CheckpointRecord(
            epoch=epoch,
            step=step,
            val_loss=val_loss,
            sha256=sha,
            path=str(ckpt_path),
            is_pre_joint_epoch=is_pre,
            flagged_healthy=True,
        )
        self._records.append(record)
        self._write_manifest()
        return str(ckpt_path)

    def last_pre_joint_checkpoint(self) -> str | None:
        """Return path of the last checkpoint before joint_epoch, or None."""
        pre_joint = [r for r in self._records if r.is_pre_joint_epoch]
        if len(pre_joint) == 0:
            return None
        return pre_joint[-1].path

    def _write_manifest(self) -> None:
        manifest = {
            "spec_version": "1",
            "run_id": self._run_id,
            "joint_epoch": self._joint_epoch,
            "checkpoints": [asdict(r) for r in self._records],
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
