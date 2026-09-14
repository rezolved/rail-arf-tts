"""JSONL metrics logger for Kokoro Stage 2 training (REQ-7, library component).

Appends one JSON record per training step or validation pass to <log_dir>/metrics.jsonl.
Designed to be imported by train_second_safeguarded.py.
"""

import json
from datetime import UTC, datetime
from pathlib import Path


class StepLogger:
    """Appends one JSONL record per training step or validation pass.

    Args:
        log_dir: Directory where metrics.jsonl is written.
        run_id: Identifier for this run (used in records).
    """

    def __init__(self, log_dir: Path, run_id: str) -> None:
        self._path = log_dir / "metrics.jsonl"
        self._run_id = run_id
        log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, step_dict: dict[str, object]) -> None:
        """Append one record.

        Args:
            step_dict: Dict with any subset of the standard fields listed below.
                Required: epoch, step.
                Optional: loss_total, disc_loss, dur_loss, ce_loss, mel_loss,
                          f0_loss, val_loss, acoustic_norm, grad_norm_msd,
                          grad_norm_mpd, grad_norm_decoder, grad_norm_style_encoder,
                          skip_count, lr.
        """
        record: dict[str, object] = {
            "run_id": self._run_id,
            "epoch": step_dict.get("epoch"),
            "step": step_dict.get("step"),
            "loss_total": step_dict.get("loss_total"),
            "disc_loss": step_dict.get("disc_loss"),
            "dur_loss": step_dict.get("dur_loss"),
            "ce_loss": step_dict.get("ce_loss"),
            "mel_loss": step_dict.get("mel_loss"),
            "f0_loss": step_dict.get("f0_loss"),
            "val_loss": step_dict.get("val_loss"),
            "acoustic_norm": step_dict.get("acoustic_norm"),
            "grad_norm_msd": step_dict.get("grad_norm_msd"),
            "grad_norm_mpd": step_dict.get("grad_norm_mpd"),
            "grad_norm_decoder": step_dict.get("grad_norm_decoder"),
            "grad_norm_style_encoder": step_dict.get("grad_norm_style_encoder"),
            "skip_count": step_dict.get("skip_count"),
            "lr": step_dict.get("lr"),
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        }
        with self._path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def path(self) -> Path:
        """Path to the JSONL file."""
        return self._path
