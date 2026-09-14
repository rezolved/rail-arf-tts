"""Health gates for Kokoro Stage 2 training (REQ-7, library component).

Thresholds:
  dur_loss_step1_max: 2.0  — Dur Loss at first step >= 2.0 → unstable
  acoustic_norm_max: 20.0  — acoustic_norm after any epoch >= 20 → diverging
  val_spike_max: 0.05      — val_loss increase > 0.05 post-joint_epoch → diverging
  consecutive_skip_max: 50 — consecutive NaN skips >= 50 → NaN storm

Usage in training loop:
    gate = HealthGate(joint_epoch=6, last_healthy_ckpt="logs/run/epoch_00005.pth")
    result = gate.check(epoch=1, step=1, metrics={"dur_loss": 9.7})
    if result.fired:
        sys.exit(3)
"""

from dataclasses import dataclass

from tasks.t0009_stage2_training_failure_forensics.code.constants import (
    ACOUSTIC_NORM_MAX,
    CONSECUTIVE_SKIP_MAX,
    DUR_LOSS_STEP1_MAX,
    VAL_SPIKE_MAX,
)
from tasks.t0009_stage2_training_failure_forensics.code.jsonl_logger import StepLogger


@dataclass(frozen=True, slots=True)
class GateResult:
    fired: bool
    gate_name: str
    value: float | None
    threshold: float | None
    last_healthy_ckpt: str | None
    message: str


class HealthGate:
    """Monitors training metrics and fires when a health gate threshold is exceeded.

    Args:
        joint_epoch: The epoch at which GAN losses activate (0-based).
        last_healthy_ckpt: Path to the last checkpoint saved before the gate fired.
            Updated by the training loop each time a checkpoint is saved.
        logger: Optional StepLogger to write gate events to JSONL.
    """

    def __init__(
        self,
        joint_epoch: int,
        last_healthy_ckpt: str | None = None,
        logger: StepLogger | None = None,
    ) -> None:
        self._joint_epoch = joint_epoch
        self._last_healthy_ckpt = last_healthy_ckpt
        self._logger = logger
        self._prev_val_loss: float | None = None
        self._consecutive_skips: int = 0
        self._is_first_step_of_epoch: dict[int, bool] = {}

    def update_last_checkpoint(self, ckpt_path: str) -> None:
        """Call this whenever a checkpoint is saved."""
        self._last_healthy_ckpt = ckpt_path

    def record_skip(self) -> GateResult | None:
        """Call when a training step is skipped due to NaN loss."""
        self._consecutive_skips += 1
        if self._consecutive_skips >= CONSECUTIVE_SKIP_MAX:
            return self._fire(
                gate_name="consecutive_skip",
                value=float(self._consecutive_skips),
                threshold=float(CONSECUTIVE_SKIP_MAX),
                message=(
                    f"Consecutive NaN skips reached {self._consecutive_skips} "
                    f"(threshold: {CONSECUTIVE_SKIP_MAX}). NaN storm — aborting."
                ),
            )
        return None

    def reset_skips(self) -> None:
        """Call when a step completes normally (no NaN)."""
        self._consecutive_skips = 0

    def check(
        self,
        epoch: int,
        step: int,
        metrics: dict[str, float | None],
    ) -> GateResult:
        """Check all applicable health gates for this step.

        Args:
            epoch: Current epoch (1-based to match log format).
            step: Current step (1-based within epoch).
            metrics: Dict with any of: dur_loss, acoustic_norm, val_loss.

        Returns:
            GateResult. If `fired` is True, the training loop should exit(3).
        """
        # Gate 1: dur_loss at first step of epoch 2+ (1-based epoch >= 2)
        if step == 1 and epoch >= 2:
            dur = metrics.get("dur_loss")
            if dur is not None and dur >= DUR_LOSS_STEP1_MAX:
                return self._fire(
                    gate_name="dur_loss_step1",
                    value=dur,
                    threshold=DUR_LOSS_STEP1_MAX,
                    message=(
                        f"Dur Loss at step 1 of epoch {epoch} = {dur:.5f} "
                        f">= threshold {DUR_LOSS_STEP1_MAX}. Unstable — aborting."
                    ),
                )

        # Gate 2: acoustic_norm (val record, step == 0)
        if step == 0:
            norm = metrics.get("acoustic_norm")
            if norm is not None and norm >= ACOUSTIC_NORM_MAX:
                return self._fire(
                    gate_name="acoustic_norm",
                    value=norm,
                    threshold=ACOUSTIC_NORM_MAX,
                    message=(
                        f"acoustic_norm after epoch {epoch} = {norm:.4f} "
                        f">= threshold {ACOUSTIC_NORM_MAX}. Diverging — aborting."
                    ),
                )

            # Gate 3: val_loss spike post-joint_epoch (1-based epoch > joint_epoch)
            if epoch > self._joint_epoch:
                val = metrics.get("val_loss")
                if val is not None and self._prev_val_loss is not None:
                    spike = val - self._prev_val_loss
                    if spike > VAL_SPIKE_MAX:
                        return self._fire(
                            gate_name="val_spike",
                            value=spike,
                            threshold=VAL_SPIKE_MAX,
                            message=(
                                f"val_loss spike at epoch {epoch}: "
                                f"{self._prev_val_loss:.3f} -> {val:.3f} "
                                f"(+{spike:.4f} > threshold {VAL_SPIKE_MAX}). Diverging — aborting."
                            ),
                        )
                if val is not None:
                    self._prev_val_loss = val

        return GateResult(
            fired=False,
            gate_name="",
            value=None,
            threshold=None,
            last_healthy_ckpt=self._last_healthy_ckpt,
            message="",
        )

    def _fire(
        self,
        gate_name: str,
        value: float,
        threshold: float,
        message: str,
    ) -> GateResult:
        result = GateResult(
            fired=True,
            gate_name=gate_name,
            value=value,
            threshold=threshold,
            last_healthy_ckpt=self._last_healthy_ckpt,
            message=message,
        )
        print(f"\n[HealthGate FIRED] {message}")
        if self._last_healthy_ckpt is not None:
            print(f"[HealthGate] Last healthy checkpoint: {self._last_healthy_ckpt}")
        if self._logger is not None:
            self._logger.log(
                {
                    "gate_fired": gate_name,
                    "gate_value": value,
                    "gate_threshold": threshold,
                    "gate_message": message,
                    "last_healthy_ckpt": self._last_healthy_ckpt,
                }
            )
        return result
