"""Aggregate t0010 training results into metrics.json and charts.

Reads:
  - data/run_v10/metrics.jsonl  (per-step training log from StepLogger)
  - data/run_v10/eval_results/eval_summary.json  (batch eval summary)

Produces:
  - data/run_v10/per_epoch_summary.csv   (per-epoch table)
  - results/metrics.json                 (explicit variant format, one per epoch + best)
  - results/images/speaker_sim_curve.png (trajectory with reference lines)
  - results/images/loss_timeline.png     (val_loss, dur_loss, acoustic_norm vs epoch)

Usage:
    python aggregate_results.py \\
        --jsonl data/run_v10/metrics.jsonl \\
        --eval-dir data/run_v10/eval_results \\
        --output-dir results \\
        --task-root tasks/t0010_stage2_safeguarded_training
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Baselines from t0008 (source: tasks/t0008_tts_eval_harness_baselines/results/metrics.json)
BASELINE_V3_BUNDLE_SPEAKER_SIM: float = 0.631  # kokoro_v3_bundle best on fillers set
ELEVENLABS_TARGET_SPEAKER_SIM: float = 0.85  # project success criterion


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TrainingStep:
    epoch: int  # 1-based
    step: int
    val_loss: float | None
    dur_loss_step1: float | None
    acoustic_norm: float | None
    health_gate_fired: bool
    health_gate_name: str | None


@dataclass(frozen=True, slots=True)
class EvalEpoch:
    epoch: int
    speaker_sim_mean: float | None
    speaker_sim_std: float | None
    ttfb_p50_ms: float | None
    rtf_mean: float | None
    n_success: int
    n_total: int


@dataclass(frozen=True, slots=True)
class EpochRow:
    epoch: int
    val_loss: float | None
    dur_loss_step1: float | None
    acoustic_norm: float | None
    speaker_sim_mean: float | None
    speaker_sim_std: float | None
    ttfb_p50_ms: float | None
    rtf_mean: float | None
    n_success: int | None
    n_total: int | None


# ── Loaders ───────────────────────────────────────────────────────────────────


def load_training_steps(jsonl_path: Path) -> list[TrainingStep]:
    """Parse StepLogger JSONL and return one record per epoch (last step of each epoch)."""
    steps: list[TrainingStep] = []
    with open(str(jsonl_path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                rec: dict[str, object] = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("epoch") is None:
                continue  # sentinel / null record
            epoch = int(str(rec.get("epoch", 0)))
            step = int(str(rec.get("step", 0)))

            val_loss_val = rec.get("val_loss")
            dur_loss_val = rec.get("dur_loss_step1") or rec.get("dur_loss")
            acoustic_val = rec.get("acoustic_norm")
            gate_fired = bool(rec.get("health_gate_fired", False))
            gate_name_raw = rec.get("health_gate_name")
            gate_name = str(gate_name_raw) if gate_name_raw is not None else None

            steps.append(
                TrainingStep(
                    epoch=epoch,
                    step=step,
                    val_loss=float(str(val_loss_val)) if val_loss_val is not None else None,
                    dur_loss_step1=float(str(dur_loss_val)) if dur_loss_val is not None else None,
                    acoustic_norm=float(str(acoustic_val)) if acoustic_val is not None else None,
                    health_gate_fired=gate_fired,
                    health_gate_name=gate_name,
                )
            )

    return steps


def load_eval_summary(eval_dir: Path) -> list[EvalEpoch]:
    """Load batch eval summary."""
    summary_path = eval_dir / "eval_summary.json"
    if not summary_path.exists():
        logger.warning("eval_summary.json not found at %s — no speaker_sim data", summary_path)
        return []

    raw: list[dict[str, object]] = json.loads(summary_path.read_text(encoding="utf-8"))
    result: list[EvalEpoch] = []
    for rec in raw:
        sim_mean = rec.get("speaker_sim_mean")
        sim_std = rec.get("speaker_sim_std")
        ttfb = rec.get("ttfb_p50_ms")
        rtf = rec.get("rtf_mean")
        result.append(
            EvalEpoch(
                epoch=int(str(rec["epoch"])),
                speaker_sim_mean=float(str(sim_mean)) if sim_mean is not None else None,
                speaker_sim_std=float(str(sim_std)) if sim_std is not None else None,
                ttfb_p50_ms=float(str(ttfb)) if ttfb is not None else None,
                rtf_mean=float(str(rtf)) if rtf is not None else None,
                n_success=int(str(rec.get("n_success", 0))),
                n_total=int(str(rec.get("n_total", 0))),
            )
        )
    return result


def build_epoch_rows(
    training_steps: list[TrainingStep],
    eval_epochs: list[EvalEpoch],
) -> list[EpochRow]:
    """Merge training and eval data into per-epoch rows."""
    # Take last step per epoch for training metrics
    epoch_train: dict[int, TrainingStep] = {}
    for step in training_steps:
        if step.epoch not in epoch_train or step.step > epoch_train[step.epoch].step:
            epoch_train[step.epoch] = step

    # Index eval by epoch
    epoch_eval: dict[int, EvalEpoch] = {e.epoch: e for e in eval_epochs}

    all_epochs = sorted(set(epoch_train.keys()) | set(epoch_eval.keys()))
    rows: list[EpochRow] = []
    for epoch in all_epochs:
        train = epoch_train.get(epoch)
        ev = epoch_eval.get(epoch)
        rows.append(
            EpochRow(
                epoch=epoch,
                val_loss=train.val_loss if train is not None else None,
                dur_loss_step1=train.dur_loss_step1 if train is not None else None,
                acoustic_norm=train.acoustic_norm if train is not None else None,
                speaker_sim_mean=ev.speaker_sim_mean if ev is not None else None,
                speaker_sim_std=ev.speaker_sim_std if ev is not None else None,
                ttfb_p50_ms=ev.ttfb_p50_ms if ev is not None else None,
                rtf_mean=ev.rtf_mean if ev is not None else None,
                n_success=ev.n_success if ev is not None else None,
                n_total=ev.n_total if ev is not None else None,
            )
        )
    return rows


def write_csv(rows: list[EpochRow], out_path: Path) -> None:
    """Write per-epoch summary CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "val_loss",
        "dur_loss_step1",
        "acoustic_norm",
        "speaker_sim_mean",
        "speaker_sim_std",
        "ttfb_p50_ms",
        "rtf_mean",
        "n_success",
        "n_total",
    ]
    with open(str(out_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "val_loss": row.val_loss,
                    "dur_loss_step1": row.dur_loss_step1,
                    "acoustic_norm": row.acoustic_norm,
                    "speaker_sim_mean": row.speaker_sim_mean,
                    "speaker_sim_std": row.speaker_sim_std,
                    "ttfb_p50_ms": row.ttfb_p50_ms,
                    "rtf_mean": row.rtf_mean,
                    "n_success": row.n_success,
                    "n_total": row.n_total,
                }
            )
    logger.info("Wrote per-epoch CSV → %s (%d rows)", out_path, len(rows))


def write_speaker_sim_curve(rows: list[EpochRow], out_path: Path) -> None:
    """Plot speaker_sim per epoch with reference lines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in rows if r.speaker_sim_mean is not None]
    sims = [r.speaker_sim_mean for r in rows if r.speaker_sim_mean is not None]

    if len(epochs) == 0:
        logger.warning("No speaker_sim data — skipping curve chart")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        epochs, sims, "o-", color="steelblue", linewidth=2, markersize=6, label="v10 speaker_sim"
    )

    # Reference lines
    ax.axhline(
        BASELINE_V3_BUNDLE_SPEAKER_SIM,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"v3_bundle baseline ({BASELINE_V3_BUNDLE_SPEAKER_SIM:.3f})",
    )
    ax.axhline(
        ELEVENLABS_TARGET_SPEAKER_SIM,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"ElevenLabs target ({ELEVENLABS_TARGET_SPEAKER_SIM:.2f})",
    )

    # Mark best epoch
    best_idx = int(max(range(len(sims)), key=lambda i: sims[i] or -1.0))
    ax.annotate(
        f"best: {sims[best_idx]:.4f}\nepoch {epochs[best_idx]}",
        xy=(epochs[best_idx], sims[best_idx]),
        xytext=(epochs[best_idx] + 0.3, sims[best_idx] + 0.01),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "grey"},
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Speaker Similarity (GE2E cosine)")
    ax.set_title(
        "Kokoro v10 — Speaker Similarity per Epoch\n(100-filler set vs ElevenLabs David centroid)"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.0, min(sims) - 0.05))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote speaker_sim curve → %s", out_path)


def write_loss_timeline(rows: list[EpochRow], out_path: Path) -> None:
    """Plot val_loss, dur_loss_step1, acoustic_norm vs epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r.epoch for r in rows]
    val_losses = [r.val_loss for r in rows]
    dur_losses = [r.dur_loss_step1 for r in rows]
    acoustic_norms = [r.acoustic_norm for r in rows]

    has_data = any(v is not None for v in val_losses + dur_losses + acoustic_norms)
    if not has_data:
        logger.warning("No training loss data — skipping loss timeline chart")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    def _plot(
        ax: Any,
        values: list[float | None],
        label: str,
        color: str,
        threshold: float | None,
    ) -> None:
        """Plot a single metric line."""
        xs = [e for e, v in zip(epochs, values, strict=False) if v is not None]
        ys = [v for v in values if v is not None]
        if len(xs) == 0:
            return
        ax.plot(xs, ys, "o-", color=color, linewidth=2, markersize=4, label=label)
        if threshold is not None:
            ax.axhline(
                threshold,
                color="red",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label=f"gate={threshold}",
            )
        ax.set_ylabel(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(axes[0], val_losses, "val_loss", "steelblue", None)
    _plot(axes[1], dur_losses, "dur_loss_step1", "darkorange", 2.0)  # gate=2.0
    _plot(axes[2], acoustic_norms, "acoustic_norm", "green", 20.0)  # gate=20.0

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Kokoro v10 — Training Loss Timeline", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote loss timeline → %s", out_path)


def write_metrics_json(rows: list[EpochRow], out_path: Path) -> None:
    """Write metrics.json in explicit variant format.

    One variant per epoch, plus a 'best' variant for peak speaker_sim epoch.
    Registered metrics: speaker_sim, ttfb_ms, rtf.
    """
    variants: list[dict[str, object]] = []

    best_epoch: EpochRow | None = None
    best_sim: float = -1.0

    for row in rows:
        variant_id = f"epoch-{row.epoch:02d}"
        label = f"Epoch {row.epoch}"
        variants.append(
            {
                "variant_id": variant_id,
                "label": label,
                "dimensions": {
                    "epoch": row.epoch,
                    "model": "kokoro-v10",
                    "run": "v10",
                },
                "metrics": {
                    "speaker_sim": row.speaker_sim_mean,
                    "ttfb_ms": row.ttfb_p50_ms,
                    "rtf": row.rtf_mean,
                },
            }
        )
        if row.speaker_sim_mean is not None and row.speaker_sim_mean > best_sim:
            best_sim = row.speaker_sim_mean
            best_epoch = row

    # Add 'best' variant
    if best_epoch is not None:
        variants.append(
            {
                "variant_id": "best",
                "label": f"Best (epoch {best_epoch.epoch})",
                "dimensions": {
                    "epoch": best_epoch.epoch,
                    "model": "kokoro-v10",
                    "run": "v10",
                    "selection": "peak_speaker_sim",
                },
                "metrics": {
                    "speaker_sim": best_epoch.speaker_sim_mean,
                    "ttfb_ms": best_epoch.ttfb_p50_ms,
                    "rtf": best_epoch.rtf_mean,
                },
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"variants": variants}, indent=2), encoding="utf-8")
    logger.info("Wrote metrics.json → %s (%d variants, incl. best)", out_path, len(variants))

    if best_epoch is not None:
        logger.info(
            "Best: epoch %d  speaker_sim=%.4f  ttfb_p50=%.1fms  rtf=%.3f",
            best_epoch.epoch,
            best_epoch.speaker_sim_mean or 0.0,
            best_epoch.ttfb_p50_ms or 0.0,
            best_epoch.rtf_mean or 0.0,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate t0010 training results")
    p.add_argument("--jsonl", type=Path, required=True, help="Path to metrics.jsonl training log")
    p.add_argument("--eval-dir", type=Path, required=True, help="Directory with eval_summary.json")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory (results/)")
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Per-epoch CSV output path (default: inferred from eval-dir)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    jsonl_path: Path = args.jsonl.resolve()
    eval_dir: Path = args.eval_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if args.csv_out is not None:
        csv_path = args.csv_out.resolve()
    else:
        csv_path = eval_dir.parent / "per_epoch_summary.csv"

    # Validate
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Training log not found: {jsonl_path}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    # Load
    training_steps = load_training_steps(jsonl_path)
    eval_epochs = load_eval_summary(eval_dir)
    logger.info("Loaded %d training steps, %d eval epochs", len(training_steps), len(eval_epochs))

    # Merge
    rows = build_epoch_rows(training_steps, eval_epochs)
    logger.info("Built %d epoch rows", len(rows))

    # Validate before charting
    sim_values = [r.speaker_sim_mean for r in rows if r.speaker_sim_mean is not None]
    if len(sim_values) == 0:
        logger.error(
            "VALIDATION GATE: No speaker_sim values found in eval data. "
            "Charts will be empty or missing. Investigate eval_summary.json."
        )
    elif all(v == 0.0 for v in sim_values):
        logger.error(
            "VALIDATION GATE: All speaker_sim values are 0.0. "
            "Likely a scoring failure. Do not treat these as valid measurements."
        )

    # Write outputs
    write_csv(rows=rows, out_path=csv_path)
    write_speaker_sim_curve(rows=rows, out_path=output_dir / "images" / "speaker_sim_curve.png")
    write_loss_timeline(rows=rows, out_path=output_dir / "images" / "loss_timeline.png")
    write_metrics_json(rows=rows, out_path=output_dir / "metrics.json")

    logger.info("Aggregation complete.")


if __name__ == "__main__":
    main()
