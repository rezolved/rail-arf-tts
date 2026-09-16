"""Generate speaker_sim_curve.png placeholder using val_loss as proxy.

Speaker_sim eval was deferred (VM disk full); this chart shows val_loss
per epoch with reference lines indicating where speaker_sim targets sit.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TASK_DIR = Path(__file__).resolve().parents[1]
JSONL_PATH = TASK_DIR / "data/run_v10/metrics.jsonl"
OUT_PATH = TASK_DIR / "results/images/speaker_sim_curve.png"


def load_epoch_val_loss(jsonl_path: Path) -> list[tuple[int, float]]:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            rec = json.loads(line)
            epoch = rec.get("epoch")
            val_loss = rec.get("val_loss")
            if epoch is not None and val_loss is not None:
                rows.append((int(epoch), float(val_loss)))
    # Deduplicate: keep latest record per epoch (epoch-end record has step=0)
    seen: dict[int, float] = {}
    for epoch, vl in rows:
        seen[epoch] = vl
    return sorted(seen.items())


def main() -> None:
    pairs = load_epoch_val_loss(JSONL_PATH)
    epochs = [p[0] for p in pairs]
    val_losses = [p[1] for p in pairs]

    best_epoch = epochs[val_losses.index(min(val_losses))]
    best_val = min(val_losses)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        epochs,
        val_losses,
        "o-",
        color="#2176AE",
        linewidth=2,
        markersize=5,
        label="val_loss (proxy)",
    )
    ax.axvline(
        x=8.5, color="#888888", linestyle="--", linewidth=1, label="GAN activation (joint_epoch=8)"
    )
    ax.scatter(
        [best_epoch],
        [best_val],
        color="#E84855",
        zorder=5,
        s=80,
        label=f"Best val_loss {best_val:.3f} @ ep {best_epoch}",
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_title(
        "Speaker Sim Proxy: val_loss per Epoch (speaker_sim eval deferred)\n"
        "Kokoro-v10 run, joint_epoch=8 | speaker_sim eval pending",
        fontsize=11,
    )
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotation box
    ax.annotate(
        "speaker_sim eval deferred\n(VM disk full — see\nintervention/eval_deferred_disk_full.md)",
        xy=(0.98, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        color="#AA0000",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#FFF0F0",
            "edgecolor": "#AA0000",
            "alpha": 0.8,
        },
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
