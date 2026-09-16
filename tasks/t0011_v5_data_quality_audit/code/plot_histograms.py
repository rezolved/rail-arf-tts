"""Distribution histograms for t0011_v5_data_quality_audit.

Reads per_clip_stats.jsonl (train) and val_clip_stats.jsonl (val) and produces
four PNG histograms in results/images/:
  - peak_distribution.png   (peak dBFS)
  - lufs_distribution.png   (LUFS)
  - duration_distribution.png (duration in seconds)
  - phoneme_distribution.png  (phoneme string length from manifest)

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0011_v5_data_quality_audit -- \
        uv run python -u tasks/t0011_v5_data_quality_audit/code/plot_histograms.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.t0011_v5_data_quality_audit.code.constants import (
    DURATION_MAX_S,
    DURATION_MIN_S,
    FIELD_DURATION_S,
    FIELD_LUFS,
    FIELD_PEAK_DBFS,
    LUFS_MAX,
    LUFS_MIN,
    PEAK_DBFS_MAX,
)
from tasks.t0011_v5_data_quality_audit.code.paths import (
    IMAGES_DIR,
    PER_CLIP_STATS_JSONL,
    V5_TRAIN_LIST,
    V5_VAL_LIST,
    VAL_CLIP_STATS_JSONL,
)


def _load_field(
    jsonl_path: Path,
    field: str,
) -> list[float]:
    """Load non-null numeric values for a single field from a JSONL file."""
    values: list[float] = []
    if not jsonl_path.exists():
        return values
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            record = json.loads(line)
            val = record.get(field)
            if isinstance(val, float | int):
                values.append(float(val))
    return values


def _load_phoneme_lengths(list_path: Path) -> list[int]:
    """Return phoneme string lengths from a manifest file."""
    lengths: list[int] = []
    if not list_path.exists():
        return lengths
    with list_path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if len(line) == 0:
                continue
            parts = line.split("|")
            phonemes = parts[1] if len(parts) > 1 else ""
            lengths.append(len(phonemes))
    return lengths


def plot_histogram(
    train_vals: list[float],
    val_vals: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
    flag_lines: list[float] | None = None,
) -> None:
    """Plot overlapping histograms for train vs val distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        train_vals,
        bins=40,
        alpha=0.6,
        label=f"train (n={len(train_vals)})",
        color="steelblue",
    )
    ax.hist(
        val_vals,
        bins=40,
        alpha=0.6,
        label=f"val (n={len(val_vals)})",
        color="orange",
    )
    if flag_lines is not None:
        for thresh in flag_lines:
            ax.axvline(x=thresh, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    print("=== plot_histograms.py: distribution charts ===")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load metric arrays
    train_peak = _load_field(jsonl_path=PER_CLIP_STATS_JSONL, field=FIELD_PEAK_DBFS)
    val_peak = _load_field(jsonl_path=VAL_CLIP_STATS_JSONL, field=FIELD_PEAK_DBFS)

    train_lufs = _load_field(jsonl_path=PER_CLIP_STATS_JSONL, field=FIELD_LUFS)
    val_lufs = _load_field(jsonl_path=VAL_CLIP_STATS_JSONL, field=FIELD_LUFS)

    train_dur = _load_field(jsonl_path=PER_CLIP_STATS_JSONL, field=FIELD_DURATION_S)
    val_dur = _load_field(jsonl_path=VAL_CLIP_STATS_JSONL, field=FIELD_DURATION_S)

    train_plen = [float(x) for x in _load_phoneme_lengths(list_path=V5_TRAIN_LIST)]
    val_plen = [float(x) for x in _load_phoneme_lengths(list_path=V5_VAL_LIST)]

    # Chart 1: Peak dBFS
    plot_histogram(
        train_vals=train_peak,
        val_vals=val_peak,
        title="Peak Amplitude Distribution (v5 train vs val)",
        xlabel="Peak dBFS",
        out_path=IMAGES_DIR / "peak_distribution.png",
        flag_lines=[PEAK_DBFS_MAX],
    )

    # Chart 2: LUFS
    plot_histogram(
        train_vals=train_lufs,
        val_vals=val_lufs,
        title="Integrated Loudness Distribution (v5 train vs val)",
        xlabel="LUFS (dBFS for short clips via RMS fallback)",
        out_path=IMAGES_DIR / "lufs_distribution.png",
        flag_lines=[LUFS_MIN, LUFS_MAX],
    )

    # Chart 3: Duration
    plot_histogram(
        train_vals=train_dur,
        val_vals=val_dur,
        title="Clip Duration Distribution (v5 train vs val)",
        xlabel="Duration (seconds)",
        out_path=IMAGES_DIR / "duration_distribution.png",
        flag_lines=[DURATION_MIN_S, DURATION_MAX_S],
    )

    # Chart 4: Phoneme string length
    plot_histogram(
        train_vals=train_plen,
        val_vals=val_plen,
        title="Phoneme String Length Distribution (v5 train vs val)",
        xlabel="Phoneme string length (characters)",
        out_path=IMAGES_DIR / "phoneme_distribution.png",
        flag_lines=None,
    )

    print("Done.")


if __name__ == "__main__":
    main()
