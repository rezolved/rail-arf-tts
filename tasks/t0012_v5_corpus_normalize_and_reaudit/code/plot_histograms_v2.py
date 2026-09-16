"""Before/after distribution histograms for t0012_v5_corpus_normalize_and_reaudit (Step 4).

Reads data/per_clip_stats_v2.jsonl and produces 3 PNGs in results/images/ (REQ-13):
  - peak_before_after.png
  - lufs_before_after.png
  - clipped_fraction_distribution.png

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0012_v5_corpus_normalize_and_reaudit -- \
        uv run python -u tasks/t0012_v5_corpus_normalize_and_reaudit/code/plot_histograms_v2.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.t0012_v5_corpus_normalize_and_reaudit.code.constants import (
    CLIPPED_FRACTION_MAX,
    TARGET_LUFS,
)
from tasks.t0012_v5_corpus_normalize_and_reaudit.code.paths import (
    IMAGES_DIR,
    PER_CLIP_STATS_V2_JSONL,
)


def _load_field(records: list[dict[str, object]], field: str) -> list[float]:
    """Load non-null numeric values for a single field from loaded records."""
    values: list[float] = []
    for record in records:
        val = record.get(field)
        if isinstance(val, float | int):
            values.append(float(val))
    return values


def plot_before_after(
    before_vals: list[float],
    after_vals: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
    flag_lines: list[float] | None = None,
    log_yscale: bool = False,
) -> None:
    """Plot overlapping histograms for pre- vs post-normalization distributions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        before_vals,
        bins=40,
        alpha=0.6,
        label=f"before (n={len(before_vals)})",
        color="steelblue",
    )
    ax.hist(
        after_vals,
        bins=40,
        alpha=0.6,
        label=f"after (n={len(after_vals)})",
        color="orange",
    )
    if flag_lines is not None:
        for thresh in flag_lines:
            ax.axvline(x=thresh, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    if log_yscale:
        ax.set_yscale("log")
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
    print("=== plot_histograms_v2.py: before/after distribution charts ===")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    with PER_CLIP_STATS_V2_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                records.append(json.loads(line))

    plot_before_after(
        before_vals=_load_field(records, "pre_peak_dbfs"),
        after_vals=_load_field(records, "post_peak_dbfs"),
        title="Peak Amplitude: Before vs After LUFS Normalization",
        xlabel="Peak dBFS",
        out_path=IMAGES_DIR / "peak_before_after.png",
        flag_lines=[0.0],
    )

    plot_before_after(
        before_vals=_load_field(records, "pre_lufs"),
        after_vals=_load_field(records, "post_lufs"),
        title="Integrated Loudness: Before vs After LUFS Normalization",
        xlabel="LUFS (dBFS for short clips via RMS fallback)",
        out_path=IMAGES_DIR / "lufs_before_after.png",
        flag_lines=[TARGET_LUFS],
    )

    plot_before_after(
        before_vals=_load_field(records, "pre_clipped_fraction"),
        after_vals=_load_field(records, "post_clipped_fraction"),
        title="Clipped-Sample Fraction: Before vs After LUFS Normalization",
        xlabel="clipped_fraction (fraction of samples within 1 LSB of full scale)",
        out_path=IMAGES_DIR / "clipped_fraction_distribution.png",
        flag_lines=[CLIPPED_FRACTION_MAX],
        log_yscale=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
