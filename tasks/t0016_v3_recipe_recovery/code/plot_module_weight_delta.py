"""Milestone 5 Step 18 (REQ-13, REQ-7): render `results/images/v3_module_weight_delta.png`.

Bar chart of relative weight-norm change per module, Stage 1 -> best, from
`results/v3_module_weight_delta.raw.json` (written by `checkpoint_forensics.py`). Answers Key
Question 3 (task_description.md): whether the decoder changed at all, or whether v3's speaker_sim
gain comes from `predictor`/`text_encoder` alone.

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.plot_module_weight_delta
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.t0016_v3_recipe_recovery.code.checkpoint_forensics import (
    NEAR_ZERO_SHIFT_THRESHOLD,
    TARGET_MODULES,
)
from tasks.t0016_v3_recipe_recovery.code.paths import (
    MODULE_WEIGHT_DELTA_PNG,
    MODULE_WEIGHT_DELTA_RAW_JSON,
)


def main() -> None:
    raw = json.loads(MODULE_WEIGHT_DELTA_RAW_JSON.read_text())
    relative_deltas: dict[str, float] = raw["relative_deltas"]

    modules = [m for m in TARGET_MODULES if m in relative_deltas]
    values_pct = [relative_deltas[m] * 100.0 for m in modules]
    threshold_pct = NEAR_ZERO_SHIFT_THRESHOLD * 100
    colors = ["#c0392b" if abs(v) >= threshold_pct else "#7f8c8d" for v in values_pct]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(modules, values_pct, color=colors)
    ax.axhline(
        NEAR_ZERO_SHIFT_THRESHOLD * 100,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"{NEAR_ZERO_SHIFT_THRESHOLD:.0%} near-zero-shift threshold",
    )
    ax.axhline(-NEAR_ZERO_SHIFT_THRESHOLD * 100, color="black", linestyle="--", linewidth=1)
    ax.set_title("v3 Stage 2: relative weight-norm change per module, Stage 1 → best checkpoint")
    ax.set_xlabel("Module")
    ax.set_ylabel("Relative weight-norm change (%)")
    ax.legend(loc="upper left")
    for bar, value in zip(bars, values_pct, strict=True):
        ax.annotate(
            f"{value:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5 if value >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()

    MODULE_WEIGHT_DELTA_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(MODULE_WEIGHT_DELTA_PNG, dpi=150)
    print(f"Wrote {MODULE_WEIGHT_DELTA_PNG}")


if __name__ == "__main__":
    main()
