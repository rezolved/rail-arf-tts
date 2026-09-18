"""Append a running GPU-billing checkpoint to `results/cost_tracking.json` (REQ-9).

Copied from `tasks.t0018_zero_shot_cloning_calibration.code.track_cost` (copied, not imported),
updated to this task's own `VM_HOURLY_COST_USD`/`BUDGET_HARD_CAP_USD`/`VM_BILLING_ANCHOR_ISO`.

Usage::

    uv run python -u tasks/t0021_zero_shot_latency_reduction/code/track_cost.py \\
        --milestone "cosyvoice2 baseline_new_ref done"
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tasks.t0021_zero_shot_latency_reduction.code.constants import (
    BUDGET_HARD_CAP_USD,
    VM_BILLING_ANCHOR_ISO,
    VM_CONFIRMED_DOWNTIME_SECONDS,
    VM_HOURLY_COST_USD,
)

COST_TRACKING_PATH = Path("tasks/t0021_zero_shot_latency_reduction/results/cost_tracking.json")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--milestone", required=True)
    p.add_argument("--note", default="")
    args = p.parse_args()

    anchor = datetime.fromisoformat(VM_BILLING_ANCHOR_ISO.replace("Z", "+00:00"))
    now = datetime.now(UTC)
    # Subtract the confirmed non-billing window (VM was stopped by the idle watchdog after the
    # mid-implementation crash, then re-acquired) so wall-clock-since-anchor does not overcount
    # actual billed GPU time. See VM_CONFIRMED_DOWNTIME_SECONDS's docstring in constants.py.
    elapsed_h = (now - anchor).total_seconds() / 3600.0 - VM_CONFIRMED_DOWNTIME_SECONDS / 3600.0
    cost_usd = elapsed_h * VM_HOURLY_COST_USD

    entries: list[dict[str, object]] = []
    if COST_TRACKING_PATH.exists():
        entries = json.loads(COST_TRACKING_PATH.read_text(encoding="utf-8"))

    entries.append(
        {
            "timestamp_utc": now.isoformat(),
            "elapsed_hours_since_billing_anchor": round(elapsed_h, 4),
            "estimated_cost_usd": round(cost_usd, 2),
            "budget_hard_cap_usd": BUDGET_HARD_CAP_USD,
            "milestone": args.milestone,
            "note": args.note,
        }
    )
    COST_TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_TRACKING_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(
        f"[{now.isoformat()}] elapsed={elapsed_h:.3f}h cost=${cost_usd:.2f} "
        f"(cap=${BUDGET_HARD_CAP_USD:.2f}) milestone={args.milestone!r}"
    )
    if cost_usd >= BUDGET_HARD_CAP_USD:
        print("WARNING: cumulative cost has reached or exceeded the task's hard cap!")


if __name__ == "__main__":
    main()
