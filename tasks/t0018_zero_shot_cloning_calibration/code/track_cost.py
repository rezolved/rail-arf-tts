"""Append a running GPU-billing checkpoint to `results/cost_tracking.json`.

Per the implementation step's budget-management instructions: log cumulative wall-clock/cost after
each major milestone so the orchestrator's later results/reporting steps (and this step's own final
report) can cite the real, measured spend rather than the plan's pre-run estimate. This data is
folded into `results/tables.json`'s `cost_tracking` field by `report_zeroshot.py` at the end: this
file is the append-only log kept during the run itself.

Usage::

    uv run python -u tasks/t0018_zero_shot_cloning_calibration/code/track_cost.py \\
        --milestone "f5tts ref_single smoke gate passed"
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    VM_BILLING_ANCHOR_ISO,
    VM_HOURLY_COST_USD,
)

COST_TRACKING_PATH = "tasks/t0018_zero_shot_cloning_calibration/results/cost_tracking.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--milestone", required=True)
    p.add_argument("--note", default="")
    args = p.parse_args()

    anchor = datetime.fromisoformat(VM_BILLING_ANCHOR_ISO.replace("Z", "+00:00"))
    now = datetime.now(UTC)
    elapsed_h = (now - anchor).total_seconds() / 3600.0
    cost_usd = elapsed_h * VM_HOURLY_COST_USD

    from pathlib import Path

    path = Path(COST_TRACKING_PATH)
    entries: list[dict[str, object]] = []
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))

    entries.append(
        {
            "timestamp_utc": now.isoformat(),
            "elapsed_hours_since_billing_anchor": round(elapsed_h, 4),
            "estimated_cost_usd": round(cost_usd, 2),
            "milestone": args.milestone,
            "note": args.note,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(
        f"[{now.isoformat()}] elapsed={elapsed_h:.3f}h cost=${cost_usd:.2f} "
        f"milestone={args.milestone!r}"
    )


if __name__ == "__main__":
    main()
