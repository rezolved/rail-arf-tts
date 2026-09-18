"""Run the hardened audible-speech gate on every synthesized clip (plan Step 10, REQ-7).

Adapted from `tasks.t0018_zero_shot_cloning_calibration.code.run_gate_check` (copied, not
imported). Imports `check_audio_quality` from THIS task's own copy of the checker
(`code/audio_quality_check.py`, copied from t0015 per plan.md Step 10 — t0018's cross-task import
of t0015's module directly is no longer valid under the current spec, so this task carries its own
copy).

Per Cross-Step Decision (owner correction #7 / REQ-17): this gate is necessary, not sufficient. A
PASS here does not certify audio quality.

Usage::

    uv run python -u tasks/t0021_zero_shot_latency_reduction/code/run_gate_check.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tasks.t0021_zero_shot_latency_reduction.code.audio_quality_check import (
    LONGEST_NONSILENT_RUN_THRESHOLD_S,
    check_audio_quality,
)
from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    RESULTS_GATE_FAILURES,
    RESULTS_PER_CLIP_METRICS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def hardened_gate_pass(result: object) -> bool:
    return not (
        result.is_likely_noise  # type: ignore[attr-defined]
        or result.duration_sanity_pass is False  # type: ignore[attr-defined]
        or result.longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S  # type: ignore[attr-defined]
    )


def main() -> None:
    records: list[dict[str, object]] = json.loads(
        RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    )
    logger.info("Loaded %d records", len(records))

    per_variant_failures: dict[str, dict[str, object]] = {}

    for rec in records:
        audio_path_str = rec.get("audio_path")
        if audio_path_str is None:
            continue
        audio_path = Path(str(audio_path_str))
        if not audio_path.exists():
            logger.warning("Missing audio file: %s", audio_path)
            continue

        system = str(rec["system"])
        variant = rec.get("acceleration_variant")
        condition = rec.get("condition")
        prompt_set = str(rec["prompt_set"])
        variant_id = f"{system}_{variant}_{condition}_{prompt_set}"

        try:
            result = check_audio_quality(audio_path, text=str(rec.get("text", "")))
            passed = hardened_gate_pass(result)
        except Exception as exc:
            logger.warning("Gate check failed for %s: %s", audio_path, exc)
            passed = False

        rec["hardened_gate_pass"] = passed

        entry = per_variant_failures.setdefault(
            variant_id, {"failure_count": 0, "n_checked": 0, "failing_audio_paths": []}
        )
        entry["n_checked"] = int(entry["n_checked"]) + 1
        if not passed:
            entry["failure_count"] = int(entry["failure_count"]) + 1
            entry["failing_audio_paths"].append(str(audio_path))  # type: ignore[union-attr]

    RESULTS_PER_CLIP_METRICS.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Updated per_clip_metrics.json with hardened_gate_pass")

    RESULTS_GATE_FAILURES.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_GATE_FAILURES.write_text(json.dumps(per_variant_failures, indent=2), encoding="utf-8")
    logger.info(
        "Saved gate_failures.json -> %s (%d variants)",
        RESULTS_GATE_FAILURES,
        len(per_variant_failures),
    )


if __name__ == "__main__":
    main()
