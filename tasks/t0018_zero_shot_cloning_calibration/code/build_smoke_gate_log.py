"""Write `results/smoke_gate_log.md` from `results/smoke_gate_results.json` (plan Step 4).

`smoke_gate_results.json` is a small hand/script-maintained ledger (one entry per system) recording
install time, pass/fail, and (F5-TTS only) the effective `ref_concat` duration consumed --
maintained directly during implementation since the smoke gates run as separate remote-shell
invocations, not as a single Python process.
"""

from __future__ import annotations

import json
import logging

from tasks.t0018_zero_shot_cloning_calibration.code.paths import RESULTS_SMOKE_GATE_LOG, TASK_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

SMOKE_GATE_RESULTS_JSON = TASK_ROOT / "results" / "smoke_gate_results.json"


def main() -> None:
    raw_text = SMOKE_GATE_RESULTS_JSON.read_text(encoding="utf-8")
    entries: list[dict[str, object]] = json.loads(raw_text)

    lines: list[str] = []
    lines.append("# Smoke Gate Log")
    lines.append("")
    lines.append(
        "Per-system install time, `ref_single` smoke-gate pass/fail, and (F5-TTS only) the "
        "dedicated `ref_concat` pre-check result (plan Step 4)."
    )
    lines.append("")
    lines.append("| System | Install Time | Smoke Gate (ref_single) | Notes |")
    lines.append("| --- | --- | --- | --- |")
    for e in entries:
        lines.append(
            f"| {e['system']} | {e.get('install_time', 'n/a (pre-installed prior step)')} "
            f"| {e['smoke_gate_result']} | {e.get('notes', '')} |"
        )

    lines.append("")
    lines.append("## F5-TTS `ref_concat` pre-check detail")
    lines.append("")
    f5_entry = next((e for e in entries if e["system"] == "f5_tts"), None)
    if f5_entry is not None and "ref_concat_check" in f5_entry:
        lines.append(str(f5_entry["ref_concat_check"]))
    else:
        lines.append(
            "Never reached: F5-TTS's `ref_single` smoke gate itself hung indefinitely across all "
            "three attempts (see `intervention/f5_tts_smoke_gate_failed.md`), so the dedicated "
            "`ref_concat` pre-check this section documents (Step 4) was never executed. This is a "
            "documented consequence of the REQ-7 smoke-gate failure, not a separate omission."
        )

    RESULTS_SMOKE_GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_SMOKE_GATE_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote smoke gate log -> %s", RESULTS_SMOKE_GATE_LOG)


if __name__ == "__main__":
    main()
