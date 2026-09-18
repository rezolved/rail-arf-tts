"""Write `results/listening_guide.md` (plan.md Step 16, owner-correction REQ-16/REQ-17).

3 columns per comparison text: this task's own best-setting output (new reference), t0018's
wrong-voice-reference output (continuity only), and the actual production-voice val96 original.
"""

from __future__ import annotations

import json
import logging

from tasks.t0021_zero_shot_latency_reduction.code.build_comparison_set import (
    select_comparison_texts,
)
from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    RESULTS_AUDIO_COMPARISON_DIR,
    RESULTS_LISTENING_GUIDE,
    RESULTS_PER_CLIP_METRICS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

HEADER_NOTE = (
    "**Necessary, not sufficient (owner correction #7):** the automated `hardened_gate_pass` "
    "check (this task's own copy of t0015's `audio_quality_check.py`) is known to pass clips a "
    "human listener would describe as 'voice plus strong noise.' A PASS here does not certify "
    "audio quality -- the owner will listen to this guide's clips directly before any "
    "production-readiness conclusion is drawn."
)


def _cell(text: str, records_for_text: list[dict[str, object]], filename: str) -> str:
    fpath = RESULTS_AUDIO_COMPARISON_DIR / filename
    if not fpath.exists():
        return "-"
    match = None
    for r in records_for_text:
        # `new_ref` cells are looked up by filename fragment already encoding system+variant;
        # for those the caller passes in a pre-filtered `records_for_text` (see main()).
        match = r
        break
    link = f"[listen](audio_samples/comparison_set/{filename})"
    if match is None:
        return link
    sim = match.get("speaker_sim")
    wer = match.get("wer")
    gate = match.get("hardened_gate_pass")
    sim_str = f"{sim:.3f}" if isinstance(sim, int | float) else "n/a"
    wer_str = f"{wer:.2f}" if isinstance(wer, int | float) else "n/a"
    gate_str = "PASS" if gate is True else ("FAIL" if gate is False else "n/a")
    return f"{link} sim={sim_str} wer={wer_str} gate={gate_str}"


def main() -> None:
    records: list[dict[str, object]] = json.loads(
        RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    )
    text_to_records: dict[str, list[dict[str, object]]] = {}
    for r in records:
        text_to_records.setdefault(str(r.get("text", "")).strip(), []).append(r)

    comparison_texts = select_comparison_texts()

    lines: list[str] = []
    lines.append("# Listening Guide -- Zero-Shot TTFB Latency Reduction (Corrected Voice)")
    lines.append("")
    lines.append(
        "One row per comparison text (3 fixed gate texts + 7 seeded val96 prompts). Three "
        "columns: this task's own best-setting output (built from the CORRECTED "
        "`data/v4/val/wavs` reference), t0018's wrong-voice-reference output (continuity only, "
        "NOT comparable as a quality baseline), and the actual production-voice val96 original."
    )
    lines.append("")
    lines.append(HEADER_NOTE)
    lines.append("")
    lines.append(
        "**Deviation, documented (not silent):** the `t0018_old_ref` column is entirely absent "
        "below (`-` in every cell) because `dvc pull` for t0018's `results/audio_samples/"
        "comparison_set.dvc` failed in this task's environment (an Azure credential-chain issue "
        "outside this task's control -- see `code/build_comparison_set.py`'s module docstring "
        "for the full explanation). The 3 fixed `GATE_TEXT_NAMES` "
        "(`lining_up_suggestions_17`, `lining_up_suggestions_10`, `putting_them_head_to_head_15`) "
        "are also not among the 100 sampled filler prompts this session actually synthesized "
        "(same structural gap t0018 itself documented) -- their `new_ref` cells are `-` too, and "
        "they have no val96 original by construction (they are filler-corpus phrases, not val96 "
        "prompts)."
    )
    lines.append("")

    columns = [
        ("cosyvoice2 (new ref)", "new_ref_cosyvoice2"),
        ("cosyvoice2 (t0018 old ref)", "old_ref_cosyvoice2"),
        ("chatterbox (new ref)", "new_ref_chatterbox"),
        ("chatterbox (t0018 old ref)", "old_ref_chatterbox"),
        ("val96 original", "val96_original"),
    ]
    header = "| Text | " + " | ".join(c[0] for c in columns) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(columns)) + " |"
    lines.append(header)
    lines.append(sep)

    for text_id, text in comparison_texts:
        recs_for_text = text_to_records.get(text.strip(), [])
        row_cells = [f"**{text}**"]
        for _, col_key in columns:
            if col_key == "val96_original":
                fname = f"{text_id}__val96_original.wav"
                # The val96 original is a ground-truth recording, not a synthesis output -- it
                # has no corresponding per-clip speaker_sim/wer/gate record of its own (those
                # fields describe SYNTHESIZED clips scored against it, not the clip itself).
                cell_records = []
            elif col_key.startswith("new_ref_"):
                system = col_key[len("new_ref_") :]
                matches = sorted(
                    RESULTS_AUDIO_COMPARISON_DIR.glob(f"{text_id}__{system}_*__new_ref.wav")
                )
                if len(matches) == 0:
                    row_cells.append("-")
                    continue
                fname = matches[0].name
                cell_records = [r for r in recs_for_text if r.get("system") == system]
            else:  # old_ref_<system>
                system = col_key[len("old_ref_") :]
                fname = f"{text_id}__{system}__t0018_old_ref.wav"
                cell_records = []  # t0018's own per-clip metrics are not this task's data
            row_cells.append(_cell(text, cell_records, fname))
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append("## What to listen for")
    lines.append("")
    for text_id, text in comparison_texts:
        recs_for_text = text_to_records.get(text.strip(), [])
        if len(recs_for_text) == 0:
            lines.append(
                f"* **{text}** ({text_id}): not synthesized this session (see the deviation "
                "note above)."
            )
            continue
        high_wer = [
            r for r in recs_for_text if isinstance(r.get("wer"), int | float) and r["wer"] > 0.3
        ]
        note = (
            "Compare timbre match against the production-voice val96 original and listen for the "
            "t0015 gate's known 'voice plus strong noise' failure mode even on a PASS."
        )
        if len(high_wer) > 0:
            bad = ", ".join(sorted({str(r["system"]) for r in high_wer}))
            note = f"High WER on this text for: {bad} -- listen for mispronounced words."
        lines.append(f"* **{text}** ({text_id}): {note}")

    RESULTS_LISTENING_GUIDE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_LISTENING_GUIDE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote listening guide -> %s", RESULTS_LISTENING_GUIDE)


if __name__ == "__main__":
    main()
