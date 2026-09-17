"""Write `results/listening_guide.md` (plan Step 16, REQ-8).

One row per comparison text (10 rows), one column per system/condition (up to 6) plus ElevenLabs
and `kokoro_v3_bundle`, each cell a clickable relative markdown link with per-clip `speaker_sim`/
`wer`/gate verdict, plus a "what to listen for" line per row.
"""

from __future__ import annotations

import json
import logging

from tasks.t0018_zero_shot_cloning_calibration.code.build_comparison_set import (
    select_comparison_texts,
)
from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    CLONING_SYSTEMS,
    REFERENCE_CONDITIONS,
)
from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
    RESULTS_AUDIO_COMPARISON_DIR,
    RESULTS_LISTENING_GUIDE,
    RESULTS_PER_CLIP_METRICS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _all_variant_slugs() -> list[str]:
    slugs = []
    for s in CLONING_SYSTEMS:
        for c in REFERENCE_CONDITIONS:
            slugs.append(f"{s}_{c}")
    slugs.append("elevenlabs_david")
    slugs.append("kokoro_v3_bundle")
    return slugs


def main() -> None:
    records: list[dict[str, object]] = json.loads(
        RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    )
    text_to_records: dict[str, list[dict[str, object]]] = {}
    for r in records:
        text_to_records.setdefault(str(r.get("text", "")).strip(), []).append(r)

    comparison_texts = select_comparison_texts()
    variant_slugs = _all_variant_slugs()
    present_slugs = [
        s for s in variant_slugs if any((RESULTS_AUDIO_COMPARISON_DIR).glob(f"*__{s}.wav"))
    ]

    lines: list[str] = []
    lines.append("# Listening Guide -- Zero-Shot Voice-Cloning Calibration")
    lines.append("")
    lines.append(
        "One row per comparison text (3 fixed gate texts + 7 seeded val96 prompts), one column "
        "per system/condition variant that produced audio, plus ElevenLabs (original reference) "
        "and `kokoro_v3_bundle`. Each cell links to the clip, with `speaker_sim` / `wer` / gate "
        "verdict in the cell text. Clips are relative to `results/audio_samples/comparison_set/`."
    )
    lines.append("")
    lines.append(
        "**Deviation, documented (not silent):** the 3 fixed `GATE_TEXT_NAMES` "
        "(`lining_up_suggestions_17`, `lining_up_suggestions_10`, `putting_them_head_to_head_15`) "
        'are **not** among the 100 filler prompts `get_prompts_by_set("both")` actually returned '
        "and synthesized this session -- verified directly: none of these 3 texts appear anywhere "
        "in `results/per_clip_metrics.json`. t0008's `filler_prompts_100.json` samples only 100 "
        "of the 1364 corpus phrases, and these 3 happen not to be in that sample. Their rows "
        "below are therefore all `-` (no clip exists to link). `kokoro_v3_bundle` is also "
        "entirely absent from every row: it was not re-synthesized this session (see "
        "`intervention/kokoro_v3_bundle_not_remeasured.md`), so no comparison clips exist for it "
        "either. `cosyvoice2_ref_concat` is absent for the same structural reason as its column "
        "never appearing: it produced 0 successful clips (REQ-6 null, see `results/tables.json`)."
    )
    lines.append("")
    header = "| Text | " + " | ".join(present_slugs) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(present_slugs)) + " |"
    lines.append(header)
    lines.append(sep)

    for text_id, text in comparison_texts:
        row_cells = [f"**{text}**"]
        recs_for_text = text_to_records.get(text.strip(), [])
        for slug in present_slugs:
            match = None
            for r in recs_for_text:
                system = str(r["system"])
                condition = r.get("condition")
                rec_slug = f"{system}_{condition}" if condition else system
                if rec_slug == slug:
                    match = r
                    break
            fname = f"{text_id}__{slug}.wav"
            fpath = RESULTS_AUDIO_COMPARISON_DIR / fname
            if not fpath.exists():
                row_cells.append("-")
                continue
            sim = match.get("speaker_sim") if match else None
            wer = match.get("wer") if match else None
            gate = match.get("hardened_gate_pass") if match else None
            sim_str = f"{sim:.3f}" if isinstance(sim, int | float) else "n/a"
            wer_str = f"{wer:.2f}" if isinstance(wer, int | float) else "n/a"
            gate_str = "PASS" if gate is True else ("FAIL" if gate is False else "n/a")
            link = f"[listen](audio_samples/comparison_set/{fname})"
            row_cells.append(f"{link} sim={sim_str} wer={wer_str} gate={gate_str}")
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append("## What to listen for")
    lines.append("")
    for text_id, text in comparison_texts:
        recs_for_text = text_to_records.get(text.strip(), [])
        if len(recs_for_text) == 0:
            lines.append(
                f"* **{text}** ({text_id}): not synthesized this session -- this fixed gate text "
                "is not among the 100 filler prompts actually sampled by t0008's "
                "filler_prompts_100.json (see the deviation note above)."
            )
            continue
        high_wer = [
            r for r in recs_for_text if isinstance(r.get("wer"), int | float) and r["wer"] > 0.3
        ]
        note = "Compare timbre match and accent drift against the ElevenLabs original."
        if len(high_wer) > 0:
            bad_systems = ", ".join(sorted({str(r["system"]) for r in high_wer}))
            note = f"High WER on this text for: {bad_systems} -- listen for mispronounced words."
        lines.append(f"* **{text}** ({text_id}): {note}")

    RESULTS_LISTENING_GUIDE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_LISTENING_GUIDE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote listening guide -> %s", RESULTS_LISTENING_GUIDE)


if __name__ == "__main__":
    main()
