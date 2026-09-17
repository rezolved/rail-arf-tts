"""Build the comparison audio set (plan Step 15, REQ-8): 10 fixed texts x every non-null variant.

10 texts = 3 fixed `GATE_TEXT_NAMES` (filler prompts) + 7 val96 prompts sampled with
`random.Random(COMPARISON_SET_SEED).sample(val96_texts, COMPARISON_SET_VAL96_COUNT)`.

Usage::

    uv run python -u tasks/t0018_zero_shot_cloning_calibration/code/build_comparison_set.py
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

from tasks.t0008_tts_eval_harness_baselines.code.harness import (
    load_val96_prompts,
)
from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    COMPARISON_SET_SEED,
    COMPARISON_SET_VAL96_COUNT,
    GATE_TEXT_NAMES,
)
from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
    RESULTS_AUDIO_COMPARISON_DIR,
    RESULTS_PER_CLIP_METRICS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _slugify_for_lookup(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def select_comparison_texts() -> list[tuple[str, str]]:
    """Return [(text_id, text), ...] -- 3 gate texts + 7 seeded val96 samples."""
    selected: list[tuple[str, str]] = []
    for name in GATE_TEXT_NAMES:
        # name is e.g. "lining_up_suggestions_17" -> text "lining up suggestions 17"
        text = name.replace("_", " ")
        selected.append((name, text))

    val96_items = load_val96_prompts()
    val96_texts = [item.text for item in val96_items]
    sampled = random.Random(COMPARISON_SET_SEED).sample(val96_texts, COMPARISON_SET_VAL96_COUNT)
    for text in sampled:
        text_id = _slugify_for_lookup(text)[:40]
        selected.append((text_id, text))

    return selected


def main() -> None:
    RESULTS_AUDIO_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = json.loads(
        RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    )

    comparison_texts = select_comparison_texts()
    logger.info("Comparison texts: %s", [t for _, t in comparison_texts])

    n_copied = 0
    for text_id, text in comparison_texts:
        matches = [
            r
            for r in records
            if str(r.get("text", "")).strip() == text.strip() and r.get("audio_path")
        ]
        for rec in matches:
            system = str(rec["system"])
            condition = rec.get("condition")
            variant_slug = f"{system}_{condition}" if condition else system
            src = Path(str(rec["audio_path"]))
            if not src.exists():
                logger.warning("Missing source audio: %s", src)
                continue
            dest = RESULTS_AUDIO_COMPARISON_DIR / f"{text_id}__{variant_slug}.wav"
            shutil.copy2(src, dest)
            n_copied += 1

    logger.info("Copied %d comparison clips -> %s", n_copied, RESULTS_AUDIO_COMPARISON_DIR)


if __name__ == "__main__":
    main()
