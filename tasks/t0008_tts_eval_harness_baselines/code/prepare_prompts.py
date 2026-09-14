"""Prepare val96_prompts.json and filler_prompts_100.json from source data.

Steps:
1. Parse val_list.txt → data/val96_prompts.json
2. Sample 100 clips from data/11labs_david/ → data/filler_prompts_100.json

Both JSON files contain arrays of:
    {"text": "...", "ref_wav": "path/or/null", "ref_duration_s": float_or_null}
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_wav_duration(wav_path: Path) -> float | None:
    """Return duration in seconds or None if unreadable."""
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        return float(info.duration)
    except Exception as exc:
        logger.warning("Cannot read duration from %s: %s", wav_path, exc)
        return None


def build_val96_prompts(
    val_list_path: Path,
    repo_root: Path,
) -> list[dict[str, object]]:
    """Parse val_list.txt and build val96_prompts.json."""
    lines = val_list_path.read_text(encoding="utf-8").strip().split("\n")
    items: list[dict[str, object]] = []
    for line in lines:
        parts = line.split("|")
        wav_rel = parts[0].strip()
        wav_path = repo_root / wav_rel

        # Derive text from filename (strip 6-char hash suffix)
        stem = Path(wav_rel).stem
        text = stem.rsplit("_", 1)[0].replace("_", " ")

        ref_duration: float | None = None
        if wav_path.exists():
            ref_duration = load_wav_duration(wav_path)

        items.append(
            {
                "text": text,
                "ref_wav": str(wav_rel),
                "ref_duration_s": ref_duration,
            }
        )
    logger.info("Built %d val96 prompt items", len(items))
    return items


def build_filler_prompts(
    corpus_dir: Path,
    n: int = 100,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Sample n filler prompts from the 11labs_david corpus."""
    wav_files: list[Path] = sorted(corpus_dir.glob("*.wav"))
    logger.info("Found %d WAV files in %s", len(wav_files), corpus_dir)

    rng = random.Random(seed)
    sampled: list[Path] = wav_files.copy()
    rng.shuffle(sampled)
    sampled = sampled[:n]

    items: list[dict[str, object]] = []
    for wav_path in sampled:
        stem = wav_path.stem
        # Try to decode text from filename (underscore-separated, no hash suffix for 11labs)
        text = stem.replace("_", " ")

        ref_duration = load_wav_duration(wav_path)

        items.append(
            {
                "text": text,
                "ref_wav": str(wav_path),
                "ref_duration_s": ref_duration,
            }
        )

    logger.info("Built %d filler prompt items", len(items))
    return items


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare prompt manifests")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repo root. Defaults to 6 levels above this file.",
    )
    p.add_argument(
        "--filler-n",
        type=int,
        default=100,
        help="Number of filler prompts to sample",
    )
    args = p.parse_args()

    from tasks.t0008_tts_eval_harness_baselines.code.paths import (
        DATA_11LABS_DAVID_DIR,
        DATA_FILLER_PROMPTS,
        DATA_VAL96_PROMPTS,
        REPO_ROOT,
        VAL_LIST,
    )

    repo_root = args.repo_root or REPO_ROOT

    # val96
    if VAL_LIST.exists():
        val_items = build_val96_prompts(val_list_path=VAL_LIST, repo_root=repo_root)
        DATA_VAL96_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
        DATA_VAL96_PROMPTS.write_text(json.dumps(val_items, indent=2), encoding="utf-8")
        logger.info("Saved → %s", DATA_VAL96_PROMPTS)
    else:
        logger.error("val_list.txt not found: %s", VAL_LIST)

    # fillers
    if DATA_11LABS_DAVID_DIR.exists():
        filler_items = build_filler_prompts(
            corpus_dir=DATA_11LABS_DAVID_DIR,
            n=args.filler_n,
        )
        DATA_FILLER_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILLER_PROMPTS.write_text(json.dumps(filler_items, indent=2), encoding="utf-8")
        logger.info("Saved → %s", DATA_FILLER_PROMPTS)
    else:
        logger.error(
            "11labs_david directory not found: %s — run ElevenLabs API regeneration first",
            DATA_11LABS_DAVID_DIR,
        )


if __name__ == "__main__":
    main()
