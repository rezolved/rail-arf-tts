"""Harness orchestrator: prompt-set loading, reference split, per-system evaluation.

This module provides:
- build_reference_split(): build ElevenLabs centroid from half-A, return half-B paths
- load_val96_prompts(): load val_96 prompt texts + ref wav paths
- load_filler_prompts(): load filler prompt manifest
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.constants import (
    RANDOM_SEED,
    REFERENCE_HALF_SIZE,
)
from tasks.t0008_tts_eval_harness_baselines.code.paths import (
    DATA_FILLER_PROMPTS,
    DATA_VAL96_PROMPTS,
    VAL_LIST,
)

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PromptItem:
    text: str
    ref_wav: Path | None  # reference audio for duration ratio (None for fillers)
    ref_duration_s: float | None  # pre-computed reference duration


# ── Reference split ───────────────────────────────────────────────────────────


def build_reference_split(
    corpus_dir: Path,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, list[Path]]:
    """Split the ElevenLabs corpus 679/679, build centroid from half-A.

    Args:
        corpus_dir: Directory containing 1358 WAV files.
        seed: Random seed for reproducible split.

    Returns:
        (centroid_array, half_b_paths) where centroid_array is a (256,) float32 ndarray
        and half_b_paths is a list of 679 Path objects.

    Raises:
        RuntimeError: if fewer than 2 WAV files found.
        ImportError: if resemblyzer is not installed.
    """
    from tasks.t0008_tts_eval_harness_baselines.code.scoring import build_centroid

    wav_files: list[Path] = sorted(corpus_dir.glob("*.wav"))
    assert len(wav_files) >= 2, f"Corpus too small: {len(wav_files)} WAV files in {corpus_dir}"

    rng = random.Random(seed)
    shuffled = wav_files.copy()
    rng.shuffle(shuffled)

    half_a = shuffled[:REFERENCE_HALF_SIZE]
    half_b = shuffled[REFERENCE_HALF_SIZE:]

    logger.info(
        "Reference split: half_A=%d clips, half_B=%d clips (seed=%d)",
        len(half_a),
        len(half_b),
        seed,
    )

    centroid = build_centroid(half_a)
    return centroid, half_b


# ── Prompt loading ────────────────────────────────────────────────────────────


def load_val96_prompts() -> list[PromptItem]:
    """Load val_96 prompts from the cached JSON manifest.

    Falls back to deriving text from WAV filenames in val_list.txt if JSON not found.
    """
    if DATA_VAL96_PROMPTS.exists():
        raw: list[dict[str, object]] = json.loads(DATA_VAL96_PROMPTS.read_text(encoding="utf-8"))
        return [
            PromptItem(
                text=str(item["text"]),
                ref_wav=Path(str(item["ref_wav"])) if item.get("ref_wav") else None,
                ref_duration_s=(
                    float(item["ref_duration_s"]) if item.get("ref_duration_s") else None
                ),
            )
            for item in raw
        ]

    # Fallback: derive from val_list.txt
    return _load_val96_from_manifest()


def _load_val96_from_manifest() -> list[PromptItem]:
    """Parse val_list.txt to extract prompt texts and reference WAV paths."""
    lines = VAL_LIST.read_text(encoding="utf-8").strip().split("\n")
    items: list[PromptItem] = []
    for line in lines:
        parts = line.split("|")
        wav_path = Path(parts[0])
        # Decode text from filename: strip hash suffix (last 6 chars after final _)
        stem = wav_path.stem
        text_part = stem.rsplit("_", 1)[0].replace("_", " ")
        items.append(PromptItem(text=text_part, ref_wav=wav_path, ref_duration_s=None))
    return items


def load_filler_prompts() -> list[PromptItem]:
    """Load filler prompts from the cached JSON manifest."""
    if not DATA_FILLER_PROMPTS.exists():
        raise FileNotFoundError(
            f"Filler prompt manifest not found: {DATA_FILLER_PROMPTS}. "
            "Run prepare_prompts.py first."
        )
    raw: list[dict[str, object]] = json.loads(DATA_FILLER_PROMPTS.read_text(encoding="utf-8"))
    return [
        PromptItem(
            text=str(item["text"]),
            ref_wav=Path(str(item["ref_wav"])) if item.get("ref_wav") else None,
            ref_duration_s=float(item["ref_duration_s"]) if item.get("ref_duration_s") else None,
        )
        for item in raw
    ]


def get_prompts(prompt_set: str) -> list[PromptItem]:
    """Return prompts for a given prompt set name.

    Args:
        prompt_set: One of 'val96', 'fillers', or 'both'.

    Returns:
        Combined list of PromptItems.
    """
    from tasks.t0008_tts_eval_harness_baselines.code.constants import (
        PROMPT_SET_BOTH,
        PROMPT_SET_FILLERS,
        PROMPT_SET_VAL96,
    )

    if prompt_set == PROMPT_SET_VAL96:
        return load_val96_prompts()
    elif prompt_set == PROMPT_SET_FILLERS:
        return load_filler_prompts()
    elif prompt_set == PROMPT_SET_BOTH:
        return load_val96_prompts() + load_filler_prompts()
    else:
        raise ValueError(f"Unknown prompt_set: {prompt_set!r}")


def get_prompts_by_set(prompt_set: str) -> dict[str, list[PromptItem]]:
    """Return prompts grouped by their set name (for per-set metrics).

    Args:
        prompt_set: One of 'val96', 'fillers', or 'both'.

    Returns:
        Dict mapping set name → prompts.
    """
    from tasks.t0008_tts_eval_harness_baselines.code.constants import (
        PROMPT_SET_BOTH,
        PROMPT_SET_FILLERS,
        PROMPT_SET_VAL96,
    )

    if prompt_set == PROMPT_SET_VAL96:
        return {PROMPT_SET_VAL96: load_val96_prompts()}
    elif prompt_set == PROMPT_SET_FILLERS:
        return {PROMPT_SET_FILLERS: load_filler_prompts()}
    elif prompt_set == PROMPT_SET_BOTH:
        return {
            PROMPT_SET_VAL96: load_val96_prompts(),
            PROMPT_SET_FILLERS: load_filler_prompts(),
        }
    else:
        raise ValueError(f"Unknown prompt_set: {prompt_set!r}")
