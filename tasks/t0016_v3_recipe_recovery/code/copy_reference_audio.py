"""Milestone 3 Step 13 (REQ-11): copy (not re-synthesize) per-epoch and ElevenLabs reference audio
into `results/audio_samples/` so the shipped-bundle, per-epoch, and reference clips sit side by
side for human listening.

* `results/audio_samples/v3_per_epoch/`: the 20 `epoch{0..3}_phrase{1..5}.wav` files plus the 15
  `v3/ep{6,7,8}_p{1..5}.wav` and 20 `v3b/ep{6,7,8,9}_p{1..5}.wav` files (55 total), preserving
  their relative subpaths.
* `results/audio_samples/elevenlabs_reference/`: the matching original ElevenLabs David clips for
  the 3 fixed gate texts and any val96 texts synthesized in `synthesize_v3_shipped.py` that have a
  same-text ElevenLabs clip in `data/11labs_david/` (matched by filename stem).

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.copy_reference_audio
"""

from __future__ import annotations

import shutil

from tasks.t0016_v3_recipe_recovery.code.paths import (
    ELEVENLABS_DAVID_DIR,
    RESULTS_AUDIO_ELEVENLABS_REFERENCE_DIR,
    RESULTS_AUDIO_V3_PER_EPOCH_DIR,
    RESULTS_AUDIO_V3_SHIPPED_DIR,
    V3_AUDIO_DIR,
)

GATE_TEXT_SLUGS: tuple[str, ...] = (
    "lining_up_suggestions_17",
    "lining_up_suggestions_10",
    "putting_them_head_to_head_15",
)


def copy_per_epoch_audio() -> int:
    count = 0
    RESULTS_AUDIO_V3_PER_EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    for wav_path in sorted(V3_AUDIO_DIR.glob("epoch*_phrase*.wav")):
        shutil.copy2(wav_path, RESULTS_AUDIO_V3_PER_EPOCH_DIR / wav_path.name)
        count += 1
    for subdir_name in ("v3", "v3b"):
        subdir = V3_AUDIO_DIR / subdir_name
        if not subdir.is_dir():
            continue
        out_subdir = RESULTS_AUDIO_V3_PER_EPOCH_DIR / subdir_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        for wav_path in sorted(subdir.glob("*.wav")):
            shutil.copy2(wav_path, out_subdir / wav_path.name)
            count += 1
    return count


def copy_elevenlabs_reference() -> list[str]:
    """Copy ElevenLabs clips matching every text synthesized into `v3_shipped/`, matched by
    filename stem (the shipped-bundle output filenames ARE the source text's slug for the gate
    texts; val96 clips are matched only when a same-slug ElevenLabs clip happens to exist, which
    is not expected for `val96_seed42_*` -- those are noted as "no matching reference" rather than
    silently skipped)."""
    RESULTS_AUDIO_ELEVENLABS_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    shipped_slugs = [p.stem for p in sorted(RESULTS_AUDIO_V3_SHIPPED_DIR.glob("*.wav"))]
    for slug in shipped_slugs:
        candidate = ELEVENLABS_DAVID_DIR / f"{slug}.wav"
        if candidate.exists():
            shutil.copy2(candidate, RESULTS_AUDIO_ELEVENLABS_REFERENCE_DIR / candidate.name)
            copied.append(slug)
    return copied


def main() -> None:
    n_per_epoch = copy_per_epoch_audio()
    print(f"Copied {n_per_epoch} per-epoch/v3/v3b files to {RESULTS_AUDIO_V3_PER_EPOCH_DIR}")

    matched = copy_elevenlabs_reference()
    print(f"Copied {len(matched)} matching ElevenLabs reference clips: {matched}")
    all_shipped = [p.stem for p in sorted(RESULTS_AUDIO_V3_SHIPPED_DIR.glob("*.wav"))]
    unmatched = [s for s in all_shipped if s not in matched]
    if len(unmatched) > 0:
        print(
            f"No matching ElevenLabs reference clip for: {unmatched} "
            "(expected for val96_seed42_* -- those slugs are synthetic sample identifiers, not "
            "ElevenLabs corpus filenames)"
        )


if __name__ == "__main__":
    main()
