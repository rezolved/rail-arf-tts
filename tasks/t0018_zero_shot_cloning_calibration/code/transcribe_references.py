"""Derive `ref_single`/`ref_concat` reference transcripts from the corpus's own filename convention.

**Deviation from an ASR-based approach (discovered during implementation):** an earlier version of
this script transcribed the reference clips with faster-whisper, but Whisper badly hallucinates on
these clips — each is a single short phrase, and `ref_single`/`ref_concat` concatenate 10-24 near-
identical-length clips with only 0.2s silence gaps, which sends Whisper into repetition loops (e.g.
"absolutely absolutely absolutely ..." x150). This is useless as `ref_text` for F5-TTS or
`prompt_text` for CosyVoice2 (both need the reference audio's *actual* transcript to align
generation).

Instead: `data/filler_prompts_100.json` cross-references show that every `11labs_david/*.wav`
filename **is** its own ground-truth transcript, verbatim, once underscores are replaced with
spaces (verified directly against every source clip used in `ref_single`/`ref_concat`: e.g.
`bringing_that_up_13.wav` -> `"bringing that up 13"`, `absolutely_08.wav` -> `"absolutely 08"`).
This is exact, not an ASR guess, and is used instead.

Usage::

    uv run python -u tasks/t0018_zero_shot_cloning_calibration/code/transcribe_references.py
"""

from __future__ import annotations

import json
import logging

from tasks.t0018_zero_shot_cloning_calibration.code.paths import REFERENCES_MANIFEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _filename_to_text(filename: str) -> str:
    """`bringing_that_up_13.wav` -> `"bringing that up 13"` (verified ground truth, not ASR)."""
    stem = filename.removesuffix(".wav")
    return stem.replace("_", " ")


def main() -> None:
    manifest = json.loads(REFERENCES_MANIFEST.read_text(encoding="utf-8"))

    ref_single_text = " ".join(_filename_to_text(f) for f in manifest["ref_single_filenames"])
    ref_concat_text = " ".join(_filename_to_text(f) for f in manifest["ref_concat_filenames"])

    manifest["ref_single_transcript"] = ref_single_text
    manifest["ref_concat_transcript"] = ref_concat_text
    manifest["transcript_source"] = "filename_ground_truth"  # not ASR — see module docstring

    logger.info("ref_single_transcript -> %r", ref_single_text)
    logger.info("ref_concat_transcript -> %r", ref_concat_text)

    REFERENCES_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Updated manifest -> %s", REFERENCES_MANIFEST)


if __name__ == "__main__":
    main()
