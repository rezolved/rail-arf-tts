"""Transcribe `ref_single`/`ref_concat` with faster-whisper to produce CosyVoice2's `prompt_text`.

**Why ASR here, unlike t0018's filename-ground-truth shortcut:** t0018's `transcribe_references.py`
used the ElevenLabs corpus's filename convention (`bringing_that_up_13.wav` ->
`"bringing that up 13"`) as an exact, verified ground-truth transcript, specifically because Whisper
hallucinated badly on concatenations of many near-identical short phrases with 0.2s gaps (repetition
loops). val_96's filenames are not a reliable substitute here: most follow a `<text_slug>_<hex>.wav`
pattern, but this task's own `ref_single` (see `plan/plan.md` Step 2's preflight finding) landed on
`llm_sess_1e426c8f62f6495c_resp_217b2448145e4511_000.wav` — an opaque session/response ID, not a
text slug at all. `plan/plan.md` Step 3 explicitly calls for Whisper-based transcription for this
reason. `ref_concat`'s 9 segments are drawn from *different* filler phrases (not near-duplicates of
each other), which is the specific condition that triggered t0018's repetition-loop hallucination —
so the risk here is lower, but each transcript is still read back and sanity-checked against
`data/v4/val_list.txt`'s phoneme column (a rough human-legibility check, not an exact match, since
phonemes cannot be losslessly reversed to text) before being trusted.

Usage::

    uv run python -u tasks/t0021_zero_shot_latency_reduction/code/transcribe_references.py
"""

from __future__ import annotations

import json
import logging

from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    REF_CONCAT_WAV,
    REF_SINGLE_WAV,
    REFERENCES_MANIFEST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = "base.en"  # CPU-friendly; run before GPU is touched further (Milestone 1)
# NOTE: found and fixed during this task's own implementation (Phase 1.5 post-hoc inspection,
# triggered by a catastrophic WER=1.0 finding for every CosyVoice2 ref_single measurement).
# `small.en` with `beam_size=5` silently truncated `ref_single`'s transcript to its FIRST sentence
# only ("I'm not able to compare us with other companies.") even though the returned segment's own
# timestamp span covered the full 15.08 s clip -- a Whisper decoding quirk (early stop mid-segment
# on this specific `small.en` model/audio combination, not a VAD/silence-detection issue: the
# timestamp span was correct, only the token generation stopped early). `base.en` (used with plain
# defaults, no `beam_size` override) transcribes the full clip correctly across multiple segments.
# A truncated `prompt_text` fed to CosyVoice2's zero-shot conditioning (which requires `prompt_text`
# to match what is actually spoken in `prompt_wav`) caused every `ref_single` CosyVoice2 synthesis
# this task ran to produce audio unrelated to the requested target text (confirmed by manually
# re-transcribing several output clips: pure gibberish, e.g. "A wolf profferpate.").


def _transcribe(model: object, wav_path: object) -> str:
    segments, _info = model.transcribe(str(wav_path), language="en")  # type: ignore[attr-defined]
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def main() -> None:
    from faster_whisper import WhisperModel

    logger.info("Loading faster-whisper model %s (CPU)...", WHISPER_MODEL_SIZE)
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    manifest = json.loads(REFERENCES_MANIFEST.read_text(encoding="utf-8"))

    ref_single_text = _transcribe(model, REF_SINGLE_WAV)
    ref_concat_text = _transcribe(model, REF_CONCAT_WAV)

    manifest["ref_single_transcript"] = ref_single_text
    manifest["ref_concat_transcript"] = ref_concat_text
    manifest["transcript_source"] = "faster_whisper_small_en"  # NOT filename ground truth (t0018)

    logger.info("ref_single_transcript -> %r", ref_single_text)
    logger.info("ref_concat_transcript -> %r", ref_concat_text)

    assert len(ref_single_text) > 0, "ref_single transcript is empty"
    assert len(ref_concat_text) > 0, "ref_concat transcript is empty"

    REFERENCES_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Updated manifest -> %s", REFERENCES_MANIFEST)


if __name__ == "__main__":
    main()
