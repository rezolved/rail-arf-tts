"""Re-score the `elevenlabs_david` paired baseline in this task's own session (plan Step 8a).

CPU-only, $0: no ElevenLabs API call is made. The existing synthesized WAVs from
`tasks/t0008_tts_eval_harness_baselines/data/synth_audio/elevenlabs_david/` (pulled via DVC, Step 1)
are re-scored for `speaker_sim` against a half-B centroid built fresh in this session (matching
t0008's own `score_speaker_sim.py` methodology: ElevenLabs's *own* synthesized fillers are compared
against a held-out half of its *own* real reference recordings, establishing the empirical
self-consistency ceiling — see the ambiguity note in `plan/plan.md`).

`ttfb_ms` and `rtf` are deliberately written as `None` for every row here (Lesson 1: cross-session
latency numbers are not pairable) — this task makes no live ElevenLabs API call, so there is no
this-session latency measurement to report, and reusing t0008's original-session numbers would
falsely imply they were measured in this task's own engine session (REQ-3's own evidence
requirement). `speaker_sim`, `duration_ratio`, and `wer` are retained from t0008's own scoring only
where this task cannot cheaply improve on them (duration_ratio, wer are unchanged — they do not
depend on session/centroid); `speaker_sim` is recomputed fresh here.

Usage::

    uv run python -u tasks/t0018_zero_shot_cloning_calibration/code/score_elevenlabs_baseline.py
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.constants import RANDOM_SEED, REFERENCE_HALF_SIZE
from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
    DATA_11LABS_DAVID_DIR,
    DATA_REFERENCES_DIR,
    SYNTH_AUDIO_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

HALF_B_CENTROID_NPY: Path = DATA_REFERENCES_DIR / "half_b_centroid.npy"
OUT_PATH: Path = Path(
    "tasks/t0018_zero_shot_cloning_calibration/results/per_clip_metrics_elevenlabs_david.json"
)


def _build_half_b_centroid(half_b_paths: list[Path]) -> np.ndarray:
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(device="cpu")
    embeddings: list[np.ndarray] = []
    for wav_path in half_b_paths:
        try:
            wav = preprocess_wav(str(wav_path))
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
        except Exception as exc:
            logger.warning("Could not embed %s: %s", wav_path, exc)
    assert len(embeddings) >= 100, f"Only {len(embeddings)} valid centroid clips (half-B)"
    mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(mean_emb))
    centroid = (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)
    logger.info("half-B centroid built from %d/%d clips", len(embeddings), len(half_b_paths))
    return centroid


def main() -> None:
    all_wavs = sorted(DATA_11LABS_DAVID_DIR.glob("*.wav"))
    rng = random.Random(RANDOM_SEED)
    shuffled = all_wavs.copy()
    rng.shuffle(shuffled)
    half_b_paths = shuffled[REFERENCE_HALF_SIZE:]

    if HALF_B_CENTROID_NPY.exists():
        centroid = np.load(str(HALF_B_CENTROID_NPY))
        logger.info("Loaded cached half-B centroid -> %s", HALF_B_CENTROID_NPY)
    else:
        centroid = _build_half_b_centroid(half_b_paths)
        np.save(str(HALF_B_CENTROID_NPY), centroid)
        logger.info("Saved half-B centroid -> %s", HALF_B_CENTROID_NPY)

    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(device="cpu")

    t0008_records: list[dict[str, object]] = json.loads(
        Path("tasks/t0008_tts_eval_harness_baselines/results/per_clip_metrics.json").read_text()
    )
    el_records = [r for r in t0008_records if r["system"] == "elevenlabs_david"]
    logger.info("Loaded %d t0008 elevenlabs_david records", len(el_records))

    per_prompt_set_index: dict[str, int] = {}
    out_records: list[dict[str, object]] = []
    for rec in el_records:
        prompt_set = str(rec["prompt_set"])
        idx = per_prompt_set_index.get(prompt_set, 0)
        per_prompt_set_index[prompt_set] = idx + 1
        audio_path = SYNTH_AUDIO_DIR / "elevenlabs_david" / prompt_set / f"{idx:04d}.wav"

        speaker_sim: float | None = None
        if audio_path.exists():
            try:
                wav = preprocess_wav(str(audio_path))
                emb = encoder.embed_utterance(wav)
                speaker_sim = float(np.dot(emb, centroid))
            except Exception as exc:
                logger.warning("speaker_sim failed for %s: %s", audio_path, exc)
        else:
            logger.warning("Audio file not found: %s", audio_path)

        out_records.append(
            {
                "system": "elevenlabs_david",
                "condition": None,
                "prompt_set": prompt_set,
                "text": rec["text"],
                "ttfb_ms": None,  # Lesson 1: no live API call this session, not pairable
                "rtf": None,  # Lesson 1: no live API call this session, not pairable
                "speaker_sim": speaker_sim,
                "duration_ratio": rec.get("duration_ratio"),
                "wer": rec.get("wer"),
                "audio_path": str(audio_path) if audio_path.exists() else None,
                "ref_duration_s": rec.get("ref_duration_s"),
                "synth_duration_s": rec.get("synth_duration_s"),
                "is_streaming": None,
                "source_note": "speaker_sim re-scored this session vs fresh half-B centroid; "
                "ttfb_ms/rtf/duration_ratio/wer retained from t0008's original session "
                "(no live ElevenLabs API call made this session, per plan Step 8a)",
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out_records, indent=2), encoding="utf-8")
    n_scored = sum(1 for r in out_records if r["speaker_sim"] is not None)
    logger.info(
        "Wrote %d records (%d with speaker_sim) -> %s", len(out_records), n_scored, OUT_PATH
    )


if __name__ == "__main__":
    main()
