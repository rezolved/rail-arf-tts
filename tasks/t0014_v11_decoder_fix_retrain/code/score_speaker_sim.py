"""Score `speaker_sim` (GE2E cosine vs. 11labs_david mean embedding) for v11's audio samples
(REQ-9).

Copied forward (paths only changed) from
`tasks/t0013_v10_synthesis_quality_forensics/code/score_speaker_sim.py`, itself adapted from
`tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`'s centroid-building logic.
`resemblyzer` is kept out of the main project dependencies (per `overview/metrics/speaker_sim.md`)
-- run through the isolated training venv on the VM (also has torch+CUDA already), not the main
project `.venv`.

Usage::

    venv/bin/python code/score_speaker_sim.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tasks.t0014_v11_decoder_fix_retrain.code.paths import (
    ELEVENLABS_DAVID_DIR,
    RESULTS_AUDIO_DIR,
    RESULTS_DIR,
)

MIN_CORPUS_SIZE = 1000
CENTROID_SEED = 42
CENTROID_SAMPLE_SIZE = 679  # matches t0008's half-A split size, for methodology consistency


@dataclass(frozen=True, slots=True)
class SpeakerSimResult:
    wav: str
    speaker_sim: float | None
    error: str | None


def build_centroid(corpus_dir: Path, encoder: object) -> np.ndarray:
    import random

    from resemblyzer import preprocess_wav  # type: ignore[import-untyped]

    wav_files = sorted(corpus_dir.glob("*.wav"))
    assert len(wav_files) >= MIN_CORPUS_SIZE, (
        f"Corpus has only {len(wav_files)} WAVs -- expected >= {MIN_CORPUS_SIZE}. Halt."
    )
    rng = random.Random(CENTROID_SEED)
    shuffled = wav_files.copy()
    rng.shuffle(shuffled)
    sample = shuffled[:CENTROID_SAMPLE_SIZE]

    embeddings: list[np.ndarray] = []
    for wav_path in sample:
        try:
            wav = preprocess_wav(str(wav_path))
            emb = encoder.embed_utterance(wav)  # type: ignore[attr-defined]
            embeddings.append(emb)
        except Exception:  # noqa: BLE001 -- best-effort centroid build, matches t0008's pattern
            continue
    assert len(embeddings) >= 100, f"Only {len(embeddings)} valid centroid clips"
    mean_emb = np.mean(np.stack(embeddings), axis=0)
    norm = float(np.linalg.norm(mean_emb))
    return (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)


def score_wav(wav_path: Path, centroid: np.ndarray, encoder: object) -> SpeakerSimResult:
    from resemblyzer import preprocess_wav  # type: ignore[import-untyped]

    if not wav_path.exists():
        return SpeakerSimResult(wav=str(wav_path), speaker_sim=None, error="file not found")
    try:
        wav = preprocess_wav(str(wav_path))
        emb = encoder.embed_utterance(wav)  # type: ignore[attr-defined]
        sim = float(np.dot(emb, centroid))
        return SpeakerSimResult(wav=str(wav_path), speaker_sim=sim, error=None)
    except Exception as exc:  # noqa: BLE001 -- record failure per-clip, don't abort the whole run
        return SpeakerSimResult(wav=str(wav_path), speaker_sim=None, error=str(exc))


def main() -> None:
    import torch
    from resemblyzer import VoiceEncoder  # type: ignore[import-untyped]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder(device=device)
    print(f"VoiceEncoder on {device}")

    centroid = build_centroid(ELEVENLABS_DAVID_DIR, encoder)
    print(f"Centroid built from up to {CENTROID_SAMPLE_SIZE} clips of {ELEVENLABS_DAVID_DIR}")

    targets = [
        RESULTS_AUDIO_DIR / "ft" / "v11_best.wav",
    ]
    results = [score_wav(t, centroid, encoder) for t in targets]
    for r in results:
        print(f"{r.wav}: speaker_sim={r.speaker_sim} error={r.error}")

    out_path = RESULTS_DIR / "speaker_sim_scores.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
