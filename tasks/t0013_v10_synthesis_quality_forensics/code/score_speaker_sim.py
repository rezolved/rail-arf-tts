"""Score `speaker_sim` (GE2E cosine vs. 11labs_david mean embedding) for this task's audio samples.

Adapted (copied, not imported -- cross-task code reuse rule, `research/research_summary.md` point
9) from `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`'s centroid-building
logic. Simplified for this task's scale: a handful of forensic clips scored once, not a
100+-request benchmark protocol (`plan/plan.md` Approach section explains why the full
`tts-benchmark-run` protocol does not apply here).

Per `overview/metrics/speaker_sim.md`, `resemblyzer` is kept out of the main project dependencies
(`pip install .[speaker-sim]`) -- here it is already installed in the isolated
`code/.venv-styletts2` venv (Milestone C step 7), so this script runs through that interpreter, not
the main project `.venv`.

Usage::

    .venv-styletts2/bin/python code/score_speaker_sim.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
from tasks.t0013_v10_synthesis_quality_forensics.code.paths import (  # noqa: E402
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
        RESULTS_AUDIO_DIR / "control_epochs_2nd_00020.wav",
        RESULTS_AUDIO_DIR / "v10_epoch16_primary.wav",
        RESULTS_AUDIO_DIR / "v10_epoch14_backup.wav",
    ]
    results = [score_wav(t, centroid, encoder) for t in targets]
    for r in results:
        print(f"{r.wav}: speaker_sim={r.speaker_sim} error={r.error}")

    out_path = RESULTS_DIR / "speaker_sim_scores.json"
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
