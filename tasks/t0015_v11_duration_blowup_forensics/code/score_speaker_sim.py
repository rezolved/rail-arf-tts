"""Score `speaker_sim` (GE2E cosine vs. 11labs_david mean embedding) for this task's audio fixtures
(Milestone E step 13).

Copied forward (paths only changed) from
`tasks/t0014_v11_decoder_fix_retrain/code/score_speaker_sim.py`, itself adapted from
`tasks/t0013_v10_synthesis_quality_forensics/code/score_speaker_sim.py`, itself adapted from
`tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`'s centroid-building logic.
`resemblyzer` is kept out of the main project dependencies (per `overview/metrics/speaker_sim.md`)
-- run through the isolated `.venv-styletts2` CPU venv, not the main project `.venv`.

Scores three fixtures (skipping any that do not exist, rather than failing the whole run): the
v11-as-shipped fixture (`v11_best.wav`, reused from t0014), the v10 fixture (reused from t0013), and
`results/audio_samples/ft/v11_corrected.wav` if the REQ-6 cheap-fix path produced one.

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.score_speaker_sim
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
    ELEVENLABS_DAVID_DIR,
    RESULTS_AUDIO_DIR,
    RESULTS_DIR,
    SPEAKER_SIM_SCORES_JSON,
    T0013_V10_AUDIO_DIR,
    V11_BEST_WAV,
)

MIN_CORPUS_SIZE = 1000
CENTROID_SEED = 42
CENTROID_SAMPLE_SIZE = 679  # matches t0008's half-A split size, for methodology consistency

V10_EPOCH16_PRIMARY_WAV = T0013_V10_AUDIO_DIR / "v10_epoch16_primary.wav"
V11_CORRECTED_WAV = RESULTS_AUDIO_DIR / "ft" / "v11_corrected.wav"


@dataclass(frozen=True, slots=True)
class SpeakerSimResult:
    label: str
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


def score_wav(
    label: str, wav_path: Path, centroid: np.ndarray, encoder: object
) -> SpeakerSimResult:
    from resemblyzer import preprocess_wav  # type: ignore[import-untyped]

    if not wav_path.exists():
        return SpeakerSimResult(
            label=label, wav=str(wav_path), speaker_sim=None, error="file not found"
        )
    try:
        wav = preprocess_wav(str(wav_path))
        emb = encoder.embed_utterance(wav)  # type: ignore[attr-defined]
        sim = float(np.dot(emb, centroid))
        return SpeakerSimResult(label=label, wav=str(wav_path), speaker_sim=sim, error=None)
    except Exception as exc:  # noqa: BLE001 -- record failure per-clip, don't abort the whole run
        return SpeakerSimResult(label=label, wav=str(wav_path), speaker_sim=None, error=str(exc))


def main() -> None:
    import torch
    from resemblyzer import VoiceEncoder  # type: ignore[import-untyped]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder(device=device)
    print(f"VoiceEncoder on {device}")

    centroid = build_centroid(ELEVENLABS_DAVID_DIR, encoder)
    print(f"Centroid built from up to {CENTROID_SAMPLE_SIZE} clips of {ELEVENLABS_DAVID_DIR}")

    targets: list[tuple[str, Path]] = [
        ("v11-as-shipped", V11_BEST_WAV),
        ("v10-primary", V10_EPOCH16_PRIMARY_WAV),
    ]
    if V11_CORRECTED_WAV.exists():
        targets.append(("v11-corrected", V11_CORRECTED_WAV))

    results = [score_wav(label, t, centroid, encoder) for label, t in targets]
    for r in results:
        print(f"{r.label} ({r.wav}): speaker_sim={r.speaker_sim} error={r.error}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SPEAKER_SIM_SCORES_JSON.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    print(f"Wrote {SPEAKER_SIM_SCORES_JSON}")


if __name__ == "__main__":
    main()
