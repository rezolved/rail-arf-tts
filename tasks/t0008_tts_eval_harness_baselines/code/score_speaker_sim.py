"""Score speaker_sim and WER for per-clip records from run_eval.py.

This script is designed to run on the VM with the resemblyzer venv available.
It loads existing per_clip_metrics JSON files (which have ttfb/rtf already),
computes speaker_sim and WER for each record, and writes updated JSON.

Usage:
    /mnt/tmp/t0008-resemblyzer-venv/bin/python \\
        score_speaker_sim.py \\
        --per-clip-in results/per_clip_metrics_kokoro.json \\
        --corpus-dir data/11labs_david/ \\
        --per-clip-out results/per_clip_metrics_kokoro_scored.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="Score speaker_sim and WER for per-clip records")
    p.add_argument("--per-clip-in", type=Path, required=True, help="Input per_clip_metrics.json")
    p.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Path to 11labs_david corpus dir for centroid",
    )
    p.add_argument(
        "--per-clip-out",
        type=Path,
        required=True,
        help="Output per_clip_metrics.json (with speaker_sim filled in)",
    )
    p.add_argument(
        "--centroid-path",
        type=Path,
        default=None,
        help="If provided, load centroid from this .npy file instead of recomputing",
    )
    p.add_argument(
        "--half-b-paths",
        type=Path,
        default=None,
        help="JSON list of half-B WAV paths (for ElevenLabs scoring)",
    )
    p.add_argument(
        "--skip-wer",
        action="store_true",
        help="Skip WER computation (faster, saves time)",
    )
    args = p.parse_args()

    # Load per-clip records
    records: list[dict[str, object]] = json.loads(args.per_clip_in.read_text(encoding="utf-8"))
    logger.info("Loaded %d records from %s", len(records), args.per_clip_in)

    # Build or load centroid
    import torch
    from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore[import-untyped]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = VoiceEncoder(device=device)
    logger.info("VoiceEncoder on %s", device)

    if args.centroid_path is not None and args.centroid_path.exists():
        centroid = np.load(str(args.centroid_path))
        logger.info("Loaded centroid from %s", args.centroid_path)
    else:
        logger.info("Building centroid from %s ...", args.corpus_dir)
        wav_files: list[Path] = sorted(args.corpus_dir.glob("*.wav"))
        assert len(wav_files) >= 1000, (
            f"Corpus has only {len(wav_files)} WAVs — expected >= 1000. Halt."
        )

        # Split 679/679 with seed=42
        import random

        rng = random.Random(42)
        shuffled = wav_files.copy()
        rng.shuffle(shuffled)
        half_a = shuffled[:679]
        half_b = shuffled[679:]

        embeddings: list[np.ndarray] = []
        for wav_path in half_a:
            try:
                wav = preprocess_wav(str(wav_path))
                # resemblyzer pads short clips automatically — no minimum length needed
                emb = encoder.embed_utterance(wav)
                embeddings.append(emb)
            except Exception as exc:
                logger.warning("Could not embed %s: %s", wav_path, exc)

        assert len(embeddings) >= 100, f"Only {len(embeddings)} valid centroid clips"
        mean_emb = np.mean(np.stack(embeddings), axis=0)
        norm = float(np.linalg.norm(mean_emb))
        centroid = (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)
        logger.info("Centroid built from %d clips", len(embeddings))

        # Save centroid
        if args.centroid_path is not None:
            args.centroid_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(args.centroid_path), centroid)
            logger.info("Saved centroid → %s", args.centroid_path)

        # Save half-B paths
        if args.half_b_paths is not None:
            args.half_b_paths.parent.mkdir(parents=True, exist_ok=True)
            args.half_b_paths.write_text(
                json.dumps([str(p) for p in half_b], indent=2), encoding="utf-8"
            )
            logger.info("Saved half-B paths → %s", args.half_b_paths)

    # Load half-B paths for ElevenLabs scoring
    half_b_embeddings: np.ndarray | None = None
    if args.half_b_paths is not None and args.half_b_paths.exists():
        half_b_paths = [Path(p) for p in json.loads(args.half_b_paths.read_text())]
        logger.info("Computing half-B embeddings for ElevenLabs (%d clips)...", len(half_b_paths))
        hb_embs: list[np.ndarray] = []
        for wav_path in half_b_paths:
            try:
                wav = preprocess_wav(str(wav_path))
                # resemblyzer pads short clips automatically
                emb = encoder.embed_utterance(wav)
                hb_embs.append(emb)
            except Exception:
                pass
        if len(hb_embs) > 0:
            half_b_embeddings = np.stack(hb_embs)

    # Score each record
    updated_records: list[dict[str, object]] = []

    for record in records:
        audio_path_str = record.get("audio_path")
        system = str(record.get("system", ""))
        updated = dict(record)

        if audio_path_str is None:
            updated_records.append(updated)
            continue

        audio_path = Path(str(audio_path_str))
        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            updated_records.append(updated)
            continue

        # Speaker sim
        try:
            wav = preprocess_wav(str(audio_path))
            # resemblyzer pads short clips automatically — no minimum length guard needed
            emb = encoder.embed_utterance(wav)
            # ElevenLabs scored vs half-B centroid; all others vs half-A centroid
            if system == "elevenlabs_david" and half_b_embeddings is not None:
                hb_centroid = half_b_embeddings.mean(axis=0)
                norm = float(np.linalg.norm(hb_centroid))
                if norm > 0:
                    hb_centroid = hb_centroid / norm
                sim = float(np.dot(emb, hb_centroid))
            else:
                sim = float(np.dot(emb, centroid))
            updated["speaker_sim"] = sim
        except Exception as exc:
            logger.warning("speaker_sim failed for %s: %s", audio_path, exc)

        # Duration ratio
        synth_dur = record.get("synth_duration_s")
        ref_dur = record.get("ref_duration_s")
        if synth_dur is not None and ref_dur is not None and float(ref_dur) > 0:
            updated["duration_ratio"] = float(synth_dur) / float(ref_dur)

        updated_records.append(updated)

    logger.info("Scored %d records", len(updated_records))

    # WER scoring (optional)
    if not args.skip_wer:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            model = WhisperModel("base.en", device="cpu", compute_type="int8")
            import re

            def norm(t: str) -> str:
                return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()

            def wer(ref: str, hyp: str) -> float:
                ref_w = ref.split()
                hyp_w = hyp.split()
                if len(ref_w) == 0:
                    return 0.0 if len(hyp_w) == 0 else 1.0
                n, m = len(ref_w), len(hyp_w)
                dp = [[0] * (m + 1) for _ in range(n + 1)]
                for i in range(n + 1):
                    dp[i][0] = i
                for j in range(m + 1):
                    dp[0][j] = j
                for i in range(1, n + 1):
                    for j in range(1, m + 1):
                        if ref_w[i - 1] == hyp_w[j - 1]:
                            dp[i][j] = dp[i - 1][j - 1]
                        else:
                            dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                return dp[n][m] / n

            for record in updated_records:
                # Duration gate
                dur_ratio = record.get("duration_ratio")
                if dur_ratio is not None and (float(dur_ratio) > 2.0 or float(dur_ratio) < 0.5):
                    continue
                audio_path_str = record.get("audio_path")
                text = str(record.get("text", ""))
                if audio_path_str is None:
                    continue
                audio_path = Path(str(audio_path_str))
                if not audio_path.exists():
                    continue
                try:
                    segments, _ = model.transcribe(str(audio_path), language="en")
                    hyp = " ".join(seg.text for seg in segments).strip()
                    record["wer"] = wer(norm(text), norm(hyp))
                except Exception as exc:
                    logger.warning("WER failed for %s: %s", audio_path, exc)

            logger.info("WER scoring complete")
        except ImportError:
            logger.warning("faster_whisper not installed — skipping WER")

    # Write output
    args.per_clip_out.parent.mkdir(parents=True, exist_ok=True)
    args.per_clip_out.write_text(json.dumps(updated_records, indent=2), encoding="utf-8")
    logger.info("Saved scored records → %s", args.per_clip_out)


if __name__ == "__main__":
    main()
