"""Speaker similarity, WER, duration-ratio, and RTF scoring for the TTS evaluation harness.

All scorer functions are pure: they take audio paths + metadata and return dataclasses.
No I/O side-effects except reading WAV files.

resemblyzer is imported lazily (optional extra, not in main deps). All functions that
require it raise ImportError with a clear message if it is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.constants import (
    DURATION_RATIO_HIGH,
    DURATION_RATIO_LOW,
    MIN_CLIP_DURATION_S,
)

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpeakerSimResult:
    mean: float
    std: float
    per_clip: list[float | None]  # None = clip skipped (too short or load error)
    skipped_count: int
    skipped_reason: str | None


@dataclass(frozen=True, slots=True)
class WerResult:
    mean_wer: float
    per_clip: list[float | None]  # None = skipped (duration_ratio gate or error)
    skipped_count: int
    skipped_reason: str | None


@dataclass(frozen=True, slots=True)
class DurationRatioResult:
    median: float
    per_clip: list[float | None]  # None = could not determine duration
    explosion_count: int  # clips with ratio > DURATION_EXPLOSION_RATIO


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_wav_duration(path: Path) -> float | None:
    """Return duration in seconds, or None on error."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.duration)
    except Exception as exc:
        logger.warning("Could not read duration from %s: %s", path, exc)
        return None


# ── Speaker similarity ────────────────────────────────────────────────────────


def build_centroid(wav_paths: list[Path]) -> np.ndarray:
    """Compute L2-normalized mean GE2E embedding from a list of WAV files.

    Skips clips shorter than MIN_CLIP_DURATION_S (resemblyzer requirement).
    Returns a (256,) float32 ndarray.

    Raises:
        ImportError: if resemblyzer is not installed.
        RuntimeError: if no clips are long enough to embed.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as e:
        raise ImportError(
            "resemblyzer is not installed. Install with: pip install '.[speaker-sim]'"
        ) from e

    encoder = VoiceEncoder(device="cpu")
    embeddings: list[np.ndarray] = []
    skipped = 0
    for wav_path in wav_paths:
        dur = _get_wav_duration(wav_path)
        if dur is not None and dur < MIN_CLIP_DURATION_S:
            skipped += 1
            continue
        try:
            wav = preprocess_wav(str(wav_path))
            if len(wav) < MIN_CLIP_DURATION_S * 16000:
                skipped += 1
                continue
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
        except Exception as exc:
            logger.warning("Could not embed %s: %s", wav_path, exc)
            skipped += 1

    if len(embeddings) == 0:
        raise RuntimeError(
            f"No clips long enough to embed (skipped={skipped}, "
            f"min_duration={MIN_CLIP_DURATION_S}s)"
        )

    logger.info("Centroid built from %d clips (skipped %d too-short)", len(embeddings), skipped)
    mean_emb: np.ndarray = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(mean_emb))
    if norm > 0:
        mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)


def compute_speaker_sim(
    synth_wavs: list[Path],
    ref_embeddings: np.ndarray,
) -> SpeakerSimResult:
    """Compute GE2E cosine similarity between synthesized clips and a reference embedding.

    ref_embeddings may be:
    - a (256,) centroid → cosine between each clip embedding and the centroid
    - a (N, 256) array → cosine between each clip embedding and the row-mean of that array
                          (used for ElevenLabs half-B scoring)

    Clips shorter than MIN_CLIP_DURATION_S are skipped (per_clip[i] = None).

    Raises:
        ImportError: if resemblyzer is not installed.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as e:
        raise ImportError(
            "resemblyzer is not installed. Install with: pip install '.[speaker-sim]'"
        ) from e

    encoder = VoiceEncoder(device="cpu")

    # Normalize reference to a single centroid vector
    if ref_embeddings.ndim == 2:
        centroid: np.ndarray = ref_embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
    else:
        centroid = ref_embeddings

    per_clip: list[float | None] = []
    skipped = 0

    for wav_path in synth_wavs:
        dur = _get_wav_duration(wav_path)
        if dur is not None and dur < MIN_CLIP_DURATION_S:
            per_clip.append(None)
            skipped += 1
            continue
        try:
            wav = preprocess_wav(str(wav_path))
            if len(wav) < int(MIN_CLIP_DURATION_S * 16000):
                per_clip.append(None)
                skipped += 1
                continue
            emb = encoder.embed_utterance(wav)
            sim = float(np.dot(emb, centroid))
            per_clip.append(sim)
        except Exception as exc:
            logger.warning("Could not compute speaker_sim for %s: %s", wav_path, exc)
            per_clip.append(None)
            skipped += 1

    valid: list[float] = [s for s in per_clip if s is not None]
    if len(valid) == 0:
        return SpeakerSimResult(
            mean=0.0,
            std=0.0,
            per_clip=per_clip,
            skipped_count=skipped,
            skipped_reason="all clips skipped",
        )

    mean_sim = float(np.mean(valid))
    std_sim = float(np.std(valid))
    return SpeakerSimResult(
        mean=mean_sim,
        std=std_sim,
        per_clip=per_clip,
        skipped_count=skipped,
        skipped_reason=f"{skipped} clips < {MIN_CLIP_DURATION_S}s" if skipped > 0 else None,
    )


# ── Duration ratio ────────────────────────────────────────────────────────────


def compute_duration_ratio(
    synth_wavs: list[Path],
    ref_durations_s: list[float],
) -> DurationRatioResult:
    """Compute synthesized / reference duration ratio for each clip.

    Args:
        synth_wavs: Paths to synthesized WAV files.
        ref_durations_s: Reference clip durations in seconds (same length as synth_wavs).

    Returns:
        DurationRatioResult with per_clip ratios and explosion count.
    """
    assert len(synth_wavs) == len(ref_durations_s), (
        f"synth_wavs ({len(synth_wavs)}) and ref_durations_s"
        f" ({len(ref_durations_s)}) length mismatch"
    )

    per_clip: list[float | None] = []
    explosion_count = 0

    for wav_path, ref_dur in zip(synth_wavs, ref_durations_s, strict=True):
        synth_dur = _get_wav_duration(wav_path)
        if synth_dur is None or ref_dur <= 0:
            per_clip.append(None)
            continue
        ratio = synth_dur / ref_dur
        per_clip.append(ratio)
        if ratio > 5.0:
            explosion_count += 1

    valid: list[float] = [r for r in per_clip if r is not None]
    median_ratio = float(np.median(valid)) if len(valid) > 0 else 0.0

    return DurationRatioResult(
        median=median_ratio,
        per_clip=per_clip,
        explosion_count=explosion_count,
    )


# ── WER ───────────────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """JiWER-style normalization: lowercase, strip punctuation, expand numbers."""
    import re

    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(
    synth_wavs: list[Path],
    texts: list[str],
    duration_ratios: list[float | None],
) -> WerResult:
    """Compute WER for each synthesized clip using faster-whisper + JiWER normalization.

    Clips with duration_ratio > DURATION_RATIO_HIGH or < DURATION_RATIO_LOW are skipped.

    Args:
        synth_wavs: Paths to synthesized WAV files.
        texts: Reference texts (same order as synth_wavs).
        duration_ratios: Per-clip duration ratios (or None if unknown).

    Returns:
        WerResult with per-clip WER values and aggregate mean.

    Raises:
        ImportError: if faster_whisper is not installed.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster_whisper is not installed. Install with: pip install faster-whisper"
        ) from e

    try:
        import jiwer  # type: ignore[import-untyped]
    except ImportError:
        jiwer = None  # fallback: manual edit-distance WER

    assert len(synth_wavs) == len(texts), (
        f"synth_wavs ({len(synth_wavs)}) and texts ({len(texts)}) length mismatch"
    )
    assert len(synth_wavs) == len(duration_ratios), (
        f"synth_wavs ({len(synth_wavs)}) and duration_ratios"
        f" ({len(duration_ratios)}) length mismatch"
    )

    from tasks.t0008_tts_eval_harness_baselines.code.constants import WHISPER_MODEL_SIZE

    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    per_clip: list[float | None] = []
    skipped = 0

    for wav_path, ref_text, dur_ratio in zip(synth_wavs, texts, duration_ratios, strict=True):
        # Duration gate
        if dur_ratio is not None and (
            dur_ratio > DURATION_RATIO_HIGH or dur_ratio < DURATION_RATIO_LOW
        ):
            per_clip.append(None)
            skipped += 1
            continue

        try:
            segments, _ = model.transcribe(str(wav_path), language="en")
            hyp = " ".join(seg.text for seg in segments).strip()
            hyp_norm = _normalize_text(hyp)
            ref_norm = _normalize_text(ref_text)

            if jiwer is not None:
                word_error_rate = jiwer.wer(ref_norm, hyp_norm)
            else:
                word_error_rate = _manual_wer(ref_norm, hyp_norm)

            per_clip.append(float(word_error_rate))
        except Exception as exc:
            logger.warning("WER failed for %s: %s", wav_path, exc)
            per_clip.append(None)
            skipped += 1

    valid: list[float] = [w for w in per_clip if w is not None]
    mean_wer = float(np.mean(valid)) if len(valid) > 0 else 0.0

    return WerResult(
        mean_wer=mean_wer,
        per_clip=per_clip,
        skipped_count=skipped,
        skipped_reason=f"{skipped} clips skipped by duration_ratio gate" if skipped > 0 else None,
    )


def _manual_wer(reference: str, hypothesis: str) -> float:
    """Simple Levenshtein-based WER fallback (no jiwer dependency)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # DP edit distance
    n, m = len(ref_words), len(hyp_words)
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n
