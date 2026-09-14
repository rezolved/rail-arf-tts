"""Per-system synthesis adapters for the TTS evaluation harness.

Each adapter has the signature:
    def synth(text: str, **kwargs) -> SynthResult

where SynthResult contains:
    audio_array_16khz: np.ndarray  (float32, 16 kHz for resemblyzer)
    audio_array_native: np.ndarray (float32, native sample rate for WAV save)
    native_sample_rate: int
    ttfb_s: float                  (time-to-first-byte in seconds)
    rtf: float                     (wall_time / audio_duration)

All Kokoro adapters call build_pipeline from t0003 with lang_code="b"
(except floor control which uses lang_code="a").

IMPORTANT: The CALLER must load models before passing them as kwargs. Adapters do NOT
load models themselves to avoid re-loading on every call.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SynthResult:
    audio_array_16khz: np.ndarray  # float32, 16 kHz, for resemblyzer
    audio_array_native: np.ndarray  # float32, native sample rate, for WAV saving
    native_sample_rate: int
    ttfb_s: float
    rtf: float
    audio_duration_s: float


# ── Audio helpers ─────────────────────────────────────────────────────────────


def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample float32 audio to 16 kHz."""
    if src_rate == 16_000:
        return audio
    try:
        import soxr  # type: ignore[import-untyped]

        return soxr.resample(audio, src_rate, 16_000).astype(np.float32)
    except ImportError:
        pass
    # Fallback: use scipy
    from math import gcd

    from scipy.signal import resample_poly  # type: ignore[import-untyped]

    g = gcd(src_rate, 16_000)
    return resample_poly(audio, 16_000 // g, src_rate // g).astype(np.float32)


def save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Save float32 audio as WAV."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="FLOAT")


# ── ElevenLabs adapter ────────────────────────────────────────────────────────


def elevenlabs_david(
    text: str,
    *,
    api_key: str,
    voice_id: str,
    session: object,  # requests.Session
) -> SynthResult:
    """Synthesize via ElevenLabs streaming API, measuring TTFB."""

    import numpy as np
    import requests  # type: ignore[import-untyped]

    from tasks.t0008_tts_eval_harness_baselines.code.constants import (
        ELEVENLABS_API_BASE,
        ELEVENLABS_MODEL_ID,
    )

    url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    t_start = time.perf_counter()

    assert isinstance(session, requests.Session)
    response = session.post(url, headers=headers, json=payload, stream=True, timeout=30)
    response.raise_for_status()

    chunks: list[bytes] = []
    ttfb_s: float | None = None
    for chunk in response.iter_content(chunk_size=1024):
        if len(chunk) > 0:
            if ttfb_s is None:
                ttfb_s = time.perf_counter() - t_start
            chunks.append(chunk)

    t_end = time.perf_counter()
    wall_time = t_end - t_start
    assert ttfb_s is not None, "ElevenLabs returned empty response"

    # Decode MP3 → float32 at native rate
    try:
        import pydub  # type: ignore[import-untyped]

        mp3_bytes = b"".join(chunks)
        audio_seg = pydub.AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
        # Convert to mono float32
        samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
        if audio_seg.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        samples = samples / 32768.0
        native_rate = audio_seg.frame_rate
    except ImportError:
        # Fallback: use pydub is not available, try audioread via soundfile
        mp3_bytes = b"".join(chunks)
        import soundfile as sf

        data, native_rate = sf.read(io.BytesIO(mp3_bytes), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        samples = data

    audio_duration_s = len(samples) / native_rate
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(samples, native_rate)

    return SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=samples,
        native_sample_rate=native_rate,
        ttfb_s=ttfb_s,
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )


# ── Kokoro adapters ───────────────────────────────────────────────────────────


def _kokoro_synth_via_pipeline(
    text: str,
    *,
    pipeline: object,  # KPipeline
    voice: str | Path,
) -> SynthResult:
    """Common Kokoro synthesis timing via an already-loaded KPipeline."""
    from tasks.t0008_tts_eval_harness_baselines.code.constants import KOKORO_SAMPLE_RATE

    chunks: list[np.ndarray] = []
    ttfb_s: float | None = None
    t_start = time.perf_counter()

    for _, _, audio in pipeline(text, voice=voice):  # type: ignore[call-arg]
        if audio is not None and len(audio) > 0:
            if ttfb_s is None:
                ttfb_s = time.perf_counter() - t_start
            # Pipeline may return torch.Tensor or np.ndarray depending on kokoro version
            try:
                import torch  # type: ignore[import-untyped]

                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
            except ImportError:
                pass
            if isinstance(audio, np.ndarray):
                chunks.append(audio.astype(np.float32))

    t_end = time.perf_counter()
    wall_time = t_end - t_start

    assert ttfb_s is not None and len(chunks) > 0, (
        f"Pipeline returned no audio chunks for text: {text!r}"
    )

    full_audio: np.ndarray = np.concatenate(chunks, axis=0)
    audio_duration_s = len(full_audio) / KOKORO_SAMPLE_RATE
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(full_audio, KOKORO_SAMPLE_RATE)

    return SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=full_audio,
        native_sample_rate=KOKORO_SAMPLE_RATE,
        ttfb_s=ttfb_s,
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )


def kokoro_george(
    text: str,
    *,
    pipeline: object,
) -> SynthResult:
    """Kokoro base with bm_george voice."""
    from tasks.t0008_tts_eval_harness_baselines.code.constants import KOKORO_VOICE_GEORGE

    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=KOKORO_VOICE_GEORGE)


def kokoro_lewis(
    text: str,
    *,
    pipeline: object,
) -> SynthResult:
    """Kokoro base with bm_lewis voice."""
    from tasks.t0008_tts_eval_harness_baselines.code.constants import KOKORO_VOICE_LEWIS

    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=KOKORO_VOICE_LEWIS)


def kokoro_base_v3_voicepack(
    text: str,
    *,
    pipeline: object,
    voicepack_path: Path,
) -> SynthResult:
    """Kokoro with stock decoder + v3 David voicepack."""
    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=str(voicepack_path))


def kokoro_v3_bundle(
    text: str,
    *,
    pipeline: object,
    voicepack_path: Path,
) -> SynthResult:
    """Kokoro with fine-tuned v3 five-module decoder + v3 David voicepack."""
    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=str(voicepack_path))


def kokoro_t0006_v6d(
    text: str,
    *,
    pipeline: object,
    voicepack_path: Path,
) -> SynthResult:
    """Kokoro with packaged t0006 v6d epoch-6 checkpoint + v3 David voicepack."""
    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=str(voicepack_path))


def kokoro_t0005_best(
    text: str,
    *,
    pipeline: object,
    voicepack_path: Path,
) -> SynthResult:
    """Kokoro with packaged t0005 run06 epoch-3 checkpoint + v3 David voicepack."""
    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=str(voicepack_path))


def kokoro_floor_control(
    text: str,
    *,
    pipeline: object,
) -> SynthResult:
    """Kokoro base af_heart voice (female American) — floor control."""
    from tasks.t0008_tts_eval_harness_baselines.code.constants import KOKORO_VOICE_FLOOR_CONTROL

    return _kokoro_synth_via_pipeline(text, pipeline=pipeline, voice=KOKORO_VOICE_FLOOR_CONTROL)


# ── Model loader ──────────────────────────────────────────────────────────────


def load_kokoro_model_with_checkpoint(
    ckpt_path: Path | None,
) -> object:
    """Load a KModel and apply a five-module checkpoint if provided.

    Args:
        ckpt_path: Path to a packaged five-module .pth dict. None → stock weights.

    Returns:
        KModel instance.
    """
    import torch
    from kokoro import KModel

    from tasks.t0008_tts_eval_harness_baselines.code.constants import CHECKPOINT_MODULES

    # Disable complex numbers for compatibility (as in t0002)
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

    if ckpt_path is not None:
        sd: object = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        assert isinstance(sd, dict), f"Expected dict checkpoint, got {type(sd)}"
        assert sorted(sd.keys()) == sorted(CHECKPOINT_MODULES), (
            f"Checkpoint keys mismatch: {sorted(sd.keys())} != {sorted(CHECKPOINT_MODULES)}"
        )
        # Load each module into the corresponding model sub-module
        merged: dict[str, object] = {}
        for module_name, module_sd in sd.items():
            for param_name, param_val in module_sd.items():
                full_key = f"{module_name}.{param_name}"
                merged[full_key] = param_val
        missing, unexpected = model.load_state_dict(merged, strict=False)
        logger.info(
            "Loaded %s: %d missing, %d unexpected keys",
            ckpt_path.name,
            len(missing),
            len(unexpected),
        )

    return model
