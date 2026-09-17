"""Per-system synthesis adapters for the three zero-shot voice-cloning systems under test.

Follows the exact `SynthResult` contract from
`tasks.t0008_tts_eval_harness_baselines.code.adapters` (REQ-1). Each adapter's model/loader is
called from that system's own isolated venv (`.venv-f5tts`, `.venv-cosyvoice2`,
`.venv-chatterbox`) — see `plan/plan.md`'s "Isolated venvs" approach note. The CALLER loads the
model once and passes it in; adapters never load models themselves (matches t0008's convention).

F5-TTS and Chatterbox are whole-utterance (non-streaming): `ttfb_s` == wall time to the complete
synthesized waveform, explicitly labelled `is_streaming=False` in the per-clip record (task's own
instruction: "whole-utterance latency for non-streaming systems, labelled as such").
CosyVoice2 streams (`inference_zero_shot(..., stream=True)`): `ttfb_s` is measured from the first
non-empty streamed chunk, matching `_kokoro_synth_via_pipeline`'s pattern, `is_streaming=True`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.adapters import SynthResult, _resample_to_16k

# ── F5-TTS ─────────────────────────────────────────────────────────────────────


def load_f5tts_model(hf_cache_dir: str | None = None) -> object:
    """Load the F5-TTS base English checkpoint (F5TTS_v1_Base). Whole-utterance, non-streaming."""
    from f5_tts.api import F5TTS

    return F5TTS(model="F5TTS_v1_Base", hf_cache_dir=hf_cache_dir)


def f5tts_transcribe_ref(model: object, ref_wav_path: Path) -> str:
    """Auto-transcribe the reference clip once per condition (F5TTS.transcribe uses Whisper)."""
    return str(model.transcribe(str(ref_wav_path)))  # type: ignore[attr-defined]


def f5_tts_synth(
    text: str,
    *,
    ref_wav_path: Path,
    ref_text: str,
    model: object,
) -> SynthResult:
    """Whole-utterance synthesis via F5-TTS's `F5TTS.infer`."""
    t_start = time.perf_counter()
    wav, sr, _spec = model.infer(  # type: ignore[attr-defined]
        ref_file=str(ref_wav_path),
        ref_text=ref_text,
        gen_text=text,
        show_info=lambda *a, **kw: None,
        progress=None,
    )
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    audio = np.asarray(wav, dtype=np.float32)
    audio_duration_s = len(audio) / sr if sr > 0 else 0.0
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(audio, sr)

    return SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=audio,
        native_sample_rate=sr,
        ttfb_s=wall_time,  # whole-utterance, non-streaming
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )


# ── CosyVoice 2 ────────────────────────────────────────────────────────────────


def load_cosyvoice2_model(model_dir: str) -> object:
    """Load CosyVoice2-0.5B. Genuine chunked streaming via `inference_zero_shot(stream=True)`."""
    from cosyvoice.cli.cosyvoice import CosyVoice2

    return CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)


def cosyvoice2_synth(
    text: str,
    *,
    ref_wav_path: Path,
    prompt_text: str,
    model: object,
) -> SynthResult:
    """Streaming synthesis via CosyVoice2's `inference_zero_shot`, TTFB from first chunk.

    **Bug found during implementation:** CosyVoice2's own `frontend_zero_shot` internally calls
    `load_wav(prompt_wav, N)` itself, THREE times, at three different target rates (24000 for
    `_extract_speech_feat`, 16000 for `_extract_speech_token` and `_extract_spk_embedding`) -- it
    expects `prompt_wav` to be a file PATH, not a pre-loaded tensor. Passing an already-loaded
    16kHz tensor (as CosyVoice's own top-level example scripts appear to suggest at a glance) makes
    the internal 24kHz `load_wav` call crash with
    `TypeError: Invalid file: tensor(...)` (soundfile receiving a tensor where it expects a path).
    """
    chunks: list[np.ndarray] = []
    ttfb_s: float | None = None
    t_start = time.perf_counter()

    for out in model.inference_zero_shot(  # type: ignore[attr-defined]
        text, prompt_text, str(ref_wav_path), stream=True
    ):
        chunk = out["tts_speech"]
        try:
            import torch

            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().cpu().numpy()
        except ImportError:
            pass
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if len(chunk) > 0:
            if ttfb_s is None:
                ttfb_s = time.perf_counter() - t_start
            chunks.append(chunk)

    t_end = time.perf_counter()
    wall_time = t_end - t_start
    assert ttfb_s is not None and len(chunks) > 0, f"CosyVoice2 returned no audio for: {text!r}"

    full_audio = np.concatenate(chunks, axis=0)
    sample_rate = int(getattr(model, "sample_rate", 24000))
    audio_duration_s = len(full_audio) / sample_rate
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(full_audio, sample_rate)

    return SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=full_audio,
        native_sample_rate=sample_rate,
        ttfb_s=ttfb_s,
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )


# ── Chatterbox ─────────────────────────────────────────────────────────────────


def load_chatterbox_model(device: str = "cuda") -> object:
    """Load the official (non-streaming) `resemble-ai/chatterbox` package."""
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device=device)


def chatterbox_synth(
    text: str,
    *,
    ref_wav_path: Path,
    model: object,
) -> SynthResult:
    """Whole-utterance synthesis via Chatterbox's single `generate()` call."""
    t_start = time.perf_counter()
    wav = model.generate(text, audio_prompt_path=str(ref_wav_path))  # type: ignore[attr-defined]
    t_end = time.perf_counter()
    wall_time = t_end - t_start

    try:
        import torch

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
    except ImportError:
        pass
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    sample_rate = int(getattr(model, "sr", 24000))
    audio_duration_s = len(audio) / sample_rate if sample_rate > 0 else 0.0
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(audio, sample_rate)

    return SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=audio,
        native_sample_rate=sample_rate,
        ttfb_s=wall_time,  # whole-utterance, non-streaming
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )
