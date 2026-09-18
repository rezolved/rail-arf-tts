"""Per-system synthesis adapters, extended with per-stage `StageTiming` instrumentation (REQ-1).

Copied from `tasks.t0018_zero_shot_cloning_calibration.code.adapters_zeroshot` (t0018 is not a
registered library, so this is a copy, not an import) and extended per `plan/plan.md` Step 7/8.
None of t0018's three adapters recorded intermediate timestamps; this is genuinely new
instrumentation.

**CosyVoice2 stage boundaries (found by reading the installed package source directly, per
plan.md Step 7 — `cosyvoice/cli/cosyvoice.py::CosyVoice2.inference_zero_shot` and
`cosyvoice/cli/frontend.py::CosyVoiceFrontEnd`, on the VM under
`/mnt/cache/persist/t0021_zero_shot_latency_reduction/CosyVoice/`)**: `inference_zero_shot()` is
just a thin wrapper around `frontend.text_normalize()` -> `frontend.frontend_zero_shot()` ->
`model.tts(**model_input, stream=...)`. This module re-implements that wrapper's logic directly
(no source-file edits to the vendored `cosyvoice` package) so each stage can be timed:

* `text_frontend_ms` — `frontend.text_normalize()` on both the prompt text and the tts text.
* `ref_encoding_ms` — `frontend.frontend_zero_shot()`: speaker embedding + prompt speech token/feat
  extraction from the reference WAV (the "per-voice, cacheable" work in Key Question 2). When
  `use_ref_cache=True`, this becomes a cheap `spk2info` dict lookup instead of a fresh
  extraction — see `cosyvoice2_register_ref_cache()`, which uses CosyVoice2's own native
  `add_zero_shot_spk()`/`zero_shot_spk_id` caching mechanism (no custom caching layer needed).
* `lm_prefill_decode_ms` + `flow_matching_vocoder_ms` — both happen inside the `model.tts(...)`
  generator. `CosyVoice2Model.tts()` spawns the LM decode as a background thread
  (`self.llm_job`) and the main thread polls (`time.sleep(0.1)`) until enough tokens exist for the
  first chunk, then calls the bound method `model.model.token2wav(...)` (flow-matching THEN
  HiFT/vocoder internally) to render it. **Merged-stage documentation (plan.md Step 7's explicit
  fallback):** `token2wav()` performs flow-matching and vocoding as one opaque call from the
  adapter's vantage point; separating those two would require patching two levels deep into the
  vendored `cosyvoice.flow`/`cosyvoice.hifigan` modules, which this task treats as the "invasive
  monkey-patching" the plan says to avoid. Instead, `model.model.token2wav` (a bound method on the
  already-instantiated model object) is wrapped with a timing closure at the ADAPTER level — this
  patches a live object attribute at runtime, not the vendored source file, and is undone
  (restored to the original) immediately after each call. `lm_prefill_decode_ms` is then derived
  as `(time to first yielded chunk) - (wrapped token2wav's own duration for that first call)`.

**Chatterbox stage boundaries** (found by reading
`chatterbox/tts.py::ChatterboxTTS.generate()`/`prepare_conditionals()` directly in the installed
package under `.venv-chatterbox/lib/python3.10/site-packages/chatterbox/tts.py`): unlike
CosyVoice2, all four stages ARE cleanly separable without any patching — `generate()`'s own body
already calls four independent, directly-callable pieces in sequence: `prepare_conditionals()`
(ref conditioning — self.ve/self.s3gen embedding extraction, exactly Key Question 2's cacheable
work), `punc_norm()` + `self.tokenizer.text_to_tokens()` (text frontend), `self.t3.inference()`
(T3 LM decode), `self.s3gen.inference()` (S3Gen vocoder). This module re-implements `generate()`'s
body directly (again, no source-file edits) with timing around each of the four pieces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.adapters import SynthResult, _resample_to_16k

COSYVOICE2_CACHED_SPK_ID = "t0021_cached_ref"


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Per-stage first-chunk latency breakdown (REQ-1). Fields not applicable to a given
    system's architecture are left `None` — never populated with a zero or a guess."""

    ref_encoding_ms: float | None = None  # CosyVoice2: speaker embedding + prompt token extraction
    ref_conditioning_ms: float | None = None  # Chatterbox: prepare_conditionals()
    text_frontend_ms: float | None = None  # both systems
    lm_prefill_decode_ms: float | None = None  # CosyVoice2: derived (see module docstring)
    lm_decode_ms: float | None = None  # Chatterbox: t3.inference()
    flow_matching_vocoder_ms: float | None = None  # CosyVoice2: merged token2wav (documented)
    vocoder_ms: float | None = None  # Chatterbox: s3gen.inference()
    total_ms: float | None = None  # sanity-check: should ~= sum of populated stages above
    notes: str = ""


@dataclass(frozen=True, slots=True)
class TimedSynthResult:
    result: SynthResult
    stage_timing: StageTiming


# ── F5-TTS (unchanged from t0018; used only for the S-0018-01 closure, Milestone 4) ────────────


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
) -> TimedSynthResult:
    """Whole-utterance synthesis via F5-TTS's `F5TTS.infer`. No per-stage breakdown (out of this
    task's acceleration scope — F5-TTS is a cheap closure item, REQ-6, not a variant-sweep target).
    """
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

    result = SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=audio,
        native_sample_rate=sr,
        ttfb_s=wall_time,  # whole-utterance, non-streaming
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )
    timing = StageTiming(
        total_ms=wall_time * 1000.0, notes="F5-TTS: no per-stage breakdown (out of scope)"
    )
    return TimedSynthResult(result=result, stage_timing=timing)


# ── CosyVoice 2 ────────────────────────────────────────────────────────────────


def load_cosyvoice2_model(
    model_dir: str,
    *,
    load_jit: bool = False,
    load_trt: bool = False,
    fp16: bool = False,
    load_vllm: bool = False,
) -> object:
    """Load CosyVoice2-0.5B with the requested acceleration flags (Step 6/8)."""
    from cosyvoice.cli.cosyvoice import CosyVoice2

    return CosyVoice2(
        model_dir, load_jit=load_jit, load_trt=load_trt, load_vllm=load_vllm, fp16=fp16
    )


def cosyvoice2_register_ref_cache(
    model: object, *, prompt_text: str, ref_wav_path: Path, spk_id: str = COSYVOICE2_CACHED_SPK_ID
) -> str:
    """Register the reference clip once via CosyVoice2's own native `add_zero_shot_spk()` cache.

    This is the `ref_cache` lever (REQ-2): subsequent `cosyvoice2_synth(..., use_ref_cache=True)`
    calls skip `_extract_speech_token`/`_extract_spk_embedding`/`_extract_speech_feat` entirely,
    reusing the dict stored in `model.frontend.spk2info[spk_id]` (a plain in-memory dict lookup).
    No custom caching layer is added — this is CosyVoice2's own documented mechanism
    (`list_available_spks()`/`add_zero_shot_spk()` in `cosyvoice/cli/cosyvoice.py`).
    """
    model.add_zero_shot_spk(prompt_text, str(ref_wav_path), spk_id)  # type: ignore[attr-defined]
    return spk_id


def cosyvoice2_synth(
    text: str,
    *,
    ref_wav_path: Path,
    prompt_text: str,
    model: object,
    use_ref_cache: bool = False,
    cached_spk_id: str = COSYVOICE2_CACHED_SPK_ID,
) -> TimedSynthResult:
    """Streaming synthesis, re-implementing `CosyVoice2.inference_zero_shot()`'s own logic
    directly (see module docstring) so each stage can be timed independently.
    """
    frontend = model.frontend  # type: ignore[attr-defined]
    inner_model = model.model  # type: ignore[attr-defined]
    sample_rate = int(getattr(model, "sample_rate", 24000))

    t_call_start = time.perf_counter()

    # ── text_frontend_ms: text_normalize on prompt text (skipped when ref-cached, since the
    # cached spk2info entry already has no `text`/`text_len` keys tied to a stale prompt) and
    # on the tts text itself. ──
    t0 = time.perf_counter()
    if not use_ref_cache:
        prompt_text_norm = frontend.text_normalize(prompt_text, split=False)
    else:
        prompt_text_norm = ""
    texts = frontend.text_normalize(text, split=True)
    t1 = time.perf_counter()
    text_frontend_ms = (t1 - t0) * 1000.0

    chunks: list[np.ndarray] = []
    ttfb_s: float | None = None
    ref_encoding_ms_total = 0.0
    lm_prefill_decode_ms_total = 0.0
    flow_vocoder_ms_total = 0.0
    first_chunk_lm_ms: float | None = None
    first_chunk_flow_vocoder_ms: float | None = None

    for i in texts:
        # ── ref_encoding_ms: frontend_zero_shot() — full extraction, or a spk2info dict lookup
        # when ref-cached (CosyVoice2's own native cache mechanism). ──
        t_fe0 = time.perf_counter()
        if use_ref_cache:
            model_input = frontend.frontend_zero_shot(i, "", "", sample_rate, cached_spk_id)
        else:
            model_input = frontend.frontend_zero_shot(
                i, prompt_text_norm, str(ref_wav_path), sample_rate, ""
            )
        t_fe1 = time.perf_counter()
        ref_encoding_ms_total += (t_fe1 - t_fe0) * 1000.0

        # ── Wrap the bound method `token2wav` with a timing closure (adapter-level runtime
        # patch of a live object attribute, restored immediately after — NOT an edit to the
        # vendored cosyvoice source file). Measures flow-matching + vocoder combined per call
        # (merged-stage documentation, see module docstring). ──
        original_token2wav = inner_model.token2wav
        call_durations_s: list[float] = []

        def _wrapped_token2wav(
            *args: Any,
            _orig: Any = original_token2wav,
            _durations: list[float] = call_durations_s,
            **kwargs: Any,
        ) -> Any:
            ts = time.perf_counter()
            out = _orig(*args, **kwargs)
            te = time.perf_counter()
            _durations.append(te - ts)
            return out

        inner_model.token2wav = _wrapped_token2wav
        t_tts_start = time.perf_counter()
        try:
            for out in inner_model.tts(**model_input, stream=True):
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
                        ttfb_s = time.perf_counter() - t_call_start
                        # This call's own token2wav duration so far is the flow+vocoder share of
                        # the first chunk; the rest of the elapsed time is LM prefill/decode.
                        this_flow_vocoder_s = call_durations_s[-1] if call_durations_s else 0.0
                        first_chunk_flow_vocoder_ms = this_flow_vocoder_s * 1000.0
                        first_chunk_lm_ms = max(
                            0.0,
                            (time.perf_counter() - t_tts_start) * 1000.0
                            - first_chunk_flow_vocoder_ms,
                        )
                    chunks.append(chunk)
        finally:
            inner_model.token2wav = original_token2wav

        flow_vocoder_ms_total += sum(call_durations_s) * 1000.0

    lm_prefill_decode_ms_total = first_chunk_lm_ms if first_chunk_lm_ms is not None else 0.0

    t_end = time.perf_counter()
    wall_time = t_end - t_call_start
    assert ttfb_s is not None and len(chunks) > 0, f"CosyVoice2 returned no audio for: {text!r}"

    full_audio = np.concatenate(chunks, axis=0)
    audio_duration_s = len(full_audio) / sample_rate
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(full_audio, sample_rate)

    result = SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=full_audio,
        native_sample_rate=sample_rate,
        ttfb_s=ttfb_s,
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )
    timing = StageTiming(
        ref_encoding_ms=ref_encoding_ms_total,
        text_frontend_ms=text_frontend_ms,
        lm_prefill_decode_ms=lm_prefill_decode_ms_total,
        flow_matching_vocoder_ms=first_chunk_flow_vocoder_ms,
        total_ms=ttfb_s * 1000.0,
        notes=(
            "flow_matching_vocoder_ms merges flow-matching + HiFT vocoder (both happen inside "
            "one opaque token2wav() call; separating further would require patching two levels "
            "into vendored cosyvoice.flow/cosyvoice.hifigan, treated as invasive per plan.md "
            "Step 7). Stage sums are for the FIRST chunk only (TTFB decomposition); "
            "flow_vocoder_ms_total across all chunks in this call was "
            f"{flow_vocoder_ms_total:.2f}ms if multiple chunks were produced."
        ),
    )
    return TimedSynthResult(result=result, stage_timing=timing)


# ── Chatterbox ─────────────────────────────────────────────────────────────────


def load_chatterbox_model(device: str = "cuda") -> object:
    """Load the official (non-streaming) `resemble-ai/chatterbox` package."""
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device=device)


def chatterbox_register_ref_cache(model: object, *, ref_wav_path: Path) -> None:
    """Register the reference clip once via Chatterbox's own native conditioning cache.

    `ChatterboxTTS.generate()` already reuses `self.conds` whenever `audio_prompt_path=None` is
    passed (see `chatterbox/tts.py`) — this is the `ref_cache` lever (REQ-2): call
    `prepare_conditionals()` once here, then pass `ref_wav_path=None`-equivalent
    (`use_ref_cache=True`) on every subsequent `chatterbox_synth()` call to skip re-conditioning.
    """
    model.prepare_conditionals(str(ref_wav_path))  # type: ignore[attr-defined]


def chatterbox_synth(
    text: str,
    *,
    ref_wav_path: Path,
    model: object,
    use_ref_cache: bool = False,
) -> TimedSynthResult:
    """Whole-utterance synthesis, re-implementing `ChatterboxTTS.generate()`'s own logic
    directly (see module docstring) so each of its four stages can be timed independently.
    """
    import torch
    import torch.nn.functional as functional
    from chatterbox.tts import punc_norm

    t_call_start = time.perf_counter()

    # ── ref_conditioning_ms: prepare_conditionals() (skipped entirely when ref-cached). ──
    t0 = time.perf_counter()
    if not use_ref_cache:
        model.prepare_conditionals(str(ref_wav_path))  # type: ignore[attr-defined]
    else:
        assert model.conds is not None, (
            "use_ref_cache=True requires prepare_conditionals() to have run first"
        )  # type: ignore[attr-defined]
    t1 = time.perf_counter()
    ref_conditioning_ms = (t1 - t0) * 1000.0

    # ── text_frontend_ms: punc_norm + tokenize + CFG duplication + BOS/EOS padding. ──
    t2 = time.perf_counter()
    norm_text = punc_norm(text)
    text_tokens = model.tokenizer.text_to_tokens(norm_text).to(model.device)  # type: ignore[attr-defined]
    cfg_weight = 0.5
    if cfg_weight > 0.0:
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token  # type: ignore[attr-defined]
    eot = model.t3.hp.stop_text_token  # type: ignore[attr-defined]
    text_tokens = functional.pad(text_tokens, (1, 0), value=sot)
    text_tokens = functional.pad(text_tokens, (0, 1), value=eot)
    t3 = time.perf_counter()
    text_frontend_ms = (t3 - t2) * 1000.0

    # ── lm_decode_ms: T3 autoregressive decode. ──
    # NOTE: found via this task's own precision-variant smoke gate. `model.conds.t3` (a `T3Cond`
    # dataclass) is computed once at ref-conditioning time and stays whatever dtype it was built
    # in (fp32) even after the `precision_bf16_or_fp16` variant casts `model.t3`'s own weights to
    # bf16/fp16 — the resulting dtype mismatch ("mat1 and mat2 must have the same dtype") failed
    # ALL 196/196 measured calls the first time this variant ran. `T3Cond` already provides its
    # own `.to(dtype=...)` (int/long fields are left alone; see cond_enc.py), so re-cast it to
    # whatever dtype `model.t3`'s own parameters currently are, every call — a no-op when T3 is
    # fp32 (baseline and every other variant), and the actual fix when T3 has been cast down.
    t3_param_dtype = next(model.t3.parameters()).dtype  # type: ignore[attr-defined]
    model.conds.t3.to(dtype=t3_param_dtype)  # type: ignore[attr-defined]

    t4 = time.perf_counter()
    with torch.inference_mode():
        speech_tokens = model.t3.inference(  # type: ignore[attr-defined]
            t3_cond=model.conds.t3,  # type: ignore[attr-defined]
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=cfg_weight,
            repetition_penalty=1.2,
            min_p=0.05,
            top_p=1.0,
        )
        speech_tokens = speech_tokens[0]
        # NOTE: at the pinned chatterbox-tts==0.1.7 version, `drop_invalid_tokens` lives in
        # `chatterbox.models.s3tokenizer` (re-exported into `chatterbox.tts`'s own module
        # namespace), not `chatterbox.models.utils` — verified by reading the installed
        # package's `tts.py` directly (`from .models.s3tokenizer import ... drop_invalid_tokens`)
        # after this task's smoke gate hit an ImportError on the wrong module path.
        from chatterbox.models.s3tokenizer import (  # type: ignore[import-not-found]
            drop_invalid_tokens,
        )

        speech_tokens = drop_invalid_tokens(speech_tokens)
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(model.device)  # type: ignore[attr-defined]
        t5 = time.perf_counter()
        lm_decode_ms = (t5 - t4) * 1000.0

        # ── vocoder_ms: S3Gen. ──
        wav, _ = model.s3gen.inference(  # type: ignore[attr-defined]
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,  # type: ignore[attr-defined]
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)  # type: ignore[attr-defined]
    t6 = time.perf_counter()
    vocoder_ms = (t6 - t5) * 1000.0

    t_end = time.perf_counter()
    wall_time = t_end - t_call_start

    audio = np.asarray(watermarked_wav, dtype=np.float32).reshape(-1)
    sample_rate = int(getattr(model, "sr", 24000))
    audio_duration_s = len(audio) / sample_rate if sample_rate > 0 else 0.0
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0
    audio_16k = _resample_to_16k(audio, sample_rate)

    result = SynthResult(
        audio_array_16khz=audio_16k,
        audio_array_native=audio,
        native_sample_rate=sample_rate,
        ttfb_s=wall_time,  # whole-utterance, non-streaming
        rtf=rtf,
        audio_duration_s=audio_duration_s,
    )
    timing = StageTiming(
        ref_conditioning_ms=ref_conditioning_ms,
        text_frontend_ms=text_frontend_ms,
        lm_decode_ms=lm_decode_ms,
        vocoder_ms=vocoder_ms,
        total_ms=wall_time * 1000.0,
        notes="Chatterbox: all four stages independently timed, no merge needed.",
    )
    return TimedSynthResult(result=result, stage_timing=timing)
