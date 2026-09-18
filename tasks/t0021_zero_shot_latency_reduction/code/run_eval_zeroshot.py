"""CLI entry point for the zero-shot latency-reduction acceleration-variant sweep (plan Step 8-9).

Adapted from `tasks.t0018_zero_shot_cloning_calibration.code.run_eval_zeroshot` (copied, not
imported — t0018 is not a registered library). New in this task: the `--acceleration-variant`
flag, per-variant loader/synth kwargs (`VARIANT_KWARGS` below), and `StageTiming` persistence
(`results/latency_breakdown_<system>_<variant>.json`) alongside the existing per-clip metrics.

Run from EACH system's own isolated venv interpreter directly, e.g.::

    .venv-cosyvoice2/bin/python -m tasks.t0021_zero_shot_latency_reduction.code.run_eval_zeroshot \\
        --system cosyvoice2 --acceleration-variant fp16 --conditions ref_single \\
        --prompt-set both --n-warmup 50 --cosyvoice-model-dir /mnt/.../pretrained/cosyvoice2

    .venv-chatterbox/bin/python -m tasks.t0021_zero_shot_latency_reduction.code.run_eval_zeroshot \\
        --system chatterbox --acceleration-variant torch_compile --conditions ref_single \\
        --prompt-set both --n-warmup 50

Saves synthesized WAVs to `results/audio_samples/harness/<system>_<variant>_<condition>/
<prompt_set>/<i>.wav` and per-clip records to
`results/per_clip_metrics_<system>_<variant>_<condition>.json`, plus aggregated stage-timing sums
to `results/latency_breakdown_<system>_<variant>.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from statistics import mean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

VALID_VARIANTS = {
    # F5-TTS is a cheap closure item only (S-0018-01), not a variant-sweep target — a single
    # baseline setting, no acceleration levers (out of this task's scope per plan.md Step 12).
    "f5_tts": ["baseline_new_ref"],
    "cosyvoice2": [
        "baseline_new_ref",
        "ref_cache",
        "fp16",
        "load_jit",
        "load_trt",
        "vllm_backend",
    ],
    "chatterbox": [
        "baseline_new_ref",
        "ref_cache",
        "precision_bf16_or_fp16",
        "torch_compile",
        "sentence_chunking",
        "streaming_api",
    ],
}

# ── Cumulative-stack loader kwargs per CosyVoice2 variant (plan.md Step 8) ─────────────────────
COSYVOICE2_LOADER_KWARGS: dict[str, dict[str, bool]] = {
    "baseline_new_ref": {"load_jit": False, "load_trt": False, "fp16": False, "load_vllm": False},
    "ref_cache": {"load_jit": False, "load_trt": False, "fp16": False, "load_vllm": False},
    "fp16": {"load_jit": False, "load_trt": False, "fp16": True, "load_vllm": False},
    "load_jit": {"load_jit": True, "load_trt": False, "fp16": True, "load_vllm": False},
    "load_trt": {"load_jit": True, "load_trt": True, "fp16": True, "load_vllm": False},
    "vllm_backend": {"load_jit": False, "load_trt": False, "fp16": False, "load_vllm": True},
}
COSYVOICE2_USE_REF_CACHE: dict[str, bool] = {
    "baseline_new_ref": False,
    "ref_cache": True,
    "fp16": True,
    "load_jit": True,
    "load_trt": True,
    "vllm_backend": True,
}


def _write_environment_record(
    system: str, variant: str, *, extra: dict[str, object] | None = None
) -> None:
    """Capture model version/commit + torch/CUDA versions at synthesis time (Lesson 4)."""
    from tasks.t0021_zero_shot_latency_reduction.code.paths import RESULTS_ENVIRONMENT

    record: dict[str, object] = {
        "system": system,
        "acceleration_variant": variant,
        "tensorrt_version": None,
        "vllm_version": None,
        "cosyvoice2_git_sha": None,
        **(extra or {}),
    }
    try:
        import torch

        record["torch_version"] = torch.__version__
        record["torch_cuda_version"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            record["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        record["torch_error"] = str(exc)

    if system == "cosyvoice2":
        record["model_checkpoint"] = "FunAudioLLM/CosyVoice2-0.5B"
        try:
            import tensorrt  # type: ignore[import-not-found]

            record["tensorrt_version"] = tensorrt.__version__
        except ImportError:
            pass
        try:
            import vllm  # type: ignore[import-not-found]

            record["vllm_version"] = vllm.__version__
        except ImportError:
            pass
    elif system == "chatterbox":
        try:
            import chatterbox

            record["package_version"] = getattr(chatterbox, "__version__", "unknown")
        except Exception:
            pass
        record["model_checkpoint"] = "ResembleAI/chatterbox"

    RESULTS_ENVIRONMENT.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if RESULTS_ENVIRONMENT.exists():
        existing = json.loads(RESULTS_ENVIRONMENT.read_text(encoding="utf-8"))
    existing[f"{system}_{variant}"] = record
    RESULTS_ENVIRONMENT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("Environment recorded for %s/%s -> %s", system, variant, RESULTS_ENVIRONMENT)


def _split_sentences(text: str) -> list[str]:
    """Minimal sentence splitter for the `sentence_chunking` Chatterbox variant."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p) > 0] or [text]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot latency-reduction acceleration sweep runner")
    p.add_argument("--system", required=True, choices=["f5_tts", "cosyvoice2", "chatterbox"])
    p.add_argument("--acceleration-variant", required=True)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["ref_single"],
        help="One or more of: ref_single, ref_concat.",
    )
    p.add_argument("--prompt-set", default="both", choices=["val96", "fillers", "both"])
    p.add_argument("--n-warmup", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="Limit prompts per set (smoke tests)")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--cosyvoice-model-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _run_one_condition(
    *,
    system: str,
    variant: str,
    condition: str,
    synth: object,
    is_streaming: bool,
    prompts_by_set: dict[str, list[object]],
    n_warmup: int,
    limit: int | None,
    audio_out_dir: Path,
    out_dir: Path,
) -> None:
    from tasks.t0008_tts_eval_harness_baselines.code.adapters import save_wav
    from tasks.t0008_tts_eval_harness_baselines.code.harness import PromptItem
    from tasks.t0021_zero_shot_latency_reduction.code.constants import WARMUP_TEXT

    variant_slug = f"{system}_{variant}_{condition}"

    logger.info("[%s] Running %d warmup synthesis(es)...", variant_slug, n_warmup)
    n_warmup_ok = 0
    for _ in range(n_warmup):
        try:
            synth(WARMUP_TEXT)  # type: ignore[operator]
            n_warmup_ok += 1
        except Exception as exc:
            logger.warning("[%s] Warmup failed: %s", variant_slug, exc)
    logger.info("[%s] Warmup: %d/%d succeeded", variant_slug, n_warmup_ok, n_warmup)

    records: list[dict[str, object]] = []
    stage_sums: dict[str, list[float]] = {}
    variant_start_time = time.perf_counter()

    for set_name, prompts_list in prompts_by_set.items():
        prompts: list[PromptItem] = (
            prompts_list[:limit] if limit is not None else list(prompts_list)
        )  # type: ignore[index]
        logger.info("[%s/%s] Synthesizing %d prompts...", variant_slug, set_name, len(prompts))

        for i, prompt in enumerate(prompts):
            assert isinstance(prompt, PromptItem)
            audio_path = audio_out_dir / set_name / f"{i:04d}.wav"
            try:
                timed = synth(prompt.text)  # type: ignore[operator]
                result = timed.result
                st = timed.stage_timing
                save_wav(
                    audio=result.audio_array_native,
                    sample_rate=result.native_sample_rate,
                    path=audio_path,
                )
                for field in (
                    "ref_encoding_ms",
                    "ref_conditioning_ms",
                    "text_frontend_ms",
                    "lm_prefill_decode_ms",
                    "lm_decode_ms",
                    "flow_matching_vocoder_ms",
                    "vocoder_ms",
                    "total_ms",
                ):
                    val = getattr(st, field, None)
                    if val is not None:
                        stage_sums.setdefault(field, []).append(val)
                records.append(
                    {
                        "system": system,
                        "acceleration_variant": variant,
                        "condition": condition,
                        "prompt_set": set_name,
                        "text": prompt.text,
                        "ttfb_ms": result.ttfb_s * 1000.0,
                        "rtf": result.rtf,
                        "speaker_sim": None,
                        "speaker_sim_radiohost_control": None,
                        "duration_ratio": None,
                        "wer": None,
                        "audio_path": str(audio_path),
                        "ref_duration_s": prompt.ref_duration_s,
                        "synth_duration_s": result.audio_duration_s,
                        "is_streaming": is_streaming,
                    }
                )
            except Exception as exc:
                logger.error(
                    "[%s/%s] Synthesis failed for prompt %d (%r): %s",
                    variant_slug,
                    set_name,
                    i,
                    prompt.text[:40],
                    exc,
                )
                records.append(
                    {
                        "system": system,
                        "acceleration_variant": variant,
                        "condition": condition,
                        "prompt_set": set_name,
                        "text": prompt.text,
                        "ttfb_ms": None,
                        "rtf": None,
                        "speaker_sim": None,
                        "speaker_sim_radiohost_control": None,
                        "duration_ratio": None,
                        "wer": None,
                        "audio_path": None,
                        "ref_duration_s": prompt.ref_duration_s,
                        "synth_duration_s": None,
                        "is_streaming": is_streaming,
                    }
                )

    variant_wall_clock_s = time.perf_counter() - variant_start_time
    n_clips = len(records)
    n_successful = sum(1 for r in records if r["ttfb_ms"] is not None)
    logger.info(
        "[%s] Done: %d/%d successful, wall_clock=%.1fs",
        variant_slug,
        n_successful,
        n_clips,
        variant_wall_clock_s,
    )

    out_path = out_dir / f"per_clip_metrics_{variant_slug}.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Saved %d records -> %s", len(records), out_path)

    timing_path = out_dir / f"timing_{variant_slug}.json"
    timing_path.write_text(
        json.dumps(
            {
                "variant_slug": variant_slug,
                "system": system,
                "acceleration_variant": variant,
                "condition": condition,
                "n_clips": n_clips,
                "n_successful": n_successful,
                "wall_clock_s": variant_wall_clock_s,
                "n_warmup": n_warmup,
                "n_warmup_ok": n_warmup_ok,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved timing -> %s", timing_path)

    # NOTE: filename MUST include `condition` — a prior run without it caused the ref_concat
    # closure's (Milestone 4) write to silently clobber the main sweep's (Milestone 3) ref_single
    # breakdown file for the same (system, variant) pair, since both share acceleration_variant
    # "baseline_new_ref". Found and fixed during this task's own implementation.
    breakdown_path = out_dir / f"latency_breakdown_{system}_{variant}_{condition}.json"
    breakdown: dict[str, object] = {
        field: {"mean_ms": mean(vals), "n": len(vals)} for field, vals in stage_sums.items()
    }
    breakdown_path.write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
    logger.info("Saved stage-timing breakdown -> %s", breakdown_path)


def main() -> None:
    args = parse_args()

    from tasks.t0008_tts_eval_harness_baselines.code.harness import get_prompts_by_set
    from tasks.t0021_zero_shot_latency_reduction.code import adapters_zeroshot as az
    from tasks.t0021_zero_shot_latency_reduction.code.paths import (
        REF_CONCAT_WAV,
        REF_SINGLE_WAV,
        REFERENCES_MANIFEST,
        RESULTS_AUDIO_HARNESS_DIR,
        RESULTS_DIR,
    )

    assert args.acceleration_variant in VALID_VARIANTS[args.system], (
        f"Unknown variant {args.acceleration_variant!r} for system {args.system!r}"
    )

    out_dir = args.out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_by_set = get_prompts_by_set(args.prompt_set)
    logger.info("Prompts loaded: %s", {k: len(v) for k, v in prompts_by_set.items()})

    manifest = (
        json.loads(REFERENCES_MANIFEST.read_text(encoding="utf-8"))
        if REFERENCES_MANIFEST.exists()
        else {}
    )

    conditions: list[str] = list(args.conditions)
    variant = args.acceleration_variant

    # ── Load the model ONCE per process, with the variant's loader kwargs ──────
    is_streaming = False
    use_ref_cache = False
    if args.system == "f5_tts":
        model = az.load_f5tts_model(hf_cache_dir=args.hf_cache_dir)
    elif args.system == "cosyvoice2":
        assert args.cosyvoice_model_dir is not None, "--cosyvoice-model-dir required for cosyvoice2"
        loader_kwargs = COSYVOICE2_LOADER_KWARGS[variant]
        use_ref_cache = COSYVOICE2_USE_REF_CACHE[variant]
        logger.info("Loading CosyVoice2 (variant=%s, kwargs=%s)...", variant, loader_kwargs)
        model = az.load_cosyvoice2_model(args.cosyvoice_model_dir, **loader_kwargs)
        is_streaming = True
    elif args.system == "chatterbox":
        logger.info("Loading Chatterbox (variant=%s, device=%s)...", variant, args.device)
        model = az.load_chatterbox_model(device=args.device)
        use_ref_cache = variant != "baseline_new_ref"
        precision_dtype_used: str | None = None

        if variant == "precision_bf16_or_fp16":
            import torch

            from tasks.t0021_zero_shot_latency_reduction.code.paths import REF_SINGLE_WAV

            # NOTE (found via this task's own smoke gate, both attempts documented in
            # intervention/): casting `model.s3gen` wholesale to bf16 OR fp16 breaks ref
            # conditioning regardless of which reduced precision is chosen. `s3gen.embed_ref()`
            # runs the xvector speaker encoder's mel-frontend through torchaudio's
            # `Kaldi.fbank()`, which calls `torch.fft.rfft()` — and `torch.fft` only supports
            # float32/float64 on the pinned torch/torchaudio versions, not bf16 *or* fp16. This is
            # not fixable by "try bf16, fall back to fp16" (both fail the same way); the
            # architecturally correct fix is to leave S3Gen (vocoder + speaker encoder) at fp32 and
            # only cast T3 (the autoregressive LM decode stage — the part this variant is actually
            # targeting) to reduced precision. `speaker_sim`/`ttfb_ms` still reflect a genuine
            # precision variant: the decode stage (`lm_decode_ms`) is what actually runs in bf16.
            dtype_used = "bf16"
            try:
                model.t3.to(dtype=torch.bfloat16)
                # Probe with a real decode call (not just ref-conditioning, which never touches
                # T3) so an unsupported-dtype failure inside T3 itself would also be caught here,
                # before any warmup/measured call.
                model.prepare_conditionals(str(REF_SINGLE_WAV))
            except Exception as exc_bf16:
                logger.warning("bf16 cast/probe failed (%s); falling back to fp16", exc_bf16)
                dtype_used = "fp16"
                model.t3.to(dtype=torch.float16)
                model.prepare_conditionals(str(REF_SINGLE_WAV))
            logger.info(
                "Chatterbox precision variant using dtype=%s (T3 decoder only; S3Gen kept "
                "fp32 — see code comment for why)",
                dtype_used,
            )
            precision_dtype_used = dtype_used
        elif variant == "torch_compile":
            import torch

            model.t3 = torch.compile(model.t3)  # type: ignore[attr-defined]
            logger.info("Chatterbox T3 decoder wrapped with torch.compile (first call recompiles)")
    else:
        raise ValueError(f"Unknown system: {args.system}")

    env_extra = (
        {"precision_dtype_used": precision_dtype_used}
        if args.system == "chatterbox" and precision_dtype_used is not None
        else None
    )
    _write_environment_record(args.system, variant, extra=env_extra)

    for condition in conditions:
        ref_wav_path = REF_SINGLE_WAV if condition == "ref_single" else REF_CONCAT_WAV
        ref_text = manifest.get(f"{condition}_transcript", "")

        if use_ref_cache and args.system == "cosyvoice2":
            az.cosyvoice2_register_ref_cache(model, prompt_text=ref_text, ref_wav_path=ref_wav_path)
        elif use_ref_cache and args.system == "chatterbox":
            az.chatterbox_register_ref_cache(model, ref_wav_path=ref_wav_path)

        if args.system == "f5_tts":

            def synth(text: str, _ref: Path = ref_wav_path, _txt: str = ref_text) -> object:
                return az.f5_tts_synth(text, ref_wav_path=_ref, ref_text=_txt, model=model)
        elif args.system == "cosyvoice2":

            def synth(text: str, _ref: Path = ref_wav_path, _txt: str = ref_text) -> object:
                return az.cosyvoice2_synth(
                    text,
                    ref_wav_path=_ref,
                    prompt_text=_txt,
                    model=model,
                    use_ref_cache=use_ref_cache,
                )
        elif args.system == "chatterbox" and variant == "sentence_chunking":

            def synth(text: str, _ref: Path = ref_wav_path) -> object:
                sentences = _split_sentences(text)
                t_start = time.perf_counter()
                first: object | None = None
                chunks_audio = []
                stage_agg: dict[str, float] = {}
                for sent in sentences:
                    timed = az.chatterbox_synth(
                        sent, ref_wav_path=_ref, model=model, use_ref_cache=True
                    )
                    if first is None:
                        first = timed
                    chunks_audio.append(timed.result.audio_array_native)
                    for f in (
                        "ref_conditioning_ms",
                        "text_frontend_ms",
                        "lm_decode_ms",
                        "vocoder_ms",
                    ):
                        v = getattr(timed.stage_timing, f, None)
                        if v is not None:
                            stage_agg[f] = stage_agg.get(f, 0.0) + v
                t_end = time.perf_counter()
                total_wall_s = t_end - t_start
                import numpy as np

                full_audio = np.concatenate(chunks_audio, axis=0)
                assert first is not None
                sr = first.result.native_sample_rate  # type: ignore[union-attr]
                audio_duration_s = len(full_audio) / sr if sr > 0 else 0.0
                rtf = total_wall_s / audio_duration_s if audio_duration_s > 0 else 0.0
                from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
                    SynthResult,
                    _resample_to_16k,
                )

                result = SynthResult(
                    audio_array_16khz=_resample_to_16k(full_audio, sr),
                    audio_array_native=full_audio,
                    native_sample_rate=sr,
                    ttfb_s=first.result.ttfb_s,  # type: ignore[union-attr]
                    rtf=rtf,
                    audio_duration_s=audio_duration_s,
                )
                timing = az.StageTiming(
                    **stage_agg,
                    total_ms=first.result.ttfb_s * 1000.0,  # type: ignore[union-attr]
                    notes=(
                        f"sentence_chunking: {len(sentences)} chunk(s); TTFB is the first "
                        "chunk's own latency, stage sums are summed across all chunks."
                    ),
                )
                return az.TimedSynthResult(result=result, stage_timing=timing)
        elif args.system == "chatterbox":

            def synth(text: str, _ref: Path = ref_wav_path) -> object:
                return az.chatterbox_synth(
                    text, ref_wav_path=_ref, model=model, use_ref_cache=use_ref_cache
                )
        else:
            raise ValueError(f"Unknown system: {args.system}")

        variant_slug = f"{args.system}_{variant}_{condition}"
        audio_out_dir = RESULTS_AUDIO_HARNESS_DIR / variant_slug

        _run_one_condition(
            system=args.system,
            variant=variant,
            condition=condition,
            synth=synth,
            is_streaming=is_streaming,
            prompts_by_set=prompts_by_set,  # type: ignore[arg-type]
            n_warmup=args.n_warmup,
            limit=args.limit,
            audio_out_dir=audio_out_dir,
            out_dir=out_dir,
        )

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
