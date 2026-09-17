"""CLI entry point for the zero-shot voice-cloning calibration harness (plan Step 6).

Run from EACH system's own isolated venv interpreter directly, e.g.::

    .venv-f5tts/bin/python -m tasks.t0018....code.run_eval_zeroshot \\
        --system f5_tts --conditions ref_single ref_concat --prompt-set both --n-warmup 50

    .venv-cosyvoice2/bin/python -m tasks.t0018....code.run_eval_zeroshot \\
        --system cosyvoice2 --conditions ref_single ref_concat --prompt-set both \\
        --n-warmup 50 --cosyvoice-model-dir /mnt/.../pretrained/cosyvoice2

    .venv-chatterbox/bin/python -m tasks.t0018....code.run_eval_zeroshot \\
        --system chatterbox --conditions ref_single ref_concat --prompt-set both --n-warmup 50

For `kokoro_v3_bundle` (paired baseline, GPU, no `--conditions`), run from the MAIN project venv
(it already has `kokoro` installed):

    uv run python -m tasks.t0018_zero_shot_cloning_calibration.code.run_eval_zeroshot \\
        --system kokoro_v3_bundle --prompt-set both --n-warmup 50

**Cost note (discovered during implementation):** the VM's model-loading overhead (many minutes,
network-filesystem-bound Python imports/weight loads) dominates the per-variant cost far more than
the plan's original estimate assumed. `--conditions` accepts BOTH `ref_single` and `ref_concat` in
one process invocation (one model load, looped over both reference conditions) instead of two
separate invocations, to avoid paying the load overhead twice per cloning system.

Saves synthesized WAVs to `results/audio_samples/harness/<system>_<condition>/<prompt_set>/<i>.wav`
(REQ-8: every clip, nothing discarded) and per-clip records to
`results/per_clip_metrics_<system>_<condition>.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot voice-cloning calibration harness runner")
    p.add_argument(
        "--system",
        required=True,
        choices=["f5_tts", "cosyvoice2", "chatterbox", "kokoro_v3_bundle"],
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=[None],
        help="One or more of: ref_single, ref_concat. Omit (or pass nothing) for baselines.",
    )
    p.add_argument("--prompt-set", default="both", choices=["val96", "fillers", "both"])
    p.add_argument("--n-warmup", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="Limit prompts per set (smoke tests)")
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--cosyvoice-model-dir", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _write_environment_record(system: str) -> None:
    """Capture model version/commit + torch/CUDA versions at synthesis time (Lesson 4)."""
    from tasks.t0018_zero_shot_cloning_calibration.code.paths import RESULTS_ENVIRONMENT

    record: dict[str, object] = {"system": system}
    try:
        import torch

        record["torch_version"] = torch.__version__
        record["torch_cuda_version"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            record["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        record["torch_error"] = str(exc)

    if system == "f5_tts":
        try:
            import f5_tts

            record["package_version"] = getattr(f5_tts, "__version__", "unknown")
            record["model_checkpoint"] = "F5TTS_v1_Base (SWivid/F5-TTS)"
        except Exception:
            pass
    elif system == "cosyvoice2":
        record["model_checkpoint"] = "FunAudioLLM/CosyVoice2-0.5B"
    elif system == "chatterbox":
        try:
            import chatterbox

            record["package_version"] = getattr(chatterbox, "__version__", "unknown")
        except Exception:
            pass
        record["model_checkpoint"] = "ResembleAI/chatterbox"
    elif system == "kokoro_v3_bundle":
        record["model_checkpoint"] = "t0006 v3 best decoder + voicepack"

    RESULTS_ENVIRONMENT.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, object] = {}
    if RESULTS_ENVIRONMENT.exists():
        existing = json.loads(RESULTS_ENVIRONMENT.read_text(encoding="utf-8"))
    existing[system] = record
    RESULTS_ENVIRONMENT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    logger.info("Environment recorded for %s -> %s", system, RESULTS_ENVIRONMENT)


def _run_one_condition(
    *,
    system: str,
    condition: str | None,
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
    from tasks.t0018_zero_shot_cloning_calibration.code.constants import WARMUP_TEXT

    variant_slug = f"{system}_{condition}" if condition is not None else system

    logger.info("[%s/%s] Running %d warmup synthesis(es)...", system, condition, n_warmup)
    n_warmup_ok = 0
    for _ in range(n_warmup):
        try:
            synth(WARMUP_TEXT)  # type: ignore[operator]
            n_warmup_ok += 1
        except Exception as exc:
            logger.warning("[%s] Warmup failed: %s", system, exc)
    logger.info("[%s] Warmup: %d/%d succeeded", system, n_warmup_ok, n_warmup)

    records: list[dict[str, object]] = []
    variant_start_time = time.perf_counter()

    for set_name, prompts_list in prompts_by_set.items():
        prompts: list[PromptItem] = (
            prompts_list[:limit] if limit is not None else list(prompts_list)
        )  # type: ignore[index]
        logger.info("[%s/%s] Synthesizing %d prompts...", system, set_name, len(prompts))

        for i, prompt in enumerate(prompts):
            assert isinstance(prompt, PromptItem)
            audio_path = audio_out_dir / set_name / f"{i:04d}.wav"
            try:
                result = synth(prompt.text)  # type: ignore[operator]
                save_wav(
                    audio=result.audio_array_native,  # type: ignore[union-attr]
                    sample_rate=result.native_sample_rate,  # type: ignore[union-attr]
                    path=audio_path,
                )
                records.append(
                    {
                        "system": system,
                        "condition": condition,
                        "prompt_set": set_name,
                        "text": prompt.text,
                        "ttfb_ms": result.ttfb_s * 1000.0,  # type: ignore[union-attr]
                        "rtf": result.rtf,  # type: ignore[union-attr]
                        "speaker_sim": None,
                        "duration_ratio": None,
                        "wer": None,
                        "audio_path": str(audio_path),
                        "ref_duration_s": prompt.ref_duration_s,
                        "synth_duration_s": result.audio_duration_s,  # type: ignore[union-attr]
                        "is_streaming": is_streaming,
                    }
                )
            except Exception as exc:
                logger.error(
                    "[%s/%s] Synthesis failed for prompt %d (%r): %s",
                    system,
                    set_name,
                    i,
                    prompt.text[:40],
                    exc,
                )
                records.append(
                    {
                        "system": system,
                        "condition": condition,
                        "prompt_set": set_name,
                        "text": prompt.text,
                        "ttfb_ms": None,
                        "rtf": None,
                        "speaker_sim": None,
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
        "[%s/%s] Done: %d/%d successful, wall_clock=%.1fs",
        system,
        condition,
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


def main() -> None:
    args = parse_args()

    from tasks.t0008_tts_eval_harness_baselines.code.harness import get_prompts_by_set
    from tasks.t0018_zero_shot_cloning_calibration.code import adapters_zeroshot as az
    from tasks.t0018_zero_shot_cloning_calibration.code.constants import N_WARMUP
    from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
        REF_CONCAT_WAV,
        REF_SINGLE_WAV,
        REFERENCES_MANIFEST,
        RESULTS_AUDIO_HARNESS_DIR,
        RESULTS_DIR,
    )

    n_warmup = args.n_warmup if args.n_warmup is not None else N_WARMUP
    out_dir = args.out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_by_set = get_prompts_by_set(args.prompt_set)
    logger.info("Prompts loaded: %s", {k: len(v) for k, v in prompts_by_set.items()})

    manifest = (
        json.loads(REFERENCES_MANIFEST.read_text(encoding="utf-8"))
        if REFERENCES_MANIFEST.exists()
        else {}
    )

    conditions: list[str | None] = list(args.conditions) if args.conditions != [None] else [None]

    # ── Load the model ONCE (dominant cost on this VM's network filesystem) ───
    is_streaming = False
    if args.system == "f5_tts":
        logger.info("Loading F5-TTS model (hf_cache_dir=%s)...", args.hf_cache_dir)
        model = az.load_f5tts_model(hf_cache_dir=args.hf_cache_dir)
    elif args.system == "cosyvoice2":
        assert args.cosyvoice_model_dir is not None, "--cosyvoice-model-dir required for cosyvoice2"
        logger.info("Loading CosyVoice2 model from %s...", args.cosyvoice_model_dir)
        model = az.load_cosyvoice2_model(args.cosyvoice_model_dir)
        is_streaming = True
    elif args.system == "chatterbox":
        logger.info("Loading Chatterbox model (device=%s)...", args.device)
        model = az.load_chatterbox_model(device=args.device)
    elif args.system == "kokoro_v3_bundle":
        from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline
        from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
            load_kokoro_model_with_checkpoint,
        )
        from tasks.t0018_zero_shot_cloning_calibration.code.paths import V3_DECODER

        logger.info("Loading kokoro_v3_bundle model (V3_DECODER=%s)...", V3_DECODER)
        kmodel = load_kokoro_model_with_checkpoint(V3_DECODER)
        model = build_pipeline(model=kmodel)
    else:
        raise ValueError(f"Unknown system: {args.system}")

    _write_environment_record(args.system)

    for condition in conditions:
        ref_wav_path = REF_SINGLE_WAV if condition == "ref_single" else REF_CONCAT_WAV
        ref_text = manifest.get(f"{condition}_transcript", "") if condition is not None else ""

        if args.system == "f5_tts":

            def synth(text: str, _ref=ref_wav_path, _txt=ref_text) -> object:
                return az.f5_tts_synth(text, ref_wav_path=_ref, ref_text=_txt, model=model)
        elif args.system == "cosyvoice2":

            def synth(text: str, _ref=ref_wav_path, _txt=ref_text) -> object:
                return az.cosyvoice2_synth(text, ref_wav_path=_ref, prompt_text=_txt, model=model)
        elif args.system == "chatterbox":

            def synth(text: str, _ref=ref_wav_path) -> object:
                return az.chatterbox_synth(text, ref_wav_path=_ref, model=model)
        elif args.system == "kokoro_v3_bundle":
            from tasks.t0008_tts_eval_harness_baselines.code.adapters import kokoro_v3_bundle
            from tasks.t0018_zero_shot_cloning_calibration.code.paths import V3_VOICEPACK

            def synth(text: str) -> object:
                return kokoro_v3_bundle(text, pipeline=model, voicepack_path=V3_VOICEPACK)
        else:
            raise ValueError(f"Unknown system: {args.system}")

        variant_slug = f"{args.system}_{condition}" if condition is not None else args.system
        audio_out_dir = RESULTS_AUDIO_HARNESS_DIR / variant_slug

        _run_one_condition(
            system=args.system,
            condition=condition,
            synth=synth,
            is_streaming=is_streaming,
            prompts_by_set=prompts_by_set,  # type: ignore[arg-type]
            n_warmup=n_warmup,
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
