"""CLI entry point for the TTS evaluation harness.

Usage:
    python -m tasks.t0008_tts_eval_harness_baselines.code.run_eval \\
        --systems elevenlabs_david kokoro_base_george \\
        --prompt-set both \\
        --out-dir results/ \\
        --n-warmup 1 \\
        --elevenlabs-api-key sk_... \\
        --v3-voicepack-path tasks/t0006.../david_v3_best_voicepack.pt \\
        --v3-decoder-path tasks/t0006.../david_v3_best_decoder_kokoro.pth \\
        --t0006-packaged-path tasks/t0008.../data/packaged/t0006_v6d_epoch6.pth \\
        --t0005-packaged-path tasks/t0008.../data/packaged/t0005_run06_epoch3.pth

Produces:
    per_clip_metrics_<system_group>.json  (array of per-clip dicts)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTS evaluation harness runner")
    p.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Systems to evaluate. Default: all.",
    )
    p.add_argument(
        "--prompt-set",
        default="both",
        choices=["val96", "fillers", "both"],
        help="Prompt set to use.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for per-clip JSON files.",
    )
    p.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        help="Warmup synthesis count per system (discarded before timing).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N prompts per prompt-set (for smoke tests).",
    )
    p.add_argument(
        "--elevenlabs-api-key",
        default=None,
        help="ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.",
    )
    p.add_argument(
        "--v3-voicepack-path",
        type=Path,
        default=None,
        help="Path to david_v3_best_voicepack.pt",
    )
    p.add_argument(
        "--v3-decoder-path",
        type=Path,
        default=None,
        help="Path to david_v3_best_decoder_kokoro.pth (packaged five-module).",
    )
    p.add_argument(
        "--t0006-packaged-path",
        type=Path,
        default=None,
        help="Path to packaged t0006 v6d checkpoint.",
    )
    p.add_argument(
        "--t0005-packaged-path",
        type=Path,
        default=None,
        help="Path to packaged t0005 run06 checkpoint.",
    )
    p.add_argument(
        "--output-prefix",
        default="per_clip_metrics",
        help="Output JSON file prefix (default: per_clip_metrics → per_clip_metrics_<system>.json)",
    )
    p.add_argument(
        "--audio-out-dir",
        type=Path,
        default=None,
        help="Directory to save synthesized WAV files. If None, a temp dir is used.",
    )
    return p.parse_args()


def get_elevenlabs_voice_id(api_key: str) -> str:
    """Look up 'David' voice ID from ElevenLabs API."""
    import requests  # type: ignore[import-untyped]

    from tasks.t0008_tts_eval_harness_baselines.code.constants import (
        ELEVENLABS_API_BASE,
        ELEVENLABS_DAVID_VOICE_NAME,
    )

    resp = requests.get(
        f"{ELEVENLABS_API_BASE}/voices",
        headers={"xi-api-key": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    voices: list[dict[str, object]] = resp.json()["voices"]
    for voice in voices:
        if voice.get("name") == ELEVENLABS_DAVID_VOICE_NAME:
            voice_id = str(voice["voice_id"])
            logger.info("Found ElevenLabs voice '%s' → %s", ELEVENLABS_DAVID_VOICE_NAME, voice_id)
            return voice_id
    # Fallback: case-insensitive match
    for voice in voices:
        name = str(voice.get("name", "")).lower()
        if "david" in name:
            voice_id = str(voice["voice_id"])
            logger.warning("Fuzzy-matched ElevenLabs voice '%s' → %s", voice["name"], voice_id)
            return voice_id
    available = [v.get("name") for v in voices]
    raise RuntimeError(f"ElevenLabs voice 'David' not found. Available: {available}")


def run_system(
    system_name: str,
    prompts_by_set: dict[str, list[object]],
    *,
    n_warmup: int,
    audio_out_dir: Path,
    limit: int | None,
    # System-specific kwargs (populated per-system)
    elevenlabs_api_key: str | None = None,
    elevenlabs_voice_id: str | None = None,
    elevenlabs_session: object = None,
    kokoro_pipeline: object = None,
    voicepack_path: Path | None = None,
) -> list[dict[str, object]]:
    """Run synthesis for a system across all prompt sets, return per-clip records."""

    from tasks.t0008_tts_eval_harness_baselines.code import adapters
    from tasks.t0008_tts_eval_harness_baselines.code.adapters import SynthResult, save_wav
    from tasks.t0008_tts_eval_harness_baselines.code.constants import (
        ELEVENLABS_MAX_RETRIES,
        ELEVENLABS_RATE_LIMIT_SLEEP_S,
        SYSTEM_ELEVENLABS_DAVID,
        SYSTEM_KOKORO_BASE_GEORGE,
        SYSTEM_KOKORO_BASE_LEWIS,
        SYSTEM_KOKORO_BASE_V3_VOICEPACK,
        SYSTEM_KOKORO_FLOOR_CONTROL,
        SYSTEM_KOKORO_T0005_BEST,
        SYSTEM_KOKORO_T0006_V6D,
        SYSTEM_KOKORO_V3_BUNDLE,
    )
    from tasks.t0008_tts_eval_harness_baselines.code.harness import PromptItem

    # Build synthesis callable
    def synth(text: str) -> SynthResult:
        if system_name == SYSTEM_ELEVENLABS_DAVID:
            assert elevenlabs_api_key and elevenlabs_voice_id and elevenlabs_session
            for attempt in range(ELEVENLABS_MAX_RETRIES):
                try:
                    result = adapters.elevenlabs_david(
                        text=text,
                        api_key=elevenlabs_api_key,
                        voice_id=elevenlabs_voice_id,
                        session=elevenlabs_session,
                    )
                    time.sleep(ELEVENLABS_RATE_LIMIT_SLEEP_S)
                    return result
                except Exception as exc:
                    if attempt == ELEVENLABS_MAX_RETRIES - 1:
                        raise
                    logger.warning("ElevenLabs retry %d: %s", attempt + 1, exc)
                    time.sleep(2.0**attempt)
            raise RuntimeError("unreachable")
        elif system_name == SYSTEM_KOKORO_BASE_GEORGE:
            assert kokoro_pipeline
            return adapters.kokoro_george(text=text, pipeline=kokoro_pipeline)
        elif system_name == SYSTEM_KOKORO_BASE_LEWIS:
            assert kokoro_pipeline
            return adapters.kokoro_lewis(text=text, pipeline=kokoro_pipeline)
        elif system_name in (SYSTEM_KOKORO_BASE_V3_VOICEPACK, SYSTEM_KOKORO_V3_BUNDLE):
            assert kokoro_pipeline and voicepack_path
            if system_name == SYSTEM_KOKORO_BASE_V3_VOICEPACK:
                return adapters.kokoro_base_v3_voicepack(
                    text=text, pipeline=kokoro_pipeline, voicepack_path=voicepack_path
                )
            else:
                return adapters.kokoro_v3_bundle(
                    text=text, pipeline=kokoro_pipeline, voicepack_path=voicepack_path
                )
        elif system_name == SYSTEM_KOKORO_T0006_V6D:
            assert kokoro_pipeline and voicepack_path
            return adapters.kokoro_t0006_v6d(
                text=text, pipeline=kokoro_pipeline, voicepack_path=voicepack_path
            )
        elif system_name == SYSTEM_KOKORO_T0005_BEST:
            assert kokoro_pipeline and voicepack_path
            return adapters.kokoro_t0005_best(
                text=text, pipeline=kokoro_pipeline, voicepack_path=voicepack_path
            )
        elif system_name == SYSTEM_KOKORO_FLOOR_CONTROL:
            assert kokoro_pipeline
            return adapters.kokoro_floor_control(text=text, pipeline=kokoro_pipeline)
        else:
            raise ValueError(f"Unknown system: {system_name}")

    # Warmup runs (discard results — REQ-20)
    logger.info("[%s] Running %d warmup synthesis(es)...", system_name, n_warmup)
    warmup_text = "Checking the latest press release."
    for _ in range(n_warmup):
        try:
            synth(warmup_text)
        except Exception as exc:
            logger.warning("[%s] Warmup failed: %s", system_name, exc)

    # Measurement runs
    records: list[dict[str, object]] = []
    for set_name, prompts_list in prompts_by_set.items():
        prompts: list[PromptItem] = (
            prompts_list[:limit] if limit is not None else list(prompts_list)  # type: ignore[index]
        )
        logger.info("[%s/%s] Synthesizing %d prompts...", system_name, set_name, len(prompts))

        for i, prompt in enumerate(prompts):
            assert isinstance(prompt, PromptItem)
            audio_path = audio_out_dir / system_name / set_name / f"{i:04d}.wav"
            try:
                result = synth(prompt.text)
                save_wav(
                    audio=result.audio_array_native,
                    sample_rate=result.native_sample_rate,
                    path=audio_path,
                )
                records.append(
                    {
                        "system": system_name,
                        "prompt_set": set_name,
                        "text": prompt.text,
                        "ttfb_ms": result.ttfb_s * 1000.0,
                        "rtf": result.rtf,
                        "speaker_sim": None,  # filled in after scoring
                        "duration_ratio": None,  # filled in after scoring
                        "wer": None,  # filled in after scoring
                        "audio_path": str(audio_path),
                        "ref_duration_s": prompt.ref_duration_s,
                        "synth_duration_s": result.audio_duration_s,
                    }
                )
            except Exception as exc:
                logger.error(
                    "[%s/%s] Synthesis failed for prompt %d (%r): %s",
                    system_name,
                    set_name,
                    i,
                    prompt.text[:40],
                    exc,
                )
                records.append(
                    {
                        "system": system_name,
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
                    }
                )

    return records


def main() -> None:
    args = parse_args()

    from tasks.t0008_tts_eval_harness_baselines.code.constants import (
        ALL_SYSTEMS,
        KOKORO_SYSTEMS,
        SYSTEM_ELEVENLABS_DAVID,
    )
    from tasks.t0008_tts_eval_harness_baselines.code.harness import get_prompts_by_set

    systems: list[str] = args.systems if args.systems is not None else ALL_SYSTEMS
    logger.info("Systems to evaluate: %s", systems)
    logger.info("Prompt set: %s, warmup: %d", args.prompt_set, args.n_warmup)

    prompts_by_set = get_prompts_by_set(args.prompt_set)
    logger.info(
        "Prompts loaded: %s",
        {k: len(v) for k, v in prompts_by_set.items()},
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_out_dir = args.audio_out_dir or (args.out_dir.parent / "data" / "synth_audio")
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # ── ElevenLabs setup ──────────────────────────────────────────────────────
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_session: object = None

    if SYSTEM_ELEVENLABS_DAVID in systems:
        import os

        import requests  # type: ignore[import-untyped]

        elevenlabs_api_key = args.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
        if elevenlabs_api_key is None:
            logger.error(
                "ElevenLabs API key not provided (--elevenlabs-api-key or ELEVENLABS_API_KEY env)",
            )
            sys.exit(1)
        elevenlabs_voice_id = get_elevenlabs_voice_id(elevenlabs_api_key)
        elevenlabs_session = requests.Session()
        logger.info("ElevenLabs voice_id: %s", elevenlabs_voice_id)

    # ── Kokoro model loading ──────────────────────────────────────────────────
    kokoro_systems_to_run = [s for s in systems if s in KOKORO_SYSTEMS]
    # Group by which checkpoint they use to avoid loading the same model twice
    # Order: george/lewis (base), floor_control (base, lang_code=a), base_v3_voicepack
    # (base + voicepack), v3_bundle (v3 decoder), t0006 (t0006 decoder), t0005 (t0005 decoder)

    all_records: list[dict[str, object]] = []

    # Run ElevenLabs first (no GPU needed)
    if SYSTEM_ELEVENLABS_DAVID in systems:
        logger.info("=== ElevenLabs David ===")
        records = run_system(
            system_name=SYSTEM_ELEVENLABS_DAVID,
            prompts_by_set=prompts_by_set,  # type: ignore[arg-type]
            n_warmup=args.n_warmup,
            audio_out_dir=audio_out_dir,
            limit=args.limit,
            elevenlabs_api_key=elevenlabs_api_key,
            elevenlabs_voice_id=elevenlabs_voice_id,
            elevenlabs_session=elevenlabs_session,
        )
        all_records.extend(records)
        out_path = args.out_dir / f"{args.output_prefix}_elevenlabs.json"
        out_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
        logger.info("Saved %d ElevenLabs records → %s", len(records), out_path)

    # Run Kokoro systems
    if len(kokoro_systems_to_run) > 0:
        from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline
        from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
            load_kokoro_model_with_checkpoint,
        )
        from tasks.t0008_tts_eval_harness_baselines.code.constants import (
            SYSTEM_KOKORO_BASE_GEORGE,
            SYSTEM_KOKORO_BASE_LEWIS,
            SYSTEM_KOKORO_BASE_V3_VOICEPACK,
            SYSTEM_KOKORO_FLOOR_CONTROL,
            SYSTEM_KOKORO_T0005_BEST,
            SYSTEM_KOKORO_T0006_V6D,
            SYSTEM_KOKORO_V3_BUNDLE,
        )
        from tasks.t0008_tts_eval_harness_baselines.code.paths import (
            T0005_PACKAGED,
            T0006_V6D_PACKAGED,
            V3_DECODER,
            V3_VOICEPACK,
        )

        # (ckpt_path, voicepack_path, is_floor_control)
        kokoro_run_groups = [
            # base voices — stock model
            (SYSTEM_KOKORO_BASE_GEORGE, None, V3_VOICEPACK, False),
            (SYSTEM_KOKORO_BASE_LEWIS, None, V3_VOICEPACK, False),
            (SYSTEM_KOKORO_BASE_V3_VOICEPACK, None, V3_VOICEPACK, False),
            (SYSTEM_KOKORO_FLOOR_CONTROL, None, V3_VOICEPACK, True),
            # v3 bundle — fine-tuned decoder
            (SYSTEM_KOKORO_V3_BUNDLE, V3_DECODER, V3_VOICEPACK, False),
            # t0006
            (SYSTEM_KOKORO_T0006_V6D, T0006_V6D_PACKAGED, V3_VOICEPACK, False),
            # t0005
            (SYSTEM_KOKORO_T0005_BEST, T0005_PACKAGED, V3_VOICEPACK, False),
        ]

        for system_name, ckpt_path, voicepack_path, is_floor_control in kokoro_run_groups:
            if system_name not in kokoro_systems_to_run:
                continue

            logger.info("=== %s ===", system_name)

            # Validate paths
            if ckpt_path is not None and not ckpt_path.exists():
                logger.error("Checkpoint not found: %s — skipping %s", ckpt_path, system_name)
                continue
            if voicepack_path is not None and not voicepack_path.exists():
                logger.error("Voicepack not found: %s — skipping %s", voicepack_path, system_name)
                continue

            # Load model
            logger.info("Loading model for %s (ckpt=%s)...", system_name, ckpt_path)
            model = load_kokoro_model_with_checkpoint(ckpt_path)

            # Build pipeline — floor control uses lang_code="a" (American)
            if is_floor_control:
                from kokoro import KPipeline

                pipeline = KPipeline(lang_code="a", model=model)
                logger.info("Floor control: using lang_code='a'")
            else:
                pipeline = build_pipeline(model=model)
                logger.info("David voice: using build_pipeline (lang_code='b')")

            records = run_system(
                system_name=system_name,
                prompts_by_set=prompts_by_set,  # type: ignore[arg-type]
                n_warmup=args.n_warmup,
                audio_out_dir=audio_out_dir,
                limit=args.limit,
                kokoro_pipeline=pipeline,
                voicepack_path=voicepack_path,
            )
            all_records.extend(records)

            # Save incrementally
            kokoro_path = args.out_dir / f"{args.output_prefix}_kokoro.json"
            kokoro_records = [r for r in all_records if r.get("system") in KOKORO_SYSTEMS]
            kokoro_path.write_text(json.dumps(kokoro_records, indent=2), encoding="utf-8")
            logger.info("Saved %d Kokoro records → %s", len(kokoro_records), kokoro_path)

            # Free GPU memory
            del model, pipeline
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    # Save combined output
    combined_path = args.out_dir / f"{args.output_prefix}.json"
    combined_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    logger.info("Saved %d total records → %s", len(all_records), combined_path)


if __name__ == "__main__":
    main()
