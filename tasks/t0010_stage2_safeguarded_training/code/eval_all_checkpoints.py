"""Batch checkpoint evaluation for t0010 Stage 2 v10 run.

For each epoch checkpoint in --checkpoint-dir:
1. Extract 5-module state dict (extract_decoder.extract)
2. Synthesize 100 filler prompts via Kokoro pipeline
3. Score speaker_sim via score_speaker_sim.py in the resemblyzer venv
4. Write per-epoch eval JSON to --output-dir/epoch_NNN_eval.json

Implements early stopping: if speaker_sim has peaked and declined for 3 consecutive epochs,
stop evaluation (the remaining checkpoints are unlikely to improve).

Design: synthesis (Kokoro) runs in main Python env; speaker_sim scoring runs via subprocess
in the resemblyzer venv to avoid webrtcvad conflicts.

Usage (run on VM inside ~/kokoro-finetune/):
    python eval_all_checkpoints.py \\
        --checkpoint-dir logs/v10 \\
        --output-dir data/run_v10/eval_results \\
        --config configs/config_david_v10.yml \\
        --voicepack /path/to/david_v3_best_voicepack.pt \\
        --centroid /path/to/reference_centroid.npy \\
        --resemblyzer-venv ~/resemblyzer-venv \\
        --repo-root /path/to/rail-arf-tts \\
        --skip-wer
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Early-stopping: stop if speaker_sim declines for this many consecutive epochs after peak
EARLY_STOP_PATIENCE: int = 3

# Validation gate: first epoch speaker_sim must exceed this (else extraction failed)
MIN_SPEAKER_SIM_GATE: float = 0.35

# Duration explosion gate: clip > 30 s → failure
MAX_CLIP_DURATION_S: float = 30.0

# Warmup prompts (discarded from timing/scoring)
N_WARMUP: int = 1
WARMUP_TEXT: str = "Checking the latest press release."


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FillerPrompt:
    text: str
    ref_duration_s: float | None


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    checkpoint_name: str
    speaker_sim_mean: float | None
    speaker_sim_std: float | None
    ttfb_p50_ms: float | None
    rtf_mean: float | None
    n_success: int
    n_total: int
    eval_json_path: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_filler_prompts(filler_json: Path) -> list[FillerPrompt]:
    """Load filler prompts from the cached JSON manifest."""
    raw: list[dict[str, object]] = json.loads(filler_json.read_text(encoding="utf-8"))
    return [
        FillerPrompt(
            text=str(item["text"]),
            ref_duration_s=(
                float(str(item["ref_duration_s"])) if item.get("ref_duration_s") else None
            ),
        )
        for item in raw
    ]


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    """Return (epoch_index, path) pairs sorted by epoch number.

    Kokoro Stage 2 checkpoint files are named: epoch_2nd_NNNNN.pth
    where NNNNN is the step count (not epoch number).
    The manifest tracks which step corresponds to which epoch.
    We sort by step counter as a proxy for epoch order.
    """
    ckpts = sorted(checkpoint_dir.glob("epoch_2nd_*.pth"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No epoch_2nd_*.pth files found in {checkpoint_dir}")

    result: list[tuple[int, Path]] = []
    for i, ckpt in enumerate(ckpts):
        result.append((i + 1, ckpt))

    logger.info("Found %d checkpoints in %s", len(result), checkpoint_dir)
    return result


def extract_checkpoint(ckpt_path: Path, out_path: Path, repo_root: Path) -> dict[str, int]:
    """Extract 5-module state dict from raw StyleTTS2 checkpoint."""
    sys.path.insert(0, str(repo_root))
    from tasks.t0008_tts_eval_harness_baselines.code.extract_decoder import extract

    return extract(ckpt_path=ckpt_path, out_path=out_path)


def synthesize_fillers(
    packaged_ckpt: Path,
    voicepack: Path,
    prompts: list[FillerPrompt],
    audio_dir: Path,
    repo_root: Path,
) -> list[dict[str, object]]:
    """Synthesize all filler prompts and return per-clip records (no speaker_sim yet)."""
    sys.path.insert(0, str(repo_root))
    from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline
    from tasks.t0008_tts_eval_harness_baselines.code.adapters import (
        load_kokoro_model_with_checkpoint,
        save_wav,
    )

    audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Kokoro model from %s ...", packaged_ckpt.name)
    model = load_kokoro_model_with_checkpoint(packaged_ckpt)
    pipeline = build_pipeline(model=model)

    # Warmup run (discard)
    logger.info("Warming up (%d run)...", N_WARMUP)
    for _ in range(N_WARMUP):
        try:
            for _, _, _ in pipeline(WARMUP_TEXT, voice=str(voicepack)):
                pass
        except Exception as exc:
            logger.warning("Warmup failed: %s", exc)

    records: list[dict[str, object]] = []
    for i, prompt in enumerate(prompts):
        wav_path = audio_dir / f"{i:04d}.wav"
        try:
            chunks = []
            ttfb_s: float | None = None
            t_start = time.perf_counter()
            for _, _, audio in pipeline(prompt.text, voice=str(voicepack)):
                if audio is not None and len(audio) > 0:
                    if ttfb_s is None:
                        ttfb_s = time.perf_counter() - t_start
                    import numpy as np

                    try:
                        import torch

                        if isinstance(audio, torch.Tensor):
                            audio = audio.detach().cpu().numpy()
                    except ImportError:
                        pass
                    if isinstance(audio, np.ndarray):
                        chunks.append(audio.astype(np.float32))
            t_end = time.perf_counter()

            if ttfb_s is None or len(chunks) == 0:
                raise RuntimeError("No audio output from pipeline")

            import numpy as np

            full_audio = np.concatenate(chunks)
            audio_duration_s = len(full_audio) / 24000
            wall_time = t_end - t_start
            rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0

            # Duration explosion gate
            if audio_duration_s > MAX_CLIP_DURATION_S:
                logger.warning(
                    "Clip %d duration %.1fs > %.1fs (explosion) — marking failed",
                    i,
                    audio_duration_s,
                    MAX_CLIP_DURATION_S,
                )
                records.append(
                    {
                        "prompt_idx": i,
                        "text": prompt.text,
                        "ttfb_ms": None,
                        "rtf": None,
                        "speaker_sim": None,
                        "synth_duration_s": audio_duration_s,
                        "ref_duration_s": prompt.ref_duration_s,
                        "audio_path": None,
                        "error": f"duration_explosion:{audio_duration_s:.1f}s",
                    }
                )
                continue

            save_wav(audio=full_audio, sample_rate=24000, path=wav_path)
            records.append(
                {
                    "prompt_idx": i,
                    "text": prompt.text,
                    "ttfb_ms": ttfb_s * 1000.0,
                    "rtf": rtf,
                    "speaker_sim": None,  # filled by score_speaker_sim
                    "synth_duration_s": audio_duration_s,
                    "ref_duration_s": prompt.ref_duration_s,
                    "audio_path": str(wav_path),
                    "error": None,
                }
            )

        except Exception as exc:
            logger.error("Synthesis failed for prompt %d (%r): %s", i, prompt.text[:40], exc)
            records.append(
                {
                    "prompt_idx": i,
                    "text": prompt.text,
                    "ttfb_ms": None,
                    "rtf": None,
                    "speaker_sim": None,
                    "synth_duration_s": None,
                    "ref_duration_s": prompt.ref_duration_s,
                    "audio_path": None,
                    "error": str(exc),
                }
            )

    del model, pipeline
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    n_ok = sum(1 for r in records if r.get("audio_path") is not None)
    logger.info("Synthesized %d/%d clips successfully", n_ok, len(records))
    return records


def score_speaker_sim(
    records: list[dict[str, object]],
    per_clip_json: Path,
    centroid_path: Path,
    resemblyzer_python: Path,
    score_speaker_sim_script: Path,
) -> list[dict[str, object]]:
    """Write per-clip JSON, invoke score_speaker_sim.py in resemblyzer venv, read back."""
    per_clip_json.parent.mkdir(parents=True, exist_ok=True)
    per_clip_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

    scored_path = per_clip_json.with_suffix(".scored.json")

    cmd = [
        str(resemblyzer_python),
        str(score_speaker_sim_script),
        "--per-clip-in",
        str(per_clip_json),
        "--corpus-dir",
        "UNUSED_CENTROID_PROVIDED",  # not used when --centroid-path provided
        "--per-clip-out",
        str(scored_path),
        "--centroid-path",
        str(centroid_path),
        "--skip-wer",
    ]
    logger.info("Running score_speaker_sim.py ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(
            "score_speaker_sim.py failed:\nstdout: %s\nstderr: %s",
            result.stdout,
            result.stderr,
        )
        raise RuntimeError(f"score_speaker_sim.py exited {result.returncode}")

    scored: list[dict[str, object]] = json.loads(scored_path.read_text(encoding="utf-8"))
    return scored


def aggregate_epoch(records: list[dict[str, object]]) -> dict[str, float | int | None]:
    """Aggregate per-clip records into epoch-level metrics."""
    import statistics

    speaker_sims: list[float] = [
        float(str(r["speaker_sim"])) for r in records if r.get("speaker_sim") is not None
    ]
    ttfbs: list[float] = [float(str(r["ttfb_ms"])) for r in records if r.get("ttfb_ms") is not None]
    rtfs: list[float] = [float(str(r["rtf"])) for r in records if r.get("rtf") is not None]
    n_success = sum(1 for r in records if r.get("audio_path") is not None)

    return {
        "speaker_sim_mean": statistics.mean(speaker_sims) if len(speaker_sims) > 0 else None,
        "speaker_sim_std": statistics.stdev(speaker_sims) if len(speaker_sims) > 1 else None,
        "ttfb_p50_ms": statistics.median(ttfbs) if len(ttfbs) > 0 else None,
        "rtf_mean": statistics.mean(rtfs) if len(rtfs) > 0 else None,
        "n_success": n_success,
        "n_total": len(records),
    }


def run_epoch_eval(
    *,
    epoch_idx: int,
    ckpt_path: Path,
    output_dir: Path,
    voicepack: Path,
    centroid_path: Path,
    prompts: list[FillerPrompt],
    repo_root: Path,
    resemblyzer_python: Path,
    score_speaker_sim_script: Path,
) -> EpochResult:
    """Full pipeline for one checkpoint: extract → synth → score → aggregate."""
    epoch_tag = f"epoch_{epoch_idx:03d}"
    epoch_dir = output_dir / epoch_tag
    epoch_dir.mkdir(parents=True, exist_ok=True)

    packaged_path = epoch_dir / f"{ckpt_path.stem}_packaged.pth"

    # Step 1: extract
    logger.info("[%s] Extracting 5-module checkpoint from %s ...", epoch_tag, ckpt_path.name)
    module_counts = extract_checkpoint(
        ckpt_path=ckpt_path, out_path=packaged_path, repo_root=repo_root
    )
    total_params = sum(module_counts.values())
    logger.info(
        "[%s] Extracted %d total params across %d modules",
        epoch_tag,
        total_params,
        len(module_counts),
    )

    # Step 2: synthesize
    audio_dir = epoch_dir / "audio"
    records = synthesize_fillers(
        packaged_ckpt=packaged_path,
        voicepack=voicepack,
        prompts=prompts,
        audio_dir=audio_dir,
        repo_root=repo_root,
    )

    # Step 3: score speaker_sim
    per_clip_json = epoch_dir / "per_clip_pre_score.json"
    try:
        records = score_speaker_sim(
            records=records,
            per_clip_json=per_clip_json,
            centroid_path=centroid_path,
            resemblyzer_python=resemblyzer_python,
            score_speaker_sim_script=score_speaker_sim_script,
        )
    except Exception as exc:
        logger.error(
            "[%s] speaker_sim scoring failed: %s — speaker_sim will be None", epoch_tag, exc
        )

    # Step 4: aggregate
    agg = aggregate_epoch(records)

    # Write final per-clip and epoch summary
    per_clip_final = epoch_dir / "per_clip_scored.json"
    per_clip_final.write_text(json.dumps(records, indent=2), encoding="utf-8")

    epoch_result_path = output_dir / f"{epoch_tag}_eval.json"
    epoch_json: dict[str, object] = {
        "epoch": epoch_idx,
        "checkpoint_name": ckpt_path.name,
        "speaker_sim_mean": agg["speaker_sim_mean"],
        "speaker_sim_std": agg["speaker_sim_std"],
        "ttfb_p50_ms": agg["ttfb_p50_ms"],
        "rtf_mean": agg["rtf_mean"],
        "n_success": agg["n_success"],
        "n_total": agg["n_total"],
        "module_param_counts": module_counts,
        "per_clip_path": str(per_clip_final),
    }
    epoch_result_path.write_text(json.dumps(epoch_json, indent=2), encoding="utf-8")
    logger.info(
        "[%s] speaker_sim=%.4f, ttfb_p50=%.1fms, rtf=%.3f (%d/%d ok)",
        epoch_tag,
        agg["speaker_sim_mean"] or 0.0,
        agg["ttfb_p50_ms"] or 0.0,
        agg["rtf_mean"] or 0.0,
        agg["n_success"],
        agg["n_total"],
    )

    sim_mean = agg["speaker_sim_mean"]
    sim_std = agg["speaker_sim_std"]
    ttfb = agg["ttfb_p50_ms"]
    rtf_mean = agg["rtf_mean"]

    return EpochResult(
        epoch=epoch_idx,
        checkpoint_name=ckpt_path.name,
        speaker_sim_mean=float(sim_mean) if isinstance(sim_mean, (int, float)) else None,
        speaker_sim_std=float(sim_std) if isinstance(sim_std, (int, float)) else None,
        ttfb_p50_ms=float(ttfb) if isinstance(ttfb, (int, float)) else None,
        rtf_mean=float(rtf_mean) if isinstance(rtf_mean, (int, float)) else None,
        n_success=int(str(agg["n_success"])),
        n_total=int(str(agg["n_total"])),
        eval_json_path=str(epoch_result_path),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch checkpoint evaluation for t0010 v10 run")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory with epoch_2nd_*.pth files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write epoch_NNN_eval.json files",
    )
    p.add_argument("--voicepack", type=Path, required=True, help="david_v3_best_voicepack.pt path")
    p.add_argument(
        "--centroid", type=Path, required=True, help="Precomputed 11labs centroid .npy path"
    )
    p.add_argument("--filler-json", type=Path, required=True, help="filler_prompts_100.json path")
    p.add_argument(
        "--resemblyzer-venv",
        type=Path,
        required=True,
        help="Path to resemblyzer virtualenv dir",
    )
    p.add_argument("--repo-root", type=Path, required=True, help="repo root (for task imports)")
    p.add_argument(
        "--score-script",
        type=Path,
        default=None,
        help="Path to score_speaker_sim.py (default: autodetect from repo-root)",
    )
    p.add_argument("--skip-wer", action="store_true", help="Skip WER scoring")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    checkpoint_dir: Path = args.checkpoint_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    voicepack: Path = args.voicepack.resolve()
    centroid_path: Path = args.centroid.resolve()
    filler_json: Path = args.filler_json.resolve()
    repo_root: Path = args.repo_root.resolve()
    resemblyzer_python: Path = (args.resemblyzer_venv / "bin" / "python").resolve()

    if args.score_script is not None:
        score_speaker_sim_script = args.score_script.resolve()
    else:
        score_speaker_sim_script = (
            repo_root
            / "tasks"
            / "t0008_tts_eval_harness_baselines"
            / "code"
            / "score_speaker_sim.py"
        )

    # Validate inputs
    for path, name in [
        (checkpoint_dir, "checkpoint-dir"),
        (voicepack, "voicepack"),
        (centroid_path, "centroid"),
        (filler_json, "filler-json"),
        (resemblyzer_python, "resemblyzer venv python"),
        (score_speaker_sim_script, "score_speaker_sim.py"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_filler_prompts(filler_json)
    logger.info("Loaded %d filler prompts", len(prompts))

    checkpoints = find_checkpoints(checkpoint_dir)
    logger.info("Found %d checkpoints to evaluate", len(checkpoints))

    results: list[EpochResult] = []
    peak_speaker_sim: float = -1.0
    declining_count: int = 0

    for epoch_idx, ckpt_path in checkpoints:
        logger.info("=" * 60)
        logger.info("Evaluating checkpoint %d/%d: %s", epoch_idx, len(checkpoints), ckpt_path.name)

        result = run_epoch_eval(
            epoch_idx=epoch_idx,
            ckpt_path=ckpt_path,
            output_dir=output_dir,
            voicepack=voicepack,
            centroid_path=centroid_path,
            prompts=prompts,
            repo_root=repo_root,
            resemblyzer_python=resemblyzer_python,
            score_speaker_sim_script=score_speaker_sim_script,
        )
        results.append(result)

        # Validation gate: first epoch must have reasonable speaker_sim
        sim1 = result.speaker_sim_mean
        if epoch_idx == 1 and sim1 is not None and sim1 < MIN_SPEAKER_SIM_GATE:
            logger.error(
                "VALIDATION GATE FAILED: epoch 1 speaker_sim=%.4f < %.2f threshold. "
                "5-module extraction may have failed. Halting evaluation.",
                sim1,
                MIN_SPEAKER_SIM_GATE,
            )
            # Write partial summary and exit with error
            _write_summary(results, output_dir)
            sys.exit(1)

        # Early stopping check
        if result.speaker_sim_mean is not None:
            if result.speaker_sim_mean > peak_speaker_sim:
                peak_speaker_sim = result.speaker_sim_mean
                declining_count = 0
            else:
                declining_count += 1
                if declining_count >= EARLY_STOP_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d: speaker_sim has declined for %d consecutive "
                        "epochs since peak %.4f",
                        epoch_idx,
                        EARLY_STOP_PATIENCE,
                        peak_speaker_sim,
                    )
                    break

    _write_summary(results, output_dir)
    logger.info("Batch evaluation complete. %d epochs evaluated.", len(results))


def _write_summary(results: list[EpochResult], output_dir: Path) -> None:
    """Write eval_summary.json with all epoch results."""
    summary: list[dict[str, object]] = [
        {
            "epoch": r.epoch,
            "checkpoint_name": r.checkpoint_name,
            "speaker_sim_mean": r.speaker_sim_mean,
            "speaker_sim_std": r.speaker_sim_std,
            "ttfb_p50_ms": r.ttfb_p50_ms,
            "rtf_mean": r.rtf_mean,
            "n_success": r.n_success,
            "n_total": r.n_total,
            "eval_json_path": r.eval_json_path,
        }
        for r in results
    ]
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote eval summary → %s", summary_path)

    if len(results) > 0:
        valid = [r for r in results if r.speaker_sim_mean is not None]
        if len(valid) > 0:
            best = max(valid, key=lambda r: r.speaker_sim_mean or -1.0)
            logger.info(
                "Best epoch: %d  speaker_sim=%.4f  ttfb_p50=%.1fms  rtf=%.3f",
                best.epoch,
                best.speaker_sim_mean or 0.0,
                best.ttfb_p50_ms or 0.0,
                best.rtf_mean or 0.0,
            )


if __name__ == "__main__":
    main()
