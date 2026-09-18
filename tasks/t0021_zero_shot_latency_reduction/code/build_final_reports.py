"""Produce results/metrics.json, results/tables.json, results/latency_breakdown.json, and the 3
required charts (plan Step 13-14).

Adapted from `tasks.t0018_zero_shot_cloning_calibration.code.build_final_reports` (copied, not
imported), extended to the 4-dimension (system, acceleration_variant, condition, prompt_set)
variant space and to fold in the 3 pre-registered null variants
(`cosyvoice2_vllm_backend`, `chatterbox_streaming_api`, `f5_tts_baseline_new_ref`) explicitly, so a
reader sees they were attempted and why they are null, rather than silently absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tasks.t0021_zero_shot_latency_reduction.code.constants import (
    CHATTERBOX_VARIANTS,
    COSYVOICE2_VARIANTS,
)
from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    CHART_LATENCY_BREAKDOWN_STACKED,
    CHART_TTFB_P50_P95_BY_VARIANT,
    CHART_TTFB_VS_SPEAKER_SIM_VARIANTS,
    RESULTS_DIR,
    RESULTS_ENVIRONMENT,
    RESULTS_GATE_FAILURES,
    RESULTS_LATENCY_BREAKDOWN,
    RESULTS_METRICS,
    RESULTS_PER_CLIP_METRICS,
    RESULTS_TABLES,
)
from tasks.t0021_zero_shot_latency_reduction.code.report_zeroshot import (
    VariantKey,
    build_metrics_json,
    build_tables_json,
    plot_latency_breakdown_stacked,
    plot_ttfb_p50_p95_by_variant,
    plot_ttfb_vs_speaker_sim_variants,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

COST_TRACKING_PATH = RESULTS_DIR / "cost_tracking.json"

NOTES = [
    "cosyvoice2_vllm_backend: NULL (never run). The `.venv-cosyvoice2-vllm` install hit its "
    "pre-authorized 20-minute cutoff mid-unpack during setup-machines (torch installed, vllm "
    "itself not reached; no CUDA/dependency incompatibility was ever observed, only a wall-clock "
    "cutoff). Not retried during implementation given the task's remaining time/budget. See "
    "intervention/cosyvoice2_vllm_install_timeout.md.",
    "chatterbox_streaming_api: NULL (never run). Verified by reading the installed "
    "chatterbox-tts==0.1.7 package source directly (`grep -n stream chatterbox/tts.py`): no "
    "`stream` token appears anywhere in `tts.py` at this pinned version. No native streaming API "
    "exists to test, per the plan's own pre-registered fallback ('if no such API exists at this "
    "version, mark null with that reason recorded').",
    "f5_tts (all conditions/prompt_sets): NULL. Retried (S-0018-01) in a fresh shell session on a "
    "freshly re-provisioned VM, with py-spy installed. Hung again inside `F5TTS.__init__`'s "
    "`datasets`-package import chain, confirmed via a working `py-spy dump` stack trace (t0018's "
    "own attempt could not get a working stack trace) to be blocked on `importlib` reading a "
    "source file from the VM's Azure Files SMB mount -- an infrastructure-level mount stall, not a "
    "task-code bug. Killed after ~7 minutes (well inside the 45-minute cap) per the plan's 'do not "
    "retry a third time' rule. See intervention/f5_tts_retry_still_hangs.md.",
    "cosyvoice2_baseline_new_ref / ref_single: this task's own `latency_breakdown_<system>_"
    "<variant>.json` filename briefly omitted `condition`, so the ref_concat closure run "
    "(Milestone 4) clobbered the ref_single sweep's breakdown file for this one (system, variant) "
    "pair. Found during implementation; the filename now includes `condition` "
    "(`latency_breakdown_<system>_<variant>_<condition>.json`), and the ref_single "
    "baseline_new_ref cell was re-run once (196/196 successful) to regenerate the correct file. "
    "No other cell was affected -- every other (system, variant) pair only ever ran under "
    "`ref_single`.",
    "chatterbox_precision_bf16_or_fp16: casting `s3gen` (vocoder + xvector speaker encoder) "
    "wholesale to bf16 OR fp16 breaks ref-conditioning on the pinned torch/torchaudio versions "
    "(`torch.fft.rfft`, used by the xvector mel-frontend, supports only float32/float64). Both "
    "were tried live during implementation and both failed identically. Fixed by casting only "
    "`model.t3` (the T3 LM decoder -- the stage this variant is actually meant to target) to "
    "bf16/fp16 and leaving S3Gen at fp32; `results/environment.json`'s "
    "`chatterbox_precision_bf16_or_fp16.precision_dtype_used` records which reduced-precision "
    "dtype was actually used (`bf16` -- the fp32-fallback branch was never triggered once S3Gen "
    "was correctly excluded from the cast).",
]

CONDITION_FOR_VARIANT_SUFFIX = "_ref_single"


def _load_json(path: Path, default: object) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _build_variant_keys() -> list[VariantKey]:
    keys: list[VariantKey] = []
    for variant in COSYVOICE2_VARIANTS:
        for ps in ("val96", "fillers"):
            keys.append(("cosyvoice2", variant, "ref_single", ps))
    for ps in ("val96", "fillers"):
        keys.append(("cosyvoice2", "baseline_new_ref", "ref_concat", ps))
    for variant in CHATTERBOX_VARIANTS:
        for ps in ("val96", "fillers"):
            keys.append(("chatterbox", variant, "ref_single", ps))
    return keys


def _null_variant_row(
    system: str, variant: str, condition: str | None, prompt_set: str, reason: str
) -> dict[str, object]:
    vid = f"{system}_{variant}_{condition}_{prompt_set}" if condition else f"{system}_{variant}"
    return {
        "variant_id": vid,
        "system": system,
        "acceleration_variant": variant,
        "condition": condition,
        "prompt_set": prompt_set,
        "speaker_sim": None,
        "speaker_sim_std": None,
        "speaker_sim_radiohost_control": None,
        "ttfb_ms": None,
        "ttfb_ms_p50": None,
        "ttfb_ms_p95": None,
        "ttfb_ms_p99": None,
        "rtf": None,
        "rtf_std": None,
        "rejected_reason": reason,
        "duration_ratio_median": None,
        "duration_explosion_fraction": None,
        "wer_mean": None,
        "n_clips": None,
        "n_successful": None,
        "success_rate": None,
        "efficiency_inference_time_per_item_seconds": None,
        "efficiency_inference_cost_per_item_usd": None,
        "gate_failure_count": None,
        "t0018_baseline_speaker_sim_wrong_voice_reference": None,
        "t0018_baseline_ttfb_ms_wrong_voice_reference": None,
        "meets_ttfb_target_300ms": False,
        "latency_breakdown": None,
        "environment": None,
    }


def _load_latency_breakdown() -> dict[str, object]:
    """Key by `<system>_<variant>` (report_zeroshot.py's plotting/table lookup scheme). Only the
    ref_single condition is included here -- ref_concat is a one-off closure item, not part of the
    main 6-variant-per-system sweep this dict feeds."""
    out: dict[str, object] = {}
    for fpath in sorted(RESULTS_DIR.glob("latency_breakdown_*_ref_single.json")):
        stem = fpath.stem  # latency_breakdown_<system>_<variant>_ref_single
        rest = stem[len("latency_breakdown_") :]
        key = rest[: -len(CONDITION_FOR_VARIANT_SUFFIX)]
        out[key] = json.loads(fpath.read_text(encoding="utf-8"))
    return out


def main() -> None:
    records: list[dict[str, object]] = json.loads(
        RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    )
    logger.info("Loaded %d per-clip records", len(records))

    environment = _load_json(RESULTS_ENVIRONMENT, {})
    gate_failures = _load_json(RESULTS_GATE_FAILURES, {})
    cost_tracking = _load_json(COST_TRACKING_PATH, [])

    timing_by_variant: dict[str, dict[str, object]] = {}
    for fpath in RESULTS_DIR.glob("timing_*.json"):
        data = json.loads(fpath.read_text(encoding="utf-8"))
        vid_base = f"{data['system']}_{data['acceleration_variant']}_{data['condition']}"
        for ps in ("val96", "fillers"):
            timing_by_variant[f"{vid_base}_{ps}"] = data

    latency_breakdown = _load_latency_breakdown()
    RESULTS_LATENCY_BREAKDOWN.write_text(json.dumps(latency_breakdown, indent=2), encoding="utf-8")
    logger.info(
        "Wrote latency_breakdown.json -> %s (%d cells)",
        RESULTS_LATENCY_BREAKDOWN,
        len(latency_breakdown),
    )

    variant_keys = _build_variant_keys()
    metrics_data = build_metrics_json(records, variant_keys)
    RESULTS_METRICS.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    logger.info(
        "Wrote metrics.json -> %s (%d variants)", RESULTS_METRICS, len(metrics_data["variants"])
    )

    tables_data = build_tables_json(
        records,
        variant_keys,
        timing_by_variant,
        latency_breakdown,
        environment,  # type: ignore[arg-type]
        gate_failures,  # type: ignore[arg-type]
        cost_tracking,  # type: ignore[arg-type]
        NOTES,
    )
    tables_data["rows"].append(  # type: ignore[union-attr]
        _null_variant_row(
            "cosyvoice2",
            "vllm_backend",
            "ref_single",
            "val96",
            "vllm install timed out during setup (20-minute cutoff, no CUDA incompatibility "
            "observed) -- see intervention/cosyvoice2_vllm_install_timeout.md",
        )
    )
    tables_data["rows"].append(  # type: ignore[union-attr]
        _null_variant_row(
            "cosyvoice2",
            "vllm_backend",
            "ref_single",
            "fillers",
            "vllm install timed out during setup (20-minute cutoff, no CUDA incompatibility "
            "observed) -- see intervention/cosyvoice2_vllm_install_timeout.md",
        )
    )
    tables_data["rows"].append(  # type: ignore[union-attr]
        _null_variant_row(
            "chatterbox",
            "streaming_api",
            "ref_single",
            "val96",
            "no native streaming API exists in chatterbox-tts==0.1.7 (verified: no `stream` token "
            "in the installed package's tts.py)",
        )
    )
    tables_data["rows"].append(  # type: ignore[union-attr]
        _null_variant_row(
            "chatterbox",
            "streaming_api",
            "ref_single",
            "fillers",
            "no native streaming API exists in chatterbox-tts==0.1.7 (verified: no `stream` token "
            "in the installed package's tts.py)",
        )
    )
    for condition in ("ref_single", "ref_concat"):
        for ps in ("val96", "fillers"):
            tables_data["rows"].append(  # type: ignore[union-attr]
                _null_variant_row(
                    "f5_tts",
                    "baseline_new_ref",
                    condition,
                    ps,
                    "S-0018-01 retry hung again inside F5TTS.__init__'s datasets-package import "
                    "chain (confirmed via py-spy stack trace: blocked on an Azure Files SMB mount "
                    "read, not a task-code bug) -- see intervention/f5_tts_retry_still_hangs.md",
                )
            )

    RESULTS_TABLES.write_text(json.dumps(tables_data, indent=2), encoding="utf-8")
    logger.info("Wrote tables.json -> %s (%d rows)", RESULTS_TABLES, len(tables_data["rows"]))  # type: ignore[arg-type]

    plot_latency_breakdown_stacked(latency_breakdown, CHART_LATENCY_BREAKDOWN_STACKED)
    plot_ttfb_vs_speaker_sim_variants(records, variant_keys, CHART_TTFB_VS_SPEAKER_SIM_VARIANTS)
    plot_ttfb_p50_p95_by_variant(records, variant_keys, CHART_TTFB_P50_P95_BY_VARIANT)
    logger.info("Charts written to results/images/")


if __name__ == "__main__":
    main()
