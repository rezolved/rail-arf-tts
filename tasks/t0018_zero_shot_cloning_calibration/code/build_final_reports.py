"""Produce results/metrics.json, results/tables.json, and the 4 required charts (plan Step 13-14).

Also folds in `kokoro_v3_bundle`'s stored (not re-measured) baseline numbers and records the
`cosyvoice2_ref_concat` / `f5_tts` null-variant reasons explicitly in tables.json notes.
"""

from __future__ import annotations

import json
import logging

from tasks.t0008_tts_eval_harness_baselines.code.report import REGISTERED_METRIC_KEYS
from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    CLONING_SYSTEMS,
    KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS,
    KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96,
    REFERENCE_CONDITIONS,
)
from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
    CHART_REF_CONDITION_EFFECT,
    CHART_SPEAKER_SIM_BY_SYSTEM,
    CHART_TTFB_VS_SPEAKER_SIM,
    CHART_WER_BY_SYSTEM,
    RESULTS_METRICS,
    RESULTS_PER_CLIP_METRICS,
    RESULTS_TABLES,
)
from tasks.t0018_zero_shot_cloning_calibration.code.report_zeroshot import (
    build_metrics_json,
    build_tables_json,
    plot_ref_condition_effect,
    plot_speaker_sim_by_system,
    plot_ttfb_vs_speaker_sim,
    plot_wer_by_system,
    variant_id_for,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

NOTES = [
    "f5_tts: NULL for all variants (ref_single, ref_concat). Smoke gate hung indefinitely across "
    "3 attempts (~30+ min total) inside F5TTS.__init__ with zero progress signal, despite an "
    "earlier bare model-class load succeeding once. Marked null per REQ-7. See "
    "intervention/f5_tts_smoke_gate_failed.md.",
    "cosyvoice2_ref_concat: NULL (0/196 successful, REQ-6 rejection rule, well below the 80% "
    "threshold). Root cause: CosyVoice2's own frontend hard-rejects any reference/prompt audio "
    "longer than 30s (assert in cosyvoice/cli/frontend.py: 'do not support extract speech token "
    "for audio longer than 30s'). This task's ref_concat clip is 30.57s -- 0.57s over the limit. "
    "This is a genuine system limitation (hard error), not a corpus/harness bug, and is a valid, "
    "reportable finding for Key Question 4 (ref_single vs ref_concat): CosyVoice2 cannot use "
    "reference audio anywhere near 30s at all, unlike F5-TTS (which silently crops long "
    "references, per the plan's own pre-registered caveat) and Chatterbox (which accepted the "
    "full 30.57s clip without issue, 196/196 successful on both conditions).",
    "kokoro_v3_bundle: NOT RE-MEASURED this session. Two independent attempts to load the v3 "
    "decoder checkpoint (including one after copying it to local disk to rule out network-"
    "filesystem I/O as the cause) both hung indefinitely, mirroring the f5_tts failure. Falls "
    "back to t0008's stored numbers (speaker_sim only; ttfb_ms/rtf omitted per Lesson 1 -- "
    "cross-session latency is not pairable). Source: "
    "tasks/t0008_tts_eval_harness_baselines/results/metrics.json "
    "(verified directly, not from memory). See "
    "intervention/kokoro_v3_bundle_not_remeasured.md.",
    "elevenlabs_david: speaker_sim re-scored fresh this session against a half-B centroid "
    "(matching t0008's own self-consistency-ceiling methodology); ttfb_ms/rtf omitted (no live "
    "API call made this session, per plan Step 8a -- Lesson 1).",
    "ref_single (all 3 cloning systems): built as a concatenation of 10 half-A clips (~10.8s "
    "total), not a single natural utterance -- the real 11labs_david corpus has no single clip "
    "near 10s (max 1.67s corpus-wide). See code/build_references.py's module docstring for the "
    "full preflight-inspection finding. Do not present as a literal single-utterance condition.",
    "GPU-bound synthesis budget: ~$46 of the $70 hard cap was consumed by the time all attempts "
    "(successful and failed) completed. See results/cost_tracking.json for the full timestamped "
    "ledger.",
]


def _kokoro_stored_variant(prompt_set: str, speaker_sim: float) -> dict[str, object]:
    return {
        "variant_id": f"kokoro_v3_bundle_{prompt_set}",
        "system": "kokoro_v3_bundle",
        "condition": None,
        "prompt_set": prompt_set,
        "speaker_sim": speaker_sim,
        "speaker_sim_std": None,
        "ttfb_ms": None,
        "ttfb_ms_p50": None,
        "ttfb_ms_p95": None,
        "ttfb_ms_p99": None,
        "rtf": None,
        "rtf_std": None,
        "rejected_reason": None,
        "duration_ratio_median": None,
        "duration_explosion_fraction": None,
        "wer_mean": None,
        "n_clips": None,
        "n_successful": None,
        "success_rate": None,
        "efficiency_inference_time_per_item_seconds": None,
        "efficiency_inference_cost_per_item_usd": None,
        "gate_failure_count": None,
        "delta_speaker_sim_vs_elevenlabs": None,
        "delta_speaker_sim_vs_kokoro_v3_bundle": 0.0,
        "environment": None,
        "source": "t0008 stored, not re-measured this session",
    }


def main() -> None:
    raw_text = RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    records: list[dict[str, object]] = json.loads(raw_text)
    logger.info("Loaded %d per-clip records", len(records))

    # Timing lookup keyed by (system, condition, prompt_set) variant_id for tables.json use.
    timing_lookup: dict[str, dict[str, object]] = {}
    for system in CLONING_SYSTEMS:
        for condition in REFERENCE_CONDITIONS:
            slug = f"{system}_{condition}"
            tpath = RESULTS_TABLES.parent / f"timing_{slug}.json"
            if tpath.exists():
                data = json.loads(tpath.read_text(encoding="utf-8"))
                for ps in ("val96", "fillers"):
                    timing_lookup[variant_id_for(system, condition, ps)] = data

    env_path = RESULTS_TABLES.parent / "environment.json"
    environment = json.loads(env_path.read_text(encoding="utf-8"))

    gate_failures: dict[str, object] = {}
    gate_failures_path = RESULTS_TABLES.parent / "gate_failures.json"
    if gate_failures_path.exists():
        gate_failures = json.loads(gate_failures_path.read_text(encoding="utf-8"))

    variant_keys: list[tuple[str, str | None, str]] = []
    for system in CLONING_SYSTEMS:
        for condition in REFERENCE_CONDITIONS:
            for ps in ("val96", "fillers"):
                variant_keys.append((system, condition, ps))
    for ps in ("val96", "fillers"):
        variant_keys.append(("elevenlabs_david", None, ps))

    metrics_data = build_metrics_json(records, variant_keys, timing_lookup, environment)

    # Add the kokoro_v3_bundle stored (registered-keys-only) entries.
    metrics_data["variants"].append(
        {
            "variant_id": "kokoro_v3_bundle_fillers",
            "label": "kokoro_v3_bundle / fillers (t0008 stored, not re-measured)",
            "dimensions": {
                "system": "kokoro_v3_bundle",
                "condition": None,
                "prompt_set": "fillers",
            },
            "metrics": {
                k: v
                for k, v in {"speaker_sim": KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS}.items()
                if k in REGISTERED_METRIC_KEYS
            },
        }
    )
    metrics_data["variants"].append(
        {
            "variant_id": "kokoro_v3_bundle_val96",
            "label": "kokoro_v3_bundle / val96 (t0008 stored, not re-measured)",
            "dimensions": {"system": "kokoro_v3_bundle", "condition": None, "prompt_set": "val96"},
            "metrics": {
                k: v
                for k, v in {"speaker_sim": KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96}.items()
                if k in REGISTERED_METRIC_KEYS
            },
        }
    )

    RESULTS_METRICS.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    logger.info(
        "Wrote metrics.json -> %s (%d variants)", RESULTS_METRICS, len(metrics_data["variants"])
    )

    tables_data = build_tables_json(
        records, variant_keys, timing_lookup, environment, gate_failures, {}, NOTES
    )
    tables_data["rows"].append(
        _kokoro_stored_variant("fillers", KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS)
    )
    tables_data["rows"].append(_kokoro_stored_variant("val96", KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96))

    cost_tracking_path = RESULTS_TABLES.parent / "cost_tracking.json"
    if cost_tracking_path.exists():
        tables_data["cost_tracking"] = json.loads(cost_tracking_path.read_text(encoding="utf-8"))

    RESULTS_TABLES.write_text(json.dumps(tables_data, indent=2), encoding="utf-8")
    logger.info("Wrote tables.json -> %s (%d rows)", RESULTS_TABLES, len(tables_data["rows"]))

    plot_speaker_sim_by_system(records, variant_keys, CHART_SPEAKER_SIM_BY_SYSTEM)
    plot_ttfb_vs_speaker_sim(records, variant_keys, CHART_TTFB_VS_SPEAKER_SIM)
    plot_ref_condition_effect(records, CLONING_SYSTEMS, CHART_REF_CONDITION_EFFECT)
    plot_wer_by_system(records, variant_keys, CHART_WER_BY_SYSTEM)
    logger.info("Charts written to results/images/")


if __name__ == "__main__":
    main()
