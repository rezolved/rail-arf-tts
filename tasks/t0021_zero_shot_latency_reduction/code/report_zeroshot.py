"""Report writer: metrics.json/tables.json + the three required charts (plan Step 13-14).

Adapted from `tasks.t0018_zero_shot_cloning_calibration.code.report_zeroshot` (copied, not
imported), extended with a 4th dimension, `acceleration_variant`, per plan Step 13.
`variant_id` pattern: `<system>_<acceleration_variant>_<condition>_<prompt_set>`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.report import REGISTERED_METRIC_KEYS, _percentile
from tasks.t0021_zero_shot_latency_reduction.code.constants import (
    T0018_CHATTERBOX_SPEAKER_SIM_FILLERS,
    T0018_CHATTERBOX_SPEAKER_SIM_VAL96,
    T0018_CHATTERBOX_TTFB_MS_FILLERS,
    T0018_CHATTERBOX_TTFB_MS_VAL96,
    T0018_COSYVOICE2_SPEAKER_SIM_FILLERS,
    T0018_COSYVOICE2_SPEAKER_SIM_VAL96,
    T0018_COSYVOICE2_TTFB_MS_FILLERS,
    T0018_COSYVOICE2_TTFB_MS_VAL96,
    TTFB_TARGET_MS,
)

logger = logging.getLogger(__name__)

VariantKey = tuple[str, str, str, str]  # (system, acceleration_variant, condition, prompt_set)


def compute_variant_metrics(
    records: list[dict[str, object]],
    system: str,
    variant: str,
    condition: str,
    prompt_set: str,
) -> dict[str, float | None]:
    """Compute aggregate metrics for one (system, variant, condition, prompt_set) cell."""
    subset = [
        r
        for r in records
        if r["system"] == system
        and r.get("acceleration_variant") == variant
        and r.get("condition") == condition
        and r["prompt_set"] == prompt_set
    ]
    if len(subset) == 0:
        return {}

    ttfb_values = [float(r["ttfb_ms"]) for r in subset if r.get("ttfb_ms") is not None]  # type: ignore[arg-type]
    sim_values = [float(r["speaker_sim"]) for r in subset if r.get("speaker_sim") is not None]  # type: ignore[arg-type]
    sim_control_values = [
        float(r["speaker_sim_radiohost_control"])  # type: ignore[arg-type]
        for r in subset
        if r.get("speaker_sim_radiohost_control") is not None
    ]
    rtf_values = [float(r["rtf"]) for r in subset if r.get("rtf") is not None]  # type: ignore[arg-type]
    dur_values = [float(r["duration_ratio"]) for r in subset if r.get("duration_ratio") is not None]  # type: ignore[arg-type]
    wer_values = [float(r["wer"]) for r in subset if r.get("wer") is not None]  # type: ignore[arg-type]

    n_clips = len(subset)
    n_successful = sum(
        1 for r in subset if r.get("ttfb_ms") is not None or r.get("speaker_sim") is not None
    )
    success_rate = n_successful / n_clips if n_clips > 0 else 0.0

    metrics: dict[str, float | None] = {}

    # Lesson 3 rejection rule (pre-registered, plan.md Rejection Criteria).
    rejected = success_rate < 0.8
    if rejected:
        metrics["speaker_sim"] = None
        metrics["speaker_sim_std"] = None
        metrics["ttfb_ms"] = None
        metrics["ttfb_ms_p50"] = None
        metrics["ttfb_ms_p95"] = None
        metrics["ttfb_ms_p99"] = None
        metrics["rtf"] = None
        metrics["rtf_std"] = None
        metrics["rejected_reason"] = "successful_requests/total_requests < 0.8"
    else:
        metrics["speaker_sim"] = float(np.mean(sim_values)) if len(sim_values) > 0 else None
        metrics["speaker_sim_std"] = float(np.std(sim_values)) if len(sim_values) > 0 else None
        if len(ttfb_values) > 0:
            metrics["ttfb_ms"] = _percentile(ttfb_values, 50)
            metrics["ttfb_ms_p50"] = _percentile(ttfb_values, 50)
            metrics["ttfb_ms_p95"] = _percentile(ttfb_values, 95)
            metrics["ttfb_ms_p99"] = _percentile(ttfb_values, 99)
        else:
            metrics["ttfb_ms"] = None
            metrics["ttfb_ms_p50"] = None
            metrics["ttfb_ms_p95"] = None
            metrics["ttfb_ms_p99"] = None
        metrics["rtf"] = float(np.mean(rtf_values)) if len(rtf_values) > 0 else None
        metrics["rtf_std"] = float(np.std(rtf_values)) if len(rtf_values) > 0 else None
        metrics["rejected_reason"] = None

    # Non-registered fields (tables.json only, never metrics.json — REQ-12's constraint).
    metrics["speaker_sim_radiohost_control"] = (
        float(np.mean(sim_control_values)) if len(sim_control_values) > 0 else None
    )
    metrics["duration_ratio_median"] = float(np.median(dur_values)) if len(dur_values) > 0 else None
    metrics["duration_explosion_fraction"] = (
        sum(1 for v in dur_values if v > 5.0) / len(dur_values) if len(dur_values) > 0 else None
    )
    metrics["wer_mean"] = float(np.mean(wer_values)) if len(wer_values) > 0 else None
    metrics["n_clips"] = float(n_clips)
    metrics["n_successful"] = float(n_successful)
    metrics["success_rate"] = success_rate

    return metrics


def variant_id_for(system: str, variant: str, condition: str, prompt_set: str) -> str:
    return f"{system}_{variant}_{condition}_{prompt_set}"


def build_metrics_json(
    records: list[dict[str, object]],
    variant_keys: list[VariantKey],
) -> dict[str, object]:
    """Explicit multi-variant metrics.json — only the 3 registered keys per variant (REQ-4)."""
    variants: list[dict[str, object]] = []
    for system, variant, condition, prompt_set in variant_keys:
        all_metrics = compute_variant_metrics(records, system, variant, condition, prompt_set)
        if len(all_metrics) == 0:
            continue
        registered = {k: v for k, v in all_metrics.items() if k in REGISTERED_METRIC_KEYS}
        vid = variant_id_for(system, variant, condition, prompt_set)
        variants.append(
            {
                "variant_id": vid,
                "label": f"{system} / {variant} / {condition} / {prompt_set}",
                "dimensions": {
                    "system": system,
                    "acceleration_variant": variant,
                    "condition": condition,
                    "prompt_set": prompt_set,
                },
                "metrics": registered,
            }
        )
    return {"variants": variants}


def build_tables_json(
    records: list[dict[str, object]],
    variant_keys: list[VariantKey],
    timing_by_variant: dict[str, dict[str, object]],
    latency_breakdown: dict[str, object],
    environment: dict[str, object],
    gate_failures: dict[str, object],
    cost_tracking: dict[str, object],
    notes: list[str],
) -> dict[str, object]:
    """tables.json: full metrics INCLUDING both speaker_sim columns side by side (REQ-12)."""
    from tasks.t0021_zero_shot_latency_reduction.code.constants import VM_HOURLY_COST_USD

    rows: list[dict[str, object]] = []
    t0018_lookup = {
        ("cosyvoice2", "fillers"): (
            T0018_COSYVOICE2_SPEAKER_SIM_FILLERS,
            T0018_COSYVOICE2_TTFB_MS_FILLERS,
        ),
        ("cosyvoice2", "val96"): (
            T0018_COSYVOICE2_SPEAKER_SIM_VAL96,
            T0018_COSYVOICE2_TTFB_MS_VAL96,
        ),
        ("chatterbox", "fillers"): (
            T0018_CHATTERBOX_SPEAKER_SIM_FILLERS,
            T0018_CHATTERBOX_TTFB_MS_FILLERS,
        ),
        ("chatterbox", "val96"): (
            T0018_CHATTERBOX_SPEAKER_SIM_VAL96,
            T0018_CHATTERBOX_TTFB_MS_VAL96,
        ),
    }

    for system, variant, condition, prompt_set in variant_keys:
        all_metrics = compute_variant_metrics(records, system, variant, condition, prompt_set)
        if len(all_metrics) == 0:
            continue
        vid = variant_id_for(system, variant, condition, prompt_set)
        timing = timing_by_variant.get(vid, {})
        n_clips = int(all_metrics.get("n_clips") or 0)

        efficiency_time_per_item_s: float | None = None
        efficiency_cost_per_item_usd: float | None = None
        wall_clock_s = timing.get("wall_clock_s")
        if wall_clock_s is not None and n_clips > 0:
            efficiency_time_per_item_s = float(wall_clock_s) / n_clips
            efficiency_cost_per_item_usd = (
                VM_HOURLY_COST_USD * (float(wall_clock_s) / 3600.0) / n_clips
            )

        t0018_sim, t0018_ttfb = t0018_lookup.get((system, prompt_set), (None, None))

        row: dict[str, object] = {
            "variant_id": vid,
            "system": system,
            "acceleration_variant": variant,
            "condition": condition,
            "prompt_set": prompt_set,
            **all_metrics,
            "efficiency_inference_time_per_item_seconds": efficiency_time_per_item_s,
            "efficiency_inference_cost_per_item_usd": efficiency_cost_per_item_usd,
            "gate_failure_count": (
                gate_failures.get(vid, {}).get("failure_count")
                if isinstance(gate_failures.get(vid), dict)
                else None
            ),
            "t0018_baseline_speaker_sim_wrong_voice_reference": t0018_sim,
            "t0018_baseline_ttfb_ms_wrong_voice_reference": t0018_ttfb,
            "meets_ttfb_target_300ms": (
                all_metrics.get("ttfb_ms") is not None and all_metrics["ttfb_ms"] <= TTFB_TARGET_MS
            ),
            "latency_breakdown": latency_breakdown.get(f"{system}_{variant}"),
            "environment": environment.get(f"{system}_{variant}"),
        }
        rows.append(row)

    return {"rows": rows, "cost_tracking": cost_tracking, "notes": notes}


def plot_latency_breakdown_stacked(latency_breakdown: dict[str, object], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(latency_breakdown.keys())
    if len(keys) == 0:
        return
    stage_fields = [
        "ref_encoding_ms",
        "ref_conditioning_ms",
        "text_frontend_ms",
        "lm_prefill_decode_ms",
        "lm_decode_ms",
        "flow_matching_vocoder_ms",
        "vocoder_ms",
    ]
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(10, len(keys) * 0.9), 7))
    bottoms = np.zeros(len(keys))
    for i, field in enumerate(stage_fields):
        vals = np.array(
            [
                float(latency_breakdown[k].get(field, {}).get("mean_ms", 0.0) or 0.0)  # type: ignore[union-attr]
                if isinstance(latency_breakdown[k].get(field), dict)
                else 0.0
                for k in keys
            ]
        )
        if vals.sum() == 0:
            continue
        ax.bar(keys, vals, bottom=bottoms, label=field, color=colors(i % 10))
        bottoms += vals
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Stage First-Chunk Latency Breakdown by Variant")
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ttfb_vs_speaker_sim_variants(
    records: list[dict[str, object]], variant_keys: list[VariantKey], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = sorted({s for s, _, _, _ in variant_keys})
    colors = plt.get_cmap("tab10")
    color_by_system = {s: colors(i) for i, s in enumerate(systems)}

    fig, ax = plt.subplots(figsize=(11, 7))
    seen = sorted({(s, v, c) for s, v, c, _ in variant_keys})
    for s, v, c in seen:
        m = compute_variant_metrics(records, s, v, c, "fillers")
        if m.get("ttfb_ms") is None or m.get("speaker_sim") is None:
            continue
        ax.scatter(
            m["ttfb_ms"],
            m["speaker_sim"],
            color=color_by_system.get(s, "gray"),
            marker="o",
            s=110,
            edgecolor="black",
            linewidth=0.5,
            label=f"{s}/{v}/{c}",
        )

    t0018_points = [
        (
            "cosyvoice2 (t0018 baseline, wrong-voice ref, continuity only)",
            T0018_COSYVOICE2_TTFB_MS_FILLERS,
            T0018_COSYVOICE2_SPEAKER_SIM_FILLERS,
        ),
        (
            "chatterbox (t0018 baseline, wrong-voice ref, continuity only)",
            T0018_CHATTERBOX_TTFB_MS_FILLERS,
            T0018_CHATTERBOX_SPEAKER_SIM_FILLERS,
        ),
    ]
    for label, ttfb, sim in t0018_points:
        ax.scatter(ttfb, sim, color="black", marker="x", s=140, label=label)

    ax.axvline(
        x=TTFB_TARGET_MS, color="red", linestyle="--", linewidth=1.5, label="TTFB target: 300ms"
    )
    ax.set_xlabel("TTFB p50 (ms)")
    ax.set_ylabel("speaker_sim (fillers, corrected val96 centroid)")
    ax.set_title("TTFB vs Speaker Similarity, one point per variant")
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ttfb_p50_p95_by_variant(
    records: list[dict[str, object]], variant_keys: list[VariantKey], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seen = sorted({(s, v, c) for s, v, c, _ in variant_keys})
    labels = [f"{s}/{v}/{c}" for s, v, c in seen]
    p50s, p95s = [], []
    for s, v, c in seen:
        m = compute_variant_metrics(records, s, v, c, "fillers")
        p50s.append(m.get("ttfb_ms_p50") or 0.0)
        p95s.append(m.get("ttfb_ms_p95") or 0.0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 7))
    ax.bar(x - width / 2, p50s, width, label="p50", color="steelblue")
    ax.bar(x + width / 2, p95s, width, label="p95", color="darkorange")
    ax.axhline(y=TTFB_TARGET_MS, color="red", linestyle="--", linewidth=1.5, label="300ms target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TTFB (ms)")
    ax.set_title("TTFB p50/p95 by Variant (fillers)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
