"""Report writer: merge per-clip results, produce metrics.json/tables.json and the four charts.

Extends `tasks.t0008_tts_eval_harness_baselines.code.report`'s `compute_variant_metrics` pattern to
a third dimension (`condition`), per plan Step 13. Cloning systems have a `condition`
(`ref_single`/`ref_concat`); the two baselines (`elevenlabs_david`, `kokoro_v3_bundle`) do not
(`condition=None`), formatted as `<system>_<prompt_set>` to match t0008's own convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.report import REGISTERED_METRIC_KEYS, _percentile
from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    ELEVENLABS_SPEAKER_SIM_FILLERS,
    ELEVENLABS_SPEAKER_SIM_VAL96,
    KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS,
    KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96,
)

logger = logging.getLogger(__name__)


def compute_variant_metrics_zeroshot(
    records: list[dict[str, object]],
    system: str,
    condition: str | None,
    prompt_set: str,
) -> dict[str, float | None]:
    """Compute aggregate metrics for one (system, condition, prompt_set) variant."""
    subset = [
        r
        for r in records
        if r["system"] == system
        and r.get("condition") == condition
        and r["prompt_set"] == prompt_set
    ]
    if len(subset) == 0:
        return {}

    ttfb_values = [float(r["ttfb_ms"]) for r in subset if r.get("ttfb_ms") is not None]  # type: ignore[arg-type]
    sim_values = [float(r["speaker_sim"]) for r in subset if r.get("speaker_sim") is not None]  # type: ignore[arg-type]
    rtf_values = [float(r["rtf"]) for r in subset if r.get("rtf") is not None]  # type: ignore[arg-type]
    dur_values = [float(r["duration_ratio"]) for r in subset if r.get("duration_ratio") is not None]  # type: ignore[arg-type]
    wer_values = [float(r["wer"]) for r in subset if r.get("wer") is not None]  # type: ignore[arg-type]

    n_clips = len(subset)
    n_successful = sum(
        1 for r in subset if r.get("ttfb_ms") is not None or r.get("speaker_sim") is not None
    )
    success_rate = n_successful / n_clips if n_clips > 0 else 0.0

    metrics: dict[str, float | None] = {}

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
        if len(sim_values) > 0:
            metrics["speaker_sim"] = float(np.mean(sim_values))
            metrics["speaker_sim_std"] = float(np.std(sim_values))
        else:
            metrics["speaker_sim"] = None
            metrics["speaker_sim_std"] = None

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

        if len(rtf_values) > 0:
            metrics["rtf"] = float(np.mean(rtf_values))
            metrics["rtf_std"] = float(np.std(rtf_values))
        else:
            metrics["rtf"] = None
            metrics["rtf_std"] = None
        metrics["rejected_reason"] = None

    if len(dur_values) > 0:
        metrics["duration_ratio_median"] = float(np.median(dur_values))
        metrics["duration_explosion_fraction"] = sum(1 for v in dur_values if v > 5.0) / len(
            dur_values
        )
    else:
        metrics["duration_ratio_median"] = None
        metrics["duration_explosion_fraction"] = None

    metrics["wer_mean"] = float(np.mean(wer_values)) if len(wer_values) > 0 else None
    metrics["n_clips"] = float(n_clips)
    metrics["n_successful"] = float(n_successful)
    metrics["success_rate"] = success_rate

    return metrics


def variant_id_for(system: str, condition: str | None, prompt_set: str) -> str:
    if condition is None:
        return f"{system}_{prompt_set}"
    return f"{system}_{condition}_{prompt_set}"


def build_metrics_json(
    records: list[dict[str, object]],
    variant_keys: list[tuple[str, str | None, str]],
    timing_by_variant: dict[str, dict[str, object]],
    environment: dict[str, object],
) -> dict[str, object]:
    """Build explicit-variant-format metrics.json (only the 3 registered keys per variant)."""
    variants: list[dict[str, object]] = []
    for system, condition, prompt_set in variant_keys:
        all_metrics = compute_variant_metrics_zeroshot(records, system, condition, prompt_set)
        if len(all_metrics) == 0:
            continue
        registered = {k: v for k, v in all_metrics.items() if k in REGISTERED_METRIC_KEYS}
        vid = variant_id_for(system, condition, prompt_set)
        label = (
            f"{system} / {condition} / {prompt_set}" if condition else f"{system} / {prompt_set}"
        )
        variants.append(
            {
                "variant_id": vid,
                "label": label,
                "dimensions": {"system": system, "condition": condition, "prompt_set": prompt_set},
                "metrics": registered,
            }
        )
    return {"variants": variants}


def build_tables_json(
    records: list[dict[str, object]],
    variant_keys: list[tuple[str, str | None, str]],
    timing_by_variant: dict[str, dict[str, object]],
    environment: dict[str, object],
    gate_failures: dict[str, object],
    cost_tracking: dict[str, object],
    notes: list[str],
) -> dict[str, object]:
    """Build tables.json with full metrics including non-registered sanity/efficiency metrics."""
    rows: list[dict[str, object]] = []

    baseline_lookup = {
        ("elevenlabs_david", "fillers"): ELEVENLABS_SPEAKER_SIM_FILLERS,
        ("elevenlabs_david", "val96"): ELEVENLABS_SPEAKER_SIM_VAL96,
        ("kokoro_v3_bundle", "fillers"): KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS,
        ("kokoro_v3_bundle", "val96"): KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96,
    }

    for system, condition, prompt_set in variant_keys:
        all_metrics = compute_variant_metrics_zeroshot(records, system, condition, prompt_set)
        if len(all_metrics) == 0:
            continue
        vid = variant_id_for(system, condition, prompt_set)
        timing = timing_by_variant.get(vid, {})
        n_clips = int(all_metrics.get("n_clips") or 0)

        efficiency_time_per_item_s: float | None = None
        efficiency_cost_per_item_usd: float | None = None
        wall_clock_s = timing.get("wall_clock_s")
        if wall_clock_s is not None and n_clips > 0:
            efficiency_time_per_item_s = float(wall_clock_s) / n_clips
            from tasks.t0018_zero_shot_cloning_calibration.code.constants import VM_HOURLY_COST_USD

            efficiency_cost_per_item_usd = (
                VM_HOURLY_COST_USD * (float(wall_clock_s) / 3600.0) / n_clips
            )

        el_sim = baseline_lookup.get(("elevenlabs_david", prompt_set))
        delta_vs_elevenlabs = (
            float(all_metrics["speaker_sim"]) - el_sim
            if all_metrics.get("speaker_sim") is not None and el_sim is not None
            else None
        )
        v3_sim = baseline_lookup.get(("kokoro_v3_bundle", prompt_set))
        delta_vs_v3 = (
            float(all_metrics["speaker_sim"]) - v3_sim
            if all_metrics.get("speaker_sim") is not None and v3_sim is not None
            else None
        )

        row: dict[str, object] = {
            "variant_id": vid,
            "system": system,
            "condition": condition,
            "prompt_set": prompt_set,
            **all_metrics,
            "efficiency_inference_time_per_item_seconds": efficiency_time_per_item_s,
            "efficiency_inference_cost_per_item_usd": efficiency_cost_per_item_usd,
            "gate_failure_count": gate_failures.get(vid, {}).get("failure_count")
            if isinstance(gate_failures.get(vid), dict)
            else None,
            "delta_speaker_sim_vs_elevenlabs": delta_vs_elevenlabs,
            "delta_speaker_sim_vs_kokoro_v3_bundle": delta_vs_v3,
            "environment": environment.get(system),
        }
        rows.append(row)

    return {"rows": rows, "cost_tracking": cost_tracking, "notes": notes}


def plot_speaker_sim_by_system(
    records: list[dict[str, object]],
    variant_keys: list[tuple[str, str | None, str]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems_conditions = sorted({(s, c) for s, c, _ in variant_keys})
    labels = [f"{s}\n{c}" if c else s for s, c in systems_conditions]
    x = np.arange(len(systems_conditions))
    width = 0.35

    fillers_vals = []
    val96_vals = []
    for s, c in systems_conditions:
        fm = compute_variant_metrics_zeroshot(records, s, c, "fillers")
        vm = compute_variant_metrics_zeroshot(records, s, c, "val96")
        fillers_vals.append(fm.get("speaker_sim") if fm.get("speaker_sim") is not None else 0.0)
        val96_vals.append(vm.get("speaker_sim") if vm.get("speaker_sim") is not None else 0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, fillers_vals, width, label="fillers", color="steelblue")
    ax.bar(x + width / 2, val96_vals, width, label="val96", color="darkorange")

    ax.axhline(
        y=ELEVENLABS_SPEAKER_SIM_FILLERS,
        color="green",
        linestyle="--",
        linewidth=1.2,
        label=f"ElevenLabs fillers ({ELEVENLABS_SPEAKER_SIM_FILLERS})",
    )
    ax.axhline(
        y=ELEVENLABS_SPEAKER_SIM_VAL96,
        color="green",
        linestyle=":",
        linewidth=1.2,
        label=f"ElevenLabs val96 ({ELEVENLABS_SPEAKER_SIM_VAL96})",
    )
    ax.axhline(
        y=KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"kokoro_v3_bundle fillers ({KOKORO_V3_BUNDLE_SPEAKER_SIM_FILLERS})",
    )
    ax.axhline(
        y=KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96,
        color="red",
        linestyle=":",
        linewidth=1.2,
        label=f"kokoro_v3_bundle val96 ({KOKORO_V3_BUNDLE_SPEAKER_SIM_VAL96})",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("GE2E Cosine Similarity (speaker_sim)")
    ax.set_title("Speaker Similarity by System/Condition (fillers vs val96)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ttfb_vs_speaker_sim(
    records: list[dict[str, object]],
    variant_keys: list[tuple[str, str | None, str]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = sorted({s for s, _, _ in variant_keys})
    colors = plt.get_cmap("tab10")
    color_by_system = {s: colors(i) for i, s in enumerate(systems)}
    marker_by_condition = {"ref_single": "o", "ref_concat": "^", None: "s"}

    fig, ax = plt.subplots(figsize=(11, 7))
    seen_systems_conditions = sorted({(s, c) for s, c, _ in variant_keys})
    for s, c in seen_systems_conditions:
        m = compute_variant_metrics_zeroshot(records, s, c, "fillers")
        if m.get("ttfb_ms") is None or m.get("speaker_sim") is None:
            continue
        ax.scatter(
            m["ttfb_ms"],
            m["speaker_sim"],
            color=color_by_system.get(s, "gray"),
            marker=marker_by_condition.get(c, "s"),
            s=120,
            edgecolor="black",
            linewidth=0.5,
            label=f"{s}/{c}" if c else s,
        )

    ax.axvline(x=300, color="red", linestyle="--", linewidth=1.5, label="TTFB target: 300ms")
    ax.set_xlabel("TTFB p50 (ms) -- whole-utterance latency labelled per-system in tables.json")
    ax.set_ylabel("speaker_sim (fillers)")
    ax.set_title("TTFB vs Speaker Similarity (one point per variant, fillers prompt set)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ref_condition_effect(
    records: list[dict[str, object]], cloning_systems: list[str], out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(cloning_systems))
    width = 0.35
    single_vals = []
    concat_vals = []
    for s in cloning_systems:
        ms = compute_variant_metrics_zeroshot(records, s, "ref_single", "fillers")
        mc = compute_variant_metrics_zeroshot(records, s, "ref_concat", "fillers")
        single_vals.append(ms.get("speaker_sim") if ms.get("speaker_sim") is not None else 0.0)
        concat_vals.append(mc.get("speaker_sim") if mc.get("speaker_sim") is not None else 0.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, single_vals, width, label="ref_single (~10s)", color="steelblue")
    ax.bar(x + width / 2, concat_vals, width, label="ref_concat (~30s)", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(cloning_systems)
    ax.set_ylabel("speaker_sim (fillers)")
    ax.set_title("Reference-Audio Duration Effect (ref_single vs ref_concat)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_wer_by_system(
    records: list[dict[str, object]],
    variant_keys: list[tuple[str, str | None, str]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems_conditions = sorted({(s, c) for s, c, _ in variant_keys})
    labels = [f"{s}/{c}" if c else s for s, c in systems_conditions]
    wer_vals = []
    for s, c in systems_conditions:
        m = compute_variant_metrics_zeroshot(records, s, c, "fillers")
        wer_vals.append(m.get("wer_mean") if m.get("wer_mean") is not None else 0.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(labels)), wer_vals, color="mediumpurple")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean WER (fillers)")
    ax.set_title("Word Error Rate by System/Condition")
    ax.grid(axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
