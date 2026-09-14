"""Report writer: merge per-clip results, produce metrics.json and charts.

Reads results/per_clip_metrics.json and writes:
- results/metrics.json (explicit variant format, registered keys only)
- results/images/speaker_sim_boxplot.png
- results/images/ttfb_cdf.png
- results/images/speaker_sim_wer_scatter.png
- results/tables.json (machine-readable summary tables)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Registered project metric keys ───────────────────────────────────────────
# Only these keys are written to metrics.json; all others go in tables.json
REGISTERED_METRIC_KEYS: set[str] = {"speaker_sim", "ttfb_ms", "rtf"}


def _percentile(values: list[float], p: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, p))


def compute_variant_metrics(
    records: list[dict[str, object]],
    system: str,
    prompt_set: str,
) -> dict[str, float | None]:
    """Compute aggregate metrics for one (system, prompt_set) variant."""
    subset = [r for r in records if r["system"] == system and r["prompt_set"] == prompt_set]
    if len(subset) == 0:
        return {}

    # TTFB
    ttfb_values: list[float] = [
        float(r["ttfb_ms"])  # type: ignore[arg-type]
        for r in subset
        if r.get("ttfb_ms") is not None
    ]
    # Speaker sim
    sim_values: list[float] = [
        float(r["speaker_sim"])  # type: ignore[arg-type]
        for r in subset
        if r.get("speaker_sim") is not None
    ]
    # RTF
    rtf_values: list[float] = [
        float(r["rtf"])  # type: ignore[arg-type]
        for r in subset
        if r.get("rtf") is not None
    ]
    # Duration ratio
    dur_values: list[float] = [
        float(r["duration_ratio"])  # type: ignore[arg-type]
        for r in subset
        if r.get("duration_ratio") is not None
    ]
    # WER
    wer_values: list[float] = [
        float(r["wer"])  # type: ignore[arg-type]
        for r in subset
        if r.get("wer") is not None
    ]

    metrics: dict[str, float | None] = {}

    # Registered metrics
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

    # Non-registered sanity metrics
    if len(dur_values) > 0:
        metrics["duration_ratio_median"] = float(np.median(dur_values))
        explosion = sum(1 for v in dur_values if v > 5.0)
        metrics["duration_explosion_fraction"] = explosion / len(dur_values)
    else:
        metrics["duration_ratio_median"] = None
        metrics["duration_explosion_fraction"] = None

    if len(wer_values) > 0:
        metrics["wer_mean"] = float(np.mean(wer_values))
    else:
        metrics["wer_mean"] = None

    metrics["n_clips"] = len(subset)
    metrics["n_successful"] = sum(1 for r in subset if r.get("ttfb_ms") is not None)

    return metrics


def build_metrics_json(
    records: list[dict[str, object]],
    systems: list[str],
    prompt_sets: list[str],
) -> dict[str, object]:
    """Build explicit variant format metrics.json."""
    variants: list[dict[str, object]] = []

    for system in systems:
        for ps in prompt_sets:
            all_metrics = compute_variant_metrics(records, system, ps)
            if len(all_metrics) == 0:
                continue

            # Only registered keys go in the metrics field
            registered: dict[str, float | None] = {
                k: v for k, v in all_metrics.items() if k in REGISTERED_METRIC_KEYS
            }

            variant: dict[str, object] = {
                "variant_id": f"{system}_{ps}",
                "label": f"{system} / {ps}",
                "dimensions": {"system": system, "prompt_set": ps},
                "metrics": registered,
            }
            variants.append(variant)

    return {"variants": variants}


def build_tables_json(
    records: list[dict[str, object]],
    systems: list[str],
    prompt_sets: list[str],
    elevenlabs_metrics: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Build tables.json with full metrics including non-registered sanity metrics."""
    rows: list[dict[str, object]] = []

    for system in systems:
        for ps in prompt_sets:
            all_metrics = compute_variant_metrics(records, system, ps)
            if len(all_metrics) == 0:
                continue

            # Compute delta vs ElevenLabs on same prompt set
            el_key = f"elevenlabs_david_{ps}"
            el_sim = elevenlabs_metrics.get(el_key, {}).get("speaker_sim")
            delta_sim: float | None = None
            if el_sim is not None and all_metrics.get("speaker_sim") is not None:
                delta_sim = float(all_metrics["speaker_sim"]) - float(el_sim)  # type: ignore[arg-type]

            row: dict[str, object] = {
                "system": system,
                "prompt_set": ps,
                **all_metrics,
                "delta_speaker_sim_vs_elevenlabs": delta_sim,
            }
            rows.append(row)

    return {"rows": rows}


def plot_speaker_sim_boxplot(
    records: list[dict[str, object]],
    systems: list[str],
    out_path: Path,
) -> None:
    """Box-and-whisker per system for speaker_sim, with 0.85 success threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tasks.t0008_tts_eval_harness_baselines.code.constants import SUCCESS_SPEAKER_SIM

    data_by_system: dict[str, list[float]] = {}
    for system in systems:
        vals = [
            float(r["speaker_sim"])  # type: ignore[arg-type]
            for r in records
            if r["system"] == system and r.get("speaker_sim") is not None
        ]
        if len(vals) > 0:
            data_by_system[system] = vals

    if len(data_by_system) == 0:
        logger.warning("No speaker_sim data for boxplot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    positions = list(range(1, len(data_by_system) + 1))
    labels = list(data_by_system.keys())
    box_data = [data_by_system[s] for s in labels]

    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.axhline(
        y=SUCCESS_SPEAKER_SIM,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Target: {SUCCESS_SPEAKER_SIM}",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("GE2E Cosine Similarity")
    ax.set_title("Speaker Similarity Distribution by System")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved speaker_sim boxplot → %s", out_path)


def plot_ttfb_cdf(
    records: list[dict[str, object]],
    systems: list[str],
    out_path: Path,
) -> None:
    """ECDF of TTFB per system, with 300 ms success threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tasks.t0008_tts_eval_harness_baselines.code.constants import SUCCESS_TTFB_MS

    fig, ax = plt.subplots(figsize=(12, 6))

    for system in systems:
        vals = sorted(
            [
                float(r["ttfb_ms"])  # type: ignore[arg-type]
                for r in records
                if r["system"] == system and r.get("ttfb_ms") is not None
            ]
        )
        if len(vals) == 0:
            continue
        n = len(vals)
        cdf = [(i + 1) / n for i in range(n)]
        ax.step(vals, cdf, where="post", label=system, linewidth=1.5)

    ax.axvline(
        x=SUCCESS_TTFB_MS,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Target: {SUCCESS_TTFB_MS} ms",
    )

    ax.set_xlabel("TTFB (ms)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("TTFB CDF by System")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved TTFB CDF → %s", out_path)


def plot_speaker_sim_wer_scatter(
    records: list[dict[str, object]],
    systems: list[str],
    out_path: Path,
) -> None:
    """Scatter plot: x = WER, y = speaker_sim, colored by system."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tasks.t0008_tts_eval_harness_baselines.code.constants import SUCCESS_SPEAKER_SIM

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.get_cmap("tab10")

    for i, system in enumerate(systems):
        subset = [
            r
            for r in records
            if r["system"] == system
            and r.get("speaker_sim") is not None
            and r.get("wer") is not None
        ]
        if len(subset) == 0:
            continue
        x = [float(r["wer"]) for r in subset]  # type: ignore[arg-type]
        y = [float(r["speaker_sim"]) for r in subset]  # type: ignore[arg-type]
        ax.scatter(x, y, label=system, color=colors(i), alpha=0.6, s=20)

    ax.axhline(
        y=SUCCESS_SPEAKER_SIM,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"speaker_sim target: {SUCCESS_SPEAKER_SIM}",
    )

    ax.set_xlabel("WER")
    ax.set_ylabel("GE2E Cosine Similarity")
    ax.set_title("Speaker Similarity vs WER (per clip, all systems)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved speaker_sim vs WER scatter → %s", out_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Generate metrics.json and charts from per-clip results",
    )
    p.add_argument(
        "--per-clip-path",
        type=Path,
        default=Path("results/per_clip_metrics.json"),
        help="Path to per_clip_metrics.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )
    args = p.parse_args()

    # Load per-clip data
    assert args.per_clip_path.exists(), f"Per-clip file not found: {args.per_clip_path}"
    records: list[dict[str, object]] = json.loads(args.per_clip_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d per-clip records", len(records))

    # Determine systems and prompt sets present in the data
    systems: list[str] = sorted({str(r["system"]) for r in records})
    prompt_sets: list[str] = sorted({str(r["prompt_set"]) for r in records})
    logger.info("Systems: %s", systems)
    logger.info("Prompt sets: %s", prompt_sets)

    # Build ElevenLabs metrics for delta computation
    el_metrics: dict[str, dict[str, object]] = {}
    for ps in prompt_sets:
        key = f"elevenlabs_david_{ps}"
        el_metrics[key] = compute_variant_metrics(records, "elevenlabs_david", ps)

    # Write metrics.json
    metrics_data = build_metrics_json(records, systems, prompt_sets)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    logger.info("Saved metrics.json → %s", metrics_path)

    # Write tables.json
    tables_data = build_tables_json(records, systems, prompt_sets, el_metrics)
    tables_path = args.out_dir / "tables.json"
    tables_path.write_text(json.dumps(tables_data, indent=2), encoding="utf-8")
    logger.info("Saved tables.json → %s", tables_path)

    # Generate charts
    from tasks.t0008_tts_eval_harness_baselines.code.paths import (
        CHART_SPEAKER_SIM_BOXPLOT,
        CHART_SPEAKER_SIM_WER_SCATTER,
        CHART_TTFB_CDF,
    )

    plot_speaker_sim_boxplot(records, systems, CHART_SPEAKER_SIM_BOXPLOT)
    plot_ttfb_cdf(records, systems, CHART_TTFB_CDF)
    plot_speaker_sim_wer_scatter(records, systems, CHART_SPEAKER_SIM_WER_SCATTER)

    # Validation
    n_variants = len(metrics_data["variants"])  # type: ignore[arg-type]
    logger.info("metrics.json: %d variants", n_variants)

    success_speaker_sim = 0.85
    success_ttfb_ms = 300.0
    logger.info("=== Pass/Fail Summary ===")
    for variant in metrics_data["variants"]:  # type: ignore[union-attr]
        v_id = variant["variant_id"]
        m = variant["metrics"]
        sim = m.get("speaker_sim")
        ttfb = m.get("ttfb_ms")
        sim_pass = "PASS" if (sim is not None and float(sim) >= success_speaker_sim) else "FAIL"  # type: ignore[arg-type]
        ttfb_pass = "PASS" if (ttfb is not None and float(ttfb) <= success_ttfb_ms) else "FAIL"  # type: ignore[arg-type]
        logger.info(
            "  %-45s speaker_sim=%s [%s]  ttfb_ms=%s [%s]",
            v_id,
            f"{float(sim):.3f}" if sim is not None else "N/A",
            sim_pass,
            f"{float(ttfb):.1f}" if ttfb is not None else "N/A",
            ttfb_pass,
        )


if __name__ == "__main__":
    main()
