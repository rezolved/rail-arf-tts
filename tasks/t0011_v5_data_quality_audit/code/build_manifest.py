"""Flag clips and build clean manifest for t0011_v5_data_quality_audit.

Reads per_clip_stats.jsonl, applies all flag thresholds, writes:
  - data/flagged_clips.txt         (clipped paths with reasons, tab-separated)
  - data/train_list_v5_clean.txt   (original manifest minus flagged entries)
  - data/distribution_stats.json   (percentile stats per metric)
  - data/flag_counts.json          (counts per flag category)
  - results/results_detailed.md    (human-readable report with flag table + dist stats)

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0011_v5_data_quality_audit -- \
        uv run python -u tasks/t0011_v5_data_quality_audit/code/build_manifest.py
"""

import json
import math
import sys

from tasks.t0011_v5_data_quality_audit.code.constants import (
    DURATION_MAX_S,
    DURATION_MIN_S,
    EXPECTED_CHANNELS,
    EXPECTED_SAMPLE_RATE,
    EXPECTED_TRAIN_CLIPS,
    FIELD_CHANNELS,
    FIELD_DURATION_S,
    FIELD_ERROR,
    FIELD_LUFS,
    FIELD_OOV_FRACTION,
    FIELD_PEAK_DBFS,
    FIELD_SAMPLE_RATE,
    FIELD_SILENCE_FRACTION,
    FIELD_WAV_PATH,
    LUFS_MAX,
    LUFS_MIN,
    MIN_CLEAN_CLIPS,
    OOV_FRACTION_MAX,
    PEAK_DBFS_MAX,
    SILENCE_FRACTION_MAX,
)
from tasks.t0011_v5_data_quality_audit.code.paths import (
    CLEAN_MANIFEST_TXT,
    DATA_DIR,
    DISTRIBUTION_STATS_JSON,
    FLAG_COUNTS_JSON,
    FLAGGED_CLIPS_TXT,
    PER_CLIP_STATS_JSONL,
    RESULTS_DIR,
    V5_TRAIN_LIST,
)


def _get_flag_reasons(record: dict[str, object]) -> list[str]:
    """Return a list of flag reason strings for a clip record."""
    reasons: list[str] = []

    if record.get(FIELD_ERROR) is not None:
        reasons.append("error")
        return reasons  # Skip metric checks for unreadable files

    peak_dbfs = record.get(FIELD_PEAK_DBFS)
    if isinstance(peak_dbfs, float | int) and peak_dbfs > PEAK_DBFS_MAX:
        reasons.append("clipping")

    silence_frac = record.get(FIELD_SILENCE_FRACTION)
    if isinstance(silence_frac, float | int) and silence_frac > SILENCE_FRACTION_MAX:
        reasons.append("silence")

    duration_s = record.get(FIELD_DURATION_S)
    if isinstance(duration_s, float | int):
        if duration_s < DURATION_MIN_S:
            reasons.append("duration_low")
        if duration_s > DURATION_MAX_S:
            reasons.append("duration_high")

    sample_rate = record.get(FIELD_SAMPLE_RATE)
    if isinstance(sample_rate, int) and sample_rate != EXPECTED_SAMPLE_RATE:
        reasons.append("samplerate")

    channels = record.get(FIELD_CHANNELS)
    if isinstance(channels, int) and channels > EXPECTED_CHANNELS:
        reasons.append("channels")

    lufs = record.get(FIELD_LUFS)
    if isinstance(lufs, float | int):
        if lufs < LUFS_MIN:
            reasons.append("lufs_low")
        if lufs > LUFS_MAX:
            reasons.append("lufs_high")

    oov_fraction = record.get(FIELD_OOV_FRACTION)
    if isinstance(oov_fraction, float | int) and oov_fraction > OOV_FRACTION_MAX:
        reasons.append("oov")

    return reasons


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if len(values) == 0:
        return float("nan")
    n = len(values)
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _compute_metric_stats(values: list[float]) -> dict[str, object]:
    """Compute descriptive stats for a list of float values."""
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / n
    return {
        "count": n,
        "mean": round(mean, 4),
        "std": round(math.sqrt(variance), 4),
        "min": round(sorted_vals[0], 4),
        "p5": round(_percentile(sorted_vals, 5), 4),
        "p50": round(_percentile(sorted_vals, 50), 4),
        "p95": round(_percentile(sorted_vals, 95), 4),
        "p99": round(_percentile(sorted_vals, 99), 4),
        "max": round(sorted_vals[-1], 4),
    }


def main() -> None:
    print("=== build_manifest.py: flag clips and build clean manifest ===")

    if not PER_CLIP_STATS_JSONL.exists():
        print(
            f"ERROR: {PER_CLIP_STATS_JSONL} not found — "
            "run audit_audio.py and audit_transcripts.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load all records
    records: list[dict[str, object]] = []
    with PER_CLIP_STATS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            records.append(json.loads(line))

    print(f"  Loaded {len(records)} clip records from JSONL")

    # Build flagged set
    category_counts: dict[str, int] = {
        "clipping": 0,
        "silence": 0,
        "duration_low": 0,
        "duration_high": 0,
        "samplerate": 0,
        "channels": 0,
        "lufs_low": 0,
        "lufs_high": 0,
        "oov": 0,
        "error": 0,
    }
    flagged_paths: set[str] = set()
    flagged_reasons: dict[str, list[str]] = {}

    for record in records:
        reasons = _get_flag_reasons(record=record)
        if len(reasons) > 0:
            wav_path = str(record.get(FIELD_WAV_PATH, ""))
            flagged_paths.add(wav_path)
            flagged_reasons[wav_path] = reasons
            for reason in reasons:
                if reason in category_counts:
                    category_counts[reason] += 1

    # Print flag table
    print("\n  Flag counts:")
    for cat, cnt in sorted(category_counts.items()):
        print(f"    {cat:20s}: {cnt}")

    # Write flagged_clips.txt
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FLAGGED_CLIPS_TXT.open("w", encoding="utf-8") as f:
        for wav_path in sorted(flagged_paths):
            reasons_str = ",".join(flagged_reasons[wav_path])
            f.write(f"{wav_path}\t{reasons_str}\n")

    print(f"\n  Wrote: {FLAGGED_CLIPS_TXT} ({len(flagged_paths)} flagged clips)")

    # Write clean manifest
    train_entries: list[str] = []
    with V5_TRAIN_LIST.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if len(line) > 0:
                train_entries.append(raw.rstrip("\n"))

    clean_entries = [line for line in train_entries if line.split("|")[0] not in flagged_paths]

    with CLEAN_MANIFEST_TXT.open("w", encoding="utf-8") as f:
        for line in clean_entries:
            f.write(line + "\n")

    print(f"  Wrote: {CLEAN_MANIFEST_TXT} ({len(clean_entries)} clean clips)")
    print(
        f"  Consistency check: {len(flagged_paths)} flagged + {len(clean_entries)} clean = "
        f"{len(flagged_paths) + len(clean_entries)} (expected {EXPECTED_TRAIN_CLIPS})"
    )

    # Reject if too few clean clips remain
    if len(clean_entries) < MIN_CLEAN_CLIPS:
        print(
            f"WARNING: Only {len(clean_entries)} clips remain after flagging "
            f"(threshold: {MIN_CLEAN_CLIPS}). Thresholds may be miscalibrated.",
            file=sys.stderr,
        )
        print("Creating intervention file...", file=sys.stderr)
        intervention_path = CLEAN_MANIFEST_TXT.parent.parent / "intervention"
        intervention_path.mkdir(parents=True, exist_ok=True)
        (intervention_path / "too_few_clean_clips.md").write_text(
            f"# Intervention: Too Few Clean Clips\n\n"
            f"Only {len(clean_entries)} clips remain after quality flagging "
            f"(threshold: {MIN_CLEAN_CLIPS}).\n\n"
            f"Flag counts per category:\n"
            + "".join(f"- {k}: {v}\n" for k, v in sorted(category_counts.items()))
            + "\nPlease review thresholds in `code/constants.py`.",
            encoding="utf-8",
        )
        sys.exit(1)

    # Compute distribution stats
    metric_fields: list[str] = [
        FIELD_PEAK_DBFS,
        FIELD_LUFS,
        FIELD_DURATION_S,
        FIELD_SILENCE_FRACTION,
    ]
    dist_stats: dict[str, object] = {}
    for field in metric_fields:
        values: list[float] = []
        for rec in records:
            val = rec.get(field)
            if isinstance(val, float | int):
                values.append(float(val))
        dist_stats[field] = _compute_metric_stats(values=values)

    with DISTRIBUTION_STATS_JSON.open("w", encoding="utf-8") as f:
        json.dump(dist_stats, f, indent=2)
    print(f"  Wrote: {DISTRIBUTION_STATS_JSON}")

    # Write flag counts JSON
    flag_counts = {
        "total_clips": EXPECTED_TRAIN_CLIPS,
        "flagged_total": len(flagged_paths),
        "flagged_by_category": dict(sorted(category_counts.items())),
        "clean_manifest_clips": len(clean_entries),
    }
    with FLAG_COUNTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(flag_counts, f, indent=2)
    print(f"  Wrote: {FLAG_COUNTS_JSON}")

    # Write human-readable results_detailed.md
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_md_path = RESULTS_DIR / "results_detailed.md"
    with results_md_path.open("w", encoding="utf-8") as md:
        md.write("# t0011 v5 Data Quality Audit — Results\n\n")
        md.write(f"Total clips in v5 train manifest: **{EXPECTED_TRAIN_CLIPS}**\n\n")
        md.write(f"Clean manifest clips: **{len(clean_entries)}**\n\n")
        md.write(f"Flagged clips: **{len(flagged_paths)}**\n\n")
        md.write("## Flag Counts by Category\n\n")
        md.write("| Category | Count |\n")
        md.write("| --- | --- |\n")
        for cat, cnt in sorted(category_counts.items()):
            md.write(f"| {cat} | {cnt} |\n")
        md.write(f"| **total flagged** | **{len(flagged_paths)}** |\n\n")
        md.write("## Distribution Statistics\n\n")
        for field, stats in dist_stats.items():
            assert isinstance(stats, dict)
            md.write(f"### {field}\n\n")
            md.write("| Stat | Value |\n")
            md.write("| --- | --- |\n")
            for k, v in stats.items():
                md.write(f"| {k} | {v} |\n")
            md.write("\n")
        md.write("## Thresholds Used\n\n")
        md.write("| Metric | Threshold | Direction |\n")
        md.write("| --- | --- | --- |\n")
        md.write(f"| peak_dbfs | {PEAK_DBFS_MAX} | > flags as clipping |\n")
        md.write(f"| silence_fraction | {SILENCE_FRACTION_MAX} | > flags as silence |\n")
        md.write(f"| duration_s | {DURATION_MIN_S} | < flags as duration_low |\n")
        md.write(f"| duration_s | {DURATION_MAX_S} | > flags as duration_high |\n")
        md.write(f"| lufs | {LUFS_MIN} | < flags as lufs_low |\n")
        md.write(f"| lufs | {LUFS_MAX} | > flags as lufs_high |\n")
        md.write(f"| oov_fraction | {OOV_FRACTION_MAX} | > flags as oov |\n")
    print(f"  Wrote: {results_md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
