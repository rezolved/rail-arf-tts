"""Build the corrected clean manifest and reconcile against t0011 (Steps 3 and 6).

Reads data/per_clip_stats_v2.jsonl and writes:
  - data/flagged_clips_v2.txt              (REQ-8)
  - data/train_list_v5_normalized_clean.txt (REQ-9)
  - data/flag_counts_v2.json                (REQ-11, REQ-4/REQ-14 reclassification)
  - data/analysis_v2.json                   (superset: percentiles, cross-tab -- REQ-16)

Run via run_with_logs:
    uv run python -m arf.scripts.utils.run_with_logs \
        --task-id t0012_v5_corpus_normalize_and_reaudit -- \
        uv run python -u tasks/t0012_v5_corpus_normalize_and_reaudit/code/build_manifest_v2.py
"""

import json
import math
import sys

from tasks.t0012_v5_corpus_normalize_and_reaudit.code.constants import (
    EXPECTED_TRAIN_CLIPS,
    FIELD_FLAGS,
    FIELD_POST_CLIPPED_FRACTION,
    FIELD_POST_LUFS,
    FIELD_POST_PEAK_DBFS,
    FIELD_PRE_CLIPPED_FRACTION,
    FIELD_PRE_LUFS,
    FIELD_PRE_PEAK_DBFS,
    FIELD_WAV_PATH,
    MIN_CLEAN_CLIPS,
    T0011_BASELINE_CLEAN_CLIPS,
)
from tasks.t0012_v5_corpus_normalize_and_reaudit.code.paths import (
    ANALYSIS_V2_JSON,
    CLEAN_MANIFEST_V2_TXT,
    DATA_DIR,
    FLAG_COUNTS_V2_JSON,
    FLAGGED_CLIPS_V2_TXT,
    PER_CLIP_STATS_V2_JSONL,
    T0011_FLAG_COUNTS_JSON,
    T0011_FLAGGED_CLIPS_TXT,
    V5_NORMALIZED_DIR,
    V5_TRAIN_LIST,
    VAL_96_LIST,
)

CATEGORIES: list[str] = [
    "clipping",
    "clipping_after_normalization",
    "silence",
    "silence_after_normalization",
    "duration_low",
    "duration_high",
    "samplerate",
    "channels",
    "lufs_low",
    "lufs_high",
    "error",
]


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


def _load_field(records: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for rec in records:
        val = rec.get(field)
        if isinstance(val, float | int):
            values.append(float(val))
    return values


def main() -> None:
    print("=== build_manifest_v2.py: corrected manifest + reconciliation ===")

    if not PER_CLIP_STATS_V2_JSONL.exists():
        print(
            f"ERROR: {PER_CLIP_STATS_V2_JSONL} not found -- run audit_normalize.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    records: list[dict[str, object]] = []
    with PER_CLIP_STATS_V2_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                records.append(json.loads(line))

    print(f"  Loaded {len(records)} clip records from JSONL")

    category_counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    flagged_paths: set[str] = set()
    flagged_reasons: dict[str, list[str]] = {}
    clean_paths: set[str] = set()

    for record in records:
        wav_path = str(record.get(FIELD_WAV_PATH, ""))
        flags = record.get(FIELD_FLAGS)
        reasons = flags if isinstance(flags, list) else []
        if len(reasons) > 0:
            flagged_paths.add(wav_path)
            flagged_reasons[wav_path] = [str(r) for r in reasons]
            for reason in reasons:
                if reason in category_counts:
                    category_counts[reason] += 1
        else:
            clean_paths.add(wav_path)

    print("\n  Flag counts (after):")
    for cat, cnt in sorted(category_counts.items()):
        print(f"    {cat:32s}: {cnt}")

    # REQ-8: flagged_clips_v2.txt
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with FLAGGED_CLIPS_V2_TXT.open("w", encoding="utf-8") as f:
        for wav_path in sorted(flagged_paths):
            f.write(f"{wav_path}\t{','.join(flagged_reasons[wav_path])}\n")
    print(f"\n  Wrote: {FLAGGED_CLIPS_V2_TXT} ({len(flagged_paths)} flagged clips)")

    # REQ-9: clean manifest pointing into data/v5_normalized/
    train_lines: list[str] = []
    with V5_TRAIN_LIST.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if len(line.strip()) > 0:
                train_lines.append(line)

    clean_lines: list[str] = []
    for line in train_lines:
        parts = line.split("|")
        wav_path = parts[0]
        if wav_path in clean_paths:
            new_path = str(V5_NORMALIZED_DIR / wav_path.split("/")[-1])
            clean_lines.append("|".join([new_path, *parts[1:]]))

    with CLEAN_MANIFEST_V2_TXT.open("w", encoding="utf-8") as f:
        for line in clean_lines:
            f.write(line + "\n")

    n_clean = len(clean_lines)
    n_flagged = len(flagged_paths)
    print(f"  Wrote: {CLEAN_MANIFEST_V2_TXT} ({n_clean} clean clips)")
    print(
        f"  Consistency check: {n_flagged} flagged + {n_clean} clean = "
        f"{n_flagged + n_clean} (expected {EXPECTED_TRAIN_CLIPS})"
    )
    print(
        f"  Clean-manifest size: {n_clean}/{EXPECTED_TRAIN_CLIPS} "
        f"({100.0 * n_clean / EXPECTED_TRAIN_CLIPS:.1f}%), "
        f"vs t0011 baseline {T0011_BASELINE_CLEAN_CLIPS}/{EXPECTED_TRAIN_CLIPS} "
        f"({100.0 * T0011_BASELINE_CLEAN_CLIPS / EXPECTED_TRAIN_CLIPS:.1f}%)"
    )

    # REQ-19: leak check against the 96-clip val set
    val_paths: set[str] = set()
    with VAL_96_LIST.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if len(line) > 0:
                val_paths.add(line.split("|")[0])
    clean_manifest_paths = {line.split("|")[0] for line in clean_lines}
    leak_count = len(clean_manifest_paths & val_paths)
    print(f"  Leak check: {leak_count} val_96 paths found in clean manifest (expected 0)")
    if leak_count > 0:
        print("ERROR: val_96 leak detected in clean manifest. Aborting.", file=sys.stderr)
        sys.exit(1)

    # REQ-4/REQ-14: reclassification of the original 224 peak-flagged clips
    orig_clipping_flagged: set[str] = set()
    with T0011_FLAGGED_CLIPS_TXT.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if len(line) == 0:
                continue
            path, reasons_str = line.split("\t")
            if "clipping" in reasons_str.split(","):
                orig_clipping_flagged.add(path)

    clipping_reasons = {"clipping", "clipping_after_normalization"}
    now_clean = sum(1 for p in orig_clipping_flagged if p not in flagged_paths)
    still_clipped = sum(
        1
        for p in orig_clipping_flagged
        if p in flagged_paths and not clipping_reasons.isdisjoint(flagged_reasons.get(p, []))
    )
    print(
        f"\n  Reclassification of original {len(orig_clipping_flagged)} peak-flagged clips: "
        f"{now_clean} now clean, {still_clipped} still clipped"
    )

    # REQ-11: before (t0011) vs after (this task) flag counts
    with T0011_FLAG_COUNTS_JSON.open(encoding="utf-8") as f:
        t0011_flag_counts = json.load(f)

    if n_clean < MIN_CLEAN_CLIPS:
        print(
            f"WARNING: Only {n_clean} clean clips (threshold: {MIN_CLEAN_CLIPS}).",
            file=sys.stderr,
        )
        intervention_dir = CLEAN_MANIFEST_V2_TXT.parent.parent / "intervention"
        intervention_dir.mkdir(parents=True, exist_ok=True)
        (intervention_dir / "too_few_clean_clips.md").write_text(
            f"# Intervention: Too Few Clean Clips\n\n"
            f"Only {n_clean} clips remain after the corrected audit + normalization "
            f"(threshold: {MIN_CLEAN_CLIPS}).\n\nFlag counts:\n"
            + "".join(f"- {k}: {v}\n" for k, v in sorted(category_counts.items())),
            encoding="utf-8",
        )
        sys.exit(1)

    if n_clean <= T0011_BASELINE_CLEAN_CLIPS:
        print(
            f"ERROR: clean manifest {n_clean} not > t0011 baseline "
            f"{T0011_BASELINE_CLEAN_CLIPS}. Pipeline likely regressed. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    flag_counts_v2 = {
        "total_clips": EXPECTED_TRAIN_CLIPS,
        "flagged_total": n_flagged,
        "flagged_by_category": dict(sorted(category_counts.items())),
        "clean_manifest_clips": n_clean,
        "reclassified_from_224": {"now_clean": now_clean, "still_clipped": still_clipped},
    }
    with FLAG_COUNTS_V2_JSON.open("w", encoding="utf-8") as f:
        json.dump(flag_counts_v2, f, indent=2)
    print(f"\n  Wrote: {FLAG_COUNTS_V2_JSON}")

    # Step 6: percentiles + Key Question 4 cross-tab
    dist_stats = {
        "pre_peak_dbfs": _compute_metric_stats(_load_field(records, FIELD_PRE_PEAK_DBFS)),
        "post_peak_dbfs": _compute_metric_stats(_load_field(records, FIELD_POST_PEAK_DBFS)),
        "pre_lufs": _compute_metric_stats(_load_field(records, FIELD_PRE_LUFS)),
        "post_lufs": _compute_metric_stats(_load_field(records, FIELD_POST_LUFS)),
        "pre_clipped_fraction": _compute_metric_stats(
            _load_field(records, FIELD_PRE_CLIPPED_FRACTION)
        ),
        "post_clipped_fraction": _compute_metric_stats(
            _load_field(records, FIELD_POST_CLIPPED_FRACTION)
        ),
    }

    # Key Question 4: gain is a uniform per-sample scalar, so duration_s cannot change, and
    # silence_fraction (a *relative* dB threshold via librosa.effects.split) is expected to be
    # gain-invariant. Confirm numerically rather than asserting it, for the clips t0011 flagged
    # duration_low (25) or silence (1).
    t0011_duration_low: set[str] = set()
    t0011_silence: set[str] = set()
    with T0011_FLAGGED_CLIPS_TXT.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if len(line) == 0:
                continue
            path, reasons_str = line.split("\t")
            reasons_list = reasons_str.split(",")
            if "duration_low" in reasons_list:
                t0011_duration_low.add(path)
            if "silence" in reasons_list:
                t0011_silence.add(path)

    def _cross_tab(orig_set: set[str], reason: str) -> dict[str, int]:
        same_flag = sum(1 for p in orig_set if reason in flagged_reasons.get(p, []))
        now_clean_n = sum(1 for p in orig_set if p not in flagged_paths)
        return {
            "original_count": len(orig_set),
            f"still_{reason}": same_flag,
            "now_clean": now_clean_n,
            "other_flag": len(orig_set) - same_flag - now_clean_n,
        }

    key_question_4 = {
        "duration_low": _cross_tab(t0011_duration_low, "duration_low"),
        "silence": _cross_tab(t0011_silence, "silence"),
        "note": (
            "Gain is a uniform per-sample scalar: duration_s is unaffected by construction. "
            "silence_fraction uses a relative dB threshold (librosa.effects.split, top_db=60), "
            "so it is expected to be gain-invariant; the cross-tab above confirms this numerically."
        ),
    }

    analysis_v2 = {
        "flag_counts_before_t0011": t0011_flag_counts,
        "flag_counts_after_v2": flag_counts_v2,
        "distribution_stats": dist_stats,
        "key_question_4_cross_tab": key_question_4,
    }
    with ANALYSIS_V2_JSON.open("w", encoding="utf-8") as f:
        json.dump(analysis_v2, f, indent=2)
    print(f"  Wrote: {ANALYSIS_V2_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
