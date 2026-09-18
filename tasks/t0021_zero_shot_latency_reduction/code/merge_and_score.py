"""Merge per-variant per-clip files; dual-centroid speaker_sim, duration_ratio, WER (Step 10).

Adapted from `tasks.t0018_zero_shot_cloning_calibration.code.merge_and_score` (copied, not
imported). New in this task (owner correction REQ-12): every row is scored TWICE — once against
the corrected `data/references/val96_centroid.npy` (written to the row's `speaker_sim` field, the
one that becomes the registered `speaker_sim` metric), and once against
`data/references/old_wrongvoice_centroid.npy` (written to `speaker_sim_radiohost_control`, a
non-registered per-clip field that lives only in `per_clip_metrics.json`/`tables.json`, never in
`metrics.json`).
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.constants import (
    DURATION_RATIO_HIGH,
    DURATION_RATIO_LOW,
)
from tasks.t0008_tts_eval_harness_baselines.code.scoring import (
    compute_duration_ratio,
    compute_speaker_sim,
    compute_wer,
)
from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    OLD_WRONGVOICE_CENTROID_NPY,
    RESULTS_DIR,
    RESULTS_PER_CLIP_METRICS,
    VAL96_CENTROID_NPY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PER_CLIP_GLOB = "per_clip_metrics_*.json"

# Same VM-absolute -> local-worktree path rewrite as t0018's version (rsync-pull path mismatch).
AUDIO_PATH_MARKER = "results/audio_samples/"


def _localize_audio_path(audio_path_str: str) -> str:
    idx = audio_path_str.find(AUDIO_PATH_MARKER)
    if idx == -1:
        return audio_path_str
    relative = audio_path_str[idx + len(AUDIO_PATH_MARKER) :]
    return str(RESULTS_DIR / "audio_samples" / relative)


def main() -> None:
    all_records: list[dict[str, object]] = []
    for fpath_str in sorted(glob.glob(str(RESULTS_DIR / PER_CLIP_GLOB))):
        fpath = Path(fpath_str)
        if fpath.name == "per_clip_metrics.json":
            continue  # the merged output itself, if re-run
        records = json.loads(fpath.read_text(encoding="utf-8"))
        for rec in records:
            if rec.get("audio_path") is not None:
                rec["audio_path"] = _localize_audio_path(str(rec["audio_path"]))
        logger.info("Loaded %d records from %s", len(records), fpath.name)
        all_records.extend(records)

    logger.info("Total merged records: %d", len(all_records))
    n_missing = sum(
        1
        for r in all_records
        if r.get("audio_path") is not None and not Path(str(r["audio_path"])).exists()
    )
    if n_missing > 0:
        logger.warning("%d records have audio_path that does not exist on disk!", n_missing)

    val96_centroid = np.load(str(VAL96_CENTROID_NPY))
    old_centroid = np.load(str(OLD_WRONGVOICE_CENTROID_NPY))

    scoreable = [r for r in all_records if r.get("audio_path")]
    paths = [Path(str(r["audio_path"])) for r in scoreable]

    # ── Corrected val96 centroid: the registered speaker_sim ──
    sim_result = compute_speaker_sim(paths, val96_centroid)
    for rec, sim in zip(scoreable, sim_result.per_clip, strict=True):
        rec["speaker_sim"] = sim
    logger.info(
        "speaker_sim (val96 centroid): mean=%.4f, skipped=%d (%s)",
        sim_result.mean,
        sim_result.skipped_count,
        sim_result.skipped_reason,
    )

    # ── Old wrong-voice (radiohost) control centroid: non-registered per-clip field only ──
    control_result = compute_speaker_sim(paths, old_centroid)
    for rec, sim in zip(scoreable, control_result.per_clip, strict=True):
        rec["speaker_sim_radiohost_control"] = sim
    logger.info(
        "speaker_sim_radiohost_control (old centroid): mean=%.4f, skipped=%d",
        control_result.mean,
        control_result.skipped_count,
    )

    # ── duration_ratio (val96 rows only, against ref_duration_s) ──
    dur_eligible = [
        r for r in scoreable if r.get("ref_duration_s") is not None and r["prompt_set"] == "val96"
    ]
    if len(dur_eligible) > 0:
        dur_paths = [Path(str(r["audio_path"])) for r in dur_eligible]
        dur_refs = [float(r["ref_duration_s"]) for r in dur_eligible]  # type: ignore[arg-type]
        dur_result = compute_duration_ratio(dur_paths, dur_refs)
        for rec, ratio in zip(dur_eligible, dur_result.per_clip, strict=True):
            rec["duration_ratio"] = ratio
        logger.info(
            "duration_ratio: median=%.3f, explosions=%d/%d",
            dur_result.median,
            dur_result.explosion_count,
            len(dur_eligible),
        )

    # ── WER (gated by duration_ratio, t0008 convention) ──
    wer_eligible = [
        r
        for r in scoreable
        if r.get("duration_ratio") is None
        or (DURATION_RATIO_LOW <= float(r["duration_ratio"]) <= DURATION_RATIO_HIGH)  # type: ignore[arg-type]
    ]
    if len(wer_eligible) > 0:
        wer_paths = [Path(str(r["audio_path"])) for r in wer_eligible]
        wer_texts = [str(r["text"]) for r in wer_eligible]
        wer_ratios = [r.get("duration_ratio") for r in wer_eligible]  # type: ignore[misc]
        wer_result = compute_wer(wer_paths, wer_texts, wer_ratios)  # type: ignore[arg-type]
        for rec, wer_val in zip(wer_eligible, wer_result.per_clip, strict=True):
            rec["wer"] = wer_val
        logger.info("WER: mean=%.4f, skipped=%d", wer_result.mean_wer, wer_result.skipped_count)

    RESULTS_PER_CLIP_METRICS.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    logger.info(
        "Wrote merged per_clip_metrics.json -> %s (%d rows)",
        RESULTS_PER_CLIP_METRICS,
        len(all_records),
    )


if __name__ == "__main__":
    main()
