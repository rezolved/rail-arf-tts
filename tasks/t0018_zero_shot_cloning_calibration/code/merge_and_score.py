"""Merge per-system per-clip files, compute speaker_sim/duration_ratio/WER (plan Steps 10-11).

Merges:
- results/per_clip_metrics_chatterbox_ref_single.json
- results/per_clip_metrics_chatterbox_ref_concat.json
- results/per_clip_metrics_cosyvoice2_ref_single.json
- results/per_clip_metrics_cosyvoice2_ref_concat.json (all-null: REQ-6 rejection, hard 30s
  CosyVoice2 limit exceeded by the ref_concat clip -- see results/tables.json notes)
- results/per_clip_metrics_elevenlabs_david.json (already speaker_sim-scored by
  score_elevenlabs_baseline.py; duration_ratio/wer retained from t0008)

f5_tts and kokoro_v3_bundle contribute NO rows (see intervention/ files) -- not synthesized this
session.

For every row with a non-null audio_path, computes:
- speaker_sim vs the half-A centroid (built in build_references.py)
- duration_ratio vs the matching reference duration (val96 prompts only; fillers use the
  ElevenLabs half-A/half-B corpus's own natural durations where available -- t0008 convention:
  fillers without a resolvable natural reference are left with duration_ratio=None)
- wer (gated by DURATION_RATIO_LOW/HIGH, per t0008's own convention)
"""

from __future__ import annotations

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
from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
    DATA_REFERENCES_DIR,
    RESULTS_DIR,
    RESULTS_PER_CLIP_METRICS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

PER_CLIP_FILES = [
    "per_clip_metrics_chatterbox_ref_single.json",
    "per_clip_metrics_chatterbox_ref_concat.json",
    "per_clip_metrics_cosyvoice2_ref_single.json",
    "per_clip_metrics_cosyvoice2_ref_concat.json",
    "per_clip_metrics_elevenlabs_david.json",
]

HALF_A_CENTROID_NPY = DATA_REFERENCES_DIR / "half_a_centroid.npy"

# Marker directory name shared by both the VM's absolute audio_path values (recorded during
# synthesis, e.g. "/mnt/cache/persist/.../repo/tasks/t0018.../results/audio_samples/harness/...")
# and this local worktree's own results dir -- used to rewrite paths after rsync-pulling the audio
# back from the VM, since the two machines' absolute paths differ upstream of this marker.
AUDIO_PATH_MARKER = "results/audio_samples/"


def _localize_audio_path(audio_path_str: str) -> str:
    """Rewrite a VM-absolute audio_path to the equivalent path in this local worktree."""
    idx = audio_path_str.find(AUDIO_PATH_MARKER)
    if idx == -1:
        return audio_path_str
    relative = audio_path_str[idx + len(AUDIO_PATH_MARKER) :]
    return str(RESULTS_DIR / "audio_samples" / relative)


def main() -> None:
    all_records: list[dict[str, object]] = []
    for fname in PER_CLIP_FILES:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            logger.warning("Missing per-clip file (skipping): %s", fpath)
            continue
        records = json.loads(fpath.read_text(encoding="utf-8"))
        for rec in records:
            if rec.get("audio_path") is not None:
                rec["audio_path"] = _localize_audio_path(str(rec["audio_path"]))
        logger.info("Loaded %d records from %s", len(records), fname)
        all_records.extend(records)

    logger.info("Total merged records: %d", len(all_records))
    n_missing = sum(
        1
        for r in all_records
        if r.get("audio_path") is not None and not Path(str(r["audio_path"])).exists()
    )
    if n_missing > 0:
        logger.warning("%d records have audio_path that does not exist on disk!", n_missing)

    centroid = np.load(str(HALF_A_CENTROID_NPY))

    # ── speaker_sim (skip elevenlabs_david -- already scored vs half-B in its own script) ──
    non_el_records = [
        r for r in all_records if r["system"] != "elevenlabs_david" and r.get("audio_path")
    ]
    paths = [Path(str(r["audio_path"])) for r in non_el_records]
    sim_result = compute_speaker_sim(paths, centroid)
    for rec, sim in zip(non_el_records, sim_result.per_clip, strict=True):
        rec["speaker_sim"] = sim
    logger.info(
        "speaker_sim: mean=%.4f, skipped=%d (%s)",
        sim_result.mean,
        sim_result.skipped_count,
        sim_result.skipped_reason,
    )

    # ── duration_ratio (val96 rows have ref_duration_s from the prompt manifest) ──
    dur_eligible = [
        r for r in non_el_records if r.get("audio_path") and r.get("ref_duration_s") is not None
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
        for r in non_el_records
        if r.get("audio_path")
        and (
            r.get("duration_ratio") is None
            or (DURATION_RATIO_LOW <= float(r["duration_ratio"]) <= DURATION_RATIO_HIGH)  # type: ignore[arg-type]
        )
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
