"""Milestone D step 11 (REQ-9): evaluate an ASR-round-trip check as an optional third gate layer.

Imports `compute_wer` from `tasks.t0008_tts_eval_harness_baselines.code.scoring` (the
`tts_eval_harness` library) and runs it once against a handful of Step 5's characterization outputs
-- a mix of a normal-ish-duration one (if any) and the blown-up v11 default-parameter ones -- to
observe its behavior in practice per plan.md Milestone D step 11: does it complete without new
dependency friction (`faster-whisper`'s `base.en` model download), does its
`DURATION_RATIO_LOW=0.5`/`DURATION_RATIO_HIGH=2.0` gating skip the blown-up clips entirely, and how
long ASR transcription takes per clip on CPU.

This is an EVALUATION, not a permanent gate wiring -- results go to
`results/asr_roundtrip_evaluation.md` (prose verdict) and this script's own stdout/JSON, not into
`audio_quality_check.py`, unless the evaluation concludes it is trivially easy to wire in as a fully
optional (default-off) parameter (plan.md's explicit conditional).

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.run_asr_roundtrip_eval
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from tasks.t0008_tts_eval_harness_baselines.code.constants import (
    DURATION_RATIO_HIGH,
    DURATION_RATIO_LOW,
    WHISPER_MODEL_SIZE,
)
from tasks.t0015_v11_duration_blowup_forensics.code.paths import DURATION_CHARACTERIZATION_JSON

ASR_ROUNDTRIP_RAW_JSON = DURATION_CHARACTERIZATION_JSON.parent / "asr_roundtrip_raw.json"
# How many characterization records to sample for this evaluation (a small, representative subset
# per plan.md step 11 -- "run it once against 2-3 ... outputs", not the full batch).
NUM_SAMPLES = 3


@dataclass(frozen=True, slots=True)
class AsrEvalRow:
    slug: str
    text: str
    output_duration_s: float
    expected_duration_s: float
    duration_ratio: float | None
    gated_out_by_duration_ratio: bool
    wer: float | None
    transcription_wall_time_s: float | None
    error: str | None


def main() -> None:
    with DURATION_CHARACTERIZATION_JSON.open() as f:
        records = json.load(f)

    valid_records = [r for r in records if r["error"] is None]
    # Sample: the record with the smallest duration_ratio (closest to "normal", if any) plus two of
    # the most blown-up ones, so the evaluation covers both ends of the observed range.
    sorted_by_ratio = sorted(valid_records, key=lambda r: r["duration_ratio"] or 0.0)
    sample = ([sorted_by_ratio[0]] if len(sorted_by_ratio) > 0 else []) + sorted_by_ratio[-2:]
    sample = sample[:NUM_SAMPLES]

    print(f"faster-whisper model size: {WHISPER_MODEL_SIZE}")
    print(f"DURATION_RATIO gating: [{DURATION_RATIO_LOW}, {DURATION_RATIO_HIGH}]")

    try:
        from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_wer
    except ImportError as exc:
        print(f"compute_wer import failed: {exc}")
        return

    rows: list[AsrEvalRow] = []
    for record in sample:
        wav_path = (
            DURATION_CHARACTERIZATION_JSON.parent
            / "audio_samples"
            / "characterization"
            / f"v11_{record['index']:02d}_{record['slug']}.wav"
        )
        duration_ratio = record["duration_ratio"]
        gated_out = duration_ratio is not None and (
            duration_ratio > DURATION_RATIO_HIGH or duration_ratio < DURATION_RATIO_LOW
        )
        start = time.perf_counter()
        try:
            result = compute_wer([wav_path], [record["text"]], [duration_ratio])
            elapsed = time.perf_counter() - start
            wer = result.per_clip[0] if len(result.per_clip) > 0 else None
            error = None
        except Exception as exc:  # noqa: BLE001 -- record the failure, this is an evaluation script
            elapsed = time.perf_counter() - start
            wer = None
            error = str(exc)
        rows.append(
            AsrEvalRow(
                slug=record["slug"],
                text=record["text"],
                output_duration_s=record["output_duration_s"],
                expected_duration_s=record["expected_duration_s"],
                duration_ratio=duration_ratio,
                gated_out_by_duration_ratio=gated_out,
                wer=wer,
                transcription_wall_time_s=elapsed,
                error=error,
            )
        )
        print(
            f"{record['slug']}: duration_ratio={duration_ratio} gated_out={gated_out} wer={wer} "
            f"wall_time_s={elapsed:.1f} error={error}"
        )

    ASR_ROUNDTRIP_RAW_JSON.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
    print(f"Wrote {ASR_ROUNDTRIP_RAW_JSON}")


if __name__ == "__main__":
    main()
