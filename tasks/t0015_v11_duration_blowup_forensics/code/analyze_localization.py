"""Milestone B step 6 (REQ-2): localize the duration blowup using Step 5's characterization data.

Reads `results/duration_characterization.json` (written by `code/run_characterization.py`) and, for
each record, computes:

(a) mean per-token `pred_dur` (`pred_dur_sum / input_token_count`);
(b) whether any single per-token `pred_dur` value is at or near the `model_params.max_dur = 50`
    ceiling (`config_david_v11.yml:79`) -- saturation there would indicate the sigmoid-sum output is
    pinned at its per-token cap, i.e. a predictor-calibration failure;
(c) whether `input_token_count` is wildly disproportionate to `word_count`
    (`input_token_count / word_count > 8`) -- a phonemizer over-segmentation, plumbing-adjacent
    hypothesis;
(d) whether the frame-count-implied duration matches the actual `output_duration_s`, accounting for
    a DECODER-INTRINSIC 2x time-domain upsample this task's own source reading and an empirical
    control-checkpoint validation run confirmed (not assumed): `code/kikiri-tts/StyleTTS2/Modules/
    {hifigan,istftnet}.py`'s shared `Decoder` class applies `encode` + 3 non-upsampling `decode`
    blocks + 1 `decode` block with `upsample=True` (`UpSample1d` -> `F.interpolate(x,
    scale_factor=2, mode="nearest")`) BEFORE the final `Generator` (whose own `upsample_rates`
    product already equals `hop_length` exactly, 10*5*3*2=300). Running the instrumented
    `synthesize()` against the LibriTTS control checkpoint (`epochs_2nd_00020.pth`, same decoder
    architecture, `model_params.decoder.type: hifigan` in its own `config.yml`) on the gate text
    gave `pred_dur_sum=186`, `output_duration_s=4.65` -- `4.65 / (186 * 300 / 24000) = 2.00`
    exactly, the SAME ratio observed in every v11 characterization record. **This means a
    `frame_vs_output_ratio` of ~2.0 is the model's NORMAL, healthy behavior, not a defect
    signal** -- the naive `pred_dur_sum * hop_length / sample_rate` formula undercounts by this
    intrinsic factor; `expected_output_duration_s` below corrects for it. A `frame_vs_output_ratio`
    that
    DEVIATES from ~2.0 (not one that equals it) would be the actual plumbing-bug signal, and no v11
    record shows such a deviation -- ruling out a `pred_aln_trg`/decoder-upsampling plumbing bug as
    a contributor, and pointing the localization squarely at `pred_dur` itself (or its upstream
    inputs) being elevated.

Writes a summary JSON (`results/localization_summary.json`) that
`results/duration_blowup_diagnosis.md` (Step 14) cites directly, per plan.md Milestone B step 6
("folded into Step 12's [now Step 14's] duration_blowup_diagnosis.md, not a separate file" -- the
raw per-text numbers are folded into the diagnosis doc; this JSON is the intermediate computation
artifact backing it).

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.analyze_localization
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
    DURATION_CHARACTERIZATION_JSON,
    RESULTS_DIR,
)

MAX_DUR_CEILING = 50  # config_david_v11.yml:79 model_params.max_dur
MAX_DUR_NEAR_CEILING_FRACTION = 0.9  # a per-token value >= 90% of max_dur counts as "near ceiling"
OVERSEGMENTATION_TOKEN_PER_WORD_RATIO = 8.0  # plan.md step 6's generous multiple
HOP_LENGTH = 300  # config_david_v11.yml preprocess_params.spect_params.hop_length
SAMPLE_RATE = 24000  # config_david_v11.yml preprocess_params.sr
HOP_LENGTH_SECONDS = HOP_LENGTH / SAMPLE_RATE
# Confirmed empirically (control-checkpoint validation run, see module docstring) and by reading
# `Decoder.decode`'s last `AdainResBlk1d(..., upsample=True)` block in both hifigan.py and
# istftnet.py -- NOT assumed. A healthy StyleTTS2-native pipeline's output is 2x longer than the
# naive `pred_dur_sum * hop_length / sample_rate` estimate, by design.
DECODER_INTRINSIC_UPSAMPLE_FACTOR = 2.0
PLUMBING_DEVIATION_TOLERANCE = 0.05  # 5% -- generous, given float rounding in wav-length trimming

LOCALIZATION_SUMMARY_JSON = RESULTS_DIR / "localization_summary.json"


@dataclass(frozen=True, slots=True)
class LocalizationRow:
    index: int
    slug: str
    category: str
    word_count: int
    input_token_count: int
    pred_dur_sum: int
    mean_pred_dur_per_token: float | None
    max_pred_dur_token: float | None
    num_tokens_near_ceiling: int
    fraction_tokens_near_ceiling: float | None
    tokens_per_word_ratio: float | None
    oversegmentation_flag: bool
    frame_implied_duration_s: float
    expected_output_duration_s: float
    output_duration_s: float
    frame_vs_output_ratio: float | None
    plumbing_deviation_flag: bool
    duration_ratio: float | None
    error: str | None


def analyze_record(record: dict[str, object]) -> LocalizationRow:
    pred_dur = record["pred_dur"]
    assert isinstance(pred_dur, list)
    input_token_count = int(record["input_token_count"])  # type: ignore[arg-type]
    pred_dur_sum = int(record["pred_dur_sum"])  # type: ignore[arg-type]
    word_count = int(record["word_count"])  # type: ignore[arg-type]
    output_duration_s = float(record["output_duration_s"])  # type: ignore[arg-type]
    error = record["error"]

    mean_pred_dur_per_token = pred_dur_sum / input_token_count if input_token_count > 0 else None
    max_pred_dur_token = max(pred_dur) if len(pred_dur) > 0 else None
    ceiling_threshold = MAX_DUR_CEILING * MAX_DUR_NEAR_CEILING_FRACTION
    num_near_ceiling = sum(1 for v in pred_dur if v >= ceiling_threshold)
    fraction_near_ceiling = num_near_ceiling / len(pred_dur) if len(pred_dur) > 0 else None

    tokens_per_word_ratio = input_token_count / word_count if word_count > 0 else None
    oversegmentation_flag = (
        tokens_per_word_ratio is not None
        and tokens_per_word_ratio > OVERSEGMENTATION_TOKEN_PER_WORD_RATIO
    )

    frame_implied_duration_s = pred_dur_sum * HOP_LENGTH_SECONDS
    expected_output_duration_s = frame_implied_duration_s * DECODER_INTRINSIC_UPSAMPLE_FACTOR
    frame_vs_output_ratio = (
        output_duration_s / frame_implied_duration_s if frame_implied_duration_s > 0 else None
    )
    plumbing_deviation_flag = (
        frame_vs_output_ratio is not None
        and abs(frame_vs_output_ratio - DECODER_INTRINSIC_UPSAMPLE_FACTOR)
        > DECODER_INTRINSIC_UPSAMPLE_FACTOR * PLUMBING_DEVIATION_TOLERANCE
    )

    return LocalizationRow(
        index=int(record["index"]),  # type: ignore[arg-type]
        slug=str(record["slug"]),
        category=str(record["category"]),
        word_count=word_count,
        input_token_count=input_token_count,
        pred_dur_sum=pred_dur_sum,
        mean_pred_dur_per_token=mean_pred_dur_per_token,
        max_pred_dur_token=max_pred_dur_token,
        num_tokens_near_ceiling=num_near_ceiling,
        fraction_tokens_near_ceiling=fraction_near_ceiling,
        tokens_per_word_ratio=tokens_per_word_ratio,
        oversegmentation_flag=oversegmentation_flag,
        frame_implied_duration_s=frame_implied_duration_s,
        expected_output_duration_s=expected_output_duration_s,
        output_duration_s=output_duration_s,
        frame_vs_output_ratio=frame_vs_output_ratio,
        plumbing_deviation_flag=plumbing_deviation_flag,
        duration_ratio=(
            float(record["duration_ratio"]) if record["duration_ratio"] is not None else None  # type: ignore[arg-type]
        ),
        error=str(error) if error is not None else None,
    )


def main() -> None:
    with DURATION_CHARACTERIZATION_JSON.open() as f:
        records = json.load(f)

    rows = [analyze_record(r) for r in records]

    valid_rows = [r for r in rows if r.error is None]
    mean_fraction_near_ceiling = (
        sum(r.fraction_tokens_near_ceiling or 0.0 for r in valid_rows) / len(valid_rows)
        if len(valid_rows) > 0
        else None
    )
    any_oversegmentation = any(r.oversegmentation_flag for r in valid_rows)
    mean_frame_vs_output_ratio = (
        sum(r.frame_vs_output_ratio or 0.0 for r in valid_rows) / len(valid_rows)
        if len(valid_rows) > 0
        else None
    )
    any_plumbing_deviation = any(r.plumbing_deviation_flag for r in valid_rows)

    summary = {
        "rows": [asdict(r) for r in rows],
        "aggregate": {
            "num_records": len(rows),
            "num_valid_records": len(valid_rows),
            "mean_fraction_tokens_near_ceiling": mean_fraction_near_ceiling,
            "any_oversegmentation_flagged": any_oversegmentation,
            "mean_frame_vs_output_duration_ratio": mean_frame_vs_output_ratio,
            "decoder_intrinsic_upsample_factor_expected": DECODER_INTRINSIC_UPSAMPLE_FACTOR,
            "any_plumbing_deviation_flagged": any_plumbing_deviation,
            "control_checkpoint_validation_ratio": (
                "epochs_2nd_00020.pth (LibriTTS control) on the gate text: pred_dur_sum=186, "
                "output_duration_s=4.65 -> frame_vs_output_ratio=2.00, matching every v11 record "
                "-- see results/duration_blowup_diagnosis.md for the full validation run."
            ),
        },
    }
    LOCALIZATION_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {LOCALIZATION_SUMMARY_JSON}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
