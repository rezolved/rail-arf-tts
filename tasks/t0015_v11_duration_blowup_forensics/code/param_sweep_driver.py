"""Milestone C step 8 (REQ-5): sweep `alpha`/`beta`/`diffusion_steps`/`embedding_scale`.

Pre-registered pass criteria (restated in `results/duration_blowup_diagnosis.md`, written before
any sweep run per plan.md Milestone C step 8): a parameter combination counts as "the cheap fix
worked" only if ALL of:

    (a) `check_audio_quality()`'s `is_likely_noise == False` (hardened signals, once
        `code/audio_quality_check.py` is extended in Milestone D step 10 -- this driver applies the
        same duration-sanity/non-silent-run logic ad hoc here via direct computation, formalized in
        the gate module afterward, per plan.md's explicit sequencing note);
    (b) `duration_ratio` (`output_duration_s / naive_upper_bound_s`,
        `naive_upper_bound_s = word_count / NAIVE_MIN_WORDS_PER_SECOND`, matching
        `audio_quality_check.py`'s `estimate_naive_duration_bound_s`) <= `PASS_DURATION_RATIO_MAX`;
    (c) `longest_nonsilent_run_s <= PASS_LONGEST_NONSILENT_RUN_MAX_S`.

No partial improvement (e.g. duration cut in half but still far too long) is reported as "fixed" --
a combination must satisfy all three simultaneously.

Grid (fixed test text, matching t0013's/t0014's own gate text for direct comparability): `alpha` in
{0.1, 0.3, 0.5} x `beta` in {0.3, 0.7, 0.9} at fixed `diffusion_steps=5, embedding_scale=1.0` (9
combinations); then, using whichever `(alpha, beta)` pair produced the shortest `output_duration_s`
in that grid (even if not passing), `embedding_scale` in {0.5, 2.0} and `diffusion_steps` in {10} (3
more combinations) -- 12 total. Every combination's full result is recorded, including a baseline
(default-parameter) run for the same text, and any crashed combination's error message rather than
a silently omitted record.

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.param_sweep_driver
    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.param_sweep_driver --validate-first-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf
import torch
from munch import Munch

from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import (
    check_audio_quality,
)
from tasks.t0015_v11_duration_blowup_forensics.code.infer_styletts2 import (
    build_harness,
    compute_style,
    synthesize,
)
from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
    PARAM_SWEEP_JSON,
    REFERENCE_CONCAT_WAV,
    RESULTS_AUDIO_DIR,
    STYLETTS2_DIR,
    V11_CHECKPOINT,
    V11_CONFIG,
)

SAMPLE_RATE = 24000
# Identical to t0013's/t0014's own gate text, for direct comparability with their recorded numbers.
SWEEP_TEXT = "This is a test of the Style T T S two inference harness."

ALPHA_GRID: tuple[float, ...] = (0.1, 0.3, 0.5)
BETA_GRID: tuple[float, ...] = (0.3, 0.7, 0.9)
FIXED_DIFFUSION_STEPS = 5
FIXED_EMBEDDING_SCALE = 1.0
EXTENSION_EMBEDDING_SCALES: tuple[float, ...] = (0.5, 2.0)
EXTENSION_DIFFUSION_STEPS: tuple[int, ...] = (10,)

DEFAULT_ALPHA = 0.3
DEFAULT_BETA = 0.7
DEFAULT_DIFFUSION_STEPS = 5
DEFAULT_EMBEDDING_SCALE = 1.0

# Pre-registered pass criteria (plan.md Milestone C step 8) -- fixed before any sweep run.
NAIVE_MIN_WORDS_PER_SECOND = 1.5  # matches audio_quality_check.py's hardened constant
PASS_DURATION_RATIO_MAX = 3.0
PASS_LONGEST_NONSILENT_RUN_MAX_S = 12.0
SILENCE_FRAME_DBFS_THRESHOLD = -40.0
SILENCE_FRAME_SECONDS = 0.02


@dataclass(frozen=True, slots=True)
class SweepRecord:
    label: str
    alpha: float
    beta: float
    diffusion_steps: int
    embedding_scale: float
    output_duration_s: float | None
    pred_dur_sum: int | None
    is_likely_noise: bool | None
    longest_nonsilent_run_s: float | None
    duration_ratio: float | None
    passed: bool
    error: str | None


def longest_nonsilent_run_seconds(wav: np.ndarray, sr: int) -> float:
    """Longest contiguous run of 20ms frames NOT below the silence dBFS threshold, in seconds.

    Standalone pre-Milestone-D copy of the logic `audio_quality_check.py`'s hardened signal will
    formalize in Step 10 -- kept here so this sweep can apply the pass criteria before that module
    edit lands, per plan.md's explicit step-ordering note.
    """
    frame_len = max(1, int(SILENCE_FRAME_SECONDS * sr))
    n_frames = len(wav) // frame_len
    max_run = 0
    current_run = 0
    for i in range(n_frames):
        frame = wav[i * frame_len : (i + 1) * frame_len]
        frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) + 1e-9
        frame_dbfs = 20 * np.log10(frame_rms)
        if frame_dbfs >= SILENCE_FRAME_DBFS_THRESHOLD:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run * frame_len / sr


def run_combination(
    model: Munch,
    model_params: Munch,
    ref_s: torch.Tensor,
    *,
    label: str,
    alpha: float,
    beta: float,
    diffusion_steps: int,
    embedding_scale: float,
) -> SweepRecord:
    word_count = len(SWEEP_TEXT.split())
    naive_upper_bound_s = word_count / NAIVE_MIN_WORDS_PER_SECOND
    try:
        result = synthesize(
            model,
            model_params,
            SWEEP_TEXT,
            ref_s,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )
    except Exception as exc:  # noqa: BLE001 -- record the failure, don't abort the sweep
        return SweepRecord(
            label=label,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            output_duration_s=None,
            pred_dur_sum=None,
            is_likely_noise=None,
            longest_nonsilent_run_s=None,
            duration_ratio=None,
            passed=False,
            error=str(exc),
        )

    out_wav = RESULTS_AUDIO_DIR / "sweep" / f"{label}.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), result.wav, SAMPLE_RATE)

    output_duration_s = len(result.wav) / SAMPLE_RATE
    duration_ratio = output_duration_s / naive_upper_bound_s if naive_upper_bound_s > 0 else None
    quality = check_audio_quality(out_wav)
    longest_run_s = longest_nonsilent_run_seconds(result.wav, SAMPLE_RATE)

    passed = (
        not quality.is_likely_noise
        and duration_ratio is not None
        and duration_ratio <= PASS_DURATION_RATIO_MAX
        and longest_run_s <= PASS_LONGEST_NONSILENT_RUN_MAX_S
    )

    return SweepRecord(
        label=label,
        alpha=alpha,
        beta=beta,
        diffusion_steps=diffusion_steps,
        embedding_scale=embedding_scale,
        output_duration_s=output_duration_s,
        pred_dur_sum=result.pred_dur_sum,
        is_likely_noise=quality.is_likely_noise,
        longest_nonsilent_run_s=longest_run_s,
        duration_ratio=duration_ratio,
        passed=passed,
        error=None,
    )


def load_existing_records() -> dict[str, SweepRecord]:
    """Resume support: a full 12-combination sweep on this CPU harness costs ~60-90 minutes (see
    plan.md Time Estimation), so a prior partial run's already-computed, error-free records
    (matched by `label`, which encodes the exact parameter combination) are reused rather than
    re-synthesized identically (seeds are fixed, so a re-run would produce the same numbers) --
    this is a resume, not a substitution: every label still gets exactly one recorded result.
    """
    if not PARAM_SWEEP_JSON.exists():
        return {}
    try:
        existing = json.loads(PARAM_SWEEP_JSON.read_text())
    except json.JSONDecodeError:
        return {}
    result: dict[str, SweepRecord] = {}
    for r in existing.get("records", []):
        if r.get("error") is None:
            result[r["label"]] = SweepRecord(**r)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-first-2", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    sys.path.insert(0, str(STYLETTS2_DIR))
    os.chdir(STYLETTS2_DIR)

    resumable = load_existing_records() if args.resume else {}
    if resumable:
        print(f"Resuming: reusing {len(resumable)} already-computed record(s): {sorted(resumable)}")

    model, model_params, _log_path = build_harness(V11_CONFIG, V11_CHECKPOINT)
    ref_s = compute_style(model, REFERENCE_CONCAT_WAV)

    records: list[SweepRecord] = []

    def record_and_save(rec: SweepRecord) -> None:
        records.append(rec)
        print(
            f"{rec.label}: alpha={rec.alpha} beta={rec.beta} steps={rec.diffusion_steps} "
            f"scale={rec.embedding_scale} -> duration_s={rec.output_duration_s} "
            f"duration_ratio={rec.duration_ratio} noise={rec.is_likely_noise} "
            f"longest_run_s={rec.longest_nonsilent_run_s} passed={rec.passed} error={rec.error}"
        )
        PARAM_SWEEP_JSON.parent.mkdir(parents=True, exist_ok=True)
        PARAM_SWEEP_JSON.write_text(
            json.dumps(
                {
                    "pass_criteria": {
                        "is_likely_noise": False,
                        "duration_ratio_max": PASS_DURATION_RATIO_MAX,
                        "longest_nonsilent_run_s_max": PASS_LONGEST_NONSILENT_RUN_MAX_S,
                        "naive_min_words_per_second": NAIVE_MIN_WORDS_PER_SECOND,
                        "note": (
                            "A combination passes only if ALL three criteria are satisfied "
                            "simultaneously. No partial improvement counts as a pass."
                        ),
                    },
                    "sweep_text": SWEEP_TEXT,
                    "records": [asdict(r) for r in records],
                },
                indent=2,
            )
            + "\n"
        )

    def execute(
        *, label: str, alpha: float, beta: float, diffusion_steps: int, embedding_scale: float
    ) -> None:
        if label in resumable:
            print(f"{label}: reusing resumed record (skipping re-synthesis)")
            record_and_save(resumable[label])
            return
        record_and_save(
            run_combination(
                model,
                model_params,
                ref_s,
                label=label,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            )
        )

    # Explicit baseline re-run (default params) on the exact sweep text, for direct comparison.
    execute(
        label="baseline_default",
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        diffusion_steps=DEFAULT_DIFFUSION_STEPS,
        embedding_scale=DEFAULT_EMBEDDING_SCALE,
    )

    grid_combinations = [(a, b) for a in ALPHA_GRID for b in BETA_GRID]
    if args.validate_first_2:
        grid_combinations = grid_combinations[:2]

    for alpha, beta in grid_combinations:
        execute(
            label=f"grid_a{alpha}_b{beta}",
            alpha=alpha,
            beta=beta,
            diffusion_steps=FIXED_DIFFUSION_STEPS,
            embedding_scale=FIXED_EMBEDDING_SCALE,
        )

    if args.validate_first_2:
        print("Validation-only run (first 2 grid combinations) complete -- stopping here.")
        return

    # Extension phase: use the (alpha, beta) pair with the shortest output_duration_s from the
    # grid phase, even if it did not pass, per plan.md Milestone C step 8.
    grid_records = [
        r for r in records if r.label.startswith("grid_") and r.output_duration_s is not None
    ]
    if len(grid_records) == 0:
        print("No successful grid combinations -- skipping extension phase.")
        return
    best = min(grid_records, key=lambda r: r.output_duration_s)  # type: ignore[arg-type,return-value]
    print(f"Extension phase base: alpha={best.alpha} beta={best.beta} (shortest duration in grid)")

    for embedding_scale in EXTENSION_EMBEDDING_SCALES:
        execute(
            label=f"ext_scale{embedding_scale}",
            alpha=best.alpha,
            beta=best.beta,
            diffusion_steps=FIXED_DIFFUSION_STEPS,
            embedding_scale=embedding_scale,
        )
    for diffusion_steps in EXTENSION_DIFFUSION_STEPS:
        execute(
            label=f"ext_steps{diffusion_steps}",
            alpha=best.alpha,
            beta=best.beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=FIXED_EMBEDDING_SCALE,
        )

    print(f"Wrote {PARAM_SWEEP_JSON} ({len(records)} records)")


if __name__ == "__main__":
    main()
