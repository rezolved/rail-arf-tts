"""Milestone D step 12 (REQ-10, [CRITICAL]): three-way v10/v11-as-shipped/corrected gate regression.

Runs the hardened `check_audio_quality()` (Milestone D step 10) against exactly three fixtures:

    (a) `tasks/t0013_v10_synthesis_quality_forensics/results/audio_samples/v10_epoch16_primary.wav`
        -- the confirmed-broken v10 checkpoint's output. Expect `is_likely_noise=True` (unchanged,
        via the original clip/flatness signals -- this fixture was never a duration problem).
    (b) `tasks/t0014_v11_decoder_fix_retrain/results/audio_samples/ft/v11_best.wav` -- the exact
        73.95s clip this whole task investigates. Expect `is_likely_noise=False` under the
        *original* 3-signal logic (as t0014 already found) but the *hardened* gate
        (`hardened_gate_pass`) to flip to False -- the specific blind-spot closure this task exists
        to prove.
    (c) `results/audio_samples/ft/v11_corrected.wav`, only if the REQ-6 cheap-fix path produced one.
        Expect it to pass all signals. If the REQ-7 (no-fix) path was taken instead, this fixture is
        skipped and noted explicitly as absent, rather than silently omitted.

`hardened_gate_pass` is defined as `not (is_likely_noise or duration_sanity_pass is False or
longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S)` -- `longest_nonsilent_run_s` is
already folded into `is_likely_noise` by Step 10's edit, so this is mostly `not is_likely_noise and
duration_sanity_pass is not False`, spelled out explicitly here so the derived field's provenance is
legible without re-reading `audio_quality_check.py`.

All three fixtures use the same known gate text ("This is a test of the Style T T S two inference
harness.") that originally produced (a) and (b), per plan.md Milestone D step 12.

Usage::

    code/.venv-styletts2/bin/python -m \
        tasks.t0015_v11_duration_blowup_forensics.code.run_gate_regression
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import (
    LONGEST_NONSILENT_RUN_THRESHOLD_S,
    check_audio_quality,
)
from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
    GATE_REGRESSION_JSON,
    RESULTS_AUDIO_DIR,
    T0013_V10_AUDIO_DIR,
    V11_BEST_WAV,
)

GATE_TEXT = "This is a test of the Style T T S two inference harness."
V10_EPOCH16_PRIMARY_WAV = T0013_V10_AUDIO_DIR / "v10_epoch16_primary.wav"
V11_CORRECTED_WAV = RESULTS_AUDIO_DIR / "ft" / "v11_corrected.wav"


@dataclass(frozen=True, slots=True)
class RegressionRow:
    fixture: str
    wav_path: str
    exists: bool
    rms: float | None
    peak: float | None
    silence_fraction: float | None
    spectral_flatness: float | None
    clip_fraction: float | None
    longest_nonsilent_run_s: float | None
    duration_sanity_pass: bool | None
    is_likely_noise: bool | None
    hardened_gate_pass: bool | None
    note: str | None


def evaluate_fixture(fixture: str, wav_path: object, note: str | None = None) -> RegressionRow:
    if not wav_path.exists():  # type: ignore[attr-defined]
        return RegressionRow(
            fixture=fixture,
            wav_path=str(wav_path),
            exists=False,
            rms=None,
            peak=None,
            silence_fraction=None,
            spectral_flatness=None,
            clip_fraction=None,
            longest_nonsilent_run_s=None,
            duration_sanity_pass=None,
            is_likely_noise=None,
            hardened_gate_pass=None,
            note=note or "fixture does not exist",
        )

    result = check_audio_quality(wav_path, text=GATE_TEXT)  # type: ignore[arg-type]
    hardened_gate_pass = not (
        result.is_likely_noise
        or result.duration_sanity_pass is False
        or result.longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S
    )
    return RegressionRow(
        fixture=fixture,
        wav_path=str(wav_path),
        exists=True,
        rms=result.rms,
        peak=result.peak,
        silence_fraction=result.silence_fraction,
        spectral_flatness=result.spectral_flatness,
        clip_fraction=result.clip_fraction,
        longest_nonsilent_run_s=result.longest_nonsilent_run_s,
        duration_sanity_pass=result.duration_sanity_pass,
        is_likely_noise=result.is_likely_noise,
        hardened_gate_pass=hardened_gate_pass,
        note=note,
    )


def main() -> None:
    rows = [
        evaluate_fixture("v10", V10_EPOCH16_PRIMARY_WAV),
        evaluate_fixture("v11_as_shipped", V11_BEST_WAV),
    ]
    if V11_CORRECTED_WAV.exists():
        rows.append(evaluate_fixture("v11_corrected", V11_CORRECTED_WAV))
    else:
        rows.append(
            evaluate_fixture(
                "v11_corrected",
                V11_CORRECTED_WAV,
                note="REQ-7 (no cheap fix) path was taken -- no corrected fixture exists to test",
            )
        )

    for r in rows:
        print(
            f"{r.fixture}: exists={r.exists} is_likely_noise={r.is_likely_noise} "
            f"hardened_gate_pass={r.hardened_gate_pass} "
            f"longest_nonsilent_run_s={r.longest_nonsilent_run_s} "
            f"duration_sanity_pass={r.duration_sanity_pass} note={r.note}"
        )

    GATE_REGRESSION_JSON.parent.mkdir(parents=True, exist_ok=True)
    GATE_REGRESSION_JSON.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
    print(f"Wrote {GATE_REGRESSION_JSON}")


if __name__ == "__main__":
    main()
