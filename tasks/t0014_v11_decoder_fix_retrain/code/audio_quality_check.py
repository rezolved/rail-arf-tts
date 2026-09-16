"""Reusable "does this sound like noise?" regression check for the project's eval-harness pipeline.

Copied unchanged (logic-for-logic) from
`tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py` -- this task's mandatory
audible-speech gate (REQ-6/REQ-7): `check_audio_quality(...).is_likely_noise` on the v11 synthesis
output is the SOLE pass/fail criterion for this task's completion claim, per `plan/plan.md`
Milestone C step 9 and the Rejection Criteria section. Only `demo()` below is adapted (points at
this task's own `results/audio_samples/` instead of t0013's).

Three signals distinguish "noise" from "speech" cheaply. This task's own v10 checkpoints (see
`results/v10_diagnosis.md`) turned out NOT to be classic white noise -- they are rail-to-rail
CLIPPED/SATURATED output (75-81% of samples pinned at +-1.0, dominant frequency component at 0 Hz
DC) with an almost perfectly FLAT (near-zero) spectral flatness, the opposite signature from what
"noise" first suggests. A flatness-only heuristic would have missed this entirely -- catching it
required adding the clipping-fraction signal below:

* `silence_fraction`: fraction of 20ms frames below -40dBFS. Near-silent output (a crashed/
  degenerate synth) has this close to 1.0.
* `spectral_flatness`: geometric-mean / arithmetic-mean of the power spectrum, per frame (Wiener
  entropy). Near 1.0 for white-noise-like signals (energy spread evenly across frequencies, no
  formant structure); well below 1.0 for voiced speech OR for a saturated/DC-dominated signal
  (both have concentrated, non-flat spectra) -- flatness alone cannot tell speech and saturation
  apart, which is why `clip_fraction` exists as a second, independent signal.
* `clip_fraction`: fraction of samples with `abs(sample) > 0.99`. Near 0 for real speech (which has
  a healthy crest factor -- loud syllables and quiet gaps); empirically 0.75-0.81 for both of this
  task's v10 checkpoints' output vs. 0.0 for the validated control checkpoint (see
  `results/control_test.md`, `results/v10_diagnosis.md`).

Usage (importable)::

    from tasks.t0013_v10_synthesis_quality_forensics.code.audio_quality_check import (
        check_audio_quality,
    )
    result = check_audio_quality(Path("some.wav"))
    if result.is_likely_noise:
        ...

Usage (CLI)::

    uv run python tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py some.wav
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# Thresholds: a clip is flagged "likely noise" when EITHER (a) its spectral content is nearly flat
# (white-noise-like, no formant/harmonic structure) OR (b) most samples are clipped/saturated (a
# broken vocoder screaming at full scale, e.g. this task's v10 checkpoints) -- AND it is not mostly
# silence (silence is a separate failure mode, "no audio" not "audio that sounds like noise";
# callers should check silence_fraction on its own too).
SPECTRAL_FLATNESS_NOISE_THRESHOLD = 0.35
CLIP_FRACTION_NOISE_THRESHOLD = 0.3
SILENCE_FRAME_DBFS_THRESHOLD = -40.0
SILENCE_FRAME_SECONDS = 0.02
CLIP_SAMPLE_ABS_THRESHOLD = 0.99


@dataclass(frozen=True, slots=True)
class AudioQualityResult:
    rms: float
    peak: float
    silence_fraction: float
    spectral_flatness: float
    clip_fraction: float
    is_likely_noise: bool


def check_audio_quality(wav_path: Path) -> AudioQualityResult:
    y, sr = librosa.load(str(wav_path), sr=None)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if len(y) > 0 else 0.0
    peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0

    frame_len = max(1, int(SILENCE_FRAME_SECONDS * sr))
    n_frames = len(y) // frame_len
    silence_frames = 0
    for i in range(n_frames):
        frame = y[i * frame_len : (i + 1) * frame_len]
        frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) + 1e-9
        frame_dbfs = 20 * np.log10(frame_rms)
        if frame_dbfs < SILENCE_FRAME_DBFS_THRESHOLD:
            silence_frames += 1
    silence_fraction = silence_frames / n_frames if n_frames > 0 else 1.0

    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = float(flatness.mean()) if flatness.size > 0 else 1.0
    clip_fraction = float(np.mean(np.abs(y) > CLIP_SAMPLE_ABS_THRESHOLD)) if len(y) > 0 else 0.0

    not_silent = silence_fraction < 0.95
    is_likely_noise = not_silent and (
        spectral_flatness >= SPECTRAL_FLATNESS_NOISE_THRESHOLD
        or clip_fraction >= CLIP_FRACTION_NOISE_THRESHOLD
    )
    return AudioQualityResult(
        rms=rms,
        peak=peak,
        silence_fraction=silence_fraction,
        spectral_flatness=spectral_flatness,
        clip_fraction=clip_fraction,
        is_likely_noise=is_likely_noise,
    )


def demo() -> None:
    """Runnable self-check (ponytail: one assert-based check for this module's branch logic, not
    a full test suite) -- asserts the heuristic classifies a known-noise v10 clip (t0013's
    deliverable) and a known-good control clip (also t0013's) correctly. Reuses t0013's
    audio_samples/ rather than duplicating those WAVs into this task's folder (no data
    duplication across task folders, per CLAUDE.md Key Rule 8).
    """
    from tasks.t0013_v10_synthesis_quality_forensics.code.paths import (
        RESULTS_AUDIO_DIR as T0013_RESULTS_AUDIO_DIR,
    )

    control_wav = T0013_RESULTS_AUDIO_DIR / "control_epochs_2nd_00020.wav"
    v10_wav = T0013_RESULTS_AUDIO_DIR / "v10_epoch16_primary.wav"

    if not control_wav.exists() or not v10_wav.exists():
        print(
            "demo() requires t0013's results/audio_samples/ -- run `dvc pull` in that task first."
        )
        return

    control_result = check_audio_quality(control_wav)
    v10_result = check_audio_quality(v10_wav)
    print(f"control: {control_result}")
    print(f"v10 primary: {v10_result}")

    assert not control_result.is_likely_noise, (
        f"known-good control clip misclassified as noise: {control_result}"
    )
    assert v10_result.is_likely_noise, f"known-noise v10 clip misclassified as speech: {v10_result}"
    print("demo() OK: heuristic correctly separates the known-good control from known-noise v10.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    if args.wav_path is None:
        demo()
        return

    result = check_audio_quality(args.wav_path)
    print(result)


if __name__ == "__main__":
    main()
