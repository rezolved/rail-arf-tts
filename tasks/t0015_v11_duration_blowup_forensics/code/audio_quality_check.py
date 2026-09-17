"""Reusable "does this sound like noise?" regression check for the project's eval-harness pipeline.

Copied forward (paths only) from `tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py`,
itself copied unchanged (logic-for-logic) from
`tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py`. This task
(t0015, Milestone D step 10, REQ-8) extends the module in place -- same module, one source of truth,
per this module's own established design history below -- with two new signals that close the exact
blind spot this task exists to fix: `kokoro-v11-best`'s 73.95s synthesis output for a ~10-word test
sentence passed the original 3-signal gate (`is_likely_noise=False`) yet a human listener confirmed
it is droning babble, not intelligible speech.

Four signals now distinguish "noise"/"not real speech" from genuine, well-formed speech cheaply.
This project's history so far, in the order each blind spot was found and closed:

* `silence_fraction`: fraction of 20ms frames below -40dBFS. Near-silent output (a crashed/
  degenerate synth) has this close to 1.0.
* `spectral_flatness`: geometric-mean / arithmetic-mean of the power spectrum, per frame (Wiener
  entropy). Near 1.0 for white-noise-like signals (energy spread evenly across frequencies, no
  formant structure); well below 1.0 for voiced speech OR for a saturated/DC-dominated signal
  (both have concentrated, non-flat spectra) -- flatness alone cannot tell speech and saturation
  apart, which is why `clip_fraction` exists as a second, independent signal.
* `clip_fraction`: fraction of samples with `abs(sample) > 0.99`. Near 0 for real speech (which has
  a healthy crest factor -- loud syllables and quiet gaps); empirically 0.75-0.81 for t0013's v10
  checkpoints' output vs. 0.0 for the validated control checkpoint. Added after `spectral_flatness`
  alone missed v10's rail-to-rail clipped/saturated failure mode (t0013).
* `duration_sanity_pass` / `longest_nonsilent_run_s` (t0015, this task): v11's failure mode is
  neither near-silent, near-flat-spectrum, nor clipped -- every 3-second window of its 73.9s output
  has speech-like *local* spectral texture and near-zero clipping, exactly what the first three
  signals check for. What it lacks is silence gaps between words/phrases: real speech for a short
  sentence has pauses; this signal never drops in level for 74 continuous seconds.
  `duration_sanity_pass` (synthesized duration vs. a generous multiple of a naive words-per-second
  estimate for the input text) and `longest_nonsilent_run_s` (max contiguous non-silent run,
  distinct from aggregate `silence_fraction`) are the two cheap, fast checks that catch this class
  of defect.

Design choice: NEITHER new signal is folded into `is_likely_noise`, which stays exactly the
original 3-signal (silence/flatness/clip) computation on purpose -- this preserves backward
compatibility for every existing caller that only checks `is_likely_noise` (it keeps returning
exactly what it always has, on exactly the same inputs) and is what makes the three-way regression
(`results/gate_regression.json`, Milestone D step 12) a real proof: `v11_best.wav` reads
`is_likely_noise=False` even after hardening (the old gate genuinely would still ship it), while
the two NEW fields below (combined by the caller into an overall pass/fail, e.g.
`code/run_gate_regression.py`'s `hardened_gate_pass`) are what catch it.
`duration_sanity_pass` requires the caller to pass `text` (the source text has no audio-only
equivalent), so it is `None` when `text` is not supplied. `longest_nonsilent_run_s` needs no extra
input -- it is computed from the audio alone, in the same 20ms-frame loop that already computes
`silence_fraction` -- and is always a concrete float.

Usage (importable)::

    from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import (
        check_audio_quality,
    )
    result = check_audio_quality(Path("some.wav"), text="the original input text")
    hardened_gate_pass = not (
        result.is_likely_noise
        or result.duration_sanity_pass is False
        or result.longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S
    )
    if not hardened_gate_pass:
        ...

Usage (CLI)::

    uv run python tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py \\
        some.wav --text "..."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

# Thresholds: a clip is flagged "likely noise" (`is_likely_noise`, unchanged 3-signal computation)
# when EITHER (a) its spectral content is nearly flat (white-noise-like, no formant/harmonic
# structure) OR (b) most samples are clipped/saturated (a broken vocoder screaming at full scale,
# e.g. t0013's v10 checkpoints) -- AND it is not mostly silence (silence is a separate failure
# mode, "no audio" not "audio that sounds like noise";
# callers should check silence_fraction on its own too).
SPECTRAL_FLATNESS_NOISE_THRESHOLD = 0.35
CLIP_FRACTION_NOISE_THRESHOLD = 0.3
SILENCE_FRAME_DBFS_THRESHOLD = -40.0
SILENCE_FRAME_SECONDS = 0.02
CLIP_SAMPLE_ABS_THRESHOLD = 0.99

# t0015 Milestone D step 10 (REQ-8) additions.
#
# NAIVE_MIN_WORDS_PER_SECOND: a deliberately SLOW floor (90 words/minute -- well below typical
# conversational English at 120-170 wpm) chosen to minimize false positives on genuinely slow,
# deliberate filler narration. `estimate_naive_duration_bound_s` divides word count by this floor,
# so it returns a generously LONG "reasonable" duration -- multiplying by DURATION_SANITY_MULTIPLIER
# on top of that gives a lot of headroom before a real (if unusually slow) narration would trip it.
NAIVE_MIN_WORDS_PER_SECOND = 1.5
# Per task_description.md's own suggested "~3x a reasonable upper bound".
DURATION_SANITY_MULTIPLIER = 3.0
# A flat, generous ceiling -- longer than any single breath group/phrase in normal speech (typical
# spoken phrases rarely run past a few seconds without a pause). Intentionally generous to avoid
# false positives; a text-length-scaled threshold is a possible future refinement, not implemented
# here to keep this signal simple and cheap.
LONGEST_NONSILENT_RUN_THRESHOLD_S = 12.0


@dataclass(frozen=True, slots=True)
class AudioQualityResult:
    rms: float
    peak: float
    silence_fraction: float
    spectral_flatness: float
    clip_fraction: float
    longest_nonsilent_run_s: float
    duration_sanity_pass: bool | None
    is_likely_noise: bool


def estimate_naive_duration_bound_s(text: str) -> float:
    """A generous, naive upper bound on how long `text` should take to speak, in seconds."""
    return len(text.split()) / NAIVE_MIN_WORDS_PER_SECOND


def check_audio_quality(wav_path: Path, *, text: str | None = None) -> AudioQualityResult:
    y, sr = librosa.load(str(wav_path), sr=None)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if len(y) > 0 else 0.0
    peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0

    frame_len = max(1, int(SILENCE_FRAME_SECONDS * sr))
    n_frames = len(y) // frame_len
    silence_frames = 0
    longest_nonsilent_run_frames = 0
    current_nonsilent_run_frames = 0
    for i in range(n_frames):
        frame = y[i * frame_len : (i + 1) * frame_len]
        frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) + 1e-9
        frame_dbfs = 20 * np.log10(frame_rms)
        if frame_dbfs < SILENCE_FRAME_DBFS_THRESHOLD:
            silence_frames += 1
            current_nonsilent_run_frames = 0
        else:
            current_nonsilent_run_frames += 1
            longest_nonsilent_run_frames = max(
                longest_nonsilent_run_frames, current_nonsilent_run_frames
            )
    silence_fraction = silence_frames / n_frames if n_frames > 0 else 1.0
    longest_nonsilent_run_s = longest_nonsilent_run_frames * frame_len / sr

    flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = float(flatness.mean()) if flatness.size > 0 else 1.0
    clip_fraction = float(np.mean(np.abs(y) > CLIP_SAMPLE_ABS_THRESHOLD)) if len(y) > 0 else 0.0

    duration_sanity_pass: bool | None = None
    if text is not None:
        duration_sanity_bound_s = estimate_naive_duration_bound_s(text) * DURATION_SANITY_MULTIPLIER
        actual_duration_s = len(y) / sr if sr > 0 else 0.0
        duration_sanity_pass = actual_duration_s <= duration_sanity_bound_s

    # `is_likely_noise` intentionally stays exactly the original 3-signal computation (silence/
    # flatness/clip only) -- it is NOT extended to fold in `longest_nonsilent_run_s` here, even
    # though it is a new signal this task adds. Reasoning: v11's `v11_best.wav` fixture must
    # continue to read `is_likely_noise=False` post-hardening (this is what makes the three-way
    # regression (Step 12) a real proof that the OLD 3-signal gate still would have shipped this
    # clip -- the new signals are what catch it, not a silently-redefined `is_likely_noise`).
    # Callers combine `is_likely_noise` with the two new fields into an overall pass/fail decision
    # (see `code/run_gate_regression.py`'s `hardened_gate_pass`) rather than this function folding
    # them together, so existing callers that only check `is_likely_noise` keep their exact prior
    # behavior -- a second backward-compatibility guarantee alongside `duration_sanity_pass`'s.
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
        longest_nonsilent_run_s=longest_nonsilent_run_s,
        duration_sanity_pass=duration_sanity_pass,
        is_likely_noise=is_likely_noise,
    )


def demo() -> None:
    """Runnable self-check (ponytail: one assert-based check for this module's branch logic, not
    a full test suite) -- asserts the heuristic classifies a known-noise v10 clip and a known-good
    control clip correctly (t0013's deliverables), reusing t0013's `results/audio_samples/` rather
    than duplicating those WAVs into this task's folder (no data duplication across task folders,
    per CLAUDE.md Key Rule 8). For the new duration/silence-gap signals' discriminating proof
    against v11's own blowup, see `results/gate_regression.json` (Milestone D step 12) instead of
    this demo -- that is the reproducible three-way regression this task's REQ-10 requires.
    """
    from tasks.t0015_v11_duration_blowup_forensics.code.paths import (
        T0013_CONTROL_WAV,
        V10_EPOCH16_PRIMARY_WAV,
    )

    if not T0013_CONTROL_WAV.exists() or not V10_EPOCH16_PRIMARY_WAV.exists():
        print(
            "demo() requires t0013's results/audio_samples/ -- run `dvc pull` in that task first."
        )
        return

    control_result = check_audio_quality(T0013_CONTROL_WAV)
    v10_result = check_audio_quality(V10_EPOCH16_PRIMARY_WAV)
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
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    if args.wav_path is None:
        demo()
        return

    result = check_audio_quality(args.wav_path, text=args.text)
    print(result)


if __name__ == "__main__":
    main()
