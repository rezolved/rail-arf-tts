"""Reusable "does this sound like noise?" regression check for the project's eval-harness pipeline.

**Copied directly (not imported) from
`tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check`** per
`research/research_code.md` and `plan/plan.md`'s "Code to copy" list: t0018's own cross-task
import of this file
(`from tasks.t0015_v11_duration_blowup_forensics.code.audio_quality_check import (...)`) is no
longer valid under the current spec (t0015 is not a registered library), so this task keeps its own
copy instead of repeating that now-invalid pattern.

Original module history/design rationale (preserved verbatim from t0015; not this task's own
authorship):

Copied forward (paths only) from `tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py`,
itself copied unchanged (logic-for-logic) from
`tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py`. t0015 (Milestone D step
10, REQ-8) extended the module in place -- same module, one source of truth, per this module's own
established design history below -- with two new signals that close the exact blind spot that task
existed to fix: `kokoro-v11-best`'s 73.95s synthesis output for a ~10-word test sentence passed the
original 3-signal gate (`is_likely_noise=False`) yet a human listener confirmed it is droning
babble, not intelligible speech.

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
* `duration_sanity_pass` / `longest_nonsilent_run_s` (t0015): v11's failure mode is neither
  near-silent, near-flat-spectrum, nor clipped -- every 3-second window of its 73.9s output has
  speech-like *local* spectral texture and near-zero clipping, exactly what the first three signals
  check for. What it lacks is silence gaps between words/phrases: real speech for a short sentence
  has pauses; this signal never drops in level for 74 continuous seconds. `duration_sanity_pass`
  (synthesized duration vs. a generous multiple of a naive words-per-second estimate for the input
  text) and `longest_nonsilent_run_s` (max contiguous non-silent run, distinct from aggregate
  `silence_fraction`) are the two cheap, fast checks that catch this class of defect.

**t0021 caveat (owner correction, REQ-17):** this gate is known to pass clips a human listener
would describe as "voice plus strong noise" -- a PASS here is necessary, not sufficient, for a
production-quality claim. See `results/listening_guide.md`'s header and
`intervention/owner_correction_wrong_david_voice.md`.

Usage (importable)::

    from tasks.t0021_zero_shot_latency_reduction.code.audio_quality_check import (
        check_audio_quality,
    )
    result = check_audio_quality(Path("some.wav"), text="the original input text")
    hardened_gate_pass = not (
        result.is_likely_noise
        or result.duration_sanity_pass is False
        or result.longest_nonsilent_run_s > LONGEST_NONSILENT_RUN_THRESHOLD_S
    )

Usage (CLI)::

    uv run python tasks/t0021_zero_shot_latency_reduction/code/audio_quality_check.py \\
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
# mode, "no audio" not "audio that sounds like noise"; callers should check silence_fraction too).
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
# Per t0015's task_description.md's own suggested "~3x a reasonable upper bound".
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
    # flatness/clip only) -- see module docstring for the backward-compatibility rationale.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    result = check_audio_quality(args.wav_path, text=args.text)
    print(result)


if __name__ == "__main__":
    main()
