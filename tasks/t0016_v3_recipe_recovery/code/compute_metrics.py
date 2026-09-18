"""Milestone 5 Step 19 (applicable-metrics coverage): cross-check speaker_sim/rtf/ttfb_ms against
t0008's recorded numbers and write the legacy-flat `results/metrics.json`.

Reuses the registered `tts_eval_harness` library (t0008_tts_eval_harness_baselines, v0.1.0):
`code/scoring.py::compute_speaker_sim` for the actual GE2E cosine scoring against a centroid
(library import, not a copy, per Critical Rule 8 -- `tts_eval_harness` is a registered library
asset at `tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/`).

**Deviation from the library's own `build_centroid()`, documented rather than silent**: this
task's `data/11labs_david/` corpus (1364 clips, pulled fresh via `dvc pull` in this task) has a
mean duration of 1.04s and a max of 1.67s (measured directly, see
`## Metrics Cross-Check` in `results/v3_checkpoint_forensics.md`) -- almost the entire corpus
falls under `scoring.py`'s own `MIN_CLIP_DURATION_S = 1.6` pre-filter, so calling
`build_centroid()` unmodified raises `RuntimeError: No clips long enough to embed` (verified: 1364/
1364 skipped). `resemblyzer.VoiceEncoder.embed_utterance()` itself works fine on clips well under
1.6s (verified directly: a 0.84s post-VAD-trim clip embeds without error) -- `MIN_CLIP_DURATION_S`
is this project's own conservative extra safety filter in `scoring.py`, not a resemblyzer
requirement. Since `t0008`'s own file is a completed task's file (Key Rule 5: nothing in a
completed task folder may be changed) and copying `scoring.py::build_centroid` in full would
duplicate ~30 lines for one filter difference, `build_local_centroid()` below reimplements only
the centroid-averaging loop directly against `resemblyzer`'s own (third-party, not project) API,
with no minimum-length pre-filter, and hands the resulting plain `(256,) ndarray` to the library's
own unmodified `compute_speaker_sim()` for the actual scoring -- `compute_speaker_sim()` accepts
any centroid array regardless of how it was built.

Timing (`ttfb_ms`, `rtf`) is read back from the print output of
`synthesize_v3_shipped.py`'s run (captured in this task's own command logs) rather than
re-synthesizing, since the WAV files it already wrote are the exact same artifacts.

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.compute_metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_speaker_sim
from tasks.t0016_v3_recipe_recovery.code.paths import (
    CHECKPOINT_FORENSICS_MD,
    ELEVENLABS_DAVID_DIR,
    METRICS_JSON,
    RESULTS_AUDIO_V3_SHIPPED_DIR,
)


def build_local_centroid(wav_paths: list[Path]) -> tuple[np.ndarray, int, int]:
    """Mean GE2E embedding over `wav_paths`, with no minimum-clip-length pre-filter.

    Returns (centroid, n_embedded, n_skipped). See module docstring for why this does not reuse
    `tasks.t0008_tts_eval_harness_baselines.code.scoring.build_centroid` unmodified.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(device="cpu")
    embeddings: list[np.ndarray] = []
    skipped = 0
    for wav_path in wav_paths:
        try:
            wav = preprocess_wav(str(wav_path))
            if len(wav) == 0:
                skipped += 1
                continue
            embeddings.append(encoder.embed_utterance(wav))
        except Exception as exc:  # noqa: BLE001 -- matches scoring.py's own broad catch-and-skip
            print(f"Could not embed {wav_path}: {exc}")
            skipped += 1
    if len(embeddings) == 0:
        raise RuntimeError(f"No clips could be embedded (skipped={skipped})")
    centroid = np.mean(np.stack(embeddings), axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid, len(embeddings), skipped


# Source: tasks/t0008_tts_eval_harness_baselines/results/metrics.json (t0008's own recorded
# numbers for the shipped v3 bundle, `kokoro_v3_bundle` system) -- read directly, not from memory.
T0008_SPEAKER_SIM_FILLERS = 0.631
T0008_SPEAKER_SIM_VAL96 = 0.588
SPEAKER_SIM_TOLERANCE = 0.02

# Per-clip timings captured directly from `synthesize_v3_shipped.py`'s own printed output
# (`logs/commands/` for this task has the raw run) -- not re-measured here, since the WAV files
# written by that run are the exact metrics artifacts being scored below.
TTFB_S_BY_SLUG: dict[str, float] = {
    "lining_up_suggestions_17": 4.741,
    "lining_up_suggestions_10": 3.539,
    "putting_them_head_to_head_15": 4.426,
    "val96_seed42_00": 5.685,
    "val96_seed42_01": 5.890,
    "val96_seed42_02": 4.890,
    "val96_seed42_03": 3.855,
    "val96_seed42_04": 17.766,
}
RTF_BY_SLUG: dict[str, float] = {
    "lining_up_suggestions_17": 2.107,
    "lining_up_suggestions_10": 1.887,
    "putting_them_head_to_head_15": 2.299,
    "val96_seed42_00": 1.944,
    "val96_seed42_01": 2.014,
    "val96_seed42_02": 1.956,
    "val96_seed42_03": 1.590,
    "val96_seed42_04": 1.692,
}


def main() -> None:
    synth_wavs = sorted(RESULTS_AUDIO_V3_SHIPPED_DIR.glob("*.wav"))
    assert len(synth_wavs) > 0, f"No synthesized WAVs found in {RESULTS_AUDIO_V3_SHIPPED_DIR}"

    ref_wavs = sorted(ELEVENLABS_DAVID_DIR.glob("*.wav"))
    assert len(ref_wavs) > 0, f"No ElevenLabs reference WAVs found in {ELEVENLABS_DAVID_DIR}"

    print(f"Building GE2E centroid from {len(ref_wavs)} ElevenLabs reference clips...")
    centroid, n_embedded, n_skipped = build_local_centroid(ref_wavs)
    print(f"Centroid built from {n_embedded} clips ({n_skipped} skipped/failed).")

    print(f"Scoring {len(synth_wavs)} synthesized clips against the reference centroid...")
    result = compute_speaker_sim(synth_wavs, centroid)
    print(f"speaker_sim mean={result.mean:.4f} std={result.std:.4f} skipped={result.skipped_count}")

    mean_ttfb_s = sum(TTFB_S_BY_SLUG.values()) / len(TTFB_S_BY_SLUG)
    mean_rtf = sum(RTF_BY_SLUG.values()) / len(RTF_BY_SLUG)

    metrics = {
        "speaker_sim": round(result.mean, 4),
        "rtf": round(mean_rtf, 4),
        "ttfb_ms": round(mean_ttfb_s * 1000.0, 1),
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {METRICS_JSON}: {metrics}")

    within_tolerance_fillers = abs(result.mean - T0008_SPEAKER_SIM_FILLERS) <= SPEAKER_SIM_TOLERANCE
    consistency_note = (
        f"## Metrics Cross-Check (Milestone 5, applicable-metrics coverage)\n\n"
        f"**Centroid-building deviation (documented, not silent)**: this task's freshly-`dvc "
        f"pull`ed `data/11labs_david/` corpus (1364 clips) measures a mean duration of ~1.04s and "
        f"a max of ~1.67s -- almost the entire corpus falls under "
        f"`tasks/t0008_tts_eval_harness_baselines/code/scoring.py`'s own `MIN_CLIP_DURATION_S = "
        f"1.6` pre-filter, so calling that module's `build_centroid()` unmodified on this corpus "
        f"raises `RuntimeError: No clips long enough to embed (skipped=1364, min_duration=1.6s)` "
        f"-- verified directly. Since `resemblyzer.VoiceEncoder.embed_utterance()` itself embeds "
        f"short clips without error (verified: a 0.84s post-VAD-trim clip embeds fine) and t0008's "
        f"own completed-task file cannot be modified (Key Rule 5), `code/compute_metrics.py`'s "
        f"`build_local_centroid()` reimplements the same GE2E-mean-embedding logic directly "
        f"against `resemblyzer`'s own API with no minimum-length pre-filter, then hands the "
        f"resulting centroid to the library's own unmodified `compute_speaker_sim()`. This "
        f"centroid was built from {n_embedded} of {len(ref_wavs)} reference clips "
        f"({n_skipped} failed to embed for other reasons). This is itself a reproducibility "
        f"finding worth flagging upstream: re-running t0008's own `build_reference_split()` "
        f"today, unmodified, against the current `data/11labs_david/` would also fail with the "
        f"same error, which is inconsistent with t0008's own recorded success on this corpus -- "
        f"either the corpus changed since t0008 ran, or `MIN_CLIP_DURATION_S` was not actually "
        f"binding in t0008's execution environment for a reason not identified here.\n\n"
        f"This task's own local CPU re-synthesis of the shipped v3 bundle scores "
        f"`speaker_sim={result.mean:.4f}` (mean over {len(synth_wavs)} clips: 3 fixed gate texts "
        f"+ 5 seed-42 val96 prompts, {result.skipped_count} skipped) against this locally-built "
        f"ElevenLabs reference centroid. t0008's own recorded numbers for the same bundle "
        f"(`tasks/t0008_tts_eval_harness_baselines/results/metrics.json`) are "
        f"`speaker_sim={T0008_SPEAKER_SIM_FILLERS}` (fillers) / `{T0008_SPEAKER_SIM_VAL96}` "
        f"(val96). This run is "
        f"{'CONSISTENT' if within_tolerance_fillers else 'NOT consistent'} with the fillers number "
        f"within a ±0.02 tolerance "
        f"(chosen as well above expected floating-point/CPU-vs-original-hardware nondeterminism "
        f"for a deterministic decode), "
        f"{
            'so no further investigation is needed for the multispeaker/module-diff conclusions '
            'above.'
            if within_tolerance_fillers
            else 'which given the centroid-building deviation documented above is only weak '
            'corroborating evidence either way -- treat this task audio-quality conclusions '
            '(Milestones 2-3) as resting primarily on the checkpoint-shape forensics and the '
            'audio-quality gate, not on this speaker_sim cross-check.'
        } "
        f"`rtf` (mean {mean_rtf:.3f}) and `ttfb_ms` (mean {mean_ttfb_s * 1000.0:.1f}) are measured "
        f"on this workstation's CPU, not t0008's original (possibly GPU-backed) hardware, so they "
        f"are recorded for completeness and are not expected to match t0008's numbers -- higher "
        f"RTF/TTFB here is not a regression finding.\n"
    )
    existing = CHECKPOINT_FORENSICS_MD.read_text()
    CHECKPOINT_FORENSICS_MD.write_text(existing.rstrip() + "\n\n" + consistency_note)
    print(f"Appended metrics cross-check section to {CHECKPOINT_FORENSICS_MD}")


if __name__ == "__main__":
    main()
