"""Build the two reference-audio conditions (`ref_single`, `ref_concat`) for zero-shot cloning.

Both conditions are drawn only from half-A of the ElevenLabs David corpus, so the reference clips
fed to the cloning models and the clips used to build the scoring centroid are the same split
(REQ-2). Concatenation approach (0.2s silence gaps) follows the pattern in
`tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py`, adapted to source from
half-A and to target two different durations.

**Preflight-inspection deviation from `plan/plan.md` Step 3 (documented, not silent — implementation
SKILL.md Phase 1.5):** the plan specified `ref_single` as "the single longest clean half-A clip
under ~10s", expecting `ref_single_duration_s` in `[8, 12]`. Inspecting the real
`data/11labs_david/` corpus (1364 raw clips) shows every clip is a short single phrase: durations
range 0.51s-1.67s, with only 2 of 1364 clips at or above 1.6s and **none** anywhere near 10s. No
single raw clip can satisfy the plan's literal ~10s target. Rather than silently reporting a 1.67s
"ref_single" against a table/chart implying ~10s (which would misrepresent the `ref_single` vs
`ref_concat` duration-effect comparison, REQ-2/Key Question 4), `ref_single` is built the same way
`ref_concat` is built — a filename-sorted concatenation of half-A clips with 0.2s silence gaps,
stopping once total duration first exceeds a target — just with a ~10s target instead of ~30s. This
preserves the *intent* of REQ-2 (two reference-audio conditions differing in amount of reference
material, ~10s vs ~30s) using the same construction method for both, given the corpus's actual
composition. This deviation is recorded in `data/references/manifest.json`'s `ref_single_is_concat`
field and must be called out in `results/tables.json`, `results/listening_guide.md`, and the answer
asset's Q4 discussion — never silently absorbed into a chart/caption implying a literal single
utterance.

Usage::

    uv run python -u tasks/t0018_zero_shot_cloning_calibration/code/build_references.py
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from tasks.t0008_tts_eval_harness_baselines.code.constants import RANDOM_SEED, REFERENCE_HALF_SIZE
from tasks.t0018_zero_shot_cloning_calibration.code.constants import (
    REF_CONCAT_SILENCE_GAP_S,
    REF_CONCAT_TARGET_DURATION_S,
    REF_SINGLE_MAX_DURATION_S,
)
from tasks.t0018_zero_shot_cloning_calibration.code.paths import (
    DATA_11LABS_DAVID_DIR,
    DATA_REFERENCES_DIR,
    HALF_A_CENTROID_NPY,
    REF_CONCAT_WAV,
    REF_SINGLE_WAV,
    REFERENCES_MANIFEST,
    RESULTS_AUDIO_REFERENCES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReferenceManifest:
    ref_single_filenames: list[str]
    ref_single_duration_s: float
    ref_single_is_concat: bool
    ref_single_deviation_note: str
    ref_concat_filenames: list[str]
    ref_concat_duration_s: float
    seed: int


def _build_half_a_centroid(half_a_paths: list[Path]) -> np.ndarray:
    """Build the half-A GE2E centroid the way t0008's actual, working entry point does.

    **Known t0008 discrepancy (discovered here, documented for the record — Key Rule 5 forbids
    modifying t0008's files, so this task works around it in its own code instead):**
    `tasks.t0008_tts_eval_harness_baselines.code.harness.build_reference_split` (the officially
    reusable library function, listed in the `tts_eval_harness` library's `entry_points`) delegates
    to `scoring.build_centroid`, which hard-skips any clip under `MIN_CLIP_DURATION_S=1.6s` before
    embedding it. The real `data/11labs_david/` corpus (1364 raw single-phrase clips, verified here
    via `soundfile.info`) has only 2 of 1364 clips at or above 1.6s — calling
    `build_reference_split` on this corpus raises `RuntimeError: No clips long enough to embed`.
    t0008's own committed results (`speaker_sim=0.832` for `elevenlabs_david_fillers`, etc.) were
    NOT produced via this reusable function; they were produced by
    `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py`'s own inline centroid-builder
    (lines ~93-107 of that file), which has no minimum-duration pre-filter and calls
    `encoder.embed_utterance` directly on every clip's `preprocess_wav` output (resemblyzer pads
    short clips automatically; comment in that file: "no minimum length needed"). This function
    replicates that proven-working approach exactly (same split, same resemblyzer calls), not the
    broken reusable one, so this task's centroid is built the same way t0008's actual numbers were.
    """
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(device="cpu")
    embeddings: list[np.ndarray] = []
    for wav_path in half_a_paths:
        try:
            wav = preprocess_wav(str(wav_path))
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
        except Exception as exc:
            logger.warning("Could not embed %s: %s", wav_path, exc)

    assert len(embeddings) >= 100, f"Only {len(embeddings)} valid centroid clips (half-A)"
    mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(mean_emb))
    centroid = (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)
    logger.info("half-A centroid built from %d/%d clips", len(embeddings), len(half_a_paths))
    return centroid


def _build_concat(
    half_a_paths: list[Path],
    *,
    target_duration_s: float,
    exclude_names: set[str],
) -> tuple[np.ndarray, int, list[str]]:
    """Concatenate half-A clips (filename-sorted) with silence gaps until > target duration."""
    sorted_paths = [
        p for p in sorted(half_a_paths, key=lambda p: p.name) if p.name not in exclude_names
    ]
    audio_parts: list[np.ndarray] = []
    used_names: list[str] = []
    total_duration_s = 0.0
    sample_rate: int | None = None

    for p in sorted_paths:
        data, sr = sf.read(str(p))
        if sample_rate is None:
            sample_rate = sr
        assert sr == sample_rate, f"Sample rate mismatch: {p} has {sr}, expected {sample_rate}"
        clip_duration_s = len(data) / sr
        audio_parts.append(data)
        audio_parts.append(np.zeros(int(REF_CONCAT_SILENCE_GAP_S * sr)))
        used_names.append(p.name)
        total_duration_s += clip_duration_s + REF_CONCAT_SILENCE_GAP_S
        if total_duration_s > target_duration_s:
            break

    assert sample_rate is not None, "No half-A clips found"
    concat = np.concatenate(audio_parts)
    return concat, sample_rate, used_names


def build_references() -> ReferenceManifest:
    """Build ref_single.wav, ref_concat.wav, and the half-A centroid; write manifest.json."""
    DATA_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_AUDIO_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    # Same 679/rest split t0008 uses (both its reusable harness.build_reference_split and its
    # actual working score_speaker_sim.py agree on this part: Random(seed).shuffle, first 679 =
    # half-A). See REFERENCE_HALF_SIZE in t0008's constants.py.
    all_wavs = sorted(DATA_11LABS_DAVID_DIR.glob("*.wav"))
    rng = random.Random(RANDOM_SEED)
    shuffled = all_wavs.copy()
    rng.shuffle(shuffled)
    half_a_paths = shuffled[:REFERENCE_HALF_SIZE]
    half_b_paths = shuffled[REFERENCE_HALF_SIZE:]
    logger.info("half-A: %d clips, half-B: %d clips", len(half_a_paths), len(half_b_paths))

    centroid = _build_half_a_centroid(half_a_paths)
    np.save(str(HALF_A_CENTROID_NPY), centroid)
    logger.info("Saved half-A centroid -> %s", HALF_A_CENTROID_NPY)

    # Preflight check (documented in module docstring): does a single raw half-A clip come close
    # to REF_SINGLE_MAX_DURATION_S? If not, build ref_single as a short concat instead.
    durations = [(p, float(sf.info(str(p)).duration)) for p in half_a_paths]
    longest_single = max(durations, key=lambda pd: pd[1])
    ref_single_is_concat = longest_single[1] < REF_SINGLE_MAX_DURATION_S * 0.5
    deviation_note = ""

    if ref_single_is_concat:
        deviation_note = (
            f"Corpus has no single half-A clip near {REF_SINGLE_MAX_DURATION_S:.0f}s "
            f"(longest half-A clip = {longest_single[1]:.2f}s, longest clip in the FULL 1364-clip "
            "corpus is 1.67s). ref_single was built as a short concatenation (same method as "
            "ref_concat, ~10s target instead of ~30s) rather than a single natural utterance, per "
            "the documented deviation in build_references.py. Do not present this as a literal "
            "single utterance in any table/chart caption."
        )
        logger.warning(deviation_note)
        ref_single_data, ref_single_sr, ref_single_filenames = _build_concat(
            half_a_paths, target_duration_s=REF_SINGLE_MAX_DURATION_S, exclude_names=set()
        )
        ref_single_duration_s = len(ref_single_data) / ref_single_sr
    else:
        ref_single_filenames = [longest_single[0].name]
        ref_single_data, ref_single_sr = sf.read(str(longest_single[0]))
        ref_single_duration_s = longest_single[1]

    sf.write(str(REF_SINGLE_WAV), ref_single_data, ref_single_sr, subtype="FLOAT")
    sf.write(
        str(RESULTS_AUDIO_REFERENCES_DIR / "ref_single.wav"),
        ref_single_data,
        ref_single_sr,
        subtype="FLOAT",
    )
    logger.info(
        "ref_single: %d clip(s), %.2fs -> %s",
        len(ref_single_filenames),
        ref_single_duration_s,
        REF_SINGLE_WAV,
    )

    # ref_concat excludes any clips already consumed by ref_single so the two conditions never
    # share source material (keeps the duration-effect comparison clean).
    ref_concat_data, ref_concat_sr, ref_concat_filenames = _build_concat(
        half_a_paths,
        target_duration_s=REF_CONCAT_TARGET_DURATION_S,
        exclude_names=set(ref_single_filenames),
    )
    ref_concat_duration_s = len(ref_concat_data) / ref_concat_sr
    sf.write(str(REF_CONCAT_WAV), ref_concat_data, ref_concat_sr, subtype="FLOAT")
    sf.write(
        str(RESULTS_AUDIO_REFERENCES_DIR / "ref_concat.wav"),
        ref_concat_data,
        ref_concat_sr,
        subtype="FLOAT",
    )
    logger.info(
        "ref_concat: %d clips, %.2fs -> %s",
        len(ref_concat_filenames),
        ref_concat_duration_s,
        REF_CONCAT_WAV,
    )

    manifest = ReferenceManifest(
        ref_single_filenames=ref_single_filenames,
        ref_single_duration_s=ref_single_duration_s,
        ref_single_is_concat=ref_single_is_concat,
        ref_single_deviation_note=deviation_note,
        ref_concat_filenames=ref_concat_filenames,
        ref_concat_duration_s=ref_concat_duration_s,
        seed=RANDOM_SEED,
    )
    REFERENCES_MANIFEST.write_text(
        json.dumps(
            {
                "ref_single_filenames": manifest.ref_single_filenames,
                "ref_single_duration_s": manifest.ref_single_duration_s,
                "ref_single_is_concat": manifest.ref_single_is_concat,
                "ref_single_deviation_note": manifest.ref_single_deviation_note,
                "ref_concat_filenames": manifest.ref_concat_filenames,
                "ref_concat_duration_s": manifest.ref_concat_duration_s,
                "seed": manifest.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved manifest -> %s", REFERENCES_MANIFEST)
    return manifest


if __name__ == "__main__":
    m = build_references()
    assert 8.0 <= m.ref_single_duration_s <= 12.0, (
        f"ref_single duration out of range: {m.ref_single_duration_s}"
    )
    assert 28.0 <= m.ref_concat_duration_s <= 34.0, (
        f"ref_concat duration out of range: {m.ref_concat_duration_s}"
    )
    print("OK:", m)
