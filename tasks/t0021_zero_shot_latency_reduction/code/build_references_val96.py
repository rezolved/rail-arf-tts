"""Build `ref_single`/`ref_concat` and the corrected speaker-sim centroid from `data/v4/val/wavs`.

**Owner correction (binding, see `checkpoint.md` and `plan/plan.md`'s "Owner Correction"
section):** `data/11labs_david` (used by t0008 and therefore by t0018's references/centroid) is the
WRONG ElevenLabs "David" voice. From this task onward, every reference clip and every
speaker-similarity centroid is built from `data/v4/val/wavs` (val_96, the correct production
voice), never from `data/11labs_david`.

**Disjoint-halves design (deliberate change from t0018):** t0018 built both its centroid and its
references from the *same* half-A split. The owner correction (item 2) explicitly requires the
centroid to come from "the half not used for references." This script therefore splits the 96
val_96 clips into two disjoint 48/48 halves: `ref_source_half` (references drawn from here) and
`centroid_half` (corrected centroid built from here). Construction *method* (filename-sorted
concatenation, 0.2s silence gaps, seed=42, first-N-until-target-duration) is identical to t0018's
`_build_concat()` — only the source corpus and the half-disjointness differ (REQ-10 ambiguity
resolution, documented in `plan/plan.md`).

**Preflight finding (Phase 1.5, documented, not silent):** `soundfile.info()` over all 96
`data/v4/val/wavs/*.wav` files shows durations ranging 1.21s-15.0s, mean 4.33s (96 clips, ~415.4s
total) — short filler-style phrases, similar in character to the ElevenLabs filler corpus t0018
used, NOT long sentences. A 48-clip half sums to roughly half of 415.4s (~200s+), far more than the
29.5s `ref_concat` target needs. No single clip in `ref_source_half` is expected to be close enough
to the ~10s `REF_SINGLE_MAX_DURATION_S` target to use directly (mirroring t0018's own finding for
its corpus); this is verified explicitly below and the actual decision is recorded in
`ref_single_is_concat`, never assumed.

Usage::

    uv run python -u tasks/t0021_zero_shot_latency_reduction/code/build_references_val96.py
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from tasks.t0008_tts_eval_harness_baselines.code.constants import RANDOM_SEED
from tasks.t0021_zero_shot_latency_reduction.code.constants import (
    CENTROID_HALF_SIZE,
    REF_CONCAT_HARD_CEILING_S,
    REF_CONCAT_SILENCE_GAP_S,
    REF_CONCAT_TARGET_DURATION_S,
    REF_SINGLE_MAX_DURATION_S,
)
from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    DATA_REFERENCES_DIR,
    DATA_V4_VAL_WAVS_DIR,
    OLD_WRONGVOICE_CENTROID_NPY,
    REF_CONCAT_WAV,
    REF_SINGLE_WAV,
    REFERENCES_MANIFEST,
    RESULTS_AUDIO_REFERENCES_DIR,
    T0018_HALF_A_CENTROID_NPY,
    VAL96_CENTROID_NPY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReferenceManifestV96:
    source_corpus: str
    ref_single_filenames: list[str]
    ref_single_duration_s: float
    ref_single_is_concat: bool
    ref_single_deviation_note: str
    ref_concat_filenames: list[str]
    ref_concat_duration_s: float
    centroid_half_filenames: list[str]
    seed: int
    old_control_centroid_source: str


def _build_concat(
    paths: list[Path],
    *,
    target_duration_s: float,
    exclude_names: set[str],
    hard_ceiling_s: float | None = None,
) -> tuple[np.ndarray, int, list[str]]:
    """Concatenate clips (filename-sorted) with silence gaps until total > target duration.

    Base method (no `hard_ceiling_s`) matches
    `tasks.t0018_zero_shot_cloning_calibration.code.build_references._build_concat` (copied
    verbatim, not imported — t0018 is not a registered library): keep adding clips until the
    running total first EXCEEDS `target_duration_s`, then stop.

    **`hard_ceiling_s` (new in this task, not in t0018's version):** t0018's "exceeds target, then
    stop" rule can overshoot by up to one clip's duration. That is exactly how S-0018-02 happened
    (target 30.0s -> actual 30.57s, over CosyVoice2's internal 30.0s hard assertion). Naively
    lowering the target to 29.5s does not fix this by itself: this task's own preflight run against
    val_96's real (longer) clips overshot to 31.16s, still over 30s. When `hard_ceiling_s` is set,
    this function does a look-ahead check and refuses to add a clip that would push the running
    total past it, even if `target_duration_s` has not yet been reached — staying safely under the
    hard ceiling always wins over hitting the soft target exactly.
    """
    sorted_paths = [p for p in sorted(paths, key=lambda p: p.name) if p.name not in exclude_names]
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
        prospective_total_s = total_duration_s + clip_duration_s + REF_CONCAT_SILENCE_GAP_S

        if hard_ceiling_s is not None and prospective_total_s > hard_ceiling_s:
            # Adding this clip would breach the hard ceiling — stop here, even if we have not
            # yet reached target_duration_s. Safety (staying under the ceiling) wins.
            break

        audio_parts.append(data)
        audio_parts.append(np.zeros(int(REF_CONCAT_SILENCE_GAP_S * sr)))
        used_names.append(p.name)
        total_duration_s = prospective_total_s

        if total_duration_s > target_duration_s:
            break

    assert sample_rate is not None, "No clips found to concatenate"
    assert len(used_names) > 0, "hard_ceiling_s rejected every clip - target/ceiling too tight"
    concat = np.concatenate(audio_parts)
    return concat, sample_rate, used_names


def _build_val96_centroid(centroid_paths: list[Path]) -> np.ndarray:
    """Build the corrected val_96 centroid, preferring t0008's reusable library function.

    val_96 clips average ~4.3s (preflight-verified), well above t0008's
    `MIN_CLIP_DURATION_S=1.6s` floor that broke this function on t0018's short-phrase
    `data/11labs_david` corpus (see t0018's `build_references.py` docstring). Try the reusable
    `build_centroid()` first (per `plan/plan.md` Step 2); fall back to the inline t0018-style
    approach only if it raises.
    """
    from tasks.t0008_tts_eval_harness_baselines.code.scoring import build_centroid

    try:
        centroid = build_centroid(centroid_paths)
        logger.info(
            "val96 centroid built via t0008 library build_centroid() (%d clips)",
            len(centroid_paths),
        )
        return centroid
    except RuntimeError as exc:
        logger.warning(
            "t0008 library build_centroid() failed (%s); falling back to inline approach", exc
        )

    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder(device="cpu")
    embeddings: list[np.ndarray] = []
    for wav_path in centroid_paths:
        try:
            wav = preprocess_wav(str(wav_path))
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
        except Exception as exc:  # noqa: BLE001 - mirrors t0018's own broad catch, documented
            logger.warning("Could not embed %s: %s", wav_path, exc)

    assert len(embeddings) >= 20, f"Only {len(embeddings)} valid centroid clips (val96 half)"
    mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = float(np.linalg.norm(mean_emb))
    centroid = (mean_emb / norm if norm > 0 else mean_emb).astype(np.float32)
    logger.info(
        "val96 centroid built from %d/%d clips (inline fallback)",
        len(embeddings),
        len(centroid_paths),
    )
    return centroid


def build_references_val96() -> ReferenceManifestV96:
    """Build ref_single.wav, ref_concat.wav, the val96 centroid; copy the old control centroid."""
    DATA_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_AUDIO_REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    all_wavs = sorted(DATA_V4_VAL_WAVS_DIR.glob("*.wav"))
    assert len(all_wavs) == 96, f"Expected 96 val_96 clips, found {len(all_wavs)}"

    # Preflight duration check (Phase 1.5) — logged, not assumed.
    durations = [(p, float(sf.info(str(p)).duration)) for p in all_wavs]
    dur_values = [d for _, d in durations]
    logger.info(
        "val_96 duration preflight: n=%d min=%.2fs max=%.2fs mean=%.2fs total=%.2fs",
        len(dur_values),
        min(dur_values),
        max(dur_values),
        sum(dur_values) / len(dur_values),
        sum(dur_values),
    )

    rng = random.Random(RANDOM_SEED)
    shuffled = all_wavs.copy()
    rng.shuffle(shuffled)
    ref_source_half = shuffled[:CENTROID_HALF_SIZE]
    centroid_half = shuffled[CENTROID_HALF_SIZE:]
    assert len(ref_source_half) == 48
    assert len(centroid_half) == 48
    assert not set(p.name for p in ref_source_half) & set(p.name for p in centroid_half), (
        "ref_source_half and centroid_half must be disjoint (owner correction #2)"
    )
    logger.info(
        "ref_source_half: %d clips, centroid_half: %d clips (disjoint, seed=%d)",
        len(ref_source_half),
        len(centroid_half),
        RANDOM_SEED,
    )

    # ── Corrected val96 centroid (built from centroid_half, NOT ref_source_half) ──────────
    centroid = _build_val96_centroid(centroid_half)
    np.save(str(VAL96_CENTROID_NPY), centroid)
    logger.info("Saved corrected val96 centroid -> %s", VAL96_CENTROID_NPY)

    # ── Radiohost (wrong voice) control centroid: copy t0018's verbatim, never rebuild ─────
    shutil.copy2(str(T0018_HALF_A_CENTROID_NPY), str(OLD_WRONGVOICE_CENTROID_NPY))
    logger.info(
        "Copied t0018's half_a_centroid.npy (verbatim, read-only) -> %s",
        OLD_WRONGVOICE_CENTROID_NPY,
    )

    # ── ref_single: preflight-check whether a single natural clip is close to the target ──
    ref_source_durations = [(p, d) for p, d in durations if p in ref_source_half]
    longest_single = max(ref_source_durations, key=lambda pd: pd[1])
    ref_single_is_concat = longest_single[1] < REF_SINGLE_MAX_DURATION_S * 0.5
    deviation_note = ""

    if ref_single_is_concat:
        deviation_note = (
            f"ref_source_half has no single clip near {REF_SINGLE_MAX_DURATION_S:.0f}s "
            f"(longest clip in ref_source_half = {longest_single[1]:.2f}s, longest clip in the "
            f"FULL 96-clip val_96 corpus is {max(dur_values):.2f}s). ref_single was built as a "
            "short concatenation (same method as ref_concat, ~10s target instead of ~29.5s) "
            "rather than a single natural utterance, matching t0018's own documented deviation "
            "pattern for its corpus. Do not present this as a literal single utterance in any "
            "table/chart caption."
        )
        logger.warning(deviation_note)
        ref_single_data, ref_single_sr, ref_single_filenames = _build_concat(
            ref_source_half, target_duration_s=REF_SINGLE_MAX_DURATION_S, exclude_names=set()
        )
        ref_single_duration_s = len(ref_single_data) / ref_single_sr
    else:
        ref_single_filenames = [longest_single[0].name]
        ref_single_data, ref_single_sr = sf.read(str(longest_single[0]))
        ref_single_duration_s = longest_single[1]
        deviation_note = (
            f"ref_source_half's longest single clip ({longest_single[0].name}) is "
            f"{ref_single_duration_s:.2f}s, notably longer than the "
            f"{REF_SINGLE_MAX_DURATION_S:.0f}s "
            "target (unlike t0018's corpus, val_96 does contain a clip long enough to clear the "
            f"{REF_SINGLE_MAX_DURATION_S * 0.5:.1f}s single-clip-vs-concat threshold). Per "
            "plan.md Step 2's pre-registered rule, a single natural clip is preferred over a "
            "concatenation whenever one is available; the resulting ref_single is therefore a "
            f"real {ref_single_duration_s:.2f}s single utterance, not a ~10s one. Never caption "
            "ref_single as '~10s' in any table/chart; use the actual value from this manifest."
        )
        logger.warning(deviation_note)

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

    # ref_concat excludes any clips already consumed by ref_single (non-overlap rule, t0018-style).
    # hard_ceiling_s enforced (see _build_concat docstring / S-0018-02) so this can never again
    # exceed CosyVoice2's internal 30.0s assertion.
    ref_concat_data, ref_concat_sr, ref_concat_filenames = _build_concat(
        ref_source_half,
        target_duration_s=REF_CONCAT_TARGET_DURATION_S,
        exclude_names=set(ref_single_filenames),
        hard_ceiling_s=REF_CONCAT_HARD_CEILING_S,
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

    manifest = ReferenceManifestV96(
        source_corpus="data/v4/val/wavs",
        ref_single_filenames=ref_single_filenames,
        ref_single_duration_s=ref_single_duration_s,
        ref_single_is_concat=ref_single_is_concat,
        ref_single_deviation_note=deviation_note,
        ref_concat_filenames=ref_concat_filenames,
        ref_concat_duration_s=ref_concat_duration_s,
        centroid_half_filenames=sorted(p.name for p in centroid_half),
        seed=RANDOM_SEED,
        old_control_centroid_source=(
            "tasks/t0018_zero_shot_cloning_calibration/data/references/half_a_centroid.npy"
        ),
    )
    REFERENCES_MANIFEST.write_text(
        json.dumps(
            {
                "source_corpus": manifest.source_corpus,
                "ref_single_filenames": manifest.ref_single_filenames,
                "ref_single_duration_s": manifest.ref_single_duration_s,
                "ref_single_is_concat": manifest.ref_single_is_concat,
                "ref_single_deviation_note": manifest.ref_single_deviation_note,
                "ref_concat_filenames": manifest.ref_concat_filenames,
                "ref_concat_duration_s": manifest.ref_concat_duration_s,
                "centroid_half_filenames": manifest.centroid_half_filenames,
                "seed": manifest.seed,
                "old_control_centroid_source": manifest.old_control_centroid_source,
                "val_96_duration_preflight": {
                    "n": len(dur_values),
                    "min_s": min(dur_values),
                    "max_s": max(dur_values),
                    "mean_s": sum(dur_values) / len(dur_values),
                    "total_s": sum(dur_values),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved manifest -> %s", REFERENCES_MANIFEST)
    return manifest


if __name__ == "__main__":
    m = build_references_val96()
    if m.ref_single_is_concat:
        # Concat method stops as soon as the running total first exceeds the target, so it stays
        # in a tight band above REF_SINGLE_MAX_DURATION_S (matches t0018's own precedent range).
        assert 8.0 <= m.ref_single_duration_s <= 12.0, (
            f"ref_single (concat) duration out of range: {m.ref_single_duration_s}"
        )
    else:
        # Single natural clip: whatever duration the corpus happens to provide. val_96's real
        # distribution (preflight: max 15.00s) produced a 15.0s natural clip here — wider than the
        # ~10s target but still a reasonable single zero-shot reference utterance (well within
        # CosyVoice2/Chatterbox's typical 3-30s zero-shot prompt range). This is the documented
        # preflight deviation `plan/plan.md` Step 2 anticipated ("if ... a single clip near 10s
        # exists ... prefer a single natural clip"); the exact value is recorded in
        # `ref_single_deviation_note` and must never be captioned as "~10s" downstream.
        assert 5.0 <= m.ref_single_duration_s <= 20.0, (
            f"ref_single (single clip) duration implausible: {m.ref_single_duration_s}"
        )
    assert 20.0 <= m.ref_concat_duration_s < 30.0, (
        f"ref_concat duration out of range or breaches the CosyVoice2 30.0s hard limit "
        f"(S-0018-02): {m.ref_concat_duration_s}"
    )
    assert len(m.centroid_half_filenames) == 48
    print("OK:", m)
