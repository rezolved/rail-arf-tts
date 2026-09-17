"""Milestone 3 Step 12 (REQ-11): synthesize the shipped v3 bundle on CPU for human listening.

Loads `best/david_v3_best_decoder_kokoro.pth` (five-module packaged checkpoint) +
`best/david_v3_best_voicepack.pt` through `kokoro.KModel`, exactly as t0002 and t0008 did (CPU is
fine -- t0008 already proved this exact bundle produces clean audio), and synthesizes:

* The 3 fixed gate texts (`lining_up_suggestions_17`, `lining_up_suggestions_10`,
  `putting_them_head_to_head_15`), text sourced from the matching ElevenLabs David reference clip
  filenames.
* 5 val96 prompts chosen with `random.Random(42)` from `data/v4/val_list.txt`'s 96 entries, text
  derived the same way `build_val96_prompts()` does in
  `tasks/t0008_tts_eval_harness_baselines/code/prepare_prompts.py` (strip the wav filename's
  trailing hash suffix).

The 5 v3 sample phrase texts are NOT re-synthesized here (their original text is unrecoverable, per
`score_per_epoch_samples.py`'s finding) -- those samples are copied, not re-synthesized, in
`copy_reference_audio.py` (Milestone 3 Step 13).

Validation gate (REQ-11 / Phase 1.5): synthesizes the 3 gate texts first and checks
`is_likely_noise` on all 3 before proceeding to the 5 val96 prompts -- if any of the 3 is flagged
noise, this indicates a loading/config bug in this task's code (t0008 already validated this exact
bundle is clean), not a property of the bundle, and synthesis stops.

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.synthesize_v3_shipped
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

from tasks.t0016_v3_recipe_recovery.code.audio_quality_check import check_audio_quality
from tasks.t0016_v3_recipe_recovery.code.paths import (
    ELEVENLABS_DAVID_DIR,
    RESULTS_AUDIO_V3_SHIPPED_DIR,
    V3_BEST_DECODER_CKPT,
    V3_BEST_VOICEPACK,
    VAL96_LIST,
)

GATE_TEXT_SLUGS: tuple[str, ...] = (
    "lining_up_suggestions_17",
    "lining_up_suggestions_10",
    "putting_them_head_to_head_15",
)
VAL96_SAMPLE_SEED = 42
VAL96_SAMPLE_SIZE = 5


@dataclass(frozen=True, slots=True)
class SynthTiming:
    slug: str
    ttfb_s: float
    rtf: float
    audio_duration_s: float


def gate_text_for_slug(slug: str) -> str:
    """Derive display text for a gate slug from its matching ElevenLabs reference filename stem.

    The ElevenLabs corpus filenames ARE the slug (see `ELEVENLABS_DAVID_DIR/<slug>.wav`); the
    literal spoken text is not stored in a separate manifest for this fixed 3-text gate set, so the
    slug (underscores replaced with spaces) is used as the synthesis input, matching the
    convention `build_val96_prompts()` uses for deriving text from filenames.
    """
    return slug.replace("_", " ")


def sample_val96_prompts(val_list_path: Path, *, seed: int, n: int) -> list[str]:
    """Parse `val_list.txt` and return `n` prompt texts chosen with `random.Random(seed)`.

    Text derivation matches `build_val96_prompts()` in
    `tasks/t0008_tts_eval_harness_baselines/code/prepare_prompts.py`: split each line on `|`, take
    the first field's filename stem, strip the trailing 6-char hash suffix by splitting on the last
    underscore, and replace remaining underscores with spaces.
    """
    lines = val_list_path.read_text(encoding="utf-8").strip().split("\n")
    texts: list[str] = []
    for line in lines:
        wav_rel = line.split("|")[0].strip()
        stem = Path(wav_rel).stem
        text = stem.rsplit("_", 1)[0].replace("_", " ")
        texts.append(text)
    rng = random.Random(seed)
    return rng.sample(texts, k=n)


def load_model_and_pipeline() -> object:
    import torch
    from kokoro import KModel

    from tasks.t0016_v3_recipe_recovery.code.kokoro_pipeline import build_pipeline

    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    sd: object = torch.load(str(V3_BEST_DECODER_CKPT), map_location="cpu", weights_only=False)
    assert isinstance(sd, dict)
    merged: dict[str, object] = {}
    for module_name, module_sd in sd.items():
        assert isinstance(module_sd, dict)
        for param_name, param_val in module_sd.items():
            merged[f"{module_name}.{param_name}"] = param_val
    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"Loaded v3 best decoder: {len(missing)} missing, {len(unexpected)} unexpected keys")

    return build_pipeline(model=model)


def synthesize_one(pipeline: object, text: str, out_path: Path) -> SynthTiming:
    import numpy as np
    import soundfile as sf
    import torch

    chunks: list[np.ndarray] = []
    ttfb_s: float | None = None
    t_start = time.perf_counter()
    for _, _, audio in pipeline(text, voice=str(V3_BEST_VOICEPACK)):  # type: ignore[operator]
        if audio is not None and len(audio) > 0:
            if ttfb_s is None:
                ttfb_s = time.perf_counter() - t_start
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            chunks.append(audio.astype("float32"))
    wall_time = time.perf_counter() - t_start
    assert ttfb_s is not None and len(chunks) > 0, f"No audio produced for: {text!r}"

    full_audio = np.concatenate(chunks, axis=0)
    sample_rate = 24_000
    audio_duration_s = len(full_audio) / sample_rate
    rtf = wall_time / audio_duration_s if audio_duration_s > 0 else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), full_audio, sample_rate, subtype="FLOAT")

    return SynthTiming(
        slug=out_path.stem, ttfb_s=ttfb_s, rtf=rtf, audio_duration_s=audio_duration_s
    )


def main() -> None:
    assert V3_BEST_DECODER_CKPT.exists(), f"missing {V3_BEST_DECODER_CKPT}"
    assert V3_BEST_VOICEPACK.exists(), f"missing {V3_BEST_VOICEPACK}"
    assert ELEVENLABS_DAVID_DIR.exists(), f"missing {ELEVENLABS_DAVID_DIR}"

    pipeline = load_model_and_pipeline()

    timings: list[SynthTiming] = []

    print("Phase 1.5 validation gate: synthesizing 3 fixed gate texts first...")
    for slug in GATE_TEXT_SLUGS:
        text = gate_text_for_slug(slug)
        out_path = RESULTS_AUDIO_V3_SHIPPED_DIR / f"{slug}.wav"
        timing = synthesize_one(pipeline, text, out_path)
        timings.append(timing)
        quality = check_audio_quality(out_path, text=text)
        print(f"  {slug}: is_likely_noise={quality.is_likely_noise}, ttfb_s={timing.ttfb_s:.3f}")
        if quality.is_likely_noise:
            raise RuntimeError(
                f"Gate text {slug!r} synthesized as likely noise -- this indicates a "
                "loading/config bug in this task's synthesis code (t0008 already proved this "
                "exact bundle produces clean audio). Stopping before the remaining 5 val96 "
                "prompts per Phase 1.5's validation gate."
            )

    print("Gate passed. Synthesizing 5 seed-42 val96 prompts...")
    val96_texts = sample_val96_prompts(VAL96_LIST, seed=VAL96_SAMPLE_SEED, n=VAL96_SAMPLE_SIZE)
    for i, text in enumerate(val96_texts):
        slug = f"val96_seed42_{i:02d}"
        out_path = RESULTS_AUDIO_V3_SHIPPED_DIR / f"{slug}.wav"
        timing = synthesize_one(pipeline, text, out_path)
        timings.append(timing)
        quality = check_audio_quality(out_path, text=text)
        print(f"  {slug} ({text!r}): is_likely_noise={quality.is_likely_noise}")

    print(f"\nWrote {len(timings)} WAV files to {RESULTS_AUDIO_V3_SHIPPED_DIR}")
    for t in timings:
        print(f"  {t.slug}: ttfb_s={t.ttfb_s:.3f} rtf={t.rtf:.3f} dur_s={t.audio_duration_s:.3f}")


if __name__ == "__main__":
    main()
