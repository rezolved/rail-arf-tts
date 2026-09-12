"""Compare predicted durations across Stage 1 checkpoints, without training anything.

Stage 1 val_loss cannot be compared across v3 and v4 -- different val sets (29 vs 96 clips).
Synthesized duration for a fixed sentence can: it is dataset-independent and it is the exact
symptom t0002 hit (89 s of audio for a ~3.6 s sentence).

Each checkpoint's five text-conditioned modules are loaded into a stock `KModel` and the same
sentence is synthesized with the same voicepack, so the predictor is the only thing that varies.
The stock model with no modules swapped is the baseline.

Usage:
    uv run python3 code/check_stage1_durations.py
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from tasks.t0003_kokoro_v5_phoneme_data.code import constants as C
from tasks.t0003_kokoro_v5_phoneme_data.code import paths as P

# Five text-conditioned modules; the style encoder is excluded because it never sees text.
MODULES = ("bert", "bert_encoder", "predictor", "text_encoder", "decoder")

SENTENCE = "Understood. If you need anything else later, feel free to reach out."
EXPECTED_SECONDS = 3.6
EXPLOSION_FACTOR = 3.0

BENCH = P.BENCHMARKS_ROOT
T0002 = P.TASK_DIR.parent / "t0002_kokoro_v4_voicepack_decoder_package" / "results"


@dataclass(frozen=True, slots=True)
class Candidate:
    """A checkpoint to test, and where in it the module state dicts live."""

    label: str
    path: Path
    net_key: str | None


CANDIDATES: tuple[Candidate, ...] = (
    # Stage 1 that fed v3's SUCCESSFUL Stage 2 (written 13:57 Sep 7, val 0.770).
    Candidate("v3-stage1-1357", BENCH / "v3" / "stage1" / "first_stage.pth", "net"),
    # Later v3 Stage 1 (21:13 Sep 7, val 0.545) -- better val, unknown downstream behaviour.
    Candidate("v3-stage1-2113", BENCH / "checkpoints" / "stage1_v3" / "first_stage.pth", "net"),
    # v4's Stage 1, the one t0001's Stage 2 started from (val 0.594).
    Candidate("v4-stage1", BENCH / "best_v4" / "first_stage.pth", "net"),
    # Positive control: v3's shipped Stage 2 bundle, already in Kokoro key format.
    Candidate(
        "v3-stage2-SHIPPED", BENCH / "v3" / "best" / "david_v3_best_decoder_kokoro.pth", None
    ),
)

VOICEPACKS: tuple[tuple[str, Path], ...] = (
    ("v3vp", BENCH / "v3" / "best" / "david_v3_best_voicepack.pt"),
    ("v4vp", T0002 / "david_v4_voicepack.pt"),
)


def convert_key(key: str) -> str:
    """StyleTTS2 checkpoint key -> Kokoro key (same mapping t0002 used)."""
    if key.startswith("module."):
        key = key[7:]
    key = key.replace(".parametrizations.weight.original0", ".weight_g")
    key = key.replace(".parametrizations.weight.original1", ".weight_v")
    return key


def load_modules(candidate: Candidate) -> dict[str, dict[str, torch.Tensor]]:
    raw = torch.load(str(candidate.path), map_location="cpu", weights_only=False)
    source = raw if candidate.net_key is None else raw[candidate.net_key]
    return {
        name: {convert_key(k): v for k, v in source[name].items()}
        for name in MODULES
        if name in source
    }


def synthesize_seconds(model: object, voicepack: Path) -> float:
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=C.KOKORO_LANG_CODE, model=model, repo_id=C.KOKORO_REPO_ID)
    chunks = [
        chunk if isinstance(chunk, torch.Tensor) else torch.tensor(chunk)
        for _, _, chunk in pipeline(SENTENCE, voice=str(voicepack), speed=1.0)
        if chunk is not None
    ]
    if len(chunks) == 0:
        return 0.0
    return float(torch.cat(chunks).shape[0]) / 24000.0


def main() -> None:
    from kokoro import KModel

    print(f"Sentence: {SENTENCE!r}  (expected ~{EXPECTED_SECONDS} s)\n")

    baseline = KModel(repo_id=C.KOKORO_REPO_ID, disable_complex=True).eval()
    for vp_label, vp_path in VOICEPACKS:
        seconds = synthesize_seconds(baseline, vp_path)
        print(f"{'BASELINE (stock Kokoro)':26s} {vp_label}  {seconds:7.2f} s")
    print()

    for candidate in CANDIDATES:
        if not candidate.path.exists():
            print(f"{candidate.label:26s} MISSING {candidate.path}")
            continue
        states = load_modules(candidate)
        for vp_label, vp_path in VOICEPACKS:
            model = KModel(repo_id=C.KOKORO_REPO_ID, disable_complex=True)
            for name, state in states.items():
                getattr(model, name).load_state_dict(state, strict=False)
            model.eval()
            seconds = synthesize_seconds(model, vp_path)
            verdict = "EXPLOSION" if seconds > EXPECTED_SECONDS * EXPLOSION_FACTOR else "sane"
            loaded = ",".join(sorted(states.keys()))
            print(f"{candidate.label:26s} {vp_label}  {seconds:7.2f} s  {verdict:9s} [{loaded}]")


if __name__ == "__main__":
    main()
