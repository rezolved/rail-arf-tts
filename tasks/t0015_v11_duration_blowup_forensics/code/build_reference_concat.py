"""Build the 3-clip-concatenated reference audio used by the audible-speech gate (Milestone C step
9), matching t0013's Milestone E reference exactly (`results/v10_diagnosis.md` "Setup":
`lining_up_suggestions_17`, `lining_up_suggestions_10`, `putting_them_head_to_head_15`, joined with
0.2s silence gaps, 5.48s total) for direct comparability -- this is the reference t0013's own
`v10_diagnosis.md` clip_fraction/spectral_flatness numbers were measured against, and the one
`plan/plan.md` step 9 refers to as "the same ... reference audio ... t0013 used". (t0013's later
`code/random_decoder_probe.py` used a different, alphabetically-first-3 selection for its own
narrower falsification-probe purpose -- not reused here.) Single `11labs_david` clips are too short
(<1s) for `StyleEncoder`, hence the concatenation.

Usage::

    .venv-styletts2/bin/python code/build_reference_concat.py <out_wav>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from tasks.t0015_v11_duration_blowup_forensics.code.paths import ELEVENLABS_DAVID_DIR

REFERENCE_CLIP_NAMES: tuple[str, ...] = (
    "lining_up_suggestions_17.wav",
    "lining_up_suggestions_10.wav",
    "putting_them_head_to_head_15.wav",
)


def build_reference_concat(out_path: Path) -> Path:
    ref_wavs = [ELEVENLABS_DAVID_DIR / name for name in REFERENCE_CLIP_NAMES]
    missing = [p for p in ref_wavs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing reference clips: {missing}")
    audio_parts = []
    sr = None
    for p in ref_wavs:
        data, sr = sf.read(str(p))
        audio_parts.append(data)
        audio_parts.append(np.zeros(int(0.2 * sr)))
    concat = np.concatenate(audio_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), concat, sr)
    print(f"Wrote {out_path} ({len(concat) / sr:.2f}s) from {[p.name for p in ref_wavs]}")
    return out_path


if __name__ == "__main__":
    build_reference_concat(Path(sys.argv[1]))
