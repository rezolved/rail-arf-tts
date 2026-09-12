"""Build a Kokoro pipeline whose G2P matches the v5 training manifests.

The manifests were phonemized with British misaki plus the brand/name lexicon in
`lexicon.py`. Inference must use the SAME accent and the SAME lexicon, or the model is fed a
token sequence it never trained on: `lang_code="a"` swaps every vowel symbol, and a missing
lexicon turns "Rezolve" back into `❓`, which is not in Kokoro's vocab and gets dropped.

Usage:
    from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline
    pipe = build_pipeline(model=model)
    for _, _, audio in pipe("Rezolve supports agentic commerce.", voice=voicepack_path):
        ...
"""

from typing import Any

from tasks.t0003_kokoro_v5_phoneme_data.code import constants as C
from tasks.t0003_kokoro_v5_phoneme_data.code.prepare_v5_data import (
    install_lexicon,
    load_kokoro_vocab,
)


def build_pipeline(model: Any) -> Any:
    """KPipeline in David's British accent, with the corpus brand lexicon installed."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=C.KOKORO_LANG_CODE, model=model, repo_id=C.KOKORO_REPO_ID)
    install_lexicon(pipeline.g2p, load_kokoro_vocab())
    return pipeline
