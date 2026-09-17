"""Milestone 3 Step 12 (REQ-11): build a Kokoro `KPipeline` matching David's training accent.

Copied and adapted (not imported -- `t0003_kokoro_v5_phoneme_data` is not a registered library, so
Critical Rule 8 requires copying rather than
`from tasks.t0003_kokoro_v5_phoneme_data.code.build_pipeline import build_pipeline`) from:

* `tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py` (`build_pipeline()`)
* `tasks/t0003_kokoro_v5_phoneme_data/code/prepare_v5_data.py` (`install_lexicon()`,
  `load_kokoro_vocab()`, `phonemize()`)
* `tasks/t0003_kokoro_v5_phoneme_data/code/lexicon.py` (`RESPELLINGS`, `IPA_PREFIX`)
* `tasks/t0003_kokoro_v5_phoneme_data/code/constants.py` (the handful of Kokoro/G2P constants used
  below)

David's reference voice is British English; the v5+ manifests were phonemized with
`misaki.en.G2P(british=True)` plus the brand/name lexicon below, so inference must use the SAME
`lang_code="b"` and the SAME lexicon or the model sees a token sequence it never trained on
(`tasks/t0003_kokoro_v5_phoneme_data/code/build_pipeline.py`'s own module docstring).

Usage::

    from tasks.t0016_v3_recipe_recovery.code.kokoro_pipeline import build_pipeline
    pipe = build_pipeline(model=model)
    for _, _, audio in pipe("Rezolve supports agentic commerce.", voice=voicepack_path):
        ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

KOKORO_REPO_ID: Final[str] = "hexgrad/Kokoro-82M"
KOKORO_CONFIG_FILENAME: Final[str] = "config.json"
KOKORO_VOCAB_KEY: Final[str] = "vocab"
KOKORO_LANG_CODE: Final[str] = "b"
OOV_MARKER: Final[str] = "❓"
MISSING_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"([.!?])([A-Za-z])")

IPA_PREFIX: Final[str] = "/"

# Copied verbatim from `tasks/t0003_kokoro_v5_phoneme_data/code/lexicon.py::RESPELLINGS` (dumped
# via `repr()` from the live module to guarantee exact IPA character fidelity).
RESPELLINGS: Final[dict[str, str]] = {
    "McElla": "/məkˈɛlə",
    "agentic": "/Aʤˈɛntɪk",
    "agio": "/ˈaʤiQ",
    "ahmad": "ah mahd",
    "ahmed": "ah med",
    "ai": "A.I.",
    "amad": "/ˈamad",
    "anthony": "/ˈantəni",
    "auditability": "audit ability",
    "azos": "/ˈAzɒs",
    "banerjee": "banner jee",
    "barrell": "barrel",
    "binlaurie": "bin lorry",
    "brainpowa": "brain power",
    "burchill": "bur chill",
    "creighton": "cray ton",
    "derek": "derrick",
    "efisher": "/ˈiːfɪʃə",
    "ermecarnatrezolve": "/ˈɜːmi kˈɑːnə tˈiː ɹɪzˈɒlv",
    "ermei": "ur may",
    "ermy": "ur mee",
    "ferrero": "ferry row",
    "frontend": "front end",
    "geo-fencing": "gee oh fencing",
    "ingram": "in gram",
    "kahn": "khan",
    "ll": "ell",
    "magento": "/məʤˈɛntQ",
    "magento's": "/məʤˈɛntQz",
    "mahindra": "/məhˈɪndɹə",
    "michelle": "/mɪʃˈɛl",
    "microsoft": "micro soft",
    "microsoft's": "micro softs",
    "mintra": "/mˈɪntɹə",
    "multimodal": "/mˌʌltɪmˈQdᵊl",
    "mv": "M.V.",
    "optimisation": "optimization",
    "personalisation": "personalization",
    "poseable": "pose able",
    "powa": "power",
    "pq": "P.Q.",
    "presley": "press lee",
    "rakuten": "/ɹˌakˈuːtɛn",
    "reebug": "ree bug",
    "replatforming": "ree platforming",
    "rezolve": "resolve",
    "rezolve's": "resolves",
    "sackville": "sack vill",
    "salman": "/sˈalmɑːn",
    "salvik": "/sˈalvɪk",
    "shoeby": "shoe bee",
    "shortlist": "short list",
    "st": "ess tee",
    "subsquid": "sub squid",
    "telecom": "tele com",
    "th": "tee aitch",
    "unif-": "uniform",
    "urmee": "ur mee",
    "verified": "vairy fide",
    "wagner": "/vˈaɡnə",
    "walkthrough": "walk through",
    "wavell": "way vell",
}


def phonemize(text: str, g2p: object) -> str:
    """British-English IPA for `text`. Raises on any G2P failure -- no silent fallback."""
    repaired = MISSING_SPACE_RE.sub(r"\1 \2", text)
    result, _ = g2p(repaired)  # type: ignore[operator]
    if isinstance(result, str):
        return result.strip()
    parts: list[str] = []
    for token in result:
        phoneme = token.phonemes if token.phonemes is not None else token.text
        parts.append(phoneme + (token.whitespace if token.whitespace else ""))
    return "".join(parts).strip()


def load_kokoro_vocab() -> frozenset[str]:
    """Kokoro's 114 token symbols -- the exact character set the model accepts at inference."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=KOKORO_REPO_ID, filename=KOKORO_CONFIG_FILENAME)
    config = json.loads(Path(config_path).read_text())
    return frozenset(config[KOKORO_VOCAB_KEY].keys())


def install_lexicon(g2p: object, vocab: frozenset[str]) -> int:
    """Teach misaki the corpus's brand and person names. Returns how many were added."""
    lexicon = g2p.lexicon  # type: ignore[attr-defined]
    failures: list[str] = []
    installed = 0
    for word, spelling in RESPELLINGS.items():
        if spelling.startswith(IPA_PREFIX):
            phonemes = spelling[len(IPA_PREFIX) :]
        else:
            phonemes = phonemize(spelling, g2p)
            if OOV_MARKER in phonemes:
                failures.append(f"{word!r}: respelling {spelling!r} is itself unknown")
                continue
        illegal = set(phonemes) - vocab
        if len(illegal) > 0:
            failures.append(f"{word!r}: {''.join(sorted(illegal))!r} not in Kokoro vocab")
            continue
        for key in (word, word.capitalize(), word.title(), word.upper()):
            lexicon.golds[key] = phonemes
        installed += 1
    if len(failures) > 0:
        raise ValueError(f"Lexicon install failures: {failures}")
    return installed


def build_pipeline(model: Any) -> Any:
    """KPipeline in David's British accent, with the corpus brand lexicon installed."""
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code=KOKORO_LANG_CODE, model=model, repo_id=KOKORO_REPO_ID)
    install_lexicon(pipeline.g2p, load_kokoro_vocab())
    return pipeline
