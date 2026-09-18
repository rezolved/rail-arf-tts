"""Build the 3-way comparison audio set (plan.md Step 15, owner-correction REQ-16).

For each of 10 fixed comparison texts (3 gate texts + 7 seeded val96 samples, same
seed/count/selection method as t0018 for continuity), copies up to THREE files per
(text, system) pair into `results/audio_samples/comparison_set/`:

* `<text_id>__<system>_<best_variant>__new_ref.wav` -- this task's own best-setting output
  (lowest `ttfb_ms_p50` among that system's non-null `ref_single` variants).
* `<text_id>__<system>__t0018_old_ref.wav` -- t0018's own (wrong-voice-reference) output for the
  matching text, copied read-only from t0018's `results/audio_samples/comparison_set/`.
* `<text_id>__val96_original.wav` -- the actual production-voice ground-truth clip (only for the 7
  val96-sampled texts; the 3 fixed gate texts are filler-corpus phrases with no val96 original).

**Documented deviation (not silent):** `dvc pull` for t0018's `results/audio_samples/
comparison_set.dvc` failed in this task's environment with an Azure `DefaultAzureCredential`
failure (`az account show`/`az account get-access-token` succeed locally, but `dvc`'s constrained
credential chain -- EnvironmentCredential, WorkloadIdentityCredential, ManagedIdentityCredential
only, no AzureCliCredential in the error trace -- does not pick up the same session). This is an
infrastructure/credential-configuration issue outside this task's control (`CLAUDE.md` Key Rule 0:
framework/infrastructure issues are not task work), not something this task's code can silently
work around. The `t0018_old_ref` column is therefore left absent (`-`) in
`results/listening_guide.md`, with this note as the recorded reason, rather than fabricated or
skipped without explanation.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

from tasks.t0008_tts_eval_harness_baselines.code.harness import load_val96_prompts
from tasks.t0021_zero_shot_latency_reduction.code.constants import (
    COMPARISON_SET_SEED,
    COMPARISON_SET_VAL96_COUNT,
    GATE_TEXT_NAMES,
)
from tasks.t0021_zero_shot_latency_reduction.code.paths import (
    REPO_ROOT,
    RESULTS_AUDIO_COMPARISON_DIR,
    RESULTS_PER_CLIP_METRICS,
    T0018_COMPARISON_SET_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEMS = ("cosyvoice2", "chatterbox")


def _slugify_for_lookup(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def select_comparison_texts() -> list[tuple[str, str]]:
    """Return [(text_id, text), ...] -- 3 gate texts + 7 seeded val96 samples."""
    selected: list[tuple[str, str]] = []
    for name in GATE_TEXT_NAMES:
        text = name.replace("_", " ")
        selected.append((name, text))

    val96_items = load_val96_prompts()
    val96_texts = [item.text for item in val96_items]
    sampled = random.Random(COMPARISON_SET_SEED).sample(val96_texts, COMPARISON_SET_VAL96_COUNT)
    for text in sampled:
        text_id = _slugify_for_lookup(text)[:40]
        selected.append((text_id, text))
    return selected


def _best_variant_per_system(records: list[dict[str, object]]) -> dict[str, str | None]:
    """Lowest ttfb_ms_p50 (fillers, ref_single) among variants with success_rate >= 0.8."""
    from tasks.t0021_zero_shot_latency_reduction.code.constants import (
        CHATTERBOX_VARIANTS,
        COSYVOICE2_VARIANTS,
    )
    from tasks.t0021_zero_shot_latency_reduction.code.report_zeroshot import (
        compute_variant_metrics,
    )

    variants_by_system = {"cosyvoice2": COSYVOICE2_VARIANTS, "chatterbox": CHATTERBOX_VARIANTS}
    best: dict[str, str | None] = {}
    for system in SYSTEMS:
        best_variant: str | None = None
        best_ttfb = float("inf")
        for variant in variants_by_system[system]:
            m = compute_variant_metrics(records, system, variant, "ref_single", "fillers")
            ttfb = m.get("ttfb_ms_p50")
            if ttfb is None:
                continue
            if float(ttfb) < best_ttfb:
                best_ttfb = float(ttfb)
                best_variant = variant
        best[system] = best_variant
        logger.info("Best variant for %s: %s (ttfb_ms_p50=%.1f)", system, best_variant, best_ttfb)
    return best


def _val96_original_path(text: str) -> Path | None:
    for item in load_val96_prompts():
        if item.text.strip() == text.strip() and item.ref_wav is not None:
            return REPO_ROOT / item.ref_wav
    return None


def main() -> None:
    RESULTS_AUDIO_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = json.loads(
        RESULTS_PER_CLIP_METRICS.read_text(encoding="utf-8")
    )
    comparison_texts = select_comparison_texts()
    logger.info("Comparison texts: %s", [t for _, t in comparison_texts])

    best_variant = _best_variant_per_system(records)

    n_new_ref = 0
    n_old_ref = 0
    n_val96_original = 0
    t0018_available = T0018_COMPARISON_SET_DIR.exists() and any(T0018_COMPARISON_SET_DIR.iterdir())
    if not t0018_available:
        logger.warning(
            "t0018 comparison_set not materialized locally (dvc pull failed for this task's "
            "environment) -- t0018_old_ref column will be entirely absent. See this module's "
            "docstring for the documented reason."
        )

    for text_id, text in comparison_texts:
        for system in SYSTEMS:
            variant = best_variant.get(system)
            if variant is None:
                continue
            matches = [
                r
                for r in records
                if str(r.get("text", "")).strip() == text.strip()
                and r.get("system") == system
                and r.get("acceleration_variant") == variant
                and r.get("condition") == "ref_single"
                and r.get("audio_path")
            ]
            for rec in matches:
                src = Path(str(rec["audio_path"]))
                if not src.exists():
                    logger.warning("Missing new-ref source audio: %s", src)
                    continue
                dest = RESULTS_AUDIO_COMPARISON_DIR / f"{text_id}__{system}_{variant}__new_ref.wav"
                shutil.copy2(src, dest)
                n_new_ref += 1

            if t0018_available:
                # t0018's own comparison_set naming: `<text_id>__<system>_ref_single.wav` (its
                # slugification of the same GATE_TEXT_NAMES/val96-sample method, seed=42).
                candidate = T0018_COMPARISON_SET_DIR / f"{text_id}__{system}_ref_single.wav"
                if candidate.exists():
                    dest = RESULTS_AUDIO_COMPARISON_DIR / f"{text_id}__{system}__t0018_old_ref.wav"
                    shutil.copy2(candidate, dest)
                    n_old_ref += 1

        val96_src = _val96_original_path(text)
        if val96_src is not None and val96_src.exists():
            dest = RESULTS_AUDIO_COMPARISON_DIR / f"{text_id}__val96_original.wav"
            shutil.copy2(val96_src, dest)
            n_val96_original += 1

    logger.info(
        "Copied %d new_ref, %d t0018_old_ref, %d val96_original clips -> %s",
        n_new_ref,
        n_old_ref,
        n_val96_original,
        RESULTS_AUDIO_COMPARISON_DIR,
    )


if __name__ == "__main__":
    main()
