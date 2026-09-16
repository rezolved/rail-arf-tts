# Control Test (Milestone D)

Validates the instrumented harness (`code/infer_styletts2.py`) against the official pretrained
StyleTTS2 LibriTTS checkpoint (`yl4579/StyleTTS2-LibriTTS`, `Models/LibriTTS/epochs_2nd_00020.pth`)
**before** ever running it against a v10 checkpoint, per `task_description.md` Scope §2 and
`plan/plan.md` Milestone D. This checkpoint is independently, externally known-good (the official
released StyleTTS2 model, published and used by thousands outside this project) -- it is not itself
under investigation.

Note: this "control" checkpoint (used to validate the harness/environment) is a different file from
`first_stage_v3.pth` (the Kokoro-project reference used in Milestone B's tensor forensics as the
comparison point for v10's decoder architecture). The two serve different purposes: this one asks
"is our harness/environment broken", `first_stage_v3.pth` asks "what was v10's decoder initialized
from."

## Setup

* Checkpoint: `Models/LibriTTS/epochs_2nd_00020.pth` (771,390,526 bytes), config
  `Models/LibriTTS/config.yml`, both downloaded via
  `git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS` (per the Colab notebooks in this
  fork's own repo, `Colab/StyleTTS2_Demo_LibriTTS.ipynb`, which link this exact HF repo).
* Reference audio: `Demo/reference_audio/1221-135767-0014.wav` (3.0s, unzipped from
  `reference_audio.zip`, same HF repo).
* Text: "This is a test of the Style T T S two inference harness." (same prompt used for all runs in
  this task, for comparability).
* Harness: `code/infer_styletts2.py` run through `code/.venv-styletts2` (`torch==2.5.1` CPU).

## A real harness bug found and fixed here

The first control run **failed the load-instrumentation hard-fail gate** (working exactly as
designed -- REQ-2): `predictor` mismatched 90/122 params, and after fixing that,
`predictor_encoder`/`style_encoder` mismatched further. Root cause, confirmed by direct key
inspection (not guessed): `models.py` (line 13) constructs modules with the newer
`torch.nn.utils.parametrizations.{weight_norm,spectral_norm}` API, but the officially released 2023
checkpoint was saved under the classic `torch.nn.utils.{weight_norm,spectral_norm}` API, which uses
different state-dict key names for the mathematically identical reparametrization (`weight_g`/
`weight_v` vs. `parametrizations.weight.original0`/`original1`; `weight_orig`/`weight_u`/`weight_v`
vs. `parametrizations.weight.original`/`.0._u`/`.0._v`). Confirmed empirically these are pure
renames (identical shapes at every position, e.g. `predictor.F0.1.conv1`: control's
`weight_g`/`weight_v` are `(256,1,1)`/`(256,512,3)`, exactly matching v10's `original0`/`original1`
shapes for the same submodule) -- **not** a real architecture mismatch. Fixed by adding
`_rename_legacy_parametrization_keys()` to `code/infer_styletts2.py`'s loader (applied as part of
the `module.`-stripped fallback attempt). v10's own checkpoints were trained under this fork's
current `models.py`, so they already use the new naming and this rename is a no-op for them -- it
exists solely to make this external control checkpoint loadable.

This is exactly the kind of harness/environment bug Milestone D exists to catch before ever
attributing anything to a project checkpoint (`task_description.md` Scope §2), and it was caught by
the instrumented per-module loader (REQ-2), not by silently accepting a `strict=False` load.

## Load instrumentation (final attempt, after the fix)

All 13 modules loaded with **0 missing, 0 unexpected** (full detail:
`results/load_log_epochs_2nd_00020.json`). All required the `module.`-stripped (+
legacy-parametrization-renamed) fallback path -- consistent across every checkpoint tested in this
task, since all are DataParallel-trained (`module.`-prefixed).

| Module | Final missing | Final unexpected |
| --- | ---: | ---: |
| bert | 0 | 0 |
| bert_encoder | 0 | 0 |
| predictor | 0 | 0 |
| decoder | 0 | 0 |
| text_encoder | 0 | 0 |
| predictor_encoder | 0 | 0 |
| style_encoder | 0 | 0 |
| diffusion | 0 | 0 |
| text_aligner | 0 | 0 |
| pitch_extractor | 0 | 0 |
| mpd | 0 | 0 |
| msd | 0 | 0 |
| wd | 0 | 0 |

## Audio output stats

`results/audio_samples/control_epochs_2nd_00020.wav` (4.92s):

| Metric | Value |
| --- | --- |
| RMS | 0.0370 |
| Peak | 0.3796 |
| Clip fraction (`\|sample\| > 0.99`) | 0.0 |
| Silence fraction (20ms frames < -40dBFS) | 0.309 |
| Spectral flatness (mean) | 0.0615 |

## Verdict: PASS

Spectral flatness (0.061) is close to 0 (concentrated formant/harmonic structure, the opposite of
white noise); RMS/peak show a healthy crest factor (~10x, i.e. loud syllables and quiet gaps, not a
saturated signal); silence fraction (0.31) is consistent with natural pauses in a short spoken
sentence, not near-total silence. By `code/audio_quality_check.py`'s heuristic:
`is_likely_noise=False`. **The harness produces real, non-noise, non-silent, non-clipped audio from
a known-good checkpoint.** Milestone E (v10 diagnosis) is therefore trustworthy evidence, not
harness noise -- proceeding per `plan/plan.md` step 11's gate check.

Per the Rejection Criteria pre-registered in `plan/plan.md`: since this control run PASSES, v10's
`speaker_sim`/`rtf` numbers in `results/metrics.json` are diagnostic (not null-by-rejection).
