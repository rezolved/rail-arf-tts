# v10 Checkpoint Synthesis Quality Forensics

## Motivation

t0010's own harness eval (`speaker_sim`, `ttfb_ms`, `rtf`) was deferred because the training VM's
root disk was 100% full at teardown — so **nobody has ever listened to audio synthesized from
`kokoro-v10-best` / `epoch_2nd_00016.pth` until this session**, and that listen happened outside
the normal ARF pipeline, via an ad hoc local reproduction, not the project's own eval harness.

While preparing an audio comparison for the user (originals vs. fine-tuned), Claude found that the
`kokoro` pip package (and the `semidark/kokoro` fork used by `kikiri-tts`) can only load the
**ISTFTNet** decoder — but v10 was trained with `model_params.decoder.type: hifigan`
(`config_david_v10.yml`), a completely different architecture. Loading the packaged 5-module
checkpoint (`kokoro-v10-best.pth`) through either package fails with tensor shape mismatches on
`decoder.generator.*`.

The correct inference path for a `hifigan`-decoder StyleTTS2 checkpoint is StyleTTS2's own native
code (`StyleTTS2/Demo/Inference_LibriTTS.ipynb` in `github.com/semidark/StyleTTS2`, a submodule of
`github.com/semidark/kikiri-tts`), not Kokoro's `KModel`/`KPipeline`. Claude adapted that notebook
into a standalone script and ran it locally (CPU) against the **raw** training checkpoint
`epoch_2nd_00016.pth` (which has the full StyleTTS2 module set — `style_encoder`,
`predictor_encoder`, `diffusion`, `text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd` — none of
which survive the "packaged 5-module" conversion used for the Kokoro-format asset).

**Result: the user reports the synthesized audio is pure noise, no voice at all.**

This could mean either of two very different things:

1. **The reproduction is broken.** The script was assembled in one pass from notebook cells and
   was never validated against a known-good checkpoint. Critically, **the checkpoint-loading loop
   never logged `missing`/`unexpected` key counts per module** — it only printed `"{key}: loading"`
   on success and fell back to a `strict=False` load (which silently tolerates a poor key match) on
   any exception. All 13 modules printed as loaded with no exception raised, but that does **not**
   prove every tensor actually got the right weights — a `strict=False` fallback could have left
   large parts of `decoder`/`diffusion` at random initialization while reporting success.
2. **The checkpoint itself is broken.** t0010's `results_summary.md` reports a healthy val_loss
   trajectory (0.797 best, 0.853 primary checkpoint) and **zero health-gate events** across all 17
   epochs — but val_loss is an aggregate reconstruction-style loss; it would not necessarily catch a
   defect isolated to the adversarial decoder or the diffusion-based style sampler. Given this
   checkpoint's audio has literally never been listened to before, it is entirely possible training
   silently produced a checkpoint whose vocoder or style sampler never converged, even with a
   plausible-looking val_loss curve.

**Do not assume either answer.** This task's job is to tell the two apart with evidence, not
opinion, then act on whichever is true.

## Key Questions

1. Does the ad hoc inference script actually load every module's weights correctly? (Log
   `missing`/`unexpected` key counts explicitly, per module, for both the primary load attempt and
   the `module.`-stripped fallback — do not let a silent `strict=False` pass as success.)
2. With a **known-good control** — the base pretrained StyleTTS2 LibriTTS checkpoint the demo
   notebook ships against (multi-speaker, untuned) — does the *same* inference harness produce
   intelligible speech? This isolates "the harness code is wrong" from "this specific checkpoint is
   wrong." If the harness fails on the control too, the bug is in the reproduction, full stop.
3. If the harness is validated correct on the control: does `epoch_2nd_00016.pth` (primary, epoch
   17) produce noise while `epoch_2nd_00014.pth` (backup, epoch 15, val_loss 0.818, also DVC-tracked
   in `tasks/t0010_stage2_safeguarded_training/data/run_v10/`) produces intelligible speech, or do
   both fail? This isolates a late-training regression from a pipeline-wide defect.
4. Do any of the loaded modules contain NaN/Inf weights, or weight-norm statistics wildly divergent
   from the pre-trained base model or from `t0009`'s reference checkpoints? (`torch.load` the raw
   checkpoint and check `torch.isfinite` + per-module weight-norm summaries before ever running
   inference — a cheap, fast check that should run first.)
5. Does `tasks/t0010_stage2_safeguarded_training/logs/steps/.../metrics.jsonl` or
   `data/run_v10/metrics.jsonl` show any anomaly in a per-step signal that the epoch-level val_loss
   average could have masked (discriminator loss spikes, gradient-norm blowups, F0/duration
   predictor divergence) around or before epoch 15-17?
6. Is `compute_style`'s reference-clip requirement (needs several seconds of audio; short filler
   clips under ~2s crash the style encoder's conv kernel — discovered this session) itself a sign
   of a config/style-encoder mismatch, or expected StyleTTS2 behavior unrelated to the noise
   problem?

## Scope

### 1. Reproduce the ad hoc setup properly, with instrumentation

The exploratory work this session lived entirely under `/tmp/tts_compare/` (ephemeral, gone by the
next session) — recreate it under this task's `code/` directory instead:

```bash
git clone --depth 1 https://github.com/semidark/kikiri-tts.git <task>/code/kikiri-tts
cd <task>/code/kikiri-tts && git submodule update --init --recursive
```

This pulls in `StyleTTS2` (`github.com/semidark/StyleTTS2`) and `kokoro`
(`github.com/semidark/kokoro`) as submodules. `StyleTTS2/Utils/{ASR,JDC,PLBERT}/` ship pretrained
weights directly in the repo (~134MB total) — no separate download needed.

Build a CPU venv with `torch==2.5.1 torchaudio==2.5.1` (torch >=2.6 changes `weights_only`
defaults and breaks `StyleTTS2/models.py`'s un-flagged `torch.load` calls — either pin torch <2.6
or patch those call sites), plus `soundfile munch pydub pyyaml librosa nltk matplotlib accelerate
transformers einops einops-exts tqdm typing-extensions phonemizer` and
`git+https://github.com/resemble-ai/monotonic_align.git` (imported at module level by
`StyleTTS2/utils.py` even though unused at inference time). `brew install espeak-ng` for the
phonemizer backend.

Rewrite the checkpoint-loading loop (adapted from
`StyleTTS2/Demo/Inference_LibriTTS.ipynb`) to print, per module: parameter count, `missing` key
count, `unexpected` key count, and whether the primary or the `module.`-stripped fallback path was
used. Treat any nonzero missing/unexpected count on a module that should be fully covered by the
checkpoint (`bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`, `predictor_encoder`,
`style_encoder`, `diffusion`) as a hard failure worth investigating before synthesizing anything.

### 2. Control test on the base pretrained checkpoint

Before touching v10 again, run the exact same harness against the base pretrained StyleTTS2
LibriTTS checkpoint (URL and reference audio paths are in the notebook's own cells —
`Models/LibriTTS/epochs_2nd_00020.pth`, `Demo/reference_audio/*.wav`). If this produces garbled
noise too, the bug is in the harness/environment (torch version, phonemizer setup, sampler
parameters) — fix that first and re-verify on the control before ever touching v10 again.

### 3. Diagnose v10 specifically

Once the harness is proven correct on the control:

* Re-run inference on `epoch_2nd_00016.pth` (primary) and `epoch_2nd_00014.pth` (backup) with the
  instrumented loader from step 1. Compare missing/unexpected counts between the two — a stark
  difference points at a load-time bug specific to one file's structure, not the harness.
* If both load cleanly (0 missing/unexpected) and both still produce noise, the defect is in
  training, not loading. Check `torch.isfinite` across every tensor in the raw checkpoint's `net`
  dict, and compare per-module weight-norm summaries between `epoch_2nd_00014` and
  `epoch_2nd_00016` — a sudden discontinuity between the two nearby epochs would point at a specific
  late-training event.
* Cross-reference `data/run_v10/metrics.jsonl` for any adversarial-loss or gradient-norm signal
  around epochs 15-17 that the epoch-level val_loss average in `results_summary.md` didn't surface.

### 4. Root-cause and recommend a fix

Depending on what step 3 finds, write up one of:

* **Reproduction bug**: fix the inference script, confirm v10 actually sounds correct, update the
  earlier finding (audio comparison delivered to the user was invalid — say so plainly).
* **Real training defect isolated to late epochs**: recommend using `epoch_2nd_00014.pth` (or an
  even earlier checkpoint) instead of the "best" pick, or recommend a retraining run with the root
  cause fixed, as a new suggestion for a future task.
* **Real training defect from the start**: recommend the same t0010 training config never actually
  produced working audio at any epoch, however good val_loss looked — this would be a significant
  finding for how "training success" is judged in future runs (val_loss alone is not sufficient;
  the eval harness — text-to-audio + a basic intelligibility/noise check — must run before a
  training task can claim `completed`, not be deferrable to "later").

Whichever it is, produce a short, objective, automatable check (e.g. spectral flatness / silence
fraction / a tiny ASR round-trip on the synthesized audio) that a future training task can run to
catch "this checkpoint only produces noise" *before* claiming completion — closing the gap that let
this go unnoticed through all of t0010's verificators.

## Expected Outputs

- `code/` — the instrumented inference harness (kikiri-tts clone + submodules under `code/`,
  gitignored; the instrumented script itself committed).
- `results/control_test.md` — control-checkpoint result (pass/fail, with audio duration/RMS/peak
  stats as an objective substitute for "I listened and it sounded like X").
- `results/v10_diagnosis.md` — findings from step 3: is it the loader, is it training, which
  checkpoint (if any) actually produces speech.
- `results/results_summary.md` / `results_detailed.md` — root cause + recommendation, written so a
  future training task (or a correction against t0010) knows exactly what to do next.
- If a quick regression check is designed in step 4, save it to `code/` so a future task can reuse
  it, and name it in `results/suggestions.json` as a follow-up for the eval-harness pipeline itself.

## Dependencies

- `t0010_stage2_safeguarded_training` — produces both checkpoints under investigation
  (`data/run_v10/epoch_2nd_00016.pth`, `epoch_2nd_00014.pth`) and the packaged asset
  `assets/model/kokoro-v10-best`.
- Not dependent on `t0012` (LUFS normalization) — this is about synthesis quality of the trained
  model, unrelated to training-data audio quality.
