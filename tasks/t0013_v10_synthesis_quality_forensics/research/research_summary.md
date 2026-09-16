# Research Summary — t0013_v10_synthesis_quality_forensics

## Key Findings (top 10 insights directly actionable for this task)

1. **Leading hypothesis:** `config_david_v10.yml` is the only Stage 2 config in the project with
   `model_params.decoder.type: hifigan` -- every other config (v4, v5, v6c, v6d) uses `istftnet`.
   `hifigan` has no `gen_istft_*` keys and a fully-learned 4-stage upsampler (`[10,5,3,2]`),
   structurally different submodule names/shapes than `istftnet`.
2. `train_second_v10.py:load_checkpoint()` loads `first_stage_v3.pth` (an `istftnet`-shaped
   checkpoint) into `decoder`, which is not in `ignore_modules`. The loader only raises on
   `len(matched)==0`, so a *partial* match (shared convs match, generator/vocoder layers don't)
   passes silently -- the HiFi-GAN vocoder proper likely trained from random init for only 17
   epochs, consistent with "pure noise."
3. **Fast falsifier, do first before any inference code:** `torch.load` the raw checkpoint, check
   `net["decoder"]` key names (HiFi-GAN uses `ups.*`/`resblocks.*`, ISTFTNet doesn't). Also run
   `torch.isfinite` + per-module weight-norm on all 13 modules, diffed between epoch 14 and 16.
4. The `strict=False` + aggregate-only missing/unexpected-count anti-pattern already caused real
   bugs twice ([t0009] training loader, [t0008] `adapters.load_kokoro_model_with_checkpoint`) -- the
   new loader must log missing/unexpected **per module** and hard-fail on any nonzero count for
   `bert`, `bert_encoder`, `predictor`, `decoder`, `text_encoder`, `predictor_encoder`,
   `style_encoder`, `diffusion`.
5. `kokoro.KModel`/`KPipeline` and t0008's `adapters.load_kokoro_model_with_checkpoint()` are
   independently broken for a `hifigan`-decoder checkpoint, regardless of t0010's disk-full issue.
   Use StyleTTS2's native `models.py` path per `task_description.md`, not these.
6. `data/run_v10/metrics.jsonl` has `"step": 0` on every record (validation-time only) --
   `disc_loss`, `grad_norm_*`, `loss_total`, `skip_count`, `lr` are null throughout, so per-step
   signal is unavailable. `acoustic_norm` stayed smooth (6.4-8.5) epochs 9-17, unlike v5's/v6's
   clear divergence -- weakly supports architecture-mismatch over late-training GAN collapse.
7. Prior precedent ([t0002], v4 packaged-bundle investigation) shows the only method that reliably
   separated "harness bug" from "real training defect" is A/B testing the identical inference code
   path against a known-good control checkpoint. Follow it rigorously (Scope 2) before concluding
   anything about v10.
8. Two registered libraries are relevant but neither is directly importable into the new
   StyleTTS2-native CPU harness: `tts_eval_harness` (t0008, ISTFTNet/KModel-coupled) and
   `t0009_training_safeguards` (training-loop-coupled) -- reference patterns only, or import
   `t0009_training_safeguards` read-only for replaying gate checks over `run_v10`'s JSONL.
9. Cross-task code reuse rule: copy, don't import, logic from `extract_decoder.py` (t0008) and
   `train_second_v10.py:load_checkpoint()` (t0010) -- only the two libraries above are
   import-eligible, and both need rewriting for this task's purposes.
10. `dvc pull` is required before starting: neither `epoch_2nd_00016.pth` nor `epoch_2nd_00014.pth`
    (identical 2,086,820,872 bytes, consistent with same architecture) is pulled locally yet.

## Best Approaches (top 3 recommended implementation approaches from research)

### Approach 1: Cheap tensor forensics before any inference code

`torch.load` the raw checkpoint (no GPU/venv needed), check `net["decoder"]` submodule naming
(`ups.*`/`resblocks.*` = HiFi-GAN), run `torch.isfinite` and per-module weight-norm across all 13
modules, diffed between epoch 14 and 16 and against `first_stage_v3.pth`'s decoder. Can
confirm/refute the architecture-mismatch hypothesis in minutes, before building the CPU venv.

### Approach 2: Instrumented StyleTTS2-native harness, control-validated

Clone `kikiri-tts` + submodules under `code/`, build the CPU venv per `task_description.md`
(`torch==2.5.1`, pin <2.6), adapt `Inference_LibriTTS.ipynb` into a script whose loader logs
per-module missing/unexpected key counts (extend, don't copy, `load_checkpoint()`'s shape-matching
structure -- change the failure condition from "zero matched" to "any missing/unexpected on a core
module"). Validate against the base pretrained LibriTTS checkpoint (control) before touching v10,
per [t0002]'s precedent.

### Approach 3: Log cross-reference as corroborating, not primary, evidence

Use `data/run_v10/metrics.jsonl`'s epoch-level `val_loss`/`dur_loss`/`acoustic_norm`/`f0_loss` only
as secondary support (smooth, no divergence) -- primary diagnosis must come from checkpoint tensor
inspection (Approach 1) and control-validated inference (Approach 2), since the JSONL log lacks
per-step discriminator/gradient signal entirely.

## Reusable Code / Assets

* `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py` (~45 lines,
  `_convert_key`/`extract`) -- key-renaming/DDP-prefix-stripping pattern; copy and extend with
  per-module missing/unexpected logging, no shape/architecture validation as-is.
* `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:87-116` (`load_checkpoint`, ~30
  lines) -- DP-aware shape-matching loader; copy as reference only, raise the failure bar from "zero
  matched" to "any missing/unexpected on core modules."
* `tasks/t0009_stage2_training_failure_forensics/code/jsonl_logger.py` (`StepLogger`) and
  `health_gates.py` (`HealthGate`/`GateResult`) -- importable via `t0009_training_safeguards`
  library for read-only replay of gate checks over `data/run_v10/metrics.jsonl`; not needed for the
  inference harness itself.
* `tasks/t0010_stage2_safeguarded_training/code/paths.py` (26 lines) -- copy the one-`paths.py`-
  per-task pattern for `data/run_v10/*` and this task's `results/audio_samples/`.
* Not reusable, reference only: `eval_all_checkpoints.py` (t0010) and `adapters.py`
  (`load_kokoro_model_with_checkpoint`, t0008) -- both hard-coupled to `kokoro.KModel`
  (ISTFTNet-only), the wrong path for a `hifigan`-decoder checkpoint.

## Key Papers (top 5, with finding most relevant to this task)

(not generated -- research-papers step skipped)

## Risks Flagged in Research

* Silent `strict=False` / aggregate-only missing-key counts is a recurring project-wide failure mode
  (already hit [t0009], [t0008]) -- log per-module counts or risk a third repeat.
* `t0010`'s deferred eval harness couldn't have caught this even without the disk-full issue, since
  it routes through the same broken `kokoro.KModel` path -- disk-full isn't the sole cause.
* `data/run_v10/metrics.jsonl` step-level fields (`disc_loss`, `grad_norm_*`, `lr`, `skip_count`)
  are null throughout -- don't expect new signal from Key Question 5 beyond what's summarized here.
* Config drift has caused undetected defects before ([t0006]'s `epochs_2nd` left at 10 instead of
  15; [t0009]'s three lost code patches after VM wipes) -- the `hifigan`/`istftnet` drift in
  `config_david_v10.yml` fits the same pattern.

## Full Detail Available In

* `tasks/t0013_v10_synthesis_quality_forensics/research/research_papers.md` -- (not generated --
  step skipped)
* `tasks/t0013_v10_synthesis_quality_forensics/research/research_internet.md` -- (not generated --
  step skipped)
* `tasks/t0013_v10_synthesis_quality_forensics/research/research_code.md` -- 12 tasks reviewed, 7
  cited, 2 libraries found/relevant
