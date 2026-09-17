---
spec_version: "1"
task_id: "t0015_v11_duration_blowup_forensics"
research_stage: "code"
tasks_reviewed: 8
tasks_cited: 6
libraries_found: 2
libraries_relevant: 2
date_completed: "2026-09-17"
status: "complete"
---
## Task Objective

t0015 must root-cause why `kokoro-v11-best`'s StyleTTS2-native synthesis runs 73.95s for a ~10-word
sentence (15-30x the 2.5-4.9s control/v10 baseline) and sounds like droning babble to a human
despite passing `audio_quality_check.py`'s `is_likely_noise` gate (`clip_fraction=0.0048`,
`spectral_flatness=0.0058`). Per `task_description.md`, the task must: (1) log raw `pred_dur` values
and `pred_aln_trg` frame counts directly to localize the defect as predictor-calibration vs.
alignment-plumbing; (2) determine whether `predictor`/`predictor_encoder` actually received gradient
updates during t0014's 50-epoch run or rode along at LibriTTS-scale init; (3) run the harness across
5-10+ varied texts to see if the blowup is universal or text-dependent; (4) sweep
`alpha`/`beta`/`diffusion_steps`/`embedding_scale` as a cheap, no-retrain fix; (5) if that fails,
recommend (not execute) targeted `predictor`/`predictor_encoder` fine-tuning as a follow-up; and (6)
harden `audio_quality_check.py` with a duration-sanity signal, a longest-contiguous-non-silent-run
signal, and an evaluation of an ASR-round-trip check, then re-run the hardened gate as a
v10/v11/fixed three-way regression proving the new signals discriminate correctly. Direct
dependencies are `t0013_v10_synthesis_quality_forensics` (source of `audio_quality_check.py` and
`infer_styletts2.py`) and `t0014_v11_decoder_fix_retrain` (source of `kokoro-v11-best` and the
disclosed duration anomaly).

## Library Landscape

Two libraries exist in the project (`aggregate_libraries --format json --detail short`, 2 found, 2
relevant):

* **`tts_eval_harness`** (v0.1.0, created by `t0008_tts_eval_harness_baselines`, no corrections
  applied). Directly relevant: it already implements a **reference-based** duration-ratio explosion
  signal (`code/scoring.py:203-241`, `compute_duration_ratio(synth_wavs, ref_durations_s)` flags
  `ratio > 5.0` as an "explosion" against a *paired reference clip's* duration) and a WER/ASR
  round-trip via faster-whisper (`code/scoring.py:259+`, `compute_wer`, using
  `faster_whisper.WhisperModel` at `constants.WHISPER_MODEL_SIZE = "base.en"`, gated by
  `DURATION_RATIO_LOW=0.5`/`DURATION_RATIO_HIGH=2.0`). Both are precedent for, but not identical to,
  Key Question 6's requested checks — the new gate needs a **text-only** words-per-second estimate
  (no reference clip required, since `infer_styletts2.py` runs are one-off text->audio, not paired
  against a same-content reference recording) and a **contiguous-silence-run** signal neither
  function computes. Import path:
  `from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_wer, compute_duration_ratio`.
  Also exposes ready-made varied-text prompt sets (see Dataset Landscape) directly useful for Scope
  step 1's "5-10 short and long texts" requirement.
* **`t0009_training_safeguards`** (v0.1.0, created by `t0009_stage2_training_failure_forensics`, no
  corrections applied). Provides `HealthGate` (`code/health_gates.py`) with three gates (dur_loss at
  step 1, acoustic_norm, val_loss spike) — relevant background showing the *existing* Stage-2
  training-time gates never monitored final synthesized-audio duration or a text-length-normalized
  duration signal, only intermediate loss scalars, which is why a predictor-calibration defect could
  train to completion with 0 gate firings (confirmed in t0014's `data/run_v11/metrics.jsonl`, see
  Key Findings). Not directly imported by this task (t0015 does no training), but informs why the
  duration defect wasn't caught earlier and is cited as precedent for "gates must watch the actual
  output, not just aggregate losses."

## Key Findings

### The duration/frame-count plumbing that must be instrumented is already fully visible in `infer_styletts2.py`

`tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py:synthesize()` (lines 282-371, copied
unchanged from `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py`) is the exact
code path Key Question 1 asks about. The duration computation is:

```python
duration = model.predictor.duration_proj(x)          # line 342
duration = torch.sigmoid(duration).sum(axis=-1)        # line 343
pred_dur = torch.round(duration.squeeze()).clamp(min=1)  # line 344
pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))  # line 346
```

Neither `pred_dur` nor `pred_aln_trg`'s resulting shape is logged or returned anywhere in the
current script — `synthesize()` only returns `(wav, wall_time)` (line 371). This is the single,
minimal change needed for Scope step 1/2: instrument this exact function (or a copy of it) to also
return/log `pred_dur.tolist()`, `pred_dur.sum().item()` (the total frame count), and
`input_lengths.item()` (token count), which immediately distinguishes "predictor calibration
problem" (per-token `pred_dur` values individually plausible but summed over many tokens, or
individually huge) from "plumbing bug" (e.g. `pred_aln_trg` shape or the `for i in range(...)` fill
loop at lines 347-350 double-counting frames) [t0013] [t0014].
`model_params.model_params.max_dur = 50` in
`tasks/t0014_v11_decoder_fix_retrain/code/config_david_v11.yml:79` bounds what a single well-formed
`duration_proj` sigmoid-sum can output per token (StyleTTS2's predictor sums a `max_dur`-length
sigmoid vector per token) — a `pred_dur` value here that exceeds ~50 per token, or a token count
that is much larger than the ~10-word input implies (e.g. from phonemizer/IPA over-segmentation),
are the two concrete hypotheses this instrumentation should discriminate between [t0014].

### `alpha`/`beta`/`diffusion_steps`/`embedding_scale` are hardcoded, untuned LibriTTS-demo defaults — exactly as the task description suspects

`synthesize()`'s signature (`infer_styletts2.py:282-292`) is
`alpha: float = 0.3, beta: float = 0.7, diffusion_steps: int = 5, embedding_scale: float = 1.0`.
Only `diffusion_steps` is exposed as a CLI flag in `main()` (`--diffusion-steps`, default 5, line
381); `alpha`, `beta`, and `embedding_scale` are not CLI-configurable at all in either t0013's or
t0014's copy of this script — Key Question 4's "sweep" requires either adding CLI flags or calling
`synthesize()` directly in a small Python driver script, both of which are cheap, no-GPU changes
[t0013] [t0014]. These four values feed the diffusion sampler (`ADPM2Sampler`/`KarrasSchedule`,
lines 303-308, 327-333) that produces the style vector `s` used by
`model.predictor.text_encoder`/`lstm`/`duration_proj` — i.e., they are upstream of the duration
computation, so a bad style vector could itself skew `pred_dur` even if `duration_proj`'s weights
are reasonable, which is a third hypothesis worth instrumenting alongside Key Question 1/2.

### `predictor` and `predictor_encoder` DID receive gradient updates for all 50 epochs — evidence against the "rode along frozen" hypothesis, but the loss trajectory is flat, evidence for calibration failure

Key Question 2 asks whether `predictor`/`predictor_encoder` were "loaded but never meaningfully
fine-tuned." Reading `tasks/t0014_v11_decoder_fix_retrain/code/train_second_v11.py` directly answers
part of this with code, not speculation:

* `ignore_modules=["predictor_encoder", "msd", "mpd", "wd", "diffusion"]` at the `load_checkpoint()`
  call site (lines 316-321) means `predictor_encoder` is **excluded** from the `first_stage_path`
  load — it is instead initialized via
  `model.predictor_encoder = copy.deepcopy(model.style_encoder)` (line 330), i.e., copied from the
  (checkpoint-loaded) `style_encoder`, not left at LibriTTS/`build_model()` random init. `predictor`
  (unqualified) is **not** in `ignore_modules`, so it **is** loaded from `first_stage_path`
  (`epochs_2nd_00020.pth`, confirmed HiFi-GAN-shaped and tensor-finite by
  `results/checkpoint_forensics_v11.md`).
* `optimizer.step("predictor")` and `optimizer.step("predictor_encoder")` (lines 733-734) execute
  **unconditionally** on every training step where the generator gradients are finite — not gated
  behind `epoch >= joint_epoch` or `epoch >= diff_epoch` the way `style_encoder`/`decoder` (line
  742-743, gated `epoch >= joint_epoch`) and `diffusion` (line 736, gated `epoch >= diff_epoch`)
  are. Both modules received gradient updates from step 1 of epoch 1 through the end of epoch 50.

This rules out "predictor rode along completely unchanged from init" as the mechanism, but reading
the actual per-epoch `dur_loss` recorded in
`tasks/t0014_v11_decoder_fix_retrain/data/run_v11/metrics.jsonl` (620 lines, `StepLogger`-format
JSONL) shows the eval-time duration loss (`loss_align / iters_test`, `train_second_v11.py:1015`,
logged as `dur_loss` at `step==0`) **plateaus at 0.53-0.62 across the entire 50-epoch run** with no
clear downward trend after roughly epoch 8 (epoch 1: 0.620, epoch 8: 0.550, epoch 25: 0.570, epoch
50: 0.529) — the loss never converges to a small value the way `val_loss`/`dur_loss` did in
`t0009`'s reference v6c run (which reached 0.034 by epoch 6, per the `t0009-stage2-forensics-answer`
answer asset). Combined with the fact that the aggregate `val_loss` reported in t0014's
`results_summary.md` (best 0.3394 at epoch 38) looked healthy throughout, this is direct evidence
that the aggregate reconstruction loss masked a duration-specific calibration failure that never
actually converged — consistent with a **predictor calibration problem**, not a "never trained" or
pure "downstream plumbing" bug, though the instrumentation in Key Finding 1 above is still required
to confirm the failure is in `duration_proj`'s output scale itself rather than in how
`pred_aln_trg`/frame-count is constructed from otherwise-reasonable per-token values [t0009]
[t0014]. The `optimizer.step()` gating structure and the DP-aware `load_checkpoint()` both trace
back unchanged to `t0010_stage2_safeguarded_training`'s `joint_epoch=8` fix and DP-aware loader
(`train_second_v10.py`, carried forward per `train_second_v11.py`'s own module docstring) — only
`first_stage_path`, `train_data`, and the epoch/`joint_epoch`/`diff_epoch` schedule changed between
v10 and v11, so the unconditional `predictor`/`predictor_encoder` optimizer-step behavior documented
above was already true in t0010's run, not something introduced by t0014 [t0010].

### `audio_quality_check.py`'s existing three-signal design is a template to extend, not replace

`tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py` (copied logic-for-logic from
`t0013`'s version, 156 lines) implements `check_audio_quality(wav_path) -> AudioQualityResult`
(frozen dataclass: `rms`, `peak`, `silence_fraction`, `spectral_flatness`, `clip_fraction`,
`is_likely_noise`) with three independent, OR-combined noise signals
(`SPECTRAL_FLATNESS_NOISE_THRESHOLD=0.35`, `CLIP_FRACTION_NOISE_THRESHOLD=0.3`, plus a
`silence_fraction < 0.95` "not mostly silent" guard). The module's own docstring explicitly
documents that it was designed by adding a second signal (`clip_fraction`) after `spectral_flatness`
alone missed v10's clipping failure — the same "one signal misses one failure mode" pattern this
task's gate extension must repeat for the duration/silence-gap failure mode. The existing
`silence_fraction` computation (lines 79-88: 20ms frames, `-40dBFS` threshold, fraction below
threshold) already computes per-frame silence flags in a loop — this loop is the natural place to
also track the **longest contiguous run of non-silent frames** (Key Question 6's second signal) with
a small, local change (track a running counter and its max instead of only a total count), rather
than writing a new silence-detection pass from scratch [t0013] [t0014]. The module's `demo()`
function (lines 109-138) establishes the project's existing pattern for a runnable self-check: load
one known-good and one known-bad clip from a prior task's `results/audio_samples/`, assert the
heuristic classifies both correctly. This task's three-way regression check (Scope step 6: v10
should still fail, v11-as-shipped should now fail on the new signal, corrected output should pass
all signals) is a direct extension of that same pattern, now with three fixtures instead of two.

### Checkpoint-load instrumentation and tensor-forensics patterns are mature and directly reusable

Three independent, well-tested patterns exist for the kind of "confirm with evidence, not assertion"
diagnostics this task needs:

* **Per-module missing/unexpected key logging with a primary+fallback strategy**:
  `load_checkpoint_instrumented()` in `infer_styletts2.py` (lines 159-218) and the analogous
  `load_checkpoint()` in `train_second_v11.py` (lines 146-179) both load raw and `module.`-stripped
  variants, pick whichever has fewer missing/unexpected keys, and hard-fail (`RuntimeError`) on any
  core-module mismatch. `results/load_log_epoch_00048.json` (t0014) already confirms 0
  missing/unexpected on all 13 modules for the exact checkpoint under investigation here — ruling
  out a checkpoint-loading bug as the cause of the duration blowup, so this task's instrumentation
  can focus purely on the forward pass (`synthesize()`), not the load path [t0013] [t0014].
* **Tensor-level pre-flight forensics without running inference**:
  `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` (14KB) reads raw
  checkpoint tensors via `torch.load` only (no StyleTTS2 import, no inference) to classify decoder
  architecture (`hifigan` vs `istftnet`, via `generator.alphas.*` key presence — empirically
  confirmed, not guessed, per its own docstring) and check `torch.isfinite` + per-module
  weight-norm. This pattern (cheap, no-GPU, run-first) is exactly what Scope step 2 asks for and
  should be the first script written, before any inference-parameter sweep [t0013].
* **Isolating one module's contribution via a falsification probe**:
  `tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py` (7.3KB) loads every
  module normally from a checkpoint *except* one target module (there, `decoder`), which is left at
  fresh `build_model()` random init, to test whether that module alone is sufficient to reproduce a
  failure signature. The same pattern — e.g., swap in the LibriTTS control checkpoint's `predictor`
  while keeping v11's `decoder`/`style_encoder`, or vice versa — is a strong template for isolating
  whether the duration blowup traces specifically to `predictor`/`predictor_encoder`'s fine-tuned
  weights vs. some other module's influence on the style vector [t0013].

### Varied-text prompt sets already exist and cover Scope step 1's "5-10 short and long texts" requirement

`tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_100.json` (100 entries) and
`tasks/t0008_tts_eval_harness_baselines/data/val96_prompts.json` are both pre-built
`[{"text": ..., "ref_wav": ..., "ref_duration_s": ...}]` arrays, produced by
`code/prepare_prompts.py`'s `build_val96_prompts()`/filler-sampling logic from `data/val_list.txt`
and `data/11labs_david/` respectively. These give both short filler phrases and (via `val_list.txt`,
sourced from the same normalized corpus format as `t0012`'s `train_list_v5_normalized_clean.txt`)
longer sentences, with reference durations already computed — directly usable as the varied-text
sample for Scope step 1, and their `ref_duration_s` field is exactly the denominator needed for a
naive duration-ratio sanity check. Caveat: `ref_wav` values in the existing JSON are absolute paths
into `t0008`'s own worktree (e.g.
`/home/azureuser/rail-metarepo/real-repos/rail-arf-tts-worktrees/t0008_tts_eval_harness_baselines/...`)
which may no longer exist if that worktree was cleaned up — re-resolve `ref_wav` against
`tasks/t0008_tts_eval_harness_baselines/data/11labs_david/<basename>` (repo-relative) rather than
trusting the stored absolute path [t0008].

## Reusable Code and Assets

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/scoring.py` (part of the
  `tts_eval_harness` library). **What it does**:
  `compute_duration_ratio(synth_wavs: list[Path], ref_durations_s: list[float]) -> DurationRatioResult`
  (dataclass with `median`, `mean`, `std`, `explosion_count`, `per_clip`), flags `ratio > 5.0`.
  `compute_wer(synth_wavs, texts, duration_ratios) -> WerResult` using
  `faster_whisper.WhisperModel(constants.WHISPER_MODEL_SIZE)` (`"base.en"`) + JiWER, skipping clips
  outside `DURATION_RATIO_LOW=0.5`/`DURATION_RATIO_HIGH=2.0`. **Reuse method**: **import via
  library**
  (`from tasks.t0008_tts_eval_harness_baselines.code.scoring import compute_wer, compute_duration_ratio`).
  **Adaptation needed**: `compute_duration_ratio` requires a paired reference-clip duration; the new
  text-only duration-sanity signal for `audio_quality_check.py` needs a words-per-second estimate
  computed from input text alone (no reference clip), so this is a design reference, not a drop-in
  call, for that specific signal — but `compute_wer` is directly callable as-is for evaluating the
  optional ASR-round-trip layer (Key Question 6). **Line count**: ~40 lines each function.

* **Source**: `tasks/t0014_v11_decoder_fix_retrain/code/infer_styletts2.py` (428 lines) and
  `tasks/t0014_v11_decoder_fix_retrain/code/audio_quality_check.py` (156 lines) — the two files the
  task description explicitly says to extend. **What it does**: `infer_styletts2.py` is the
  instrumented StyleTTS2-native inference harness (`build_harness`, `compute_style`, `synthesize`,
  `load_checkpoint_instrumented`); `audio_quality_check.py` is the `check_audio_quality()` noise
  gate. **Reuse method**: **copy into task** (per the Cross-Task Code Reuse Rule — neither is a
  registered library; both are plain task-folder scripts). Copy from `t0014`'s versions specifically
  (not `t0013`'s), since `t0014`'s are the ones already adapted for the checkpoint/config under
  investigation and carry the accurate module docstrings about what changed. **Adaptation needed**:
  (1) in `synthesize()`, add `pred_dur`/`pred_aln_trg` frame-count logging and return them; (2)
  expose `alpha`/`beta`/`embedding_scale` as CLI args alongside the existing `--diffusion-steps`;
  (3) in `check_audio_quality()`, add a duration-sanity field (needs the input text or a
  words-per-second constant passed in) and a `longest_nonsilent_run_s`-style field computed inside
  the existing 20ms-frame loop (lines 79-88). **Line count**: instrumentation additions are small
  (~20-40 lines each); the base files being copied are 428 and 156 lines respectively.

* **Source**: `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` (14KB) and
  `tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py` (7.3KB). **What it
  does**: cheap (no-GPU, no-inference) tensor-level checkpoint forensics and a module-isolation
  falsification-probe pattern. **Reuse method**: **copy into task** and adapt (not registered as
  library code). **Adaptation needed**: retarget `CORE_MODULES`/checkpoint paths at
  `kokoro-v11-best`/`epoch_00048.pth` and the LibriTTS control; for a predictor-isolation probe,
  change which module is swapped (from `decoder` to `predictor`/`predictor_encoder`). **Line
  count**: ~350 and ~180 lines respectively (estimated from file size).

* **Source**: `tasks/t0014_v11_decoder_fix_retrain/code/build_reference_concat.py` (53 lines).
  **What it does**: `build_reference_concat(out_path: Path) -> Path` concatenates the same 3 named
  `11labs_david` reference clips (`lining_up_suggestions_17/10`, `putting_them_head_to_head_15`)
  with 0.2s silence gaps into a single 5.48s reference-style WAV, reused unchanged across
  t0013/t0014 for direct comparability of gate results. **Reuse method**: **copy into task** and
  reuse unchanged if re-testing the same gate texts (Scope step 4 explicitly asks to re-synthesize
  `lining_up_suggestions_17/10`, `putting_them_head_to_head_15` for direct comparability). **Line
  count**: 53 lines, trivial to copy verbatim.

* **Source**: `tasks/t0008_tts_eval_harness_baselines/data/filler_prompts_100.json` and
  `data/val96_prompts.json`. **What it does**: pre-built varied-text prompt lists with reference
  durations. **Reuse method**: **copy into task** (data file, not library code — the library only
  registers the code that produces these, not the JSON outputs themselves) or regenerate via
  `code/prepare_prompts.py` (importable from the `tts_eval_harness` library) against this repo's
  current `data/v4/val_list.txt` / `data/11labs_david/`. **Adaptation needed**: re-resolve `ref_wav`
  paths to be worktree-independent (see Key Findings caveat). **Line count**: N/A (data, not code).

## Lessons Learned

* **Aggregate losses hide localized failures — this exact pattern has now recurred twice.** t0013
  found `val_loss` looked healthy for 17 epochs while the decoder produced 75-81%-clipped noise
  (`ignore_modules` bug). t0014's own `dur_loss` plateau (0.53-0.62 across 50 epochs, see Key
  Findings) shows the same pattern for duration: the aggregate `val_loss` (best 0.3394) looked
  healthy while the duration-specific component never converged. Both t0013 and t0009 independently
  concluded that per-component signals, not aggregate loss curves, are needed to catch defects —
  directly supporting this task's Scope step 6 mandate to add gate signals that inspect the actual
  output rather than trust an upstream metric [t0009] [t0013] [t0014].
* **"Gate passed" claims in this project chain are narrowly pre-registered and explicitly scoped —
  t0014's own `v11_gate_verdict.md` already disclosed the duration anomaly as a known, non-blocking
  limitation rather than hiding it**, which is why this task exists as a planned follow-up
  (`source_suggestion: S-0014-01`) rather than a surprise regression. `plan/plan.md`'s "no other
  metric may be substituted" Rejection Criteria pattern (seen in both t0013 and t0014) is worth
  carrying into this task's own plan: pre-register exactly which signal(s) constitute "the cheap fix
  worked" before running the alpha/beta/diffusion_steps/embedding_scale sweep, so a
  partially-improved but still-anomalous duration can't be silently reported as success [t0014].
* **CPU StyleTTS2-native inference is slow but reliable**: t0013/t0014's `.venv-styletts2` CPU
  environment (torch 2.5.1) has already been built and validated twice; `rtf` figures of 3.2-5.5x
  (`t0013`, `t0014` `results_summary.md`) mean a single synthesis of a short sentence takes seconds
  to tens of seconds on CPU, and t0014's own 73.95s-audio synthesis took 235.2s wall time — a
  10-30-item text sweep plus a parameter sweep is feasible on CPU without provisioning a GPU,
  matching Key Question 4's "no GPU required for this step" framing [t0013] [t0014].
* **Weight-norm parametrization key renaming is a recurring gotcha** when loading checkpoints
  trained under different torch/StyleTTS2-fork versions — `infer_styletts2.py`'s
  `_rename_legacy_parametrization_keys()` (lines 99-144) documents the exact `weight_g`/`weight_v`
  vs. `parametrizations.weight.original0/1` key-naming difference discovered empirically; only
  needed for the external LibriTTS control checkpoint, not for this project's own checkpoints
  (already trained under the current fork's `models.py`) — worth knowing this function exists rather
  than re-deriving it if a predictor-isolation probe needs to load an external control's `predictor`
  module [t0013].

## Recommendations for This Task

1. **Copy `t0014`'s `infer_styletts2.py` and `audio_quality_check.py` into `t0015/code/`** (not
   `t0013`'s — `t0014`'s are the checkpoint/config-current versions) and instrument `synthesize()`
   to log/return `pred_dur` (per-token values + sum) and the resulting `pred_aln_trg` frame count
   before doing anything else — this directly answers Key Question 1/Scope step 2 and requires no
   GPU, no parameter sweep, and no new inference run beyond what Scope step 1 already requires.
2. **Copy `t0013`'s `inspect_checkpoint.py` pattern** to add a `predictor`/`predictor_encoder`
   tensor-level check (weight-norm summary vs. the LibriTTS control's
   `predictor`/`predictor_encoder` values already recorded in
   `checkpoint_forensics_v11.md`/`checkpoint_forensics.md`) as a fast, no-inference cross-check on
   Key Finding 3's training-log evidence (flat `dur_loss`) before spending any time on the parameter
   sweep.
3. **Expose `alpha`/`beta`/`embedding_scale` as CLI flags** on the copied `infer_styletts2.py`
   alongside the existing `--diffusion-steps`, and run the Scope step 3 sweep as a small grid (a
   Python driver script calling `synthesize()` directly is simpler than shelling out per
   combination). Pre-register what "fixed" means (e.g., duration within Nx of the naive
   words-per-second estimate AND `is_likely_noise=False`) before running the sweep, per the Lessons
   Learned rejection-criteria pattern.
4. **Extend `audio_quality_check.py`'s existing 20ms-frame silence loop** (lines 79-88) to also
   track longest contiguous non-silent run, and add a separate text-length-aware duration-sanity
   function in the same module (following the module's own stated convention of "kept in the same
   module for one source of truth", cited from S-0013-04). Do not build a reference-clip-based check
   for this — reuse `compute_duration_ratio`'s reference-based approach only if a paired reference
   recording of the same content exists; otherwise the new signal must be text-only
   (words-per-second), which is not what either existing `tts_eval_harness` function computes.
5. **Import `compute_wer` from the `tts_eval_harness` library** (already wraps faster-whisper) to
   evaluate the optional ASR-round-trip layer from Key Question 6/Scope step 6, rather than
   re-deriving a transcription pipeline — this directly answers the task description's own question
   about whether faster-whisper is practical to reuse here.
6. **Copy `build_reference_concat.py` unchanged** and reuse `filler_prompts_100.json`/
   `val96_prompts.json` (re-resolving `ref_wav` to repo-relative paths) for the varied-text sample
   required by Scope step 1, rather than hand-picking new texts.
7. **Run the hardened gate as a v10/v11/corrected three-way regression** using `t0013`'s
   `results/audio_samples/ft/v10_epoch16_primary.wav` (or equivalent) and `t0014`'s
   `results/audio_samples/ft/v11_best.wav` as the two known fixtures, following the exact pattern
   `audio_quality_check.py`'s own `demo()` function already establishes, extended to three signals
   and three fixtures instead of two.

## Task Index

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Source of the `tts_eval_harness` library (`compute_duration_ratio`, `compute_wer`)
  and the pre-built `filler_prompts_100.json`/`val96_prompts.json` varied-text prompt sets reusable
  for this task's Scope step 1.

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Source of the `HealthGate`/`StepLogger` training safeguards library and the
  `t0009-stage2-forensics-answer` answer asset, which establishes the "aggregate loss hides
  localized failure" pattern this task's duration-plateau finding repeats, and gives a converged
  `dur_loss` reference point (0.034 by epoch 6 in v6c) to contrast against v11's flat 0.53-0.62.

### [t0012]

* **Task ID**: `t0012_v5_corpus_normalize_and_reaudit`
* **Name**: v5 corpus LUFS normalization and clipped-fraction re-audit
* **Status**: completed
* **Relevance**: Source of `train_list_v5_normalized_clean.txt` (1,531 clips), the corpus t0014
  trained on and referenced by the task description as a source of varied text for Scope step 1.

### [t0013]

* **Task ID**: `t0013_v10_synthesis_quality_forensics`
* **Name**: v10 checkpoint synthesis quality forensics
* **Status**: completed
* **Relevance**: Direct dependency. Original author of `infer_styletts2.py`,
  `audio_quality_check.py`, `inspect_checkpoint.py`, and `random_decoder_probe.py` — the exact
  tools/patterns this task extends and reuses, plus the control-checkpoint audio samples needed for
  the three-way gate regression.

### [t0014]

* **Task ID**: `t0014_v11_decoder_fix_retrain`
* **Name**: Kokoro Stage 2 v11: decoder-init fix, full normalized corpus retrain
* **Status**: completed
* **Relevance**: Direct dependency. Source of `kokoro-v11-best`, the checkpoint under investigation;
  its copies of `infer_styletts2.py`/`audio_quality_check.py`, `config_david_v11.yml`,
  `train_second_v11.py` (with the `ignore_modules`/`optimizer.step` evidence used in Key Finding 3),
  `data/run_v11/metrics.jsonl` (per-epoch `dur_loss` trajectory), and the gate-passing texts this
  task must re-test for direct comparability.

### [t0010]

* **Task ID**: `t0010_stage2_safeguarded_training`
* **Name**: Kokoro Stage 2: safeguarded training with joint_epoch=8
* **Status**: completed
* **Relevance**: Ancestor of `train_second_v10.py`/`train_second_v11.py`'s shared training-loop
  structure (DP-aware loader, per-epoch checkpointing); background context for why `joint_epoch` and
  `ignore_modules` are structured the way they are in the config this task investigates.
