---
spec_version: "1"
task_id: "t0013_v10_synthesis_quality_forensics"
research_stage: "code"
tasks_reviewed: 12
tasks_cited: 7
libraries_found: 2
libraries_relevant: 2
date_completed: "2026-09-16"
status: "complete"
---
## Task Objective

Determine whether `kokoro-v10-best` (raw checkpoint `epoch_2nd_00016.pth`, produced by
`t0010_stage2_safeguarded_training`) produces noise instead of speech because of a bug in an ad hoc,
never-committed inference reproduction, or because of a real training defect, and act on whichever
is true. The task must build a reusable, instrumented StyleTTS2-native inference harness (missing/
unexpected key counts logged per module, not swallowed by `strict=False`), validate that harness
against a known-good control checkpoint, diagnose `epoch_2nd_00016.pth` (primary, epoch 17) and
`epoch_2nd_00014.pth` (backup, epoch 15) with it, check the raw checkpoint tensors for NaN/Inf and
weight-norm anomalies, cross-reference `t0010`'s per-epoch JSONL log for anomalies val_loss could
have masked, and ship a cheap automatable "does this checkpoint even produce a voice" regression
check for future training tasks — closing the gap that let `t0010` claim `completed` without ever
listening to its own output.

## Library Landscape

Two libraries are registered, discovered via `aggregate_libraries --format json --detail short` (2
found, 2 relevant):

* **`tts_eval_harness`** (`tasks/t0008_tts_eval_harness_baselines/code/`, v0.1.0, created by
  `t0008_tts_eval_harness_baselines`). No corrections/replacements present in the aggregator output.
  Relevant because it contains `extract_decoder.extract()` — the exact five-module packaging
  function used to turn `epoch_2nd_00016.pth` into the `kokoro-v10-best.pth` asset that (per the
  task's own motivation) fails to load through `kokoro.KModel` — and
  `adapters.load_kokoro_model_with_checkpoint()`, which is the code path that already performs a
  silent `strict=False` load against the wrong decoder architecture (see Key Findings). Import path:
  `from tasks.t0008_tts_eval_harness_baselines.code.extract_decoder import extract`.
* **`t0009_training_safeguards`** (`tasks/t0009_stage2_training_failure_forensics/code/`, v0.1.0,
  created by `t0009_stage2_training_failure_forensics`). No corrections present. Relevant because
  its `HealthGate`/`StepLogger`/`CheckpointManager` produced the exact `data/run_v10/metrics.jsonl`
  and `checkpoint_manifest.json` this task's Key Question 5 must cross-reference, and its "fail
  loudly on checkpoint mismatch" design principle is the direct precedent for the
  missing/unexpected-key instrumentation this task's Scope §1 asks for. Import path:
  `from tasks.t0009_stage2_training_failure_forensics.code.health_gates import HealthGate` (and
  siblings `jsonl_logger`, `checkpoint_manager`, `run_config`).

Neither library can be imported into this task's CPU/StyleTTS2-native harness as-is: both depend on
`kokoro.KModel`/PyTorch training internals (ISTFTNet-only for the former, training-loop-coupled for
the latter). They are documented here as **prior art to imitate**, not as direct imports for the new
`kikiri-tts`/native-StyleTTS2 script — see Reusable Code and Assets.

An answer asset, `t0009-stage2-forensics-answer` (`t0009_stage2_training_failure_forensics`,
confidence: high), is also directly relevant and is treated as prior synthesized finding throughout
Key Findings below.

## Key Findings

### The "silent `strict=False` checkpoint load" anti-pattern has already bitten this project twice — this task is set up to hit it a third time if not instrumented carefully

[t0009]'s entire root-cause finding was that upstream StyleTTS2's `load_checkpoint` uses
`strict=False` and can silently load **zero** parameters when a DataParallel-saved (`module.`-
prefixed) checkpoint is loaded into a non-wrapped model — "all 13 modules printed as loaded with no
exception raised" is structurally the same failure class task_description.md warns about for the ad
hoc reproduction. [t0009]'s fix, applied in `train_second_v10.py:load_checkpoint()` (used by
[t0010]), only raises `RuntimeError` when **zero** parameters match
(`tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:87-116`) — a *partial* mismatch
(e.g. every generator/vocoder key failing to match while encoder/predictor keys match fine) still
passes silently, logging only `"{key} loaded: {matched}/{total} params"` with no per-key
missing/unexpected breakdown. [t0008]'s `adapters.load_kokoro_model_with_checkpoint()`
(`tasks/t0008_tts_eval_harness_baselines/code/adapters.py:291-330`) repeats the exact same shape: it
calls `model.load_state_dict(merged, strict=False)` and logs only aggregate `missing`/`unexpected`
**counts**, never which modules they belong to. This task's Key Question 1 explicitly asks for
*per-module* missing/unexpected counts, which is a strictly stronger bar than either existing
implementation meets — do not copy either loader verbatim; use them only as a starting point and add
the per-module breakdown task_description.md requires.

### The five-module packaging pipeline (`extract_decoder.extract()`) performs a pure key rename, not an architecture reconciliation — it cannot detect a decoder-type mismatch

[t0002] discovered the "five-module, not decoder-only" packaging requirement (loading only `decoder`
into `KModel` causes ~10x duration explosions) and [t0008] generalized it into `extract_decoder.py`
(96 lines). `extract()` (`tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py:28-71`)
does exactly three things per module: strip `module.` DDP prefixes, rename
`.parametrizations.weight.original0/1` → `.weight_g/.weight_v`, and assert the five top-level keys
are present. It performs **no shape or architecture validation** — it copies whatever tensors are
under `net["decoder"]` verbatim. This is precisely why the packaged `kokoro-v10-best.pth` produced
from `epoch_2nd_00016.pth` fails to load into `kokoro.KModel` with "tensor shape mismatches on
`decoder.generator.*`" (per this task's own motivation): `extract()` has no way to know the source
checkpoint's decoder was built as a HiFi-GAN generator, not the ISTFTNet generator `kokoro.KModel`
expects.

### `config_david_v10.yml` is the only StyleTTS2 Stage 2 config in this entire project with `model_params.decoder.type: hifigan` — every other config, including the one that produced its own `first_stage_path`, uses `istftnet`

This is the most significant new finding of this research pass, not previously documented anywhere
in [t0010]'s plan, research, or results files
(`grep -n hifigan tasks/t0010_stage2_safeguarded_training/{plan,research,results}/**` returns
nothing outside the config file itself and the model asset's mechanical copy of the config). A
repo-wide grep for `type: hifigan` / `type: istftnet` across every committed Stage 1/Stage 2 config
shows:

| Config | Task | `model_params.decoder.type` |
| --- | --- | --- |
| `config_david_v4.yml` | [t0001] | `istftnet` |
| `config_david_v5.yml` / `config_david_v5_stage2.yml` | [t0005] | `istftnet` |
| `config_david_v6_stage2.yml`, `v6b`, `v6c`, `v6d` | [t0006] | `istftnet` (all four) |
| `data/configs/t0006_run03_v6c.yml`, `t0006_run04_v6d.yml` (forensic copies) | [t0009] | `istftnet` |
| `config_david_v10.yml` | [t0010] | **`hifigan`** |

The `istftnet` decoder block (e.g. `config_david_v6c_stage2.yml`) carries `gen_istft_hop_size: 5`,
`gen_istft_n_fft: 20`, and a two-stage `upsample_rates: [10, 6]` / `upsample_kernel_sizes: [20, 12]`
(total 60x learned upsampling, the rest done by inverse-STFT). The `hifigan` block in
`config_david_v10.yml` has **no** `gen_istft_*` keys and a four-stage
`upsample_rates: [10, 5, 3, 2]` / `upsample_kernel_sizes: [20, 10, 6, 4]` (300x, fully learned
upsampling to raw waveform, no ISTFT stage) — a structurally different generator with different
submodule names and parameter shapes, not a superset/subset of ISTFTNet's.

Critically, [t0010]'s own `train_second_v10.py:load_checkpoint()` loads `first_stage_path` (the
Stage 1 checkpoint `first_stage_v3.pth`, the *same* file [t0006]'s `v6c`/`v6d` runs — both
`istftnet` configs — loaded successfully) into the `decoder` module **without** `decoder` in
`ignore_modules` (`tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:248-266`, the
`ignore_modules` list is `["predictor_encoder", "msd", "mpd", "wd", "diffusion"]` — `decoder` is not
excluded). Because `first_stage_v3.pth`'s decoder was almost certainly saved from an
`istftnet`-shaped model (every other run that loaded it used `istftnet`), and `train_second_v10.py`
builds its `model["decoder"]` as a `hifigan` generator (`build_model(model_params, ...)` with
`model_params.decoder.type == "hifigan"`), the shape-matching logic in `load_checkpoint()`
(`matched = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}`)
would only match whatever decoder submodule keys happen to be architecture-agnostic (e.g. shared
pre-generator convs), while the vocoder/generator layers proper — the part that actually renders a
waveform — would be left at `build_model`'s fresh random initialization, not the converged Stage 1
weights. `len(matched) > 0` (some keys do match), so `load_checkpoint`'s "raise on 0/N matched"
guard never fires and training proceeded silently for all 17 epochs with the parameter-count
assertion reported as "passed" ([t0010] `results_summary.md`) — that assertion only checks aggregate
match rate, not per-module correctness, so it would not have caught a decoder-only mismatch either.
A randomly-initialized HiFi-GAN vocoder then had only 17 epochs (8 pre-GAN warmup + 9 post-GAN,
`joint_epoch=8`) to converge from scratch — vocoders typically need substantially more steps than
that to produce intelligible audio, which is fully consistent with "pure noise, no voice at all."
This hypothesis is falsifiable cheaply (Key Question 4/step 4 of the task): compare per-module
weight-norm statistics between `first_stage_v3.pth`'s `decoder` and `epoch_2nd_00016.pth`'s
`decoder`, and check whether the generator submodule names in `epoch_2nd_00016.pth`'s `decoder`
state dict match `hifigan`-style names (`ups.*`, `resblocks.*`) rather than ISTFTNet-style names —
this should be the first thing the instrumented harness checks, before spending time on full
inference.

### `t0010`'s own deferred eval harness could not have caught a decoder architecture mismatch even if the VM disk had not filled up

[t0010]'s `eval_all_checkpoints.py` (596 lines,
`tasks/t0010_stage2_safeguarded_training/code/eval_all_checkpoints.py`) calls `extract_checkpoint()`
→ [t0008]'s `extract_decoder.extract()` → `synthesize_fillers()` → [t0008]'s
`adapters.load_kokoro_model_with_checkpoint()` — the same `kokoro.KModel` (ISTFTNet-only) path this
task's own motivation identifies as incompatible with a `hifigan`-decoder checkpoint. Even with
`MIN_SPEAKER_SIM_GATE = 0.35` and `MAX_CLIP_DURATION_S = 30.0` gates defined in
`eval_all_checkpoints.py`, the harness would have hit the same tensor-shape-mismatch failure (or, if
`KModel`'s `load_state_dict(strict=False)` degrades gracefully instead of raising, silently scored a
partially-random decoder and likely failed the `MIN_SPEAKER_SIM_GATE` check, which *would* have
surfaced the problem, just not with a diagnosis). This means the "harness eval never ran, disk full"
framing in [t0010]'s `results_summary.md` is necessary-but-not-sufficient context: **even a
successful disk-space fix would not have produced usable audio from this specific harness path**,
because the eval harness and the training config disagree on decoder architecture. The reusable
harness this task builds must go around `kokoro.KModel` entirely, as `task_description.md` already
specifies (native `StyleTTS2/models.py` inference, not the `kokoro` pip package).

### The safeguard JSONL log has the schema for per-step diagnostics but `train_second_v10.py` never populates it — Key Question 5 has a smaller answer space than the task description assumes

`tasks/t0010_stage2_safeguarded_training/data/run_v10/metrics.jsonl` (17 non-final records + 1
terminal null record) was inspected directly. Every record has `"step": 0` and non-null values only
for `dur_loss`, `f0_loss`, `val_loss`, `acoustic_norm`; `loss_total`, `disc_loss`, `ce_loss`,
`mel_loss`, `grad_norm_msd`, `grad_norm_mpd`, `grad_norm_decoder`, `grad_norm_style_encoder`,
`skip_count`, and `lr` are `null` in every record. [t0009]'s `StepLogger`
(`tasks/t0009_stage2_training_failure_forensics/code/jsonl_logger.py`, 62 lines) supports an
arbitrary per-call `record` dict and does not enforce which fields are populated — the schema
exists, but `train_second_v10.py` only calls `step_logger.log()` once per epoch, at the validation
checkpoint (`"step": 0` on every row), never at intra-epoch training steps. This means Key Question
5's "does `metrics.jsonl` show a per-step signal that the epoch-level val_loss average could have
masked" has a documentable, negative-leaning answer from inspection alone: **no per-step
discriminator loss or gradient-norm data was ever recorded**, so that specific signal is unavailable
in this run's log — the diagnosis has to come from the checkpoint tensors themselves (NaN/Inf,
weight-norm per module, generator submodule naming) rather than from step-level training dynamics.
`acoustic_norm` (the one populated post-GAN health-gate signal) stayed in a narrow 6.4–8.5 band
across epochs 9-17 with no spike, consistent with [t0010]'s "0 health gate events" claim and
*inconsistent* with a GAN-divergence explanation for the noise — this weakly supports the
architecture-mismatch hypothesis over a late-training-regression hypothesis, since a genuine
adversarial-training collapse would typically show in `acoustic_norm` or `val_loss` (both stayed
smooth: 0.797 best at epoch 16, 0.853 at epoch 17 — nothing resembling [t0005]'s 0.848→2.070
divergence or [t0006]'s repeated diverge-at-epoch-7 pattern).

### Prior "noisy audio" incidents in this project were consistently architecture/packaging bugs, not training divergence, once someone actually listened

[t0002]'s results (the only prior task to synthesize audio from an intermediate checkpoint and
listen/spectrogram-inspect it) found v4's packaged bundle produced a persistent 3-3.5 kHz tonal
artifact and 10x-89x duration explosions — root-caused to (a) an extraction bug that skipped four of
five modules (fixed by generalizing to `extract_decoder_generic.py`, the direct ancestor of
[t0008]'s `extract_decoder.py`) and (b) a genuine defect in v4's own predictor, isolated by
A/B-testing against v3's known-good bundle through the *identical* code path. That A/B-against-a-
known-good-control methodology is exactly what this task's Scope §2 (control test against the base
pretrained StyleTTS2 checkpoint) replicates, and it is the right pattern: [t0002] proved the
harness/extraction code was not the source of v4's problem only by showing it produced clean audio
on v3's checkpoint through the same path. This task should hold itself to the same standard before
concluding anything about `v10`.

## Reusable Code and Assets

* **Source**: `tasks/t0008_tts_eval_harness_baselines/code/extract_decoder.py` (96 lines). **What it
  does**: `extract(*, ckpt_path: Path, out_path: Path) -> dict[str, int]` strips `module.` DDP
  prefixes and renames `.parametrizations.weight.original0/1` → `.weight_g/.weight_v` for the five
  modules in `CHECKPOINT_MODULES` (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`),
  asserting exact key coverage; `validate_packaged()` re-checks a saved packaged file. **Reuse
  method**: copy into task (do not import — see Cross-task rule) as the starting point for a
  key-renaming helper in the new StyleTTS2-native loader, but it must be extended, not used as-is,
  because it never checks shape/architecture compatibility (see Key Findings). **Adaptation
  needed**: add per-module `missing`/`unexpected` key-count logging against the StyleTTS2-native
  `models.py` module objects (not `kokoro.KModel`), and do not assume the five-module Kokoro-format
  split applies — the native StyleTTS2 checkpoint has 13 modules (`style_encoder`,
  `predictor_encoder`, `diffusion`, `text_aligner`, `pitch_extractor`, `mpd`, `msd`, `wd`, plus the
  five above) per `task_description.md`. **Line count**: ~45 lines of directly adaptable logic (the
  `_convert_key`/`extract` functions).

* **Source**: `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:87-116`
  (`load_checkpoint`). **What it does**: DP-aware loader — strips `module.` prefix per key, computes
  a `matched` dict filtered by both key presence and shape equality, raises `RuntimeError` if zero
  keys matched for any module, otherwise loads with `strict=False` and prints
  `"{key} loaded: {matched}/{total} params"`. **Reuse method**: copy into task as a reference
  pattern only — do not reuse verbatim; its "raise only if fully zero matched" threshold is weaker
  than this task's requirement (any nonzero missing/unexpected on a should-be-fully-covered module
  is a hard failure per `task_description.md`). **Adaptation needed**: change the failure condition
  from `len(matched) == 0` to `len(missing) > 0 or len(unexpected) > 0` for the modules
  `task_description.md` names, and log per-module `missing`/`unexpected` *lists* (or counts), not
  just a matched/total ratio. **Line count**: ~30 lines.

* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/jsonl_logger.py` (62 lines,
  `StepLogger` class) and `tasks/t0009_stage2_training_failure_forensics/code/health_gates.py` (183
  lines, `HealthGate`/`GateResult`). **What it does**: `StepLogger.log(record: dict) -> None`
  appends timestamped JSON records; `HealthGate.check(epoch, step, metrics) -> GateResult` evaluates
  threshold gates (`dur_loss`, `acoustic_norm`, `val_spike`) and returns a frozen dataclass with
  `fired`, `gate_name`, `value`, `threshold`, `last_healthy_ckpt`. **Reuse method**: import via
  library (`t0009_training_safeguards`) if this task ever needs to re-derive or sanity-check
  `t0010`'s existing gate behavior against `data/run_v10/metrics.jsonl` — no adaptation needed for
  read-only replay use, since `test_replay.py`
  (`tasks/t0009_stage2_training_failure_forensics/code/test_replay.py`) already demonstrates this
  usage pattern against a committed log. Not needed for the new inference harness itself (training-
  loop-coupled), but directly usable if step 4/5 of this task wants to programmatically re-run gate
  checks over `run_v10`'s JSONL rather than reading them by hand.

* **Source**: `tasks/t0010_stage2_safeguarded_training/code/paths.py` (26 lines). **What it does**:
  centralizes path constants (`RUN_V10_METRICS_JSONL`, `RUN_V10_CHECKPOINT_MANIFEST`,
  `RUN_V10_CHECKPOINTS_DIR`, `MODEL_ASSET_FILES_DIR`, etc.) relative to `TASK_ROOT`. **Reuse
  method**: copy the pattern (one `paths.py` per task, `Path(__file__).parent.parent`-anchored) into
  this task's `code/` for referencing `tasks/t0010_stage2_safeguarded_training/data/run_v10/*` and
  `tasks/t0013_.../results/audio_samples/` locations — do not import `t0010`'s `paths.py` directly
  (non-library code, cross-task import forbidden). **Line count**: ~10 lines worth of constants to
  replicate for this task's own file layout.

* **Not reusable, reference only**:
  `tasks/t0010_stage2_safeguarded_training/code/eval_all_checkpoints.py` (596 lines) and
  `tasks/t0008_tts_eval_harness_baselines/code/adapters.py` (`load_kokoro_model_with_checkpoint`,
  lines 291-330) — both are hard-coupled to `kokoro.KModel` (ISTFTNet-only) and are explicitly the
  wrong inference path for a `hifigan`-decoder checkpoint per this task's own premise. Useful only
  as documentation of the failure mode to avoid repeating (see Key Findings) and as the source of
  the `CHECKPOINT_MODULES` five-key constant if this task ever needs to re-verify the Kokoro-format
  packaged asset (`kokoro-v10-best.pth`) separately from the raw checkpoint diagnosis.

## Lessons Learned

* **Listening/inspecting audio from an intermediate checkpoint through the *actual* production code
  path, A/B'd against a known-good control, is the only method in this project's history that has
  reliably separated "packaging bug" from "training defect."** [t0002] used it to isolate v4's
  predictor defect from its own extraction bug; this task's Scope §2 (control test) applies the same
  method to the harness/checkpoint question at hand — follow it rigorously rather than skipping the
  control step.
* **`strict=False` + aggregate-only missing/unexpected counts is a project-wide recurring failure
  mode**, not a one-off ad hoc-script mistake: it caused [t0009]'s root-caused 19+ training failures
  (checkpoint loader), and the same shape reappears in [t0008]'s eval-harness loader and (per this
  research) plausibly explains v10's silent decoder-architecture mismatch too. Any new loader this
  task writes must log missing/unexpected **per module**, not aggregate, and must treat
  nonzero-but-not-total mismatches on core modules as hard failures — a bar stricter than any
  existing loader in the codebase currently meets.
* **`val_loss` and the existing health gates (`acoustic_norm`, `val_spike`, `dur_loss`) did not fire
  during v10's run and stayed smooth throughout** — this is itself evidence against a late-training
  GAN-divergence explanation (contrast [t0005]'s and [t0006]'s clearly visible divergence
  signatures) and toward a structural/architecture-level defect that a reconstruction- style
  aggregate loss would not be sensitive to, exactly as `task_description.md` Key Question 2
  hypothesizes.
* **Config drift between task iterations has gone undetected before**: [t0006]'s results note
  `epochs_2nd` was "accidentally left at 10" instead of the intended 15, and [t0009]'s forensics
  found three separately-lost code patches (`slmadv`/`lambda_slm` guard, `train_LM` guard, gradient
  clipping) that had to be rediscovered from `t0001`'s patched script after VM disk wipes. The
  apparent `decoder.type` drift in `config_david_v10.yml` (found in this research pass) fits the
  same pattern — an unreviewed config diff introducing an unintended architecture change alongside
  the intended `joint_epoch`/`epochs_2nd` changes.
* **JSONL step logging infrastructure exists but is easy to under-populate**: `t0010`'s
  `metrics.jsonl` has columns for `grad_norm_decoder`, `disc_loss`, etc. that were never filled in,
  because `train_second_v10.py` only logs once per epoch at validation time. A future safeguards
  library revision should assert that declared-but-always-null fields are flagged, not silently
  accepted.

## Recommendations for This Task

1. **Build the checkpoint-tensor diagnostic (Key Question 4) before writing any inference code.**
   Per Lesson 2 in `LESSONS.md` ("smoke-gate before measurement saves ~10% of VM spend" — the
   general principle, not the VM-specific mechanics, applies here too), a `torch.load` +
   `torch.isfinite` + per-module weight-norm + submodule-name inspection of `epoch_2nd_00016.pth`'s
   raw `net["decoder"]` dict is minutes of CPU work and can likely confirm or refute the
   architecture-mismatch hypothesis from this research (Key Findings §3) before any StyleTTS2 venv
   is even built. Specifically: check whether `net["decoder"]` keys look like `ups.*`/`resblocks.*`
   (HiFi-GAN) — this is a direct, cheap test of the central finding of this research.
2. **When writing the instrumented loader (Scope §1), start from
   `train_second_v10.py:load_checkpoint()`'s shape-matching structure but raise the failure bar**:
   log missing/unexpected **per module**, and fail loudly on any nonzero count for the modules
   `task_description.md` names as should-be-fully-covered (`bert`, `bert_encoder`, `predictor`,
   `decoder`, `text_encoder`, `predictor_encoder`, `style_encoder`, `diffusion`) — do not reuse the
   "raise only if literally zero matched" threshold from either existing loader.
3. **Do not reuse `kokoro.KModel`/`KPipeline` or [t0008]'s
   `adapters.load_kokoro_model_with_checkpoint()` for this task at all** — per Key Findings §4, that
   path is independently broken for a `hifigan`-decoder checkpoint regardless of the disk-space
   issue that blocked it in [t0010]. `task_description.md`'s instruction to use StyleTTS2's own
   `Demo/Inference_LibriTTS.ipynb` code path is correct and should not be second-guessed.
4. **When running the control test (Scope §2)**, follow [t0002]'s precedent: score the *harness*,
   not just the checkpoint, by first confirming the identical code path produces clean audio on a
   known-good reference before trusting any verdict about `v10`.
5. **For Key Question 5, do not expect much from `data/run_v10/metrics.jsonl` beyond the epoch-level
   `val_loss`/`dur_loss`/`acoustic_norm`/`f0_loss` fields already inspected in this research** — the
   discriminator-loss and gradient-norm columns were never populated during the run. State this
   explicitly in `results/v10_diagnosis.md` rather than re-discovering it from scratch.
6. **Copy, do not import**, any code taken from `extract_decoder.py`, `train_second_v10.py`, or the
   `t0009_training_safeguards` library modules into this task's `code/`, per the cross-task code
   reuse rule — only the two registered libraries (`tts_eval_harness`, `t0009_training_safeguards`)
   are import-eligible, and neither is directly usable for the new StyleTTS2-native harness without
   substantial rewriting (see Reusable Code and Assets).
7. **`dvc pull` both `epoch_2nd_00016.pth` and `epoch_2nd_00014.pth`** before starting — both `.dvc`
   pointers are present in `tasks/t0010_stage2_safeguarded_training/data/run_v10/` (identical
   2,086,820,872-byte size, consistent with same architecture/config for both), but neither raw file
   is pulled to local disk yet in this worktree.

## Task Index

### [t0001]

* **Task ID**: `t0001_kokoro_v4_stage2_finetune`
* **Name**: Kokoro v4 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: First Stage 2 training attempt; its `config_david_v4.yml` establishes the
  `decoder.type: istftnet` baseline that every subsequent config (except v10) follows, making the
  v10 anomaly identifiable.

### [t0002]

* **Task ID**: `t0002_kokoro_v4_voicepack_decoder_package`
* **Name**: Package Kokoro v4 voicepack + decoder
* **Status**: completed
* **Relevance**: Originated the five-module packaging requirement and the extraction bug/genuine-
  defect disambiguation methodology (A/B against a known-good checkpoint through the identical code
  path) that this task's control-test step directly replicates.

### [t0005]

* **Task ID**: `t0005_kokoro_v5_stage2_train`
* **Name**: Kokoro v5 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: Establishes the project's clearest example of genuine GAN-divergence signal shape
  (val_loss 0.848→2.070 across post-GAN epochs) — the contrasting pattern against which v10's
  smooth, non-diverging loss curve is evaluated in this research.

### [t0006]

* **Task ID**: `t0006_kokoro_v5_stage2_subset`
* **Name**: Kokoro v5 Stage 2: 250-clip subset, multispeaker: false
* **Status**: completed
* **Relevance**: `v6c`/`v6d` configs are the last successful loads of `first_stage_v3.pth` before
  v10, both using `decoder.type: istftnet` — direct comparison point for the architecture-mismatch
  finding.

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Produces the `tts_eval_harness` library, including `extract_decoder.py` (reusable
  packaging pattern to adapt) and `adapters.load_kokoro_model_with_checkpoint()` (the ISTFTNet-only
  load path this task's motivation shows is broken for v10, and which this research confirms is
  independently blocked regardless of the VM disk issue).

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Root-caused the project's recurring silent-checkpoint-load failure class and
  shipped the safeguard library (`StepLogger`, `HealthGate`, `CheckpointManager`) this task must
  cross-reference (`data/run_v10/metrics.jsonl`) and whose loader pattern this task must extend, not
  copy verbatim.

### [t0010]

* **Task ID**: `t0010_stage2_safeguarded_training`
* **Name**: Kokoro Stage 2: safeguarded training with joint_epoch=8
* **Status**: completed
* **Relevance**: Direct dependency; produced both checkpoints under investigation
  (`epoch_2nd_00016.pth`, `epoch_2nd_00014.pth`), the packaged `kokoro-v10-best` model asset, and
  `config_david_v10.yml` — whose `decoder.type: hifigan` (vs. every other project config's
  `istftnet`) is this research's central new finding.
