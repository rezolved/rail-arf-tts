---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
research_stage: "code"
tasks_reviewed: 13
tasks_cited: 6
libraries_found: 2
libraries_relevant: 2
date_completed: "2026-09-16"
status: "complete"
---
# Research Code: t0014 v11 Decoder-Init Fix, Full Normalized Corpus Retrain

## Task Objective

t0014 fixes the `ignore_modules` bug that left `kokoro-v10-best`'s HiFi-GAN decoder worse than
random init, retrains on `t0012`'s 1,531-clip LUFS-normalized corpus, and gates the task's own
`completed` status on an objective audible-speech check (`is_likely_noise=False`) rather than
`val_loss` alone. Per step 6's assignment, this file reviews `t0009`'s safeguard library, `t0010`'s
`train_second_v10.py` (the `joint_epoch=8` fix and the DP-aware checkpoint loader), and `t0013`'s
`inspect_checkpoint.py` / `audio_quality_check.py` / `infer_styletts2.py` for direct reuse, and —
per the cross-step decision recorded after `research-internet` — explicitly checks whether any of
that prior code bakes in the now-superseded assumption that the fix is "add `\"decoder\"` to
`ignore_modules` and accept random init" rather than "repoint `first_stage_path` at a genuinely
`hifigan`-shaped pretrained checkpoint."

## Library Landscape

Two libraries exist in the project (`aggregate_libraries --format json --detail short`, 2 of 2
assessed relevant):

* **`tts_eval_harness`** (v0.1.0, created by `t0008_tts_eval_harness_baselines`) — GE2E speaker
  similarity, TTFB, RTF, WER, duration-ratio scoring across Kokoro/ElevenLabs backends. Import path:
  `from tasks.t0008_tts_eval_harness_baselines.code.<module> import <name>` (e.g.
  `code/scoring.py`'s `build_centroid`, `compute_speaker_sim`). **Relevant but with a caveat**: its
  synthesis adapters (`code/adapters.py`) all route through `kokoro.KModel`/`KPipeline`, which
  `[t0013]` proved cannot load a `hifigan`-decoder StyleTTS2 checkpoint at all (tensor shape
  mismatches on `decoder.generator.*`). Only the **scoring** half of this library (`build_centroid`,
  `compute_speaker_sim`, both pure-`resemblyzer` functions with no Kokoro dependency) is directly
  usable for a `v11` checkpoint; the synthesis half is not. No aggregator correction or replacement
  is present for this library — the `full_description` field is the original, unmodified text.
* **`t0009_training_safeguards`** (v0.1.0, created by `t0009_stage2_training_failure_forensics`) —
  `StepLogger`, `CheckpointManager`, `HealthGate`, `capture_run_config`. Import path:
  `from tasks.t0009_stage2_training_failure_forensics.code.<module> import <name>`. **Directly
  relevant and already integrated**: `[t0010]`'s `train_second_v10.py` imports all four components
  at the top of the file (`checkpoint_manager.CheckpointManager`, `health_gates.HealthGate`,
  `jsonl_logger.StepLogger`, `run_config.capture_run_config`), so v11 inherits this wiring for free
  by copying `train_second_v10.py` forward. No aggregator correction is present.

No other libraries exist in the project. `libraries_found: 2`, `libraries_relevant: 2`.

## Key Findings

### The decoder-init bug is a five-line, precisely located omission

`tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py:253-259` calls
`load_checkpoint(model, None, first_stage_path, load_only_params=True, ignore_modules=[ "predictor_encoder", "msd", "mpd", "wd", "diffusion"])`
— `"decoder"` is absent from that list, so `load_checkpoint()` (lines 87-118 of the same file)
partially loads `first_stage_v3.pth`'s decoder weights into `config_david_v10.yml`'s
`hifigan`-shaped decoder. The loader's own `matched` filter
(`ckpt_sd[k] for k in ckpt_sd if k in model_sd and model_sd[k].shape == v.shape`, line 103-105) only
keeps tensors with matching shapes, so this is not a crash — it silently loads whichever `decoder`
sub-tensors happen to have identical shapes between an ISTFTNet-shaped checkpoint (375 keys, 2 `ups`
stages, no `generator.alphas`) and a HiFi-GAN-shaped model (678 keys, 4 `ups` stages,
`generator.alphas.{0..4}`) and leaves the rest at `build_model()`'s random init `[t0010]`,
`[t0013]`. `[t0013]`'s `code/inspect_checkpoint.py` confirmed this architecture mismatch empirically
by `torch.load`-ing both checkpoints directly (no StyleTTS2 source imported) and classifying each
`decoder` state dict by the presence of `generator.alphas.*` keys (the only reliable discriminator
found by reading `hifigan.py`/`istftnet.py` source — `ups`/`resblocks` naming and even the
`stft`/`reflection_pad` attributes are shared or absent from `state_dict()` on both architectures,
so naive key-name diffing does not work). `[t0013]`'s `code/random_decoder_probe.py` went further
and showed the partial-shape-match load is actively **worse** than doing nothing at all: leaving
`decoder` at pure `build_model()` random init (every other module loaded normally) produces
`clip_fraction=0.004`, versus `0.750-0.807` for the real v10 checkpoints — the partial load is not a
harmless no-op, it corrupts the decoder into a worse-than-random state.

### Fix direction changed after this research began — v10/t0013 code assumes the wrong fix

`[t0013]`'s own `results/v10_diagnosis.md` framed the fix as "add `\"decoder\"` to `ignore_modules`"
(equivalent to accepting random decoder init and training it from scratch in `epochs_2nd: 20`). Step
5 (`research-internet`, this task) found upstream `train_second.py`'s own default `ignore_modules`
**also** omits `"decoder"` — the upstream design assumes `first_stage_path` always matches
`config.yml`'s `decoder.type` by construction, and never anticipated a cross-architecture mismatch.
The recommended fix is now to repoint `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s
`Models/LibriTTS/epochs_2nd_00020.pth` — a checkpoint whose `config.yml` decoder block
(`resblock_kernel_sizes: [3,7,11]`, `upsample_initial_channel: 512`, `upsample_rates: [10,5,3,2]`,
`multispeaker: true`) matches `config_david_v10.yml`'s decoder block field-for-field — and fine-tune
the decoder from real converged weights, not random init. **This is the exact same checkpoint file**
`[t0013]`'s own `code/infer_styletts2.py` harness already used as its validated "known-good control"
(`CONTROL_CKPT` in `code/paths.py`, resolved to
`code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`, sourced via the
`semidark/kikiri-tts` submodule clone) — `[t0013]`'s own control test (`results/control_test.md`)
already proved this checkpoint produces intelligible, non-noise speech through the project's own
instrumented harness. That clone is gitignored
(`tasks/t0013_v10_synthesis_quality_forensics/code/.gitignore` excludes `kikiri-tts/`), so it is not
present in this task's worktree and must be re-cloned, but the fact that it is already
harness-validated in this exact repo is strong corroborating evidence for the internet-research
recommendation, not just an external claim.

### `t0010`'s checkpoint-evaluation script bakes in two now-invalid assumptions

`[t0010]`'s `code/eval_all_checkpoints.py` evaluates each epoch checkpoint by (1) calling
`[t0008]`'s `extract_decoder.extract` to package a 5-module Kokoro-format state dict, then (2)
synthesizing "100 filler prompts via Kokoro pipeline" (module docstring, lines 1-8). `[t0013]`
proved step (2) cannot work for a `hifigan`-decoder checkpoint — `kokoro.KModel`/`KPipeline` only
implements ISTFTNet decoding. This script was never actually run to completion against a working
Kokoro-loadable checkpoint in `[t0010]` (harness eval was deferred to disk-full teardown), so the
assumption went undetected until `[t0013]`'s forensics. Separately, `eval_all_checkpoints.py`
defines `MIN_SPEAKER_SIM_GATE: float = 0.35` (line 50) as a "validation gate: first epoch
speaker_sim must exceed this (else extraction failed)" — but `[t0013]`'s Milestone E measured
`speaker_sim=0.35` and `0.31` for the two confirmed-garbage v10 checkpoints (clipped/saturated
noise, `clip_fraction` 0.75-0.81), i.e. right at or above this exact threshold. This gate would not
have caught the real defect even if the script had run. Both assumptions are now superseded: v11
evaluation must use `[t0013]`'s StyleTTS2-native `infer_styletts2.py` harness (or a fixed-decoder
successor of it) for synthesis, and must use `audio_quality_check.py`'s `clip_fraction`/spectral
flatness gate, not a bare `speaker_sim` threshold, as the pass/fail signal.

### `t0013`'s instrumented-loading pattern is the correct template for the pre-flight verification step

`[t0013]`'s `code/infer_styletts2.py:load_checkpoint_instrumented()` (lines 158-217) logs
`matched`/`missing`/`unexpected` counts **per module, per attempt** (raw and `module.`-stripped
fallback) and raises `RuntimeError` if any `CORE_MODULES` entry (`bert`, `bert_encoder`,
`predictor`, `decoder`, `text_encoder`, `predictor_encoder`, `style_encoder`, `diffusion`) has a
nonzero final missing/unexpected count — a strictly stronger check than `[t0010]`'s
`train_second_v10.py:load_checkpoint()`, which only raises on a **zero**-match module and silently
accepts any partial match above zero (exactly the condition that let the decoder bug through). Per
the task description's Scope item 1, the pre-training verification step should reuse `[t0013]`'s
`inspect_checkpoint.py`-style tensor forensics (architecture classification via the
`generator.alphas` marker, weight-norm/finite checks) rather than `train_second_v10.py`'s own
loader, since the loader's pass condition is exactly what was previously insufficient.

### `t0009`'s safeguards are corpus-size-agnostic by design but were calibrated on a 6x smaller run

`HealthGate`'s thresholds (`DUR_LOSS_STEP1_MAX=2.0`, `ACOUSTIC_NORM_MAX=20.0`, `VAL_SPIKE_MAX=0.05`,
`CONSECUTIVE_SKIP_MAX=50`, all in `tasks/t0009_stage2_training_failure_forensics/code/constants.py`)
were calibrated from a single successful run, `v6c`, on the same 250-clip subset `[t0010]` also
trained on — not on anything close to `[t0012]`'s 1,531-clip corpus. The library's own
`description.md` documents this limitation explicitly ("If future runs use a different dataset,
batch size, or learning rate, thresholds may need to be adjusted"). Nothing about switching corpora
requires new thresholds a priori, but `v11` is the first run to combine both changes (decoder-init
fix and 6x more data) at once, so gate firings should be interpreted with this calibration gap in
mind, per Key Question 4 in `task_description.md`.

### Corpus manifest lineage is a clean three-hop chain

`[t0011]` produced `per_clip_stats.jsonl` (1,557 clips audited, 1,311 clean by a since-superseded
peak-dBFS clipping heuristic). `[t0012]` corrected the clipping metric to `clipped_fraction` and
LUFS-normalized the full corpus to -14 LUFS, producing
`tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt` — verified by
direct line count to contain exactly **1,531** entries, matching `task_description.md`'s stated
corpus size. `val_data` is unchanged across every run in this lineage (`t0008`/`t0009`/`t0010`
confirmed `val_5 == val_96`, the project's permanent held-out set) and must stay that way per
`CLAUDE.md`'s "NEVER train on val_96" rule.

## Reusable Code and Assets

* **Source**: `tasks/t0010_stage2_safeguarded_training/code/train_second_v10.py` (43,000 bytes).
  **What it does**: full Stage 2 training script — DP-aware `load_checkpoint()` (lines 87-118),
  `main()` CLI (`-p config_path --run-id`), integrates the `t0009` safeguard library
  (`CheckpointManager`, `HealthGate`, `StepLogger`, `capture_run_config`), sets `joint_epoch=8` via
  config. **Reuse method**: **copy into task** (per the spec's cross-task rule — this is task-local
  code, not a registered library) as `code/train_second_v11.py`, per `task_description.md`'s
  Expected Outputs. **Adaptation needed**: fix the `ignore_modules=[...]` list at line 253-259 (add
  `"decoder"` only if falling back to random init) or, per the recommended direction, change
  `first_stage_path` in the config to point at the LibriTTS checkpoint and verify `ignore_modules`
  no longer needs a `"decoder"` exclusion at all since the architectures now match. **Line count to
  copy**: ~700 lines total file; the load-checkpoint and first-stage-loading block needing the fix
  is lines 87-269.
* **Source**: `t0009_training_safeguards` library (`checkpoint_manager.py`, `health_gates.py`,
  `jsonl_logger.py`, `run_config.py`, `constants.py`). **What it does**: see Library Landscape.
  **Reuse method**: **import via library** — already wired into `train_second_v10.py`'s imports
  (`from tasks.t0009_stage2_training_failure_forensics.code.checkpoint_manager import CheckpointManager`,
  etc.), so copying `train_second_v10.py` forward carries these imports with it unchanged.
  **Function signatures**: `CheckpointManager(log_dir: Path, run_id: str, joint_epoch: int)`,
  `.save(model, optimizer, epoch, step, val_loss=None) -> str`;
  `HealthGate(joint_epoch: int, last_healthy_ckpt=None, logger=None)`,
  `.check(epoch, step, metrics) -> GateResult`;
  `capture_run_config(config_path, log_dir, run_id, extra_info=None) -> None`. **Adaptation
  needed**: none for direct reuse; consider widening `constants.py` thresholds if `v11`'s 1,531-clip
  run produces systematically different loss magnitudes (see Key Findings).
* **Source**: `tasks/t0013_v10_synthesis_quality_forensics/code/inspect_checkpoint.py` (14,436
  bytes). **What it does**: `torch.load`s a raw checkpoint with no StyleTTS2 source imported,
  classifies `decoder` architecture (`hifigan`/`istftnet`/`ambiguous`) via the `generator.alphas`
  marker and `ups` stage count, checks every module for NaN/Inf and reports weight norms. **Reuse
  method**: **copy into task**, adapted to run against v11's checkpoint and (if the
  pretrained-checkpoint path is taken) the LibriTTS `epochs_2nd_00020.pth` as the new expected match
  target instead of `first_stage_v3.pth`. **Function signatures**:
  `inspect_checkpoint(path, label) -> CheckpointForensics`;
  `classify_decoder(decoder_sd: dict) -> DecoderArchitectureVerdict`. **Line count**: ~300 lines;
  `CORE_MODULES` tuple and `classify_decoder` (lines 33-112) are the directly reusable core.
* **Source**: `tasks/t0013_v10_synthesis_quality_forensics/code/audio_quality_check.py` (6,298
  bytes). **What it does**: `check_audio_quality(wav_path) -> AudioQualityResult` with `rms`,
  `peak`, `silence_fraction`, `spectral_flatness`, `clip_fraction`, `is_likely_noise` (fires on
  `spectral_flatness >= 0.35` OR `clip_fraction >= 0.3`, guarded by `silence_fraction < 0.95`).
  **Reuse method**: **copy into task** — this is the literal Milestone F deliverable
  `task_description.md` names as the completion gate ("`is_likely_noise=False` before this task may
  claim `status: completed`"). **Adaptation needed**: none functionally; run as-is against v11's
  synthesized samples. **Line count**: 149 lines total, ~35 lines of actual logic
  (`check_audio_quality`, lines 73-105).
* **Source**: `tasks/t0013_v10_synthesis_quality_forensics/code/infer_styletts2.py` (17,682 bytes).
  **What it does**: StyleTTS2-native inference harness (not Kokoro's `KModel`) —
  `load_checkpoint_instrumented(model, checkpoint_path) -> list[ModuleLoadResult]` (per-module
  missing/unexpected logging, hard-fails on core-module mismatch),
  `compute_style(model, path) -> Tensor`,
  `synthesize(model, model_params, text, ref_s, ...) -> (wav, wall_time)`. **Reuse method**: **copy
  into task**, per `task_description.md`'s Expected Outputs item 3 ("a working inference recipe ...
  committed under `code/`"). **Adaptation needed**: point `--checkpoint-path`/`--config-path` at the
  new v11 checkpoint; the `code/kikiri-tts` clone (StyleTTS2 submodule + CPU venv) must be
  re-created since `[t0013]`'s copy is gitignored, not committed. **Line count**: 424 lines;
  `load_checkpoint_instrumented` (56 lines), `_rename_legacy_parametrization_keys` (46 lines, needed
  only when loading externally-sourced checkpoints such as the LibriTTS one), and `synthesize` (90
  lines) are the reusable core.
* **Source**: `tasks/t0013_v10_synthesis_quality_forensics/code/random_decoder_probe.py` (7,299
  bytes). **What it does**: falsification-probe pattern — loads every module except `decoder` from a
  real checkpoint, leaves `decoder` at fresh `build_model()` random init, verifies via `torch.equal`
  that it was never touched, then synthesizes and runs `audio_quality_check`. **Reuse method**:
  **copy into task**, useful as a pre-training sanity check if the pretrained `first_stage_path`
  route is adopted: confirm the *old* config (decoder loaded from `first_stage_v3.pth`) still
  reproduces the worse-than-random signature, and/or confirm the *new* config (decoder loaded from
  the LibriTTS checkpoint) does not. **Line count**: 178 lines.
* **Source**: `tts_eval_harness` library, `code/scoring.py`. **What it does**:
  `build_centroid(wav_paths) -> np.ndarray`,
  `compute_speaker_sim(synth_wavs, ref_embeddings) -> SpeakerSimResult`. **Reuse method**: **import
  via library**
  (`from tasks.t0008_tts_eval_harness_baselines.code.scoring import build_centroid, compute_speaker_sim`)
  for the scoring half only — do **not** import `code/adapters.py`'s Kokoro-based synthesis
  functions, which cannot load a `hifigan` checkpoint. `[t0013]`'s `code/score_speaker_sim.py`
  (copied, not imported, 117 lines) already demonstrates the correct pattern: reuse only the
  centroid-building/scoring logic, pair it with `infer_styletts2.py`'s StyleTTS2-native synthesis
  instead of `adapters.py`.

## Lessons Learned

`[t0009]` established that the upstream loader's `strict=False` silent-failure mode is a recurring
project-wide anti-pattern — it caused the original 19+-run divergence crisis (DataParallel prefix
mismatch, zero params loaded) and, per this research, the same permissive-partial-match pattern
(different failure mode, same root cause: a loader that "succeeds" without verifying it loaded the
*right* weights) is what let the decoder-init bug through `[t0010]`'s training and evaluation
undetected. `[t0010]` shows that "training completed with zero health-gate events and a plausible
`val_loss` curve" is not sufficient evidence of a working checkpoint — `val_loss=0.797` best,
`0.853` primary, with **zero** gate firings across all 17 epochs, yet the checkpoint's audio was
never listened to before `[t0013]` and turned out to be 75-81% clipped noise. `[t0013]` shows the
fix for this class of failure: an objective, automatable, cheap audio-quality check
(`is_likely_noise`) run before a training task can claim `completed`, plus instrumented
missing/unexpected key logging on every checkpoint load, at every stage (not just training-time
loading, but evaluation-time loading too). `[t0012]` shows LUFS normalization plus a corrected
`clipped_fraction` metric recovers most of the corpus that a naive peak-dBFS check over-flagged
(1,311 to 1,531 clean clips), a pattern of "the audit heuristic itself was the bug" that recurred
across `[t0011]`→`[t0012]` similarly to how the `ignore_modules` list was the training bug in
`[t0010]`→`[t0013]`.

## Recommendations for This Task

1. **Copy `train_second_v10.py` forward as `train_second_v11.py`** and apply the fix at the
   `first_stage_path` load-checkpoint call site (lines 244-269), following the direction Step 5
   settled on: repoint `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth`
   rather than adding `"decoder"` to `ignore_modules`. Point `data_params.train_data` at `[t0012]`'s
   `train_list_v5_normalized_clean.txt` (1,531 clips); leave `val_data` unchanged.
2. **Import the `t0009_training_safeguards` library unchanged** (already wired into the file being
   copied) — no code changes needed, but recognize its gate thresholds were calibrated on a
   6x-smaller corpus and treat any gate firing as needing manual review rather than an
   automatic-correctness signal, per Key Findings.
3. **Do not reuse `[t0010]`'s `eval_all_checkpoints.py` as-is.** Its Kokoro-pipeline synthesis path
   cannot load a `hifigan`-decoder checkpoint at all, and its `MIN_SPEAKER_SIM_GATE=0.35` sanity
   check would not have caught (and did not catch) the exact failure this task exists to fix. Build
   v11's evaluation on `[t0013]`'s `infer_styletts2.py` (StyleTTS2-native synthesis, instrumented
   loading) instead, pairing it with `tts_eval_harness`'s `scoring.py` functions (imported, not the
   Kokoro adapters) for `speaker_sim` if a numeric comparison to the ElevenLabs baseline is still
   wanted alongside the pass/fail audible-speech gate.
4. **Run `[t0013]`'s `inspect_checkpoint.py`-style tensor forensics before any GPU training starts**
   (per `task_description.md` Scope item 1 and Key Question 3), adapted to check the *new*
   `first_stage_path` target: confirm `decoder` now shows full key/shape overlap with the LibriTTS
   checkpoint (not the partial match `first_stage_v3.pth` produced), using the same
   `generator.alphas`-marker classification method.
5. **Gate task completion on `audio_quality_check.py`'s `is_likely_noise=False`**, copied unchanged
   from `[t0013]`, exactly as `task_description.md` mandates — this is the process fix `[t0013]`
   filed as S-0013-02 and this task must not regress it.
6. **Gap**: no prior task's code handles fine-tuning *from* an externally-sourced pretrained
   checkpoint (all prior runs used `first_stage_v3.pth`, a project-local artifact). Downloading and
   verifying `yl4579/StyleTTS2-LibriTTS`'s checkpoint, and documenting its usage-terms disclosure
   per `research/research_internet.md`, is new work this task's planning step must scope explicitly
   — no existing script in this project does it.

## Task Index

### [t0008]

* **Task ID**: `t0008_tts_eval_harness_baselines`
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Source of the `tts_eval_harness` library; its `scoring.py` centroid/speaker_sim
  functions are reusable, but its Kokoro-based synthesis adapters are not (cannot load a
  `hifigan`-decoder checkpoint, per `[t0013]`).

### [t0009]

* **Task ID**: `t0009_stage2_training_failure_forensics`
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Source of the `t0009_training_safeguards` library (`StepLogger`,
  `CheckpointManager`, `HealthGate`, `capture_run_config`) already wired into `train_second_v10.py`
  and carried forward unchanged for v11, per this task's dependency and scope.

### [t0010]

* **Task ID**: `t0010_stage2_safeguarded_training`
* **Name**: Kokoro Stage 2: safeguarded training with joint_epoch=8
* **Status**: completed
* **Relevance**: Direct dependency; source of `train_second_v10.py` (copied forward as the base for
  `train_second_v11.py`) and the exact `ignore_modules` bug site (lines 253-259) this task fixes.
  Also the source of `eval_all_checkpoints.py`, whose Kokoro-synthesis assumption and
  `MIN_SPEAKER_SIM_GATE` are both now superseded.

### [t0011]

* **Task ID**: `t0011_v5_data_quality_audit`
* **Name**: v5 training data audio quality audit and clean manifest
* **Status**: completed
* **Relevance**: First hop of the corpus-manifest lineage this task's training data descends from;
  its over-strict peak-dBFS clipping heuristic is the reason `[t0012]` exists.

### [t0012]

* **Task ID**: `t0012_v5_corpus_normalize_and_reaudit`
* **Name**: v5 corpus LUFS normalization and clipped-fraction re-audit
* **Status**: completed
* **Relevance**: Direct dependency; source of the 1,531-clip `train_list_v5_normalized_clean.txt`
  manifest this task trains on, verified by direct line count.

### [t0013]

* **Task ID**: `t0013_v10_synthesis_quality_forensics`
* **Name**: v10 checkpoint synthesis quality forensics
* **Status**: completed
* **Relevance**: Direct dependency; source of the root-cause diagnosis, the tensor-forensics
  (`inspect_checkpoint.py`), the audible-speech gate (`audio_quality_check.py`), the instrumented
  StyleTTS2-native inference harness (`infer_styletts2.py`), and the falsification probe
  (`random_decoder_probe.py`) this task reuses directly, plus the evidence (`control_test.md`) that
  the LibriTTS `epochs_2nd_00020.pth` checkpoint already produces valid speech through this
  project's own harness.
