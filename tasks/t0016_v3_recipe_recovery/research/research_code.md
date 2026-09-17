---
spec_version: "1"
task_id: "t0016_v3_recipe_recovery"
research_stage: "code"
tasks_reviewed: 15
tasks_cited: 13
libraries_found: 2
libraries_relevant: 2
date_completed: "2026-09-17"
status: "complete"
---
## Task Objective

t0016 must reconstruct the exact Kokoro-82M v3 Stage 2 StyleTTS2 training recipe — config
(`joint_epoch`, `diff_epoch`, `lambda_gen`, `lambda_slm`, `lr`, `batch_size`, `epochs_2nd`,
`multispeaker`, `train_LM`, `max_len`, `first_stage_path`), the 266-clip training list, the applied
patches, epoch count/selection criterion, and the training environment — from VM files, DVC
artifacts, and checkpoint forensics, because v3's original author is unavailable and v3 is the only
Kokoro fine-tune in this project's 15-task history with clean, production-usable audio
(`speaker_sim=0.631` fillers / `0.588` val96, TTFB p50 185 ms per [t0008]). Every field must be
labelled confirmed (file/hash-backed), inferred (derived from checkpoints/samples/logs), or unknown
— no field may be silently upgraded past what its evidence supports. The task closes suggestion
S-0009-04 from [t0009], reconciles a live contradiction about `multispeaker` between
`best/config.json` (`true`) and [t0006]'s task description (claims `false`), and must produce a
reconstructed config plus a full human-listenable audio set (shipped v3, per-epoch samples,
ElevenLabs reference) so the answer asset is a working baseline for a future reproduction task
(t0017, out of scope here) rather than an unverifiable claim.

## Library Landscape

Two libraries exist in the project, both discovered via
`uv run python -u -m arf.scripts.aggregators.aggregate_libraries --format json --detail full --include-full-description --ids tts_eval_harness t0009_training_safeguards`.
Neither aggregator entry shows a correction or replacement overlay (`details.json`/`description.md`
match the raw task folder content).

* **`tts_eval_harness`** (v0.1.0, created by [t0008]). Modules: `code/harness.py`, `adapters.py`,
  `scoring.py`, `report.py`, `extract_decoder.py`, `constants.py`, `paths.py`, `run_eval.py`,
  `prepare_prompts.py`, `score_speaker_sim.py`, all under
  `tasks/t0008_tts_eval_harness_baselines/code/`. **Relevant**: this task's Scope §4 mandates
  synthesizing the shipped v3 bundle on gate texts, v3 sample phrases, and 5 seeded val96 prompts —
  exactly what `adapters.kokoro_v3_bundle(text, *, pipeline)` and
  `load_kokoro_model_with_checkpoint()` already do (t0008 loaded `david_v3_best_decoder_kokoro.pth`
  \+ `david_v3_best_voicepack.pt` through `kokoro.KModel`, the identical bundle this task must
  re-synthesize). Import path:
  `from tasks.t0008_tts_eval_harness_baselines.code.adapters import (kokoro_v3_bundle, load_kokoro_model_with_checkpoint, save_wav, SynthResult)`.
  `score_speaker_sim.py` is also reusable if the answer asset wants a numeric speaker_sim
  confirmation of the shipped bundle reproduced locally, cross-checked against t0008's recorded
  0.631/0.588.
* **`t0009_training_safeguards`** (v0.1.0, created by [t0009]). Modules: `jsonl_logger.py`,
  `checkpoint_manager.py`, `health_gates.py`, `run_config.py`, `constants.py`, `paths.py`, under
  `tasks/t0009_stage2_training_failure_forensics/code/`. **Relevant but not to import**: t0016 runs
  no training, so `StepLogger`/`HealthGate`/`CheckpointManager.save()` have no live use here. It is
  relevant as a **reference pattern** — `CheckpointManager`'s per-checkpoint `_sha256()` helper
  (`code/checkpoint_manager.py:100-119`, SHA-256 over the saved `.pth` via `hashlib.sha256`) is the
  exact hashing convention this task must apply to every VM-recovered file and to
  `stage1/first_stage.pth` vs. `first_stage_v3.pth` to settle the byte-identity question Key
  Question 6 raises. The gate thresholds (`ACOUSTIC_NORM_MAX=20.0`, `DUR_LOSS_STEP1_MAX=2.0`,
  `VAL_SPIKE_MAX=0.05`) are also useful context for the reconstructed config's inline
  confirmed/inferred/unknown comments, since t0009's confound table calibrated them off v6c, not v3.

## Key Findings

### The `multispeaker` contradiction is real and traceable to a version mismatch, not a typo

[t0006]'s `data/reference/v3/best/config.json` (the shipped Kokoro-API bundle config) states
`"multispeaker": true`. [t0009]'s `results/confound_table.md` lists the v3 row as
`multispeaker: true (assumed)` and separately records that t0006_run03 (v6c) used
`first_stage_v3.pth` with `multispeaker: true` in the *checkpoint*. However,
`tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml` — the actual Stage 2 launch
config file for that same v6c run — sets `model_params.multispeaker: false` (line 76) while loading
`first_stage_path: first_stage_v3.pth` and annotating it inline as "v3 Stage 1 checkpoint
(multispeaker:false)". This is a three-way disagreement across three sources that all claim to
describe the same checkpoint: the packaged bundle config says `true`, the confound table says
`true`, and the actual Stage 2 YAML that loaded that checkpoint says `false`. None of these three
files carries a SHA-256 to prove they describe byte-identical checkpoints — t0009 explicitly left
all hashes null (`results/confound_table.md` header: "SHA-256 hashes not available (no local
checkpoint files; VM not accessible)"). This is the central unresolved confound t0016 must settle
with hashes, not by picking the source that "sounds more authoritative." The forensic approach:
checkpoint-level (does `stage1/first_stage.pth`'s state dict contain style-conditioning layers whose
shape implies multispeaker support?) settles it independently of any config file, using the exact
raw-`torch.load` pattern in [t0015]'s `predictor_tensor_forensics.py`.

### Checkpoint forensics: the raw-`torch.load` module-diff pattern is the load-bearing prior art

Three tasks built increasingly refined versions of the same idea — read a checkpoint's `net` dict
directly with `torch.load(path, map_location="cpu", weights_only=False)`, never construct a live
model — and each fixed a blind spot in the version before it:

* [t0013]'s `inspect_checkpoint.py` first established the pattern for the decoder module, root-
  causing v10's clipped/saturated noise (`clip_fraction` 0.750-0.807) to an `ignore_modules` bug
  that let a HiFi-GAN-shaped decoder silently absorb an ISTFTNet-shaped Stage-1 checkpoint.
* [t0015]'s `code/predictor_tensor_forensics.py` (266 lines) retargeted the same raw-`torch.load`
  pattern at `predictor`/`predictor_encoder`, adding a `TensorForensics` dataclass
  (`key, num_params, finite, weight_norm, mean_abs, max_abs, nonfinite_keys`) computed per module
  and per named submodule prefix (`duration_proj`), plus a Markdown table renderer
  (`render_markdown()`) and a relative weight-norm-delta verdict with a documented 5% "near-zero
  shift" threshold. This is the exact tool t0016's `results/v3_module_weight_delta.png` chart and
  Key Question 3 (did the decoder change at all, or did the 0.03 speaker_sim gain over "base Kokoro
  \+ v3 voicepack" come from predictor/text_encoder alone, per [t0008]'s 0.631 vs. 0.603 numbers)
  require — comparing `stage1/first_stage.pth` against `best/david_v3_best_decoder_kokoro.pth`
  module by module (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`) is a direct
  generalization of `predictor_tensor_forensics.py`'s two-checkpoint, N-module comparison to five
  modules instead of two.
* All raw-checkpoint code in this family strips DataParallel `module.` prefixes with
  `k.removeprefix("module.")` before comparing keys (`predictor_tensor_forensics.py:100`) — the same
  prefix-stripping Key Question 6 asks t0016 to check for on `first_stage.pth`.

### Audio quality gating evolved through three tasks and directly answers the "is this epoch's audio clean?" question

`code/audio_quality_check.py`'s `check_audio_quality()` function was written once in [t0013], copied
unchanged (logic-for-logic) into [t0014], then extended in place by [t0015] (233 lines) with two new
signals — `duration_sanity_pass` (needs `text`) and `longest_nonsilent_run_s` — while deliberately
keeping `is_likely_noise` computed from exactly the original three signals (`silence_fraction`,
`spectral_flatness >= 0.35`, `clip_fraction >= 0.3`) for backward compatibility. This is precisely
the "hardened gate" t0016's task_description.md names for scoring the `epoch{0..3}_phrase{1..5}.wav`
per-epoch samples and the shipped v3 bundle's fresh synthesis: `AudioQualityResult` returns
`rms, peak, silence_fraction, spectral_flatness, clip_fraction, longest_nonsilent_run_s, duration_sanity_pass, is_likely_noise`,
and callers combine the two new fields with `is_likely_noise` into their own pass/fail (see
`run_gate_regression.py`'s `hardened_gate_pass` pattern quoted in the module's own docstring).
Feeding `text=` the five v3 sample phrases makes `duration_sanity_pass` meaningful for the per-epoch
table t0016's `results/v3_checkpoint_forensics.md` requires (`is_likely_noise`,
`duration_sanity_pass`, `longest_nonsilent_run_s` columns are named verbatim in
task_description.md's Outputs section).

### Packaging pipeline provenance: v3's five-module bundle format was reverse-engineered once, in [t0002], and is the spec t0016 must follow exactly

[t0002] confirms the v3 packaging recipe by direct inspection of `david_v3_best_decoder.pth`: it is
**not** decoder-only despite its name — it bundles five modules (`bert`, `bert_encoder`,
`predictor`, `text_encoder`, `decoder`). [t0002]'s `code/extract_decoder_generic.py` (38 lines) is
the exact extraction logic: it loads a raw checkpoint, remaps
`*.parametrizations.weight.original0/1` keys to `*.weight_g`/`*.weight_v` (`convert_key()`), strips
a `module.` DataParallel prefix, and saves
`{mod: {converted_key: tensor} for mod in ["bert","bert_encoder","predictor","text_encoder", "decoder"]}`.
[t0002] also proved the resulting bundle loads cleanly into `kokoro.KModel` with a clean spectrogram
when built from v3's proven-good artifacts, confirming the packaging code itself is not the source
of any noise seen elsewhere — a fact this task can rely on rather than re-verify. This is the
packaging contract `best/config.json`'s five-module `.pth` already satisfies; t0016 does not need to
repackage v3, only to load and diff it, but any future reconstruction (t0017) must follow this exact
five-module, key-remapped shape.

### The Stage 2 confound landscape: `joint_epoch` and Stage-1-checkpoint alignment dominate every known success or failure, but v3 predates the whole confound table

[t0009]'s `results/confound_table.md` is the master cross-run ledger (7 rows: t0001_run01,
t0005_run06, t0006_run01/02/03/04, v3). Its two firmest conclusions, both evidenced across multiple
runs: (1) a Stage-1-checkpoint/`multispeaker`-flag mismatch causes a **silent zero-parameter load**
(t0006_run01/02, "config mismatch with checkpoint" → "likely 0-param load (trained from scratch)" —
i.e. the run silently trains from random init while reporting normal-looking losses); (2)
`joint_epoch` escalation from 3 (t0001, t0005, failing) to 6 (t0006_run03/04 v6c/v6d, succeeding,
val 0.849/0.846) is "the main stabilizing factor besides checkpoint fix." Critically, the v3 row
itself is the *worst-evidenced* row in the table: `multispeaker` "true (assumed)",
`first_stage_path` "first_stage_v3.pth (assumed)", `joint_epoch` "unknown (speculated: 0)",
`lambda_gen` "1.0 (assumed)", `lr` "unknown" — every field either assumed or unknown, with the
table's own header stating "SHA-256 hashes not available (no local checkpoint files; VM not
accessible)." t0009 also notes only `lambda_slm > 0` guard is confirmed as part of v3's patch set
(`train_second_patch.diff`, [t0006]) — "no istftnet clamp, no skip guard" — meaning v3's 10-epoch,
266-clip, `val=0.506` run needed none of the seven crash-mitigation patches [t0005]/[t0006] later
stacked. This strongly suggests v3's success is structural (data scale, `joint_epoch` timing,
single-speaker-scale training or a genuinely different `first_stage.pth`) rather than a
patch-dependent stability fix, reinforcing why the checkpoint forensics track (not log/patch
archaeology) is the most decisive evidence source available locally.

### Data list provenance: v3's 266 clips are provably distinct from every other Kokoro-Stage-2 corpus in the project, and val_96 overlap must be re-checked from scratch

No project task, including [t0011] and [t0012] (which audited and LUFS-normalized the *v5* corpus —
1557 raw clips, 1531 after cleaning, unrelated to v3), ever recovered or referenced v3's own
266-clip list. [t0006]'s own `code/train_list_250.txt` is explicitly a **seed-42 random sample of
the 1557-clip v5 set** (used to match v3's *data scale*, not its actual content — confirmed by
[t0006]'s own hypothesis text: "250 clips (seed=42 random sample from v5's 1557 ... to match v3's
data scale)"). [t0009]'s data audit ([t0009] `results/data_audit_summary.md`, confirmed in its
results_summary "0 train/val overlap; v5 val == val_96") only covers the v5/v11 lineage — it never
asserts anything about v3's 266-clip list versus val_96, because v3's list was never in scope for
that audit. Task Question 7 ("does the 266-clip list overlap val_96?") is therefore **fully open**
going into t0016 — there is no code or data anywhere in the repo that already answers it, and it
must be answered by whatever list recovery (VM) or provenance reconstruction (voicepack embedding
inspection) t0016 manages.

### VM persistence and idle-billing lessons directly bound this task's Scope §1

`LESSONS.md` Lesson 10 (Azure ML `/mnt` is ephemeral; only a `/mnt/cache/persist` symlink to Azure
Files survives a stop/start) is the reason task_description.md prioritizes `~/kokoro-finetune/` (OS
disk, survives) over `/mnt/kikiri-tts/` (ephemeral, likely gone) as the VM evidence source, and why
[t0010]'s $272.78 overrun (VM left running overnight after `/mnt` filled up, no watchdog stop) and
[t0014]'s explicit proactive disk-monitoring fix are cited precedents for treating VM time as a hard
budget line, not a soft one. Lesson 8 (fire-and-forget handoffs cause idle billing; the orchestrator
must own step liveness) is the reason this task's Scope caps VM time at 90 minutes and requires an
explicit stop, mirroring the `setup-remote-machine` skill's teardown discipline used throughout
[t0009] (~30 min VM retrieval budget) and [t0008] (~1.5 h harness run budget).

## Reusable Code and Assets

* **Source**: `tasks/t0015_v11_duration_blowup_forensics/code/predictor_tensor_forensics.py` (266
  lines). **What it does**: raw-`torch.load` module-by-module weight-norm forensics between two
  checkpoints, with a `TensorForensics` dataclass
  (`tensor_forensics(label, module_sd) -> TensorForensics` at line 55) and a Markdown table renderer
  (`render_markdown(v11_results, control_results, v11_duration_proj, control_duration_proj) -> str`
  at line 110). **Reuse method**: copy into task (not a registered library). **Adaptation needed**:
  generalize `TARGET_MODULES` from `("predictor", "predictor_encoder")` to all five v3 bundle
  modules (`bert, bert_encoder, predictor, text_encoder, decoder`); replace
  `load_module_state_dicts()`'s single hard-coded `net["module"]` key access with a loader that
  tolerates `stage1/first_stage.pth` (which may not share the same top-level structure as a Stage-2
  `epoch_*.pth`); add SHA-256 hashing of each loaded file (pattern below) since this task must
  record hashes, which t0015 did not need to.
* **Source**: `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` (233 lines).
  **What it does**:
  `check_audio_quality(wav_path: Path, *, text: str | None = None) -> AudioQualityResult` computes
  `rms, peak, silence_fraction, spectral_flatness, clip_fraction, longest_nonsilent_run_s, duration_sanity_pass, is_likely_noise`.
  **Reuse method**: copy into task (explicitly named in task_description.md as the script to reuse
  for per-epoch sample scoring). **Adaptation needed**: none functionally — call with `text=` set to
  each of the five v3 sample phrases and the three fixed gate texts to populate
  `duration_sanity_pass`; the CLI entry point (`main()`) already supports ad hoc single-file
  scoring.
* **Source**: `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py` (38
  lines). **What it does**: `convert_key(k: str) -> str` remaps parametrization keys and strips
  `module.` prefixes; `main()` extracts the five bundle modules
  (`MODULES = ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]`) from a raw
  checkpoint into a Kokoro-API loadable `.pth`. **Reuse method**: copy into task, read-only
  reference — t0016 does not need to re-run extraction since `best/david_v3_best_decoder_kokoro.pth`
  already exists (DVC), but this is the exact key-remapping contract to validate the shipped
  bundle's shapes against during checkpoint forensics (Key Question 3), and is the one-line-of-truth
  for what "five-module bundle" means when writing the answer asset's packaging pointer back to
  [t0002]. **Adaptation needed**: none if used read-only for validation; trivial path edits if
  re-extraction from a different checkpoint is needed.
* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py`, method
  `CheckpointManager._sha256()` (private helper, invoked at line ~80 inside `save()`, class spans
  119 lines total). **What it does**: computes a SHA-256 digest of a saved checkpoint file for the
  manifest. **Reuse method**: copy into task as a small standalone `sha256_file(path: Path) -> str`
  helper (do not depend on the full `CheckpointManager` class, which is training-loop-oriented and
  not needed here). **Adaptation needed**: extract just the hashing logic; apply it uniformly across
  `data/vm_inventory/inventory.json`'s `sha256` field and the checkpoint-forensics hash comparison
  of `stage1/first_stage.pth` vs. VM-recovered `first_stage_v3.pth`.
* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/build_confound_table.py` (305
  lines, `ConfoundRow` dataclass with 20 fields:
  `run_id, task, data_list, n_clips, multispeaker, first_stage_path, first_stage_sha256, joint_epoch, lambda_gen, lambda_slm, lr, ft_lr, bert_lr, batch_size, gpu_count, parallelism_mode, train_lm, patches_active, outcome, notes`)
  and `code/build_inventory.py` (233 lines, `RunEntry` dataclass:
  `run_id, task, log_present, config_present, stage1_ckpt_path, stage1_ckpt_sha256, data_list, outcome`).
  **What they do**: produce the exact confound-table and log-inventory schema this project already
  uses for cross-run config bookkeeping. **Reuse method**: copy into task (adapt field set — t0016
  has one run, v3, not a table of runs, so these become single-row/single-entry schemas, or the
  dataclasses are reused directly for `data/vm_inventory/inventory.json`'s
  `path, size, mtime, sha256` schema by trimming unused fields). **Adaptation needed**: substantial
  trimming since t0009 tracked 7 runs and t0016 tracks one; keep the dataclass-plus-Markdown-table
  pattern (`asdict()` + hand-built `| --- |` rows), which matches this project's established
  results-table style.
* **Source**: `tasks/t0009_stage2_training_failure_forensics/code/collect_configs.py` (145 lines).
  **What it does**: copies committed config YAMLs from other tasks' `code/` directories into a local
  `data/configs/` folder by path list (`_GIT_CONFIGS`), and reconstructs unrecoverable ones from
  README notes. **Reuse method**: copy into task as a reference pattern only — task_description
  already names the one file to diff against
  (`tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml`, 119 lines, read
  directly), so a full multi-run collector is unnecessary; the pattern of "copy read-only, note what
  could not be recovered" is what to follow for `data/v3_train_list_UNRECOVERED.md` if the 266-clip
  list cannot be found.
* **Source**: `tts_eval_harness` library, `code/adapters.py`'s
  `kokoro_v3_bundle(text, *, pipeline)`, `load_kokoro_model_with_checkpoint(...)`, `save_wav(...)`,
  and `SynthResult` dataclass
  (`audio_array_16khz, audio_array_native, native_sample_rate, ttfb_s, rtf, audio_duration_s`).
  **What it does**: synthesizes text through the shipped v3 bundle exactly as [t0008] benchmarked
  it. **Reuse method**: import via library —
  `from tasks.t0008_tts_eval_harness_baselines.code.adapters import kokoro_v3_bundle, load_kokoro_model_with_checkpoint, save_wav`.
  **Adaptation needed**: point the loader at this task's local copy of
  `best/david_v3_best_decoder_kokoro.pth` / `best/david_v3_best_voicepack.pt` (already the same DVC
  path [t0008] used, under `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/`) and call
  for the three gate texts, five v3 sample phrases, and five seed-42 val96 prompts required by Scope
  §4.
* **Source**: `tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml` (119 lines).
  **What it does**: the closest surviving Stage 2 config to v3 (ISTFTNet decoder,
  `multispeaker: false`, `first_stage_path: first_stage_v3.pth`, `joint_epoch: 3`,
  `lambda_gen: 0.2`, `lr: 1.0e-04`, `batch_size: 8`, `train_data: v5/train_list_250.txt`). **Reuse
  method**: copy into task as the literal template `data/config_david_v3_reconstructed.yml` must be
  diffed against (task_description.md names this file explicitly as "the template to diff the
  reconstruction against"). **Adaptation needed**: every field must be re-annotated per t0016's own
  evidence, not inherited from v6c wholesale — v6c is itself an inference chain away from v3 (it
  *loads* v3's Stage-1 checkpoint but is a 2026-09-13 run with its own `joint_epoch`/`lambda_gen`
  choices, not v3's own recipe).
* **Source**: `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff` (152
  lines, DVC-committed as plain text, not `.dvc`-pointed). **What it does**: v3's only two confirmed
  patches to upstream `train_second.py` — the `loss_params.lambda_slm > 0` guard around `slmadv()`,
  and a `monotonic_align` `sys.path.insert` fix in `utils.py`. **Reuse method**: read directly
  (already in this task's evidence set per task_description.md, not code to execute) — cite verbatim
  in the reconstructed config's inline patch-provenance comment and in
  `results/v3_checkpoint_forensics.md`.

## Lessons Learned

* **Config files edited in place, not committed per-run, destroyed the historical record for every
  training task before [t0009]'s safeguards existed.** [t0009]'s own results state only 1 of 13
  audited training logs survived in git (7.7%), and [t0005]'s launch command actively `rm -f`'d the
  remote log between launches. This is precisely the failure mode t0016 must not repeat when
  documenting v3 — every recovered artifact needs a hash and a copy in `data/vm_inventory/`
  immediately, because the VM window is one-shot (90 minutes, then stopped per Scope §1).
* **A checkpoint that "loads without error" is not proof it loaded correctly.** [t0009]'s
  `load_checkpoint` root cause (silent zero-param load on DataParallel key mismatch, `strict=False`)
  and [t0013]'s `ignore_modules` root cause (partial Frankenstein load between
  architecture-mismatched decoders) are two independently-discovered instances of the same
  meta-lesson: verify shapes and parameter counts explicitly, never trust a clean exit code. t0016's
  checkpoint forensics must check this for `stage1/first_stage.pth` specifically (does it carry a
  `module.` DP prefix? does loading it into either a `multispeaker: true` or `multispeaker: false`
  shaped model actually match parameter counts, or silently zero-match one of them?).
* **`val_loss` does not predict audible quality, and is not even comparable across runs with
  different loss configurations.** [t0009]'s results_summary explicitly flags that v3's
  `val_loss=0.506` uses "a different loss configuration; not directly comparable" to v6c's 0.849.
  [t0003] independently found Stage-1 `val_loss` inversely tracked outcome quality (0.770 preceded a
  working Stage 2, 0.545 did not). t0016 should not treat v3's quoted 0.569/0.549/0.506 trajectory
  as validating anything about the reconstructed config beyond epoch ordering — audible/gate
  evidence (per-epoch WAV scoring) is the load-bearing signal.
* **A gate that catches one failure mode will not catch the next one.** The three-stage evolution of
  `audio_quality_check.py` ([t0013] → [t0014] → [t0015], each closing a distinct blind spot:
  clipping/saturation, then near-silent, then locally-speech-shaped-but-structurally-endless audio)
  is a direct warning that even the current hardened gate may have an undiscovered blind spot; a
  human listening pass (mandated by task_description.md Scope §4) is not optional decoration, it is
  how two of the three prior blind spots were actually found.
* **Data-scale and `multispeaker` are both plausible, unconfirmed causal stories for v3's success,
  and [t0006]/[t0009] never ran the controlled experiment that would isolate them.** [t0006]'s
  v6c/v6d runs held `first_stage_v3.pth` fixed but varied `lambda_gen`/`joint_epoch`, never testing
  266 clips vs. 250 clips vs. 1557 clips with everything else held constant. t0016 should not
  present "266 clips" or "multispeaker: false" as *the* explanation for v3's quality without
  flagging that this project has never isolated either variable experimentally — they are associated
  with the one successful run, not proven causal.

## Recommendations for This Task

1. **Import `tts_eval_harness`'s `adapters.kokoro_v3_bundle` / `load_kokoro_model_with_checkpoint` /
   `save_wav`** for the mandatory `results/audio_samples/v3_shipped/` synthesis (Scope §4) rather
   than writing a fresh Kokoro-loading script — this is a registered library specifically to avoid
   re-deriving this exact call path, which [t0008] already validated against this exact bundle.
2. **Copy and generalize [t0015]'s `predictor_tensor_forensics.py`** to all five v3 bundle modules
   for `results/v3_checkpoint_forensics.md` and the `v3_module_weight_delta.png` chart — it is the
   most mature module-diff tool in the repo and already produces the weight-norm-delta table shape
   task_description.md asks for.
3. **Copy [t0015]'s `audio_quality_check.py` unmodified** and call it with `text=` set for every
   per-epoch sample and the shipped-bundle synthesis; do not re-derive a noise gate from scratch,
   and do not fold `longest_nonsilent_run_s`/`duration_sanity_pass` into `is_likely_noise` (breaks
   the project's established backward-compatibility contract for that field).
4. **Extract a standalone `sha256_file()` helper from `checkpoint_manager.py`'s `_sha256()`** (do
   not import the full `CheckpointManager`, which is training-oriented) and hash every VM artifact
   and every local reference file (`stage1/first_stage.pth`,
   `best/david_v3_best_decoder_kokoro.pth`) before writing any confirmed/inferred/unknown label —
   this is the single check that resolves the `multispeaker` and byte-identity questions with actual
   evidence instead of by preferring one already-contradictory source over another.
5. **Do not trust [t0006]'s task_description.md claim that v3 used `multispeaker: false`** at face
   value, and do not trust `best/config.json`'s `true` at face value either — both are secondary
   artifacts (a task description's prose, and a packaged bundle's config) one inference step removed
   from v3's actual Stage 2 launch YAML, which no longer exists locally. Resolve via checkpoint
   shape forensics (style-conditioning layer presence/shape in `stage1/first_stage.pth`) as the
   primary evidence source, with the VM's surviving `logs/kokoro-david-v3*/config.yml` (if it
   exists) as the only source strong enough to override a forensic finding.
6. **Treat the 266-clip list as very likely unrecoverable locally** — no task in this project's
   history ever captured it (v5's 1557/1531/250-clip lists are all unrelated samples), so budget the
   VM window's list-recovery step accordingly and have `data/v3_train_list_UNRECOVERED.md` ready as
   the documented fallback per task_description.md's own contingency plan, rather than treating a VM
   miss as a task failure.
7. **Cap and log VM time exactly as [t0009] (30 min) and [t0008] (1.5 h) did**, using the
   `setup-remote-machine` skill for start/stop — Lessons 8 and 10 and [t0010]'s $272.78 overrun are
   direct precedent for why the 90-minute/$21 cap in this task's Scope §1 must be enforced by an
   explicit stop call, not a "the watchdog will catch it" assumption.
8. **Do not present v3's data-scale or `multispeaker` setting as a proven cause of its success** in
   the answer asset's short answer — per Lessons Learned, this project has never run the controlled
   ablation that would isolate these variables from `joint_epoch`/`lambda_gen`/checkpoint-alignment
   confounds. State them as "associated, unconfirmed" and let `results/suggestions.json`'s
   two-arm-ablation suggestion (already required by task_description.md if `multispeaker` stays
   ambiguous) carry the follow-up.

## Task Index

### [t0001]

* **Task ID**: t0001_kokoro_v4_stage2_finetune
* **Name**: Kokoro v4 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: First Stage 2 attempt (1557 clips, `joint_epoch=3`); diverged at epoch 7 after a
  best checkpoint at epoch 6 (val 0.751). Establishes the pre-t0009-safeguards failure baseline and
  the `joint_epoch=3`-fails pattern the confound table later generalizes.

### [t0002]

* **Task ID**: t0002_kokoro_v4_voicepack_decoder_package
* **Name**: Package Kokoro v4 voicepack + decoder
* **Status**: completed
* **Relevance**: Reverse-engineered and confirmed v3's five-module Kokoro-API packaging format
  (`bert, bert_encoder, predictor, text_encoder, decoder`) by inspecting `david_v3_best_decoder.pth`
  directly; proved v3's packaging pipeline and inference path are correct and noise-free when run on
  v3's own artifacts. Named explicitly in task_description.md as the packaging-recipe confirmation
  this task's answer asset must point back to.

### [t0003]

* **Task ID**: t0003_kokoro_v5_phoneme_data
* **Name**: Regenerate v5 phoneme manifests
* **Status**: completed
* **Relevance**: Found that Stage-1 `val_loss` does not predict Stage-2 success (0.770 preceded a
  working run, 0.545 did not), directly informing this task's caution about trusting v3's quoted
  val_loss trajectory as a quality signal.

### [t0005]

* **Task ID**: t0005_kokoro_v5_stage2_train
* **Name**: Kokoro v5 Stage 2 fine-tune
* **Status**: completed
* **Relevance**: Six Stage 2 runs, all diverged post-GAN; accumulated the seven crash-mitigation
  patches later carried into [t0006]. Its confound-table row (`joint_epoch=3`, 1557 clips, diverged)
  is a key contrast point against v3's `joint_epoch` unknown/speculated-0 and 266 clips.

### [t0006]

* **Task ID**: t0006_kokoro_v5_stage2_subset
* **Name**: Kokoro v5 Stage 2: 250-clip subset, multispeaker: false
* **Status**: completed
* **Relevance**: Owns the DVC-tracked v3 reference artifacts (`best/`, `stage1/`,
  `train_second_patch.diff`, per-epoch audio) this whole task is built on; its
  `config_david_v6c_stage2.yml` is the closest surviving relative of v3's config and the explicit
  diff template; its task_description.md's `multispeaker: false` claim is the contradiction this
  task must resolve.

### [t0008]

* **Task ID**: t0008_tts_eval_harness_baselines
* **Name**: TTS evaluation harness and baselines
* **Status**: completed
* **Relevance**: Created the `tts_eval_harness` library and produced the only existing
  speaker_sim/TTFB numbers for the shipped v3 bundle (0.631/0.588, TTFB p50 185 ms) that
  task_description.md quotes as the benchmark v3's recipe must be judged against; its
  `adapters.kokoro_v3_bundle` is directly reusable for this task's mandatory audio synthesis.

### [t0009]

* **Task ID**: t0009_stage2_training_failure_forensics
* **Name**: Stage 2 training failure forensics and safeguards
* **Status**: completed
* **Relevance**: Owns the confound table with the v3 row this task must fill in, filed suggestion
  S-0009-04 ("Recover v3 training configuration") that this task closes, and established the
  checkpoint-hashing and confound-table-schema conventions this task reuses.

### [t0010]

* **Task ID**: t0010_stage2_safeguarded_training
* **Name**: Kokoro Stage 2: safeguarded training with joint_epoch=8
* **Status**: completed
* **Relevance**: $272.78 VM overrun from an unmonitored full ephemeral disk is the direct precedent
  motivating this task's hard 90-minute VM cap and explicit-stop discipline.

### [t0011]

* **Task ID**: t0011_v5_data_quality_audit
* **Name**: v5 training data audio quality audit and clean manifest
* **Status**: completed
* **Relevance**: Establishes that the only data-quality auditing this project has done covers the v5
  lineage (1557 clips), never v3's 266-clip list — relevant as negative evidence that no existing
  task's data audit already answers t0016's val_96-overlap question.

### [t0012]

* **Task ID**: t0012_v5_corpus_normalize_and_reaudit
* **Name**: v5 corpus LUFS normalization and clipped-fraction re-audit
* **Status**: completed
* **Relevance**: Confirms the v5/v11 corpus lineage (1531 clean clips) is entirely distinct from and
  postdates v3's 266-clip list, reinforcing that v3's list must come from the VM or be documented
  unrecovered rather than approximated from any v5-derived list.

### [t0013]

* **Task ID**: t0013_v10_synthesis_quality_forensics
* **Name**: v10 checkpoint synthesis quality forensics
* **Status**: completed
* **Relevance**: Originated both the raw-`torch.load` checkpoint-forensics pattern (via
  `inspect_checkpoint.py`) and the first version of `audio_quality_check.py`, the two tool lineages
  this task's checkpoint and audio forensics work directly extends.

### [t0014]

* **Task ID**: t0014_v11_decoder_fix_retrain
* **Name**: Kokoro Stage 2 v11: decoder-init fix, full normalized corpus retrain
* **Status**: completed
* **Relevance**: Carried `audio_quality_check.py` forward unchanged and demonstrated proactive
  disk-usage monitoring that avoided repeating [t0010]'s overrun — a pattern this task's bounded VM
  inspection should follow.

### [t0015]

* **Task ID**: t0015_v11_duration_blowup_forensics
* **Name**: v11 duration-blowup forensics and audible-speech gate hardening
* **Status**: completed
* **Relevance**: Owns `predictor_tensor_forensics.py` and the hardened `audio_quality_check.py`,
  both explicitly named in task_description.md as the scripts to reuse for this task's checkpoint
  forensics and per-epoch audio scoring.
