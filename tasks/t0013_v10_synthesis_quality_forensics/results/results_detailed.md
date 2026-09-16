---
spec_version: "2"
task_id: "t0013_v10_synthesis_quality_forensics"
---
# Results Detailed: v10 Checkpoint Synthesis Quality Forensics

## Summary

This task determined, with instrumented and control-validated evidence rather than opinion, why
`kokoro-v10-best` produces noise instead of speech. The verdict is **a real training defect from the
start**, not a bug in the ad hoc reproduction that first surfaced the problem. A cheap tensor-level
falsifier (Milestone B) confirmed an architecture mismatch between `first_stage_v3.pth`
(ISTFTNet-shaped) and `config_david_v10.yml`'s HiFi-GAN decoder. A full instrumented
StyleTTS2-native inference harness (`code/infer_styletts2.py`), validated against a known-good
external control checkpoint before ever touching v10, then showed both `epoch_2nd_00016.pth`
(primary) and `epoch_2nd_00014.pth` (backup) load cleanly (0 missing/0 unexpected keys on all 13
modules) but produce 75-81%-clipped, DC-saturated audio — ruling out a loader bug. Root cause,
traced to a specific line: `train_second_v10.py:load_checkpoint()`'s `ignore_modules` list (lines
244-260) omits `"decoder"`, so a HiFi-GAN vocoder was initialized from a partial, shape-mismatched
load of an ISTFTNet checkpoint. A follow-up falsification probe (step 11) shows this is actively
**worse** than pure random initialization (clip fraction 0.004 for a randomly-initialized decoder
vs. 0.750-0.807 for the real, partially-mismatched-loaded checkpoints). Recommendation: do not use
`kokoro-v10-best` for anything requiring audible output; retrain with `"decoder"` added to
`ignore_modules`.

## Methodology

* **Machine**: local Azure VM worktree host (`claude-vm-vlad`), Linux 6.8.0-1044-azure x86_64, Intel
  Xeon E5-2673 v4 @ 2.30GHz, 4 vCPUs. CPU-only throughout — no GPU was used or required, per
  `plan/plan.md`'s Objective ("No GPU, no remote machine, and no paid API calls are required — this
  is a CPU-only local forensics task").
* **Environment**: isolated venv at `code/.venv-styletts2/` pinning `torch==2.5.1 torchaudio==2.5.1`
  (CPU wheels), built specifically because `StyleTTS2/models.py`'s un-flagged `torch.load` calls
  break under `torch>=2.6`'s new `weights_only` default. `espeak-ng` installed via `apt-get` for the
  phonemizer backend (substituted for `task_description.md`'s macOS-authored `brew install`
  instruction; see `plan/plan.md`'s Ambiguity Note).
* **Code under test**: `github.com/semidark/kikiri-tts` cloned with submodules (`code/kikiri-tts/`,
  gitignored) providing `StyleTTS2` and `kokoro` source; the instrumented harness
  `code/infer_styletts2.py` is this task's own committed deliverable, adapted from
  `StyleTTS2/Demo/Inference_LibriTTS.ipynb`.
* **Runtime**: implementation step (step 9) ran from **2026-09-16T12:49:53Z** to
  **2026-09-16T13:30:00Z** (~40 minutes wall clock: venv build, DVC pulls, tensor forensics, harness
  build, control gate, v10 diagnosis, DVC push). Creative-thinking step (step 11, the random-decoder
  falsification probe) ran from **2026-09-16T13:35:58Z** to **2026-09-16T13:43:15Z** (~7 minutes).
  Individual inference runs on CPU took 13.7-22.3 seconds wall time each (see
  `results/audio_samples/*.timing.json`).
* **Task window**: task created at **2026-09-16T12:19:30Z**; this `results` step started at
  **2026-09-16T13:45:22Z**.

## Verification

* `verify_task_dependencies.py` (step 2) — PASSED, 0 errors, 0 warnings
  (`logs/steps/002_check-deps/deps_report.json`); confirmed `t0010_stage2_safeguarded_training`
  completed and produced the checkpoints under investigation.
* `verify_research_code.py` (step 6) — PASSED, 0 errors, 0 warnings.
* `verify_plan.py` (step 7) — PASSED, 0 errors, 0 warnings.
* Harness self-validation gate (Milestone D, step 10, `results/control_test.md`) — **PASS**:
  `code/audio_quality_check.py` on the known-good external StyleTTS2-LibriTTS control checkpoint
  gives `is_likely_noise=False` (spectral flatness 0.0615, clip fraction 0.0, silence fraction
  0.309). This is the load-bearing gate: without it passing first, the v10 diagnosis below would not
  be trustworthy evidence.
* `verify_task_results.py` (this step) — PASSED, 0 errors, 0 warnings.
* `verify_task_metrics.py` (this step) — PASSED, 0 errors, 0 warnings. Command logs for both runs
  are under `logs/commands/` for this step.

## Limitations

* **Single fixed text prompt and reference-audio construction across every run** (control, v10 x2,
  and the falsification probe all use the same sentence and the same concatenated-reference-clip
  recipe). The clipping mechanism identified (a structurally mismatched decoder architecture) is
  input-independent by construction, so this is assessed as low risk, but "does v10 clip on every
  input" is technically untested beyond this one input — flagged explicitly in
  `results/creative_thinking.md` (c).
* **Prosody/style quality downstream of the vocoder is untested.** Because the decoder failure
  dominates and saturates the output, whether `diffusion`/`predictor_encoder`/`style_encoder` are
  themselves well-trained for style/prosody (as opposed to just not causing the clipping symptom,
  which the random-decoder probe rules out) remains a genuinely open question — a different question
  from the one this task answers.
* **`speaker_sim` is a weak signal for this specific failure mode.** Confirmed-garbage,
  75-81%-clipped v10 output still scores 0.31-0.35 (vs. 0.48 for clean control audio) —
  non-trivially positive, not a near-zero outlier. A reader who only skims `results/metrics.json`
  without this narrative could mistake that for partial success; it is not.
* **The `clip_fraction` regression-check threshold (0.3 in `code/audio_quality_check.py`) has only
  three independent calibration points**: control (0.0), real v10 checkpoints (0.750, 0.807), and
  the random-decoder probe (0.004). It classifies all three correctly, but has no broader
  calibration set from other checkpoints or speakers.
* **`rtf` was measured on CPU**, not this project's registered H100 target hardware, so it is only a
  same-hardware, across-variant comparison within this task, not a production-latency estimate.
* **`ttfb_ms` is deliberately not measured** — the harness is an offline batch script producing one
  complete WAV per invocation; there is no streaming/serving endpoint in scope, so "time to first
  audio byte" is not a meaningful quantity here (per `plan/plan.md` step 17). This is an intentional
  omission, documented in `results/metrics.json`'s absence of a `ttfb_ms` key and in
  `results/v10_diagnosis.md`'s Metrics note, not an oversight.

## Analysis

**Plan assumption check** (re-reading `plan/plan.md`'s Objective and Approach against actual
results): the plan's leading hypothesis, formed during planning by directly reading
`train_second_v10.py:load_checkpoint()` and `config_david_v10.yml`, was that `ignore_modules`
omitting `"decoder"` caused a partial, architecture-mismatched load of the HiFi-GAN vocoder from an
ISTFTNet-shaped `first_stage_v3.pth`, leaving it "randomly-initialized-then-barely-touched." Actual
results **confirm this hypothesis exactly** — no assumption in the plan's Objective or Approach was
contradicted. One refinement, not a contradiction: the plan's own phrasing anticipated a vocoder
that was merely under-trained from a near-random start ("barely touched"). Step 11's falsification
probe (`code/random_decoder_probe.py`) shows the real mechanism is worse than that — a decoder left
at *pure* random init (never touched by any checkpoint at all) does **not** reproduce the clipping
failure (clip fraction 0.004, `is_likely_noise=False`), while the real, partially-mismatched-loaded
decoder does (0.750-0.807). The partial load is an actively bad "Frankenstein" initialization — some
`decoder.*` keys overwritten with weights trained for a structurally different architecture, the
rest left at fresh random init — not equivalent to a clean random start. This sharpens *why* 17
epochs were not enough (the model was fighting an inconsistent starting point, not merely starting
from scratch), but does not change the plan's root-cause attribution or its recommended fix
(`train_second_v10.py`'s `ignore_modules`).

A second contradicted informal assumption, flagged for correction rather than found in the written
plan: `task_description.md`'s Motivation section speculated the earlier ad hoc audio comparison
delivered to the user this session might itself have been invalid (a harness bug rather than a real
defect). This task's evidence shows the opposite — the ad hoc finding ("v10 produces no intelligible
voice") was directionally correct; what it lacked was the missing/unexpected-key instrumentation and
control validation to *prove* it wasn't a harness bug, which this task now supplies
(`results/v10_diagnosis.md`, final paragraph before "Metrics note").

## Files Created

* `code/infer_styletts2.py` — the reusable, instrumented StyleTTS2-native inference harness
  (deliverable 1 of `task_description.md`'s Expected Outputs). Logs per-module parameter count,
  missing/unexpected key counts, and primary-vs-fallback load path to both stdout and a JSON file.
* `code/inspect_checkpoint.py` — the cheap tensor-level falsifier (Milestone B): reads
  `net["decoder"]` key names, checks `torch.isfinite`, and computes per-module weight-norm summaries
  directly from a raw `.pth`, no inference environment required.
* `code/audio_quality_check.py` — the automatable noise/clipping regression check
  (`task_description.md` Scope §4's fourth deliverable), combining spectral flatness, silence
  fraction, and clip fraction into `is_likely_noise`. Includes a runnable `__main__` self-check.
* `code/random_decoder_probe.py` — the step-11 falsification probe: loads a real checkpoint into
  every module except `decoder`, which is left at fresh random init, to isolate the decoder's
  specific contribution to the clipping symptom.
* `code/score_speaker_sim.py` — GE2E cosine speaker-similarity scorer against the 11labs David
  reference corpus (reused pattern from `t0008_tts_eval_harness_baselines`).
* `code/paths.py` — shared path constants for the harness scripts.
* `code/kikiri-tts/` (gitignored, not committed — `code/.gitignore`) — the cloned upstream
  StyleTTS2/kokoro source the harness imports from.
* `results/checkpoint_forensics.md` / `results/checkpoint_forensics_raw.json` — Milestone B tensor
  forensics: per-module parameter count, finiteness, and weight-norm for both v10 checkpoints and
  the project reference `first_stage_v3.pth`.
* `results/control_test.md` — Milestone D control-validation gate result (PASS) against the external
  base pretrained StyleTTS2 LibriTTS checkpoint, including the real harness bug it caught and fixed
  (legacy `weight_norm`/`spectral_norm` parametrization key naming).
* `results/v10_diagnosis.md` — the root-cause verdict: real training defect from the start, traced
  to `train_second_v10.py:load_checkpoint()`'s `ignore_modules` list.
* `results/creative_thinking.md` — step 11's falsification probe and its refinement of the failure
  mechanism (worse than random, not merely under-trained).
* `results/load_log_epochs_2nd_00020.json`, `results/load_log_epoch_2nd_00016.json`,
  `results/load_log_epoch_2nd_00014.json` — full per-module load instrumentation for the control and
  both v10 checkpoints.
* `results/random_decoder_probe.json` — structured output of the step-11 probe (load report + audio
  quality stats).
* `results/speaker_sim_scores.json` — raw GE2E cosine scores per audio sample.
* `results/audio_samples/` (DVC-tracked, `results/audio_samples.dvc`, pushed) —
  `control_epochs_2nd_00020.wav`, `v10_epoch16_primary.wav`, `v10_epoch14_backup.wav`,
  `probe_random_decoder.wav`, plus a `.timing.json` per synthesis run — deliverable 2 of
  `task_description.md`'s Expected Outputs.
* `results/metrics.json` — `rtf` and `speaker_sim` in explicit multi-variant format for all three
  primary variants (control, v10 primary, v10 backup); `ttfb_ms` intentionally omitted (see
  Limitations).
* `results/costs.json`, `results/remote_machines_used.json` — zero-cost, no-remote-machine records
  (CPU-only local work).

## Examples

Ten-plus concrete input/output instances across categories, per
`arf/specifications/task_results_specification.md`'s Examples requirement for `code-reproduction`
task types. All values below are copied verbatim from this task's own result files — none are
fabricated.

### Example 1 (best case) — control checkpoint synthesis, harness validation gate

Input (identical harness invocation used for every checkpoint in this task, via
`code/infer_styletts2.py`):

```text
checkpoint:    code/kikiri-tts/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth
config:        code/kikiri-tts/StyleTTS2/Models/LibriTTS/config.yml
reference_wav: code/kikiri-tts/StyleTTS2/Demo/reference_audio/1221-135767-0014.wav  (3.0s)
text:          "This is a test of the Style T T S two inference harness."
alpha=0.3  beta=0.7  diffusion_steps=5  embedding_scale=1.0
```

Raw output (`results/audio_samples/control_epochs_2nd_00020.timing.json` +
`code/audio_quality_check.py` stats cited in `results/control_test.md`):

```json
{
  "wall_time_seconds": 22.33084505100851,
  "duration_seconds": 4.922916666666667,
  "rtf": 4.536100560509557,
  "rms": 0.0370,
  "peak": 0.3796,
  "clip_fraction": 0.0,
  "silence_fraction": 0.309,
  "spectral_flatness": 0.0615,
  "is_likely_noise": false
}
```

Illustrates: what a genuinely working checkpoint's output looks like through this harness — low clip
fraction, healthy crest factor, low spectral flatness (formant structure, not noise). This is the
baseline every other example is judged against.

### Example 2 (worst case) — v10 primary checkpoint (`epoch_2nd_00016.pth`, the "best" pick)

Input: identical harness invocation and text as Example 1, checkpoint swapped to
`tasks/t0010_stage2_safeguarded_training/data/run_v10/epoch_2nd_00016.pth`, reference audio swapped
to a 5.48s concatenation of three `11labs_david` clips (required — see Example 9).

Raw output (`results/audio_samples/v10_epoch16_primary.timing.json` + stats from
`results/v10_diagnosis.md`'s table):

```json
{
  "wall_time_seconds": 18.01375235900923,
  "duration_seconds": 3.4229166666666666,
  "rtf": 5.262690890033129,
  "rms": 0.983,
  "peak": 1.000,
  "clip_fraction": 0.750,
  "silence_fraction": 0.0,
  "spectral_flatness": 0.00048,
  "dominant_frequency": "0 Hz (DC)",
  "is_likely_noise": true
}
```

Illustrates: the failure signature this task exists to explain — 75% of samples pinned at full
scale, near-zero spectral flatness dominated by a DC offset, not classic broadband white noise.

### Example 3 (worst case, contrastive with Example 2) — v10 backup checkpoint (`epoch_2nd_00014.pth`)

Input: identical to Example 2, checkpoint swapped to `epoch_2nd_00014.pth` (epoch 15, val_loss
0.818, two epochs earlier than the primary pick).

Raw output (`results/audio_samples/v10_epoch14_backup.timing.json`):

```json
{
  "wall_time_seconds": 13.701156025999808,
  "duration_seconds": 2.497916666666667,
  "rtf": 5.485033271459472,
  "rms": 0.991,
  "peak": 1.000,
  "clip_fraction": 0.807,
  "silence_fraction": 0.0,
  "spectral_flatness": 0.00019,
  "dominant_frequency": "0 Hz (DC)",
  "is_likely_noise": true
}
```

Illustrates: Key Question 3's direct answer — the backup checkpoint (two epochs earlier) fails
**nearly identically** to the primary (clip fraction 0.807 vs. 0.750, both DC-dominated), ruling out
"just use the backup" as a fix and ruling out a late-training-only regression.

### Example 4 (boundary/contrastive) — random-decoder falsification probe

Input: `code/random_decoder_probe.py` builds the v10-primary model, loads `epoch_2nd_00016.pth`'s
real trained weights into every module **except** `decoder`, which is left at `build_model()`'s
fresh random initialization (verified untouched via a before/after state-dict diff).

Raw output (`results/random_decoder_probe.json`):

```json
{
  "decoder_confirmed_untouched_by_checkpoint": true,
  "wall_time_seconds": 17.526220996995107,
  "audio_quality": {
    "rms": 0.5974076119144697,
    "peak": 0.999786376953125,
    "silence_fraction": 0.0,
    "spectral_flatness": 0.24122372269630432,
    "clip_fraction": 0.004055188962207559,
    "is_likely_noise": false
  }
}
```

Illustrates: the pivotal finding from step 11 — a decoder that was **never touched by any checkpoint
at all** (clip fraction 0.004) is qualitatively closer to the clean control (Example 1) than to the
real, partially-loaded v10 decoder (Examples 2-3, clip fraction 0.750-0.807). This proves the real
decoder's partial-mismatch initialization is actively worse than pure random, not merely equivalent
to "undertrained."

### Example 5 (contrastive table) — same text/reference recipe, four decoder states side by side

| Variant | Checkpoint decoder state | Clip fraction | Spectral flatness | `is_likely_noise` |
| --- | --- | ---: | ---: | --- |
| Control | fully pretrained, correct architecture | 0.0 | 0.0615 | false |
| v10 primary | partial ISTFTNet-into-HiFi-GAN mismatched load | 0.750 | 0.00048 | true |
| v10 backup | partial ISTFTNet-into-HiFi-GAN mismatched load | 0.807 | 0.00019 | true |
| Random-decoder probe | pure random init, never checkpoint-loaded | 0.004 | 0.2412 | false |

Illustrates: the full four-way contrast in one place — a mismatched partial load is worse than
having no checkpoint applied to `decoder` at all.

### Example 6 (boundary case) — the harness bug caught and fixed by the control gate

Input: first control-checkpoint load attempt, before the fix in `code/infer_styletts2.py`'s
`_rename_legacy_parametrization_keys()`. The officially released checkpoint uses the classic
`torch.nn.utils.weight_norm` state-dict key names; this fork's `models.py` constructs modules with
the newer `torch.nn.utils.parametrizations.weight_norm` API.

Raw output (first attempt, narrated in `results/control_test.md` — the module-level counts before
the fix):

```text
module=predictor  num_model_params=122  num_ckpt_params=122
primary:  matched=0   missing=122  unexpected=122
fallback: matched=32  missing=90   unexpected=90   <- HARD FAILURE, raised (REQ-2 working as designed)
```

Illustrates: the instrumented loader's hard-failure gate (Key Question 1's requirement) catching a
real bug in the harness/environment itself — a false attribution to "v10 is broken" was avoided
because this ran against the control, not v10, first.

### Example 7 (boundary case, contrastive with Example 6) — the same module, after the fix

Raw output (`results/load_log_epochs_2nd_00020.json`, the `predictor` module entry, final state):

```json
{
  "module": "predictor",
  "num_model_params": 122,
  "num_ckpt_params": 122,
  "primary": {"matched": 0, "missing": 122, "unexpected": 122},
  "fallback": {"matched": 122, "missing": 0, "unexpected": 0},
  "used_fallback": true,
  "final_missing": 0,
  "final_unexpected": 0
}
```

Illustrates: after adding the legacy-parametrization key rename to the `module.`-stripped fallback
path, the same module loads cleanly (0 missing, 0 unexpected) — confirming the mismatch was a pure
key-naming rename, not a real architecture difference (identical tensor shapes at every position,
verified directly per `results/control_test.md`).

### Example 8 (random/representative) — v10 primary's `decoder` module load entry

Raw output (`results/load_log_epoch_2nd_00016.json`, `decoder` module entry):

```json
{
  "module": "decoder",
  "num_model_params": 678,
  "num_ckpt_params": 678,
  "primary": {"matched": 0, "missing": 678, "unexpected": 678},
  "fallback": {"matched": 678, "missing": 0, "unexpected": 0},
  "used_fallback": true,
  "final_missing": 0,
  "final_unexpected": 0
}
```

Illustrates: the loader itself reports a clean 0/0 load for `decoder` on v10 primary — this is
precisely why Key Question 1 (reproduction bug) is ruled out: the *loader* did its job correctly
given the checkpoint it was told to load (`epoch_2nd_00016.pth`, produced by training, not by this
task's harness). The defect is upstream, in what `train_second_v10.py` wrote into that checkpoint
during Stage 2 training, not in how this task's harness reads it.

### Example 9 (boundary case) — `StyleEncoder`'s minimum reference-clip duration

Input: synthetic mel-spectrogram inputs of increasing duration fed directly to
`code/kikiri-tts/StyleTTS2/models.py`'s `StyleEncoder.compute_style` in isolation (Key Question 6
investigation, `results/v10_diagnosis.md`).

Raw output:

```text
input_duration=0.7s -> RuntimeError: Calculated padded input size per channel: (5 x N).
                        Kernel size: (5 x 5). Kernel size can't be greater than actual input size
input_duration=1.0s -> succeeds, style vector produced, shape (1, 128)
```

Illustrates: the short-reference-clip crash discovered during the earlier ad hoc session is an
**architectural minimum-duration requirement of `StyleEncoder`** (four halving `ResBlk` stages
feeding an unpadded `kernel_size=5` conv), unrelated to the v10 noise defect — this is the direct
answer to Key Question 6. All `11labs_david` filler clips top out at ~1.67s, too short individually,
which is why this task's v10 runs use a 5.48s concatenation of three clips instead of one.

### Example 10 (per-module boundary comparison) — decoder weight-norm across three checkpoints

Raw output (`results/checkpoint_forensics.md`'s per-checkpoint tables, `decoder` row extracted):

```text
epoch_2nd_00016 (v10 primary): params=54,289,492  finite=yes  weight_norm=213.2920
epoch_2nd_00014 (v10 backup):  params=54,289,492  finite=yes  weight_norm=213.2222
first_stage_v3 (control ref):  params=53,276,190  finite=yes  weight_norm=196.9789
```

Illustrates: Key Question 4's direct answer — no NaN/Inf anywhere, and weight norms are close across
all three (213.2 vs. 213.2 vs. 197.0), ruling out a catastrophic mid-training blowup. The defect is
not weight divergence; it is that the weights, however numerically stable, encode the wrong
architecture's semantics.

### Example 11 (decisive architecture classification signal)

Raw output (`results/checkpoint_forensics.md`, ground-truth discriminator found by reading
`code/kikiri-tts/StyleTTS2/Modules/{hifigan,istftnet}.py` directly):

```text
v10 primary decoder state_dict:    678 keys, includes "generator.alphas.0".."generator.alphas.4"
v10 backup decoder state_dict:     678 keys, includes "generator.alphas.0".."generator.alphas.4"
first_stage_v3 decoder state_dict: 375 keys, zero "alphas.*" keys present
```

Illustrates: the tensor-level, pre-inference confirmation (Milestone B, run before any venv/audio
code) that v10's decoder is HiFi-GAN-shaped while the checkpoint `train_second_v10.py` loaded it
from (`first_stage_v3.pth`) is ISTFTNet-shaped — the root architecture mismatch, established from
raw tensor keys alone, independent of and prior to any audio evidence.

### Example 12 (training-log cross-reference, Key Question 5)

Raw output (42 records read directly from
`tasks/t0010_stage2_safeguarded_training/data/run_v10/metrics.jsonl`, epoch-level `val_loss` values
for epochs 9-17):

```text
epoch=9   val_loss=1.079
epoch=16  val_loss=0.797   <- "best", this is epoch_2nd_00016.pth, the primary checkpoint
epoch=17  val_loss=0.853   <- last epoch, worse than epoch 16
disc_loss, grad_norm_total, loss_total, skip_count, lr: None on every record (never populated)
```

Illustrates: no per-step signal in this run's own training log could have surfaced the vocoder
defect — `val_loss` decreased smoothly with no divergence spike, and the finer-grained diagnostic
fields this project's metrics schema supports were simply never populated for this run. This
directly answers Key Question 5 and explains how a smooth, good-looking loss curve coexisted with a
vocoder that never produced valid audio.

## Task Requirement Coverage

Operative task text, quoted verbatim from `task.json` and `task_description.md`:

> **name**: "v10 checkpoint synthesis quality forensics" **short_description**: "Determine whether
> kokoro-v10-best produces noise instead of speech due to a bug in an ad hoc inference reproduction,
> or a real checkpoint defect, and act accordingly."

> **Key Questions** (`task_description.md`):
> 1. Does the ad hoc inference script actually load every module's weights correctly? (Log
>    missing/unexpected key counts explicitly, per module, for both the primary load attempt and the
>    `module.`-stripped fallback.)
> 2. With a known-good control (base pretrained StyleTTS2 LibriTTS checkpoint), does the same
>    inference harness produce intelligible speech?
> 3. If the harness is validated correct: does `epoch_2nd_00016.pth` (primary) produce noise while
>    `epoch_2nd_00014.pth` (backup) produces speech, or do both fail?
> 4. Do any loaded modules contain NaN/Inf weights, or weight-norm statistics wildly divergent from
>    a known-good reference?
> 5. Does `metrics.jsonl` show any per-step anomaly the epoch-level val_loss average could have
>    masked?
> 6. Is `compute_style`'s reference-clip-length requirement itself a sign of mismatch, or expected
>    StyleTTS2 behavior?
> 
> **Scope**: (1) reproduce the ad hoc setup properly, with instrumentation, under `code/`; (2) run a
> control test on the base pretrained checkpoint before touching v10 again; (3) diagnose v10
> specifically (primary vs. backup, NaN/Inf, weight-norm, metrics.jsonl cross-reference); (4)
> root-cause and recommend a fix, and produce a short automatable regression check.
> 
> **Expected Outputs**: (1) a reusable inference recipe — committed, documented, instrumented script
> under `code/`, not a throwaway `/tmp` script; (2) audio samples saved as real files under
> `results/audio_samples/` for the control checkpoint and v10 primary/backup.

Requirement-by-requirement coverage, reusing `REQ-*` IDs from `plan/plan.md`'s Task Requirement
Checklist:

| ID | Requirement | Status | Direct answer / result | Evidence |
| --- | --- | --- | --- | --- |
| REQ-1 | Cheap tensor-level falsifier runs first, before any inference/venv code | **Done** | Confirmed architecture mismatch at the tensor level: v10 decoder is HiFi-GAN-shaped (678 keys, `generator.alphas.*` present), `first_stage_v3.pth` is ISTFTNet-shaped (375 keys, no `alphas`) | `results/checkpoint_forensics.md`, `code/inspect_checkpoint.py`, Example 11 |
| REQ-2 | Instrumented harness logs missing/unexpected key counts per module, primary + fallback (Key Question 1) | **Done** | Both v10 checkpoints load with 0 missing/0 unexpected on all 13 modules; a real harness bug (legacy parametrization keys) was caught and fixed via this exact instrumentation on the control before ever trusting a v10 result | `code/infer_styletts2.py`, `results/load_log_*.json`, Examples 6-8 |
| REQ-3 | Control test against base pretrained StyleTTS2 LibriTTS checkpoint before touching v10 (Key Question 2) | **Done** | PASS: `is_likely_noise=False`, spectral flatness 0.0615, clip fraction 0.0 | `results/control_test.md`, `results/audio_samples/control_epochs_2nd_00020.wav`, Example 1 |
| REQ-4 | Compare v10 primary vs. backup outputs (Key Question 3) | **Done** | Both fail nearly identically (clip fraction 0.750 vs. 0.807, both DC-dominated) — not a late-training-only regression, backup is not a viable fallback | `results/audio_samples/v10_epoch16_primary.wav`, `v10_epoch14_backup.wav`, `results/v10_diagnosis.md`, Examples 2-3 |
| REQ-5 | NaN/Inf + weight-norm comparison across v10 checkpoints and control (Key Question 4) | **Done** | No NaN/Inf in any of the 13 modules of either checkpoint; decoder weight norm close across all three (213.29 / 213.22 / 196.98) — rules out a training blowup, points to a semantically-wrong-but-numerically-stable initialization instead | `results/checkpoint_forensics.md`, `results/checkpoint_forensics_raw.json`, Example 10 |
| REQ-6 | Cross-reference `metrics.jsonl` for per-step signal missed by epoch-level val_loss (Key Question 5) | **Done** | 42 records inspected directly; `disc_loss`/`grad_norm_total`/`loss_total`/`skip_count`/`lr` are `None` on every record for this run, and `val_loss` decreases smoothly with no divergence spike — no per-step signal could have surfaced the vocoder defect | `results/v10_diagnosis.md` ("Training-log cross-reference" section), Example 12 |
| REQ-7 | Document whether `compute_style`'s reference-clip-length requirement is a mismatch signal or expected behavior (Key Question 6) | **Done** | Expected StyleTTS2 architectural behavior (`StyleEncoder`'s 4 halving stages + unpadded kernel-size-5 conv require ~1s+ input), unrelated to the noise defect | `results/v10_diagnosis.md` ("Key Question 6" section), Example 9 |
| REQ-8 | Reusable inference recipe: kikiri-tts clone + submodules under `code/` (gitignored), instrumented script committed | **Done** | `code/infer_styletts2.py` committed; `code/kikiri-tts/` present but gitignored via `code/.gitignore` | `code/infer_styletts2.py`, `code/.gitignore` |
| REQ-9 | Audio sample deliverables under `results/audio_samples/`, DVC-tracked | **Done** | 7 files (4 WAVs + 3 timing JSONs, plus the step-11 probe WAV) DVC-tracked and pushed | `results/audio_samples/`, `results/audio_samples.dvc` |
| REQ-10 | `results/control_test.md` with pass/fail + objective stats | **Done** | PASS verdict with RMS/peak/clip/silence/flatness stats | `results/control_test.md` |
| REQ-11 | `results/v10_diagnosis.md` with root cause and recommended action | **Done** | Verdict: real defect from the start; recommendation: discard/retrain with `ignore_modules` fix | `results/v10_diagnosis.md` |
| REQ-12 | Short, objective, automatable regression check saved to `code/` | **Done** | `code/audio_quality_check.py` (spectral flatness + silence fraction + clip fraction, with `is_likely_noise` verdict and a runnable self-check) | `code/audio_quality_check.py` |
| REQ-13 | Applicable registered metrics (`speaker_sim`, `rtf`) measured; `ttfb_ms` explicitly omitted with reasoning | **Done** | All three variants' `rtf`/`speaker_sim` recorded in explicit variant format; `ttfb_ms` omitted with reasoning in `results/v10_diagnosis.md`'s Metrics note and this file's Limitations | `results/metrics.json` |

Both Expected Outputs from `task_description.md` are satisfied: the reusable inference recipe
(`code/infer_styletts2.py`, REQ-8) and the audio sample files (`results/audio_samples/`, REQ-9). All
six Key Questions have a direct, evidenced answer (REQ-2, REQ-3, REQ-4, REQ-5, REQ-6, REQ-7 above).
The task's core ask — "act accordingly" — is satisfied by `results/v10_diagnosis.md`'s
Recommendation section (do not use `kokoro-v10-best`; retrain with the `ignore_modules` fix) and by
filing the eval-harness pre-completion regression-check follow-up for step 14 (`suggestions`), per
`checkpoint.md`'s Cross-Step Decisions.
