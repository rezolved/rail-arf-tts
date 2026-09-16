# v10 Diagnosis (Milestone E)

Synthesizes Milestone B (tensor forensics), Milestone D (control validation, passed --
`results/control_test.md`), and this milestone's own inference runs on both v10 checkpoints into one
root-cause verdict, per `task_description.md` Scope §4.

## Setup

* Checkpoints: `epoch_2nd_00016.pth` (primary, "best", epoch 17) and `epoch_2nd_00014.pth` (backup,
  epoch 15), both from `tasks/t0010_stage2_safeguarded_training/data/run_v10/`.
* Config: `tasks/t0010_stage2_safeguarded_training/data/run_v10/config_david_v10.yml`.
* Reference audio: a concatenation of three
  `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` clips (`lining_up_suggestions_17`,
  `lining_up_suggestions_10`, `putting_them_head_to_head_15`, joined with 0.2s silence gaps, total
  5.48s) -- see Key Question 6 below for why a single clip could not be used directly.
* Text and harness: identical to the control run (`results/control_test.md`) for comparability.

## Load instrumentation: both checkpoints load cleanly

Both `epoch_2nd_00016.pth` and `epoch_2nd_00014.pth` loaded with **0 missing, 0 unexpected on all 13
modules** (`results/load_log_epoch_2nd_00016.json`, `results/load_log_epoch_2nd_00014.json`), same
as the validated control. **The loader is not the problem for either checkpoint.** This rules out
Key Question 1's "reproduction bug in the loader" as the explanation -- the instrumented,
hard-failing loader (REQ-2) accepts both checkpoints cleanly, and it already proved itself capable
of catching a real mismatch (the control's initial `parametrizations` API drift, see
`results/control_test.md`) rather than silently passing bad loads.

## Audio output: both checkpoints produce saturated, clipped garbage -- not classic "noise"

| Checkpoint | Duration | RMS | Peak | Clip fraction (`\|x\|>0.99`) | Silence fraction | Spectral flatness | Dominant freq. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Control (`epochs_2nd_00020`) | 4.92s | 0.037 | 0.380 | 0.0 | 0.309 | 0.0615 | speech-like, broadband |
| v10 primary (`epoch_2nd_00016`) | 3.42s | 0.983 | 1.000 | **0.750** | 0.0 | 0.00048 | 0 Hz (DC) |
| v10 backup (`epoch_2nd_00014`) | 2.50s | 0.991 | 1.000 | **0.807** | 0.0 | 0.00019 | 0 Hz (DC) |

Both v10 outputs are **75-81% clipped at full scale**, with the dominant frequency component at 0 Hz
(a DC offset with magnitude ~100x every other frequency bin) and near-zero silence. This is a
qualitatively different failure signature from generic white noise (which would show HIGH spectral
flatness, not near-zero): it is the signature of a vocoder emitting values far outside the valid
[-1, 1] range, saturating on write. `code/audio_quality_check.py` was extended with a
`clip_fraction` signal specifically because the original flatness-only heuristic (correct for
classic white noise) would have scored these clips as "not noise" (flatness 0.0002-0.0005, far below
any white-noise threshold) -- a false negative that would have missed exactly the failure mode this
task exists to catch. With the clip-fraction signal added,
`check_audio_quality(...).is_likely_noise == True` for both v10 outputs.

**Both epoch 16 and epoch 14 fail nearly identically** (clip fraction 0.750 vs. 0.807, both
dominant-DC, both zero silence). This is the key discriminator for Key Question 3: it is NOT a sharp
regression between epoch 15 and epoch 17 -- whatever is broken was already broken by epoch 14 (and,
by the reasoning below, essentially from the start of Stage 2 training).

## Tensor forensics cross-reference (Milestone B)

`results/checkpoint_forensics.md`'s verdict (architecture-mismatch hypothesis confirmed at the
tensor level) is corroborated, not contradicted, by the audio evidence:

* `train_second_v10.py:load_checkpoint()`'s `ignore_modules` list (lines 253-259) excludes
  `predictor_encoder`, `msd`, `mpd`, `wd`, `diffusion` from the Stage-1-checkpoint load -- but
  **not** `decoder`. `first_stage_v3.pth`'s decoder is istftnet-shaped (confirmed: 375 keys, 2 `ups`
  stages, no `generator.alphas`); `config_david_v10.yml`'s decoder is `hifigan`-shaped (678 keys, 4
  `ups` stages, `generator.alphas.{0..4}` present). The loader's `len(matched) == 0` guard only
  raises on a fully-zero match; a *partial* shape match (shared architecture-agnostic layers
  overlap, the HiFi-GAN generator's `ups`/`resblocks`/`alphas`/`conv_post`/`noise_convs` do not)
  passes silently. This leaves the actual vocoder-generating layers at random initialization
  entering Stage 2 training.
* No NaN/Inf anywhere in either v10 checkpoint (`results/checkpoint_forensics_raw.json` -- all 13
  modules `finite=True`). The decoder's weight-norm is close between epoch 14 (213.22) and epoch 16
  (213.29), and both are close to the control's (196.98) -- **the weights are not diverging or
  blowing up**. **Refined by step 11's falsification probe** (`results/creative_thinking.md`,
  `code/random_decoder_probe.py`): this is not merely a HiFi-GAN vocoder that started near a random
  initialization and made limited progress in 17 epochs. Loading v10 primary's real checkpoint into
  every module except `decoder` -- left at pure `build_model()` random init -- gives
  `clip_fraction=0.004` (`is_likely_noise=False`), close to the control, while the real,
  partially-mismatched-loaded decoder gives `clip_fraction=0.750-0.807`. A purely random decoder
  therefore does **not** reproduce the failure; the actual initialization is a "Frankenstein" mix of
  freshly-random layers and layers overwritten with weights trained for a structurally different
  (ISTFTNet) architecture, which is an actively **worse** starting point than plain random init, not
  an equivalent one. This is consistent with a catastrophic-blowup-free, numerically-stable-but-
  semantically-wrong initialization (weight-norm alone would not distinguish the two, which is why
  the probe was necessary), rather than a mid-training blowup (which would show as NaN/Inf or a
  weight-norm spike -- neither is observed).

## Training-log cross-reference (Key Question 5, re-verified directly, not just cited from research)

Read `tasks/t0010_stage2_safeguarded_training/data/run_v10/metrics.jsonl` directly (42 records,
`step` values `{0, 10, 20, 30, None}` -- validation-epoch snapshots, no true per-optimizer-step
signal). Confirmed for epochs 9-17: `disc_loss`, `grad_norm_total`, `loss_total`, `skip_count`, `lr`
are `None` on every record (never populated for this run); `acoustic_norm` stays in a narrow, smooth
band (6.38-7.32) with no divergence spike; `val_loss` decreases smoothly from 1.079 (epoch 9) to
0.797 (epoch 16, the "best"/primary pick) before ticking back up to 0.853 (epoch 17, actually the
LAST epoch, not the best-val one -- `epoch_2nd_00016.pth`'s filename tracks epoch count, its
`val_loss=0.797` is genuinely the run's best). **No per-step signal in this log could have surfaced
the vocoder defect** -- confirms `research/research_summary.md` point 6 exactly, and confirms it is
not merely a training-log logging gap specific to this run: `val_loss` (whatever mixture of
reconstruction/duration/F0 losses it aggregates) is simply not sensitive to "does the adversarial
decoder produce valid-range audio," which is precisely why a smooth, good-looking val_loss curve
coexisted with a vocoder that never worked.

## Key Question 6: the short-reference-clip crash is expected StyleTTS2 behavior, not a mismatch signal

Every clip in `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` tops out at ~1.67s (the
corpus is short voice-commerce filler phrases) -- too short for a single clip to safely drive
`compute_style`. Confirmed by direct, isolated testing of `models.py`'s `StyleEncoder` (dim_in=64,
style_dim=128, max_conv_dim=512, matching `config_david_v10.yml`) on synthetic mel-spectrogram
inputs of increasing duration: the encoder **crashes below ~1.0s** input
(`RuntimeError: Calculated padded input size per channel: (5 x N). Kernel size: (5 x 5)...` for
0.1-0.7s inputs; works cleanly at 1.0s+). Root cause read directly from `models.py`'s
`StyleEncoder.__init__`: `repeat_num=4` half-downsampling `ResBlk` stages followed by a
`Conv2d(..., kernel_size=5, padding=0)` with no padding -- an 80-mel spectrogram whose time axis is
halved 4 times must retain at least 5 frames afterward for that final conv to have a valid input,
which requires roughly 80+ frames (~1s at `hop_length=300`, `sr=24000`) before downsampling. This is
an **architectural minimum-duration requirement of `StyleEncoder`, unrelated to the v10 noise
defect** -- it explains why this session's earlier ad hoc reproduction (using a single short filler
clip) crashed on `compute_style`, a separate, already-understood issue distinct from the "produces
noise" finding. This task worked around it by concatenating three short David clips into one 5.48s
reference (see Setup), safely above the ~1s floor.

## Verdict

**Real defect from the start**, not a reproduction bug and not a late-training-specific regression:

1. **Not a reproduction/loader bug** (rules out Key Question 1/Milestone D-adjacent explanations):
   the instrumented loader loads both v10 checkpoints with 0 missing/0 unexpected on all 13 modules,
   using the same harness code already validated against a known-good external control
   (`results/control_test.md`).
2. **Not isolated to late epochs** (Key Question 3): `epoch_2nd_00014` (epoch 15) and
   `epoch_2nd_00016` (epoch 17, the "best" pick) produce near-identical saturated-clipping failures
   (75% vs. 81% samples clipped, both DC-dominated, both zero silence). Using the backup checkpoint
   instead of primary would **not** fix this.
3. **Root cause, traced to a specific line**: `train_second_v10.py:load_checkpoint()`'s call site
   (lines 244-260) passes `ignore_modules=["predictor_encoder", "msd", "mpd", "wd", "diffusion"]`
   when loading `first_stage_v3.pth` into the Stage-2 model -- `decoder` is not excluded, but
   `first_stage_v3.pth`'s decoder is istftnet-shaped while `config_david_v10.yml` builds a
   hifigan-shaped decoder. The loader's `len(matched) == 0` guard does not catch a *partial* shape
   match, so the HiFi-GAN generator's actual vocoder layers (`ups`, `resblocks`, `alphas`,
   `conv_post`, `noise_convs`) entered Stage 2 training at random initialization instead of a
   pretrained starting point, and 17 epochs (a normal budget for FINE-TUNING an already-working
   vocoder, not for training one from scratch) were not enough to recover.
4. Because the initialization-time bug predates epoch 1, and epochs 14 and 16 fail identically, the
   most defensible statement is that **this training run never produced a working vocoder at any
   point** -- `config_david_v10.yml`'s specific combination (`decoder.type: hifigan` +
   `train_second_v10.py`'s `ignore_modules` list not excluding `decoder`) never had a working
   starting point to begin with.

**The earlier ad hoc audio comparison delivered to the user this session was NOT invalid** in the
sense of "the reproduction was buggy" -- the underlying finding (v10 produces no intelligible voice)
is now confirmed independently, through an instrumented, control-validated harness, with objective
audio-stats evidence rather than a single unvalidated listen. What WAS incomplete about the ad hoc
session is exactly what `task_description.md`'s Motivation section says: it never logged per-module
missing/unexpected counts and never validated against a control, so it could not have distinguished
"harness bug" from "real defect" on its own -- this task supplies that missing evidence and the two
lines of evidence agree.

## Metrics note (`results/metrics.json`)

`rtf` and `speaker_sim` are recorded per checkpoint in the explicit multi-variant format. Both are
**corroborating diagnostics, not benchmark claims**: `rtf` was measured single-clip on CPU (not the
registered target's H100), so it is only a same-hardware, across-variant comparison (control vs.
v10, all measured identically) -- all three variants land around RTF 4.5-5.5, unsurprising for
unbatched CPU inference of a ~54M-parameter decoder plus a 5-step diffusion sampler, and not
indicative of anything checkpoint-specific. `speaker_sim` is notable for being a **weak, not
reliable, signal** in this case: even the confirmed-broken, 75-81%-clipped v10 outputs scored
0.31-0.35 against the David centroid (vs. 0.48 for the control) -- non-trivially positive despite
being audibly-by-stats garbage. `resemblyzer`'s GE2E embedding is not a noise detector; a a clipped
DC-saturated signal still has *some* spectral content resemblyzer maps somewhere in embedding space,
so `speaker_sim` alone would NOT have caught this defect (it produced a plausible-looking, merely
lower, number rather than an obvious near-zero outlier). This is exactly why the audio-stats
heuristic (`clip_fraction`, `spectral_flatness`) in `code/audio_quality_check.py` -- not
`speaker_sim` -- is the deliverable recommended as a pre-completion smoke gate.

`ttfb_ms` is **deliberately not measured**: this task's harness is an offline batch script that
writes one complete WAV per invocation; there is no streaming/production serving endpoint in scope,
so "time to first audio byte" is not a meaningful quantity here (per `plan/plan.md` step 17). This
is an intentional omission, not an oversight.

## Recommendation

* **Do not use `kokoro-v10-best` (`epoch_2nd_00016.pth`) or `epoch_2nd_00014.pth` for anything
  requiring audible output.** Both are confirmed non-functional at the vocoder level.
* **Retrain**, starting from a corrected `train_second_v10.py:load_checkpoint()` call: add
  `"decoder"` to the `ignore_modules` list for the `hifigan`-decoder config path (leaving it at its
  own random initialization explicitly, rather than a silently-partial-matched one, OR -- better --
  source a HiFi-GAN-shaped pretrained first-stage checkpoint for `config_david_v10.yml` to start
  from, if one exists/can be produced, so Stage 2 is genuine fine-tuning rather than training a
  vocoder from scratch in 17 epochs).
* **Process recommendation** (per `task_description.md` Scope §4's third bullet): `val_loss` alone
  is not sufficient evidence of training success for this project's pipeline. A synthesis smoke
  check -- text-to-audio plus the noise/clipping heuristic in `code/audio_quality_check.py` -- must
  run and pass before a Stage 2 training task can claim `completed`, not be deferrable to "later"
  eval-harness work. This is the gap that let `kokoro-v10-best` ship with nobody having listened to
  its audio (`task_description.md` Motivation). Filed as a follow-up in `results/suggestions.json`.
