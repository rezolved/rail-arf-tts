# v3 Checkpoint Forensics (Milestone 2, REQ-2/REQ-6/REQ-7/REQ-10)

No-GPU tensor-level cross-check of the v3 Stage 2 bundle, comparing `stage1/first_stage.pth` against
`best/david_v3_best_decoder_kokoro.pth` across all 5 Kokoro-API bundle modules. Produced by
`code/checkpoint_forensics.py`.

## Module-by-Module Weight-Norm Comparison (REQ-2, REQ-7)

| Module | Params | Finite | Weight norm (stage1) | Weight norm (best) | Relative delta | Changed (yes/no) | DP prefix present |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| bert | 6,292,480 | yes | 186.5264 | 174.3075 | -6.55% | yes | yes |
| bert_encoder | 393,728 | yes | 13.0890 | 46.2956 | +253.70% | yes | yes |
| predictor | 16,194,612 | yes | 141.6734 | 378.5873 | +167.23% | yes | yes |
| text_encoder | 5,606,400 | yes | 310.7809 | 373.4580 | +20.17% | yes | yes |
| decoder | 53,276,190 | yes | 196.9789 | 640.0961 | +224.96% | yes | yes |

## Byte-Identity Check (REQ-10)

* `stage1/first_stage.pth` local SHA-256:
  `8a3375e306bc5a7522e21bd996ed9317ea1bb7ebbafc99a8e485a4263803bcac`
* `best/david_v3_best_decoder_kokoro.pth` local SHA-256:
  `a49f12dd5b9fbfdafaa94ae766253be3dc9626dd498bbc9494f2dc7c5bc27b3a`
* VM `first_stage_v3.pth` SHA-256: **not obtainable** — `first_stage_v3.pth` was not found anywhere
  on `LLM-T1-NC80` (searched `~/kokoro-finetune` and the resolved `/mnt/cache/persist` share; see
  `data/vm_inventory/inventory.json`'s `first_stage_v3_search_result` field, and the "Critical
  finding" in `data/v3_train_list_UNRECOVERED.md`). Byte-identity between the local
  `stage1/first_stage.pth`, the VM's (unfound) copy, and whatever v6c/v6d actually loaded remains
  **unconfirmed**.
* No prior task's own artifacts (t0006, t0009) record a SHA-256 for any of these files either —
  confirmed by inspection, not assumed.

## All Top-Level `net` Keys in `stage1/first_stage.pth` (multispeaker scope)

| Top-level key | Raw param count | In 5-module bundle? |
| --- | ---: | --- |
| bert | 6,292,480 | yes |
| bert_encoder | 393,728 | yes |
| decoder | 53,276,190 | yes |
| diffusion | 43,836,160 | no |
| mpd | 41,105,770 | no |
| msd | 280,902 | no |
| pitch_extractor | 5,251,148 | no |
| predictor | 16,194,612 | yes |
| predictor_encoder | 13,880,813 | no |
| style_encoder | 13,880,813 | no |
| text_aligner | 7,868,452 | no |
| text_encoder | 5,606,400 | yes |
| wd | 1,173,634 | no |

## Multispeaker Resolution (REQ-6)

**Contradiction restated**: `best/config.json` says `multispeaker: true`; t0009's
`results/confound_table.md` says `true (assumed)`; `t0006`'s `code/config_david_v6c_stage2.yml` line
76 says `multispeaker: false` (with an inline comment claiming the v3 Stage 1 checkpoint itself is
`multispeaker:false`). None of the three carries a SHA-256 or file citation.

**Step 1 -- where the flag acts (VM `models.py`, found and copied to
`data/vm_inventory/kokoro_finetune_current/models.py.txt`)**: `models.py:808`,
`if args.multispeaker:` selects
`StyleTransformer1d(channels=args.style_dim*2, context_embedding_features=bert.config.hidden_size, context_features=args.style_dim*2, **kwargs)`
for the `else` branch it selects
`Transformer1d(channels=args.style_dim*2, context_embedding_features=bert.config.hidden_size, **kwargs)`
-- i.e. `StyleTransformer1d` additionally accepts a second conditioning input, `context_features` (a
style-vector pathway), that `Transformer1d` does not. This transformer becomes
`nets.diffusion.diffusion.net` AND `nets.diffusion.unet` (both point at the same object,
`models.py:839-840`). `diffusion` **is** a top-level key of the checkpoint's `net` dict (confirmed
above, 43,836,160 raw params) -- so, unlike a flag that only touches a component entirely absent
from the checkpoint, this one is in scope for shape forensics. It is outside the 5 Kokoro-API bundle
modules, however, so the module-by-module table above cannot show it; a dedicated inspection of
`net["diffusion"]` was required (see `code/checkpoint_forensics.py`'s `all_top_level_keys()` plus an
ad hoc key-shape dump run during this task, not re-run in `main()` since
`StyleTransformer1d`/`Transformer1d`'s own class source lives in an external,
unavailable-in-this-repo package).

**Step 2 -- shape evidence**: `net["diffusion"]`'s `diffusion.net.*` keys show exactly ONE
cross-attention conditioning pathway per transformer block (`attention.to_q`/`to_kv`/`norm_context`,
shapes `(512,1024)`/`(1024,1024)`/`(1024,)`), one `fixed_embedding.embedding.weight` of shape
`(512, 768)` (matching `embedding_max_length=512`, `embedding_features=768` -- the shared BERT
`context_embedding_features` conditioning both branches take), and no second, separately-shaped
conditioning block that would correspond to a distinct `context_features=style_dim*2=256` pathway
`StyleTransformer1d` alone accepts. This is *suggestive* of the simpler `Transformer1d` branch
(`multispeaker: false`): if the diffusion model had been built with `multispeaker: true`, a second,
structurally distinct conditioning pathway would be expected somewhere in this state dict for the
style-vector input `StyleTransformer1d` alone accepts.

**Verdict: `inferred false`.** This does NOT meet the "confirmed" bar defined in
`task_description.md`'s Rejection Criteria (checkpoint-shape finding AND either `models.py`
confirming which parameters the flag controls, OR a surviving launch YAML) because `models.py`
explains *where* the flag acts (which branch of `build_model()`) but not the exact resulting
parameter-shape signature of `StyleTransformer1d` vs. `Transformer1d` -- their actual class
definitions live in an external pip package (only re-exported via `from .utils import *` /
`from .sampler import *` in the VM's `Modules/diffusion/diffusion.py`, itself copied to
`data/vm_inventory/kokoro_finetune_current/Modules/diffusion.py.txt`) that was not recovered from
the VM within the bounded inventory window. The absence of a second conditioning pathway is
consistent with, but not proof of, `multispeaker: false`. **This directly favors t0006's
`config_david_v6c_stage2.yml` (`multispeaker: false`) over `best/config.json` (`true`) and t0009's
`confound_table.md` (`true`, assumed) --** both of the latter two are secondary artifacts (a
packaged inference-time bundle config, and an admittedly-assumed confound-table entry) with no shape
or file evidence behind them, whereas this verdict is grounded in the actual Stage 1 checkpoint's
own tensor shapes plus the model-building source. `data/config_david_v3_reconstructed.yml` records
`multispeaker: false # inferred: net["diffusion"] shape shows a single conditioning pathway, consistent with Transformer1d (multispeaker=false) per models.py:808; see Multispeaker Resolution in results/v3_checkpoint_forensics.md`.

## Config Diff vs v6c and t0009 (REQ-5)

Comparing `data/config_david_v3_reconstructed.yml` against
`tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml`:

| Field | Reconstructed v3 | v6c | Agreement | Better-evidenced value |
| --- | --- | --- | --- | --- |
| `data_params.train_data` | `../data/v3/train_list_266.txt` (naming guess, unrecovered) | `../data/v5/train_list_250.txt` | differs (expected -- different corpora) | Neither; v3's real list is unrecoverable (see `data/v3_train_list_UNRECOVERED.md`) |
| `loss_params.joint_epoch` | `6` (unknown, default from v6c) | `6` | same | Neither confirmed for v3; carried from v6c for lack of anything better |
| `loss_params.lambda_gen` | `0.2` (unknown, default from v6c's real file) | `0.2` | same | v6c's file, not t0009's confound table (see finding below) |
| `loss_params.lambda_slm` | `0.0` (inferred from `train_second_patch.diff`) | `0.0` | same | v3's own patch evidence, independently arrives at the same value as v6c |
| `model_params.multispeaker` | `false` (inferred from checkpoint-shape forensics) | `false` (inline-commented, no hash) | **same** | v3's own checkpoint-shape forensics -- an independent confirmation of v6c's claim, using primary evidence v6c's own file did not cite |

**Key finding surfaced by this diff, not merely restated**: cross-checking `lambda_gen` against the
primary source (`config_david_v6c_stage2.yml` line 27) rather than trusting
`tasks/t0009_stage2_training_failure_forensics/results/confound_table.md`'s secondary claim revealed
the confound table is **factually wrong about v6c's own row**: it lists v6c's `lambda_gen` as `1.0`,
but v6c's actual committed file states `0.2`. Since the confound table misstates a value for a run
whose config file is directly available and checkable, its separate, unhedged-looking
"`1.0 (assumed)`" claim for v3's `lambda_gen` is correspondingly less trustworthy than it first
appears -- `# unknown: default from v6c's ACTUAL committed value` was used instead of propagating
the confound table's number, per this task's Rejection Criteria (`# confirmed` and even `# inferred`
labels must trace to a real file, not to a secondary table found to already be wrong once).

**`multispeaker` agreement, independently arrived at**: this task's own checkpoint-shape forensics
(Milestone 2 Step 9, `## Multispeaker Resolution` above) landed on `multispeaker: false` via a route
entirely independent of v6c's inline comment (tensor shapes in `stage1/first_stage.pth`, not trust
in v6c's uncited claim) -- the two sources now agree, which raises confidence in `false` over
`best/config.json`'s and t0009's confound table's `true`, without promoting the label past
`inferred` (the checkpoint-shape evidence alone still lacks the `models.py`-confirmed
parameter-shape signature the Rejection Criteria requires for `confirmed`).

Comparing against t0009's recommended next-run config
(`tasks/t0009_stage2_training_failure_forensics/assets/answer/t0009-stage2-forensics-answer/full_answer.md`:
"a copy of v6c with `joint_epoch=8` (extra margin), epochs=20, per-epoch checkpoint retention, and
health-gate integration"): that recommendation is itself a *forward-looking* config for a future
run, not a historical claim about v3, so it is not "disagreed with" in the same sense as v6c -- but
it is worth noting explicitly that t0009's `joint_epoch=8` recommendation is **more conservative**
than this reconstruction's `joint_epoch=6` (carried from v6c, unconfirmed for v3 itself), consistent
with t0009's own reasoning that v6c's `joint_epoch=6` was necessary but not proven sufficient (v6c
itself eventually diverged at epoch 9). A future reproduction (t0017) choosing between v3's
`joint_epoch=6` (this reconstruction, unconfirmed) and t0009's `joint_epoch=8` (recommended, also
unconfirmed for v3 specifically) has no first-party evidence to prefer one over the other for
reproducing v3 exactly -- t0009's `joint_epoch=8` is a safety margin recommendation for future runs,
not a claim about what v3 did.

## Per-Epoch Sample Gate Scores (REQ-3, REQ-8)

Phrase texts were not recoverable from any evidence source (no `phrases.txt` manifest survived in
DVC or the VM inventory), so `text=None` was passed for every file and `duration_sanity_pass` is
`null` throughout -- honest per the data-analysis instruction, not a guessed value.

| File | is_likely_noise | duration_sanity_pass | longest_nonsilent_run_s |
| --- | --- | --- | ---: |
| audio/epoch0_phrase1.wav | False | None | 1.500 |
| audio/epoch0_phrase2.wav | False | None | 1.200 |
| audio/epoch0_phrase3.wav | False | None | 2.100 |
| audio/epoch0_phrase4.wav | False | None | 1.160 |
| audio/epoch0_phrase5.wav | False | None | 1.900 |
| audio/epoch1_phrase1.wav | False | None | 0.800 |
| audio/epoch1_phrase2.wav | False | None | 2.300 |
| audio/epoch1_phrase3.wav | False | None | 1.580 |
| audio/epoch1_phrase4.wav | False | None | 1.320 |
| audio/epoch1_phrase5.wav | False | None | 1.480 |
| audio/epoch2_phrase1.wav | False | None | 1.020 |
| audio/epoch2_phrase2.wav | False | None | 1.000 |
| audio/epoch2_phrase3.wav | False | None | 0.560 |
| audio/epoch2_phrase4.wav | False | None | 1.100 |
| audio/epoch2_phrase5.wav | False | None | 0.960 |
| audio/epoch3_phrase1.wav | False | None | 1.480 |
| audio/epoch3_phrase2.wav | False | None | 2.000 |
| audio/epoch3_phrase3.wav | False | None | 1.480 |
| audio/epoch3_phrase4.wav | False | None | 1.660 |
| audio/epoch3_phrase5.wav | False | None | 2.560 |
| audio/v3/ep6_p1.wav | True | None | 1.440 |
| audio/v3/ep6_p2.wav | True | None | 0.960 |
| audio/v3/ep6_p3.wav | True | None | 0.660 |
| audio/v3/ep6_p4.wav | True | None | 2.160 |
| audio/v3/ep6_p5.wav | True | None | 0.820 |
| audio/v3/ep7_p1.wav | True | None | 0.680 |
| audio/v3/ep7_p2.wav | True | None | 0.840 |
| audio/v3/ep7_p3.wav | True | None | 0.460 |
| audio/v3/ep7_p4.wav | True | None | 1.200 |
| audio/v3/ep7_p5.wav | True | None | 0.900 |
| audio/v3/ep8_p1.wav | True | None | 5.300 |
| audio/v3/ep8_p2.wav | True | None | 4.020 |
| audio/v3/ep8_p3.wav | True | None | 4.720 |
| audio/v3/ep8_p4.wav | True | None | 7.700 |
| audio/v3/ep8_p5.wav | True | None | 5.060 |
| audio/v3b/ep6_p1.wav | False | None | 1.420 |
| audio/v3b/ep6_p2.wav | False | None | 0.800 |
| audio/v3b/ep6_p3.wav | False | None | 0.860 |
| audio/v3b/ep6_p4.wav | False | None | 1.600 |
| audio/v3b/ep6_p5.wav | False | None | 1.400 |
| audio/v3b/ep7_p1.wav | False | None | 1.440 |
| audio/v3b/ep7_p2.wav | False | None | 0.820 |
| audio/v3b/ep7_p3.wav | False | None | 0.840 |
| audio/v3b/ep7_p4.wav | False | None | 1.620 |
| audio/v3b/ep7_p5.wav | False | None | 1.400 |
| audio/v3b/ep8_p1.wav | False | None | 1.380 |
| audio/v3b/ep8_p2.wav | False | None | 0.820 |
| audio/v3b/ep8_p3.wav | False | None | 0.860 |
| audio/v3b/ep8_p4.wav | False | None | 1.640 |
| audio/v3b/ep8_p5.wav | False | None | 1.360 |
| audio/v3b/ep9_p1.wav | False | None | 1.400 |
| audio/v3b/ep9_p2.wav | False | None | 0.820 |
| audio/v3b/ep9_p3.wav | False | None | 0.840 |
| audio/v3b/ep9_p4.wav | False | None | 1.640 |
| audio/v3b/ep9_p5.wav | False | None | 1.340 |

**Summary**: 55 files scored; 15 flagged `is_likely_noise=True`. Epoch range observed across
`epoch{0..3}_phrase{1..5}.wav` plus `v3/ep{6,7,8}` and `v3b/ep{6,7,8,9}` is epochs 0-9 (10 distinct
epoch checkpoints sampled at 5 phrases each), consistent with `task_description.md`'s note that
`v3/audio/v3b/ep9_p5.wav` implies at least 10 epochs and a `v3b` variant -- confirmed here directly
from the file listing, not inferred.

## Metrics Cross-Check (Milestone 5, applicable-metrics coverage)

**Centroid-building deviation (documented, not silent)**: this task's freshly-`dvc pull`ed
`data/11labs_david/` corpus (1364 clips) measures a mean duration of ~1.04s and a max of ~1.67s --
almost the entire corpus falls under `tasks/t0008_tts_eval_harness_baselines/code/scoring.py`'s own
`MIN_CLIP_DURATION_S = 1.6` pre-filter, so calling that module's `build_centroid()` unmodified on
this corpus raises `RuntimeError: No clips long enough to embed (skipped=1364, min_duration=1.6s)`
-- verified directly. Since `resemblyzer.VoiceEncoder.embed_utterance()` itself embeds short clips
without error (verified: a 0.84s post-VAD-trim clip embeds fine) and t0008's own completed-task file
cannot be modified (Key Rule 5), `code/compute_metrics.py`'s `build_local_centroid()` reimplements
the same GE2E-mean-embedding logic directly against `resemblyzer`'s own API with no minimum-length
pre-filter, then hands the resulting centroid to the library's own unmodified
`compute_speaker_sim()`. This centroid was built from 1364 of 1364 reference clips (0 failed to
embed for other reasons). This is itself a reproducibility finding worth flagging upstream:
re-running t0008's own `build_reference_split()` today, unmodified, against the current
`data/11labs_david/` would also fail with the same error, which is inconsistent with t0008's own
recorded success on this corpus -- either the corpus changed since t0008 ran, or
`MIN_CLIP_DURATION_S` was not actually binding in t0008's execution environment for a reason not
identified here.

This task's own local CPU re-synthesis of the shipped v3 bundle scores `speaker_sim=0.5660` (mean
over 8 clips: 3 fixed gate texts + 5 seed-42 val96 prompts, 0 skipped) against this locally-built
ElevenLabs reference centroid. t0008's own recorded numbers for the same bundle
(`tasks/t0008_tts_eval_harness_baselines/results/metrics.json`) are `speaker_sim=0.631` (fillers) /
`0.588` (val96). This run is NOT consistent with the fillers number within a ±0.02 tolerance (chosen
as well above expected floating-point/CPU-vs-original-hardware nondeterminism for a deterministic
decode), which given the centroid-building deviation documented above is only weak corroborating
evidence either way -- treat this task audio-quality conclusions (Milestones 2-3) as resting
primarily on the checkpoint-shape forensics and the audio-quality gate, not on this speaker_sim
cross-check. `rtf` (mean 1.936) and `ttfb_ms` (mean 6349.0) are measured on this workstation's CPU,
not t0008's original (possibly GPU-backed) hardware, so they are recorded for completeness and are
not expected to match t0008's numbers -- higher RTF/TTFB here is not a regression finding.
