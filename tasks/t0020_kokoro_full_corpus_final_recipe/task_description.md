# Scale the Reproduced v3 Recipe to the Full Clean Corpus and Publish the Final Recipe

## Motivation

The owner's decision after t0016 and t0018: the project keeps training Kokoro on the David audio we
already own. Zero-shot cloning proved speaker similarity is reachable, but the product needs dynamic
text at TTFB ≤ 300 ms, which only Kokoro delivers today. We have 1557 David clips (1531 after
t0012's cleaning) and every failure so far was a bug or a recipe change, not a data limitation. What
the project still lacks is a **written, reproducible recipe** that takes a set of clips of a given
voice and produces a Kokoro bundle that sounds like that voice.

t0017 establishes the baseline on the 266-clip scale. This task takes that baseline and changes one
variable at a time until the full clean corpus trains cleanly, then writes the recipe down as an
answer asset that a future engineer can follow for a different voice.

If t0017 did **not** reproduce v3, this task does not start; it is re-planned from t0017's finding.

## Hypothesis

The v3 recipe that works on 266 clips works on 1531 clips provided the GAN activation point is
expressed in optimizer steps rather than epochs, since one epoch on 1531 clips is about 6x more
steps than on 266 (t0006: 31 vs 194 steps/epoch) and every earlier full-corpus run activated the GAN
far earlier in step terms than v3 or v6c did.

## Runs (in order, each gated by the previous)

Held fixed in every run: ISTFTNet decoder, v3 Stage 1 checkpoint, t0017's exact config, phoneme
manifests from t0012 (`train_list_v5_normalized_clean.txt`, 1531 clips, LUFS-normalized), the t0009
safeguards, the per-epoch bundle synthesis + hardened gate + duration-sanity loop from t0017, disk
watch, watchdog.

1. **Run A, data only.** t0017's config verbatim on the full 1531-clip list. The only change is
   `train_data`. Run all planned epochs unless the per-epoch gate trips. Whatever happens, record
   the epoch and the step count at which it happens.
2. **Run B, only if A trips the gate after GAN activation.** Same as A with `joint_epoch` (and
   `diff_epoch` if used) scaled so the GAN activates at the same **step count** as in t0017, not the
   same epoch count. One variable changed from A.
3. **Run C, only if B still fails.** Same as B with `lambda_gen` lowered to the v6d value (0.05),
   which t0006 found kept the GAN stable. One variable changed from B. No further runs in this task;
   a third failure is a documented negative result and a re-plan.

Each run: save every epoch, synthesize the thirteen fixed texts per epoch (three gate texts, five v3
phrases, five val96 seed-42 prompts), gate them, log `dur_loss`, `val_loss`, per-module weight-norm
deltas. Select the best checkpoint per run by harness `speaker_sim` on fillers among gate-passing
epochs, never by `val_loss` alone.

## Evaluation

The fixed harness from t0019, same session, on: the selected checkpoint of the last run that passed,
`kokoro_v3_repro_best` from t0017, `kokoro_v3_bundle`, `kokoro_base_v3_voicepack`, and
`elevenlabs_david` self-consistency. Prompt sets: val96, 100 fillers, gate texts. Report every
registered metric (`speaker_sim`, `ttfb_ms`, `rtf`) plus WER, duration ratio, gate-failure count,
`efficiency_training_time_seconds`, `efficiency_inference_time_per_item_seconds`,
`efficiency_inference_cost_per_item_usd`.

## Pre-registered success criteria

The full-corpus checkpoint **succeeds** if all hold:

1. Hardened gate passes on all thirteen texts at the selected epoch.
2. Zero duration explosions (ratio > 5.0) across all 196 harness prompts.
3. `speaker_sim` on fillers ≥ t0017's `kokoro_v3_repro_best` + 0.02 in the same session (more data
   must buy something), and ≥ 0.65 absolute.
4. TTFB p50 ≤ 300 ms on fillers through the Kokoro API path.
5. The owner has listened to the listening guide and not vetoed it (recorded in
   `results/human_listening_verdict.md`, written by the owner or transcribed from their message).

Rejection: `successful_prompts / total_prompts < 0.8` for any system nulls that system (Lesson 3).

## Audio for human listening (mandatory)

DVC-track and index in `results/listening_guide.md`: every epoch of every run (thirteen texts), the
same texts for t0017's checkpoint, the shipped v3, base + voicepack, ElevenLabs originals, and all
harness clips per system. One row per text, one column per epoch per run, with gate verdict and
per-clip `speaker_sim` in each cell; failed epochs kept on purpose.

## The final recipe (answer asset)

`assets/answer/kokoro-voice-training-recipe/` answers: **"Given N clips of a target voice with
transcripts, what exact steps produce a Kokoro bundle that speaks in that voice, and what checks
tell you it worked?"** The full answer is a runbook, voice-agnostic, with every step pointing at the
script that implements it:

1. Data: minimum clip count and duration observed to work (from this task and t0017), phonemization
   with the t0003 pipeline and brand lexicon, the t0011/t0012 audit and LUFS normalization, the
   held-out split rule.
2. Stage 1: whether a new voice needs its own Stage 1 run (t0004 did one for v5; v3 and this task
   reused `first_stage_v3.pth`). If this task cannot answer it from evidence, mark it as the
   recipe's one open step and say what experiment closes it.
3. Stage 2: the config that worked (checked in next to the answer), the GAN activation rule in
   steps, the loader and safeguard patches, the per-epoch gate loop, the stop rule.
4. Packaging: five-module extraction plus voicepack (t0002), KModel load test.
5. Evaluation: the t0019 harness, the numbers a good result shows, and the listening guide.
6. Known failure modes and what they sound like, with links to the DVC audio from t0013, t0015,
   t0016 and this task, so the next person recognizes noise, duration blowup and GAN divergence by
   ear.

## Expected Outputs

* `assets/model/kokoro-david-full-best/` (DVC) and `assets/answer/kokoro-voice-training-recipe/`.
* `data/run_{A,B,C}/` configs, `metrics.jsonl`, `epoch_gates.json`.
* `results/metrics.json` (variants per system × prompt set), `results/per_clip_metrics.json`,
  `results/costs.json`, `results/remote_machines_used.json`, `results/human_listening_verdict.md`.
* Charts in `results/images/`, embedded in `results_detailed.md`: `losses_per_step_all_runs.png` (x:
  optimizer step, y: `val_loss` and `dur_loss`, one series per run, vertical lines at GAN
  activation; tests the hypothesis), `gate_verdicts_per_epoch_all_runs.png`,
  `speaker_sim_five_systems.png` (grouped bars, fillers vs val96),
  `module_weight_delta_per_run.png`.
* `results/suggestions.json`.

## Compute and budget

* `LLM-T1-NC80`. From t0017's measured GPU-hours per epoch (Key Question 5 there) times 5.8x more
  steps per epoch. Planning estimate before that number exists: Run A about 3 h, Run B about 3 h,
  Run C about 3 h, evaluation 1 h, setup and teardown 0.5 h. **About $105 per run actually needed;
  $150 if two runs; hard cap $200** for the whole task. Stop and write the negative result at the
  cap. Watchdog armed and PID confirmed before launch (Lesson 8).

## Dependencies

* `t0017_v3_reproduction_istftnet_266` — the baseline config, checkpoint, per-epoch gate loop, and
  the GPU-hours-per-epoch measurement.
* `t0019_harness_centroid_filter_fix` — the trustworthy harness and corrected baseline table.
* `t0012_v5_corpus_normalize_and_reaudit` — the 1531-clip clean, normalized corpus.
