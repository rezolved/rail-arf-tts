# Reproduce v3 Inside ARF: ISTFTNet Stage 2 on 266 Clips, Zero Variables Changed

## Motivation

v3 is the only Kokoro fine-tune this project has ever produced that (a) synthesizes clean speech,
(b) loads into the Kokoro API and hits the latency target (TTFB p50 185 ms), and (c) beats base
Kokoro on speaker similarity (0.631 vs 0.603 on fillers, t0008). It was run outside ARF, its config
was lost, and nobody has reproduced it. Every later run changed several things at once and failed.

This task changes **nothing**. It takes the recipe t0016 reconstructs and runs it inside ARF with
full instrumentation: committed config, JSONL step logs, per-epoch checkpoints, per-epoch synthesis
samples, the hardened audible-speech gate, and t0008 harness scoring. Both possible outcomes are
valuable:

* **It reproduces.** The project finally has a known-good baseline with logs, from which follow-up
  scaling runs can change one variable at a time (data 266 → ~800 → 1531, `joint_epoch` in steps
  rather than epochs, and so on).
* **It does not reproduce.** Then the recipe as reconstructed is incomplete, or the environment
  (library versions, phonemizer, StyleTTS2 commit) has drifted, and that becomes the most important
  finding of the project so far. Lesson 4 applies: pin versions before, not after.

The HiFi-GAN line (v10, v11) is explicitly **not** continued here. Those checkpoints cannot load
into `kokoro.KModel`, cannot be latency-benchmarked, and score below base Kokoro. Their tooling
(gate, inference harness, checkpoint inspector) is reused; their architecture is not.

## Hypothesis

Running the v3 recipe on the v3 data with the v3 Stage 1 checkpoint produces a checkpoint whose
Kokoro-API bundle scores within noise of the shipped v3 bundle on the t0008 harness.

## What is held fixed (do not touch)

Taken verbatim from `tasks/t0016_v3_recipe_recovery/data/config_david_v3_reconstructed.yml`:

* Decoder `type: istftnet` with v3's `upsample_rates: [10, 6]`, `gen_istft_n_fft: 20`,
  `gen_istft_hop_size: 5`. **Never HiFi-GAN.**
* `first_stage_path`: the v3 Stage 1 checkpoint, SHA-256 verified against t0016's ledger before
  launch.
* Training list: t0016's `data/v3_train_list_266.txt`. If t0016 could not recover it, use the subset
  rule t0016's suggestion prescribes, record that as the single known deviation, and say so in every
  results file.
* `multispeaker`, `joint_epoch`, `diff_epoch`, `lambda_*`, `lr`, `batch_size`, `epochs_2nd`,
  `train_LM`, `max_len`: as reconstructed. Fields t0016 marks "unknown" take the value t0016 chose
  for the reconstructed config; list them in `plan/plan.md` under a heading "Unknown fields and the
  value used".
* Patches: only the two in `train_second_patch.diff` plus the safeguards below, which are logging
  and loading fixes, not recipe changes.
* Environment: the versions t0016 captured. Where t0016 could not capture a version, pin to the
  version t0006 v6c ran under (the closest surviving relative) and record it.

## What is added (instrumentation only, no recipe change)

* DP-aware `load_checkpoint` with the parameter-count assertion (t0009 patch 7 and safeguard library
  `t0009_training_safeguards`): a zero-param or partial-shape load must abort before any GPU hour is
  spent. Run `tasks/t0014_v11_decoder_fix_retrain/code/inspect_checkpoint.py` (or the t0013
  original) as the pre-flight so the decoder-shape match is proven on paper first.
* `StepLogger` JSONL, `HealthGate`, `CheckpointManager` with per-epoch saves to
  `/mnt/cache/persist/` (Lesson 10), `capture_run_config`.
* **Per-epoch evaluation loop, on the VM, during training**: after every epoch checkpoint, run the
  five-module extraction from `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/` into a Kokoro
  bundle, synthesize the three fixed gate texts (`lining_up_suggestions_17`,
  `lining_up_suggestions_10`, `putting_them_head_to_head_15`, the same ones t0013-t0015 used) plus
  the five v3 sample phrases from t0016, and run the hardened gate
  (`tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py`: `is_likely_noise`,
  `duration_sanity_pass`, `longest_nonsilent_run_s`). Log the verdict per epoch. The first epoch
  whose bundle fails `duration_sanity_pass` on any gate text stops training, keeps the last passing
  checkpoint, and is reported as the divergence epoch. This is the check t0002 asked for ("verify
  predictor duration sanity first") and no training task has run it during training yet.
* Disk watch: `df -h /` and `df -h /mnt/cache/persist` every 15 minutes with the 85% cleanup rule
  from t0014's plan. t0010 lost its evaluation to a full disk; that must not recur.

## Runs

Exactly one training run, `v3_repro`. Run **all** planned epochs from the recipe (v3 evidence
implies 10; use t0016's value), save every epoch, then select the best checkpoint by **harness
`speaker_sim` on the 100-filler prompt set** among epochs that pass the hardened gate on all gate
texts. Do not select by val_loss alone: t0009 showed v3's 0.506 and v6c's 0.849 are not comparable
across loss configurations, and t0013 showed val_loss cannot see a broken decoder.

If the pre-flight or the first epoch's gate fails in a way that is clearly an environment error
(missing espeak, wrong torch), fix and relaunch once. A second failure is an intervention, not a
third launch.

## Evaluation

Score with the `tts_eval_harness` library from t0008, same protocol as t0008 (smoke gate, 50 warmup
requests, val96 + 100 fillers, half-A centroid, half-B ElevenLabs self-score), on these systems in
the same session:

1. `kokoro_v3_repro_best` — this task's selected bundle.
2. `kokoro_v3_bundle` — the shipped v3 bundle (re-scored, same session, so the comparison is paired,
   Lesson 1).
3. `kokoro_base_v3_voicepack` — base Kokoro with the v3 voicepack (isolates the Stage 2 gain).

Report every registered metric (`speaker_sim`, `ttfb_ms`, `rtf`) plus WER and duration ratio for
each system and prompt set, as explicit metrics variants. Also report
`efficiency_training_time_seconds`, `efficiency_inference_time_per_item_seconds`, and
`efficiency_inference_cost_per_item_usd` (machine hourly price × wall-clock / items).

## Pre-registered success criteria

Reproduction **succeeds** if all hold for `kokoro_v3_repro_best`:

1. Hardened gate passes on all eight gate/sample texts (`is_likely_noise=False`,
   `duration_sanity_pass=True`, `longest_nonsilent_run_s <= 12`).
2. Harness duration ratio explosions (ratio > 5.0) on 0 of 196 prompts.
3. `speaker_sim` on fillers ≥ 0.60 and within 0.03 of the same-session `kokoro_v3_bundle` score.
4. TTFB p50 on fillers ≤ 300 ms through the Kokoro API path.

Anything less is a **non-reproduction**, reported with the same rigor: which criterion failed, at
which epoch the gate first tripped, and the diff between the reconstructed config and the closest
thing that did work (v3 bundle forensics from t0016).

Rejection: if `successful_prompts / total_prompts < 0.8` for any system, that system's metrics are
null (Lesson 3).

## Key Questions

1. Does the reconstructed recipe reproduce v3 within the criteria above?
2. At which epoch, if any, does duration sanity first fail, and does that coincide with
   `joint_epoch` (GAN activation)?
3. Which modules change most during Stage 2 (same weight-norm delta table as t0016, computed on this
   run), and does that match v3's fingerprint?
4. How much of the final `speaker_sim` is the voicepack (system 3) versus Stage 2 (system 1 minus
   system 3)?
5. What did this run cost in GPU-hours per epoch, so that scaling to 1531 clips can be budgeted from
   measurement rather than guess?

## Expected Outputs

* `assets/model/kokoro-v3-repro-best/` — Kokoro-API bundle (five-module `.pth` + voicepack, DVC),
  the exact config used, `launch_info.json`, environment pins.
* `data/run_v3_repro/` — `metrics.jsonl`, per-epoch gate verdicts `epoch_gates.json`, config copy.
* `results/audio_samples/{v3_repro,v3_shipped}/` — the eight texts per epoch for the repro, the same
  texts for the shipped bundle (DVC).
* `results/per_clip_metrics.json`, `results/metrics.json` (variants: three systems × two prompt
  sets), `results/costs.json`, `results/remote_machines_used.json`.
* Charts in `results/images/`, embedded in `results_detailed.md`:
  `val_loss_and_dur_loss_per_epoch.png` (x: epoch, y: loss, vertical line at `joint_epoch`; answers
  Q2), `gate_verdicts_per_epoch.png` (x: epoch, y: pass/fail per signal; Q2),
  `speaker_sim_three_systems.png` (grouped bars, fillers vs val96; Q1, Q4),
  `module_weight_delta_repro_vs_v3.png` (Q3).
* Tables: per-epoch (epoch, val_loss, dur_loss, gate pass, filler speaker_sim if scored); per-system
  (speaker_sim mean ± std, TTFB p50/p95/p99, RTF, WER, explosion count) for each prompt set.
* `results/suggestions.json` — if reproduced: the first single-variable scaling run (data size); if
  not: the narrowest next experiment that separates recipe error from environment drift.

## Compute and budget

* `LLM-T1-NC80` (2×H100), needed because the v3 Stage 1 checkpoint and environment live there and
  because per-epoch bundle synthesis plus harness scoring share the session.
* Estimate: v3 data is 266 clips ≈ 31 steps/epoch; 10 epochs of Stage 2 plus per-epoch bundle
  synthesis ≈ 1.5 h; harness on three systems ≈ 0.75 h (t0008 ran eight systems in ~2.5 h); setup
  and teardown ≈ 0.5 h. **≈ 2.75 h × $13.96 ≈ $40.** Hard cap: **$70** (one relaunch allowed).
* The idle watchdog must be armed and its PID confirmed before the training command is issued
  (Lesson 8, t0010's $273 overrun).

## Dependencies

* `t0016_v3_recipe_recovery` — the reconstructed config, train list, environment pins and the Stage
  1 checkpoint hash this task must verify. Cannot start before it completes.
* `t0002_kokoro_v4_voicepack_decoder_package` — five-module extraction and voicepack scripts, and
  the KModel load test.
* `t0008_tts_eval_harness_baselines` — `tts_eval_harness` library and the shipped v3 bundle system
  definitions.
* `t0015_v11_duration_blowup_forensics` — hardened `audio_quality_check.py`.
