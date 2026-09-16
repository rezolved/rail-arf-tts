---
spec_version: "3"
task_id: "t0014_v11_decoder_fix_retrain"
step_number: 9
step_name: "implementation"
status: "completed"
started_at: "2026-09-16T16:59:32Z"
completed_at: "2026-09-16T23:05:00Z"
---
## Summary

Executed `plan/plan.md` in full across three sub-invocations (`pause_count: 0, 1, 2`): fixed the
decoder-init bug and verified it with a pre-flight tensor check before any GPU spend (Milestone A),
trained 50 epochs on `LLM-T1-NC80` with proactive disk monitoring (Milestone B), and, on this final
resume, independently re-confirmed training finished cleanly and ran the mandatory audible-speech
gate (Milestone C), which **passed** — producing the `kokoro-v11-best` model asset (Milestone D).

## Actions Taken

1. **Milestone A.** Repointed `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s
   `epochs_2nd_00020.pth` in `code/config_david_v11.yml` (leaving `ignore_modules` unchanged — the
   fix is the checkpoint choice, not the exclusion list). Ran `inspect_checkpoint.py`'s
   `classify_decoder()` pre-flight, which confirmed `hifigan`-shaped decoder tensors with 0
   missing/0 unexpected keys before any GPU spend. Confirmed 0 overlap between the new 1,531-clip
   `train_list_v11_normalized.txt` and `val_96`. Found and fixed a second, plan-unanticipated bug:
   the checkpoint's classic `weight_norm`/`spectral_norm` key naming did not match this fork's
   `parametrizations.*` naming, silently partial-loading `decoder`/`style_encoder`/`predictor`
   despite the architecture pre-flight passing; ported `_rename_legacy_parametrization_keys()` (from
   t0013's `infer_styletts2.py`) into `code/train_second_v11.py`'s `load_checkpoint()`.
2. **Milestone B.** Provisioned `LLM-T1-NC80`, reclaimed root disk from 98%/3GB free to 87%/16GB
   free before launch, launched 50-epoch/`diff_epoch=10`/`joint_epoch=30` training inside `tmux`,
   and paused twice via `heartbeat.pause_step` (watchdog-protected, explicit `resume_sentinel` each
   time) rather than holding a warm context idle across the ~3-hour GPU run, per the sanctioned
   async-wait mechanism — never a fire-and-forget background poller.
3. **Resume (this invocation).** `resume_check` returned `job_dead` (exit 3). Recognized this as a
   known false-negative of the crude `tmux has-session -t v11train` liveness probe, which closes
   naturally on clean process exit and cannot by itself distinguish "finished" from "crashed."
   Independently re-verified over SSH before treating it as a completion, not trusting the prior
   turn's summary alone: `v11_train.log` ends with a trailing `DONE` marker after "Epochs: 50",
   `metrics.jsonl` has 620 records through epoch 50 (final `val_loss=0.3505`), `grep -c gate_fired`
   returns `0`, `df -h /` is stable at 87%/16GB free, and the last actual saved checkpoint on disk
   is `epoch_00048.pth` (`val_loss=0.3468841`, `flagged_healthy: true` — only even epochs are
   saved). This matched the resume_sentinel's own pre-registered "DONE + 50 epochs + 0 gate firings
   -> proceed to Milestone C" branch, so training was correctly treated as genuinely complete, not
   re-paused or escalated as a dead job.
4. **Milestone C/D.** Spawned a fresh `/implementation` subagent (per Rule 9) with full context on
   what was already done and what remained. It ran the mandatory audible-speech gate
   (`audio_quality_check.py`'s `check_audio_quality()`) against `epoch_00048.pth`'s synthesis, which
   **passed** (`is_likely_noise=False`). It first stalled mid-run waiting on a long CPU
   diffusion-sampler synthesis without driving the wait synchronously; caught this via `SendMessage`
   and resumed it with an explicit instruction not to end its turn again with async work
   outstanding. It then completed Milestone C steps 10-11 (paired sample reuse, `speaker_sim`,
   `rtf`) and Milestone D (the `kokoro-v11-best` model asset, DVC-tracked). Independently re-ran
   `meta.asset_types.model.verificator`, `verify_task_metrics`, and `verify_task_results` myself
   rather than trusting the subagent's report alone — all passed clean (model asset: 0 errors, 2
   pre-existing warnings matching `kokoro-v10-best`'s own warning profile).

## Outputs

- `tasks/t0014_v11_decoder_fix_retrain/code/config_david_v11.yml`, `train_second_v11.py`,
  `inspect_checkpoint.py`, `infer_styletts2.py`, `audio_quality_check.py`, `score_speaker_sim.py`,
  `build_reference_concat.py`, `paths.py`.
- `tasks/t0014_v11_decoder_fix_retrain/results/checkpoint_forensics_v11.md`, `val96_leak_check.txt`,
  `audio_quality_v11.json`, `v11_gate_verdict.md`, `metrics.json`, `metrics_notes.md`,
  `speaker_sim_scores.json`, `results_summary.md`, `results_detailed.md`, `costs.json` (interim),
  `remote_machines_used.json` (interim), `load_log_epoch_00048.json`,
  `audio_samples/{ft/v11_best.wav,original/librispeech_control.wav}` (DVC-tracked).
- `tasks/t0014_v11_decoder_fix_retrain/data/run_v11/{epoch_00048.pth (DVC-tracked), checkpoints.json, metrics.jsonl}`.
- `tasks/t0014_v11_decoder_fix_retrain/assets/model/kokoro-v11-best/{details.json, description.md, files/kokoro-v11-best.pth (DVC-tracked), files/config_david_v11.yml}`
  — 6 files pushed to the `azureblob` DVC remote.

## Issues

- The `resume_check` utility's `job_dead` signal was a false negative caused by the crude
  `tmux has-session` liveness probe; resolved by independent SSH re-verification against the richer
  resume_sentinel evidence (DONE marker, epoch count, gate-fired count) before proceeding, per this
  step's own pre-registered decision tree — not a framework defect requiring an `intervention/`
  file, but worth noting for future tasks using the same probe style.
- The Milestone C/D subagent initially stopped its turn while a long-running CPU synthesis was still
  in flight without driving the wait synchronously (a near-miss of the same fire-and-forget pattern
  flagged during `setup-machines`); caught immediately via `SendMessage` before any incorrect
  terminal state was reported, and the resumed run completed correctly.
- The plan's literal `arf.scripts.verificators.verify_model_asset` module path does not exist in
  this repo (a pre-existing documentation gap — `kokoro-v10-best`'s own log shows the same issue);
  the real importable path is `meta.asset_types.model.verificator`, used successfully instead.
- One honestly-disclosed, non-blocking synthesis anomaly: v11's gate output is 73.95s for a 10-word
  sentence (vs. 2.5-4.9s for the control/v10 on the same text) — a likely duration-predictor
  calibration issue, distinct from the decoder defect this task fixes. Does not affect the
  pre-registered `is_likely_noise` pass/fail criterion; flagged in `results/v11_gate_verdict.md` as
  a follow-up candidate.
- `LLM-T1-NC80` remains live and billing; teardown (step 10) was deliberately left for the next
  step, not run here.
