---
spec_version: "3"
task_id: "t0014_v11_decoder_fix_retrain"
step_number: 5
step_name: "research-internet"
status: "completed"
started_at: "2026-09-16T15:18:04Z"
completed_at: "2026-09-16T15:29:51Z"
---
## Summary

Ran the `/research-internet` skill to resolve Key Question 1: whether a genuinely `hifigan`-shaped
pretrained StyleTTS2 first-stage checkpoint exists before committing to random-init decoder
training. Found one, and it changes the planned fix.

## Actions Taken

1. Spawned a subagent to execute `/research-internet` (per Rule 9), which ran 12 web/GitHub/Hugging
   Face searches, deep-read 11 pages, and wrote `research/research_internet.md` plus 12 search logs
   under `logs/searches/`.
2. Verified the output with
   `uv run python -m arf.scripts.verificators.verify_research_internet t0014_v11_decoder_fix_retrain`
   — PASSED, zero errors, zero warnings.
3. Checked the discovered paper ("GAN Vocoder: Multi-Resolution Discriminator Is All You Need",
   arXiv:2103.05236) against `arf.scripts.aggregators.aggregate_papers` — confirmed not already in
   the corpus (only HiFi-GAN, iSTFTNet, StyleTTS 2 are present) — and spawned an `/add-paper`
   subagent for it. That subagent runs in parallel with subsequent steps per the Paper Addition
   protocol; it is not awaited here.
4. Ran `git status` before this step's commit to confirm only this step's own output files (plus
   `step_tracker.json`) were untracked/modified — no stray parallel-agent files yet.

## Outputs

* `tasks/t0014_v11_decoder_fix_retrain/research/research_internet.md`
* `tasks/t0014_v11_decoder_fix_retrain/logs/searches/001_20260916T151830Z_web-search.json` through
  `012_20260916T152630Z_web-search.json`
* `tasks/t0014_v11_decoder_fix_retrain/logs/commands/008_*` (verificator run, auto-logged)
* `tasks/t0014_v11_decoder_fix_retrain/logs/steps/005_research-internet/step_log.md` (this file)

## Issues

No issues encountered. One paper was discovered
(`GAN Vocoder: Multi-Resolution Discriminator Is All You Need`, arXiv:2103.05236) and its
`/add-paper` subagent was spawned; that subagent's completion will be confirmed before the
`compare-literature` or `reporting` steps, per the coordinator's Paper Addition protocol — not a
blocker for this step.

## Key Finding for Downstream Steps

`yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` is a genuinely `hifigan`-shaped pretrained
checkpoint whose decoder config (`resblock_kernel_sizes: [3, 7, 11]`,
`upsample_initial_channel: 512`, `upsample_rates: [10, 5, 3, 2]`, `multispeaker: true`) matches
`config_david_v10.yml` field-for-field, directly answering Key Question 1: yes, a suitable
pretrained checkpoint exists. Additionally, upstream `train_second.py`'s own default
`ignore_modules` list also omits `"decoder"` — the bug is better framed as an architecture-mismatch
assumption violation (t0013's `first_stage_v3.pth` is istftnet-shaped while the config builds a
hifigan decoder) than a missing-exclusion bug. The recommended fix direction changes from "add
`decoder` to `ignore_modules` and accept random init" to "repoint `first_stage_path` at the LibriTTS
checkpoint." The official `Configs/config_ft.yml` recipe (50 epochs, `diff_epoch: 10`,
`joint_epoch: 30`, ~1k-sample corpora) is a much closer corpus-scale anchor for Key Question 2 than
anything found in `research_papers.md`, corroborated by two independent community fine-tunes at the
same defaults.
