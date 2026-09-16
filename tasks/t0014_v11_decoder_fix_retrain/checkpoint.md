---
spec_version: "1"
task_id: "t0014_v11_decoder_fix_retrain"
updated_at: "2026-09-16T23:14:00Z"
completed_steps: 11
next_step_number: 12
next_step_id: "results"
---
# Task Objective

Fix the ignore_modules bug that left v10's HiFi-GAN decoder worse than random, retrain on t0012's
1531-clip normalized corpus, and gate completion on an actual audible-speech check.

* * *

## Step History

### Step 1 — create-branch

Trimmed to stay within 10 KB limit.

### Step 2 — check-deps

Trimmed to stay within 10 KB limit.

### Step 3 — init-folders

Trimmed to stay within 10 KB limit.

### Step 4 — research-papers

Trimmed to stay within 10 KB limit.

### Step 5 — research-internet

Trimmed to stay within 10 KB limit.

### Step 6 — research-code

Trimmed to stay within 10 KB limit.

### Step 7 — planning

Trimmed to stay within 10 KB limit.

### Step 8 — setup-machines

Trimmed to stay within 10 KB limit.

### Step 11 — creative-thinking

Skipped: task scope is a well-defined diagnostic fix (decoder-init bug) plus corpus-expansion
retrain with an explicit audible-speech gate; the Key Questions that call for alternative approaches
(pretrained-checkpoint search, epoch-count sizing) are already covered by the research and planning
steps.

### Step 9 — implementation

Resumed from `paused_waiting`; `resume_check`'s `job_dead` was independently confirmed over SSH as a
false-negative (training actually finished cleanly: `DONE` marker, 50/50 epochs, 0 gate firings).
Milestone C's mandatory audible-speech gate **passed** (`is_likely_noise=False`, `clip_fraction`
0.0048 vs. v10's confirmed-broken 0.750-0.807) against `epoch_00048.pth`, so Milestone D's
`kokoro-v11-best` model asset was created and independently verified (0 errors). See
`results/v11_gate_verdict.md` for full evidence; `LLM-T1-NC80` is left running for `teardown`.

### Step 10 — teardown

Confirmed no job was running, then a subagent executed the Teardown Protocol on `LLM-T1-NC80`: no
further downloads needed (step 9 already pulled the verified model asset), cleaned up this task's
own `/mnt/tmp/t0014_venv` scratch, and ran `azure_ml_vm teardown`. `machine_log.json` now has
`destroyed_at: 2026-09-16T23:11:25Z`, final `total_duration_hours: 6.436`, `total_cost_usd: 89.85`
(supersedes the interim ~$86.46 estimate). `results/remote_machines_used.json`/`costs.json` updated
to match. `verify_machines_destroyed.py` — PASSED, 0 errors, 2 non-blocking warnings.

* * *

## Cross-Step Decisions

* Corpus had no paper assets or categories before this task; three papers (StyleTTS 2, HiFi-GAN,
  iSTFTNet) were added under this task's `assets/paper/` to ground Key Question 2 and 3 findings.
  `categories_consulted` is `[]` project-wide — not a fixable gap within this task's scope (Key Rule
  0).
* **Decoder-init fix direction changed by internet research.** Step 5 found that upstream
  `train_second.py`'s own default `ignore_modules` also omits `"decoder"` — t0013's bug is better
  framed as an architecture-mismatch assumption violation (istftnet-shaped `first_stage_v3.pth` vs.
  a hifigan-shaped config) than a missing-exclusion bug. The recommended fix is now to repoint
  `first_stage_path` at `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` (a real hifigan-shaped
  checkpoint), not to add `"decoder"` to `ignore_modules` and accept random init. Planning must
  resolve licensing use-terms disclosure (MIT code license; pretrained weights carry separate
  consent/disclosure terms per `research_internet.md`) and treat random-init retraining only as a
  fallback if this checkpoint path fails.
* **Epoch-count anchor updated.** If fine-tuning from the LibriTTS checkpoint, the official
  `Configs/config_ft.yml` recipe (50 epochs, `diff_epoch: 10`, `joint_epoch: 30`, ~1k-sample scale,
  corroborated by two independent community fine-tunes) is a closer corpus-scale anchor than
  `research_papers.md`'s VCTK/LibriTTS from-scratch numbers, and should supersede v10's 20/8 budget
  in planning — for the from-scratch fallback, `research_internet.md` also found 400-1,000 Stage-1
  epochs on ~150 files as order-of-magnitude confirmation the old budget was short.
* **Prior code confirmed to still carry the superseded random-init assumption.** `t0010`'s
  `eval_all_checkpoints.py` and `t0013`'s `v10_diagnosis.md` both assume the fix is "add
  `\"decoder\"` to `ignore_modules`" (random init); planning must not copy that framing forward
  as-is. `t0010`'s `train_second_v10.py` (with the `t0009` safeguards already wired in) and
  `t0013`'s
  `inspect_checkpoint.py`/`audio_quality_check.py`/`infer_styletts2.py`/`random_decoder_probe.py`
  are otherwise directly reusable (copy into task; only `tts_eval_harness`'s scoring functions, not
  its Kokoro-based synthesis adapters, are usable pre-v11).
* **Plan finalized and verified.** `plan/plan.md` is `status: "complete"`, `verify_plan.py` passes 0
  errors/0 warnings. The plan supersedes `task_description.md`'s literal `joint_epoch=8 unchanged`
  instruction with `epochs: 50, diff_epoch: 10, joint_epoch: 30`, documented explicitly as a
  resolved ambiguity per the plan spec's Task Requirement Checklist rules — implementation must use
  the plan's schedule, not the task description's literal text. Cost estimate (~$84-150) is expected
  to exceed the $100 per-task default like t0010 did; this is pre-authorized in the plan with a $300
  hard-stop escalation threshold, and project budget headroom (99.9% left) is not a constraint.
* **`LLM-T1-NC80` is live and billing.** Acquired and `ready` since 2026-09-16T16:45:14Z, watchdog
  active (3600s idle timeout, confirmed PID), `/mnt/cache/persist` verified as a real Azure Files
  mount. `implementation` must write all checkpoints under `/mnt/cache/persist/`, never bare `/mnt`.
  The `kokoro-finetune` conda/code environment on the VM needs to be re-created from scratch — its
  prior symlink target on ephemeral `/mnt/tmp` was wiped by an earlier VM stop, so
  `implementation`'s Milestone A/B setup cannot assume it still exists.
* **A subagent fire-and-forget near-miss occurred during setup-machines** (Lesson 8 pattern:
  claiming an untracked "background poller" instead of blocking synchronously or registering a
  `ScheduleWakeup`). Caught and corrected within the same step via `SendMessage` before any VM was
  billing idle. Future long-running steps on this task (especially `implementation`'s multi-hour
  training run) must use the sanctioned `paused_waiting` + `ScheduleWakeup` mechanism, never an
  agent-described background watcher, when a wait needs to outlive the current turn.
* **Step 9 (`implementation`) is `paused_waiting`, not complete.** Milestone A (REQ-1, REQ-2, REQ-3,
  REQ-5) finished and is verified: `code/config_david_v11.yml` repoints `first_stage_path` at
  `yl4579/StyleTTS2-LibriTTS`'s `epochs_2nd_00020.pth` with `epochs=50`/`diff_epoch=10`/
  `joint_epoch=30`; the mandatory pre-flight `inspect_checkpoint.py` run **passed**
  (`results/checkpoint_forensics_v11.md`: decoder classifies `hifigan`, all 14 modules finite, 0
  missing) before any GPU spend; `results/val96_leak_check.txt` confirms 0 overlap between the
  1,531-clip `data/train_list_v11_normalized.txt` and `val_96`. **A second real bug was found and
  fixed mid-flight, beyond what the plan anticipated**: the LibriTTS checkpoint's state dict uses
  the classic `torch.nn.utils.{weight_norm,spectral_norm}` key naming, but this fork's `models.py`
  builds the newer `parametrizations.*` naming, so the first training launch silently partial-loaded
  (`decoder 438/678`, `style_encoder 16/67`, `predictor 90/122` params) despite the architecture
  pre-flight passing — the pre-flight only checks decoder *shape classification*, not key-name-level
  load compatibility. Fixed by porting t0013's already-existing
  `_rename_legacy_parametrization_keys()` (from `infer_styletts2.py`) into
  `code/train_second_v11.py`'s `load_checkpoint()`; relaunched training now shows every module
  loading 100%. **Environment deviation, documented and safe**: building the training venv on
  `/mnt/cache/persist` (the CIFS-backed Azure Files mount) stalled indefinitely writing PyTorch's
  ~14k files, so the venv was built instead at `/mnt/tmp/t0014_venv/venv` (local ext4, 177GB free,
  fully regenerable) — this is **not** the root disk and does not touch the plan's Lesson-10
  persistence requirement, because all actual data (corpus, the 771MB LibriTTS checkpoint, and every
  training checkpoint/log) correctly lives under `/mnt/cache/persist/`. **Disk-space near-miss found
  and mitigated before any training spend**: root disk (`/dev/root`) was at 98%/3GB free when step 9
  started (t0010's exact stuck-teardown failure mode recurring, caused by ~68GB of unrelated
  leftover data from other tasks/projects sharing this VM pool, none of it touched) — reclaimed to
  87%/16GB free via `docker system prune -a --volumes -f` (9.8GB), `apt-get clean`/`autoremove`,
  `journalctl --vacuum-time=2d`, and disabled-snap-revision removal; independently re-verified
  stable at 87% after training launch. Training is confirmed healthy (independently re-verified, not
  just taken on the subagent's word): `tmux has-session -t v11train` on `LLM-T1-NC80` returns alive,
  log shows loss trending down with no NaNs (epoch 1, step 150/191 at last check), both H100s at
  ~73GB/96GB VRAM, idle watchdog PID confirmed still running. Step 9 was paused via
  `heartbeat.pause_step` (`resume_after: 2026-09-16T19:45:00Z`, `watchdog_active: true`,
  `liveness_probe: "ssh LLM-T1-NC80 tmux has-session -t v11train"`, `pause_count: 1`) rather than
  ridden out synchronously, because the full 50-epoch/30-joint-epoch run is estimated at ~3 hours of
  active GPU compute. Milestones B (remaining epochs), C (the mandatory audible-speech gate —
  REQ-6/REQ-7, the task's actual pass/fail criterion), and D (conditional `model` asset + DVC) are
  still open.
* **First resume check (2026-09-16T19:47Z): still training, re-paused (`pause_count: 2`).** Ran
  `resume_check` (`decision: job_alive`) then independently re-verified over SSH: epoch 27/50, step
  70/191, tmux `v11train` alive, `grep -c gate_fired metrics.jsonl` returned 0, `df -h /` stable at
  87%/16GB free (unchanged from pause time), both H100s at 11-18% util / ~73-74GB VRAM, loss curves
  flat with no NaNs. `checkpoints.json`'s last entry (epoch 24) has `flagged_healthy: true`.
  Observed pace epochs 1-26: ~4.6 min/epoch (training started 2026-09-16T17:45:20Z);
  `joint_epoch: 30` is 3 epochs away and may slow the remaining ~23 epochs (discriminator losses are
  still 0.0, confirming joint phase has not started yet). Re-paused with
  `resume_after: 2026-09-16T22:30:00Z` (~2h45m buffer over the naive linear-pace estimate of ~1h50m,
  to absorb joint-phase slowdown) and an updated `resume_sentinel` recording this checkpoint's
  readings. `jq` is not installed on the VM — use `python3 -c "import json; ..."` against
  `checkpoints.json` instead on future checks.
* **`LLM-T1-NC80` fully torn down at teardown (step 10).** Final measured cost is $89.85 over 6.436
  hours (`created_at` 16:45:14Z → `destroyed_at` 23:11:25Z) — supersedes every earlier interim
  figure in the checkpoint history above. This is the authoritative total for `results/costs.json`
  and any budget-reporting step downstream; do not recompute from the old ~$86.46 interim number.

* * *

## Next Step Notes

Step 10 (`teardown`) is now **`completed`**. `LLM-T1-NC80` is destroyed (`deallocated: true`, no
competing lock), and `results/costs.json` / `results/remote_machines_used.json` carry the final
measured total: **6.436 hours, $89.85** (not the earlier ~6.19h/$86.46 interim figure). Step 11
(`creative-thinking`) is already `skipped`. Step 12 (`results`) runs next: write
`results/results_summary.md`, `results/results_detailed.md`, `results/metrics.json` (cross-check
every number against the gate verdict and training metrics), and confirm `results/costs.json` /
`results/remote_machines_used.json` (already final, no further edits needed there). `task.json`'s
`expected_assets: {"model": 1}` is satisfied by the already-verified `kokoro-v11-best` asset (see
step 9). Also worth carrying forward to `results` or `suggestions`: `results/v11_gate_verdict.md`
flags a non-blocking anomaly (73.95s synthesis duration for a 10-word sentence, vs. 2.5-4.9s for the
control/v10 — a likely duration-predictor calibration issue); it did not block this task's gate
criterion but is worth surfacing as a follow-up-task suggestion. `code/config_david_v11.yml`,
`code/train_second_v11.py`, and all Milestone C code already exist in `code/` and do not need to be
recreated by any later step.
