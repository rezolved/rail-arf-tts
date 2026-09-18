---
spec_version: "2"
task_id: "t0016_v3_recipe_recovery"
date_completed: "2026-09-17"
status: "complete"
---
## Objective

Reconstruct the exact Kokoro-82M "v3" Stage 2 StyleTTS2 fine-tuning recipe from surviving artifacts,
because v3 is the only Kokoro fine-tune in this project's 15-task, ~$840-spend history that shipped
clean, production-usable audio (`speaker_sim=0.631` on fillers, `0.588` on val96, TTFB p50 185 ms
per t0008), and its original author is unavailable to ask directly. The recipe was never written
down: t0009 opened suggestion S-0009-04 ("Recover v3 training configuration") on 2026-09-14, and
this task (source_suggestion `S-0009-04`) closes it. "Done" means: every Stage 2 config field
(`joint_epoch`, `diff_epoch`, `lambda_gen`, `lambda_slm`, `lr`, `batch_size`, `epochs_2nd`,
`multispeaker`, `train_LM`, `max_len`, `first_stage_path`) is labelled `confirmed` (backed by a file
path or SHA-256 hash), `inferred` (derived from checkpoint/sample/log evidence), or `unknown` (no
evidence, default carried from the closest template); the standing three-way `multispeaker`
contradiction between `best/config.json` (`true`), t0009's confound table (`true`, assumed), and
t0006's `config_david_v6c_stage2.yml` (`false`, inline-commented) has been attacked with
checkpoint-shape forensics rather than resolved by picking whichever source "sounds more
authoritative"; a full human-listenable audio set (shipped v3 bundle, recovered per-epoch samples,
matching ElevenLabs reference clips) exists so a human can verify "what good sounds like"; and one
answer asset (`v3-recipe`) states the recovered recipe, its confidence per field, and what a future
reproduction (t0017, out of scope here) must hold fixed. A reconstruction that ends up labelling
most fields "inferred" or "unknown" is still a valid, useful result as long as every label is honest
about its evidence basis — no field may be upgraded to "confirmed" without a file path or hash.

## Task Requirement Checklist

Operative task text (`task.json` `short_description` plus the resolved `task_description.md`, quoted
verbatim where it matters):

> Reconstruct the exact Kokoro v3 Stage 2 recipe (config, data list, patches, epochs, environment)
> from VM files, DVC artifacts and checkpoint forensics, without access to the original author.
> 
> The question this task answers: **what exactly did v3 do, and which parts of that are confirmed
> versus inferred?**

Concrete requirements, decomposed into stable IDs used throughout this plan and reused in
`results/results_detailed.md` (written by the orchestrator from this plan's outputs):

* **REQ-1 — Bounded VM inspection (Scope §1).** Start `LLM-T1-NC80` (the project's only GPU pool
  entry, `project/azure_vm.json`, $13.96/h), read-only inventory of `~/kokoro-finetune/` (and
  `/mnt/cache/persist/` if it resolves via `readlink -f`), copy every config/log/list/small text
  artifact into `tasks/t0016_v3_recipe_recovery/data/vm_inventory/`, hash everything with SHA-256,
  stop the VM. Hard cap: 90 minutes of VM wall time (~$21), enforced by an explicit stop regardless
  of how much was found. If the home directory is gone, record that finding and proceed on sources
  2-4 only. Satisfied by Milestone 1 (Steps 3-7). Evidence: `data/vm_inventory/inventory.json` plus
  the copied files, and a VM session-time entry in the step log showing an explicit teardown call.
* **REQ-2 — Checkpoint forensics (Scope §2, local CPU).** Module-by-module diff of
  `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth` against
  `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/best/david_v3_best_decoder_kokoro.pth`
  across all five bundle modules (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`):
  parameter count, changed-vs-Stage-1 (yes/no + relative weight-norm delta), and whether
  `first_stage.pth`'s keys carry a DataParallel `module.` prefix. Satisfied by Milestone 2 (Steps
  8-10). Evidence: `results/v3_checkpoint_forensics.md` module table and
  `results/v3_module_weight_delta.raw.json`.
* **REQ-3 — Per-epoch sample gate scoring (Scope §2/§3).** Run the hardened audio-quality gate
  (`is_likely_noise`, `duration_sanity_pass`, `longest_nonsilent_run_s`) on every
  `epoch{0..3}_phrase{1..5}.wav` and on the `v3/`/`v3b/` folder contents, fixing epoch count, sample
  phrases, and checkpoint save cadence. Satisfied by Milestone 3 (Step 11). Evidence: the per-epoch
  table in `results/v3_checkpoint_forensics.md`.
* **REQ-4 — 266-clip training list recovery (Scope §1/§3, Key Question 7).** Recover
  `data/v3_train_list_266.txt` from the VM, or write `data/v3_train_list_UNRECOVERED.md` documenting
  exactly what was tried; if recovered, check for overlap against the 96-clip held-out set at
  `data/v4/val_list.txt` and flag loudly if any overlap exists. Satisfied by Milestone 1 (Step 6, VM
  search) and Milestone 4 (Step 15, overlap check). Evidence: the list file or the UNRECOVERED.md,
  plus an explicit overlap count (0 or N>0) recorded in `results/v3_checkpoint_forensics.md`.
* **REQ-5 — Reconstructed config (Scope §3).** Write `data/config_david_v3_reconstructed.yml` in the
  exact schema of `tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml`, annotating
  every field `# confirmed: <source>` / `# inferred: <reasoning>` / `# unknown: default from v6c`;
  diff it against v6c and against t0009's recommended-next-run config (in
  `tasks/t0009_stage2_training_failure_forensics/assets/answer/t0009-stage2-forensics-answer/full_answer.md`,
  "a copy of v6c with `joint_epoch=8`, epochs=20, per-epoch checkpoint retention, and health-gate
  integration"); list every disagreement. Satisfied by Milestone 4 (Steps 14-15). Evidence: the YAML
  file plus a "Config Diff vs v6c and t0009" section in `results/v3_checkpoint_forensics.md`.
* **REQ-6 — Resolve the `multispeaker` contradiction (Key Question 2).** `best/config.json` says
  `true`; t0009's `results/confound_table.md` says `true (assumed)`; `t0006`'s
  `code/config_david_v6c_stage2.yml` line 76 says `false` (inline: "v3 Stage 1 checkpoint
  (multispeaker:false)"). None of the three carries a SHA-256. Satisfied by Milestone 1 (Step 5, VM
  search for a surviving launch YAML or `models.py`/`build_model()` source) and Milestone 2 (Step 9,
  checkpoint-shape forensics as the primary evidence source). If both are inconclusive, the
  reconstructed config must label `multispeaker` `unknown (contradictory sources, unresolved)`
  rather than arbitrarily preferring one source. Evidence: a dedicated "multispeaker resolution"
  subsection in `results/v3_checkpoint_forensics.md`.
* **REQ-7 — Which modules actually trained (Key Question 3).** Determine whether the decoder changed
  at all in Stage 2, or whether v3's 0.03 speaker_sim gain over "base Kokoro + v3 voicepack" (t0008:
  0.631 vs. 0.603) comes from `predictor`/`text_encoder` alone. Satisfied by REQ-2's module diff
  plus the `v3_module_weight_delta.png` chart (Milestone 5, Step 17).
* **REQ-8 — Epoch count, best-epoch selection, and val_loss trajectory (Key Question 4).** t0006
  quotes v3 at val=0.569 (epoch 1), 0.549 (epoch 4), 0.506 (best). Satisfied by REQ-1 (VM
  logs/metrics files if they survive) and REQ-3 (sample cadence). Evidence recorded in
  `results/v3_checkpoint_forensics.md`; if VM logs do not survive, epoch count and selection
  criterion are labelled `inferred` from sample cadence, not `confirmed`.
* **REQ-9 — Environment/library versions (Key Question 5).** Capture torch, kokoro,
  phonemizer/espeak, misaki, and StyleTTS2 commit versions from the VM if they survive
  (`pip freeze`/`conda list`/`git log`/`git remote -v` inside `~/kokoro-finetune`). Satisfied by
  Milestone 1 (Step 4). Evidence: files under `data/vm_inventory/env/`.
* **REQ-10 — Byte-identity of Stage 1 checkpoints (Key Question 6).** Is `stage1/first_stage.pth`
  (DVC) byte-identical to `first_stage_v3.pth` on the VM, and to the file v6c/v6d loaded? Satisfied
  by Milestone 1 (Step 5, remote `sha256sum` without copying the large file) cross-referenced with
  Milestone 2 (Step 8, local hash). Evidence: hash comparison table in
  `results/v3_checkpoint_forensics.md`.
* **REQ-11 — Mandatory human-listenable audio (Scope §4).** Produce, DVC-track, and index:
  `results/audio_samples/v3_shipped/` (shipped bundle via `kokoro.KModel` synthesizing the 3 fixed
  gate texts, 5 v3 sample phrases, 5 seed-42 val96 prompts), `results/audio_samples/v3_per_epoch/`
  (copied, not re-synthesized, per-epoch/`v3`/`v3b` samples),
  `results/audio_samples/elevenlabs_reference/` (matching original ElevenLabs David clips), and
  `results/listening_guide.md` (one table: text, ElevenLabs original, v3 shipped, per-epoch samples,
  gate verdict, listening note; clickable relative links). `dvc add` + `dvc push` before the PR.
  Satisfied by Milestone 3 (Steps 11-13).
* **REQ-12 — Answer asset (Scope §5).** One answer asset `v3-recipe` under
  `assets/answer/v3-recipe/` with a short answer (recipe in five sentences) and a full answer
  (evidence table, confirmed/inferred/unknown ledger, environment pins, packaging-recipe pointer to
  t0002, and reproduction-fixed-variables guidance). Satisfied by Milestone 6 (Step 19). This is the
  task's single `expected_assets.answer: 1` entry in `task.json`.
* **REQ-13 — Weight-delta chart (Outputs).** One chart, `results/images/v3_module_weight_delta.png`
  (x: module, y: relative weight-norm change Stage 1 to best). Satisfied by Milestone 5 (Step 17).
* **REQ-14 — `results/suggestions.json` content (Outputs).** At minimum: a suggestion for how t0017
  should choose its training subset if the 266-clip list is unrecoverable (REQ-4), and a two-arm
  `multispeaker` ablation suggestion if REQ-6 stays ambiguous. **Ambiguity note**: writing
  `results/suggestions.json` itself is an orchestrator-managed step outside this plan's Step by Step
  (per `arf/specifications/plan_specification.md`, which explicitly excludes `suggestions.json`
  generation from the Step by Step section). This plan's steps produce the two underlying findings
  (REQ-4's recovery/non-recovery outcome, REQ-6's resolution/non-resolution outcome) that the
  orchestrator's suggestion-generation step will consume.
* **REQ-15 — `results/results_summary.md` / `results/results_detailed.md` (Outputs).** Also
  orchestrator-managed steps outside this plan's Step by Step. This plan's steps produce all the
  underlying tables (REQ-2, REQ-3, REQ-5), the chart (REQ-13), the metrics cross-check (Step 18),
  and the answer asset (REQ-12) those documents will summarize.
* **REQ-16 — Rejection criteria (Rejection criteria section of task text).** No field in
  `data/config_david_v3_reconstructed.yml` may be labelled "confirmed" without a file path or
  SHA-256 hash; an all-"inferred" reconstruction is still a valid result if honestly labelled. The
  90-minute VM cap (REQ-1) is hard: stop the VM at 90 minutes regardless of progress and report what
  was covered. See also `## Rejection Criteria` below.
* **REQ-17 — Budget cap (Compute and budget section of task text).** VM: `LLM-T1-NC80` only, ≤1.5 h
  at $13.96/h ≈ $21. Everything else is local CPU. Total task cap: $30. No training, no GPU compute
  beyond the VM being powered on for inventory retrieval. See `## Cost Estimation`.

**Ambiguity called out explicitly**: the task text names `results/v3_checkpoint_forensics.md` as the
home for two distinct tables (module diff, per-epoch samples) plus, per this plan's reading, the
config-diff and multispeaker-resolution write-ups (REQ-5, REQ-6) that Scope §3 requires but does not
name a specific output file for. This plan places all of REQ-2, REQ-3, REQ-5's diff list, REQ-6's
resolution, REQ-7, REQ-8, REQ-10 into the single `results/v3_checkpoint_forensics.md` document as
separate `##` subsections, since the task text's Outputs list does not create a second forensics
file and duplicating content across files would violate CLAUDE.md's "no duplication across task
folders" rule.

## Approach

**Grounding from research** (`tasks/t0016_v3_recipe_recovery/research/research_code.md`, verified 0
errors/0 warnings): this project has built three generations of the same no-GPU checkpoint forensics
idea (t0013 → t0015) and three generations of the same audio-noise gate (t0013 → t0014 → t0015),
each fixing a blind spot in the one before it — reusing the latest version of each (t0015's
`code/predictor_tensor_forensics.py` and `code/audio_quality_check.py`) rather than re-deriving
either is both cheaper and more correct than writing new forensics code from scratch. The
`multispeaker` contradiction is real and none of the three contradicting sources carries a hash —
`best/config.json` (shipped bundle) says `true`, t0009's confound table says `true (assumed)`, and
`config_david_v6c_stage2.yml` line 76 says `false` inline. Checkpoint-shape forensics on
`stage1/first_stage.pth` is the only evidence source that does not depend on trusting one of these
three secondary artifacts over another, so this plan schedules that forensic step explicitly (REQ-6,
Milestone 2 Step 9) rather than defaulting to whichever source "looks more authoritative" — and if
that forensic check is itself inconclusive (e.g. the `multispeaker` flag turns out to only affect
components outside the five bundled modules, such as the discriminator), the plan requires an
explicit VM-side check for the upstream StyleTTS2 `models.py`/`build_model()` source (Milestone 1
Step 5) as the tie-breaker, and if neither resolves it, the reconstructed config must say so plainly
(`unknown (contradictory sources, unresolved)`) rather than silently picking a side.

`val_loss` is not a reliable cross-run quality signal in this project (t0009: v3's loss config is
"not directly comparable" to v6c's; t0003: Stage-1 val_loss inversely tracked Stage-2 outcome in one
case) — so this plan treats v3's quoted 0.569/0.549/0.506 trajectory only as ordering evidence for
epoch count/selection (REQ-8), not as proof of the reconstructed config's correctness, and leans on
audible/gate evidence (REQ-3, REQ-11) as the load-bearing signal, consistent with the project's own
Lessons Learned that two of three prior audio-quality blind spots were only caught by human
listening, not by loss curves.

**DVC is confirmed empirically flaky in this environment**, not merely a theoretical risk: a live
test during planning
(`uv run dvc pull tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/stage1/first_stage.pth`)
failed with `DefaultAzureCredential failed to retrieve a token` (`EnvironmentCredential`
unconfigured, `WorkloadIdentityCredential` unconfigured, `ManagedIdentityCredential`: "SSO failure,
to mitigated it please try to click Jupyter/JupyterLab"). Cross-checking prior tasks' own command
logs confirms this is a known, recurring, and *transient* failure: t0015's log
(`tasks/t0015_v11_duration_blowup_forensics/logs/commands/010_20260917T082107Z_*.json`) shows the
identical error for `dvc pull tasks/t0014_v11_decoder_fix_retrain/data/run_v11/epoch_00048.pth` at
08:21:07 and again at 08:22:55, then a clean exit 0 for the same command at 08:30:30 — i.e. retrying
after ~9 minutes resolved it without any credential change. This plan's data-pull step (Step 1) is
written to retry with backoff rather than treat a single failed `dvc pull` as a hard blocker.

**Alternatives considered**:

* *Re-derive checkpoint forensics from scratch* instead of copying t0015's
  `predictor_tensor_forensics.py`. Rejected: this project has already iterated the forensics pattern
  three times (t0013→t0015), each iteration fixing a real blind spot found the hard way; writing a
  fourth version from zero would silently drop those fixes (e.g. `module.` DP-prefix stripping,
  finite-value checks) and cost more implementation time for no benefit. The only real adaptation
  needed is generalizing `TARGET_MODULES` from 2 to the 5 v3-bundle modules and making the loader
  tolerate two different top-level checkpoint shapes (see Step 8).
* *Skip the VM entirely and rely only on local checkpoint/DVC forensics.* Rejected: the task
  description ranks the VM as evidence-priority #1 specifically because it is the only source that
  could contain a literal surviving `config.yml` (StyleTTS2's `train_second.py` copies its launch
  config into `log_dir/config.yml`) or the 266-clip list — no amount of checkpoint-shape forensics
  can recover a training list. Skipping the VM would leave REQ-4 and parts of REQ-8/REQ-9
  unaddressed for no cost savings large enough to justify the coverage loss, given the VM step is
  hard-capped at 90 minutes (~$21) regardless.
* *Trust `best/config.json`'s `multispeaker: true` at face value* since it is the config of the
  actually-shipped, actually-working bundle. Rejected per research finding 5: `best/config.json` is
  a packaged Kokoro-API bundle config, one inference step removed from v3's actual Stage 2 launch
  YAML (which no longer exists locally); it is a secondary artifact exactly like t0006's contrary
  claim, not a checkpoint-level proof.

**Task types**: `task.json` already declares `["answer-question", "data-analysis"]`, which matches.
Per `meta/task_types/answer-question/instruction.md`: this plan treats the recipe recovery as the
core deliverable (not background), creates exactly one answer asset (`v3-recipe`), states the
evidence channels used (code/checkpoint experiments — VM inspection and torch-level forensics — no
papers, no internet sources), and states conflicting evidence (the `multispeaker` contradiction)
explicitly rather than forcing false certainty, per that instruction file's explicit guidance. Per
`meta/task_types/data-analysis/instruction.md`: this plan produces at least one chart
(`v3_module_weight_delta.png`) under `results/images/`, uses only registered metric keys in
`results/metrics.json`, and saves intermediate data (`v3_module_weight_delta.raw.json`,
`data/vm_inventory/inventory.json`) rather than only final tables, for research traceability.

## Cost Estimation

Task hard cap: **$30 total**. Breakdown:

* **VM compute (LLM-T1-NC80, the project's only Azure ML pool entry, $13.96/h):** capped at 90
  minutes = 1.5 h × $13.96/h ≈ **$21.00**. This covers acquire/boot (~5-10 min per the
  `setup-remote-machine` skill's own estimate for cold start + SSH setup), the read-only inventory
  scan and copy (~50-60 min budgeted, including remote `sha256sum` of `first_stage_v3.pth` without
  copying the file), environment/version capture (~5 min), and teardown (~5 min). No training, no
  inference, no GPU compute is run on the VM — it is powered on purely for filesystem/SSH access. If
  the VM home directory is found gone or the inventory completes early, the VM is stopped
  immediately rather than held for the full 90 minutes, so $21 is a ceiling, not a target.
* **Local CPU compute (this workstation/worktree):** **$0** marginal — checkpoint forensics
  (`torch.load` on CPU), audio-quality gating, matplotlib chart generation, and CPU-only
  `kokoro.KModel` synthesis (t0008 already established v3-bundle synthesis works cleanly on CPU) do
  not call any paid API and run on already-provisioned infrastructure.
* **API calls:** **$0**. No `anthropic_api`, `openai_api`, or `elevenlabs_api` calls are needed —
  the ElevenLabs reference clips are pre-recorded assets already in `data/11labs_david/` (DVC), not
  fresh ElevenLabs API synthesis.
* **DVC egress:** effectively **$0** (Azure Blob egress within the same subscription is not billed
  to this task's budget line) but is a *time* risk, not a cost risk — see Risks & Fallbacks.
* **Total estimated: ≈ $21.00**, against the task's own $30 hard cap (30% margin) and $21 VM sub-cap
  (used in full only in the worst case where the VM inventory scan runs the entire budgeted time).
  Against the project-wide budget (`project/budget.json`: `total_budget=$5000`, spent so far $397.73
  = 7.95%, no warn/stop threshold reached), this task's estimated $21 spend is immaterial and does
  not require escalation.

## Step by Step

**Milestone 1: Bounded VM inspection [CRITICAL].** This is the task's evidence-priority-#1 source
(task_description.md Scope §1) and the only source that can recover the 266-clip list or a literal
surviving launch config; it defines the task's core "read the primary evidence before inferring"
identity and is capped at 90 minutes of VM wall time.

1. **[CRITICAL] Pull local DVC reference data first (before touching the VM), with retry.** Run
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0016_v3_recipe_recovery -- uv run dvc pull tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3`
   to materialize `stage1/first_stage.pth`, `best/david_v3_best_decoder_kokoro.pth`,
   `best/david_v3_best_voicepack.pt`, `best/david_v3_voicepack_ft_enc.pt`, and the 20
   `audio/epoch{0..3}_phrase{1..5}.wav` files plus `audio/v3/` and `audio/v3b/`. Also pull
   `tasks/t0008_tts_eval_harness_baselines/data/11labs_david` (ElevenLabs reference clips) and
   confirm `data/v4/val_list.txt` (96 lines, already present in this worktree, not DVC-pointed) is
   readable. If any `dvc pull` fails with `DefaultAzureCredential failed to retrieve a token`, wait
   2 minutes and retry, up to 4 attempts (~10 minutes total) — this exact failure mode was observed
   to resolve itself after ~9 minutes in a prior task's own command log
   (`tasks/t0015_v11_duration_blowup_forensics/logs/commands/010_20260917T082107Z_*.json`), so a
   single failed attempt is not a hard blocker. Expected output:
   `find tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3 -name '*.pth' -o -name '*.wav'` lists
   real (non-`.dvc`-pointer) files with nonzero size. Satisfies: infrastructure precondition for
   REQ-2, REQ-3, REQ-11.
2. **Copy reusable code into this task's `code/` directory.** Copy
   `tasks/t0015_v11_duration_blowup_forensics/code/predictor_tensor_forensics.py` to
   `code/checkpoint_forensics.py`,
   `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` to
   `code/audio_quality_check.py` (unmodified — do not fold
   `duration_sanity_pass`/`longest_nonsilent_run_s` into `is_likely_noise`, per that module's own
   documented backward-compatibility contract), and
   `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py` to
   `code/extract_decoder_reference.py` (read-only reference for the five-module/key-remapping
   contract, not re-run). Create `code/paths.py` following the pattern in
   `tasks/t0015_v11_duration_blowup_forensics/code/paths.py`
   (`TASK_ROOT = Path(__file__).parent. parent`, `REPO_ROOT = TASK_ROOT.parent.parent`) with
   constants for: `V3_REF_DIR` (`tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3`),
   `V3_STAGE1_CKPT` (`V3_REF_DIR/"stage1"/"first_stage.pth"`), `V3_BEST_DECODER_CKPT`
   (`V3_REF_DIR/"best"/"david_v3_best_decoder_kokoro.pth"`), `V3_BEST_VOICEPACK`, `V3_BEST_CONFIG`
   (`V3_REF_DIR/"best"/"config.json"`), `V3_AUDIO_DIR` (`V3_REF_DIR/"audio"`),
   `ELEVENLABS_DAVID_DIR` (`tasks/t0008_tts_eval_harness_baselines/data/11labs_david`), `VAL96_LIST`
   (`data/v4/val_list.txt`), `VM_INVENTORY_DIR` (`data/vm_inventory`), `RESULTS_DIR`,
   `RESULTS_IMAGES_DIR`. Expected output:
   `python -c "from tasks.t0016_v3_recipe_recovery.code.paths import V3_STAGE1_CKPT; print(V3_STAGE1_CKPT.exists())"`
   prints `True`. Satisfies: infrastructure precondition for all later steps.
3. **[CRITICAL] Acquire `LLM-T1-NC80` via the `setup-remote-machine` skill.** Run
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0016_v3_recipe_recovery -- uv run python -m arf.scripts.utils.azure_ml_vm acquire t0016_v3_recipe_recovery`.
   Start a wall-clock timer at this point — the 90-minute cap (REQ-1, REQ-17) starts here. Expected
   output: JSON with `status: "ready"` (or equivalent) and a refreshed `HostName` for the
   `LLM-T1-NC80` SSH alias. Satisfies: REQ-1 (VM provisioning).
4. **[CRITICAL] Read-only inventory of `~/kokoro-finetune/` over SSH.** Wrap every command in
   `run_with_logs.py`. Run, in order, over `ssh LLM-T1-NC80`:
   * `ls -la ~/kokoro-finetune/` and `find ~/kokoro-finetune -maxdepth 3 -type d` to map the tree.
   * `find ~/kokoro-finetune/logs -iname 'config.yml' -o -iname '*.log' -o -iname 'metrics*.jsonl' 2>/dev/null`
     — StyleTTS2's `train_second.py` copies its launch config into `log_dir/config.yml`, so any
     surviving `logs/kokoro-david-v3*/config.yml` is a **confirmed** source for every field in it.
   * `find ~/kokoro-finetune/Models -iname '*v3*' -maxdepth 2 2>/dev/null` to locate any surviving
     `Models/*v3*/` checkpoint-adjacent directory (do not copy `.pth` files themselves — too large
     and already in DVC or out of scope; copy only text/config siblings).
   * `find ~/kokoro-finetune/data -iname '*v3*' -o -iname '*266*' 2>/dev/null` to search for the
     266-clip training list (REQ-4).
   * `cat ~/.bash_history` and `cat ~/.python_history` (small text files; copy directly).
   * If `~/kokoro-finetune` is a git checkout: `git -C ~/kokoro-finetune log --oneline -20` and
     `git -C ~/kokoro-finetune reflog` and `git -C ~/kokoro-finetune remote -v`.
   * Resolve `/mnt/cache/persist` with `readlink -f /mnt/cache/persist 2>/dev/null` (Lesson 10:
     `/mnt` is ephemeral, only a `/mnt/cache/persist → Azure Files` symlink survives stop/start) and
     `find` under the resolved target the same way as above if it exists.
   * Search specifically for the StyleTTS2 model-building source
     (`find ~/kokoro-finetune -iname 'models.py' 2>/dev/null`) — this is the tie-breaker source for
     REQ-6 if checkpoint-shape forensics (Step 9) is inconclusive: reading `build_model()`'s
     `multispeaker` branch shows exactly which named parameters/shapes depend on the flag. If the
     home directory does not exist at all (`ls: cannot access '/home/azureuser/kokoro-finetune'` or
     equivalent), stop searching immediately, record `"home_directory_present": false` in
     `data/vm_inventory/inventory.json`, and proceed straight to Step 6 (teardown) — do not spend
     remaining VM time retrying. Expected output: a directory listing and a set of file paths found
     (possibly empty). Satisfies: REQ-1, REQ-4 (list search), REQ-6 (models.py search), REQ-8 (log
     search).
5. **Capture environment/version pins and settle byte-identity (REQ-9, REQ-10).** Over the same SSH
   session: `pip freeze` (or `conda list` if `~/kokoro-finetune` uses a conda env — check
   `which python` first) redirected to a local copy;
   `python -c "import torch; print(torch.__version__)"`;
   `python -c "import kokoro; print(kokoro.__version__)"` if importable; `espeak-ng --version` or
   `espeak --version`; `pip show misaki 2>/dev/null`;
   `git -C ~/kokoro-finetune describe --always --dirty 2>/dev/null` for a StyleTTS2 commit pin if
   the checkout is a StyleTTS2 fork. Then, without copying the file, run
   `sha256sum ~/kokoro-finetune/first_stage_v3.pth 2>/dev/null` (and the same for any other
   `first_stage*.pth` found in Step 4) to get its hash for later comparison against the local
   `stage1/first_stage.pth`'s hash (computed in Step 8) — this settles Key Question 6 without
   transferring a large file. If `first_stage_v3.pth` is not found at that exact path, search
   `find ~/kokoro-finetune -iname 'first_stage_v3.pth' -o -iname 'first_stage.pth' 2>/dev/null`
   first. Expected output: version strings and (if the file exists) a 64-hex-char SHA-256 string.
   Satisfies: REQ-9, REQ-10.
6. **Copy every found text artifact locally and build the inventory manifest.** `scp` (or `ssh cat`
   redirected) every config/log/list/small text file found in Steps 4-5 into `data/vm_inventory/`,
   preserving relative subpaths (e.g. `data/vm_inventory/logs/kokoro-david-v3/config.yml`). Do not
   copy any `.pth` file. Write `code/build_vm_inventory.py` (adapt the `RunEntry`-style dataclass
   pattern from `tasks/t0009_stage2_training_failure_forensics/code/build_inventory.py`, trimmed to
   a single `InventoryEntry(path: str, size_bytes: int, mtime_iso: str, sha256: str)` dataclass)
   that walks `data/vm_inventory/`, computes local SHA-256 for every copied file with a standalone
   `sha256_file(path: Path) -> str` helper extracted from
   `tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py`'s `_sha256()` static
   method (copy just that hashing logic, not the full `CheckpointManager` class, which is
   training-loop-oriented), and writes `data/vm_inventory/inventory.json` as a JSON array of
   `InventoryEntry` records plus a top-level `"home_directory_present": true/false` and
   `"first_stage_v3_remote_sha256": "<hash or null>"` field. Expected output:
   `python -m json.tool data/vm_inventory/inventory.json` parses cleanly and lists every copied file
   with a nonempty `sha256`. Satisfies: REQ-1.
7. **[CRITICAL] Stop the VM — hard cap enforcement.** Regardless of how much of Steps 4-6 completed,
   check the wall-clock timer from Step 3. If 90 minutes have elapsed or all planned inventory steps
   are done (whichever comes first), immediately run
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0016_v3_recipe_recovery -- uv run python -m arf.scripts.utils.azure_ml_vm teardown t0016_v3_recipe_recovery`
   (no `--keep-running` flag). Expected output: JSON with `deallocated: true` (unless another task
   holds a lock — in that case the lock release still happened and the shared VM's own idle watchdog
   covers the rest). Record the actual VM session duration in `data/vm_inventory/inventory.json`'s
   top level as `"vm_session_minutes": <float>`. Satisfies: REQ-1, REQ-16, REQ-17. This step is not
   optional and not skippable even if Step 4 found nothing — a stuck acquire without a matching
   teardown is exactly the failure mode Lesson 8 and t0010's $272.78 overrun warn against.

**Milestone 2: Checkpoint forensics (local CPU, no VM).** Independently verifiable: the module table
and multispeaker-resolution subsection exist in `results/v3_checkpoint_forensics.md` with every cell
populated (`weight_norm`, `finite`, `dp_prefix_present`) or explicitly `N/A` with a stated reason.

8. **[CRITICAL] Generalize the checkpoint-diff tool to all five v3 bundle modules.** Edit
   `code/checkpoint_forensics.py` (copied in Step 2 from `predictor_tensor_forensics.py`): change
   `TARGET_MODULES` from `("predictor", "predictor_encoder")` to
   `("bert", "bert_encoder", "predictor", "text_encoder", "decoder")`. Rewrite
   `load_module_state_dicts()` to tolerate two different top-level checkpoint shapes, confirmed by
   direct inspection of
   `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`: (a) a raw
   StyleTTS2 checkpoint has the shape
   `{"net": {module_name: state_dict, ...}, "epoch": ..., "val_loss": ...}` — this is
   `stage1/first_stage.pth`'s expected shape; (b) the packaged Kokoro-API bundle has the shape
   `{module_name: state_dict, ...}` directly, with no `"net"` wrapper and no `module.` DP prefix
   (already stripped and key-remapped by t0002's extraction) — this is
   `best/david_v3_best_decoder_kokoro.pth`'s shape. The loader must:
   `raw = torch.load(path, map_location="cpu", weights_only=False)`; if `"net"` is a key in `raw`,
   use `raw["net"]`, otherwise use `raw` directly; then for each of the 5 target modules, if
   present, strip any `module.` prefix from its keys with `k.removeprefix("module.")` (matching the
   existing pattern at line 100 of the source file). Add a standalone
   `sha256_file(path: Path) -> str` (same helper as Step 6, reuse rather than re-derive) and compute
   it for both checkpoints at the top of `main()`. Add a `dp_prefix_present: bool` field to the
   `TensorForensics`-equivalent per-module record (True if any raw key in that module started with
   `module.` before stripping). Expected output:
   `uv run python -m tasks.t0016_v3_recipe_recovery.code.checkpoint_forensics` runs without a
   `KeyError` and prints 5 module names for each of the two checkpoints. Satisfies: REQ-2, REQ-10
   (local hash half).
9. **[CRITICAL] Compute weight-norm deltas and attempt the `multispeaker` resolution.** Run
   `code/checkpoint_forensics.py`'s `main()` to compare `V3_STAGE1_CKPT` (label `stage1`) against
   `V3_BEST_DECODER_CKPT` (label `best`) for all 5 modules, writing `weight_norm`, `mean_abs`,
   `max_abs`, `finite`, `num_params`, `dp_prefix_present` per module to
   `results/v3_module_weight_delta.raw.json`. For each module compute
   `relative_delta = (best.weight_norm - stage1.weight_norm) / stage1.weight_norm` and classify
   "changed" as `abs(relative_delta) >= 0.05` (the same 5% near-zero-shift threshold
   `predictor_tensor_forensics.py` already uses and documents as "well below what gradient updates
   would be expected to produce if the loss were actively improving"). For the `multispeaker`
   resolution: enumerate **every** top-level key present in `raw["net"]` from `V3_STAGE1_CKPT` (not
   only the 5 bundle modules — StyleTTS2 checkpoints may include `discriminator`, `style_encoder`,
   `predictor_encoder`, etc. as separate top-level entries) and record their shapes; if Step 4/5
   recovered a `models.py` copy from the VM, grep it for `multispeaker` to identify exactly which
   named parameters/layer shapes the flag controls, then check whether those specific
   parameters/shapes are present and consistent with `true` or `false` in `V3_STAGE1_CKPT`. Write
   the outcome as one of three explicit verdicts in a "## Multispeaker Resolution" subsection of
   `results/v3_checkpoint_forensics.md`: (a) "confirmed true" or "confirmed false" with the exact
   shape/parameter evidence cited; (b) "inferred [true/false]" if the shape evidence is suggestive
   but the VM's `models.py` was not recovered to confirm which parameters the flag actually
   controls; or (c) "unknown (contradictory sources, unresolved)" if the forensic check is
   inconclusive (e.g., `multispeaker` only affects components entirely outside what
   `stage1/first_stage.pth`'s `net` dict contains). Do not silently pick `true` or `false` without
   one of these three labels and its evidence. Expected output: a populated
   `results/v3_module_weight_delta.raw.json` with 5 relative-delta values, plus the "Multispeaker
   Resolution" subsection with an explicit verdict. Satisfies: REQ-6, REQ-7.
10. **Render the module table and byte-identity table into `results/v3_checkpoint_forensics.md`.**
    Adapt `predictor_tensor_forensics.py`'s `render_markdown()` pattern (Markdown table via string
    building, not a templating library) to render: (a) the 5-module table (columns: Module, Params,
    Finite, Weight norm (stage1), Weight norm (best), Relative delta, Changed (yes/no), DP prefix
    present); (b) a byte-identity table comparing the local SHA-256 of `stage1/first_stage.pth`
    (from Step 8) against the VM's `first_stage_v3_remote_sha256` (from
    `data/vm_inventory/inventory.json`, Step 6) and against any hash `t0006`'s own artifacts might
    already record (none currently do, per research — state this explicitly if so). Expected output:
    `results/v3_checkpoint_forensics.md` contains both tables with no `TBD`/placeholder cells —
    every cell is a number, "N/A", or a short reason string. Satisfies: REQ-2, REQ-10.

**Milestone 3: Audio forensics and mandatory human-listening set (local CPU).** Independently
verifiable: `results/audio_samples/{v3_shipped,v3_per_epoch,elevenlabs_reference}/` each contain
files, `results/listening_guide.md` links resolve, and every linked file has a gate verdict.

11. **[CRITICAL] Score every per-epoch sample with the hardened audio gate.** For each of the 20
    `epoch{0..3}_phrase{1..5}.wav` files under `V3_AUDIO_DIR`, plus every `.wav` found under
    `V3_AUDIO_DIR/v3/` and `V3_AUDIO_DIR/v3b/` (pulled in Step 1), call
    `check_audio_quality(wav_path, text=<phrase text>)` from `code/audio_quality_check.py` (the five
    v3 sample phrase texts must be recovered from context — if a `phrases.txt` or similar manifest
    exists among the pulled DVC files or the VM inventory, use it; otherwise, since exact original
    text is not preserved for these files, pass `text=None` and record `duration_sanity_pass: null`
    explicitly rather than guessing a text, per the data-analysis instruction "use None/null when
    data is unavailable, not 0.0/false as a stand-in"). Append a "## Per-Epoch Sample Gate Scores"
    table to `results/v3_checkpoint_forensics.md` (columns: file, `is_likely_noise`,
    `duration_sanity_pass`, `longest_nonsilent_run_s`) — this is the same file REQ-2's tables live
    in, satisfying CLAUDE.md's no-duplication rule. Expected output: 20+ rows, each with concrete
    values or explicit `null`, no exceptions raised. Satisfies: REQ-3, REQ-8 (epoch-cadence
    confirmation).
12. **[CRITICAL] Synthesize the shipped v3 bundle on the three gate texts, phrases, and 5 val96
    prompts.** Import
    `from tasks.t0008_tts_eval_harness_baselines.code.adapters import kokoro_v3_bundle, load_kokoro_model_with_checkpoint, save_wav`.
    Load the model once via
    `load_kokoro_model_with_checkpoint(decoder_path=V3_BEST_DECODER_CKPT, voicepack_path=V3_BEST_VOICEPACK)`
    (adjust kwarg names to match the actual `adapters.py` signature — read the function signature
    before calling, since this plan's research pass did not capture every kwarg). Synthesize, on
    CPU: the three fixed gate texts (derive text from filename by stripping the `.wav` extension and
    replacing underscores with spaces, matching `build_val96_prompts()`'s convention in
    `tasks/t0008_tts_eval_harness_baselines/code/prepare_prompts.py`): `lining_up_suggestions_17`,
    `lining_up_suggestions_10`, `putting_them_head_to_head_15` (sourced from
    `ELEVENLABS_DAVID_DIR`); the five v3 sample phrase texts (from Step 11, or a generic placeholder
    text noted as such if unrecoverable); and 5 val96 prompts chosen with `random.Random(42)` from
    the 96 entries in `data/v4/val_list.txt` (parsed the same way as `build_val96_prompts()`: split
    on `|`, derive text from the wav filename's stem minus its trailing hash suffix). Save each
    synthesis with `save_wav()` into `results/audio_samples/v3_shipped/<slug>.wav`. Validation gate
    (this is expensive relative to the task's local-CPU-only budget, ~13 synthesis calls): first
    synthesize only the 3 gate texts with `--limit 3` behavior (i.e., run those 3 first,
    standalone), inspect all 3 output files by ear/waveform (`check_audio_quality()` output, not
    literally listening during implementation) — if any of the 3 has `is_likely_noise: true`, STOP
    and debug the model-loading call before synthesizing the remaining 15 (baseline: t0008 already
    proved this exact bundle produces clean audio, so any `is_likely_noise: true` here indicates a
    loading/config bug in this task's code, not a property of the bundle). Run
    `check_audio_quality()` on every output file and record verdicts for
    `results/listening_guide.md` (Step 14). Satisfies: REQ-11.
13. **Copy per-epoch and ElevenLabs reference audio into the results tree (no re-synthesis).** Copy
    (do not re-synthesize) the 20 `epoch{0..3}_phrase{1..5}.wav` files plus `v3/`/`v3b/` contents
    from `V3_AUDIO_DIR` into `results/audio_samples/v3_per_epoch/`, preserving filenames. Copy the
    matching ElevenLabs original clips for the 3 gate texts (and any val96/phrase texts that have a
    same-text ElevenLabs clip in `ELEVENLABS_DAVID_DIR`) into
    `results/audio_samples/elevenlabs_reference/`. `dvc add results/audio_samples` and `dvc push`
    (retry on `DefaultAzureCredential` failure per Step 1's pattern — do not block the rest of the
    task on a single push failure; if push fails after 4 retries across ~10 minutes, proceed and
    note the pending push explicitly in the step's completion notes). Expected output: `dvc status`
    shows `results/audio_samples.dvc` committed and `dvc push` eventually exits 0 (or the retry
    exhaustion is documented). Satisfies: REQ-11.
14. **Write `results/listening_guide.md`.** One Markdown table: columns Text, ElevenLabs Original
    (relative link), v3 Shipped (relative link), Per-Epoch Samples (relative links, comma-separated
    if multiple), Gate Verdict (from Steps 11/12), Listening Note (one line each, e.g. "listen for
    breathiness on sibilants" or "compare prosody on the question-mark clause"). All file links must
    be relative paths from `results/listening_guide.md` (e.g.
    `audio_samples/v3_shipped/lining_up_suggestions_17.wav`) so they are clickable on GitHub.
    Satisfies: REQ-11.

**Milestone 4: Reconstruction and cross-check (local CPU, depends on Milestones 1-3's findings).**
Independently verifiable: `data/config_david_v3_reconstructed.yml` exists with an inline comment on
every field, and a diff list against v6c/t0009 exists.

15. **[CRITICAL] Write the annotated reconstructed config.** Create
    `data/config_david_v3_reconstructed.yml`, copying the exact key structure of
    `tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml` (same top-level keys:
    `ASR_config`, `ASR_path`, `F0_path`, `PLBERT_dir`, `batch_size`, `data_params`, `epochs`,
    `epochs_1st`, `epochs_2nd`, `first_stage_path`, `load_only_params`, `log_dir`, `loss_params`
    with its 13 sub-fields including `joint_epoch`/`lambda_gen`/`lambda_slm`/`diff_epoch`,
    `model_params` with its `multispeaker` field, `optimizer_params` with `lr`/`ft_lr`/`bert_lr`,
    `preprocess_params`, `pretrained_model`, `save_freq`, `second_stage_load_pretrained`,
    `slmadv_params`, `train_LM`). For every field, write an inline YAML comment starting with
    exactly one of `# confirmed: <file path or SHA-256>`,
    `# inferred: <reasoning citing which checkpoint/sample/log evidence>`, or
    `# unknown: default from v6c`. Populate `multispeaker` from Step 9's verdict (with its exact
    confidence label). Populate `first_stage_path` with the byte-identity finding from Step 10.
    Populate `joint_epoch`/`lambda_gen`/`lambda_slm`/`lr`/
    `batch_size`/`epochs_2nd`/`train_LM`/`max_len` from whatever VM logs, `train_second_patch.diff`
    content, or (failing those) v6c's own values were found, each labelled per its actual evidence
    (a value copied from v6c because nothing better exists is `# unknown: default from v6c`, not
    `# confirmed`). Cite
    `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/train_second_patch.diff`'s two patches
    (the `lambda_slm > 0` guard and the `monotonic_align` `sys.path` fix) as `# confirmed:` evidence
    wherever they bear on a field. Expected output:
    `uv run python -c "import yaml; yaml.safe_load(open('data/config_david_v3_reconstructed.yml'))"`
    parses without error, and every field has a `confirmed`/`inferred`/`unknown` comment (spot-check
    by grep:
    `grep -c '# confirmed:\|# inferred:\|# unknown:' data/config_david_v3_reconstructed.yml` returns
    a count close to the total field count). Satisfies: REQ-5, REQ-16.
16. **Diff against v6c and t0009's recommended-next-run config; list disagreements.** Append a "##
    Config Diff vs v6c and t0009" section to `results/v3_checkpoint_forensics.md` listing every
    field where `data/config_david_v3_reconstructed.yml` differs from
    `tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml` (e.g. `multispeaker`,
    `joint_epoch`, `lambda_gen`) and every field where it differs from t0009's recommended next-run
    config (in
    `tasks/t0009_stage2_training_failure_forensics/assets/answer/t0009-stage2-forensics-answer/full_answer.md`:
    "a copy of v6c with `joint_epoch=8`, epochs=20, per-epoch checkpoint retention, and health-gate
    integration"). For each disagreement, state which value (if either) is better-evidenced.
    Satisfies: REQ-5.
17. **Recover or document the 266-clip list; check val96 overlap.** If Step 4/6 recovered a
    training-list file from the VM, copy it to `data/v3_train_list_266.txt` and verify its line
    count is 266 (`wc -l`). Cross-check every clip filename in it against every line of
    `data/v4/val_list.txt` (96 lines) — if any filename appears in both, record the overlap count
    prominently (not buried) in `results/v3_checkpoint_forensics.md`'s summary, since
    task_description.md explicitly requires flagging this loudly if it occurs. If no list was
    recoverable, write `data/v3_train_list_UNRECOVERED.md` documenting exactly what was searched
    (the VM paths from Step 4, and the negative result that no project task's data lineage — v5's
    1557/1531/250-clip lists per t0011/t0012/t0006 — matches v3's 266-clip count or content) and
    state this explicitly as an open gap rather than a task failure, per the research finding that
    this list has never been recovered by any prior task. Satisfies: REQ-4.

**Milestone 5: Chart and metrics cross-check (local CPU).** Independently verifiable: the PNG file
exists and opens, and `results/metrics.json` contains only registered metric keys with numeric
values or explicit `null`.

18. **[CRITICAL] Generate `results/images/v3_module_weight_delta.png`.** Using `matplotlib` (per the
    data-analysis instruction's "use matplotlib and seaborn for all visualizations"), plot a bar
    chart from `results/v3_module_weight_delta.raw.json` (Step 9): x-axis = the 5 module names
    (`bert`, `bert_encoder`, `predictor`, `text_encoder`, `decoder`), y-axis = relative weight-norm
    change from `stage1` to `best` (as a percentage). Include a descriptive title ("v3 Stage 2:
    relative weight-norm change per module, Stage 1 → best checkpoint"), labeled axes ("Module",
    "Relative weight-norm change (%)"), and a horizontal reference line at the 5% near-zero-shift
    threshold used in Step 9's "changed" classification, annotated in the legend. Save to
    `results/images/v3_module_weight_delta.png`. Expected output: the PNG file exists and is
    referenced by filename in `results/v3_checkpoint_forensics.md`. Satisfies: REQ-13, REQ-7.
19. **Cross-check speaker_sim/rtf/ttfb_ms against t0008's recorded numbers and write
    `results/metrics.json`.** The three registered project metrics (`speaker_sim`, `rtf`, `ttfb_ms`,
    per `meta/metrics/`) all apply here because Step 12 already synthesizes the shipped v3 bundle
    locally on CPU. Reuse `tasks/t0008_tts_eval_harness_baselines/code/score_speaker_sim.py` to
    compute GE2E cosine similarity between the Step 12 synthesis outputs and the mean GE2E embedding
    of `ELEVENLABS_DAVID_DIR`'s reference set, and record per-call wall-clock timing already
    captured in `SynthResult.ttfb_s`/`SynthResult.rtf` from the `adapters` call in Step 12 (convert
    `ttfb_s` to `ttfb_ms` by ×1000). Use the **legacy flat** `results/metrics.json` format (per
    `arf/specifications/task_results_specification.md`) since this task measures one condition (the
    shipped bundle, re-synthesized locally) as a confirmation/cross-check of an existing number, not
    a multi-condition comparison of its own:
    `{"speaker_sim": <float>, "rtf": <float>, "ttfb_ms": <float>}`. In
    `results/v3_checkpoint_forensics.md`, state explicitly whether this task's locally re-measured
    `speaker_sim` is consistent with t0008's recorded `0.631` (fillers) / `0.588` (val96) within a
    reasonable tolerance (e.g. ±0.02, chosen because it is well above expected floating-point/CPU-
    vs-original-hardware nondeterminism for a deterministic decode) — if it is not consistent, flag
    this as evidence the local synthesis setup differs from t0008's in some way (e.g. resampling,
    model-loading kwargs) and investigate before trusting any of this task's own audio conclusions.
    This is the metric-measurement step required by the planning process for every applicable
    registered metric; `rtf`/`ttfb_ms` on CPU are expected to be materially worse than t0008's
    original (possibly GPU-backed) numbers and are recorded for completeness, not as a regression
    finding. Expected output: `python -m json.tool results/metrics.json` parses cleanly with exactly
    the 3 registered keys. Satisfies: REQ-2 (indirectly, corroborates REQ-7's "which modules
    changed" story via the observed speaker_sim gain), applicable-metrics coverage requirement.

**Milestone 6: Answer asset [CRITICAL].** This is the task's REQ-12 deliverable and the only
`expected_assets` entry in `task.json` (`{"answer": 1}`) — without it the task has not been done
regardless of how complete the forensics are.

20. **[CRITICAL] Write the `v3-recipe` answer asset.** Create `assets/answer/v3-recipe/details.json`
    (`spec_version: "2"`, `answer_id: "v3-recipe"`, `question`: "What exactly did Kokoro v3's Stage
    2 training recipe consist of, and which parts of that are confirmed versus inferred?",
    `short_answer_path: "short_answer.md"`, `full_answer_path: "full_answer.md"`,
    `answer_methods: ["code-experiment"]` (no papers, no internet sources used — state this
    explicitly per the answer-question instruction),
    `source_task_ids: ["t0002_kokoro_v4_voicepack_decoder_package", "t0006_kokoro_v5_stage2_subset", "t0008_tts_eval_harness_baselines", "t0009_stage2_training_failure_forensics", "t0013_v10_synthesis_quality_forensics", "t0015_v11_duration_blowup_forensics"]`,
    `confidence`: `"low"` if REQ-4 or REQ-6 ended unresolved, `"medium"` if one of them resolved and
    the other did not, `"high"` only if both the 266-clip list was recovered and `multispeaker` was
    confirmed with a file/hash — choose honestly based on Milestones 1-4's actual outcomes, do not
    default to `"medium"`). Write `assets/answer/v3-recipe/short_answer.md` (`## Question`,
    `## Answer` in 2-5 sentences stating the recipe directly — e.g. "v3 trained Stage 2 StyleTTS2 on
    266 David clips for at least 10 epochs from `first_stage_v3.pth`...
    [state the actual reconstructed values from Step 15, and the actual `multispeaker` verdict from Step 9]
    ... `## Sources`). Write `assets/answer/v3-recipe/full_answer.md` with all 8 mandatory sections
    per `meta/asset_types/answer/specification.md` (`## Question`, `## Short Answer`,
    `## Research Process`, `## Evidence from Papers` — state "papers method not used" —
    `## Evidence from Internet Sources` — state "internet method not used" —
    `## Evidence from Code or Experiments` — summarize Milestones 1-5's findings — `## Synthesis` —
    `## Limitations` — including the `multispeaker`/266-clip-list uncertainty and the "this project
    never ran the controlled ablation isolating data-scale or multispeaker as the actual cause of
    v3's success" caveat from the research findings — `## Sources` with markdown reference-link
    definitions to every cited task folder, e.g.
    `[t0009]: ../../../t0009_stage2_training_failure_forensics/`). Point the packaging-recipe
    pointer explicitly to t0002's five-module extraction contract (`extract_decoder_generic.py`) as
    the mechanism that must be followed unchanged by any future reproduction, and state which config
    fields a reproduction (t0017, out of scope here) must hold fixed to be a faithful test of this
    recipe. Expected output:
    `uv run python -m arf.scripts.verificators.verify_answer_asset t0016_v3_recipe_recovery v3-recipe`
    (or the project's equivalent answer-asset verificator invocation) passes with 0 errors.
    Satisfies: REQ-12.

This concludes the plan's Step by Step scope. Metric computation (Step 19) and chart generation
(Step 18) are the final implementation actions; the orchestrator's own later steps write the
project's standard follow-on summary and suggestion documentation from these outputs.

## Remote Machines

**Yes, one remote machine is required**: `LLM-T1-NC80`, the project's sole Azure ML pool entry
(`project/azure_vm.json`: 2×H100 SXM5, workspace `brainpowa-northeurope`, resource group
`rezolve-AI`, $13.96/h). It is used **only** for read-only SSH filesystem inspection of
`~/kokoro-finetune/` and `/mnt/cache/persist/` (Milestone 1) — no training, no inference, no GPU
compute is scheduled on it. Estimated runtime: ≤90 minutes wall time (hard cap per REQ-1/REQ-17),
acquired via `uv run python -m arf.scripts.utils.azure_ml_vm acquire t0016_v3_recipe_recovery` and
torn down via `uv run python -m arf.scripts.utils.azure_ml_vm teardown t0016_v3_recipe_recovery` (no
`--keep-running`), both wrapped in `run_with_logs.py` per Key Rule 1. All other work (checkpoint
forensics, audio-quality gating, CPU-only `kokoro.KModel` synthesis, chart generation) runs on the
local CPU worktree — vast.ai is not needed since the project's `available_services` does not require
it and the only Azure ML pool entry already covers the one piece of work requiring a remote machine
(filesystem access to the VM's OS disk).

## Assets Needed

* **DVC dataset** `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/` (dependency
  `t0006_kokoro_v5_stage2_subset`): `stage1/first_stage.pth`,
  `best/{config.json,david_v3_best_decoder_kokoro.pth,david_v3_best_voicepack.pt, david_v3_voicepack_ft_enc.pt}`,
  `audio/{epoch0..3_phrase1..5.wav,v3/,v3b/}`, `train_second_patch.diff` (plain-text, git-committed,
  not DVC-pointed). Must `dvc pull` before use (Step 1).
* **Config template** `tasks/t0006_kokoro_v5_stage2_subset/code/config_david_v6c_stage2.yml`
  (dependency `t0006_kokoro_v5_stage2_subset`), git-committed, no pull needed.
* **Confound table and answer asset**
  `tasks/t0009_stage2_training_failure_forensics/results/confound_table.md` and
  `tasks/t0009_stage2_training_failure_forensics/assets/answer/t0009-stage2-forensics-answer/full_answer.md`
  (dependency `t0009_stage2_training_failure_forensics`), git-committed.
* **Reusable code** (copy, not import, since none of these are registered libraries for the specific
  forensics/gating logic):
  `tasks/t0015_v11_duration_blowup_forensics/code/{predictor_tensor_forensics.py,audio_quality_check.py}`,
  `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`,
  `tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py` (hashing helper only).
* **Registered library** `tts_eval_harness` (v0.1.0, from `t0008_tts_eval_harness_baselines`):
  imported via
  `from tasks.t0008_tts_eval_harness_baselines.code.adapters import (kokoro_v3_bundle, load_kokoro_model_with_checkpoint, save_wav, SynthResult)`
  and `tasks.t0008_tts_eval_harness_baselines.code.score_speaker_sim` for the speaker-sim
  cross-check.
* **DVC dataset** `tasks/t0008_tts_eval_harness_baselines/data/11labs_david/` (ElevenLabs David
  reference corpus, 1358 clips) — must `dvc pull` before use (Step 1).
* **Project data** `data/v4/val_list.txt` (96-line held-out val set, already present, git-committed
  or DVC-pointed at the project root — confirmed present and readable during planning).
* **Remote machine**: `LLM-T1-NC80` per `project/azure_vm.json`, accessed via the
  `setup-remote- machine` skill's `acquire`/`teardown` CLI.

## Expected Assets

* **Answer asset** (`expected_assets.answer: 1` in `task.json`): `v3-recipe` at
  `tasks/t0016_v3_recipe_recovery/assets/answer/v3-recipe/` — the canonical recovered v3 recipe with
  its confirmed/inferred/unknown evidence ledger, per Milestone 6.
* Non-registered-asset-type outputs (task-folder files, not aggregator-tracked asset types):
  `data/config_david_v3_reconstructed.yml` (annotated reconstruction), `data/v3_train_list_266.txt`
  or `data/v3_train_list_UNRECOVERED.md`, `data/vm_inventory/` (copied VM text artifacts plus
  `inventory.json`), `results/v3_checkpoint_forensics.md` (module diff, per-epoch gate scores,
  config diff, multispeaker resolution, byte-identity check),
  `results/images/v3_module_weight_delta.png`,
  `results/audio_samples/{v3_shipped,v3_per_epoch, elevenlabs_reference}/` (DVC-tracked),
  `results/listening_guide.md`, `results/metrics.json`.

## Time Estimation

* Research (already done): 0 additional hours — `research/research_code.md` is complete (0 errors/0
  warnings per its own header).
* Milestone 1 (VM inspection): ≤90 minutes wall time, hard-capped (includes acquire/boot ~5-10 min,
  inventory scan/copy ~50-60 min, environment capture ~5 min, teardown ~5 min).
* Milestone 2 (checkpoint forensics): ~30-45 minutes (code adaptation + one `torch.load`-based run
  on CPU over 2 checkpoints, each well under 1 GB).
* Milestone 3 (audio forensics + synthesis): ~45-60 minutes (13 CPU synthesis calls at Kokoro-82M's
  typical CPU RTF, plus gating ~40 files, plus DVC add/push with possible retries).
* Milestone 4 (reconstruction + cross-check): ~30 minutes (mostly writing, informed by Milestones
  1-3's findings).
* Milestone 5 (chart + metrics): ~20 minutes.
* Milestone 6 (answer asset): ~20-30 minutes (writing, informed by all prior milestones).
* **Total implementation estimate: ≈4-4.5 hours wall time**, of which only the first 90 minutes
  bills VM time; the remainder is local CPU and does not affect the dollar cost estimate.

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `dvc pull`/`dvc push` fails with `DefaultAzureCredential failed to retrieve a token` (confirmed during planning: this exact error was reproduced live and is documented in multiple prior tasks' own command logs, e.g. `tasks/t0015_v11_duration_blowup_forensics/logs/commands/010_*`) | High (observed every time tested so far) | Blocks Steps 1 and 13 (data pull and DVC push of new audio) | Retry with a ~2-minute backoff, up to 4 attempts (~10 min); t0015's own logs show the identical failure resolved after ~9 minutes without any credential change. If pull still fails after retries for the *reference* data (Step 1), this is a hard blocker for Milestones 2-6 and must be escalated via an intervention file — do not fabricate forensics from memory of prior tasks' descriptions. If only the *push* (Step 13) fails after retries, proceed with the rest of the task and note the pending push explicitly rather than blocking. |
| The VM's `~/kokoro-finetune/` home directory (or the specific `logs/kokoro-david-v3*/config.yml`) is gone — Azure ML OS-disk persistence across stop/start is expected per Lesson 10 but not guaranteed, and t0014's own $272.78-overrun precedent shows this environment has had disk-state surprises before | Medium | REQ-1, REQ-4, REQ-6's VM-side tie-breaker, and REQ-8/REQ-9 lose their strongest evidence source | Record `"home_directory_present": false` immediately (Step 4) and stop the VM early rather than burning the full 90 minutes searching a directory that is not there; fall back entirely to checkpoint forensics (Milestone 2), per-epoch sample evidence (Milestone 3), and honest `unknown`/`inferred` labelling in the reconstructed config (REQ-16 explicitly allows an all-inferred result) — this is not a task failure, it is exactly the scenario the task's Rejection Criteria anticipate. |
| The 90-minute VM cap is exceeded because the inventory scan runs long (e.g. `find` over a large, unfamiliar directory tree, or an interactive `az login` prompt hangs if managed-identity auth on the VM itself is also broken) | Medium | Budget overrun beyond the $21 VM sub-cap and the task's $30 hard cap | Start the wall-clock timer at acquire (Step 3) and check it before each sub-step in Steps 4-6; if any single `find`/`git log` command seems to hang, kill it with a command-level timeout (e.g. `timeout 60 find ...`) rather than waiting indefinitely; Step 7's teardown is unconditional at the 90-minute mark regardless of what Steps 4-6 have completed. |
| Checkpoint-shape forensics (Step 9) is inconclusive for `multispeaker` because the flag only affects StyleTTS2 components (e.g. the discriminator) that are not among the 5 bundled modules present in either `stage1/first_stage.pth`'s `net` dict or the packaged bundle | Medium | REQ-6 cannot be settled by checkpoint forensics alone, undermining a headline claim of this task | This is explicitly anticipated in Step 9: the fallback is the VM's `models.py`/`build_model()` source (Step 4) as tie-breaker, and if that is also unavailable, the plan requires the honest `"unknown (contradictory sources, unresolved)"` label rather than picking a side — Key Rule 16 in the checklist (REQ-16/Rejection Criteria) makes this an acceptable, not a failing, outcome. |
| The 266-clip training list is not recoverable from the VM (research finding: no project task in this codebase's 15-task history has ever recovered or referenced v3's own list; v5's 1557/1531/250-clip lists are all confirmed-unrelated samples) | High | REQ-4 cannot be satisfied with a real list | Write `data/v3_train_list_UNRECOVERED.md` per Step 17's exact fallback — this is the explicitly pre-planned outcome, not an improvisation, and feeds directly into the orchestrator's `results/suggestions.json` step (REQ-14) rather than blocking this task. |
| CPU-only Kokoro synthesis (Step 12) is slow enough that 13 synthesis calls plus gating exceeds the local-time budget, or the `kokoro`/`espeak-ng` phonemizer dependency is missing from this worktree's `.venv` | Low-Medium | Delays Milestone 3, and by extension Milestones 4-6 | t0008 already validated this exact loading path on CPU for the same bundle, so the dependency chain is known-working; if `kokoro`/`espeak-ng`/`misaki` are missing from this specific worktree's environment, run `uv sync` first (per CLAUDE.md's standard setup command) before treating it as a blocker. |

## Verification Criteria

* Run `uv run python -u -m arf.scripts.verificators.verify_plan t0016_v3_recipe_recovery` (this
  plan's own verificator) — expect `0 errors`, and address any warnings before implementation
  begins.
* Run `find tasks/t0016_v3_recipe_recovery/data/vm_inventory -type f | wc -l` and
  `python -m json.tool tasks/t0016_v3_recipe_recovery/data/vm_inventory/inventory.json` — expect at
  least the `inventory.json` file itself to exist and parse cleanly, with a
  `"home_directory_present"` boolean and a `"vm_session_minutes"` value ≤ 90.0, confirming
  REQ-1/REQ-17 were honored.
* Run
  `grep -c '# confirmed:\|# inferred:\|# unknown:' tasks/t0016_v3_recipe_recovery/data/config_david_v3_reconstructed.yml`
  — expect a nonzero count approximately equal to the number of top-level and nested scalar fields
  in the config, confirming REQ-5/REQ-16 (no un-labelled field).
* Run
  `uv run python -c "import json; d=json.load(open('tasks/t0016_v3_recipe_recovery/results/metrics.json')); assert set(d) <= {'speaker_sim','rtf','ttfb_ms'}, d"`
  — expect no assertion error, confirming only registered metric keys were used.
* Run `ls tasks/t0016_v3_recipe_recovery/results/images/v3_module_weight_delta.png` — expect the
  file to exist (confirms REQ-13); open it and confirm it has a title, labeled axes, and 5 bars (one
  per module).
* Run
  `ls tasks/t0016_v3_recipe_recovery/results/audio_samples/v3_shipped tasks/t0016_v3_recipe_recovery/results/audio_samples/v3_per_epoch tasks/t0016_v3_recipe_recovery/results/audio_samples/elevenlabs_reference`
  — expect each directory to contain at least one `.wav` file, confirming REQ-11; then run
  `uv run dvc status` — expect the three directories to show as tracked (not "not in cache") after
  Step 13's `dvc add`. `results/listening_guide.md` must exist and every filename it links to must
  resolve
  (`grep -oP '\]\(\K[^)]+' results/listening_guide.md | xargs -I{} test -f tasks/t0016_v3_recipe_recovery/results/{}`
  should exit 0 for every link).
* Run the project's answer-asset verificator against this task's answer, e.g.
  `uv run python -u -m arf.scripts.verificators.verify_answer_asset t0016_v3_recipe_recovery v3-recipe`
  (exact module path per the project's verificator layout) — expect `0 errors`, confirming REQ-12,
  `details.json` validity, and both canonical documents' mandatory sections.
* Requirement-coverage check: for each of `REQ-1` through `REQ-13` and `REQ-16`/`REQ-17`, run
  `grep -c "REQ-<N>\b" tasks/t0016_v3_recipe_recovery/plan/plan.md` and expect a count of at least 2
  (once in the checklist, once in a Step by Step "Satisfies:" line) — confirms every executable
  checklist item maps to an executed step, not just a plan-time intention. `REQ-14` and `REQ-15` are
  the two checklist items explicitly marked orchestrator-scoped (their documents are written outside
  this plan's Step by Step by design) and are exempt from this count.

## Rejection Criteria

This task does not run a paired benchmark comparison in the Lesson-3 sense (no
successful-vs-failed-request ratio to compute), so the standard
`successful_requests / total_requests < 0.8` benchmark-nullification rule does not directly apply.
The task-specific rejection conditions, pre-registered before implementation begins:

* **No field in `data/config_david_v3_reconstructed.yml` may be labelled `confirmed` unless the Step
  15 write-up cites a specific file path or SHA-256 hash for it.** A field labelled `confirmed`
  without a traceable source in `results/v3_checkpoint_forensics.md` or
  `data/vm_inventory/inventory.json` invalidates that specific field's label (downgrade it to
  `inferred` or `unknown`) — it does not invalidate the whole task, since an honestly-labelled
  all-`inferred`/`unknown` reconstruction is, per the task's own Rejection Criteria text, "still a
  valid, useful result."
* **The 90-minute VM cap (REQ-1/REQ-17) is hard, not advisory.** If Step 7's teardown does not
  execute by 90 minutes of VM wall time for any reason (including a stuck SSH session or a failed
  `find` command), that overrun itself — not the forensic findings — is the reportable failure, and
  must be disclosed in the task's results rather than omitted because "the extra time helped."
* **A `multispeaker` verdict of `confirmed` requires BOTH a checkpoint-shape finding AND either the
  VM's `models.py` source confirming what that shape controls, OR a surviving VM launch YAML stating
  the flag directly.** A checkpoint-shape finding alone, without the `models.py` tie-breaker or a
  direct config, may only be labelled `inferred`, not `confirmed` — this directly operationalizes
  the task's own instruction that "this can only be resolved by checkpoint-shape forensics ...
  during implementation" while still requiring a second, independent confirmation before the
  strongest label is used.

The full context from research suggests this task's answer will most likely land at
`confidence: "low"` or `"medium"` (per the research findings that v3's row in every existing table —
t0009's confound table included — is "the worst-evidenced row," with every field either assumed or
unknown before this task's own forensics run). This is an acceptable, pre-registered outcome, not a
task failure to be avoided by inflating confidence post hoc.
