---
spec_version: "1"
task_id: "t0018_zero_shot_cloning_calibration"
updated_at: "2026-09-17T20:10:00Z"
completed_steps: 14
next_step_number: 15
next_step_id: "reporting"
---
# Task Objective

Benchmark three open zero-shot voice-cloning TTS models (F5-TTS, CosyVoice 2, Chatterbox) on David
reference audio with the t0008 harness to calibrate the reachable speaker_sim/TTFB envelope.

* * *

## Step History

### Step 1 — create-branch

Branch `task/t0018_zero_shot_cloning_calibration` created. Initial folder structure initialized in
`tasks/t0018_zero_shot_cloning_calibration/`. Step 1 is a mechanical setup step with no research
output.

### Step 2 — check-deps

Ran `verify_task_dependencies.py` for `t0018_zero_shot_cloning_calibration`: the single declared
dependency, `t0008_tts_eval_harness_baselines`, has `status: "completed"` in its `task.json`, so the
check passed with 0 errors and 0 warnings. Result written to
`logs/steps/002_check-deps/deps_report.json`.

### Step 3 — init-folders

Ran `init_task_folders` to create the mandatory task folder structure (`plan/`, `research/`,
`results/`, `results/images/`, `corrections/`, `intervention/`, `code/`, `logs/commands/`,
`logs/searches/`, `logs/sessions/`, `logs/steps/`, `assets/answer/`) with `.gitkeep` files; recorded
in `logs/steps/003_init-folders/folders_created.txt`. Populated the local aggregator cache at
`tasks/t0018_zero_shot_cloning_calibration/ctx/` (task_types, costs, tasks, metrics, suggestions) —
gitignored and not committed.

### Step 4 — research-papers

Reviewed the full existing paper corpus (`aggregate_papers`) for material on F5-TTS, CosyVoice 2,
Chatterbox, and GE2E speaker-similarity evaluation. The corpus currently holds only 4 papers, all
added by an unrelated prior task (`t0014_v11_decoder_fix_retrain`): HiFi-GAN, iSTFTNet, a
multi-generator vocoder study, and StyleTTS 2 — none cover the three target zero-shot cloning
systems directly, though StyleTTS 2 underlies the Kokoro baseline this task re-scores. Wrote
`research/research_papers.md` (`status: "complete"`, 4/4 papers cited) documenting this gap
explicitly; verificator passed with 0 errors, 1 expected warning (`RP-W003`, project has no category
taxonomy yet).

### Step 5 — research-internet

Wrote `research/research_internet.md` (19 web searches, 4 targeted fetches, 18 sources cited, 7
papers discovered, `status: "complete"`) covering install/checkpoints, licensing, and streaming/TTFB
behavior for F5-TTS, CosyVoice 2, and Chatterbox. Verificator passed with 0 errors, 0 warnings. Key
new findings: F5-TTS default weights are CC-BY-NC-4.0 (non-commercial risk for Key Question 6); only
CosyVoice 2 has genuine chunked streaming (F5-TTS and base Chatterbox are whole-utterance TTFB);
F5-TTS reference clips over ~20 s risk mid-word truncation, a direct risk to this task's
`ref_concat` (~30 s) condition — flag for validation before the full implementation run; Chatterbox
itself has no research paper (confirmed by the maintainer). Of 7 discovered papers, the first batch
of 3 `/add-paper` subagents (max-3-concurrent cap) completed and their assets are committed in this
step: `Chen2024-F5TTS` (`10.48550_arXiv.2410.06885`), `Du2024-CosyVoice2`
(`10.48550_arXiv.2412.10117`), and `Wan2018-GE2E` (`10.1109_ICASSP.2018.8462665`). 4 more remain
queued, not yet dispatched: `Du2024-CosyVoice1`, `Casanova2022-YourTTS`, `Zhang2025-ECAPA` (low
priority), `ChatterboxFlash2026` (low priority).

### Step 6 — research-code

Reviewed the `tts_eval_harness` library (t0008) plus t0013/t0014/t0015's audible-speech gate lineage
and `build_reference_concat.py`, and t0010's GPU cost-overrun precedent; wrote
`research/research_code.md` (5 tasks cited, `status: "complete"`, verificator passed 0
errors/warnings). In the same step, dispatched and confirmed all 4 remaining queued `/add-paper`
subagents from step 5 — the paper corpus now holds all 7 discovered papers (11 total, up from 4).
After all research steps completed, a `/research-summarize` subagent wrote
`research/research_summary.md` for downstream planning/implementation subagents to read instead of
the full research files.

### Step 7 — planning

Wrote `plan/plan.md` (spec_version 2, all 11 mandatory sections, 18 `REQ-*` items) covering the
3-system × 2-condition zero-shot cloning benchmark plus the two paired baselines; `verify_plan`
passed 0 errors/0 warnings. Cost estimate ≈$42-45 GPU wall-clock against the $70 hard cap, with a
dedicated Step 4 F5-TTS `ref_concat` smoke-gate check (confirmed the standard F5-TTS pipeline
auto-crops reference audio to ~15 s by default) run before the full 196-prompt job. Caveat: the
planning subagent initially wrote its output to the main repo instead of the task worktree
(spawn-prompt omission); the step-executor recovered the file into the worktree and cleaned the
stray main-repo files before verifying — no content was lost.

### Step 8 — setup-machines

Acquired `LLM-T1-NC80` (2xH100 NVL) on the third attempt (first two hit pool contention from a
concurrent `t0016` session; see `intervention/pool_busy_llm-t1-nc80.md`); armed the idle watchdog
(PID confirmed) before any download, then installed isolated venvs for F5-TTS, CosyVoice 2, and
Chatterbox under `/mnt/cache/persist/t0018_zero_shot_cloning_calibration/venvs/`, fixing two
install-time issues (CosyVoice2's `openai-whisper` needing `--no-build-isolation`; F5-TTS's default
resolver picking a torch/CUDA build too new for the VM's driver, repinned to `torch==2.5.1+cu121`).
All three venvs confirm `torch.cuda.is_available()` and `device_count()==2`. `machine_log.json`
written with `watchdog_active: true`. Caveat: environment prep took ~2h wall clock (well over the
plan's 0.5h line) due to uncached large wheel downloads — watch cumulative GPU spend closely in
`implementation`.

### Step 9 — implementation

Executed plan Steps 1-18 via a dedicated `/implementation` subagent. Chatterbox completed both
reference conditions at 100% success (392/392 clips); CosyVoice2 completed `ref_single` at 100%
success but `ref_concat` hard-failed at 0/196 (CosyVoice2 rejects reference audio over 30s, and this
task's `ref_concat` clip is 30.57s — a genuine system limit, not a bug); F5-TTS was null for all
variants and `kokoro_v3_bundle` was not re-measured, both due to an identical indefinite-hang
failure signature during model loading on this VM session (documented in
`intervention/f5_tts_smoke_gate_failed.md` and `intervention/kokoro_v3_bundle_not_remeasured.md`).
Total GPU spend (shared billing anchor with setup-machines) reached ~$56.15 of the $70 hard cap —
under the cap, no scope was cut for budget reasons. Key output: `results/metrics.json`,
`results/tables.json`, `results/per_clip_metrics.json` (980 rows), 4 required charts, the DVC-pushed
audio sets, and the answer asset `assets/answer/zero-shot-speaker-sim-ceiling/` (verificator
`PASSED`). Caveat: the step-executor's own turn ended once mid-run without an active
heartbeat-refresh mechanism, tripping a (false-alarm) `ST-E007` liveness alert that the coordinator
caught and the step-executor then fixed with a self-terminating background heartbeat loop for the
remainder of the wait — future long GPU-bound steps should set this up proactively rather than
reactively.

### Step 10 — teardown

Confirmed no live job remained on `LLM-T1-NC80` after implementation (SSH `tmux has-session` →
`DONE`), then ran the `/setup-remote-machine` Teardown Protocol via a dedicated subagent:
`azure_ml_vm teardown` released the task's lock and deallocated the VM (`deallocated: true`,
`other_locks_present: false`), updating `machine_log.json` with
`destroyed_at: "2026-09-17T19:31:35Z"`, `total_duration_hours: 4.173`, `total_cost_usd: 58.25`.
Wrote `results/remote_machines_used.json` and `results/costs.json` (both new — the `results` step
had not yet run) with the final `$58.25` line item. `verify_machines_destroyed` passed 0 errors/3
expected warnings (legacy `spec_version`, sandboxed-API-unreachable, no `checkpoint_path` on a
non-training job). Independently re-verified via a direct `az ml compute show --name LLM-T1-NC80`
call from this session (not just trusting the subagent): `state: "Stopped"`, last operation `Stop`
succeeded at `2026-09-17T19:31:32Z` — matches `machine_log.json` to within 3 seconds. Total task GPU
spend is final at **$58.25** of the $70 hard cap (~$11.75 unused headroom); no further GPU work will
occur in this task.

### Step 11 — creative-thinking

Wrote a five-part critique in `logs/steps/011_creative-thinking/step_log.md` (no dedicated asset for
this step): (1) the F5-TTS/`kokoro_v3_bundle` hang diagnosis left two cheap, concrete alternative
explanations unchecked (a GPU-context-contention confound in attempt 3, an untested
stale-HF-lock-file hypothesis) even though the aggregate two-codebases-same-signature pattern is
still decent evidence for a session-level cause; (2) the proposed success-criterion restatement
rests on one system/one condition/one session and should be flagged provisional pending an eventual
F5-TTS measurement; (3) a production path via zero-shot cloning realistically narrows to
CosyVoice2/Chatterbox given F5-TTS's CC-BY-NC-4.0 license, and CosyVoice2's output could instead
feed Kokoro Stage 2 as training augmentation rather than being framed only as a direct-replacement
candidate; (4) CosyVoice2's `ref_concat` null was a 0.57s miss against a greppable hard limit in its
own source that a per-system (not one-shared-clip) reference-duration design would likely have
avoided; (5) a previously unstated methodological flag — CosyVoice2 exceeding the ElevenLabs
self-consistency ceiling is worth reading as a possible GE2E-embedding/"cleaner voice" artifact, not
only as a clean win, pending a human-listening check. No committed data/tables were changed; this is
a critique layer for steps 12-15 to draw on.

### Step 12 — results

Wrote `results/results_summary.md` and `results/results_detailed.md` (`spec_version: "2"`, 12
concrete examples, `## Task Requirement Coverage` covering all 18 `REQ-*` items) on top of the
results data step 9/10 already produced — no data was recomputed. Every quoted number was
cross-checked against `results/metrics.json`/`results/tables.json` exactly; `verify_task_metrics.py`
and `verify_task_results.py` both PASSED with 0 errors/0 warnings. Step 11's five hedges (F5-TTS
attribution uncertainty, provisional success-criterion restatement, CosyVoice2/Chatterbox-only
licensing viability, the CosyVoice2 `ref_concat` 0.57s-miss framing, and the GE2E
"cleaner-voice"-artifact caveat) were carried into `## Limitations` verbatim in spirit, not
presented as more certain than step 11 showed them to be. `## Analysis` documents four
contradictions between the plan's assumptions and actual results (most notably: only 2 of 3 named
systems produced data, and `ref_single` could not be built as a literal single ~10s clip since no
such clip exists in the corpus).

### Step 13 — compare-literature

Wrote `results/compare_literature.md` comparing this task's measured `speaker_sim`/WER numbers
against the closest published values for the same three named systems: F5-TTS
([Chen2024, Table 1/2], SIM-o, WavLM-large), CosyVoice2 ([Du2024, Table 5/6], SS, ERes2Net), and
Chatterbox ([Seo2026, Table 1], SIM-o/WER, WavLM-ECAPA-TDNN — Chatterbox-Flash's own re-benchmark of
the base open-weight Chatterbox checkpoint this task uses). Every one of the 8 comparison-table rows
carries an explicit `**METRIC MISMATCH**` flag in its Notes column (GE2E-cosine `resemblyzer` vs.
WavLM/ERes2Net-based embeddings, different corpora/speakers/text) and the file's `## Summary`,
`## Analysis`, and `## Limitations` all restate that no numeric delta should be read as a validated
quality ranking — order-of-magnitude/qualitative comparison only, consistent with the constraint
repeated across steps 4-12. `verify_compare_literature.py` passed 0 errors/0 warnings (both the
producing subagent's own check and this step-executor's independent `run_with_logs`-wrapped re-run).
Two notable findings surfaced: (1) a `### Prior Task Comparison` subsection confirms this task's
re-measured `elevenlabs_david` baseline (0.8325/0.7923) matches t0008's cited 0.832/0.792 numbers,
and separately flags that CosyVoice2's `ref_single` result (0.8628 val96) contradicts the implicit
prior assumption that the 0.85 success criterion was structurally unreachable under this project's
own GE2E-cosine scoring; (2) Chatterbox's measured WER (41.01%) is far higher than [Seo2026]'s
published 1.99% for the identical checkpoint, honestly reported as a genuine negative finding but
attributed to ASR-scorer size (`faster-whisper base.en` vs. Whisper-large-v3/HuBERT-large) and this
task's ASR-adversarial filler text (brand names, alphanumeric session IDs), not audio quality —
cross-checked against the per-clip brand-name WER breakdown in `results/results_detailed.md` showing
no elevated WER specifically on brand-name text.

### Step 14 — suggestions

Spawned a dedicated `/generate-suggestions` subagent (explicit worktree/branch instruction), which
did its own review of `results/`, `checkpoint.md`, prior step logs (including step 11's
creative-thinking critique), and its own duplicate-check via `aggregate_suggestions --uncovered` (29
existing entries, none overlapping) and `aggregate_tasks` (18 tasks, none covering these angles),
then wrote `results/suggestions.json` with 7 entries covering: an F5-TTS retry with
`py-spy`/HF-lock-file hang diagnostics (S-0018-01, high); a trimmed-under-30s CosyVoice2
`ref_concat` re-run (S-0018-02, medium); a human-listening check on the CosyVoice2 "cleaner-voice"
GE2E artifact hypothesis (S-0018-03, medium); a Chatterbox WER re-score with a larger ASR model
(S-0018-04, medium, cites `Seo2026`); evaluating CosyVoice2 output as Kokoro Stage 2 training
augmentation (S-0018-05, medium); registering the two efficiency metrics via `/add-metric`
(S-0018-06, low); and a brainstorm on whether to restate the `speaker_sim >= 0.85` success criterion
(S-0018-07, high). This step-executor cross-checked the 7 entries against the coordinator's explicit
follow-up list and found one gap — creative-thinking recommendation (d), a general per-system (not
one-shared-clip) reference-duration *design guideline for future multi-system TTS benchmark tasks*,
was not captured as its own entry (`S-0018-02` only covers redoing this task's own CosyVoice2 cell).
Sent the subagent a follow-up prompt (not pre-written wording) asking it to re-analyze and add an
entry if it judged the gap genuine and non-duplicate after its own re-check; the subagent confirmed
the gap, re-checked the aggregators again, and added `S-0018-08` ("Design future multi-system TTS
benchmarks with per-system safe reference durations", `evaluation`, medium). Final file has 8
entries; `verify_suggestions.py` passed 0 errors/0 warnings both when the subagent ran it and when
this step-executor independently re-ran it via `run_with_logs`. The two remaining follow-up items
from the coordinator's list were confirmed out of scope rather than missing: the `add-paper`
`SKILL.md` verificator-path documentation bug is framework work per CLAUDE.md Rule 0 (already logged
in Cross-Step Decisions below as a future `self-improvement` candidate, not a task-level
suggestion); and F5-TTS's CC-BY-NC-4.0 licensing-risk flag was conditional on a future
production-path suggestion considering F5-TTS, which none of the 8 entries do (S-0018-01 frames the
F5-TTS retry as closing a missing measurement/research-ceiling gap, not a production proposal).

* * *

## Cross-Step Decisions

* **Pool-contention saga (setup-machines, resolved on the 3rd attempt)**: `LLM-T1-NC80` is the only
  entry in `project/azure_vm.json`, and a concurrent session on `task/t0016_v3_recipe_recovery` was
  independently starting/stopping this same VM. Attempts 1-2 both failed with `ssh_connect` timeouts
  (documented in `intervention/pool_busy_llm-t1-nc80.md`) because the VM was stopped/contended when
  `acquire` tried it. Attempt 3 succeeded because the VM happened to be free (mid-`Start`, likely
  triggered by the peer session or a tail end of attempt 2) at the moment this step-executor polled
  — no code change was needed, just re-polling with `--vm-name LLM-T1-NC80` pinned. Future
  step-executors on this task (or `t0016`) should expect this contention to recur since both tasks
  share the one-VM pool; check `az ml compute show ... --query last_operation` before assuming a
  fresh `pool_busy` intervention file means the VM is still busy right now.
* **F5-TTS's default `pip install f5-tts` is not driver-safe on this pool VM**: it resolves an
  unconstrained torch (2.14.0+cu130) that requires a newer NVIDIA driver than `LLM-T1-NC80` ships
  (535.274.02 / CUDA 12.2), silently leaving `torch.cuda.is_available()` false. Fixed by repinning
  `torch==2.5.1`/`torchaudio==2.5.1` from the `cu121` wheel index inside `.venv-f5tts` (also
  satisfies `bitsandbytes>=2.4` and `torch-einops-utils>=2.5`). CosyVoice2 and Chatterbox's own
  resolved torch versions (2.3.1+cu121, 2.6.0+cu124 respectively) were already driver-compatible.
  Any later re-install of these venvs (e.g. after a VM swap) must re-check
  `torch.cuda.is_available()` before trusting the smoke gate, not just that `pip install` exited 0.
* `~/.cache/huggingface` and `/mnt/pip_cache` are broken symlinks to an unprovisioned ephemeral
  mount on this shared pool VM (pre-existing, not caused by this task). Model-weight downloads in
  `implementation` must set
  `HF_HOME=/mnt/cache/persist/t0018_zero_shot_cloning_calibration/hf-cache` explicitly rather than
  relying on the default cache location.
* The paper corpus has zero coverage of F5-TTS, CosyVoice 2, or Chatterbox specifically —
  `research-internet` (step 5) independently sourced install/inference/checkpoint details for all
  three since there is no corpus paper to fall back on.
* F5-TTS's default checkpoints are CC-BY-NC-4.0 and remain non-commercial even after fine-tuning —
  must be flagged explicitly in `results/suggestions.json` if F5-TTS leads on `speaker_sim`/TTFB
  (Key Question 6, production viability).
* F5-TTS official docs warn reference clips over ~20 s risk mid-word truncation, with a 30 s
  prompt+generation cap — this task's `ref_concat` condition (~30 s) must be validated against this
  limit before the full implementation run (smoke-gate stage).
* Published SIM-o/SS/MOS numbers from F5-TTS, CosyVoice 2, and the Podonos Chatterbox AB study use
  different metrics (WavLM-based or MOS/preference-%) than this project's GE2E-cosine `speaker_sim`
  and must never be merged/compared directly — extends the same rule already established for CMOS-S
  in `research_papers.md`.
* **Paper-addition tracking — RESOLVED in step 6**: all 7 papers discovered in
  `research/research_internet.md` are now landed in the corpus (11 papers total, up from 4 before
  this task started); no papers remain queued. Confirmed via
  `aggregate_papers --format json --detail short`. Final DOI/commit map:
  * `Chen2024-F5TTS` — `10.48550_arXiv.2410.06885` (landed step 5)
  * `Du2024-CosyVoice2` — `10.48550_arXiv.2412.10117` (landed step 5)
  * `Wan2018-GE2E` — `10.1109_ICASSP.2018.8462665` (landed step 5)
  * `Du2024-CosyVoice1` — `10.48550_arXiv.2407.05407` (landed step 6, commit `0268535`)
  * `Casanova2022-YourTTS` — `10.48550_arXiv.2112.02418` (landed step 6, commit `a305347`)
  * `Zhang2025-ECAPA` (dispatch label) — actually **`Kunesova2025`**, "An Exploration of ECAPA-TDNN
    and x-vector Speaker Representations in Zero-shot Multi-speaker TTS" (Kunešová, Hanzlíček,
    Matoušek, TSD 2025) — `10.48550_arXiv.2506.20190` (landed step 6, commit `4bb35d8`). The
    research-internet snippet's authorship guess was wrong; the `/add-paper` subagent re-verified
    against the actual PDF and corrected the citation key.
  * `ChatterboxFlash2026` (dispatch label) — actually "Chatterbox-Flash: Prior-Calibrated Block
    Diffusion for Streaming Zero-Shot TTS" (Seo, Park, Nam, 2026) — `10.48550_arXiv.2605.30748`
    (landed step 6, commit `a5f7160`). Same situation: snippet authorship was wrong, re-verified and
    corrected on download.
  * No paper addition failed; the inline-fallback path in the paper-addition protocol was not
    needed. `compare-literature` and `reporting` can treat paper coverage as complete — no further
    dispatch action required.
* Two independent `/add-paper` subagents flagged that `arf/skills/add-paper/SKILL.md`'s documented
  verificator module path (`arf.scripts.verificators.verify_paper_asset`) does not exist in this
  repo; the real module is `meta.asset_types.paper.verificator`. This is a framework documentation
  bug (out of scope for this task per CLAUDE.md Rule 0) — worth a future `self-improvement` pass.
* **F5-TTS `ref_concat` risk — resolved as a planned smoke-gate check, not a blocker**: confirmed
  during planning (step 7) that F5-TTS's standard pipeline (`preprocess_ref_audio_text`) auto-crops
  reference audio over ~15 s by default. The plan's Step 4 requires a dedicated F5-TTS ×
  `ref_concat` synthesis check before the full 196-prompt run, recording whether the effective
  reference consumed was ~30 s or ~15 s; this is a documented caveat on that one variant, not
  grounds for nulling it, unless synthesis fails outright even after a manual 15 s-trim retry.
* **`results/metrics.json` unregistered-key ambiguity — resolved**: the task text's
  `efficiency_inference_time_per_item_seconds`/`efficiency_inference_cost_per_item_usd` keys are not
  registered in `meta/metrics/` (only `rtf`, `speaker_sim`, `ttfb_ms` are). The plan routes these
  two fields (plus WER, duration ratio, gate-failure count) into `results/tables.json` instead,
  following t0008's own `REGISTERED_METRIC_KEYS` convention, and flags that `suggestions.json`
  (orchestrator step) should recommend registering the two efficiency metrics via `/add-metric` for
  future tasks.
* **Planning-subagent worktree miss (process note for future spawns)**: the step 7 planning subagent
  was spawned without an explicit worktree `cd` instruction and defaulted to the main repo checkout
  on `main`, writing `plan/plan.md` and regenerated `ctx/` files there. The step-executor caught
  this via `git status`/`git branch` before trusting the subagent's report, copied the plan into the
  worktree, and deleted the stray main-repo files. Future step-executors in this task should include
  the explicit worktree path in subagent spawn prompts to avoid repeating this.

* * *

## Next Step Notes

Proceed to step 15 (`reporting`), the final step, per `step_tracker.json`. `LLM-T1-NC80` remains
fully torn down (no live-machine concerns remain). Final total GPU spend is unchanged and final:
**$58.25 of the $70 hard cap** (~$11.75 unused headroom), recorded in `machine_log.json`,
`results/remote_machines_used.json`, and `results/costs.json`.

`results/suggestions.json` is now written (step 14) with 8 entries (`S-0018-01` through `S-0018-08`)
and `verify_suggestions.py` passes with 0 errors/0 warnings (verified independently by both the
producing subagent and this step-executor). It covers: an F5-TTS retry with `py-spy`/HF-lock-file
hang diagnostics; a trimmed-under-30s CosyVoice2 `ref_concat` re-run; a human-listening check on the
CosyVoice2 "cleaner-voice" GE2E artifact hypothesis; a Chatterbox WER re-score with a larger ASR
model (cites `Seo2026`); evaluating CosyVoice2 output as Kokoro Stage 2 training augmentation;
registering the two ad hoc efficiency metrics via `/add-metric`; a brainstorm on restating the
`speaker_sim >= 0.85` success criterion; and a general per-system (not one-shared-clip)
reference-duration design guideline for future multi-system TTS benchmark tasks. All of step 11's
"Recommendations Carried Forward" and the coordinator's explicit follow-up list are now reflected as
concrete suggestion entries, except two items confirmed genuinely out of scope for
`suggestions.json`: the `add-paper` `SKILL.md` verificator-path documentation bug (framework work
per CLAUDE.md Rule 0, already logged above as a future `self-improvement` candidate) and the F5-TTS
CC-BY-NC-4.0 licensing-risk flag (its trigger condition — a future suggestion proposing F5-TTS for
production — never occurred, since `S-0018-01` frames the F5-TTS retry as a
missing-measurement/research-ceiling gap, not a production proposal).

`reporting` (step 15) should run all remaining verificators (`verify_task_file.py`,
`verify_task_dependencies.py`, `verify_suggestions.py`, `verify_task_metrics.py`,
`verify_task_results.py`, `verify_task_folder.py`, `verify_logs.py`, and asset verificators for the
answer asset and all 11 corpus paper assets touched by this task), capture task sessions, and
finalize the task per `arf/skills/execute-task/SKILL.md` Phase 6. No open data/GPU/budget concerns
remain going into it.
