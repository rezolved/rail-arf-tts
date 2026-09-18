---
spec_version: "3"
task_id: "t0018_zero_shot_cloning_calibration"
step_number: 11
step_name: "creative-thinking"
status: "completed"
started_at: "2026-09-17T19:36:47Z"
completed_at: "2026-09-17T19:55:00Z"
---
## Summary

Out-of-the-box review of steps 1-10's results before the `results`/`suggestions`/`reporting` steps
lock in the task's conclusions. Five angles were pressure-tested: whether the
F5-TTS/`kokoro_v3_bundle` hangs were really ruled out as VM-infra rather than code/config issues,
whether the "reachable envelope" success-criterion restatement is safe on 2-of-3-systems data, what
a production path looks like given F5-TTS's licensing risk, whether the CosyVoice2 `ref_concat` null
reflects a task design mismatch rather than a fair test, and a previously unstated methodological
red flag on CosyVoice2's headline number. This step produced no code changes — it is a written
critique for the `results` and `suggestions` steps to draw on, per the "optional, out-of-the-box
analysis" description in `arf/skills/execute-task/SKILL.md` Phase 5.

## Actions Taken

1. Re-read `checkpoint.md` in full (steps 1-10), with particular attention to step 9's
   implementation results and step 10's teardown, per the coordinator's instructions.
2. Read `intervention/f5_tts_smoke_gate_failed.md` and
   `intervention/kokoro_v3_bundle_not_remeasured.md` in full to extract the exact diagnostic
   evidence gathered (wchan states, `HF_HUB_OFFLINE` attempt, local-copy workaround) and the exact
   hedges already present in the task's own conclusions.
3. Read `assets/answer/zero-shot-speaker-sim-ceiling/full_answer.md` and `task_description.md` in
   full to see what the answer asset already claims and what it does not address.
4. Cross-checked `LESSONS.md` Lesson 10 (Azure ML VMs mount an Azure Files SMB share at
   `/mnt/batch/tasks/shared/LS_root/mounts/clusters/<vm-name>/code/`, symlinked from
   `/mnt/cache/persist`) against the `wait_for_response` (CIFS wait-channel) evidence in the F5-TTS
   intervention file, to check whether the "VM/mount" causal claim is grounded in a documented,
   pre-existing fact about this VM class rather than speculation.
5. Read `plan/plan.md`'s F5-TTS `ref_concat` smoke-gate section and `results/tables.json`'s notes to
   check whether CosyVoice2's own reference-length limit was knowable before the run, the same way
   F5-TTS's ~15s auto-crop was.
6. Wrote the five-part creative-thinking analysis below, checked it against `LESSONS.md` Lesson 8/9
   (liveness/hang-detection precedent) for consistency, and recorded it in this step log since this
   step has no dedicated asset type per the Per-Step Spec Table.

## Outputs

* This step log (`logs/steps/011_creative-thinking/step_log.md`) — the analysis is written inline
  here (see `## Analysis` below) rather than as a separate `research/creative_thinking.md`, since
  neither `logs_specification.md` nor `step_registry.py`'s `creative-thinking` entry
  (required_files: `step_log.md`, min 100 words, no other required file) specify a dedicated output
  path for this step, and `SKILL.md` Phase 5 describes it only as "Update `checkpoint.md`" plus the
  step log.
* `checkpoint.md` — Step History entry, Next Step Notes rewritten for the `results` step-executor.

## Issues

No issues encountered. This step involved no GPU/remote-machine work (confirmed already torn down in
step 10) and no code execution beyond reading existing task files.

## Analysis

### 1. Was the F5-TTS / `kokoro_v3_bundle` hang really "the VM pool," or could it be code/config?

The intervention files' own resolution language ("very likely an environment/concurrency issue...
rather than a fundamental F5-TTS incompatibility") is appropriately hedged, but the case for a pure
VM/infrastructure cause is weaker than the confident framing in `checkpoint.md`'s Next Step Notes
("both implicated in ... the `LLM-T1-NC80` pool's intermittent model-load hangs") suggests, for
three concrete reasons the diagnostic did not rule out:

* **Attempt 3 is confounded, not clean evidence.** It ran F5-TTS's load "concurrently with
  CosyVoice2's debug script on the same GPU." Two processes independently initializing CUDA contexts
  and touching the same GPU at once is itself a known source of multi-minute stalls (driver-level
  context-creation locks, NCCL/IPC handshakes, memory-fragmentation retries) independent of any
  filesystem issue. This attempt cannot cleanly support "it's the SMB mount" because a second,
  entirely different candidate cause was active in the same trial.
* **A cheap, decisive diagnostic was available and not used.** `/proc/<pid>/stack` and
  `/proc/<pid>/syscall` were blocked by permissions, and the investigation stopped there.
  `py-spy dump --pid <pid>` (or `py-spy dump --native`) does not require root for a same-user
  process and would have printed the exact Python (and native) frame the process was blocked in
  within seconds — resolving definitively whether the hang was inside `torch.load`,
  `hf_hub_download`, `huggingface_hub`'s file locking, or somewhere in F5-TTS's/misaki's own code.
  This is worth adding to `arf/scripts/utils/idle_watchdog.sh`-adjacent tooling (or the
  `diagnose-stuck-step` skill) as a standard first move for "indefinite hang, no log output" cases,
  since `/proc` inspection is not always available to the executing user.
* **A stale-lock hypothesis was proposed but never checked.** The intervention file's own leading
  guess is "a stale lock file left by an interrupted earlier attempt" — `huggingface_hub`'s
  file-locking mechanism (`.lock` files under the HF cache) is exactly the kind of thing that can
  silently deadlock a second process even under `HF_HUB_OFFLINE=1`, since offline mode still touches
  the local cache's lock files before skipping the network call. A single
  `find $HF_HOME -name "*.lock"` before killing each attempt would have taken seconds and either
  confirmed or eliminated this as the cause. It was not run.

Set against this: the fact that **two unrelated codebases** (F5-TTS's flow-matching loader and
Kokoro/misaki's lexicon-install pipeline) both produced the identical "zero incremental progress"
signature in the same session is real, and reasonably strong, evidence for a session/VM-level common
cause (shared HF Hub rate-limiting — explicitly observed as a warning on every Hub call this session
— or shared SMB-mount degradation under cumulative I/O) rather than two independent bugs in two
different projects' code. Both things can be true at once: the *aggregate pattern* across systems is
decent evidence for a session-level cause, while the *individual* root-cause diagnosis for either
hang was incomplete and left at least two cheap, concrete alternative explanations (GPU-context
contention in attempt 3, stale HF lock files) unexamined. A follow-up task should treat "it's this
VM pool" as a prior to test with `py-spy` and lock-file checks, not as an established conclusion to
build on.

### 2. Should the "reachable envelope" success criterion be restated on 2-of-3-systems data?

The answer asset's proposed restatement (decouple the similarity bar from latency/reliability) is
directionally reasonable, but the evidentiary base is thinner than "the envelope is now known"
implies:

* It rests on **one system, one condition, one session**: CosyVoice2 `ref_single` val96 (0.863).
  F5-TTS contributed zero data points, so the true upper end of the envelope (a non-autoregressive
  flow-matching architecture that the literature suggests can be strong on similarity) is simply
  unknown, not merely "not yet as good as CosyVoice2." Any restated project criterion built now is
  provisional and should say so explicitly, flagged for revisit once F5-TTS is ever measured.
* CosyVoice2's own 0.863 rests on a **gate-passing 67-of-96 subsample** — the 0.859 gate-passed-only
  mean is reassuring, but it is still a single run with no repeat/seed-variation check. Before
  writing a new number into `project/description.md` (a project-wide, hard-to-walk-back document), a
  domain expert would want either a second `ref_single` clip or a second random prompt ordering to
  confirm 0.86 is a stable estimate rather than one favorable draw.
* The `ref_single`/`ref_concat` clips used are themselves **engineered, curated concatenations from
  half-A** (the "longest clean half-A clips," per `code/build_references.py`), not a single natural
  in-the-wild reference clip. The ElevenLabs self-consistency ceiling it is being compared against
  (0.792/0.832) uses ordinary independent half-B takes with no such curation. The comparison is
  directionally fair (both sides are "David's own voice measured against itself/a clone of itself")
  but not perfectly matched in reference-quality effort, which is worth one sentence in any
  criterion-restatement proposal so it isn't read as a clean apples-to-apples number.

### 3. What does a production path look like given F5-TTS's licensing risk?

F5-TTS produced no data this session, so the licensing risk (CC-BY-NC-4.0, survives fine-tuning) is
moot for now — but it is worth stating forward-looking, so a future task doesn't spend GPU budget
re-chasing F5-TTS under the impression it is a deployment candidate:

* Even if a retry succeeds and F5-TTS's numbers turn out best, its checkpoint license makes it
  unusable in Rezolve's commercial voice-commerce product without a separate commercial license from
  the checkpoint owner or training an equivalent architecture from scratch on permissively licensed
  data — which reopens the exact cost/timeline question this task exists to shortcut. F5-TTS's
  practical role, even unblocked, is a **research ceiling reference**, not a shippable option.
* That narrows the realistic zero-shot production path to CosyVoice2 (Apache-2.0) or Chatterbox
  (MIT) — both already measured, and both 5-19x over the 300ms TTFB target. A genuine production
  path via either would require accepting a materially higher filler-latency budget (a product/UX
  decision, not a modeling one), or a distillation/streaming-optimization engineering effort
  comparable in scale to the Kokoro Stage 2 investment this task was meant to help decide against —
  which would undercut the "avoid fine-tuning cost" premise of choosing zero-shot cloning in the
  first place.
* A genuinely new angle not in the answer asset: CosyVoice2's Apache-2.0 license permits using its
  high-`speaker_sim` output as **synthetic training data for Kokoro Stage 2** (e.g., additional
  David-voice training pairs, or a distillation target), rather than as a live-serving replacement.
  This reframes "zero-shot cloning vs. fine-tuning" as a false dichotomy and is worth a line in
  `results/suggestions.json` as a distinct follow-up task idea from "should we ship CosyVoice2
  directly."

### 4. Was the CosyVoice2 `ref_concat` failure a fair test, or a task-design mismatch?

The 0/196 hard failure is a real, useful finding about CosyVoice2 (it rejects references over 30s
outright), but the specific way it surfaced is worth scrutinizing:

* The task's `ref_concat` clip landed at **30.57s** — 0.57s over CosyVoice2's hard
  `assert speech.shape[1] / 16000 <= 30` limit (`cosyvoice/cli/frontend.py`). That is a narrow miss
  against a static, greppable constant in CosyVoice2's own source, discoverable the same way
  `research-internet` (step 5) had already surfaced F5-TTS's ~15s auto-crop behavior before the run.
  Nobody checked CosyVoice2's own hard limit at planning time, even though the plan's Step 4
  dedicated a smoke-gate check specifically to F5-TTS's reference-length risk. This is an
  inconsistency in how thoroughly the two systems' reference-length constraints were vetted before
  committing GPU budget to the full run.
* The task's design used **one literal reference audio file, shared verbatim across all three
  systems**, for cross-system comparability. But the three systems have different, incompatible hard
  ceilings discovered only by running them: F5-TTS silently auto-crops at ~15s, CosyVoice2
  hard-errors at 30s, Chatterbox tolerates the full 30.57s with no apparent limit. A single shared
  duration cannot respect all three simultaneously — the "fair, identical reference" design is
  actually what turned an informative "does reference length matter" experiment into a binary
  pass/fail against an arbitrary fence for one of the three systems. A more robust design for a
  calibration task like this one would target each system's own documented safe maximum (e.g.,
  `min(published_limit - safety_margin, target_duration)` per system, with the effective duration
  actually used recorded per system-row in the results table) rather than assuming one audio file
  works uniformly.
* The fix is cheap relative to what was lost: re-running just the CosyVoice2 `ref_concat` cell with
  the same clip trimmed to (for example) 29.5s would cost roughly the same GPU-time as CosyVoice2's
  own `ref_single` run (already measured, so the cost is well-bounded) — small next to the task's
  $11.75 unused budget headroom, though that headroom is now moot since the VM is torn down. This is
  already flagged as a possible follow-up in the answer asset's Limitations section; this step
  affirms it and adds the design-level root cause (uniform-duration reference selection is not
  system-agnostic) for `suggestions.json` to generalize into future multi-system benchmark task
  guidance.

### 5. A methodological red flag not yet raised: CosyVoice2 beating the self-consistency ceiling

The full answer reports, without further comment, that CosyVoice2 `ref_single` val96 (0.863) exceeds
David's own **half-A-vs-half-B self-consistency ceiling** (0.792). Read literally, this says a
synthetic clone is a *better* match to half-B than David's own independent half-A recordings are — a
result worth treating as a flag on the metric, not simply a win for the model:

* A plausible explanation is that GE2E cosine similarity partially captures session/recording-level
  acoustic characteristics (background noise floor, microphone response, room tone) in addition to
  pure speaker identity. If CosyVoice2's synthesis (conditioned on a handful of half-A reference
  clips) produces audio that is acoustically "cleaner" or more uniform than David's own naturally
  variable real recordings, it could score artificially high on GE2E cosine against half-B without
  necessarily being a more faithful reproduction of David's idiosyncratic voice qualities in a
  perceptual sense.
* This does not contradict the task's own measured numbers, but it does mean "CosyVoice2 beat the
  ElevenLabs ceiling" should be read with an added caveat: it may reflect the model converging
  toward a cleaner/more canonical point in embedding space that happens to sit close to both half-A
  and half-B, rather than proof of superior identity fidelity. A cheap follow-up check (not
  attempted in this task, and not necessary to gate this task's completion) would be a human
  listening pass specifically asking "does this sound more like David, or does it sound like a
  slightly-averaged version of a voice" — which the task's own `results/listening_guide.md` already
  sets up nicely for a future task or the human owner to do.
* This angle is distinct from, and additional to, the reliability caveat (30% gate failures) already
  in the answer asset — it applies even to the gate-passing 67/96 subsample, since the
  near-identical gate-passed-only mean (0.859) shows the high score is not merely an artifact of
  counting broken clips as falsely similar.

## Recommendations Carried Forward

* `results/suggestions.json` (step 13) should propose: (a) a `py-spy`/lock-file-first diagnostic
  protocol for future indefinite-hang cases before attributing them to "the VM pool"; (b) treating
  any success-criterion restatement in `project/description.md` as provisional pending an F5-TTS
  measurement; (c) a distinct "CosyVoice2 output as Kokoro Stage 2 training augmentation" follow-up
  task, separate from a "should we ship CosyVoice2 directly" framing; (d) a per-system
  safe-reference-duration design for any future multi-system TTS benchmark, instead of one shared
  literal audio file; (e) a targeted human-listening check on whether CosyVoice2's high
  `speaker_sim` reflects genuine David-identity fidelity or a "cleaner/averaged voice" metric
  artifact.
* None of the above changes any file this task has already produced (`results/tables.json`, the
  answer asset, `metrics.json`) — this step is a critique layer for
  `results`/`suggestions`/`reporting` to draw on, not a correction to committed data.
