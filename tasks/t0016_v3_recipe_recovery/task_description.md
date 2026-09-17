# Recover the v3 Training Recipe From Artifacts

## Motivation

Across fifteen tasks and roughly $840 of GPU spend, this project has produced exactly one Kokoro
fine-tune that works end to end: **v3**. It was trained before the ARF project existed, on 266
ElevenLabs David clips, with the Kokoro-native ISTFTNet decoder. Packaged as a Kokoro-API bundle it
scores `speaker_sim=0.631` on fillers and `0.588` on val96 with TTFB p50 of 185 ms (t0008). Every
attempt since then has failed for a different reason (t0001 poisoned phonemes, t0005/t0006
zero-param checkpoint loads and early GAN activation, t0010 partial decoder load, t0014 duration
blowup), and every one of them changed several variables from v3 at once.

The v3 recipe was never written down. t0009 flagged this as S-0009-04 ("Recover v3 training
configuration") and it has been open since 2026-09-14. The original author is **not available**, so
this task must reconstruct the recipe from evidence alone. Without it, the next training run has no
known-good baseline to change one variable from, which is exactly the discipline t0009's answer
asset demands.

The question this task answers: **what exactly did v3 do, and which parts of that are confirmed
versus inferred?**

## What already exists (start here, do not re-derive)

Local, DVC-tracked, under `tasks/t0006_kokoro_v5_stage2_subset/data/reference/v3/`:

* `best/config.json` — Kokoro-API model config of the shipped bundle: `istftnet` decoder
  (`upsample_rates: [10, 6]`, `gen_istft_n_fft: 20`, `gen_istft_hop_size: 5`), `multispeaker: true`,
  `max_dur: 50`, `style_dim: 128`, `n_token: 178`.
* `best/david_v3_best_decoder_kokoro.pth` (327 MB, five-module bundle: `bert`, `bert_encoder`,
  `predictor`, `text_encoder`, `decoder`), `best/david_v3_best_voicepack.pt`,
  `best/david_v3_voicepack_ft_enc.pt`.
* `stage1/first_stage.pth` — the Stage 1 checkpoint v3 started Stage 2 from (byte-identity with the
  `first_stage_v3.pth` used by v6c/v6d was never confirmed; t0009 left all SHA-256 hashes null).
* `train_second_patch.diff` — the only two patches v3 applied to upstream `train_second.py`
  (`lambda_slm > 0` guard, `monotonic_align` sys.path).
* `audio/epoch{0..3}_phrase{1..5}.wav`, `audio/v3.dvc`, `audio/v3b.dvc` — per-epoch synthesis
  samples; t0002 references a clean sample `v3/audio/v3b/ep9_p5.wav`, implying at least 10 epochs
  and a `v3b` variant.

Prior forensic findings to reconcile, not repeat:

* t0009 `results/confound_table.md`: v3 row is almost entirely "assumed" or "unknown" (`joint_epoch`
  speculated 0, `lambda_gen` assumed 1.0, `lr` unknown), val_loss 0.506.
* t0006 `task_description.md`: "v3 used 266 clips (31 steps/epoch)", "v3 reached val=0.569 at epoch
  1", "v3 0.549 at epoch 4", and claims `multispeaker: false` for v3 — which **contradicts**
  `best/config.json`'s `multispeaker: true` and t0009's confound table. This contradiction must be
  settled with evidence.
* t0006's `code/config_david_v6c_stage2.yml` is the closest surviving relative of the v3 config
  (ISTFTNet, `first_stage_v3.pth`, 250 clips) and is the template to diff the reconstruction
  against.
* t0002 `results/results_summary.md` confirms the v3 packaging recipe (`make_voicepack_from_ft.py`,
  `convert_checkpoint.py`, five-module extraction) and that v3's bundle loads cleanly into
  `kokoro.KModel`.

## Evidence sources, in priority order

1. **The VM's home directory.** `~/kokoro-finetune/` on `LLM-T1-NC80` held `first_stage_v3.pth` as
   recently as t0014 (2026-09-16, see t0014 `plan/plan.md` step referencing
   `/home/azureuser/kokoro-finetune/first_stage_v3.pth`). The OS disk of an Azure ML compute
   instance survives stop/start; only `/mnt` is ephemeral (Lesson 10). Upstream StyleTTS2's
   `train_second.py` copies its config into `log_dir/config.yml` at launch, so any surviving
   `logs/kokoro-david-v3*/` or `Models/*v3*/` directory contains the literal config. Also inspect:
   `~/.bash_history`, `~/.python_history`, `~/kokoro-finetune/{configs,v3,data,logs,Models}`, any
   `*.log`, `tensorboard` event files, `git log`/`git reflog` inside `~/kokoro-finetune` if it is a
   git checkout, and file mtimes to order the runs. Then the persistent share `/mnt/cache/persist/`
   (resolve with `readlink -f`, Lesson 10).
2. **Checkpoint forensics, locally on CPU.** Compare `stage1/first_stage.pth` against
   `best/david_v3_best_decoder_kokoro.pth` module by module using the raw `torch.load` pattern from
   `tasks/t0015_v11_duration_blowup_forensics/code/predictor_tensor_forensics.py`: which modules
   changed (trained) and which are byte-identical (frozen or never loaded), the decoder architecture
   (`istftnet` key shapes), `multispeaker` (presence/shape of style-conditioned layers), and whether
   `first_stage.pth` keys carry a DataParallel `module.` prefix. Record SHA-256 of every v3 file and
   of `first_stage_v3.pth` on the VM to settle the byte-identity question t0009 left open.
3. **Per-epoch audio samples.** The `epoch{0..3}_phrase{1..5}.wav` set and the `v3`/`v3b` folders
   fix the epoch count, the sample phrases, and the checkpoint cadence (`save_freq`). Run the
   hardened gate from `tasks/t0015_v11_duration_blowup_forensics/code/audio_quality_check.py` on
   every sample so the reconstruction records what "clean" sounded like at each epoch.
4. **The 266-clip training list.** Not present locally (`data/v4/train_list.txt` is the 1557-clip v4
   list; `t0006/code/train_list_250.txt` is a seed-42 sample of v5, not v3's list). Recover it from
   the VM (`~/kokoro-finetune/data/v3/` or similar) or, failing that, reconstruct the candidate set
   from `voicepack_ft_enc.pt`'s provenance and document it as unrecovered.
5. **rail-benchmarks git history** was already checked in this session: no `kokoro` path exists in
   any ref of `rezolved/rail-benchmarks`. Do not spend time there; note it as a dead end.

## Key Questions

1. What was v3's Stage 2 config: `joint_epoch`, `diff_epoch`, `lambda_gen`, `lambda_slm`, `lr`,
   `batch_size`, `epochs_2nd`, `multispeaker`, `train_LM`, `max_len`, `first_stage_path`? For each
   field, is the value **confirmed** (found in a file), **inferred** (derived from checkpoints,
   samples or logs), or **unknown**?
2. Was `multispeaker` true or false? `best/config.json` says true, t0006's description says false.
3. Which modules did Stage 2 actually update relative to `first_stage.pth`? In particular: did the
   decoder change at all, or does v3's 0.03 speaker_sim gain over "base Kokoro + v3 voicepack"
   (t0008: 0.631 vs 0.603) come from `predictor`/`text_encoder` alone?
4. How many epochs ran, which epoch was selected as "best" and by what criterion, and what is the
   val_loss trajectory (t0006 quotes 0.569 at epoch 1, 0.549 at epoch 4, 0.506 best)?
5. Which library versions did v3 run under (torch, kokoro, phonemizer/espeak, misaki, StyleTTS2
   commit)? Capture from the VM environment if it survives (Lesson 4).
6. Is `stage1/first_stage.pth` (DVC) byte-identical to `first_stage_v3.pth` on the VM, and to the
   file v6c/v6d loaded?
7. What exactly is the 266-clip list, and does it overlap val_96? (It must not; if it does, say so
   loudly — it changes how v3's numbers should be read.)

## Scope

### 1. Bounded VM inspection (read-only)

Start `LLM-T1-NC80` through `/setup-remote-machine`, run the inventory over SSH, copy every config,
log, list and small text artifact found into this task's `data/vm_inventory/` (no checkpoints; those
are already in DVC or too large), record SHA-256 hashes, and stop the VM. Hard cap: **90 minutes of
VM time (~$21)**. Do not train, do not modify anything on the VM. If the home directory is gone,
record that as a finding and proceed with sources 2-4 only.

### 2. Checkpoint and sample forensics (local CPU)

Module-by-module diff of Stage 1 vs best bundle; architecture fingerprint; per-epoch sample gate
scores. Write `results/v3_checkpoint_forensics.md` with the tables described under Outputs.

### 3. Reconstruct and cross-check

Write `data/config_david_v3_reconstructed.yml` in the same schema as
`t0006/code/config_david_v6c_stage2.yml`, with an inline comment on every line stating
`# confirmed: <source>` / `# inferred: <reasoning>` / `# unknown: default from v6c`. Diff it against
v6c and against t0009's "recommended next run" table; list every disagreement.

### 4. Answer asset

One answer asset, `v3-recipe`, with the canonical recipe (short answer: the config in five
sentences; full answer: the evidence table, the confirmed/inferred/unknown ledger, the environment
pins, the packaging recipe pointer to t0002, and what a reproduction must hold fixed).

## Expected Outputs

* `assets/answer/v3-recipe/` — the recipe with its evidence ledger.
* `data/config_david_v3_reconstructed.yml` — annotated config, ready for t0017 to run unchanged.
* `data/v3_train_list_266.txt` — recovered list, or `data/v3_train_list_UNRECOVERED.md` explaining
  what was tried.
* `data/vm_inventory/` — copied text artifacts plus `inventory.json` (path, size, mtime, sha256).
* `results/v3_checkpoint_forensics.md` — table: module, param count, changed-vs-Stage-1 (yes/no,
  relative weight-norm delta), DP prefix present; table: per-epoch sample, `is_likely_noise`,
  `duration_sanity_pass`, `longest_nonsilent_run_s`.
* `results/results_summary.md`, `results/results_detailed.md` — with one chart,
  `results/images/v3_module_weight_delta.png` (x: module, y: relative weight-norm change Stage 1 to
  best; answers Key Question 3).
* `results/suggestions.json` — at minimum: if the 266-clip list is unrecoverable, a suggestion for
  how t0017 should choose its subset; if `multispeaker` remains ambiguous, a two-arm ablation
  suggestion.

## Rejection criteria

* No field may be labelled "confirmed" without a file path or hash that proves it. Anything else is
  "inferred" or "unknown". A reconstruction that is all "inferred" is still a valid, useful result
  if it says so.
* The VM cap is hard: at 90 minutes, stop the VM regardless of progress and report what was covered.

## Compute and budget

* VM: `LLM-T1-NC80` (the only pool entry; the artifacts live on its OS disk, so no other machine
  helps). ≤ 1.5 h at $13.96/h ≈ **$21**. Everything else is local CPU. Total cap: **$30**.
* No training, no GPU compute beyond the VM being on.

## Dependencies

* `t0006_kokoro_v5_stage2_subset` — owner of the DVC-tracked v3 reference artifacts.
* `t0009_stage2_training_failure_forensics` — confound table and the open S-0009-04 this task
  closes; its checkpoint forensics conventions are reused.
