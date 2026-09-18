# v3 266-Clip Training List — Unrecovered

## What was searched

* **VM (`LLM-T1-NC80`) home directory**, `~/kokoro-finetune/data/` (Milestone 1 Step 4):
  `find ~/kokoro-finetune/data \( -iname '*v3*' -o -iname '*266*' \)` returned only two irrelevant
  matches (a `.cancelled.wav` file and a v5-normalized clip whose content hash happens to contain
  the substring `266`) — no training-list file. `~/kokoro-finetune/data/` itself contains only
  `OOD_texts.txt`, `train_list_v11_normalized.txt` (v11's own list), `data/v4/val/` (partial), and
  `data/v5_normalized/` (1531 clips — the same corpus size referenced by t0011/t0012's v5 lineage,
  confirmed unrelated to v3's 266-clip count).
* **`/mnt/cache/persist`** (the one Azure Files share proven to survive VM stop/start, per Lesson
  10): resolved via `readlink -f` to
  `/mnt/batch/tasks/shared/LS_root/mounts/clusters/llm-t1-nc80/code`, and searched to depth 3. This
  share contains artifacts from `t0009`, `t0010`, `t0014_v11_decoder_fix_retrain`, and several
  directories belonging to an unrelated LLM fine-tuning project sharing this VM pool
  (`t0027_wise_ft_interpolation_and_val_repair`, `t0043_code`, `t0043_pkg`, `t0046repo`, `t0053`,
  `t0053repo`, `t0053_data`, `t0055_staging`) — no `v3`-named directory, no 266-line list, and no
  file whose content matches a StyleTTS2 manifest format (`wav_path|phonemes|speaker_id`) with
  exactly 266 lines.
* **Critical finding**: `~/kokoro-finetune` on the VM is a symlink to
  `/mnt/cache/persist/t0014_v11_decoder_fix_retrain/kokoro-finetune` — i.e. it is **t0014's own
  StyleTTS2 clone** used for the v11 retrain (2026-09-16), not a preserved v3-era (pre-ARF)
  environment. Nothing about v3's actual original training environment survived;
  `first_stage_v3.pth` itself was not found anywhere on the VM either (see
  `results/v3_checkpoint_forensics.md`'s Byte-Identity Check), confirming this is a wholesale loss
  of the v3-era filesystem state, not a narrower gap.
* **`/mnt/kikiri-tts`** (referenced in one relevant, non-credential-bearing line recovered from the
  VM's shared `~/.bash_history`:
  `cd /mnt/kikiri-tts/StyleTTS2 && ... python train_second.py --config_path /tmp/config_david_v4b.yml`):
  this path no longer exists (`find: '/mnt/kikiri-tts': No such file or directory`) — `/mnt`
  (excluding the `/mnt/cache/persist` symlink target) is ephemeral across VM stop/start per Lesson
  10, and this history line is for `v4b`, not `v3`, and its config lived at
  `/tmp/config_david_v4b.yml` (also ephemeral). Dead end, as anticipated by `task_description.md`'s
  own priority ordering.
* **This project's own data lineage** (cross-checked, not re-derived): `data/v4/train_list.txt` is
  the 1557-clip v4 list (poisoned phonemes, per t0001);
  `tasks/t0006_kokoro_v5_stage2_subset/code/train_list_250.txt` is a seed-42 250-clip sample of v5;
  the VM's own `data/v5_normalized/` holds 1531 clips. None of the project's v4/v5-lineage lists is
  266 clips or a subset/superset consistent with 266, and no prior task (t0001 through t0015) has
  ever referenced or recovered v3's own list — this matches the research finding cited in
  `plan/plan.md`.
* **rail-benchmarks git history**: already checked during planning (research phase) — no `kokoro`
  path exists in any ref of `rezolved/rail-benchmarks`. Not re-checked here.

## Conclusion

The 266-clip training list is **not recoverable** from any evidence source available to this task.
This is an explicit, pre-registered open gap (`plan/plan.md`'s Risks & Fallbacks table already
anticipated this as the likely outcome, citing that no project task in this codebase's history has
ever recovered v3's own list), not a task failure. `data/config_david_v3_reconstructed.yml`'s
`data_params.train_data` field is labelled `unknown` accordingly, with only a naming-convention
guess (`../data/v3/train_list_266.txt`) rather than a real recovered path.

Consequently, the val96/266-clip overlap check (Key Question 7) cannot be run against a real list
either. `data/v4/val_list.txt` (96 lines) was cross-checked against every other locally-recoverable
list this task touched (`data/v5_normalized`'s 1531 filenames, `train_list_250.txt`'s 250 filenames)
— zero overlaps found by filename in both — but this does **not** establish whether the
never-recovered 266-clip list itself overlaps val_96, since that list was never seen.

## Suggested next step (for the orchestrator's `results/suggestions.json`, REQ-14)

Since the 266-clip list cannot be recovered, a reproduction attempt (t0017, out of this task's
scope) has no principled way to reconstruct v3's exact training subset. A one-way door remains open:
`best/david_v3_voicepack_ft_enc.pt`'s provenance (an encoder-fine-tuned voicepack artifact) might,
on closer inspection by a follow-up task, reveal which utterances contributed to it, narrowing the
candidate set. Absent that, the most defensible fallback is a fresh, seed-42 stratified sample of
266 clips from the full ElevenLabs David corpus
(`tasks/t0008_tts_eval_harness_baselines/data/11labs_david/`, 1364 clips) or from the David-voice
portion of `data/v4`/`data/v5_normalized`, explicitly disjoint from `data/v4/val_list.txt`'s 96
held-out clips, documented as a substitute (not a reproduction) of v3's actual subset.
