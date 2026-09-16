---
spec_version: "2"
task_id: "t0012_v5_corpus_normalize_and_reaudit"
date_completed: "2026-09-16"
status: "complete"
---
# Plan: v5 Corpus LUFS Normalization and Clipped-Fraction Re-Audit

## Objective

t0011 (`v5 training data audio quality audit`) audited all 1557 v5 train clips and flagged 246
(15.8%) as unusable, including 224 clips (14.4%) flagged purely because ElevenLabs peak-normalizes
its output to 0 dBFS — not because the audio is actually clipped. t0011's creative-thinking step
confirmed this and recorded two follow-up suggestions: **S-0011-01** (LUFS-normalize the full corpus
to -14 LUFS to recover the 224 clips) and **S-0011-03** (replace the raw peak-dBFS clipping check
with a `clipped_fraction` filter so the audit itself stops mistaking intentional peak normalization
for real clipping). This task implements both together: the corrected audit is what makes it safe to
trust the corpus has no genuine clipping before normalizing gain up, and the post-normalization
audio is re-checked with the same corrected metric in case raising the gain pushes any clip into
real clipping for the first time. This task produces data only — it runs no training and needs no
GPU.

**Done looks like**: a new `clipped_fraction` metric (fraction of samples within 1 LSB of full
scale) replaces the `peak_dbfs > -0.1 dBFS` clipping flag; every clip that passes the corrected
audit is LUFS-normalized to -14 LUFS and written to `data/v5_normalized/`; the normalized audio is
re-checked for new clipping/silence; a final clean manifest
(`data/train_list_v5_normalized_clean.txt`) is produced and DVC-pushed; and `results_detailed.md`
reports the reclassification counts, before/after distributions, and final clean-manifest size as a
fraction of 1557, compared against t0011's 1311/1557 (84.2%) baseline.

## Task Requirement Checklist

Operative task text (from `task.json` `short_description` and `task_description.md`):

> Fix the clipping heuristic that over-flags peak-normalized clips, LUFS-normalize the full v5
> corpus to -14 LUFS, and produce a near-full-corpus clean train manifest.
> 
> 1. Reuse t0011's data and code: `dvc pull` the same v5 train (1557 clips) and v4 val (96 clips)
>    paths t0011 used. Read `tasks/t0011_v5_data_quality_audit/data/per_clip_stats.jsonl` and
>    `tasks/t0011_v5_data_quality_audit/code/audit_audio.py` as the starting point — do not modify
>    t0011's files; copy what's needed into this task's `code/`.
> 2. Corrected clipping metric (S-0011-03): add `clipped_fraction` per clip; replace
>    `peak_dbfs > -0.1 dBFS` with `clipped_fraction > 0.1%`; keep other thresholds unchanged
>    (duration < 1.5s, silence > 30%, LUFS outside [-30, -6], sample rate != 24kHz, non-mono).
>    Report how many of the original 224 peak-flagged clips are reclassified as not-clipped, and how
>    many (if any) genuinely clipped clips remain.
> 3. LUFS normalization (S-0011-01): for every clip that passes the corrected audit, normalize
>    loudness to -14 LUFS (EBU R128) with `pyloudnorm`, preserving 24kHz sample rate and mono. Write
>    to `data/v5_normalized/`.
> 4. Post-normalization re-check: re-run clipped-fraction and silence checks on normalized audio;
>    move any clip that becomes genuinely clipped after the gain change to the flagged list.
> 5. Manifest and DVC tracking: write `data/per_clip_stats_v2.jsonl`, `data/flagged_clips_v2.txt`,
>    `data/train_list_v5_normalized_clean.txt`. `dvc add data/v5_normalized/` and `dvc push` before
>    marking complete.
> 6. Comparison and distributions: report flag counts by category before (t0011) vs after; final
>    clean-manifest size as a fraction of 1557. Produce histograms (peak dBFS, LUFS,
>    clipped_fraction) before vs after normalization, saved to `results/images/`.
> 
> Key Questions: (1) how many v5 clips are genuinely clipped by `clipped_fraction`? (2) does
> normalization gain introduce new clipping? (3) final clean-manifest size vs t0011's 1311/1557 and
> the two suggestions' individual estimates (~1532 for S-0011-03 alone, ~1556 for S-0011-01 alone)?
> (4) does normalization change distributions in ways that could affect training stability?

| ID | Requirement | Satisfied by step(s) | Evidence |
| --- | --- | --- | --- |
| REQ-1 | `dvc pull` v5 train (1557) + v4 val (96) manifests/audio; read t0011's `per_clip_stats.jsonl` and `audit_audio.py` as reference only; never modify t0011's folder; copy what's needed into `code/`. | Step 1 | `dvc pull` output confirms file counts; `git diff --stat tasks/t0011_v5_data_quality_audit/` is empty |
| REQ-2 | Compute `clipped_fraction` (fraction of samples within 1 LSB of full scale) per clip. | Step 2 | `clipped_fraction` field present in every `per_clip_stats_v2.jsonl` record |
| REQ-3 | Replace `peak_dbfs > -0.1 dBFS` with `clipped_fraction > 0.1%`; keep duration/silence/LUFS/rate/channel thresholds unchanged. | Step 2, Step 3 | `code/constants.py` diff vs t0011's; `flag_counts_v2.json` categories |
| REQ-4 | Report reclassification: how many of the original 224 peak-flagged clips are now clean; how many remain genuinely clipped. | Step 3 | `results_detailed.md` "Reclassification" table |
| REQ-5 | LUFS-normalize every corrected-audit-clean clip to -14 LUFS with `pyloudnorm`, preserving 24kHz/mono, write to `data/v5_normalized/`. | Step 2 | files present under `data/v5_normalized/`; `post_lufs` ≈ -14 in stats |
| REQ-6 | Post-normalization re-check: re-run clipped-fraction and silence checks on normalized audio; move newly-failing clips to flagged list. | Step 2, Step 3 | `post_clipped_fraction`/`post_silence_fraction` fields; flag reason `clipping_after_normalization` |
| REQ-7 | Write `data/per_clip_stats_v2.jsonl` with `clipped_fraction`, pre- and post-normalization peak/LUFS. | Step 2 | `wc -l` = 1557; schema check |
| REQ-8 | Write `data/flagged_clips_v2.txt`. | Step 3 | file exists, tab-separated path+reasons |
| REQ-9 | Write `data/train_list_v5_normalized_clean.txt` pointing into `data/v5_normalized/`. | Step 3 | file exists; paths resolve under `data/v5_normalized/` |
| REQ-10 | `dvc add data/v5_normalized/` and `dvc push`; normalized audio gitignored, never committed raw. | Step 5 | `git status` shows only `.dvc` pointer staged; `dvc push` exit 0 |
| REQ-11 | Report flag counts by category, before (t0011) vs after (this task). | Step 3 | `results_detailed.md` "Flag Counts: Before vs After" table |
| REQ-12 | Report final clean-manifest size as a fraction of 1557, vs t0011's 1311/1557 (84.2%) and the two suggestions' individual estimates (~1532, ~1556). | Step 3 | `results_detailed.md` summary line |
| REQ-13 | Histograms (peak dBFS, LUFS, clipped_fraction) before vs after normalization in `results/images/`. | Step 4 | 3 PNG files present |
| REQ-14 | Key Question 1: how many clips are genuinely clipped under `clipped_fraction`? | Step 3 | `results_detailed.md` answers with exact count vs t0011's 0-10 estimate |
| REQ-15 | Key Question 2: does the gain change introduce new clipping? | Step 2, Step 3 | count of `clipping_after_normalization` flags reported |
| REQ-16 | Key Question 4: does normalization change LUFS/duration/silence distributions in ways affecting training stability, and does it interact with the short-duration/silence flags? | Step 4, Step 6 | `results_detailed.md` "Analysis" section discusses interaction |
| REQ-17 | CPU-only, $0 cost — no paid API or GPU usage in any step. | Cost Estimation section; all steps | `## Cost Estimation` states `$0.00`; no step calls a paid API or provisions compute |
| REQ-18 | Produce the reclassification counts, before/after distributions, and final clean-manifest size that `results_detailed.md` will report (the file itself is written by the orchestrator's results stage). | Step 3, Step 6 | `data/analysis_v2.json` contains all required counts |
| REQ-19 | val_96 (`data/v4/val_list.txt`, 96 clips) is read-only reference/leak-check only — never normalized or added to the clean manifest, per project rule "NEVER train on val_96." | Step 1, Step 3 | leak check: 0 val paths in `train_list_v5_normalized_clean.txt` |

**Ambiguity note**: the task text does not specify which bit depth "1 LSB of full scale" refers to.
The v5 corpus audio format is not fixed by the manifest alone (WAVs could be 16- or 24-bit PCM). The
plan resolves this in Step 2 by detecting bit depth per file via `soundfile.info().subtype` rather
than hardcoding 16-bit, so the metric is correct regardless of the corpus's actual bit depth.

## Approach

**Grounding in t0011's own analysis** (this task has no separate research phase — data-analysis
tasks reuse the prior task's audit findings as their research input, and t0011's
`results/creative_thinking.md` already worked out the exact fix):

* t0011 found the corpus has **zero errors, zero LUFS outliers, zero OOV flags, zero multi-channel
  clips** — the only real issues are the 224 peak-normalized-not-clipped clips and 1 silence
  outlier. t0011's creative-thinking section explicitly recommended `clipped_fraction > 0.001` as "a
  more principled threshold [that] distinguishes genuine saturation from deliberate normalization"
  and recommended LUFS-normalizing to -14 LUFS as "the safest path... before training."
* t0011's `results/suggestions.json` (S-0011-01) states the current mean LUFS is already -14.69 (std
  2.34), so -14 LUFS normalization is a small, low-risk gain adjustment for most clips.
* t0011's creative-thinking section includes worked pseudocode for the normalization gain formula,
  reused directly in Step 2 below: `gain = 10 ** ((TARGET_LUFS - loudness) / 20)`,
  `normalized = clip(audio * gain, -1.0, 1.0)`.
* Strategy A in t0011's creative-thinking section ("accept peak-normalized clips, flag only true
  clipping" via `clipped_fraction > 0.001`) estimated 0-10 residual clipped clips and a ~1532-clip
  clean manifest if applied alone — this is the REQ-14/REQ-4 sanity check target.

**Technical approach**:

1. Copy (not import) t0011's `code/audit_audio.py`, `code/constants.py`, `code/paths.py` into this
   task's `code/` as starting points, per the task's explicit instruction to read them as reference
   and copy what's needed rather than modify t0011's immutable folder.
2. Extend the copied per-clip loop with one new metric (`clipped_fraction`) computed from the same
   decoded waveform already used for `peak_dbfs`, so no second decode pass is needed for the
   pre-normalization stats.
3. Do the audit-pass and normalization-pass **in a single per-clip loop** (one script,
   `code/audit_normalize.py`) rather than separate audit/normalize scripts: for a clip that passes
   the corrected pre-normalization flags, compute the gain and normalized array in-memory, then
   compute post-normalization stats directly on that in-memory array (no re-read from disk needed).
   This halves the I/O compared to a "normalize, then re-open and re-audit" design.
4. Reuse t0011's flagging/manifest/reporting structure (`code/build_manifest.py`) as the template
   for `code/build_manifest_v2.py`, and its histogram structure (`code/plot_histograms.py`) as the
   template for `code/plot_histograms_v2.py`.

**Alternatives considered**:

* *Reuse t0011's `per_clip_stats.jsonl` values for peak/LUFS/silence/duration and only compute
  `clipped_fraction` in a lightweight second pass.* Rejected: this requires two full audio-decode
  passes instead of one (no time savings since decode is the expensive part, not the metric
  arithmetic — t0011 processed 1557 clips in under 60 seconds), and it creates a hidden runtime
  dependency on another task's output file staying in a specific schema instead of a self-contained,
  auditable script. Recomputing everything fresh from raw audio in one pass is simpler and matches
  the task's instruction to treat t0011's script as a "starting point," not a data source to import
  from at runtime.
* *Delete rejected normalized WAVs from `data/v5_normalized/` after the post-normalization
  re-check.* Rejected: the expected residual-flag count is 0-10 clips (per t0011's estimate), so the
  disk cost of keeping them is negligible, and keeping them lets a future task inspect exactly what
  went wrong with a specific clip without re-running normalization.

**Task type**: `data-analysis` (already set in `task.json`). Per
`meta/task_types/data-analysis/instruction.md`: metrics/chart types are listed upfront (this section
— peak dBFS, LUFS, clipped_fraction, before/after), thresholds are decided during planning (Step
2/3), intermediate data is saved as JSONL/txt (not just charts), and every chart is referenced in
`results_summary.md`/`results_detailed.md` with a description.

## Cost Estimation

**Total: $0.00.** All processing is local CPU compute (`soundfile`, `numpy`, `librosa`,
`pyloudnorm`, `matplotlib` — all already in `pyproject.toml`, confirmed via
`grep -n "pyloudnorm\|soundfile\|librosa\|matplotlib" pyproject.toml`). No paid API calls (no
`anthropic_api`, `openai_api`, or `elevenlabs_api` usage), no remote/GPU compute. Against
`project/budget.json` (`total_budget: 5000.0`, `per_task_default_limit: 100.0`), this task spends $0
of the $5000 total budget and is trivially within the $100 per-task default limit.
`results/ costs.json` will record `{"total_cost_usd": 0}` per REQ-17, matching t0011's precedent
(`tasks/t0011_v5_data_quality_audit/results/costs.json` also records $0).

## Step by Step

### Milestone A — Corrected audit + normalization (Steps 1-3)

1. **[CRITICAL] Pull data and scaffold code.** Run
   `uv run python -m arf.scripts.utils.run_with_logs --task-id t0012_v5_corpus_normalize_and_reaudit -- dvc pull`
   to fetch the v5 train audio (`data/v4/train/wavs/` — 1557 files, referenced by
   `tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt`) and v4 val audio
   (`data/v4/val_ list.txt`, 96 clips). Create `code/__init__.py`, `code/paths.py`, and
   `code/constants.py`:
   * `code/paths.py` (adapted from `tasks/t0011_v5_data_quality_audit/code/paths.py`): define
     `TASK_ROOT`, `DATA_DIR`, `RESULTS_DIR`, `IMAGES_DIR` under this task's own folder;
     `V5_TRAIN_LIST = Path("tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt")`;
     `VAL_96_LIST = Path("data/v4/val_list.txt")` (read-only reference, REQ-19);
     `V5_NORMALIZED_DIR = DATA_DIR / "v5_normalized"`;
     `PER_CLIP_STATS_V2_JSONL = DATA_DIR / "per_clip_stats_v2.jsonl"`;
     `FLAGGED_CLIPS_V2_TXT = DATA_DIR / "flagged_clips_v2.txt"`;
     `CLEAN_MANIFEST_V2_TXT = DATA_DIR / "train_list_v5_normalized_clean.txt"`;
     `T0011_FLAG_COUNTS_JSON = Path("tasks/t0011_v5_data_quality_audit/data/flag_counts.json")`
     (read-only, for the before/after comparison table).
   * `code/constants.py` (adapted from `tasks/t0011_v5_data_quality_audit/code/constants.py`): keep
     `DURATION_MIN_S = 1.5`, `SILENCE_FRACTION_MAX = 0.30`, `LUFS_MIN = -30.0`, `LUFS_MAX = -6.0`,
     `EXPECTED_SAMPLE_RATE = 24000`, `EXPECTED_CHANNELS = 1`, `SHORT_CLIP_LUFS_THRESHOLD_S = 3.0`,
     `EXPECTED_TRAIN_CLIPS = 1557` unchanged. Drop `PEAK_DBFS_MAX` as a flag threshold (keep
     `peak_dbfs` only as a reported field, not a flag). Add
     `CLIPPED_FRACTION_MAX: Final[float] = 0.001` (0.1%, REQ-3) and
     `TARGET_LUFS: Final[float] = -14.0` (REQ-5). Add field-name constants `FIELD_CLIPPED_FRACTION`,
     `FIELD_PRE_PEAK_DBFS`, `FIELD_POST_PEAK_DBFS`, `FIELD_PRE_LUFS`, `FIELD_POST_LUFS`,
     `FIELD_PRE_CLIPPED_FRACTION`, `FIELD_POST_CLIPPED_FRACTION`, `FIELD_POST_SILENCE_FRACTION`,
     `FIELD_NORMALIZED_PATH`. Expected output: `dvc pull` reports 1557 train + 96 val files present
     (or already up to date); `data/v4/train/wavs/` and `data/v4/val*` are populated with real
     audio, not `.dvc` pointers. Satisfies REQ-1, REQ-19 (path wiring).

2. **[CRITICAL] Audit + normalize in one pass.** Create `code/audit_normalize.py`. For each of the
   1557 entries in `V5_TRAIN_LIST` (parsed as `wav_path|phonemes|speaker_id`, same parsing as
   t0011's `_parse_list`):
   * Decode once via `soundfile.read(path, dtype="float32", always_2d=True)`, also read
     `soundfile.info(path)` for `duration`, `samplerate`, `channels`, `subtype`.
   * Compute pre-normalization stats identically to t0011's `compute_clip_stats`: `peak_dbfs`
     (`20*log10(abs(audio).max() + 1e-9)`), `lufs`/`lufs_method` (BS.1770 via `pyloudnorm.Meter` for
     clips >= `SHORT_CLIP_LUFS_THRESHOLD_S`, RMS fallback below that — same as t0011, so the two
     tasks' LUFS numbers stay comparable), `silence_fraction` (via
     `librosa.effects.split(y=mono, top_db=60)`, same as t0011).
   * **Compute `clipped_fraction`** (REQ-2): detect bit depth from `info.subtype` (`PCM_16`→16,
     `PCM_24`→24, `PCM_32`→32, `PCM_S8`→8; any other subtype, including `FLOAT`, defaults to 16 —
     document this fallback in a code comment since float PCM has no fixed LSB).
     `lsb = 1.0 / (2 ** (bits - 1))`; `clip_threshold = 1.0 - lsb`;
     `clipped_fraction = mean(abs(audio_2d.flatten()) >= clip_threshold)`.
   * **Apply the corrected pre-normalization flag set** (REQ-3): flag reasons from
     `clipped_fraction > CLIPPED_FRACTION_MAX` (replaces t0011's `peak_dbfs > PEAK_DBFS_MAX`),
     `silence_fraction > SILENCE_FRACTION_MAX`, `duration_s < DURATION_MIN_S`,
     `sample_rate != EXPECTED_SAMPLE_RATE`, `channels != EXPECTED_CHANNELS`,
     `lufs < LUFS_MIN or lufs > LUFS_MAX`, `error is not None`. Note t0011 did not have a
     `duration_s > 15.0` high-side flag actually fire (0 clips), keep it for parity but it is not
     part of REQ-3's explicit unchanged list — carry it over from t0011's constants unchanged either
     way since the task says "keep t0011's other thresholds unchanged."
   * **If pre-normalization-clean** (REQ-5): compute `gain = 10 ** ((TARGET_LUFS - lufs) / 20)`
     (using whichever `lufs` value was just measured, BS.1770 or RMS-fallback — this is t0011's own
     pseudocode from `results/creative_thinking.md`),
     `normalized = np.clip(audio_2d * gain, -1.0, 1.0)`. Write `normalized` to
     `V5_NORMALIZED_DIR / Path(wav_path).name` via
     `soundfile.write(path, normalized, samplerate, subtype=info.subtype)` (preserve original bit
     depth and the 24kHz rate that passed the pre-check). Then, **without re-reading the file**,
     compute post-normalization stats (`post_peak_dbfs`, `post_lufs` via the same method choice,
     `post_silence_fraction`, `post_clipped_fraction`) directly on the in-memory `normalized` array.
   * **Post-normalization re-check** (REQ-6): if `post_clipped_fraction > CLIPPED_FRACTION_MAX`, add
     flag reason `"clipping_after_normalization"`; if
     `post_silence_fraction > SILENCE_FRACTION_MAX`, add `"silence_after_normalization"`. A clip
     with either post-flag is excluded from the clean manifest in Step 3 even though it was written
     to `data/v5_normalized/`.
   * Write one JSON record per train clip to `data/per_clip_stats_v2.jsonl`
     (`PER_CLIP_STATS_V2_JSONL`) with fields: `wav_path`, `duration_s`, `sample_rate`, `channels`,
     `pre_peak_dbfs`, `pre_lufs`, `lufs_method`, `pre_silence_fraction`, `pre_clipped_fraction`,
     `post_peak_dbfs` (null if not normalized), `post_lufs` (null if not normalized),
     `post_silence_fraction` (null if not normalized), `post_clipped_fraction` (null if not
     normalized), `normalized_path` (null if not normalized), `flags` (list of reason strings, empty
     if final-clean), `error`.
   * **Validation gate** (this step processes 1557 items, over the 100-item threshold): first run
     with `--limit 20` (a CLI flag reading only the first 20 manifest lines). Trivial baseline: 0
     reclassified clips — if 0 of the 20 sampled clips that were in t0011's 224-clip
     `peak_dbfs`-flagged set come back `clipped_fraction`-clean, the fix is doing nothing; STOP and
     debug before running on all 1557. Read 5 individual JSONL records by hand (a mix of
     originally-flagged and originally-clean clips) and confirm: `clipped_fraction` is near 0 for
     clips that are merely peak-normalized (not hard-clipped), `post_lufs` is within about ±1.0 LUFS
     of -14.0 for RMS-fallback clips and tighter for BS.1770 clips, and `normalized_path` points to
     a file that actually exists on disk. Only after this inspection passes, rerun without `--limit`
     for the full 1557 clips. Expected output: `wc -l data/per_clip_stats_v2.jsonl` = 1557;
     `data/v5_normalized/` contains roughly 1500-1557 WAV files (only clips excluded before
     normalization are absent). Satisfies REQ-2, REQ-5, REQ-6, REQ-7.

3. **[CRITICAL] Build the corrected manifest and reconcile against t0011.** Create
   `code/build_manifest_v2.py`. Read `data/per_clip_stats_v2.jsonl` and:
   * Compute `flagged_paths` = clips with any non-empty `flags` list; write
     `data/flagged_clips_v2.txt` (`FLAGGED_CLIPS_V2_TXT`), tab-separated `wav_path\treason1,reason2`
     — same format as t0011's `flagged_clips.txt` (REQ-8).
   * Build `train_list_v5_normalized_clean.txt` (`CLEAN_MANIFEST_V2_TXT`): for every train-manifest
     line not in `flagged_paths`, rewrite the `wav_path` field (the part before the first `|`) to
     point at `data/v5_normalized/<filename>` while keeping the original `phonemes|speaker_id`
     suffix unchanged (REQ-9).
   * **Reclassification report** (REQ-4, REQ-14): load the original 224 peak-flagged clip paths from
     `tasks/t0011_v5_data_quality_audit/data/flagged_clips.txt` (read-only), filtered to
     `reason == "clipping"`. Cross-reference against this task's final flag set: count how many of
     those 224 are now final-clean, and how many still carry `clipping` or
     `clipping_after_normalization` (the answer to Key Question 1 / REQ-14 — compare against t0011's
     estimate of 0-10 residual).
   * **Before/after flag-count comparison** (REQ-11): load
     `tasks/t0011_v5_data_quality_audit/data/flag_counts.json` (read-only) as the "before" row and
     this task's own category counts as the "after" row.
   * **Leak check** (REQ-19): confirm 0 paths from `data/v4/val_list.txt` (96 lines) appear in
     `train_list_v5_normalized_clean.txt`.
   * Write `data/flag_counts_v2.json` with the same shape as t0011's `flag_counts.json` plus a
     `reclassified_from_224` sub-object (`now_clean`, `still_clipped`).
   * Print all counts to stdout (per `data-analysis` guidelines — ratios AND raw counts, e.g.
     "1532/1557 (98.4%)"), so they appear in the `run_with_logs` log.
   * Validation gate: after computing on the full 1557-clip output, sanity-check the final
     clean-manifest count against the trivial no-op baseline of t0011's 1311 clips — if the new
     clean-manifest count is **not greater than 1311**, STOP and debug (the whole point of this task
     is to recover clips t0011 excluded; a result at or below 1311 means the fix regressed or did
     nothing). Expected output: `data/flagged_clips_v2.txt`,
     `data/train_list_v5_normalized_clean.txt`, `data/flag_counts_v2.json` exist; clean-manifest
     count is printed and is > 1311. Satisfies REQ-4, REQ-8, REQ-9, REQ-11, REQ-12, REQ-14, REQ-15,
     REQ-19.

### Milestone B — Visualization, DVC, analysis data (Steps 4-6)

4. **Histograms.** Create `code/plot_histograms_v2.py` (adapted from t0011's
   `code/plot_histograms.py`), reading `data/per_clip_stats_v2.jsonl`. Produce exactly 3 PNGs in
   `results/images/` (REQ-13):
   * `peak_before_after.png` — overlapping histograms of `pre_peak_dbfs` (all 1557 clips) vs
     `post_peak_dbfs` (normalized subset only, nulls excluded), with a vertical line at 0 dBFS.
   * `lufs_before_after.png` — overlapping histograms of `pre_lufs` vs `post_lufs`, with a vertical
     line at `TARGET_LUFS = -14.0`.
   * `clipped_fraction_distribution.png` — overlapping histograms of `pre_clipped_fraction` vs
     `post_clipped_fraction`, with a vertical line at `CLIPPED_FRACTION_MAX = 0.001`, log-scale y
     axis (most values are 0, a log scale makes the tail visible). Every chart must have a title,
     labeled axes, and a legend distinguishing before/after (per `arf/styleguide` and
     `meta/task_types/data-analysis/instruction.md` "Charts without titles or axis labels" pitfall).
     Expected output: 3 PNG files in `results/images/`, each 15-50 KB (matching t0011's chart
     sizes).

5. **DVC-track the normalized audio.** Add `data/v5_normalized/` to this task's `.gitignore` (task-
   local `.gitignore`, not the repo-root one — only `pyproject.toml`, `uv.lock`, `ruff.toml`, and
   `.gitignore` may be touched outside the task folder per Key Rule 3, and this is a new file inside
   the task folder, so no rule conflict). Run, wrapped in `run_with_logs`:
   `dvc add tasks/t0012_v5_corpus_normalize_and_reaudit/data/v5_normalized/` then `dvc push`
   (REQ-10). This is a destructive-adjacent step only in the sense that `dvc add` rewrites the
   `.dvc` pointer file — it is safely re-runnable (re-running `dvc add` on an unchanged directory is
   a no-op; re-running after Step 2 regenerates the same content deterministically since the
   normalization pipeline has no random seed). Recovery: if `dvc push` fails partway (network),
   re-run `dvc push` — it resumes from the last successfully uploaded object. Expected output:
   `tasks/t0012_v5_corpus_normalize_and_reaudit/data/v5_normalized.dvc` created; `git status` shows
   the `.dvc` file staged and `data/v5_normalized/` itself ignored; `dvc push` exits 0.

6. **Compute and persist the analysis data.** Extend `code/build_manifest_v2.py` with a
   `_compute_metric_stats`/`_percentile` pair (copied from t0011's `code/build_manifest.py`) to
   compute pre- vs post-normalization `peak_dbfs`/`lufs`/`clipped_fraction` percentiles, and a
   cross-tabulation for Key Question 4 (REQ-16): of the clips originally flagged `duration_low` or
   `silence` in t0011 (25 + 1), how many carry the same flag in this task's final flag set versus
   how many are now clean. Note in a code comment (and print to stdout) that gain is a uniform
   per-sample scalar, so `duration_s` cannot change, and `silence_fraction` (measured via
   `librosa.effects.split`, a *relative* dB threshold) is expected to be gain-invariant — this step
   confirms that expectation numerically rather than asserting it. Write everything this step and
   Step 3 computed — flag counts before/after, reclassification counts, distribution percentiles,
   the Key Question 4 cross-tab — into `data/analysis_v2.json` (one consolidated JSON, superset of
   `data/flag_counts_v2.json`) and print a human-readable summary to stdout so it appears in the
   `run_with_logs` log. This is the last implementation step; the human-readable results write-up
   that embeds the 3 charts from Step 4 and narrates `data/analysis_v2.json`'s contents is produced
   by the orchestrator's results stage in `execute-task`, not by this task's own code (per the
   planning skill's Forbidden list — this plan's Step by Step ends at implementation work). Expected
   output: `data/analysis_v2.json` exists and contains every count required by REQ-4, REQ-11,
   REQ-12, REQ-14, REQ-15, REQ-16.

**Metrics and cost note**: none of the 3 registered metrics (`rtf`, `speaker_sim`, `ttfb_ms` — per
`tasks/t0012_v5_corpus_normalize_and_reaudit/ctx/metrics.json`) apply to a corpus
normalization/audit task, since no synthesis or inference runs here — this is a deliberate omission,
not an oversight. Total cost is $0 (Cost Estimation section above); no paid API or GPU usage occurs
in any step. Recording these facts into their respective result files is an orchestrator-managed
step (see Forbidden list in the planning skill), not part of this plan's Step by Step.

## Remote Machines

None required. This task is CPU-only local processing (`soundfile`, `numpy`, `librosa`,
`pyloudnorm`, `matplotlib` on ~1557 short WAV clips), matching t0011's precedent which needed under
60 seconds for a comparable per-clip pass. No GPU pool contention with
`t0010_stage2_safeguarded_ training`, which is running concurrently on `LLM-T1-NC80`.

## Assets Needed

* **v5 train manifest + audio** (from `t0003_kokoro_v5_phoneme_data`, via DVC):
  `tasks/t0003_kokoro_v5_phoneme_data/results/v5/train_list.txt` (1557 lines) and the WAVs it
  references under `data/v4/train/wavs/` (DVC-tracked, `dvc pull` required).
* **v4 val manifest** (project corpus): `data/v4/val_list.txt` (96 clips) — read-only, for the leak
  check only (REQ-19); never normalized or modified.
* **t0011 outputs** (from `t0011_v5_data_quality_audit`, read-only, never modified):
  `data/per_clip_ stats.jsonl`, `data/flag_counts.json`, `data/flagged_clips.txt`,
  `code/audit_audio.py`, `code/build_manifest.py`, `code/plot_histograms.py`, `code/constants.py`,
  `code/paths.py`, `results/creative_thinking.md` (source of the normalization pseudocode reused in
  Step 2), `results/suggestions.json` (source of S-0011-01 and S-0011-03's exact text and
  estimates).
* **Libraries** (already declared in `pyproject.toml`, confirmed present): `soundfile>=0.12`,
  `librosa>=0.10`, `pyloudnorm>=0.1`, `matplotlib>=3.0`, `numpy>=2.0`.

## Expected Assets

`task.json` declares `expected_assets: {}` — matching t0011's own precedent
(`tasks/t0011_v5_data_quality_audit/task.json` also has `expected_assets: {}`). This task does not
register a formal `meta/asset_types/` asset (no `dataset`, `library`, or `model` asset ID). Its
outputs are task-local files consumed directly by the future full-corpus Stage 2 training task
(tracked separately as suggestion S-0011-02):

* `data/per_clip_stats_v2.jsonl` — per-clip pre/post-normalization stats for all 1557 train clips.
* `data/v5_normalized/` — DVC-tracked normalized WAV audio (`.dvc` pointer committed to git; raw
  audio pushed to `azure://ml-dvc-datasets/datasets/rail-arf-tts`).
* `data/flagged_clips_v2.txt` — final excluded-clip list with reasons.
* `data/train_list_v5_normalized_clean.txt` — final clean manifest for the next Stage 2 task.
* `data/flag_counts_v2.json` — flag category counts plus reclassification counts.
* `results/results_detailed.md`, `results/images/peak_before_after.png`,
  `results/images/lufs_before_after.png`, `results/images/clipped_fraction_distribution.png`,
  `results/costs.json`.

## Time Estimation

* Setup + `dvc pull` (Step 1): ~5 minutes (network-dependent; files may already be cached locally).
* Audit + normalize pass, `--limit 20` validation then full 1557 (Step 2): ~5-10 minutes (t0011's
  comparable per-clip decode pass over 1557 clips took under 60 seconds; this step adds a
  normalization write and a second in-memory stats computation per clean clip, plus manual
  inspection time for the validation gate).
* Manifest build + reconciliation (Step 3): under 1 minute (pure JSONL/text processing).
* Histograms (Step 4): under 1 minute.
* `dvc add` + `dvc push` (Step 5): ~5-10 minutes, network-dependent on uploading up to ~500 MB of
  normalized WAVs to Azure Blob (per t0011's creative-thinking disk-cost estimate).
* Results writing + style/lint checks (Steps 6-7, plus `ruff`/`mypy`/`flowmark`): ~10 minutes.
* **Total**: approximately 30-40 minutes wall clock, well under t0011's ~30-minute run despite the
  added normalization and DVC-push work, since both tasks operate on the same 1557-clip corpus with
  comparable per-clip cost.

## Risks & Fallbacks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `dvc pull` is incomplete or partial (network interruption), silently truncating the 1557-clip corpus | Low | High — would silently shrink the corpus and skew every count | Reuse t0011's own guard: abort with a nonzero exit if parsed train entries < 90% of `EXPECTED_TRAIN_CLIPS` (1557), matching t0011's `MIN_PULL_FRACTION` check. Re-run `dvc pull` and verify `wc -l` before proceeding. |
| Bit-depth detection for `clipped_fraction` is wrong for a subtype not seen in preflight inspection (e.g., corpus is actually 24-bit, but the fallback assumes 16-bit), causing `clipped_fraction` to systematically over- or under-count | Medium | Medium — could mis-classify some clips as clipped/not-clipped | Step 2's validation gate requires reading 5 individual records by hand before the full run, including printing the detected `subtype`/bit-depth for those 5 files; if the detected subtype looks wrong, fix the mapping in `code/constants.py` before the full run. |
| LUFS normalization pushes a clip that was already near the loud end into new clipping in large numbers (not just 0-10 edge cases) | Low | Medium — would shrink the clean manifest below the ~1556-clip target from S-0011-01 | This is exactly what Step 2's post-normalization re-check (REQ-6) and the `>1311` validation gate in Step 3 catch — if the final clean count is not strictly greater than t0011's 1311-clip baseline, the step 3 gate halts and requires debugging before the task can be marked done. |
| `dvc push` fails or times out uploading ~500 MB of normalized audio, leaving the `.dvc` pointer committed but the blob store out of sync | Low | High — would break `dvc pull` for any teammate/future task consuming this manifest | Per repo DVC rules, `dvc push` is idempotent/resumable; re-run it. If it fails repeatedly, do not mark the task complete — this is a hard blocker per CLAUDE.md's DVC workflow rules, not something to work around by committing raw audio to git. |
| The 96-clip val set gets accidentally included in the normalization loop or the clean manifest (e.g., a path-matching bug in the leak check) | Low | High — would violate the project's hard "NEVER train on val_96" rule and silently contaminate the regression baseline | Step 1 keeps `VAL_96_LIST` used only for the Step 3 leak check (path membership test against the val list), never passed into `code/audit_normalize.py`'s per-clip loop at all. The leak check in Step 3 explicitly asserts 0 overlap and is part of REQ-19's verification. |

## Verification Criteria

* `wc -l tasks/t0012_v5_corpus_normalize_and_reaudit/data/per_clip_stats_v2.jsonl` outputs `1557`
  (REQ-7).
* `uv run python -u -m arf.scripts.verificators.verify_plan t0012_v5_corpus_normalize_and_reaudit`
  exits 0 with zero errors (plan structure check).
* After implementation, run
  `uv run python3 -c " import json clean = open('tasks/t0012_v5_corpus_normalize_and_reaudit/data/train_list_v5_normalized_clean.txt').read().splitlines() val = {l.split('|')[0] for l in open('data/v4/val_list.txt').read().splitlines() if l.strip()} clean_paths = {l.split('|')[0] for l in clean} assert len(clean) > 1311, f'clean manifest {len(clean)} not > t0011 baseline 1311' assert not (clean_paths & val), 'val_96 leak detected in clean manifest' print(f'OK: {len(clean)}/1557 clean, 0 val leaks') "`
  — expected output `OK: <N>/1557 clean, 0 val leaks` with `N > 1311` (REQ-12, REQ-19).
* `ls tasks/t0012_v5_corpus_normalize_and_reaudit/results/images/*.png | wc -l` outputs `3`
  (REQ-13).
* `uv run ruff check --fix tasks/t0012_v5_corpus_normalize_and_reaudit/code/ && uv run ruff format tasks/t0012_v5_corpus_normalize_and_reaudit/code/ && uv run mypy tasks/t0012_v5_corpus_normalize_and_reaudit/code/`
  all exit 0 with no remaining errors (Key Rule 6).
* `dvc status` (after `dvc add`/`dvc push`) shows `data/v5_normalized/` as up to date with the
  remote, confirming REQ-10.
* Grep every `REQ-` ID referenced in this plan's Step by Step section against the Task Requirement
  Checklist:
  `grep -o 'REQ-[0-9]*' tasks/t0012_v5_corpus_normalize_and_reaudit/plan/plan.md | sort -u | wc -l`
  outputs `19`, confirming all 19 checklist items are referenced somewhere in the execution steps
  (requirement-coverage check).

## Rejection Criteria

This task produces a data manifest, not a paired benchmark comparison, so Lesson 3's
`successful_requests / total_requests < 0.8` rule does not apply verbatim (there are no "requests").
The equivalent pre-registered rejection conditions for this task:

* If `dvc pull` recovers fewer than 90% of the expected 1557 train clips (matching t0011's
  `MIN_PULL_FRACTION = 0.90` guard), the run is **null** — do not report clean-manifest counts
  computed on a truncated corpus.
* If the final clean-manifest count is **not strictly greater than 1311** (t0011's baseline), the
  result is **null** — this task's entire purpose is to recover clips t0011 excluded, so a result at
  or below the baseline indicates a broken pipeline, not a valid negative finding, and must be
  debugged rather than reported as "normalization did not help."
* If any clip from `data/v4/val_list.txt` (the 96-clip held-out set) appears in
  `data/v5_normalized/` or `train_list_v5_normalized_clean.txt`, the entire manifest output is
  **null** and must not be used by any downstream training task until the leak is fixed — this is a
  hard project rule (CLAUDE.md: "NEVER train on val_96"), not a soft quality concern.

## Verification Additions (data-analysis task type)

Per `meta/task_types/data-analysis/instruction.md`:

* At least one chart exists in `results/images/` — 3 exist (Step 4).
* `results/metrics.json` contains only registered metric keys — it will be `{}` since no registered
  metric (`rtf`, `speaker_sim`, `ttfb_ms`) applies (Step 7).
* `results/results_summary.md` references all generated charts — enforced by Step 6's embed
  requirement (this file is orchestrator-managed per the plan spec, but its content depends on Step
  6's `results_detailed.md` being complete and chart-referencing).
* No hardcoded file paths in analysis scripts — all paths come from `code/paths.py` constants (Step
  1).
