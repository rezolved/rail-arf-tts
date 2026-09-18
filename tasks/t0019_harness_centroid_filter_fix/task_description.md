# Fix the TTS Harness Centroid Filter and Re-establish the speaker_sim Baseline

## Motivation

The project's primary metric, `speaker_sim`, is computed by the `tts_eval_harness` library from
t0008 against a GE2E centroid of the ElevenLabs David reference corpus. t0016 found that calling
t0008's own `code/scoring.py` `build_centroid()` on the freshly pulled `data/11labs_david/` corpus
raises `RuntimeError: No clips long enough to embed`: the filter `MIN_CLIP_DURATION_S = 1.6` rejects
every one of the 1364 clips (mean duration about 1.04 s). t0016's own local re-score of the shipped
v3 bundle therefore had to bypass the filter and landed at 0.566 instead of t0008's recorded 0.631,
and nobody can say today which number is right.

Meanwhile t0018 ran the same harness on the VM and did reproduce ElevenLabs' self-consistency (0.832
fillers / 0.792 val96), so either the VM saw a different corpus, or the filter was never binding
there, or the two runs computed different things. Until this is settled, every `speaker_sim`
comparison across tasks is suspect, and the owner has decided the Kokoro training line continues, so
the harness must be trustworthy before t0017 and t0020 spend GPU money on it.

A second, smaller harness defect surfaced in t0018's listening guide: the three fixed gate texts
(`lining_up_suggestions_17`, `lining_up_suggestions_10`, `putting_them_head_to_head_15`) that every
training task synthesizes for the audible-speech gate are not among the 100 filler prompts the
harness samples, so no harness run ever scores them and the listening guides have empty rows for
them.

## Key Questions

1. Did the `11labs_david` corpus change between t0008 (2026-09-14) and today (re-segmentation,
   normalization, a different DVC revision), or was the 1.6 s filter never binding in t0008's
   environment? Compare the DVC hash and per-clip duration distribution t0008 recorded (or can be
   reconstructed from `per_clip_metrics.json`) against the current pull, and against whatever the VM
   had during t0018.
2. What is resemblyzer's actual minimum embeddable length, and what does the embedding do on 1 s
   clips: fail, degrade, or work fine? Measure, do not assume.
3. With the filter fixed, what are the same-session `speaker_sim` numbers for `elevenlabs_david`
   (half-A centroid vs half-B), `kokoro_v3_bundle`, and `kokoro_base_v3_voicepack` on both prompt
   sets, and do they match t0008's recorded 0.832/0.792, 0.631/0.588, 0.603/0.582 within 0.02?
4. If they do not match, which of t0008's or t0016's numbers is the artifact, and what is the
   corrected baseline table every later task must cite?

## Scope

### 1. Diagnose

Reproduce the failure locally on CPU. Record the current corpus's DVC hash and duration histogram.
Recover t0008's view of the corpus from its committed artifacts and logs. Test resemblyzer on 0.5 s,
1.0 s, 1.5 s, 2.0 s, 3.0 s clips (real David clips, cropped) and report embedding stability (cosine
of the cropped clip's embedding against the full clip's).

### 2. Fix

Ship the harness as a new library asset `tts_eval_harness` version `0.2.0` in this task's
`assets/library/` with the code copied from t0008 and changed only where needed:

* Replace the hard `MIN_CLIP_DURATION_S` rejection with the empirically justified rule from step 1
  (lower threshold, zero-padding to resemblyzer's minimum, or concatenating short reference clips
  for the centroid). Whatever the rule, `build_centroid()` must never silently drop more than a
  logged fraction of the corpus, and must raise with the count and threshold when it does.
* Add the three fixed gate texts to the harness's prompt sets as a third, always-included set
  `gate_texts`, so every run scores them and listening guides have no empty rows.
* Keep every public function signature from t0008 unchanged so t0017/t0020/t0021 code written
  against t0008's API keeps working. Add unit tests for the new rule next to t0008's eleven.

Write a `corrections/` entry that marks t0008's library asset as replaced by this task's version
(correction type `replace`), so aggregators resolve `tts_eval_harness` to the fixed code.

### 3. Re-baseline

Run the fixed harness on the three systems in Key Question 3 on both prompt sets plus the gate
texts, CPU or a short GPU session, same session for all three. Publish the corrected baseline table
in `results/results_summary.md` and, if any number moves by more than 0.02 from t0008's, file a
`corrections/` `update` for t0008's `results/metrics.json` values with the explanation, rather than
leaving two contradicting tables in the repo.

## Expected Outputs

* `assets/library/tts_eval_harness/` version 0.2.0 with `description.md` documenting the exact
  change, and `corrections/` entries replacing t0008's library asset (and updating its metrics if
  needed).
* `results/filter_diagnosis.md` — corpus-change verdict, resemblyzer minimum-length measurements,
  the chosen rule and why.
* `results/metrics.json` — variants for the three systems × {fillers, val96, gate_texts}, with
  `speaker_sim`, `ttfb_ms`, `rtf`.
* `results/per_clip_metrics.json`.
* `results/images/duration_histogram_then_vs_now.png` (x: clip duration s, y: count, two series:
  t0008's corpus view and today's; answers Q1), `results/images/embedding_stability_vs_length.png`
  (x: crop length s, y: cosine to full-clip embedding; Q2),
  `results/images/baseline_then_vs_now.png` (grouped bars per system: t0008 recorded vs this task;
  Q3).
* `results/audio_samples/` (DVC) with the gate-text synthesis for the two Kokoro systems and the
  ElevenLabs originals, plus `results/listening_guide.md`.
* `results/suggestions.json`.

## Rejection criteria

* No number may be reported as "matches t0008" unless it was produced by the fixed harness in this
  task's own session; quoting t0008's stored values is allowed only in the comparison column.
* If resemblyzer genuinely cannot embed the filler corpus reliably, say so and propose the
  replacement (for example WavLM-based speaker verification) as a suggestion; do not paper over it
  with a permissive threshold.

## Compute and budget

* Diagnosis and the fix are CPU-only. Re-baselining Kokoro synthesis on CPU is slow but feasible
  (t0016 did 8 clips); if a full 196-prompt run per system is needed, use `LLM-T1-NC80` for at most
  1 h (about $14). Hard cap: **$20**.

## Dependencies

* `t0008_tts_eval_harness_baselines` — the library being corrected and the baseline numbers being
  re-established.
* `t0016_v3_recipe_recovery` — the reproduction of the failure and the 0.566 cross-check that
  motivates this task.
