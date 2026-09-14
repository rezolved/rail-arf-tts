---
spec_version: "3"
task_id: "t0008_tts_eval_harness_baselines"
step_number: 6
step_name: "research-code"
status: "completed"
started_at: "2026-09-14T15:48:28Z"
completed_at: "2026-09-14T15:53:22Z"
---
## Summary

Reviewed all 7 completed tasks (t0001–t0007) and 0 registered libraries to produce
`research/research_code.md`. Key findings: `extract_decoder_generic.py` in t0002 is the only tool
for packaging raw StyleTTS2 checkpoints as five-module Kokoro bundles; `build_pipeline.py` in t0003
is mandatory for all Kokoro synthesis; t0005 epoch 3 and t0006 v6d epoch 6 are the DVC-tracked
checkpoints to score; all checkpoint binaries require `dvc pull` before evaluation.

## Actions Taken

1. Ran library aggregator — confirmed zero registered libraries across all prior tasks.
2. Ran answer aggregator — confirmed zero answer assets.
3. Ran task aggregator for all 7 completed tasks; fetched full detail for the 4 most relevant
   (t0002, t0003, t0005, t0006).
4. Read source files for key reusable code: `extract_decoder_generic.py` (38 lines),
   `build_pipeline.py` (30 lines), `constants.py` (55 lines), `lexicon.py` (93 lines),
   `test_kokoro_inference_generic.py` (56 lines), `prepare_v5_data.py:99-140`.
5. Inspected DVC pointer files for t0005 and t0006 checkpoints and v3 reference bundle to confirm
   file names, sizes, and MD5s.
6. Reviewed LESSONS.md for latency/warmup constraints (Lessons 1, 4, 10).
7. Checked val_list.txt format and filler corpus dataset overview files.
8. Wrote `research/research_code.md` with 7 mandatory sections, 4 Key Findings subsections, Dataset
   Landscape, and 5-entry Task Index; verificator passed with zero errors/warnings.

## Outputs

* `tasks/t0008_tts_eval_harness_baselines/research/research_code.md` — code research document

## Issues

No issues encountered. The verificator passed with zero errors and zero warnings.
