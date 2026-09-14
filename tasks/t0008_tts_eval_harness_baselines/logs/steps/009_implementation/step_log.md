---
spec_version: "3"
task_id: "t0008_tts_eval_harness_baselines"
step_number: 9
step_name: "implementation"
status: "completed"
started_at: "2026-09-14T16:23:38Z"
completed_at: "2026-09-14T17:42:00Z"
---
## Summary

Built the `tts_eval_harness` reusable library and ran a full baseline evaluation of 8 TTS systems
(ElevenLabs David + 7 Kokoro variants) on two prompt sets (96 val96 + 100 filler prompts) on
LLM-T1-NC80 (2×H100 NVL). All 1568 per-clip records scored for speaker_sim (GE2E cosine GPU), WER
(faster-whisper base.en CPU), and duration ratio. Library verificator and all 11 unit tests pass.
Task results verificator passes with 0 errors.

## Actions Taken

1. Wrote core library modules: `constants.py`, `paths.py`, `adapters.py`, `scoring.py`,
   `harness.py`, `extract_decoder.py`, `report.py`, `run_eval.py`, `score_speaker_sim.py`,
   `prepare_prompts.py`.
2. Fixed Kokoro 0.9.x / Python 3.11 tensor-to-numpy incompatibility in `adapters.py` (`torch.Tensor`
   → `np.ndarray` via `.detach().cpu().numpy()`).
3. Packaged t0005_run06_epoch3.pth and t0006_v6d_epoch6.pth as five-module checkpoints via
   `extract_decoder.py`. Stored provenance in `data/packaged/metadata.json`.
4. Generated 1364 ElevenLabs David corpus clips via API (62 phrases × 22 reps). Stored in
   `data/11labs_david/`. Split 679/685 seed=42: half-A → GE2E centroid, half-B → ElevenLabs
   self-scoring.
5. Generated 100 filler prompt manifests (`data/filler_prompts_100.json`) and 96 val96 manifests
   (`data/val96_prompts.json`).
6. Ran ElevenLabs synthesis timing evaluation (196 prompts) — saved to `data/synth_audio/`.
7. Ran Kokoro synthesis for all 7 variants on both prompt sets on LLM-T1-NC80 (GPU). Fixed
   `HF_HOME=/mnt/cache/persist/hf-cache` for model downloads.
8. Fixed resemblyzer short-clip guard (was excluding all filler clips <1.6s — incorrect; removed).
   Added GPU acceleration for VoiceEncoder: 68ms/clip vs 9100ms/clip on CPU (133× speedup).
9. Fixed matplotlib `cm.get_cmap` removal in 3.11: changed to `plt.get_cmap("tab10")`.
10. Ran `score_speaker_sim.py` on resemblyzer venv (GPU): scored all 1568 records for speaker_sim
    and WER. Merged results into `results/per_clip_metrics.json`.
11. Ran `report.py` to produce `results/metrics.json` (16 explicit-format variants),
    `results/tables.json`, and 3 PNG charts.
12. Wrote `results/results_summary.md` (spec_version "1") and `results/results_detailed.md`
    (spec_version "2") with full REQ-1 through REQ-21 coverage.
13. Wrote `results/metadata.json`, `results/costs.json`, `results/remote_machines_used.json`.
14. Rewrote library description.md with all 8 mandatory sections; `* ` bullets in Main Ideas.
15. Wrote `test_harness.py` with 11 unit tests: TestDurationRatio×3, TestWer×5, TestReportJson×3.
16. Ran ruff/mypy on all code; ran flowmark on all markdown files.
17. DVC-tracked `data/11labs_david`, `data/synth_audio`, `data/packaged/*.pth`; pushed to Azure Blob
    (`azure://ml-dvc-datasets/datasets/rail-arf-tts`).
18. Called VM teardown at 17:42Z; VM stopped, cost $27.92 (2h at $13.96/hr).

## Outputs

* `tasks/t0008_tts_eval_harness_baselines/code/` — 13 modules (adapters, scoring, harness, report,
  extract_decoder, run_eval, score_speaker_sim, prepare_prompts, paths, constants, test)
* `tasks/t0008_tts_eval_harness_baselines/assets/library/tts_eval_harness/description.md` — library
  v0.1.0, verificator PASSED
* `tasks/t0008_tts_eval_harness_baselines/results/per_clip_metrics.json` — 1568 records
* `tasks/t0008_tts_eval_harness_baselines/results/metrics.json` — 16 explicit-format variants
* `tasks/t0008_tts_eval_harness_baselines/results/tables.json` — metrics with deltas
* `tasks/t0008_tts_eval_harness_baselines/results/images/` — 3 PNGs
* `tasks/t0008_tts_eval_harness_baselines/results/centroid.npy` — 679-clip GE2E centroid
* `tasks/t0008_tts_eval_harness_baselines/results/half_b_paths.json` — 685 half-B clip paths
* `tasks/t0008_tts_eval_harness_baselines/results/metadata.json`, `costs.json`,
  `remote_machines_used.json`
* `tasks/t0008_tts_eval_harness_baselines/results/results_summary.md`, `results_detailed.md`
* `tasks/t0008_tts_eval_harness_baselines/data/11labs_david.dvc`, `data/synth_audio.dvc`,
  `data/packaged/t0005_run06_epoch3.pth.dvc`, `data/packaged/t0006_v6d_epoch6.pth.dvc`

## Issues

* **Resemblyzer short clips**: 0.75–1.5s filler clips were being excluded by a `<1.6s` guard in the
  speaker_sim loop — removed, as resemblyzer pads them automatically. Affected speaker_sim would
  have been zero for all filler clips without this fix.
* **Resemblyzer GPU**: VoiceEncoder defaulted to CPU (9100ms/clip). Added
  `VoiceEncoder(device="cuda")` — 68ms/clip on H100, 133× speedup.
* **Kokoro torch.Tensor**: kokoro 0.9.x on Python 3.11 returns `torch.Tensor` from the pipeline
  iterator instead of `np.ndarray` — fixed with `.detach().cpu().numpy()` in all adapters.
* **faster-whisper cudaErrorInvalidDevice**: dual-H100 CUDA enumeration issue; used
  `device="cpu", compute_type="int8"` for WER scoring instead.
* **matplotlib 3.11 cm.get_cmap removed**: changed to `plt.get_cmap("tab10")`.
* **t0005_best explosions**: epoch-3 not converged; produces duration_ratio > 5.0 for most clips.
  Expected finding, not a harness bug.
* **DVC push authorization**: `AuthorizationPermissionMismatch` on initial push; retried with
  `AZURE_STORAGE_AUTH_MODE=login` env var.

## Key Results

* ElevenLabs David: speaker_sim=0.832 (fillers) / 0.792 (val96), TTFB_p50=132ms / 153ms
* kokoro_v3_bundle (best Kokoro sim): 0.631 / 0.588, TTFB=185ms / 282ms
* kokoro_t0006_v6d (best TTFB): 132ms (=ElevenLabs), speaker_sim=0.601 / 0.482
* kokoro_t0005_best: not viable — duration explosions, nan speaker_sim
* Gap to 0.85 target: ~0.20 GE2E cosine units for best Kokoro variant
