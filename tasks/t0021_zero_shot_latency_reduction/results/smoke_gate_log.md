# Smoke Gate Log

Per-system single-clip smoke gate result before any warmup/measurement began (plan.md Step 5, Lesson
2: never spend GPU time on the full warmup/measurement protocol before confirming the pipeline
actually produces audio end to end).

| System | Smoke Gate (`--limit 1`) | Notes |
| --- | --- | --- |
| cosyvoice2 | PASS | `baseline_new_ref` / `ref_single` / fillers, 1/1 successful. Confirmed `StageTiming` fields sum to approximately the total wall time before scaling to the full sweep. |
| chatterbox | PASS (after one code fix) | First attempt failed with `ImportError: cannot import name 'drop_invalid_tokens' from 'chatterbox.models.utils'` -- the function actually lives in `chatterbox.models.s3tokenizer` at the pinned `chatterbox-tts==0.1.7` version, found by reading the installed package's `tts.py` directly. Fixed in `code/adapters_zeroshot.py`; re-run passed 1/1. |
| f5_tts | FAIL (hung, not a code bug) | Not part of the smoke-gate-before-sweep protocol (F5-TTS has no acceleration-variant sweep, only the S-0018-01 closure item). The dedicated retry hung inside `F5TTS.__init__`'s `datasets`-package import chain; confirmed via a `py-spy dump` stack trace to be blocked reading a source file from the VM's Azure Files SMB mount. Marked null. See `intervention/f5_tts_retry_still_hangs.md`. |

Both CosyVoice2 and Chatterbox passed their smoke gate before any of Milestone 3's 50-warmup /
196-measured-prompt runs began, satisfying Lesson 2 and the REQ-7 protocol requirement. F5-TTS is a
cheap closure item (S-0018-01), not a variant-sweep target, so its smoke-gate-before-sweep step does
not apply to it the same way; its own retry (also required by the protocol before any measured run)
failed as recorded above and in `intervention/f5_tts_retry_still_hangs.md`.

## Additional bug found during the sweep itself (not a smoke-gate failure)

`chatterbox`'s `precision_bf16_or_fp16` variant initially failed with
`RuntimeError: Unsupported dtype BFloat16` inside `s3gen.embed_ref()`'s xvector mel-frontend
(`torchaudio`'s `Kaldi.fbank` calls `torch.fft.rfft`, which supports only float32/float64 on the
pinned torch/torchaudio versions -- this applies to fp16 as well as bf16, confirmed by trying both
live). Fixed by casting only `model.t3` (the T3 LM decoder) to the reduced-precision dtype and
leaving `model.s3gen` at fp32; the variant then ran successfully end to end
(`precision_dtype_used: "bf16"` in `results/environment.json`). See `results/tables.json`'s notes
field and `code/run_eval_zeroshot.py`'s inline comment for the full explanation.
