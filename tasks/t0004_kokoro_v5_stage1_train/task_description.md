# t0004 — Kokoro v5 Stage 1 fine-tune

## Objective

Run StyleTTS2 Stage 1 on t0003's phoneme-corrected corpus (`data/v5/`), producing a
`first_stage.pth` to hand off to Stage 2. Stage 1 must be redone from scratch — t0001's Stage 1 was
trained on the same corrupted grapheme/phoneme mixture as its Stage 2, so its checkpoint cannot be
reused (see t0003's Stage 1 duration probe: it explodes exactly like v4's).

## Background

t0001's Stage 2 run diverged; t0003 traced the likely cause to 21% of the training manifest holding
raw text instead of phonemes. t0003 rebuilt clean manifests but did not train anything. This task is
the actual run, plus the operational tooling t0001 lacked from the start: a sync script that also
prunes old checkpoints (both locally and on the ephemeral VM disk) and shows epoch progress, so a
long run doesn't need constant manual log-tailing.

## Approach

1. Provision `/mnt/kikiri-tts` on LLM-T1-NC80: clone the `semidark/StyleTTS2` and `semidark/kokoro`
   submodules, build a venv, upload the v4 wavs and v5 manifests.
2. Launch `train_first.py` against `config_david_v5.yml` (10 epochs, matching v3's schedule).
3. Run `code/sync_and_monitor.sh` locally: polls the VM every 30s, rsyncs the log and any new
   `epoch_1st_*.pth`, prints an epoch progress bar, and prunes to the top-2 checkpoints by
   `val_loss` — on both the local copy and the VM (the VM disk is ephemeral, but there's no reason
   to let 10 epochs of checkpoints pile up mid-run either).
4. Stop and report if the log shows a crash; otherwise stop when the configured final epoch is
   reached.

## Environment setup notes (for whoever runs this next)

Getting `train_first.py` to launch on a fresh VM took five sequential fixes, none related to the
data or config — pure environment gaps in the from-scratch `/mnt/kikiri-tts` setup:

1. `pandas` was missing from the initial pip install list (`meldataset.py` imports it directly).
2. `tensorboard` was missing (`torch.utils.tensorboard.SummaryWriter`).
3. The pip package `monotonic_align` only ships `maximum_path` — `utils.py` also imports
   `mask_from_lens`, which the real StyleTTS2 repo defines itself but this pip package doesn't.
   Patched it directly into the installed package
   (`venv/lib/python3.10/site-packages/monotonic_align/__init__.py`).
4. A stray `/mnt/hf_home_cache` default (not from any visible env var or dotfile — never fully
   traced) is not writable; launch with `HF_HOME`/`HF_HUB_CACHE` pointed at
   `/mnt/kikiri-tts/hf_cache` explicitly.
5. `transformers==5.17.0` refuses `torch.load` on torch < 2.6 for non-safetensors checkpoints
   (`microsoft/wavlm-base-plus`, loaded unconditionally by `WavLMLoss.__init__` even though
   `lambda_slm: 0.0` means the loss is never used). Downgraded to `transformers==4.46.3`, which
   predates the check.

## Verification criteria

- `train_first.py` completes 10 epochs without a traceback.
- **Stop-early signal, carried over from t0003**: this is Stage 1, so Dur/CE loss aren't logged yet
  — the real check is at Stage 2's first step (Dur ~0.8 = healthy, 8-17 = the data hypothesis is
  wrong). Recorded here for continuity into the next task.
- Exactly 2 checkpoints survive locally and on the VM at any point after epoch 3.
