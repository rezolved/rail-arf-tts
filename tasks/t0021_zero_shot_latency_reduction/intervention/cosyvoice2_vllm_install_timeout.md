# CosyVoice2 vLLM backend install: 20-minute cutoff hit, `vllm_backend` variant null

## What happened

During the `setup-machines` step, the plan's Milestone 2 Step 6 called for creating a new, separate
isolated venv `.venv-cosyvoice2-vllm` on `LLM-T1-NC80` and installing
`vllm==0.11.0`/`transformers==4.57.1`/`numpy==1.26.4` there (the dependency set the
`swulling/CosyVoice2-0.5B-vllm` backend needs), fully separate from `.venv-cosyvoice2`'s pinned
`torch==2.3.1+cu121`. The plan pre-authorized a 20-minute cutoff for this attempt.

The install was launched at 2026-09-18T11:54:52Z under `timeout 1100` (1100 s ≈ 18.3 minutes, inside
the plan's 20-minute cap) via:

```bash
cd /mnt/cache/persist/t0021_zero_shot_latency_reduction
python3 -m venv venvs/.venv-cosyvoice2-vllm
./venvs/.venv-cosyvoice2-vllm/bin/pip install --upgrade pip -q
timeout 1100 ./venvs/.venv-cosyvoice2-vllm/bin/pip install 'vllm==0.11.0' 'transformers==4.57.1' 'numpy==1.26.4' \
  > logs/install_cosyvoice2_vllm.log 2>&1
```

All dependency wheels (including the large `torch`, `xformers`, and `vllm` wheels) downloaded
successfully — `logs/install_cosyvoice2_vllm.log` shows every `Downloading ...` line completing with
a 100% progress bar and no network errors. The log then shows pip enter its final
`Installing collected packages: ...` phase (unpacking/copying ~140 packages, `torch` appears roughly
two-thirds through the alphabetical-by-dependency-order list, `vllm` itself is last). The
`timeout 1100` wrapper killed the process before this phase finished: the install process (PID
11690, and its `python3 -m venv`/`pip` parent PID 11689) was confirmed alive at 12:09:58Z (during a
`kill -0` poll) and confirmed gone at 12:13:18Z, consistent with the 1100 s cap firing at
approximately 12:12:52Z (11:54:52Z + 1100 s). No traceback, no `Error`/`Killed` line, and no
`PIP_VLLM_EXIT:<code>` marker appears anywhere in the 520-line log — the log simply stops mid
`Installing collected packages` with no further output, which is the expected signature of `timeout`
SIGTERM-ing the whole process group rather than pip itself raising an error.

## Verification of partial state

* `venvs/.venv-cosyvoice2-vllm/lib/python3.10/site-packages/` contains `torch` and `functorch` but
  **not** `vllm` — `import vllm` raises `ModuleNotFoundError: No module named 'vllm'`. The install
  reached `torch` in the package-installation order but was killed before reaching `vllm` itself.
* Disk space was not the cause: `df -h /mnt/cache/persist` showed 99 TB free (2.0 TB / 100 TB used)
  throughout.
* This was a pure wall-clock cutoff, not a dependency-resolution or CUDA-compatibility conflict — no
  incompatible-torch/CUDA error ever appeared, because the install never reached the point of
  resolving vLLM's own CUDA extension build/link step.

## Resolution (plan Milestone 2 Step 6 / Risks & Fallbacks table)

Per the plan's explicit pre-authorization ("If the install fails or produces an incompatible
torch/CUDA combination within a 20-minute cutoff, record the exact error in an intervention file,
mark the `cosyvoice2_vllm` variant null with the reason, and continue with the other 5 CosyVoice2
variants — do not let this block the rest of Milestone 3"), the `vllm_backend` acceleration variant
for CosyVoice2 is marked **null** for this task, with this file as the recorded evidence. Setup
continued and completed for everything else: the base `.venv-cosyvoice2` and `.venv-chatterbox`
venvs, the CosyVoice2 `load_jit`/`load_trt` export (`flow.encoder.fp32.zip` /
`flow.encoder.fp16.zip` under
`/mnt/cache/persist/t0021_zero_shot_latency_reduction/pretrained/cosyvoice2/`), and the idle
watchdog were all verified working before and independently of this install attempt.

A same-session, un-timed retry (e.g. a plain `pip install` with no wrapper, letting it run to
completion and simply measuring how much longer it actually needs — the download phase alone already
completed, so a retry only needs to redo the installation/unpacking phase, likely a few more
minutes) was **not** attempted in this step, to avoid extending GPU billing during environment
preparation beyond the 1.0 h Milestone 2 budget. This is left as an option for the `implementation`
step: if budget allows and the `vllm_backend` variant is judged valuable enough, a fresh,
longer-capped install attempt could still succeed, since no CUDA/dependency incompatibility was ever
actually observed — only a timing cutoff.

## Cost impact

The `.venv-cosyvoice2-vllm` install attempt ran for ~18.3 minutes of GPU wall-clock
(2026-09-18T11:54:52Z-2026-09-18T12:13:12Z, concurrently with the CosyVoice2 `load_jit`/`load_trt`
export, which itself completed successfully in ~5 minutes during the same window) — approximately
$4.26 at the $13.96/hr `LLM-T1-NC80` rate, attributed to the `setup-machines` step's Milestone 2
budget line (`VM setup: isolated venvs, load_jit/load_trt export, vLLM backend install, watchdog`,
budgeted at 1.0 h / $13.96 total). Because the TensorRT export ran concurrently on the same VM
rather than sequentially, this does not represent additional billed time beyond the setup step's
existing 1.0 h allocation.
