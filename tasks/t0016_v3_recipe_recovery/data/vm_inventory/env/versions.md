# VM Environment Versions (REQ-9)

Captured from `LLM-T1-NC80` during the bounded read-only SSH inventory (Milestone 1 Step 5). Full
`pip freeze` output is in `pip_freeze.txt` (300 packages).

**Important caveat**: this is the environment as of this task's inventory (2026-09-17), inside
`~/kokoro-finetune`'s Python environment (which resolves via a symlink to t0014's own clone — see
`data/vm_inventory/inventory.json`'s `home_directory_note`). It is **not confirmed** to be v3's
original training environment (pre-ARF, before September 2026) — no version-pin file, lockfile, or
`requirements.txt` snapshot from v3's actual run survived. These values are recorded as the closest
available evidence, labelled `unknown: current environment, not confirmed as v3's original` in
`data/config_david_v3_reconstructed.yml` where relevant.

| Package | Version |
| --- | --- |
| `torch` | 2.14.0 |
| `torchaudio` | 2.11.0 |
| `torchcodec` | 0.16.0 |
| `misaki` | 0.9.4 |
| `phonemizer` | 3.4.0 |
| `espeakng-loader` | 0.2.4 |
| `espeak-ng` (CLI, not pip) | 1.50 |

No `kokoro` pip package is installed in this training environment — expected, since Kokoro-API
packaging (via `tasks/t0002_kokoro_v4_voicepack_decoder_package/code/extract_decoder_generic.py`'s
contract) happens as a separate post-training step, not inside the StyleTTS2 training environment
itself.

`~/kokoro-finetune` is not a git checkout (`git -C ~/kokoro-finetune log` returned no repository),
so no StyleTTS2 commit hash could be captured for this or any prior run.
