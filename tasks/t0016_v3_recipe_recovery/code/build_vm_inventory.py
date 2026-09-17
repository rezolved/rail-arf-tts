"""Milestone 1 Step 6 (REQ-1): build `data/vm_inventory/inventory.json` from the files copied off
`LLM-T1-NC80` during the bounded read-only SSH inventory.

Adapts the `RunEntry`-style dataclass pattern from
`tasks/t0009_stage2_training_failure_forensics/code/build_inventory.py`, trimmed to a single
`InventoryEntry(path, size_bytes, mtime_iso, sha256)` dataclass, and reuses
`checkpoint_forensics.sha256_file()` (itself extracted from
`tasks/t0009_stage2_training_failure_forensics/code/checkpoint_manager.py`'s `_sha256()`).

Usage::

    uv run python -m tasks.t0016_v3_recipe_recovery.code.build_vm_inventory
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tasks.t0016_v3_recipe_recovery.code.checkpoint_forensics import sha256_file
from tasks.t0016_v3_recipe_recovery.code.paths import VM_INVENTORY_DIR, VM_INVENTORY_JSON

# Findings from the bounded SSH inventory (Milestone 1 Steps 4-6, see logs/commands/ for the raw
# transcripts). Recorded here as named constants rather than re-parsed from raw dump text, since
# the raw dumps are kept as `data/vm_inventory/raw_dump*.txt` for audit but are not
# machine-structured.
HOME_DIRECTORY_PRESENT = True
HOME_DIRECTORY_IS_ORIGINAL_V3_ENV = False
FIRST_STAGE_V3_REMOTE_SHA256: str | None = None
# (2026-09-17T14:47:56.755481Z teardown) - (2026-09-17T14:34:28.703377Z billing_started_at, per
# logs/steps/008_setup-machines/machine_log.json) = 13.468 minutes.
VM_SESSION_MINUTES = 13.468


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    path: str
    size_bytes: int
    mtime_iso: str
    sha256: str


def build_entries(root: Path) -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "inventory.json":
            continue
        stat = path.stat()
        entries.append(
            InventoryEntry(
                path=str(path.relative_to(root)),
                size_bytes=stat.st_size,
                mtime_iso=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                sha256=sha256_file(path),
            )
        )
    return entries


def main() -> None:
    entries = build_entries(VM_INVENTORY_DIR)
    manifest = {
        "spec_version": "1",
        "home_directory_present": HOME_DIRECTORY_PRESENT,
        "home_directory_is_original_v3_env": HOME_DIRECTORY_IS_ORIGINAL_V3_ENV,
        "home_directory_note": (
            "~/kokoro-finetune is a symlink to /mnt/cache/persist/t0014_v11_decoder_fix_retrain/"
            "kokoro-finetune -- i.e. t0014's own StyleTTS2 clone used for the v11 retrain, not a "
            "preserved v3-era environment. No first_stage_v3.pth, no v3-named logs/ directory, no "
            "266-clip data list exist anywhere under ~/kokoro-finetune or the resolved "
            "/mnt/cache/persist share. The share's other task-named directories (t0009, t0010, "
            "t0014_v11_decoder_fix_retrain, plus several directories from an unrelated LLM "
            "fine-tuning project sharing this VM pool) contain no v3 or t0002/t0006-referenced "
            "artifacts either."
        ),
        "first_stage_v3_remote_sha256": FIRST_STAGE_V3_REMOTE_SHA256,
        "first_stage_v3_search_result": "not found under ~/kokoro-finetune or /mnt/cache/persist",
        "models_py_found": True,
        "models_py_multispeaker_finding": (
            "models.py:808 `if args.multispeaker:` selects StyleTransformer1d vs. Transformer1d "
            "for `diffusion.diffusion.net`/`diffusion.unet` inside build_model(). `diffusion` IS a "
            "top-level key in the StyleTTS2 `nets` Munch (line ~850) and therefore a top-level key "
            "in any raw checkpoint's `net` dict, alongside the 5 bundled modules -- but it is NOT "
            "one of the 5 Kokoro-API bundle modules (bert/bert_encoder/predictor/text_encoder/"
            "decoder), so multispeaker's effect is invisible to a diff restricted to those 5 "
            "modules. It IS checkable via stage1/first_stage.pth's net['diffusion'] shape, per "
            "checkpoint_forensics.all_top_level_keys()."
        ),
        "266_clip_list_found": False,
        "kikiri_tts_symlink_broken": True,
        "vm_session_minutes": VM_SESSION_MINUTES,
        "vm_ready_at": (
            "2026-09-17T14:34:45.033597Z (per logs/steps/008_setup-machines/machine_log.json)"
        ),
        "vm_billing_started_at": "2026-09-17T14:34:28.703377Z",
        "vm_teardown_called_at": "2026-09-17T14:47:56.755481Z",
        "vm_teardown_by": "implementation step (this step), not a later teardown step",
        "entries": [asdict(e) for e in entries],
    }
    VM_INVENTORY_JSON.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {VM_INVENTORY_JSON} with {len(entries)} file entries.")


if __name__ == "__main__":
    main()
