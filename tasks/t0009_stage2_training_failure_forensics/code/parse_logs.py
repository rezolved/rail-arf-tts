"""Step 4: Parse training logs into per-step tidy CSV (REQ-2).

Input:  data/logs/<run_id>/*.log
Output: data/timelines/<run_id>_steps.csv

Log format (from run03_v6c_v3_stage1.log):
  Step lines:   Epoch [E/N], Step [S/T], Loss: X, Disc Loss: X, Dur Loss: X, ...
  Val lines:    Validation loss: X, Dur loss: X, F0 loss: X
  Epoch label:  Epoch N:  (0-based; appears after val line, before acoustic_norm)
  Norm lines:   acoustic_norm=X  prosodic_norm=X
  Epochs cnt:   Epochs: N  (1-based total epochs elapsed; used to derive epoch number
                            for val records that lack preceding step lines)
"""

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tasks.t0009_stage2_training_failure_forensics.code.constants import (
    COL_ACOUSTIC_NORM,
    COL_CE_LOSS,
    COL_DISC_LOSS,
    COL_DUR_LOSS,
    COL_EPOCH,
    COL_F0_LOSS,
    COL_GRAD_NORM_DECODER,
    COL_GRAD_NORM_MPD,
    COL_GRAD_NORM_MSD,
    COL_GRAD_NORM_STYLE_ENCODER,
    COL_LOSS_TOTAL,
    COL_MEL_LOSS,
    COL_RUN_ID,
    COL_SKIP_COUNT,
    COL_STEP,
    COL_VAL_LOSS,
    TIMELINE_COLUMNS,
)
from tasks.t0009_stage2_training_failure_forensics.code.paths import (
    LOGS_DIR,
    TIMELINES_DIR,
)

_STEP_RE = re.compile(
    r"Epoch \[(\d+)/\d+\], Step \[(\d+)/\d+\], "
    r"Loss: ([\d.]+), Disc Loss: ([\d.]+), Dur Loss: ([\d.]+), CE Loss: ([\d.]+)"
    r"(?:, Norm Loss: ([\d.]+))?"
    r"(?:, F0 Loss: ([\d.]+))?"
)
_VAL_RE = re.compile(r"Validation loss: ([\d.]+), Dur loss: ([\d.]+), F0 loss: ([\d.]+)")
# "Epoch N:" label — 0-based epoch number of epoch just completed
_EPOCH_LABEL_RE = re.compile(r"^Epoch (\d+):\s")
# "Epochs: N" counter — 1-based total elapsed epochs
_EPOCHS_COUNTER_RE = re.compile(r"^Epochs:\s+(\d+)\s*$")
_NORM_RE = re.compile(r"acoustic_norm=([\d.]+)")


@dataclass(frozen=False)
class StepRecord:
    run_id: str
    epoch: int
    step: int
    loss_total: float | None = None
    disc_loss: float | None = None
    dur_loss: float | None = None
    ce_loss: float | None = None
    mel_loss: float | None = None
    f0_loss: float | None = None
    val_loss: float | None = None
    acoustic_norm: float | None = None
    grad_norm_msd: float | None = None
    grad_norm_mpd: float | None = None
    grad_norm_decoder: float | None = None
    grad_norm_style_encoder: float | None = None
    skip_count: int | None = None


@dataclass
class ParseState:
    records: list[StepRecord] = field(default_factory=list)
    # Val line pending (awaiting acoustic_norm to flush)
    pending_val_epoch_1based: int | None = None
    pending_val_loss: float | None = None
    pending_val_dur: float | None = None
    pending_val_f0: float | None = None
    # Epoch label from "Epoch N:" line (0-based)
    pending_epoch_label_0based: int | None = None
    # Last known 1-based epoch from step lines or Epochs: counter
    last_epoch_1based: int = 0


def _float_or_none(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_log_file(log_path: Path, run_id: str) -> list[StepRecord]:
    state = ParseState()
    with log_path.open() as f:
        for raw_line in f:
            _process_line(line=raw_line.rstrip(), state=state, run_id=run_id)
    # Flush any trailing val record that never got its acoustic_norm
    if state.pending_val_loss is not None:
        epoch_1b = state.pending_val_epoch_1based or state.last_epoch_1based
        state.records.append(
            StepRecord(
                run_id=run_id,
                epoch=epoch_1b,
                step=0,
                val_loss=state.pending_val_loss,
                dur_loss=state.pending_val_dur,
                f0_loss=state.pending_val_f0,
            )
        )
    return state.records


def _process_line(line: str, state: ParseState, run_id: str) -> None:
    # Training step line
    m = _STEP_RE.search(line)
    if m is not None:
        epoch_1b = int(m.group(1))
        step = int(m.group(2))
        state.records.append(
            StepRecord(
                run_id=run_id,
                epoch=epoch_1b,
                step=step,
                loss_total=_float_or_none(m.group(3)),
                disc_loss=_float_or_none(m.group(4)),
                dur_loss=_float_or_none(m.group(5)),
                ce_loss=_float_or_none(m.group(6)),
                mel_loss=_float_or_none(m.group(7)),  # Norm Loss → mel_loss column
                f0_loss=_float_or_none(m.group(8)),
            )
        )
        state.last_epoch_1based = epoch_1b
        return

    # Validation line
    m = _VAL_RE.search(line)
    if m is not None:
        # Flush previous pending val if any (shouldn't happen, but defensive)
        if state.pending_val_loss is not None:
            epoch_1b = state.pending_val_epoch_1based or state.last_epoch_1based
            state.records.append(
                StepRecord(
                    run_id=run_id,
                    epoch=epoch_1b,
                    step=0,
                    val_loss=state.pending_val_loss,
                    dur_loss=state.pending_val_dur,
                    f0_loss=state.pending_val_f0,
                )
            )
        state.pending_val_loss = float(m.group(1))
        state.pending_val_dur = float(m.group(2))
        state.pending_val_f0 = float(m.group(3))
        state.pending_val_epoch_1based = state.last_epoch_1based
        return

    # "Epoch N:" label (0-based; N+1 = 1-based epoch)
    m = _EPOCH_LABEL_RE.match(line)
    if m is not None:
        state.pending_epoch_label_0based = int(m.group(1))
        # Convert to 1-based for pending val record if val is pending and epoch not yet set
        epoch_1b = int(m.group(1)) + 1
        if state.pending_val_loss is not None and state.pending_val_epoch_1based is None:
            state.pending_val_epoch_1based = epoch_1b
        return

    # "Epochs: N" counter (1-based total elapsed)
    m = _EPOCHS_COUNTER_RE.match(line)
    if m is not None:
        state.last_epoch_1based = int(m.group(1))
        return

    # acoustic_norm line — flush pending val record
    m = _NORM_RE.search(line)
    if m is not None and "prosodic_norm" in line:
        acoustic = float(m.group(1))
        if state.pending_val_loss is not None:
            # Use epoch label if available, else last known epoch
            if state.pending_epoch_label_0based is not None:
                epoch_1b = state.pending_epoch_label_0based + 1
            else:
                epoch_1b = state.pending_val_epoch_1based or state.last_epoch_1based
            state.records.append(
                StepRecord(
                    run_id=run_id,
                    epoch=epoch_1b,
                    step=0,
                    val_loss=state.pending_val_loss,
                    dur_loss=state.pending_val_dur,
                    f0_loss=state.pending_val_f0,
                    acoustic_norm=acoustic,
                )
            )
            state.pending_val_loss = None
            state.pending_val_dur = None
            state.pending_val_f0 = None
            state.pending_val_epoch_1based = None
            state.pending_epoch_label_0based = None
        else:
            # acoustic_norm without val (e.g. baseline extraction at start)
            state.records.append(
                StepRecord(
                    run_id=run_id,
                    epoch=state.last_epoch_1based,
                    step=0,
                    acoustic_norm=acoustic,
                )
            )
        return


def write_csv(records: list[StepRecord], run_id: str) -> Path:
    TIMELINES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TIMELINES_DIR / f"{run_id}_steps.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMELINE_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    COL_RUN_ID: rec.run_id,
                    COL_EPOCH: rec.epoch,
                    COL_STEP: rec.step,
                    COL_LOSS_TOTAL: rec.loss_total,
                    COL_DISC_LOSS: rec.disc_loss,
                    COL_DUR_LOSS: rec.dur_loss,
                    COL_CE_LOSS: rec.ce_loss,
                    COL_MEL_LOSS: rec.mel_loss,
                    COL_F0_LOSS: rec.f0_loss,
                    COL_VAL_LOSS: rec.val_loss,
                    COL_ACOUSTIC_NORM: rec.acoustic_norm,
                    COL_GRAD_NORM_MSD: rec.grad_norm_msd,
                    COL_GRAD_NORM_MPD: rec.grad_norm_mpd,
                    COL_GRAD_NORM_DECODER: rec.grad_norm_decoder,
                    COL_GRAD_NORM_STYLE_ENCODER: rec.grad_norm_style_encoder,
                    COL_SKIP_COUNT: rec.skip_count,
                }
            )
    return out_path


def parse_all_logs() -> dict[str, list[StepRecord]]:
    results: dict[str, list[StepRecord]] = {}
    if not LOGS_DIR.exists():
        print(f"WARN: {LOGS_DIR} does not exist")
        return results
    for run_dir in sorted(LOGS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        log_files = sorted(run_dir.glob("*.log"))
        if len(log_files) == 0:
            print(f"  SKIP: {run_id} — no .log files found")
            continue
        for log_file in log_files:
            records = parse_log_file(log_path=log_file, run_id=run_id)
            results[run_id] = records
            print(f"  Parsed {run_id}: {len(records)} records from {log_file.name}")
    return results


def main() -> None:
    print("=== parse_logs.py: parse training logs into timelines ===")
    all_records = parse_all_logs()
    if len(all_records) == 0:
        print("No logs found to parse.", file=sys.stderr)
        sys.exit(1)
    for run_id, records in all_records.items():
        out = write_csv(records=records, run_id=run_id)
        val_recs = [r for r in records if r.val_loss is not None]
        bad = [r for r in val_recs if r.val_loss is not None and r.val_loss != r.val_loss]
        print(
            f"  {run_id}: {len(records)} total, {len(val_recs)} val, {len(bad)} non-finite val_loss"
        )
        if len(val_recs) > 0:
            print("  val_loss sequence:", [f"ep{r.epoch}:{r.val_loss}" for r in val_recs])
        print(f"  -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
