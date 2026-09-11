"""Azure ML compute instance provisioner accessed via SSH.

Replaces the per-task vast.ai provisioning model with a small fixed pool of
long-lived Azure ML compute instances, defined in project/azure_vm.json (the pool
spans three Azure ML workspaces, so never assume a single --workspace-name).
The pool is shared with the finetuning team; coordination is by Slack.

Public surface:

* ``acquire(task_id)`` — pick a VM, start it if needed, verify SSH, place a
  per-task lock file on the VM. Refuses if a different task already holds a
  lock on every VM in the pool.
* ``run(task_id, command)`` — execute a shell command on the locked VM with
  periodic heartbeats and (for long jobs) checkpoint reminders.
* ``teardown(task_id, deallocate=True)`` — clear the lock, kill stray vLLM
  processes started by the task, and stop the VM if no other tasks hold
  locks on it.

Also exposes a CLI:

    uv run python -m arf.scripts.utils.azure_ml_vm acquire <task_id>
    uv run python -m arf.scripts.utils.azure_ml_vm teardown <task_id>
    uv run python -m arf.scripts.utils.azure_ml_vm run <task_id> -- <cmd>

All subprocess work goes through two thin shims (``_run_az`` and ``_run_ssh``)
that tests patch with monkeypatch so the module is exercised end-to-end
without ever touching the Azure or SSH control planes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, assert_never

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POOL_CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "project" / "azure_vm.json"
LOCK_DIR_REMOTE: str = "~/.arf-locks"

# Matches ``**Version**`` in arf/specifications/remote_machines_specification.md; the
# verificator warns (RM-W007) on entries written under an older one.
SPEC_VERSION: str = "10"
# The provider slug written into machine_log.json. Kept as the historical "azure-ml"
# spelling because aggregate_machines.py, verify_machine_log_cost_attribution.py and
# the setup-remote-machine skill all key on it; remote_machines_specification.md
# accepts it as an alias of the canonical "azure_ml".
PROVIDER: str = "azure-ml"
GPU_MODEL: str = "H100 80GB"
GPU_COUNT: int = 2
SELECTED_OFFER_GPU_DISPLAY: str = "2xH100"

VM_START_TIMEOUT_SECONDS: float = 8 * 60.0
VM_START_POLL_INTERVAL_SECONDS: float = 15.0
SSH_POLL_INTERVAL_SECONDS: float = 10.0
SSH_HANDSHAKE_TIMEOUT_SECONDS: float = 10.0
AZ_TIMEOUT_SECONDS: float = 60.0

HEARTBEAT_INTERVAL_SECONDS: float = 5 * 60.0
CHECKPOINT_INTERVAL_SECONDS: float = 30 * 60.0
LONG_JOB_THRESHOLD_SECONDS: float = 2 * 60 * 60.0

EXIT_OK: int = 0
EXIT_GENERIC_ERROR: int = 1
EXIT_POOL_BUSY: int = 75  # matches sysexits.h EX_TEMPFAIL

# VM-side guarantees checked before the task lock is placed. See the script header
# and LESSONS.md Lessons 10 (persistent storage) and 11 (systemd lingering).
PREFLIGHT_SCRIPT_PATH: Path = Path(__file__).resolve().parent / "remote_preflight.sh"
PREFLIGHT_FAILURE_PHASE: str = "preflight"
PREFLIGHT_TIMEOUT_SECONDS: float = 120.0
# The wire format the script prints. Kept as literals here and in the script rather
# than shared through a constant neither side can see, so a rename on one side shows
# up as a failure instead of staying self-consistently green.
PREFLIGHT_LINGER_ENABLED_KEY: str = "linger_enabled"
PREFLIGHT_PERSIST_PATH_KEY: str = "persist_path"
PREFLIGHT_PERSIST_WRITABLE_KEY: str = "persist_writable"

# Azure compute instance states we care about, normalized to lowercase.
_STATE_RUNNING: str = "running"
_STATE_STOPPED: str = "stopped"
_STATE_STARTING: str = "starting"
_STATE_STOPPING: str = "stopping"
_STATE_UNKNOWN: str = "unknown"

# The key az prints the compute state under in its ``-o json`` payload.
_AZ_STATE_KEY: str = "state"

# Failure phases recorded on a FailedAttempt. The two az-show phases below are the
# non-retryable ones: an RBAC denial or a missing compute never becomes a running VM by
# being polled at, so the acquire attempt gives up on them instead of waiting out
# VM_START_TIMEOUT_SECONDS and mislabelling the result a boot timeout.
AZ_SHOW_FAILURE_PHASE: str = "az_show"
AZ_SHOW_AUTHORIZATION_FAILURE_PHASE: str = "az_show_authorization"
AZ_SHOW_NOT_FOUND_FAILURE_PHASE: str = "az_show_not_found"
AZ_START_FAILURE_PHASE: str = "az_start"
VM_START_TIMEOUT_FAILURE_PHASE: str = "vm_start_timeout"
SSH_CONNECT_FAILURE_PHASE: str = "ssh_connect"
LOCK_HELD_FAILURE_PHASE: str = "lock_held"

# Substrings az prints for the two failure kinds worth distinguishing. Matched
# case-insensitively against stderr; the error code and the prose form both appear in
# the wild depending on the az version and whether ARM or the ML extension rejected the
# call.
_AUTHORIZATION_STDERR_MARKERS: tuple[str, ...] = (
    "authorizationfailed",
    "does not have authorization to perform action",
)
_NOT_FOUND_STDERR_MARKERS: tuple[str, ...] = (
    "resourcenotfound",
    "was not found",
)

# Cost attribution. A machine's cost is its hourly rate times the hours it actually
# billed, which is not the wall-clock span between acquire and teardown: a stopped
# instance does not bill, and billing starts when the start call is issued rather than
# when the provisioner finishes verifying SSH. See the "Cost Attribution" section of
# arf/specifications/remote_machines_specification.md.
BILLING_ANCHOR_AZ_START: str = "az_start_issued"
BILLING_ANCHOR_ALREADY_RUNNING: str = "already_running"
BILLING_ANCHOR_ACQUIRED_AT: str = "acquired_at_fallback"

# How much the reported figure can be trusted. `unresolved` is the honest outcome when
# nothing records when billing ended -- it reports null rather than a fabricated number.
COST_STATUS_VERIFIED: str = "verified"
COST_STATUS_ASSUMED: str = "assumed"
COST_STATUS_UNRESOLVED: str = "unresolved"

COST_METHOD_ACTIVITY_LOG: str = "activity_log"
COST_METHOD_PROVISIONER_OBSERVED: str = "provisioner_observed"
COST_METHOD_UNRESOLVED: str = "unresolved"

POWER_EVENT_START: str = "start"
POWER_EVENT_STOP: str = "stop"

SEGMENT_SOURCE_ACTIVITY_LOG: str = "activity_log"
SEGMENT_SOURCE_PROVISIONER: str = "provisioner"

# az monitor activity-log wire format. Kept as literals on this side so a schema change
# shows up as "no events found" in the machine log rather than a silent miscount.
_ACTIVITY_LOG_MAX_EVENTS: str = "1000"
_ACTIVITY_LOG_STATUS_SUCCEEDED: str = "succeeded"
_ACTIVITY_RESOURCE_ID_KEY: str = "resourceId"
_ACTIVITY_OPERATION_KEY: str = "operationName"
_ACTIVITY_STATUS_KEY: str = "status"
_ACTIVITY_TIMESTAMP_KEY: str = "eventTimestamp"
_ACTIVITY_VALUE_KEY: str = "value"
_START_OPERATION_SUFFIX: str = "/computes/start/action"
_STOP_OPERATION_SUFFIX: str = "/computes/stop/action"
_COMPUTES_PATH_SEGMENT: str = "/computes/"

_SECONDS_PER_HOUR: float = 3600.0

# Operator-facing text that lands in machine_log.json. Written once so two unresolved
# entries never differ by nothing but wording.
_UNRESOLVED_REMEDIATION_NOTE: str = (
    "Reconstruct the billable windows from this task's own step logs and record the "
    "methodology in results/costs.json."
)
# Sort key for an event whose timestamp will not parse: push it past every real one so a
# stable sort keeps it out of the way. Such events are dropped before they reach a walk.
_FAR_FUTURE: datetime = datetime.max.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Pool config (Pydantic at the I/O edge)
# ---------------------------------------------------------------------------


class VmPoolEntryFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    workspace: str
    resource_group: str
    ssh_host_alias: str
    hourly_cost_usd: float
    priority: int
    notes: str


class VmPoolFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec_version: str
    vms: list[VmPoolEntryFile]


@dataclass(frozen=True, slots=True)
class VmPoolEntry:
    name: str
    workspace: str
    resource_group: str
    ssh_host_alias: str
    hourly_cost_usd: float
    priority: int
    notes: str


def load_pool(*, config_path: Path | None = None) -> list[VmPoolEntry]:
    resolved: Path = config_path if config_path is not None else POOL_CONFIG_PATH
    pool_file: VmPoolFile = VmPoolFile.model_validate_json(
        resolved.read_text(encoding="utf-8"),
    )
    entries: list[VmPoolEntry] = [
        VmPoolEntry(
            name=v.name,
            workspace=v.workspace,
            resource_group=v.resource_group,
            ssh_host_alias=v.ssh_host_alias,
            hourly_cost_usd=v.hourly_cost_usd,
            priority=v.priority,
            notes=v.notes,
        )
        for v in pool_file.vms
    ]
    entries.sort(key=lambda e: e.priority)
    return entries


# ---------------------------------------------------------------------------
# Result dataclasses (internal, returned to callers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FailedAttempt:
    vm_name: str
    failure_reason: str
    failure_phase: str
    duration_seconds: float
    wasted_cost_usd: float
    timestamp: str
    # Verbatim provider stderr, when the failure came from a provider call. None means
    # no provider output was captured, never "the call printed nothing useful" — the
    # whole point is that the operator can read the real cause in the machine log.
    failure_reason_detail: str | None = None


@dataclass(frozen=True, slots=True)
class AcquireResult:
    task_id: str
    vm: VmPoolEntry
    acquired_at: str
    search_started_at: str
    ready_at: str
    total_provisioning_seconds: float
    started_vm: bool
    # Not the same instant as ``acquired_at``: billing begins when the start call is
    # issued, ``acquired_at`` is the later moment the VM was verified reachable.
    billing_started_at: str
    billing_anchor: str
    failed_attempts: list[FailedAttempt] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AcquireAttempt:
    """One pool entry's acquire attempt, with the billing anchor it established."""

    acquired: bool
    failure: FailedAttempt | None
    started_vm: bool
    billing_started_at: str
    billing_anchor: str


@dataclass(frozen=True, slots=True)
class PowerEvent:
    kind: str
    at: str


@dataclass(frozen=True, slots=True)
class BillableSegment:
    started_at: str
    ended_at: str
    hours: float
    source: str


@dataclass(frozen=True, slots=True)
class BillableWindow:
    segments: list[BillableSegment]
    running_at_end: bool


@dataclass(frozen=True, slots=True)
class CostAttribution:
    """Everything a reviewer needs to redo a machine's cost arithmetic by hand.

    ``billable_hours``, ``total_cost_usd`` and ``stopped_hours`` are ``None`` -- never
    ``0.0`` -- when the attribution is unresolved. Zero is a measurement; ``None`` is the
    absence of one, and inventing the difference is the defect this type exists to close.
    """

    billing_started_at: str
    billing_anchor: str
    billing_ended_at: str
    method: str
    status: str
    segments: list[BillableSegment]
    billable_hours: float | None
    wall_clock_span_hours: float | None
    stopped_hours: float | None
    hourly_cost_usd: float
    total_cost_usd: float | None
    notes: str


@dataclass(frozen=True, slots=True)
class BillingAnchor:
    started_at: str
    kind: str


@dataclass(frozen=True, slots=True)
class TeardownResult:
    task_id: str
    vm_name: str
    deallocated: bool
    other_locks_present: bool
    destroyed_at: str
    # Billable hours, not the wall-clock span, and ``None`` when unresolved.
    duration_hours: float | None
    total_cost_usd: float | None
    cost_attribution: CostAttribution


@dataclass(frozen=True, slots=True)
class RunResult:
    task_id: str
    vm_name: str
    exit_code: int
    duration_seconds: float
    heartbeats: int


# ---------------------------------------------------------------------------
# Subprocess shims (the only places that touch the outside world)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _run_az(*, args: list[str], timeout: float = AZ_TIMEOUT_SECONDS) -> CommandResult:
    full: list[str] = ["az", *args]
    proc: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        full,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _run_ssh(
    *,
    host_alias: str,
    remote_command: str,
    timeout: float = AZ_TIMEOUT_SECONDS,
    connect_timeout: float = SSH_HANDSHAKE_TIMEOUT_SECONDS,
) -> CommandResult:
    args: list[str] = [
        "ssh",
        "-o",
        f"ConnectTimeout={int(connect_timeout)}",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        host_alias,
        remote_command,
    ]
    proc: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


# ---------------------------------------------------------------------------
# VM-side preflight guarantees
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreflightResult:
    linger_enabled: bool
    persist_path: str
    persist_writable: bool


def _parse_preflight_stdout(*, stdout: str) -> PreflightResult | None:
    # The script prints one JSON line, but a login shell may print a banner or a
    # motd first, so scan for the line that parses rather than assuming position.
    for line in stdout.splitlines():
        stripped: str = line.strip()
        if len(stripped) == 0:
            continue
        try:
            payload: object = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        linger: object = payload.get(PREFLIGHT_LINGER_ENABLED_KEY)
        persist_path: object = payload.get(PREFLIGHT_PERSIST_PATH_KEY)
        writable: object = payload.get(PREFLIGHT_PERSIST_WRITABLE_KEY)
        if not isinstance(linger, bool):
            continue
        if not isinstance(persist_path, str):
            continue
        if not isinstance(writable, bool):
            continue
        return PreflightResult(
            linger_enabled=linger,
            persist_path=persist_path,
            persist_writable=writable,
        )
    return None


def run_remote_preflight(*, vm: VmPoolEntry) -> PreflightResult | None:
    """Run the VM-side guarantees on ``vm``; ``None`` means the box must not be used.

    Ships ``remote_preflight.sh`` as the SSH command itself, so nothing has to be
    staged on the box first. A ``None`` here is not a warning to log and continue
    past: the two guarantees it checks are the ones whose absence destroys a
    training run silently (``LESSONS.md`` Lessons 10 and 11).
    """

    # A sibling data file, not an import: nothing fails at build time if it is renamed,
    # and an uncaught read error here would abort the whole pool instead of one VM.
    assert PREFLIGHT_SCRIPT_PATH.is_file(), f"preflight script exists at {PREFLIGHT_SCRIPT_PATH}"
    script: str = PREFLIGHT_SCRIPT_PATH.read_text(encoding="utf-8")
    try:
        result: CommandResult = _run_ssh(
            host_alias=vm.ssh_host_alias,
            remote_command=script,
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write(
            f"preflight on {vm.name} timed out after {PREFLIGHT_TIMEOUT_SECONDS:.0f}s\n",
        )
        return None
    if result.returncode != 0:
        # The script names which guarantee failed, on which path, for which user. Put it
        # where run_with_logs will capture it — the FailedAttempt recorded upstream can
        # only say that preflight failed, not why.
        sys.stderr.write(f"preflight failed on {vm.name}: {result.stderr.strip()}\n")
        return None
    parsed: PreflightResult | None = _parse_preflight_stdout(stdout=result.stdout)
    if parsed is None:
        return None
    # Enforce the guarantees on this side too, rather than trusting the exit code to
    # agree with the payload. A detector that reads fields nothing ever varies is how the
    # first liveness bug survived for months (LESSONS Lesson 8, first follow-up).
    if not parsed.linger_enabled or not parsed.persist_writable:
        sys.stderr.write(
            f"preflight on {vm.name} reported linger_enabled={parsed.linger_enabled} "
            f"persist_writable={parsed.persist_writable}\n",
        )
        return None
    return parsed


# ---------------------------------------------------------------------------
# Azure compute control
# ---------------------------------------------------------------------------


def _normalize_state(*, raw: str | None) -> str:
    if raw is None:
        return _STATE_UNKNOWN
    lowered: str = raw.strip().lower()
    if lowered in {"running"}:
        return _STATE_RUNNING
    if lowered in {"stopped", "deallocated"}:
        return _STATE_STOPPED
    if lowered in {"starting", "creating"}:
        return _STATE_STARTING
    if lowered in {"stopping", "deallocating"}:
        return _STATE_STOPPING
    return _STATE_UNKNOWN


class ComputeStateFailure(Enum):
    """Why an ``az ml compute show`` call did not yield a state.

    The distinction exists because two of these are permanent: an authorization denial
    and a missing compute stay true no matter how long the caller waits, while an
    unknown failure (a management-endpoint blip, a truncated payload) plausibly clears
    on the next poll.
    """

    AUTHORIZATION = "authorization"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ComputeStateResult:
    state: str
    failure: ComputeStateFailure | None
    stderr: str | None


def _classify_state_failure(*, stderr: str) -> ComputeStateFailure:
    lowered: str = stderr.lower()
    if any(marker in lowered for marker in _AUTHORIZATION_STDERR_MARKERS):
        return ComputeStateFailure.AUTHORIZATION
    if any(marker in lowered for marker in _NOT_FOUND_STDERR_MARKERS):
        return ComputeStateFailure.NOT_FOUND
    return ComputeStateFailure.UNKNOWN


def _stderr_or_none(*, raw: str) -> str | None:
    """Return az stderr verbatim, or None when it carried nothing.

    Verbatim on purpose: the machine log keeps exactly what az printed, and only the
    human-facing intervention file trims it for layout.
    """
    stripped: str = raw.strip()
    if len(stripped) == 0:
        return None
    return raw


def non_retryable_state_failure_phase(
    *,
    failure: ComputeStateFailure | None,
) -> str | None:
    """Return the failure phase for a failure that waiting cannot fix, else None.

    Exhaustive on purpose. A mapping with a ``.get`` default would let a future member
    (quota exhausted, subscription disabled — both permanent) inherit "retryable" in
    silence, which is the 8-minute wait this function exists to prevent.
    """
    match failure:
        case None | ComputeStateFailure.UNKNOWN:
            return None
        case ComputeStateFailure.AUTHORIZATION:
            return AZ_SHOW_AUTHORIZATION_FAILURE_PHASE
        case ComputeStateFailure.NOT_FOUND:
            return AZ_SHOW_NOT_FOUND_FAILURE_PHASE
        case _:
            assert_never(failure)


def get_compute_state_result(*, vm: VmPoolEntry) -> ComputeStateResult:
    """Read ``vm``'s compute state, keeping the failure kind and the az stderr.

    Collapsing every non-zero exit into ``unknown`` is what let an
    ``AuthorizationFailed`` on ``computes/read`` be recorded as ``vm_start_timeout``
    after two full 8-minute waits, with the real cause never reaching the machine log.
    """
    result: CommandResult = _run_az(
        args=[
            "ml",
            "compute",
            "show",
            "--name",
            vm.name,
            "--workspace-name",
            vm.workspace,
            "--resource-group",
            vm.resource_group,
            "-o",
            "json",
        ],
    )
    if result.returncode != 0:
        return ComputeStateResult(
            state=_STATE_UNKNOWN,
            failure=_classify_state_failure(stderr=result.stderr),
            stderr=_stderr_or_none(raw=result.stderr),
        )
    unparseable: ComputeStateResult = ComputeStateResult(
        state=_STATE_UNKNOWN,
        failure=ComputeStateFailure.UNKNOWN,
        stderr=_stderr_or_none(raw=result.stderr),
    )
    try:
        payload: object = json.loads(result.stdout)
    except json.JSONDecodeError:
        return unparseable
    if not isinstance(payload, dict):
        return unparseable
    raw_state: object = payload.get(_AZ_STATE_KEY)
    if not isinstance(raw_state, str):
        return unparseable
    return ComputeStateResult(
        state=_normalize_state(raw=raw_state),
        failure=None,
        stderr=None,
    )


def get_compute_state(*, vm: VmPoolEntry) -> str:
    return get_compute_state_result(vm=vm).state


def start_compute(*, vm: VmPoolEntry) -> CommandResult:
    return _run_az(
        args=[
            "ml",
            "compute",
            "start",
            "--name",
            vm.name,
            "--workspace-name",
            vm.workspace,
            "--resource-group",
            vm.resource_group,
            "--no-wait",
        ],
    )


def stop_compute(*, vm: VmPoolEntry) -> CommandResult:
    return _run_az(
        args=[
            "ml",
            "compute",
            "stop",
            "--name",
            vm.name,
            "--workspace-name",
            vm.workspace,
            "--resource-group",
            vm.resource_group,
            "--no-wait",
        ],
    )


# ---------------------------------------------------------------------------
# SSH-level operations
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _now_monotonic() -> float:
    return time.monotonic()


def _ssh_ok(*, vm: VmPoolEntry) -> bool:
    result: CommandResult = _run_ssh(
        host_alias=vm.ssh_host_alias,
        remote_command="true",
        timeout=SSH_HANDSHAKE_TIMEOUT_SECONDS + 5.0,
        connect_timeout=SSH_HANDSHAKE_TIMEOUT_SECONDS,
    )
    return result.returncode == 0


def _wait_for_ssh(*, vm: VmPoolEntry, deadline: float) -> bool:
    while _now_monotonic() < deadline:
        if _ssh_ok(vm=vm):
            return True
        time.sleep(SSH_POLL_INTERVAL_SECONDS)
    return False


def _wait_for_state(
    *,
    vm: VmPoolEntry,
    target: str,
    deadline: float,
) -> bool:
    while _now_monotonic() < deadline:
        state_result: ComputeStateResult = get_compute_state_result(vm=vm)
        if state_result.state == target:
            return True
        # A wait needs a way to fail, not only a way to continue (LESSONS.md Lesson 8,
        # third follow-up): polling an RBAC denial or a deleted compute can never turn
        # true, so stop instead of burning the deadline.
        if non_retryable_state_failure_phase(failure=state_result.failure) is not None:
            return False
        time.sleep(VM_START_POLL_INTERVAL_SECONDS)
    return False


def _lock_path(*, task_id: str) -> str:
    return f"{LOCK_DIR_REMOTE}/{task_id}.lock"


def list_remote_locks(*, vm: VmPoolEntry) -> list[str]:
    result: CommandResult = _run_ssh(
        host_alias=vm.ssh_host_alias,
        remote_command=(
            f"mkdir -p {LOCK_DIR_REMOTE} && "
            f"ls {LOCK_DIR_REMOTE}/ 2>/dev/null | grep '\\.lock$' || true"
        ),
    )
    if result.returncode != 0:
        return []
    locks: list[str] = []
    for line in result.stdout.splitlines():
        name: str = line.strip()
        if name.endswith(".lock"):
            locks.append(name[: -len(".lock")])
    return locks


def place_remote_lock(*, vm: VmPoolEntry, task_id: str) -> None:
    payload: dict[str, str] = {
        "task_id": task_id,
        "acquired_at": _now_iso(),
    }
    body: str = json.dumps(payload)
    quoted_body: str = shlex.quote(body)
    target: str = _lock_path(task_id=task_id)
    remote_cmd: str = f"mkdir -p {LOCK_DIR_REMOTE} && printf '%s' {quoted_body} > {target}"
    result: CommandResult = _run_ssh(host_alias=vm.ssh_host_alias, remote_command=remote_cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to place lock on {vm.name}: {result.stderr.strip() or result.stdout.strip()}",
        )


def clear_remote_lock(*, vm: VmPoolEntry, task_id: str) -> bool:
    target: str = _lock_path(task_id=task_id)
    result: CommandResult = _run_ssh(
        host_alias=vm.ssh_host_alias,
        remote_command=f"rm -f {target}",
    )
    return result.returncode == 0


def kill_task_vllm_processes(*, vm: VmPoolEntry, task_id: str) -> CommandResult:
    # Match vLLM processes tagged with the task_id in their command line. The
    # task is expected to launch vLLM with --served-model-name or env containing
    # the task_id; pkill -f is best-effort, never fatal.
    return _run_ssh(
        host_alias=vm.ssh_host_alias,
        remote_command=(f"pkill -f 'vllm.*{shlex.quote(task_id)}' 2>/dev/null; true"),
    )


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------


def _parse_iso(*, value: str) -> datetime | None:
    """Parse an ISO 8601 instant into a timezone-*aware* datetime, or ``None``.

    The awareness matters: ``fromisoformat`` happily accepts a naive
    ``"2026-08-21T13:00:00"``, and subtracting a naive datetime from an aware one raises
    ``TypeError``. In ``teardown`` that fires after the lock is cleared and the VM
    stopped, so a hand-typed ``--billing-started-at`` would abort the call with no
    ``TeardownResult`` and no cost record at all. Everything this module emits is UTC, so
    a naive value is read as UTC rather than rejected.
    """
    try:
        parsed: datetime = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _hours_between(*, start: str, end: str) -> float | None:
    start_dt: datetime | None = _parse_iso(value=start)
    end_dt: datetime | None = _parse_iso(value=end)
    if start_dt is None or end_dt is None:
        return None
    return (end_dt - start_dt).total_seconds() / _SECONDS_PER_HOUR


def _power_event_sort_key(event: PowerEvent) -> datetime:
    parsed: datetime | None = _parse_iso(value=event.at)
    return parsed if parsed is not None else _FAR_FUTURE


def _power_event_kind(*, operation: str) -> str | None:
    lowered: str = operation.strip().lower()
    if lowered.endswith(_START_OPERATION_SUFFIX):
        return POWER_EVENT_START
    if lowered.endswith(_STOP_OPERATION_SUFFIX):
        return POWER_EVENT_STOP
    return None


def _activity_element_to_event(*, element: object, needle: str) -> PowerEvent | None:
    if not isinstance(element, dict):
        return None
    resource_id: object = element.get(_ACTIVITY_RESOURCE_ID_KEY)
    operation: object = element.get(_ACTIVITY_OPERATION_KEY)
    status: object = element.get(_ACTIVITY_STATUS_KEY)
    at: object = element.get(_ACTIVITY_TIMESTAMP_KEY)
    if not isinstance(resource_id, str) or not isinstance(at, str):
        return None
    if not isinstance(operation, dict) or not isinstance(status, dict):
        return None
    operation_value: object = operation.get(_ACTIVITY_VALUE_KEY)
    status_value: object = status.get(_ACTIVITY_VALUE_KEY)
    if not isinstance(operation_value, str) or not isinstance(status_value, str):
        return None
    if status_value.strip().lower() != _ACTIVITY_LOG_STATUS_SUCCEEDED:
        return None
    # az returns every event in the resource group; a sibling pool VM's power
    # transitions must not leak into this VM's ledger.
    if needle not in resource_id.lower():
        return None
    kind: str | None = _power_event_kind(operation=operation_value)
    if kind is None:
        return None
    return PowerEvent(kind=kind, at=at)


def parse_power_events(*, payload: str, vm_name: str) -> list[PowerEvent] | None:
    """Read power transitions for ``vm_name`` out of an ``az monitor activity-log`` payload.

    ``None`` means the log could not be read, which is materially different from ``[]``
    ("read it, nothing happened"). Collapsing the two is what would let an unreadable log
    masquerade as a machine that never stopped.
    """
    try:
        decoded: object = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list):
        return None
    needle: str = f"{_COMPUTES_PATH_SEGMENT}{vm_name}".lower()
    seen: set[PowerEvent] = set()
    events: list[PowerEvent] = []
    for element in decoded:
        event: PowerEvent | None = _activity_element_to_event(element=element, needle=needle)
        if event is None:
            continue
        if event in seen:
            continue
        seen.add(event)
        events.append(event)
    events.sort(key=_power_event_sort_key)
    return events


def get_power_events(*, vm: VmPoolEntry, since: str) -> list[PowerEvent] | None:
    """Ask Azure when ``vm`` was started and stopped since ``since``.

    The provider's own history is the only thing that can see a stop the provisioner was
    not present for -- an idle-watchdog shutdown, or a hand-run ``az ml compute stop``.
    """
    result: CommandResult = _run_az(
        args=[
            "monitor",
            "activity-log",
            "list",
            "--resource-group",
            vm.resource_group,
            "--start-time",
            since,
            "--max-events",
            _ACTIVITY_LOG_MAX_EVENTS,
            "-o",
            "json",
        ],
    )
    if result.returncode != 0:
        return None
    return parse_power_events(payload=result.stdout, vm_name=vm.name)


def _append_segment(
    *,
    segments: list[BillableSegment],
    started_at: str,
    ended_at: str,
    source: str,
) -> None:
    hours: float | None = _hours_between(start=started_at, end=ended_at)
    if hours is None or hours <= 0.0:
        return
    segments.append(
        BillableSegment(
            started_at=started_at,
            ended_at=ended_at,
            hours=hours,
            source=source,
        ),
    )


def build_billable_segments(
    *,
    billing_started_at: str,
    billing_ended_at: str,
    power_events: list[PowerEvent],
    source: str,
) -> list[BillableSegment]:
    """Thin wrapper over :func:`build_billable_window` returning only the segments."""
    return build_billable_window(
        billing_started_at=billing_started_at,
        billing_ended_at=billing_ended_at,
        power_events=power_events,
        source=source,
    ).segments


def build_billable_window(
    *,
    billing_started_at: str,
    billing_ended_at: str,
    power_events: list[PowerEvent],
    source: str,
) -> BillableWindow:
    """Reconstruct the intervals during which the machine was powered on and billing.

    The machine is treated as running at ``billing_started_at``; the window's own
    boundaries are authoritative, so events at or outside them are ignored.

    ``running_at_end`` reports what the walk concluded, rather than leaving callers to
    infer it by comparing the last segment's ``ended_at`` against the window end. That
    comparison is wrong in two ways this module cares about: the same instant spelled
    ``...Z`` and ``...+00:00`` compares unequal, and a machine stopped at the window's
    final instant loses its zero-length segment and would read as still running. The
    inference drives the verified/assumed/unresolved branch, so it decides whether a
    cost figure is published at all.
    """
    window_start: datetime | None = _parse_iso(value=billing_started_at)
    window_end: datetime | None = _parse_iso(value=billing_ended_at)
    if window_start is None or window_end is None:
        return BillableWindow(segments=[], running_at_end=False)
    inside: list[PowerEvent] = []
    for event in power_events:
        at: datetime | None = _parse_iso(value=event.at)
        if at is None:
            continue
        if at <= window_start or at >= window_end:
            continue
        inside.append(event)
    # Stable, so two events sharing a timestamp keep the order the provider reported them.
    inside.sort(key=_power_event_sort_key)

    segments: list[BillableSegment] = []
    open_at: str = billing_started_at
    running: bool = True
    for event in inside:
        if event.kind == POWER_EVENT_STOP and running:
            _append_segment(
                segments=segments,
                started_at=open_at,
                ended_at=event.at,
                source=source,
            )
            running = False
        elif event.kind == POWER_EVENT_START and not running:
            open_at = event.at
            running = True
    if running:
        _append_segment(
            segments=segments,
            started_at=open_at,
            ended_at=billing_ended_at,
            source=source,
        )
    return BillableWindow(segments=segments, running_at_end=running)


def _resolved_attribution(
    *,
    billing_started_at: str,
    billing_anchor: str,
    billing_ended_at: str,
    hourly_cost_usd: float,
    wall_clock_span_hours: float,
    segments: list[BillableSegment],
    method: str,
    status: str,
    notes: str,
) -> CostAttribution:
    billable_hours: float = sum((segment.hours for segment in segments), 0.0)
    stopped_hours: float = max(0.0, wall_clock_span_hours - billable_hours)
    return CostAttribution(
        billing_started_at=billing_started_at,
        billing_anchor=billing_anchor,
        billing_ended_at=billing_ended_at,
        method=method,
        status=status,
        segments=segments,
        billable_hours=billable_hours,
        wall_clock_span_hours=wall_clock_span_hours,
        stopped_hours=stopped_hours,
        hourly_cost_usd=hourly_cost_usd,
        total_cost_usd=hourly_cost_usd * billable_hours,
        notes=notes,
    )


def _unresolved_attribution(
    *,
    billing_started_at: str,
    billing_anchor: str,
    billing_ended_at: str,
    hourly_cost_usd: float,
    wall_clock_span_hours: float | None,
    segments: list[BillableSegment],
    method: str,
    notes: str,
) -> CostAttribution:
    return CostAttribution(
        billing_started_at=billing_started_at,
        billing_anchor=billing_anchor,
        billing_ended_at=billing_ended_at,
        method=method,
        status=COST_STATUS_UNRESOLVED,
        segments=segments,
        billable_hours=None,
        wall_clock_span_hours=wall_clock_span_hours,
        stopped_hours=None,
        hourly_cost_usd=hourly_cost_usd,
        total_cost_usd=None,
        notes=notes,
    )


def compute_cost_attribution(
    *,
    billing_started_at: str,
    billing_anchor: str,
    billing_ended_at: str,
    hourly_cost_usd: float,
    power_events: list[PowerEvent] | None,
    running_at_teardown: bool,
) -> CostAttribution:
    """Attribute a machine's cost to the hours it actually billed.

    ``running_at_teardown`` is what the provisioner observed about the machine's power
    state when teardown began. Cross-checking it against the provider's history is what
    catches a stop nothing recorded -- the case where the only honest answer is "unknown".
    """
    span: float | None = _hours_between(start=billing_started_at, end=billing_ended_at)
    if span is None:
        # An unparseable endpoint means the window itself is unknown. Reporting a 0.0
        # span here would flow a confident, "verified" $0.00 into machine_log.json --
        # exactly the fabricated number this whole type exists to prevent.
        return _unresolved_attribution(
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
            billing_ended_at=billing_ended_at,
            hourly_cost_usd=hourly_cost_usd,
            wall_clock_span_hours=None,
            segments=[],
            method=COST_METHOD_UNRESOLVED,
            notes=(
                f"The billing window could not be read: billing_started_at "
                f"{billing_started_at!r} and billing_ended_at {billing_ended_at!r} do not "
                f"both parse as ISO 8601 instants. {_UNRESOLVED_REMEDIATION_NOTE}"
            ),
        )
    wall_clock_span_hours: float = span

    if power_events is None:
        if not running_at_teardown:
            return _unresolved_attribution(
                billing_started_at=billing_started_at,
                billing_anchor=billing_anchor,
                billing_ended_at=billing_ended_at,
                hourly_cost_usd=hourly_cost_usd,
                wall_clock_span_hours=wall_clock_span_hours,
                segments=[],
                method=COST_METHOD_UNRESOLVED,
                notes=(
                    "The machine was found stopped at teardown and the provider activity "
                    "log could not be read, so the moment billing ended is unknown. "
                    f"{_UNRESOLVED_REMEDIATION_NOTE}"
                ),
            )
        return _resolved_attribution(
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
            billing_ended_at=billing_ended_at,
            hourly_cost_usd=hourly_cost_usd,
            wall_clock_span_hours=wall_clock_span_hours,
            segments=build_billable_segments(
                billing_started_at=billing_started_at,
                billing_ended_at=billing_ended_at,
                power_events=[],
                source=SEGMENT_SOURCE_PROVISIONER,
            ),
            method=COST_METHOD_PROVISIONER_OBSERVED,
            status=COST_STATUS_ASSUMED,
            notes=(
                "The provider activity log could not be read. The machine was running at "
                f"teardown, so the figure assumes it billed for the whole "
                f"{wall_clock_span_hours:.2f} h window; an intervening stop would make it "
                "an over-estimate."
            ),
        )

    window: BillableWindow = build_billable_window(
        billing_started_at=billing_started_at,
        billing_ended_at=billing_ended_at,
        power_events=power_events,
        source=SEGMENT_SOURCE_ACTIVITY_LOG,
    )
    segments: list[BillableSegment] = window.segments
    log_running_at_end: bool = window.running_at_end
    if log_running_at_end and not running_at_teardown:
        # Something stopped the machine that the activity log did not record. The
        # segments stay as evidence, but no end time may be invented for the last one.
        return _unresolved_attribution(
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
            billing_ended_at=billing_ended_at,
            hourly_cost_usd=hourly_cost_usd,
            wall_clock_span_hours=wall_clock_span_hours,
            segments=segments,
            method=COST_METHOD_ACTIVITY_LOG,
            notes=(
                "The activity log shows the machine still running, but it was found "
                "stopped at teardown, so nothing records when it stopped. The segments "
                f"below are evidence, not a total. {_UNRESOLVED_REMEDIATION_NOTE}"
            ),
        )
    if log_running_at_end == running_at_teardown:
        status: str = COST_STATUS_VERIFIED
        notes: str = (
            f"Reconstructed {len(segments)} billable segment(s) from the provider "
            f"activity log; the provider's final state agrees with what the provisioner "
            f"observed at teardown."
        )
    else:
        status = COST_STATUS_ASSUMED
        notes = (
            "The activity log shows the machine stopped, but it was running at teardown, "
            "so the log missed a start. The segments below are a lower bound on billable "
            "time and the total may under-count."
        )
    return _resolved_attribution(
        billing_started_at=billing_started_at,
        billing_anchor=billing_anchor,
        billing_ended_at=billing_ended_at,
        hourly_cost_usd=hourly_cost_usd,
        wall_clock_span_hours=wall_clock_span_hours,
        segments=segments,
        method=COST_METHOD_ACTIVITY_LOG,
        status=status,
        notes=notes,
    )


def resolve_billing_anchor(
    *,
    billing_started_at: str | None,
    billing_anchor: str | None,
    acquired_at: str | None,
    fallback: str,
) -> BillingAnchor:
    """Pick the instant billing began, preferring the acquire-time anchor.

    Falling back to ``acquired_at`` under-bills by the boot window, which is why the
    anchor kind travels with the timestamp instead of being inferred later.
    """
    if billing_started_at is not None:
        kind: str = billing_anchor if billing_anchor is not None else BILLING_ANCHOR_AZ_START
        return BillingAnchor(started_at=billing_started_at, kind=kind)
    if acquired_at is not None:
        return BillingAnchor(started_at=acquired_at, kind=BILLING_ANCHOR_ACQUIRED_AT)
    return BillingAnchor(started_at=fallback, kind=BILLING_ANCHOR_ACQUIRED_AT)


# ---------------------------------------------------------------------------
# Acquire / Run / Teardown
# ---------------------------------------------------------------------------


class PoolBusyError(RuntimeError):
    """Raised when no VM in the pool can be acquired."""


def _try_acquire_one(*, vm: VmPoolEntry, task_id: str) -> AcquireAttempt:
    """Attempt to acquire ``vm`` for ``task_id``, releasing it if abandoned.

    Wraps ``_attempt_acquire_one_detailed``: when that attempt started ``vm``
    itself but did not end up acquiring it, this stops the VM again before
    returning, so a failed acquire never leaves a billing VM for the caller
    to notice and stop by hand (see
    ``tasks/t0055_fix_truncation_regenerate_predictions/intervention/
    setup_machines_ft-arf-weu-v1.md``).
    """
    attempt: AcquireAttempt = _attempt_acquire_one_detailed(vm=vm, task_id=task_id)
    if not attempt.acquired and attempt.started_vm:
        stop_result: CommandResult = stop_compute(vm=vm)
        if stop_result.returncode != 0:
            sys.stderr.write(
                f"failed to stop {vm.name} after abandoning a failed acquire attempt: "
                f"{stop_result.stderr.strip() or stop_result.stdout.strip()}\n",
            )
    return attempt


def _attempt_acquire_one(
    *,
    vm: VmPoolEntry,
    task_id: str,
) -> tuple[bool, FailedAttempt | None, bool]:
    """Back-compat adapter over ``_attempt_acquire_one_detailed``.

    Kept because callers outside the pool loop only care about the three original
    outcomes; the billing anchor is only meaningful to ``acquire``.
    """
    attempt: AcquireAttempt = _attempt_acquire_one_detailed(vm=vm, task_id=task_id)
    return attempt.acquired, attempt.failure, attempt.started_vm


def _attempt_acquire_one_detailed(*, vm: VmPoolEntry, task_id: str) -> AcquireAttempt:
    """Attempt to acquire ``vm`` for ``task_id``.

    ``started_vm`` reports whether this attempt issued a *successful* az start call. It
    gates two things: whether ``_try_acquire_one`` must release the VM it started, and
    whether any wasted cost may be reported at all -- a VM this attempt never started
    billed nothing on this attempt's account.
    """
    attempt_start: float = _now_monotonic()
    timestamp: str = _now_iso()
    state_result: ComputeStateResult = get_compute_state_result(vm=vm)
    state: str = state_result.state
    started_vm: bool = False
    # A VM that was already running was started by someone else at a time nothing
    # observable records, so the honest anchor for this task is the moment it took over.
    billing_started_at: str = timestamp
    billing_anchor: str = BILLING_ANCHOR_ALREADY_RUNNING

    non_retryable_phase: str | None = non_retryable_state_failure_phase(
        failure=state_result.failure,
    )
    if non_retryable_phase is not None:
        return AcquireAttempt(
            acquired=False,
            failure=FailedAttempt(
                vm_name=vm.name,
                failure_reason=(
                    f"az ml compute show on {vm.name} failed with a non-retryable "
                    f"{non_retryable_phase} error; waiting cannot resolve it"
                ),
                failure_phase=non_retryable_phase,
                duration_seconds=_now_monotonic() - attempt_start,
                wasted_cost_usd=0.0,
                timestamp=timestamp,
                failure_reason_detail=state_result.stderr,
            ),
            started_vm=False,
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
        )

    if state == _STATE_STOPPING:
        return AcquireAttempt(
            acquired=False,
            failure=FailedAttempt(
                vm_name=vm.name,
                failure_reason=f"VM {vm.name} is stopping; will not race",
                failure_phase=AZ_SHOW_FAILURE_PHASE,
                duration_seconds=_now_monotonic() - attempt_start,
                wasted_cost_usd=0.0,
                timestamp=timestamp,
            ),
            started_vm=False,
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
        )

    if state == _STATE_STOPPED:
        # Captured before the call, not after: Azure begins charging when the start
        # request is issued, and the boot window that follows is real spend.
        billing_started_at = _now_iso()
        billing_anchor = BILLING_ANCHOR_AZ_START
        start_result: CommandResult = start_compute(vm=vm)
        if start_result.returncode != 0:
            # The call was rejected, so nothing was started and nothing billed.
            return AcquireAttempt(
                acquired=False,
                failure=FailedAttempt(
                    vm_name=vm.name,
                    failure_reason=(
                        f"az ml compute start failed: "
                        f"{start_result.stderr.strip() or start_result.stdout.strip()}"
                    ),
                    failure_phase=AZ_START_FAILURE_PHASE,
                    duration_seconds=_now_monotonic() - attempt_start,
                    wasted_cost_usd=0.0,
                    timestamp=timestamp,
                    failure_reason_detail=_stderr_or_none(raw=start_result.stderr),
                ),
                started_vm=False,
                billing_started_at=billing_started_at,
                billing_anchor=billing_anchor,
            )
        started_vm = True

    deadline: float = _now_monotonic() + VM_START_TIMEOUT_SECONDS
    if state != _STATE_RUNNING and not _wait_for_state(
        vm=vm, target=_STATE_RUNNING, deadline=deadline
    ):
        return AcquireAttempt(
            acquired=False,
            failure=FailedAttempt(
                vm_name=vm.name,
                failure_reason=(
                    f"VM {vm.name} did not reach running state within "
                    f"{int(VM_START_TIMEOUT_SECONDS)}s"
                ),
                failure_phase=VM_START_TIMEOUT_FAILURE_PHASE,
                duration_seconds=_now_monotonic() - attempt_start,
                wasted_cost_usd=_wasted_cost(
                    hourly=vm.hourly_cost_usd,
                    seconds=_now_monotonic() - attempt_start,
                    started_vm=started_vm,
                ),
                timestamp=timestamp,
            ),
            started_vm=started_vm,
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
        )

    if not _wait_for_ssh(vm=vm, deadline=deadline):
        return AcquireAttempt(
            acquired=False,
            failure=FailedAttempt(
                vm_name=vm.name,
                failure_reason=f"SSH did not come up on {vm.name} within deadline",
                failure_phase=SSH_CONNECT_FAILURE_PHASE,
                duration_seconds=_now_monotonic() - attempt_start,
                wasted_cost_usd=_wasted_cost(
                    hourly=vm.hourly_cost_usd,
                    seconds=_now_monotonic() - attempt_start,
                    started_vm=started_vm,
                ),
                timestamp=timestamp,
            ),
            started_vm=started_vm,
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
        )

    existing_locks: list[str] = list_remote_locks(vm=vm)
    foreign_locks: list[str] = [lock for lock in existing_locks if lock != task_id]
    if len(foreign_locks) > 0:
        return AcquireAttempt(
            acquired=False,
            failure=FailedAttempt(
                vm_name=vm.name,
                failure_reason=(f"VM {vm.name} already locked by: {', '.join(foreign_locks)}"),
                failure_phase=LOCK_HELD_FAILURE_PHASE,
                duration_seconds=_now_monotonic() - attempt_start,
                wasted_cost_usd=0.0,
                timestamp=timestamp,
            ),
            started_vm=started_vm,
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
        )

    # Before the lock, not after: a locked VM is one this task is committed to, and
    # a box that fails preflight is one whose training job dies on disconnect or
    # whose checkpoints land on an ephemeral disk. Failing here lets the pool loop
    # fall through to the next entry instead of stranding the task on it.
    if run_remote_preflight(vm=vm) is None:
        return AcquireAttempt(
            acquired=False,
            failure=FailedAttempt(
                vm_name=vm.name,
                failure_reason=(
                    f"VM {vm.name} failed remote preflight (systemd lingering or the "
                    f"persistent-storage mount); see the step log for the script's stderr"
                ),
                failure_phase=PREFLIGHT_FAILURE_PHASE,
                duration_seconds=_now_monotonic() - attempt_start,
                wasted_cost_usd=_wasted_cost(
                    hourly=vm.hourly_cost_usd,
                    seconds=_now_monotonic() - attempt_start,
                    started_vm=started_vm,
                ),
                timestamp=timestamp,
            ),
            started_vm=started_vm,
            billing_started_at=billing_started_at,
            billing_anchor=billing_anchor,
        )

    place_remote_lock(vm=vm, task_id=task_id)
    return AcquireAttempt(
        acquired=True,
        failure=None,
        started_vm=started_vm,
        billing_started_at=billing_started_at,
        billing_anchor=billing_anchor,
    )


def _estimate_wasted_cost(*, hourly: float, seconds: float) -> float:
    return hourly * (seconds / _SECONDS_PER_HOUR)


def _wasted_cost(*, hourly: float, seconds: float, started_vm: bool) -> float:
    """Cost this attempt actually caused by giving up on ``vm``.

    A VM this attempt never started billed nothing on this attempt's account, whoever
    else may have been paying for it. Reporting elapsed-time-times-rate regardless is
    fictional spend, and it lands in ``aggregate_machines``' wasted-cost total as though
    it were real.
    """
    if not started_vm:
        return 0.0
    return _estimate_wasted_cost(hourly=hourly, seconds=seconds)


def _find_pool_vm(*, pool: list[VmPoolEntry], vm_name: str) -> VmPoolEntry:
    matched: list[VmPoolEntry] = [vm for vm in pool if vm.name == vm_name]
    if len(matched) == 0:
        available: str = ", ".join(vm.name for vm in pool)
        raise RuntimeError(
            f"--vm-name {vm_name!r} does not match any VM in pool (available: {available})"
        )
    return matched[0]


def acquire(
    *,
    task_id: str,
    pool: list[VmPoolEntry] | None = None,
    intervention_dir: Path | None = None,
    vm_name: str | None = None,
) -> AcquireResult:
    pool_to_use: list[VmPoolEntry] = pool if pool is not None else load_pool()
    if len(pool_to_use) == 0:
        raise RuntimeError("VM pool is empty; check project/azure_vm.json")

    if vm_name is not None:
        pool_to_use = [_find_pool_vm(pool=pool_to_use, vm_name=vm_name)]

    search_started_at: str = _now_iso()
    search_started_monotonic: float = _now_monotonic()
    failed: list[FailedAttempt] = []

    for vm in pool_to_use:
        attempt: AcquireAttempt = _try_acquire_one(vm=vm, task_id=task_id)
        if attempt.acquired:
            ready_at: str = _now_iso()
            total: float = _now_monotonic() - search_started_monotonic
            return AcquireResult(
                task_id=task_id,
                vm=vm,
                acquired_at=ready_at,
                search_started_at=search_started_at,
                ready_at=ready_at,
                total_provisioning_seconds=total,
                started_vm=attempt.started_vm,
                billing_started_at=attempt.billing_started_at,
                billing_anchor=attempt.billing_anchor,
                failed_attempts=failed,
            )
        if attempt.failure is not None:
            failed.append(attempt.failure)

    _write_pool_busy_intervention(
        task_id=task_id,
        failed=failed,
        intervention_dir=intervention_dir,
    )
    summary: str = "; ".join(f"{f.vm_name}: {f.failure_reason}" for f in failed)
    raise PoolBusyError(f"all VMs in pool unavailable -> {summary}")


def _pool_busy_filename(*, failed: list[FailedAttempt]) -> str:
    # Keyed on the attempted VM name(s) so that parallel subagents provisioning different
    # --vm-name-pinned VMs for the same task_id each get their own intervention file instead of
    # overwriting one another's (see t0055's intervention/pool_busy.md note).
    vm_slugs: list[str] = sorted({f.vm_name.lower() for f in failed})
    if len(vm_slugs) == 0:
        return "pool_busy.md"
    return f"pool_busy_{'_'.join(vm_slugs)}.md"


def _write_pool_busy_intervention(
    *,
    task_id: str,
    failed: list[FailedAttempt],
    intervention_dir: Path | None,
) -> None:
    base: Path
    if intervention_dir is not None:
        base = intervention_dir
    else:
        base = POOL_CONFIG_PATH.parent.parent / "tasks" / task_id / "intervention"
    base.mkdir(parents=True, exist_ok=True)
    path: Path = base / _pool_busy_filename(failed=failed)
    lines: list[str] = [
        "# Azure ML compute pool busy",
        "",
        f"Task `{task_id}` could not acquire a VM from `project/azure_vm.json`.",
        "",
        "## Attempts",
        "",
    ]
    for f in failed:
        lines.append(f"* **{f.vm_name}** ({f.failure_phase}): {f.failure_reason}")
        if f.failure_reason_detail is not None:
            lines.extend(["", "  ```text", f"  {f.failure_reason_detail.strip()}", "  ```"])
    lines.extend(
        [
            "",
            "## Resolution",
            "",
            "* Check Slack (`#rail-arf-serving` for the westeurope entries, which share a",
            "  NCADSH100v5 quota ceiling) to see whether another team is using the pool.",
            "* Confirm each VM's state with `az ml compute show --name <vm-name> "
            "--resource-group <resource_group> --workspace-name <workspace> -o json`,",
            "  reading `<resource_group>` and `<workspace>` from that VM's entry in",
            "  `project/azure_vm.json` -- the pool spans three workspaces, so there is no",
            "  single correct `--workspace-name`.",
            "* An `az_show_authorization` phase above is an RBAC problem, not a busy pool:",
            "  the identity running `az` needs read access to the workspace's computes.",
            "* Once a VM is free, re-run the setup-machines step.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    task_id: str,
    command: str,
    vm: VmPoolEntry | None = None,
    pool: list[VmPoolEntry] | None = None,
    estimated_duration_seconds: float | None = None,
    progress_callback: Any = None,
) -> RunResult:
    """Execute ``command`` on the VM currently locked by ``task_id``.

    Streams nothing — captures stdout/stderr at the end. Heartbeats are
    surfaced via ``progress_callback`` (a ``Callable[[dict[str, Any]], None]``)
    if supplied; the test suite uses this to assert intervals without sleeping.

    Heartbeats fire every 5 minutes; for jobs estimated longer than 2 hours,
    a checkpoint reminder fires every 30 minutes alongside the heartbeat.
    """
    target_vm: VmPoolEntry = _resolve_locked_vm(task_id=task_id, vm=vm, pool=pool)
    started_monotonic: float = _now_monotonic()

    proc: subprocess.Popen[str] = subprocess.Popen(  # noqa: S603
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            target_vm.ssh_host_alias,
            command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    heartbeats: int = 0
    next_heartbeat: float = started_monotonic + HEARTBEAT_INTERVAL_SECONDS
    next_checkpoint: float = started_monotonic + CHECKPOINT_INTERVAL_SECONDS
    is_long_job: bool = (
        estimated_duration_seconds is not None
        and estimated_duration_seconds > LONG_JOB_THRESHOLD_SECONDS
    )

    while True:
        if proc.poll() is not None:
            break
        now: float = _now_monotonic()
        if now >= next_heartbeat:
            heartbeats += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "kind": "heartbeat",
                        "elapsed_seconds": now - started_monotonic,
                        "heartbeat_index": heartbeats,
                    },
                )
            next_heartbeat = now + HEARTBEAT_INTERVAL_SECONDS
        if is_long_job and now >= next_checkpoint:
            if progress_callback is not None:
                progress_callback(
                    {
                        "kind": "checkpoint_reminder",
                        "elapsed_seconds": now - started_monotonic,
                    },
                )
            next_checkpoint = now + CHECKPOINT_INTERVAL_SECONDS
        time.sleep(1.0)

    proc.wait()
    duration: float = _now_monotonic() - started_monotonic
    return RunResult(
        task_id=task_id,
        vm_name=target_vm.name,
        exit_code=proc.returncode,
        duration_seconds=duration,
        heartbeats=heartbeats,
    )


def _resolve_locked_vm(
    *,
    task_id: str,
    vm: VmPoolEntry | None,
    pool: list[VmPoolEntry] | None,
) -> VmPoolEntry:
    if vm is not None:
        return vm
    pool_to_use: list[VmPoolEntry] = pool if pool is not None else load_pool()
    for candidate in pool_to_use:
        if task_id in list_remote_locks(vm=candidate):
            return candidate
    raise RuntimeError(
        f"no VM in the pool currently holds a lock for task {task_id}",
    )


def teardown(
    *,
    task_id: str,
    deallocate: bool = True,
    vm: VmPoolEntry | None = None,
    pool: list[VmPoolEntry] | None = None,
    acquired_at: str | None = None,
    billing_started_at: str | None = None,
    billing_anchor: str | None = None,
) -> TeardownResult:
    target_vm: VmPoolEntry = _resolve_locked_vm(task_id=task_id, vm=vm, pool=pool)

    # Clearing the on-VM lock file needs SSH, which needs the VM running. A VM stopped outside
    # azure_ml_vm (by hand, or by the idle watchdog) must still be releasable through this one
    # function -- see t0055's intervention/setup_machines_ft-arf-weu-v1.md, where a hand-stopped
    # VM's stale lock could not be cleared and blocked a later task's acquire for days.
    # Read once, before anything this function does changes it. A VM found stopped here
    # stopped at a moment nothing in this process observed, which is exactly the case the
    # cost attribution must refuse to guess at.
    state_at_teardown: str = get_compute_state(vm=target_vm)
    started_from_stopped: bool = state_at_teardown == _STATE_STOPPED
    running_at_teardown: bool = state_at_teardown == _STATE_RUNNING
    if started_from_stopped:
        start_compute(vm=target_vm)
        start_deadline: float = _now_monotonic() + VM_START_TIMEOUT_SECONDS
        started_ok: bool = _wait_for_state(
            vm=target_vm,
            target=_STATE_RUNNING,
            deadline=start_deadline,
        ) and _wait_for_ssh(vm=target_vm, deadline=start_deadline)
        if not started_ok:
            raise RuntimeError(
                f"could not start {target_vm.name} to release task {task_id}'s lock; the VM "
                "must be reachable over SSH to clear the on-VM lock file",
            )

    if not clear_remote_lock(vm=target_vm, task_id=task_id):
        raise RuntimeError(
            f"failed to clear the lock for {task_id} on {target_vm.name}; the on-VM lock file "
            "may still be present -- do not treat this task's VM as released",
        )
    kill_task_vllm_processes(vm=target_vm, task_id=task_id)

    other_locks: list[str] = [lock for lock in list_remote_locks(vm=target_vm) if lock != task_id]
    other_locks_present: bool = len(other_locks) > 0

    # Restore the VM to how we found it: if this call started it just to reach it over SSH,
    # stop it again even when a sibling lock is present -- a sibling lock on a VM that was
    # stopped is necessarily stale (nobody can be actively using a stopped VM), so it never
    # protects a live task the way it does when the VM was already running. `deallocate=False`
    # (the `--keep-running` override) still wins over restoring the original state, since it is
    # an explicit request to leave the VM up.
    deallocated: bool = False
    if deallocate and (not other_locks_present or started_from_stopped):
        stop_result: CommandResult = stop_compute(vm=target_vm)
        deallocated = stop_result.returncode == 0

    destroyed_at: str = _now_iso()
    anchor: BillingAnchor = resolve_billing_anchor(
        billing_started_at=billing_started_at,
        billing_anchor=billing_anchor,
        acquired_at=acquired_at,
        fallback=destroyed_at,
    )
    attribution: CostAttribution = compute_cost_attribution(
        billing_started_at=anchor.started_at,
        billing_anchor=anchor.kind,
        billing_ended_at=destroyed_at,
        hourly_cost_usd=target_vm.hourly_cost_usd,
        power_events=get_power_events(vm=target_vm, since=anchor.started_at),
        running_at_teardown=running_at_teardown,
    )

    return TeardownResult(
        task_id=task_id,
        vm_name=target_vm.name,
        deallocated=deallocated,
        other_locks_present=other_locks_present,
        destroyed_at=destroyed_at,
        duration_hours=attribution.billable_hours,
        total_cost_usd=attribution.total_cost_usd,
        cost_attribution=attribution,
    )


# ---------------------------------------------------------------------------
# machine_log.json shaping
# ---------------------------------------------------------------------------


def to_machine_log_entry(
    *,
    acquire_result: AcquireResult,
    teardown_result: TeardownResult | None = None,
    image: str = "azure-ml-system-managed",
    disk_gb: int = 0,
    label: str | None = None,
    cuda_version: str = "unknown",
    checkpoint_path: str | None = None,
    heartbeat_path: str | None = None,
) -> dict[str, Any]:
    """Produce a machine_log.json-compatible entry from acquire/teardown results.

    The shape mirrors the vast.ai-era schema (see
    ``arf/specifications/remote_machines_specification.md``) so that
    ``aggregate_machines.py`` continues to work unchanged.
    """
    vm: VmPoolEntry = acquire_result.vm
    entry: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "provider": PROVIDER,
        "instance_id": vm.name,
        "vm_name": vm.name,
        "offer_id": 0,
        "search_criteria": {
            "gpu_name": GPU_MODEL,
            "num_gpus": GPU_COUNT,
            "min_gpu_ram": 80.0,
            "min_cpu_ram": None,
            "min_disk": None,
            "min_reliability": 1.0,
            "extra_filters": "azure-ml-pool",
        },
        "selected_offer": {
            "offer_id": 0,
            "gpu": SELECTED_OFFER_GPU_DISPLAY,
            "gpu_count": GPU_COUNT,
            "gpu_ram_gb": 80.0,
            "cpu_ram_gb": 0.0,
            "disk_gb": 0.0,
            "price_per_hour": vm.hourly_cost_usd,
            "reliability": 1.0,
            "location": "azure-eastus2",
        },
        "selection_rationale": (
            f"Acquired {vm.name} from the Azure ML pool (priority {vm.priority}) "
            f"at ${vm.hourly_cost_usd:.2f}/hr."
        ),
        "image": image,
        "disk_gb": disk_gb,
        "label": label,
        "ssh_host": vm.ssh_host_alias,
        "ssh_port": 0,
        "gpu_verified": SELECTED_OFFER_GPU_DISPLAY,
        "cuda_version": cuda_version,
        "created_at": acquire_result.acquired_at,
        "ready_at": acquire_result.ready_at,
        "destroyed_at": None,
        "total_duration_hours": None,
        "total_cost_usd": None,
        # Cost attribution. The rate lives at the top level so a reviewer can redo the
        # arithmetic from the entry alone rather than looking up the pool config.
        "hourly_cost_usd": vm.hourly_cost_usd,
        "started_vm": acquire_result.started_vm,
        "billing_started_at": acquire_result.billing_started_at,
        "billing_anchor": acquire_result.billing_anchor,
        "billing_ended_at": None,
        "cost_attribution_method": None,
        "cost_attribution_status": None,
        "cost_attribution_notes": None,
        "billable_segments": [],
        "billable_hours": None,
        "wall_clock_span_hours": None,
        "stopped_hours": None,
        "search_started_at": acquire_result.search_started_at,
        "total_provisioning_seconds": acquire_result.total_provisioning_seconds,
        "failed_attempts": [
            {
                "offer_id": 0,
                "instance_id": f.vm_name,
                "gpu": SELECTED_OFFER_GPU_DISPLAY,
                "failure_reason": f.failure_reason,
                "failure_phase": f.failure_phase,
                "failure_reason_detail": f.failure_reason_detail,
                "duration_seconds": f.duration_seconds,
                "wasted_cost_usd": f.wasted_cost_usd,
                "timestamp": f.timestamp,
            }
            for f in acquire_result.failed_attempts
        ],
        "checkpoint_path": checkpoint_path,
        "heartbeat_path": heartbeat_path,
    }
    if teardown_result is not None:
        attribution: CostAttribution = teardown_result.cost_attribution
        entry["destroyed_at"] = teardown_result.destroyed_at
        entry["total_duration_hours"] = teardown_result.duration_hours
        entry["total_cost_usd"] = teardown_result.total_cost_usd
        entry["billing_started_at"] = attribution.billing_started_at
        entry["billing_anchor"] = attribution.billing_anchor
        entry["billing_ended_at"] = attribution.billing_ended_at
        entry["cost_attribution_method"] = attribution.method
        entry["cost_attribution_status"] = attribution.status
        entry["cost_attribution_notes"] = attribution.notes
        entry["billable_segments"] = [asdict(segment) for segment in attribution.segments]
        entry["billable_hours"] = attribution.billable_hours
        entry["wall_clock_span_hours"] = attribution.wall_clock_span_hours
        entry["stopped_hours"] = attribution.stopped_hours
    return entry


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_acquire(*, result: AcquireResult) -> None:
    payload: dict[str, Any] = {
        "task_id": result.task_id,
        "vm_name": result.vm.name,
        "ssh_host_alias": result.vm.ssh_host_alias,
        "hourly_cost_usd": result.vm.hourly_cost_usd,
        "acquired_at": result.acquired_at,
        "started_vm": result.started_vm,
        "billing_started_at": result.billing_started_at,
        "billing_anchor": result.billing_anchor,
        "search_started_at": result.search_started_at,
        "total_provisioning_seconds": result.total_provisioning_seconds,
        "failed_attempts": [asdict(f) for f in result.failed_attempts],
    }
    print(json.dumps(payload, indent=2))


def _print_teardown(*, result: TeardownResult) -> None:
    print(json.dumps(asdict(result), indent=2))


def _print_run(*, result: RunResult) -> None:
    print(json.dumps(asdict(result), indent=2))


def main(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="azure_ml_vm",
        description="Azure ML compute instance provisioner via SSH",
    )
    sub: argparse._SubParsersAction[argparse.ArgumentParser] = parser.add_subparsers(
        dest="command",
        required=True,
    )

    acquire_parser: argparse.ArgumentParser = sub.add_parser(
        "acquire",
        help="Acquire a VM from the pool for a task",
    )
    acquire_parser.add_argument("task_id")
    acquire_parser.add_argument(
        "--vm-name",
        type=str,
        default=None,
        help=(
            "Pin acquisition to a single VM in the pool by name. When set, the "
            "command will not roll over to other VMs and will fail with exit 75 "
            "(pool busy) if the named VM is locked or unavailable. Use this when "
            "a task's measurements are statistically valid only on one specific VM "
            "(e.g., paired comparisons against a frozen baseline measured on that VM)."
        ),
    )

    teardown_parser: argparse.ArgumentParser = sub.add_parser(
        "teardown",
        help="Release the VM held by a task",
    )
    teardown_parser.add_argument("task_id")
    teardown_parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Do not deallocate the VM after releasing the lock",
    )
    teardown_parser.add_argument(
        "--acquired-at",
        type=str,
        default=None,
        help=(
            "ISO 8601 timestamp from acquire output. Fallback billing anchor only: it "
            "under-bills by the boot window, so prefer --billing-started-at"
        ),
    )
    teardown_parser.add_argument(
        "--billing-started-at",
        type=str,
        default=None,
        help=(
            "ISO 8601 billing_started_at from acquire output -- the instant the provider "
            "began charging, which precedes acquired_at by the boot window"
        ),
    )
    teardown_parser.add_argument(
        "--billing-anchor",
        type=str,
        default=None,
        help="billing_anchor from acquire output, recording how the anchor was determined",
    )
    teardown_parser.add_argument(
        "--vm-name",
        type=str,
        default=None,
        help=(
            "Pin release to a single VM in the pool by name, from this task's own acquire "
            "output. Without it, resolving the locked VM walks the pool over SSH, which cannot "
            "find a lock on a VM that is currently stopped (by hand, or by the idle watchdog) "
            "-- pass this when the VM might be stopped."
        ),
    )

    run_parser: argparse.ArgumentParser = sub.add_parser(
        "run",
        help="Execute a shell command on the VM held by a task",
    )
    run_parser.add_argument("task_id")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    args: argparse.Namespace = parser.parse_args(argv)

    if args.command == "acquire":
        try:
            result: AcquireResult = acquire(task_id=args.task_id, vm_name=args.vm_name)
        except PoolBusyError as err:
            print(f"pool busy: {err}", file=sys.stderr)
            return EXIT_POOL_BUSY
        except Exception as err:  # noqa: BLE001
            print(f"acquire failed: {err}", file=sys.stderr)
            return EXIT_GENERIC_ERROR
        _print_acquire(result=result)
        return EXIT_OK

    if args.command == "teardown":
        try:
            pinned_vm: VmPoolEntry | None = None
            if args.vm_name is not None:
                pinned_vm = _find_pool_vm(pool=load_pool(), vm_name=args.vm_name)
            t_result: TeardownResult = teardown(
                task_id=args.task_id,
                deallocate=not args.keep_running,
                vm=pinned_vm,
                acquired_at=args.acquired_at,
                billing_started_at=args.billing_started_at,
                billing_anchor=args.billing_anchor,
            )
        except Exception as err:  # noqa: BLE001
            print(f"teardown failed: {err}", file=sys.stderr)
            return EXIT_GENERIC_ERROR
        _print_teardown(result=t_result)
        return EXIT_OK

    if args.command == "run":
        if len(args.command) == 0:
            print("run: a remote command is required after --", file=sys.stderr)
            return EXIT_GENERIC_ERROR
        # argparse.REMAINDER keeps the leading "--" when present.
        tokens: list[str] = (
            args.command[1:] if len(args.command) > 0 and args.command[0] == "--" else args.command
        )
        joined: str = " ".join(shlex.quote(tok) for tok in tokens)
        try:
            r_result: RunResult = run(task_id=args.task_id, command=joined)
        except Exception as err:  # noqa: BLE001
            print(f"run failed: {err}", file=sys.stderr)
            return EXIT_GENERIC_ERROR
        _print_run(result=r_result)
        return r_result.exit_code

    return EXIT_GENERIC_ERROR


if __name__ == "__main__":
    sys.exit(main())
