"""Dema daemon — local presence loop.

A "tick" is a no-network, no-desktop heartbeat that:
  1. confirms the profile is present
  2. reads current mission state
  3. appends one tick entry to today's daily log
  4. emits a tick receipt

The daemon supports two modes:

  --once     Run a single tick and exit (Phase A0 baseline).
  --loop     Run ticks continuously, separated by --interval-seconds, until
             SIGINT/SIGTERM or --max-ticks is reached. A PID file under
             sovereign_state/dema/runtime/ prevents two instances from
             racing.

No autonomous action is taken. The daemon does not call out, does not write
to MEMORY.md, does not touch the network or desktop, does not post to
social.

Usage:
    python scripts/dema/dema_daemon.py --once
    python scripts/dema/dema_daemon.py --loop --interval-seconds 60 --max-ticks 5
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - native Windows is documented as later-phase.
    fcntl = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema import (  # noqa: E402
    DailyLog,
    DailyLogEntry,
    DemaReceipt,
    MissionStateMachine,
    ProfileStore,
    ReceiptWriter,
)

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"
DEFAULT_INTERVAL_SECONDS = 60.0
DEFAULT_LOOP_MAX_SECONDS = 24 * 3600  # safety ceiling on a single loop run


class _Stop:
    """Module-level flag flipped by SIGINT/SIGTERM handlers."""

    requested = False
    reason = ""


class _Lock:
    """Process-held POSIX lock state for the daemon PID file."""

    fd: int | None = None
    path: Path | None = None


def _install_signal_handlers() -> None:
    def _handle(signum: int, _frame: Any) -> None:
        _Stop.requested = True
        _Stop.reason = signal.Signals(signum).name

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (OSError, ValueError):
            # Some test harnesses or non-main threads forbid signal install;
            # in those cases _Stop.requested can still be flipped manually
            # by tests.
            pass


def _runtime_dir(root: Path) -> Path:
    d = root / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_file(root: Path) -> Path:
    return _runtime_dir(root) / "dema_daemon.pid"


def _read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _acquire_lock(root: Path) -> tuple[bool, str, Path]:
    path = _pid_file(root)
    current_pid = os.getpid()

    if fcntl is None:
        return False, "POSIX fcntl locks are required for --loop mode", path

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        existing = _read_pid(path)
        os.close(fd)
        if existing is not None:
            return False, f"another dema daemon already running (pid={existing})", path
        return False, f"daemon lock is already held at {path}", path
    except OSError:
        os.close(fd)
        raise

    existing = _read_pid(path)
    if existing is not None and existing != current_pid and _process_alive(existing):
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return False, f"another dema daemon already running (pid={existing})", path

    os.ftruncate(fd, 0)
    os.write(fd, str(current_pid).encode("utf-8"))
    os.fsync(fd)
    _Lock.fd = fd
    _Lock.path = path
    return True, "acquired", path


def _release_lock(path: Path) -> None:
    if _Lock.fd is None or _Lock.path != path:
        return
    if fcntl is None:
        return
    try:
        if _read_pid(path) == os.getpid():
            path.unlink(missing_ok=True)
    finally:
        fcntl.flock(_Lock.fd, fcntl.LOCK_UN)
        os.close(_Lock.fd)
        _Lock.fd = None
        _Lock.path = None


def _interruptible_sleep(seconds: float, check_interval: float = 0.05) -> None:
    """Sleep in small increments so SIGINT/SIGTERM is responsive."""
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if _Stop.requested:
            return
        remaining = deadline - time.monotonic()
        time.sleep(min(check_interval, max(0.0, remaining)))


def tick(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    profile = ProfileStore(root).load()
    state = MissionStateMachine(root).get()

    log = DailyLog(root)
    receipt = DemaReceipt(
        action="dema.daemon.tick",
        truth_label="MEASURED",
        touched_paths=[
            str(root / "logs"),
            str(root / "receipts"),
        ],
        not_touched_paths=[
            "network",
            "desktop",
            "MEMORY.md",
            "docs/canon/",
            "social",
        ],
        approval_required=False,
        approval_status="n/a",
        payload={
            "profile_present": profile is not None,
            "mission_truth_label": state.truth_label,
            "actionable": state.is_actionable(),
        },
    )
    rid, receipt_path = ReceiptWriter(root).write(receipt)

    log_path = log.append(
        DailyLogEntry(
            timestamp=receipt.timestamp,
            kind="tick",
            summary=(
                "tick: profile=%s state=%s"
                % (
                    "present" if profile else "missing",
                    state.truth_label,
                )
            ),
            receipt_id=rid,
            metadata={"actionable": state.is_actionable()},
        )
    )

    return {
        "ok": True,
        "kind": "dema_daemon_tick",
        "receipt_id": rid,
        "receipt_path": str(receipt_path),
        "log_path": str(log_path),
        "profile_present": profile is not None,
        "mission_truth_label": state.truth_label,
    }


def loop(
    root: Path,
    *,
    interval_seconds: float,
    max_ticks: int | None,
    max_seconds: float | None = DEFAULT_LOOP_MAX_SECONDS,
) -> dict[str, Any]:
    """Run the daemon loop until SIGINT/SIGTERM, max_ticks, or max_seconds."""
    if interval_seconds < 0:
        raise ValueError("interval_seconds must be >= 0")
    if max_ticks is not None and max_ticks < 0:
        raise ValueError("max_ticks must be >= 0")

    root.mkdir(parents=True, exist_ok=True)
    _Stop.requested = False
    _Stop.reason = ""
    acquired, lock_msg, lock_path = _acquire_lock(root)
    if not acquired:
        return {
            "ok": False,
            "kind": "dema_daemon_loop",
            "reason": lock_msg,
            "lock_path": str(lock_path),
        }

    _install_signal_handlers()
    started_at = time.monotonic()
    ticks_done = 0
    last_receipt_id: str | None = None
    last_log_path: str | None = None
    stop_reason = "max_ticks" if max_ticks is not None else "loop"

    try:
        while True:
            if _Stop.requested:
                stop_reason = f"signal:{_Stop.reason}"
                break
            if max_ticks is not None and ticks_done >= max_ticks:
                stop_reason = "max_ticks"
                break
            if (
                max_seconds is not None
                and (time.monotonic() - started_at) >= max_seconds
            ):
                stop_reason = "max_seconds"
                break

            out = tick(root)
            ticks_done += 1
            last_receipt_id = str(out["receipt_id"])
            last_log_path = str(out["log_path"])

            if max_ticks is not None and ticks_done >= max_ticks:
                stop_reason = "max_ticks"
                break

            _interruptible_sleep(interval_seconds)
    finally:
        _release_lock(lock_path)

    return {
        "ok": True,
        "kind": "dema_daemon_loop",
        "ticks_done": ticks_done,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started_at,
        "interval_seconds": interval_seconds,
        "max_ticks": max_ticks,
        "last_receipt_id": last_receipt_id,
        "last_log_path": last_log_path,
        "lock_path": str(lock_path),
        "lock_released": not lock_path.exists(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Run one tick and exit.")
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Run ticks continuously until signal or --max-ticks/--max-seconds.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=f"Seconds between ticks in --loop mode (default {DEFAULT_INTERVAL_SECONDS}).",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Stop after this many ticks (default: unbounded).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=DEFAULT_LOOP_MAX_SECONDS,
        help=f"Stop after this many seconds (default {DEFAULT_LOOP_MAX_SECONDS}).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )
    args = parser.parse_args(argv)

    if args.once:
        out = tick(args.root)
    else:
        out = loop(
            args.root,
            interval_seconds=args.interval_seconds,
            max_ticks=args.max_ticks,
            max_seconds=args.max_seconds,
        )

    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
