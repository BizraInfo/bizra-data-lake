"""Dema service — operator-facing wrapper around the ambient daemon.

Commands:
  status                       JSON report on the running daemon, last tick,
                               profile, and lock state.
  start-once                   Run a single tick (calls into dema_daemon.tick).
  print-systemd-user-unit      Emit a Linux systemd --user unit file. The
                               unit is template-only: no public network
                               exposure, no autonomous social, no MEMORY.md.
  print-windows-task-command   Emit a placeholder Task Scheduler command.
                               WSL caveat documented; native Windows ships
                               in a later phase.
  doctor                       Sanity check: profile present, lock state,
                               last tick recency, no rogue paths outside
                               the sandbox root.

Every command's output goes to stdout as JSON (or text for unit/cmd
templates). Nothing the service itself does crosses the kernel's
not_touched_paths boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema import DailyLog, MissionStateMachine, ProfileStore  # noqa: E402

# Reuse the daemon's lock + tick helpers without spawning a subprocess.
import scripts.dema.dema_daemon as dema_daemon  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"

# Constants for systemd template — kept verbatim in tests so the boundary
# is auditable.
SYSTEMD_NO_NETWORK_DECLARATION = (
    "# Boundary: no public network exposure. The unit only runs the local "
    "Dema daemon."
)
WINDOWS_NOT_NATIVE_TODAY = (
    "Windows Task Scheduler support is a placeholder in v0.1. Use WSL or "
    "run the daemon manually from a Linux shell. Native Windows ships in "
    "a later phase."
)


def cmd_status(root: Path) -> dict[str, Any]:
    pid_path = dema_daemon._pid_file(root)
    pid = dema_daemon._read_pid(pid_path)
    alive = bool(pid and dema_daemon._process_alive(pid))

    profile = ProfileStore(root).load()
    state = MissionStateMachine(root).get()
    log = DailyLog(root)
    today = log.read_today()
    last_tick: dict[str, Any] | None = None
    for entry in reversed(today):
        if entry.kind == "tick":
            last_tick = entry.to_dict()
            break

    return {
        "kind": "dema_service_status",
        "schema_version": "0.1.0",
        "running": alive,
        "pid": pid,
        "lock_path": str(pid_path),
        "profile_present": profile is not None,
        "mission_truth_label": state.truth_label,
        "log_today_count": len(today),
        "last_tick": last_tick,
        "root": str(root),
    }


def cmd_start_once(root: Path) -> dict[str, Any]:
    return dema_daemon.tick(root)


def cmd_print_systemd_user_unit(root: Path, *, python: str | None = None) -> str:
    py = python or sys.executable
    daemon_script = (
        Path(dema_daemon.__file__).resolve().relative_to(REPO_ROOT.resolve())
    )
    unit = f"""# Dema Ambient Service v0.1 — systemd --user unit
{SYSTEMD_NO_NETWORK_DECLARATION}
#
# Install:
#   mkdir -p ~/.config/systemd/user
#   cp dema.service ~/.config/systemd/user/dema.service
#   systemctl --user daemon-reload
#   systemctl --user enable --now dema.service
# Logs (operator):
#   journalctl --user -u dema.service -f
# Local Dema state lives at:
#   {root}

[Unit]
Description=Dema Ambient Service (BIZRA Phase A0.5)
After=default.target

[Service]
Type=simple
ExecStart={py} {REPO_ROOT}/{daemon_script} --loop --interval-seconds 60 --root {root}
Restart=on-failure
RestartSec=15s
# No public network exposure: the daemon does not bind any port.
PrivateNetwork=true
# Hardening: confine writes to the operator's dema state root.
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={root}

[Install]
WantedBy=default.target
"""
    return unit


def cmd_print_windows_task_command(root: Path, *, python: str | None = None) -> str:
    py = python or sys.executable
    daemon_script = (
        Path(dema_daemon.__file__).resolve().relative_to(REPO_ROOT.resolve())
    )
    note = WINDOWS_NOT_NATIVE_TODAY
    cmd = (
        'schtasks /Create /SC ONLOGON /TN "Dema Ambient Service" /TR '
        f'"\\"{py}\\" \\"{REPO_ROOT}\\\\{daemon_script}\\" --loop '
        f'--interval-seconds 60 --root \\"{root}\\""'
    )
    return f"# {note}\n{cmd}\n"


def cmd_doctor(root: Path) -> dict[str, Any]:
    status = cmd_status(root)
    profile_ok = status["profile_present"]
    lock_path = Path(status["lock_path"])
    pid_path_under_root = lock_path.resolve().is_relative_to(root.resolve())

    last_tick_recent = False
    last_tick_age_seconds: float | None = None
    if status.get("last_tick"):
        try:
            ts = status["last_tick"]["timestamp"]
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - dt).total_seconds()
            last_tick_age_seconds = age
            last_tick_recent = age < 24 * 3600
        except (KeyError, ValueError, TypeError):
            last_tick_recent = False

    findings = []
    if not profile_ok:
        findings.append("no profile yet — run scripts/dema/dema_onboarding.py --init")
    if not pid_path_under_root:
        findings.append(f"lock path {lock_path} escapes sandbox root {root}")
    if status.get("last_tick") is None:
        findings.append("no tick recorded today — run --once or --loop")
    elif not last_tick_recent:
        findings.append(f"last tick {last_tick_age_seconds:.0f}s old (>24h)")

    healthy = not findings
    return {
        "kind": "dema_service_doctor",
        "schema_version": "0.1.0",
        "healthy": healthy,
        "findings": findings,
        "status": status,
        "last_tick_age_seconds": last_tick_age_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "status",
            "start-once",
            "print-systemd-user-unit",
            "print-windows-task-command",
            "doctor",
        ],
        help="Service command to run.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )
    args = parser.parse_args(argv)

    if args.command == "status":
        print(json.dumps(cmd_status(args.root), indent=2, sort_keys=True))
    elif args.command == "start-once":
        print(json.dumps(cmd_start_once(args.root), indent=2, sort_keys=True))
    elif args.command == "print-systemd-user-unit":
        sys.stdout.write(cmd_print_systemd_user_unit(args.root))
    elif args.command == "print-windows-task-command":
        sys.stdout.write(cmd_print_windows_task_command(args.root))
    elif args.command == "doctor":
        out = cmd_doctor(args.root)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0 if out["healthy"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
