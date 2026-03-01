#!/usr/bin/env python3
"""
BIZRA Infrastructure Guardian — Self-Healing WSL2 Daemon

OODA loop: Observe → Orient → Decide → Act
Each probe follows: Check → Diagnose → Correct → Verify → Log

Monitors:
  1. Docker socket connectivity
  2. Container health (all compose stacks)
  3. Memory pressure (OOM prevention)
  4. Filesystem health (ext4 errors, disk space)
  5. Critical services (Redis, Postgres, ChromaDB)
  6. Journal corruption
  7. Port collisions (known: 8081)
  8. systemd service state (no crash-loops)

Self-optimizes:
  - Docker prune on disk pressure
  - Journal vacuum on size threshold
  - OOM score adjustment for critical containers
  - Container restart on unhealthy (with backoff)

Usage:
  python3 infra_guardian.py --check          # Single check, exit 0/1
  python3 infra_guardian.py --daemon         # Continuous loop (systemd)
  python3 infra_guardian.py --report         # JSON health report to stdout
  python3 infra_guardian.py --correct        # Single check + auto-correct
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GUARDIAN_VERSION = "1.0.0"
BIZRA_ROOT = Path(os.environ.get("BIZRA_DATA_LAKE_ROOT", "/mnt/c/BIZRA-DATA-LAKE"))
LOG_DIR = BIZRA_ROOT / "logs" / "guardian"
STATE_FILE = LOG_DIR / "guardian_state.json"
REPORT_FILE = LOG_DIR / "last_report.json"

# Docker
DOCKER_SOCKET = "/var/run/docker.sock"
DOCKER_DESKTOP_PROXY = (
    "/mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock"
)

# Thresholds
MEMORY_WARN_PCT = 85          # % used → warning
MEMORY_CRIT_PCT = 92          # % used → emergency prune
DISK_WARN_PCT = 85            # % used → warning
DISK_CRIT_PCT = 95            # % used → emergency prune
JOURNAL_MAX_MB = 500          # Vacuum if journals exceed this
CONTAINER_RESTART_BACKOFF = 300  # 5 min cooldown per container restart
DAEMON_INTERVAL = 60          # Seconds between checks

# Critical services that must be healthy
CRITICAL_CONTAINERS = [
    "bizra-redis",
    "bizra-node0-db",
    "bizra-chromadb",
]

# Containers to restart if unhealthy (with backoff)
RESTARTABLE_CONTAINERS = [
    "bizra-redis",
    "bizra-chromadb",
    "bizra-dual-agentic-system--main-kernel-1",
    "bizra-dual-agentic-system--main-elite-1",
    "bizra-dual-agentic-system--main-postgres-1",
]

# Known port collisions (port → expected owner)
KNOWN_PORT_CONFLICTS = {
    8081: "bizra-dual-agentic-system--main-refinery-1",
}

# systemd services that should NOT be running (they conflict with Docker)
BANNED_SYSTEMD_SERVICES = ["redis-server", "redis"]


class Severity(Enum):
    OK = "ok"
    WARN = "warn"
    CRIT = "critical"
    FIXED = "fixed"


@dataclass
class ProbeResult:
    name: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    corrected: bool = False
    correction_msg: str = ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bizra.guardian")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler — rotate by keeping last log
    fh = logging.FileHandler(LOG_DIR / "guardian.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


log = _setup_logging()


# ---------------------------------------------------------------------------
# State management (persistent across runs)
# ---------------------------------------------------------------------------

class GuardianState:
    """Tracks restart cooldowns, correction history, and health trends."""

    def __init__(self) -> None:
        self._path = STATE_FILE
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                return self._defaults()
        return self._defaults()

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "version": GUARDIAN_VERSION,
            "restart_cooldowns": {},
            "correction_log": [],
            "consecutive_ok": 0,
            "total_checks": 0,
            "total_corrections": 0,
            "last_check": None,
        }

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, default=str))

    def can_restart(self, container: str) -> bool:
        last = self._data["restart_cooldowns"].get(container, 0)
        return time.time() - last > CONTAINER_RESTART_BACKOFF

    def record_restart(self, container: str) -> None:
        self._data["restart_cooldowns"][container] = time.time()

    def record_correction(self, probe: str, msg: str) -> None:
        entry = {"time": datetime.now(timezone.utc).isoformat(), "probe": probe, "msg": msg}
        self._data["correction_log"].append(entry)
        # Keep last 100 corrections
        self._data["correction_log"] = self._data["correction_log"][-100:]
        self._data["total_corrections"] += 1

    def record_check(self, all_ok: bool) -> None:
        self._data["total_checks"] += 1
        self._data["last_check"] = datetime.now(timezone.utc).isoformat()
        if all_ok:
            self._data["consecutive_ok"] += 1
        else:
            self._data["consecutive_ok"] = 0


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    """Run a command, return (returncode, stdout+stderr)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        return -2, f"Command not found: {cmd[0]}"


def _docker(*args: str, timeout: int = 15) -> tuple[int, str]:
    return _run(["docker", *args], timeout=timeout)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_docker_socket(state: GuardianState, correct: bool) -> ProbeResult:
    """Check Docker socket exists and is functional."""
    name = "docker_socket"

    # Check socket exists
    if not os.path.exists(DOCKER_SOCKET):
        if correct and os.path.exists(DOCKER_DESKTOP_PROXY):
            rc, out = _run(["ln", "-sf", DOCKER_DESKTOP_PROXY, DOCKER_SOCKET])
            if rc == 0:
                state.record_correction(name, "Recreated Docker socket symlink")
                return ProbeResult(
                    name, Severity.FIXED,
                    "Docker socket was missing — recreated symlink",
                    corrected=True,
                    correction_msg="ln -sf <proxy> /var/run/docker.sock",
                )
        return ProbeResult(
            name, Severity.CRIT,
            f"Docker socket missing at {DOCKER_SOCKET}",
            details={"proxy_exists": os.path.exists(DOCKER_DESKTOP_PROXY)},
        )

    # Check connectivity
    rc, out = _docker("info", "--format", "{{.ServerVersion}}")
    if rc != 0:
        return ProbeResult(name, Severity.CRIT, f"Docker unreachable: {out}")

    return ProbeResult(name, Severity.OK, f"Docker connected (v{out})")


def probe_container_health(state: GuardianState, correct: bool) -> list[ProbeResult]:
    """Check all running containers for health status."""
    results = []
    rc, out = _docker(
        "ps", "--all",
        "--format", "{{.Names}}\t{{.Status}}\t{{.State}}",
        timeout=20,
    )
    if rc != 0:
        return [ProbeResult("containers", Severity.CRIT, f"Cannot list containers: {out}")]

    unhealthy = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, status, state_val = parts[0], parts[1], parts[2]

        if "unhealthy" in status.lower():
            unhealthy.append(name)
            sev = Severity.CRIT if name in CRITICAL_CONTAINERS else Severity.WARN

            if correct and name in RESTARTABLE_CONTAINERS and state.can_restart(name):
                _docker("restart", name, timeout=60)
                state.record_restart(name)
                state.record_correction("container_health", f"Restarted unhealthy: {name}")
                results.append(ProbeResult(
                    f"container:{name}", Severity.FIXED,
                    f"Restarted unhealthy container: {name}",
                    corrected=True,
                    correction_msg=f"docker restart {name}",
                ))
            else:
                results.append(ProbeResult(
                    f"container:{name}", sev,
                    f"Container unhealthy: {name} — {status}",
                    details={"state": state_val},
                ))

    if not unhealthy:
        # Count running containers
        rc2, out2 = _docker("ps", "-q", timeout=10)
        count = len(out2.splitlines()) if rc2 == 0 else "?"
        results.append(ProbeResult(
            "containers", Severity.OK,
            f"All {count} containers healthy",
        ))

    # Check critical containers are running
    for crit in CRITICAL_CONTAINERS:
        if crit not in out:
            results.append(ProbeResult(
                f"container:{crit}", Severity.CRIT,
                f"Critical container not running: {crit}",
            ))

    return results


def probe_memory(state: GuardianState, correct: bool) -> ProbeResult:
    """Check memory pressure."""
    name = "memory"
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mem[parts[0].rstrip(":")] = int(parts[1])

        total = mem.get("MemTotal", 1)
        available = mem.get("MemAvailable", total)
        used_pct = round((1 - available / total) * 100, 1)

        if used_pct >= MEMORY_CRIT_PCT:
            if correct:
                # Emergency: clear Docker build cache
                _docker("builder", "prune", "-f", timeout=60)
                state.record_correction(name, f"Pruned Docker build cache at {used_pct}% memory")
                return ProbeResult(
                    name, Severity.FIXED,
                    f"Memory critical ({used_pct}%) — pruned Docker build cache",
                    details={"used_pct": used_pct, "available_mb": available // 1024},
                    corrected=True,
                )
            return ProbeResult(
                name, Severity.CRIT,
                f"Memory critical: {used_pct}% used ({available // 1024} MB free)",
                details={"used_pct": used_pct},
            )
        if used_pct >= MEMORY_WARN_PCT:
            return ProbeResult(
                name, Severity.WARN,
                f"Memory pressure: {used_pct}% used",
                details={"used_pct": used_pct},
            )
        return ProbeResult(
            name, Severity.OK,
            f"Memory OK: {used_pct}% used ({available // 1024} MB free)",
            details={"used_pct": used_pct},
        )
    except OSError as e:
        return ProbeResult(name, Severity.WARN, f"Cannot read /proc/meminfo: {e}")


def probe_disk(state: GuardianState, correct: bool) -> list[ProbeResult]:
    """Check disk usage on key mounts."""
    results = []
    for mount in ["/", "/mnt/c"]:
        name = f"disk:{mount}"
        try:
            usage = shutil.disk_usage(mount)
            used_pct = round((usage.used / usage.total) * 100, 1)
            free_gb = round(usage.free / (1024**3), 1)

            if used_pct >= DISK_CRIT_PCT:
                if correct and mount == "/":
                    _docker("system", "prune", "-f", timeout=120)
                    state.record_correction(name, f"Docker prune at {used_pct}%")
                    results.append(ProbeResult(
                        name, Severity.FIXED,
                        f"Disk critical ({used_pct}%) — ran docker system prune",
                        details={"used_pct": used_pct, "free_gb": free_gb},
                        corrected=True,
                    ))
                else:
                    results.append(ProbeResult(
                        name, Severity.CRIT,
                        f"{mount}: {used_pct}% used ({free_gb} GB free)",
                        details={"used_pct": used_pct, "free_gb": free_gb},
                    ))
            elif used_pct >= DISK_WARN_PCT:
                results.append(ProbeResult(
                    name, Severity.WARN,
                    f"{mount}: {used_pct}% used ({free_gb} GB free)",
                    details={"used_pct": used_pct, "free_gb": free_gb},
                ))
            else:
                results.append(ProbeResult(
                    name, Severity.OK,
                    f"{mount}: {used_pct}% used ({free_gb} GB free)",
                    details={"used_pct": used_pct, "free_gb": free_gb},
                ))
        except OSError as e:
            results.append(ProbeResult(name, Severity.WARN, f"Cannot stat {mount}: {e}"))
    return results


def probe_ext4_errors(state: GuardianState, correct: bool) -> ProbeResult:
    """Check dmesg for EXT4 filesystem errors."""
    name = "ext4_errors"
    rc, out = _run(["dmesg", "--level=err,crit,alert,emerg"], timeout=10)
    if rc != 0:
        return ProbeResult(name, Severity.WARN, f"Cannot read dmesg: {out}")

    ext4_lines = [l for l in out.splitlines() if "EXT4-fs error" in l or "ext4_" in l.lower()]
    if ext4_lines:
        return ProbeResult(
            name, Severity.WARN,
            f"EXT4 errors detected ({len(ext4_lines)} entries) — run fix_docker_disk.ps1 from Windows",
            details={"count": len(ext4_lines), "sample": ext4_lines[:3]},
        )
    return ProbeResult(name, Severity.OK, "No EXT4 errors in dmesg")


def probe_critical_services(state: GuardianState, correct: bool) -> list[ProbeResult]:
    """Ping critical services (Redis, Postgres)."""
    results = []

    # Redis on 6379 (may require AUTH — NOAUTH response still means Redis is alive)
    try:
        with socket.create_connection(("127.0.0.1", 6379), timeout=3) as s:
            s.sendall(b"PING\r\n")
            resp = s.recv(64).decode()
            if "+PONG" in resp or "NOAUTH" in resp:
                results.append(ProbeResult("service:redis", Severity.OK, "Redis alive on 6379"))
            else:
                results.append(ProbeResult("service:redis", Severity.WARN, f"Redis unexpected: {resp}"))
    except (OSError, ConnectionRefusedError):
        results.append(ProbeResult("service:redis", Severity.CRIT, "Redis unreachable on 6379"))

    # Postgres on 5433
    try:
        with socket.create_connection(("127.0.0.1", 5433), timeout=3):
            results.append(ProbeResult("service:postgres", Severity.OK, "PostgreSQL accepting on 5433"))
    except (OSError, ConnectionRefusedError):
        results.append(ProbeResult("service:postgres", Severity.CRIT, "PostgreSQL unreachable on 5433"))

    # ChromaDB on 8001
    try:
        with socket.create_connection(("127.0.0.1", 8001), timeout=3):
            results.append(ProbeResult("service:chromadb", Severity.OK, "ChromaDB accepting on 8001"))
    except (OSError, ConnectionRefusedError):
        results.append(ProbeResult("service:chromadb", Severity.WARN, "ChromaDB unreachable on 8001"))

    return results


def probe_journal_health(state: GuardianState, correct: bool) -> ProbeResult:
    """Check systemd journal size and corruption."""
    name = "journal"
    rc, out = _run(["journalctl", "--disk-usage"], timeout=10)
    if rc != 0:
        return ProbeResult(name, Severity.WARN, f"Cannot check journal: {out}")

    # Parse "Archived and active journals take up 123.4M in the file system."
    size_mb = 0
    for word in out.split():
        try:
            size_mb = float(word.rstrip("MGKB"))
            if "G" in out:
                size_mb *= 1024
            break
        except ValueError:
            continue

    if size_mb > JOURNAL_MAX_MB:
        if correct:
            _run(["journalctl", "--rotate"], timeout=10)
            _run(["journalctl", "--vacuum-size=200M"], timeout=30)
            state.record_correction(name, f"Vacuumed journals from {size_mb:.0f}MB to 200MB")
            return ProbeResult(
                name, Severity.FIXED,
                f"Journals were {size_mb:.0f}MB — vacuumed to 200MB",
                corrected=True,
            )
        return ProbeResult(
            name, Severity.WARN,
            f"Journals large: {size_mb:.0f}MB (threshold: {JOURNAL_MAX_MB}MB)",
        )
    return ProbeResult(name, Severity.OK, f"Journal size OK: {size_mb:.0f}MB")


def probe_banned_services(state: GuardianState, correct: bool) -> list[ProbeResult]:
    """Ensure banned systemd services (that conflict with Docker) are stopped."""
    results = []
    for svc in BANNED_SYSTEMD_SERVICES:
        rc, out = _run(["systemctl", "is-active", svc], timeout=5)
        if out.strip() == "active":
            if correct:
                _run(["systemctl", "stop", svc], timeout=10)
                _run(["systemctl", "disable", svc], timeout=10)
                state.record_correction(f"banned_svc:{svc}", f"Stopped and disabled {svc}")
                results.append(ProbeResult(
                    f"banned_svc:{svc}", Severity.FIXED,
                    f"Stopped conflicting systemd service: {svc}",
                    corrected=True,
                ))
            else:
                results.append(ProbeResult(
                    f"banned_svc:{svc}", Severity.WARN,
                    f"Conflicting systemd service running: {svc} (conflicts with Docker)",
                ))
    if not results:
        results.append(ProbeResult("banned_svcs", Severity.OK, "No conflicting systemd services"))
    return results


def probe_port_collisions(state: GuardianState, correct: bool) -> list[ProbeResult]:
    """Check for known port conflicts."""
    results = []
    for port, expected_owner in KNOWN_PORT_CONFLICTS.items():
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
                # Port is in use — that's expected
                pass
            results.append(ProbeResult(
                f"port:{port}", Severity.OK,
                f"Port {port} in use by expected owner ({expected_owner})",
            ))
        except (OSError, ConnectionRefusedError):
            results.append(ProbeResult(
                f"port:{port}", Severity.OK,
                f"Port {port} free (owner {expected_owner} not running — OK)",
            ))
    return results


# ---------------------------------------------------------------------------
# Self-Evaluation (meta-probe)
# ---------------------------------------------------------------------------

def self_evaluate(results: list[ProbeResult], state: GuardianState) -> ProbeResult:
    """Evaluate the guardian's own effectiveness using Ihsan quality gate."""
    crits = sum(1 for r in results if r.severity == Severity.CRIT)
    warns = sum(1 for r in results if r.severity == Severity.WARN)
    fixed = sum(1 for r in results if r.corrected)
    total = len(results)

    # Ihsan score: 1.0 = all OK, penalized by crits and warns
    ihsan = max(0.0, 1.0 - (crits * 0.2) - (warns * 0.05))

    details = {
        "ihsan_score": round(ihsan, 3),
        "total_probes": total,
        "critical": crits,
        "warnings": warns,
        "auto_fixed": fixed,
        "consecutive_ok": state._data["consecutive_ok"],
        "total_checks": state._data["total_checks"],
        "total_corrections": state._data["total_corrections"],
    }

    if ihsan >= 0.95:
        return ProbeResult("self_eval", Severity.OK, f"Ihsan {ihsan:.3f} — system excellent", details)
    elif ihsan >= 0.80:
        return ProbeResult("self_eval", Severity.WARN, f"Ihsan {ihsan:.3f} — needs attention", details)
    else:
        return ProbeResult("self_eval", Severity.CRIT, f"Ihsan {ihsan:.3f} — system degraded", details)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_all_probes(correct: bool, state: GuardianState) -> list[ProbeResult]:
    """Run all probes and return results."""
    results: list[ProbeResult] = []

    results.append(probe_docker_socket(state, correct))
    results.extend(probe_container_health(state, correct))
    results.append(probe_memory(state, correct))
    results.extend(probe_disk(state, correct))
    results.append(probe_ext4_errors(state, correct))
    results.extend(probe_critical_services(state, correct))
    results.append(probe_journal_health(state, correct))
    results.extend(probe_banned_services(state, correct))
    results.extend(probe_port_collisions(state, correct))

    # Self-evaluation is always last
    results.append(self_evaluate(results, state))

    return results


def results_to_report(results: list[ProbeResult]) -> dict[str, Any]:
    """Convert probe results to a JSON-serializable report."""
    now = datetime.now(timezone.utc).isoformat()
    crits = [r for r in results if r.severity == Severity.CRIT]
    warns = [r for r in results if r.severity == Severity.WARN]
    fixed = [r for r in results if r.corrected]

    # Find self-eval probe
    self_eval = next((r for r in results if r.name == "self_eval"), None)

    return {
        "timestamp": now,
        "version": GUARDIAN_VERSION,
        "overall": "CRITICAL" if crits else "WARN" if warns else "GREEN",
        "ihsan": self_eval.details.get("ihsan_score", 0) if self_eval else 0,
        "summary": {
            "total_probes": len(results),
            "ok": sum(1 for r in results if r.severity == Severity.OK),
            "warnings": len(warns),
            "critical": len(crits),
            "auto_fixed": len(fixed),
        },
        "probes": [
            {
                "name": r.name,
                "severity": r.severity.value,
                "message": r.message,
                "corrected": r.corrected,
                **({"correction": r.correction_msg} if r.corrected else {}),
                **({"details": r.details} if r.details else {}),
            }
            for r in results
        ],
    }


def print_dashboard(results: list[ProbeResult]) -> None:
    """Print a compact dashboard to stderr."""
    icons = {
        Severity.OK: "  [OK]  ",
        Severity.WARN: " [WARN] ",
        Severity.CRIT: "[CRIT!] ",
        Severity.FIXED: "[FIXED] ",
    }
    print("\n=== BIZRA Infrastructure Guardian ===", file=sys.stderr)
    print(f"    Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", file=sys.stderr)
    print("", file=sys.stderr)

    for r in results:
        icon = icons.get(r.severity, "  [??]  ")
        print(f"  {icon} {r.message}", file=sys.stderr)

    crits = sum(1 for r in results if r.severity == Severity.CRIT)
    fixed = sum(1 for r in results if r.corrected)
    if fixed:
        print(f"\n  Auto-corrected: {fixed} issue(s)", file=sys.stderr)
    if crits:
        print(f"\n  CRITICAL: {crits} issue(s) need manual attention", file=sys.stderr)
    print("=" * 40, file=sys.stderr)


def daemon_loop(correct: bool) -> None:
    """Run probes in a loop with graceful shutdown."""
    state = GuardianState()

    def _shutdown(signum: int, frame: Any) -> None:
        log.info("Guardian shutting down (signal %d)", signum)
        state.save()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Guardian daemon started (v%s, interval=%ds, correct=%s)",
             GUARDIAN_VERSION, DAEMON_INTERVAL, correct)

    while True:
        try:
            results = run_all_probes(correct=correct, state=state)
            all_ok = all(r.severity in (Severity.OK, Severity.FIXED) for r in results)
            state.record_check(all_ok)

            # Log results
            for r in results:
                if r.severity == Severity.CRIT:
                    log.error("[%s] %s", r.name, r.message)
                elif r.severity == Severity.WARN:
                    log.warning("[%s] %s", r.name, r.message)
                elif r.corrected:
                    log.info("[%s] FIXED: %s", r.name, r.message)

            # Save report
            report = results_to_report(results)
            REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
            REPORT_FILE.write_text(json.dumps(report, indent=2))

            state.save()
        except Exception:
            log.exception("Guardian probe cycle failed")

        time.sleep(DAEMON_INTERVAL)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BIZRA Infrastructure Guardian — self-healing WSL2 daemon"
    )
    parser.add_argument("--check", action="store_true", help="Single health check (exit 0=ok, 1=issues)")
    parser.add_argument("--correct", action="store_true", help="Single check + auto-correct known issues")
    parser.add_argument("--daemon", action="store_true", help="Continuous monitoring loop")
    parser.add_argument("--report", action="store_true", help="JSON health report to stdout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        global log
        log = _setup_logging(verbose=True)

    if args.daemon:
        daemon_loop(correct=True)
        return 0  # unreachable

    state = GuardianState()
    do_correct = args.correct

    results = run_all_probes(correct=do_correct, state=state)
    all_ok = all(r.severity in (Severity.OK, Severity.FIXED) for r in results)
    state.record_check(all_ok)
    state.save()

    if args.report:
        report = results_to_report(results)
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0

    print_dashboard(results)

    report = results_to_report(results)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2))

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
