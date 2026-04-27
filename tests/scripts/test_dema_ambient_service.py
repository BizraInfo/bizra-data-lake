"""Dema Ambient Service v0.1 contract tests.

Covers:
  1. --loop --max-ticks 2 exits safely with 2 ticks recorded.
  2. Lock file lifecycle: acquire writes pid, release removes it.
  3. Second daemon refuses to start while a live pid is in the lock.
  4. dema_service status JSON shape.
  5. systemd --user unit text contains the no-public-network declaration.
  6. Windows command output carries the WSL placeholder note.
  7. doctor returns findings on cold-start, clean on healthy state.
  8. Daemon writes only under the supplied --root sandbox.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.dema.dema_daemon as dema_daemon  # noqa: E402


def _run(script: str, *args: str, root: Path, expect_zero: bool = True) -> dict | str:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dema" / script),
        *args,
        "--root",
        str(root),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
        check=expect_zero,
    )
    out = res.stdout
    # JSON-like outputs start with "{" or "["
    if out.lstrip().startswith(("{", "[")):
        return json.loads(out)
    return out


def _onboard(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMA_PREFERRED_NAME", "ServiceTest")
    _run("dema_onboarding.py", "--init", root=root)


# ── Daemon loop tests ──────────────────────────────────────────────


def test_loop_max_ticks_exits_safely(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)

    out = _run(
        "dema_daemon.py",
        "--loop",
        "--interval-seconds",
        "0",
        "--max-ticks",
        "2",
        "--max-seconds",
        "30",
        root=tmp_path,
    )
    assert out["ok"] is True
    assert out["kind"] == "dema_daemon_loop"
    assert out["ticks_done"] == 2
    assert out["stop_reason"] == "max_ticks"
    assert out["lock_released"] is True

    # Lock file removed after clean shutdown.
    pid_path = tmp_path / "runtime" / "dema_daemon.pid"
    assert not pid_path.exists()


def test_loop_lock_refuses_second_instance(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)

    # Plant a live pid in the lock (this process is alive by definition).
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    pid_path = runtime / "dema_daemon.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    out = _run(
        "dema_daemon.py",
        "--loop",
        "--interval-seconds",
        "0",
        "--max-ticks",
        "1",
        root=tmp_path,
        expect_zero=False,
    )
    assert out["ok"] is False
    assert "already running" in out["reason"]
    # Pre-planted lock file should NOT have been overwritten / removed.
    assert pid_path.exists()
    pid_path.unlink()  # cleanup


def test_loop_reclaims_stale_lock(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)

    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    pid_path = runtime / "dema_daemon.pid"
    pid_path.write_text("-1", encoding="utf-8")

    out = _run(
        "dema_daemon.py",
        "--loop",
        "--interval-seconds",
        "0",
        "--max-ticks",
        "1",
        root=tmp_path,
    )
    assert out["ok"] is True
    assert out["ticks_done"] == 1
    assert out["lock_released"] is True
    assert not pid_path.exists()


def test_loop_concurrent_stale_reclaim_allows_one_instance(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)

    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    pid_path = runtime / "dema_daemon.pid"
    pid_path.write_text("-1", encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dema" / "dema_daemon.py"),
        "--loop",
        "--interval-seconds",
        "2",
        "--max-ticks",
        "2",
        "--max-seconds",
        "10",
        "--root",
        str(tmp_path),
    ]
    procs = [
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
        for _ in range(2)
    ]

    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=15)
        assert stdout, stderr
        outputs.append(json.loads(stdout))

    assert sum(1 for out in outputs if out["ok"] is True) == 1
    assert sum(1 for out in outputs if out["ok"] is False) == 1
    failed = next(out for out in outputs if out["ok"] is False)
    assert "already running" in failed["reason"] or "lock" in failed["reason"]


def test_release_lock_keeps_foreign_pid(tmp_path):
    pid_path = dema_daemon._pid_file(tmp_path)
    foreign_pid = os.getpid() + 1
    pid_path.write_text(str(foreign_pid), encoding="utf-8")

    dema_daemon._release_lock(pid_path)

    assert pid_path.read_text(encoding="utf-8") == str(foreign_pid)


def test_loop_with_zero_max_ticks_is_a_no_op(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)

    out = _run(
        "dema_daemon.py",
        "--loop",
        "--interval-seconds",
        "0",
        "--max-ticks",
        "0",
        root=tmp_path,
    )
    assert out["ok"] is True
    assert out["ticks_done"] == 0
    assert out["stop_reason"] == "max_ticks"
    assert out["lock_released"] is True


# ── Service command tests ──────────────────────────────────────────


def test_service_status_shape(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)
    out = _run("dema_service.py", "status", root=tmp_path)
    assert out["kind"] == "dema_service_status"
    assert out["running"] is False
    assert out["profile_present"] is True
    assert out["lock_path"].endswith("dema_daemon.pid")
    assert out["log_today_count"] >= 1  # onboarding counted


def test_service_start_once_emits_tick_receipt(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)
    out = _run("dema_service.py", "start-once", root=tmp_path)
    assert out["ok"] is True
    assert out["kind"] == "dema_daemon_tick"


def test_systemd_unit_declares_no_public_network(tmp_path):
    out = _run("dema_service.py", "print-systemd-user-unit", root=tmp_path)
    assert isinstance(out, str)
    assert "PrivateNetwork=true" in out
    assert "no public network exposure" in out.lower()
    assert "ReadWritePaths" in out
    assert "ProtectSystem=strict" in out
    # Must reference the Dema state root.
    assert str(tmp_path) in out


def test_windows_task_command_carries_wsl_caveat(tmp_path):
    out = _run("dema_service.py", "print-windows-task-command", root=tmp_path)
    assert isinstance(out, str)
    assert "placeholder" in out.lower()
    assert "wsl" in out.lower()
    assert "schtasks" in out.lower()


def test_doctor_flags_cold_start(tmp_path):
    """Cold sandbox: doctor should fail (no profile, no tick) and exit non-zero."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dema" / "dema_service.py"),
        "doctor",
        "--root",
        str(tmp_path),
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60
    )
    assert res.returncode == 2
    out = json.loads(res.stdout)
    assert out["healthy"] is False
    assert any("no profile" in f for f in out["findings"])


def test_doctor_clean_after_onboard_and_tick(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)
    _run("dema_daemon.py", "--once", root=tmp_path)
    out = _run("dema_service.py", "doctor", root=tmp_path)
    assert out["healthy"] is True
    assert out["findings"] == []


# ── Sandbox-bound boundary ────────────────────────────────────────


def test_loop_writes_only_under_root(tmp_path, monkeypatch):
    _onboard(tmp_path, monkeypatch)
    out = _run(
        "dema_daemon.py",
        "--loop",
        "--interval-seconds",
        "0",
        "--max-ticks",
        "3",
        root=tmp_path,
    )
    assert out["ticks_done"] == 3
    last_log = Path(out["last_log_path"])
    assert last_log.is_relative_to(tmp_path)

    # Verify the onboarding receipt + 3 tick receipts all live under tmp.
    receipts_root = tmp_path / "receipts"
    written = list(receipts_root.rglob("*.json"))
    assert len(written) >= 4
    for p in written:
        assert p.is_relative_to(tmp_path)
