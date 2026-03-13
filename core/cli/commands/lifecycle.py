"""bizra start / stop / reset — Lifecycle management commands."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from typing import List

from ..registry import CommandResult
from ..shared import (
    API_PORT,
    BIZRA_HOME,
    BIZRA_LOGS,
    BIZRA_MODELS,
    OLLAMA_PORT,
    C,
    api_health,
    clear_pids,
    ensure_dirs,
    find_bizra_root,
    find_python,
    load_pids,
    port_in_use,
    print_error,
    print_info,
    print_status,
    print_warn,
    save_pid,
)


class StartCommand:
    name = "start"
    aliases = ("up",)
    description = "Start the sovereign runtime"
    category = "lifecycle"

    def execute(self, args: List[str]) -> CommandResult:
        foreground = "--foreground" in args or "-f" in args
        ensure_dirs()

        root = find_bizra_root()
        if not root:
            print_error(
                "Cannot find BIZRA source. Set BIZRA_ROOT environment variable."
            )
            return CommandResult.error("BIZRA source not found")

        health = api_health()
        if health:
            print_info(f"Sovereign runtime already running on :{API_PORT}")
            return CommandResult.ok("Already running")

        # Start Ollama if not running
        if not port_in_use(OLLAMA_PORT) and shutil.which("ollama"):
            print_info("Starting Ollama...")
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=open(BIZRA_LOGS / "ollama.log", "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            save_pid(proc.pid)
            time.sleep(2)

        py = find_python()
        venv_py = root / ".venv-linux" / "bin" / "python"
        venv_py_win = root / ".venv" / "Scripts" / "python.exe"
        if venv_py.exists():
            py = str(venv_py)
        elif venv_py_win.exists():
            py = str(venv_py_win)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["BIZRA_HOME"] = str(BIZRA_HOME)
        env["BIZRA_ENV"] = os.environ.get("BIZRA_ENV", "development")

        api_cmd = [
            py,
            "-m",
            "uvicorn",
            "core.sovereign.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(API_PORT),
            "--log-level",
            "info",
        ]

        if foreground:
            print_info(f"Starting sovereign runtime on :{API_PORT} (foreground)...")
            subprocess.run(api_cmd, cwd=str(root), env=env)
            return CommandResult.ok("Foreground process exited")

        print_info(f"Starting sovereign runtime on :{API_PORT}...")
        proc = subprocess.Popen(
            api_cmd,
            cwd=str(root),
            env=env,
            stdout=open(BIZRA_LOGS / "api.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        save_pid(proc.pid)

        for _i in range(15):
            time.sleep(1)
            if api_health():
                print_status("Sovereign runtime", f"Healthy (:{API_PORT})", True)
                return CommandResult.ok(f"Running on :{API_PORT}")
            sys.stdout.write(".")
            sys.stdout.flush()

        print()
        print_warn(
            "Runtime started but not yet healthy. Check logs: "
            + str(BIZRA_LOGS / "api.log")
        )
        return CommandResult.ok("Started (pending health)")


class StopCommand:
    name = "stop"
    aliases = ("down", "kill")
    description = "Stop all BIZRA services"
    category = "lifecycle"

    def execute(self, args: List[str]) -> CommandResult:
        pids = load_pids()
        if not pids:
            print_info("No BIZRA processes tracked.")
            return CommandResult.ok("Nothing to stop")

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                print_status(f"Process {pid}", "Stopped", True)
            except ProcessLookupError:
                print_status(f"Process {pid}", "Already stopped", True)
            except PermissionError:
                print_status(f"Process {pid}", "Permission denied", False)

        clear_pids()
        print_info("All BIZRA services stopped.")
        return CommandResult.ok("Stopped")


class ResetCommand:
    name = "reset"
    aliases = ()
    description = "Reset to factory (keeps identity, clears cache)"
    category = "lifecycle"

    def execute(self, args: List[str]) -> CommandResult:
        confirm = input(
            f"  {C.RED}This will clear cache and reflexes (identity preserved). "
            f"Continue? [y/N] {C.RESET}"
        )
        if confirm.lower() != "y":
            print_info("Reset cancelled.")
            return CommandResult.ok("Cancelled")

        for d in [BIZRA_LOGS, BIZRA_MODELS]:
            if d.exists():
                shutil.rmtree(d)
                d.mkdir()
        print_info("Cache cleared. Identity preserved.")
        return CommandResult.ok("Reset complete")
