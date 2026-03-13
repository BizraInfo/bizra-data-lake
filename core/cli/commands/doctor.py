"""bizra doctor — Diagnose the BIZRA installation."""

from __future__ import annotations

import shutil
from typing import List

from ..registry import CommandResult
from ..shared import (
    API_PORT,
    BIZRA_HOME,
    BIZRA_IDENTITY,
    OLLAMA_PORT,
    WEB_PORT,
    C,
    api_health,
    find_bizra_root,
    find_frontend_root,
    find_python,
    port_in_use,
    print_status,
    print_warn,
)


class DoctorCommand:
    name = "doctor"
    aliases = ("doc", "check", "diagnose")
    description = "Diagnose the BIZRA installation"
    category = "system"

    def execute(self, args: List[str]) -> CommandResult:
        print(f"\n{C.BOLD}{C.WHITE}BIZRA Doctor{C.RESET}")
        print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

        issues = 0

        py = find_python()
        print_status("Python", py, True)

        root = find_bizra_root()
        if root:
            print_status("BIZRA source", str(root), True)
        else:
            print_status("BIZRA source", "NOT FOUND", False)
            print_warn("Set BIZRA_ROOT env var to your BIZRA-DATA-LAKE directory")
            issues += 1

        frontend = find_frontend_root()
        if frontend:
            print_status("Frontend", str(frontend), True)
        else:
            print_status("Frontend", "NOT FOUND", False)
            print_warn(
                "Set BIZRA_FRONTEND env var to your award-winner-design directory"
            )
            issues += 1

        ollama_path = shutil.which("ollama")
        if ollama_path:
            print_status("Ollama", ollama_path, True)
        else:
            print_status("Ollama", "NOT FOUND", False)
            print_warn("Install Ollama: https://ollama.ai")
            issues += 1

        if port_in_use(OLLAMA_PORT):
            print_status("Ollama server", f"Running (:{OLLAMA_PORT})", True)
        else:
            print_status("Ollama server", "Not running", False)
            issues += 1

        health = api_health()
        if health:
            print_status("Sovereign API", f"Healthy (:{API_PORT})", True)
        else:
            print_status("Sovereign API", f"Not running (:{API_PORT})", False)
            issues += 1

        if port_in_use(WEB_PORT):
            print_status("Terminal UI", f"Running (:{WEB_PORT})", True)
        else:
            print_status("Terminal UI", f"Not running (:{WEB_PORT})", False)
            issues += 1

        node_path = shutil.which("node")
        if node_path:
            print_status("Node.js", node_path, True)
        else:
            print_status("Node.js", "NOT FOUND", False)
            issues += 1

        print_status("BIZRA home", str(BIZRA_HOME), BIZRA_HOME.exists())

        if BIZRA_IDENTITY.exists():
            print_status("Node identity", "Exists", True)
        else:
            print_status(
                "Node identity", "Not created (will generate on first run)", False
            )

        print(f"\n{C.GRAY}{'─' * 50}{C.RESET}")
        if issues == 0:
            print(f"  {C.GREEN}{C.BOLD}All systems operational.{C.RESET}")
        else:
            print(f"  {C.GOLD}{issues} issue(s) found.{C.RESET}")
        print()

        return CommandResult.ok(data={"issues": issues})
