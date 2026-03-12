"""bizra status — Show node health and sovereignty score."""

from __future__ import annotations

import json
from typing import List

from ..registry import CommandResult
from ..shared import (
    API_PORT,
    BIZRA_IDENTITY,
    BIZRA_LOGS,
    C,
    api_health,
    print_info,
    print_status,
)


class StatusCommand:
    name = "status"
    aliases = ("health", "s")
    description = "Show node health and sovereignty score"
    category = "system"

    def execute(self, args: List[str]) -> CommandResult:
        health = api_health()

        print(f"\n{C.BOLD}{C.WHITE}BIZRA Node Status{C.RESET}")
        print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

        if not health:
            print_status("Runtime", "OFFLINE", False)
            print_info(f"Start with: {C.WHITE}bizra start{C.RESET}")
            print()
            return CommandResult.ok(data={"online": False})

        print_status("Runtime", "ONLINE", True)
        print_status("Status", health.get("status", "unknown"), True)

        # Node identity
        if BIZRA_IDENTITY.exists():
            try:
                ident = json.loads(BIZRA_IDENTITY.read_text())
                pub_key = ident.get("public_key", "unknown")
                if len(pub_key) > 20:
                    pub_key = pub_key[:10] + "..." + pub_key[-6:]
                print_status("Node ID", pub_key, True)
            except (json.JSONDecodeError, ValueError, OSError):
                pass

        # Sovereignty score
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/seed/potential")
            with urllib.request.urlopen(req, timeout=3) as resp:
                pot = json.loads(resp.read())
                score = pot.get("sovereignty_score", 0)
                tier = pot.get("tier", "UNKNOWN")
                print_status("Sovereignty", f"{score:.2f}", score >= 0.5)
                print_status("Tier", tier, True)
        except (OSError, json.JSONDecodeError, ValueError, ImportError):
            pass

        # Token balance
        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/token/balance")
            with urllib.request.urlopen(req, timeout=3) as resp:
                bal = json.loads(resp.read())
                balances = bal.get("balances", {})
                seed_bal = balances.get("SEED", {}).get("balance", 0)
                print_status("SEED Balance", f"{seed_bal:.2f}", True)
        except (OSError, json.JSONDecodeError, ValueError, ImportError):
            pass

        print(f"\n{C.GRAY}{'─' * 50}{C.RESET}")
        print(f"  {C.TEAL}♥ Constitutional heartbeat: every 60s{C.RESET}")
        print(f"  {C.GRAY}Logs: {BIZRA_LOGS}{C.RESET}")
        print()

        return CommandResult.ok(data={"online": True, "status": health.get("status")})
