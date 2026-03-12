"""bizra wallet / briefing — Token balance and morning briefing."""

from __future__ import annotations

import json
from typing import List

from ..registry import CommandResult
from ..shared import API_PORT, C, api_health, print_error, print_info, print_status


class WalletCommand:
    name = "wallet"
    aliases = ("w", "balance")
    description = "Show SEED/BLOOM balance"
    category = "operations"

    def execute(self, args: List[str]) -> CommandResult:
        health = api_health()
        if not health:
            print_error("Sovereign runtime is not running.")
            print_info(f"Start with: {C.WHITE}bizra start{C.RESET}")
            return CommandResult.error("Runtime offline")

        print(f"\n{C.BOLD}{C.WHITE}BIZRA Wallet{C.RESET}")
        print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/token/balance")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            balances = data.get("balances", {})
            seed = balances.get("SEED", {})
            bloom = balances.get("BLOOM", {})

            print_status("SEED", f"{seed.get('balance', 0):.2f}", True)
            print_status("BLOOM", f"{bloom.get('balance', 0):.2f}", True)
            print(f"\n  {C.GRAY}SEED: liquid utility token (100% yours){C.RESET}")
            print(f"  {C.GRAY}BLOOM: soulbound reputation (non-transferable){C.RESET}")
            print()
            return CommandResult.ok(data={"seed": seed.get("balance", 0), "bloom": bloom.get("balance", 0)})

        except Exception as exc:
            print(f"  {C.RED}✗ Could not fetch balance: {exc}{C.RESET}\n")
            return CommandResult.error(str(exc))


class BriefingCommand:
    name = "briefing"
    aliases = ("brief", "b", "morning")
    description = "Show morning briefing from DEMA"
    category = "operations"

    def execute(self, args: List[str]) -> CommandResult:
        health = api_health()
        if not health:
            print_error("Sovereign runtime is not running.")
            print_info(f"Start with: {C.WHITE}bizra start{C.RESET}")
            return CommandResult.error("Runtime offline")

        print(f"\n{C.BOLD}{C.WHITE}DEMA Morning Briefing{C.RESET}")
        print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

        try:
            import urllib.request
            req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}/v1/briefing")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())

            briefing_text = data.get("briefing", data.get("message", "No briefing available."))
            print(f"  {C.TEAL}{briefing_text}{C.RESET}")
            print()
            return CommandResult.ok()

        except Exception as exc:
            print(f"  {C.GRAY}DEMA is not available right now: {exc}{C.RESET}\n")
            return CommandResult.error(str(exc))
