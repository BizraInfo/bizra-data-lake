"""bizra mission — Submit a mission from CLI."""

from __future__ import annotations

import json
import sys
from typing import List

from ..registry import CommandResult
from ..shared import API_PORT, C, api_health, print_error, print_info


class MissionCommand:
    name = "mission"
    aliases = ("m", "do")
    description = "Submit a mission directly from CLI"
    category = "operations"

    def execute(self, args: List[str]) -> CommandResult:
        if not args:
            print_error('Usage: bizra mission "your mission text"')
            return CommandResult.error("No mission text provided")

        text = " ".join(args)
        health = api_health()
        if not health:
            print_error("Sovereign runtime is not running.")
            print_info(f"Start with: {C.WHITE}bizra start{C.RESET}")
            return CommandResult.error("Runtime offline")

        print(f"\n{C.BOLD}{C.WHITE}Mission Submitted{C.RESET}")
        print(f"{C.GRAY}{'─' * 50}{C.RESET}")
        print(f"  {C.TEAL}» {text}{C.RESET}\n")

        try:
            import urllib.request
            data = json.dumps({"description": text}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{API_PORT}/v1/plan",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())

            status = result.get("status", "unknown")
            receipt = result.get("receipt_hash", result.get("receipt", {}).get("hash", ""))

            print(f"  {C.GREEN}✓ Status: {status}{C.RESET}")
            if receipt:
                short = receipt[:16] + "..." if len(receipt) > 16 else receipt
                print(f"  {C.GRAY}Receipt: {short}{C.RESET}")

            ihsan = result.get("ihsan_score", result.get("receipt", {}).get("ihsan_score"))
            if ihsan is not None:
                color = C.GREEN if ihsan >= 0.95 else C.GOLD if ihsan >= 0.85 else C.RED
                print(f"  {color}Ihsān: {ihsan:.4f}{C.RESET}")

            print()
            return CommandResult.ok(data={"status": status, "receipt": receipt})

        except Exception as exc:
            print(f"\n  {C.RED}✗ Mission failed: {exc}{C.RESET}\n")
            return CommandResult.error(str(exc))
