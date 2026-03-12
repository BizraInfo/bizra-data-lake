"""bizra identity — Show node identity (Ed25519 public key)."""

from __future__ import annotations

import json
from typing import List

from ..registry import CommandResult
from ..shared import BIZRA_IDENTITY, C, print_info, print_status


class IdentityCommand:
    name = "identity"
    aliases = ("id", "whoami")
    description = "Show node identity (Ed25519 public key)"
    category = "system"

    def execute(self, args: List[str]) -> CommandResult:
        print(f"\n{C.BOLD}{C.WHITE}BIZRA Node Identity{C.RESET}")
        print(f"{C.GRAY}{'─' * 50}{C.RESET}\n")

        if not BIZRA_IDENTITY.exists():
            print_info("No identity created yet. Run 'bizra start' to generate.")
            print()
            return CommandResult.ok(data={"exists": False})

        try:
            ident = json.loads(BIZRA_IDENTITY.read_text())
            pub_key = ident.get("public_key", "unknown")
            node_id = ident.get("node_id", "unknown")
            created = ident.get("created_at", "unknown")

            print_status("Public Key", pub_key, True)
            print_status("Node ID", node_id, True)
            print_status("Created", created, True)
            print(f"\n  {C.GRAY}Identity is Ed25519 + BLAKE2b (L1 protocol layer){C.RESET}")
            print()
            return CommandResult.ok(data={"public_key": pub_key, "node_id": node_id})

        except (json.JSONDecodeError, ValueError, OSError) as exc:
            print(f"  {C.RED}✗ Could not read identity: {exc}{C.RESET}\n")
            return CommandResult.error(str(exc))
