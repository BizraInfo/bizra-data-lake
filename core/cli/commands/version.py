"""bizra version — Show version info."""

from __future__ import annotations

from typing import List

from ..registry import CommandResult
from ..shared import CODENAME, VERSION, C


class VersionCommand:
    name = "version"
    aliases = ("v", "--version", "-v")
    description = "Show version info"
    category = "system"

    def execute(self, args: List[str]) -> CommandResult:
        print(f"\n  {C.BOLD}{C.WHITE}BIZRA{C.RESET} {C.TEAL}{VERSION}{C.RESET}")
        print(f"  {C.GRAY}Codename: {CODENAME}{C.RESET}")
        print(f"  {C.GRAY}The Living Organism — 12B Constitutional MOE{C.RESET}")
        print()
        return CommandResult.ok(data={"version": VERSION, "codename": CODENAME})
