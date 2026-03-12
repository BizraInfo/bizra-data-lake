"""bizra (no args) — Launch the full terminal experience."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List

from ..registry import CommandResult
from ..shared import (
    API_PORT,
    C,
    DEMA_GREETING,
    WEB_PORT,
    find_bizra_root,
    find_frontend_root,
    port_in_use,
    print_info,
)


# ASCII banner for launch
_LAUNCH_BANNER = f"""
{C.TEAL}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ██╗███████╗██████╗  █████╗                         ║
║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                        ║
║   ██████╔╝██║  ███╔╝ ██████╔╝███████║                        ║
║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                        ║
║   ██████╔╝██║███████╗██║  ██║██║  ██║                        ║
║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                        ║
║                                                              ║
║          Sovereign Autonomous Operating System               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
"""


class LaunchCommand:
    name = "launch"
    aliases = ()
    description = "Launch the full terminal experience"
    category = "lifecycle"

    def execute(self, args: List[str]) -> CommandResult:
        print(_LAUNCH_BANNER)
        print(f"  {C.TEAL}{DEMA_GREETING}{C.RESET}\n")

        root = find_bizra_root()
        frontend = find_frontend_root()

        # Start API if not running
        if not port_in_use(API_PORT) and root:
            print_info("Starting sovereign runtime...")
            from .lifecycle import StartCommand
            StartCommand().execute([])

        # Start frontend if available
        if frontend and not port_in_use(WEB_PORT):
            print_info("Starting terminal UI...")
            env = os.environ.copy()
            env["BROWSER"] = "none"
            try:
                subprocess.Popen(
                    ["npm", "run", "dev"],
                    cwd=str(frontend),
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except FileNotFoundError:
                pass

        # Open browser
        if port_in_use(WEB_PORT):
            url = f"http://localhost:{WEB_PORT}"
            print_info(f"Terminal: {C.WHITE}{url}{C.RESET}")
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:
                pass

        print(f"\n  {C.GRAY}Use 'bizra status' to check health{C.RESET}")
        print(f"  {C.GRAY}Use 'bizra stop' to shut down{C.RESET}")
        print()

        return CommandResult.ok("Launched")
