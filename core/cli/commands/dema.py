"""bizra dema - read-only DEMA operator status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from core.dema.node0_status import DEFAULT_DEMA_ROOT, read_node0_dema_status

from ..registry import CommandResult
from ..shared import C, print_info, print_status, print_warn


class DemaCommand:
    name = "dema"
    aliases: tuple[str, ...] = ()
    description = "Show local DEMA status without starting the daemon"
    category = "node0"

    def execute(self, args: List[str]) -> CommandResult:
        if not args or args[0] in {"-h", "--help", "help"}:
            self._print_usage()
            return CommandResult.ok(data={"usage": "bizra dema status"})

        subcommand = args[0]
        if subcommand != "status":
            self._print_usage()
            return CommandResult.error(
                f"Unknown dema command: {subcommand}. Use 'bizra dema status'."
            )

        root = DEFAULT_DEMA_ROOT
        json_output = False
        rest = args[1:]
        i = 0
        while i < len(rest):
            arg = rest[i]
            if arg == "--json":
                json_output = True
                i += 1
            elif arg == "--root":
                if i + 1 >= len(rest):
                    return CommandResult.error("--root requires a path")
                root = Path(rest[i + 1])
                i += 2
            else:
                return CommandResult.error(f"Unknown option for dema status: {arg}")

        try:
            report = read_node0_dema_status(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return CommandResult.error(f"Failed to read DEMA status: {exc}")
        if json_output:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            self._print_status_report(report)
        return CommandResult.ok(data=report)

    def _print_usage(self) -> None:
        print(f"\n{C.BOLD}{C.WHITE}BIZRA DEMA{C.RESET}")
        print(f"{C.GRAY}{'-' * 50}{C.RESET}")
        print("  bizra dema status [--json] [--root PATH]")
        print()
        print_info("Read-only status only. This command does not start the daemon.")

    def _print_status_report(self, report: dict) -> None:
        service = report["dema_service"]
        doctor = report["dema_doctor"]
        current_gap = report["dema_current_gap"]
        lm_studio = report["lm_studio"]

        print(f"\n{C.BOLD}{C.WHITE}BIZRA DEMA Status{C.RESET}")
        print(f"{C.GRAY}{'-' * 50}{C.RESET}\n")

        print_status(
            "Readiness", "READY" if report["ready"] else "BLOCKED", report["ready"]
        )
        print_status("Daemon", service["status"], service["running"])
        print_status(
            "Profile",
            "present" if service["profile_present"] else "missing",
            service["profile_present"],
        )
        print_status(
            "Service doctor",
            "healthy" if doctor["healthy"] else "blocked",
            doctor["healthy"],
        )
        print_status("Mission truth", service["mission_truth_label"], True)
        print_status(
            "Mission actionable",
            "yes" if current_gap["actionable"] else "no",
            current_gap["actionable"],
        )

        last_tick = service["last_tick"]
        if last_tick:
            print_status("Last wake tick", last_tick["timestamp"], True)
            if last_tick.get("receipt_id"):
                print_status("Last receipt", last_tick["receipt_id"], True)
        else:
            print_status("Last wake tick", "none today", False)

        lm_message = (
            f"{lm_studio['model_count']} model(s), "
            f"{lm_studio['loaded_count']} loaded"
            if lm_studio["connected"]
            else "not reachable"
        )
        print_status("LM Studio", lm_message, lm_studio["connected"])
        if lm_studio["loaded_model_ids"]:
            print_info("Loaded model(s): " + ", ".join(lm_studio["loaded_model_ids"]))
        elif lm_studio["connected"] and not lm_studio["load_state_known"]:
            print_warn("LM Studio responded through /v1/models; load state is unknown")

        if report["findings"]:
            print(f"\n{C.GOLD}Findings:{C.RESET}")
            for finding in report["findings"]:
                print_warn(finding)

        print(f"\n{C.GRAY}Root: {report['root']}{C.RESET}")
        print()
