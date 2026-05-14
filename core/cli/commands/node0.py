# [ENFORCEMENT: WIRED] Read-only Node0 command-center wrapper.
"""Read-only Node0 command-center CLI surface.

This module renders measured Node0/DEMA readiness from existing status probes.
It does not start or stop daemons, dispatch missions, load models, or ingest
memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.dema.node0_status import DEFAULT_DEMA_ROOT, read_node0_dema_status

from ..registry import CommandResult
from ..shared import C, print_info, print_status, print_warn


class Node0Command:
    """Render the read-only Node0 command center.

    The command defaults to status rendering and rejects unsupported verbs with
    a read-only message instead of exposing mutating actions.
    """

    name = "node0"
    aliases: tuple[str, ...] = ()
    description = "Open the read-only Node0 command center"
    category = "node0"

    def execute(self, args: list[str]) -> CommandResult:
        """Execute the read-only Node0 command.

        Args:
            args: Command arguments after `bizra node0`.

        Returns:
            CommandResult with the measured Node0/DEMA status payload on
            success, or an error result for unsupported verbs/options.

        Raises:
            No exceptions are intentionally raised; local read failures are
            converted into CommandResult errors.
        """
        if args and args[0] in {"-h", "--help", "help"}:
            self._print_usage()
            return CommandResult.ok(data={"usage": "bizra node0 [status] [--json]"})

        subcommand = "status"
        rest = args
        if rest and not rest[0].startswith("-"):
            subcommand = rest[0]
            rest = rest[1:]

        if subcommand != "status":
            self._print_usage()
            return CommandResult.error(
                f"Unknown node0 command: {subcommand}. "
                "Node0 v0.1 is read-only; use 'bizra node0' or 'bizra node0 status'."
            )

        root = DEFAULT_DEMA_ROOT
        json_output = False
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
                return CommandResult.error(f"Unknown option for node0 status: {arg}")

        try:
            report = read_node0_dema_status(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return CommandResult.error(f"Failed to read Node0 status: {exc}")

        if json_output:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            self._print_command_center(report)
        return CommandResult.ok(data=report)

    def _print_usage(self) -> None:
        """Print read-only Node0 usage.

        Args:
            None.

        Returns:
            None.
        """
        print(f"\n{C.BOLD}{C.WHITE}BIZRA Node0{C.RESET}")
        print(f"{C.GRAY}{'-' * 60}{C.RESET}")
        print("  bizra node0 [--json] [--root PATH]")
        print("  bizra node0 status [--json] [--root PATH]")
        print()
        print_info(
            "Read-only command center. No daemon, mission, or memory action runs."
        )

    def _print_command_center(self, report: dict[str, Any]) -> None:
        """Render the measured Node0/DEMA command-center screen.

        Args:
            report: Status payload returned by read_node0_dema_status().

        Returns:
            None.
        """
        service = report["dema_service"]
        doctor = report["dema_doctor"]
        current_gap = report["dema_current_gap"]
        lm_studio = report["lm_studio"]

        print(f"\n{C.BOLD}{C.WHITE}BIZRA Node0 Command Center{C.RESET}")
        print(f"{C.GRAY}{'-' * 60}{C.RESET}\n")

        print_status(
            "Readiness", "READY" if report["ready"] else "BLOCKED", report["ready"]
        )
        print_status("Truth label", report.get("truth_label", "MEASURED"), True)
        print_status("Mode", "Mumu-DEMA local relief", True)

        node_console = report.get("dema_node_console", {})
        if node_console:
            print_status(
                "Node Console",
                node_console["activation_gate"],
                node_console["ready"],
            )

        print(f"\n{C.BOLD}Runtime{C.RESET}")
        print_status("DEMA daemon", service["status"], service["running"])
        print_status(
            "DEMA doctor",
            "healthy" if doctor["healthy"] else "blocked",
            doctor["healthy"],
        )
        print_status(
            "Profile",
            "present" if service["profile_present"] else "missing",
            service["profile_present"],
        )
        print_status(
            "Mission gap",
            "actionable" if current_gap["actionable"] else "not actionable",
            current_gap["actionable"],
        )

        print(f"\n{C.BOLD}Model backend{C.RESET}")
        lm_message = (
            f"{lm_studio['model_count']} model(s), "
            f"{lm_studio['loaded_count']} loaded"
            if lm_studio["connected"]
            else "not reachable"
        )
        print_status("LM Studio", lm_message, lm_studio["connected"])
        print_status(
            "LM auth",
            "required" if lm_studio.get("auth_required") else "not required by probe",
            not lm_studio.get("auth_required") or lm_studio.get("token_present"),
        )
        if lm_studio["loaded_model_ids"]:
            print_info("Loaded model(s): " + ", ".join(lm_studio["loaded_model_ids"]))
        elif lm_studio["connected"] and not lm_studio["load_state_known"]:
            print_warn("LM Studio responded through /v1/models; load state is unknown")

        if node_console:
            print(f"\n{C.BOLD}Node Console Dependencies{C.RESET}")
            for dependency in node_console["dependencies"]:
                ok = dependency["status"] == "READY"
                status = f"{dependency['status']} — {dependency['observed']}"
                print_status(dependency["label"], status, ok)

        print(f"\n{C.BOLD}Guardrails{C.RESET}")
        print_info("This command is read-only.")
        print_info(
            "Daemon start/stop and mission dispatch require explicit confirmation."
        )
        print_info("Memory ingestion is outside this v0.1 command-center slice.")

        if report["findings"]:
            print(f"\n{C.GOLD}Blockers / findings:{C.RESET}")
            for finding in report["findings"]:
                print_warn(finding)
        else:
            print(
                f"\n{C.GREEN}No readiness findings reported by measured probes.{C.RESET}"
            )

        print(f"\n{C.BOLD}Safe next commands{C.RESET}")
        print_info("bizra doctor --json")
        print_info("bizra dema status --json")
        print(f"\n{C.GRAY}Root: {report['root']}{C.RESET}")
        print()
