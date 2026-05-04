# [ENFORCEMENT: WIRED] DEMA operator status and guarded pre-start wrapper.
"""DEMA operator CLI surfaces.

This module exposes measured DEMA status and the Relief Mode pre-start package.
The pre-start package is intentionally non-launching: it surfaces preflight,
confirmation, heartbeat, receipt-signing, and stop/recovery information without
starting a daemon.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.dema.node0_status import DEFAULT_DEMA_ROOT, read_node0_dema_status

from ..registry import CommandResult
from ..shared import C, print_info, print_status, print_warn

CONFIRM_RELIEF_START = "START MUMU-DEMA RELIEF"


class DemaCommand:
    """Operate DEMA visibility and Relief Mode pre-start checks.

    The only start-related behavior in this class is preflight packaging. It
    never calls the daemon tick or loop start primitives.
    """

    name = "dema"
    aliases: tuple[str, ...] = ()
    description = "Show local DEMA status and guarded Relief pre-start package"
    category = "node0"

    def execute(self, args: list[str]) -> CommandResult:
        """Execute a DEMA operator command.

        Args:
            args: Command arguments after `bizra dema`.

        Returns:
            CommandResult containing either measured status/pre-start data or a
            fail-closed error for unsupported options and missing confirmation.

        Raises:
            No exceptions are intentionally raised; local read failures are
            converted into CommandResult errors.
        """
        if not args or args[0] in {"-h", "--help", "help"}:
            self._print_usage()
            return CommandResult.ok(
                data={
                    "usage": (
                        "bizra dema status | "
                        "bizra dema start --mode relief [--confirm PHRASE]"
                    )
                }
            )

        subcommand = args[0]
        if subcommand == "status":
            return self._execute_status(args[1:])
        if subcommand == "start":
            return self._execute_start(args[1:])

        self._print_usage()
        return CommandResult.error(
            f"Unknown dema command: {subcommand}. "
            "Use 'bizra dema status' or 'bizra dema start --mode relief'."
        )

    def _execute_status(self, args: list[str]) -> CommandResult:
        """Render or emit measured DEMA status without mutation."""
        if args and args[0] in {"-h", "--help", "help"}:
            self._print_usage()
            return CommandResult.ok(data={"usage": "bizra dema status"})

        root = DEFAULT_DEMA_ROOT
        json_output = False
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--json":
                json_output = True
                i += 1
            elif arg == "--root":
                if i + 1 >= len(args):
                    return CommandResult.error("--root requires a path")
                root = Path(args[i + 1])
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

    def _execute_start(self, args: list[str]) -> CommandResult:
        """Build the Relief Mode pre-start package without launching."""
        if args and args[0] in {"-h", "--help", "help"}:
            self._print_usage()
            return CommandResult.ok(
                data={"usage": "bizra dema start --mode relief [--confirm PHRASE]"}
            )

        root = DEFAULT_DEMA_ROOT
        mode: str | None = None
        json_output = False
        confirmation: str | None = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--json":
                json_output = True
                i += 1
            elif arg == "--root":
                if i + 1 >= len(args):
                    return CommandResult.error("--root requires a path")
                root = Path(args[i + 1])
                i += 2
            elif arg == "--mode":
                if i + 1 >= len(args):
                    return CommandResult.error("--mode requires a value")
                mode = args[i + 1]
                i += 2
            elif arg == "--confirm":
                if i + 1 >= len(args):
                    return CommandResult.error("--confirm requires the exact phrase")
                confirmation = args[i + 1]
                i += 2
            else:
                return CommandResult.error(f"Unknown option for dema start: {arg}")

        if mode != "relief":
            return CommandResult.error(
                "Only 'bizra dema start --mode relief' is defined in v0.1."
            )

        try:
            report = read_node0_dema_status(root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return CommandResult.error(
                f"Failed to build Relief pre-start package: {exc}"
            )

        package = self._build_relief_prestart_package(
            root=root,
            report=report,
            confirmation=confirmation,
        )
        if json_output:
            print(json.dumps(package, indent=2, sort_keys=True))
        else:
            self._print_relief_prestart_package(package)

        if package["confirmation"]["accepted"]:
            return CommandResult(
                success=False,
                message="Relief Mode launch is not executed in this pre-start package.",
                exit_code=3,
                data=package,
            )
        return CommandResult(
            success=False,
            message="Explicit confirmation required before Relief Mode daemon launch.",
            exit_code=2,
            data=package,
        )

    def _build_relief_prestart_package(
        self,
        *,
        root: Path,
        report: dict[str, Any],
        confirmation: str | None,
    ) -> dict[str, Any]:
        """Build the non-launching Relief Mode pre-start package."""
        service = report["dema_service"]
        lm_studio = report["lm_studio"]
        receipt_signing = self._receipt_signing_status(root)
        preflight = [
            {
                "id": "node0_status_ready",
                "label": "Node0/DEMA readiness payload has no findings",
                "passed": bool(report["ready"]),
            },
            {
                "id": "lm_studio_connected",
                "label": "LM Studio local API is connected",
                "passed": bool(lm_studio["connected"]),
            },
            {
                "id": "model_loaded",
                "label": "At least one local model is loaded",
                "passed": int(lm_studio.get("loaded_count", 0)) > 0,
            },
            {
                "id": "profile_present",
                "label": "DEMA local profile is present",
                "passed": bool(service["profile_present"]),
            },
            {
                "id": "daemon_stopped",
                "label": "DEMA daemon is currently stopped",
                "passed": not bool(service["running"]),
            },
            {
                "id": "receipt_signing_visible",
                "label": "Receipt signing state is surfaced before start",
                "passed": True,
                "status": receipt_signing["status"],
            },
        ]
        confirmation_accepted = confirmation == CONFIRM_RELIEF_START
        return {
            "kind": "dema_relief_prestart_package",
            "schema_version": "0.1.0",
            "truth_label": "MEASURED_PRESTART_NO_LAUNCH",
            "mode": "relief",
            "root": str(root),
            "launch_executed": False,
            "ready_for_final_confirmation": all(item["passed"] for item in preflight),
            "preflight": preflight,
            "findings": list(report.get("findings", [])),
            "receipt_signing": receipt_signing,
            "confirmation": {
                "required": True,
                "phrase": CONFIRM_RELIEF_START,
                "accepted": confirmation_accepted,
                "provided": confirmation is not None,
                "boundary": (
                    "This pre-start package never launches the daemon. "
                    "A future launch slice must re-run preflight immediately "
                    "before honoring this confirmation."
                ),
            },
            "heartbeat_verification_plan": [
                "Run one bounded Relief heartbeat after launch.",
                "Re-read `bizra dema status --json` and require running=true or a fresh tick receipt.",
                "Report PID, lock path, last tick timestamp, and receipt id when available.",
            ],
            "stop_recovery": {
                "stop_command": "bizra dema stop",
                "current_safe_fallback": (
                    "No loop supervisor is started by this package; if a future "
                    "daemon launch fails, use the supervisor-specific stop path "
                    "and inspect the visible PID/lock/log paths."
                ),
                "pid": service["pid"],
                "lock_path": service["lock_path"],
                "log_path": str(root / "logs"),
            },
            "prohibited_actions": [
                "no_node1",
                "no_third_fact_publication",
                "no_bulk_memory_ingestion",
                "no_broad_file_actions",
                "no_hidden_daemon_start",
            ],
            "measured_status": report,
        }

    def _receipt_signing_status(self, root: Path) -> dict[str, Any]:
        """Return visible receipt-signing readiness for the DEMA state root."""
        key_present = bool(os.environ.get("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "").strip())
        key_registry = root.parent / "key_registry.json"
        registry_present = key_registry.exists()
        if key_present:
            return {
                "status": "SIGNING_CONFIG_VISIBLE",
                "private_key_env_present": True,
                "key_registry_present": registry_present,
                "warning": None,
            }
        return {
            "status": "LOCAL_UNSIGNED_DEV",
            "private_key_env_present": False,
            "key_registry_present": registry_present,
            "warning": (
                "BIZRA_RECEIPT_PRIVATE_KEY_HEX/key registry not found; a future "
                "Relief start must surface unsigned local-dev receipt status."
                if not registry_present
                else "key_registry.json is present, but BIZRA_RECEIPT_PRIVATE_KEY_HEX "
                "is missing; Relief receipts remain LOCAL_UNSIGNED_DEV until signer "
                "key material is configured."
            ),
        }

    def _print_relief_prestart_package(self, package: dict[str, Any]) -> None:
        """Print the Relief Mode pre-start package for operators."""
        print(f"\n{C.BOLD}{C.WHITE}Mumu-DEMA Relief Mode Pre-Start{C.RESET}")
        print(f"{C.GRAY}{'-' * 60}{C.RESET}\n")
        print_status(
            "Launch executed",
            "no - pre-start package only",
            not package["launch_executed"],
        )
        print_status(
            "Preflight",
            "ready" if package["ready_for_final_confirmation"] else "blocked",
            package["ready_for_final_confirmation"],
        )
        print_status("Receipt signing", package["receipt_signing"]["status"], True)
        if package["receipt_signing"]["warning"]:
            print_warn(package["receipt_signing"]["warning"])

        print(f"\n{C.BOLD}Preflight checklist{C.RESET}")
        for item in package["preflight"]:
            print_status(item["id"], item["label"], item["passed"])

        print(f"\n{C.BOLD}Confirmation boundary{C.RESET}")
        print_info(f"Required phrase: {package['confirmation']['phrase']}")
        print_status(
            "Confirmation provided",
            "accepted" if package["confirmation"]["accepted"] else "missing/invalid",
            package["confirmation"]["accepted"],
        )
        print_warn(package["confirmation"]["boundary"])

        print(f"\n{C.BOLD}Heartbeat verification plan{C.RESET}")
        for step in package["heartbeat_verification_plan"]:
            print_info(step)

        print(f"\n{C.BOLD}Stop / recovery visibility{C.RESET}")
        print_info(f"Future stop command: {package['stop_recovery']['stop_command']}")
        print_info(f"PID: {package['stop_recovery']['pid']}")
        print_info(f"Lock path: {package['stop_recovery']['lock_path']}")
        print_info(f"Log path: {package['stop_recovery']['log_path']}")

        if package["findings"]:
            print(f"\n{C.GOLD}Findings:{C.RESET}")
            for finding in package["findings"]:
                print_warn(finding)

        print()

    def _print_usage(self) -> None:
        """Print DEMA command usage."""
        print(f"\n{C.BOLD}{C.WHITE}BIZRA DEMA{C.RESET}")
        print(f"{C.GRAY}{'-' * 50}{C.RESET}")
        print("  bizra dema status [--json] [--root PATH]")
        print("  bizra dema start --mode relief [--json] [--root PATH]")
        print("  bizra dema start --mode relief --confirm 'START MUMU-DEMA RELIEF'")
        print()
        print_info("Status is read-only. Relief start is preflight-only in v0.1.")

    def _print_status_report(self, report: dict[str, Any]) -> None:
        """Print the measured DEMA status report."""
        service = report["dema_service"]
        doctor = report["dema_doctor"]
        current_gap = report["dema_current_gap"]
        lm_studio = report["lm_studio"]
        node_console = report.get("dema_node_console", {})

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

        if node_console:
            print(f"\n{C.BOLD}Node Console Dependencies{C.RESET}")
            print_status(
                "Activation gate",
                node_console["activation_gate"],
                node_console["ready"],
            )
            for dependency in node_console["dependencies"]:
                ok = dependency["status"] == "READY"
                status = f"{dependency['status']} — {dependency['observed']}"
                print_status(dependency["label"], status, ok)

        if report["findings"]:
            print(f"\n{C.GOLD}Findings:{C.RESET}")
            for finding in report["findings"]:
                print_warn(finding)

        print(f"\n{C.GRAY}Root: {report['root']}{C.RESET}")
        print()
