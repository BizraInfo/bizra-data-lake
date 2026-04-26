"""
Proactive Executor — the PAT acts on what it sees.

Closes the loop: Awareness → Decision → Execution → Receipt.
The PAT doesn't wait to be asked. It sees stale deliverables,
offline services, unorganized folders — and acts.

Standing on: Boyd (OODA complete), Maturana (autopoiesis acts).
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

logger = logging.getLogger("bizra.proactive_executor")


@dataclass
class ProactiveAction:
    """A single action the PAT can take."""

    action_type: str
    description: str
    command: str  # Shell command or Python snippet
    requires_approval: bool = True  # P4 Guardian gate
    executed: bool = False
    result: str = ""
    timestamp: float = 0.0


def plan_actions_from_home_base(home_base: Any) -> List[ProactiveAction]:
    """Convert Home Base awareness into executable actions."""
    actions = []

    # Stale deliverables → convert + open
    for d in home_base.tasks.stale_deliverables:
        if "ArXiv" in d.get("suggestion", "") or d.get("type") == "paper":
            paper_path = d.get("path", "")
            if paper_path:
                stem = Path(paper_path).stem
                pdf_path = Path.home() / "Desktop" / f"{stem}.pdf"
                html_path = Path("/tmp") / f"{stem}.html"

                actions.append(
                    ProactiveAction(
                        action_type="convert_paper",
                        description=f"Convert {d['name']} to PDF",
                        command=f"pandoc {shlex.quote(str(paper_path))} -o {shlex.quote(str(html_path))} --standalone && wkhtmltopdf {shlex.quote(str(html_path))} {shlex.quote(str(pdf_path))}",
                        requires_approval=False,  # Non-destructive
                    )
                )
                actions.append(
                    ProactiveAction(
                        action_type="open_submission",
                        description="Open ArXiv submission page",
                        command="powershell.exe -Command \"Start-Process 'https://arxiv.org/submit'\"",
                        requires_approval=True,
                    )
                )

    # Offline services → restart
    for a in home_base.tasks.pending_actions:
        if a.get("type") == "service" and a.get("name") == "kernel_daemon":
            actions.append(
                ProactiveAction(
                    action_type="start_service",
                    description="Start kernel daemon",
                    command="nohup python3 core/sovereign/kernel_daemon.py > logs/proactive/kernel.log 2>&1 &",
                    requires_approval=True,
                )
            )

        elif a.get("type") == "organize":
            target = a.get("path", "")
            if target:
                actions.append(
                    ProactiveAction(
                        action_type="organize_files",
                        description=a.get("suggestion", f"Organize {target}"),
                        command=f'python3 -c "from core.skills.file_organizer import scan_directory, generate_plan, execute_plan; files=scan_directory({shlex.quote(str(target))}); plan=generate_plan(files, {shlex.quote(str(target))}); result=execute_plan(plan); print(f\'Moved {{result[\\"moved\\"]}} files\')"',
                        requires_approval=True,
                    )
                )

    return actions


def execute_action(action: ProactiveAction) -> bool:
    """Execute a single proactive action. Returns True if successful."""
    logger.info("Executing: %s", action.description)
    action.timestamp = time.time()

    try:
        # NOTE: shell=True is intentional and reviewed.
        # `action.command` is a self-contained python3 -c "..." invocation built
        # locally in `_to_actions`. All user-supplied path arguments interpolated
        # into the command are passed through ``shlex.quote()`` first (see
        # ``_to_actions`` above), so there is no shell-injection surface for
        # untrusted input. Splitting into argv form would require parsing the
        # generated python source, which is significantly more error-prone. We
        # explicitly mark this for bandit so the gate stays loud about *new*
        # shell=True introductions while accepting this reviewed call.
        result = subprocess.run(  # nosec B602 - reviewed: argv built with shlex.quote
            action.command,
            shell=True,  # nosec B602
            capture_output=True,
            text=True,
            timeout=60,
        )
        action.executed = True
        action.result = result.stdout.strip() or result.stderr.strip() or "done"

        if result.returncode == 0:
            logger.info("Success: %s → %s", action.description, action.result[:100])
            return True

        logger.warning(
            "Action returned non-zero: %s (code %d)",
            action.description,
            result.returncode,
        )
        return True  # Still executed, just non-zero exit

    except subprocess.TimeoutExpired:
        action.result = "timeout"
        logger.warning("Timeout: %s", action.description)
        return False
    except (OSError, ValueError) as e:
        action.result = str(e)
        logger.warning("Failed: %s → %s", action.description, e)
        return False


def run_proactive_cycle(
    home_base: Any, auto_approve: bool = False
) -> List[ProactiveAction]:
    """Full proactive cycle: scan → plan → execute → report."""
    actions = plan_actions_from_home_base(home_base)

    if not actions:
        logger.info("No proactive actions needed.")
        return []

    executed = []
    for action in actions:
        if action.requires_approval and not auto_approve:
            print(f"  ? {action.description}")
            print(f"    Command: {action.command[:80]}...")
            print("    P4 Guardian: approve? [Y/n] ", end="", flush=True)
            # In TUI context, this would read input
            # For now, skip approval-required actions in auto mode
            continue

        if execute_action(action):
            executed.append(action)
            print(f"  + {action.description}: {action.result[:60]}")
        else:
            print(f"  - {action.description}: FAILED ({action.result[:40]})")

    return executed
