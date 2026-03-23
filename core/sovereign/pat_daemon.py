"""
PAT Daemon — the always-on sovereign awareness loop.

The PAT doesn't sleep. It watches the home base continuously,
detects changes, and queues proactive actions. When the user
opens the TUI, the suggestions are already waiting.

Runs as a background process (systemd or nohup).
Cycle: scan → detect changes → plan actions → queue → wait → repeat.

Standing on: Maturana (autopoiesis never stops), Deming (continuous improvement).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sovereign.home_base import (
    detect_changes,
    full_scan,
    load_home_base,
    save_home_base,
)
from core.sovereign.proactive_executor import plan_actions_from_home_base

logger = logging.getLogger("bizra.pat_daemon")

CYCLE_SECONDS = int(os.environ.get("BIZRA_PAT_CYCLE_SECONDS", "300"))  # 5 min default
QUEUE_PATH = Path.home() / ".bizra" / "proactive_queue.jsonl"
PID_PATH = Path.home() / ".bizra" / "pat_daemon.pid"

_running = True


def _signal_handler(signum: int, frame: object) -> None:
    global _running
    logger.info("PAT daemon stopping (signal %d)", signum)
    _running = False


def queue_action(action_dict: dict) -> None:
    """Append a proactive action to the queue for TUI pickup."""
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_PATH, "a") as f:
        f.write(json.dumps(action_dict) + "\n")


def read_queue() -> list[dict]:
    """Read and clear the proactive action queue."""
    if not QUEUE_PATH.exists():
        return []
    with open(QUEUE_PATH) as f:
        items = [json.loads(line) for line in f if line.strip()]
    QUEUE_PATH.write_text("")  # Clear after read
    return items


def run_daemon() -> None:
    """Main daemon loop — scan, detect, queue, repeat."""
    global _running

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Write PID
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))
    logger.info("PAT daemon started (PID %d, cycle %ds)", os.getpid(), CYCLE_SECONDS)

    cycle = 0
    while _running:
        cycle += 1
        try:
            # Load previous state
            previous = load_home_base()

            # Full scan
            current = full_scan()
            save_home_base(current)

            # Detect changes
            changes = []
            if previous:
                changes = detect_changes(previous, current)

            # Plan actions from current state
            actions = plan_actions_from_home_base(current)

            # Queue notifications for the TUI
            for change in changes:
                queue_action(
                    {
                        "type": "change",
                        "cycle": cycle,
                        "timestamp": time.time(),
                        "detail": change,
                    }
                )

            for action in actions:
                if not action.requires_approval:
                    queue_action(
                        {
                            "type": "auto_action",
                            "cycle": cycle,
                            "timestamp": time.time(),
                            "description": action.description,
                            "command": action.command,
                        }
                    )
                else:
                    queue_action(
                        {
                            "type": "suggestion",
                            "cycle": cycle,
                            "timestamp": time.time(),
                            "description": action.description,
                        }
                    )

            # Log cycle summary
            logger.info(
                "Cycle %d: %d changes, %d actions queued, %d stale deliverables",
                cycle,
                len(changes),
                len(actions),
                len(current.tasks.stale_deliverables),
            )

        except Exception:
            logger.exception("PAT daemon cycle %d failed", cycle)

        # Wait for next cycle
        for _ in range(CYCLE_SECONDS):
            if not _running:
                break
            time.sleep(1)

    # Cleanup
    if PID_PATH.exists():
        PID_PATH.unlink()
    logger.info("PAT daemon stopped after %d cycles", cycle)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
    )
    run_daemon()
