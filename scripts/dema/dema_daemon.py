"""Dema daemon — single-tick local presence loop.

A "tick" is a no-network, no-desktop heartbeat that:
  1. confirms the profile is present
  2. reads current mission state
  3. appends one tick entry to today's daily log
  4. emits a tick receipt

No autonomous action is taken. The daemon does not call out, does not write
to MEMORY.md, does not touch the network or desktop. It is the always-on
local presence that makes Dema feel alive.

Usage:
    python scripts/dema/dema_daemon.py --once
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema import (  # noqa: E402
    DailyLog,
    DailyLogEntry,
    DemaReceipt,
    MissionStateMachine,
    ProfileStore,
    ReceiptWriter,
)

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"


def tick(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    profile = ProfileStore(root).load()
    state = MissionStateMachine(root).get()

    log = DailyLog(root)
    receipt = DemaReceipt(
        action="dema.daemon.tick",
        truth_label="MEASURED",
        touched_paths=[
            str(root / "logs"),
            str(root / "receipts"),
        ],
        not_touched_paths=[
            "network",
            "desktop",
            "MEMORY.md",
            "docs/canon/",
            "social",
        ],
        approval_required=False,
        approval_status="n/a",
        payload={
            "profile_present": profile is not None,
            "mission_truth_label": state.truth_label,
            "actionable": state.is_actionable(),
        },
    )
    rid, receipt_path = ReceiptWriter(root).write(receipt)

    log_path = log.append(
        DailyLogEntry(
            timestamp=receipt.timestamp,
            kind="tick",
            summary=(
                "tick: profile=%s state=%s"
                % (
                    "present" if profile else "missing",
                    state.truth_label,
                )
            ),
            receipt_id=rid,
            metadata={"actionable": state.is_actionable()},
        )
    )

    return {
        "ok": True,
        "kind": "dema_daemon_tick",
        "receipt_id": rid,
        "receipt_path": str(receipt_path),
        "log_path": str(log_path),
        "profile_present": profile is not None,
        "mission_truth_label": state.truth_label,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="Run a single tick and exit."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )
    args = parser.parse_args(argv)

    if not args.once:
        parser.error(
            "Only --once is supported in v0.1 (no continuous loop yet)."
        )

    out = tick(args.root)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
