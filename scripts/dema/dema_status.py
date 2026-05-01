"""Dema status — emit Current/Ideal/Gap/Next as JSON.

Truth label: read-only over MEASURED inputs; output truth_label reflects the
underlying mission_state's truth_label (default PLANNED for fresh nodes).

Usage:
    python scripts/dema/dema_status.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema import DailyLog, MissionStateMachine, ProfileStore  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"


def status(root: Path) -> dict[str, object]:
    profile = ProfileStore(root, create=False).load()
    state = MissionStateMachine(root, create=False).get()
    log_today = DailyLog(root, create=False).read_today()

    return {
        "schema_version": "0.1.0",
        "kind": "dema_status",
        "profile_present": profile is not None,
        "profile": profile.to_dict() if profile else None,
        "mission_state": state.to_dict(),
        "log_today_count": len(log_today),
        "log_today_kinds": sorted({e.kind for e in log_today}),
        "actionable": state.is_actionable(),
        "truth_label": state.truth_label,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON (default true).")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )
    args = parser.parse_args(argv)

    out = status(args.root)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
