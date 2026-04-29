"""Dema CSL CLI — emit the canonical TypeScript mirror from the Python source.

Usage:
    # Print the TS mirror to stdout
    python scripts/dema/dema_csl.py emit-ts

    # Write to disk (default target: frontend/src/lib/dema-csl.ts)
    python scripts/dema/dema_csl.py emit-ts --write

Python is the source of truth. The drift test in
``tests/scripts/test_dema_csl.py::test_typescript_mirror_matches_python``
asserts that the on-disk TS mirror equals the freshly emitted text. If they
diverge, regenerate with ``--write`` and commit the result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema.csl import (  # noqa: E402
    APPROVAL_STATUSES,
    DECISION_VERDICTS,
    DISPLAY_TRUTH_LABELS,
    MISSION_TRUTH_LABELS,
    RECEIPT_TRUTH_LABELS,
    RISK_LEVELS,
    SCHEMA_VERSION,
)

DEFAULT_TS_PATH = REPO_ROOT / "frontend" / "src" / "lib" / "dema-csl.ts"


def _ts_array(values: tuple[str, ...]) -> str:
    quoted = ", ".join(f'"{v}"' for v in values)
    return f"[{quoted}] as const"


def emit_ts() -> str:
    return f"""// AUTO-GENERATED from core/dema/csl/labels.py — do not edit by hand.
// Regenerate with: python scripts/dema/dema_csl.py emit-ts --write
//
// Python is the source of truth. The Python drift test
// (tests/scripts/test_dema_csl.py::test_typescript_mirror_matches_python)
// fails CI if this file falls out of sync.

export const CSL_SCHEMA_VERSION = "{SCHEMA_VERSION}" as const;

export const RECEIPT_TRUTH_LABELS = {_ts_array(RECEIPT_TRUTH_LABELS)};
export type ReceiptTruthLabel = (typeof RECEIPT_TRUTH_LABELS)[number];

export const DISPLAY_TRUTH_LABELS = {_ts_array(DISPLAY_TRUTH_LABELS)};
export type DisplayTruthLabel = (typeof DISPLAY_TRUTH_LABELS)[number];

export const MISSION_TRUTH_LABELS = {_ts_array(MISSION_TRUTH_LABELS)};
export type MissionTruthLabel = (typeof MISSION_TRUTH_LABELS)[number];

export const RISK_LEVELS = {_ts_array(RISK_LEVELS)};
export type RiskLevel = (typeof RISK_LEVELS)[number];

export const APPROVAL_STATUSES = {_ts_array(APPROVAL_STATUSES)};
export type ApprovalStatus = (typeof APPROVAL_STATUSES)[number];

export const DECISION_VERDICTS = {_ts_array(DECISION_VERDICTS)};
export type DecisionVerdict = (typeof DECISION_VERDICTS)[number];
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("emit-ts", help="Emit the canonical TypeScript mirror.")
    e.add_argument(
        "--write",
        action="store_true",
        help=f"Write to {DEFAULT_TS_PATH.relative_to(REPO_ROOT)} instead of stdout.",
    )
    e.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TS_PATH,
        help="Override target path for --write.",
    )

    args = parser.parse_args(argv)

    if args.command == "emit-ts":
        text = emit_ts()
        if args.write:
            args.target.parent.mkdir(parents=True, exist_ok=True)
            args.target.write_text(text, encoding="utf-8")
            print(f"wrote {args.target}")
        else:
            sys.stdout.write(text)
        return 0

    parser.error("unknown command")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
