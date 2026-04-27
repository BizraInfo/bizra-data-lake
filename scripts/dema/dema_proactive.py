"""Dema proactive — evaluate ambient signals through the policy layer.

Truth label: MEASURED at the file level (every evaluate emits a real receipt
+ proposal artifact under sovereign_state/dema/).

Usage:
    python scripts/dema/dema_proactive.py evaluate \\
        --signal downloads_folder_large \\
        --confidence 0.87 \\
        --urgency low

The CLI does NOT auto-execute any system action. It only emits a proposal
+ receipt; the operator must accept and run the action manually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.dema.proactive import (  # noqa: E402
    AmbientSignal,
    ProposalWriter,
    VALID_SIGNAL_KINDS,
    compose_proposal,
    decide,
    predict_intent,
)

DEFAULT_ROOT = REPO_ROOT / "sovereign_state" / "dema"


def evaluate(
    *,
    root: Path,
    kind: str,
    confidence: float,
    urgency: str,
    user_preference: str,
) -> dict[str, object]:
    signal = AmbientSignal(
        kind=kind,
        confidence=confidence,
        urgency=urgency,
        source="cli",
    )
    intent = predict_intent(signal)
    decision = decide(intent, user_preference=user_preference)
    proposal = compose_proposal(signal, decision)

    rid, proposal_path, receipt_path = ProposalWriter(root).write(proposal)

    return {
        "ok": True,
        "kind": "dema_proactive_evaluate",
        "signal": signal.to_dict(),
        "intent": intent.to_dict(),
        "decision": decision.to_dict(),
        "proposal": proposal.to_dict(),
        "receipt_id": rid,
        "proposal_path": str(proposal_path),
        "receipt_path": str(receipt_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("evaluate", help="Evaluate one ambient signal.")
    e.add_argument(
        "--signal",
        required=True,
        choices=list(VALID_SIGNAL_KINDS),
        help="Signal kind to evaluate.",
    )
    e.add_argument(
        "--confidence",
        type=float,
        required=True,
        help="Confidence in [0, 1].",
    )
    e.add_argument(
        "--urgency",
        default="low",
        choices=["low", "medium", "high"],
        help="Urgency hint (default: low).",
    )
    e.add_argument(
        "--user-preference",
        default="default",
        help="Operator preference channel (default: default).",
    )
    e.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Local Dema state root (default: {DEFAULT_ROOT}).",
    )

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        out = evaluate(
            root=args.root,
            kind=args.signal,
            confidence=args.confidence,
            urgency=args.urgency,
            user_preference=args.user_preference,
        )
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    parser.error("unknown command")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
