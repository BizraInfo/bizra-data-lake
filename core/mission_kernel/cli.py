"""Mission Kernel command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.mission_kernel.chain import JsonlReceiptStore
from core.mission_kernel.identity import IdentityRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mission-kernel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-chain", help="verify a JSONL Receipt v1 chain")
    verify.add_argument("receipt_log", type=Path, help="path to receipts.jsonl")
    verify.add_argument(
        "--identity-registry",
        type=Path,
        default=None,
        help="optional identity_registry.v1 JSON file for signer binding",
    )
    verify.add_argument("--json", action="store_true", help="emit machine-readable report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-chain":
        registry = (
            IdentityRegistry.from_file(args.identity_registry)
            if args.identity_registry is not None
            else None
        )
        report = JsonlReceiptStore(args.receipt_log).verify_chain(registry)
        payload = {
            "ok": report.ok,
            "receipts_checked": report.receipts_checked,
            "errors": list(report.errors),
            "chain_tail": report.chain_tail,
        }
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            status = "OK" if report.ok else "FAIL"
            print(f"mission-kernel verify-chain: {status}")
            print(f"receipts_checked={report.receipts_checked}")
            print(f"chain_tail={report.chain_tail}")
            for error in report.errors:
                print(f"error={error}", file=sys.stderr)
        return 0 if report.ok else 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
