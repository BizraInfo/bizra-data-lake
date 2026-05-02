"""Receipt chain storage and verification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.mission_kernel.identity import IdentityRegistry
from core.mission_kernel.receipt import ReceiptV1, receipt_from_dict, receipt_to_dict, verify_receipt


@dataclass(frozen=True)
class ChainReport:
    ok: bool
    receipts_checked: int
    errors: tuple[str, ...]
    chain_tail: str | None


class JsonlReceiptStore:
    """Append-only JSONL receipt store.

    v0.1 intentionally uses JSONL to keep Node0 replay simple, local, and
    inspectable. A later storage backend may wrap this trait with SQLite or a
    content-addressed ledger.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, receipt: ReceiptV1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt_to_dict(receipt), sort_keys=True))
            handle.write("\n")

    def read_all(self) -> tuple[ReceiptV1, ...]:
        if not self.path.exists():
            return ()
        receipts: list[ReceiptV1] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    receipts.append(receipt_from_dict(json.loads(stripped)))
                except Exception as exc:  # noqa: BLE001 - verifier reports corrupted line
                    raise ValueError(f"invalid receipt jsonl line {line_number}: {exc}") from exc
        return tuple(receipts)

    def verify_chain(self, registry: IdentityRegistry | None = None) -> ChainReport:
        return verify_chain(self.read_all(), registry=registry)


def verify_chain(
    receipts: Iterable[ReceiptV1], *, registry: IdentityRegistry | None = None
) -> ChainReport:
    errors: list[str] = []
    previous_tail: str | None = None
    count = 0

    for count, receipt in enumerate(receipts, start=1):
        if receipt.prev_hash != previous_tail:
            errors.append(
                f"receipt[{count}] prev_hash mismatch: got {receipt.prev_hash!r}, expected {previous_tail!r}"
            )
        expected_public_key = None
        if registry is not None:
            expected_public_key = registry.expected_public_key(receipt.signer_id)
            if expected_public_key is None:
                errors.append(f"receipt[{count}] no active identity binding for {receipt.signer_id}")
        if not verify_receipt(receipt, expected_public_key=expected_public_key):
            errors.append(f"receipt[{count}] signature/hash verification failed")
        previous_tail = receipt.current_hash

    return ChainReport(
        ok=not errors,
        receipts_checked=count,
        errors=tuple(errors),
        chain_tail=previous_tail,
    )
