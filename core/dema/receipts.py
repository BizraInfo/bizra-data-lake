"""Dema local receipt — minimal hash-chained audit envelope.

Distinct from the canonical CanonicalReceipt (Ed25519-signed, BLAKE3-chained).
This is the lightweight local audit trail for Ambient Kernel actions:
onboarding, daily log ticks, dream phases, mission state updates.

Each receipt carries:
- receipt_id: BLAKE3 of canonical payload
- truth_label: MEASURED | DERIVED | PLANNED | SANDBOX
- action: short verb (e.g. "dema.onboarding.init")
- touched_paths: list of paths the action wrote
- not_touched_paths: explicit non-claims (e.g. network, desktop, MEMORY.md)
- approval_required / approval_status: human-in-loop discipline
- payload_digest: hash of caller-supplied detail

Receipts land under sovereign_state/dema/receipts/<date>/<receipt_id>.json
(gitignored).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import blake3

VALID_TRUTH_LABELS = ("MEASURED", "DERIVED", "PLANNED", "SANDBOX")
VALID_APPROVAL = ("granted", "pending", "n/a", "denied")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(payload: dict[str, Any]) -> str:
    return blake3.blake3(_canonical_bytes(payload)).hexdigest()


@dataclass
class DemaReceipt:
    action: str
    truth_label: str
    touched_paths: list[str]
    not_touched_paths: list[str] = field(default_factory=list)
    approval_required: bool = False
    approval_status: str = "n/a"
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)
    schema_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if self.truth_label not in VALID_TRUTH_LABELS:
            raise ValueError(
                f"truth_label must be one of {VALID_TRUTH_LABELS}, "
                f"got {self.truth_label!r}"
            )
        if self.approval_status not in VALID_APPROVAL:
            raise ValueError(
                f"approval_status must be one of {VALID_APPROVAL}, "
                f"got {self.approval_status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def receipt_id(self) -> str:
        return _digest(self.to_dict())

    def payload_digest(self) -> str:
        return _digest(self.payload)


class ReceiptWriter:
    """Writes Dema receipts to a date-partitioned local store."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "receipts"
        self.root.mkdir(parents=True, exist_ok=True)

    def _date_dir(self, date: str | None = None) -> Path:
        d = self.root / (date or _today_utc())
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, receipt: DemaReceipt) -> tuple[str, Path]:
        rid = receipt.receipt_id()
        sealed = {
            **receipt.to_dict(),
            "receipt_id": rid,
            "payload_digest": receipt.payload_digest(),
        }
        path = self._date_dir() / f"{rid[:16]}.json"
        path.write_text(
            json.dumps(sealed, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return rid, path
