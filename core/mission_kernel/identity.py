"""Identity binding for Mission Kernel receipts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IdentityRecord:
    identity_id: str
    public_key: str
    role: str = "mission-kernel-signer"
    status: str = "active"

    def __post_init__(self) -> None:
        if not self.identity_id.strip():
            raise ValueError("identity_id must be non-empty")
        if not self.public_key.strip():
            raise ValueError("public_key must be non-empty")
        if self.status not in {"active", "revoked"}:
            raise ValueError("status must be active or revoked")


class IdentityRegistry:
    """Mandatory signer-id to public-key binding.

    Receipt signatures alone are not enough; the signer public key must match
    the expected key for signer_id. This prevents identity ambiguity and
    self-declared signer keys from becoming authority.
    """

    def __init__(self, records: dict[str, IdentityRecord]) -> None:
        self._records = dict(records)

    @classmethod
    def from_file(cls, path: Path) -> "IdentityRegistry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "identity_registry.v1":
            raise ValueError("unsupported identity registry schema version")
        records: dict[str, IdentityRecord] = {}
        for item in raw.get("identities", []):
            record = IdentityRecord(
                identity_id=str(item["identity_id"]),
                public_key=str(item["public_key"]),
                role=str(item.get("role", "mission-kernel-signer")),
                status=str(item.get("status", "active")),
            )
            records[record.identity_id] = record
        return cls(records)

    def expected_public_key(self, identity_id: str) -> str | None:
        record = self._records.get(identity_id)
        if record is None or record.status != "active":
            return None
        return record.public_key

    def require_public_key(self, identity_id: str) -> str:
        expected = self.expected_public_key(identity_id)
        if expected is None:
            raise ValueError(f"no active key binding for identity: {identity_id}")
        return expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "identity_registry.v1",
            "identities": [record.__dict__ for record in self._records.values()],
        }
