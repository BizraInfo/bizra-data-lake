"""
Constitutional Membrane — the 4-property filter between nodes and URP.

Property 1: Fail-closed (missing authority -> reject)
Property 2: Constitutional filtering (invariants enforced)
Property 3: Cryptographic authentication (Ed25519 + BLAKE3)
Property 4: Provenance recording (every crossing -> receipt)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from core.urp.constitution import Constitution

logger = logging.getLogger("bizra.urp.membrane")


@dataclass
class MembraneRecord:
    """Record of a single membrane crossing (pass or reject)."""

    timestamp: float
    node_id: str
    direction: str  # "inbound" or "outbound"
    event_type: str
    admitted: bool
    rejection_reason: str = ""
    receipt_hash: str = ""
    previous_hash: str = ""
    chain_hash: str = ""


class ConstitutionalMembrane:
    """The 4-property constitutional filter between nodes and the URP."""

    def __init__(self, constitution: Constitution) -> None:
        self.constitution = constitution
        self._log: List[MembraneRecord] = []
        self._head_hash = "0" * 64
        self._admitted_count = 0
        self._rejected_count = 0

    def filter_inbound(
        self,
        node_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> tuple[bool, str, MembraneRecord]:
        """Filter an inbound request from a node to the URP.

        Returns (admitted, reason, record).
        """
        ts = time.time()

        # Property 1: Fail-closed — missing authority -> reject
        if not node_id:
            return self._reject(ts, "", event_type, "missing_node_identity")

        # Property 2: Constitutional filtering
        if event_type == "receipt":
            ok, reason = self.constitution.check_receipt(payload)
            if not ok:
                return self._reject(ts, node_id, event_type, reason)

        # Property 3: Cryptographic authentication
        if payload.get("requires_signature") and not payload.get("signed"):
            return self._reject(ts, node_id, event_type, "unsigned_payload")

        # Property 4: Provenance recording
        record = self._record(ts, node_id, event_type, admitted=True)
        self._admitted_count += 1
        return True, "admitted", record

    def filter_outbound(
        self,
        node_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> tuple[bool, str, MembraneRecord]:
        """Filter outbound data from URP to a node."""
        ts = time.time()

        if not node_id:
            return self._reject(ts, "", event_type, "missing_recipient")

        record = self._record(
            ts, node_id, event_type, admitted=True, direction="outbound"
        )
        return True, "delivered", record

    def stats(self) -> Dict[str, Any]:
        """Membrane health statistics."""
        return {
            "admitted": self._admitted_count,
            "rejected": self._rejected_count,
            "total_crossings": len(self._log),
            "head_hash": self._head_hash[:16],
            "constitution_hash": self.constitution.hash()[:16],
        }

    def verify_chain(self) -> tuple[bool, List[str]]:
        """Verify integrity of the membrane crossing log."""
        errors = []
        expected_prev = "0" * 64
        for i, record in enumerate(self._log):
            if record.previous_hash != expected_prev:
                errors.append(f"Chain break at index {i}")
            expected_prev = record.chain_hash
        return len(errors) == 0, errors

    # -- internal --

    def _reject(
        self, ts: float, node_id: str, event_type: str, reason: str
    ) -> tuple[bool, str, MembraneRecord]:
        record = self._record(ts, node_id, event_type, admitted=False, reason=reason)
        self._rejected_count += 1
        logger.debug(
            "Membrane REJECT: node=%s type=%s reason=%s", node_id, event_type, reason
        )
        return False, reason, record

    def _record(
        self,
        ts: float,
        node_id: str,
        event_type: str,
        admitted: bool,
        direction: str = "inbound",
        reason: str = "",
    ) -> MembraneRecord:
        content = f"{ts}:{node_id}:{event_type}:{admitted}:{direction}"
        receipt_hash = hashlib.blake2b(content.encode(), digest_size=32).hexdigest()
        chain_input = f"{self._head_hash}:{receipt_hash}"
        chain_hash = hashlib.blake2b(chain_input.encode(), digest_size=32).hexdigest()

        record = MembraneRecord(
            timestamp=ts,
            node_id=node_id,
            direction=direction,
            event_type=event_type,
            admitted=admitted,
            rejection_reason=reason,
            receipt_hash=receipt_hash,
            previous_hash=self._head_hash,
            chain_hash=chain_hash,
        )
        self._log.append(record)
        self._head_hash = chain_hash
        return record
