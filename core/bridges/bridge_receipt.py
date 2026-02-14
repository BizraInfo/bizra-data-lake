"""
Bridge Receipt Engine — Signed PCI receipts for every bridge command.

Every Desktop Bridge invocation (ping, status, sovereign_query, invoke_skill,
list_skills) produces a cryptographically signed receipt that is:
  1. Returned in the JSON-RPC response under "receipt"
  2. Persisted to sovereign_state/bridge_receipts/ as atomic JSON files
  3. Retrievable via get_receipt method

Standing on Giants:
- Lamport (1978): Logical clocks for causal ordering
- Saltzer & Schroeder (1975): Fail-closed — no unsigned proofs
- O'Connor et al. (2020): BLAKE3 deterministic hashing
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("bizra.bridge_receipt")

# Maximum cached receipts in memory
_MAX_CACHE = 100

# Default receipt directory
_DEFAULT_RECEIPT_DIR = Path("sovereign_state/bridge_receipts")


def load_signer() -> Any:
    """
    Load receipt signer from environment.

    Hard cutover: strict-key-only startup.
    BIZRA_RECEIPT_PRIVATE_KEY_HEX MUST be set to valid hex.
    """
    from core.proof_engine.receipt import SimpleSigner

    key_hex = os.getenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "").strip()
    if not key_hex:
        raise RuntimeError(
            "Missing BIZRA_RECEIPT_PRIVATE_KEY_HEX; bridge receipts require a signing key"
        )
    try:
        key_bytes = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise RuntimeError(
            "Invalid BIZRA_RECEIPT_PRIVATE_KEY_HEX (must be hex)"
        ) from exc
    if not key_bytes:
        raise RuntimeError("Invalid BIZRA_RECEIPT_PRIVATE_KEY_HEX (empty key)")
    return SimpleSigner(key_bytes)


class BridgeReceiptEngine:
    """
    Creates, signs, persists, and retrieves PCI receipts for bridge commands.

    Receipts are the proof of execution — every bridge invocation, whether
    accepted or rejected, produces an immutable signed record.
    """

    def __init__(
        self,
        signer: Any = None,
        receipt_dir: Optional[Path] = None,
    ) -> None:
        self._signer = signer or load_signer()
        self.receipt_dir = receipt_dir or _DEFAULT_RECEIPT_DIR
        self.receipt_dir.mkdir(parents=True, exist_ok=True)

        # LRU cache: receipt_id -> receipt dict
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._counter = 0

    def _next_id(self, method: str) -> str:
        """Generate next receipt ID."""
        self._counter += 1
        ts = int(time.time() * 1000)
        return f"br-{ts}-{self._counter:04d}-{method}"

    def create_receipt(
        self,
        method: str,
        query_data: Dict[str, Any],
        result_data: Dict[str, Any],
        fate_score: float,
        snr_score: float,
        gate_passed: str,
        status: str,
        duration_ms: float = 0.0,
        reason: Optional[str] = None,
        origin: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a signed receipt for a bridge command.

        Args:
            method: JSON-RPC method name
            query_data: Input parameters
            result_data: Output data
            fate_score: FATE gate overall score
            snr_score: SNR score
            gate_passed: Last gate passed (or failed at)
            status: "accepted", "rejected", or "amber_restricted"
            duration_ms: Execution duration
            reason: Rejection reason (if applicable)

        Returns:
            Receipt dict with receipt_id, status, digests, signature
        """
        from core.proof_engine.canonical import blake3_digest, canonical_bytes

        receipt_id = self._next_id(method)

        # Compute deterministic digests
        query_digest = blake3_digest(canonical_bytes(query_data)).hex()
        policy_digest = blake3_digest(b"bizra-bridge-v1").hex()
        payload_digest = blake3_digest(canonical_bytes(result_data)).hex()

        origin_payload = origin or {
            "designation": "ephemeral_node",
            "genesis_node": False,
            "genesis_block": False,
            "home_base_device": False,
            "authority_source": "genesis_files",
            "hash_validated": False,
        }
        origin_digest = blake3_digest(canonical_bytes(origin_payload)).hex()

        # Build receipt body
        body = {
            "receipt_id": receipt_id,
            "status": status,
            "method": method,
            "query_digest": query_digest,
            "policy_digest": policy_digest,
            "payload_digest": payload_digest,
            "origin": origin_payload,
            "origin_digest": origin_digest,
            "fate_score": fate_score,
            "snr_score": snr_score,
            "gate_passed": gate_passed,
            "reason": reason,
            "duration_ms": round(duration_ms, 2),
            "timestamp": time.time(),
        }

        # Sign
        body_bytes = canonical_bytes(body)
        signature = self._signer.sign(body_bytes)
        if not self._signer.verify(body_bytes, signature):
            raise RuntimeError("Signer self-verification failed")
        body["signature"] = signature.hex()
        body["signer_pubkey"] = self._signer.public_key_bytes().hex()
        body["receipt_digest"] = blake3_digest(body_bytes + signature).hex()

        # Cache
        self._cache_put(receipt_id, body)

        # Persist (best-effort, non-blocking)
        self._persist(receipt_id, body)

        return body

    def get_receipt(self, receipt_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a receipt by ID.

        Checks in-memory cache first, then disk.
        """
        # Cache hit
        if receipt_id in self._cache:
            self._cache.move_to_end(receipt_id)
            return self._cache[receipt_id]

        # Disk fallback
        path = self.receipt_dir / f"{receipt_id}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._cache_put(receipt_id, data)
                return data
            except (json.JSONDecodeError, OSError):
                logger.warning(f"Failed to read receipt: {path}")
        return None

    def verify_receipt(self, receipt: Dict[str, Any]) -> bool:
        """Verify a receipt's signature."""
        from core.proof_engine.canonical import canonical_bytes

        sig_hex = receipt.get("signature", "")
        if not sig_hex:
            return False

        # Rebuild body without signature fields
        body = {
            k: v
            for k, v in receipt.items()
            if k not in ("signature", "signer_pubkey", "receipt_digest")
        }
        body_bytes = canonical_bytes(body)
        return self._signer.verify(body_bytes, bytes.fromhex(sig_hex))

    def _persist(self, receipt_id: str, data: Dict[str, Any]) -> None:
        """Atomic write receipt to disk."""
        path = self.receipt_dir / f"{receipt_id}.json"
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.rename(path)
        except OSError as exc:
            logger.warning(f"Receipt persist failed: {exc}")
            # Clean up temp file
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _cache_put(self, receipt_id: str, data: Dict[str, Any]) -> None:
        """Add to LRU cache, evicting oldest if over limit."""
        self._cache[receipt_id] = data
        self._cache.move_to_end(receipt_id)
        while len(self._cache) > _MAX_CACHE:
            self._cache.popitem(last=False)
