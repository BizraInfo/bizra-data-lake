"""Identity-bound Receipt v1 bridge.

This module is the production cryptographic lane between mission execution and
receipt emission. It does not define a new truth engine; it binds the existing
Receipt v1 continuity format to an explicit identity registry lookup.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping, Protocol, runtime_checkable

from core.proof_engine.canonical import canonical_bytes, hex_digest


class IdentityBoundReceiptError(RuntimeError):
    """Base error for identity-bound receipt failures."""


class SignerMismatchError(IdentityBoundReceiptError):
    """Raised when a signer does not match the registry identity."""


class ReceiptVerificationError(IdentityBoundReceiptError):
    """Raised when a generated or supplied receipt fails cryptographic checks."""


@runtime_checkable
class IdentityRegistry(Protocol):
    """Registry contract required by the mission kernel bridge."""

    def require_public_key(self, signer_id: str) -> str | bytes:
        """Return the active public key for signer_id or raise on invalid identity."""
        ...


@runtime_checkable
class MissionReceiptSigner(Protocol):
    """Signer contract compatible with core.proof_engine.receipt.Ed25519Signer."""

    def sign(self, msg: bytes) -> bytes:
        """Sign canonical receipt body bytes."""
        ...

    def public_key_bytes(self) -> bytes:
        """Return the signer public key bytes."""
        ...


def _normalize_public_key(public_key: str | bytes) -> str:
    if isinstance(public_key, bytes):
        public_key_hex = public_key.hex()
    else:
        public_key_hex = public_key.lower()
    try:
        bytes.fromhex(public_key_hex)
    except ValueError as exc:
        raise IdentityBoundReceiptError("registry public key must be hex") from exc
    if len(public_key_hex) != 64:
        raise IdentityBoundReceiptError("registry public key must be 32 bytes")
    return public_key_hex


def _receipt_body(receipt: Mapping[str, Any]) -> bytes:
    excluded = {"signature", "receipt_hash"}
    return canonical_bytes({k: v for k, v in receipt.items() if k not in excluded})


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_receipt(
    *,
    signer_id: str,
    signer: MissionReceiptSigner,
    mission_id: str,
    status: str,
    mission_hash: str,
    payload_hash: str,
    prev_hash: str,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    creation_path: str = "test/bootstrap-only",
) -> dict[str, Any]:
    """Create a signed Receipt v1 payload.

    Direct use is retained for tests/bootstrap only. Production callers must use
    create_identity_bound_receipt(), which performs registry lookup and
    expected-public-key verification before returning the receipt.
    """

    signer_public_key = signer.public_key_bytes().hex()
    receipt: dict[str, Any] = {
        "receipt_version": "v1",
        "hash_algorithm": "sha256",
        "signature_algorithm": "ed25519-blake3-digest",
        "authority": "mission_kernel.identity_bound_receipt.v1",
        "creation_path": creation_path,
        "truth_label": "Cryptographic lane = LOCAL_VALIDATED / CI_PENDING",
        "signer_id": signer_id,
        "signer_public_key": signer_public_key,
        "mission_id": mission_id,
        "status": status,
        "mission_hash": mission_hash,
        "payload_hash": payload_hash,
        "prev_hash": prev_hash,
        "reason": reason,
        "metadata": dict(metadata or {}),
        "timestamp_ms": int(time.time() * 1000),
    }
    body = _receipt_body(receipt)
    signature = signer.sign(body).hex()
    receipt["signature"] = signature
    receipt["receipt_hash"] = _sha256_hex(body + bytes.fromhex(signature))
    return receipt


def verify_identity_bound_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_public_key: str | bytes,
) -> bool:
    """Verify receipt hash, signature, and expected signer public key."""

    expected_public_key_hex = _normalize_public_key(expected_public_key)
    signer_public_key = str(receipt.get("signer_public_key", "")).lower()
    if signer_public_key != expected_public_key_hex:
        return False

    body = _receipt_body(receipt)
    signature_hex = str(receipt.get("signature", ""))
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False

    expected_hash = _sha256_hex(body + signature)
    if receipt.get("receipt_hash") != expected_hash:
        return False

    from core.pci.crypto import verify_signature

    return verify_signature(hex_digest(body), signature_hex, expected_public_key_hex)


def create_identity_bound_receipt(
    *,
    registry: IdentityRegistry,
    signer_id: str,
    signer: MissionReceiptSigner,
    mission_id: str,
    status: str,
    mission_hash: str,
    payload_hash: str,
    prev_hash: str,
    reason: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a receipt only after signer_id is bound to an active registry key."""

    expected_public_key = _normalize_public_key(registry.require_public_key(signer_id))
    actual_public_key = signer.public_key_bytes().hex()
    if actual_public_key != expected_public_key:
        raise SignerMismatchError("signer public key does not match registry identity")

    receipt = create_receipt(
        signer_id=signer_id,
        signer=signer,
        mission_id=mission_id,
        status=status,
        mission_hash=mission_hash,
        payload_hash=payload_hash,
        prev_hash=prev_hash,
        reason=reason,
        metadata=metadata,
        creation_path="identity-bound-bridge",
    )
    if not verify_identity_bound_receipt(
        receipt,
        expected_public_key=expected_public_key,
    ):
        raise ReceiptVerificationError("identity-bound receipt verification failed")
    return receipt
