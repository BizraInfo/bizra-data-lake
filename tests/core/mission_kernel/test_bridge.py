"""Identity-bound mission receipt bridge tests."""

from __future__ import annotations

import pytest

from core.mission_kernel.bridge import (
    IdentityBoundReceiptError,
    ReceiptVerificationError,
    SignerMismatchError,
    create_identity_bound_receipt,
    create_receipt,
    verify_identity_bound_receipt,
)
from core.pci.crypto import generate_keypair
from core.proof_engine.receipt import Ed25519Signer


class FakeIdentityRegistry:
    def __init__(self) -> None:
        self._keys: dict[str, str] = {}
        self._revoked: set[str] = set()

    def register(self, signer_id: str, public_key_hex: str) -> None:
        self._keys[signer_id] = public_key_hex

    def revoke(self, signer_id: str) -> None:
        self._revoked.add(signer_id)

    def require_public_key(self, signer_id: str) -> str:
        if signer_id not in self._keys:
            raise IdentityBoundReceiptError("unknown signer")
        if signer_id in self._revoked:
            raise IdentityBoundReceiptError("revoked signer")
        return self._keys[signer_id]


@pytest.fixture
def signer() -> Ed25519Signer:
    private_key, public_key = generate_keypair()
    return Ed25519Signer(private_key_hex=private_key, public_key_hex=public_key)


@pytest.fixture
def other_signer() -> Ed25519Signer:
    private_key, public_key = generate_keypair()
    return Ed25519Signer(private_key_hex=private_key, public_key_hex=public_key)


@pytest.fixture
def registry(signer: Ed25519Signer) -> FakeIdentityRegistry:
    registry = FakeIdentityRegistry()
    registry.register("node0", signer.public_key_hex)
    return registry


def _create_receipt(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
    *,
    signer_id: str = "node0",
    status: str = "accepted",
    reason: str | None = None,
) -> dict:
    return create_identity_bound_receipt(
        registry=registry,
        signer_id=signer_id,
        signer=signer,
        mission_id="mission-001",
        status=status,
        mission_hash="a" * 64,
        payload_hash="b" * 64,
        prev_hash="c" * 64,
        reason=reason,
        metadata={"lane": "test"},
    )


def test_unknown_signer_rejected(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    with pytest.raises(IdentityBoundReceiptError, match="unknown"):
        _create_receipt(registry, signer, signer_id="missing")


def test_revoked_signer_rejected(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    registry.revoke("node0")
    with pytest.raises(IdentityBoundReceiptError, match="revoked"):
        _create_receipt(registry, signer)


def test_mismatched_key_rejected(
    registry: FakeIdentityRegistry,
    other_signer: Ed25519Signer,
) -> None:
    with pytest.raises(SignerMismatchError):
        _create_receipt(registry, other_signer)


def test_matching_key_signs_and_verifies(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    receipt = _create_receipt(registry, signer)
    assert receipt["creation_path"] == "identity-bound-bridge"
    assert verify_identity_bound_receipt(
        receipt,
        expected_public_key=signer.public_key_hex,
    )


def test_wrong_expected_public_key_fails(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
    other_signer: Ed25519Signer,
) -> None:
    receipt = _create_receipt(registry, signer)
    assert not verify_identity_bound_receipt(
        receipt,
        expected_public_key=other_signer.public_key_hex,
    )


def test_tampered_hash_fails(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    receipt = _create_receipt(registry, signer)
    receipt["receipt_hash"] = "0" * 64
    assert not verify_identity_bound_receipt(
        receipt,
        expected_public_key=signer.public_key_hex,
    )


def test_tampered_signature_fails(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    receipt = _create_receipt(registry, signer)
    receipt["signature"] = "0" * 128
    assert not verify_identity_bound_receipt(
        receipt,
        expected_public_key=signer.public_key_hex,
    )


def test_rejected_mission_receipts_still_identity_bound(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    receipt = _create_receipt(
        registry,
        signer,
        status="rejected",
        reason="policy gate failed",
    )
    assert receipt["status"] == "rejected"
    assert receipt["reason"] == "policy gate failed"
    assert verify_identity_bound_receipt(
        receipt,
        expected_public_key=signer.public_key_hex,
    )


def test_prev_hash_round_trip_preserved(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
) -> None:
    receipt = _create_receipt(registry, signer)
    assert receipt["prev_hash"] == "c" * 64


def test_direct_create_receipt_path_marked_test_bootstrap_only(
    signer: Ed25519Signer,
) -> None:
    receipt = create_receipt(
        signer_id="bootstrap",
        signer=signer,
        mission_id="mission-bootstrap",
        status="accepted",
        mission_hash="d" * 64,
        payload_hash="e" * 64,
        prev_hash="f" * 64,
    )
    assert receipt["creation_path"] == "test/bootstrap-only"
    assert verify_identity_bound_receipt(
        receipt,
        expected_public_key=signer.public_key_hex,
    )


def test_bridge_raises_when_post_sign_verify_fails(
    registry: FakeIdentityRegistry,
    signer: Ed25519Signer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "core.mission_kernel.bridge.verify_identity_bound_receipt",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(ReceiptVerificationError):
        _create_receipt(registry, signer)
