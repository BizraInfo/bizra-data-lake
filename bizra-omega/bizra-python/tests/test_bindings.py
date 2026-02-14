"""
Tests for BIZRA Python Bindings

These tests verify the Python interface to the Rust implementation.
"""

import pytest
import json


def test_import():
    """Test that bizra module can be imported."""
    from bizra import (
        NodeId, NodeIdentity, Constitution,
        IHSAN_THRESHOLD, SNR_THRESHOLD
    )
    assert IHSAN_THRESHOLD == 0.95
    assert SNR_THRESHOLD == 0.85


def test_node_identity_generation():
    """Test NodeIdentity generation."""
    from bizra import NodeIdentity

    identity = NodeIdentity()
    assert identity.node_id is not None
    assert len(identity.public_key) == 64  # 32 bytes hex


def test_node_identity_signing():
    """Test message signing and verification."""
    from bizra import NodeIdentity

    identity = NodeIdentity()
    message = b"Hello, BIZRA!"

    signature = identity.sign(message)
    assert len(signature) == 128  # 64 bytes hex

    # Verify
    assert NodeIdentity.verify(message, signature, identity.public_key)

    # Tampered message fails
    assert not NodeIdentity.verify(b"tampered", signature, identity.public_key)


def test_constitution():
    """Test Constitution validation."""
    from bizra import Constitution

    constitution = Constitution()

    # Ihsan checks
    assert constitution.check_ihsan(0.95)
    assert constitution.check_ihsan(0.99)
    assert not constitution.check_ihsan(0.94)

    # SNR checks
    assert constitution.check_snr(0.85)
    assert constitution.check_snr(0.95)
    assert not constitution.check_snr(0.84)


def test_domain_separated_digest():
    """Test domain-separated hashing."""
    from bizra import domain_separated_digest

    msg = b"test message"
    digest1 = domain_separated_digest(msg)
    digest2 = domain_separated_digest(msg)

    # Deterministic
    assert digest1 == digest2

    # Different messages give different digests
    digest3 = domain_separated_digest(b"different")
    assert digest1 != digest3


def test_node_id_validation():
    """Test NodeId validation."""
    from bizra import NodeId

    # Valid 32-char hex
    node_id = NodeId("a" * 32)
    assert node_id.id == "a" * 32

    # Invalid length should raise
    with pytest.raises(ValueError):
        NodeId("tooshort")

    with pytest.raises(ValueError):
        NodeId("a" * 64)  # Too long


def test_pci_envelope():
    """Test PCI envelope creation (if available)."""
    try:
        from bizra import NodeIdentity, PCIEnvelope

        identity = NodeIdentity()
        payload = json.dumps({"action": "test", "value": 42})

        envelope = PCIEnvelope.create(identity, payload, ttl=3600, provenance=[])

        assert envelope.id.startswith("pci_")
        assert envelope.ttl == 3600
        assert envelope.signature is not None

    except (ImportError, AttributeError):
        pytest.skip("PCIEnvelope not available in this build")


def test_performance():
    """Basic performance test."""
    from bizra import NodeIdentity, domain_separated_digest
    import time

    # Generate 100 signatures
    identity = NodeIdentity()
    message = b"performance test message" * 10

    start = time.perf_counter()
    for _ in range(100):
        identity.sign(message)
    elapsed = time.perf_counter() - start

    print(f"\n100 signatures: {elapsed*1000:.2f}ms ({100/elapsed:.0f} ops/sec)")
    assert elapsed < 5.0  # Should be fast

    # Hash 1000 messages
    start = time.perf_counter()
    for i in range(1000):
        domain_separated_digest(f"message {i}".encode())
    elapsed = time.perf_counter() - start

    print(f"1000 hashes: {elapsed*1000:.2f}ms ({1000/elapsed:.0f} ops/sec)")
    assert elapsed < 1.0  # Should be very fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
