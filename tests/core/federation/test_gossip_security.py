"""
BIZRA Federation Gossip Protocol Security Test Suite

Tests for Ed25519 cryptographic signing and signature verification (SEC-016/SEC-017).
Target: 80% coverage of gossip.py security-critical paths
"""

import pytest
import asyncio
import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.federation.gossip import (
    GossipEngine,
    GossipMessage,
    MessageType,
    NodeInfo,
    NodeState,
    GOSSIP_INTERVAL_MS,
    SUSPICION_TIMEOUT_MS,
    DEAD_TIMEOUT_MS,
    MAX_FANOUT,
)
from core.pci.crypto import generate_keypair


@pytest.fixture
def keypair():
    """Generate a test Ed25519 keypair."""
    return generate_keypair()


@pytest.fixture
def gossip_engine(keypair):
    """Create a GossipEngine with valid credentials."""
    priv, pub = keypair
    return GossipEngine(
        node_id="test_node",
        address="127.0.0.1:7777",
        public_key=pub,
        private_key=priv,
    )


@pytest.fixture
def peer_keypair():
    """Generate a keypair for a peer node."""
    return generate_keypair()


class TestNodeInfoValidation:
    """Tests for NodeInfo public key validation (SEC-017)."""

    def test_valid_public_key_accepted(self, keypair):
        """NodeInfo with valid 64-char hex public key should be accepted."""
        _, pub = keypair
        node = NodeInfo(
            node_id="valid_node",
            address="127.0.0.1:8000",
            public_key=pub,
        )
        assert node.public_key == pub

    def test_empty_public_key_rejected(self):
        """NodeInfo with empty public key should raise ValueError."""
        with pytest.raises(ValueError, match="valid 64-char hex public_key"):
            NodeInfo(
                node_id="invalid_node",
                address="127.0.0.1:8000",
                public_key="",
            )

    def test_short_public_key_rejected(self):
        """NodeInfo with too-short public key should raise ValueError."""
        with pytest.raises(ValueError, match="valid 64-char hex public_key"):
            NodeInfo(
                node_id="invalid_node",
                address="127.0.0.1:8000",
                public_key="abc123",  # Too short
            )

    def test_none_public_key_rejected(self):
        """NodeInfo with None public key should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            NodeInfo(
                node_id="invalid_node",
                address="127.0.0.1:8000",
                public_key=None,
            )


class TestSeedNodeValidation:
    """Tests for seed node public key validation (SEC-017)."""

    def test_add_seed_node_requires_public_key(self, gossip_engine):
        """add_seed_node should require valid public key."""
        with pytest.raises(ValueError, match="valid 64-char hex public_key"):
            gossip_engine.add_seed_node("127.0.0.1:8001", "seed1", public_key="")

    def test_add_seed_node_rejects_short_key(self, gossip_engine):
        """add_seed_node should reject short public keys."""
        with pytest.raises(ValueError, match="valid 64-char hex public_key"):
            gossip_engine.add_seed_node("127.0.0.1:8001", "seed1", public_key="abc")

    def test_add_seed_node_accepts_valid_key(self, gossip_engine, peer_keypair):
        """add_seed_node should accept valid 64-char public key."""
        _, peer_pub = peer_keypair
        gossip_engine.add_seed_node("127.0.0.1:8001", "seed1", public_key=peer_pub)

        assert "seed1" in gossip_engine.peers
        assert gossip_engine.peers["seed1"].public_key == peer_pub


class TestGossipMessageSigning:
    """Tests for Ed25519 message signing (SEC-016)."""

    def test_message_signed_when_private_key_provided(self, gossip_engine):
        """Messages should be signed when private key is available."""
        msg = gossip_engine._create_message(MessageType.PING, {})

        assert msg.signature != ""
        assert len(msg.signature) == 128  # 64 bytes = 128 hex chars

    def test_message_not_signed_without_private_key(self, keypair):
        """Messages should not be signed without private key."""
        _, pub = keypair
        engine = GossipEngine(
            node_id="unsigned_node",
            address="127.0.0.1:7778",
            public_key=pub,
            private_key="",  # No private key
        )

        msg = engine._create_message(MessageType.PING, {})

        assert msg.signature == ""

    def test_signature_verifies_with_correct_key(self, gossip_engine, keypair):
        """Signature should verify with correct public key."""
        _, pub = keypair
        msg = gossip_engine._create_message(MessageType.ANNOUNCE, {"test": "data"})

        assert msg.verify_signature(pub) is True

    def test_signature_fails_with_wrong_key(self, gossip_engine, peer_keypair):
        """Signature should fail with wrong public key."""
        _, wrong_pub = peer_keypair
        msg = gossip_engine._create_message(MessageType.ANNOUNCE, {"test": "data"})

        assert msg.verify_signature(wrong_pub) is False

    def test_tampered_message_fails_verification(self, gossip_engine, keypair):
        """Tampered message should fail signature verification."""
        _, pub = keypair
        msg = gossip_engine._create_message(MessageType.PING, {"original": True})

        # Tamper with payload after signing
        msg.payload["tampered"] = True

        assert msg.verify_signature(pub) is False


class TestMessageAuthentication:
    """Tests for message authentication on receive (SEC-016)."""

    @pytest.mark.asyncio
    async def test_signed_message_accepted(self, gossip_engine, peer_keypair):
        """Properly signed message from known peer should be accepted."""
        peer_priv, peer_pub = peer_keypair

        # Add peer to known nodes
        gossip_engine.add_seed_node("127.0.0.1:8002", "peer1", public_key=peer_pub)

        # Create peer's engine to sign message
        peer_engine = GossipEngine(
            node_id="peer1",
            address="127.0.0.1:8002",
            public_key=peer_pub,
            private_key=peer_priv,
        )

        ping_msg = peer_engine._create_message(MessageType.PING, {})
        msg_bytes = ping_msg.to_bytes()

        response = await gossip_engine.handle_message(msg_bytes)

        # Should get PING_ACK response (message was processed)
        assert response is not None
        response_msg = GossipMessage.from_bytes(response)
        assert response_msg.msg_type == MessageType.PING_ACK

    @pytest.mark.asyncio
    async def test_unsigned_message_rejected(self, gossip_engine, peer_keypair):
        """Unsigned message should be silently rejected."""
        _, peer_pub = peer_keypair

        # Add peer
        gossip_engine.add_seed_node("127.0.0.1:8003", "peer2", public_key=peer_pub)

        # Create unsigned message (no private key)
        unsigned_engine = GossipEngine(
            node_id="peer2",
            address="127.0.0.1:8003",
            public_key=peer_pub,
            private_key="",  # No signing
        )

        ping_msg = unsigned_engine._create_message(MessageType.PING, {})
        msg_bytes = ping_msg.to_bytes()

        response = await gossip_engine.handle_message(msg_bytes)

        # Should be rejected (None response)
        assert response is None

    @pytest.mark.asyncio
    async def test_wrong_signature_rejected(self, gossip_engine, keypair, peer_keypair):
        """Message signed with wrong key should be rejected."""
        _, peer_pub = peer_keypair
        wrong_priv, _ = keypair  # Use different keypair

        # Add peer
        gossip_engine.add_seed_node("127.0.0.1:8004", "peer3", public_key=peer_pub)

        # Create message signed with wrong key
        wrong_engine = GossipEngine(
            node_id="peer3",
            address="127.0.0.1:8004",
            public_key=peer_pub,
            private_key=wrong_priv,  # Wrong key!
        )

        ping_msg = wrong_engine._create_message(MessageType.PING, {})
        msg_bytes = ping_msg.to_bytes()

        response = await gossip_engine.handle_message(msg_bytes)

        # Should be rejected
        assert response is None


class TestCachePoisoningPrevention:
    """Tests for cache poisoning DoS prevention.

    SEC-016: Signature must be verified BEFORE deduplication check.
    Otherwise, attacker can send unsigned message with valid ID to
    poison the cache and cause legitimate signed message to be rejected.
    """

    @pytest.mark.asyncio
    async def test_unsigned_message_does_not_pollute_cache(self, gossip_engine, peer_keypair):
        """Unsigned messages should NOT be added to seen_messages cache."""
        peer_priv, peer_pub = peer_keypair

        # Add peer
        gossip_engine.add_seed_node("127.0.0.1:8005", "peer4", public_key=peer_pub)

        # Send unsigned message first
        unsigned_engine = GossipEngine(
            node_id="peer4",
            address="127.0.0.1:8005",
            public_key=peer_pub,
            private_key="",
        )
        unsigned_msg = unsigned_engine._create_message(MessageType.PING, {})
        sequence = unsigned_msg.sequence

        await gossip_engine.handle_message(unsigned_msg.to_bytes())

        # Verify it was NOT added to seen cache
        message_id = f"peer4:{sequence}"
        assert message_id not in gossip_engine._seen_messages

    @pytest.mark.asyncio
    async def test_signed_message_added_to_cache_after_verification(self, gossip_engine, peer_keypair):
        """Signed messages should be added to cache AFTER verification."""
        peer_priv, peer_pub = peer_keypair

        # Add peer
        gossip_engine.add_seed_node("127.0.0.1:8006", "peer5", public_key=peer_pub)

        # Send properly signed message
        signed_engine = GossipEngine(
            node_id="peer5",
            address="127.0.0.1:8006",
            public_key=peer_pub,
            private_key=peer_priv,
        )
        signed_msg = signed_engine._create_message(MessageType.PING, {})
        sequence = signed_msg.sequence

        await gossip_engine.handle_message(signed_msg.to_bytes())

        # Verify it WAS added to seen cache
        message_id = f"peer5:{sequence}"
        assert message_id in gossip_engine._seen_messages

    @pytest.mark.asyncio
    async def test_cache_poisoning_attack_fails(self, gossip_engine, peer_keypair):
        """Cache poisoning attack should fail - legitimate message still processed."""
        peer_priv, peer_pub = peer_keypair

        # Add peer
        gossip_engine.add_seed_node("127.0.0.1:8007", "victim", public_key=peer_pub)

        # Step 1: Attacker sends unsigned message with forged sequence
        attacker_engine = GossipEngine(
            node_id="victim",  # Pretend to be victim
            address="127.0.0.1:8007",
            public_key=peer_pub,
            private_key="",  # No valid signature
        )
        attacker_msg = attacker_engine._create_message(MessageType.PING, {})
        attack_sequence = attacker_msg.sequence

        await gossip_engine.handle_message(attacker_msg.to_bytes())

        # Step 2: Real victim sends properly signed message with same sequence
        victim_engine = GossipEngine(
            node_id="victim",
            address="127.0.0.1:8007",
            public_key=peer_pub,
            private_key=peer_priv,
        )
        # Force same sequence
        victim_engine.sequence = attack_sequence - 1
        victim_msg = victim_engine._create_message(MessageType.PING, {})

        response = await gossip_engine.handle_message(victim_msg.to_bytes())

        # Real message should still be processed (cache wasn't poisoned)
        assert response is not None


class TestNewNodeAnnouncement:
    """Tests for new node discovery with public key validation."""

    @pytest.mark.asyncio
    async def test_new_node_with_valid_pubkey_added(self, gossip_engine, peer_keypair):
        """New node with valid public key in payload should be added."""
        peer_priv, peer_pub = peer_keypair

        # New node announces itself
        new_engine = GossipEngine(
            node_id="new_node",
            address="127.0.0.1:8010",
            public_key=peer_pub,
            private_key=peer_priv,
        )

        announce_msg = new_engine._create_message(
            MessageType.ANNOUNCE,
            {"public_key": peer_pub, "ihsan_average": 0.96}
        )

        await gossip_engine.handle_message(announce_msg.to_bytes())

        # New node should be in peers
        assert "new_node" in gossip_engine.peers
        assert gossip_engine.peers["new_node"].public_key == peer_pub

    @pytest.mark.asyncio
    async def test_new_node_without_pubkey_not_added(self, gossip_engine, peer_keypair):
        """New node without public key should not be added."""
        peer_priv, peer_pub = peer_keypair

        # New node announces but public_key validation fails in _add_peer_from_message
        # We'll simulate by sending a message from unknown sender
        msg = GossipMessage(
            msg_type=MessageType.ANNOUNCE,
            sender_id="anon_node",
            sender_address="127.0.0.1:8011",
            sequence=1,
            timestamp="2026-02-01T00:00:00Z",
            payload={},  # No public_key!
            piggyback=[],
            signature="",
        )

        # Sign with valid key but don't include public_key in payload
        from core.pci.crypto import sign_message, domain_separated_digest, canonical_json
        digest = domain_separated_digest(canonical_json(msg._signable_dict()))
        msg.signature = sign_message(digest, peer_priv)

        await gossip_engine.handle_message(msg.to_bytes())

        # Should NOT be added due to missing/invalid public_key
        assert "anon_node" not in gossip_engine.peers


class TestMessageDomainSeparation:
    """Tests for domain-separated signing."""

    def test_signable_dict_excludes_signature(self, gossip_engine):
        """_signable_dict should exclude signature field."""
        msg = gossip_engine._create_message(MessageType.PING, {"data": 123})

        signable = msg._signable_dict()

        assert "signature" not in signable
        assert "msg_type" in signable
        assert "sender_id" in signable

    def test_signature_covers_all_fields(self, gossip_engine, keypair):
        """Signature should cover all message fields."""
        _, pub = keypair
        msg = gossip_engine._create_message(MessageType.PING, {"critical": True})

        # Tamper with each field and verify signature fails
        fields_to_tamper = [
            ("sender_id", "attacker"),
            ("sequence", msg.sequence + 1),
            ("payload", {"tampered": True}),
        ]

        for field_name, tampered_value in fields_to_tamper:
            original = getattr(msg, field_name)
            setattr(msg, field_name, tampered_value)

            assert msg.verify_signature(pub) is False, \
                f"Tampering with {field_name} should invalidate signature"

            setattr(msg, field_name, original)  # Restore


class TestPublicKeyInPayload:
    """Tests for public key propagation in messages."""

    def test_messages_include_sender_pubkey(self, gossip_engine, keypair):
        """All messages should include sender's public key in payload."""
        _, pub = keypair
        msg = gossip_engine._create_message(MessageType.PING, {})

        assert "public_key" in msg.payload
        assert msg.payload["public_key"] == pub

    def test_user_payload_not_overwritten(self, gossip_engine, keypair):
        """User-provided public_key in payload should not be overwritten."""
        _, pub = keypair
        custom_key = "a" * 64

        msg = gossip_engine._create_message(
            MessageType.ANNOUNCE,
            {"public_key": custom_key}  # User provides key
        )

        # Should preserve user's key (though this is unusual)
        assert msg.payload["public_key"] == custom_key


class TestGossipEngineLifecycle:
    """Tests for engine start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_leave_message_signed(self, gossip_engine):
        """Leave message should be properly signed."""
        leave_bytes = gossip_engine.create_leave_message()
        leave_msg = GossipMessage.from_bytes(leave_bytes)

        assert leave_msg.msg_type == MessageType.LEAVE
        assert leave_msg.signature != ""

    def test_announce_message_signed(self, gossip_engine):
        """Announce message should be properly signed."""
        announce_bytes = gossip_engine.create_announce_message()
        announce_msg = GossipMessage.from_bytes(announce_bytes)

        assert announce_msg.msg_type == MessageType.ANNOUNCE
        assert announce_msg.signature != ""


class TestPiggybackedUpdateSecurity:
    """Tests for piggybacked state update validation."""

    @pytest.mark.asyncio
    async def test_piggyback_with_invalid_pubkey_rejected(self, gossip_engine, peer_keypair):
        """Piggybacked node info without valid pubkey should be ignored."""
        peer_priv, peer_pub = peer_keypair

        # Add legitimate peer
        gossip_engine.add_seed_node("127.0.0.1:8020", "legit_peer", public_key=peer_pub)

        # Peer sends message with piggybacked invalid node
        peer_engine = GossipEngine(
            node_id="legit_peer",
            address="127.0.0.1:8020",
            public_key=peer_pub,
            private_key=peer_priv,
        )

        # Create message with invalid piggyback
        msg = peer_engine._create_message(MessageType.PING, {})
        msg.piggyback = [
            {
                "node_id": "fake_node",
                "address": "127.0.0.1:9999",
                "public_key": "short",  # Invalid!
                "state": "ALIVE",
                "incarnation": 1,
            }
        ]
        # Re-sign after modification
        msg.sign(peer_priv)

        await gossip_engine.handle_message(msg.to_bytes())

        # Fake node should NOT be in peers
        assert "fake_node" not in gossip_engine.peers


class TestStatistics:
    """Tests for gossip engine statistics."""

    def test_stats_include_all_fields(self, gossip_engine):
        """Stats should include all relevant fields."""
        stats = gossip_engine.get_stats()

        assert "node_id" in stats
        assert "network_size" in stats
        assert "alive_peers" in stats
        assert "network_multiplier" in stats
        assert "sequence" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
