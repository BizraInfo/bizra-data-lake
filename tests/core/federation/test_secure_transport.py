"""
BIZRA Secure Transport Layer Test Suite (P0-2)

Comprehensive tests for DTLS/Noise transport layer security implementation.
Target: 80% coverage of security-critical paths

Test Categories:
1. Key exchange and handshake flows
2. Message encryption/decryption
3. Replay attack protection
4. Session management
5. Error handling and edge cases
6. Integration with gossip protocol
"""

import pytest
import asyncio
import sys
import time
import struct
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from core.federation.secure_transport import (
    # Error types
    SecureTransportError,
    HandshakeError,
    DecryptionError,
    ReplayError,
    SessionError,
    NonceExhaustionError,
    # Data structures
    CipherState,
    SymmetricState,
    ReplayWindow,
    SecureSession,
    HandshakeState,
    MessageType,
    # Transports
    NoiseTransport,
    DTLSTransport,
    SecureTransportManager,
    # Factory
    create_secure_gossip_transport,
    # Constants
    SESSION_TIMEOUT_SECONDS,
    REPLAY_WINDOW_SIZE,
    MAX_NONCE_VALUE,
)
from core.pci.crypto import generate_keypair


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def keypair():
    """Generate Ed25519 keypair for testing."""
    return generate_keypair()


@pytest.fixture
def peer_keypair():
    """Generate second keypair for peer testing."""
    return generate_keypair()


@pytest.fixture
def cipher_key():
    """Generate random 32-byte key for cipher tests."""
    return os.urandom(32)


@pytest.fixture
def noise_transport(keypair):
    """Create NoiseTransport instance."""
    priv, pub = keypair
    return NoiseTransport(
        static_private_key=bytes.fromhex(priv),
        static_public_key=bytes.fromhex(pub),
        node_id="test_node"
    )


@pytest.fixture
def noise_transport_peer(peer_keypair):
    """Create NoiseTransport instance for peer."""
    priv, pub = peer_keypair
    return NoiseTransport(
        static_private_key=bytes.fromhex(priv),
        static_public_key=bytes.fromhex(pub),
        node_id="peer_node"
    )


@pytest.fixture
def dtls_transport(keypair):
    """Create DTLSTransport instance."""
    priv, pub = keypair
    return DTLSTransport(
        static_private_key=bytes.fromhex(priv),
        static_public_key=bytes.fromhex(pub),
        node_id="test_node"
    )


@pytest.fixture
def dtls_transport_peer(peer_keypair):
    """Create DTLSTransport instance for peer."""
    priv, pub = peer_keypair
    return DTLSTransport(
        static_private_key=bytes.fromhex(priv),
        static_public_key=bytes.fromhex(pub),
        node_id="peer_node"
    )


@pytest.fixture
def transport_manager(keypair):
    """Create SecureTransportManager with Noise protocol."""
    priv, pub = keypair
    return SecureTransportManager(
        static_private_key=priv,
        static_public_key=pub,
        node_id="test_node",
        transport_type="noise"
    )


@pytest.fixture
def transport_manager_peer(peer_keypair):
    """Create SecureTransportManager for peer."""
    priv, pub = peer_keypair
    return SecureTransportManager(
        static_private_key=priv,
        static_public_key=pub,
        node_id="peer_node",
        transport_type="noise"
    )


# =============================================================================
# CIPHER STATE TESTS
# =============================================================================

class TestCipherState:
    """Tests for CipherState symmetric encryption."""

    def test_encrypt_decrypt_roundtrip(self, cipher_key):
        """Encrypt then decrypt should return original plaintext."""
        cipher = CipherState(key=cipher_key)
        plaintext = b"Hello, BIZRA federation!"

        ciphertext = cipher.encrypt(plaintext)

        # Decrypt with nonce 0 (first encryption)
        decrypt_cipher = CipherState(key=cipher_key)
        decrypted = decrypt_cipher.decrypt(ciphertext, nonce=0)

        assert decrypted == plaintext

    def test_nonce_increments(self, cipher_key):
        """Nonce should increment after each encryption."""
        cipher = CipherState(key=cipher_key)

        assert cipher.nonce == 0
        cipher.encrypt(b"message 1")
        assert cipher.nonce == 1
        cipher.encrypt(b"message 2")
        assert cipher.nonce == 2

    def test_different_nonces_produce_different_ciphertext(self, cipher_key):
        """Same plaintext with different nonces should produce different ciphertext."""
        cipher1 = CipherState(key=cipher_key)
        cipher2 = CipherState(key=cipher_key)

        plaintext = b"same message"

        ct1 = cipher1.encrypt(plaintext)
        cipher2.nonce = 1  # Skip to nonce 1
        ct2 = cipher2.encrypt(plaintext)

        assert ct1 != ct2

    def test_decrypt_wrong_nonce_fails(self, cipher_key):
        """Decryption with wrong nonce should fail."""
        cipher = CipherState(key=cipher_key)
        ciphertext = cipher.encrypt(b"secret data")

        decrypt_cipher = CipherState(key=cipher_key)

        with pytest.raises(DecryptionError):
            decrypt_cipher.decrypt(ciphertext, nonce=999)  # Wrong nonce

    def test_decrypt_wrong_key_fails(self, cipher_key):
        """Decryption with wrong key should fail."""
        cipher = CipherState(key=cipher_key)
        ciphertext = cipher.encrypt(b"secret data")

        wrong_key = os.urandom(32)
        decrypt_cipher = CipherState(key=wrong_key)

        with pytest.raises(DecryptionError):
            decrypt_cipher.decrypt(ciphertext, nonce=0)

    def test_associated_data_authentication(self, cipher_key):
        """Associated data should be authenticated."""
        cipher = CipherState(key=cipher_key)
        plaintext = b"message"
        ad = b"metadata"

        ciphertext = cipher.encrypt(plaintext, associated_data=ad)

        # Decrypt with correct AD
        decrypt_cipher = CipherState(key=cipher_key)
        decrypted = decrypt_cipher.decrypt(ciphertext, nonce=0, associated_data=ad)
        assert decrypted == plaintext

        # Decrypt with wrong AD should fail
        with pytest.raises(DecryptionError):
            decrypt_cipher.decrypt(ciphertext, nonce=0, associated_data=b"wrong")

    def test_nonce_exhaustion_raises_error(self, cipher_key):
        """Should raise NonceExhaustionError when nonce space exhausted."""
        cipher = CipherState(key=cipher_key)
        cipher.nonce = MAX_NONCE_VALUE  # At limit

        with pytest.raises(NonceExhaustionError):
            cipher.encrypt(b"data")


# =============================================================================
# SYMMETRIC STATE TESTS
# =============================================================================

class TestSymmetricState:
    """Tests for SymmetricState handshake operations."""

    def test_initialize_with_protocol_name(self):
        """Should initialize with protocol name."""
        state = SymmetricState.initialize(b"Noise_XX_test")

        assert len(state.ck) == 32
        assert len(state.h) == 32

    def test_mix_hash_updates_hash(self):
        """mix_hash should update the handshake hash."""
        state = SymmetricState.initialize(b"test")
        original_h = state.h

        state.mix_hash(b"new data")

        assert state.h != original_h
        assert len(state.h) == 32

    def test_mix_key_updates_chaining_key(self):
        """mix_key should update chaining key and return temp key."""
        state = SymmetricState.initialize(b"test")
        original_ck = state.ck

        ikm = os.urandom(32)
        temp_key = state.mix_key(ikm)

        assert state.ck != original_ck
        assert len(temp_key) == 32

    def test_split_produces_two_cipher_states(self):
        """split should produce two independent cipher states."""
        state = SymmetricState.initialize(b"test")
        state.mix_key(os.urandom(32))  # Need to mix some key material

        initiator_cipher, responder_cipher = state.split()

        assert isinstance(initiator_cipher, CipherState)
        assert isinstance(responder_cipher, CipherState)
        assert initiator_cipher.key != responder_cipher.key

    def test_split_keys_are_deterministic(self):
        """Same state should produce same keys when split."""
        ikm = os.urandom(32)

        state1 = SymmetricState.initialize(b"test")
        state1.mix_key(ikm)
        c1_init, c1_resp = state1.split()

        state2 = SymmetricState.initialize(b"test")
        state2.mix_key(ikm)
        c2_init, c2_resp = state2.split()

        assert c1_init.key == c2_init.key
        assert c1_resp.key == c2_resp.key


# =============================================================================
# REPLAY WINDOW TESTS
# =============================================================================

class TestReplayWindow:
    """Tests for replay attack protection."""

    def test_new_nonce_accepted(self):
        """New nonces should be accepted."""
        window = ReplayWindow()

        assert window.check_and_update(1) is True
        assert window.check_and_update(2) is True
        assert window.check_and_update(3) is True

    def test_duplicate_nonce_rejected(self):
        """Duplicate nonces should be rejected."""
        window = ReplayWindow()

        assert window.check_and_update(5) is True
        assert window.check_and_update(5) is False  # Duplicate

    def test_old_nonces_in_window_tracked(self):
        """Old nonces within window should be trackable."""
        window = ReplayWindow()

        # Advance to nonce 10
        for i in range(1, 11):
            assert window.check_and_update(i) is True

        # Try nonce 5 again (should be rejected, within window)
        assert window.check_and_update(5) is False

    def test_nonces_outside_window_rejected(self):
        """Nonces older than window size should be rejected."""
        window = ReplayWindow()

        # Advance well beyond window
        window.check_and_update(1)
        window.check_and_update(REPLAY_WINDOW_SIZE + 10)

        # Nonce 1 is now outside window
        assert window.check_and_update(1) is False

    def test_out_of_order_nonces_in_window_accepted(self):
        """Out-of-order nonces within window should be accepted."""
        window = ReplayWindow()

        window.check_and_update(10)
        assert window.check_and_update(8) is True  # Out of order but in window
        assert window.check_and_update(9) is True  # Out of order but in window
        assert window.check_and_update(8) is False  # Now a duplicate

    def test_window_slides_forward(self):
        """Window should slide forward as higher nonces arrive."""
        window = ReplayWindow()

        window.check_and_update(1)
        assert window.highest_seen == 1

        window.check_and_update(100)
        assert window.highest_seen == 100


# =============================================================================
# SECURE SESSION TESTS
# =============================================================================

class TestSecureSession:
    """Tests for SecureSession management."""

    def test_session_creation(self, cipher_key):
        """Session should be created with valid parameters."""
        session = SecureSession(
            session_id="test_session",
            peer_id="peer_001",
            peer_static_public=os.urandom(32),
            send_cipher=CipherState(key=cipher_key),
            recv_cipher=CipherState(key=os.urandom(32)),
        )

        assert session.session_id == "test_session"
        assert session.messages_sent == 0
        assert session.messages_received == 0

    def test_session_not_expired_initially(self, cipher_key):
        """New session should not be expired."""
        session = SecureSession(
            session_id="test",
            peer_id="peer",
            peer_static_public=os.urandom(32),
            send_cipher=CipherState(key=cipher_key),
            recv_cipher=CipherState(key=os.urandom(32)),
        )

        assert session.is_expired is False

    def test_session_expired_after_timeout(self, cipher_key):
        """Session should be expired after timeout."""
        session = SecureSession(
            session_id="test",
            peer_id="peer",
            peer_static_public=os.urandom(32),
            send_cipher=CipherState(key=cipher_key),
            recv_cipher=CipherState(key=os.urandom(32)),
        )

        # Artificially age the session
        session.created_at = time.time() - SESSION_TIMEOUT_SECONDS - 1

        assert session.is_expired is True

    def test_touch_updates_last_activity(self, cipher_key):
        """touch() should update last_activity timestamp."""
        session = SecureSession(
            session_id="test",
            peer_id="peer",
            peer_static_public=os.urandom(32),
            send_cipher=CipherState(key=cipher_key),
            recv_cipher=CipherState(key=os.urandom(32)),
        )

        original_activity = session.last_activity
        time.sleep(0.01)
        session.touch()

        assert session.last_activity > original_activity


# =============================================================================
# NOISE TRANSPORT HANDSHAKE TESTS
# =============================================================================

class TestNoiseTransportHandshake:
    """Tests for Noise_XX handshake protocol."""

    def test_create_handshake_init(self, noise_transport):
        """Should create valid handshake init message."""
        msg, state = noise_transport.create_handshake_init()

        assert msg[0] == MessageType.HANDSHAKE_INIT
        assert len(msg) > 32  # Should contain ephemeral public key
        assert 'e_private' in state
        assert 'e_public' in state
        assert state['phase'] == 'init_sent'

    @pytest.mark.asyncio
    async def test_handshake_responder(self, noise_transport, noise_transport_peer):
        """Responder should process init and create response."""
        # Initiator creates init
        init_msg, init_state = noise_transport.create_handshake_init()

        # Responder processes init
        session, response = await noise_transport_peer.handshake_responder(
            init_msg, "127.0.0.1:8001"
        )

        assert response is not None
        assert response[0] == MessageType.HANDSHAKE_RESPONSE

    def test_full_handshake_flow(self, noise_transport, noise_transport_peer):
        """Complete handshake should establish sessions for both parties."""
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        # Step 1: Initiator creates init message
        init_msg, init_state = noise_transport.create_handshake_init()

        # Step 2: Responder processes init, creates response
        import asyncio
        loop = asyncio.new_event_loop()
        _, response = loop.run_until_complete(
            noise_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        # Step 3: Initiator processes response, creates final
        session_initiator, final_msg = noise_transport.process_handshake_response(
            response, init_state, peer_addr
        )

        assert session_initiator is not None
        assert final_msg is not None
        assert final_msg[0] == MessageType.HANDSHAKE_FINAL

        # Step 4: Responder processes final
        session_responder = noise_transport_peer.process_handshake_final(
            final_msg, initiator_addr
        )

        assert session_responder is not None

    def test_handshake_init_too_short_rejected(self, noise_transport):
        """Handshake init that's too short should be rejected."""
        short_msg = struct.pack('!B', MessageType.HANDSHAKE_INIT) + b'short'

        import asyncio
        loop = asyncio.new_event_loop()
        with pytest.raises(HandshakeError, match="too short"):
            loop.run_until_complete(
                noise_transport.handshake_responder(short_msg, "127.0.0.1:8001")
            )
        loop.close()

    def test_wrong_message_type_rejected(self, noise_transport):
        """Wrong message type should be rejected."""
        wrong_msg = struct.pack('!B', MessageType.APPLICATION_DATA) + os.urandom(32)

        import asyncio
        loop = asyncio.new_event_loop()
        with pytest.raises(HandshakeError, match="Expected HANDSHAKE_INIT"):
            loop.run_until_complete(
                noise_transport.handshake_responder(wrong_msg, "127.0.0.1:8001")
            )
        loop.close()


# =============================================================================
# NOISE TRANSPORT ENCRYPTION TESTS
# =============================================================================

class TestNoiseTransportEncryption:
    """Tests for Noise transport message encryption."""

    def test_encrypt_decrypt_roundtrip(self, noise_transport, noise_transport_peer):
        """Encrypted message should decrypt correctly."""
        # Establish session via handshake
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        init_msg, init_state = noise_transport.create_handshake_init()

        import asyncio
        loop = asyncio.new_event_loop()
        _, response = loop.run_until_complete(
            noise_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        session_init, final_msg = noise_transport.process_handshake_response(
            response, init_state, peer_addr
        )
        session_resp = noise_transport_peer.process_handshake_final(
            final_msg, initiator_addr
        )

        # Initiator sends message
        plaintext = b"Hello from initiator!"
        encrypted = noise_transport.encrypt(session_init, plaintext)

        # Responder decrypts
        decrypted = noise_transport_peer.decrypt(session_resp, encrypted)

        assert decrypted == plaintext

    def test_replay_attack_detected(self, noise_transport, noise_transport_peer):
        """Replay attack should be detected."""
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        # Establish session
        init_msg, init_state = noise_transport.create_handshake_init()

        import asyncio
        loop = asyncio.new_event_loop()
        _, response = loop.run_until_complete(
            noise_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        session_init, final_msg = noise_transport.process_handshake_response(
            response, init_state, peer_addr
        )
        session_resp = noise_transport_peer.process_handshake_final(
            final_msg, initiator_addr
        )

        # Send and decrypt a message
        encrypted = noise_transport.encrypt(session_init, b"message")
        noise_transport_peer.decrypt(session_resp, encrypted)

        # Try to replay the same message
        with pytest.raises(ReplayError):
            noise_transport_peer.decrypt(session_resp, encrypted)

    def test_tampered_message_rejected(self, noise_transport, noise_transport_peer):
        """Tampered message should be rejected."""
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        # Establish session
        init_msg, init_state = noise_transport.create_handshake_init()

        import asyncio
        loop = asyncio.new_event_loop()
        _, response = loop.run_until_complete(
            noise_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        session_init, final_msg = noise_transport.process_handshake_response(
            response, init_state, peer_addr
        )
        session_resp = noise_transport_peer.process_handshake_final(
            final_msg, initiator_addr
        )

        # Create and tamper with message
        encrypted = noise_transport.encrypt(session_init, b"secret")
        tampered = encrypted[:-1] + bytes([encrypted[-1] ^ 0xFF])  # Flip last byte

        with pytest.raises(DecryptionError):
            noise_transport_peer.decrypt(session_resp, tampered)

    def test_encrypted_message_format(self, noise_transport, noise_transport_peer):
        """Encrypted message should have correct format."""
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        # Establish session
        init_msg, init_state = noise_transport.create_handshake_init()

        import asyncio
        loop = asyncio.new_event_loop()
        _, response = loop.run_until_complete(
            noise_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        session_init, final_msg = noise_transport.process_handshake_response(
            response, init_state, peer_addr
        )

        encrypted = noise_transport.encrypt(session_init, b"test")

        # Check format: [type:1][nonce:8][ciphertext:N]
        assert encrypted[0] == MessageType.APPLICATION_DATA
        nonce = struct.unpack('>Q', encrypted[1:9])[0]
        assert nonce == 0  # First message


# =============================================================================
# DTLS TRANSPORT TESTS
# =============================================================================

class TestDTLSTransportHandshake:
    """Tests for DTLS handshake protocol."""

    def test_create_handshake_init(self, dtls_transport):
        """Should create valid DTLS ClientHello."""
        msg, state = dtls_transport.create_handshake_init()

        assert msg[0] == MessageType.HANDSHAKE_INIT
        assert 'client_random' in state
        assert 'e_private' in state

    @pytest.mark.asyncio
    async def test_handshake_responder(self, dtls_transport, dtls_transport_peer):
        """Responder should process ClientHello and create ServerHello."""
        init_msg, _ = dtls_transport.create_handshake_init()

        session, response = await dtls_transport_peer.handshake_responder(
            init_msg, "127.0.0.1:8001"
        )

        assert session is not None
        assert response is not None
        assert response[0] == MessageType.HANDSHAKE_RESPONSE

    def test_full_handshake_flow(self, dtls_transport, dtls_transport_peer):
        """Complete DTLS handshake should establish sessions."""
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        # Step 1: Client creates ClientHello
        init_msg, init_state = dtls_transport.create_handshake_init()

        # Step 2: Server processes and responds
        import asyncio
        loop = asyncio.new_event_loop()
        session_resp, response = loop.run_until_complete(
            dtls_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        # Step 3: Client processes ServerHello
        session_init, _ = dtls_transport.process_handshake_response(
            response, init_state, peer_addr
        )

        assert session_init is not None
        assert session_resp is not None

    def test_dtls_version_mismatch_rejected(self, dtls_transport):
        """Wrong DTLS version should be rejected."""
        # Create message with wrong version
        bad_version = struct.pack('!H', 0x0000)  # Invalid version
        wrong_msg = struct.pack('!B', MessageType.HANDSHAKE_INIT)
        wrong_msg += bad_version
        wrong_msg += os.urandom(32 + 1 + 1 + 32)  # Padding

        import asyncio
        loop = asyncio.new_event_loop()
        with pytest.raises(HandshakeError):
            loop.run_until_complete(
                dtls_transport.handshake_responder(wrong_msg, "127.0.0.1:8001")
            )
        loop.close()


class TestDTLSTransportEncryption:
    """Tests for DTLS transport encryption."""

    def test_encrypt_decrypt_roundtrip(self, dtls_transport, dtls_transport_peer):
        """DTLS encrypted message should decrypt correctly."""
        peer_addr = "127.0.0.1:8001"
        initiator_addr = "127.0.0.1:8000"

        # Establish session
        init_msg, init_state = dtls_transport.create_handshake_init()

        import asyncio
        loop = asyncio.new_event_loop()
        session_resp, response = loop.run_until_complete(
            dtls_transport_peer.handshake_responder(init_msg, initiator_addr)
        )
        loop.close()

        session_init, _ = dtls_transport.process_handshake_response(
            response, init_state, peer_addr
        )

        # Client sends message
        plaintext = b"DTLS secure message"
        encrypted = dtls_transport.encrypt(session_init, plaintext)

        # Server decrypts
        decrypted = dtls_transport_peer.decrypt(session_resp, encrypted)

        assert decrypted == plaintext


# =============================================================================
# TRANSPORT MANAGER TESTS
# =============================================================================

class TestSecureTransportManager:
    """Tests for SecureTransportManager high-level API."""

    def test_create_with_noise(self, keypair):
        """Should create manager with Noise transport."""
        priv, pub = keypair
        manager = SecureTransportManager(
            static_private_key=priv,
            static_public_key=pub,
            node_id="test",
            transport_type="noise"
        )

        assert manager.transport_type == "noise"

    def test_create_with_dtls(self, keypair):
        """Should create manager with DTLS transport."""
        priv, pub = keypair
        manager = SecureTransportManager(
            static_private_key=priv,
            static_public_key=pub,
            node_id="test",
            transport_type="dtls"
        )

        assert manager.transport_type == "dtls"

    def test_invalid_transport_type_rejected(self, keypair):
        """Invalid transport type should raise error."""
        priv, pub = keypair

        with pytest.raises(ValueError, match="Unknown transport type"):
            SecureTransportManager(
                static_private_key=priv,
                static_public_key=pub,
                node_id="test",
                transport_type="invalid"
            )

    def test_create_handshake(self, transport_manager):
        """Should create handshake init message."""
        msg = transport_manager.create_handshake(peer_address="127.0.0.1:8001")

        assert msg is not None
        assert msg[0] == MessageType.HANDSHAKE_INIT

    def test_no_session_encrypt_fails(self, transport_manager):
        """Encrypt should fail without established session."""
        with pytest.raises(SessionError):
            transport_manager.encrypt_message("127.0.0.1:8001", b"data")

    def test_has_session_false_initially(self, transport_manager):
        """has_session should return False initially."""
        assert transport_manager.has_session("127.0.0.1:8001") is False

    def test_get_stats(self, transport_manager):
        """Should return stats dictionary."""
        stats = transport_manager.get_stats()

        assert 'transport_type' in stats
        assert 'active_sessions' in stats
        assert 'pending_handshakes' in stats

    def test_cleanup_expired_sessions(self, transport_manager, cipher_key):
        """cleanup_expired should remove expired sessions."""
        # Manually add an expired session
        session = SecureSession(
            session_id="test",
            peer_id="127.0.0.1:8001",
            peer_static_public=os.urandom(32),
            send_cipher=CipherState(key=cipher_key),
            recv_cipher=CipherState(key=os.urandom(32)),
        )
        session.created_at = time.time() - SESSION_TIMEOUT_SECONDS - 1

        transport_manager._session_cache["127.0.0.1:8001"] = session

        removed = transport_manager.cleanup_expired()

        assert removed == 1
        assert "127.0.0.1:8001" not in transport_manager._session_cache


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestCreateSecureGossipTransport:
    """Tests for factory function."""

    def test_creates_noise_transport_by_default(self, keypair):
        """Should create Noise transport by default."""
        priv, pub = keypair
        manager = create_secure_gossip_transport(priv, pub, "node_001")

        assert manager.transport_type == "noise"

    def test_creates_dtls_transport(self, keypair):
        """Should create DTLS transport when specified."""
        priv, pub = keypair
        manager = create_secure_gossip_transport(
            priv, pub, "node_001", transport_type="dtls"
        )

        assert manager.transport_type == "dtls"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullIntegration:
    """End-to-end integration tests."""

    def test_full_noise_communication(self, keypair, peer_keypair):
        """Test complete Noise communication flow."""
        priv1, pub1 = keypair
        priv2, pub2 = peer_keypair

        manager1 = create_secure_gossip_transport(priv1, pub1, "node_1")
        manager2 = create_secure_gossip_transport(priv2, pub2, "node_2")

        addr1 = "127.0.0.1:8001"
        addr2 = "127.0.0.1:8002"

        # Node 1 initiates handshake
        init_msg = manager1.create_handshake(addr2)

        # Node 2 responds
        session2, response = manager2.process_handshake(init_msg, addr1)

        # Node 1 processes response
        if response:
            session1, final_msg = manager1.process_handshake(response, addr2)

            # Node 2 processes final if present
            if final_msg:
                manager2.process_handshake(final_msg, addr1)

        # Now both should have sessions
        assert manager1.has_session(addr2)
        assert manager2.has_session(addr1)

        # Test bidirectional communication
        msg1 = b"Hello from node 1"
        encrypted1 = manager1.encrypt_message(addr2, msg1)
        decrypted1 = manager2.decrypt_message(addr1, encrypted1)
        assert decrypted1 == msg1

        msg2 = b"Hello from node 2"
        encrypted2 = manager2.encrypt_message(addr1, msg2)
        decrypted2 = manager1.decrypt_message(addr2, encrypted2)
        assert decrypted2 == msg2

    def test_full_dtls_communication(self, keypair, peer_keypair):
        """Test complete DTLS communication flow."""
        priv1, pub1 = keypair
        priv2, pub2 = peer_keypair

        manager1 = create_secure_gossip_transport(
            priv1, pub1, "node_1", transport_type="dtls"
        )
        manager2 = create_secure_gossip_transport(
            priv2, pub2, "node_2", transport_type="dtls"
        )

        addr1 = "127.0.0.1:8001"
        addr2 = "127.0.0.1:8002"

        # Node 1 initiates
        init_msg = manager1.create_handshake(addr2)

        # Node 2 responds (DTLS completes in 2 messages)
        session2, response = manager2.process_handshake(init_msg, addr1)
        assert session2 is not None

        # Node 1 processes response
        session1, _ = manager1.process_handshake(response, addr2)
        assert session1 is not None

        # Test communication
        msg = b"DTLS test message"
        encrypted = manager1.encrypt_message(addr2, msg)
        decrypted = manager2.decrypt_message(addr1, encrypted)
        assert decrypted == msg

    def test_multiple_sessions(self, keypair, peer_keypair):
        """Test managing multiple concurrent sessions."""
        priv1, pub1 = keypair
        manager = create_secure_gossip_transport(priv1, pub1, "central_node")

        # Create multiple peer managers
        peers = []
        for i in range(5):
            peer_priv, peer_pub = generate_keypair()
            peers.append({
                'manager': create_secure_gossip_transport(
                    peer_priv, peer_pub, f"peer_{i}"
                ),
                'addr': f"127.0.0.1:{9000+i}"
            })

        central_addr = "127.0.0.1:8000"

        # Establish sessions with all peers
        for peer in peers:
            init_msg = manager.create_handshake(peer['addr'])
            _, response = peer['manager'].process_handshake(init_msg, central_addr)

            if response:
                session, final = manager.process_handshake(response, peer['addr'])
                if final:
                    peer['manager'].process_handshake(final, central_addr)

        # Verify all sessions established
        for peer in peers:
            assert manager.has_session(peer['addr'])

        stats = manager.get_stats()
        assert stats['active_sessions'] == 5


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance benchmarks (marked as slow)."""

    @pytest.mark.slow
    def test_encryption_throughput(self, keypair, peer_keypair):
        """Benchmark encryption throughput."""
        priv1, pub1 = keypair
        priv2, pub2 = peer_keypair

        manager1 = create_secure_gossip_transport(priv1, pub1, "node_1")
        manager2 = create_secure_gossip_transport(priv2, pub2, "node_2")

        addr1 = "127.0.0.1:8001"
        addr2 = "127.0.0.1:8002"

        # Establish session
        init_msg = manager1.create_handshake(addr2)
        _, response = manager2.process_handshake(init_msg, addr1)
        session1, final = manager1.process_handshake(response, addr2)
        if final:
            manager2.process_handshake(final, addr1)

        # Benchmark
        message = b"X" * 1024  # 1KB message
        iterations = 1000

        start = time.time()
        for _ in range(iterations):
            encrypted = manager1.encrypt_message(addr2, message)
        elapsed = time.time() - start

        throughput_mbps = (iterations * len(message) * 8) / (elapsed * 1_000_000)
        print(f"\nEncryption throughput: {throughput_mbps:.2f} Mbps")
        print(f"Messages per second: {iterations / elapsed:.0f}")

        # Should achieve reasonable throughput (>10 Mbps on modern hardware)
        assert throughput_mbps > 1  # Very conservative threshold

    @pytest.mark.slow
    def test_handshake_latency(self, keypair, peer_keypair):
        """Benchmark handshake latency."""
        priv1, pub1 = keypair
        priv2, pub2 = peer_keypair

        iterations = 100
        latencies = []

        for _ in range(iterations):
            manager1 = create_secure_gossip_transport(priv1, pub1, "node_1")
            manager2 = create_secure_gossip_transport(priv2, pub2, "node_2")

            addr1 = "127.0.0.1:8001"
            addr2 = "127.0.0.1:8002"

            start = time.time()

            init_msg = manager1.create_handshake(addr2)
            _, response = manager2.process_handshake(init_msg, addr1)
            session1, final = manager1.process_handshake(response, addr2)
            if final:
                manager2.process_handshake(final, addr1)

            latencies.append(time.time() - start)

        avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
        print(f"\nAverage handshake latency: {avg_latency_ms:.2f} ms")

        # Should complete handshake in reasonable time (<50ms on modern hardware)
        assert avg_latency_ms < 100  # Conservative threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
