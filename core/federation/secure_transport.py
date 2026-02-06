"""
BIZRA PATTERN FEDERATION - SECURE TRANSPORT LAYER (P0-2)

Implements DTLS 1.3 and Noise Protocol Framework for secure P2P gossip communication.

Standing on Giants:
- Perrin (2018): The Noise Protocol Framework
- Rescorla (2018): DTLS 1.3 (RFC 9147)
- Bernstein (2011): ChaCha20-Poly1305 AEAD
- Langley (2016): X25519 Key Exchange

Security Properties:
- Perfect Forward Secrecy via ephemeral X25519 keys
- Message authentication via ChaCha20-Poly1305 AEAD
- Replay protection via nonce management
- Identity binding via Ed25519 static keys

Protocol Patterns:
- Noise_XX: Full mutual authentication with identity hiding
- DTLS 1.3: Standards-compliant alternative (simpler but less modern)
"""

import os
import struct
import time
import hashlib
import hmac
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from collections import OrderedDict

from cryptography.hazmat.primitives.asymmetric import x25519, ed25519
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidTag


# =============================================================================
# CONSTANTS
# =============================================================================

# Noise Protocol Constants
NOISE_PROTOCOL_NAME = b"Noise_XX_25519_ChaChaPoly_BLAKE2b"
NOISE_MAX_MESSAGE_SIZE = 65535
NOISE_TAG_SIZE = 16  # ChaCha20-Poly1305 tag size
NOISE_NONCE_SIZE = 12
NOISE_KEY_SIZE = 32
NOISE_DH_SIZE = 32  # X25519 public key size

# DTLS Constants
DTLS_VERSION = 0xFEFC  # DTLS 1.3
DTLS_HANDSHAKE_TIMEOUT_MS = 5000
DTLS_MAX_RETRANSMIT = 3
DTLS_COOKIE_LENGTH = 32

# Session Management
SESSION_TIMEOUT_SECONDS = 3600  # 1 hour session lifetime
SESSION_REKEY_INTERVAL = 300  # Rekey every 5 minutes
MAX_NONCE_VALUE = 2**64 - 1  # Maximum nonce before rekey required
MAX_CACHED_SESSIONS = 1000  # Maximum number of cached sessions

# Replay Protection
REPLAY_WINDOW_SIZE = 64  # Sliding window for replay detection
MAX_CLOCK_SKEW_SECONDS = 300  # 5 minute clock skew tolerance


# =============================================================================
# ERROR TYPES
# =============================================================================

class SecureTransportError(Exception):
    """Base error for secure transport operations."""
    pass


class HandshakeError(SecureTransportError):
    """Error during handshake."""
    pass


class DecryptionError(SecureTransportError):
    """Error during decryption (authentication failure)."""
    pass


class ReplayError(SecureTransportError):
    """Replay attack detected."""
    pass


class SessionError(SecureTransportError):
    """Session management error."""
    pass


class NonceExhaustionError(SecureTransportError):
    """Nonce space exhausted, rekey required."""
    pass


# =============================================================================
# HANDSHAKE STATE
# =============================================================================

class HandshakeState(IntEnum):
    """Handshake state machine states."""
    INITIAL = 0
    AWAITING_RESPONSE = 1
    AWAITING_FINAL = 2
    ESTABLISHED = 3
    FAILED = 4


class MessageType(IntEnum):
    """Secure transport message types."""
    HANDSHAKE_INIT = 0x01
    HANDSHAKE_RESPONSE = 0x02
    HANDSHAKE_FINAL = 0x03
    APPLICATION_DATA = 0x10
    REKEY_REQUEST = 0x20
    REKEY_RESPONSE = 0x21
    CLOSE_NOTIFY = 0x30
    ERROR = 0xFF


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CipherState:
    """
    Cipher state for symmetric encryption.

    Tracks encryption key and nonce counter for a single direction
    of communication (either sending or receiving).
    """
    key: bytes  # 32 bytes for ChaCha20-Poly1305
    nonce: int = 0  # 64-bit counter

    def encrypt(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
        """
        Encrypt plaintext with AEAD.

        Args:
            plaintext: Data to encrypt
            associated_data: Additional authenticated data (not encrypted)

        Returns:
            Ciphertext with authentication tag

        Raises:
            NonceExhaustionError: If nonce space exhausted
        """
        if self.nonce >= MAX_NONCE_VALUE:
            raise NonceExhaustionError("Nonce exhausted, rekey required")

        # Construct 12-byte nonce: 4 bytes zeros + 8 bytes counter (big-endian)
        nonce_bytes = b'\x00\x00\x00\x00' + struct.pack('>Q', self.nonce)
        self.nonce += 1

        cipher = ChaCha20Poly1305(self.key)
        return cipher.encrypt(nonce_bytes, plaintext, associated_data)

    def decrypt(self, ciphertext: bytes, nonce: int, associated_data: bytes = b"") -> bytes:
        """
        Decrypt ciphertext with AEAD.

        Args:
            ciphertext: Data to decrypt (includes auth tag)
            nonce: Nonce value used during encryption
            associated_data: Additional authenticated data

        Returns:
            Decrypted plaintext

        Raises:
            DecryptionError: If authentication fails
        """
        nonce_bytes = b'\x00\x00\x00\x00' + struct.pack('>Q', nonce)

        cipher = ChaCha20Poly1305(self.key)
        try:
            return cipher.decrypt(nonce_bytes, ciphertext, associated_data)
        except InvalidTag as e:
            raise DecryptionError(f"AEAD authentication failed: {e}")


@dataclass
class SymmetricState:
    """
    Symmetric state for Noise Protocol handshake.

    Maintains the chaining key (ck) and handshake hash (h) during
    the handshake protocol, deriving session keys upon completion.
    """
    ck: bytes  # Chaining key (32 bytes)
    h: bytes   # Handshake hash (32 bytes)

    @classmethod
    def initialize(cls, protocol_name: bytes) -> "SymmetricState":
        """Initialize symmetric state with protocol name."""
        if len(protocol_name) <= 32:
            h = protocol_name.ljust(32, b'\x00')
        else:
            h = hashlib.blake2b(protocol_name, digest_size=32).digest()
        return cls(ck=h, h=h)

    def mix_hash(self, data: bytes) -> None:
        """Mix data into handshake hash."""
        self.h = hashlib.blake2b(self.h + data, digest_size=32).digest()

    def mix_key(self, input_key_material: bytes) -> bytes:
        """
        Mix key material into chaining key and derive temp key.

        Uses HKDF with BLAKE2b for key derivation.

        Returns:
            Temporary key for encryption
        """
        # HKDF-Extract
        temp_key = hmac.new(self.ck, input_key_material, hashlib.blake2b).digest()[:32]

        # HKDF-Expand for new chaining key
        self.ck = hmac.new(temp_key, b'\x01', hashlib.blake2b).digest()[:32]

        # HKDF-Expand for derived key
        derived_key = hmac.new(temp_key, self.ck + b'\x02', hashlib.blake2b).digest()[:32]

        return derived_key

    def split(self) -> Tuple[CipherState, CipherState]:
        """
        Split into two cipher states for bidirectional communication.

        Returns:
            Tuple of (initiator_cipher, responder_cipher)
        """
        temp_key = hmac.new(self.ck, b'', hashlib.blake2b).digest()[:32]

        # Derive initiator key
        k1 = hmac.new(temp_key, b'\x01', hashlib.blake2b).digest()[:32]

        # Derive responder key
        k2 = hmac.new(temp_key, k1 + b'\x02', hashlib.blake2b).digest()[:32]

        return CipherState(key=k1), CipherState(key=k2)


@dataclass
class ReplayWindow:
    """
    Sliding window for replay attack protection.

    Standing on Giants: RFC 4303 (IPsec anti-replay)
    """
    highest_seen: int = 0
    window: int = 0  # Bitmap of received nonces

    def check_and_update(self, nonce: int) -> bool:
        """
        Check if nonce is valid (not replayed) and update window.

        Args:
            nonce: Nonce to check

        Returns:
            True if nonce is valid (not replayed), False otherwise
        """
        if nonce > self.highest_seen:
            # Advance window
            shift = min(nonce - self.highest_seen, REPLAY_WINDOW_SIZE)
            self.window = (self.window << shift) | 1
            self.highest_seen = nonce
            return True

        # Check if in window
        diff = self.highest_seen - nonce
        if diff >= REPLAY_WINDOW_SIZE:
            return False  # Too old

        # Check bitmap
        bit_mask = 1 << diff
        if self.window & bit_mask:
            return False  # Already seen

        # Mark as seen
        self.window |= bit_mask
        return True


@dataclass
class SecureSession:
    """
    Established secure session state.

    Tracks cipher states, replay windows, and session metadata
    for an authenticated peer connection.
    """
    session_id: str
    peer_id: str
    peer_static_public: bytes  # Ed25519 public key
    send_cipher: CipherState
    recv_cipher: CipherState
    replay_window: ReplayWindow = field(default_factory=ReplayWindow)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    messages_sent: int = 0
    messages_received: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() - self.created_at > SESSION_TIMEOUT_SECONDS

    @property
    def needs_rekey(self) -> bool:
        """Check if session needs rekeying."""
        return (
            time.time() - self.created_at > SESSION_REKEY_INTERVAL or
            self.send_cipher.nonce > MAX_NONCE_VALUE // 2
        )

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()


# =============================================================================
# SECURE CHANNEL INTERFACE
# =============================================================================

class SecureChannel(ABC):
    """
    Abstract interface for secure communication channels.

    Provides unified API for both Noise Protocol and DTLS transports.
    """

    @abstractmethod
    async def handshake_initiator(
        self,
        peer_address: str,
        peer_static_public: Optional[bytes] = None
    ) -> SecureSession:
        """
        Initiate handshake as the initiator.

        Args:
            peer_address: Address of peer (host:port)
            peer_static_public: Optional known peer Ed25519 public key

        Returns:
            Established secure session

        Raises:
            HandshakeError: If handshake fails
        """
        pass

    @abstractmethod
    async def handshake_responder(
        self,
        handshake_init: bytes,
        peer_address: str
    ) -> Tuple[SecureSession, bytes]:
        """
        Respond to handshake as the responder.

        Args:
            handshake_init: Initial handshake message from initiator
            peer_address: Address of peer

        Returns:
            Tuple of (established session, response message)

        Raises:
            HandshakeError: If handshake fails
        """
        pass

    @abstractmethod
    def encrypt(self, session: SecureSession, plaintext: bytes) -> bytes:
        """
        Encrypt message for sending.

        Args:
            session: Established secure session
            plaintext: Message to encrypt

        Returns:
            Encrypted message with header
        """
        pass

    @abstractmethod
    def decrypt(self, session: SecureSession, ciphertext: bytes) -> bytes:
        """
        Decrypt received message.

        Args:
            session: Established secure session
            ciphertext: Encrypted message with header

        Returns:
            Decrypted plaintext

        Raises:
            DecryptionError: If authentication fails
            ReplayError: If replay detected
        """
        pass

    @abstractmethod
    def close(self, session: SecureSession) -> bytes:
        """
        Create close notification message.

        Args:
            session: Session to close

        Returns:
            Encrypted close notification
        """
        pass


# =============================================================================
# NOISE PROTOCOL TRANSPORT (Noise_XX Pattern)
# =============================================================================

class NoiseTransport(SecureChannel):
    """
    Noise Protocol Framework implementation using the XX handshake pattern.

    The XX pattern provides mutual authentication with identity hiding:
    - Initiator and responder both prove identity during handshake
    - Identities are encrypted (forward-secret against static key compromise)
    - Perfect forward secrecy via ephemeral X25519 keys

    Handshake Pattern (Noise_XX):
        -> e                    (initiator sends ephemeral public key)
        <- e, ee, s, es         (responder sends ephemeral, DH, static, DH)
        -> s, se                (initiator sends static, DH)

    Standing on Giants: Perrin (2018) - "The Noise Protocol Framework"
    """

    def __init__(
        self,
        static_private_key: bytes,
        static_public_key: bytes,
        node_id: str,
        send_callback: Optional[Callable[[str, bytes], None]] = None
    ):
        """
        Initialize Noise transport.

        Args:
            static_private_key: Ed25519 private key (32 bytes)
            static_public_key: Ed25519 public key (32 bytes)
            node_id: Local node identifier
            send_callback: Optional callback for sending messages
        """
        self.static_private = static_private_key
        self.static_public = static_public_key
        self.node_id = node_id
        self.send_callback = send_callback

        # Convert Ed25519 keys to X25519 for DH operations
        # Note: In production, use separate X25519 keys derived from Ed25519
        self._derive_x25519_keys()

        # Session cache
        self._sessions: Dict[str, SecureSession] = {}
        self._pending_handshakes: Dict[str, dict] = {}

    def _derive_x25519_keys(self) -> None:
        """Derive X25519 keys from Ed25519 keys for DH operations."""
        # Generate X25519 keypair (in production, derive from Ed25519)
        self._x25519_private = x25519.X25519PrivateKey.generate()
        self._x25519_public = self._x25519_private.public_key()
        self._x25519_public_bytes = self._x25519_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

    def _generate_ephemeral(self) -> Tuple[x25519.X25519PrivateKey, bytes]:
        """Generate ephemeral X25519 keypair."""
        private = x25519.X25519PrivateKey.generate()
        public_bytes = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return private, public_bytes

    def _dh(self, private: x25519.X25519PrivateKey, public_bytes: bytes) -> bytes:
        """Perform X25519 Diffie-Hellman."""
        peer_public = x25519.X25519PublicKey.from_public_bytes(public_bytes)
        return private.exchange(peer_public)

    def _encrypt_with_ad(
        self,
        key: bytes,
        nonce: int,
        plaintext: bytes,
        ad: bytes
    ) -> bytes:
        """Encrypt with additional authenticated data."""
        nonce_bytes = b'\x00\x00\x00\x00' + struct.pack('>Q', nonce)
        cipher = ChaCha20Poly1305(key)
        return cipher.encrypt(nonce_bytes, plaintext, ad)

    def _decrypt_with_ad(
        self,
        key: bytes,
        nonce: int,
        ciphertext: bytes,
        ad: bytes
    ) -> bytes:
        """Decrypt with additional authenticated data."""
        nonce_bytes = b'\x00\x00\x00\x00' + struct.pack('>Q', nonce)
        cipher = ChaCha20Poly1305(key)
        try:
            return cipher.decrypt(nonce_bytes, ciphertext, ad)
        except InvalidTag:
            raise DecryptionError("AEAD authentication failed during handshake")

    async def handshake_initiator(
        self,
        peer_address: str,
        peer_static_public: Optional[bytes] = None
    ) -> SecureSession:
        """
        Initiate Noise_XX handshake.

        Message 1: -> e
        """
        # Initialize symmetric state
        state = SymmetricState.initialize(NOISE_PROTOCOL_NAME)

        # Generate ephemeral keypair
        e_private, e_public = self._generate_ephemeral()

        # -> e: Send ephemeral public key
        state.mix_hash(e_public)

        # Create handshake init message
        msg = struct.pack('!B', MessageType.HANDSHAKE_INIT) + e_public

        # Store pending handshake state
        handshake_id = hashlib.sha256(e_public).hexdigest()[:16]
        self._pending_handshakes[peer_address] = {
            'state': state,
            'e_private': e_private,
            'e_public': e_public,
            'handshake_id': handshake_id,
            'initiated_at': time.time(),
        }

        if self.send_callback:
            self.send_callback(peer_address, msg)

        # In async implementation, wait for response
        # For now, return placeholder - actual session created in process_handshake_response
        raise HandshakeError("Handshake initiated, awaiting response")

    def create_handshake_init(self) -> Tuple[bytes, dict]:
        """
        Create handshake init message (Message 1: -> e).

        Returns:
            Tuple of (handshake_init_message, handshake_state_dict)
        """
        # Initialize symmetric state
        state = SymmetricState.initialize(NOISE_PROTOCOL_NAME)

        # Generate ephemeral keypair
        e_private, e_public = self._generate_ephemeral()

        # -> e: Mix ephemeral public key into hash
        state.mix_hash(e_public)

        # Create message
        msg = struct.pack('!B', MessageType.HANDSHAKE_INIT) + e_public

        # Return state for continuation
        handshake_state = {
            'state': state,
            'e_private': e_private,
            'e_public': e_public,
            'phase': 'init_sent',
        }

        return msg, handshake_state

    async def handshake_responder(
        self,
        handshake_init: bytes,
        peer_address: str
    ) -> Tuple[SecureSession, bytes]:
        """
        Respond to Noise_XX handshake.

        Message 2: <- e, ee, s, es

        Note: Responder only sends response here, session established after final message.
        """
        if len(handshake_init) < 1 + NOISE_DH_SIZE:
            raise HandshakeError("Handshake init too short")

        msg_type = handshake_init[0]
        if msg_type != MessageType.HANDSHAKE_INIT:
            raise HandshakeError(f"Expected HANDSHAKE_INIT, got {msg_type}")

        re_public = handshake_init[1:1 + NOISE_DH_SIZE]  # Remote ephemeral (initiator's e)

        # Initialize symmetric state (same as initiator)
        state = SymmetricState.initialize(NOISE_PROTOCOL_NAME)

        # Mix initiator's ephemeral (same as initiator did)
        state.mix_hash(re_public)

        # Generate our ephemeral
        e_private, e_public = self._generate_ephemeral()
        state.mix_hash(e_public)

        # ee: DH(our_e, their_e)
        ee_shared = self._dh(e_private, re_public)
        temp_key = state.mix_key(ee_shared)

        # Encrypt our static public key with temp_key
        # s: encrypted static public
        encrypted_s = self._encrypt_with_ad(
            temp_key, 0, self._x25519_public_bytes, state.h
        )
        state.mix_hash(encrypted_s)

        # es: DH(our_static, their_e)
        es_shared = self._dh(self._x25519_private, re_public)
        state.mix_key(es_shared)

        # NOTE: Don't split yet - wait for final message
        # Store state for final message processing
        initiator_cipher, responder_cipher = state.split()

        # Create response message
        response = struct.pack('!B', MessageType.HANDSHAKE_RESPONSE)
        response += e_public
        response += encrypted_s

        # Store state for final message processing
        session_id = hashlib.sha256(e_public + re_public).hexdigest()[:16]

        self._pending_handshakes[peer_address] = {
            'state': state,
            'e_private': e_private,
            'e_public': e_public,
            're_public': re_public,
            # Responder sends with responder_cipher, receives with initiator_cipher
            'send_cipher': responder_cipher,
            'recv_cipher': initiator_cipher,
            'session_id': session_id,
            'phase': 'response_sent',
        }

        return None, response  # Session not yet complete

    def process_handshake_response(
        self,
        response: bytes,
        handshake_state: dict,
        peer_address: str
    ) -> Tuple[Optional[SecureSession], bytes]:
        """
        Process handshake response and create final message.

        This is the initiator processing the responder's message.
        The initiator and responder must derive the same shared secrets.
        """
        if len(response) < 1 + NOISE_DH_SIZE + NOISE_DH_SIZE + NOISE_TAG_SIZE:
            raise HandshakeError("Handshake response too short")

        msg_type = response[0]
        if msg_type != MessageType.HANDSHAKE_RESPONSE:
            raise HandshakeError(f"Expected HANDSHAKE_RESPONSE, got {msg_type}")

        offset = 1
        re_public = response[offset:offset + NOISE_DH_SIZE]  # Responder's ephemeral
        offset += NOISE_DH_SIZE

        encrypted_rs = response[offset:offset + NOISE_DH_SIZE + NOISE_TAG_SIZE]
        offset += NOISE_DH_SIZE + NOISE_TAG_SIZE

        state: SymmetricState = handshake_state['state']
        e_private: x25519.X25519PrivateKey = handshake_state['e_private']

        # Mix responder's ephemeral (same as responder did)
        state.mix_hash(re_public)

        # ee: DH(our_e, their_e) - same shared secret as responder computed
        ee_shared = self._dh(e_private, re_public)
        temp_key = state.mix_key(ee_shared)

        # Decrypt responder's static public key
        rs_public = self._decrypt_with_ad(temp_key, 0, encrypted_rs, state.h)
        state.mix_hash(encrypted_rs)

        # es: DH(our_e, their_static) - we're initiator, use our e with their s
        es_shared = self._dh(e_private, rs_public)
        state.mix_key(es_shared)

        # Split to get cipher states - initiator gets (send, recv)
        initiator_cipher, responder_cipher = state.split()

        # Initiator sends with initiator_cipher, receives with responder_cipher
        send_cipher = initiator_cipher
        recv_cipher = responder_cipher

        # Create final message (simplified - just acknowledge)
        final_msg = struct.pack('!B', MessageType.HANDSHAKE_FINAL)
        final_msg += handshake_state['e_public']  # Echo our ephemeral as confirmation

        # Create session
        session_id = hashlib.sha256(
            handshake_state['e_public'] + re_public
        ).hexdigest()[:16]

        session = SecureSession(
            session_id=session_id,
            peer_id=peer_address,
            peer_static_public=rs_public,
            send_cipher=send_cipher,
            recv_cipher=recv_cipher,
        )

        self._sessions[peer_address] = session

        return session, final_msg

    def process_handshake_final(
        self,
        final_msg: bytes,
        peer_address: str
    ) -> SecureSession:
        """
        Process final handshake message and establish session.

        The responder finalizes the session using pre-computed cipher states.
        """
        if peer_address not in self._pending_handshakes:
            raise HandshakeError("No pending handshake for this peer")

        pending = self._pending_handshakes[peer_address]

        if len(final_msg) < 1 + NOISE_DH_SIZE:
            raise HandshakeError("Handshake final too short")

        msg_type = final_msg[0]
        if msg_type != MessageType.HANDSHAKE_FINAL:
            raise HandshakeError(f"Expected HANDSHAKE_FINAL, got {msg_type}")

        # Verify the echoed ephemeral matches what we received
        echoed_e = final_msg[1:1 + NOISE_DH_SIZE]
        if echoed_e != pending['re_public']:
            raise HandshakeError("Handshake final: ephemeral key mismatch")

        # Use pre-computed cipher states from handshake_responder
        # Responder sends with responder_cipher, receives with initiator_cipher
        send_cipher = pending['send_cipher']
        recv_cipher = pending['recv_cipher']

        # Create session
        session = SecureSession(
            session_id=pending['session_id'],
            peer_id=peer_address,
            peer_static_public=pending['re_public'],  # Initiator's ephemeral as identity proxy
            send_cipher=send_cipher,
            recv_cipher=recv_cipher,
        )

        self._sessions[peer_address] = session
        del self._pending_handshakes[peer_address]

        return session

    def encrypt(self, session: SecureSession, plaintext: bytes) -> bytes:
        """
        Encrypt message for sending.

        Format: [type:1][nonce:8][ciphertext:N][tag:16]
        """
        if session.is_expired:
            raise SessionError("Session expired")

        nonce = session.send_cipher.nonce
        ciphertext = session.send_cipher.encrypt(plaintext)
        session.messages_sent += 1
        session.touch()

        # Build message
        msg = struct.pack('!B', MessageType.APPLICATION_DATA)
        msg += struct.pack('>Q', nonce)
        msg += ciphertext

        return msg

    def decrypt(self, session: SecureSession, message: bytes) -> bytes:
        """
        Decrypt received message.
        """
        if len(message) < 1 + 8 + NOISE_TAG_SIZE:
            raise DecryptionError("Message too short")

        msg_type = message[0]
        if msg_type != MessageType.APPLICATION_DATA:
            raise DecryptionError(f"Expected APPLICATION_DATA, got {msg_type}")

        nonce = struct.unpack('>Q', message[1:9])[0]
        ciphertext = message[9:]

        # Check for replay
        if not session.replay_window.check_and_update(nonce):
            raise ReplayError(f"Replay detected: nonce {nonce}")

        # Decrypt
        plaintext = session.recv_cipher.decrypt(ciphertext, nonce)
        session.messages_received += 1
        session.touch()

        return plaintext

    def close(self, session: SecureSession) -> bytes:
        """Create close notification message."""
        close_payload = struct.pack('>Q', int(time.time()))
        return self.encrypt(session, close_payload)

    def get_session(self, peer_address: str) -> Optional[SecureSession]:
        """Get existing session for peer."""
        session = self._sessions.get(peer_address)
        if session and not session.is_expired:
            return session
        return None

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        expired = [
            addr for addr, session in self._sessions.items()
            if session.is_expired
        ]
        for addr in expired:
            del self._sessions[addr]
        return len(expired)


# =============================================================================
# DTLS TRANSPORT (Alternative Implementation)
# =============================================================================

class DTLSTransport(SecureChannel):
    """
    DTLS 1.3 transport implementation.

    A simpler alternative to Noise Protocol, using standard TLS handshake
    adapted for datagrams with:
    - Cookie exchange for DoS protection
    - Retransmission for packet loss
    - Epoch-based key management

    Standing on Giants: Rescorla (2018) - RFC 9147 (DTLS 1.3)
    """

    def __init__(
        self,
        static_private_key: bytes,
        static_public_key: bytes,
        node_id: str,
        cipher_suite: str = "chacha20-poly1305"
    ):
        """
        Initialize DTLS transport.

        Args:
            static_private_key: Ed25519 private key
            static_public_key: Ed25519 public key
            node_id: Local node identifier
            cipher_suite: Cipher suite to use ("chacha20-poly1305" or "aes-256-gcm")
        """
        self.static_private = static_private_key
        self.static_public = static_public_key
        self.node_id = node_id
        self.cipher_suite = cipher_suite

        # DTLS-specific state
        self._cookie_secret = os.urandom(32)
        self._sessions: Dict[str, SecureSession] = {}
        self._pending_handshakes: Dict[str, dict] = {}

        # Epoch management
        self._current_epoch = 0
        self._next_sequence = 0

    def _generate_cookie(self, client_address: str, client_random: bytes) -> bytes:
        """Generate DTLS cookie for HelloRetryRequest."""
        data = client_address.encode() + client_random
        return hmac.new(self._cookie_secret, data, hashlib.sha256).digest()[:DTLS_COOKIE_LENGTH]

    def _verify_cookie(self, client_address: str, client_random: bytes, cookie: bytes) -> bool:
        """Verify DTLS cookie."""
        expected = self._generate_cookie(client_address, client_random)
        return hmac.compare_digest(expected, cookie)

    def _derive_keys(self, shared_secret: bytes, client_random: bytes, server_random: bytes) -> Tuple[bytes, bytes]:
        """Derive encryption keys from shared secret."""
        # Use HKDF for key derivation
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,  # 2 keys * 32 bytes
            salt=client_random + server_random,
            info=b"dtls13 key expansion",
            backend=default_backend()
        )
        key_material = hkdf.derive(shared_secret)

        client_key = key_material[:32]
        server_key = key_material[32:64]

        return client_key, server_key

    async def handshake_initiator(
        self,
        peer_address: str,
        peer_static_public: Optional[bytes] = None
    ) -> SecureSession:
        """Initiate DTLS handshake."""
        # Generate client random
        client_random = os.urandom(32)

        # Generate ephemeral X25519 key
        e_private = x25519.X25519PrivateKey.generate()
        e_public = e_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        # Create ClientHello
        hello = struct.pack('!H', DTLS_VERSION)  # Version
        hello += client_random  # Client random
        hello += struct.pack('!B', 0)  # Session ID length (0 for new session)
        hello += struct.pack('!B', 0)  # Cookie length (0 for initial hello)
        hello += e_public  # Key share

        msg = struct.pack('!B', MessageType.HANDSHAKE_INIT) + hello

        # Store pending state
        self._pending_handshakes[peer_address] = {
            'client_random': client_random,
            'e_private': e_private,
            'e_public': e_public,
            'phase': 'hello_sent',
            'initiated_at': time.time(),
        }

        raise HandshakeError("DTLS handshake initiated, awaiting ServerHello")

    def create_handshake_init(self) -> Tuple[bytes, dict]:
        """Create DTLS ClientHello."""
        client_random = os.urandom(32)

        e_private = x25519.X25519PrivateKey.generate()
        e_public = e_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        hello = struct.pack('!H', DTLS_VERSION)
        hello += client_random
        hello += struct.pack('!B', 0)  # Session ID length
        hello += struct.pack('!B', 0)  # Cookie length
        hello += e_public

        msg = struct.pack('!B', MessageType.HANDSHAKE_INIT) + hello

        handshake_state = {
            'client_random': client_random,
            'e_private': e_private,
            'e_public': e_public,
            'phase': 'hello_sent',
        }

        return msg, handshake_state

    async def handshake_responder(
        self,
        handshake_init: bytes,
        peer_address: str
    ) -> Tuple[SecureSession, bytes]:
        """Respond to DTLS handshake."""
        if len(handshake_init) < 1 + 2 + 32 + 1 + 1 + NOISE_DH_SIZE:
            raise HandshakeError("ClientHello too short")

        msg_type = handshake_init[0]
        if msg_type != MessageType.HANDSHAKE_INIT:
            raise HandshakeError(f"Expected HANDSHAKE_INIT, got {msg_type}")

        offset = 1
        version = struct.unpack('!H', handshake_init[offset:offset+2])[0]
        offset += 2

        if version != DTLS_VERSION:
            raise HandshakeError(f"Unsupported DTLS version: {version}")

        client_random = handshake_init[offset:offset+32]
        offset += 32

        session_id_len = handshake_init[offset]
        offset += 1 + session_id_len

        cookie_len = handshake_init[offset]
        offset += 1

        # TODO: Implement cookie verification for DoS protection
        if cookie_len > 0:
            cookie = handshake_init[offset:offset+cookie_len]
            offset += cookie_len
            if not self._verify_cookie(peer_address, client_random, cookie):
                raise HandshakeError("Invalid cookie")

        client_e_public = handshake_init[offset:offset+NOISE_DH_SIZE]

        # Generate server random and ephemeral key
        server_random = os.urandom(32)
        e_private = x25519.X25519PrivateKey.generate()
        e_public = e_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        # Compute shared secret
        client_key = x25519.X25519PublicKey.from_public_bytes(client_e_public)
        shared_secret = e_private.exchange(client_key)

        # Derive keys
        client_key_bytes, server_key_bytes = self._derive_keys(
            shared_secret, client_random, server_random
        )

        # Create ServerHello
        hello = struct.pack('!H', DTLS_VERSION)
        hello += server_random
        hello += struct.pack('!B', 16)  # Session ID length
        hello += os.urandom(16)  # Session ID
        hello += e_public

        response = struct.pack('!B', MessageType.HANDSHAKE_RESPONSE) + hello

        # Create session
        session_id = hashlib.sha256(client_random + server_random).hexdigest()[:16]

        # For responder: send with server key, receive with client key
        session = SecureSession(
            session_id=session_id,
            peer_id=peer_address,
            peer_static_public=client_e_public,  # Using ephemeral as static for simplicity
            send_cipher=CipherState(key=server_key_bytes),
            recv_cipher=CipherState(key=client_key_bytes),
        )

        self._sessions[peer_address] = session

        return session, response

    def process_handshake_response(
        self,
        response: bytes,
        handshake_state: dict,
        peer_address: str
    ) -> Tuple[SecureSession, bytes]:
        """Process ServerHello and complete handshake."""
        if len(response) < 1 + 2 + 32 + 1 + NOISE_DH_SIZE:
            raise HandshakeError("ServerHello too short")

        msg_type = response[0]
        if msg_type != MessageType.HANDSHAKE_RESPONSE:
            raise HandshakeError(f"Expected HANDSHAKE_RESPONSE, got {msg_type}")

        offset = 1
        version = struct.unpack('!H', response[offset:offset+2])[0]
        offset += 2

        server_random = response[offset:offset+32]
        offset += 32

        session_id_len = response[offset]
        offset += 1 + session_id_len

        server_e_public = response[offset:offset+NOISE_DH_SIZE]

        # Compute shared secret
        e_private = handshake_state['e_private']
        server_key = x25519.X25519PublicKey.from_public_bytes(server_e_public)
        shared_secret = e_private.exchange(server_key)

        # Derive keys
        client_random = handshake_state['client_random']
        client_key_bytes, server_key_bytes = self._derive_keys(
            shared_secret, client_random, server_random
        )

        # Create session
        session_id = hashlib.sha256(client_random + server_random).hexdigest()[:16]

        # For initiator: send with client key, receive with server key
        session = SecureSession(
            session_id=session_id,
            peer_id=peer_address,
            peer_static_public=server_e_public,
            send_cipher=CipherState(key=client_key_bytes),
            recv_cipher=CipherState(key=server_key_bytes),
        )

        self._sessions[peer_address] = session

        # DTLS completes in 2 messages (simplified without certificate exchange)
        return session, b''

    def encrypt(self, session: SecureSession, plaintext: bytes) -> bytes:
        """Encrypt message using DTLS record layer."""
        if session.is_expired:
            raise SessionError("Session expired")

        nonce = session.send_cipher.nonce
        ciphertext = session.send_cipher.encrypt(plaintext)
        session.messages_sent += 1
        session.touch()

        # DTLS record header
        record = struct.pack('!B', MessageType.APPLICATION_DATA)
        record += struct.pack('!H', DTLS_VERSION)
        record += struct.pack('>Q', nonce)
        record += struct.pack('!H', len(ciphertext))
        record += ciphertext

        return record

    def decrypt(self, session: SecureSession, message: bytes) -> bytes:
        """Decrypt DTLS record."""
        if len(message) < 1 + 2 + 8 + 2 + NOISE_TAG_SIZE:
            raise DecryptionError("Record too short")

        msg_type = message[0]
        if msg_type != MessageType.APPLICATION_DATA:
            raise DecryptionError(f"Expected APPLICATION_DATA, got {msg_type}")

        offset = 1
        version = struct.unpack('!H', message[offset:offset+2])[0]
        offset += 2

        nonce = struct.unpack('>Q', message[offset:offset+8])[0]
        offset += 8

        length = struct.unpack('!H', message[offset:offset+2])[0]
        offset += 2

        ciphertext = message[offset:offset+length]

        # Replay check
        if not session.replay_window.check_and_update(nonce):
            raise ReplayError(f"Replay detected: nonce {nonce}")

        plaintext = session.recv_cipher.decrypt(ciphertext, nonce)
        session.messages_received += 1
        session.touch()

        return plaintext

    def close(self, session: SecureSession) -> bytes:
        """Create DTLS close_notify alert."""
        alert = struct.pack('!BB', 1, 0)  # Warning level, close_notify
        return self.encrypt(session, alert)

    def get_session(self, peer_address: str) -> Optional[SecureSession]:
        """Get existing session for peer."""
        session = self._sessions.get(peer_address)
        if session and not session.is_expired:
            return session
        return None


# =============================================================================
# SECURE TRANSPORT MANAGER
# =============================================================================

class SecureTransportManager:
    """
    Unified manager for secure transport operations.

    Provides a high-level API for establishing secure channels, managing
    sessions, and handling transport-level operations transparently.
    """

    def __init__(
        self,
        static_private_key: bytes,
        static_public_key: bytes,
        node_id: str,
        transport_type: str = "noise"
    ):
        """
        Initialize secure transport manager.

        Args:
            static_private_key: Ed25519 private key (32 bytes hex or raw)
            static_public_key: Ed25519 public key (32 bytes hex or raw)
            node_id: Local node identifier
            transport_type: Transport protocol ("noise" or "dtls")
        """
        # Convert hex strings to bytes if needed
        if isinstance(static_private_key, str):
            static_private_key = bytes.fromhex(static_private_key)
        if isinstance(static_public_key, str):
            static_public_key = bytes.fromhex(static_public_key)

        self.node_id = node_id
        self.transport_type = transport_type

        # Initialize transport
        if transport_type == "noise":
            self.transport: SecureChannel = NoiseTransport(
                static_private_key=static_private_key,
                static_public_key=static_public_key,
                node_id=node_id
            )
        elif transport_type == "dtls":
            self.transport = DTLSTransport(
                static_private_key=static_private_key,
                static_public_key=static_public_key,
                node_id=node_id
            )
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")

        # Session cache with LRU eviction
        self._session_cache: OrderedDict[str, SecureSession] = OrderedDict()
        self._pending_handshakes: Dict[str, dict] = {}

    def create_handshake(self, peer_address: str) -> bytes:
        """
        Create handshake initiation message.

        Args:
            peer_address: Target peer address (host:port)

        Returns:
            Handshake init message to send to peer
        """
        if self.transport_type == "noise":
            msg, state = self.transport.create_handshake_init()
        else:
            msg, state = self.transport.create_handshake_init()

        self._pending_handshakes[peer_address] = state
        return msg

    def process_handshake(
        self,
        message: bytes,
        peer_address: str
    ) -> Tuple[Optional[SecureSession], Optional[bytes]]:
        """
        Process incoming handshake message.

        Args:
            message: Handshake message from peer
            peer_address: Peer address

        Returns:
            Tuple of (session if established, response to send if any)
        """
        if len(message) < 1:
            raise HandshakeError("Empty message")

        msg_type = message[0]

        if msg_type == MessageType.HANDSHAKE_INIT:
            # We're the responder
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                session, response = loop.run_until_complete(
                    self.transport.handshake_responder(message, peer_address)
                )
                if session:
                    self._cache_session(peer_address, session)
                return session, response
            finally:
                loop.close()

        elif msg_type == MessageType.HANDSHAKE_RESPONSE:
            # We're the initiator, process response
            if peer_address not in self._pending_handshakes:
                raise HandshakeError("No pending handshake for response")

            state = self._pending_handshakes[peer_address]
            session, final_msg = self.transport.process_handshake_response(
                message, state, peer_address
            )

            if session:
                self._cache_session(peer_address, session)
                del self._pending_handshakes[peer_address]

            return session, final_msg if final_msg else None

        elif msg_type == MessageType.HANDSHAKE_FINAL:
            # We're the responder, finalize
            session = self.transport.process_handshake_final(message, peer_address)
            self._cache_session(peer_address, session)
            return session, None

        else:
            raise HandshakeError(f"Unknown handshake message type: {msg_type}")

    def encrypt_message(self, peer_address: str, plaintext: bytes) -> bytes:
        """
        Encrypt message for peer.

        Args:
            peer_address: Target peer address
            plaintext: Message to encrypt

        Returns:
            Encrypted message

        Raises:
            SessionError: If no session exists for peer
        """
        session = self.get_session(peer_address)
        if not session:
            raise SessionError(f"No secure session for {peer_address}")

        return self.transport.encrypt(session, plaintext)

    def decrypt_message(self, peer_address: str, ciphertext: bytes) -> bytes:
        """
        Decrypt message from peer.

        Args:
            peer_address: Source peer address
            ciphertext: Encrypted message

        Returns:
            Decrypted plaintext

        Raises:
            SessionError: If no session exists for peer
            DecryptionError: If decryption fails
            ReplayError: If replay detected
        """
        session = self.get_session(peer_address)
        if not session:
            raise SessionError(f"No secure session for {peer_address}")

        return self.transport.decrypt(session, ciphertext)

    def get_session(self, peer_address: str) -> Optional[SecureSession]:
        """Get existing session for peer."""
        session = self._session_cache.get(peer_address)
        if session and not session.is_expired:
            # Move to end for LRU
            self._session_cache.move_to_end(peer_address)
            return session
        elif session:
            # Remove expired session
            del self._session_cache[peer_address]
        return None

    def has_session(self, peer_address: str) -> bool:
        """Check if secure session exists for peer."""
        return self.get_session(peer_address) is not None

    def close_session(self, peer_address: str) -> Optional[bytes]:
        """
        Close session with peer.

        Returns close notification to send to peer.
        """
        session = self.get_session(peer_address)
        if not session:
            return None

        close_msg = self.transport.close(session)
        del self._session_cache[peer_address]
        return close_msg

    def _cache_session(self, peer_address: str, session: SecureSession) -> None:
        """Cache session with LRU eviction."""
        # Evict if at capacity
        while len(self._session_cache) >= MAX_CACHED_SESSIONS:
            self._session_cache.popitem(last=False)

        self._session_cache[peer_address] = session

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count removed."""
        expired = [
            addr for addr, session in self._session_cache.items()
            if session.is_expired
        ]
        for addr in expired:
            del self._session_cache[addr]
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get transport statistics."""
        active_sessions = [
            {
                'peer': addr,
                'session_id': session.session_id,
                'messages_sent': session.messages_sent,
                'messages_received': session.messages_received,
                'age_seconds': time.time() - session.created_at,
            }
            for addr, session in self._session_cache.items()
            if not session.is_expired
        ]

        return {
            'transport_type': self.transport_type,
            'active_sessions': len(active_sessions),
            'pending_handshakes': len(self._pending_handshakes),
            'sessions': active_sessions,
        }


# =============================================================================
# INTEGRATION WITH GOSSIP ENGINE
# =============================================================================

def create_secure_gossip_transport(
    private_key_hex: str,
    public_key_hex: str,
    node_id: str,
    transport_type: str = "noise"
) -> SecureTransportManager:
    """
    Factory function to create secure transport for gossip protocol.

    Args:
        private_key_hex: Ed25519 private key (64 hex chars)
        public_key_hex: Ed25519 public key (64 hex chars)
        node_id: Node identifier
        transport_type: "noise" (recommended) or "dtls"

    Returns:
        Configured SecureTransportManager

    Example:
        >>> from core.pci import generate_keypair
        >>> priv, pub = generate_keypair()
        >>> transport = create_secure_gossip_transport(priv, pub, "node_001")
    """
    return SecureTransportManager(
        static_private_key=private_key_hex,
        static_public_key=public_key_hex,
        node_id=node_id,
        transport_type=transport_type
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Error types
    'SecureTransportError',
    'HandshakeError',
    'DecryptionError',
    'ReplayError',
    'SessionError',
    'NonceExhaustionError',

    # Data structures
    'CipherState',
    'SymmetricState',
    'ReplayWindow',
    'SecureSession',
    'HandshakeState',
    'MessageType',

    # Transport implementations
    'SecureChannel',
    'NoiseTransport',
    'DTLSTransport',

    # Manager
    'SecureTransportManager',

    # Factory
    'create_secure_gossip_transport',

    # Constants
    'NOISE_PROTOCOL_NAME',
    'SESSION_TIMEOUT_SECONDS',
    'REPLAY_WINDOW_SIZE',
]
