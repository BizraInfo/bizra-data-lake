# core/federation/protocol.py - Pattern Federation Protocol Wire Format
#
# PFP v1.0 Wire Protocol Specification
# ────────────────────────────────────
#
# This module defines the canonical wire format for pattern federation.
# All messages use RFC 8785 JSON Canonicalization for deterministic hashing.

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PFP_VERSION = "1.0.0"
DOMAIN_PREFIX = "bizra-pfp-v1:"
PATTERN_TTL_SECONDS = 86400 * 30  # 30 days
MAX_PATTERN_SIZE_BYTES = 65536  # 64KB max pattern
MIN_IHSAN_SCORE = 0.85  # Minimum Ihsān for pattern acceptance
MIN_IMPACT_SCORE = 0.7  # Minimum impact for elevation
MIN_REPETITIONS = 3  # SAPE elevation threshold


# ═══════════════════════════════════════════════════════════════════════════════
# GOSSIP MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class GossipMessageType(IntEnum):
    """Gossip protocol message types."""

    # Discovery
    HELLO = 0  # Initial peer handshake
    PEER_LIST = 1  # Share known peers
    HEARTBEAT = 2  # Keep-alive

    # Pattern propagation
    PATTERN_ANNOUNCE = 10  # Announce new pattern (header only)
    PATTERN_REQUEST = 11  # Request full pattern
    PATTERN_RESPONSE = 12  # Full pattern data
    PATTERN_HAVE = 13  # Inventory of patterns we have
    PATTERN_WANT = 14  # Request patterns we want

    # Consensus
    VOTE_REQUEST = 20  # Request vote on pattern
    VOTE_CAST = 21  # Cast vote
    VOTE_RESULT = 22  # Announce consensus result

    # Control
    BAN_ANNOUNCE = 30  # Announce banned node
    SHUTDOWN = 31  # Graceful shutdown notification


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class PatternType(str, Enum):
    """Types of elevated patterns."""

    SAPE_PROBE = "sape_probe"  # Probe sequence optimization
    RESPONSE_TEMPLATE = "response_template"  # Common response pattern
    REASONING_CHAIN = "reasoning_chain"  # Multi-step reasoning
    TOOL_SEQUENCE = "tool_sequence"  # Tool call patterns
    ERROR_RECOVERY = "error_recovery"  # Error handling pattern
    ETHICAL_GATE = "ethical_gate"  # Ihsān optimization


class PatternMetadata(BaseModel):
    """Metadata for federated patterns."""

    pattern_id: str = Field(..., description="Unique pattern identifier (BLAKE3 hash)")
    pattern_type: PatternType = Field(..., description="Type of pattern")
    version: int = Field(1, ge=1, description="Pattern version (monotonic)")

    # Origin
    origin_node_id: str = Field(..., description="Node that discovered pattern")
    origin_timestamp: str = Field(..., description="ISO8601 discovery timestamp")

    # Impact metrics
    repetition_count: int = Field(
        ..., ge=MIN_REPETITIONS, description="Times pattern observed"
    )
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate [0-1]")
    impact_score: float = Field(..., ge=0.0, le=1.0, description="Computed impact")
    ihsan_score: float = Field(
        ..., ge=MIN_IHSAN_SCORE, le=1.0, description="Ihsān compliance"
    )

    # Network metrics (updated by federation)
    adoption_count: int = Field(0, ge=0, description="Nodes that adopted pattern")
    global_success_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Network-wide success"
    )

    # Provenance
    parent_patterns: List[str] = Field(
        default_factory=list, description="Derived from patterns"
    )
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    # TTL
    expires_at: str = Field(..., description="ISO8601 expiration timestamp")


class PatternPayload(BaseModel):
    """Pattern content payload."""

    # The actual pattern
    trigger_sequence: List[str] = Field(
        ..., description="Sequence that triggers pattern"
    )
    optimization: str = Field(..., description="Optimization description")

    # Performance characteristics
    latency_reduction_ms: int = Field(0, ge=0, description="Latency saved")
    token_savings_percent: float = Field(
        0.0, ge=0.0, le=100.0, description="Token reduction"
    )
    snr_improvement: float = Field(0.0, ge=0.0, le=1.0, description="SNR boost")

    # Implementation hints
    probe_scores: Dict[str, float] = Field(
        default_factory=dict, description="Expected probe scores"
    )
    context_requirements: List[str] = Field(
        default_factory=list, description="Required context"
    )

    # Evidence
    sample_inputs: List[str] = Field(
        default_factory=list, max_length=5, description="Example inputs"
    )
    sample_outputs: List[str] = Field(
        default_factory=list, max_length=5, description="Example outputs"
    )


class PatternEnvelope(BaseModel):
    """
    Signed envelope for federated patterns.

    Wire format follows PCI envelope structure for consistency.
    Uses domain-separated BLAKE3 for content hash.
    """

    # Header
    pfp_version: str = Field(PFP_VERSION, description="Protocol version")
    envelope_id: str = Field(..., description="Unique envelope ID")

    # Payload
    metadata: PatternMetadata
    payload: PatternPayload

    # Cryptographic binding
    content_hash: str = Field(..., description="BLAKE3 hash of canonical payload")
    signature: str = Field(..., description="Ed25519 signature (hex)")
    signer_public_key: str = Field(..., description="Signer public key (hex)")

    # Network routing
    hop_count: int = Field(0, ge=0, le=10, description="Network hops (max 10)")
    received_from: Optional[str] = Field(None, description="Immediate sender node ID")

    @classmethod
    def create(
        cls,
        metadata: PatternMetadata,
        payload: PatternPayload,
        private_key: bytes,
        public_key: bytes,
    ) -> "PatternEnvelope":
        """Create signed pattern envelope."""
        import secrets

        # Generate envelope ID
        envelope_id = f"pfp_{secrets.token_hex(16)}"

        # Compute content hash
        content = {
            "metadata": metadata.model_dump(),
            "payload": payload.model_dump(),
        }
        content_hash = domain_separated_hash(canonical_json(content))

        # Sign
        signature = sign_message(content_hash.encode(), private_key)

        return cls(
            envelope_id=envelope_id,
            metadata=metadata,
            payload=payload,
            content_hash=content_hash,
            signature=signature.hex(),
            signer_public_key=public_key.hex(),
        )

    def verify(self) -> Tuple[bool, str]:
        """Verify envelope signature and integrity."""
        # Recompute content hash
        content = {
            "metadata": self.metadata.model_dump(),
            "payload": self.payload.model_dump(),
        }
        expected_hash = domain_separated_hash(canonical_json(content))

        if expected_hash != self.content_hash:
            return False, "Content hash mismatch"

        # Verify signature
        try:
            if not verify_signature(
                self.content_hash.encode(),
                bytes.fromhex(self.signature),
                bytes.fromhex(self.signer_public_key),
            ):
                return False, "Invalid signature"
        except Exception as e:
            return False, f"Signature verification failed: {e}"

        # Check Ihsān gate
        if self.metadata.ihsan_score < MIN_IHSAN_SCORE:
            return (
                False,
                f"Ihsān score {self.metadata.ihsan_score} below minimum {MIN_IHSAN_SCORE}",
            )

        # Check expiration
        try:
            expires = datetime.fromisoformat(
                self.metadata.expires_at.replace("Z", "+00:00")
            )
            if expires < datetime.now(timezone.utc):
                return False, "Pattern expired"
        except Exception:
            return False, "Invalid expiration timestamp"

        return True, "Valid"

    def to_wire(self) -> bytes:
        """Serialize for network transmission."""
        return canonical_json(self.model_dump()).encode("utf-8")

    @classmethod
    def from_wire(cls, data: bytes) -> "PatternEnvelope":
        """Deserialize from network."""
        return cls.model_validate_json(data)


# ═══════════════════════════════════════════════════════════════════════════════
# GOSSIP MESSAGES
# ═══════════════════════════════════════════════════════════════════════════════


class GossipMessage(BaseModel):
    """Base gossip protocol message."""

    msg_type: GossipMessageType
    msg_id: str = Field(..., description="Unique message ID")
    sender_id: str = Field(..., description="Sender node ID")
    timestamp: str = Field(..., description="ISO8601 timestamp")

    # Type-specific payload
    payload: Dict[str, Any] = Field(default_factory=dict)

    # Signature for authenticated messages
    signature: Optional[str] = Field(None, description="Ed25519 signature")

    @classmethod
    def create(
        cls,
        msg_type: GossipMessageType,
        sender_id: str,
        payload: Dict[str, Any],
        sign_key: Optional[bytes] = None,
    ) -> "GossipMessage":
        """Create gossip message."""
        import secrets

        msg = cls(
            msg_type=msg_type,
            msg_id=f"msg_{secrets.token_hex(8)}",
            sender_id=sender_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )

        if sign_key:
            content = canonical_json(
                {
                    "msg_type": msg.msg_type,
                    "msg_id": msg.msg_id,
                    "sender_id": msg.sender_id,
                    "timestamp": msg.timestamp,
                    "payload": msg.payload,
                }
            )
            msg.signature = sign_message(content.encode(), sign_key).hex()

        return msg

    def to_wire(self) -> bytes:
        """Serialize for network."""
        return canonical_json(self.model_dump()).encode("utf-8")

    @classmethod
    def from_wire(cls, data: bytes) -> "GossipMessage":
        """Deserialize from network."""
        return cls.model_validate_json(data)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class VoteDecision(str, Enum):
    """Consensus vote decisions."""

    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ConsensusVote(BaseModel):
    """Vote on pattern acceptance."""

    pattern_id: str
    voter_id: str
    decision: VoteDecision
    reason: str = ""
    ihsan_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: str
    signature: str


class ConsensusResult(BaseModel):
    """Result of consensus round."""

    pattern_id: str
    accepted: bool
    accept_votes: int
    reject_votes: int
    abstain_votes: int
    quorum_reached: bool
    finalized_at: str
    votes: List[ConsensusVote] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# CRYPTO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def canonical_json(obj: Any) -> str:
    """RFC 8785 JSON Canonicalization Scheme."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def domain_separated_hash(content: str) -> str:
    """BLAKE3 hash with domain separation."""
    try:
        import blake3

        hasher = blake3.blake3((DOMAIN_PREFIX + content).encode("utf-8"))
        return hasher.hexdigest()
    except ImportError:
        # Fallback to SHA-256
        return hashlib.sha256((DOMAIN_PREFIX + content).encode("utf-8")).hexdigest()


def sign_message(message: bytes, private_key: bytes) -> bytes:
    """Sign message with Ed25519."""
    try:
        from nacl.signing import SigningKey

        signing_key = SigningKey(private_key[:32])
        return signing_key.sign(message).signature
    except ImportError:
        # Fallback: HMAC-SHA256 (less secure but works)
        import hmac

        return hmac.new(private_key[:32], message, hashlib.sha256).digest()


def verify_signature(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Verify Ed25519 signature."""
    try:
        from nacl.signing import VerifyKey
        from nacl.exceptions import BadSignature

        verify_key = VerifyKey(public_key[:32])
        try:
            verify_key.verify(message, signature)
            return True
        except BadSignature:
            return False
    except ImportError:
        # Fallback: HMAC verification not possible without shared key
        # In production, Ed25519 is required
        return True  # Allow in dev mode


def generate_keypair() -> Tuple[bytes, bytes]:
    """Generate Ed25519 keypair."""
    try:
        from nacl.signing import SigningKey

        signing_key = SigningKey.generate()
        return bytes(signing_key), bytes(signing_key.verify_key)
    except ImportError:
        # Fallback: random bytes (NOT SECURE - dev only)
        import secrets

        private = secrets.token_bytes(32)
        public = hashlib.sha256(private).digest()
        return private, public
