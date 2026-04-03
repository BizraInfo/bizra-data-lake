"""
BIZRA PCI Protocol — PCIEnvelope
================================
Proof-Carrying Inference envelope with cryptographic signing.

Status: FROZEN — Changes require version bump + test vector update
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .crypto import (
    canonical_json,
    envelope_digest,
    generate_nonce,
    generate_keypair,
    sign_envelope,
    verify_envelope_signature,
    KeyPair,
)
from .types import (
    PCI_VERSION,
    AgentType,
    Metadata,
    Payload,
    Sender,
    Signature,
    SignatureAlgorithm,
    Urgency,
    generate_envelope_id,
    utc_now_iso,
)


@dataclass
class PCIEnvelope:
    """
    Proof-Carrying Inference Envelope.
    
    Every inference/action in the dual-agent architecture is wrapped
    in this cryptographically signed envelope.
    
    Wire Format: RFC 8785 JSON Canonicalization Scheme (JCS)
    Signature: Ed25519 over domain-separated BLAKE3 digest
    """
    version: str
    envelope_id: str
    timestamp: str
    nonce: str
    sender: Sender
    payload: Payload
    metadata: Metadata
    signature: Optional[Signature] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for wire format."""
        result = {
            "version": self.version,
            "envelope_id": self.envelope_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "sender": self.sender.to_dict(),
            "payload": self.payload.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
        if self.signature:
            result["signature"] = self.signature.to_dict()
        return result

    def to_canonical_bytes(self) -> bytes:
        """Serialize to canonical JSON bytes."""
        return canonical_json(self.to_dict())

    def to_canonical_json(self) -> str:
        """Serialize to canonical JSON string."""
        return self.to_canonical_bytes().decode('utf-8')

    def compute_digest(self) -> str:
        """Compute the domain-separated BLAKE3 digest."""
        return envelope_digest(self.to_dict())

    def sign(self, private_key: bytes) -> "PCIEnvelope":
        """
        Sign the envelope with Ed25519.
        
        Returns a new envelope with the signature field populated.
        """
        # Create data dict without signature
        data = {
            "version": self.version,
            "envelope_id": self.envelope_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
            "sender": self.sender.to_dict(),
            "payload": self.payload.to_dict(),
            "metadata": self.metadata.to_dict(),
        }
        
        signed_fields = ["version", "envelope_id", "timestamp", "nonce", "sender", "payload", "metadata"]
        signature_value = sign_envelope(data, private_key, signed_fields)
        
        return PCIEnvelope(
            version=self.version,
            envelope_id=self.envelope_id,
            timestamp=self.timestamp,
            nonce=self.nonce,
            sender=self.sender,
            payload=self.payload,
            metadata=self.metadata,
            signature=Signature(
                algorithm=SignatureAlgorithm.ED25519,
                value=signature_value,
                signed_fields=signed_fields,
            ),
        )

    def verify_signature(self) -> bool:
        """Verify the envelope signature."""
        if self.signature is None:
            return False
        return verify_envelope_signature(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PCIEnvelope":
        """Deserialize from dictionary."""
        signature = None
        if "signature" in data:
            signature = Signature.from_dict(data["signature"])
        
        return cls(
            version=data["version"],
            envelope_id=data["envelope_id"],
            timestamp=data["timestamp"],
            nonce=data["nonce"],
            sender=Sender.from_dict(data["sender"]),
            payload=Payload.from_dict(data["payload"]),
            metadata=Metadata.from_dict(data["metadata"]),
            signature=signature,
        )

    @classmethod
    def create(
        cls,
        agent_type: AgentType,
        agent_id: str,
        public_key: str,
        action: str,
        data: Dict[str, Any],
        policy_hash: str,
        state_hash: str,
        ihsan_score: float,
        snr_score: float,
        urgency: Urgency = Urgency.NEAR_REAL_TIME,
    ) -> "PCIEnvelope":
        """
        Create a new unsigned envelope.
        
        Call .sign(private_key) to add the signature.
        """
        return cls(
            version=PCI_VERSION,
            envelope_id=generate_envelope_id(),
            timestamp=utc_now_iso(),
            nonce=generate_nonce(),
            sender=Sender(
                agent_type=agent_type,
                agent_id=agent_id,
                public_key=public_key,
            ),
            payload=Payload(
                action=action,
                data=data,
                policy_hash=policy_hash,
                state_hash=state_hash,
            ),
            metadata=Metadata(
                ihsan_score=ihsan_score,
                snr_score=snr_score,
                urgency=urgency,
            ),
            signature=None,
        )


# =============================================================================
# ENVELOPE BUILDER (Fluent API)
# =============================================================================

class EnvelopeBuilder:
    """
    Fluent builder for PCIEnvelope.
    
    Example:
        envelope = (EnvelopeBuilder()
            .with_sender(AgentType.PAT, "pat-001", public_key)
            .with_action("propose", {"task": "analyze"})
            .with_policy(policy_hash)
            .with_state(state_hash)
            .with_scores(ihsan=0.97, snr=0.85)
            .build()
            .sign(private_key))
    """
    
    def __init__(self):
        self._agent_type: Optional[AgentType] = None
        self._agent_id: Optional[str] = None
        self._public_key: Optional[str] = None
        self._action: Optional[str] = None
        self._data: Dict[str, Any] = {}
        self._policy_hash: Optional[str] = None
        self._state_hash: Optional[str] = None
        self._ihsan_score: float = 0.0
        self._snr_score: float = 0.0
        self._urgency: Urgency = Urgency.NEAR_REAL_TIME

    def with_sender(
        self,
        agent_type: AgentType,
        agent_id: str,
        public_key: str,
    ) -> "EnvelopeBuilder":
        """Set the sender information."""
        self._agent_type = agent_type
        self._agent_id = agent_id
        self._public_key = public_key
        return self

    def with_action(self, action: str, data: Optional[Dict[str, Any]] = None) -> "EnvelopeBuilder":
        """Set the action and data."""
        self._action = action
        if data:
            self._data = data
        return self

    def with_data(self, data: Dict[str, Any]) -> "EnvelopeBuilder":
        """Set the payload data."""
        self._data = data
        return self

    def with_policy(self, policy_hash: str) -> "EnvelopeBuilder":
        """Set the policy hash."""
        self._policy_hash = policy_hash
        return self

    def with_state(self, state_hash: str) -> "EnvelopeBuilder":
        """Set the state hash."""
        self._state_hash = state_hash
        return self

    def with_scores(self, ihsan: float, snr: float) -> "EnvelopeBuilder":
        """Set the Ihsān and SNR scores."""
        self._ihsan_score = ihsan
        self._snr_score = snr
        return self

    def with_urgency(self, urgency: Urgency) -> "EnvelopeBuilder":
        """Set the urgency level."""
        self._urgency = urgency
        return self

    def build(self) -> PCIEnvelope:
        """Build the unsigned envelope."""
        if self._agent_type is None:
            raise ValueError("Sender agent_type is required")
        if self._agent_id is None:
            raise ValueError("Sender agent_id is required")
        if self._public_key is None:
            raise ValueError("Sender public_key is required")
        if self._action is None:
            raise ValueError("Action is required")
        if self._policy_hash is None:
            raise ValueError("Policy hash is required")
        if self._state_hash is None:
            raise ValueError("State hash is required")
        
        return PCIEnvelope.create(
            agent_type=self._agent_type,
            agent_id=self._agent_id,
            public_key=self._public_key,
            action=self._action,
            data=self._data,
            policy_hash=self._policy_hash,
            state_hash=self._state_hash,
            ihsan_score=self._ihsan_score,
            snr_score=self._snr_score,
            urgency=self._urgency,
        )


# =============================================================================
# ENVELOPE VALIDATION
# =============================================================================

def validate_envelope_schema(envelope_data: Dict[str, Any]) -> List[str]:
    """
    Validate envelope against schema requirements.
    
    Returns: List of validation errors (empty if valid)
    """
    errors = []
    
    # Required top-level fields
    required_fields = ["version", "envelope_id", "timestamp", "nonce", "sender", "payload", "metadata"]
    for field in required_fields:
        if field not in envelope_data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors
    
    # Version check
    if envelope_data["version"] != PCI_VERSION:
        errors.append(f"Unsupported version: {envelope_data['version']} (expected {PCI_VERSION})")
    
    # Nonce format
    nonce = envelope_data.get("nonce", "")
    if len(nonce) != 64:  # 32 bytes hex = 64 chars
        errors.append(f"Invalid nonce length: {len(nonce)} (expected 64)")
    
    # Sender validation
    sender = envelope_data.get("sender", {})
    if "agent_type" not in sender:
        errors.append("Missing sender.agent_type")
    elif sender["agent_type"] not in ["PAT", "SAT"]:
        errors.append(f"Invalid sender.agent_type: {sender['agent_type']}")
    
    if "public_key" not in sender:
        errors.append("Missing sender.public_key")
    elif len(sender.get("public_key", "")) != 64:  # 32 bytes hex
        errors.append(f"Invalid public_key length: {len(sender.get('public_key', ''))}")
    
    # Payload validation
    payload = envelope_data.get("payload", {})
    if "action" not in payload:
        errors.append("Missing payload.action")
    if "policy_hash" not in payload:
        errors.append("Missing payload.policy_hash")
    if "state_hash" not in payload:
        errors.append("Missing payload.state_hash")
    
    # Metadata validation
    metadata = envelope_data.get("metadata", {})
    if "ihsan_score" not in metadata:
        errors.append("Missing metadata.ihsan_score")
    elif not isinstance(metadata["ihsan_score"], (int, float)):
        errors.append("metadata.ihsan_score must be a number")
    elif not (0.0 <= metadata["ihsan_score"] <= 1.0):
        errors.append(f"metadata.ihsan_score out of range: {metadata['ihsan_score']}")
    
    if "snr_score" not in metadata:
        errors.append("Missing metadata.snr_score")
    elif not isinstance(metadata["snr_score"], (int, float)):
        errors.append("metadata.snr_score must be a number")
    elif not (0.0 <= metadata["snr_score"] <= 1.0):
        errors.append(f"metadata.snr_score out of range: {metadata['snr_score']}")
    
    # Signature validation (if present)
    if "signature" in envelope_data:
        sig = envelope_data["signature"]
        if "algorithm" not in sig:
            errors.append("Missing signature.algorithm")
        elif sig["algorithm"] not in ["ed25519", "dilithium5"]:
            errors.append(f"Unsupported signature.algorithm: {sig['algorithm']}")
        
        if "value" not in sig:
            errors.append("Missing signature.value")
        elif len(sig.get("value", "")) != 128:  # 64 bytes hex
            errors.append(f"Invalid signature.value length: {len(sig.get('value', ''))}")
    
    return errors
