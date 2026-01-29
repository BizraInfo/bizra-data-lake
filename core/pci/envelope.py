"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Envelope Definitions
"""

import uuid
import time
import secrets
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from enum import Enum

from .reject_codes import RejectCode
from .crypto import canonical_json, domain_separated_digest, sign_message

class AgentType(str, Enum):
    PAT = "PAT"
    SAT = "SAT"

@dataclass
class EnvelopeSender:
    agent_type: AgentType
    agent_id: str
    public_key: str

@dataclass
class EnvelopePayload:
    action: str
    data: Dict[str, Any]
    policy_hash: str
    state_hash: str

@dataclass
class EnvelopeMetadata:
    ihsan_score: float
    snr_score: float
    urgency: str = "REAL_TIME"
    
    def __post_init__(self):
        if not (0.0 <= self.ihsan_score <= 1.0):
            raise ValueError("Ihsan score must be between 0.0 and 1.0")

@dataclass
class EnvelopeSignature:
    algorithm: str
    value: str
    signed_fields: List[str]

@dataclass
class PCIEnvelope:
    version: str
    envelope_id: str
    timestamp: str
    nonce: str
    sender: EnvelopeSender
    payload: EnvelopePayload
    metadata: EnvelopeMetadata
    signature: Optional[EnvelopeSignature] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def compute_digest(self) -> str:
        """Compute the canonical digest of the envelope (excluding signature)."""
        # Create a dict without signature
        d = asdict(self)
        if 'signature' in d:
            del d['signature']
        
        # Canonicalize and hash
        return domain_separated_digest(canonical_json(d))
        
    def sign(self, private_key_hex: str) -> 'PCIEnvelope':
        """Sign the envelope and attach signature."""
        digest = self.compute_digest()
        sig_hex = sign_message(digest, private_key_hex)
        
        self.signature = EnvelopeSignature(
            algorithm="ed25519",
            value=sig_hex,
            signed_fields=["version", "envelope_id", "timestamp", "nonce", "sender", "payload", "metadata"]
        )
        return self

class EnvelopeBuilder:
    """Builder for PCI Envelopes."""
    
    def __init__(self):
        self._sender = None
        self._payload = None
        self._metadata = None
        
    def with_sender(self, agent_type: str, agent_id: str, public_key: str):
        self._sender = EnvelopeSender(AgentType(agent_type), agent_id, public_key)
        return self
        
    def with_payload(self, action: str, data: Dict, policy_hash: str, state_hash: str):
        self._payload = EnvelopePayload(action, data, policy_hash, state_hash)
        return self
        
    def with_metadata(self, ihsan: float, snr: float, urgency: str = "REAL_TIME"):
        self._metadata = EnvelopeMetadata(ihsan, snr, urgency)
        return self
        
    def build(self) -> PCIEnvelope:
        if not all([self._sender, self._payload, self._metadata]):
            raise ValueError("Sender, Payload, and Metadata are required")
            
        return PCIEnvelope(
            version="1.0.0",
            envelope_id=str(uuid.uuid4()),
            timestamp=datetime_now_iso(),
            nonce=secrets.token_hex(32),
            sender=self._sender,
            payload=self._payload,
            metadata=self._metadata
        )

def datetime_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
