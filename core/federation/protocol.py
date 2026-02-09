"""
BIZRA PATTERN FEDERATION PROTOCOL (PFP)
Allows nodes to share SAPE-elevated patterns with Proof-of-Impact.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from core.pci import EnvelopeBuilder, PCIEnvelope  # type: ignore[attr-defined]
from core.pci.gates import DEFAULT_CONSTITUTION_HASH, PCIGateKeeper


@dataclass
class PatternImpact:
    """Proof of a pattern's value."""

    success_count: int
    total_uses: int
    average_snr_boost: float
    ihsan_score: float

    @property
    def impact_score(self) -> float:
        """Calculate impact score 0.0-1.0."""
        # Simple heuristic: Success Rate * Ihsan * (1 + SNR_Boost)
        rate = self.success_count / max(1, self.total_uses)
        return min(1.0, rate * self.ihsan_score * (1.0 + self.average_snr_boost))


@dataclass
class FederatedPattern:
    """A pattern shared across the network."""

    pattern_id: str
    source_node_id: str
    pattern_logic: str  # JSON or DSL
    impact_proof: PatternImpact
    signatures: List[str] = field(default_factory=list)  # Validator signatures


class FederationProtocol:
    """
    Manages the exchange of trusted patterns.
    """

    def __init__(self, node_id: str, private_key: str):
        self.node_id = node_id
        self.private_key = private_key
        self.public_key = self._derive_public_key(private_key)
        if not self.public_key:
            raise ValueError("Invalid Ed25519 private key; cannot derive public key")
        self.known_patterns: Dict[str, FederatedPattern] = {}
        self.gatekeeper = PCIGateKeeper(policy_enforcement=True)

    def _derive_public_key(self, private_key_hex: str) -> str:
        """Derive public key from Ed25519 private key hex."""
        try:
            private_bytes = bytes.fromhex(private_key_hex)
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
            public_key = private_key.public_key()
            return public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            ).hex()
        except (ValueError, TypeError, AttributeError):
            # Key derivation failed - return empty (validation will catch this)
            return ""

    def create_pattern_proposal(self, logic: str, impact: PatternImpact) -> PCIEnvelope:
        """Wrap a pattern in a PCI Envelope for federation."""
        pattern = FederatedPattern(
            pattern_id=str(uuid.uuid4()),
            source_node_id=self.node_id,
            pattern_logic=logic,
            impact_proof=impact,
        )

        # Build PCI Envelope
        builder = EnvelopeBuilder()
        builder.with_sender("PAT", self.node_id, self.public_key)
        builder.with_payload(
            action="FEDERATE_PATTERN",
            data=self._serialize_pattern(pattern),
            policy_hash=DEFAULT_CONSTITUTION_HASH,
            state_hash="current_state",
        )
        # Ihsan score must meet threshold
        builder.with_metadata(
            ihsan=impact.ihsan_score,
            snr=impact.average_snr_boost,  # Use boost as SNR proxy
        )

        envelope = builder.build()
        envelope.sign(self.private_key)

        return envelope

    def receive_proposal(self, envelope: PCIEnvelope) -> bool:
        """
        Receive and validate a pattern proposal from the network.
        Should use PCIGateKeeper for verification.
        """
        # 1. PCI verification (GateKeeper)
        result = self.gatekeeper.verify(envelope)
        if not result.passed:
            print(f"Refused pattern {envelope.envelope_id}: {result.reject_code}")
            return False

        # 2. Extract pattern

        # 3. Store
        print(f"Accepted pattern from {envelope.sender.agent_id}")
        return True

    def _serialize_pattern(self, pattern: FederatedPattern) -> Dict:
        return {
            "id": pattern.pattern_id,
            "logic": pattern.pattern_logic,
            "impact": {"score": pattern.impact_proof.impact_score},
        }
