"""
BIZRA PATTERN FEDERATION PROTOCOL (PFP)
Allows nodes to share SAPE-elevated patterns with Proof-of-Impact.
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from core.pci import PCIEnvelope, EnvelopeBuilder

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
    pattern_logic: str # JSON or DSL
    impact_proof: PatternImpact
    signatures: List[str] = field(default_factory=list) # Validator signatures

class FederationProtocol:
    """
    Manages the exchange of trusted patterns.
    """
    def __init__(self, node_id: str, private_key: str):
        self.node_id = node_id
        self.private_key = private_key
        self.known_patterns: Dict[str, FederatedPattern] = {}
        
    def create_pattern_proposal(self, logic: str, impact: PatternImpact) -> PCIEnvelope:
        """Wrap a pattern in a PCI Envelope for federation."""
        pattern = FederatedPattern(
            pattern_id=str(uuid.uuid4()),
            source_node_id=self.node_id,
            pattern_logic=logic,
            impact_proof=impact
        )
        
        # Build PCI Envelope
        builder = EnvelopeBuilder()
        builder.with_sender("PAT", self.node_id, "lookup_public_key") # Simplify pubkey lookup
        builder.with_payload(
            action="FEDERATE_PATTERN",
            data=self._serialize_pattern(pattern),
            policy_hash="constitution_v1",
            state_hash="current_state"
        )
        # Ihsan score must meet threshold
        builder.with_metadata(
            ihsan=impact.ihsan_score,
            snr=impact.average_snr_boost # Use boost as SNR proxy
        )
        
        envelope = builder.build()
        envelope.sign(self.private_key)
        
        return envelope
    
    def receive_proposal(self, envelope: PCIEnvelope) -> bool:
        """
        Receive and validate a pattern proposal from the network.
        Should use PCIGateKeeper for verification.
        """
        # 1. basic PCI verification (GateKeeper) would happen here
        # 2. Extract pattern
        data = envelope.payload.data
        if envelope.metadata.ihsan_score < 0.95:
             print(f"Refused pattern {envelope.envelope_id}: Low Ihsan")
             return False
             
        # 3. Store
        print(f"Accepted pattern from {envelope.sender.agent_id}")
        return True

    def _serialize_pattern(self, pattern: FederatedPattern) -> Dict:
        return {
            "id": pattern.pattern_id,
            "logic": pattern.pattern_logic,
            "impact": {
                "score": pattern.impact_proof.impact_score
            }
        }
