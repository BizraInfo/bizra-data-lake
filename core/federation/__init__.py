"""
BIZRA Pattern Federation Package

P2P pattern sharing for DDAGI network effect.

Components:
- gossip: SWIM-style node discovery and health monitoring
- consensus: Byzantine fault-tolerant consensus (PBFT)
- propagation: Pattern elevation and network sharing
- secure_transport: DTLS/Noise encrypted transport layer (P0-2)
- node: Main federation node integration
- protocol: Protocol definitions and wire formats
- interaction_boundary: Axiom 1.6 — Pool-mediated interaction (Phase 61)
- pool_consensus: Amended Theorem 2.4 — Pool-mediated BFT (Phase 61)
"""

from .consensus import ConsensusEngine, Proposal, Vote
from .gossip import GossipEngine, GossipMessage, MessageType, NodeInfo, NodeState
from .node import FederationNode, SyncFederationNode
from .propagation import (
    ElevatedPattern,
    PatternMetrics,
    PatternStatus,
    PatternStore,
    PropagationEngine,
)
from .protocol import FederatedPattern, FederationProtocol, PatternImpact
from .secure_transport import (  # Error types; Data structures; Transports; Factory
    CipherState,
    DecryptionError,
    DTLSTransport,
    HandshakeError,
    NoiseTransport,
    ReplayError,
    ReplayWindow,
    SecureChannel,
    SecureSession,
    SecureTransportError,
    SecureTransportManager,
    SessionError,
    create_secure_gossip_transport,
)

__all__ = [
    # Gossip
    "GossipEngine",
    "NodeInfo",
    "NodeState",
    "GossipMessage",
    "MessageType",
    # Propagation
    "PatternStore",
    "PropagationEngine",
    "ElevatedPattern",
    "PatternStatus",
    "PatternMetrics",
    # Consensus
    "ConsensusEngine",
    "Vote",
    "Proposal",
    # Node
    "FederationNode",
    "SyncFederationNode",
    # Protocol
    "FederatedPattern",
    "PatternImpact",
    "FederationProtocol",
    # Secure Transport (P0-2)
    "SecureTransportError",
    "HandshakeError",
    "DecryptionError",
    "ReplayError",
    "SessionError",
    "SecureSession",
    "CipherState",
    "ReplayWindow",
    "SecureChannel",
    "NoiseTransport",
    "DTLSTransport",
    "SecureTransportManager",
    "create_secure_gossip_transport",
]

__version__ = "1.2.0"  # Phase 61: interaction boundary + pool consensus
