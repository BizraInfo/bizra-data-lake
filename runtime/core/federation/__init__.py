# core/federation/__init__.py - Pattern Federation Protocol (PFP) v1.0
#
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PATTERN FEDERATION PROTOCOL - Network Effect Activation Layer            ║
# ╠════════════════════════════════════════════════════════════════════════════╣
# ║                                                                            ║
# ║  The missing piece that transforms isolated PAT nodes into a              ║
# ║  collectively intelligent network. Each node's learning becomes           ║
# ║  every node's learning.                                                    ║
# ║                                                                            ║
# ║  Network Architecture:                                                     ║
# ║  ─────────────────────                                                     ║
# ║                                                                            ║
# ║    Node₁ ←──────→ Node₂                                                   ║
# ║      ↕      ↘   ↙      ↕                                                  ║
# ║      ↕        ✕        ↕          Gossip Protocol                         ║
# ║      ↕      ↙   ↘      ↕          (Epidemic broadcast)                    ║
# ║    Node₃ ←──────→ Node₄                                                   ║
# ║                                                                            ║
# ║  Pattern Flow:                                                             ║
# ║  ─────────────                                                             ║
# ║                                                                            ║
# ║    1. SAPE elevates pattern (>3 repetitions, score > 0.7)                 ║
# ║    2. Pattern wrapped in PatternEnvelope (signed, timestamped)            ║
# ║    3. Gossip broadcasts to known peers                                    ║
# ║    4. Peers validate (Ihsān gate + signature check)                       ║
# ║    5. Consensus reached (3/5 validators)                                  ║
# ║    6. Pattern committed to local SAPE cache                               ║
# ║    7. BlockGraph records Proof-of-Impact                                  ║
# ║                                                                            ║
# ║  Security Model:                                                           ║
# ║  ──────────────                                                            ║
# ║                                                                            ║
# ║    - Ed25519 signatures on all patterns                                   ║
# ║    - BLAKE3 content hashing for integrity                                 ║
# ║    - Ihsān ≥ 0.85 gate (prevent malicious patterns)                       ║
# ║    - Rate limiting (max 100 patterns/min per node)                        ║
# ║    - Reputation tracking (bad actors get isolated)                        ║
# ║                                                                            ║
# ║  إحسان Standard: Excellence through collective intelligence               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

"""
Pattern Federation Protocol (PFP) - Activates BIZRA Network Effects

Usage:
    from core.federation import PatternFederation, GossipProtocol

    # Initialize federation
    fed = PatternFederation(node_id="node_001", port=9999)
    await fed.start()

    # When SAPE elevates a pattern
    await fed.broadcast_pattern(elevated_pattern)

    # Patterns from network automatically integrate with local SAPE
"""

from core.federation.protocol import (
    PatternEnvelope,
    PatternMetadata,
    PatternPayload,
    GossipMessage,
    GossipMessageType,
    ConsensusVote,
    ConsensusResult,
)

from core.federation.gossip import (
    GossipProtocol,
    PeerInfo,
    PeerState,
)

from core.federation.consensus import (
    PatternConsensus,
    ConsensusState,
    CONSENSUS_QUORUM,
)

from core.federation.federation import (
    PatternFederation,
    FederationConfig,
    FederationStats,
)

from core.federation.sape_bridge import (
    SAPEFederationBridge,
    get_federation_bridge,
    start_federation,
    share_pattern,
)

__all__ = [
    # Protocol types
    "PatternEnvelope",
    "PatternMetadata",
    "PatternPayload",
    "GossipMessage",
    "GossipMessageType",
    "ConsensusVote",
    "ConsensusResult",
    # Gossip
    "GossipProtocol",
    "PeerInfo",
    "PeerState",
    # Consensus
    "PatternConsensus",
    "ConsensusState",
    "CONSENSUS_QUORUM",
    # Federation
    "PatternFederation",
    "FederationConfig",
    "FederationStats",
    # SAPE Bridge
    "SAPEFederationBridge",
    "get_federation_bridge",
    "start_federation",
    "share_pattern",
]

__version__ = "1.0.0"
__protocol_version__ = "pfp-v1"
