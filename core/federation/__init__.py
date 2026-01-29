"""
BIZRA Pattern Federation Package

P2P pattern sharing for DDAGI network effect.
"""

from .gossip import GossipEngine, NodeInfo, NodeState, GossipMessage, MessageType
from .propagation import (
    PatternStore, 
    PropagationEngine, 
    ElevatedPattern, 
    PatternStatus,
    PatternMetrics
)
from .consensus import ConsensusEngine, ConsensusRound, Vote, VoteType
from .node import FederationNode, SyncFederationNode
from .protocol import FederatedPattern, PatternImpact, FederationProtocol

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
    "ConsensusRound",
    "Vote",
    "VoteType",
    # Node
    "FederationNode",
    "SyncFederationNode",
    # Protocol
    "FederatedPattern",
    "PatternImpact",
    "FederationProtocol",
]

__version__ = "1.0.0"
