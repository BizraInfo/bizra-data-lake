"""
BIZRA Constitutional Engine Package
====================================

The Omega Point - Unified Constitutional Framework.

This package provides the mathematical and operational foundation for
BIZRA's constitutional constraints:

- GAP-C1: IhsanProjector - O(1) projection from 8D Ihsan to 3D NTU
- GAP-C2: AdlInvariant - Protocol-level justice enforcement gate
- GAP-C3: ByzantineConsensus - f < n/3 fault tolerant consensus
- GAP-C4: TreasuryController - Graceful mode degradation

Usage:
    from core.constitutional import (
        ConstitutionalEngine,
        create_constitutional_engine,
        IhsanVector,
        TreasuryMode,
    )

    engine = create_constitutional_engine(
        node_id="node_001",
        private_key=private_key,
        public_key=public_key,
        total_nodes=7,
    )

    # Evaluate an action
    permitted, details = engine.evaluate_action(
        ihsan_vector=IhsanVector(
            correctness=0.98,
            safety=0.97,
            user_benefit=0.95,
            efficiency=0.92,
            auditability=0.94,
            anti_centralization=0.88,
            robustness=0.91,
            adl_fairness=0.96,
        ),
        distribution={"node_001": 100, "node_002": 150},
    )

Standing on Giants: Shannon, Lamport, Landauer, Al-Ghazali

Created: 2026-02-03 | BIZRA Constitutional Engine v1.0.0
"""

from core.constitutional.omega_engine import (
    # Core Types
    IhsanVector,
    NTUState,
    # GAP-C1: Ihsan Projector
    IhsanProjector,
    # GAP-C2: Adl Invariant
    AdlInvariant,
    AdlInvariantResult,
    AdlViolation,
    AdlViolationType,
    AdlViolationError,
    # GAP-C3: Byzantine Consensus
    ByzantineConsensus,
    ByzantineVoteType,
    SignedVote,
    ConsensusState,
    ConsensusProposal,
    # GAP-C4: Treasury Controller
    TreasuryMode,
    TreasuryModeConfig,
    TreasuryController,
    TREASURY_MODES,
    # Unified Engine
    ConstitutionalEngine,
    create_constitutional_engine,
    # Constants
    IHSAN_DIMENSIONS,
    ADL_GINI_THRESHOLD,
    ADL_GINI_EMERGENCY,
    BFT_QUORUM_FRACTION,
    LANDAUER_LIMIT_JOULES,
)

__all__ = [
    # Core Types
    "IhsanVector",
    "NTUState",
    # GAP-C1
    "IhsanProjector",
    # GAP-C2
    "AdlInvariant",
    "AdlInvariantResult",
    "AdlViolation",
    "AdlViolationType",
    "AdlViolationError",
    # GAP-C3
    "ByzantineConsensus",
    "ByzantineVoteType",
    "SignedVote",
    "ConsensusState",
    "ConsensusProposal",
    # GAP-C4
    "TreasuryMode",
    "TreasuryModeConfig",
    "TreasuryController",
    "TREASURY_MODES",
    # Unified
    "ConstitutionalEngine",
    "create_constitutional_engine",
    # Constants
    "IHSAN_DIMENSIONS",
    "ADL_GINI_THRESHOLD",
    "ADL_GINI_EMERGENCY",
    "BFT_QUORUM_FRACTION",
    "LANDAUER_LIMIT_JOULES",
]

__version__ = "1.0.0"
__author__ = "BIZRA Sovereignty Team"
