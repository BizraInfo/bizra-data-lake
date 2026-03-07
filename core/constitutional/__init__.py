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

from core.constitutional.omega_engine import (  # Core Types; GAP-C1: Ihsan Projector; GAP-C2: Adl Invariant; GAP-C3: Byzantine Consensus; GAP-C4: Treasury Controller; Unified Engine; Constants
    ADL_GINI_EMERGENCY,
    ADL_GINI_THRESHOLD,
    BFT_QUORUM_FRACTION,
    IHSAN_DIMENSIONS,
    LANDAUER_LIMIT_JOULES,
    TREASURY_MODES,
    AdlInvariant,
    AdlInvariantResult,
    AdlViolation,
    AdlViolationError,
    AdlViolationType,
    ByzantineConsensus,
    ByzantineVoteType,
    ConsensusProposal,
    ConsensusState,
    ConstitutionalEngine,
    IhsanProjector,
    IhsanVector,
    NTUState,
    SignedVote,
    TreasuryController,
    TreasuryMode,
    TreasuryModeConfig,
    create_constitutional_engine,
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

# ═══════════════════════════════════════════════════════════════════
# Phase 67: Fixed-Point Arithmetic Kernel
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.fixed_point import (
    FP_MAX,
    FP_ONE,
    FP_PRECISION,
    FP_ZERO,
    fp,
    fp_add,
    fp_clamp,
    fp_div,
    fp_float,
    fp_gini_threshold,
    fp_ihsan_floor,
    fp_mul,
    fp_percentage,
    fp_sub,
    fp_weighted_avg,
)

__all__ += [
    # Fixed-Point Kernel (Phase 67)
    "FP_PRECISION",
    "FP_ONE",
    "FP_ZERO",
    "FP_MAX",
    "fp",
    "fp_float",
    "fp_add",
    "fp_sub",
    "fp_mul",
    "fp_div",
    "fp_clamp",
    "fp_weighted_avg",
    "fp_percentage",
    "fp_ihsan_floor",
    "fp_gini_threshold",
]

# ═══════════════════════════════════════════════════════════════════
# Phase 67: Constitutional Types
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.types import (
    ActionReceipt,
    Attestation,
    Event,
    Proposal,
    Reflex,
    WalletState,
)

__all__ += [
    "ActionReceipt",
    "WalletState",
    "Proposal",
    "Reflex",
    "Attestation",
    "Event",
]

# ═══════════════════════════════════════════════════════════════════
# Phase 67: 15 Native Algorithms
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.algorithms import (
    accrue_bloom,
    append_event,
    apply_demurrage,
    asabiyyah_adjustment,
    asabiyyah_score,
    backing_ratio,
    compile_reflex,
    compute_gini,
    compute_zakat,
    decay_bloom,
    full_ihsan_check,
    ghazali_equity_factor,
    ihsan_score,
    intent_gate,
    khaldunian_throttle,
    mint_seed,
    network_asabiyyah,
    progressive_mint,
    reflex_lookup,
    shura_resolve,
    shura_vote,
    trust_score,
    verify_event_chain,
)

__all__ += [
    # A1: Ihsan
    "intent_gate",
    "ihsan_score",
    "full_ihsan_check",
    # A2: SEED
    "mint_seed",
    # A3: BLOOM
    "accrue_bloom",
    "decay_bloom",
    # A4: Gini
    "compute_gini",
    "khaldunian_throttle",
    "ghazali_equity_factor",
    "progressive_mint",
    # A5: Zakat
    "compute_zakat",
    # A6: Backing
    "backing_ratio",
    # A7: Demurrage
    "apply_demurrage",
    # A8: Shura
    "shura_vote",
    "shura_resolve",
    # A9: Trust
    "trust_score",
    # A10: Reflex
    "compile_reflex",
    "reflex_lookup",
    # A14: Events
    "append_event",
    "verify_event_chain",
    # A15: Asabiyyah
    "asabiyyah_adjustment",
    "asabiyyah_score",
    "network_asabiyyah",
]

# ═══════════════════════════════════════════════════════════════════
# Phase 67: Declaration Genesis
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.declaration import (
    DECLARATION_BLAKE2B_256,
    DECLARATION_PATH,
    INVARIANTS,
    ConstitutionalInvariant,
    ConstitutionalViolation,
    compute_declaration_hash,
    create_genesis_event,
    load_declaration,
    verify_covenant_chain,
    verify_declaration_hash,
)

__all__ += [
    "DECLARATION_BLAKE2B_256",
    "DECLARATION_PATH",
    "INVARIANTS",
    "ConstitutionalInvariant",
    "ConstitutionalViolation",
    "load_declaration",
    "compute_declaration_hash",
    "verify_declaration_hash",
    "create_genesis_event",
    "verify_covenant_chain",
]

# ═══════════════════════════════════════════════════════════════════
# Phase 67: Constitutional Ticker (12-Step Heartbeat)
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.ticker import TickResult, process_tick

__all__ += [
    "TickResult",
    "process_tick",
]

# ═══════════════════════════════════════════════════════════════════
# Phase 67: Sovereignty CLI (Production Genesis Interface)
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.cli import (
    AttestResult,
    InitResult,
    NodeState,
    StatusResult,
    WorkResult,
    attest_peer,
    get_status,
    init_node,
    load_node_state,
    process_work,
    save_node_state,
)

__all__ += [
    "InitResult",
    "WorkResult",
    "AttestResult",
    "StatusResult",
    "NodeState",
    "init_node",
    "process_work",
    "attest_peer",
    "get_status",
    "save_node_state",
    "load_node_state",
]

# ═══════════════════════════════════════════════════════════════════
# Phase 67: Sovereign Network Simulation
# ═══════════════════════════════════════════════════════════════════
from core.constitutional.simulation import (
    SimulationConfig,
    SimulationMilestone,
    SimulationReport,
    SovereignNetworkSimulation,
    render_simulation_report,
    run_simulation,
)

__all__ += [
    "SimulationConfig",
    "SimulationMilestone",
    "SimulationReport",
    "SovereignNetworkSimulation",
    "run_simulation",
    "render_simulation_report",
]

__version__ = "2.0.0"
__author__ = "BIZRA Sovereignty Team"
