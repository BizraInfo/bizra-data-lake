"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA PROOF-OF-IMPACT BOUNTY ECONOMY                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   The Four Layers:                                                           ║
║   ════════════════                                                           ║
║   L1: DISCOVERY    — UERS 5D manifold scans for entropy anomalies           ║
║   L2: PROOF        — ZK-proofs + cryptographic impact verification           ║
║   L3: PAYMENT      — Smart contract instant payouts                          ║
║   L4: REPUTATION   — SNR-scored sovereign reputation ledger                  ║
║                                                                              ║
║   Traditional: Find → Report → Wait 90 days → Maybe get paid                 ║
║   BIZRA PoI:   Find → Prove → Get Paid (6 seconds, trustless)               ║
║                                                                              ║
║   Standing on: Bugcrowd • Ethereum • BIZRA Omega (47.9M ops/sec)            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import TYPE_CHECKING

# Version
BOUNTY_VERSION = "1.0.0"

# Severity Levels
SEVERITY_LEVELS = {
    "informational": {"multiplier": 0.5, "min_payout": 0},
    "low": {"multiplier": 1, "min_payout": 100},
    "medium": {"multiplier": 5, "min_payout": 500},
    "high": {"multiplier": 20, "min_payout": 2500},
    "critical": {"multiplier": 100, "min_payout": 10000},
}

# Vulnerability Categories
VULN_CATEGORIES = [
    "reentrancy",
    "flash_loan",
    "oracle_manipulation",
    "access_control",
    "integer_overflow",
    "logic_error",
    "front_running",
    "denial_of_service",
    "upgrade_vulnerability",
    "signature_malleability",
]

# UERS Vector Mapping for Security Analysis
SECURITY_VECTORS = {
    "surface": "bytecode_entropy",  # Static analysis
    "structural": "cfg_patterns",  # Control flow
    "behavioral": "state_transitions",  # Dynamic analysis
    "hypothetical": "symbolic_execution",  # Formal verification
    "contextual": "spec_vs_implementation",  # Semantic gaps
}

# Agent Roles
AGENT_ROLES = {
    "surveyor": {"model": "slither", "vectors": ["surface", "structural"]},
    "fuzzer": {"model": "echidna", "vectors": ["behavioral"]},
    "prover": {"model": "z3", "vectors": ["hypothetical"]},
    "ethicist": {"model": "constitutional", "vectors": ["contextual"]},
}

# Base payout per entropy unit
BASE_PAYOUT_PER_DELTA_E = 500  # $500 per entropy unit reduced

# SNR threshold for valid bounty claims
BOUNTY_SNR_THRESHOLD = 0.90

# Ihsān threshold for ethical validation
BOUNTY_IHSAN_THRESHOLD = 0.95

# Lazy imports
if TYPE_CHECKING:
    from .bridge import BountyBridge, PlatformAdapter
    from .hunter import HunterAgent, HunterSwarm
    from .impact_proof import ImpactProof, ImpactProofBuilder
    from .oracle import BountyCalculation, BountyOracle


def __getattr__(name: str):
    if name == "ImpactProof":
        from .impact_proof import ImpactProof

        return ImpactProof
    elif name == "BountyOracle":
        from .oracle import BountyOracle

        return BountyOracle
    elif name == "HunterAgent":
        from .hunter import HunterAgent

        return HunterAgent
    elif name == "BountyBridge":
        from .bridge import BountyBridge

        return BountyBridge
    raise AttributeError(f"module 'core.bounty' has no attribute '{name}'")


__all__ = [
    "BOUNTY_VERSION",
    "SEVERITY_LEVELS",
    "VULN_CATEGORIES",
    "SECURITY_VECTORS",
    "AGENT_ROLES",
    "BASE_PAYOUT_PER_DELTA_E",
    "BOUNTY_SNR_THRESHOLD",
    "BOUNTY_IHSAN_THRESHOLD",
    "ImpactProof",
    "BountyOracle",
    "HunterAgent",
    "BountyBridge",
]
