"""Dema core — Ambient Kernel components.

Public surface:
- DemaProfile, ProfileStore         — onboarding identity
- DailyLogEntry, DailyLog           — append-only operator log
- FourStateModel, MissionStateMachine — Current/Ideal/Gap/Next §9 model
- DemaReceipt, ReceiptWriter        — local hash-chained audit trail
- semantic_transducer types         — RawParsedClaim -> Claim trust boundary

All persistent state writes under sovereign_state/dema/ which is gitignored.
No network listener, no desktop control, no autonomous social.
"""

from __future__ import annotations

from core.dema.daily_log import DailyLog, DailyLogEntry
from core.dema.mission_state import FourStateModel, MissionStateMachine
from core.dema.profile import DemaProfile, ProfileStore
from core.dema.proof_convergence import (
    ProofConvergenceResult,
    ProofConvergenceVerifier,
    ProofSignal,
    converge_proofs,
)
from core.dema.proof_surface import (
    ClaimSource,
    ProofSurface,
    build_proof_surface,
    proof_surface_from_convergence,
)
from core.dema.receipts import DemaReceipt, ReceiptWriter
from core.dema.semantic_transducer import (
    Claim,
    ConstitutionalPolicy,
    GateDecision,
    GateVerdict,
    IntentType,
    MissionReceiptDescriptor,
    RawParsedClaim,
    ResourceScope,
    ResourceType,
    SemanticSurface,
    StepDescriptor,
    build_semantic_surface,
    compute_evidence_weight,
    describe_receipt_process,
    fate_gate,
    validate_raw_claim,
)

__all__ = [
    "DemaProfile",
    "ProfileStore",
    "DailyLogEntry",
    "DailyLog",
    "FourStateModel",
    "MissionStateMachine",
    "ProofSignal",
    "ProofConvergenceResult",
    "ProofConvergenceVerifier",
    "converge_proofs",
    "ClaimSource",
    "ProofSurface",
    "build_proof_surface",
    "proof_surface_from_convergence",
    "DemaReceipt",
    "ReceiptWriter",
    "RawParsedClaim",
    "Claim",
    "GateVerdict",
    "IntentType",
    "ResourceType",
    "ResourceScope",
    "StepDescriptor",
    "ConstitutionalPolicy",
    "GateDecision",
    "MissionReceiptDescriptor",
    "SemanticSurface",
    "compute_evidence_weight",
    "validate_raw_claim",
    "fate_gate",
    "describe_receipt_process",
    "build_semantic_surface",
]
