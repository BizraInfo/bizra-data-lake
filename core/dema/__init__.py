"""Dema core — Ambient Kernel components.

Public surface:
- DemaProfile, ProfileStore         — onboarding identity
- DailyLogEntry, DailyLog           — append-only operator log
- FourStateModel, MissionStateMachine — Current/Ideal/Gap/Next §9 model
- DemaReceipt, ReceiptWriter        — local hash-chained audit trail

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
from core.dema.receipts import DemaReceipt, ReceiptWriter

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
    "DemaReceipt",
    "ReceiptWriter",
]
