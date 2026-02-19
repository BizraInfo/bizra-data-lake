"""Phase 47.1 — Safe Activation rollout infrastructure.

Canary routing, HMM caller isolation, metrics, and strict rollback.

Standing on Giants: Fowler (canary, 2010) · Nygard (Release It!, 2007)
"""

from core.rollout.canary import CanaryRouter
from core.rollout.hmm_gate import HMMCallerGate
from core.rollout.metrics import Phase46Metrics
from core.rollout.rollback import RollbackEngine, RollbackReceipt

__all__ = [
    "CanaryRouter",
    "HMMCallerGate",
    "Phase46Metrics",
    "RollbackEngine",
    "RollbackReceipt",
]
