"""Re-export from canonical location: core.sovereign.autonomy"""

# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.autonomy import *  # noqa: F401,F403
from core.sovereign.autonomy import (
    AutonomousLoop,
    DecisionCandidate,
    DecisionGate,
    DecisionOutcome,
    DecisionType,
    GateResult,
    LoopState,
    SystemMetrics,
    create_autonomous_loop,
)
