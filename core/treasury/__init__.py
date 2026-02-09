"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA TREASURY — Resource Management & Justice Enforcement                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Treasury operations, ADL (عدل) justice enforcement, and market mechanisms. ║
║                                                                              ║
║   Components:                                                                ║
║   - treasury_mode: Graceful degradation and resource management              ║
║   - adl_kernel: Full antitrust kernel with causal drag                       ║
║   - adl_invariant: Gini coefficient enforcement (≤ 0.40)                     ║
║   - market_integration: Harberger tax and compute market                     ║
║                                                                              ║
║   Constitutional Constraint: Gini ≤ 0.40 (ADL justice invariant)             ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - Gini (1912): Income Inequality Measurement                               ║
║   - Harberger (1965): Optimal Taxation Theory                                ║
║   - Rawls (1971): Justice as Fairness                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Sovereign Module Decomposition
Migrated: 2026-02-05 | Files now in dedicated treasury package
"""

from .adl_invariant import (  # type: ignore[attr-defined]
    AdlGate,
    AdlInvariant,
    calculate_gini,
)
from .adl_kernel import (  # type: ignore[attr-defined]
    AdlEnforcer,
)
from .adl_kernel import AdlInvariant as AdlInvariantKernel  # type: ignore[attr-defined]
from .adl_kernel import (  # type: ignore[attr-defined]
    IncrementalGini,
    NetworkGiniTracker,
)
from .adl_kernel import calculate_gini as calculate_gini_kernel  # type: ignore[attr-defined]
from .market_integration import (  # type: ignore[attr-defined]
    MarketAwareMuraqabah,
)
from .treasury_controller import TreasuryController
from .treasury_mode import create_treasury_controller
from .treasury_persistence import TreasuryPersistence

# Direct imports from local files
from .treasury_types import (
    EthicsAssessment,
    TransitionEvent,
    TransitionTrigger,
    TreasuryEvent,
    TreasuryMode,
    TreasuryState,
)

__all__ = [
    # Treasury Types & Enums
    "TreasuryMode",
    "TreasuryState",
    "TreasuryEvent",
    "TransitionTrigger",
    "TransitionEvent",
    "EthicsAssessment",
    # Treasury Operations
    "TreasuryPersistence",
    "TreasuryController",
    "create_treasury_controller",
    # ADL Kernel
    "AdlEnforcer",
    "AdlInvariantKernel",
    "IncrementalGini",
    "NetworkGiniTracker",
    # ADL Invariant
    "AdlInvariant",
    "AdlGate",
    "calculate_gini",
    # Market Integration
    "MarketAwareMuraqabah",
]
