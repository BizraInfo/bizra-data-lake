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

# Direct imports from local files
from .treasury_types import (
    TreasuryMode,
    TreasuryState,
    TreasuryEvent,
    TransitionTrigger,
    TransitionEvent,
    EthicsAssessment,
)
from .treasury_persistence import TreasuryPersistence
from .treasury_controller import TreasuryController
from .treasury_mode import create_treasury_controller
from .adl_kernel import (
    AdlEnforcer,
    AdlInvariant as AdlInvariantKernel,
    IncrementalGini,
    NetworkGiniTracker,
    calculate_gini as calculate_gini_kernel,
)
from .adl_invariant import (
    AdlInvariant,
    AdlGate,
    calculate_gini,
)
from .market_integration import (
    MarketAwareMuraqabah,
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
