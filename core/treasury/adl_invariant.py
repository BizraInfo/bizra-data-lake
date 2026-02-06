"""
BIZRA ADL INVARIANT — Re-export from canonical source.

The canonical implementation lives in core.sovereign.adl_invariant.
This module re-exports for backwards compatibility with treasury imports.
"""

from core.sovereign.adl_invariant import (  # noqa: F401
    AdlGate,
    AdlInvariant,
    AdlRejectCode,
    AdlValidationResult,
    RedistributionResult,
    Transaction,
    assert_adl_invariant,
    calculate_gini,
    calculate_gini_components,
    create_adl_extended_gatekeeper,
    simulate_transaction_impact,
)
