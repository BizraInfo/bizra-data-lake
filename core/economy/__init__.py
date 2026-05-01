"""BIZRA economic constitution primitives.

[ENFORCEMENT: WIRED] This package exposes deterministic, non-mutating economic
ledger and gate helpers. It does not execute transfers or perform signing.
"""

from __future__ import annotations

from core.economy.ledger import (
    CAP_PRECISION,
    NISAB_THRESHOLD_NC,
    ZAKAT_RATE_BPS,
    EconomicPolicyView,
    Identity,
    IdentityRegistry,
    InMemoryIdentityRegistry,
    LedgerEntry,
    LedgerState,
    RibaDetector,
    RibaPattern,
    TransactionType,
    ZakatAssessment,
    assess_zakat,
    build_entry,
    economic_fate_gate,
    enforce,
    gini,
    simulate_gini,
)

__all__ = [
    "CAP_PRECISION",
    "NISAB_THRESHOLD_NC",
    "ZAKAT_RATE_BPS",
    "EconomicPolicyView",
    "Identity",
    "IdentityRegistry",
    "InMemoryIdentityRegistry",
    "LedgerEntry",
    "LedgerState",
    "RibaDetector",
    "RibaPattern",
    "TransactionType",
    "ZakatAssessment",
    "assess_zakat",
    "build_entry",
    "economic_fate_gate",
    "enforce",
    "gini",
    "simulate_gini",
]
