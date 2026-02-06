"""
BIZRA PROOF-CARRYING INFERENCE (PCI) PROTOCOL
Rejection Code Registry - Stable Numeric IDs

Standing on Giants: Lamport (BFT), Shannon (SNR), Gini (Adl)

Code Ranges:
- 0-19: Core PCI gates (schema, signature, timestamp, quality)
- 20-49: Policy and state gates
- 50-99: Reserved for future core gates
- 100-199: Adl (Justice) invariant codes
- 200-299: Reserved for future invariants
"""

from enum import IntEnum


class RejectCode(IntEnum):
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE PCI GATES (0-19)
    # ═══════════════════════════════════════════════════════════════════════════
    SUCCESS = 0
    REJECT_SCHEMA = 1
    REJECT_SIGNATURE = 2
    REJECT_NONCE_REPLAY = 3
    REJECT_TIMESTAMP_STALE = 4
    REJECT_TIMESTAMP_FUTURE = 5
    REJECT_IHSAN_BELOW_MIN = 6
    REJECT_SNR_BELOW_MIN = 7
    REJECT_BUDGET_EXCEEDED = 8
    REJECT_POLICY_MISMATCH = 9
    REJECT_STATE_MISMATCH = 10
    REJECT_ROLE_VIOLATION = 11
    REJECT_QUORUM_FAILED = 12
    REJECT_FATE_VIOLATION = 13
    REJECT_INVARIANT_FAILED = 14
    REJECT_RATE_LIMITED = 15

    # ═══════════════════════════════════════════════════════════════════════════
    # ADL (JUSTICE) INVARIANT CODES (100-199)
    # Standing on Giants: Gini (1912), Harberger (1962), Rawls (1971)
    # ═══════════════════════════════════════════════════════════════════════════
    REJECT_ADL_GINI_EXCEEDED = 100  # Post-tx Gini > threshold (0.40)
    REJECT_ADL_CONSERVATION = 101  # Total value changed (conservation law)
    REJECT_ADL_NEGATIVE_HOLDING = 102  # Would create negative balance
    REJECT_ADL_INVALID_TX = 103  # Malformed transaction
    REJECT_ADL_DUST_AMOUNT = 104  # Amount below minimum holding

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL (99)
    # ═══════════════════════════════════════════════════════════════════════════
    REJECT_INTERNAL_ERROR = 99
