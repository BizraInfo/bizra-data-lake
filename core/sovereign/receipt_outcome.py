"""
Receipt Outcome — Canonical decision/status computation
========================================================
Single source of truth for computing receipt decisions from query results.
Used by both SovereignRuntime and SpearPointPipeline.

Standing on Giants: Shannon (1948) — SNR gating determines decision.
"""

from __future__ import annotations

from typing import Any

# SNR floor below which results are quarantined (not rejected).
_SNR_QUARANTINE_THRESHOLD: float = 0.85


def receipt_outcome(result: Any) -> tuple[str, str, list[str]]:
    """Compute canonical receipt decision/status/reason codes.

    Args:
        result: A query result object with ``validation_passed`` (bool)
                and ``snr_score`` (float) attributes.

    Returns:
        A 3-tuple of ``(decision, status, reason_codes)`` where:
        - decision: "APPROVED" | "REJECTED" | "QUARANTINED"
        - status: "accepted" | "rejected" | "quarantined"
        - reason_codes: list of machine-readable rejection reasons
    """
    decision = "APPROVED"
    reason_codes: list[str] = []
    status = "accepted"

    if not getattr(result, "validation_passed", False):
        decision = "REJECTED"
        reason_codes.append("IHSAN_BELOW_THRESHOLD")
        status = "rejected"

    snr = float(getattr(result, "snr_score", 0.0))
    if snr < _SNR_QUARANTINE_THRESHOLD:
        if "SNR_BELOW_THRESHOLD" not in reason_codes:
            reason_codes.append("SNR_BELOW_THRESHOLD")
        if decision == "APPROVED":
            decision = "QUARANTINED"
            status = "quarantined"

    return decision, status, reason_codes
