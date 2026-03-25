"""
Epistemic Admissibility — Claim tagging and binding.

Definition 10.1 (Claim Tag):
    tag(c) in {VERIFIED, PLANNED, DERIVED, HYPOTHETICAL}

Definition 10.2 (Claim Binding):
    bound(c) := exists e: evidence(e) AND supports(e, c)

Definition 10.3 (Admissibility):
    admissible(c) := tag(c) in {VERIFIED, PLANNED, DERIVED} AND bound(c)

Theorem 10.1 (Export Restriction):
    Untagged or unbound claims are not exportable through the membrane.

This implements the CLAIM_MUST_BIND epistemology: no claim exits the
node without evidence. HYPOTHETICAL claims are allowed locally but
cannot cross the membrane to the URP.

Standing on Giants:
- Popper (1934): Falsifiability — claims must be testable
- Al-Ghazali (1095): Knowledge classification — verified vs. opinion
- Dijkstra (1970): "Testing shows presence, not absence, of bugs"
- BIZRA Constitution: ZANN_ZERO — no unverified claims in URP
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class ClaimTag(Enum):
    """Epistemic status of a claim."""

    VERIFIED = "verified"  # evidence exists and has been checked
    PLANNED = "planned"  # future work, bound to a roadmap item
    DERIVED = "derived"  # logically follows from verified premises
    HYPOTHETICAL = "hypothetical"  # speculation — cannot cross membrane


# Tags that are admissible for membrane crossing
ADMISSIBLE_TAGS = frozenset({ClaimTag.VERIFIED, ClaimTag.PLANNED, ClaimTag.DERIVED})


@dataclass
class Evidence:
    """A piece of evidence supporting a claim."""

    evidence_id: str
    source: str  # where the evidence comes from
    receipt_hash: str = ""  # BLAKE3 hash linking to evidence ledger


@dataclass
class Claim:
    """A tagged, potentially bound claim."""

    claim_id: str
    text: str
    tag: ClaimTag
    evidence: List[Evidence]

    @property
    def bound(self) -> bool:
        """A claim is bound if it has at least one evidence item."""
        return len(self.evidence) > 0


@dataclass
class AdmissibilityResult:
    """Result of admissibility check."""

    claim_id: str
    admissible: bool
    tag_ok: bool
    bound_ok: bool
    reason: str


def check_admissibility(claim: Claim) -> AdmissibilityResult:
    """Check if a claim is admissible for membrane crossing.

    Theorem 10.1: admissible(c) := tag(c) in ADMISSIBLE_TAGS AND bound(c)
    """
    tag_ok = claim.tag in ADMISSIBLE_TAGS
    bound_ok = claim.bound

    if not tag_ok and not bound_ok:
        reason = f"tag={claim.tag.value} not admissible AND no evidence"
    elif not tag_ok:
        reason = f"tag={claim.tag.value} not admissible for export"
    elif not bound_ok:
        reason = "claim has no evidence (CLAIM_MUST_BIND)"
    else:
        reason = "admissible"

    return AdmissibilityResult(
        claim_id=claim.claim_id,
        admissible=tag_ok and bound_ok,
        tag_ok=tag_ok,
        bound_ok=bound_ok,
        reason=reason,
    )


def filter_exportable(
    claims: List[Claim],
) -> tuple[List[Claim], List[AdmissibilityResult]]:
    """Filter a list of claims, returning only admissible ones.

    Returns (exportable_claims, all_results) so the caller can
    inspect why claims were rejected.
    """
    exportable: List[Claim] = []
    results: List[AdmissibilityResult] = []

    for claim in claims:
        result = check_admissibility(claim)
        results.append(result)
        if result.admissible:
            exportable.append(claim)

    return exportable, results
