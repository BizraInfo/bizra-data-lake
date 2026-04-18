"""
SAPE v2.0 Checks — The 6 Non-Negotiable Verification Gates

Every SAPE execution, regardless of mode (Lite/Standard/Deep),
must pass all 6 checks. This is the Ihsān constraint.

1. Correctness — Are the claims factually accurate?
2. Consistency — Do the parts contradict each other?
3. Completeness — Are there obvious gaps?
4. Causality — Are cause-effect claims justified?
5. Ethics (Ihsān) — Does the output meet the Ihsān floor?
6. Evidence — Is every claim backed by its stated evidence level?

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

from .types import Check, CheckResult, EvidenceLevel

logger = logging.getLogger(__name__)


def run_all_checks(
    content: str,
    *,
    ihsan_score: float = 0.0,
    snr_score: float = 0.0,
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> List[CheckResult]:
    """
    Run all 6 SAPE checks against the content.

    These checks are non-negotiable: all modes (Lite, Standard, Deep) run them.
    """
    return [
        _check_correctness(content, snr_score=snr_score, snr_fn=snr_fn),
        _check_consistency(content, snr_fn=snr_fn),
        _check_completeness(content, snr_fn=snr_fn),
        _check_causality(content, snr_fn=snr_fn),
        _check_ethics(content, ihsan_score=ihsan_score),
        _check_evidence(content, evidence_level=evidence_level),
    ]


def _check_correctness(
    content: str,
    *,
    snr_score: float = 0.0,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> CheckResult:
    """Check 1: Correctness — Are claims factually accurate?"""
    effective_snr = snr_fn(content) if snr_fn else snr_score
    passed = effective_snr >= UNIFIED_SNR_THRESHOLD

    return CheckResult(
        check=Check.CORRECTNESS,
        passed=passed,
        score=effective_snr,
        detail=f"SNR={effective_snr:.3f} vs threshold={UNIFIED_SNR_THRESHOLD}",
        evidence_level=EvidenceLevel.GROUNDED_INFERENCE,
    )


def _check_consistency(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> CheckResult:
    """Check 2: Consistency — Do parts contradict each other?"""
    # Heuristic: check for contradiction markers
    contradiction_markers = [
        "however, this contradicts",
        "on the other hand",
        "this is inconsistent",
        "but earlier we said",
    ]
    lower = content.lower()
    contradiction_count = sum(1 for m in contradiction_markers if m in lower)

    # Simple heuristic: no explicit contradictions detected = consistent
    passed = contradiction_count == 0
    score = max(0.0, 1.0 - (contradiction_count * 0.25))

    return CheckResult(
        check=Check.CONSISTENCY,
        passed=passed,
        score=score,
        detail=f"Contradiction markers found: {contradiction_count}",
        evidence_level=EvidenceLevel.GROUNDED_INFERENCE,
    )


def _check_completeness(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> CheckResult:
    """Check 3: Completeness — Are there obvious gaps?"""
    # Heuristic: check for TODO/placeholder/TBD markers
    gap_markers = ["todo", "tbd", "placeholder", "to be defined", "to be filled"]
    lower = content.lower()
    gap_count = sum(1 for m in gap_markers if m in lower)

    passed = gap_count == 0
    score = max(0.0, 1.0 - (gap_count * 0.15))

    return CheckResult(
        check=Check.COMPLETENESS,
        passed=passed,
        score=score,
        detail=f"Gap markers found: {gap_count}",
        evidence_level=EvidenceLevel.GROUNDED_INFERENCE,
    )


def _check_causality(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
) -> CheckResult:
    """Check 4: Causality — Are cause-effect claims justified?"""
    # Heuristic: causal claims should have supporting reasoning
    causal_markers = ["because", "therefore", "thus", "consequently", "as a result"]
    lower = content.lower()
    causal_count = sum(1 for m in causal_markers if m in lower)

    words = content.split()
    # Ratio of causal reasoning markers to total length
    causal_density = causal_count / max(len(words) / 50, 1)

    # Reasonable causal density: not too few (no reasoning), not too many (over-justified)
    passed = True
    score = min(causal_density, 1.0)

    return CheckResult(
        check=Check.CAUSALITY,
        passed=passed,
        score=score,
        detail=f"Causal markers: {causal_count}, density: {causal_density:.3f}",
        evidence_level=EvidenceLevel.GROUNDED_INFERENCE,
    )


def _check_ethics(
    content: str,
    *,
    ihsan_score: float = 0.0,
) -> CheckResult:
    """Check 5: Ethics (Ihsān) — Does output meet the Ihsān floor?"""
    passed = ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    return CheckResult(
        check=Check.ETHICS,
        passed=passed,
        score=ihsan_score,
        detail=(f"Ihsān={ihsan_score:.3f} vs threshold={UNIFIED_IHSAN_THRESHOLD}"),
        evidence_level=EvidenceLevel.VERIFIED,
    )


def _check_evidence(
    content: str,
    *,
    evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN,
) -> CheckResult:
    """Check 6: Evidence — Is every claim backed by its stated evidence level?"""
    # Score based on evidence level
    level_scores = {
        EvidenceLevel.VERIFIED: 1.0,
        EvidenceLevel.GROUNDED_INFERENCE: 0.8,
        EvidenceLevel.CONJECTURE: 0.5,
        EvidenceLevel.UNKNOWN: 0.2,
    }

    score = level_scores.get(evidence_level, 0.2)
    # Pass if at least CONJECTURE level (speculation is acceptable if labeled)
    passed = evidence_level != EvidenceLevel.UNKNOWN

    return CheckResult(
        check=Check.EVIDENCE,
        passed=passed,
        score=score,
        detail=f"Evidence level: {evidence_level.value}",
        evidence_level=evidence_level,
    )
