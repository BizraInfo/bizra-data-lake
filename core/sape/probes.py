"""
SAPE v2.0 Probes — The 9 Divergence Probes

The probes fire during the Diverge pass to expand the search space
before convergence. Deep mode runs all 9; Standard runs a targeted
subset; Lite runs none (relies on checks alone).

1. Counterfactual — What if the opposite were true?
2. Boundary — What happens at the edges?
3. Analogical — What similar problem has been solved?
4. Formalization — Can we express this mathematically?
5. Program Sketch — Can we write pseudocode with pre/postconditions?
6. Compression — What is the minimal representation?
7. Expansion — What are the full implications?
8. Adversarial — What is the strongest attack on this?
9. Ethical Overlay — Does this serve human flourishing?

Created: 2026-04-10 | BIZRA SAPE v2.0
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

from .types import Probe, ProbeResult

logger = logging.getLogger(__name__)


def run_probes(
    content: str,
    probes: List[Probe],
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> List[ProbeResult]:
    """
    Run the specified probes against content.

    The probe list is determined by the execution mode:
    - Lite: no probes
    - Standard: [Counterfactual, Boundary, Adversarial]
    - Deep: all 9
    """
    dispatch = {
        Probe.COUNTERFACTUAL: _probe_counterfactual,
        Probe.BOUNDARY: _probe_boundary,
        Probe.ANALOGICAL: _probe_analogical,
        Probe.FORMALIZATION: _probe_formalization,
        Probe.PROGRAM_SKETCH: _probe_program_sketch,
        Probe.COMPRESSION: _probe_compression,
        Probe.EXPANSION: _probe_expansion,
        Probe.ADVERSARIAL: _probe_adversarial,
        Probe.ETHICAL_OVERLAY: _probe_ethical_overlay,
    }

    results = []
    for probe in probes:
        fn = dispatch.get(probe)
        if fn:
            results.append(fn(content, snr_fn=snr_fn, ihsan_score=ihsan_score))
        else:
            logger.warning("Unknown probe: %s", probe)

    return results


# ═══════════════════════════════════════════════════════════════
# Individual Probe Implementations
# ═══════════════════════════════════════════════════════════════


def _probe_counterfactual(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 1: What if the opposite were true?"""
    findings = (
        "Counterfactual probe: Invert the primary claim. "
        "If the negation is plausible, the original claim needs stronger evidence."
    )
    return ProbeResult(
        probe=Probe.COUNTERFACTUAL,
        findings=findings,
        score=0.5,
        flagged=False,
    )


def _probe_boundary(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 2: What happens at the edges?"""
    # Check for boundary-aware language
    boundary_markers = [
        "edge case",
        "limit",
        "overflow",
        "underflow",
        "maximum",
        "minimum",
    ]
    lower = content.lower()
    has_boundary_awareness = any(m in lower for m in boundary_markers)

    flagged = not has_boundary_awareness
    findings = (
        "Boundary probe: Check extreme inputs, empty cases, maximum scale. "
        f"Boundary awareness detected: {has_boundary_awareness}."
    )
    return ProbeResult(
        probe=Probe.BOUNDARY,
        findings=findings,
        score=0.8 if has_boundary_awareness else 0.3,
        flagged=flagged,
    )


def _probe_analogical(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 3: What similar problem has been solved?"""
    findings = (
        "Analogical probe: Search for solved analogues in adjacent domains. "
        "Transfer the solution structure, not surface details."
    )
    return ProbeResult(
        probe=Probe.ANALOGICAL,
        findings=findings,
        score=0.5,
        flagged=False,
    )


def _probe_formalization(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 4: Can we express this mathematically?"""
    # Check for formal/mathematical language
    formal_markers = [
        "∀",
        "∃",
        "→",
        "⇒",
        "≤",
        "≥",
        "proof",
        "theorem",
        "lemma",
        "invariant",
    ]
    lower = content.lower()
    has_formalism = any(m in lower or m in content for m in formal_markers)

    findings = (
        "Formalization probe: Attempt to express core claims in formal notation. "
        f"Existing formal language detected: {has_formalism}."
    )
    return ProbeResult(
        probe=Probe.FORMALIZATION,
        findings=findings,
        score=0.7 if has_formalism else 0.3,
        flagged=not has_formalism,
    )


def _probe_program_sketch(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 5: Can we write pseudocode with pre/postconditions?"""
    code_markers = ["def ", "function", "class ", "if ", "for ", "while ", "return"]
    lower = content.lower()
    has_code = any(m in lower for m in code_markers)

    findings = (
        "Program Sketch probe: Express the solution as pseudocode with "
        "explicit preconditions and postconditions. "
        f"Code-like content detected: {has_code}."
    )
    return ProbeResult(
        probe=Probe.PROGRAM_SKETCH,
        findings=findings,
        score=0.6 if has_code else 0.4,
        flagged=False,
    )


def _probe_compression(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 6: What is the minimal representation?"""
    words = content.split()
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)

    # Flag if content seems unnecessarily verbose (low unique ratio)
    flagged = unique_ratio < 0.4 and word_count > 50

    findings = (
        f"Compression probe: {word_count} words, {unique_ratio:.2f} unique ratio. "
        f"Can the core message be expressed in fewer words without loss?"
    )
    return ProbeResult(
        probe=Probe.COMPRESSION,
        findings=findings,
        score=min(unique_ratio + 0.2, 1.0),
        flagged=flagged,
    )


def _probe_expansion(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 7: What are the full implications?"""
    findings = (
        "Expansion probe: Trace second and third-order consequences. "
        "What downstream effects does this create? "
        "What feedback loops emerge?"
    )
    return ProbeResult(
        probe=Probe.EXPANSION,
        findings=findings,
        score=0.5,
        flagged=False,
    )


def _probe_adversarial(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 8: What is the strongest attack on this?"""
    findings = (
        "Adversarial probe: Assume a competent adversary. "
        "What is the single strongest objection or attack vector? "
        "Can the solution survive it?"
    )
    return ProbeResult(
        probe=Probe.ADVERSARIAL,
        findings=findings,
        score=0.5,
        flagged=False,
    )


def _probe_ethical_overlay(
    content: str,
    *,
    snr_fn: Optional[Callable[[str], float]] = None,
    ihsan_score: float = 0.0,
) -> ProbeResult:
    """Probe 9: Does this serve human flourishing?"""
    passed = ihsan_score >= UNIFIED_IHSAN_THRESHOLD
    flagged = not passed

    findings = (
        f"Ethical Overlay probe: Ihsān score = {ihsan_score:.3f} "
        f"(threshold = {UNIFIED_IHSAN_THRESHOLD}). "
        f"{'Meets' if passed else 'BELOW'} ethical excellence floor."
    )
    return ProbeResult(
        probe=Probe.ETHICAL_OVERLAY,
        findings=findings,
        score=ihsan_score,
        flagged=flagged,
    )
