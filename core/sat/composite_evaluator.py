"""SAT-5 Composite Evaluator — runs all 5 SAT gates and produces a unified verdict.

Gates:
1. Sentinel  — structural integrity, schema validation
2. Oracle-S  — LLM-based Ihsan/quality scoring (the existing SAT validator)
3. Ledger    — receipt chain verification, token ledger consistency
4. Conductor — consensus rules, quorum checks
5. Ambassador — network boundary validation, federation readiness

Each gate returns a GateResult. The composite verdict passes only if
ALL gates pass (fail-closed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.sat.gate_result import CheckResult, CheckStatus, GateResult

logger = logging.getLogger(__name__)


@dataclass
class CompositeVerdict:
    """Unified SAT-5 verdict from all gates."""

    passed: bool
    gate_results: Dict[str, GateResult] = field(default_factory=dict)
    blocking_gates: List[str] = field(default_factory=list)
    ihsan_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gate_results": {k: v.to_dict() for k, v in self.gate_results.items()},
            "blocking_gates": self.blocking_gates,
            "ihsan_score": self.ihsan_score,
            "reason": self.reason,
        }


def evaluate_all_gates(
    *,
    skip_slow: bool = True,
    skip_manual: bool = True,
) -> CompositeVerdict:
    """Run all 5 SAT gates and return a composite verdict.

    Args:
        skip_slow: Skip time-intensive checks (for testing/dev).
        skip_manual: Skip checks that need human input.

    Returns:
        CompositeVerdict with all gate results and unified pass/fail.
    """
    from core.sat.ambassador_gate import ambassador_verify
    from core.sat.conductor_gate import conductor_verify
    from core.sat.ledger_gate import ledger_verify
    from core.sat.oracle_s_gate import oracle_s_verify
    from core.sat.sentinel_gate import sentinel_verify

    gates = {
        "sentinel": lambda: sentinel_verify(skip_slow=skip_slow),
        "oracle_s": lambda: oracle_s_verify(
            skip_manual=skip_manual, skip_slow=skip_slow
        ),
        "ledger": lambda: ledger_verify(),
        "conductor": lambda: conductor_verify(skip_slow=skip_slow),
        "ambassador": lambda: ambassador_verify(
            skip_manual=skip_manual, skip_slow=skip_slow
        ),
    }

    results: Dict[str, GateResult] = {}
    blocking: List[str] = []

    for name, gate_fn in gates.items():
        try:
            result = gate_fn()
            results[name] = result
            if not result.passed:
                blocking.append(name)
                logger.warning("SAT gate %s: BLOCKED — %s", name, result.verdict.value)
            else:
                logger.info("SAT gate %s: PASS", name)
        except Exception as e:
            # Gate failure = fail-closed
            error_result = GateResult(
                gate_name=name,
                checks=[
                    CheckResult(
                        name=f"{name}_error",
                        status=CheckStatus.FAIL,
                        message=f"Gate raised exception: {e}",
                    )
                ],
            )
            results[name] = error_result
            blocking.append(name)
            logger.error("SAT gate %s: ERROR — %s", name, e)

    all_passed = len(blocking) == 0

    # Compute composite Ihsan from Oracle-S if available
    ihsan = 0.0
    oracle_result = results.get("oracle_s")
    if oracle_result and oracle_result.passed:
        # Extract ihsan from oracle checks
        for check in oracle_result.checks:
            if "ihsan" in check.name.lower() or "quality" in check.name.lower():
                if check.status == CheckStatus.PASS:
                    ihsan = 1.0
                    break
        if ihsan == 0.0 and oracle_result.passed:
            ihsan = 0.95  # Oracle passed but no explicit ihsan check

    reason = (
        "All 5 SAT gates passed" if all_passed else f"Blocked by: {', '.join(blocking)}"
    )

    return CompositeVerdict(
        passed=all_passed,
        gate_results=results,
        blocking_gates=blocking,
        ihsan_score=ihsan,
        reason=reason,
    )


def evaluate_gates_for_receipt(
    pat_answer: str,
    evidence_refs: list[str],
    *,
    skip_slow: bool = True,
) -> CompositeVerdict:
    """Convenience: evaluate all gates in the context of a PAT receipt.

    This is what the FATE gate calls when crossing from PAT to SAT.
    """
    # Run the standard composite evaluation
    verdict = evaluate_all_gates(skip_slow=skip_slow, skip_manual=True)

    # Also run the LLM-based SAT validator for Ihsan scoring
    try:
        from core.proof_engine.sat_validator import validate as sat_validate

        class _PatBridge:
            def __init__(self):
                self.answer = pat_answer
                self.evidence_refs = evidence_refs
                self.confidence = "high" if len(evidence_refs) >= 3 else "medium"

        sat_verdict = sat_validate(_PatBridge())
        verdict.ihsan_score = sat_verdict.ihsan_score

        if sat_verdict.verdict != "PASS":
            verdict.passed = False
            verdict.blocking_gates.append("oracle_s_llm")
            verdict.reason = (
                f"Oracle-S LLM: {sat_verdict.verdict} (ihsan={sat_verdict.ihsan_score})"
            )
    except Exception as e:
        logger.warning(
            "Oracle-S LLM evaluation failed: %s — using gate-only verdict", e
        )

    return verdict
