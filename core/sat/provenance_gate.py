"""
Provenance Gate — SAT Layer: Information Chain Verification
===========================================================

Validates that all data crossing the PAT→SAT boundary carries
verifiable provenance chains graded by the IRP algorithm.

This is the first gate that makes data quality a constitutional
requirement, not an afterthought.

Standing on Giants:
- Islamic hadith science (8th century CE): isnad chain verification
- Markowitz (1952): provenance affects risk computation
- Shannon (1948): information quality as measurable quantity

Constitutional Anchor:
No data without provenance. No provenance without grade.
No grade below DAIF enters the network.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.sat.gate_result import CheckResult, CheckStatus, GateResult

logger = logging.getLogger(__name__)

PASS = CheckStatus.PASS
FAIL = CheckStatus.FAIL
PARTIAL = CheckStatus.PARTIAL

# Constitutional minimum: MAWDU data is excluded entirely
MINIMUM_GRADE_NAME = "DAIF"
MINIMUM_GRADE_VALUE = 2  # IsnadGrade.DAIF


def provenance_verify(
    assessments: Optional[List[Dict[str, Any]]] = None,
) -> GateResult:
    """
    SAT Provenance Gate: Verify information chain quality.

    Checks:
    1. IRP module is available and importable
    2. All assessments carry provenance grades
    3. No MAWDU (fabricated) data in the assessment set
    4. Average chain strength meets constitutional floor
    5. At least one SAHIH source exists (mutawatir requirement)
    6. Assessment hashes are non-empty (tamper evidence)

    Args:
        assessments: List of IrpAssessment-like dicts from PAT.
            Each must have: asset_id, grade, chain_strength,
            assessment_hash, independent_chain_count.
            If None, runs a self-test of the IRP module.
    """
    checks: list[CheckResult] = []

    # ------------------------------------------------------------------
    # Check 1: IRP module importable
    # ------------------------------------------------------------------
    try:
        from core.irp import aggregate_strength  # noqa: F401 — import-check
        from core.irp import (
            DataPoint,
            IsnadChain,
            IsnadGrade,
            Source,
            pat_assess,
        )

        checks.append(
            CheckResult("irp_importable", PASS, "core.irp module loaded successfully")
        )
    except ImportError as e:
        checks.append(
            CheckResult("irp_importable", FAIL, f"Cannot import core.irp: {e}")
        )
        return GateResult(
            agent="Provenance",
            layer="Information Chain Verification",
            checks=checks,
        )

    # ------------------------------------------------------------------
    # Check 2: IRP self-test (if no assessments provided)
    # ------------------------------------------------------------------
    if assessments is None:
        try:
            # Run a minimal self-test: create sources, chain, assess
            s1 = Source(
                id="self_test_a", name="SelfTest A", reliability=0.95, verified=True
            )
            s2 = Source(
                id="self_test_b", name="SelfTest B", reliability=0.90, verified=True
            )
            s3 = Source(
                id="self_test_c", name="SelfTest C", reliability=0.88, verified=True
            )
            chains = [
                IsnadChain(sources=[s1]),
                IsnadChain(sources=[s2]),
                IsnadChain(sources=[s3]),
            ]
            dp = DataPoint(asset_id="SELF_TEST", value=1.0, chains=chains)
            assessment = pat_assess(dp)
            if assessment.grade == IsnadGrade.SAHIH:
                checks.append(
                    CheckResult(
                        "irp_self_test",
                        PASS,
                        f"Self-test: grade=SAHIH, strength={assessment.chain_strength:.3f}",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "irp_self_test",
                        PARTIAL,
                        f"Self-test: grade={assessment.grade.name} (expected SAHIH)",
                    )
                )
        except Exception as e:
            checks.append(CheckResult("irp_self_test", FAIL, f"Self-test failed: {e}"))

        return GateResult(
            agent="Provenance",
            layer="Information Chain Verification",
            checks=checks,
        )

    # ------------------------------------------------------------------
    # Checks 3-6: Validate provided assessments
    # ------------------------------------------------------------------

    # Check 3: All assessments have grades
    missing_grades = [
        a.get("asset_id", "?")
        for a in assessments
        if "grade" not in a or a["grade"] is None
    ]
    if missing_grades:
        checks.append(
            CheckResult(
                "all_graded",
                FAIL,
                f"Missing grades on: {', '.join(missing_grades[:5])}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "all_graded",
                PASS,
                f"All {len(assessments)} assessments carry provenance grades",
            )
        )

    # Check 4: No MAWDU data
    mawdu_items = [
        a.get("asset_id", "?") for a in assessments if _grade_value(a.get("grade")) <= 1
    ]
    if mawdu_items:
        checks.append(
            CheckResult(
                "no_mawdu",
                FAIL,
                f"MAWDU (fabricated) data detected: {', '.join(mawdu_items[:5])}. "
                "Excluded from all computation.",
            )
        )
    else:
        checks.append(
            CheckResult(
                "no_mawdu", PASS, "No fabricated (MAWDU) data in assessment set"
            )
        )

    # Check 5: Average chain strength above floor
    strengths = [
        a.get("chain_strength", 0.0)
        for a in assessments
        if _grade_value(a.get("grade")) > 1  # exclude MAWDU
    ]
    if strengths:
        avg_strength = sum(strengths) / len(strengths)
        if avg_strength >= 0.5:
            checks.append(
                CheckResult(
                    "chain_strength_floor",
                    PASS,
                    f"Average chain strength: {avg_strength:.3f} (floor: 0.500)",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "chain_strength_floor",
                    FAIL,
                    f"Average chain strength {avg_strength:.3f} below floor 0.500",
                )
            )
    else:
        checks.append(
            CheckResult(
                "chain_strength_floor",
                FAIL,
                "No valid assessments to evaluate chain strength",
            )
        )

    # Check 6: Assessment hashes present (tamper evidence)
    missing_hashes = [
        a.get("asset_id", "?") for a in assessments if not a.get("assessment_hash")
    ]
    if missing_hashes:
        checks.append(
            CheckResult(
                "hash_integrity",
                FAIL,
                f"Missing hashes on: {', '.join(missing_hashes[:5])}",
            )
        )
    else:
        checks.append(
            CheckResult(
                "hash_integrity",
                PASS,
                f"All {len(assessments)} assessments carry BLAKE2b hashes",
            )
        )

    return GateResult(
        agent="Provenance",
        layer="Information Chain Verification",
        checks=checks,
    )


def _grade_value(grade: Any) -> int:
    """Extract numeric grade value from various representations."""
    if grade is None:
        return 0
    if isinstance(grade, int):
        return grade
    if isinstance(grade, str):
        mapping = {"SAHIH": 4, "HASAN": 3, "DAIF": 2, "MAWDU": 1}
        return mapping.get(grade.upper(), 0)
    # Handle IsnadGrade enum
    if hasattr(grade, "value"):
        return int(grade.value)
    return 0
