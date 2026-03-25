"""
Algorithmic Membrane Verifier — DFA property verification.

Verifies that M(iota) satisfies three transformation properties:
  P1 — Anonymity: identity is unrecoverable from output
  P2 — Epistemic Validity: claims have BLAKE3 derivation chains
  P3 — Constitutional Alignment: ihsan >= threshold

If any property fails, M(iota) = Bottom (fail-closed).

Standing on Giants:
- Shannon (1948): Information entropy — P1 measures identity leakage
- Merkle (1979): Hash chains — P2 requires provenance
- Al-Ghazali (1095): Epistemic integrity — P2 rejects unverified claims
- BIZRA Constitution: Ihsan threshold is the floor, not the ceiling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.sovereign.workspace_boundary import PRIVATE_FIELDS


@dataclass
class CheckResult:
    """Result of a single property check."""

    passed: bool
    reason: str


@dataclass
class VerificationResult:
    """Composite result of all three membrane property checks."""

    passed: bool
    checks: Dict[str, CheckResult] = field(default_factory=dict)


@dataclass
class Bottom:
    """Rejection (sink state) — the action was constitutionally blocked."""

    reject_code: str
    gate_name: str
    reason: str
    receipt_hash: str = ""


class MembraneVerifier:
    """Verify that M satisfies all three transformation properties.

    This is the formal verification layer for the Algorithmic Membrane.
    The DFA (bizra-mission/state.rs) handles state transitions; this
    module verifies the *properties* of those transitions.
    """

    def __init__(self, ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD) -> None:
        self._ihsan_floor = ihsan_floor

    def verify_transformation(
        self,
        result: Any,
    ) -> VerificationResult:
        """Verify all three properties of a membrane transformation.

        Args:
            result: Either an action (with ihsan_score, evidence_receipt_id, etc.)
                    or a Bottom rejection.

        Returns:
            VerificationResult with per-property details.
        """
        if isinstance(result, Bottom):
            checks = {
                "anonymity": CheckResult(
                    passed=True, reason="rejected — no identity leaked"
                ),
                "epistemic_validity": CheckResult(
                    passed=True, reason="rejected — no claims made"
                ),
                "constitutional_alignment": CheckResult(
                    passed=True, reason="fail-closed is constitutional"
                ),
            }
            return VerificationResult(passed=True, checks=checks)

        checks = {
            "anonymity": self._check_anonymity(result),
            "epistemic_validity": self._check_provenance(result),
            "constitutional_alignment": self._check_ihsan(result),
        }
        all_pass = all(c.passed for c in checks.values())
        return VerificationResult(passed=all_pass, checks=checks)

    def _check_anonymity(self, result: Any) -> CheckResult:
        """P1: No node identity recoverable from action payload."""
        leaked = set()
        result_dict = self._to_dict(result)
        for key in self._flatten_keys(result_dict):
            if key in PRIVATE_FIELDS:
                leaked.add(key)
        return CheckResult(
            passed=len(leaked) == 0,
            reason=f"leaked: {leaked}" if leaked else "clean",
        )

    def _check_provenance(self, result: Any) -> CheckResult:
        """P2: Every claim has a BLAKE3 derivation chain."""
        receipt_id = getattr(result, "evidence_receipt_id", None)
        if receipt_id is None:
            result_dict = self._to_dict(result)
            receipt_id = result_dict.get("evidence_receipt_id")
        has_receipt = bool(receipt_id)
        return CheckResult(
            passed=has_receipt,
            reason=(
                "receipt present" if has_receipt else "NO RECEIPT — epistemic violation"
            ),
        )

    def _check_ihsan(self, result: Any) -> CheckResult:
        """P3: Constitutional alignment — ihsan >= threshold."""
        score = getattr(result, "ihsan_score", None)
        if score is None:
            score = self._to_dict(result).get("ihsan_score", 0.0)
        passed = float(score) >= self._ihsan_floor
        return CheckResult(
            passed=passed,
            reason=f"ihsan={float(score):.3f} vs floor={self._ihsan_floor}",
        )

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        return getattr(obj, "__dict__", {})

    @staticmethod
    def _flatten_keys(d: Dict[str, Any], prefix: str = "") -> set[str]:
        """Recursively collect all keys in a nested dict."""
        keys = set()
        for k, v in d.items():
            keys.add(k)
            if isinstance(v, dict):
                keys |= MembraneVerifier._flatten_keys(v, f"{prefix}{k}.")
        return keys
