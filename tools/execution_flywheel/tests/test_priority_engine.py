"""Tests for the adaptive priority engine."""

from __future__ import annotations

import json
import unittest

from tools.execution_flywheel.priority_engine import recommend_from_json, recommend_priority
from tools.execution_flywheel.schemas import PrioritySignal


class RecommendPriorityTests(unittest.TestCase):
    def test_security_wins_on_secret_finding(self) -> None:
        signal = recommend_priority({"secret_findings": 3})
        self.assertEqual(signal.priority, "SECURITY")
        self.assertIn("secret_findings=3", signal.evidence)

    def test_security_wins_on_rotation_required(self) -> None:
        signal = recommend_priority({"secret_findings": 0, "rotation_required": True})
        self.assertEqual(signal.priority, "SECURITY")

    def test_runtime_hardening_over_public_claims(self) -> None:
        signal = recommend_priority(
            {
                "secret_findings": 0,
                "runtime_defaults_insecure": True,
                "public_claims_risky": True,
            }
        )
        self.assertEqual(signal.priority, "RUNTIME_HARDENING")

    def test_ci_baseline_when_main_red(self) -> None:
        signal = recommend_priority({"main_branch_red": True})
        self.assertEqual(signal.priority, "CI_BASELINE")
        self.assertIn("main_branch_red=True", signal.evidence)

    def test_ci_baseline_when_ci_failing(self) -> None:
        signal = recommend_priority({"ci_failing_count": 2})
        self.assertEqual(signal.priority, "CI_BASELINE")

    def test_supply_chain_on_vulns(self) -> None:
        signal = recommend_priority({"dependency_vulnerabilities": 2})
        self.assertEqual(signal.priority, "SUPPLY_CHAIN")

    def test_supply_chain_on_sbom_stale(self) -> None:
        signal = recommend_priority({"sbom_stale": True})
        self.assertEqual(signal.priority, "SUPPLY_CHAIN")

    def test_public_claims_when_secrets_cleared(self) -> None:
        signal = recommend_priority(
            {
                "secret_findings": 0,
                "rotation_required": False,
                "public_claims_risky": True,
            }
        )
        self.assertEqual(signal.priority, "PUBLIC_CLAIMS")
        self.assertIn("public_claims_risky=True", signal.evidence)

    def test_node0_activation_when_rows_blocked(self) -> None:
        signal = recommend_priority({"node0_activation_blocked_rows": 7})
        self.assertEqual(signal.priority, "NODE0_ACTIVATION")
        self.assertIn("node0_activation_blocked_rows=7", signal.evidence)

    def test_stop_and_land_on_clean_context(self) -> None:
        signal = recommend_priority({})
        self.assertEqual(signal.priority, "STOP_AND_LAND")

    def test_invalid_priority_rejected_in_schema(self) -> None:
        with self.assertRaises(ValueError):
            PrioritySignal(priority="NEBULA", reason="?", confidence=0.5)

    def test_confidence_range_validated(self) -> None:
        with self.assertRaises(ValueError):
            PrioritySignal(priority="SECURITY", reason="x", confidence=1.5)
        with self.assertRaises(ValueError):
            PrioritySignal(priority="SECURITY", reason="x", confidence=-0.1)

    def test_non_dict_context_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recommend_priority([])  # type: ignore[arg-type]


class RecommendFromJsonTests(unittest.TestCase):
    def test_parses_json_payload(self) -> None:
        payload = json.dumps({"secret_findings": 1})
        signal = recommend_from_json(payload)
        self.assertEqual(signal.priority, "SECURITY")

    def test_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            recommend_from_json(json.dumps([1, 2]))


if __name__ == "__main__":
    unittest.main()
