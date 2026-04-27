"""Tests for the flywheel runner combining guard + priority."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tools.execution_flywheel.flywheel_runner import (
    run_flywheel,
    run_from_json,
)
from tools.execution_flywheel.pattern_registry import load_patterns


class FlywheelRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.patterns = load_patterns(root / "patterns.yaml")

    def test_clean_context_proceeds_and_stops(self) -> None:
        result = run_flywheel({}, self.patterns)
        self.assertEqual(result.guard.decision, "PROCEED")
        self.assertEqual(result.priority.priority, "STOP_AND_LAND")

    def test_stale_review_fix_present_aborts(self) -> None:
        ctx = {
            "action_type": "edit_file",
            "triggers_detected": ["review_requests_change"],
            "metadata": {"fix_already_present": True},
            "priority_context": {},
        }
        result = run_flywheel(ctx, self.patterns)
        self.assertEqual(result.guard.decision, "ABORT")
        self.assertIn(
            "PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT",
            result.guard.matched_patterns,
        )
        self.assertEqual(result.priority.priority, "STOP_AND_LAND")

    def test_yaml_parse_context_revalidates(self) -> None:
        ctx = {
            "triggers_detected": ["audit_engine_crash", "yaml_typeerror_int_vs_str"],
            "metadata": {"reviewed_sha": "aaa", "head_sha": "bbb"},
            "priority_context": {"secret_findings": 0},
        }
        result = run_flywheel(ctx, self.patterns)
        self.assertEqual(result.guard.decision, "REVALIDATE")
        self.assertIn(
            "AUDIT_YAML_INLINE_COMMENT_PARSE_FAILURE",
            result.guard.matched_patterns,
        )

    def test_scanner_noise_context_revalidates_and_security(self) -> None:
        ctx = {
            "triggers_detected": ["high_secret_finding_count", "self_scan_matches"],
            "metadata": {},
            "priority_context": {"secret_findings": 42},
        }
        result = run_flywheel(ctx, self.patterns)
        self.assertEqual(result.guard.decision, "REVALIDATE")
        self.assertIn(
            "SECRET_SCANNER_SNR_NOISE_COLLAPSE",
            result.guard.matched_patterns,
        )
        self.assertEqual(result.priority.priority, "SECURITY")

    def test_credential_fallback_context_aborts(self) -> None:
        ctx = {
            "triggers_detected": ["default_dsn_or_redis_or_neo4j_fallback"],
            "metadata": {},
            "priority_context": {"runtime_defaults_insecure": True},
        }
        result = run_flywheel(ctx, self.patterns)
        self.assertEqual(result.guard.decision, "ABORT")
        self.assertIn(
            "DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT",
            result.guard.matched_patterns,
        )
        self.assertEqual(result.priority.priority, "RUNTIME_HARDENING")

    def test_public_claims_flow(self) -> None:
        ctx = {
            "triggers_detected": ["secret_findings_zero", "public_claims_risky"],
            "metadata": {},
            "priority_context": {"secret_findings": 0, "public_claims_risky": True},
        }
        result = run_flywheel(ctx, self.patterns)
        self.assertEqual(result.priority.priority, "PUBLIC_CLAIMS")
        self.assertEqual(result.guard.decision, "REVALIDATE")

    def test_run_from_json(self) -> None:
        payload = json.dumps({"priority_context": {"secret_findings": 1}})
        result = run_from_json(payload, self.patterns)
        self.assertEqual(result.priority.priority, "SECURITY")

    def test_invalid_context_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_flywheel([], self.patterns)  # type: ignore[arg-type]

    def test_invalid_priority_context_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_flywheel({"priority_context": "not a dict"}, self.patterns)

    def test_to_dict_roundtrip(self) -> None:
        result = run_flywheel({}, self.patterns)
        d = result.to_dict()
        self.assertEqual(d["guard"]["decision"], "PROCEED")
        self.assertEqual(d["priority"]["priority"], "STOP_AND_LAND")
        self.assertIn("explanations", d)


if __name__ == "__main__":
    unittest.main()
