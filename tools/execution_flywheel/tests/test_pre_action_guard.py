"""Tests for the pre-action guard decision engine."""

from __future__ import annotations

import json
import unittest

from tools.execution_flywheel.pre_action_guard import evaluate, evaluate_json
from tools.execution_flywheel.schemas import ActionContext, GuardDecision, Pattern


def _critical_pattern() -> Pattern:
    return Pattern.from_dict(
        {
            "pattern_id": "PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT",
            "name": "Verify origin branch before editing from review feedback",
            "severity": "critical",
            "triggers": [
                {"keyword": "review_requests_change"},
                {"keyword": "pr_has_commits_after_reviewed_sha"},
            ],
            "risks": ["duplicate fix"],
            "guard_actions": ["fetch origin", "inspect origin/<branch>:<file>"],
            "source": ["PR #49"],
        }
    )


def _minor_pattern() -> Pattern:
    return Pattern.from_dict(
        {
            "pattern_id": "STYLE_NITPICK",
            "name": "Cosmetic nitpick pattern",
            "severity": "minor",
            "triggers": [{"keyword": "style_nitpick_noted"}],
        }
    )


class EvaluateTests(unittest.TestCase):
    def test_already_fixed_returns_abort(self) -> None:
        ctx = ActionContext(
            action_type="edit_file",
            target_files=["core/bus/subscribers.py"],
            triggers_detected=["review_requests_change"],
            metadata={"fix_already_present": True},
        )
        d = evaluate(ctx, [_critical_pattern()])
        self.assertEqual(d.decision, "ABORT")
        self.assertIn("PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT", d.matched_patterns)
        self.assertIn("already present", d.reason)

    def test_stale_review_returns_revalidate(self) -> None:
        ctx = ActionContext(
            action_type="edit_file",
            target_files=["core/bus/subscribers.py"],
            triggers_detected=["review_requests_change"],
            metadata={"reviewed_sha": "34b27bec", "head_sha": "c09cc95c"},
        )
        d = evaluate(ctx, [_critical_pattern()])
        self.assertEqual(d.decision, "REVALIDATE")
        self.assertIn("34b27bec", d.reason)
        self.assertIn("c09cc95c", d.reason)

    def test_unrelated_context_returns_proceed(self) -> None:
        ctx = ActionContext(
            action_type="edit_file",
            target_files=["docs/foo.md"],
            triggers_detected=["docs_update"],
            metadata={},
        )
        d = evaluate(ctx, [_critical_pattern()])
        self.assertEqual(d.decision, "PROCEED")
        self.assertEqual(d.matched_patterns, [])

    def test_critical_match_insufficient_metadata_requires_operator(self) -> None:
        ctx = ActionContext(
            action_type="edit_file",
            target_files=["x.py"],
            triggers_detected=["review_requests_change"],
            metadata={},
        )
        d = evaluate(ctx, [_critical_pattern()])
        self.assertEqual(d.decision, "NEEDS_OPERATOR_CONFIRMATION")
        self.assertIn("PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT", d.matched_patterns)

    def test_minor_match_proceeds_without_operator(self) -> None:
        ctx = ActionContext(
            action_type="edit_file",
            target_files=["x.py"],
            triggers_detected=["style_nitpick_noted"],
            metadata={},
        )
        d = evaluate(ctx, [_minor_pattern()])
        self.assertEqual(d.decision, "PROCEED")
        self.assertEqual(d.matched_patterns, ["STYLE_NITPICK"])

    def test_fix_present_wins_over_sha_mismatch(self) -> None:
        ctx = ActionContext(
            action_type="edit_file",
            target_files=["x.py"],
            triggers_detected=["review_requests_change"],
            metadata={
                "fix_already_present": True,
                "reviewed_sha": "aaa",
                "head_sha": "bbb",
            },
        )
        d = evaluate(ctx, [_critical_pattern()])
        self.assertEqual(d.decision, "ABORT")

    def test_guard_decision_validates_value(self) -> None:
        with self.assertRaises(ValueError):
            GuardDecision(decision="MAYBE", reason="?", matched_patterns=[])


class EvaluateJsonTests(unittest.TestCase):
    def test_evaluates_from_json_payload(self) -> None:
        payload = json.dumps(
            {
                "action_type": "edit_file",
                "target_files": ["x.py"],
                "triggers_detected": ["review_requests_change"],
                "metadata": {"fix_already_present": True},
            }
        )
        d = evaluate_json(payload, [_critical_pattern()])
        self.assertEqual(d.decision, "ABORT")

    def test_rejects_non_object_payload(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_json(json.dumps([1, 2, 3]), [_critical_pattern()])


if __name__ == "__main__":
    unittest.main()
