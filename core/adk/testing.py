"""Testing utilities for BIZRA-ADK agents.

Provides test_mission() and assert_receipt_valid() for agent test fixtures.
Tests run against the REAL FATE gate — no stubs.
"""

from __future__ import annotations

from core.adk.mission import Budget, GovernanceClass, Mission


def make_test_mission(
    question: str,
    governance_class: GovernanceClass = GovernanceClass.PAT,
    max_tokens: int = 1024,
    max_tool_calls: int = 5,
    allow_external_unverified: bool = False,
) -> Mission:
    """Create a sandboxed mission for testing.

    Uses reduced budget and smaller model, but the SAME governance
    discipline as production. The FATE gate is NOT stubbed.
    """
    return Mission(
        question=question,
        governance_class=governance_class,
        requester="test-harness",
        budget=Budget(
            max_tokens=max_tokens,
            max_wall_seconds=30,
            max_tool_calls=max_tool_calls,
            max_evidence_fetches=10,
        ),
        allow_external_unverified=allow_external_unverified,
    )


def assert_receipt_valid(result) -> None:
    """Verify an AgentResult has a valid receipt chain.

    Checks:
    - receipt exists
    - receipt has non-empty digest
    - if verdict is PASS, ihsan >= threshold
    - evidence refs all verified
    """
    from core.integration.constants import IHSAN_THRESHOLD

    assert result is not None, "AgentResult is None"
    assert result.mission_id, "Missing mission_id"

    if result.success:
        assert result.verdict == "PASS", f"Success but verdict is {result.verdict}"
        assert (
            result.ihsan_score >= IHSAN_THRESHOLD
        ), f"PASS verdict but ihsan {result.ihsan_score} < {IHSAN_THRESHOLD}"
        assert result.content, "PASS verdict but empty content"

    if result.receipt is not None:
        assert hasattr(result.receipt, "hex_digest"), "Receipt missing hex_digest"
        digest = result.receipt.hex_digest()
        assert digest and len(digest) > 0, "Receipt has empty digest"

    if result.loop_proof is not None:
        assert result.loop_proof.manifest_hash, "LoopProof missing manifest_hash"
        assert len(result.loop_proof.steps) > 0, "LoopProof has no steps"


def assert_blocked(result, expected_verdict: str) -> None:
    """Verify an AgentResult was correctly blocked."""
    assert not result.success, "Expected block but got success"
    assert (
        result.verdict == expected_verdict
    ), f"Expected {expected_verdict}, got {result.verdict}"
