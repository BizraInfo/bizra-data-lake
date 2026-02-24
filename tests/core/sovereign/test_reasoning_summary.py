"""
Tests for ReasoningSummary and GoTNodeSnapshot (Task 3.1).

Covers:
- GoTNodeSnapshot creation and serialization
- ReasoningSummary construction and to_dict()
- Payload capping (10 nodes max, 256 chars per node)
- Integration with SovereignResult.to_dict()
- Reasoning summary present in pipeline results

Standing on Giants: Besta (GoT, 2024) · Shannon (SNR) · Al-Ghazali (Ihsān)
"""

from __future__ import annotations

from core.sovereign.runtime_types import (
    GoTNodeSnapshot,
    ReasoningSummary,
    SovereignResult,
)


class TestGoTNodeSnapshot:
    def test_creation(self) -> None:
        node = GoTNodeSnapshot(
            node_id="got_abc_0",
            content="Analyze the user query for intent",
            score=0.85,
            depth=0,
            is_conclusion=False,
        )
        assert node.node_id == "got_abc_0"
        assert node.score == 0.85
        assert not node.is_conclusion

    def test_conclusion_node(self) -> None:
        node = GoTNodeSnapshot(
            node_id="got_abc_3",
            content="Final synthesis of all reasoning paths",
            score=0.95,
            depth=3,
            is_conclusion=True,
            parent_id="got_abc_2",
        )
        assert node.is_conclusion
        assert node.parent_id == "got_abc_2"


class TestReasoningSummary:
    def test_empty_summary(self) -> None:
        summary = ReasoningSummary()
        d = summary.to_dict()
        assert d["got_nodes"] == []
        assert d["alternatives_considered"] == 0
        assert d["confidence"] == 0.0

    def test_populated_summary(self) -> None:
        nodes = [
            GoTNodeSnapshot(
                node_id=f"n{i}",
                content=f"Thought {i}",
                score=0.7 + i * 0.05,
                depth=i,
                is_conclusion=(i == 2),
                parent_id=f"n{i-1}" if i > 0 else None,
            )
            for i in range(3)
        ]
        summary = ReasoningSummary(
            got_nodes=nodes,
            agent_scores={"strategist": 0.92, "guardian": 0.88},
            alternatives_considered=2,
            convergence_reason="GoT depth 3, SNR 0.91",
            total_reasoning_ms=1234.5,
            confidence=0.87,
            guardian_verdicts={"constitutional": "APPROVED"},
            model_used="qwen2.5:7b",
        )
        d = summary.to_dict()
        assert len(d["got_nodes"]) == 3
        assert d["got_nodes"][2]["is_conclusion"]
        assert d["agent_scores"]["strategist"] == 0.92
        assert d["alternatives_considered"] == 2
        assert d["confidence"] == 0.87
        assert d["guardian_verdicts"]["constitutional"] == "APPROVED"
        assert d["model_used"] == "qwen2.5:7b"

    def test_payload_capping_10_nodes(self) -> None:
        """More than 10 GoT nodes are capped at 10 in serialization."""
        nodes = [
            GoTNodeSnapshot(node_id=f"n{i}", content=f"Node {i}", depth=i)
            for i in range(15)
        ]
        summary = ReasoningSummary(got_nodes=nodes)
        d = summary.to_dict()
        assert len(d["got_nodes"]) == 10

    def test_content_truncation_256_chars(self) -> None:
        """Node content is truncated to 256 chars in serialization."""
        long_content = "x" * 500
        node = GoTNodeSnapshot(node_id="n0", content=long_content)
        summary = ReasoningSummary(got_nodes=[node])
        d = summary.to_dict()
        assert len(d["got_nodes"][0]["content"]) == 256

    def test_convergence_reason_truncation(self) -> None:
        """Convergence reason is truncated to 512 chars."""
        long_reason = "r" * 1000
        summary = ReasoningSummary(convergence_reason=long_reason)
        d = summary.to_dict()
        assert len(d["convergence_reason"]) == 512


class TestSovereignResultIntegration:
    def test_to_dict_includes_reasoning_summary(self) -> None:
        summary = ReasoningSummary(
            confidence=0.9,
            alternatives_considered=3,
            convergence_reason="Test convergence",
        )
        result = SovereignResult(
            query_id="test-123",
            success=True,
            response="Test answer",
            reasoning_summary=summary,
        )
        d = result.to_dict()
        assert d["reasoning_summary"] is not None
        assert d["reasoning_summary"]["confidence"] == 0.9
        assert d["reasoning_summary"]["alternatives_considered"] == 3

    def test_to_dict_without_reasoning_summary(self) -> None:
        result = SovereignResult(
            query_id="test-456",
            success=True,
            response="Simple answer",
        )
        d = result.to_dict()
        assert d["reasoning_summary"] is None

    def test_reasoning_summary_in_full_result(self) -> None:
        """Full SovereignResult with all fields populated."""
        nodes = [
            GoTNodeSnapshot(node_id="g0", content="Analyze", score=0.8, depth=0),
            GoTNodeSnapshot(
                node_id="g1",
                content="Conclude",
                score=0.95,
                depth=1,
                is_conclusion=True,
                parent_id="g0",
            ),
        ]
        summary = ReasoningSummary(
            got_nodes=nodes,
            agent_scores={"analyst": 0.91},
            alternatives_considered=1,
            convergence_reason="Direct path",
            total_reasoning_ms=500.0,
            confidence=0.92,
            guardian_verdicts={"constitutional": "APPROVED"},
            model_used="qwen2.5:7b",
        )
        result = SovereignResult(
            query_id="full-test",
            success=True,
            response="Full answer",
            reasoning_used=True,
            reasoning_depth=2,
            thoughts=["Analyze", "Conclude"],
            ihsan_score=0.96,
            snr_score=0.91,
            snr_ok=True,
            reasoning_summary=summary,
        )
        d = result.to_dict()
        assert d["success"]
        assert d["reasoning"]["depth"] == 2
        assert d["reasoning_summary"]["got_nodes"][1]["is_conclusion"]
        assert d["quality"]["ihsan_score"] == 0.96
