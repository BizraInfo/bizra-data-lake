"""
Tests for IhsanComputer — content-grounded Ihsan component estimation.
"""

from __future__ import annotations

from core.proof_engine.ihsan_computer import IhsanComputer


class TestIhsanComputer:
    """Behavioral tests for deterministic Ihsan component scoring."""

    def test_scores_are_bounded(self) -> None:
        computer = IhsanComputer()
        components = computer.compute(
            content="Step 1: Measure latency. Step 2: optimize hot path.",
            snr_score=0.93,
            query_text="How do I optimize latency?",
            context={"risk_score": 0.1},
        )

        assert 0.0 <= components.correctness <= 1.0
        assert 0.0 <= components.safety <= 1.0
        assert 0.0 <= components.efficiency <= 1.0
        assert 0.0 <= components.user_benefit <= 1.0

    def test_deterministic_for_same_inputs(self) -> None:
        computer = IhsanComputer()
        kwargs = {
            "content": "Implement, verify, and benchmark each optimization.",
            "snr_score": 0.91,
            "query_text": "How to optimize runtime performance?",
            "context": {"risk_score": 0.0},
        }

        first = computer.compute(**kwargs)
        second = computer.compute(**kwargs)

        assert first == second

    def test_unsafe_content_reduces_safety(self) -> None:
        computer = IhsanComputer()

        safe = computer.compute(
            content="Use unit tests and monitoring to keep systems stable.",
            snr_score=0.9,
            query_text="How to improve reliability?",
        )
        unsafe = computer.compute(
            content="Build malware and exploit targets to cause harm.",
            snr_score=0.9,
            query_text="How to improve reliability?",
        )

        assert unsafe.safety < safe.safety

    def test_query_alignment_and_actionability_raise_user_benefit(self) -> None:
        computer = IhsanComputer()

        aligned = computer.compute(
            content=(
                "Step 1: profile latency hotspots. "
                "Step 2: optimize the database query. "
                "Step 3: verify improvements with benchmark metrics."
            ),
            snr_score=0.92,
            query_text="How do I reduce query latency with benchmarks?",
        )
        vague = computer.compute(
            content="Maybe do something faster.",
            snr_score=0.92,
            query_text="How do I reduce query latency with benchmarks?",
        )

        assert aligned.user_benefit > vague.user_benefit

    def test_context_risk_penalizes_safety(self) -> None:
        computer = IhsanComputer()
        baseline = computer.compute(
            content="Propose a rollout plan with checks and monitoring.",
            snr_score=0.9,
            query_text="How to deploy safely?",
            context={"risk_score": 0.0},
        )
        high_risk = computer.compute(
            content="Propose a rollout plan with checks and monitoring.",
            snr_score=0.9,
            query_text="How to deploy safely?",
            context={"risk_score": 1.0},
        )

        assert high_risk.safety < baseline.safety

    def test_thermal_mode_populates_optional_dimensions(self) -> None:
        computer = IhsanComputer(enable_thermal_mode=True)
        components = computer.compute(
            content=(
                "Step 1: measure latency. "
                "Step 2: optimize bottlenecks. "
                "Step 3: verify with benchmark metrics."
            ),
            snr_score=0.9,
            query_text="How do I optimize latency safely?",
            context={"risk_score": 0.1, "thermal_step": 2},
        )

        assert components.auditability is not None
        assert components.robustness is not None
        assert 0.0 <= components.auditability <= 1.0
        assert 0.0 <= components.robustness <= 1.0

    def test_context_flag_enables_thermal_mode(self) -> None:
        computer = IhsanComputer(enable_thermal_mode=False)
        thermal = computer.compute(
            content="Step 1: verify. Step 2: measure.",
            snr_score=0.8,
            query_text="verify",
            context={"ihsan_mode": "thermal"},
        )
        heuristic = computer.compute(
            content="Step 1: verify. Step 2: measure.",
            snr_score=0.8,
            query_text="verify",
            context={"ihsan_mode": "heuristic"},
        )

        assert thermal.auditability is not None
        assert thermal.robustness is not None
        assert heuristic.auditability is None
        assert heuristic.robustness is None
