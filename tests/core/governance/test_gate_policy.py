"""
Tests for core.governance.gate_policy — Python mirror of bizra-core gate_policy tests.

14 tests matching 1:1 with Rust: 6 apply_gate + 3 env_gate_policy + 1 verdict fields
+ 4 maturation (Wire 5).
Cross-language parity: identical (score, threshold, policy) → identical (passed, action).
"""

import pytest

from core.governance.gate_policy import (
    GateAction,
    GateMaturationPolicy,
    GatePolicy,
    MaturationThresholds,
    apply_gate,
    env_gate_policy,
)


class TestApplyGate:
    """Tests for the apply_gate() canonical decision function."""

    def test_passing_score_always_allows(self) -> None:
        v = apply_gate(0.96, 0.95, GatePolicy.REJECT)
        assert v.passed is True
        assert v.action == GateAction.ALLOW

    def test_failing_observe_allows_with_warning(self) -> None:
        v = apply_gate(0.90, 0.95, GatePolicy.OBSERVE)
        assert v.passed is False
        assert v.action == GateAction.ALLOW_WITH_WARNING

    def test_failing_flag_returns_flagged(self) -> None:
        v = apply_gate(0.90, 0.95, GatePolicy.FLAG)
        assert v.passed is False
        assert v.action == GateAction.FLAGGED

    def test_failing_throttle_returns_throttled(self) -> None:
        v = apply_gate(0.90, 0.95, GatePolicy.THROTTLE)
        assert v.passed is False
        assert v.action == GateAction.THROTTLED

    def test_failing_reject_returns_rejected(self) -> None:
        v = apply_gate(0.90, 0.95, GatePolicy.REJECT)
        assert v.passed is False
        assert v.action == GateAction.REJECTED

    def test_exact_threshold_passes(self) -> None:
        v = apply_gate(0.95, 0.95, GatePolicy.REJECT)
        assert v.passed is True
        assert v.action == GateAction.ALLOW


class TestEnvGatePolicy:
    """Tests for env_gate_policy() environment resolution."""

    def test_env_gate_default_is_observe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BIZRA_ENV", raising=False)
        assert env_gate_policy() == GatePolicy.OBSERVE

    def test_env_gate_prod_is_reject(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIZRA_ENV", "prod")
        assert env_gate_policy() == GatePolicy.REJECT

    def test_env_gate_production_long_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BIZRA_ENV", "production")
        assert env_gate_policy() == GatePolicy.REJECT


class TestVerdictFields:
    """Test that GateVerdict fields are populated correctly."""

    def test_verdict_fields_populated(self) -> None:
        v = apply_gate(0.93, 0.95, GatePolicy.REJECT)
        assert v.score == pytest.approx(0.93)
        assert v.threshold == pytest.approx(0.95)
        assert v.passed is False
        assert v.policy == GatePolicy.REJECT
        assert v.action == GateAction.REJECTED


# ── Wire 5: Maturation tests ─────────────────────────────────


class TestGateMaturation:
    """Tests for GateMaturationPolicy — Deming's PDCA applied to enforcement."""

    def test_starts_at_observe(self) -> None:
        m = GateMaturationPolicy()
        assert m.current == GatePolicy.OBSERVE
        assert m.cycle_count == 0
        assert m.is_mature is False

    def test_promotes_through_all_stages(self) -> None:
        t = MaturationThresholds(
            observe_to_flag=3, flag_to_throttle=6, throttle_to_reject=10
        )
        m = GateMaturationPolicy(t)

        # Cycles 1-2: still Observe
        for _ in range(2):
            assert m.tick() == GatePolicy.OBSERVE

        # Cycle 3: promotes to Flag
        assert m.tick() == GatePolicy.FLAG

        # Cycles 4-5: still Flag
        for _ in range(2):
            assert m.tick() == GatePolicy.FLAG

        # Cycle 6: promotes to Throttle
        assert m.tick() == GatePolicy.THROTTLE
        assert m.is_mature is False

        # Cycles 7-9: still Throttle
        for _ in range(3):
            assert m.tick() == GatePolicy.THROTTLE

        # Cycle 10: promotes to Reject (terminal)
        assert m.tick() == GatePolicy.REJECT
        assert m.is_mature is True
        assert m.cycle_count == 10

        # Further ticks stay at Reject (monotonic)
        assert m.tick() == GatePolicy.REJECT

    def test_never_softens(self) -> None:
        t = MaturationThresholds(
            observe_to_flag=1, flag_to_throttle=2, throttle_to_reject=3
        )
        m = GateMaturationPolicy(t)
        for _ in range(5):
            m.tick()
        assert m.current == GatePolicy.REJECT
        for _ in range(100):
            assert m.tick() == GatePolicy.REJECT

    def test_default_thresholds(self) -> None:
        t = MaturationThresholds()
        assert t.observe_to_flag == 100
        assert t.flag_to_throttle == 500
        assert t.throttle_to_reject == 1000
