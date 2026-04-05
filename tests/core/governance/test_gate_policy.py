"""
Tests for core.governance.gate_policy — Python mirror of bizra-core gate_policy tests.

10 tests matching 1:1 with Rust: 6 apply_gate + 3 env_gate_policy + 1 verdict fields.
Cross-language parity: identical (score, threshold, policy) → identical (passed, action).
"""

import pytest

from core.governance.gate_policy import (
    GateAction,
    GatePolicy,
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
