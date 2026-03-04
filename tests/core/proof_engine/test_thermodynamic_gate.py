from __future__ import annotations

from core.proof_engine.thermodynamic_gate import ThermodynamicIhsanGate


class TestThermodynamicIhsanGate:
    def test_rejects_when_below_threshold(self) -> None:
        gate = ThermodynamicIhsanGate(threshold=0.99)
        decision = gate.evaluate(
            content="maybe maybe do something",
            snr_score=0.4,
            query_text="precise implementation plan",
            context={"risk_score": 0.8},
        )
        assert decision.approved is False
        assert "below threshold" in decision.reason.lower()

    def test_approves_when_threshold_is_met(self) -> None:
        gate = ThermodynamicIhsanGate(threshold=0.60)
        decision = gate.evaluate(
            content=(
                "Step 1: measure baseline latency. "
                "Step 2: optimize bottlenecks. "
                "Step 3: verify with benchmark metrics."
            ),
            snr_score=0.92,
            query_text="How do I optimize and verify latency?",
        )
        assert decision.approved is True
        assert decision.profile.composite_ihsan >= 0.60

    def test_rejects_on_lyapunov_violation(self) -> None:
        gate = ThermodynamicIhsanGate(threshold=0.0, lyapunov_constant=0.05)
        decision = gate.evaluate(
            content="malware exploit harm",
            snr_score=0.2,
            query_text="safe rollout plan",
            context={"risk_score": 1.0},
            previous_energy=0.0,
            step=0,
        )
        assert decision.approved is False
        assert "lyapunov" in decision.reason.lower()
        assert decision.delta_energy is not None
        assert decision.lyapunov_bound is not None

    def test_to_dict_shape(self) -> None:
        gate = ThermodynamicIhsanGate(threshold=0.60)
        decision = gate.evaluate(
            content="Step 1: inspect. Step 2: verify.",
            snr_score=0.8,
            query_text="verify",
        )
        payload = decision.to_dict()
        assert "approved" in payload
        assert "energies" in payload
        assert "ihsan_dimensions" in payload
        assert "composite_ihsan" in payload
