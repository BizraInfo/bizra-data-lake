"""
Diffusion Reasoning Amplifier tests.

Validates HMM-informed T1->T2 amplification behavior:
- observation symbol mapping
- fail-closed activation threshold
- query augmentation
- router context payload
"""

from core.integration.constants import SNR_THRESHOLD_T1_HIGH, UNIFIED_SNR_THRESHOLD
from core.prediction import HMMState, PredictionResult
from core.reasoning.diffusion_reasoning_amplifier import (
    DiffusionReasoningAmplifier,
)


def _prediction(state: HMMState, confidence: float) -> PredictionResult:
    return PredictionResult(
        most_likely_state=state,
        state_probabilities={s.value: 0.0 for s in HMMState},
        predicted_next_state=state,
        prediction_confidence=confidence,
        observation_likelihood=-1.0,
    )


class TestDiffusionReasoningAmplifier:
    def test_query_to_observation_symbol(self) -> None:
        amp = DiffusionReasoningAmplifier()
        symbol = amp.query_to_observation_symbol("Please search and find this file.")
        assert symbol == "search"

    def test_amplify_fail_closed_without_prediction(self) -> None:
        amp = DiffusionReasoningAmplifier()
        ctx = amp.amplify(query="Analyze this architecture", prediction=None)
        assert ctx.activated is False
        assert ctx.complexity_hint == 0.0
        assert ctx.got_hypotheses == 1
        assert ctx.snr_target == UNIFIED_SNR_THRESHOLD

    def test_amplify_blocks_below_threshold(self) -> None:
        amp = DiffusionReasoningAmplifier()
        ctx = amp.amplify(
            query="Compare system designs",
            prediction=_prediction(HMMState.ANALYZING, 0.60),
        )
        assert ctx.activated is False
        assert "confidence_below_threshold" in ctx.reasons
        assert ctx.got_depth == 1

    def test_amplify_activates_above_threshold(self) -> None:
        amp = DiffusionReasoningAmplifier()
        ctx = amp.amplify(
            query="Compare and evaluate distributed consensus options",
            prediction=_prediction(HMMState.ANALYZING, 0.93),
        )
        assert ctx.activated is True
        assert ctx.complexity_hint > 0.0
        assert ctx.got_hypotheses >= 2
        assert ctx.got_depth >= 2
        assert ctx.snr_target >= SNR_THRESHOLD_T1_HIGH
        assert ctx.focus == "verify"

    def test_augment_query_preserves_baseline_when_inactive(self) -> None:
        amp = DiffusionReasoningAmplifier()
        ctx = amp.amplify(query="hello", prediction=None)
        assert amp.augment_query("hello", ctx) == "hello"

    def test_augment_query_injects_diffusion_block_when_active(self) -> None:
        amp = DiffusionReasoningAmplifier()
        ctx = amp.amplify(
            query="design a migration plan",
            prediction=_prediction(HMMState.CREATING, 0.95),
        )
        augmented = amp.augment_query("design a migration plan", ctx)
        assert "[DIFFUSION_CONTEXT]" in augmented
        assert "got_hypotheses=" in augmented
        assert augmented.endswith("design a migration plan")

    def test_context_for_router_shape(self) -> None:
        amp = DiffusionReasoningAmplifier()
        ctx = amp.amplify(
            query="explain threat model trade-offs",
            prediction=_prediction(HMMState.EXPLORING, 0.90),
        )
        payload = amp.context_for_router(ctx)
        assert "complexity_hint" in payload
        assert "got_hypotheses" in payload
        assert "got_depth" in payload
        assert "snr_target" in payload
        assert payload["diffusion_active"] is True

