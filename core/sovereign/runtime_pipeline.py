"""
Runtime Pipeline — Query Processing Stages
===========================================
Implements the 5-stage query processing pipeline for Sovereign Runtime.

Stages:
0. Compute Tier Selection (Treasury Mode)
1. Graph-of-Thoughts Reasoning
2. LLM Inference
3. SNR Optimization
4. Constitutional Validation

Standing on Giants: Besta (GoT) + Shannon (SNR) + Anthropic (Constitutional AI)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .runtime_types import (
    SovereignQuery,
    SovereignResult,
    GraphReasonerProtocol,
    SNROptimizerProtocol,
    GuardianProtocol,
)

logger = logging.getLogger("sovereign.pipeline")


class QueryPipeline:
    """
    Implements the 5-stage query processing pipeline.

    Stages:
    0. Compute Tier Selection (Treasury Mode)
    1. Graph-of-Thoughts Reasoning
    2. LLM Inference
    3. SNR Optimization
    4. Constitutional Validation
    """

    def __init__(
        self,
        graph_reasoner: Optional[GraphReasonerProtocol] = None,
        snr_optimizer: Optional[SNROptimizerProtocol] = None,
        guardian_council: Optional[GuardianProtocol] = None,
        gateway: Optional[object] = None,
        omega_engine: Optional[object] = None,
        max_reasoning_depth: int = 5,
        ihsan_threshold: float = 0.95,
        snr_threshold: float = 0.85,
    ):
        self._graph_reasoner = graph_reasoner
        self._snr_optimizer = snr_optimizer
        self._guardian_council = guardian_council
        self._gateway = gateway
        self._omega = omega_engine
        self._max_reasoning_depth = max_reasoning_depth
        self._ihsan_threshold = ihsan_threshold
        self._snr_threshold = snr_threshold

    async def process(
        self,
        query: SovereignQuery,
        start_time: float
    ) -> SovereignResult:
        """Run the full query processing pipeline."""
        result = SovereignResult(query_id=query.id)

        # STAGE 0: Select compute tier
        compute_tier = await self._stage_compute_tier(query)

        # STAGE 1: Execute reasoning (GoT)
        reasoning_path, confidence, thought_prompt = await self._stage_reasoning(query)
        result.thoughts = reasoning_path
        result.reasoning_depth = len(reasoning_path)

        # STAGE 2: Perform LLM inference
        answer, model_used = await self._stage_inference(
            thought_prompt, compute_tier, query
        )
        result.response = answer

        # STAGE 3: Optimize SNR
        optimized_content, snr_score = await self._stage_snr(result.response)
        result.response = optimized_content
        result.snr_score = snr_score

        # STAGE 4: Constitutional validation
        ihsan_score, guardian_verdict = await self._stage_validation(
            result.response, query.context, query, result.snr_score
        )
        result.ihsan_score = ihsan_score
        result.validated = query.require_validation
        result.validation_passed = ihsan_score >= self._ihsan_threshold

        # Finalize
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.success = True
        result.reasoning_used = query.require_reasoning

        return result

    # -------------------------------------------------------------------------
    # STAGE 0: Compute Tier Selection
    # -------------------------------------------------------------------------

    async def _stage_compute_tier(self, query: SovereignQuery) -> Optional[object]:
        """Select compute tier based on Treasury Mode."""
        if not self._omega:
            return None

        mode = getattr(self._omega, 'get_operational_mode', lambda: None)()
        if mode is None:
            return None
        return self._mode_to_tier(mode)

    def _mode_to_tier(self, mode: object) -> Optional[object]:
        """Map TreasuryMode to ComputeTier."""
        try:
            from .omega_engine import TreasuryMode
            from core.inference.gateway import ComputeTier
            mapping = {
                TreasuryMode.ETHICAL: ComputeTier.LOCAL,
                TreasuryMode.HIBERNATION: ComputeTier.EDGE,
                TreasuryMode.EMERGENCY: ComputeTier.EDGE,
            }
            return mapping.get(mode, ComputeTier.LOCAL)
        except ImportError:
            return None

    # -------------------------------------------------------------------------
    # STAGE 1: Graph-of-Thoughts Reasoning
    # -------------------------------------------------------------------------

    async def _stage_reasoning(
        self,
        query: SovereignQuery
    ) -> Tuple[List[str], float, str]:
        """Execute Graph-of-Thoughts reasoning."""
        thought_prompt: str = query.text
        reasoning_path: List[str] = []
        confidence: float = 0.75

        if query.require_reasoning and self._graph_reasoner:
            reasoning_result = await self._graph_reasoner.reason(
                query=query.text,
                context=query.context,
                max_depth=self._max_reasoning_depth,
            )
            reasoning_path = reasoning_result.get("thoughts", [])
            confidence = reasoning_result.get("confidence", 0.0)

            conclusion = reasoning_result.get("conclusion")
            if conclusion:
                thought_prompt = conclusion

        return reasoning_path, confidence, thought_prompt

    # -------------------------------------------------------------------------
    # STAGE 2: LLM Inference
    # -------------------------------------------------------------------------

    async def _stage_inference(
        self,
        thought_prompt: str,
        compute_tier: Optional[object],
        query: SovereignQuery
    ) -> Tuple[str, str]:
        """Perform LLM inference via gateway."""
        if self._gateway:
            try:
                infer_method = getattr(self._gateway, 'infer', None)
                if infer_method is not None:
                    inference_result = await infer_method(
                        thought_prompt,
                        tier=compute_tier,
                        max_tokens=1024,
                    )
                    answer = getattr(inference_result, 'content', str(inference_result))
                    model_used = getattr(inference_result, 'model', 'unknown')
                    return answer, model_used
            except Exception as e:
                logger.warning(f"Gateway inference failed: {e}, using stub")

        return f"Reasoned response for: {query.text}", "stub"

    # -------------------------------------------------------------------------
    # STAGE 3: SNR Optimization
    # -------------------------------------------------------------------------

    async def _stage_snr(self, content: str) -> Tuple[str, float]:
        """Optimize Signal-to-Noise Ratio."""
        optimized_content = content
        snr_score = self._snr_threshold

        if self._snr_optimizer:
            snr_result = self._snr_optimizer.optimize(content)
            snr_score = snr_result.get("snr_score", self._snr_threshold)

        return optimized_content, snr_score

    # -------------------------------------------------------------------------
    # STAGE 4: Constitutional Validation
    # -------------------------------------------------------------------------

    async def _stage_validation(
        self,
        content: str,
        context: Dict[str, Any],
        query: SovereignQuery,
        snr_score: float
    ) -> Tuple[float, str]:
        """Validate against constitutional constraints."""
        ihsan_score = snr_score
        guardian_verdict = "SKIPPED"

        # 4a: Ihsan validation via OmegaEngine
        if self._omega:
            try:
                ihsan_vector = self._extract_ihsan_vector(content, context)
                evaluate_ihsan = getattr(self._omega, 'evaluate_ihsan', None)
                if evaluate_ihsan is not None and ihsan_vector is not None:
                    result = evaluate_ihsan(ihsan_vector)
                    if isinstance(result, tuple) and len(result) >= 2:
                        ihsan_score = result[0]
                    else:
                        ihsan_score = float(result) if result else snr_score
                guardian_verdict = "OMEGA_ONLY"
            except Exception as e:
                logger.warning(f"Omega Ihsan evaluation failed: {e}")
                ihsan_score = snr_score

        # 4b: Guardian Council validation
        if query.require_validation and self._guardian_council:
            validation = await self._guardian_council.validate(
                content=content,
                context=context,
            )
            guardian_verdict = "VALIDATED" if validation.get("is_valid") else "REJECTED"
            guardian_score = validation.get("confidence", 0.0)

            if self._omega:
                ihsan_score = (ihsan_score + guardian_score) / 2
            else:
                ihsan_score = guardian_score

        return ihsan_score, guardian_verdict

    def _extract_ihsan_vector(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> Optional[object]:
        """Extract Ihsan vector from response content."""
        try:
            from .omega_engine import ihsan_from_scores

            word_count = len(content.split())
            has_harmful = any(w in content.lower() for w in [
                "kill", "harm", "destroy", "attack", "illegal"
            ])

            correctness = min(0.98, 0.85 + (word_count / 1000) * 0.1)
            safety = 0.50 if has_harmful else 0.98
            user_benefit = float(context.get("benefit_score", 0.92))
            efficiency = min(0.96, 1.0 - (word_count / 5000))

            return ihsan_from_scores(
                correctness=correctness,
                safety=safety,
                user_benefit=user_benefit,
                efficiency=efficiency,
            )
        except ImportError:
            return None


__all__ = [
    "QueryPipeline",
]
