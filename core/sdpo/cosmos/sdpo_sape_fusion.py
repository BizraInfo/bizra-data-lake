"""
SAPE-SDPO Cognitive Fusion — Wisdom Layer Enhancement
═══════════════════════════════════════════════════════════════════════════════

Integrates Self-Distillation with SAPE's 4-layer intelligence pyramid:
- Layer 1: Data extraction (SNR 0.90)
- Layer 2: Information structuring (SNR 0.95)
- Layer 3: Knowledge synthesis (SNR 0.99)
- Layer 4: Wisdom application (SNR 0.999) <- SDPO enhancement here

The fusion enables self-teaching at the wisdom layer, where complex reasoning
decisions benefit most from rich feedback-driven improvement.

Standing on Giants: Shannon + Anthropic + SDPO Paper
Genesis Strict Synthesis v2.2.2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo import (
    SAPE_DATA_SNR,
    SAPE_INFORMATION_SNR,
    SAPE_KNOWLEDGE_SNR,
    SAPE_WISDOM_SNR,
    SDPO_FEEDBACK_CONFIDENCE_THRESHOLD,
)
from core.sdpo.optimization import (
    BIZRAFeedbackGenerator,
    SDPOAdvantage,
    SDPOAdvantageCalculator,
    SDPOFeedback,
)


@dataclass
class SAPELayerOutput:
    """Output from a SAPE processing layer."""

    layer: str  # "data", "information", "knowledge", "wisdom"
    content: Any
    snr_score: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "content": self.content,
            "snr_score": self.snr_score,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class SDPO_SAPE_Result:
    """Result from SAPE-SDPO fusion processing."""

    output: Any
    layer_outputs: List[SAPELayerOutput]
    sdpo_advantage: Optional[SDPOAdvantage]
    feedback_applied: bool
    total_snr: float
    ihsan_score: float
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "layer_outputs": [lo.to_dict() for lo in self.layer_outputs],
            "sdpo_advantage": (
                self.sdpo_advantage.to_dict() if self.sdpo_advantage else None
            ),
            "feedback_applied": self.feedback_applied,
            "total_snr": self.total_snr,
            "ihsan_score": self.ihsan_score,
            "processing_time_ms": self.processing_time_ms,
        }

    @property
    def passes_ihsan(self) -> bool:
        return self.ihsan_score >= UNIFIED_IHSAN_THRESHOLD


class SAPEProcessor(ABC):
    """Abstract interface for SAPE layer processing."""

    @abstractmethod
    async def extract_data(self, input_text: str) -> SAPELayerOutput:
        """Layer 1: Data extraction."""
        pass

    @abstractmethod
    async def structure_information(self, data: SAPELayerOutput) -> SAPELayerOutput:
        """Layer 2: Information structuring."""
        pass

    @abstractmethod
    async def synthesize_knowledge(self, info: SAPELayerOutput) -> SAPELayerOutput:
        """Layer 3: Knowledge synthesis."""
        pass

    @abstractmethod
    async def apply_wisdom(self, knowledge: SAPELayerOutput) -> SAPELayerOutput:
        """Layer 4: Wisdom application."""
        pass


class DefaultSAPEProcessor(SAPEProcessor):
    """Default SAPE processor implementation."""

    async def extract_data(self, input_text: str) -> SAPELayerOutput:
        """Layer 1: Extract raw data from input."""
        start = datetime.now(timezone.utc)

        # Simple data extraction - tokenize and structure
        tokens = input_text.split()
        entities = [t for t in tokens if t[0].isupper()] if tokens else []

        content = {
            "raw_text": input_text,
            "token_count": len(tokens),
            "entities": entities,
        }

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SAPELayerOutput(
            layer="data",
            content=content,
            snr_score=SAPE_DATA_SNR,
            processing_time_ms=elapsed_ms,
        )

    async def structure_information(self, data: SAPELayerOutput) -> SAPELayerOutput:
        """Layer 2: Structure data into information."""
        start = datetime.now(timezone.utc)

        raw = data.content.get("raw_text", "")
        entities = data.content.get("entities", [])

        # Structure into meaningful information
        content = {
            "structured_text": raw,
            "entity_count": len(entities),
            "key_concepts": entities[:5],  # Top 5 entities as concepts
            "data_layer": data.content,
        }

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SAPELayerOutput(
            layer="information",
            content=content,
            snr_score=SAPE_INFORMATION_SNR,
            processing_time_ms=elapsed_ms,
        )

    async def synthesize_knowledge(self, info: SAPELayerOutput) -> SAPELayerOutput:
        """Layer 3: Synthesize information into knowledge."""
        start = datetime.now(timezone.utc)

        concepts = info.content.get("key_concepts", [])
        text = info.content.get("structured_text", "")

        # Synthesize knowledge (simplified)
        content = {
            "synthesized": text,
            "concepts": concepts,
            "relationships": [],  # Would contain concept relationships
            "confidence": 0.85,
            "info_layer": info.content,
        }

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SAPELayerOutput(
            layer="knowledge",
            content=content,
            snr_score=SAPE_KNOWLEDGE_SNR,
            processing_time_ms=elapsed_ms,
        )

    async def apply_wisdom(self, knowledge: SAPELayerOutput) -> SAPELayerOutput:
        """Layer 4: Apply wisdom to knowledge."""
        start = datetime.now(timezone.utc)

        synthesized = knowledge.content.get("synthesized", "")
        confidence = knowledge.content.get("confidence", 0.5)

        # Wisdom application (simplified)
        content = {
            "wisdom_output": synthesized,
            "decision_confidence": confidence,
            "reasoning": "Applied domain knowledge and ethical constraints",
            "knowledge_layer": knowledge.content,
        }

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SAPELayerOutput(
            layer="wisdom",
            content=content,
            snr_score=SAPE_WISDOM_SNR,
            processing_time_ms=elapsed_ms,
        )


class ImplicitPRM:
    """
    Implicit Process Reward Model for SDPO.

    Caches feedback patterns to avoid redundant self-teaching on
    similar error patterns. Acts as a "teacher memory" that accumulates
    wisdom from past corrections.
    """

    def __init__(self, max_cache_size: int = 1000):
        self._cache: Dict[str, SDPOFeedback] = {}
        self._max_size = max_cache_size

    def lookup(self, error_signature: str) -> Optional[SDPOFeedback]:
        """Look up cached feedback for an error pattern."""
        return self._cache.get(error_signature)

    def store(self, error_signature: str, feedback: SDPOFeedback):
        """Store feedback for future reference."""
        if len(self._cache) >= self._max_size:
            # Evict oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[error_signature] = feedback

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "utilization": len(self._cache) / self._max_size,
        }


class SAPE_SDPO_Fusion:
    """
    Unifies SDPO's self-teaching with SAPE's 4-layer reasoning.

    At the wisdom layer, if quality checks fail, SDPO correction is applied
    using rich feedback from BIZRA's quality gates (Ihsān, SNR, FATE).

    Usage:
        fusion = SAPE_SDPO_Fusion()
        result = await fusion.process("What is sovereignty?")

        if not result.passes_ihsan:
            print("Quality threshold not met")
        else:
            print(f"Wisdom output: {result.output}")
    """

    def __init__(
        self,
        sape_processor: Optional[SAPEProcessor] = None,
        advantage_calculator: Optional[SDPOAdvantageCalculator] = None,
        feedback_generator: Optional[BIZRAFeedbackGenerator] = None,
        llm_callback: Optional[Callable[[str], str]] = None,
    ):
        self.sape_processor = sape_processor or DefaultSAPEProcessor()
        self.advantage_calculator = advantage_calculator or SDPOAdvantageCalculator()
        self.feedback_generator = feedback_generator or BIZRAFeedbackGenerator()
        self.llm_callback = llm_callback  # Optional LLM for correction

        # Implicit Process Reward Model
        self._implicit_prm = ImplicitPRM()

    async def process(self, input_text: str) -> SDPO_SAPE_Result:
        """
        Process input through SAPE-SDPO fusion pipeline.

        Returns complete result with layer outputs and SDPO metrics.
        """
        start = datetime.now(timezone.utc)
        layer_outputs: List[SAPELayerOutput] = []

        # Layer 1: Data extraction
        data_output = await self.sape_processor.extract_data(input_text)
        layer_outputs.append(data_output)

        # Layer 2: Information structuring
        info_output = await self.sape_processor.structure_information(data_output)
        layer_outputs.append(info_output)

        # Layer 3: Knowledge synthesis
        knowledge_output = await self.sape_processor.synthesize_knowledge(info_output)
        layer_outputs.append(knowledge_output)

        # Layer 4: Wisdom application with SDPO enhancement
        wisdom_output, sdpo_advantage, feedback_applied = (
            await self._apply_wisdom_with_sdpo(knowledge_output, input_text)
        )
        layer_outputs.append(wisdom_output)

        # Calculate aggregate metrics
        total_snr = self._calculate_aggregate_snr(layer_outputs)
        ihsan_score = self._calculate_ihsan_score(wisdom_output)
        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return SDPO_SAPE_Result(
            output=wisdom_output.content.get("wisdom_output", ""),
            layer_outputs=layer_outputs,
            sdpo_advantage=sdpo_advantage,
            feedback_applied=feedback_applied,
            total_snr=total_snr,
            ihsan_score=ihsan_score,
            processing_time_ms=elapsed_ms,
        )

    async def _apply_wisdom_with_sdpo(
        self,
        knowledge: SAPELayerOutput,
        original_input: str,
    ) -> tuple[SAPELayerOutput, Optional[SDPOAdvantage], bool]:
        """
        Wisdom layer enhanced with SDPO self-teaching mechanism.

        If initial wisdom output fails quality checks, SDPO correction is applied.
        """
        # Base wisdom application
        base_wisdom = await self.sape_processor.apply_wisdom(knowledge)

        # Check quality
        quality_check = self._check_quality(base_wisdom)

        if quality_check["passes"]:
            return base_wisdom, None, False

        # Generate feedback
        feedback = self.feedback_generator.generate_feedback(quality_check)

        # Check implicit PRM cache
        error_sig = self._get_error_signature(quality_check)
        cached_feedback = self._implicit_prm.lookup(error_sig)

        if cached_feedback and cached_feedback.confidence >= feedback.confidence:
            feedback = cached_feedback

        # Only apply SDPO correction if we have LLM callback
        if (
            self.llm_callback
            and feedback.confidence >= SDPO_FEEDBACK_CONFIDENCE_THRESHOLD
        ):
            corrected_wisdom = await self._apply_sdpo_correction(
                base_wisdom, feedback, original_input
            )

            # Calculate advantage
            advantage = await self.advantage_calculator.calculate_advantages(
                question=original_input,
                failed_attempt=str(base_wisdom.content.get("wisdom_output", "")),
                feedback=feedback.text,
                corrected_attempt=str(
                    corrected_wisdom.content.get("wisdom_output", "")
                ),
            )

            # Store in implicit PRM
            self._implicit_prm.store(error_sig, feedback)

            return corrected_wisdom, advantage, True

        return base_wisdom, None, False

    async def _apply_sdpo_correction(
        self,
        base_wisdom: SAPELayerOutput,
        feedback: SDPOFeedback,
        original_input: str,
    ) -> SAPELayerOutput:
        """Apply SDPO correction using LLM."""
        correction_prompt = f"""Original question: {original_input}

Previous response that needs improvement:
{base_wisdom.content.get("wisdom_output", "")}

Feedback:
{feedback.text}

Please provide a corrected, improved response:"""

        # Call LLM for correction
        assert self.llm_callback is not None
        corrected_text = self.llm_callback(correction_prompt)

        # Create corrected wisdom output
        corrected_content = base_wisdom.content.copy()
        corrected_content["wisdom_output"] = corrected_text
        corrected_content["sdpo_corrected"] = True

        return SAPELayerOutput(
            layer="wisdom",
            content=corrected_content,
            snr_score=min(SAPE_WISDOM_SNR, base_wisdom.snr_score + 0.01),
            processing_time_ms=base_wisdom.processing_time_ms,
            metadata={"sdpo_corrected": True},
        )

    def _check_quality(self, wisdom: SAPELayerOutput) -> Dict[str, Any]:
        """Check wisdom output against quality gates."""
        snr = wisdom.snr_score
        ihsan = self._calculate_ihsan_score(wisdom)

        return {
            "passes": snr >= UNIFIED_SNR_THRESHOLD and ihsan >= UNIFIED_IHSAN_THRESHOLD,
            "snr": snr,
            "ihsan_score": ihsan,
            "fate_compliant": True,  # Simplified - would integrate with FATE engine
        }

    def _calculate_aggregate_snr(self, layers: List[SAPELayerOutput]) -> float:
        """Calculate aggregate SNR across all layers."""
        if not layers:
            return 0.0

        # Weighted geometric mean (wisdom layer weighted highest)
        weights = {"data": 0.1, "information": 0.2, "knowledge": 0.3, "wisdom": 0.4}
        weighted_sum = sum(
            weights.get(layer.layer, 0.1) * layer.snr_score for layer in layers
        )
        return weighted_sum

    def _calculate_ihsan_score(self, wisdom: SAPELayerOutput) -> float:
        """Calculate Ihsān score from wisdom output."""
        confidence = wisdom.content.get("decision_confidence", 0.5)
        snr = wisdom.snr_score

        # Ihsān = weighted combination of confidence and SNR
        return 0.6 * confidence + 0.4 * snr

    def _get_error_signature(self, quality_check: Dict[str, Any]) -> str:
        """Generate signature for error pattern (for PRM caching)."""
        issues = []
        if quality_check.get("snr", 1.0) < UNIFIED_SNR_THRESHOLD:
            issues.append("low_snr")
        if quality_check.get("ihsan_score", 1.0) < UNIFIED_IHSAN_THRESHOLD:
            issues.append("low_ihsan")
        if not quality_check.get("fate_compliant", True):
            issues.append("fate_violation")
        return "_".join(sorted(issues)) if issues else "unknown"

    def get_prm_stats(self) -> Dict[str, Any]:
        """Get implicit PRM statistics."""
        return self._implicit_prm.get_stats()
