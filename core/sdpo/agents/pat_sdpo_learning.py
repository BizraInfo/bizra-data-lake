"""
PAT Agent Self-Distillation Learning — SDPO Integration for PAT Agents
===============================================================================

Enhances PAT (Proof-Augmented Trust) agents with SDPO learning capabilities:
- Context compression via self-teaching
- Continuous learning from rich feedback
- Pattern extraction and reuse

Standing on Giants: Shannon + Anthropic + SDPO Paper
Genesis Strict Synthesis v2.2.2
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo import (
    SDPO_COMPRESSION_TARGET,
    SDPO_FEEDBACK_CONFIDENCE_THRESHOLD,
    SDPO_MAX_ITERATIONS,
)
from core.sdpo.optimization import (
    BIZRAFeedbackGenerator,
    SDPOAdvantage,
    SDPOAdvantageCalculator,
    SDPOFeedback,
)


@dataclass
class PAT_SDPO_Config:
    """Configuration for PAT-SDPO learning."""

    compression_target: float = SDPO_COMPRESSION_TARGET
    max_iterations: int = SDPO_MAX_ITERATIONS
    feedback_threshold: float = SDPO_FEEDBACK_CONFIDENCE_THRESHOLD
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    learning_rate: float = 1e-5
    memory_window: int = 100  # Recent interactions to consider


@dataclass
class PAT_SDPO_State:
    """State tracking for PAT-SDPO learner."""

    total_interactions: int = 0
    successful_compressions: int = 0
    learning_cycles: int = 0
    average_compression_ratio: float = 1.0
    accumulated_advantage: float = 0.0
    last_update: Optional[datetime] = None
    pattern_cache: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_interactions": self.total_interactions,
            "successful_compressions": self.successful_compressions,
            "learning_cycles": self.learning_cycles,
            "average_compression_ratio": self.average_compression_ratio,
            "accumulated_advantage": self.accumulated_advantage,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "pattern_cache_size": len(self.pattern_cache),
        }


@dataclass
class SelfTeachingCycle:
    """Result from a single self-teaching cycle."""

    original_context: str
    compressed_context: str
    compression_ratio: float
    quality_preserved: float
    feedback: Optional[SDPOFeedback]
    advantage: Optional[SDPOAdvantage]
    iteration: int
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compression_ratio": self.compression_ratio,
            "quality_preserved": self.quality_preserved,
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "advantage": self.advantage.to_dict() if self.advantage else None,
            "iteration": self.iteration,
            "success": self.success,
        }


class ContextCompressionEngine:
    """
    Context compression via SDPO self-teaching.

    Implements the SDPO advantage function to iteratively compress
    context while preserving semantic fidelity above Ihsān threshold.
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        config: Optional[PAT_SDPO_Config] = None,
    ):
        self.llm_callback = llm_callback
        self.config = config or PAT_SDPO_Config()
        self.advantage_calculator = SDPOAdvantageCalculator()
        self.feedback_generator = BIZRAFeedbackGenerator()

    async def compress(
        self,
        context: str,
        preserve_patterns: Optional[List[str]] = None,
    ) -> Tuple[str, float, List[SelfTeachingCycle]]:
        """
        Compress context via iterative self-teaching.

        Args:
            context: Original context to compress
            preserve_patterns: Patterns that must be preserved

        Returns:
            (compressed_context, final_ratio, teaching_cycles)
        """
        cycles: List[SelfTeachingCycle] = []
        current = context
        preserve_patterns = preserve_patterns or []

        for iteration in range(self.config.max_iterations):
            # Attempt compression
            compressed, quality = await self._compress_iteration(
                current, preserve_patterns
            )

            ratio = len(compressed) / len(context) if context else 1.0

            # Check if target reached
            if ratio <= self.config.compression_target:
                cycle = SelfTeachingCycle(
                    original_context=current,
                    compressed_context=compressed,
                    compression_ratio=ratio,
                    quality_preserved=quality,
                    feedback=None,
                    advantage=None,
                    iteration=iteration,
                    success=True,
                )
                cycles.append(cycle)
                return compressed, ratio, cycles

            # Quality check
            quality_check = self._check_quality(compressed, context, preserve_patterns)

            if quality_check["passes"]:
                # Good compression, continue
                cycle = SelfTeachingCycle(
                    original_context=current,
                    compressed_context=compressed,
                    compression_ratio=ratio,
                    quality_preserved=quality,
                    feedback=None,
                    advantage=None,
                    iteration=iteration,
                    success=True,
                )
                cycles.append(cycle)
                current = compressed
            else:
                # Generate feedback and try self-teaching
                feedback = self.feedback_generator.generate_feedback(quality_check)

                if (
                    self.llm_callback
                    and feedback.confidence >= self.config.feedback_threshold
                ):
                    # Apply SDPO correction
                    corrected = await self._apply_sdpo_correction(
                        compressed, feedback, context
                    )

                    # Calculate advantage
                    advantage = await self.advantage_calculator.calculate_advantages(
                        question=f"Compress: {context[:100]}...",
                        failed_attempt=compressed,
                        feedback=feedback.text,
                        corrected_attempt=corrected,
                    )

                    cycle = SelfTeachingCycle(
                        original_context=current,
                        compressed_context=corrected,
                        compression_ratio=len(corrected) / len(context),
                        quality_preserved=quality,
                        feedback=feedback,
                        advantage=advantage,
                        iteration=iteration,
                        success=advantage.is_beneficial,
                    )
                    cycles.append(cycle)

                    if advantage.is_beneficial:
                        current = corrected
                    else:
                        # Feedback didn't help, stop
                        break
                else:
                    # No LLM or low confidence, accept current
                    cycle = SelfTeachingCycle(
                        original_context=current,
                        compressed_context=compressed,
                        compression_ratio=ratio,
                        quality_preserved=quality,
                        feedback=feedback,
                        advantage=None,
                        iteration=iteration,
                        success=False,
                    )
                    cycles.append(cycle)
                    break

        final_ratio = len(current) / len(context) if context else 1.0
        return current, final_ratio, cycles

    async def _compress_iteration(
        self,
        text: str,
        preserve_patterns: List[str],
    ) -> Tuple[str, float]:
        """Single compression iteration."""
        if self.llm_callback:
            prompt = self._create_compression_prompt(text, preserve_patterns)
            compressed = self.llm_callback(prompt)
            quality = self._estimate_quality(text, compressed, preserve_patterns)
            return compressed, quality
        else:
            # Heuristic compression without LLM
            return self._heuristic_compress(text, preserve_patterns)

    def _heuristic_compress(
        self,
        text: str,
        preserve_patterns: List[str],
    ) -> Tuple[str, float]:
        """Heuristic compression when LLM unavailable."""
        # Simple sentence extraction
        sentences = text.split(". ")
        if len(sentences) <= 1:
            return text, 1.0

        # Keep sentences containing preserve patterns
        kept = []
        for sentence in sentences:
            if any(
                pattern.lower() in sentence.lower() for pattern in preserve_patterns
            ):
                kept.append(sentence)
            elif len(sentence) > 50:  # Keep substantial sentences
                kept.append(sentence)

        # Keep at least half
        if len(kept) < len(sentences) // 2:
            kept = sentences[: len(sentences) // 2]

        compressed = ". ".join(kept)
        quality = len(kept) / len(sentences) if sentences else 1.0

        return compressed, quality

    def _create_compression_prompt(
        self,
        text: str,
        preserve_patterns: List[str],
    ) -> str:
        """Create prompt for LLM compression."""
        patterns_str = ", ".join(preserve_patterns) if preserve_patterns else "none"
        return f"""Compress the following text while preserving its essential meaning.
Target: Reduce to {int(self.config.compression_target * 100)}% of original length.
Must preserve these patterns/concepts: {patterns_str}

Original text:
{text}

Compressed version:"""

    def _estimate_quality(
        self,
        original: str,
        compressed: str,
        preserve_patterns: List[str],
    ) -> float:
        """Estimate compression quality (0-1)."""
        # Check pattern preservation
        pattern_score = 1.0
        if preserve_patterns:
            preserved = sum(
                1 for p in preserve_patterns if p.lower() in compressed.lower()
            )
            pattern_score = preserved / len(preserve_patterns)

        # Check length ratio
        length_ratio = len(compressed) / len(original) if original else 1.0
        length_score = min(1.0, length_ratio / self.config.compression_target)

        # Weighted combination
        return 0.7 * pattern_score + 0.3 * length_score

    def _check_quality(
        self,
        compressed: str,
        original: str,
        preserve_patterns: List[str],
    ) -> Dict[str, Any]:
        """Check compressed output quality."""
        quality = self._estimate_quality(original, compressed, preserve_patterns)
        snr = quality  # Simplified - would use SNR calculator in production

        return {
            "passes": quality >= self.config.ihsan_threshold,
            "snr": snr,
            "ihsan_score": quality,
            "pattern_preservation": quality,
        }

    async def _apply_sdpo_correction(
        self,
        failed_compression: str,
        feedback: SDPOFeedback,
        original: str,
    ) -> str:
        """Apply SDPO correction to failed compression."""
        if not self.llm_callback:
            return failed_compression

        correction_prompt = f"""Original text: {original[:500]}...

Failed compression attempt:
{failed_compression}

Feedback:
{feedback.text}

Please provide a better compression that addresses the feedback:"""

        return self.llm_callback(correction_prompt)


class PAT_SDPO_Learner:
    """
    PAT Agent enhanced with SDPO self-teaching capabilities.

    Provides:
    - Context compression with quality preservation
    - Continuous learning from feedback
    - Pattern extraction and reuse

    Usage:
        learner = PAT_SDPO_Learner(llm_callback=my_llm)

        # Compress context
        compressed, ratio = await learner.compress_context(
            "Long agent context...",
            preserve=["sovereignty", "ihsan"]
        )

        # Learn from interaction
        await learner.learn_from_interaction(
            task="answer query",
            response="...",
            feedback="improve clarity",
            quality_score=0.87
        )
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        config: Optional[PAT_SDPO_Config] = None,
    ):
        self.config = config or PAT_SDPO_Config()
        self.compression_engine = ContextCompressionEngine(llm_callback, self.config)
        self.state = PAT_SDPO_State()
        self.advantage_calculator = SDPOAdvantageCalculator()
        self.feedback_generator = BIZRAFeedbackGenerator()

        # Learning memory
        self._interaction_history: List[Dict[str, Any]] = []
        self._pattern_library: Dict[str, float] = {}  # pattern -> success_rate

    async def compress_context(
        self,
        context: str,
        preserve: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        Compress agent context using SDPO self-teaching.

        Args:
            context: Full context to compress
            preserve: Patterns that must be preserved

        Returns:
            (compressed_context, compression_ratio)
        """
        compressed, ratio, cycles = await self.compression_engine.compress(
            context, preserve
        )

        # Update state
        self.state.total_interactions += 1
        if ratio <= self.config.compression_target:
            self.state.successful_compressions += 1

        # Update running average
        n = self.state.total_interactions
        self.state.average_compression_ratio = (
            self.state.average_compression_ratio * (n - 1) + ratio
        ) / n

        self.state.last_update = datetime.now(timezone.utc)

        return compressed, ratio

    async def learn_from_interaction(
        self,
        task: str,
        response: str,
        feedback: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> Optional[SDPOAdvantage]:
        """
        Learn from a PAT agent interaction.

        Args:
            task: The task that was performed
            response: The agent's response
            feedback: Optional feedback on the response
            quality_score: Optional quality score (0-1)

        Returns:
            SDPOAdvantage if learning occurred, None otherwise
        """
        # Record interaction
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "response_length": len(response),
            "had_feedback": feedback is not None,
            "quality_score": quality_score,
        }
        self._interaction_history.append(interaction)

        # Trim history
        if len(self._interaction_history) > self.config.memory_window:
            self._interaction_history = self._interaction_history[
                -self.config.memory_window :
            ]

        # If we have feedback and it's actionable, learn
        if feedback and quality_score and quality_score < self.config.ihsan_threshold:
            quality_check = {
                "passes": False,
                "ihsan_score": quality_score,
                "snr": quality_score,
            }

            self.feedback_generator.generate_feedback(quality_check)

            # Calculate advantage (heuristic since we don't have corrected response)
            advantage = await self.advantage_calculator.calculate_advantages(
                question=task,
                failed_attempt=response[:500],
                feedback=feedback,
                corrected_attempt=response[:500],  # Placeholder
            )

            self.state.learning_cycles += 1
            self.state.accumulated_advantage += advantage.overall_advantage
            self.state.last_update = datetime.now(timezone.utc)

            # Extract patterns from feedback
            self._extract_patterns(feedback, quality_score)

            return advantage

        return None

    def _extract_patterns(self, feedback: str, quality_score: float):
        """Extract reusable patterns from feedback."""
        # Simple keyword extraction
        keywords = ["improve", "add", "remove", "clarify", "expand", "reduce"]
        words = feedback.lower().split()

        for i, word in enumerate(words):
            if word in keywords and i + 1 < len(words):
                pattern = f"{word}_{words[i + 1]}"
                # Update pattern success rate
                if pattern in self._pattern_library:
                    old_rate = self._pattern_library[pattern]
                    self._pattern_library[pattern] = (old_rate + quality_score) / 2
                else:
                    self._pattern_library[pattern] = quality_score

    def get_successful_patterns(self, threshold: float = 0.8) -> List[str]:
        """Get patterns with high success rate."""
        return [
            pattern
            for pattern, rate in self._pattern_library.items()
            if rate >= threshold
        ]

    def get_state(self) -> Dict[str, Any]:
        """Get current learner state."""
        return {
            **self.state.to_dict(),
            "pattern_library_size": len(self._pattern_library),
            "interaction_history_size": len(self._interaction_history),
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self._interaction_history:
            return {"total_interactions": 0}

        quality_scores = [
            i["quality_score"]
            for i in self._interaction_history
            if i.get("quality_score") is not None
        ]

        return {
            "total_interactions": len(self._interaction_history),
            "learning_cycles": self.state.learning_cycles,
            "accumulated_advantage": self.state.accumulated_advantage,
            "average_quality": (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            ),
            "compression_success_rate": (
                self.state.successful_compressions / self.state.total_interactions
                if self.state.total_interactions > 0
                else 0
            ),
            "average_compression_ratio": self.state.average_compression_ratio,
        }
