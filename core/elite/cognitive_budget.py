"""
Thinking Budget Allocation — Dynamic Cognitive Resource Management

Implements dynamic cognitive budget allocation based on task type
using the 7-3-6-9 DNA signature pattern.

Standing on Giants:
- Kahneman (2011): System 1/2 thinking and cognitive load
- Shannon (1948): Information theory for complexity estimation
- Anthropic: Extended thinking / reasoning depth control
- BIZRA DNA: 7-3-6-9 signature (spiritual numerology meets computation)

The 7-3-6-9 Pattern:
- 7: Cognitive depth (layers of reasoning)
- 3: Parallel tracks (convergent paths)
- 6: Balance point (effort equilibrium)
- 9: Completion threshold (finalization)

Budget Categories:
1. NANO: Quick reflexive responses (< 1s, 100 tokens)
2. MICRO: Simple lookups and classifications (1-5s, 500 tokens)
3. MESO: Multi-step reasoning (5-30s, 2K tokens)
4. MACRO: Deep analysis (30s-5m, 10K tokens)
5. MEGA: Full deliberation (5m+, 50K+ tokens)

The DNA signature modulates budget allocation:
- Task complexity score (7-based)
- Convergence requirement (3-based)
- Effort calibration (6-based)
- Completion confidence (9-based)

Created: 2026-02-03 | BIZRA Elite Integration v1.1.0
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DNA SIGNATURE CONSTANTS
# ============================================================================

# The sacred 7-3-6-9 signature
DNA_DEPTH = 7  # Maximum cognitive depth layers
DNA_TRACKS = 3  # Parallel reasoning tracks
DNA_BALANCE = 6  # Balance/equilibrium point
DNA_COMPLETE = 9  # Completion/finalization

# DNA-derived constants
DNA_RATIO = (7 * 9) / (3 * 6)  # = 3.5 - depth amplification ratio
DNA_HARMONY = (7 + 3 + 6 + 9) / 4  # = 6.25 - harmonic mean for budget scaling


# ============================================================================
# BUDGET CATEGORIES
# ============================================================================


class BudgetTier(str, Enum):
    """Cognitive budget tiers."""

    NANO = "nano"  # Reflexive, instant
    MICRO = "micro"  # Quick lookup
    MESO = "meso"  # Standard reasoning
    MACRO = "macro"  # Deep analysis
    MEGA = "mega"  # Full deliberation


@dataclass
class BudgetAllocation:
    """
    Budget allocation for a task.

    Specifies computational resources across dimensions.
    """

    tier: BudgetTier

    # Time budget (seconds)
    time_budget_s: float

    # Token budget (approximate)
    token_budget: int

    # Cognitive depth (1-7 based on DNA_DEPTH)
    depth: int

    # Parallel tracks (1-3 based on DNA_TRACKS)
    parallel_tracks: int

    # Iteration limit (DNA_COMPLETE = 9 max)
    max_iterations: int

    # Confidence threshold for completion
    confidence_threshold: float

    # Extended thinking enabled
    extended_thinking: bool = False

    # NTU integration
    ntu_window_size: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize allocation."""
        return {
            "tier": self.tier.value,
            "time_budget_s": self.time_budget_s,
            "token_budget": self.token_budget,
            "depth": self.depth,
            "parallel_tracks": self.parallel_tracks,
            "max_iterations": self.max_iterations,
            "confidence_threshold": self.confidence_threshold,
            "extended_thinking": self.extended_thinking,
            "ntu_window_size": self.ntu_window_size,
        }


# Tier specifications
TIER_SPECS: Dict[BudgetTier, Dict[str, Any]] = {
    BudgetTier.NANO: {
        "time_budget_s": 1.0,
        "token_budget": 100,
        "depth": 1,
        "parallel_tracks": 1,
        "max_iterations": 1,
        "confidence_threshold": 0.9,
        "extended_thinking": False,
        "ntu_window_size": 3,
    },
    BudgetTier.MICRO: {
        "time_budget_s": 5.0,
        "token_budget": 500,
        "depth": 2,
        "parallel_tracks": 1,
        "max_iterations": 3,
        "confidence_threshold": 0.92,
        "extended_thinking": False,
        "ntu_window_size": 5,
    },
    BudgetTier.MESO: {
        "time_budget_s": 30.0,
        "token_budget": 2000,
        "depth": 4,
        "parallel_tracks": 2,
        "max_iterations": 6,
        "confidence_threshold": 0.95,
        "extended_thinking": False,
        "ntu_window_size": 7,
    },
    BudgetTier.MACRO: {
        "time_budget_s": 300.0,  # 5 minutes
        "token_budget": 10000,
        "depth": 6,
        "parallel_tracks": 3,
        "max_iterations": 9,
        "confidence_threshold": 0.97,
        "extended_thinking": True,
        "ntu_window_size": 9,
    },
    BudgetTier.MEGA: {
        "time_budget_s": 1800.0,  # 30 minutes
        "token_budget": 50000,
        "depth": 7,
        "parallel_tracks": 3,
        "max_iterations": 9,
        "confidence_threshold": 0.99,
        "extended_thinking": True,
        "ntu_window_size": 12,
    },
}


# ============================================================================
# TASK TYPES
# ============================================================================


class TaskType(str, Enum):
    """Task type classification."""

    # Reflexive tasks (NANO)
    ECHO = "echo"  # Simple echo/repeat
    CLASSIFY = "classify"  # Single-label classification

    # Simple tasks (MICRO)
    LOOKUP = "lookup"  # Information retrieval
    TRANSFORM = "transform"  # Data transformation
    VALIDATE = "validate"  # Validation check

    # Standard tasks (MESO)
    SUMMARIZE = "summarize"  # Content summarization
    ANALYZE = "analyze"  # Basic analysis
    GENERATE = "generate"  # Content generation
    REASON = "reason"  # Multi-step reasoning

    # Complex tasks (MACRO)
    PLAN = "plan"  # Strategic planning
    SYNTHESIZE = "synthesize"  # Multi-source synthesis
    DEBUG = "debug"  # Complex debugging
    OPTIMIZE = "optimize"  # Optimization

    # Deep tasks (MEGA)
    ARCHITECT = "architect"  # System design
    RESEARCH = "research"  # Deep research
    PROOF = "proof"  # Formal proof
    CONSENSUS = "consensus"  # Multi-agent consensus


# Task type to tier mapping
TASK_TIER_MAP: Dict[TaskType, BudgetTier] = {
    # NANO
    TaskType.ECHO: BudgetTier.NANO,
    TaskType.CLASSIFY: BudgetTier.NANO,
    # MICRO
    TaskType.LOOKUP: BudgetTier.MICRO,
    TaskType.TRANSFORM: BudgetTier.MICRO,
    TaskType.VALIDATE: BudgetTier.MICRO,
    # MESO
    TaskType.SUMMARIZE: BudgetTier.MESO,
    TaskType.ANALYZE: BudgetTier.MESO,
    TaskType.GENERATE: BudgetTier.MESO,
    TaskType.REASON: BudgetTier.MESO,
    # MACRO
    TaskType.PLAN: BudgetTier.MACRO,
    TaskType.SYNTHESIZE: BudgetTier.MACRO,
    TaskType.DEBUG: BudgetTier.MACRO,
    TaskType.OPTIMIZE: BudgetTier.MACRO,
    # MEGA
    TaskType.ARCHITECT: BudgetTier.MEGA,
    TaskType.RESEARCH: BudgetTier.MEGA,
    TaskType.PROOF: BudgetTier.MEGA,
    TaskType.CONSENSUS: BudgetTier.MEGA,
}


# ============================================================================
# COMPLEXITY ESTIMATION
# ============================================================================


@dataclass
class ComplexitySignal:
    """
    Complexity signals for budget estimation.

    Uses 7-3-6-9 DNA pattern for scoring.
    """

    # Input signals
    input_length: int = 0  # Characters
    input_entropy: float = 0.0  # Shannon entropy (normalized)
    domain_specificity: float = 0.0  # How specialized (0-1)

    # Task signals
    reasoning_depth: int = 1  # Estimated reasoning layers (1-7)
    convergence_required: int = 1  # Parallel paths needed (1-3)

    # Context signals
    context_size: int = 0  # Relevant context tokens
    knowledge_gap: float = 0.0  # Estimated knowledge gap (0-1)

    # History signals
    previous_attempts: int = 0  # Failed attempts
    snr_history: List[float] = field(default_factory=list)

    def compute_complexity_score(self) -> float:
        """
        Compute overall complexity score using DNA pattern.

        Score = (depth_factor * 7 + convergence_factor * 3 +
                 effort_factor * 6 + completion_factor * 9) / 25

        Normalized to [0, 1] range.
        """
        # Depth factor (7-based): reasoning complexity
        depth_factor = min(1.0, self.reasoning_depth / DNA_DEPTH)

        # Convergence factor (3-based): parallel exploration needed
        convergence_factor = min(1.0, self.convergence_required / DNA_TRACKS)

        # Effort factor (6-based): overall resource requirement
        length_factor = min(1.0, self.input_length / 10000)
        entropy_factor = self.input_entropy
        gap_factor = self.knowledge_gap
        effort_factor = (length_factor + entropy_factor + gap_factor) / 3

        # Completion factor (9-based): difficulty to finalize
        retry_penalty = min(1.0, self.previous_attempts / DNA_COMPLETE)
        snr_factor = (
            1.0 - (sum(self.snr_history) / max(len(self.snr_history), 1))
            if self.snr_history
            else 0.5
        )
        completion_factor = (retry_penalty + snr_factor) / 2

        # DNA-weighted score
        score = (
            depth_factor * DNA_DEPTH
            + convergence_factor * DNA_TRACKS
            + effort_factor * DNA_BALANCE
            + completion_factor * DNA_COMPLETE
        ) / (DNA_DEPTH + DNA_TRACKS + DNA_BALANCE + DNA_COMPLETE)

        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize signals."""
        return {
            "input_length": self.input_length,
            "input_entropy": self.input_entropy,
            "domain_specificity": self.domain_specificity,
            "reasoning_depth": self.reasoning_depth,
            "convergence_required": self.convergence_required,
            "context_size": self.context_size,
            "knowledge_gap": self.knowledge_gap,
            "previous_attempts": self.previous_attempts,
            "snr_history_mean": (
                sum(self.snr_history) / max(len(self.snr_history), 1)
                if self.snr_history
                else 0
            ),
            "complexity_score": self.compute_complexity_score(),
        }


# ============================================================================
# BUDGET ALLOCATOR
# ============================================================================


class CognitiveBudgetAllocator:
    """
    Allocates cognitive budget based on task complexity.

    Uses the 7-3-6-9 DNA pattern to:
    1. Estimate task complexity
    2. Select appropriate tier
    3. Allocate resources within tier
    4. Adapt based on execution feedback
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        enable_adaptive: bool = True,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.enable_adaptive = enable_adaptive

        # Allocation history for adaptation
        self._history: List[Dict[str, Any]] = []
        self._max_history = 100

        # Statistics
        self._total_allocations = 0
        self._tier_distribution: Dict[BudgetTier, int] = {t: 0 for t in BudgetTier}

    def estimate_complexity(
        self,
        task_type: TaskType,
        input_text: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> ComplexitySignal:
        """
        Estimate task complexity.

        Args:
            task_type: Type of task
            input_text: Input text (for entropy calculation)
            context: Additional context

        Returns:
            ComplexitySignal with computed metrics
        """
        context = context or {}

        signal = ComplexitySignal()

        # Input signals
        signal.input_length = len(input_text)
        signal.input_entropy = self._compute_entropy(input_text) if input_text else 0.5

        # Task-based signals
        tier = TASK_TIER_MAP.get(task_type, BudgetTier.MESO)
        tier_index = list(BudgetTier).index(tier)

        signal.reasoning_depth = min(DNA_DEPTH, tier_index + 2)
        signal.convergence_required = min(DNA_TRACKS, 1 + tier_index // 2)

        # Context signals
        signal.context_size = context.get("context_tokens", 0)
        signal.knowledge_gap = context.get("knowledge_gap", 0.3)

        # History signals
        signal.previous_attempts = context.get("retry_count", 0)
        signal.snr_history = context.get("snr_history", [])

        return signal

    def _compute_entropy(self, text: str) -> float:
        """Compute normalized Shannon entropy of text."""
        from collections import Counter

        if not text:
            return 0.0

        # Character-level entropy
        counts = Counter(text)
        total = len(text)
        probs = [count / total for count in counts.values()]

        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Normalize by max possible entropy (log2 of unique chars)
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def allocate(
        self,
        task_type: TaskType,
        input_text: str = "",
        context: Optional[Dict[str, Any]] = None,
        override_tier: Optional[BudgetTier] = None,
    ) -> BudgetAllocation:
        """
        Allocate cognitive budget for a task.

        Args:
            task_type: Type of task
            input_text: Input text
            context: Additional context
            override_tier: Force specific tier

        Returns:
            BudgetAllocation with resource limits
        """
        self._total_allocations += 1

        # Estimate complexity
        complexity = self.estimate_complexity(task_type, input_text, context)
        complexity_score = complexity.compute_complexity_score()

        # Determine tier
        if override_tier:
            tier = override_tier
        else:
            tier = self._select_tier(task_type, complexity_score)

        # Get base specs
        specs = TIER_SPECS[tier].copy()

        # Apply DNA modulation
        if self.enable_adaptive:
            specs = self._modulate_by_dna(specs, complexity)

        # Create allocation
        allocation = BudgetAllocation(
            tier=tier,
            time_budget_s=specs["time_budget_s"],
            token_budget=specs["token_budget"],
            depth=specs["depth"],
            parallel_tracks=specs["parallel_tracks"],
            max_iterations=specs["max_iterations"],
            confidence_threshold=specs["confidence_threshold"],
            extended_thinking=specs["extended_thinking"],
            ntu_window_size=specs["ntu_window_size"],
        )

        # Track statistics
        self._tier_distribution[tier] += 1

        # Record history
        self._history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_type": task_type.value,
                "complexity_score": complexity_score,
                "tier": tier.value,
                "allocation": allocation.to_dict(),
            }
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        logger.info(
            f"Budget allocated: {task_type.value} -> {tier.value} "
            f"(complexity={complexity_score:.3f}, time={allocation.time_budget_s}s, "
            f"tokens={allocation.token_budget})"
        )

        return allocation

    def _select_tier(self, task_type: TaskType, complexity_score: float) -> BudgetTier:
        """
        Select tier based on task type and complexity.

        Task type provides base tier, complexity can promote.
        """
        base_tier = TASK_TIER_MAP.get(task_type, BudgetTier.MESO)
        base_index = list(BudgetTier).index(base_tier)

        # Complexity can promote tier (but not demote)
        # High complexity (>0.7) promotes by 1, very high (>0.9) by 2
        promotion = 0
        if complexity_score > 0.9:
            promotion = 2
        elif complexity_score > 0.7:
            promotion = 1

        new_index = min(len(BudgetTier) - 1, base_index + promotion)
        return list(BudgetTier)[new_index]

    def _modulate_by_dna(
        self,
        specs: Dict[str, Any],
        complexity: ComplexitySignal,
    ) -> Dict[str, Any]:
        """
        Modulate budget specs using DNA pattern.

        DNA modulation adjusts:
        - Time: scaled by depth factor (7-based)
        - Tokens: scaled by convergence factor (3-based)
        - Iterations: scaled by effort factor (6-based)
        - Confidence: adjusted by completion factor (9-based)
        """
        # DNA factors from complexity
        depth_factor = min(1.0, complexity.reasoning_depth / DNA_DEPTH)
        convergence_factor = min(1.0, complexity.convergence_required / DNA_TRACKS)

        # Apply DNA ratio scaling
        time_scale = 1.0 + (depth_factor * (DNA_RATIO - 1))
        token_scale = 1.0 + (convergence_factor * (DNA_RATIO - 1))

        # Modulated specs
        modulated = specs.copy()
        modulated["time_budget_s"] = specs["time_budget_s"] * time_scale
        modulated["token_budget"] = int(specs["token_budget"] * token_scale)

        # Adjust depth based on complexity reasoning depth
        modulated["depth"] = max(specs["depth"], complexity.reasoning_depth)

        # Adjust tracks based on convergence requirement
        modulated["parallel_tracks"] = max(
            specs["parallel_tracks"], complexity.convergence_required
        )

        return modulated

    def record_execution(
        self,
        allocation: BudgetAllocation,
        actual_time_s: float,
        actual_tokens: int,
        success: bool,
        snr_achieved: float,
    ) -> None:
        """
        Record execution results for adaptation.

        Args:
            allocation: The allocation used
            actual_time_s: Actual execution time
            actual_tokens: Actual tokens used
            success: Whether task succeeded
            snr_achieved: SNR score achieved
        """
        efficiency = {
            "time_efficiency": allocation.time_budget_s / max(actual_time_s, 0.001),
            "token_efficiency": allocation.token_budget / max(actual_tokens, 1),
            "success": success,
            "snr_achieved": snr_achieved,
            "ihsan_achieved": snr_achieved >= self.ihsan_threshold,
        }

        # Update last history entry
        if self._history:
            self._history[-1]["execution"] = efficiency

        logger.debug(
            f"Budget execution recorded: time_eff={efficiency['time_efficiency']:.2f}, "
            f"token_eff={efficiency['token_efficiency']:.2f}, success={success}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get allocator statistics."""
        total = self._total_allocations or 1

        # Success rate from history
        executions = [h for h in self._history if "execution" in h]
        success_rate = sum(1 for e in executions if e["execution"]["success"]) / max(
            len(executions), 1
        )

        # Average efficiency
        time_eff = sum(
            e["execution"]["time_efficiency"] for e in executions if "execution" in e
        ) / max(len(executions), 1)
        token_eff = sum(
            e["execution"]["token_efficiency"] for e in executions if "execution" in e
        ) / max(len(executions), 1)

        return {
            "total_allocations": self._total_allocations,
            "tier_distribution": {
                t.value: c / total for t, c in self._tier_distribution.items()
            },
            "success_rate": success_rate,
            "avg_time_efficiency": time_eff,
            "avg_token_efficiency": token_eff,
            "dna_signature": {
                "depth": DNA_DEPTH,
                "tracks": DNA_TRACKS,
                "balance": DNA_BALANCE,
                "complete": DNA_COMPLETE,
                "ratio": DNA_RATIO,
                "harmony": DNA_HARMONY,
            },
        }


# ============================================================================
# BUDGET TRACKER
# ============================================================================


@dataclass
class BudgetUsage:
    """Tracks budget usage during execution."""

    allocation: BudgetAllocation

    # Usage tracking
    start_time: float = field(default_factory=time.time)
    tokens_used: int = 0
    iterations_completed: int = 0
    current_confidence: float = 0.0

    # State
    exhausted: bool = False
    early_exit: bool = False

    def time_elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    def time_remaining_s(self) -> float:
        """Get remaining time budget."""
        return max(0, self.allocation.time_budget_s - self.time_elapsed_s())

    def tokens_remaining(self) -> int:
        """Get remaining token budget."""
        return max(0, self.allocation.token_budget - self.tokens_used)

    def iterations_remaining(self) -> int:
        """Get remaining iterations."""
        return max(0, self.allocation.max_iterations - self.iterations_completed)

    def is_budget_available(self) -> bool:
        """Check if budget is still available."""
        return (
            self.time_remaining_s() > 0
            and self.tokens_remaining() > 0
            and self.iterations_remaining() > 0
            and not self.exhausted
        )

    def should_continue(self) -> bool:
        """
        Check if computation should continue.

        Continue if:
        - Budget available AND
        - Confidence below threshold (not yet converged)
        """
        if not self.is_budget_available():
            return False

        return self.current_confidence < self.allocation.confidence_threshold

    def consume_tokens(self, count: int) -> None:
        """Record token consumption."""
        self.tokens_used += count
        if self.tokens_used >= self.allocation.token_budget:
            self.exhausted = True
            logger.warning("Token budget exhausted")

    def complete_iteration(self, confidence: float) -> None:
        """Record iteration completion."""
        self.iterations_completed += 1
        self.current_confidence = confidence

        if self.iterations_completed >= self.allocation.max_iterations:
            self.exhausted = True
            logger.warning("Iteration budget exhausted")

        if confidence >= self.allocation.confidence_threshold:
            self.early_exit = True
            logger.info(f"Confidence threshold reached: {confidence:.4f}")

    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            "tier": self.allocation.tier.value,
            "time_used_s": self.time_elapsed_s(),
            "time_budget_s": self.allocation.time_budget_s,
            "time_utilization": self.time_elapsed_s() / self.allocation.time_budget_s,
            "tokens_used": self.tokens_used,
            "token_budget": self.allocation.token_budget,
            "token_utilization": self.tokens_used / self.allocation.token_budget,
            "iterations": self.iterations_completed,
            "max_iterations": self.allocation.max_iterations,
            "final_confidence": self.current_confidence,
            "exhausted": self.exhausted,
            "early_exit": self.early_exit,
        }


class BudgetTracker:
    """
    Context manager for tracking budget usage.

    Usage:
        allocator = CognitiveBudgetAllocator()
        allocation = allocator.allocate(TaskType.ANALYZE, text)

        with BudgetTracker(allocation) as tracker:
            while tracker.should_continue():
                result = process_step()
                tracker.consume_tokens(result.tokens)
                tracker.complete_iteration(result.confidence)
    """

    def __init__(self, allocation: BudgetAllocation):
        self.allocation = allocation
        self.usage: Optional[BudgetUsage] = None

    def __enter__(self) -> BudgetUsage:
        """Start tracking."""
        self.usage = BudgetUsage(allocation=self.allocation)
        logger.info(
            f"Budget tracking started: {self.allocation.tier.value} "
            f"(time={self.allocation.time_budget_s}s, tokens={self.allocation.token_budget})"
        )
        return self.usage

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End tracking and log summary."""
        if self.usage:
            summary = self.usage.get_summary()
            logger.info(
                f"Budget tracking ended: "
                f"time={summary['time_utilization']:.1%}, "
                f"tokens={summary['token_utilization']:.1%}, "
                f"confidence={summary['final_confidence']:.4f}"
            )


# ============================================================================
# NTU INTEGRATION
# ============================================================================


class NTUBudgetAdapter:
    """
    Adapts NTU temporal patterns to budget decisions.

    Uses NTU belief/entropy to inform budget allocation:
    - High belief + low entropy: stable pattern, can reduce budget
    - Low belief + high entropy: uncertain, increase budget
    """

    def __init__(self, allocator: CognitiveBudgetAllocator):
        self.allocator = allocator
        self._ntu = None

    @property
    def ntu(self):
        """Lazy-load NTU."""
        if self._ntu is None:
            try:
                from core.ntu import NTU, NTUConfig

                self._ntu = NTU(
                    NTUConfig(ihsan_threshold=self.allocator.ihsan_threshold)
                )
            except ImportError:
                logger.warning("NTU not available")
        return self._ntu

    def allocate_with_ntu(
        self,
        task_type: TaskType,
        input_text: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> BudgetAllocation:
        """
        Allocate budget with NTU-informed adjustments.

        NTU state modulates the base allocation:
        - High belief: slight budget reduction (confident)
        - High entropy: budget increase (uncertain)
        """
        # Get base allocation
        allocation = self.allocator.allocate(task_type, input_text, context)

        if self.ntu is None:
            return allocation

        # Get NTU state
        state = self.ntu.state

        # Compute adjustment factor
        # Range: 0.8 (confident) to 1.5 (uncertain)
        confidence = state.belief * (1.0 - state.entropy)
        adjustment = 1.0 + (0.5 - confidence) * 0.7

        # Apply adjustment
        allocation.time_budget_s *= adjustment
        allocation.token_budget = int(allocation.token_budget * adjustment)

        logger.debug(
            f"NTU adjustment: belief={state.belief:.3f}, entropy={state.entropy:.3f}, "
            f"adjustment={adjustment:.2f}"
        )

        return allocation

    def observe_execution(self, usage: BudgetUsage) -> None:
        """
        Observe execution result in NTU.

        Records the efficiency of budget usage as a quality signal.
        """
        if self.ntu is None:
            return

        summary = usage.get_summary()

        # Quality signal: high confidence with low utilization = efficient
        efficiency = summary["final_confidence"] * (
            1.0 - (summary["time_utilization"] + summary["token_utilization"]) / 2
        )
        efficiency = max(0.0, min(1.0, efficiency + 0.5))  # Shift to [0, 1]

        self.ntu.observe(
            efficiency,
            {
                "source": "cognitive_budget",
                "tier": summary["tier"],
                "utilization": summary["time_utilization"],
            },
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_allocator(
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    adaptive: bool = True,
) -> CognitiveBudgetAllocator:
    """Create a cognitive budget allocator."""
    return CognitiveBudgetAllocator(ihsan_threshold, adaptive)


def allocate_budget(
    task_type: TaskType,
    input_text: str = "",
    context: Optional[Dict[str, Any]] = None,
) -> BudgetAllocation:
    """
    Quick budget allocation using default allocator.

    Example:
        budget = allocate_budget(TaskType.ANALYZE, "analyze this text")
        print(f"Time budget: {budget.time_budget_s}s")
    """
    allocator = create_allocator()
    return allocator.allocate(task_type, input_text, context)


def track_budget(allocation: BudgetAllocation) -> BudgetTracker:
    """
    Create a budget tracker.

    Example:
        budget = allocate_budget(TaskType.REASON, prompt)
        with track_budget(budget) as tracker:
            while tracker.should_continue():
                # Process step
                tracker.complete_iteration(confidence)
    """
    return BudgetTracker(allocation)
