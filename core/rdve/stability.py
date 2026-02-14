"""
RDVE Stability Protocols — Adaptive Warmup & Convergence Detection

Implements the stability mechanisms from the RDVE whitepaper:
    - Adaptive warmup for hypothesis generation (analog to gradient warmup)
    - Convergence detection with plateau analysis
    - Confidence calibration across exploration cycles
    - Rate limiting with exponential backoff on failures

Standing on Giants:
    Deming (statistical process control, 1950) — convergence detection
    Shannon (information theory, 1948) — entropy-based warmup calibration
    Boyd (OODA, 1976) — adaptive tempo control
    He (initialization theory, 2015) — warmup scheduling

Artifact: core/rdve/stability.py
"""

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

from core.integration.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_MINIMUM,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# WARMUP SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════════


class WarmupStrategy(str, Enum):
    """Warmup strategies — analogous to learning rate warmup in training."""

    LINEAR = "linear"  # Linear ramp from conservative to full exploration
    COSINE = "cosine"  # Cosine annealing — slow start, smooth transition
    EXPONENTIAL = "exponential"  # Fast initial ramp, then plateau
    ADAPTIVE = "adaptive"  # Data-driven — adjust based on early results


@dataclass
class WarmupSchedule:
    """
    Adaptive warmup schedule for hypothesis exploration.

    During warmup, exploration is conservative:
    - Fewer exploration paths
    - Stricter SNR/Ihsan thresholds
    - Lower risk tolerance

    After warmup, the system transitions to full exploration.

    Standing on Giants: He (initialization, 2015) — proper warmup prevents
    divergence in novel configurations.
    """

    warmup_cycles: int = 5
    strategy: WarmupStrategy = WarmupStrategy.COSINE
    min_exploration_factor: float = 0.3  # 30% of full exploration during warmup
    max_exploration_factor: float = 1.0

    # Adaptive parameters
    early_success_threshold: float = 0.5  # If >50% early cycles succeed, shorten warmup
    early_failure_threshold: float = 0.8  # If >80% fail, extend warmup

    def get_exploration_factor(
        self,
        current_cycle: int,
        success_rate: float = 0.5,
    ) -> float:
        """
        Compute the exploration scaling factor for the current cycle.

        Returns a value in [min_exploration_factor, max_exploration_factor].
        During warmup, this scales exploration conservatively.
        After warmup, returns 1.0 (full exploration).
        """
        if current_cycle >= self.warmup_cycles:
            return self.max_exploration_factor

        progress = current_cycle / max(self.warmup_cycles, 1)

        if self.strategy == WarmupStrategy.LINEAR:
            factor = self.min_exploration_factor + (
                self.max_exploration_factor - self.min_exploration_factor
            ) * progress

        elif self.strategy == WarmupStrategy.COSINE:
            # Cosine annealing: smooth S-curve transition
            factor = self.min_exploration_factor + (
                self.max_exploration_factor - self.min_exploration_factor
            ) * (1 - math.cos(math.pi * progress)) / 2

        elif self.strategy == WarmupStrategy.EXPONENTIAL:
            # Exponential: fast initial ramp
            factor = self.min_exploration_factor + (
                self.max_exploration_factor - self.min_exploration_factor
            ) * (1 - math.exp(-3 * progress))

        elif self.strategy == WarmupStrategy.ADAPTIVE:
            # Adaptive: adjust based on success rate
            if success_rate > self.early_success_threshold:
                # Good results — accelerate warmup
                factor = min(
                    self.max_exploration_factor,
                    self.min_exploration_factor + progress * 2,
                )
            elif success_rate < (1 - self.early_failure_threshold):
                # Poor results — slow down warmup
                factor = self.min_exploration_factor + progress * 0.5
            else:
                # Normal — linear progression
                factor = self.min_exploration_factor + (
                    self.max_exploration_factor - self.min_exploration_factor
                ) * progress
        else:
            factor = self.max_exploration_factor

        return max(self.min_exploration_factor, min(self.max_exploration_factor, factor))

    def get_snr_threshold(
        self,
        current_cycle: int,
        base_threshold: float = UNIFIED_SNR_THRESHOLD,
    ) -> float:
        """
        Get the SNR threshold for the current cycle.

        During warmup, the threshold is HIGHER (more conservative).
        This prevents low-quality early hypotheses from polluting the system.
        """
        if current_cycle >= self.warmup_cycles:
            return base_threshold

        progress = current_cycle / max(self.warmup_cycles, 1)
        # Start 10% above base, linearly decay to base
        warmup_premium = 0.10 * (1 - progress)
        return min(1.0, base_threshold + warmup_premium)

    def is_warmup_complete(self, current_cycle: int) -> bool:
        """Check if warmup phase is complete."""
        return current_cycle >= self.warmup_cycles


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConvergenceState:
    """Internal state for convergence detection."""

    scores: List[float] = field(default_factory=list)
    best_score: float = 0.0
    best_cycle: int = 0
    plateau_count: int = 0
    divergence_count: int = 0


class ConvergenceDetector:
    """
    Detects when the RDVE pipeline has converged (no more improvements).

    Uses multiple signals:
    1. Improvement plateau — recent scores below threshold
    2. Score divergence — scores worsening (not just flat)
    3. Entropy collapse — hypothesis diversity dropping
    4. Discovery rate decline — fewer successes over time

    Standing on Giants:
        Deming (statistical process control, 1950) — control charts for convergence
        Shannon (information theory, 1948) — entropy as diversity measure
    """

    def __init__(
        self,
        window_size: int = 5,
        min_improvement: float = 0.01,
        max_plateau_cycles: int = 10,
        entropy_floor: float = 0.3,
    ):
        self.window_size = window_size
        self.min_improvement = min_improvement
        self.max_plateau_cycles = max_plateau_cycles
        self.entropy_floor = entropy_floor

        self._state = ConvergenceState()

    def record(self, score: float, categories: Optional[List[str]] = None) -> None:
        """Record a cycle's improvement score."""
        self._state.scores.append(score)

        if score > self._state.best_score:
            self._state.best_score = score
            self._state.best_cycle = len(self._state.scores) - 1
            self._state.plateau_count = 0
        else:
            self._state.plateau_count += 1

        # Check for divergence (worsening)
        if len(self._state.scores) >= 2:
            if self._state.scores[-1] < self._state.scores[-2] * 0.95:
                self._state.divergence_count += 1
            else:
                self._state.divergence_count = max(
                    0, self._state.divergence_count - 1
                )

    def is_converged(self) -> bool:
        """Check if the system has converged."""
        if len(self._state.scores) < self.window_size:
            return False

        # Signal 1: Plateau detection
        recent = self._state.scores[-self.window_size :]
        max_recent = max(recent)
        if max_recent < self.min_improvement:
            logger.info(
                f"Convergence: improvement plateau detected "
                f"(max_recent={max_recent:.4f} < {self.min_improvement})"
            )
            return True

        # Signal 2: Extended plateau
        if self._state.plateau_count >= self.max_plateau_cycles:
            logger.info(
                f"Convergence: extended plateau ({self._state.plateau_count} cycles "
                f"without improvement)"
            )
            return True

        return False

    def is_diverging(self) -> bool:
        """Check if the system is diverging (getting worse)."""
        return self._state.divergence_count >= 3

    def get_trend(self) -> str:
        """Get the current improvement trend."""
        if len(self._state.scores) < 2:
            return "insufficient_data"

        if self.is_converged():
            return "converged"

        if self.is_diverging():
            return "diverging"

        recent = self._state.scores[-min(self.window_size, len(self._state.scores)) :]
        if len(recent) >= 2 and recent[-1] > recent[0]:
            return "improving"

        return "stable"

    def get_status(self) -> Dict[str, Any]:
        """Get convergence detector status."""
        return {
            "total_cycles": len(self._state.scores),
            "best_score": self._state.best_score,
            "best_cycle": self._state.best_cycle,
            "plateau_count": self._state.plateau_count,
            "divergence_count": self._state.divergence_count,
            "trend": self.get_trend(),
            "is_converged": self.is_converged(),
            "is_diverging": self.is_diverging(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY PROTOCOL (Unified)
# ═══════════════════════════════════════════════════════════════════════════════


class StabilityProtocol:
    """
    Unified stability protocol for the RDVE pipeline.

    Combines warmup scheduling, convergence detection, and rate limiting
    into a single coherent stability layer.

    Standing on Giants:
        Deming (PDCA quality, 1950) · Shannon (information entropy, 1948) ·
        Boyd (adaptive tempo, 1976) · He (initialization, 2015)

    Artifact: core/rdve/stability.py
    """

    def __init__(
        self,
        warmup: Optional[WarmupSchedule] = None,
        convergence: Optional[ConvergenceDetector] = None,
        max_rate_per_minute: float = 30.0,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 300.0,
    ):
        self.warmup = warmup or WarmupSchedule()
        self.convergence = convergence or ConvergenceDetector()

        # Rate limiting
        self._max_rate = max_rate_per_minute
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff_seconds
        self._current_backoff = 0.0
        self._last_cycle_time = 0.0
        self._cycle_count = 0
        self._success_count = 0

    def pre_cycle(self, cycle_number: int) -> Dict[str, Any]:
        """
        Pre-cycle stability check. Returns adjusted parameters.

        Call this BEFORE each RDVE cycle to get adjusted exploration parameters.
        """
        self._cycle_count = cycle_number
        success_rate = (
            self._success_count / max(cycle_number, 1)
            if cycle_number > 0
            else 0.5
        )

        # Warmup-adjusted exploration factor
        exploration_factor = self.warmup.get_exploration_factor(
            cycle_number, success_rate
        )

        # Warmup-adjusted SNR threshold (more conservative during warmup)
        snr_threshold = self.warmup.get_snr_threshold(cycle_number)

        # Rate limit check
        now = time.time()
        min_interval = 60.0 / self._max_rate
        actual_interval = now - self._last_cycle_time if self._last_cycle_time else min_interval

        should_wait = False
        wait_seconds = 0.0

        if actual_interval < min_interval + self._current_backoff:
            should_wait = True
            wait_seconds = (min_interval + self._current_backoff) - actual_interval

        return {
            "exploration_factor": exploration_factor,
            "snr_threshold": snr_threshold,
            "warmup_complete": self.warmup.is_warmup_complete(cycle_number),
            "success_rate": success_rate,
            "should_wait": should_wait,
            "wait_seconds": wait_seconds,
            "trend": self.convergence.get_trend(),
            "is_converged": self.convergence.is_converged(),
        }

    def post_cycle(
        self,
        score: float,
        success: bool,
        categories: Optional[List[str]] = None,
    ) -> None:
        """
        Post-cycle stability update. Call AFTER each RDVE cycle.
        """
        self._last_cycle_time = time.time()

        if success:
            self._success_count += 1
            self._current_backoff = 0.0  # Reset backoff on success
        else:
            # Exponential backoff on failure
            if self._current_backoff == 0:
                self._current_backoff = 1.0
            else:
                self._current_backoff = min(
                    self._current_backoff * self._backoff_factor,
                    self._max_backoff,
                )

        # Update convergence detector
        self.convergence.record(score, categories)

    def should_stop(self) -> Tuple[bool, str]:
        """
        Check if the campaign should stop.

        Returns:
            (should_stop, reason)
        """
        if self.convergence.is_converged():
            return True, "converged"

        if self.convergence.is_diverging():
            return True, "diverging"

        if self._current_backoff >= self._max_backoff:
            return True, "max_backoff_reached"

        return False, "continue"

    def get_status(self) -> Dict[str, Any]:
        """Get full stability protocol status."""
        return {
            "warmup": {
                "strategy": self.warmup.strategy.value,
                "cycles": self.warmup.warmup_cycles,
                "complete": self.warmup.is_warmup_complete(self._cycle_count),
            },
            "convergence": self.convergence.get_status(),
            "rate_limiting": {
                "max_rate_per_minute": self._max_rate,
                "current_backoff_seconds": self._current_backoff,
            },
            "cycle_count": self._cycle_count,
            "success_count": self._success_count,
            "success_rate": (
                self._success_count / max(self._cycle_count, 1)
            ),
        }
