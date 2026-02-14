"""
NK-Landscape Autopoietic Monitor — Evolutionary Fitness Topology

╔══════════════════════════════════════════════════════════════════════════════╗
║   HP-03: Autopoietic Fitness Landscape Topology (SNR 0.94)                   ║
║   Discovery: 4-component fitness creates Pareto surface on NK landscape.     ║
║   Insight: Selection pressure σ²/s ≈ 2.3 → punctuated equilibrium regime.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Kauffman (1993): "The Origins of Order" — NK fitness landscapes
  - N = number of genes (traits)
  - K = epistatic interactions per gene
  - K=0: smooth landscape (Mt. Fuji)
  - K=N-1: fully rugged (random)
  - Phase transition at K ≈ 2-4 for most N
- Holland (1975): "Adaptation in Natural and Artificial Systems"
- Maturana & Varela (1980): Autopoiesis — self-creating systems
- Gould & Eldredge (1972): Punctuated equilibrium
- Wright (1932): Adaptive landscapes in population genetics
- Fisher (1930): Fundamental theorem of natural selection

Mathematical Foundation:

    NK-Landscape:
        f(x) = (1/N) × Σᵢ fᵢ(xᵢ, xᵢ₁, ..., xᵢₖ)

    where:
        N = 8 (Ihsān dimensions)
        K = epistatic interactions (how many dimensions affect each other)
        fᵢ = fitness contribution of gene i, depending on K neighbors
        x = genome (configuration of dimension weights)

    Ruggedness Measure (Weinberger, 1990):
        ρ(d) = corr(f(x), f(x')) for ||x - x'|| = d
        
        Correlation length: τ = -1/ln(ρ(1))
        Short τ = rugged landscape (many local optima)
        Long τ = smooth landscape (few local optima)

    Selection Pressure (Fisher, 1930):
        s = σ²_fitness / mean_fitness
        
        σ²/s ≈ 2.3 → punctuated equilibrium:
        - Long periods of stasis (stuck on local peak)
        - Brief bursts of rapid change (crossing valleys)
        - Innovation triggers needed when plateau detected

BIZRA Integration:
    N = 8 (Ihsān dimensions)
    K = determined by IHSAN_WEIGHTS dependency structure
    Fitness = 0.4×Ihsān + 0.3×SNR + 0.2×Novelty + 0.1×Efficiency

Complexity: O(N × 2^K) for landscape evaluation, O(N²) for ruggedness
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Autopoiesis fitness components (from __init__.py)
# ═══════════════════════════════════════════════════════════════════════════════

N_DIMENSIONS: Final[int] = 8                    # Ihsān dimension count
FITNESS_IHSAN_WEIGHT: Final[float] = 0.4        # Ihsān score weight
FITNESS_SNR_WEIGHT: Final[float] = 0.3          # SNR weight
FITNESS_NOVELTY_WEIGHT: Final[float] = 0.2      # Novelty weight
FITNESS_EFFICIENCY_WEIGHT: Final[float] = 0.1   # Efficiency weight

# Punctuated equilibrium detection thresholds
PLATEAU_WINDOW: Final[int] = 20                 # Generations to detect stasis
PLATEAU_THRESHOLD: Final[float] = 0.001         # Minimum improvement to not be stasis
INNOVATION_BURST_STRENGTH: Final[float] = 0.3   # Mutation rate during innovation burst
INNOVATION_BURST_DURATION: Final[int] = 5       # Generations of elevated mutation

# NK landscape parameters
DEFAULT_K: Final[int] = 2                       # Epistatic interactions (moderate ruggedness)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FitnessSnapshot:
    """A single fitness measurement at a point in evolution time."""
    generation: int
    ihsan_score: float
    snr_score: float
    novelty_score: float
    efficiency_score: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    @property
    def composite_fitness(self) -> float:
        """Weighted composite fitness (the selection target)."""
        return (
            FITNESS_IHSAN_WEIGHT * self.ihsan_score
            + FITNESS_SNR_WEIGHT * self.snr_score
            + FITNESS_NOVELTY_WEIGHT * self.novelty_score
            + FITNESS_EFFICIENCY_WEIGHT * self.efficiency_score
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "composite_fitness": round(self.composite_fitness, 6),
            "ihsan": round(self.ihsan_score, 4),
            "snr": round(self.snr_score, 4),
            "novelty": round(self.novelty_score, 4),
            "efficiency": round(self.efficiency_score, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class LandscapeMetrics:
    """
    Computed metrics about the NK fitness landscape topology.

    These metrics determine whether the system is in:
    - Smooth regime (K low): Gradient ascent works, steady improvement
    - Rugged regime (K moderate): Local optima traps, need exploration
    - Chaotic regime (K high): Random search, innovation bursts needed
    """
    N: int                              # Dimension count
    K: int                              # Epistatic interaction count
    ruggedness: float                   # Correlation length inverse (0=smooth, 1=maximally rugged)
    correlation_length: float           # τ — how far you can move and predict fitness
    num_local_optima_estimate: int      # Estimated local peaks
    selection_pressure: float           # σ²/s — Fisher's selection gradient
    is_punctuated_equilibrium: bool     # σ²/s in [1.5, 3.5] range
    regime: str                         # "smooth", "rugged", or "chaotic"
    plateau_detected: bool              # Currently in fitness stasis
    plateau_length: int                 # How many generations on plateau
    innovation_recommended: bool        # Should we trigger burst?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "K": self.K,
            "ruggedness": round(self.ruggedness, 4),
            "correlation_length": round(self.correlation_length, 4),
            "num_local_optima_estimate": self.num_local_optima_estimate,
            "selection_pressure": round(self.selection_pressure, 4),
            "is_punctuated_equilibrium": self.is_punctuated_equilibrium,
            "regime": self.regime,
            "plateau_detected": self.plateau_detected,
            "plateau_length": self.plateau_length,
            "innovation_recommended": self.innovation_recommended,
        }


@dataclass
class InnovationTrigger:
    """
    Record of an innovation burst trigger event.

    When the system detects a fitness plateau, it triggers an innovation
    burst — elevated mutation rates to escape local optima.
    """
    generation: int
    plateau_length: int
    fitness_at_trigger: float
    burst_strength: float
    burst_duration: int
    landscape_metrics: LandscapeMetrics
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "plateau_length": self.plateau_length,
            "fitness_at_trigger": round(self.fitness_at_trigger, 6),
            "burst_strength": self.burst_strength,
            "burst_duration": self.burst_duration,
            "landscape_metrics": self.landscape_metrics.to_dict(),
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NK LANDSCAPE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class NKLandscapeMonitor:
    """
    NK-Landscape fitness topology monitor for BIZRA autopoiesis.

    Tracks the evolutionary fitness landscape of the agent ecosystem
    and detects when innovation bursts are needed to escape local optima.

    The monitor computes:
    1. Landscape ruggedness (from fitness correlation)
    2. Selection pressure (from fitness variance)
    3. Plateau detection (from fitness stagnation)
    4. Innovation triggers (when plateau + ruggedness thresholds met)

    Standing on Giants:
    - Kauffman (1993): NK fitness landscapes
    - Wright (1932): Adaptive landscapes
    - Gould & Eldredge (1972): Punctuated equilibrium
    - Fisher (1930): Fundamental theorem of natural selection

    Usage:
        monitor = NKLandscapeMonitor(N=8, K=2)
        
        for generation in range(100):
            snapshot = FitnessSnapshot(
                generation=generation,
                ihsan_score=0.96, snr_score=0.92,
                novelty_score=0.85, efficiency_score=0.88,
            )
            metrics = monitor.observe(snapshot)
            
            if metrics.innovation_recommended:
                # Trigger innovation burst
                mutation_rate = metrics_to_mutation_rate(metrics)
    """

    def __init__(
        self,
        N: int = N_DIMENSIONS,
        K: int = DEFAULT_K,
        plateau_window: int = PLATEAU_WINDOW,
        plateau_threshold: float = PLATEAU_THRESHOLD,
    ) -> None:
        self._N = N
        self._K = K
        self._plateau_window = plateau_window
        self._plateau_threshold = plateau_threshold

        self._history: List[FitnessSnapshot] = []
        self._triggers: List[InnovationTrigger] = []
        self._active_burst: Optional[InnovationTrigger] = None
        self._burst_remaining: int = 0

    def observe(self, snapshot: FitnessSnapshot) -> LandscapeMetrics:
        """
        Observe a new fitness measurement and update landscape analysis.

        This is the main entry point — call once per generation.

        Returns:
            LandscapeMetrics with current landscape topology analysis
        """
        self._history.append(snapshot)

        # Compute metrics
        metrics = self._compute_metrics()

        # Check for innovation trigger
        if metrics.innovation_recommended and self._burst_remaining <= 0:
            trigger = InnovationTrigger(
                generation=snapshot.generation,
                plateau_length=metrics.plateau_length,
                fitness_at_trigger=snapshot.composite_fitness,
                burst_strength=INNOVATION_BURST_STRENGTH,
                burst_duration=INNOVATION_BURST_DURATION,
                landscape_metrics=metrics,
            )
            self._triggers.append(trigger)
            self._active_burst = trigger
            self._burst_remaining = INNOVATION_BURST_DURATION

            logger.info(
                f"INNOVATION BURST triggered at gen {snapshot.generation}: "
                f"plateau={metrics.plateau_length}, ruggedness={metrics.ruggedness:.3f}"
            )

        # Decrement burst counter
        if self._burst_remaining > 0:
            self._burst_remaining -= 1

        return metrics

    def _compute_metrics(self) -> LandscapeMetrics:
        """Compute full landscape metrics from observation history."""
        history = self._history
        n = len(history)

        if n < 3:
            return LandscapeMetrics(
                N=self._N, K=self._K,
                ruggedness=0.0, correlation_length=float("inf"),
                num_local_optima_estimate=1,
                selection_pressure=0.0,
                is_punctuated_equilibrium=False,
                regime="smooth",
                plateau_detected=False, plateau_length=0,
                innovation_recommended=False,
            )

        fitnesses = np.array([s.composite_fitness for s in history])

        # ── Ruggedness (autocorrelation of fitness series) ──────────────
        ruggedness, corr_length = self._compute_ruggedness(fitnesses)

        # ── Selection pressure (Fisher's theorem) ───────────────────────
        selection_pressure = self._compute_selection_pressure(fitnesses)
        is_pe = 1.5 <= selection_pressure <= 3.5

        # ── Regime classification ───────────────────────────────────────
        if ruggedness < 0.2:
            regime = "smooth"
        elif ruggedness < 0.6:
            regime = "rugged"
        else:
            regime = "chaotic"

        # ── Local optima estimate (Kauffman approximation) ──────────────
        # For NK model: expected peaks ≈ 2^N / (N+1) for K = N-1
        # For small K: much fewer
        if self._K == 0:
            num_optima = 1
        else:
            num_optima = max(1, int(2 ** (self._N * self._K / (self._N - 1 + 1e-10)) / (self._N + 1)))
            num_optima = min(num_optima, 2 ** self._N)  # Cap

        # ── Plateau detection ───────────────────────────────────────────
        plateau_detected, plateau_length = self._detect_plateau(fitnesses)

        # ── Innovation recommendation ───────────────────────────────────
        innovation_recommended = (
            plateau_detected
            and plateau_length >= self._plateau_window
            and ruggedness > 0.1  # Only if landscape is actually rugged
        )

        return LandscapeMetrics(
            N=self._N,
            K=self._K,
            ruggedness=ruggedness,
            correlation_length=corr_length,
            num_local_optima_estimate=num_optima,
            selection_pressure=selection_pressure,
            is_punctuated_equilibrium=is_pe,
            regime=regime,
            plateau_detected=plateau_detected,
            plateau_length=plateau_length,
            innovation_recommended=innovation_recommended,
        )

    def _compute_ruggedness(self, fitnesses: np.ndarray) -> Tuple[float, float]:
        """
        Compute landscape ruggedness via autocorrelation.

        Ruggedness ≈ 1 - ρ(1) where ρ(1) is lag-1 autocorrelation.
        Correlation length τ = -1/ln(|ρ(1)|).

        Standing on: Weinberger (1990), Kauffman (1993).
        """
        n = len(fitnesses)
        if n < 5:
            return 0.0, float("inf")

        mean = np.mean(fitnesses)
        var = np.var(fitnesses)

        if var < 1e-12:
            return 0.0, float("inf")  # Constant fitness = perfectly smooth

        # Lag-1 autocorrelation
        cov_1 = np.mean((fitnesses[:-1] - mean) * (fitnesses[1:] - mean))
        rho_1 = cov_1 / var

        # Ruggedness
        ruggedness = float(max(0.0, min(1.0, 1.0 - abs(rho_1))))

        # Correlation length
        if abs(rho_1) < 1e-10 or abs(rho_1) >= 1.0:
            corr_length = float("inf") if abs(rho_1) >= 1.0 - 1e-10 else 1.0
        else:
            corr_length = -1.0 / math.log(abs(rho_1))

        return ruggedness, corr_length

    def _compute_selection_pressure(self, fitnesses: np.ndarray) -> float:
        """
        Compute selection pressure σ²/s (Fisher's theorem).

        σ² = variance of fitness
        s = mean fitness (selection coefficient)

        σ²/s ≈ 2.3 → punctuated equilibrium regime
        σ²/s < 1.0 → weak selection (drift dominates)
        σ²/s > 4.0 → strong selection (rapid adaptation)
        """
        if len(fitnesses) < 3:
            return 0.0

        # Use recent window for current pressure
        recent = fitnesses[-min(len(fitnesses), 50):]
        variance = float(np.var(recent))
        mean = float(np.mean(recent))

        if mean < 1e-10:
            return 0.0

        return variance / mean

    def _detect_plateau(self, fitnesses: np.ndarray) -> Tuple[bool, int]:
        """
        Detect if the system is in a fitness plateau (stasis).

        A plateau is detected when the best fitness hasn't improved
        by more than PLATEAU_THRESHOLD over the last PLATEAU_WINDOW generations.
        """
        n = len(fitnesses)
        if n < self._plateau_window:
            return False, 0

        recent = fitnesses[-self._plateau_window:]
        improvement = float(np.max(recent) - np.min(recent))

        if improvement < self._plateau_threshold:
            # Count how far back the plateau extends
            plateau_length = self._plateau_window
            for i in range(self._plateau_window, n):
                idx = n - i - 1
                if abs(fitnesses[idx] - fitnesses[-1]) > self._plateau_threshold:
                    break
                plateau_length = i + 1

            return True, plateau_length

        return False, 0

    # ─────────────────────────────────────────────────────────────────────────
    # MUTATION RATE CONTROL
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def current_mutation_rate(self) -> float:
        """
        Get current recommended mutation rate.

        During innovation bursts, mutation rate is elevated.
        Otherwise, returns the baseline rate.
        """
        from core.autopoiesis import MUTATION_RATE

        if self._burst_remaining > 0:
            return INNOVATION_BURST_STRENGTH
        return MUTATION_RATE

    @property
    def is_in_burst(self) -> bool:
        """Whether an innovation burst is currently active."""
        return self._burst_remaining > 0

    # ─────────────────────────────────────────────────────────────────────────
    # ANALYSIS & DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    def evolution_trace(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent evolution trace for visualization."""
        return [s.to_dict() for s in self._history[-limit:]]

    def trigger_history(self) -> List[Dict[str, Any]]:
        """Get history of all innovation triggers."""
        return [t.to_dict() for t in self._triggers]

    def landscape_summary(self) -> Dict[str, Any]:
        """Comprehensive landscape summary."""
        if not self._history:
            return {"status": "no_observations", "generations": 0}

        fitnesses = [s.composite_fitness for s in self._history]
        metrics = self._compute_metrics()

        return {
            "generations": len(self._history),
            "current_fitness": round(fitnesses[-1], 6),
            "best_fitness": round(max(fitnesses), 6),
            "mean_fitness": round(sum(fitnesses) / len(fitnesses), 6),
            "fitness_std": round(float(np.std(fitnesses)), 6),
            "landscape": metrics.to_dict(),
            "innovation_triggers": len(self._triggers),
            "burst_active": self.is_in_burst,
            "burst_remaining": self._burst_remaining,
            "current_mutation_rate": round(self.current_mutation_rate, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "NKLandscapeMonitor",
    "FitnessSnapshot",
    "LandscapeMetrics",
    "InnovationTrigger",
    # Constants
    "N_DIMENSIONS",
    "DEFAULT_K",
    "PLATEAU_WINDOW",
    "PLATEAU_THRESHOLD",
    "INNOVATION_BURST_STRENGTH",
    "INNOVATION_BURST_DURATION",
]
