"""
DP-FedLoRA — Differential Privacy for Federated LoRA Updates

Implements privacy-safe federated fine-tuning by applying differential
privacy guarantees to LoRA weight updates before federation.

Algorithm:
    1. Compute LoRA weight delta: Delta_B = B_new - B_old
    2. Clip gradient: Delta_B_clipped = Delta_B * min(1, C / ||Delta_B||_2)
    3. Add Gaussian noise: Delta_B_dp = Delta_B_clipped + N(0, sigma^2 * C^2 * I)
    4. Send Delta_B_dp to federation (privacy-safe)

Privacy Guarantee:
    (epsilon, delta)-differential privacy where:
    epsilon = C * sqrt(2 * ln(1.25 / delta)) / sigma

Standing on Giants:
- Cynthia Dwork (2006): Differential Privacy
- Abadi et al. (2016): Deep Learning with Differential Privacy
- Hu et al. (2021): LoRA — Low-Rank Adaptation
- McMahan et al. (2017): Federated Learning
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default DP parameters
DEFAULT_CLIP_THRESHOLD = 1.0  # L2 norm clipping bound (C)
DEFAULT_NOISE_MULTIPLIER = 1.1  # sigma multiplier
DEFAULT_DELTA = 1e-5  # Privacy failure probability


@dataclass
class DPConfig:
    """Differential privacy configuration."""

    clip_threshold: float = DEFAULT_CLIP_THRESHOLD
    noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER
    delta: float = DEFAULT_DELTA
    target_epsilon: Optional[float] = None  # If set, auto-tune sigma

    @property
    def epsilon(self) -> float:
        """Compute the privacy budget (epsilon) for this configuration."""
        if self.noise_multiplier <= 0:
            return float("inf")
        return (
            self.clip_threshold
            * math.sqrt(2.0 * math.log(1.25 / self.delta))
            / self.noise_multiplier
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_threshold": self.clip_threshold,
            "noise_multiplier": self.noise_multiplier,
            "delta": self.delta,
            "epsilon": round(self.epsilon, 4),
        }


@dataclass
class LoRAWeightDelta:
    """A LoRA weight update (matrix delta)."""

    values: List[List[float]]  # 2D matrix of weight deltas
    rank: int = 0
    layer_name: str = ""

    def __post_init__(self):
        if self.values and not self.rank:
            self.rank = len(self.values[0]) if self.values else 0

    @property
    def rows(self) -> int:
        return len(self.values)

    @property
    def cols(self) -> int:
        return len(self.values[0]) if self.values else 0


@dataclass
class DPResult:
    """Result of applying differential privacy to a weight update."""

    success: bool
    original_norm: float = 0.0
    clipped_norm: float = 0.0
    was_clipped: bool = False
    noise_std: float = 0.0
    epsilon: float = 0.0
    delta: float = 0.0
    privatized_delta: Optional[LoRAWeightDelta] = None
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )


class GradientClipper:
    """
    Clips gradient updates to bound their L2 norm.

    This is the first step of the DP mechanism: ensuring bounded
    sensitivity before noise injection.
    """

    def __init__(self, clip_threshold: float = DEFAULT_CLIP_THRESHOLD):
        if clip_threshold <= 0:
            raise ValueError("Clip threshold must be positive")
        self.clip_threshold = clip_threshold

    def compute_l2_norm(self, matrix: List[List[float]]) -> float:
        """Compute the Frobenius (L2) norm of a matrix."""
        total = 0.0
        for row in matrix:
            for val in row:
                total += val * val
        return math.sqrt(total)

    def clip(self, delta: LoRAWeightDelta) -> Tuple[LoRAWeightDelta, float, bool]:
        """
        Clip the weight delta to bound its L2 norm.

        If ||delta||_2 > C, scale delta by C / ||delta||_2.

        Args:
            delta: The weight delta to clip

        Returns:
            Tuple of (clipped_delta, original_norm, was_clipped)
        """
        norm = self.compute_l2_norm(delta.values)

        if norm <= self.clip_threshold:
            return delta, norm, False

        # Scale factor: C / ||delta||_2
        scale = self.clip_threshold / norm

        clipped_values = [[val * scale for val in row] for row in delta.values]

        clipped = LoRAWeightDelta(
            values=clipped_values,
            rank=delta.rank,
            layer_name=delta.layer_name,
        )

        return clipped, norm, True


class NoiseInjector:
    """
    Injects calibrated Gaussian noise for differential privacy.

    The noise standard deviation is sigma * C where:
    - sigma is the noise multiplier
    - C is the clipping threshold
    """

    def __init__(
        self,
        noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
        clip_threshold: float = DEFAULT_CLIP_THRESHOLD,
    ):
        if noise_multiplier < 0:
            raise ValueError("Noise multiplier must be non-negative")
        self.noise_multiplier = noise_multiplier
        self.clip_threshold = clip_threshold

    @property
    def noise_std(self) -> float:
        """Standard deviation of the injected noise."""
        return self.noise_multiplier * self.clip_threshold

    def inject(
        self,
        delta: LoRAWeightDelta,
        rng: Optional[Any] = None,
    ) -> LoRAWeightDelta:
        """
        Add Gaussian noise to the weight delta.

        Args:
            delta: The (clipped) weight delta
            rng: Optional random number generator (for deterministic testing)

        Returns:
            Noisy weight delta
        """
        import random

        if rng is None:
            rng = random

        std = self.noise_std

        noisy_values = [
            [val + rng.gauss(0.0, std) for val in row] for row in delta.values
        ]

        return LoRAWeightDelta(
            values=noisy_values,
            rank=delta.rank,
            layer_name=delta.layer_name,
        )


class DPFedLoRAUpdate:
    """
    End-to-end DP-FedLoRA pipeline.

    Combines gradient clipping and noise injection to produce
    differentially private LoRA weight updates suitable for
    federation.

    Usage:
        dp = DPFedLoRAUpdate(config=DPConfig(clip_threshold=1.0, noise_multiplier=1.1))
        result = dp.privatize(weight_delta)
        if result.success:
            send_to_federation(result.privatized_delta)
    """

    def __init__(self, config: Optional[DPConfig] = None):
        self._config = config or DPConfig()
        self._clipper = GradientClipper(self._config.clip_threshold)
        self._injector = NoiseInjector(
            noise_multiplier=self._config.noise_multiplier,
            clip_threshold=self._config.clip_threshold,
        )
        self._update_count = 0

    @property
    def config(self) -> DPConfig:
        return self._config

    @property
    def update_count(self) -> int:
        return self._update_count

    def privatize(
        self,
        delta: LoRAWeightDelta,
        rng: Optional[Any] = None,
    ) -> DPResult:
        """
        Apply differential privacy to a LoRA weight update.

        Pipeline:
            1. Clip gradient to bound sensitivity
            2. Inject calibrated Gaussian noise
            3. Return privatized delta with privacy accounting

        Args:
            delta: Raw LoRA weight delta
            rng: Optional RNG for deterministic testing

        Returns:
            DPResult with the privatized delta and privacy metrics
        """
        try:
            # Step 1: Clip
            clipped, original_norm, was_clipped = self._clipper.clip(delta)

            clipped_norm = self._clipper.compute_l2_norm(clipped.values)

            # Step 2: Add noise
            noisy = self._injector.inject(clipped, rng=rng)

            self._update_count += 1

            return DPResult(
                success=True,
                original_norm=original_norm,
                clipped_norm=clipped_norm,
                was_clipped=was_clipped,
                noise_std=self._injector.noise_std,
                epsilon=self._config.epsilon,
                delta=self._config.delta,
                privatized_delta=noisy,
            )

        except Exception as e:
            logger.error("DP-FedLoRA privatization failed: %s", e)
            return DPResult(
                success=False,
                error=str(e),
            )

    def compose_privacy(self, num_updates: int) -> Dict[str, float]:
        """
        Compute composed privacy budget over multiple updates.

        Uses basic composition theorem: epsilon_total = sqrt(n) * epsilon_per_step.
        (Advanced composition for Gaussian mechanism)

        Args:
            num_updates: Number of updates to compose

        Returns:
            Dict with composed epsilon and delta
        """
        per_step_epsilon = self._config.epsilon
        # Advanced composition for Gaussian mechanism
        composed_epsilon = per_step_epsilon * math.sqrt(
            2.0 * num_updates * math.log(1.0 / self._config.delta)
        )
        composed_delta = num_updates * self._config.delta

        return {
            "per_step_epsilon": round(per_step_epsilon, 4),
            "composed_epsilon": round(composed_epsilon, 4),
            "composed_delta": composed_delta,
            "num_updates": num_updates,
        }


__all__ = [
    "DPFedLoRAUpdate",
    "DPConfig",
    "GradientClipper",
    "NoiseInjector",
    "LoRAWeightDelta",
    "DPResult",
    "DEFAULT_CLIP_THRESHOLD",
    "DEFAULT_NOISE_MULTIPLIER",
]
