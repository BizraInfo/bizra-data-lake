"""
Domain Mixing Engine — Optimal Data Blending for LLM Training

Standing on Giants:
- DoReMi (Xie et al., 2024): Domain Reweighting with Minimax Optimization
- DRO (Distributionally Robust Optimization) for data mixing
- SlimPajama mixing ratios (Cerebras, 2023)
- DoGE: Domain Reweighting with Generalization Estimation

"Data mixing refers to determining the proportion (mixing ratio) of
 different data sources in a training corpus to optimize performance."
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MixingStrategy(Enum):
    """Data mixing strategies from DATA4LLM."""

    UNIFORM = "uniform"  # Equal proportions
    PROPORTIONAL = "proportional"  # Match source distribution
    TEMPERATURE = "temperature"  # Softmax with temperature
    DOREMI = "doremi"  # Domain Reweighting with Minimax
    DRO = "dro"  # Distributionally Robust Optimization
    INVERSE_PERPLEXITY = "inverse_ppl"  # Weight by inverse perplexity
    SKILL_BALANCED = "skill_balanced"  # Balance by skill requirements


@dataclass
class DomainStats:
    """Statistics for a single domain."""

    name: str
    count: int
    total_tokens: int
    mean_perplexity: float = 50.0
    mean_quality: float = 0.8
    mean_diversity: float = 0.7
    source_weight: float = 1.0  # Natural proportion in source data

    @property
    def token_density(self) -> float:
        return self.total_tokens / max(self.count, 1)


@dataclass
class MixingResult:
    """Result of mixing operation."""

    original_distribution: Dict[str, float]
    target_distribution: Dict[str, float]
    sample_indices: Dict[str, List[int]]  # Domain -> sampled indices
    total_samples: int
    strategy: MixingStrategy

    @property
    def mixing_ratios(self) -> Dict[str, float]:
        """Return the mixing ratios (target distribution)."""
        return self.target_distribution


class DomainMixer:
    """
    Domain-aware data mixing engine.

    Implements multiple mixing strategies:
    1. Uniform: Equal proportions across domains
    2. Proportional: Match natural distribution
    3. Temperature: Softmax smoothing with temperature
    4. Inverse Perplexity: Weight by quality (low perplexity = high quality)
    5. Skill-balanced: Balance by downstream task requirements
    """

    def __init__(
        self,
        strategy: MixingStrategy = MixingStrategy.TEMPERATURE,
        temperature: float = 1.0,
        min_domain_fraction: float = 0.01,  # Minimum 1% per domain
        max_domain_fraction: float = 0.50,  # Maximum 50% per domain
    ):
        self.strategy = strategy
        self.temperature = temperature
        self.min_fraction = min_domain_fraction
        self.max_fraction = max_domain_fraction

    def compute_domain_stats(
        self,
        texts: List[str],
        domains: List[str],
        perplexities: Optional[np.ndarray] = None,
        qualities: Optional[np.ndarray] = None,
    ) -> Dict[str, DomainStats]:
        """Compute statistics for each domain."""
        domain_texts = defaultdict(list)
        domain_indices = defaultdict(list)

        for i, (text, domain) in enumerate(zip(texts, domains)):
            domain_texts[domain].append(text)
            domain_indices[domain].append(i)

        stats = {}
        total_count = len(texts)

        for domain, indices in domain_indices.items():
            texts_in_domain = domain_texts[domain]
            count = len(texts_in_domain)
            total_tokens = sum(len(t.split()) for t in texts_in_domain)

            # Compute mean perplexity if available
            mean_ppl = 50.0
            if perplexities is not None:
                domain_ppls = perplexities[indices]
                mean_ppl = float(np.mean(domain_ppls))

            # Compute mean quality if available
            mean_qual = 0.8
            if qualities is not None:
                domain_quals = qualities[indices]
                mean_qual = float(np.mean(domain_quals))

            stats[domain] = DomainStats(
                name=domain,
                count=count,
                total_tokens=total_tokens,
                mean_perplexity=mean_ppl,
                mean_quality=mean_qual,
                source_weight=count / total_count,
            )

        return stats

    def _uniform_mixing(
        self,
        stats: Dict[str, DomainStats],
    ) -> Dict[str, float]:
        """Equal weight to all domains."""
        n_domains = len(stats)
        return {domain: 1.0 / n_domains for domain in stats}

    def _proportional_mixing(
        self,
        stats: Dict[str, DomainStats],
    ) -> Dict[str, float]:
        """Match natural source distribution."""
        return {domain: s.source_weight for domain, s in stats.items()}

    def _temperature_mixing(
        self,
        stats: Dict[str, DomainStats],
    ) -> Dict[str, float]:
        """
        Softmax with temperature.

        Temperature < 1: Sharper distribution (emphasize dominant)
        Temperature = 1: Natural distribution
        Temperature > 1: Smoother distribution (more uniform)
        """
        log_counts = np.array([math.log(max(s.count, 1)) for s in stats.values()])

        # Apply temperature
        scaled = log_counts / self.temperature

        # Softmax
        exp_scaled = np.exp(scaled - np.max(scaled))
        probs = exp_scaled / np.sum(exp_scaled)

        return {domain: float(p) for domain, p in zip(stats.keys(), probs)}

    def _inverse_perplexity_mixing(
        self,
        stats: Dict[str, DomainStats],
    ) -> Dict[str, float]:
        """
        Weight domains by inverse perplexity (quality proxy).

        Lower perplexity = higher quality = higher weight.
        """
        inv_ppls = np.array([1.0 / max(s.mean_perplexity, 1.0) for s in stats.values()])

        # Normalize
        probs = inv_ppls / np.sum(inv_ppls)

        return {domain: float(p) for domain, p in zip(stats.keys(), probs)}

    def _skill_balanced_mixing(
        self,
        stats: Dict[str, DomainStats],
        skill_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Balance by skill requirements.

        Default skill weights based on downstream task importance.
        """
        default_weights = {
            "code": 0.25,
            "math": 0.15,
            "science": 0.15,
            "writing": 0.20,
            "conversation": 0.15,
            "factual": 0.10,
        }

        weights = skill_weights or default_weights

        # Map domains to skills (simplified mapping)
        domain_weights = {}
        for domain in stats:
            # Check if domain matches any skill
            matched = False
            for skill, weight in weights.items():
                if skill in domain.lower():
                    domain_weights[domain] = weight
                    matched = True
                    break
            if not matched:
                domain_weights[domain] = 0.1  # Default weight

        # Normalize
        total = sum(domain_weights.values())
        return {d: w / total for d, w in domain_weights.items()}

    def compute_mixing_ratios(
        self,
        stats: Dict[str, DomainStats],
        skill_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute mixing ratios based on strategy."""
        if self.strategy == MixingStrategy.UNIFORM:
            ratios = self._uniform_mixing(stats)
        elif self.strategy == MixingStrategy.PROPORTIONAL:
            ratios = self._proportional_mixing(stats)
        elif self.strategy == MixingStrategy.TEMPERATURE:
            ratios = self._temperature_mixing(stats)
        elif self.strategy == MixingStrategy.INVERSE_PERPLEXITY:
            ratios = self._inverse_perplexity_mixing(stats)
        elif self.strategy == MixingStrategy.SKILL_BALANCED:
            ratios = self._skill_balanced_mixing(stats, skill_weights)
        else:
            ratios = self._proportional_mixing(stats)

        # Apply min/max constraints
        ratios = self._apply_constraints(ratios)

        return ratios

    def _apply_constraints(
        self,
        ratios: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply min/max fraction constraints."""
        constrained = {}

        for domain, ratio in ratios.items():
            constrained[domain] = max(self.min_fraction, min(self.max_fraction, ratio))

        # Renormalize
        total = sum(constrained.values())
        return {d: r / total for d, r in constrained.items()}

    def sample_by_ratio(
        self,
        texts: List[str],
        domains: List[str],
        target_ratios: Dict[str, float],
        total_samples: int,
    ) -> MixingResult:
        """
        Sample from each domain according to target ratios.

        Returns indices for each domain to include in the mixed dataset.
        """
        # Group indices by domain
        domain_indices = defaultdict(list)
        for i, domain in enumerate(domains):
            domain_indices[domain].append(i)

        # Compute samples per domain
        samples_per_domain = {}
        for domain, ratio in target_ratios.items():
            n_samples = int(total_samples * ratio)
            samples_per_domain[domain] = n_samples

        # Sample indices
        sample_indices = {}
        for domain, n_samples in samples_per_domain.items():
            available = domain_indices.get(domain, [])
            if len(available) >= n_samples:
                # Random sample without replacement
                sampled = np.random.choice(available, n_samples, replace=False).tolist()
            else:
                # Take all available, with replacement if needed
                if available:
                    sampled = list(available)
                    if n_samples > len(available):
                        extra = np.random.choice(
                            available, n_samples - len(available), replace=True
                        ).tolist()
                        sampled.extend(extra)
                else:
                    sampled = []
            sample_indices[domain] = sampled

        # Compute original distribution
        total_original = len(texts)
        original_dist = {
            domain: len(indices) / total_original
            for domain, indices in domain_indices.items()
        }

        return MixingResult(
            original_distribution=original_dist,
            target_distribution=target_ratios,
            sample_indices=sample_indices,
            total_samples=sum(len(v) for v in sample_indices.values()),
            strategy=self.strategy,
        )

    def mix(
        self,
        texts: List[str],
        domains: List[str],
        total_samples: Optional[int] = None,
        perplexities: Optional[np.ndarray] = None,
        qualities: Optional[np.ndarray] = None,
    ) -> MixingResult:
        """
        Execute the full mixing pipeline.

        1. Compute domain statistics
        2. Determine mixing ratios
        3. Sample according to ratios
        """
        # Compute stats
        stats = self.compute_domain_stats(texts, domains, perplexities, qualities)

        logger.info(f"Domain stats: {len(stats)} domains")
        for domain, s in stats.items():
            logger.debug(f"  {domain}: {s.count} samples, PPL={s.mean_perplexity:.1f}")

        # Compute ratios
        ratios = self.compute_mixing_ratios(stats)

        logger.info(f"Mixing ratios ({self.strategy.value}):")
        for domain, ratio in sorted(ratios.items(), key=lambda x: -x[1]):
            logger.info(f"  {domain}: {ratio:.2%}")

        # Sample
        n_samples = total_samples or len(texts)
        result = self.sample_by_ratio(texts, domains, ratios, n_samples)

        return result


class DistributionallyRobustOptimizer:
    """
    Distributionally Robust Optimization (DRO) for data mixing.

    DRO optimizes mixing ratios to minimize worst-case loss across domains,
    ensuring the model performs well even on underrepresented domains.

    Algorithm (simplified from DoReMi):
    1. Train proxy model on uniform mixture
    2. Compute per-domain losses
    3. Reweight to emphasize high-loss domains
    4. Iterate until convergence

    For this implementation, we use quality scores as proxy for loss
    (low quality = high "loss" = needs more weight).
    """

    def __init__(
        self,
        max_iterations: int = 10,
        learning_rate: float = 0.1,
        min_weight: float = 0.01,
        convergence_threshold: float = 0.001,
    ):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.convergence_threshold = convergence_threshold

    def _compute_domain_losses(
        self,
        stats: Dict[str, DomainStats],
    ) -> Dict[str, float]:
        """
        Compute proxy losses for each domain.

        Uses inverse quality as loss proxy.
        """
        losses = {}
        for domain, s in stats.items():
            # Lower quality = higher loss
            loss = 1.0 - s.mean_quality
            # Also factor in perplexity (high perplexity = harder = higher loss)
            loss += (s.mean_perplexity - 30) / 100  # Normalize PPL contribution
            losses[domain] = max(0.01, loss)

        return losses

    def optimize(
        self,
        stats: Dict[str, DomainStats],
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Run DRO optimization to find optimal mixing weights.

        Returns optimized weights that minimize worst-case performance.
        """
        n_domains = len(stats)

        # Initialize weights
        if initial_weights:
            weights = np.array([initial_weights.get(d, 1.0 / n_domains) for d in stats])
        else:
            weights = np.ones(n_domains) / n_domains

        domains = list(stats.keys())

        for iteration in range(self.max_iterations):
            # Compute current losses
            losses = self._compute_domain_losses(stats)
            loss_array = np.array([losses[d] for d in domains])

            # Compute weighted average loss
            weighted_loss = np.sum(weights * loss_array)

            # Update weights using exponentiated gradient
            # Domains with higher loss get more weight
            gradients = loss_array - weighted_loss
            weights = weights * np.exp(self.learning_rate * gradients)

            # Project onto simplex (normalize)
            weights = np.maximum(weights, self.min_weight)
            weights = weights / np.sum(weights)

            # Check convergence
            if iteration > 0 and np.max(np.abs(gradients)) < self.convergence_threshold:
                logger.info(f"DRO converged at iteration {iteration}")
                break

        # Convert to dict
        optimized = {domain: float(w) for domain, w in zip(domains, weights)}

        logger.info("DRO optimization complete:")
        for domain, w in sorted(optimized.items(), key=lambda x: -x[1]):
            original = stats[domain].source_weight
            logger.info(f"  {domain}: {original:.2%} -> {w:.2%}")

        return optimized

    def mix_with_dro(
        self,
        texts: List[str],
        domains: List[str],
        perplexities: Optional[np.ndarray] = None,
        qualities: Optional[np.ndarray] = None,
        total_samples: Optional[int] = None,
    ) -> MixingResult:
        """
        Full DRO-based mixing pipeline.
        """
        # Compute domain stats
        mixer = DomainMixer(strategy=MixingStrategy.DRO)
        stats = mixer.compute_domain_stats(texts, domains, perplexities, qualities)

        # Run DRO optimization
        optimized_weights = self.optimize(stats)

        # Sample according to optimized weights
        n_samples = total_samples or len(texts)
        result = mixer.sample_by_ratio(texts, domains, optimized_weights, n_samples)
        result.strategy = MixingStrategy.DRO

        return result
