"""
Interactive Denoising — Bayesian Belief Updates from User Corrections

Implements the SAPE v1.∞ Interactive Denoising layer that updates
L2 HHMM beliefs from user corrections. When a user says "that meeting
moved" or "actually, focus on X", the denoiser adjusts belief
distributions using Bayes' theorem.

Math:
    P(intent | observation) = P(observation | intent) * P(intent) / P(observation)

    Where:
    - P(intent) is the prior belief about what the user wants
    - P(observation | intent) is the likelihood of seeing this correction
      given the true intent
    - P(observation) is the marginal (normalizing constant)

Standing on Giants:
- Thomas Bayes (1763): Bayesian inference
- Claude Shannon (1948): Information-theoretic noise reduction
- Rudolf Kalman (1960): Optimal state estimation
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default likelihood for user corrections
CORRECTION_LIKELIHOOD = 0.9  # P(correction | intent_changed)
FALSE_ALARM_RATE = 0.05  # P(correction | intent_unchanged)

# Minimum belief threshold before pruning
MIN_BELIEF_THRESHOLD = 0.001

# Maximum tracked priorities
MAX_PRIORITIES = 50

# Decay rate for stale beliefs (per hour)
BELIEF_DECAY_RATE = 0.02


class CorrectionType(Enum):
    """Types of user corrections."""

    DISMISS = "dismiss"  # User dismisses a suggestion
    PROMOTE = "promote"  # User promotes a topic
    RESCHEDULE = "reschedule"  # User reschedules something
    CANCEL = "cancel"  # User cancels entirely
    CLARIFY = "clarify"  # User clarifies intent
    REDIRECT = "redirect"  # User redirects to different topic


@dataclass
class BeliefState:
    """
    Belief distribution over user priorities.

    Each priority has a probability representing our belief that
    it is the user's current focus.
    """

    priorities: Dict[str, float] = field(default_factory=dict)
    last_updated: str = ""
    update_count: int = 0

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def normalize(self) -> None:
        """Normalize beliefs to sum to 1.0."""
        total = sum(self.priorities.values())
        if total > 0:
            for key in self.priorities:
                self.priorities[key] /= total

    def prune(self, threshold: float = MIN_BELIEF_THRESHOLD) -> int:
        """Remove beliefs below threshold. Returns number pruned."""
        before = len(self.priorities)
        self.priorities = {k: v for k, v in self.priorities.items() if v >= threshold}
        pruned = before - len(self.priorities)
        if pruned > 0 and self.priorities:
            self.normalize()
        return pruned

    def top_k(self, k: int = 5) -> List[tuple]:
        """Return top k priorities by belief strength."""
        sorted_items = sorted(self.priorities.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def get_belief(self, priority: str) -> float:
        """Get belief for a specific priority."""
        return self.priorities.get(priority, 0.0)

    def set_belief(self, priority: str, value: float) -> None:
        """Set belief for a specific priority."""
        self.priorities[priority] = max(0.0, min(1.0, value))

    def entropy(self) -> float:
        """Calculate Shannon entropy of the belief distribution."""
        h = 0.0
        for p in self.priorities.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priorities": dict(self.priorities),
            "last_updated": self.last_updated,
            "update_count": self.update_count,
            "entropy": round(self.entropy(), 4),
        }


@dataclass
class CorrectionEvent:
    """A user correction event."""

    correction_type: CorrectionType
    target_priority: str
    context: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )


@dataclass
class DenoisingResult:
    """Result of applying a denoising update."""

    success: bool
    prior_belief: float  # Belief before update
    posterior_belief: float  # Belief after update
    correction_type: CorrectionType
    target_priority: str
    belief_delta: float = 0.0
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        self.belief_delta = self.posterior_belief - self.prior_belief


class InteractiveDenoiser:
    """
    Bayesian belief updater for user intent denoising.

    Maintains a probability distribution over user priorities and
    updates it using Bayes' theorem when user corrections arrive.
    """

    def __init__(
        self,
        correction_likelihood: float = CORRECTION_LIKELIHOOD,
        false_alarm_rate: float = FALSE_ALARM_RATE,
        decay_rate: float = BELIEF_DECAY_RATE,
    ):
        self._correction_likelihood = correction_likelihood
        self._false_alarm_rate = false_alarm_rate
        self._decay_rate = decay_rate
        self._belief = BeliefState()
        self._history: List[CorrectionEvent] = []
        self._last_decay_time = time.monotonic()

    @property
    def belief_state(self) -> BeliefState:
        return self._belief

    @property
    def correction_history(self) -> List[CorrectionEvent]:
        return list(self._history)

    def initialize_beliefs(self, priorities: Dict[str, float]) -> None:
        """
        Initialize the belief distribution.

        Args:
            priorities: Mapping of priority_name -> initial_probability
        """
        self._belief = BeliefState(priorities=dict(priorities))
        self._belief.normalize()
        logger.info("Beliefs initialized with %d priorities", len(priorities))

    def add_priority(self, name: str, initial_belief: float = 0.1) -> None:
        """Add a new priority to the belief state."""
        if len(self._belief.priorities) >= MAX_PRIORITIES:
            self._belief.prune()

        self._belief.priorities[name] = initial_belief
        self._belief.normalize()

    def apply_correction(
        self,
        correction_type: CorrectionType,
        target_priority: str,
        redirect_to: Optional[str] = None,
        context: str = "",
    ) -> DenoisingResult:
        """
        Apply a user correction using Bayesian update.

        Args:
            correction_type: Type of correction
            target_priority: The priority being corrected
            redirect_to: For REDIRECT corrections, the new target
            context: Additional context string

        Returns:
            DenoisingResult with before/after beliefs
        """
        # Record correction event
        event = CorrectionEvent(
            correction_type=correction_type,
            target_priority=target_priority,
            context=context,
        )
        self._history.append(event)

        # Apply time decay first
        self._apply_decay()

        # Get current belief
        entropy_before = self._belief.entropy()
        prior = self._belief.get_belief(target_priority)

        # If priority doesn't exist yet, add it
        if target_priority not in self._belief.priorities:
            self._belief.priorities[target_priority] = 0.1
            self._belief.normalize()
            prior = self._belief.get_belief(target_priority)

        # Apply Bayesian update based on correction type
        if correction_type in (CorrectionType.DISMISS, CorrectionType.CANCEL):
            posterior = self._bayes_decrease(prior)
        elif correction_type in (CorrectionType.PROMOTE, CorrectionType.CLARIFY):
            posterior = self._bayes_increase(prior)
        elif correction_type == CorrectionType.RESCHEDULE:
            # Reschedule: moderate decrease (still relevant, just not now)
            posterior = self._bayes_moderate_decrease(prior)
        elif correction_type == CorrectionType.REDIRECT:
            posterior = self._bayes_decrease(prior)
            if redirect_to:
                if redirect_to not in self._belief.priorities:
                    self._belief.priorities[redirect_to] = 0.1
                redirect_posterior = self._bayes_increase(
                    self._belief.get_belief(redirect_to)
                )
                self._belief.set_belief(redirect_to, redirect_posterior)
        else:
            posterior = prior

        # Update belief state
        self._belief.set_belief(target_priority, posterior)
        self._belief.normalize()
        self._belief.update_count += 1
        self._belief.last_updated = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        # Prune negligible beliefs
        self._belief.prune()

        entropy_after = self._belief.entropy()

        return DenoisingResult(
            success=True,
            prior_belief=prior,
            posterior_belief=self._belief.get_belief(target_priority),
            correction_type=correction_type,
            target_priority=target_priority,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
        )

    def _bayes_increase(self, prior: float) -> float:
        """Bayesian update that increases belief (user confirmed/promoted)."""
        # P(intent_true | observation) using Bayes' theorem
        likelihood_true = self._correction_likelihood
        likelihood_false = self._false_alarm_rate

        posterior = (likelihood_true * prior) / (
            likelihood_true * prior + likelihood_false * (1 - prior)
        )
        return posterior

    def _bayes_decrease(self, prior: float) -> float:
        """Bayesian update that decreases belief (user dismissed/cancelled)."""
        # Invert the likelihoods: dismissal is evidence AGAINST the priority
        likelihood_true = self._false_alarm_rate  # Low chance of dismissing true intent
        likelihood_false = (
            self._correction_likelihood
        )  # High chance of dismissing false intent

        if prior <= 0:
            return 0.0

        marginal = likelihood_true * prior + likelihood_false * (1 - prior)
        if marginal <= 0:
            return 0.0

        posterior = (likelihood_true * prior) / marginal
        return posterior

    def _bayes_moderate_decrease(self, prior: float) -> float:
        """Moderate decrease for rescheduling (still relevant, just delayed)."""
        # Use a softer likelihood ratio
        likelihood_true = 0.3
        likelihood_false = 0.7

        if prior <= 0:
            return 0.0

        marginal = likelihood_true * prior + likelihood_false * (1 - prior)
        if marginal <= 0:
            return 0.0

        return (likelihood_true * prior) / marginal

    def _apply_decay(self) -> None:
        """Apply time-based decay to all beliefs."""
        now = time.monotonic()
        elapsed_hours = (now - self._last_decay_time) / 3600.0
        self._last_decay_time = now

        if elapsed_hours <= 0:
            return

        decay_factor = math.exp(-self._decay_rate * elapsed_hours)

        # Decay all beliefs toward uniform
        n = len(self._belief.priorities)
        if n == 0:
            return
        uniform = 1.0 / n

        for key in self._belief.priorities:
            current = self._belief.priorities[key]
            # Exponential decay toward uniform
            self._belief.priorities[key] = uniform + (current - uniform) * decay_factor

    def get_morning_brief_priorities(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top priorities for a morning brief.

        Returns list of priorities sorted by belief strength,
        suitable for morning brief generation.
        """
        top = self._belief.top_k(top_k)
        return [
            {"priority": name, "belief": round(belief, 4), "rank": i + 1}
            for i, (name, belief) in enumerate(top)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_state": self._belief.to_dict(),
            "history_count": len(self._history),
            "correction_likelihood": self._correction_likelihood,
            "false_alarm_rate": self._false_alarm_rate,
        }


__all__ = [
    "InteractiveDenoiser",
    "BeliefState",
    "CorrectionType",
    "CorrectionEvent",
    "DenoisingResult",
    "CORRECTION_LIKELIHOOD",
    "FALSE_ALARM_RATE",
]
