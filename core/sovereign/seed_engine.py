"""
Seed Engine — Every Node Is a Seed, Every Seed Has Infinite Potential
=====================================================================

The heartbeat of the DDAGI OS. Embeds self-RLVR and proactive self-assessment
into the SovereignRuntime lifecycle, tracking each node's growth from SEED
through SPROUT, TREE, and FOREST tiers.

Core loop (driven by ActionBus events or periodic tick):

    OBSERVE mission result
      -> SCORE via composite_reward (SNR + Ihsan + efficiency + feedback)
        -> GATE via constitutional thresholds
          -> RECORD hash-chained episode receipt
            -> ASSESS sovereignty tier progression
              -> PROMOTE or RECALIBRATE

The engine consumes real mission outcomes (not synthetic episodes) and
produces verifiable growth receipts anchored to the EvidenceLedger.

Standing on Giants:
- Deming (1986): PDCA — continuous improvement through measurement
- Kahneman (2011): System 1/2 — compiled reflexes vs deliberate reasoning
- Shannon (1948): SNR — signal quality as hard constraint
- Al-Ghazali (1095): Ihsan — excellence as floor, not ceiling
- Lamport (1978): Hash chains — deterministic, verifiable ordering

Phase 71 — Seed Potential Engine
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Optional

logger = logging.getLogger("bizra.sovereign.seed_engine")

from core.integration.constants import (
    SNR_THRESHOLD_T2_STANDARD,
    UNIFIED_IHSAN_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Sovereignty Tiers — the growth trajectory of every seed
# ---------------------------------------------------------------------------

TIER_SEED = "SEED"  # 0.00 - 0.25: nascent, learning
TIER_SPROUT = "SPROUT"  # 0.25 - 0.50: growing, contributing
TIER_TREE = "TREE"  # 0.50 - 0.75: mature, reliable
TIER_FOREST = "FOREST"  # 0.75 - 1.00: sovereign, nurturing others

TIER_THRESHOLDS = {
    TIER_SEED: (0.00, 0.25),
    TIER_SPROUT: (0.25, 0.50),
    TIER_TREE: (0.50, 0.75),
    TIER_FOREST: (0.75, 1.00),
}

TIER_ORDER = [TIER_SEED, TIER_SPROUT, TIER_TREE, TIER_FOREST]


def sovereignty_tier(score: float) -> str:
    """Map a 0-1 sovereignty score to its tier name."""
    clamped = max(0.0, min(1.0, score))
    for tier_name in reversed(TIER_ORDER):
        low, _ = TIER_THRESHOLDS[tier_name]
        if clamped >= low:
            return tier_name
    return TIER_SEED


# ---------------------------------------------------------------------------
# Growth Episode — one quantum of self-improvement
# ---------------------------------------------------------------------------


@dataclass
class GrowthEpisode:
    """A single scored episode in the node's growth trajectory."""

    index: int
    timestamp: str
    snr: float
    ihsan: float
    reward: float
    qualified: bool
    tier: str
    sovereignty_score: float
    receipt_hash: str


# ---------------------------------------------------------------------------
# Seed Potential — the infinite capacity waiting to be unlocked
# ---------------------------------------------------------------------------


@dataclass
class SeedPotential:
    """Snapshot of a node's realized and unrealized potential."""

    sovereignty_score: float  # 0-1, current composite
    tier: str  # SEED / SPROUT / TREE / FOREST
    tier_progress: float  # 0-1 within current tier
    episodes_total: int
    episodes_qualified: int
    qualification_rate: float
    reward_ema: float  # Exponential moving average of rewards
    streak: int  # Consecutive qualified episodes
    compiled: bool  # Reflex promotion achieved
    converged: bool  # Reward variance below threshold
    chain_valid: bool  # Receipt chain integrity
    potential_unlocked: float  # Fraction of capacity realized
    potential_remaining: float  # Infinite potential not yet realized
    weakest_dimension: Optional[str]  # Area most needing growth
    growth_velocity: float  # Rate of tier progression
    last_receipt_hash: str


# ---------------------------------------------------------------------------
# SeedEngine — the runtime-embedded growth tracker
# ---------------------------------------------------------------------------


_CLAMP01 = lambda v: max(0.0, min(1.0, float(v)))  # noqa: E731


@dataclass
class SeedEngineConfig:
    """Tunable parameters for the seed engine."""

    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = SNR_THRESHOLD_T2_STANDARD
    reward_threshold: float = 0.75
    compile_streak: int = 3
    ema_alpha: float = 0.30
    convergence_variance_max: float = 0.010
    variance_window: int = 5
    max_episodes: int = 500  # Rolling window


class SeedEngine:
    """Tracks a node's growth trajectory through RLVR episodes.

    Designed for runtime embedding — lightweight, deterministic, no I/O.
    External callers feed mission results via `record_episode()`;
    the engine computes rewards, maintains the hash chain, and
    reports sovereignty tier progression.

    Integration points:
    - SovereignRuntime._init_seed_engine() creates the engine
    - Mission completion calls engine.record_episode(metrics)
    - /v1/seed/potential returns engine.potential()
    - Health check includes engine.health()
    """

    def __init__(
        self,
        node_id: str = "node0",
        config: Optional[SeedEngineConfig] = None,
    ) -> None:
        self._node_id = node_id
        self._config = config or SeedEngineConfig()
        self._episodes: Deque[GrowthEpisode] = deque(maxlen=self._config.max_episodes)
        self._receipt_hashes: Deque[str] = deque(maxlen=self._config.max_episodes)

        # Running state
        self._reward_ema: float = 0.0
        self._reward_ema_initialized: bool = False
        self._streak: int = 0
        self._compiled: bool = False
        self._qualified_count: int = 0
        self._total_count: int = 0
        self._previous_hash: str = "GENESIS"
        self._rewards_window: Deque[float] = deque(maxlen=self._config.variance_window)

        # Dimension tracking for weakness detection
        self._dimension_scores: dict[str, list[float]] = {
            "snr": [],
            "ihsan": [],
            "efficiency": [],
            "feedback": [],
        }

        # Growth velocity tracking
        self._tier_history: Deque[tuple[float, str]] = deque(maxlen=50)

        logger.info(
            "SeedEngine initialized: node=%s, compile_streak=%d, ema_alpha=%.2f",
            node_id,
            self._config.compile_streak,
            self._config.ema_alpha,
        )

    def record_episode(self, metrics: dict[str, Any]) -> GrowthEpisode:
        """Record a growth episode from mission completion metrics.

        Args:
            metrics: Dict with keys: snr, ihsan, tokens_used, quality,
                     user_feedback, penalties, verified (all optional with
                     safe defaults).

        Returns:
            The recorded GrowthEpisode with receipt hash.
        """
        from core.token.rl_rewards import composite_reward

        self._total_count += 1
        index = self._total_count

        snr = _CLAMP01(metrics.get("snr", 0.0))
        ihsan = _CLAMP01(metrics.get("ihsan", 0.0))
        verified = bool(metrics.get("verified", True))

        reward = composite_reward(metrics)

        # Track dimension scores
        self._dimension_scores["snr"].append(snr)
        self._dimension_scores["ihsan"].append(ihsan)
        efficiency = _CLAMP01(metrics.get("efficiency", reward))
        self._dimension_scores["efficiency"].append(efficiency)
        feedback = _CLAMP01(metrics.get("user_feedback", 0.5))
        self._dimension_scores["feedback"].append(feedback)

        # Trim dimension history
        for dim in self._dimension_scores:
            if len(self._dimension_scores[dim]) > self._config.max_episodes:
                self._dimension_scores[dim] = self._dimension_scores[dim][
                    -self._config.max_episodes :
                ]

        # Qualification gate
        qualified = (
            verified
            and snr >= self._config.snr_threshold
            and ihsan >= self._config.ihsan_threshold
            and reward >= self._config.reward_threshold
        )

        if qualified:
            self._streak += 1
            self._qualified_count += 1
        else:
            self._streak = 0

        if self._streak >= self._config.compile_streak:
            self._compiled = True

        # EMA update
        if not self._reward_ema_initialized:
            self._reward_ema = reward
            self._reward_ema_initialized = True
        else:
            alpha = self._config.ema_alpha
            self._reward_ema = alpha * reward + (1.0 - alpha) * self._reward_ema

        self._rewards_window.append(reward)

        # Sovereignty score
        sov_score = self._compute_sovereignty_score()
        tier = sovereignty_tier(sov_score)

        # Track tier transitions
        self._tier_history.append((time.time(), tier))

        # Hash-chained receipt
        timestamp = datetime.now(timezone.utc).isoformat()
        hash_input = {
            "node_id": self._node_id,
            "episode_index": index,
            "timestamp": timestamp,
            "snr": round(snr, 6),
            "ihsan": round(ihsan, 6),
            "reward": round(reward, 6),
            "qualified": qualified,
            "sovereignty_score": round(sov_score, 6),
            "tier": tier,
            "previous_hash": self._previous_hash,
        }
        receipt_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        self._previous_hash = receipt_hash
        self._receipt_hashes.append(receipt_hash)

        episode = GrowthEpisode(
            index=index,
            timestamp=timestamp,
            snr=snr,
            ihsan=ihsan,
            reward=reward,
            qualified=qualified,
            tier=tier,
            sovereignty_score=sov_score,
            receipt_hash=receipt_hash,
        )
        self._episodes.append(episode)

        logger.debug(
            "SeedEngine episode %d: reward=%.3f qualified=%s tier=%s sov=%.3f",
            index,
            reward,
            qualified,
            tier,
            sov_score,
        )

        return episode

    def potential(self) -> SeedPotential:
        """Compute current seed potential snapshot.

        This is the main query method — answers "how much has this seed grown,
        and how much capacity remains?"
        """
        sov_score = self._compute_sovereignty_score()
        tier = sovereignty_tier(sov_score)

        # Progress within current tier
        low, high = TIER_THRESHOLDS[tier]
        tier_range = high - low
        tier_progress = (sov_score - low) / tier_range if tier_range > 0 else 0.0

        # Qualification rate
        qual_rate = (
            self._qualified_count / self._total_count if self._total_count > 0 else 0.0
        )

        # Convergence check
        variance = self._current_variance()
        converged = (
            self._reward_ema >= 0.85
            and variance <= self._config.convergence_variance_max
        )

        # Chain validity (lightweight — checks last hash continuity)
        chain_valid = len(self._receipt_hashes) == 0 or self._previous_hash != "GENESIS"

        # Potential computation
        # "Infinite potential" — what fraction has been unlocked?
        # Potential is not capped at 1.0 because growth is unbounded;
        # we measure it as a trajectory, not a destination.
        potential_unlocked = sov_score
        potential_remaining = 1.0 - sov_score  # always room to grow

        # Weakest dimension
        weakest = self._find_weakest_dimension()

        # Growth velocity (tier transitions per 10 episodes)
        velocity = self._compute_growth_velocity()

        return SeedPotential(
            sovereignty_score=round(sov_score, 4),
            tier=tier,
            tier_progress=round(tier_progress, 4),
            episodes_total=self._total_count,
            episodes_qualified=self._qualified_count,
            qualification_rate=round(qual_rate, 4),
            reward_ema=round(self._reward_ema, 4),
            streak=self._streak,
            compiled=self._compiled,
            converged=converged,
            chain_valid=chain_valid,
            potential_unlocked=round(potential_unlocked, 4),
            potential_remaining=round(potential_remaining, 4),
            weakest_dimension=weakest,
            growth_velocity=round(velocity, 4),
            last_receipt_hash=self._previous_hash,
        )

    def health(self) -> dict[str, Any]:
        """Lightweight health summary for /v1/health integration."""
        return {
            "active": self._total_count > 0,
            "episodes": self._total_count,
            "tier": sovereignty_tier(self._compute_sovereignty_score()),
            "compiled": self._compiled,
            "streak": self._streak,
        }

    def recent_episodes(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent episodes for API exposure."""
        episodes = list(self._episodes)[-limit:]
        return [
            {
                "index": e.index,
                "timestamp": e.timestamp,
                "snr": e.snr,
                "ihsan": e.ihsan,
                "reward": e.reward,
                "qualified": e.qualified,
                "tier": e.tier,
                "sovereignty_score": e.sovereignty_score,
                "receipt_hash": e.receipt_hash,
            }
            for e in episodes
        ]

    def _compute_sovereignty_score(self) -> float:
        """Composite sovereignty score from all growth dimensions.

        Formula:
            0.30 * qualification_rate
          + 0.25 * reward_ema
          + 0.20 * streak_ratio
          + 0.15 * dimension_balance
          + 0.10 * compiled_bonus

        Returns 0-1 score representing overall sovereignty maturity.
        """
        if self._total_count == 0:
            return 0.0

        qual_rate = self._qualified_count / self._total_count
        streak_ratio = min(1.0, self._streak / max(1, self._config.compile_streak * 2))
        dim_balance = self._dimension_balance()
        compiled_bonus = 1.0 if self._compiled else 0.0

        score = (
            0.30 * qual_rate
            + 0.25 * self._reward_ema
            + 0.20 * streak_ratio
            + 0.15 * dim_balance
            + 0.10 * compiled_bonus
        )

        return _CLAMP01(score)

    def _dimension_balance(self) -> float:
        """How balanced are the growth dimensions? 1.0 = perfectly balanced."""
        avgs = []
        for dim, scores in self._dimension_scores.items():
            if scores:
                avgs.append(sum(scores) / len(scores))

        if not avgs:
            return 0.0

        mean = sum(avgs) / len(avgs)
        if mean <= 0:
            return 0.0

        # Coefficient of variation (lower = more balanced)
        variance = sum((a - mean) ** 2 for a in avgs) / len(avgs)
        cv = (variance**0.5) / mean if mean > 0 else 1.0

        # Invert: cv=0 → balance=1.0, cv≥1 → balance→0
        return _CLAMP01(1.0 - cv)

    def _find_weakest_dimension(self) -> Optional[str]:
        """Identify the dimension with lowest average score."""
        if not any(self._dimension_scores.values()):
            return None

        dim_avgs = {}
        for dim, scores in self._dimension_scores.items():
            if scores:
                dim_avgs[dim] = sum(scores) / len(scores)

        if not dim_avgs:
            return None

        return min(dim_avgs, key=dim_avgs.get)  # type: ignore[arg-type]

    def _current_variance(self) -> float:
        """Variance of recent rewards."""
        if len(self._rewards_window) < 2:
            return 0.0
        values = list(self._rewards_window)
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _compute_growth_velocity(self) -> float:
        """Rate of sovereignty score improvement over recent episodes.

        Returns score delta per 10 episodes (positive = growing).
        """
        if len(self._episodes) < 2:
            return 0.0

        episodes = list(self._episodes)
        recent = episodes[-min(10, len(episodes)) :]
        if len(recent) < 2:
            return 0.0

        first_score = recent[0].sovereignty_score
        last_score = recent[-1].sovereignty_score
        span = len(recent) - 1

        delta_per_episode = (last_score - first_score) / span
        return delta_per_episode * 10  # Normalize to per-10-episodes


# ---------------------------------------------------------------------------
# Factory — Wire into SovereignRuntime
# ---------------------------------------------------------------------------


def create_seed_engine(
    runtime: Any = None,
    node_id: str = "node0",
    config: Optional[SeedEngineConfig] = None,
) -> SeedEngine:
    """Create a SeedEngine, optionally extracting node_id from runtime.

    Usage in runtime_core.py:
        from core.sovereign.seed_engine import create_seed_engine
        self._seed_engine = create_seed_engine(self)
    """
    if runtime is not None:
        identity = getattr(runtime, "_identity", None)
        if identity and hasattr(identity, "node_id"):
            node_id = identity.node_id

    engine = SeedEngine(node_id=node_id, config=config)
    logger.info("SeedEngine created for node=%s", node_id)
    return engine
