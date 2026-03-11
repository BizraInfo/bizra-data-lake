"""
BIZRA Impact Tracker — Sovereignty Growth Engine

Bridges the Accumulator's PoI (Proof of Impact) events to sovereignty score
updates on the user's IdentityCard. This is the growth engine from
Article IX § 9.5 and § 9.7.

Flow:
    Contribution → ImpactScorer → Accumulator.record_impact() → bloom
    bloom → ImpactTracker.evaluate() → sovereignty_delta
    sovereignty_delta → IdentityCard.sovereignty_score update → re-sign

Sovereignty Tiers (Constitution Article IX § 9.5):
    SEED    (بذرة, 0.00–0.25): New node, learning phase
    SPROUT  (نبتة, 0.25–0.50): Active participation
    TREE    (شجرة, 0.50–0.75): Established contributor
    FOREST  (غابة, 0.75–1.00): Network elder

Standing on Giants: Al-Khwarizmi (algorithms) + Shannon (information) + Anthropic (alignment)
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.proof_engine.canonical import hex_digest

from .identity_card import IdentityCard, IdentityStatus, SovereigntyTier

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# UERS DIMENSIONS — Article IX § 9.7
# ═══════════════════════════════════════════════════════════════════════════════


class UERSDimension(str, Enum):
    """Five dimensions of impact measurement."""

    UTILITY = "utility"  # Practical value delivered
    EFFICIENCY = "efficiency"  # Resource optimization
    RESILIENCE = "resilience"  # System strengthening
    SUSTAINABILITY = "sustainability"  # Long-term viability
    ETHICS = "ethics"  # Ethical alignment


# Weights for each UERS dimension (must sum to 1.0)
UERS_WEIGHTS: Dict[str, float] = {
    UERSDimension.UTILITY: 0.25,
    UERSDimension.EFFICIENCY: 0.20,
    UERSDimension.RESILIENCE: 0.20,
    UERSDimension.SUSTAINABILITY: 0.15,
    UERSDimension.ETHICS: 0.20,
}

# Score normalization: max bloom that maps to sovereignty_score = 1.0
# Based on sustained contribution over ~1 year
BLOOM_SCORE_CEILING = 10_000.0

# Minimum bloom delta to trigger a sovereignty update
MIN_BLOOM_DELTA = 0.01

# Metadata safety: whitelist of allowed keys and max serialized size (bytes)
SAFE_METADATA_KEYS = frozenset(
    {
        "query_id",
        "processing_time_ms",
        "reasoning_depth",
        "snr_score",
        "ihsan_score",
        "source",
        "event_type",
    }
)
MAX_METADATA_SIZE = 1024  # 1 KB per event

# Achievement thresholds
TIER_THRESHOLDS = {
    SovereigntyTier.SEED: 0.0,
    SovereigntyTier.SPROUT: 0.25,
    SovereigntyTier.TREE: 0.50,
    SovereigntyTier.FOREST: 0.75,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ACHIEVEMENT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════


class Achievement(str, Enum):
    """Milestone achievements for sovereignty progression."""

    FIRST_QUERY = "first_query"
    FIRST_DAY = "first_day"
    WEEK_STREAK = "week_streak"
    MONTH_STREAK = "month_streak"
    SPROUT_REACHED = "sprout_reached"
    TREE_REACHED = "tree_reached"
    FOREST_REACHED = "forest_reached"
    FIRST_HARVEST = "first_harvest"
    COMMUNITY_HELPER = "community_helper"
    ETHICS_GUARDIAN = "ethics_guardian"


# Achievement bonuses (added to sovereignty_score)
ACHIEVEMENT_BONUSES: Dict[str, float] = {
    Achievement.FIRST_QUERY: 0.01,
    Achievement.FIRST_DAY: 0.02,
    Achievement.WEEK_STREAK: 0.03,
    Achievement.MONTH_STREAK: 0.05,
    Achievement.SPROUT_REACHED: 0.0,  # Tier transitions are natural, no bonus
    Achievement.TREE_REACHED: 0.0,
    Achievement.FOREST_REACHED: 0.0,
    Achievement.FIRST_HARVEST: 0.02,
    Achievement.COMMUNITY_HELPER: 0.03,
    Achievement.ETHICS_GUARDIAN: 0.04,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class UERSScore:
    """Scores across all five UERS dimensions."""

    utility: float = 0.0
    efficiency: float = 0.0
    resilience: float = 0.0
    sustainability: float = 0.0
    ethics: float = 0.0

    @property
    def weighted_total(self) -> float:
        """Compute weighted UERS score (0.0–1.0 range)."""
        raw = (
            self.utility * UERS_WEIGHTS[UERSDimension.UTILITY]
            + self.efficiency * UERS_WEIGHTS[UERSDimension.EFFICIENCY]
            + self.resilience * UERS_WEIGHTS[UERSDimension.RESILIENCE]
            + self.sustainability * UERS_WEIGHTS[UERSDimension.SUSTAINABILITY]
            + self.ethics * UERS_WEIGHTS[UERSDimension.ETHICS]
        )
        return min(1.0, max(0.0, raw))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ImpactEvent:
    """A recorded impact event with UERS scoring."""

    event_id: str
    contributor: str
    category: str
    action: str
    bloom_amount: float
    uers_scores: UERSScore
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["uers_scores"] = self.uers_scores.to_dict()
        return d


@dataclass
class ProgressSnapshot:
    """Current sovereignty progression state."""

    node_id: str
    sovereignty_score: float
    sovereignty_tier: str
    total_bloom: float
    total_events: int
    uers_aggregate: UERSScore
    achievements: List[str]
    streak_days: int
    last_activity: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["uers_aggregate"] = self.uers_aggregate.to_dict()
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# IMPACT TRACKER — The Growth Engine
# ═══════════════════════════════════════════════════════════════════════════════


def compute_query_bloom(
    processing_time_ms: float,
    reasoning_depth: int,
    validated: bool = False,
) -> float:
    """
    Compute bloom amount from query metrics — single source of truth.

    Formula:
        time_bloom = min(5.0, processing_time_ms / 1000 * 2.0)
        depth_bloom = reasoning_depth * 0.5
        validation_bloom = 1.0 if validated else 0.5
        total = time_bloom + depth_bloom + validation_bloom
    """
    time_bloom = min(5.0, processing_time_ms / 1000 * 2.0)
    depth_bloom = reasoning_depth * 0.5
    validation_bloom = 1.0 if validated else 0.5
    return time_bloom + depth_bloom + validation_bloom


class ImpactTracker:
    """
    Tracks impact events, computes UERS scores, and updates sovereignty.

    This is the bridge between:
        - BizraAccumulator (bloom/fruit/zakat cycle)
        - IdentityCard (sovereignty_score, sovereignty_tier)
        - Credential persistence (re-sign + save on tier change)

    Usage:
        tracker = ImpactTracker(node_id="BIZRA-A1B2C3D4")
        tracker.record_event("computation", "llm_query", bloom=1.5,
                             uers=UERSScore(utility=0.8, efficiency=0.7))
        progress = tracker.get_progress()
    """

    # Batch save: persist at most once per SAVE_INTERVAL seconds
    SAVE_INTERVAL: float = 5.0
    # Always persist on tier transitions (critical state change)

    def __init__(
        self,
        node_id: str,
        state_dir: Optional[Path] = None,
    ):
        self._node_id = node_id
        self._state_dir = state_dir or (Path.home() / ".bizra-node")
        self._events: List[ImpactEvent] = []
        self._achievements: List[str] = []
        self._total_bloom: float = 0.0
        self._uers_aggregate = UERSScore()
        self._streak_days: int = 0
        self._last_activity: str = ""
        self._sovereignty_score: float = 0.0
        self._dirty: bool = False
        self._last_save_time: float = 0.0
        # Incremental category counters — avoids O(N) scan in _check_achievements
        self._category_counts: Dict[str, int] = {}

        # Load persisted state if available
        self._load_state()

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def sovereignty_score(self) -> float:
        return self._sovereignty_score

    @property
    def sovereignty_tier(self) -> SovereigntyTier:
        if self._sovereignty_score < 0.25:
            return SovereigntyTier.SEED
        elif self._sovereignty_score < 0.50:
            return SovereigntyTier.SPROUT
        elif self._sovereignty_score < 0.75:
            return SovereigntyTier.TREE
        else:
            return SovereigntyTier.FOREST

    @property
    def achievements(self) -> List[str]:
        return list(self._achievements)

    @property
    def total_bloom(self) -> float:
        return self._total_bloom

    # ─── Core Recording ───────────────────────────────────────────────

    def record_event(
        self,
        category: str,
        action: str,
        bloom: float,
        uers: Optional[UERSScore] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImpactEvent:
        """
        Record an impact event and update sovereignty.

        Args:
            category: Impact category (computation, knowledge, code, etc.)
            action: Specific action (llm_query, doc_synthesis, etc.)
            bloom: Bloom amount from accumulator
            uers: UERS dimension scores (auto-estimated if None)
            metadata: Optional event metadata

        Returns:
            The recorded ImpactEvent
        """
        if uers is None:
            uers = self._estimate_uers(category, bloom)

        # Sanitize metadata: whitelist keys and enforce size cap
        if metadata:
            unsafe_keys = set(metadata.keys()) - SAFE_METADATA_KEYS
            if unsafe_keys:
                logger.debug(f"Removing non-whitelisted metadata keys: {unsafe_keys}")
                metadata = {
                    k: v for k, v in metadata.items() if k in SAFE_METADATA_KEYS
                }
            # Enforce size cap
            try:
                meta_json = json.dumps(metadata)
                if len(meta_json) > MAX_METADATA_SIZE:
                    logger.warning(
                        f"Metadata truncated: {len(meta_json)} > {MAX_METADATA_SIZE}"
                    )
                    metadata = {}
            except (TypeError, ValueError):
                metadata = {}

        event_id = hex_digest(
            f"{self._node_id}:{category}:{action}:{time.time()}".encode()
        )[:16]

        event = ImpactEvent(
            event_id=event_id,
            contributor=self._node_id,
            category=category,
            action=action,
            bloom_amount=bloom,
            uers_scores=uers,
            metadata=metadata or {},
        )

        self._events.append(event)
        self._total_bloom += bloom
        self._last_activity = event.timestamp
        self._category_counts[category] = self._category_counts.get(category, 0) + 1

        # Update aggregate UERS scores (running weighted average)
        n = len(self._events)
        if n == 1:
            self._uers_aggregate = UERSScore(
                utility=uers.utility,
                efficiency=uers.efficiency,
                resilience=uers.resilience,
                sustainability=uers.sustainability,
                ethics=uers.ethics,
            )
        else:
            prev = n - 1
            self._uers_aggregate.utility = (
                self._uers_aggregate.utility * prev + uers.utility
            ) / n
            self._uers_aggregate.efficiency = (
                self._uers_aggregate.efficiency * prev + uers.efficiency
            ) / n
            self._uers_aggregate.resilience = (
                self._uers_aggregate.resilience * prev + uers.resilience
            ) / n
            self._uers_aggregate.sustainability = (
                self._uers_aggregate.sustainability * prev + uers.sustainability
            ) / n
            self._uers_aggregate.ethics = (
                self._uers_aggregate.ethics * prev + uers.ethics
            ) / n

        # Compute new sovereignty score
        old_tier = self.sovereignty_tier
        self._sovereignty_score = self._compute_sovereignty()

        # Check achievements (may unlock new ones)
        prev_achievement_count = len(self._achievements)
        self._check_achievements()

        # Recompute if new achievements were unlocked (they affect score)
        if len(self._achievements) > prev_achievement_count:
            self._sovereignty_score = self._compute_sovereignty()

        # Detect tier transition
        new_tier = self.sovereignty_tier
        tier_changed = new_tier != old_tier
        if tier_changed:
            logger.info(
                f"Tier transition: {old_tier.value} → {new_tier.value} "
                f"(score: {self._sovereignty_score:.4f})"
            )

        # Batched persistence: save immediately on tier transitions,
        # otherwise defer to save interval to avoid blocking the hot path
        self._dirty = True
        now = time.time()
        if tier_changed or (now - self._last_save_time >= self.SAVE_INTERVAL):
            self._save_state()

        return event

    def _compute_sovereignty(self) -> float:
        """
        Compute sovereignty score from bloom + UERS.

        Formula:
            base = min(1.0, total_bloom / BLOOM_SCORE_CEILING)
            uers_multiplier = uers_aggregate.weighted_total
            achievement_bonus = sum(bonuses for unlocked achievements)
            sovereignty = min(1.0, base * 0.6 + uers_multiplier * 0.3 + achievement_bonus)

        The 60/30/10 split ensures:
            - 60% from actual contributions (bloom)
            - 30% from quality of contributions (UERS)
            - 10% cap from achievements
        """
        # Base from accumulated bloom
        base = min(1.0, self._total_bloom / BLOOM_SCORE_CEILING)

        # UERS quality multiplier
        uers_factor = self._uers_aggregate.weighted_total

        # Achievement bonus
        achievement_bonus = sum(
            ACHIEVEMENT_BONUSES.get(a, 0.0) for a in self._achievements
        )

        # Weighted combination
        score = base * 0.6 + uers_factor * 0.3 + min(0.1, achievement_bonus)
        return min(1.0, max(0.0, score))

    # Class-level UERS weight templates — avoids dict allocation per call
    _UERS_TEMPLATES: Dict[str, tuple] = {
        "computation": (0.7, 0.8, 0.3, 0.4, 0.5),
        "knowledge": (0.8, 0.5, 0.6, 0.7, 0.6),
        "code": (0.9, 0.7, 0.7, 0.6, 0.5),
        "ethics": (0.5, 0.4, 0.6, 0.8, 0.9),
        "community": (0.8, 0.5, 0.7, 0.8, 0.7),
        "orchestration": (0.7, 0.8, 0.5, 0.5, 0.5),
    }
    _UERS_DEFAULT = (0.5, 0.5, 0.5, 0.5, 0.5)

    def _estimate_uers(self, category: str, bloom: float) -> UERSScore:
        """Estimate UERS scores from category when not provided."""
        bf = min(1.0, bloom / 10.0)
        w = self._UERS_TEMPLATES.get(category, self._UERS_DEFAULT)
        return UERSScore(
            utility=w[0] * bf,
            efficiency=w[1] * bf,
            resilience=w[2] * bf,
            sustainability=w[3] * bf,
            ethics=w[4] * bf,
        )

    # ─── Achievements ─────────────────────────────────────────────────

    def _check_achievements(self) -> None:
        """Check and unlock new achievements."""
        if Achievement.FIRST_QUERY not in self._achievements and len(self._events) >= 1:
            self._achievements.append(Achievement.FIRST_QUERY)
            logger.info(f"Achievement unlocked: {Achievement.FIRST_QUERY}")

        if (
            Achievement.SPROUT_REACHED not in self._achievements
            and self._sovereignty_score >= 0.25
        ):
            self._achievements.append(Achievement.SPROUT_REACHED)
            logger.info(f"Achievement unlocked: {Achievement.SPROUT_REACHED}")

        if (
            Achievement.TREE_REACHED not in self._achievements
            and self._sovereignty_score >= 0.50
        ):
            self._achievements.append(Achievement.TREE_REACHED)
            logger.info(f"Achievement unlocked: {Achievement.TREE_REACHED}")

        if (
            Achievement.FOREST_REACHED not in self._achievements
            and self._sovereignty_score >= 0.75
        ):
            self._achievements.append(Achievement.FOREST_REACHED)
            logger.info(f"Achievement unlocked: {Achievement.FOREST_REACHED}")

        # Community helper: 10+ community events (O(1) via incremental counter)
        if (
            Achievement.COMMUNITY_HELPER not in self._achievements
            and self._category_counts.get("community", 0) >= 10
        ):
            self._achievements.append(Achievement.COMMUNITY_HELPER)
            logger.info(f"Achievement unlocked: {Achievement.COMMUNITY_HELPER}")

        # Ethics guardian: 5+ ethics events (O(1) via incremental counter)
        if (
            Achievement.ETHICS_GUARDIAN not in self._achievements
            and self._category_counts.get("ethics", 0) >= 5
        ):
            self._achievements.append(Achievement.ETHICS_GUARDIAN)
            logger.info(f"Achievement unlocked: {Achievement.ETHICS_GUARDIAN}")

    def unlock_achievement(self, achievement: str) -> bool:
        """Manually unlock an achievement (for external triggers like streaks)."""
        if achievement in self._achievements:
            return False
        self._achievements.append(achievement)
        # Recompute sovereignty with new achievement bonus
        self._sovereignty_score = self._compute_sovereignty()
        self._save_state()
        return True

    # ─── Progress Reporting ───────────────────────────────────────────

    def get_progress(self) -> ProgressSnapshot:
        """Get the current sovereignty progression snapshot."""
        return ProgressSnapshot(
            node_id=self._node_id,
            sovereignty_score=self._sovereignty_score,
            sovereignty_tier=self.sovereignty_tier.value,
            total_bloom=self._total_bloom,
            total_events=len(self._events),
            uers_aggregate=UERSScore(
                utility=self._uers_aggregate.utility,
                efficiency=self._uers_aggregate.efficiency,
                resilience=self._uers_aggregate.resilience,
                sustainability=self._uers_aggregate.sustainability,
                ethics=self._uers_aggregate.ethics,
            ),
            achievements=list(self._achievements),
            streak_days=self._streak_days,
            last_activity=self._last_activity,
        )

    def get_tier_progress(self) -> Dict[str, Any]:
        """Get progress toward the next sovereignty tier."""
        current = self.sovereignty_tier
        score = self._sovereignty_score

        next_tier = None
        next_threshold = 1.0
        if current == SovereigntyTier.SEED:
            next_tier = SovereigntyTier.SPROUT
            next_threshold = 0.25
        elif current == SovereigntyTier.SPROUT:
            next_tier = SovereigntyTier.TREE
            next_threshold = 0.50
        elif current == SovereigntyTier.TREE:
            next_tier = SovereigntyTier.FOREST
            next_threshold = 0.75

        current_threshold = TIER_THRESHOLDS[current]
        tier_range = next_threshold - current_threshold
        progress_in_tier = (
            (score - current_threshold) / tier_range if tier_range > 0 else 1.0
        )

        return {
            "current_tier": current.value,
            "current_score": score,
            "next_tier": next_tier.value if next_tier else None,
            "next_threshold": next_threshold,
            "progress_percent": round(min(100.0, progress_in_tier * 100), 1),
            "bloom_to_next": max(
                0, (next_threshold - score) * BLOOM_SCORE_CEILING / 0.6
            ),
        }

    # ─── Identity Card Integration ───────────────────────────────────

    def update_identity_card(
        self,
        card: IdentityCard,
        private_key: str,
    ) -> IdentityCard:
        """
        Update an IdentityCard's sovereignty_score and re-sign.

        IMPORTANT: sovereignty_score is part of the canonical digest.
        Changing it invalidates existing signatures, so we must re-sign.

        Args:
            card: The IdentityCard to update
            private_key: Owner's Ed25519 private key for re-signing

        Returns:
            Updated and re-signed IdentityCard
        """
        card.sovereignty_score = self._sovereignty_score
        card.status = IdentityStatus.ACTIVE
        card.self_signature = None  # Clear old signature before re-signing
        card.sign_as_owner(private_key)
        return card

    # ─── Persistence ──────────────────────────────────────────────────

    @property
    def _tracker_file(self) -> Path:
        return self._state_dir / "impact_tracker.json"

    def _load_state(self) -> None:
        """Load persisted tracker state."""
        if not self._tracker_file.exists():
            return
        try:
            data = json.loads(self._tracker_file.read_text())
            self._total_bloom = data.get("total_bloom", 0.0)
            self._sovereignty_score = data.get("sovereignty_score", 0.0)
            self._achievements = data.get("achievements", [])
            self._streak_days = data.get("streak_days", 0)
            self._last_activity = data.get("last_activity", "")

            agg = data.get("uers_aggregate", {})
            self._uers_aggregate = UERSScore(
                utility=agg.get("utility", 0.0),
                efficiency=agg.get("efficiency", 0.0),
                resilience=agg.get("resilience", 0.0),
                sustainability=agg.get("sustainability", 0.0),
                ethics=agg.get("ethics", 0.0),
            )

            # Restore category counters
            self._category_counts = data.get("category_counts", {})

            # Reconstruct events
            for ev in data.get("events", []):
                uers_data = ev.get("uers_scores", {})
                self._events.append(
                    ImpactEvent(
                        event_id=ev["event_id"],
                        contributor=ev["contributor"],
                        category=ev["category"],
                        action=ev["action"],
                        bloom_amount=ev["bloom_amount"],
                        uers_scores=UERSScore(**uers_data),
                        timestamp=ev.get("timestamp", ""),
                        metadata=ev.get("metadata", {}),
                    )
                )

            # Rebuild counters from events if not persisted (backwards compat)
            if not self._category_counts and self._events:
                for ev in self._events:
                    self._category_counts[ev.category] = (
                        self._category_counts.get(ev.category, 0) + 1
                    )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load impact tracker state: {e}")

    def flush(self) -> None:
        """Force-save if there are unsaved changes."""
        if self._dirty:
            self._save_state()

    def _save_state(self) -> None:
        """Persist tracker state to disk."""
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # Keep only the last 1000 events in the persisted file
        recent_events = (
            self._events[-1000:] if len(self._events) > 1000 else self._events
        )

        data = {
            "node_id": self._node_id,
            "sovereignty_score": self._sovereignty_score,
            "sovereignty_tier": self.sovereignty_tier.value,
            "total_bloom": self._total_bloom,
            "achievements": self._achievements,
            "streak_days": self._streak_days,
            "last_activity": self._last_activity,
            "uers_aggregate": self._uers_aggregate.to_dict(),
            "events": [e.to_dict() for e in recent_events],
            "total_events": len(self._events),
            "category_counts": self._category_counts,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        self._tracker_file.write_text(json.dumps(data, indent=2))
        self._dirty = False
        self._last_save_time = time.time()
