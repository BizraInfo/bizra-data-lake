"""
Takaful Security Gate — Sybil-Resistant Cold-Start Bootstrap
=============================================================
Golden Gem α3: Takaful pool admission requires Proof-of-Humanity gate
with minimum impact history threshold to prevent pool poisoning.

Standing on Giants:
  - XZ Backdoor — Social engineering the trust infrastructure
  - تكافل (Takaful) — Islamic mutual cooperation principle
  - k-anonymity (Sweeney, 2002) — Privacy through aggregation
  - Federated Learning (McMahan et al., 2017) — Learn without sharing data

Core Insight: The Alpha-100 nodes are not just early adopters.
They are the cold-start training corpus for every future user.
If Sybil nodes poison the Takaful pool, EVERY new user gets
corrupted bootstrap intelligence. This is the XZ attack pattern
applied to collective intelligence.

Defense: Probationary period. New nodes can RECEIVE Takaful
but cannot CONTRIBUTE to it until behavioral integrity is
verified through sustained positive impact.

Constitutional Principle: RIBA_ZERO (no exploitation of newcomers)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final, Optional

logger = logging.getLogger("sovereign.takaful_gate")


# ═══════════════════════════════════════════════════════════════════════════════
# TAKAFUL MEMBERSHIP TIERS
# ═══════════════════════════════════════════════════════════════════════════════


class TakafulTier(Enum):
    """Node membership tier in the Takaful pool."""
    OBSERVER = auto()      # Can RECEIVE, cannot CONTRIBUTE
    CONTRIBUTOR = auto()   # Can receive AND contribute
    ANCHOR = auto()        # Verified high-impact, weights cluster centroid


class HumanityStatus(Enum):
    """Proof-of-Humanity verification status."""
    UNVERIFIED = auto()
    PENDING = auto()
    VERIFIED = auto()
    REVOKED = auto()


@dataclass
class TakafulProfile:
    """A node's Takaful pool membership profile."""
    node_id: str
    tier: TakafulTier = TakafulTier.OBSERVER
    humanity_status: HumanityStatus = HumanityStatus.UNVERIFIED

    # Impact history
    total_interactions: int = 0
    verified_impact_score: float = 0.0
    ihsan_history: list[float] = field(default_factory=list)
    consecutive_ihsan_above_floor: int = 0

    # Cluster assignment
    cluster_id: Optional[str] = None
    behavioral_hash: Optional[str] = None

    # Timestamps
    joined_at_ns: int = field(default_factory=time.time_ns)
    promoted_at_ns: Optional[int] = None

    @property
    def ihsan_mean(self) -> float:
        if not self.ihsan_history:
            return 0.0
        return sum(self.ihsan_history) / len(self.ihsan_history)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMOTION THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum interactions before a node can contribute to Takaful pool
MIN_INTERACTIONS_FOR_CONTRIBUTOR: Final[int] = 50

# Minimum verified impact score for contributor status
MIN_IMPACT_FOR_CONTRIBUTOR: Final[float] = 0.5

# Minimum consecutive إحسان-above-floor periods
MIN_IHSAN_PERIODS_FOR_CONTRIBUTOR: Final[int] = 5

# Minimum contributors in cluster before centroid is shareable
# This enforces k-anonymity: individual profiles dissolve into aggregate
MIN_CLUSTER_SIZE_FOR_SHARING: Final[int] = 50  # k=50 anonymity

# Anchor threshold (top performers who anchor cluster centroids)
MIN_INTERACTIONS_FOR_ANCHOR: Final[int] = 500
MIN_IMPACT_FOR_ANCHOR: Final[float] = 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# TAKAFUL GATE
# ═══════════════════════════════════════════════════════════════════════════════


class TakafulSecurityGate:
    """
    Manages Takaful pool membership and prevents Sybil poisoning.

    New nodes enter as OBSERVERS: they receive bootstrap intelligence
    from the cluster centroid but cannot contribute their behavioral
    profile to the pool.

    Promotion to CONTRIBUTOR requires:
    1. Proof-of-Humanity verification
    2. Minimum interaction count (sustained engagement)
    3. Minimum verified impact score (genuine contribution)
    4. Consistent إحسان history above floor

    This creates a probationary period where:
    - Legitimate users experience no degradation (they receive Takaful)
    - Sybil nodes cannot poison the pool (they can't contribute)
    - The pool only grows with verified, high-integrity profiles
    """

    def __init__(self) -> None:
        self._profiles: dict[str, TakafulProfile] = {}
        self._clusters: dict[str, list[str]] = {}  # cluster_id → [node_ids]
        self._cluster_centroids: dict[str, dict[str, Any]] = {}

    def register_node(self, node_id: str) -> TakafulProfile:
        """Register a new node as OBSERVER."""
        if node_id in self._profiles:
            return self._profiles[node_id]

        profile = TakafulProfile(node_id=node_id)
        self._profiles[node_id] = profile
        logger.info("Takaful: Registered node '%s' as OBSERVER", node_id)
        return profile

    def record_interaction(
        self,
        node_id: str,
        ihsan_score: float,
        impact_delta: float = 0.0,
    ) -> TakafulProfile:
        """
        Record an interaction for a node and check for promotion.

        Args:
            node_id: The node that completed an interaction.
            ihsan_score: إحسان score for this interaction.
            impact_delta: Verified impact contribution.

        Returns:
            Updated TakafulProfile (may have been promoted).
        """
        profile = self._profiles.get(node_id)
        if not profile:
            profile = self.register_node(node_id)

        # Update stats
        profile.total_interactions += 1
        profile.verified_impact_score += max(0.0, impact_delta)
        profile.ihsan_history.append(ihsan_score)

        # Keep bounded history
        if len(profile.ihsan_history) > 100:
            profile.ihsan_history = profile.ihsan_history[-100:]

        # Track consecutive إحسان above floor
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

        if ihsan_score >= UNIFIED_IHSAN_THRESHOLD:
            profile.consecutive_ihsan_above_floor += 1
        else:
            profile.consecutive_ihsan_above_floor = 0

        # Check for promotion
        self._check_promotion(profile)

        return profile

    def _check_promotion(self, profile: TakafulProfile) -> None:
        """Check if node qualifies for tier promotion."""
        # OBSERVER → CONTRIBUTOR
        if profile.tier == TakafulTier.OBSERVER:
            if (
                profile.humanity_status == HumanityStatus.VERIFIED
                and profile.total_interactions >= MIN_INTERACTIONS_FOR_CONTRIBUTOR
                and profile.verified_impact_score >= MIN_IMPACT_FOR_CONTRIBUTOR
                and profile.consecutive_ihsan_above_floor
                >= MIN_IHSAN_PERIODS_FOR_CONTRIBUTOR
            ):
                profile.tier = TakafulTier.CONTRIBUTOR
                profile.promoted_at_ns = time.time_ns()
                logger.info(
                    "Takaful: Node '%s' PROMOTED to CONTRIBUTOR "
                    "(interactions=%d, impact=%.2f, ihsan_streak=%d)",
                    profile.node_id,
                    profile.total_interactions,
                    profile.verified_impact_score,
                    profile.consecutive_ihsan_above_floor,
                )

        # CONTRIBUTOR → ANCHOR
        elif profile.tier == TakafulTier.CONTRIBUTOR:
            if (
                profile.total_interactions >= MIN_INTERACTIONS_FOR_ANCHOR
                and profile.verified_impact_score >= MIN_IMPACT_FOR_ANCHOR
            ):
                profile.tier = TakafulTier.ANCHOR
                logger.info(
                    "Takaful: Node '%s' PROMOTED to ANCHOR",
                    profile.node_id,
                )

    def verify_humanity(self, node_id: str, verified: bool = True) -> None:
        """Mark a node as humanity-verified (or revoked)."""
        profile = self._profiles.get(node_id)
        if not profile:
            return
        profile.humanity_status = (
            HumanityStatus.VERIFIED if verified else HumanityStatus.REVOKED
        )
        if not verified and profile.tier != TakafulTier.OBSERVER:
            # Demotion on revocation
            profile.tier = TakafulTier.OBSERVER
            logger.warning(
                "Takaful: Node '%s' DEMOTED to OBSERVER — humanity revoked",
                node_id,
            )

    def can_contribute(self, node_id: str) -> bool:
        """Check if a node can contribute to the Takaful pool."""
        profile = self._profiles.get(node_id)
        if not profile:
            return False
        return profile.tier in (TakafulTier.CONTRIBUTOR, TakafulTier.ANCHOR)

    def can_receive(self, node_id: str) -> bool:
        """Check if a node can receive from the Takaful pool."""
        profile = self._profiles.get(node_id)
        if not profile:
            return False
        # ALL tiers can receive — this is mutual cooperation
        return True

    def get_cluster_centroid(
        self, cluster_id: str
    ) -> Optional[dict[str, Any]]:
        """
        Get the cluster centroid for bootstrap.

        The centroid is computed ONLY from CONTRIBUTOR and ANCHOR profiles,
        never from OBSERVERS. This prevents Sybil poisoning.

        Returns None if cluster has fewer than MIN_CLUSTER_SIZE_FOR_SHARING
        contributors (k-anonymity guarantee).
        """
        cluster_nodes = self._clusters.get(cluster_id, [])
        contributors = [
            nid for nid in cluster_nodes
            if self.can_contribute(nid)
        ]

        if len(contributors) < MIN_CLUSTER_SIZE_FOR_SHARING:
            logger.debug(
                "Takaful: Cluster '%s' has %d contributors "
                "(need %d for sharing)",
                cluster_id,
                len(contributors),
                MIN_CLUSTER_SIZE_FOR_SHARING,
            )
            return None

        return self._cluster_centroids.get(cluster_id)

    def assign_cluster(self, node_id: str, cluster_id: str) -> None:
        """Assign a node to a behavioral cluster."""
        profile = self._profiles.get(node_id)
        if not profile:
            return

        # Remove from old cluster
        if profile.cluster_id and profile.cluster_id in self._clusters:
            self._clusters[profile.cluster_id] = [
                nid for nid in self._clusters[profile.cluster_id]
                if nid != node_id
            ]

        # Add to new cluster
        profile.cluster_id = cluster_id
        if cluster_id not in self._clusters:
            self._clusters[cluster_id] = []
        self._clusters[cluster_id].append(node_id)

    def get_profile(self, node_id: str) -> Optional[TakafulProfile]:
        """Get a node's Takaful profile."""
        return self._profiles.get(node_id)

    def pool_stats(self) -> dict[str, Any]:
        """Get Takaful pool statistics."""
        profiles = list(self._profiles.values())
        return {
            "total_nodes": len(profiles),
            "observers": sum(1 for p in profiles if p.tier == TakafulTier.OBSERVER),
            "contributors": sum(
                1 for p in profiles if p.tier == TakafulTier.CONTRIBUTOR
            ),
            "anchors": sum(1 for p in profiles if p.tier == TakafulTier.ANCHOR),
            "verified_humans": sum(
                1
                for p in profiles
                if p.humanity_status == HumanityStatus.VERIFIED
            ),
            "clusters": len(self._clusters),
            "shareable_clusters": sum(
                1
                for cid, nodes in self._clusters.items()
                if sum(1 for n in nodes if self.can_contribute(n))
                >= MIN_CLUSTER_SIZE_FOR_SHARING
            ),
        }


__all__ = [
    "HumanityStatus",
    "TakafulProfile",
    "TakafulSecurityGate",
    "TakafulTier",
    "MIN_CLUSTER_SIZE_FOR_SHARING",
    "MIN_IHSAN_PERIODS_FOR_CONTRIBUTOR",
    "MIN_IMPACT_FOR_CONTRIBUTOR",
    "MIN_INTERACTIONS_FOR_CONTRIBUTOR",
]
