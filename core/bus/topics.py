"""
Topic Registry — Canonical Event Namespace for Bus Architecture
═══════════════════════════════════════════════════════════════

Hierarchical dot-separated topic scheme shared by Python and Rust.
Tiered activation based on node degradation level.

Standing on Giants:
- Hewitt (1973): Actor model messaging
- Lamport (1978): Logical clocks and ordering
- van Steen & Tanenbaum (2017): Distributed Systems, 3rd ed.

Phase 68.06 — Sovereign Instantiation
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum


class TopicTier(IntEnum):
    """Activation tiers — lower tiers are more critical."""

    CONSTITUTIONAL = 0  # Always active
    COGNITIVE = 1  # Active at degradation >= 2
    LIFECYCLE = 2  # Always active
    ECONOMIC = 3  # Active during ticker
    FEDERATION = 4  # Active when peers > 0
    POLICY = 5  # Always active
    MISSION = 6  # Active during orchestration
    OMEGA = 7  # Active during Omega loops


class Priority(IntEnum):
    """Event priority levels."""

    NORMAL = 0
    HIGH = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass(frozen=True)
class TopicDef:
    """Definition of a canonical bus topic."""

    tier: TopicTier
    schema: str
    min_priority: Priority = Priority.NORMAL


# ═══════════════════════════════════════════════════════════════
# Canonical Topic Registry — 45 topics across 8 tiers
# ═══════════════════════════════════════════════════════════════

TOPIC_REGISTRY: dict[str, TopicDef] = {
    # Tier 0: Constitutional (always active, never deactivatable)
    "action.intent": TopicDef(tier=TopicTier.CONSTITUTIONAL, schema="action_intent_v1"),
    "action.receipt": TopicDef(
        tier=TopicTier.CONSTITUTIONAL, schema="action_receipt_v1"
    ),
    "action.receipt.failed": TopicDef(
        tier=TopicTier.CONSTITUTIONAL, schema="action_receipt_v1"
    ),
    "action.cancelled": TopicDef(
        tier=TopicTier.CONSTITUTIONAL, schema="action_cancelled_v1"
    ),
    "critical.acknowledged": TopicDef(
        tier=TopicTier.CONSTITUTIONAL,
        schema="critical_ack_v1",
    ),
    "ihsan.breach": TopicDef(
        tier=TopicTier.CONSTITUTIONAL,
        schema="ihsan_breach_v1",
        min_priority=Priority.EMERGENCY,
    ),
    "tick.completed": TopicDef(
        tier=TopicTier.LIFECYCLE,
        schema="tick_completed_v1",
    ),
    "poi.credit": TopicDef(tier=TopicTier.CONSTITUTIONAL, schema="poi_credit_v1"),
    # Tier 1: Cognitive
    "memory.promoted": TopicDef(tier=TopicTier.COGNITIVE, schema="memory_event_v1"),
    "memory.retrieved": TopicDef(tier=TopicTier.COGNITIVE, schema="memory_event_v1"),
    "reflex.compiled": TopicDef(tier=TopicTier.COGNITIVE, schema="reflex_event_v1"),
    "reflex.cache_hit": TopicDef(tier=TopicTier.COGNITIVE, schema="reflex_event_v1"),
    "reflex.pruned": TopicDef(tier=TopicTier.COGNITIVE, schema="reflex_event_v1"),
    # Tier 2: Lifecycle
    "node.lifecycle.boot": TopicDef(tier=TopicTier.LIFECYCLE, schema="lifecycle_v1"),
    "node.lifecycle.shutdown": TopicDef(
        tier=TopicTier.LIFECYCLE, schema="lifecycle_v1"
    ),
    "node.lifecycle.upgrade": TopicDef(tier=TopicTier.LIFECYCLE, schema="lifecycle_v1"),
    "session.end": TopicDef(tier=TopicTier.LIFECYCLE, schema="session_v1"),
    "system.lifecycle": TopicDef(tier=TopicTier.LIFECYCLE, schema="lifecycle_v1"),
    # Tier 3: Economic
    "economy.seed_minted": TopicDef(tier=TopicTier.ECONOMIC, schema="economy_v1"),
    "economy.bloom_accrued": TopicDef(tier=TopicTier.ECONOMIC, schema="economy_v1"),
    "economy.zakat": TopicDef(tier=TopicTier.ECONOMIC, schema="economy_v1"),
    "economy.demurrage": TopicDef(tier=TopicTier.ECONOMIC, schema="economy_v1"),
    "economy.asabiyyah": TopicDef(tier=TopicTier.ECONOMIC, schema="economy_v1"),
    # Tier 4: Federation
    "federation.peer_seen": TopicDef(tier=TopicTier.FEDERATION, schema="federation_v1"),
    "federation.attestation.sent": TopicDef(
        tier=TopicTier.FEDERATION, schema="attestation_v1"
    ),
    "federation.attestation.received": TopicDef(
        tier=TopicTier.FEDERATION, schema="attestation_v1"
    ),
    "federation.attestation.reciprocal": TopicDef(
        tier=TopicTier.FEDERATION, schema="attestation_v1"
    ),
    "federation.diffusion": TopicDef(tier=TopicTier.FEDERATION, schema="diffusion_v1"),
    # Tier 5: Policy
    "policy.fate.vetoed": TopicDef(tier=TopicTier.POLICY, schema="policy_v1"),
    "policy.telescript.denied": TopicDef(tier=TopicTier.POLICY, schema="policy_v1"),
    "auth.boundary.crossed": TopicDef(
        tier=TopicTier.POLICY,
        schema="auth_boundary_v1",
        min_priority=Priority.HIGH,
    ),
    "invariant.violation": TopicDef(
        tier=TopicTier.POLICY,
        schema="invariant_v1",
        min_priority=Priority.CRITICAL,
    ),
    "policy.invariant.violation": TopicDef(
        tier=TopicTier.POLICY,
        schema="invariant_v1",
        min_priority=Priority.CRITICAL,
    ),
    # Tier 6: Mission
    "mission.created": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.planned": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.executed": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.verified": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.failed": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.started": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.decomposed": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.completed": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "mission.system_ready": TopicDef(tier=TopicTier.MISSION, schema="mission_v1"),
    "receipt.generated": TopicDef(tier=TopicTier.MISSION, schema="receipt_v1"),
    "receipt.verified": TopicDef(tier=TopicTier.MISSION, schema="receipt_v1"),
    # Tier 7: Omega
    "omega.started": TopicDef(tier=TopicTier.OMEGA, schema="omega_v1"),
    "omega.iteration": TopicDef(tier=TopicTier.OMEGA, schema="omega_v1"),
    "omega.proved": TopicDef(tier=TopicTier.OMEGA, schema="omega_v1"),
    "omega.cancelled": TopicDef(tier=TopicTier.OMEGA, schema="omega_v1"),
    "omega.paused": TopicDef(tier=TopicTier.OMEGA, schema="omega_v1"),
    "omega.completed": TopicDef(tier=TopicTier.OMEGA, schema="omega_v1"),
}

# Tiers that CANNOT be deactivated (constitutional invariant)
_IMMUTABLE_TIERS: frozenset[TopicTier] = frozenset(
    {TopicTier.CONSTITUTIONAL, TopicTier.LIFECYCLE, TopicTier.POLICY}
)


class TopicRegistry:
    """Validates topic names and manages tier activation.

    Fail-closed: unknown topics are rejected. Constitutional, Lifecycle,
    and Policy tiers are always active and cannot be deactivated.
    """

    __slots__ = ("_topics", "_active_tiers")

    def __init__(self) -> None:
        self._topics: dict[str, TopicDef] = dict(TOPIC_REGISTRY)
        self._active_tiers: set[TopicTier] = set(_IMMUTABLE_TIERS)

    def activate_tier(self, tier: TopicTier) -> None:
        """Activate a topic tier for routing."""
        self._active_tiers.add(tier)

    def deactivate_tier(self, tier: TopicTier) -> None:
        """Deactivate a topic tier. Constitutional tiers cannot be deactivated."""
        if tier in _IMMUTABLE_TIERS:
            raise ValueError(f"Cannot deactivate immutable tier: {tier.name}")
        self._active_tiers.discard(tier)

    def validate(self, topic: str) -> bool:
        """Check if a topic is known and its tier is currently active."""
        defn = self._topics.get(topic)
        if defn is not None:
            return defn.tier in self._active_tiers
        # Wildcard parent match (e.g., "economy.*")
        prefix = topic.rstrip("*").rstrip(".")
        if prefix:
            return any(t.startswith(prefix + ".") for t in self._topics)
        return False

    def get_min_priority(self, topic: str) -> Priority:
        """Return minimum priority for a topic."""
        defn = self._topics.get(topic)
        return defn.min_priority if defn else Priority.NORMAL

    def active_topics(self) -> list[str]:
        """Return all currently active topic names."""
        return [
            name
            for name, defn in self._topics.items()
            if defn.tier in self._active_tiers
        ]

    def export_json(self) -> str:
        """Export registry as JSON for Rust cross-validation."""
        return json.dumps(
            {
                topic: {"tier": defn.tier.value, "schema": defn.schema}
                for topic, defn in sorted(self._topics.items())
            },
            sort_keys=True,
            indent=2,
        )
