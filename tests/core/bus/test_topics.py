"""
Topic Registry Tests — Phase 68.06
═══════════════════════════════════

TDD anchors for TopicTier, TopicRegistry, TOPIC_REGISTRY.

Standing on Giants:
- Beck (2002): TDD by Example
"""

from __future__ import annotations

import json

import pytest

from core.bus.topics import (
    TOPIC_REGISTRY,
    Priority,
    TopicDef,
    TopicRegistry,
    TopicTier,
)


class TestTopicValidation:
    """Topic name resolution and validation."""

    def test_known_topic_validates(self) -> None:
        r = TopicRegistry()
        assert r.validate("action.intent") is True

    def test_unknown_topic_rejected(self) -> None:
        r = TopicRegistry()
        assert r.validate("not.a.real.topic") is False

    def test_wildcard_parent_validates(self) -> None:
        r = TopicRegistry()
        assert r.validate("economy.*") is True

    def test_constitutional_always_active(self) -> None:
        r = TopicRegistry()
        assert r.validate("action.intent") is True
        assert r.validate("ihsan.breach") is True
        assert r.validate("poi.credit") is True

    def test_federation_inactive_by_default(self) -> None:
        r = TopicRegistry()
        assert r.validate("federation.peer_seen") is False


class TestTierActivation:
    """Tier activation and deactivation."""

    def test_activate_economic_tier(self) -> None:
        r = TopicRegistry()
        assert r.validate("economy.zakat") is False
        r.activate_tier(TopicTier.ECONOMIC)
        assert r.validate("economy.zakat") is True

    def test_cannot_deactivate_constitutional(self) -> None:
        r = TopicRegistry()
        with pytest.raises(ValueError, match="immutable tier"):
            r.deactivate_tier(TopicTier.CONSTITUTIONAL)

    def test_cannot_deactivate_lifecycle(self) -> None:
        r = TopicRegistry()
        with pytest.raises(ValueError, match="immutable tier"):
            r.deactivate_tier(TopicTier.LIFECYCLE)

    def test_cannot_deactivate_policy(self) -> None:
        r = TopicRegistry()
        with pytest.raises(ValueError, match="immutable tier"):
            r.deactivate_tier(TopicTier.POLICY)

    def test_deactivate_mission_tier(self) -> None:
        r = TopicRegistry()
        r.activate_tier(TopicTier.MISSION)
        assert r.validate("mission.created") is True
        r.deactivate_tier(TopicTier.MISSION)
        assert r.validate("mission.created") is False


class TestTopicProperties:
    """Topic metadata and properties."""

    def test_ihsan_breach_emergency_priority(self) -> None:
        r = TopicRegistry()
        assert r.get_min_priority("ihsan.breach") == Priority.EMERGENCY

    def test_invariant_violation_critical_priority(self) -> None:
        r = TopicRegistry()
        assert r.get_min_priority("policy.invariant.violation") == Priority.CRITICAL

    def test_normal_topic_normal_priority(self) -> None:
        r = TopicRegistry()
        assert r.get_min_priority("action.intent") == Priority.NORMAL

    def test_registry_has_44_topics(self) -> None:
        assert len(TOPIC_REGISTRY) == 44

    def test_all_8_tiers_represented(self) -> None:
        tiers = {defn.tier for defn in TOPIC_REGISTRY.values()}
        assert len(tiers) == 8
        for tier in TopicTier:
            assert tier in tiers


class TestExportJson:
    """Cross-runtime JSON export."""

    def test_export_json_parseable(self) -> None:
        r = TopicRegistry()
        exported = r.export_json()
        data = json.loads(exported)
        assert len(data) == 44

    def test_export_json_sorted(self) -> None:
        r = TopicRegistry()
        data = json.loads(r.export_json())
        keys = list(data.keys())
        assert keys == sorted(keys)

    def test_export_includes_tier_and_schema(self) -> None:
        r = TopicRegistry()
        data = json.loads(r.export_json())
        entry = data["ihsan.breach"]
        assert entry["tier"] == TopicTier.CONSTITUTIONAL.value
        assert entry["schema"] == "ihsan_breach_v1"
