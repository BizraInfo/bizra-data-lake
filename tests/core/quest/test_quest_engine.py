"""
Tests: Quest Engine
===================

Covers: QuestEngine, Quest, QuestReward, QuestDifficulty, QuestStatus

Standing on Giants:
- Beck (2002, TDD): Tests drive behavior specification
- Szabo (1997): Smart contract reward logic is deterministic
"""

from __future__ import annotations

import pytest

from core.quest import (
    Quest,
    QuestAcceptResult,
    QuestDifficulty,
    QuestEngine,
    QuestReward,
    QuestStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module Import Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuestImports:
    def test_import_engine(self):
        from core.quest import QuestEngine
        assert QuestEngine is not None

    def test_import_types(self):
        from core.quest import Quest, QuestAcceptResult, QuestDifficulty, QuestReward, QuestStatus
        assert Quest is not None

    def test_module_version(self):
        import core.quest as q
        assert q.__version__ == "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# QuestStatus and QuestDifficulty Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuestEnums:
    def test_status_values(self):
        assert QuestStatus.AVAILABLE == "available"
        assert QuestStatus.ACCEPTED == "accepted"
        assert QuestStatus.COMPLETED == "completed"
        assert QuestStatus.FAILED == "failed"

    def test_difficulty_values(self):
        assert QuestDifficulty.SEED == "seed"
        assert QuestDifficulty.SPROUT == "sprout"
        assert QuestDifficulty.BLOOM == "bloom"
        assert QuestDifficulty.FOREST == "forest"

    def test_enums_are_str(self):
        assert isinstance(QuestStatus.AVAILABLE, str)
        assert isinstance(QuestDifficulty.BLOOM, str)


# ─────────────────────────────────────────────────────────────────────────────
# QuestReward Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuestReward:
    def test_defaults_zero(self):
        r = QuestReward()
        assert r.seed_amount == 0.0
        assert r.bloom_amount == 0.0
        assert r.impt_amount == 0.0

    def test_custom_values(self):
        r = QuestReward(seed_amount=25.0, impt_amount=50.0, description="50 IMPT")
        assert r.seed_amount == 25.0
        assert r.impt_amount == 50.0
        assert r.description == "50 IMPT"


# ─────────────────────────────────────────────────────────────────────────────
# Quest Dataclass Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuest:
    def test_basic_creation(self):
        q = Quest(quest_id="test-001", title="Test Quest")
        assert q.quest_id == "test-001"
        assert q.title == "Test Quest"
        assert q.status == QuestStatus.AVAILABLE
        assert q.accepted_by is None

    def test_prerequisites_default_empty(self):
        q = Quest(quest_id="x", title="X")
        assert q.prerequisites == []

    def test_difficulty_default_seed(self):
        q = Quest(quest_id="x", title="X")
        assert q.difficulty == QuestDifficulty.SEED


# ─────────────────────────────────────────────────────────────────────────────
# QuestEngine Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuestEngine:
    def test_creation_seeds_defaults(self):
        e = QuestEngine()
        quests = e.list_all()
        assert len(quests) >= 5

    def test_default_quest_ids(self):
        e = QuestEngine()
        ids = {q.quest_id for q in e.list_all()}
        assert "001-sustainable-water" in ids
        assert "002-open-curriculum" in ids
        assert "003-health-data-sovereignty" in ids
        assert "004-solar-microgrid" in ids
        assert "005-cooperative-lending" in ids

    def test_all_default_quests_are_available(self):
        e = QuestEngine()
        for q in e.list_all():
            assert q.status == QuestStatus.AVAILABLE

    def test_accept_quest_success(self):
        e = QuestEngine()
        result = e.accept_quest("001-sustainable-water", "BIZRA-00000001")
        assert result.success
        assert result.quest is not None
        assert result.quest.accepted_by == "BIZRA-00000001"
        assert result.quest.status == QuestStatus.ACCEPTED

    def test_accept_nonexistent_quest_fails(self):
        e = QuestEngine()
        result = e.accept_quest("does-not-exist", "BIZRA-00000001")
        assert not result.success
        assert "not found" in result.message

    def test_accept_already_accepted_quest_fails(self):
        e = QuestEngine()
        e.accept_quest("002-open-curriculum", "BIZRA-FIRST")
        result = e.accept_quest("002-open-curriculum", "BIZRA-SECOND")
        assert not result.success
        assert "not available" in result.message

    def test_accepted_at_set_on_accept(self):
        e = QuestEngine()
        result = e.accept_quest("001-sustainable-water", "BIZRA-00000001")
        assert result.quest.accepted_at is not None
        assert result.quest.accepted_at.endswith("Z")

    def test_list_available_filters_accepted(self):
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-00000001")
        available = e.list_available()
        ids = {q.quest_id for q in available}
        assert "001-sustainable-water" not in ids

    def test_list_available_by_guild(self):
        e = QuestEngine()
        agri = e.list_available("agriculture")
        assert all(q.guild_id == "agriculture" for q in agri)

    def test_get_accepted_for_node(self):
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-MYNODE")
        e.accept_quest("002-open-curriculum", "BIZRA-MYNODE")
        accepted = e.get_accepted("BIZRA-MYNODE")
        assert len(accepted) == 2

    def test_get_accepted_other_node_empty(self):
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-OTHER")
        accepted = e.get_accepted("BIZRA-MYNODE")
        assert len(accepted) == 0

    def test_complete_quest_ihsan_gated(self):
        """Complete quest below Ihsan threshold — should return None."""
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-NODE")
        reward = e.complete_quest("001-sustainable-water", "BIZRA-NODE", ihsan_score=0.50)
        assert reward is None

    def test_complete_quest_above_threshold(self):
        """Complete quest above Ihsan threshold — should return reward."""
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-NODE2")
        reward = e.complete_quest("001-sustainable-water", "BIZRA-NODE2", ihsan_score=0.97)
        assert reward is not None
        assert reward.impt_amount > 0

    def test_complete_quest_marks_completed(self):
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-COMPLETE")
        e.complete_quest("001-sustainable-water", "BIZRA-COMPLETE", ihsan_score=0.97)
        quest = e.get_quest("001-sustainable-water")
        assert quest.status == QuestStatus.COMPLETED
        assert quest.completed_at is not None

    def test_complete_quest_wrong_node_fails(self):
        e = QuestEngine()
        e.accept_quest("001-sustainable-water", "BIZRA-RIGHT")
        reward = e.complete_quest("001-sustainable-water", "BIZRA-WRONG", ihsan_score=0.97)
        assert reward is None

    def test_register_custom_quest(self):
        e = QuestEngine()
        q = Quest(quest_id="custom-001", title="Custom", guild_id="energy")
        e.register_quest(q)
        assert e.get_quest("custom-001") is not None
