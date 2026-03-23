"""
BIZRA Quest System — Smoke Tests
===================================

8 tests covering quest registration, acceptance, and rewards.

Test naming: test_XX_descriptive_name
Coverage: QuestEngine, Quest, QuestReward, QuestAcceptResult
"""

from core.quest import (
    Quest,
    QuestDifficulty,
    QuestEngine,
    QuestReward,
    QuestStatus,
)


class TestQuestSmoke:
    """Quest system smoke tests."""

    # ── test_01: quest registration ──────────────────────────────────────
    def test_01_register_quest(self) -> None:
        """A new quest can be registered."""
        engine = QuestEngine()
        quest = Quest(
            quest_id="test-quest-001",
            title="Test Quest",
            description="A test quest for validation",
            guild_id="education",
            difficulty=QuestDifficulty.SEED,
            reward=QuestReward(impt_amount=10.0, description="10 IMPT"),
        )
        result = engine.register_quest(quest)
        assert result.quest_id == "test-quest-001"
        assert result.title == "Test Quest"

    # ── test_02: accept quest ────────────────────────────────────────────
    def test_02_accept_quest(self) -> None:
        """A node can accept an available quest."""
        engine = QuestEngine()
        result = engine.accept_quest(
            quest_id="001-sustainable-water",
            node_id="BIZRA-00000001",
        )
        assert result.success is True
        assert result.quest is not None
        assert result.quest.status == QuestStatus.ACCEPTED
        assert result.quest.accepted_by == "BIZRA-00000001"
        assert (
            "reward" in result.message.lower() or "accepted" in result.message.lower()
        )

    # ── test_03: list available quests by guild ──────────────────────────
    def test_03_list_available_by_guild(self) -> None:
        """Available quests can be filtered by guild."""
        engine = QuestEngine()
        agriculture_quests = engine.list_available("agriculture")
        assert len(agriculture_quests) >= 1
        assert all(q.guild_id == "agriculture" for q in agriculture_quests)

    # ── test_04: pre-seeded sustainable-water quest ──────────────────────
    def test_04_sustainable_water_quest_exists(self) -> None:
        """The 001-sustainable-water quest is pre-seeded."""
        engine = QuestEngine()
        quest = engine.get_quest("001-sustainable-water")
        assert quest is not None
        assert quest.title == "Sustainable Water Management"
        assert quest.guild_id == "agriculture"
        assert quest.difficulty == QuestDifficulty.BLOOM
        assert quest.reward.impt_amount == 50.0
        assert quest.reward.seed_amount == 25.0

    # ── test_05: quest reward structure ──────────────────────────────────
    def test_05_quest_reward_structure(self) -> None:
        """Quest rewards have correct token/IMPT amounts."""
        engine = QuestEngine()
        # Check all pre-seeded quests have valid rewards
        for quest in engine.list_all():
            assert quest.reward is not None
            assert quest.reward.impt_amount >= 0
            assert quest.reward.seed_amount >= 0
            assert quest.reward.bloom_amount >= 0
            assert quest.reward.description != ""

    # ── test_06: quest status transitions ────────────────────────────────
    def test_06_quest_status_transitions(self) -> None:
        """Quest status transitions: AVAILABLE -> ACCEPTED -> COMPLETED."""
        engine = QuestEngine()
        quest = engine.get_quest("002-open-curriculum")
        assert quest is not None
        assert quest.status == QuestStatus.AVAILABLE

        # Accept
        result = engine.accept_quest("002-open-curriculum", "BIZRA-00000001")
        assert result.success is True
        assert quest.status == QuestStatus.ACCEPTED

        # Complete (with sufficient Ihsan)
        reward = engine.complete_quest(
            "002-open-curriculum",
            "BIZRA-00000001",
            ihsan_score=0.96,
        )
        assert reward is not None
        assert reward.impt_amount == 30.0
        assert quest.status == QuestStatus.COMPLETED

    # ── test_07: quest not found ─────────────────────────────────────────
    def test_07_quest_not_found(self) -> None:
        """Accepting a non-existent quest returns failure."""
        engine = QuestEngine()
        result = engine.accept_quest("nonexistent-quest", "BIZRA-00000001")
        assert result.success is False
        assert "not found" in result.message

    # ── test_08: accepted quests per node ─────────────────────────────────
    def test_08_accepted_quests_per_node(self) -> None:
        """Can query all quests accepted by a specific node."""
        engine = QuestEngine()
        engine.accept_quest("001-sustainable-water", "BIZRA-00000001")
        engine.accept_quest("004-solar-microgrid", "BIZRA-00000001")
        engine.accept_quest("005-cooperative-lending", "BIZRA-00000002")

        node1_quests = engine.get_accepted("BIZRA-00000001")
        assert len(node1_quests) == 2
        assert all(q.accepted_by == "BIZRA-00000001" for q in node1_quests)

        node2_quests = engine.get_accepted("BIZRA-00000002")
        assert len(node2_quests) == 1
