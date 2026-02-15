"""
BIZRA Quest Engine — Impact Mission Management
=================================================

Manages quest registration, acceptance, completion, and rewards.
Pre-seeds default quests for each guild domain. Quest completion
is gated by Ihsan threshold — nodes must meet constitutional
excellence standards to claim rewards.

Standing on Giants:
- McGonigal (2011): Games for social good
- Szabo (1997): Automated contract execution
- Shannon (1948): SNR as quest quality gate
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

from .types import (
    Quest,
    QuestAcceptResult,
    QuestDifficulty,
    QuestReward,
    QuestStatus,
)

logger = logging.getLogger(__name__)


# Pre-seeded quests aligned with guild domains
DEFAULT_QUESTS = [
    Quest(
        quest_id="001-sustainable-water",
        title="Sustainable Water Management",
        description="Design and implement a community water conservation system "
        "using IoT sensors and BIZRA data pipeline for monitoring.",
        guild_id="agriculture",
        difficulty=QuestDifficulty.BLOOM,
        reward=QuestReward(
            seed_amount=25.0,
            impt_amount=50.0,
            description="50 IMPT + 25 SEED for water conservation",
        ),
    ),
    Quest(
        quest_id="002-open-curriculum",
        title="Open Curriculum Builder",
        description="Create a modular, community-owned learning curriculum "
        "using Graph-of-Thoughts knowledge representation.",
        guild_id="education",
        difficulty=QuestDifficulty.SPROUT,
        reward=QuestReward(
            seed_amount=15.0,
            impt_amount=30.0,
            description="30 IMPT + 15 SEED for open education",
        ),
    ),
    Quest(
        quest_id="003-health-data-sovereignty",
        title="Health Data Sovereignty Framework",
        description="Build a privacy-preserving health data collection system "
        "where patients own and control their records.",
        guild_id="healthcare",
        difficulty=QuestDifficulty.FOREST,
        reward=QuestReward(
            seed_amount=50.0,
            bloom_amount=5.0,
            impt_amount=100.0,
            description="100 IMPT + 50 SEED + 5 BLOOM for health sovereignty",
        ),
    ),
    Quest(
        quest_id="004-solar-microgrid",
        title="Community Solar Microgrid",
        description="Design a peer-to-peer energy trading system for "
        "community solar installations with fair pricing.",
        guild_id="energy",
        difficulty=QuestDifficulty.BLOOM,
        reward=QuestReward(
            seed_amount=30.0,
            impt_amount=60.0,
            description="60 IMPT + 30 SEED for renewable energy",
        ),
    ),
    Quest(
        quest_id="005-cooperative-lending",
        title="Cooperative Lending Circle",
        description="Implement a rotating savings and credit association "
        "(ROSCA) system with constitutional zakat compliance.",
        guild_id="finance",
        difficulty=QuestDifficulty.SPROUT,
        reward=QuestReward(
            seed_amount=20.0,
            impt_amount=40.0,
            description="40 IMPT + 20 SEED for economic justice",
        ),
    ),
]


class QuestEngine:
    """
    Quest management engine.

    Handles the full quest lifecycle: registration, acceptance,
    progress tracking, and reward distribution. Quest completion
    is constitutionally gated by Ihsan threshold.

    Usage:
        engine = QuestEngine()
        result = engine.accept_quest("001-sustainable-water", "BIZRA-00000000")
        available = engine.list_available("agriculture")
    """

    def __init__(self) -> None:
        self._quests: Dict[str, Quest] = {}
        self._seed_default_quests()

    def _seed_default_quests(self) -> None:
        """Pre-seed default impact quests."""
        for quest in DEFAULT_QUESTS:
            if quest.quest_id not in self._quests:
                self._quests[quest.quest_id] = Quest(
                    quest_id=quest.quest_id,
                    title=quest.title,
                    description=quest.description,
                    guild_id=quest.guild_id,
                    difficulty=quest.difficulty,
                    reward=quest.reward,
                    status=QuestStatus.AVAILABLE,
                )

    def register_quest(self, quest: Quest) -> Quest:
        """Register a new quest."""
        self._quests[quest.quest_id] = quest
        logger.info("Quest registered: %s (%s)", quest.quest_id, quest.title)
        return quest

    def accept_quest(
        self,
        quest_id: str,
        node_id: str,
    ) -> QuestAcceptResult:
        """
        Accept a quest.

        Args:
            quest_id: ID of the quest to accept
            node_id: Node ID of the accepting node

        Returns:
            QuestAcceptResult with success/failure and details
        """
        quest = self._quests.get(quest_id)
        if quest is None:
            return QuestAcceptResult(
                success=False,
                message=f"Quest '{quest_id}' not found",
            )

        if quest.status != QuestStatus.AVAILABLE:
            return QuestAcceptResult(
                success=False,
                quest=quest,
                message=f"Quest '{quest_id}' is not available (status: {quest.status.value})",
            )

        # Accept the quest
        quest.accepted_by = node_id
        quest.accepted_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        quest.status = QuestStatus.ACCEPTED

        logger.info("Node %s accepted quest %s", node_id, quest_id)

        return QuestAcceptResult(
            success=True,
            quest=quest,
            message=f"Quest '{quest.title}' accepted (reward: {quest.reward.description})",
        )

    def complete_quest(
        self,
        quest_id: str,
        node_id: str,
        ihsan_score: float = 0.0,
    ) -> Optional[QuestReward]:
        """
        Complete a quest and claim rewards.

        Gated by Ihsan threshold — node must meet constitutional
        excellence standards to claim rewards.

        Args:
            quest_id: Quest to complete
            node_id: Node claiming completion
            ihsan_score: Current Ihsan score of the node

        Returns:
            QuestReward if successful, None if failed
        """
        quest = self._quests.get(quest_id)
        if quest is None:
            logger.warning("Quest not found: %s", quest_id)
            return None

        if quest.accepted_by != node_id:
            logger.warning(
                "Node %s cannot complete quest %s (accepted by %s)",
                node_id, quest_id, quest.accepted_by,
            )
            return None

        # Constitutional gate: Ihsan threshold
        if ihsan_score < UNIFIED_IHSAN_THRESHOLD:
            logger.warning(
                "Ihsan gate failed for quest %s: %.4f < %.4f",
                quest_id, ihsan_score, UNIFIED_IHSAN_THRESHOLD,
            )
            return None

        # Complete the quest
        quest.status = QuestStatus.COMPLETED
        quest.completed_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

        logger.info(
            "Quest %s completed by %s (reward: %s)",
            quest_id, node_id, quest.reward.description,
        )

        return quest.reward

    def get_quest(self, quest_id: str) -> Optional[Quest]:
        """Get a quest by ID."""
        return self._quests.get(quest_id)

    def list_available(self, guild_id: Optional[str] = None) -> List[Quest]:
        """List available quests, optionally filtered by guild."""
        quests = [
            q for q in self._quests.values()
            if q.status == QuestStatus.AVAILABLE
        ]
        if guild_id:
            quests = [q for q in quests if q.guild_id == guild_id]
        return quests

    def get_accepted(self, node_id: str) -> List[Quest]:
        """Get all quests accepted by a specific node."""
        return [
            q for q in self._quests.values()
            if q.accepted_by == node_id
            and q.status in (QuestStatus.ACCEPTED, QuestStatus.IN_PROGRESS)
        ]

    def list_all(self) -> List[Quest]:
        """List all quests regardless of status."""
        return list(self._quests.values())
