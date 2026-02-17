"""
BIZRA Quest Types — Impact Mission Data Structures
=====================================================

Quests are structured impact missions within guilds. Each quest
has clear objectives, difficulty tiers, and token/IMPT rewards.
Completing quests builds reputation (IMPT) and earns SEED tokens.

Quest difficulty maps to BIZRA's growth metaphor:
    SEED   → Entry-level, learning-focused
    SPROUT → Intermediate, collaboration required
    BLOOM  → Advanced, measurable impact
    FOREST → Expert, systemic change

Standing on Giants:
- McGonigal (2011): Gameful design for real-world impact
- Szabo (1997): Smart contract rewards (automated, trustless)
- Al-Ghazali (1058-1111): Ihsan gates on quest completion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class QuestStatus(Enum):
    """Quest lifecycle states."""

    AVAILABLE = "available"  # Open for acceptance
    ACCEPTED = "accepted"  # Node has accepted, not started
    IN_PROGRESS = "in_progress"  # Work underway
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Abandoned or expired


class QuestDifficulty(Enum):
    """Quest difficulty tiers — growth metaphor."""

    SEED = "seed"  # Entry-level
    SPROUT = "sprout"  # Intermediate
    BLOOM = "bloom"  # Advanced
    FOREST = "forest"  # Expert/systemic


@dataclass(frozen=True)
class QuestReward:
    """Reward structure for quest completion."""

    seed_amount: float = 0.0  # SEED tokens earned
    bloom_amount: float = 0.0  # BLOOM tokens earned
    impt_amount: float = 0.0  # IMPT reputation points
    description: str = ""  # Human-readable reward description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_amount": self.seed_amount,
            "bloom_amount": self.bloom_amount,
            "impt_amount": self.impt_amount,
            "description": self.description,
        }


@dataclass
class Quest:
    """A structured impact mission within a guild."""

    quest_id: str
    title: str
    description: str
    guild_id: str
    difficulty: QuestDifficulty = QuestDifficulty.SEED
    reward: QuestReward = field(default_factory=QuestReward)
    status: QuestStatus = QuestStatus.AVAILABLE
    prerequisites: List[str] = field(default_factory=list)
    accepted_by: Optional[str] = None  # node_id of acceptor
    accepted_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quest_id": self.quest_id,
            "title": self.title,
            "description": self.description,
            "guild_id": self.guild_id,
            "difficulty": self.difficulty.value,
            "reward": self.reward.to_dict(),
            "status": self.status.value,
            "prerequisites": self.prerequisites,
            "accepted_by": self.accepted_by,
            "accepted_at": self.accepted_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
        }


@dataclass
class QuestAcceptResult:
    """Result of accepting a quest."""

    success: bool
    quest: Optional[Quest] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "quest": self.quest.to_dict() if self.quest else None,
            "message": self.message,
        }
