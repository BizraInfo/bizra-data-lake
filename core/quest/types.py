"""
Quest Data Types — Impact Mission Primitives
==============================================

Standing on Giants:
- McGonigal (2011): Quest structures for social good
- Szabo (1997): Smart contract reward logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class QuestStatus(str, Enum):
    """Status of a quest."""

    AVAILABLE = "available"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestDifficulty(str, Enum):
    """Quest difficulty tiers (growth metaphor)."""

    SEED = "seed"  # Beginner
    SPROUT = "sprout"  # Intermediate
    BLOOM = "bloom"  # Advanced
    FOREST = "forest"  # Expert


@dataclass
class QuestReward:
    """Reward for completing a quest."""

    seed_amount: float = 0.0
    bloom_amount: float = 0.0
    impt_amount: float = 0.0
    description: str = ""


@dataclass
class Quest:
    """An impact mission aligned with a guild domain."""

    quest_id: str
    title: str
    description: str = ""
    guild_id: str = ""
    difficulty: QuestDifficulty = QuestDifficulty.SEED
    reward: QuestReward = field(default_factory=QuestReward)
    status: QuestStatus = QuestStatus.AVAILABLE
    prerequisites: List[str] = field(default_factory=list)
    accepted_by: Optional[str] = None
    accepted_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class QuestAcceptResult:
    """Result of attempting to accept a quest."""

    success: bool = False
    quest: Optional[Quest] = None
    message: str = ""
