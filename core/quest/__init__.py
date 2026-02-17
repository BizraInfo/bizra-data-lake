"""
BIZRA Quest Module — Impact Mission Engine
=============================================

Quests are structured impact missions within guilds. Each quest
has clear objectives, difficulty tiers (SEED/SPROUT/BLOOM/FOREST),
and token/IMPT rewards gated by constitutional Ihsan thresholds.

v1.0.0 — Genesis Quest System

Standing on Giants:
- McGonigal (2011): Gameful design for real-world impact
- Szabo (1997): Smart contract theory for automated rewards
- Al-Ghazali (1058-1111): Ihsan as completion gate
"""

from core.quest.engine import DEFAULT_QUESTS, QuestEngine
from core.quest.types import (
    Quest,
    QuestAcceptResult,
    QuestDifficulty,
    QuestReward,
    QuestStatus,
)

__version__ = "1.0.0"

__all__ = [
    # Types
    "Quest",
    "QuestReward",
    "QuestAcceptResult",
    "QuestStatus",
    "QuestDifficulty",
    # Engine
    "QuestEngine",
    # Constants
    "DEFAULT_QUESTS",
]
