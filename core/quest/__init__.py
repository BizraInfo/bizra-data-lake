"""
BIZRA Quest System — Impact Mission Management
================================================
Quests are structured impact missions aligned with guild domains.
Nodes accept quests, complete them, and earn rewards — gated
by Ihsan constitutional thresholds.

Standing on Giants:
- McGonigal (2011): Games for social good
- Szabo (1997): Automated contract execution
- Shannon (1948): SNR as quest quality gate

v1.0.0
"""

from __future__ import annotations

from .engine import QuestEngine
from .types import (
    Quest,
    QuestAcceptResult,
    QuestDifficulty,
    QuestReward,
    QuestStatus,
)

__version__ = "1.0.0"

__all__ = [
    "Quest",
    "QuestAcceptResult",
    "QuestDifficulty",
    "QuestEngine",
    "QuestReward",
    "QuestStatus",
]
