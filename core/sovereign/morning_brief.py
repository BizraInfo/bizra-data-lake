"""
Morning Brief Generator — Proactive Daily Intelligence

Generates a prioritized morning brief using the Interactive Denoiser's
belief state. The brief surfaces the user's top priorities, recent
corrections, and suggested focus areas.

Standing on Giants:
- Shannon (1948): Information-theoretic relevance ranking
- Kahneman (2011): System 1/2 cognitive load management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BriefItem:
    """A single item in the morning brief."""

    title: str
    priority_score: float
    category: str = "general"
    context: str = ""
    action_suggested: str = ""


@dataclass
class MorningBrief:
    """A complete morning brief."""

    items: List[BriefItem] = field(default_factory=list)
    generated_at: str = ""
    belief_entropy: float = 0.0
    total_priorities: int = 0

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "title": item.title,
                    "priority_score": round(item.priority_score, 4),
                    "category": item.category,
                    "context": item.context,
                    "action_suggested": item.action_suggested,
                }
                for item in self.items
            ],
            "generated_at": self.generated_at,
            "belief_entropy": round(self.belief_entropy, 4),
            "total_priorities": self.total_priorities,
        }


class MorningBriefGenerator:
    """
    Generates morning briefs from the interactive denoiser's belief state.

    Uses the denoiser's probability distribution to rank and surface
    the most relevant priorities for the user.
    """

    def __init__(self, max_items: int = 7):
        self._max_items = max_items

    def generate(
        self,
        denoiser: Optional[Any] = None,
        priorities: Optional[List[Dict[str, Any]]] = None,
    ) -> MorningBrief:
        """
        Generate a morning brief.

        Args:
            denoiser: InteractiveDenoiser instance (preferred)
            priorities: Pre-computed priority list (fallback)

        Returns:
            MorningBrief with ranked items
        """
        if denoiser is not None:
            priority_list = denoiser.get_morning_brief_priorities(self._max_items)
            entropy = denoiser.belief_state.entropy()
            total = len(denoiser.belief_state.priorities)
        elif priorities is not None:
            priority_list = priorities[: self._max_items]
            entropy = 0.0
            total = len(priorities)
        else:
            return MorningBrief(belief_entropy=0.0, total_priorities=0)

        items = []
        for p in priority_list:
            items.append(
                BriefItem(
                    title=p.get("priority", "unknown"),
                    priority_score=p.get("belief", 0.0),
                    category=p.get("category", "general"),
                )
            )

        brief = MorningBrief(
            items=items,
            belief_entropy=entropy,
            total_priorities=total,
        )

        logger.info(
            "Morning brief generated: %d items, entropy=%.3f",
            len(items),
            entropy,
        )
        return brief


__all__ = [
    "MorningBriefGenerator",
    "MorningBrief",
    "BriefItem",
]
