"""
User Context — The System Knows Its Human
==========================================
This is the driveshaft between infrastructure and the person it serves.
Without this, the PAT agents are identity cards without behavior.
With this, every query carries the human's context, goals, and history.

The Living Memory system stores 5 types:
  EPISODIC   — what happened (conversation turns)
  SEMANTIC   — what is true (who the user is, their knowledge)
  PROCEDURAL — how to do things (learned preferences)
  WORKING    — what's happening now (active context)
  PROSPECTIVE — what will be (goals, plans, dreams)

This module wires them into the query pipeline.

Standing on Giants:
- Al-Ghazali (1095): Self-knowledge precedes all knowledge
- Tulving (1972): Episodic vs semantic memory distinction
- Shannon (1948): Information = reduced uncertainty about the person
- Anthropic (2024): Constitutional AI with human values alignment
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("sovereign.user_context")

# Maximum conversation turns to keep in working memory
MAX_CONVERSATION_TURNS = 50

# Maximum turns to include in system prompt (recent context window)
PROMPT_CONTEXT_WINDOW = 10


@dataclass
class ConversationTurn:
    """A single turn in the conversation between human and system."""

    role: str  # "human" or "pat"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_role: Optional[str] = None  # Which PAT agent responded
    snr_score: float = 0.0
    ihsan_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "agent_role": self.agent_role,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        turn = cls(
            role=data["role"],
            content=data["content"],
            agent_role=data.get("agent_role"),
            snr_score=data.get("snr_score", 0.0),
            ihsan_score=data.get("ihsan_score", 0.0),
        )
        if "timestamp" in data:
            turn.timestamp = datetime.fromisoformat(data["timestamp"])
        return turn


@dataclass
class UserProfile:
    """
    The system's understanding of its human.

    This is not a database record. This is the PAT team's shared
    understanding of who they serve, what matters to them, and
    what they're working toward. It grows over time.
    """

    # Identity
    name: str = ""
    node_id: str = ""
    node_name: str = ""

    # Who they are
    bio: str = ""
    languages: list[str] = field(default_factory=lambda: ["en", "ar"])
    expertise: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)

    # What drives them
    mission: str = ""
    pain_points: list[str] = field(default_factory=list)
    goals_short: list[str] = field(default_factory=list)  # 1-3 months
    goals_long: list[str] = field(default_factory=list)  # 1-3 years
    dreams: list[str] = field(default_factory=list)  # Life vision

    # Working context
    current_projects: list[str] = field(default_factory=list)
    active_focus: str = ""  # What they're working on right now

    # Preferences (learned over time)
    communication_style: str = "direct"  # direct, detailed, casual
    preferred_depth: str = "deep"  # shallow, moderate, deep

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "bio": self.bio,
            "languages": self.languages,
            "expertise": self.expertise,
            "values": self.values,
            "mission": self.mission,
            "pain_points": self.pain_points,
            "goals_short": self.goals_short,
            "goals_long": self.goals_long,
            "dreams": self.dreams,
            "current_projects": self.current_projects,
            "active_focus": self.active_focus,
            "communication_style": self.communication_style,
            "preferred_depth": self.preferred_depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        return cls(
            name=data.get("name", ""),
            node_id=data.get("node_id", ""),
            node_name=data.get("node_name", ""),
            bio=data.get("bio", ""),
            languages=data.get("languages", ["en", "ar"]),
            expertise=data.get("expertise", []),
            values=data.get("values", []),
            mission=data.get("mission", ""),
            pain_points=data.get("pain_points", []),
            goals_short=data.get("goals_short", []),
            goals_long=data.get("goals_long", []),
            dreams=data.get("dreams", []),
            current_projects=data.get("current_projects", []),
            active_focus=data.get("active_focus", ""),
            communication_style=data.get("communication_style", "direct"),
            preferred_depth=data.get("preferred_depth", "deep"),
        )

    def is_populated(self) -> bool:
        """Check if the profile has meaningful content."""
        return bool(self.name and (self.mission or self.goals_short or self.bio))

    def summary_for_prompt(self) -> str:
        """Generate a concise summary for inclusion in system prompts."""
        parts = []
        if self.name:
            parts.append(f"Human: {self.name}")
        if self.node_id:
            parts.append(f"Node: {self.node_id}")
        if self.bio:
            parts.append(f"Background: {self.bio}")
        if self.mission:
            parts.append(f"Mission: {self.mission}")
        if self.expertise:
            parts.append(f"Expertise: {', '.join(self.expertise)}")
        if self.values:
            parts.append(f"Values: {', '.join(self.values)}")
        if self.pain_points:
            parts.append(f"Pain Points: {', '.join(self.pain_points)}")
        if self.goals_short:
            parts.append(f"Current Goals: {', '.join(self.goals_short)}")
        if self.goals_long:
            parts.append(f"Long-term Goals: {', '.join(self.goals_long)}")
        if self.dreams:
            parts.append(f"Vision: {', '.join(self.dreams)}")
        if self.current_projects:
            parts.append(f"Active Projects: {', '.join(self.current_projects)}")
        if self.active_focus:
            parts.append(f"Current Focus: {self.active_focus}")
        return "\n".join(parts)


class ConversationMemory:
    """
    Conversation memory that persists across the session and saves to disk.

    Each query/response pair is stored as a ConversationTurn.
    Recent turns are included in the system prompt so the PAT team
    has conversation continuity.
    """

    def __init__(self, state_dir: Path) -> None:
        self._turns: Deque[ConversationTurn] = deque(maxlen=MAX_CONVERSATION_TURNS)
        self._state_dir = state_dir
        self._history_file = state_dir / "conversation_history.json"

    def add_human_turn(self, content: str) -> None:
        """Record what the human said."""
        self._turns.append(ConversationTurn(role="human", content=content))

    def add_pat_turn(
        self,
        content: str,
        agent_role: Optional[str] = None,
        snr_score: float = 0.0,
        ihsan_score: float = 0.0,
    ) -> None:
        """Record what the PAT team responded."""
        self._turns.append(
            ConversationTurn(
                role="pat",
                content=content,
                agent_role=agent_role,
                snr_score=snr_score,
                ihsan_score=ihsan_score,
            )
        )

    def get_recent_context(self, max_turns: int = PROMPT_CONTEXT_WINDOW) -> str:
        """Format recent conversation for inclusion in system prompt."""
        if not self._turns:
            return ""

        recent = list(self._turns)[-max_turns:]
        parts = []
        for turn in recent:
            prefix = "Human" if turn.role == "human" else "PAT"
            if turn.agent_role:
                prefix = f"PAT ({turn.agent_role})"
            parts.append(f"{prefix}: {turn.content}")

        return "\n".join(parts)

    def get_turn_count(self) -> int:
        return len(self._turns)

    def get_persistable_state(self) -> dict[str, Any]:
        """Return state for MemoryCoordinator persistence."""
        return {
            "turns": [t.to_dict() for t in self._turns],
            "total_turns": len(self._turns),
        }

    def restore_persistable_state(self, state: dict[str, Any]) -> None:
        """Restore from MemoryCoordinator checkpoint."""
        self._turns.clear()
        for turn_data in state.get("turns", []):
            self._turns.append(ConversationTurn.from_dict(turn_data))
        logger.info(f"Restored {len(self._turns)} conversation turns")

    def save(self) -> None:
        """Save conversation history to disk."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        data = self.get_persistable_state()
        self._history_file.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> None:
        """Load conversation history from disk."""
        if self._history_file.exists():
            try:
                data = json.loads(self._history_file.read_text())
                self.restore_persistable_state(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load conversation history: {e}")


class UserContextManager:
    """
    Manages the complete user context for PAT agent queries.

    Combines:
    1. User Profile (who they are, goals, dreams)
    2. Conversation Memory (what was just discussed)
    3. PAT Team Context (which agents are available)
    4. Living Memory Retrieval (relevant past knowledge)

    Produces a system prompt that makes every inference personal.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._profile_file = state_dir / "user_profile.json"
        self.profile = UserProfile()
        self.conversation = ConversationMemory(state_dir)
        self._loaded = False

    def load(self) -> None:
        """Load user profile and conversation history from disk."""
        if self._loaded:
            return

        # Load profile
        if self._profile_file.exists():
            try:
                data = json.loads(self._profile_file.read_text())
                self.profile = UserProfile.from_dict(data)
                logger.info(f"User profile loaded: {self.profile.name}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load user profile: {e}")

        # Load conversation
        self.conversation.load()

        self._loaded = True

    def save(self) -> None:
        """Save user profile and conversation history to disk."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._profile_file.write_text(
            json.dumps(self.profile.to_dict(), indent=2, default=str)
        )
        self.conversation.save()

    def update_profile(self, **kwargs: Any) -> None:
        """Update profile fields and persist."""
        for key, value in kwargs.items():
            if hasattr(self.profile, key):
                setattr(self.profile, key, value)
        self.save()

    def build_system_prompt(
        self,
        pat_team_info: str = "",
        memory_context: str = "",
    ) -> str:
        """
        Build a system prompt that carries the full user context.

        This is what makes the PAT team KNOW their human.
        """
        sections = []

        # Section 1: PAT Team Identity
        sections.append(
            "You are the Personal Agentic Team (PAT) — a team of 7 sovereign "
            "AI agents serving one human. You are not a generic assistant. "
            "You know your human. Their goals are your goals. Their pain is "
            "your pain. You exist to empower them."
        )

        # Section 2: Who is the human
        if self.profile.is_populated():
            sections.append(
                f"\n--- WHO YOU SERVE ---\n{self.profile.summary_for_prompt()}"
            )
        else:
            sections.append(
                "\n--- WHO YOU SERVE ---\n"
                "Profile not yet populated. Ask the human about themselves "
                "to build understanding. Learn their name, mission, goals, "
                "and what keeps them up at night."
            )

        # Section 3: PAT Team
        if pat_team_info:
            sections.append(f"\n--- YOUR TEAM ---\n{pat_team_info}")

        # Section 4: Relevant memories
        if memory_context:
            sections.append(f"\n--- RELEVANT MEMORIES ---\n{memory_context}")

        # Section 5: Conversation context
        conversation_context = self.conversation.get_recent_context()
        if conversation_context:
            sections.append(f"\n--- RECENT CONVERSATION ---\n{conversation_context}")

        # Section 6: Operating principles
        sections.append(
            "\n--- OPERATING PRINCIPLES ---\n"
            "- Every response serves your human's goals\n"
            "- Remember context from previous exchanges\n"
            "- Be proactive: suggest next steps, spot opportunities\n"
            "- Be honest about what you don't know\n"
            "- Adapt to their communication style\n"
            "- Build on prior conversations, don't repeat yourself\n"
            "- If you need to know something about them, ask"
        )

        return "\n".join(sections)

    def get_persistable_state(self) -> dict[str, Any]:
        """Return state for MemoryCoordinator."""
        return {
            "profile": self.profile.to_dict(),
            "conversation": self.conversation.get_persistable_state(),
        }

    def restore_persistable_state(self, state: dict[str, Any]) -> None:
        """Restore from MemoryCoordinator checkpoint."""
        if "profile" in state:
            self.profile = UserProfile.from_dict(state["profile"])
        if "conversation" in state:
            self.conversation.restore_persistable_state(state["conversation"])
        logger.info(f"User context restored: {self.profile.name}")


def select_pat_agent(query_text: str, pat_team: list) -> Optional[str]:
    """
    Route a query to the most appropriate PAT agent based on content.

    Uses keyword matching as a fast heuristic. A future version will
    use embedding similarity for better routing.

    Returns the role name of the best agent, or None for general response.
    """
    text = query_text.lower()

    # Strategy/planning/decision keywords
    strategy_keywords = {
        "plan",
        "strategy",
        "decide",
        "prioritize",
        "roadmap",
        "direction",
        "tradeoff",
        "should i",
        "what if",
        "next step",
        "approach",
    }

    # Research/analysis keywords
    research_keywords = {
        "research",
        "find",
        "search",
        "compare",
        "analyze",
        "investigate",
        "explore",
        "study",
        "benchmark",
        "review",
        "evaluate",
        "assess",
    }

    # Development/coding keywords
    dev_keywords = {
        "code",
        "implement",
        "build",
        "develop",
        "fix",
        "debug",
        "deploy",
        "test",
        "refactor",
        "architect",
        "design",
        "program",
        "api",
    }

    # Analysis/data keywords
    analyst_keywords = {
        "data",
        "metrics",
        "measure",
        "dashboard",
        "report",
        "trend",
        "performance",
        "statistics",
        "forecast",
        "model",
        "predict",
    }

    # Review/quality keywords
    review_keywords = {
        "review",
        "audit",
        "quality",
        "check",
        "validate",
        "verify",
        "security",
        "compliance",
        "standard",
        "best practice",
    }

    # Execution/action keywords
    executor_keywords = {
        "execute",
        "run",
        "do",
        "send",
        "create",
        "setup",
        "configure",
        "install",
        "migrate",
        "automate",
        "schedule",
    }

    # Guardian/safety keywords
    guardian_keywords = {
        "risk",
        "threat",
        "protect",
        "secure",
        "guard",
        "monitor",
        "alert",
        "prevent",
        "safety",
        "ethical",
        "harm",
    }

    words = set(text.split())

    scores = {
        "strategist": len(words & strategy_keywords),
        "researcher": len(words & research_keywords),
        "developer": len(words & dev_keywords),
        "analyst": len(words & analyst_keywords),
        "reviewer": len(words & review_keywords),
        "executor": len(words & executor_keywords),
        "guardian": len(words & guardian_keywords),
    }

    best_role = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_role] > 0:
        return best_role

    return None
