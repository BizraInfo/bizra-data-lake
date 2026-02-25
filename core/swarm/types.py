"""Swarm Coordination Types — Topology, Agent Specs, Events.

ADR-004 Phase 1: Defines the shared vocabulary for all swarm operations.

Standing on Giants: Lamport (distributed state) . Burns (K8s patterns) . Hamilton (operations)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

# -- Topology ------------------------------------------------------------------


class SwarmTopology(str, Enum):
    """Execution topology for a swarm mission."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL_MESH = "hierarchical_mesh"


class AgentRole(str, Enum):
    """Unified agent role taxonomy."""

    STRATEGIST = "strategist"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CREATOR = "creator"
    EXECUTOR = "executor"
    GUARDIAN = "guardian"
    COORDINATOR = "coordinator"


class SwarmPhase(str, Enum):
    """Lifecycle phases of a swarm mission."""

    INITIALIZING = "initializing"
    PRELOADING = "preloading"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    SCORING = "scoring"
    EQUALIZING = "equalizing"
    COMPLETE = "complete"
    FAILED = "failed"


# -- Agent Spec ----------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """Unified agent descriptor — bridges PAT_AGENTS and AgentConfig."""

    id: str
    role: AgentRole
    model_purpose: str
    system_prompt: str = ""
    max_tokens: int = 600
    timeout_seconds: float = 30.0
    is_thinking_model: bool = False

    _ROLE_MAP: dict[str, AgentRole] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    @staticmethod
    def from_pat_agent(agent_id: str, pat_dict: dict[str, Any]) -> AgentSpec:
        """Convert a PAT_AGENTS entry to AgentSpec.

        Args:
            agent_id: Key from PAT_AGENTS dict (e.g. "strategist").
            pat_dict: Value dict with keys: name, role, giants, model_purpose.
        """
        role_map = {
            "strategist": AgentRole.STRATEGIST,
            "researcher": AgentRole.RESEARCHER,
            "analyst": AgentRole.ANALYST,
            "creator": AgentRole.CREATOR,
            "executor": AgentRole.EXECUTOR,
            "guardian": AgentRole.GUARDIAN,
            "coordinator": AgentRole.COORDINATOR,
        }
        purpose = pat_dict.get("model_purpose", "reasoning")
        is_thinking = purpose in ("thinking", "reasoning", "reasoning_large")
        return AgentSpec(
            id=agent_id,
            role=role_map.get(agent_id, AgentRole.COORDINATOR),
            model_purpose=purpose,
            system_prompt=(
                f"You are the PAT {pat_dict.get('name', agent_id)}. "
                f"Your role is {pat_dict.get('role', '')}.\n"
                f"Standing on Giants: {pat_dict.get('giants', '')}.\n"
                f"Be concise (2-3 paragraphs). Focus on actionable insights."
            ),
            max_tokens=1200 if is_thinking else 600,
            timeout_seconds=120.0 if is_thinking else 30.0,
            is_thinking_model=is_thinking,
        )


# -- Swarm Config --------------------------------------------------------------


@dataclass
class SwarmConfig:
    """Single configuration surface for all swarm operations."""

    topology: SwarmTopology = SwarmTopology.SEQUENTIAL
    max_concurrent: int = 3
    preload_models: bool = True
    equalizer_enabled: bool = True
    got_synthesis: bool = True
    ihsan_threshold: float = 0.95
    agent_timeout: float = 120.0
    mission_timeout: float = 600.0


# -- Swarm Events --------------------------------------------------------------


class SwarmEventKind(str, Enum):
    """Event types emitted by SwarmEngine."""

    SWARM_CREATED = "swarm_created"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    MODEL_PRELOADED = "model_preloaded"
    EQUALIZER_ACTION = "equalizer_action"
    PHASE_CHANGED = "phase_changed"
    MISSION_COMPLETE = "mission_complete"


@dataclass(frozen=True)
class SwarmEvent:
    """Immutable event emitted by SwarmEngine at each lifecycle transition."""

    kind: SwarmEventKind
    swarm_id: str
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
