"""
Agent Activator — Genesis PAT/SAT → Active Execution Instances
================================================================
Bridges the gap between genesis identity (static agent definitions)
and live agent execution (dispatching tasks through InferenceGateway).

This is the True Spearpoint: the single piece that transforms BIZRA
from "a framework with inert subsystems" into "a working proactive
sovereign AI node."

Architecture:
    GenesisState.pat_team → AgentActivator → ActiveAgent instances
    PEK proposals → AgentExecutor → ActiveAgent.execute() → InferenceGateway

Standing on Giants:
- Al-Ghazali (1095): Agent identity as spiritual covenant
- Boyd (1976): OODA loop — activated agents enable the ACT phase
- Wiener (1948): Cybernetic control loop requires actuators
- Deming (1950): Quality through systematic activation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger("sovereign.agent_activator")


# ---------------------------------------------------------------------------
# Agent Activation Types
# ---------------------------------------------------------------------------


class ActivationStatus(str, Enum):
    """Status of an activated agent."""

    DORMANT = "dormant"  # Genesis-loaded but not activated
    ACTIVATING = "activating"  # Activation in progress
    READY = "ready"  # Activated and ready for dispatch
    BUSY = "busy"  # Currently executing a task
    FAILED = "failed"  # Activation or execution failed
    DEACTIVATED = "deactivated"  # Explicitly shut down


@dataclass
class ActiveAgent:
    """A genesis-loaded agent activated for live execution.

    Bridges AgentIdentity (from genesis_identity.py) with execution
    capability via InferenceGateway.
    """

    agent_id: str
    role: str
    capabilities: List[str]
    giants: List[str]
    public_key: str
    status: ActivationStatus = ActivationStatus.DORMANT

    # Execution context
    system_prompt: str = ""
    model_purpose: str = "reasoning"  # reasoning | general | agentic

    # Performance tracking
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    last_active: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "model_purpose": self.model_purpose,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ActivationResult:
    """Result of activating a set of agents."""

    activated: int = 0
    failed: int = 0
    agents: List[ActiveAgent] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.activated > 0


# Role → model purpose mapping (aligned with scripts/node0_activate.py)
_ROLE_MODEL_MAP: Dict[str, str] = {
    "worker": "general",
    "researcher": "reasoning",
    "guardian": "reasoning",
    "synthesizer": "reasoning",
    "validator": "reasoning",
    "coordinator": "reasoning",
    "executor": "agentic",
}

# Role → system prompt template
_ROLE_PROMPTS: Dict[str, str] = {
    "worker": (
        "You are a PAT Worker agent. Your role is general task execution. "
        "Be concise, actionable, and precise. Focus on completing the task."
    ),
    "researcher": (
        "You are a PAT Researcher agent. Your role is deep investigation "
        "and evidence gathering. Standing on Giants: Vannevar Bush, Claude Shannon, "
        "Douglas Engelbart. Provide thorough, evidence-based analysis."
    ),
    "guardian": (
        "You are a PAT Guardian agent. Your role is ethical oversight and security. "
        "Standing on Giants: Al-Ghazali, John Rawls, Anthropic. "
        "Evaluate safety, ethics, and constitutional compliance."
    ),
    "synthesizer": (
        "You are a PAT Synthesizer agent. Your role is data integration and "
        "insight generation. Identify patterns, connect disparate signals, "
        "and produce high-SNR synthesis."
    ),
    "validator": (
        "You are a PAT Validator agent. Your role is proof verification "
        "and quality assurance. Verify claims, assess quality, and ensure "
        "Ihsan compliance."
    ),
    "coordinator": (
        "You are a PAT Coordinator agent. Your role is team synthesis "
        "and integration. Standing on Giants: Norbert Wiener, Peter Senge. "
        "Orchestrate multi-agent workflows and synthesize results."
    ),
    "executor": (
        "You are a PAT Executor agent. Your role is task execution "
        "and automation. Standing on Giants: Frederick Taylor, W. Edwards Deming. "
        "Execute precisely, report outcomes, handle failures gracefully."
    ),
}


# ---------------------------------------------------------------------------
# Agent Activator
# ---------------------------------------------------------------------------


class AgentActivator:
    """
    Activates genesis PAT/SAT agents into live execution instances.

    The activator:
    1. Takes AgentIdentity objects from GenesisState
    2. Creates ActiveAgent instances with system prompts and model routing
    3. Registers them as dispatch targets for PEK and the orchestrator
    4. Tracks execution metrics for each agent

    Usage:
        activator = AgentActivator()
        result = activator.activate_from_genesis(genesis_state)
        agent = activator.get_agent_for_role("researcher")
        response = await activator.dispatch(agent, "Analyze this data")
    """

    def __init__(
        self,
        gateway: Optional[object] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ) -> None:
        self._agents: Dict[str, ActiveAgent] = {}
        self._gateway = gateway
        self._ihsan_threshold = ihsan_threshold
        self._snr_threshold = snr_threshold
        self._activated = False

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def ready_count(self) -> int:
        return sum(
            1 for a in self._agents.values()
            if a.status == ActivationStatus.READY
        )

    @property
    def agents(self) -> Dict[str, ActiveAgent]:
        return dict(self._agents)

    def set_gateway(self, gateway: object) -> None:
        """Inject or update the InferenceGateway reference."""
        self._gateway = gateway

    def activate_from_genesis(self, genesis_state: object) -> ActivationResult:
        """
        Activate agents from a GenesisState object.

        Creates ActiveAgent instances for each PAT and SAT agent in the
        genesis identity, with appropriate system prompts and model routing.

        Args:
            genesis_state: A GenesisState with pat_team and sat_team lists.

        Returns:
            ActivationResult with counts and activated agent references.
        """
        result = ActivationResult()

        pat_team = getattr(genesis_state, "pat_team", [])
        sat_team = getattr(genesis_state, "sat_team", [])

        for agent_identity in pat_team + sat_team:
            try:
                active = self._activate_agent(agent_identity)
                self._agents[active.agent_id] = active
                result.activated += 1
                result.agents.append(active)
            except Exception as e:
                result.failed += 1
                agent_id = getattr(agent_identity, "agent_id", "unknown")
                result.errors.append(f"{agent_id}: {e}")
                logger.warning("Failed to activate agent %s: %s", agent_id, e)

        self._activated = result.activated > 0

        logger.info(
            "Agent activation: %d activated, %d failed (total: %d)",
            result.activated,
            result.failed,
            len(pat_team) + len(sat_team),
        )
        return result

    def _activate_agent(self, agent_identity: object) -> ActiveAgent:
        """Activate a single agent from its genesis identity."""
        agent_id = getattr(agent_identity, "agent_id", "")
        role = getattr(agent_identity, "role", "worker").lower()
        capabilities = getattr(agent_identity, "capabilities", [])
        giants = getattr(agent_identity, "giants", [])
        public_key = getattr(agent_identity, "public_key", "")

        # Build system prompt from role
        base_prompt = _ROLE_PROMPTS.get(role, _ROLE_PROMPTS["worker"])
        if giants:
            giant_str = ", ".join(giants[:5])
            system_prompt = f"{base_prompt}\nStanding on Giants: {giant_str}."
        else:
            system_prompt = base_prompt

        model_purpose = _ROLE_MODEL_MAP.get(role, "general")

        agent = ActiveAgent(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            giants=giants,
            public_key=public_key,
            status=ActivationStatus.READY,
            system_prompt=system_prompt,
            model_purpose=model_purpose,
        )

        logger.debug("Activated agent: %s (%s) → %s", agent_id, role, model_purpose)
        return agent

    def get_agent(self, agent_id: str) -> Optional[ActiveAgent]:
        """Get an activated agent by ID."""
        return self._agents.get(agent_id)

    def get_agent_for_role(self, role: str) -> Optional[ActiveAgent]:
        """Get the first ready agent matching a role."""
        role_lower = role.lower()
        for agent in self._agents.values():
            if agent.role == role_lower and agent.status == ActivationStatus.READY:
                return agent
        return None

    def get_agents_by_status(self, status: ActivationStatus) -> List[ActiveAgent]:
        """Get all agents with a given status."""
        return [a for a in self._agents.values() if a.status == status]

    def select_agents_for_task(self, task_description: str) -> List[ActiveAgent]:
        """
        Select appropriate agents for a task based on description keywords.

        Follows the same routing logic as scripts/node0_activate.py but
        uses genesis-loaded agents instead of static dicts.
        """
        desc = task_description.lower()
        selected: List[ActiveAgent] = []

        # Always include coordinator if available
        coord = self.get_agent_for_role("coordinator")
        if coord:
            selected.append(coord)

        # Keyword-based routing
        _role_keywords = {
            "researcher": ["research", "investigate", "find", "search", "discover"],
            "guardian": ["security", "safe", "risk", "ethic", "validate", "audit"],
            "synthesizer": ["analyze", "data", "pattern", "insight", "synthesize"],
            "worker": ["create", "build", "implement", "execute", "generate"],
            "validator": ["verify", "proof", "quality", "check", "assess"],
            "executor": ["run", "deploy", "automate", "tool", "api"],
        }

        for role, keywords in _role_keywords.items():
            if any(kw in desc for kw in keywords):
                agent = self.get_agent_for_role(role)
                if agent and agent not in selected:
                    selected.append(agent)

        # Default fallback: coordinator + researcher + guardian
        if len(selected) <= 1:
            for fallback_role in ["researcher", "guardian"]:
                agent = self.get_agent_for_role(fallback_role)
                if agent and agent not in selected:
                    selected.append(agent)

        return selected

    def deactivate_all(self) -> int:
        """Deactivate all agents. Returns count deactivated."""
        count = 0
        for agent in self._agents.values():
            if agent.status in (ActivationStatus.READY, ActivationStatus.BUSY):
                agent.status = ActivationStatus.DEACTIVATED
                count += 1
        self._activated = False
        logger.info("Deactivated %d agents", count)
        return count

    def summary(self) -> Dict[str, Any]:
        """Activation summary for status display."""
        by_status: Dict[str, int] = {}
        by_role: Dict[str, int] = {}
        for agent in self._agents.values():
            by_status[agent.status.value] = by_status.get(agent.status.value, 0) + 1
            by_role[agent.role] = by_role.get(agent.role, 0) + 1

        return {
            "total_agents": len(self._agents),
            "ready": self.ready_count,
            "activated": self._activated,
            "by_status": by_status,
            "by_role": by_role,
        }
