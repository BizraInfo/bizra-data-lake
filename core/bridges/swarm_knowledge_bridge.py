"""
Swarm Knowledge Bridge — Agent-to-Knowledge Interface
======================================================
Provides the interface for swarm agents to access the integrated
BIZRA Data Lake knowledge base with role-based access and caching.

Enables:
- Agent-specific knowledge retrieval
- Role-based access control
- Knowledge injection into agent context
- Cross-agent knowledge sharing
- MoMo R&D context propagation

Standing on Giants: RAG + Multi-Agent Systems + Knowledge Graphs
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .knowledge_integrator import (
    KnowledgeIntegrator,
    KnowledgeQuery,
    KnowledgeResult,
    create_knowledge_integrator,
)
from core.orchestration.team_planner import AgentRole
from core.orchestration.event_bus import EventBus, EventPriority, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class AgentKnowledgeContext:
    """Knowledge context for a specific agent."""
    agent_role: AgentRole = AgentRole.MASTER_REASONER
    accessible_categories: Set[str] = field(default_factory=set)
    preloaded_knowledge: Dict[str, Any] = field(default_factory=dict)
    query_history: List[KnowledgeQuery] = field(default_factory=list)
    last_refresh: Optional[datetime] = None


# Role-based knowledge access matrix
ROLE_KNOWLEDGE_ACCESS = {
    # PAT Agents
    AgentRole.MASTER_REASONER: {
        "categories": {"memory", "graph", "session", "foundation", "integration", "insights"},
        "priority_access": True,
        "can_write": True,
    },
    AgentRole.DATA_ANALYZER: {
        "categories": {"corpus", "embedding", "graph", "patterns", "insights"},
        "priority_access": True,
        "can_write": False,
    },
    AgentRole.EXECUTION_PLANNER: {
        "categories": {"session", "integration", "patterns", "index"},
        "priority_access": False,
        "can_write": False,
    },
    AgentRole.ETHICS_GUARDIAN: {
        "categories": {"foundation", "framework", "identity"},
        "priority_access": True,
        "can_write": False,
    },
    AgentRole.COMMUNICATOR: {
        "categories": {"session", "insights", "index"},
        "priority_access": False,
        "can_write": False,
    },
    AgentRole.MEMORY_ARCHITECT: {
        "categories": {"memory", "graph", "embedding", "corpus", "index"},
        "priority_access": True,
        "can_write": True,
    },
    AgentRole.FUSION: {
        "categories": {"memory", "graph", "session", "integration", "insights"},
        "priority_access": True,
        "can_write": True,
    },

    # SAT Validators
    AgentRole.SECURITY_GUARDIAN: {
        "categories": {"session", "integration", "identity"},
        "priority_access": True,
        "can_write": False,
    },
    AgentRole.ETHICS_VALIDATOR: {
        "categories": {"foundation", "framework", "identity"},
        "priority_access": True,
        "can_write": False,
    },
    AgentRole.PERFORMANCE_MONITOR: {
        "categories": {"session", "index"},
        "priority_access": False,
        "can_write": False,
    },
    AgentRole.CONSISTENCY_CHECKER: {
        "categories": {"integration", "index", "patterns"},
        "priority_access": False,
        "can_write": False,
    },
    AgentRole.RESOURCE_OPTIMIZER: {
        "categories": {"session", "index"},
        "priority_access": False,
        "can_write": False,
    },
}


@dataclass
class KnowledgeInjection:
    """Knowledge to inject into agent context."""
    agent_role: AgentRole = AgentRole.MASTER_REASONER
    knowledge_type: str = "context"  # context, reference, constraint
    content: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    snr_score: float = 0.0
    expires_at: Optional[datetime] = None


class SwarmKnowledgeBridge:
    """
    Bridge between swarm agents and the BIZRA knowledge base.

    Features:
    - Role-based access control
    - Automatic context enrichment
    - Cross-agent knowledge sharing
    - MoMo R&D context injection
    - Event-driven knowledge updates
    """

    def __init__(
        self,
        integrator: Optional[KnowledgeIntegrator] = None,
        event_bus: Optional[EventBus] = None,
        ihsan_threshold: float = 0.95,
    ):
        self.integrator = integrator
        self.event_bus = event_bus or get_event_bus()
        self.ihsan_threshold = ihsan_threshold

        # Agent contexts
        self._agent_contexts: Dict[AgentRole, AgentKnowledgeContext] = {}

        # Knowledge injections queue
        self._injection_queue: Dict[AgentRole, List[KnowledgeInjection]] = {}

        # Statistics
        self._queries_served = 0
        self._injections_made = 0
        self._cross_agent_shares = 0

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the bridge and knowledge integrator."""
        if not self.integrator:
            self.integrator = await create_knowledge_integrator(
                ihsan_threshold=self.ihsan_threshold
            )

        # Initialize agent contexts
        for role in AgentRole:
            access = ROLE_KNOWLEDGE_ACCESS.get(role, {})
            self._agent_contexts[role] = AgentKnowledgeContext(
                agent_role=role,
                accessible_categories=access.get("categories", set()),
            )
            self._injection_queue[role] = []

        # Preload critical knowledge for priority agents
        await self._preload_priority_knowledge()

        # Subscribe to knowledge events
        self._setup_event_handlers()

        return {
            "integrator_initialized": True,
            "agents_configured": len(self._agent_contexts),
            "integrator_stats": self.integrator.stats(),
        }

    async def _preload_priority_knowledge(self) -> None:
        """Preload knowledge for priority access agents."""
        # Load MoMo context for all agents
        momo_context = self.integrator.get_momo_context()
        giants = self.integrator.get_standing_on_giants()

        for role, access in ROLE_KNOWLEDGE_ACCESS.items():
            if access.get("priority_access"):
                ctx = self._agent_contexts[role]
                ctx.preloaded_knowledge["momo_context"] = momo_context
                ctx.preloaded_knowledge["standing_on_giants"] = giants
                ctx.last_refresh = datetime.now(timezone.utc)

        logger.info("Priority knowledge preloaded for privileged agents")

    def _setup_event_handlers(self) -> None:
        """Set up event bus handlers for knowledge updates."""

        async def handle_knowledge_request(event):
            """Handle knowledge request from agent."""
            role_str = event.payload.get("requester_role")
            query_text = event.payload.get("query")

            if role_str and query_text:
                try:
                    role = AgentRole(role_str)
                    result = await self.query_for_agent(role, query_text)
                    await self.event_bus.emit(
                        topic="knowledge.response",
                        payload={
                            "query_id": result.query_id,
                            "requester_role": role_str,
                            "results": result.results[:5],  # Limit for event
                            "snr_score": result.snr_score,
                        },
                        priority=EventPriority.HIGH,
                    )
                except Exception as e:
                    logger.error(f"Knowledge request failed: {e}")

        self.event_bus.subscribe("knowledge.request", handle_knowledge_request)

    async def query_for_agent(
        self,
        role: AgentRole,
        query: str,
        max_results: int = 10,
        min_snr: float = 0.85,
    ) -> KnowledgeResult:
        """Execute a knowledge query on behalf of an agent."""
        self._queries_served += 1

        # Get agent's accessible categories
        access = ROLE_KNOWLEDGE_ACCESS.get(role, {})
        categories = list(access.get("categories", []))

        kq = KnowledgeQuery(
            query=query,
            max_results=max_results,
            min_snr=min_snr,
            categories=categories,
            requester=role.value,
        )

        # Record in agent context
        ctx = self._agent_contexts.get(role)
        if ctx:
            ctx.query_history.append(kq)
            if len(ctx.query_history) > 100:
                ctx.query_history = ctx.query_history[-50:]

        result = await self.integrator.query(kq)
        return result

    async def inject_knowledge(
        self,
        injection: KnowledgeInjection,
    ) -> bool:
        """Inject knowledge into an agent's context."""
        role = injection.agent_role
        ctx = self._agent_contexts.get(role)

        if not ctx:
            return False

        # Validate SNR
        if injection.snr_score < self.ihsan_threshold - 0.1:
            logger.warning(f"Injection rejected: SNR {injection.snr_score} too low")
            return False

        # Add to context
        key = f"{injection.knowledge_type}:{injection.source}"
        ctx.preloaded_knowledge[key] = injection.content
        self._injections_made += 1

        # Queue for agent
        self._injection_queue[role].append(injection)

        # Emit event
        await self.event_bus.emit(
            topic="knowledge.injected",
            payload={
                "agent_role": role.value,
                "knowledge_type": injection.knowledge_type,
                "source": injection.source,
            },
            priority=EventPriority.NORMAL,
        )

        return True

    async def share_knowledge(
        self,
        from_role: AgentRole,
        to_role: AgentRole,
        knowledge_key: str,
    ) -> bool:
        """Share knowledge from one agent to another."""
        from_ctx = self._agent_contexts.get(from_role)
        to_ctx = self._agent_contexts.get(to_role)

        if not from_ctx or not to_ctx:
            return False

        if knowledge_key not in from_ctx.preloaded_knowledge:
            return False

        # Check target can receive this category
        # (simplified - in production would check category)

        to_ctx.preloaded_knowledge[f"shared:{from_role.value}:{knowledge_key}"] = \
            from_ctx.preloaded_knowledge[knowledge_key]

        self._cross_agent_shares += 1

        await self.event_bus.emit(
            topic="knowledge.shared",
            payload={
                "from_role": from_role.value,
                "to_role": to_role.value,
                "key": knowledge_key,
            },
            priority=EventPriority.NORMAL,
        )

        return True

    def get_agent_context(self, role: AgentRole) -> Dict[str, Any]:
        """Get the current knowledge context for an agent."""
        ctx = self._agent_contexts.get(role)
        if not ctx:
            return {}

        return {
            "role": role.value,
            "accessible_categories": list(ctx.accessible_categories),
            "preloaded_keys": list(ctx.preloaded_knowledge.keys()),
            "query_count": len(ctx.query_history),
            "last_refresh": ctx.last_refresh.isoformat() if ctx.last_refresh else None,
            "pending_injections": len(self._injection_queue.get(role, [])),
        }

    def get_pending_injections(self, role: AgentRole) -> List[KnowledgeInjection]:
        """Get and clear pending knowledge injections for an agent."""
        injections = self._injection_queue.get(role, [])
        self._injection_queue[role] = []
        return injections

    async def refresh_agent_context(self, role: AgentRole) -> bool:
        """Refresh an agent's knowledge context."""
        ctx = self._agent_contexts.get(role)
        if not ctx:
            return False

        # Reload MoMo context
        ctx.preloaded_knowledge["momo_context"] = self.integrator.get_momo_context()
        ctx.last_refresh = datetime.now(timezone.utc)

        return True

    def get_momo_context(self) -> Dict[str, Any]:
        """Get MoMo's R&D context for agents."""
        return self.integrator.get_momo_context()

    def stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "queries_served": self._queries_served,
            "injections_made": self._injections_made,
            "cross_agent_shares": self._cross_agent_shares,
            "agents_with_context": len(self._agent_contexts),
            "total_pending_injections": sum(
                len(q) for q in self._injection_queue.values()
            ),
            "integrator_stats": self.integrator.stats() if self.integrator else None,
        }


async def create_swarm_knowledge_bridge(
    ihsan_threshold: float = 0.95,
) -> SwarmKnowledgeBridge:
    """Create and initialize a swarm knowledge bridge."""
    bridge = SwarmKnowledgeBridge(ihsan_threshold=ihsan_threshold)
    await bridge.initialize()
    return bridge


__all__ = [
    "AgentKnowledgeContext",
    "KnowledgeInjection",
    "ROLE_KNOWLEDGE_ACCESS",
    "SwarmKnowledgeBridge",
    "create_swarm_knowledge_bridge",
]
