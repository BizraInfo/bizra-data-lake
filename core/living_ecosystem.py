"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA LIVING ECOSYSTEM — Unified Self-Sustaining Intelligence              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   • Maturana & Varela (Autopoiesis - Living Systems)                         ║
║   • Shannon (Information Theory)                                             ║
║   • Anthropic (Constitutional AI / Ihsān)                                    ║
║   • Holland (Genetic Algorithms)                                             ║
║   • OpenClaw/PAT (Personal AI Team)                                          ║
║   • DATA4LLM (Data Processing for LLMs)                                      ║
║                                                                              ║
║   The Living Ecosystem integrates:                                           ║
║   • Living Memory: Self-organizing, self-healing knowledge store             ║
║   • Agentic System: Autonomous decision-making agents                        ║
║   • PAT Bridge: Multi-channel communication                                  ║
║   • Autopoiesis: Continuous self-evolution                                   ║
║   • Proactive Engineering: Anticipatory information management               ║
║                                                                              ║
║   This is BIZRA's cognitive core — a self-sustaining, self-optimizing        ║
║   intelligence that operates within strict constitutional bounds.            ║
║                                                                              ║
║   Created: 2026-02-02 | BIZRA Genesis v2.2.3                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


class EcosystemState(str, Enum):
    """State of the living ecosystem."""

    DORMANT = "dormant"
    INITIALIZING = "initializing"
    RUNNING = "running"
    EVOLVING = "evolving"
    HEALING = "healing"
    OPTIMIZING = "optimizing"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class EcosystemHealth:
    """Health metrics for the ecosystem."""

    overall_health: float = 1.0
    memory_health: float = 1.0
    agent_health: float = 1.0
    evolution_health: float = 1.0

    ihsan_compliance: float = 1.0
    snr_average: float = 1.0

    last_check: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_health": self.overall_health,
            "memory_health": self.memory_health,
            "agent_health": self.agent_health,
            "evolution_health": self.evolution_health,
            "ihsan_compliance": self.ihsan_compliance,
            "snr_average": self.snr_average,
            "last_check": self.last_check.isoformat() if self.last_check else None,
        }


@dataclass
class EcosystemConfig:
    """Configuration for the living ecosystem."""

    # Storage
    data_path: Path = field(default_factory=lambda: Path("/var/lib/bizra"))

    # Thresholds
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD

    # Timing
    maintenance_interval_seconds: float = 60.0
    evolution_interval_seconds: float = 300.0
    memory_consolidation_interval_seconds: float = 180.0

    # Limits
    max_agents: int = 10
    max_memory_entries: int = 100_000

    # Features
    enable_evolution: bool = True
    enable_proactive: bool = True
    enable_pat: bool = True


class LivingEcosystem:
    """
    The unified BIZRA Living Ecosystem.

    Orchestrates all subsystems into a coherent, self-sustaining intelligence:
    - Living Memory for knowledge persistence
    - Agent Orchestrator for task execution
    - Autopoietic Loop for continuous evolution
    - PAT Bridge for multi-channel communication
    - Self-Healing for automatic error recovery
    """

    def __init__(
        self,
        config: Optional[EcosystemConfig] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        embedding_fn: Optional[Callable[[str], Any]] = None,
    ):
        self.config = config or EcosystemConfig()
        self.llm_fn = llm_fn
        self.embedding_fn = embedding_fn

        # State
        self.state = EcosystemState.DORMANT
        self._start_time: Optional[datetime] = None
        self._health = EcosystemHealth()

        # Subsystems (lazy initialized)
        self._memory = None
        self._memory_healer = None
        self._proactive_retriever = None
        self._orchestrator = None
        self._autopoietic_loop = None
        self._pat_bridge = None

        # Background tasks
        self._tasks: List[asyncio.Task] = []

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def initialize(self) -> None:
        """Initialize all ecosystem components."""
        self.state = EcosystemState.INITIALIZING
        self._start_time = datetime.now(timezone.utc)

        logger.info("Initializing BIZRA Living Ecosystem...")

        # Create data directory
        self.config.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize Living Memory
        from core.living_memory.core import LivingMemoryCore

        self._memory = LivingMemoryCore(
            storage_path=self.config.data_path / "memory",
            embedding_fn=self.embedding_fn,
            llm_fn=self.llm_fn,
            max_entries=self.config.max_memory_entries,
            ihsan_threshold=self.config.ihsan_threshold,
        )
        await self._memory.initialize()

        # Initialize Memory Healer
        from core.living_memory.healing import MemoryHealer

        self._memory_healer = MemoryHealer(
            memory=self._memory,
            llm_fn=self.llm_fn,
            ihsan_threshold=self.config.ihsan_threshold,
            snr_threshold=self.config.snr_threshold,
        )

        # Initialize Proactive Retriever
        if self.config.enable_proactive:
            from core.living_memory.proactive import ProactiveRetriever

            self._proactive_retriever = ProactiveRetriever(
                memory=self._memory,
                llm_fn=self.llm_fn,
            )

        # Initialize Agent Orchestrator
        from core.agentic.orchestrator import AgentOrchestrator

        self._orchestrator = AgentOrchestrator(
            memory=self._memory,
            llm_fn=self.llm_fn,
            max_agents=self.config.max_agents,
            ihsan_threshold=self.config.ihsan_threshold,
        )
        await self._orchestrator.initialize()

        # Initialize Autopoietic Loop
        if self.config.enable_evolution:
            try:
                from core.autopoiesis.loop import AutopoiesisConfig, AutopoieticLoop

                autopoiesis_config = AutopoiesisConfig(
                    ihsan_threshold=self.config.ihsan_threshold,
                    snr_threshold=self.config.snr_threshold,
                )
                self._autopoietic_loop = AutopoieticLoop(config=autopoiesis_config)
            except ImportError as e:
                logger.warning(f"Autopoiesis not available: {e}")

        # Initialize PAT Bridge
        if self.config.enable_pat:
            from core.pat.bridge import PATBridge

            self._pat_bridge = PATBridge(
                memory=self._memory,
                llm_fn=self.llm_fn,
                ihsan_threshold=self.config.ihsan_threshold,
            )

        self.state = EcosystemState.RUNNING
        logger.info("BIZRA Living Ecosystem initialized successfully")

    async def start(self) -> None:
        """Start all background processes."""
        if self.state != EcosystemState.RUNNING:
            await self.initialize()

        # Start maintenance loop
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))

        # Start memory consolidation loop
        self._tasks.append(asyncio.create_task(self._memory_consolidation_loop()))

        # Start evolution loop
        if self._autopoietic_loop and self.config.enable_evolution:
            self._tasks.append(asyncio.create_task(self._evolution_loop()))

        # Connect PAT bridge
        if self._pat_bridge and self.config.enable_pat:
            await self._pat_bridge.connect()

        logger.info("Living Ecosystem started - all background processes running")

    async def stop(self) -> None:
        """Stop all background processes."""
        self.state = EcosystemState.SHUTTING_DOWN

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Disconnect PAT
        if self._pat_bridge:
            await self._pat_bridge.disconnect()

        # Shutdown orchestrator
        if self._orchestrator:
            await self._orchestrator.shutdown()

        # Final memory save
        if self._memory:
            await self._memory._save_memories()

        self.state = EcosystemState.DORMANT
        logger.info("Living Ecosystem stopped")

    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while self.state == EcosystemState.RUNNING:
            try:
                await self._run_maintenance()
                await asyncio.sleep(self.config.maintenance_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance error: {e}")
                await asyncio.sleep(10)

    async def _memory_consolidation_loop(self) -> None:
        """Background memory consolidation loop."""
        while self.state == EcosystemState.RUNNING:
            try:
                await asyncio.sleep(self.config.memory_consolidation_interval_seconds)
                await self._consolidate_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory consolidation error: {e}")

    async def _evolution_loop(self) -> None:
        """Background evolution loop."""
        while self.state == EcosystemState.RUNNING:
            try:
                await asyncio.sleep(self.config.evolution_interval_seconds)
                await self._run_evolution()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evolution error: {e}")

    async def _run_maintenance(self) -> None:
        """Run maintenance cycle."""
        prev_state = self.state
        self.state = EcosystemState.HEALING

        # Check memory health
        if self._memory_healer:
            health_report = self._memory_healer.get_health_report()
            self._health.memory_health = health_report.get("health_score", 1.0)

            # Auto-heal if needed
            if self._health.memory_health < 0.8:
                await self._memory_healer.optimize()

        # Check agent health
        if self._orchestrator:
            await self._orchestrator.run_maintenance()
            stats = self._orchestrator.get_stats()
            if stats.total_agents > 0:
                self._health.agent_health = stats.active_agents / stats.total_agents

        # Update overall health
        self._health.overall_health = (
            0.4 * self._health.memory_health
            + 0.3 * self._health.agent_health
            + 0.2 * self._health.ihsan_compliance
            + 0.1 * (self._health.snr_average / self.config.snr_threshold)
        )
        self._health.last_check = datetime.now(timezone.utc)

        self.state = prev_state

    async def _consolidate_memory(self) -> None:
        """Consolidate memory."""
        if not self._memory:
            return

        self.state = EcosystemState.OPTIMIZING
        stats = await self._memory.consolidate()
        logger.debug(f"Memory consolidation: {stats}")
        self.state = EcosystemState.RUNNING

    async def _run_evolution(self) -> None:
        """Run evolution cycle."""
        if not self._autopoietic_loop:
            return

        self.state = EcosystemState.EVOLVING

        try:
            await self._autopoietic_loop.run_cycle()
            self._health.evolution_health = 1.0
        except Exception as e:
            logger.error(f"Evolution cycle failed: {e}")
            self._health.evolution_health = 0.5

        self.state = EcosystemState.RUNNING

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def think(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Process a query through the living ecosystem.

        Combines:
        - Memory retrieval
        - Proactive suggestions
        - LLM reasoning
        - Constitutional validation
        """
        if not self.llm_fn:
            return "LLM not available"

        # Update proactive context
        if self._proactive_retriever:
            self._proactive_retriever.update_context(query=query)

        # Retrieve relevant memories
        memories = []
        if self._memory:
            memories = await self._memory.retrieve(
                query=query,
                top_k=5,
                min_score=0.3,
            )

        memory_context = "\n".join(m.content for m in memories)

        # Get proactive suggestions
        suggestions = []
        if self._proactive_retriever:
            suggestions = await self._proactive_retriever.get_proactive_suggestions()

        suggestion_context = "\n".join(
            f"- {s.reason}: {s.memory.content[:100]}" for s in suggestions[:3]
        )

        # Build prompt
        full_context = f"""
Relevant Knowledge:
{memory_context[:2000]}

Proactive Suggestions:
{suggestion_context}

Additional Context:
{context or 'None'}
""".strip()

        prompt = f"""You are BIZRA, an intelligent assistant with a living memory system.

{full_context}

User Query: {query}

Respond helpfully, drawing on your knowledge and proactive suggestions."""

        # Generate response
        response = self.llm_fn(prompt)

        # Store interaction in memory
        if self._memory:
            await self._memory.encode(
                content=f"Q: {query[:200]}",
                memory_type="episodic",
                source="user",
                importance=0.7,
            )
            await self._memory.encode(
                content=f"A: {response[:500]}",
                memory_type="episodic",
                source="bizra",
                importance=0.6,
            )

        return response

    async def learn(
        self,
        content: str,
        source: str = "user",
        importance: float = 0.7,
    ) -> bool:
        """Add new knowledge to living memory."""
        if not self._memory:
            return False

        from core.living_memory.core import MemoryType

        entry = await self._memory.encode(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            source=source,
            importance=importance,
        )
        return entry is not None

    async def chat(
        self,
        message: str,
        channel: str = "internal",
    ) -> str:
        """Chat interface (routes through PAT if available)."""
        if self._pat_bridge and self.config.enable_pat:
            from core.pat.bridge import ChannelType

            try:
                channel_type = ChannelType(channel)
            except ValueError:
                channel_type = ChannelType.INTERNAL

            return await self._pat_bridge.process_local(
                content=message,
                channel=channel_type,
            )
        else:
            return await self.think(message)

    async def submit_task(
        self,
        name: str,
        description: str,
        priority: str = "normal",
    ) -> Optional[str]:
        """Submit a task to the agent orchestrator."""
        if not self._orchestrator:
            return None

        from core.agentic.agent import TaskPriority

        try:
            task_priority = TaskPriority(priority)
        except ValueError:
            task_priority = TaskPriority.NORMAL

        task = await self._orchestrator.submit_task(
            name=name,
            description=description,
            priority=task_priority,
        )
        return task.id

    def get_status(self) -> Dict[str, Any]:
        """Get ecosystem status."""
        status = {
            "state": self.state.value,
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self._start_time).total_seconds()
                if self._start_time
                else 0
            ),
            "health": self._health.to_dict(),
        }

        if self._memory:
            status["memory"] = self._memory.get_stats().to_dict()

        if self._orchestrator:
            status["orchestrator"] = self._orchestrator.get_stats().__dict__

        if self._pat_bridge:
            status["pat"] = self._pat_bridge.get_status()

        return status


# Factory function
def create_living_ecosystem(
    llm_fn: Optional[Callable[[str], str]] = None,
    embedding_fn: Optional[Callable[[str], Any]] = None,
    config: Optional[EcosystemConfig] = None,
) -> LivingEcosystem:
    """Create a new living ecosystem instance."""
    return LivingEcosystem(
        config=config,
        llm_fn=llm_fn,
        embedding_fn=embedding_fn,
    )
