"""
Agent Orchestrator — Multi-Agent Coordination System

Coordinates multiple autonomous agents:
- Task distribution and load balancing
- Inter-agent communication
- Global state management
- Self-healing agent ecosystem

Standing on Giants: Swarm Intelligence + Autopoiesis + BFT Consensus
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

from core.agentic.agent import (
    AutonomousAgent,
    AgentTask,
    AgentState,
    TaskPriority,
    TaskStatus,
    SimpleAgent,
)
from core.living_memory.core import LivingMemoryCore, MemoryType
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


class OrchestratorState(str, Enum):
    """State of the orchestrator."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    HEALING = "healing"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class AgentHealth:
    """Health metrics for an agent."""
    agent_id: str
    is_alive: bool = True
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    consecutive_failures: int = 0
    tasks_completed: int = 0
    avg_task_time_ms: float = 0.0


@dataclass
class OrchestratorStats:
    """Statistics for the orchestrator."""
    active_agents: int = 0
    total_agents: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    uptime_seconds: float = 0.0
    healing_events: int = 0


class AgentOrchestrator:
    """
    Central coordinator for the BIZRA agentic system.

    Manages agent lifecycle, task distribution, and system health.
    """

    def __init__(
        self,
        memory: Optional[LivingMemoryCore] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_agents: int = 10,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.memory = memory
        self.llm_fn = llm_fn
        self.max_agents = max_agents
        self.ihsan_threshold = ihsan_threshold

        # State
        self.state = OrchestratorState.INITIALIZING
        self._start_time: Optional[datetime] = None

        # Agent management
        self._agents: Dict[str, AutonomousAgent] = {}
        self._agent_health: Dict[str, AgentHealth] = {}
        self._agent_tasks: Dict[str, asyncio.Task] = {}

        # Task management
        self._global_task_queue: List[AgentTask] = []
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self._completed_tasks: Dict[str, AgentTask] = {}

        # Communication
        self._message_queue: asyncio.Queue = asyncio.Queue()

        # Self-healing
        self._healing_lock = asyncio.Lock()
        self._healing_count = 0

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        self.state = OrchestratorState.INITIALIZING
        self._start_time = datetime.now(timezone.utc)

        if self.memory:
            await self.memory.initialize()

        # Start default agents
        await self._spawn_default_agents()

        self.state = OrchestratorState.RUNNING
        logger.info("Agent Orchestrator initialized")

    async def _spawn_default_agents(self) -> None:
        """Spawn default agent team."""
        # Create a simple agent for basic tasks
        basic_agent = SimpleAgent(
            name="BasicAgent",
            llm_fn=self.llm_fn,
            ihsan_threshold=self.ihsan_threshold,
        )
        await self.register_agent(basic_agent)

    async def register_agent(
        self,
        agent: AutonomousAgent,
    ) -> bool:
        """Register a new agent with the orchestrator."""
        if len(self._agents) >= self.max_agents:
            logger.warning(f"Cannot register agent {agent.id}: max agents reached")
            return False

        self._agents[agent.id] = agent
        self._agent_health[agent.id] = AgentHealth(
            agent_id=agent.id,
            last_heartbeat=datetime.now(timezone.utc),
        )

        # Start agent background task
        self._agent_tasks[agent.id] = asyncio.create_task(
            self._run_agent(agent)
        )

        logger.info(f"Registered agent: {agent.name} ({agent.id})")
        return True

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id not in self._agents:
            return False

        # Cancel agent task
        if agent_id in self._agent_tasks:
            self._agent_tasks[agent_id].cancel()
            del self._agent_tasks[agent_id]

        # Reassign pending tasks
        for task_id, assigned_agent in list(self._task_assignments.items()):
            if assigned_agent == agent_id:
                # Return task to global queue
                task = self._find_task(task_id)
                if task and task.status == TaskStatus.IN_PROGRESS:
                    task.status = TaskStatus.PENDING
                    self._global_task_queue.append(task)
                del self._task_assignments[task_id]

        del self._agents[agent_id]
        del self._agent_health[agent_id]

        logger.info(f"Unregistered agent: {agent_id}")
        return True

    async def _run_agent(self, agent: AutonomousAgent) -> None:
        """Background task to run an agent."""
        try:
            while self.state == OrchestratorState.RUNNING:
                # Heartbeat
                self._agent_health[agent.id].last_heartbeat = datetime.now(timezone.utc)

                # Check for assigned tasks
                if agent.state == AgentState.IDLE:
                    task = await self._get_task_for_agent(agent.id)
                    if task:
                        success = await agent.run_task(task)
                        self._record_task_completion(agent.id, task, success)

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Agent {agent.id} crashed: {e}")
            self._agent_health[agent.id].is_alive = False
            self._agent_health[agent.id].error_count += 1

    def _find_task(self, task_id: str) -> Optional[AgentTask]:
        """Find a task by ID."""
        for task in self._global_task_queue:
            if task.id == task_id:
                return task
        return self._completed_tasks.get(task_id)

    async def _get_task_for_agent(self, agent_id: str) -> Optional[AgentTask]:
        """Get next task for an agent."""
        # Find unassigned task
        for task in self._global_task_queue:
            if task.id not in self._task_assignments and task.status == TaskStatus.PENDING:
                self._task_assignments[task.id] = agent_id
                self._global_task_queue.remove(task)
                return task
        return None

    def _record_task_completion(
        self,
        agent_id: str,
        task: AgentTask,
        success: bool,
    ) -> None:
        """Record task completion."""
        self._completed_tasks[task.id] = task

        if task.id in self._task_assignments:
            del self._task_assignments[task.id]

        health = self._agent_health.get(agent_id)
        if health:
            health.tasks_completed += 1
            if success:
                health.consecutive_failures = 0
            else:
                health.consecutive_failures += 1

        # Store in memory if available
        if self.memory:
            asyncio.create_task(self._store_task_memory(task, agent_id, success))

    async def _store_task_memory(
        self,
        task: AgentTask,
        agent_id: str,
        success: bool,
    ) -> None:
        """Store task execution in living memory."""
        content = f"Task '{task.name}' executed by agent {agent_id}. "
        content += f"Status: {'SUCCESS' if success else 'FAILED'}. "
        if task.error:
            content += f"Error: {task.error}"

        await self.memory.encode(
            content=content,
            memory_type=MemoryType.EPISODIC,
            source=f"orchestrator:{agent_id}",
            importance=0.7 if success else 0.9,
        )

    async def submit_task(
        self,
        name: str,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        deadline: Optional[datetime] = None,
    ) -> AgentTask:
        """Submit a new task to the orchestrator."""
        task = AgentTask(
            name=name,
            description=description,
            priority=priority,
            deadline=deadline,
        )

        self._global_task_queue.append(task)
        self._global_task_queue.sort(key=lambda t: (
            -list(TaskPriority).index(t.priority),
            t.created_at,
        ))

        logger.debug(f"Task submitted: {name} (priority={priority.value})")
        return task

    async def _health_check(self) -> List[str]:
        """Check health of all agents, return unhealthy agent IDs."""
        unhealthy = []
        now = datetime.now(timezone.utc)

        for agent_id, health in self._agent_health.items():
            # Check for dead agents
            if not health.is_alive:
                unhealthy.append(agent_id)
                continue

            # Check for stale heartbeat (30 seconds)
            if health.last_heartbeat:
                since_heartbeat = (now - health.last_heartbeat).total_seconds()
                if since_heartbeat > 30:
                    health.is_alive = False
                    unhealthy.append(agent_id)
                    continue

            # Check for repeated failures
            if health.consecutive_failures >= 3:
                unhealthy.append(agent_id)

        return unhealthy

    async def _heal_agents(self, unhealthy_ids: List[str]) -> int:
        """Attempt to heal unhealthy agents."""
        async with self._healing_lock:
            healed = 0

            for agent_id in unhealthy_ids:
                agent = self._agents.get(agent_id)
                if not agent:
                    continue

                logger.info(f"Healing agent: {agent_id}")

                # Try to restart
                try:
                    # Cancel existing task
                    if agent_id in self._agent_tasks:
                        self._agent_tasks[agent_id].cancel()

                    # Reset agent state
                    agent.state = AgentState.IDLE

                    # Restart
                    self._agent_tasks[agent_id] = asyncio.create_task(
                        self._run_agent(agent)
                    )

                    # Reset health
                    self._agent_health[agent_id] = AgentHealth(
                        agent_id=agent_id,
                        last_heartbeat=datetime.now(timezone.utc),
                    )

                    healed += 1
                    self._healing_count += 1

                except Exception as e:
                    logger.error(f"Failed to heal agent {agent_id}: {e}")
                    await self.unregister_agent(agent_id)

            return healed

    async def run_maintenance(self) -> Dict[str, int]:
        """Run maintenance cycle."""
        stats = {"checked": 0, "unhealthy": 0, "healed": 0}

        if self.state != OrchestratorState.RUNNING:
            return stats

        # Health check
        unhealthy = await self._health_check()
        stats["checked"] = len(self._agents)
        stats["unhealthy"] = len(unhealthy)

        # Heal if needed
        if unhealthy:
            self.state = OrchestratorState.HEALING
            stats["healed"] = await self._heal_agents(unhealthy)
            self.state = OrchestratorState.RUNNING

        return stats

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self.state = OrchestratorState.SHUTTING_DOWN

        # Cancel all agent tasks
        for task in self._agent_tasks.values():
            task.cancel()

        # Wait for cancellation
        if self._agent_tasks:
            await asyncio.gather(*self._agent_tasks.values(), return_exceptions=True)

        self._agents.clear()
        self._agent_tasks.clear()
        self._agent_health.clear()

        self.state = OrchestratorState.STOPPED
        logger.info("Agent Orchestrator shut down")

    def get_stats(self) -> OrchestratorStats:
        """Get orchestrator statistics."""
        stats = OrchestratorStats()

        stats.total_agents = len(self._agents)
        stats.active_agents = sum(
            1 for h in self._agent_health.values() if h.is_alive
        )
        stats.pending_tasks = len(self._global_task_queue)
        stats.running_tasks = len(self._task_assignments)
        stats.completed_tasks = sum(
            1 for t in self._completed_tasks.values()
            if t.status == TaskStatus.COMPLETED
        )
        stats.failed_tasks = sum(
            1 for t in self._completed_tasks.values()
            if t.status == TaskStatus.FAILED
        )
        stats.healing_events = self._healing_count

        if self._start_time:
            stats.uptime_seconds = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()

        return stats

    def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents."""
        return [
            {
                **agent.get_status(),
                "health": self._agent_health.get(agent_id, AgentHealth(agent_id)).is_alive,
            }
            for agent_id, agent in self._agents.items()
        ]
