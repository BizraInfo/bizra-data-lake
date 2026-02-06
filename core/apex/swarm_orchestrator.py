"""
BIZRA Swarm Orchestrator — Autonomous Deployment & Scaling
═══════════════════════════════════════════════════════════════════════════════

Deployment manager with horizontal scaling, health monitoring, and
self-healing capabilities for proactive agent swarms.

Standing on the Shoulders of Giants:
- Lamport (1982): Distributed systems, Byzantine fault tolerance
- Verma et al. (2015): Google Borg, large-scale cluster management
- Burns et al. (2016): Kubernetes design principles
- Hamilton (2007): Azure operations, 3-5-9 availability
- Brewer (2000): CAP theorem, consistency-availability tradeoff

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    SwarmOrchestrator                          │
    │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
    │  │ Deployment │  │ Health     │  │ Scaling                │ │
    │  │ Manager    │  │ Monitor    │  │ Manager                │ │
    │  └──────┬─────┘  └──────┬─────┘  └──────────┬─────────────┘ │
    │         └───────────────┼───────────────────┘               │
    │                         ▼                                   │
    │              ┌──────────────────────┐                       │
    │              │   Self-Healing Loop  │                       │
    │              └──────────────────────┘                       │
    └──────────────────────────────────────────────────────────────┘

Availability Target: 99.9% (three nines)
Scaling Response: <30 seconds

Created: 2026-02-04 | BIZRA Apex System v1.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Deque
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (Standing on Giants)
# =============================================================================

# Availability target (Hamilton's Azure operations)
AVAILABILITY_TARGET = 0.999  # Three nines

# Health check interval (seconds)
HEALTH_CHECK_INTERVAL = 10

# Unhealthy threshold (consecutive failures before action)
UNHEALTHY_THRESHOLD = 3

# Scale-up latency target (Borg-inspired)
SCALE_UP_LATENCY_SECONDS = 30

# Scale-down cool-off period (prevent thrashing)
SCALE_DOWN_COOLOFF_MINUTES = 5

# Maximum agents per swarm
MAX_AGENTS_PER_SWARM = 100

# Minimum agents for fault tolerance (Byzantine: 3f+1)
MIN_AGENTS_FAULT_TOLERANT = 4

# CPU threshold for scaling
CPU_SCALE_UP_THRESHOLD = 0.80
CPU_SCALE_DOWN_THRESHOLD = 0.40

# Memory threshold for scaling
MEMORY_SCALE_UP_THRESHOLD = 0.85
MEMORY_SCALE_DOWN_THRESHOLD = 0.50


# =============================================================================
# ENUMS
# =============================================================================

class AgentStatus(str, Enum):
    """Status of deployed agents."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ScalingAction(str, Enum):
    """Types of scaling actions."""
    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REPLACE = "replace"


class HealthStatus(str, Enum):
    """Health check results."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SwarmTopology(str, Enum):
    """Swarm network topologies."""
    STAR = "star"      # Central coordinator
    MESH = "mesh"      # All-to-all
    RING = "ring"      # Circular
    HIERARCHY = "hierarchy"  # Tree structure


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for deploying an agent."""
    agent_type: str
    name: str
    capabilities: Set[str] = field(default_factory=set)

    # Resource allocation
    cpu_limit: float = 1.0
    memory_limit_mb: int = 512
    gpu_enabled: bool = False

    # Networking
    port: int = 0  # 0 = auto-assign
    public_endpoint: bool = False

    # Behavior
    autonomy_level: int = 2  # 0-5 scale
    ihsan_threshold: float = 0.95

    # Environment (secrets from env vars, not hardcoded)
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentInstance:
    """A deployed agent instance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    config: AgentConfig = field(default_factory=AgentConfig)
    status: AgentStatus = AgentStatus.PENDING

    # Process info
    process_id: Optional[int] = None
    host: str = "localhost"
    port: int = 0

    # Health tracking
    health_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    last_health_check: Optional[datetime] = None

    # Metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    tasks_completed: int = 0
    uptime_seconds: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    @property
    def is_healthy(self) -> bool:
        return self.status == AgentStatus.RUNNING and self.health_status == HealthStatus.HEALTHY

    @property
    def is_running(self) -> bool:
        return self.status in [AgentStatus.RUNNING, AgentStatus.DEGRADED]

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class SwarmConfig:
    """Configuration for an agent swarm."""
    name: str
    agent_config: AgentConfig
    topology: SwarmTopology = SwarmTopology.MESH

    # Scaling parameters
    min_agents: int = MIN_AGENTS_FAULT_TOLERANT
    max_agents: int = MAX_AGENTS_PER_SWARM
    desired_agents: int = MIN_AGENTS_FAULT_TOLERANT

    # Auto-scaling
    auto_scale_enabled: bool = True
    scale_up_threshold: float = CPU_SCALE_UP_THRESHOLD
    scale_down_threshold: float = CPU_SCALE_DOWN_THRESHOLD

    # Health
    health_check_interval: int = HEALTH_CHECK_INTERVAL
    unhealthy_threshold: int = UNHEALTHY_THRESHOLD


@dataclass
class Swarm:
    """A swarm of deployed agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: SwarmConfig = field(default_factory=lambda: SwarmConfig(name="default", agent_config=AgentConfig(agent_type="default", name="agent")))
    agents: Dict[str, AgentInstance] = field(default_factory=dict)

    # State
    is_running: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_scale_action: Optional[datetime] = None

    @property
    def healthy_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.is_healthy)

    @property
    def running_count(self) -> int:
        return sum(1 for a in self.agents.values() if a.is_running)

    @property
    def availability(self) -> float:
        if not self.agents:
            return 0.0
        return self.healthy_count / len(self.agents)


@dataclass
class ScalingDecision:
    """A scaling decision with reasoning."""
    action: ScalingAction
    target_count: int
    current_count: int
    reason: str
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class HealthReport:
    """Health report for a swarm."""
    swarm_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    overall_health: HealthStatus = HealthStatus.UNKNOWN
    availability: float = 0.0
    agent_statuses: Dict[str, HealthStatus] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthMonitor:
    """
    Monitors agent health with self-healing triggers.

    Implements Hamilton's operations principles:
    - Design for failure
    - Automate recovery
    - Monitor everything
    """

    def __init__(self, check_interval: int = HEALTH_CHECK_INTERVAL):
        self.check_interval = check_interval
        self._health_history: Dict[str, Deque[HealthStatus]] = {}
        self._check_callbacks: Dict[str, Callable] = {}

    def register_health_check(self, agent_id: str, callback: Callable):
        """Register health check callback for agent."""
        self._check_callbacks[agent_id] = callback
        self._health_history[agent_id] = deque(maxlen=10)

    async def check_health(self, agent: AgentInstance) -> HealthStatus:
        """Perform health check on agent."""
        try:
            # Call registered health check
            if agent.id in self._check_callbacks:
                is_healthy = await self._check_callbacks[agent.id]()
            else:
                # Default: check if process exists and responsive
                is_healthy = agent.is_running and agent.cpu_usage < 0.95

            if is_healthy:
                status = HealthStatus.HEALTHY
                agent.consecutive_failures = 0
            else:
                agent.consecutive_failures += 1
                if agent.consecutive_failures >= UNHEALTHY_THRESHOLD:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.DEGRADED

        except Exception as e:
            logger.warning(f"Health check failed for {agent.id}: {e}")
            agent.consecutive_failures += 1
            status = HealthStatus.UNHEALTHY

        # Record history
        if agent.id not in self._health_history:
            self._health_history[agent.id] = deque(maxlen=10)
        self._health_history[agent.id].append(status)

        agent.health_status = status
        agent.last_health_check = datetime.now(timezone.utc)

        return status

    async def check_swarm_health(self, swarm: Swarm) -> HealthReport:
        """Check health of entire swarm."""
        report = HealthReport(swarm_id=swarm.id)
        issues = []
        recommendations = []

        # Check each agent
        for agent_id, agent in swarm.agents.items():
            status = await self.check_health(agent)
            report.agent_statuses[agent_id] = status

            if status == HealthStatus.UNHEALTHY:
                issues.append(f"Agent {agent_id} is unhealthy")
                recommendations.append(f"Consider replacing agent {agent_id}")

        # Calculate overall health
        report.availability = swarm.availability

        if report.availability >= AVAILABILITY_TARGET:
            report.overall_health = HealthStatus.HEALTHY
        elif report.availability >= 0.9:
            report.overall_health = HealthStatus.DEGRADED
            issues.append(f"Availability {report.availability:.2%} below target {AVAILABILITY_TARGET:.2%}")
        else:
            report.overall_health = HealthStatus.UNHEALTHY
            issues.append(f"Critical: Availability {report.availability:.2%}")
            recommendations.append("Immediate scale-up recommended")

        # Check fault tolerance (Byzantine: 3f+1)
        if swarm.healthy_count < MIN_AGENTS_FAULT_TOLERANT:
            issues.append(f"Below fault tolerance threshold ({swarm.healthy_count}/{MIN_AGENTS_FAULT_TOLERANT})")
            recommendations.append("Scale up to maintain fault tolerance")

        report.issues = issues
        report.recommendations = recommendations

        return report


# =============================================================================
# SCALING MANAGER
# =============================================================================

class ScalingManager:
    """
    Manages auto-scaling decisions.

    Implements Borg/Kubernetes scaling principles:
    - Horizontal pod autoscaling
    - Resource-based triggers
    - Cool-off periods to prevent thrashing
    """

    def __init__(self):
        self._last_scale_up: Dict[str, datetime] = {}
        self._last_scale_down: Dict[str, datetime] = {}
        self._scaling_history: Deque[ScalingDecision] = deque(maxlen=100)

    def decide_scaling(self, swarm: Swarm) -> ScalingDecision:
        """Decide if scaling is needed."""
        config = swarm.config
        current = swarm.running_count
        healthy = swarm.healthy_count

        # Calculate average resource usage
        if swarm.agents:
            avg_cpu = sum(a.cpu_usage for a in swarm.agents.values()) / len(swarm.agents)
            avg_memory = sum(a.memory_usage for a in swarm.agents.values()) / len(swarm.agents)
        else:
            avg_cpu = 0.0
            avg_memory = 0.0

        metrics = {
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
            "current_count": current,
            "healthy_count": healthy,
            "availability": swarm.availability
        }

        # Check if we need to scale up
        if self._should_scale_up(swarm, avg_cpu, avg_memory, healthy):
            target = min(current + 1, config.max_agents)
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                target_count=target,
                current_count=current,
                reason=f"Resource pressure (CPU: {avg_cpu:.1%}, Memory: {avg_memory:.1%}) or low availability",
                metrics=metrics
            )

        # Check if we can scale down
        if self._should_scale_down(swarm, avg_cpu, avg_memory):
            target = max(current - 1, config.min_agents)
            if target < current:
                return ScalingDecision(
                    action=ScalingAction.SCALE_DOWN,
                    target_count=target,
                    current_count=current,
                    reason=f"Low resource usage (CPU: {avg_cpu:.1%}, Memory: {avg_memory:.1%})",
                    metrics=metrics
                )

        # Check if we need to replace unhealthy agents
        unhealthy_agents = [a for a in swarm.agents.values() if a.health_status == HealthStatus.UNHEALTHY]
        if unhealthy_agents:
            return ScalingDecision(
                action=ScalingAction.REPLACE,
                target_count=current,
                current_count=current,
                reason=f"Replacing {len(unhealthy_agents)} unhealthy agents",
                metrics=metrics
            )

        return ScalingDecision(
            action=ScalingAction.NONE,
            target_count=current,
            current_count=current,
            reason="No scaling needed",
            metrics=metrics
        )

    def _should_scale_up(self, swarm: Swarm, avg_cpu: float, avg_memory: float, healthy: int) -> bool:
        """Determine if scale-up is needed."""
        config = swarm.config

        # Don't scale if at max
        if swarm.running_count >= config.max_agents:
            return False

        # Check cool-off period
        last_up = self._last_scale_up.get(swarm.id)
        if last_up and (datetime.now(timezone.utc) - last_up).total_seconds() < 60:
            return False

        # Scale up conditions
        if avg_cpu >= config.scale_up_threshold:
            return True
        if avg_memory >= MEMORY_SCALE_UP_THRESHOLD:
            return True
        if healthy < config.min_agents:
            return True
        if swarm.availability < AVAILABILITY_TARGET:
            return True

        return False

    def _should_scale_down(self, swarm: Swarm, avg_cpu: float, avg_memory: float) -> bool:
        """Determine if scale-down is possible."""
        config = swarm.config

        # Don't scale below minimum
        if swarm.running_count <= config.min_agents:
            return False

        # Check cool-off period (longer for scale-down to prevent thrashing)
        last_down = self._last_scale_down.get(swarm.id)
        cooloff = timedelta(minutes=SCALE_DOWN_COOLOFF_MINUTES)
        if last_down and (datetime.now(timezone.utc) - last_down) < cooloff:
            return False

        # Only scale down if resources are low
        if avg_cpu <= config.scale_down_threshold and avg_memory <= MEMORY_SCALE_DOWN_THRESHOLD:
            return True

        return False

    def record_scale_action(self, swarm_id: str, decision: ScalingDecision):
        """Record a scaling action."""
        self._scaling_history.append(decision)

        if decision.action == ScalingAction.SCALE_UP:
            self._last_scale_up[swarm_id] = datetime.now(timezone.utc)
        elif decision.action == ScalingAction.SCALE_DOWN:
            self._last_scale_down[swarm_id] = datetime.now(timezone.utc)


# =============================================================================
# SWARM ORCHESTRATOR (UNIFIED)
# =============================================================================

class SwarmOrchestrator:
    """
    Unified swarm orchestration with deployment, health, and scaling.

    Integrates:
    - HealthMonitor for continuous health tracking
    - ScalingManager for auto-scaling decisions
    - Self-healing loop for automatic recovery

    Standing on Giants:
    - Lamport: Byzantine fault tolerance
    - Borg: Large-scale orchestration
    - Kubernetes: Declarative desired state
    """

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.scaling_manager = ScalingManager()

        self._swarms: Dict[str, Swarm] = {}
        self._agent_factory: Optional[Callable[[AgentConfig], AgentInstance]] = None

        # Background tasks
        self._health_loop_task: Optional[asyncio.Task] = None
        self._scaling_loop_task: Optional[asyncio.Task] = None

        self._is_running = False

        logger.info("SwarmOrchestrator initialized")

    def set_agent_factory(self, factory: Callable[[AgentConfig], AgentInstance]):
        """Set the factory function for creating agents."""
        self._agent_factory = factory

    # -------------------------------------------------------------------------
    # SWARM MANAGEMENT
    # -------------------------------------------------------------------------

    async def create_swarm(self, config: SwarmConfig) -> Swarm:
        """Create a new swarm with initial agents."""
        swarm = Swarm(config=config)
        self._swarms[swarm.id] = swarm

        logger.info(f"Creating swarm {swarm.id} with {config.desired_agents} agents")

        # Deploy initial agents
        for i in range(config.desired_agents):
            await self._deploy_agent(swarm)

        swarm.is_running = True
        return swarm

    async def _deploy_agent(self, swarm: Swarm) -> Optional[AgentInstance]:
        """Deploy a single agent to the swarm."""
        if not self._agent_factory:
            # Default factory - creates mock agent
            agent = AgentInstance(
                config=swarm.config.agent_config,
                status=AgentStatus.PENDING,
                port=8000 + len(swarm.agents)
            )
        else:
            agent = self._agent_factory(swarm.config.agent_config)

        swarm.agents[agent.id] = agent

        # Simulate startup
        agent.status = AgentStatus.STARTING
        await asyncio.sleep(0.1)  # Simulated startup time

        agent.status = AgentStatus.RUNNING
        agent.started_at = datetime.now(timezone.utc)
        agent.health_status = HealthStatus.HEALTHY

        logger.info(f"Deployed agent {agent.id} to swarm {swarm.id}")
        return agent

    async def _terminate_agent(self, swarm: Swarm, agent_id: str):
        """Terminate an agent."""
        agent = swarm.agents.get(agent_id)
        if not agent:
            return

        agent.status = AgentStatus.STOPPING
        await asyncio.sleep(0.05)  # Graceful shutdown

        agent.status = AgentStatus.STOPPED
        agent.stopped_at = datetime.now(timezone.utc)

        del swarm.agents[agent_id]
        logger.info(f"Terminated agent {agent_id} from swarm {swarm.id}")

    async def scale_swarm(self, swarm_id: str, target_count: int) -> bool:
        """Scale swarm to target agent count."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return False

        current = swarm.running_count
        config = swarm.config

        # Enforce limits
        target_count = max(config.min_agents, min(config.max_agents, target_count))

        if target_count == current:
            return True

        if target_count > current:
            # Scale up
            for _ in range(target_count - current):
                await self._deploy_agent(swarm)
        else:
            # Scale down - remove least healthy agents first
            agents_to_remove = sorted(
                swarm.agents.values(),
                key=lambda a: (a.is_healthy, a.tasks_completed)
            )[:current - target_count]

            for agent in agents_to_remove:
                await self._terminate_agent(swarm, agent.id)

        swarm.last_scale_action = datetime.now(timezone.utc)
        logger.info(f"Scaled swarm {swarm_id}: {current} → {target_count}")
        return True

    # -------------------------------------------------------------------------
    # SELF-HEALING LOOP
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the orchestrator background loops."""
        if self._is_running:
            return

        self._is_running = True
        self._health_loop_task = asyncio.create_task(self._health_loop())
        self._scaling_loop_task = asyncio.create_task(self._scaling_loop())

        logger.info("SwarmOrchestrator started")

    async def stop(self):
        """Stop the orchestrator."""
        self._is_running = False

        if self._health_loop_task:
            self._health_loop_task.cancel()
        if self._scaling_loop_task:
            self._scaling_loop_task.cancel()

        logger.info("SwarmOrchestrator stopped")

    async def _health_loop(self):
        """Continuous health monitoring loop."""
        while self._is_running:
            try:
                for swarm in self._swarms.values():
                    if swarm.is_running:
                        report = await self.health_monitor.check_swarm_health(swarm)

                        # Auto-heal unhealthy agents
                        for agent_id, status in report.agent_statuses.items():
                            if status == HealthStatus.UNHEALTHY:
                                logger.warning(f"Auto-healing agent {agent_id}")
                                await self._replace_agent(swarm, agent_id)

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health loop error: {e}")
                await asyncio.sleep(1)

    async def _scaling_loop(self):
        """Continuous auto-scaling loop."""
        while self._is_running:
            try:
                for swarm in self._swarms.values():
                    if swarm.is_running and swarm.config.auto_scale_enabled:
                        decision = self.scaling_manager.decide_scaling(swarm)

                        if decision.action != ScalingAction.NONE:
                            logger.info(f"Scaling decision for {swarm.id}: {decision.action.value} - {decision.reason}")

                            if decision.action == ScalingAction.SCALE_UP:
                                await self.scale_swarm(swarm.id, decision.target_count)
                            elif decision.action == ScalingAction.SCALE_DOWN:
                                await self.scale_swarm(swarm.id, decision.target_count)
                            elif decision.action == ScalingAction.REPLACE:
                                # Replace unhealthy agents
                                for agent in list(swarm.agents.values()):
                                    if agent.health_status == HealthStatus.UNHEALTHY:
                                        await self._replace_agent(swarm, agent.id)

                            self.scaling_manager.record_scale_action(swarm.id, decision)

                await asyncio.sleep(SCALE_UP_LATENCY_SECONDS)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                await asyncio.sleep(1)

    async def _replace_agent(self, swarm: Swarm, agent_id: str):
        """Replace an unhealthy agent."""
        await self._terminate_agent(swarm, agent_id)
        await self._deploy_agent(swarm)

    # -------------------------------------------------------------------------
    # STATUS & METRICS
    # -------------------------------------------------------------------------

    def get_swarm_status(self, swarm_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a swarm."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return None

        return {
            "id": swarm.id,
            "name": swarm.config.name,
            "is_running": swarm.is_running,
            "agents": {
                "total": len(swarm.agents),
                "running": swarm.running_count,
                "healthy": swarm.healthy_count,
            },
            "availability": swarm.availability,
            "topology": swarm.config.topology.value,
            "created_at": swarm.created_at.isoformat(),
            "agent_list": [
                {
                    "id": a.id,
                    "status": a.status.value,
                    "health": a.health_status.value,
                    "cpu": a.cpu_usage,
                    "memory": a.memory_usage,
                    "endpoint": a.endpoint,
                }
                for a in swarm.agents.values()
            ]
        }

    def get_all_swarms(self) -> List[Dict[str, Any]]:
        """Get status of all swarms."""
        return [
            {
                "id": s.id,
                "name": s.config.name,
                "agents": len(s.agents),
                "healthy": s.healthy_count,
                "availability": s.availability,
            }
            for s in self._swarms.values()
        ]


# =============================================================================
# FACTORY
# =============================================================================

_swarm_orchestrator: Optional[SwarmOrchestrator] = None


def get_swarm_orchestrator() -> SwarmOrchestrator:
    """Get singleton swarm orchestrator."""
    global _swarm_orchestrator
    if _swarm_orchestrator is None:
        _swarm_orchestrator = SwarmOrchestrator()
    return _swarm_orchestrator
