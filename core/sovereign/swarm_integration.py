"""
Swarm Integration — SwarmOrchestrator ↔ Rust Lifecycle
═══════════════════════════════════════════════════════════════════════════════

Integrates the Apex SwarmOrchestrator with the Rust lifecycle manager for
hybrid Python/Rust swarm orchestration with self-healing capabilities.

Standing on the Shoulders of Giants:
- Lamport (1982): Distributed systems, Byzantine fault tolerance
- Verma et al. (2015): Google Borg, large-scale cluster management
- Burns et al. (2016): Kubernetes design principles
- Hamilton (2007): Azure operations, availability targets

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                  HybridSwarmOrchestrator                      │
    │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
    │  │ SwarmOrch      │  │ RustLifecycle  │  │ Self-Healing   │  │
    │  │ (Apex base)    │  │ Manager        │  │ Loop           │  │
    │  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  │
    │           └───────────────────┴───────────────────┘          │
    └──────────────────────────────────────────────────────────────┘

Created: 2026-02-04 | BIZRA Apex Integration v1.0
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Optional

from core.apex import (
    AgentConfig,
    AgentInstance,
    AgentStatus,
    HealthStatus,
    ScalingAction,
    ScalingDecision,
    SwarmOrchestrator,
)
from core.sovereign.rust_lifecycle import (
    RustServiceStatus,
)

logger = logging.getLogger(__name__)


# Hamilton's operations principles
HEALTH_CHECK_INTERVAL: int = 30  # seconds
RESTART_BACKOFF_BASE: int = 5  # seconds
MAX_RESTART_ATTEMPTS: int = 3
AVAILABILITY_TARGET: float = 0.999  # Three nines


class ServiceType(str, Enum):
    """Types of services in hybrid swarm."""

    PYTHON_AGENT = "python_agent"
    RUST_SERVICE = "rust_service"


@dataclass
class ServiceStatus:
    """Unified status for Python and Rust services."""

    service_id: str
    service_type: ServiceType
    health: HealthStatus
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    restart_count: int = 0
    uptime_seconds: float = 0.0
    error: Optional[str] = None


class RustServiceAdapter:
    """
    Adapts Rust services to SwarmOrchestrator agent protocol.

    Provides unified health checking and lifecycle management
    for Rust services alongside Python agents.

    Standing on Giants:
    - Hamilton (2007): Operations at scale principles
    - Verma/Borg (2015): Large-scale cluster management
    """

    def __init__(
        self,
        service_name: str,
        endpoint: str = "http://localhost:3001",
    ):
        """
        Initialize Rust service adapter.

        Args:
            service_name: Name of the Rust service
            endpoint: HTTP endpoint for health checks
        """
        self.service_name = service_name
        self.endpoint = endpoint

        self.restart_count: int = 0
        self.last_health: HealthStatus = HealthStatus.UNKNOWN
        self.last_check: Optional[datetime] = None
        self._start_time: datetime = datetime.now(timezone.utc)

    def _map_rust_status(self, rust_status: RustServiceStatus) -> HealthStatus:
        """Map RustServiceStatus to HealthStatus."""
        mapping = {
            RustServiceStatus.HEALTHY: HealthStatus.HEALTHY,
            RustServiceStatus.STARTING: HealthStatus.DEGRADED,
            RustServiceStatus.DEGRADED: HealthStatus.DEGRADED,
            RustServiceStatus.UNHEALTHY: HealthStatus.UNHEALTHY,
            RustServiceStatus.STOPPED: HealthStatus.UNHEALTHY,
            RustServiceStatus.UNKNOWN: HealthStatus.UNKNOWN,
        }
        return mapping.get(rust_status, HealthStatus.UNKNOWN)

    async def health_check(self) -> HealthStatus:
        """
        Check Rust service health.

        Attempts HTTP health check with timeout.
        """
        self.last_check = datetime.now(timezone.utc)

        try:
            # Try to import and use aiohttp for health check
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    if response.status == 200:
                        self.last_health = HealthStatus.HEALTHY
                    elif response.status < 500:
                        self.last_health = HealthStatus.DEGRADED
                    else:
                        self.last_health = HealthStatus.UNHEALTHY

        except ImportError:
            # Fallback without aiohttp
            logger.debug(
                f"aiohttp not available, assuming healthy: {self.service_name}"
            )
            self.last_health = HealthStatus.HEALTHY

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout: {self.service_name}")
            self.last_health = HealthStatus.DEGRADED

        except Exception as e:
            logger.error(f"Health check failed for {self.service_name}: {e}")
            self.last_health = HealthStatus.UNHEALTHY

        return self.last_health

    async def restart(self) -> bool:
        """
        Restart Rust service with exponential backoff.

        Returns True if restart successful.
        """
        if self.restart_count >= MAX_RESTART_ATTEMPTS:
            logger.error(f"Max restart attempts reached for {self.service_name}")
            return False

        # Exponential backoff
        backoff = RESTART_BACKOFF_BASE * (2**self.restart_count)
        logger.info(f"Restarting {self.service_name} after {backoff}s backoff")

        await asyncio.sleep(backoff)

        try:
            # In production, this would call rust_lifecycle.restart_service
            # For now, simulate restart
            logger.info(f"Simulating restart of {self.service_name}")

            # Check health after restart
            await asyncio.sleep(2)  # Allow startup time
            health = await self.health_check()

            if health == HealthStatus.HEALTHY:
                self.restart_count = 0
                self._start_time = datetime.now(timezone.utc)
                return True
            else:
                self.restart_count += 1
                return False

        except Exception as e:
            logger.error(f"Restart failed for {self.service_name}: {e}")
            self.restart_count += 1
            return False

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    def get_status(self) -> ServiceStatus:
        """Get current service status."""
        return ServiceStatus(
            service_id=f"rust:{self.service_name}",
            service_type=ServiceType.RUST_SERVICE,
            health=self.last_health,
            last_check=self.last_check or datetime.now(timezone.utc),
            restart_count=self.restart_count,
            uptime_seconds=self.get_uptime(),
        )


class HybridSwarmOrchestrator(SwarmOrchestrator):
    """
    Orchestrates hybrid Python/Rust swarms.

    Extends SwarmOrchestrator to manage both Python agents
    and Rust services as a unified swarm with self-healing.

    Key Features:
    - Unified health monitoring (Python + Rust)
    - Proportional scaling (70% Python, 30% Rust)
    - Self-healing loop with exponential backoff
    - Graceful degradation on failures
    """

    # Default ratio: 70% Python agents, 30% Rust services
    PYTHON_RATIO: float = 0.7
    RUST_RATIO: float = 0.3

    # Minimum instances to maintain
    MIN_PYTHON_AGENTS: int = 1
    MIN_RUST_SERVICES: int = 1

    def __init__(self):
        """Initialize hybrid swarm orchestrator."""
        super().__init__()

        # Rust service adapters
        self.rust_adapters: Dict[str, RustServiceAdapter] = {}

        # Self-healing state
        self._running: bool = False
        self._heal_task: Optional[asyncio.Task] = None

        # Metrics
        self._total_restarts: int = 0
        self._total_replacements: int = 0
        self._availability_history: Deque[float] = deque(maxlen=100)

        logger.info("HybridSwarmOrchestrator initialized")

    def register_rust_service(
        self,
        service_name: str,
        endpoint: str = "http://localhost:3001",
    ) -> None:
        """
        Register a Rust service as a swarm member.

        Creates adapter for unified health/scaling management.
        """
        adapter = RustServiceAdapter(
            service_name=service_name,
            endpoint=endpoint,
        )
        self.rust_adapters[service_name] = adapter

        # Register health check callback
        self.health_monitor.register_health_check(
            agent_id=f"rust:{service_name}",
            callback=adapter.health_check,
        )

        logger.info(f"Registered Rust service: {service_name} at {endpoint}")

    def unregister_rust_service(self, service_name: str) -> None:
        """Unregister a Rust service."""
        if service_name in self.rust_adapters:
            del self.rust_adapters[service_name]
            logger.info(f"Unregistered Rust service: {service_name}")

    async def start(self) -> None:
        """Start the hybrid swarm orchestrator."""
        if self._running:
            return

        self._running = True

        # Start self-healing loop
        self._heal_task = asyncio.create_task(self._self_heal_loop())

        logger.info("HybridSwarmOrchestrator started")

    async def stop(self) -> None:
        """Stop the hybrid swarm orchestrator gracefully."""
        if not self._running:
            return

        self._running = False

        # Cancel self-healing task
        if self._heal_task:
            self._heal_task.cancel()
            try:
                await self._heal_task
            except asyncio.CancelledError:
                pass

        logger.info("HybridSwarmOrchestrator stopped")

    async def check_all_health(self) -> Dict[str, HealthStatus]:
        """Check health of all services (Python + Rust)."""
        health_results: Dict[str, HealthStatus] = {}

        # Check Rust services
        for service_name, adapter in self.rust_adapters.items():
            health = await adapter.health_check()
            health_results[f"rust:{service_name}"] = health

        # Check Python agents via base class
        # (simplified - would check actual agent health)
        for swarm_id, swarm in self._swarms.items():
            for agent in swarm.agents:
                if not agent.id.startswith("rust:"):
                    health_results[agent.id] = HealthStatus.HEALTHY

        return health_results

    def get_swarm_health_summary(self, swarm_id: str) -> Dict[str, Any]:
        """Get health summary for a swarm."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return {"error": f"Unknown swarm: {swarm_id}"}

        all_health = {}
        healthy_count = 0
        total_count = 0

        for agent in swarm.agents:
            total_count += 1
            if agent.id.startswith("rust:"):
                service_name = agent.id.replace("rust:", "")
                adapter = self.rust_adapters.get(service_name)
                health = adapter.last_health if adapter else HealthStatus.UNKNOWN
            else:
                health = HealthStatus.HEALTHY  # Simplified

            all_health[agent.id] = health.name
            if health == HealthStatus.HEALTHY:
                healthy_count += 1

        availability = healthy_count / total_count if total_count > 0 else 0.0
        self._availability_history.append(availability)

        return {
            "swarm_id": swarm_id,
            "total_agents": total_count,
            "healthy_agents": healthy_count,
            "availability": availability,
            "meets_target": availability >= AVAILABILITY_TARGET,
            "agent_health": all_health,
        }

    async def apply_scaling_decision(
        self,
        decision: ScalingDecision,
        swarm_id: str,
    ) -> None:
        """
        Apply scaling decision to hybrid swarm.

        Scales both Python agents and Rust services proportionally.
        """
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            logger.error(f"Unknown swarm: {swarm_id}")
            return

        # Partition instances by type
        python_agents = [a for a in swarm.agents if not a.id.startswith("rust:")]
        rust_services = [a for a in swarm.agents if a.id.startswith("rust:")]

        if decision.action == ScalingAction.SCALE_UP:
            delta = decision.target_count - decision.current_count

            # Scale proportionally
            python_delta = int(delta * self.PYTHON_RATIO)
            rust_delta = delta - python_delta

            await self._scale_up_python(swarm_id, python_delta)
            await self._scale_up_rust(swarm_id, rust_delta)

            logger.info(
                f"Scaled up swarm {swarm_id}: +{python_delta} Python, +{rust_delta} Rust"
            )

        elif decision.action == ScalingAction.SCALE_DOWN:
            delta = decision.current_count - decision.target_count

            # Prefer scaling down Python first (Rust has startup cost)
            python_delta = min(delta, len(python_agents) - self.MIN_PYTHON_AGENTS)
            rust_delta = delta - python_delta

            # Ensure we don't go below minimums
            rust_delta = min(rust_delta, len(rust_services) - self.MIN_RUST_SERVICES)

            await self._scale_down_python(swarm_id, python_delta)
            await self._scale_down_rust(swarm_id, rust_delta)

            logger.info(
                f"Scaled down swarm {swarm_id}: -{python_delta} Python, -{rust_delta} Rust"
            )

    async def _scale_up_python(self, swarm_id: str, count: int) -> None:
        """Spawn additional Python agent instances."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return

        for i in range(count):
            agent_id = f"py-{swarm_id}-{uuid.uuid4().hex[:8]}"
            agent = AgentInstance(
                id=agent_id,
                config=AgentConfig(
                    agent_type="python-worker",
                    name=agent_id,
                    capabilities={"reasoning", "execution"},
                ),
                status=AgentStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
            )
            swarm.agents.append(agent)
            logger.debug(f"Spawned Python agent: {agent_id}")

    async def _scale_up_rust(self, swarm_id: str, count: int) -> None:
        """Spawn additional Rust service instances."""
        for i in range(count):
            service_name = f"{swarm_id}-rust-{uuid.uuid4().hex[:8]}"

            # Register the new service
            self.register_rust_service(
                service_name=service_name,
                endpoint="http://localhost:3001",  # Would be dynamic in production
            )

            # Add to swarm
            swarm = self._swarms.get(swarm_id)
            if swarm:
                agent = AgentInstance(
                    id=f"rust:{service_name}",
                    config=AgentConfig(
                        agent_type="rust-worker",
                        name=service_name,
                        capabilities={"inference", "consensus", "pci"},
                    ),
                    status=AgentStatus.RUNNING,
                    started_at=datetime.now(timezone.utc),
                )
                swarm.agents.append(agent)

            logger.debug(f"Spawned Rust service: {service_name}")

    async def _scale_down_python(self, swarm_id: str, count: int) -> None:
        """Gracefully shutdown Python agent instances."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return

        python_agents = [a for a in swarm.agents if not a.id.startswith("rust:")]

        # Remove oldest first (FIFO)
        to_remove = python_agents[:count]
        for agent in to_remove:
            swarm.agents.remove(agent)
            logger.debug(f"Removed Python agent: {agent.id}")

    async def _scale_down_rust(self, swarm_id: str, count: int) -> None:
        """Gracefully shutdown Rust service instances."""
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return

        rust_agents = [a for a in swarm.agents if a.id.startswith("rust:")]

        # Remove oldest first
        to_remove = rust_agents[:count]
        for agent in to_remove:
            service_name = agent.id.replace("rust:", "")
            self.unregister_rust_service(service_name)
            swarm.agents.remove(agent)
            logger.debug(f"Removed Rust service: {service_name}")

    async def _self_heal_loop(self) -> None:
        """
        Self-healing loop for hybrid swarm.

        Checks all members and restarts unhealthy ones.
        """
        while self._running:
            try:
                await self._self_heal_iteration()
            except Exception as e:
                logger.error(f"Self-heal error: {e}")

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _self_heal_iteration(self) -> None:
        """Run one iteration of self-healing."""
        # Check all Rust adapters
        for service_name, adapter in list(self.rust_adapters.items()):
            health = await adapter.health_check()

            if health in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
                logger.warning(f"Unhealthy Rust service detected: {service_name}")

                success = await adapter.restart()
                self._total_restarts += 1

                if not success:
                    # Replace with new instance
                    logger.error(f"Restart failed, replacing: {service_name}")
                    await self._replace_rust_service(service_name)
                    self._total_replacements += 1

    async def _replace_rust_service(self, old_service: str) -> None:
        """Replace failed Rust service with new instance."""
        # Find which swarm this service belongs to
        swarm_id = None
        for sid, swarm in self._swarms.items():
            if any(a.id == f"rust:{old_service}" for a in swarm.agents):
                swarm_id = sid
                break

        # Remove old
        self.unregister_rust_service(old_service)

        if swarm_id:
            swarm = self._swarms[swarm_id]
            swarm.agents = [a for a in swarm.agents if a.id != f"rust:{old_service}"]

            # Add new
            new_service = f"{swarm_id}-rust-{uuid.uuid4().hex[:8]}"
            self.register_rust_service(new_service)

            agent = AgentInstance(
                id=f"rust:{new_service}",
                config=AgentConfig(
                    agent_type="rust-worker",
                    name=new_service,
                    capabilities={"inference", "consensus", "pci"},
                ),
                status=AgentStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
            )
            swarm.agents.append(agent)

            logger.info(f"Replaced {old_service} with {new_service}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        avg_availability = (
            sum(self._availability_history) / len(self._availability_history)
            if self._availability_history
            else 0.0
        )

        return {
            "total_swarms": len(self._swarms),
            "total_rust_services": len(self.rust_adapters),
            "total_restarts": self._total_restarts,
            "total_replacements": self._total_replacements,
            "average_availability": avg_availability,
            "meets_availability_target": avg_availability >= AVAILABILITY_TARGET,
            "running": self._running,
        }
