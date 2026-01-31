"""
BIZRA Integration Bridge

Unified coordination layer that connects all core BIZRA modules
into a coherent, production-ready system.

Integration Points:
1. A2A <-> Transport: Task result delivery via transport layer
2. A2A <-> Federation: Capability-based pattern discovery
3. Federation <-> Inference: Pattern execution feedback
4. PCI <-> All: Unified verification gates

Created: 2026-01-30
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Awaitable

from .constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    DEFAULT_FEDERATION_BIND,
    A2A_PORT_OFFSET,
)

# Core module imports
from core.pci import generate_keypair, PCIEnvelope
from core.pci.gates import PCIGateKeeper
from core.federation.node import FederationNode
from core.federation.propagation import ElevatedPattern
from core.a2a.engine import A2AEngine
from core.a2a.schema import AgentCard, Capability, CapabilityType, TaskCard, MessageType
from core.a2a.transport import A2ATransport, HybridTransport

logger = logging.getLogger("INTEGRATION")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BridgeConfig:
    """Configuration for the IntegrationBridge."""

    # Identity
    node_id: str = ""
    agent_name: str = "BIZRA Agent"
    agent_description: str = "Integrated BIZRA node"

    # Network
    federation_bind: str = DEFAULT_FEDERATION_BIND
    a2a_bind: str = ""  # Auto-calculated if empty
    seed_nodes: List[str] = field(default_factory=list)

    # Quality thresholds
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD

    # Features
    enable_federation: bool = True
    enable_a2a: bool = True
    enable_inference: bool = True

    # Capabilities
    capabilities: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.a2a_bind and self.federation_bind:
            # Auto-calculate A2A bind address
            host, port = self.federation_bind.rsplit(":", 1)
            self.a2a_bind = f"{host}:{int(port) + A2A_PORT_OFFSET}"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

class IntegrationBridge:
    """
    Central integration point for all BIZRA core modules.

    Responsibilities:
    - Coordinate lifecycle of all subsystems
    - Route messages between A2A and Federation
    - Feed inference results to consensus
    - Provide unified verification via PCI gates

    Architecture:

        ┌─────────────┐     ┌─────────────┐
        │   A2A       │ <── │ Transport   │ ──> Network
        │   Engine    │     │   Layer     │
        └──────┬──────┘     └─────────────┘
               │
               ▼
        ┌─────────────────────────────────┐
        │       INTEGRATION BRIDGE        │
        │   - Capability Discovery        │
        │   - Result Delivery             │
        │   - Pattern Matching            │
        │   - Ihsan Verification          │
        └──────┬───────────────────┬──────┘
               │                   │
               ▼                   ▼
        ┌─────────────┐     ┌─────────────┐
        │ Federation  │     │  Inference  │
        │    Node     │     │   Gateway   │
        └─────────────┘     └─────────────┘
    """

    def __init__(self, config: BridgeConfig):
        self.config = config

        # Generate identity
        self.private_key, self.public_key = generate_keypair()
        self.node_id = config.node_id or f"bizra_{self.public_key[:8]}"

        # PCI gatekeeper for unified verification
        self.gatekeeper = PCIGateKeeper()

        # Initialize subsystems (lazy)
        self._federation: Optional[FederationNode] = None
        self._a2a: Optional[A2AEngine] = None
        self._transport: Optional[A2ATransport] = None

        # State
        self._running = False

        # Callbacks for external integration
        self._on_task_completed: Optional[Callable[[TaskCard], Awaitable[None]]] = None
        self._on_pattern_elevated: Optional[Callable[[ElevatedPattern], Awaitable[None]]] = None

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self):
        """Start all enabled subsystems in correct order."""
        logger.info(f"Starting IntegrationBridge {self.node_id}")

        # 1. Start Federation (P2P network layer)
        if self.config.enable_federation:
            await self._start_federation()

        # 2. Start A2A with transport
        if self.config.enable_a2a:
            await self._start_a2a()

        self._running = True
        logger.info(f"IntegrationBridge {self.node_id} started")

    async def stop(self):
        """Stop all subsystems gracefully."""
        logger.info(f"Stopping IntegrationBridge {self.node_id}")
        self._running = False

        if self._transport:
            await self._transport.stop()

        if self._federation:
            await self._federation.stop()

        logger.info(f"IntegrationBridge {self.node_id} stopped")

    async def _start_federation(self):
        """Initialize and start Federation node."""
        self._federation = FederationNode(
            node_id=self.node_id,
            bind_address=self.config.federation_bind,
            public_key=self.public_key,
            private_key=self.private_key,
            ihsan_score=self.config.ihsan_threshold,
        )

        await self._federation.start(self.config.seed_nodes)
        logger.info(f"Federation started on {self.config.federation_bind}")

    async def _start_a2a(self):
        """Initialize and start A2A engine with transport."""
        # Create capabilities
        capabilities = [
            Capability(
                name=c.get("name", "unknown"),
                type=CapabilityType(c.get("type", "custom")),
                description=c.get("description", ""),
                parameters=c.get("parameters", {}),
                ihsan_floor=c.get("ihsan_floor", self.config.ihsan_threshold)
            )
            for c in self.config.capabilities
        ]

        # Create agent card
        card = AgentCard(
            agent_id=self.node_id,
            name=self.config.agent_name,
            description=self.config.agent_description,
            public_key=self.public_key,
            capabilities=capabilities,
            ihsan_score=self.config.ihsan_threshold,
        )

        # Create engine with task handler
        self._a2a = A2AEngine(
            agent_card=card,
            private_key=self.private_key,
            on_task_received=self._handle_task,
        )

        # Create transport with message handler
        self._transport = HybridTransport(
            agent_id=self.node_id,
            bind_address=self.config.a2a_bind,
            on_message=self._a2a.handle_message,
        )

        # Wire transport to engine for result delivery
        self._a2a._transport = self._transport

        await self._transport.start()
        logger.info(f"A2A started on {self.config.a2a_bind}")

    # ─────────────────────────────────────────────────────────────────────────
    # TASK HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_task(self, task: TaskCard) -> Any:
        """
        Handle incoming A2A task.

        This is where Federation patterns can inform task execution.
        """
        logger.info(f"Handling task {task.task_id}: {task.capability_required}")

        # Check for applicable federation patterns
        if self._federation:
            patterns = self._federation.get_applicable_patterns({
                "capability": task.capability_required,
                "prompt": task.prompt,
            })
            if patterns:
                logger.info(f"Found {len(patterns)} applicable patterns")
                # Patterns can inform execution strategy
                task.parameters["_patterns"] = [p.trigger for p in patterns]

        # Execute via callback or return placeholder
        if self._on_task_completed:
            return await self._on_task_completed(task)

        return {"status": "completed", "message": "Task received by bridge"}

    # ─────────────────────────────────────────────────────────────────────────
    # CROSS-MODULE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    async def delegate_task(
        self,
        capability: str,
        prompt: str,
        parameters: Optional[Dict] = None,
        target_agent: Optional[str] = None,
    ) -> Optional[TaskCard]:
        """
        Delegate a task to another agent.

        If target_agent is not specified, finds the best agent
        for the capability.
        """
        if not self._a2a:
            logger.warning("A2A not enabled, cannot delegate task")
            return None

        # Find best agent if not specified
        if not target_agent:
            agent = self._a2a.find_best_agent(capability)
            if not agent:
                logger.warning(f"No agent found for capability: {capability}")
                return None
            target_agent = agent.agent_id

        # Create task
        task = self._a2a.create_task(
            capability=capability,
            prompt=prompt,
            parameters=parameters,
            target_agent=target_agent,
        )

        # Send via transport
        if self._transport:
            msg = self._a2a.create_task_message(task, target_agent)
            success = await self._transport.send(msg, target_agent)
            if success:
                logger.info(f"Delegated task {task.task_id} to {target_agent}")
            else:
                logger.warning(f"Failed to send task to {target_agent}")

        return task

    def record_pattern(self, trigger: str, success: bool, snr_delta: float):
        """
        Record a pattern use for potential elevation.

        Integrates local pattern tracking with federation propagation.
        """
        if self._federation:
            self._federation.record_pattern_use(trigger, success, snr_delta)

    def verify_envelope(self, envelope: PCIEnvelope) -> bool:
        """
        Verify a PCI envelope using unified gates.
        """
        result = self.gatekeeper.verify(envelope)
        if not result.passed:
            logger.warning(f"Envelope verification failed: {result.reject_code.name}")
        return result.passed

    # ─────────────────────────────────────────────────────────────────────────
    # DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    async def discover_agents(self):
        """
        Broadcast discovery to find other agents.
        """
        if not self._a2a or not self._transport:
            return

        msg = self._a2a.create_discover_message()
        count = await self._transport.broadcast(msg)
        logger.info(f"Discovery broadcast to {count} agents")

    async def announce_self(self):
        """
        Announce this agent's capabilities to the network.
        """
        if not self._a2a or not self._transport:
            return

        msg = self._a2a.create_announce_message()
        count = await self._transport.broadcast(msg)
        logger.info(f"Announced to {count} agents")

    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get comprehensive statistics from all subsystems."""
        stats = {
            "node_id": self.node_id,
            "running": self._running,
            "ihsan_threshold": self.config.ihsan_threshold,
        }

        if self._federation:
            stats["federation"] = self._federation.get_stats()

        if self._a2a:
            stats["a2a"] = self._a2a.get_stats()

        return stats

    def get_health(self) -> Dict:
        """Quick health check across all subsystems."""
        health = {
            "status": "healthy" if self._running else "stopped",
            "node_id": self.node_id,
        }

        if self._federation:
            health["federation"] = self._federation.get_health()

        if self._a2a:
            health["a2a"] = {
                "registered_agents": len(self._a2a.registry),
                "pending_tasks": len(self._a2a.pending_tasks),
            }

        return health


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_integrated_system(
    node_id: Optional[str] = None,
    name: str = "BIZRA Agent",
    description: str = "Integrated BIZRA node",
    capabilities: Optional[List[Dict]] = None,
    federation_bind: str = DEFAULT_FEDERATION_BIND,
    seed_nodes: Optional[List[str]] = None,
    enable_federation: bool = True,
    enable_a2a: bool = True,
) -> IntegrationBridge:
    """
    Factory function to create a fully configured IntegrationBridge.

    Example:
        bridge = create_integrated_system(
            name="MyAgent",
            capabilities=[
                {"name": "summarize", "type": "inference"},
                {"name": "search", "type": "retrieval"},
            ],
            seed_nodes=["192.168.1.100:7654"],
        )
        await bridge.start()
    """
    config = BridgeConfig(
        node_id=node_id or "",
        agent_name=name,
        agent_description=description,
        federation_bind=federation_bind,
        seed_nodes=seed_nodes or [],
        capabilities=capabilities or [],
        enable_federation=enable_federation,
        enable_a2a=enable_a2a,
    )

    return IntegrationBridge(config)
