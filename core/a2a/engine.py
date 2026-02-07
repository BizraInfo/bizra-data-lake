"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA A2A — CORE ENGINE                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Agent registry, capability discovery, and message routing:                 ║
║   - Agent registration with PCI verification                                 ║
║   - Capability-based agent discovery                                         ║
║   - Message signing and verification                                         ║
║   - Task routing to capable agents                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from core.integration.constants import (
    UNIFIED_AGENT_TIMEOUT_MS,
    UNIFIED_IHSAN_THRESHOLD,
)
from core.pci import (
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
)

from .schema import (
    A2AMessage,
    AgentCard,
    Capability,
    CapabilityType,
    MessageType,
    TaskCard,
    TaskStatus,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (imported from single source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

IHSAN_MINIMUM = UNIFIED_IHSAN_THRESHOLD
AGENT_TIMEOUT_MS = UNIFIED_AGENT_TIMEOUT_MS


# ═══════════════════════════════════════════════════════════════════════════════
# A2A ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class A2AEngine:
    """
    Core A2A protocol engine.

    Responsibilities:
    - Maintain agent registry
    - Route messages to appropriate agents
    - Verify message signatures (PCI)
    - Enforce Ihsān requirements
    """

    def __init__(
        self,
        agent_card: AgentCard,
        private_key: str,
        on_task_received: Optional[Callable[[TaskCard], Awaitable[Any]]] = None,
        on_message_received: Optional[Callable[[A2AMessage], Awaitable[None]]] = None,
    ):
        """
        Initialize A2A engine with agent identity.

        Args:
            agent_card: This agent's identity and capabilities
            private_key: Ed25519 private key for signing
            on_task_received: Handler for incoming task requests
            on_message_received: Handler for other messages
        """
        self.agent_card = agent_card
        self.private_key = private_key

        # Callbacks
        self.on_task_received = on_task_received
        self.on_message_received = on_message_received

        # Agent registry (discovered agents)
        self.registry: Dict[str, AgentCard] = {}

        # Capability index for fast lookup
        self._capability_index: Dict[str, List[str]] = (
            {}
        )  # capability_name -> [agent_ids]

        # Pending tasks
        self.pending_tasks: Dict[str, TaskCard] = {}

        # Transport layer (set by IntegrationBridge for result delivery)
        self._transport: Optional[Any] = None

        # Message handlers by type
        self._handlers: Dict[MessageType, Callable] = {
            MessageType.DISCOVER: self._handle_discover,
            MessageType.ANNOUNCE: self._handle_announce,
            MessageType.TASK_REQUEST: self._handle_task_request,
            MessageType.TASK_ACCEPT: self._handle_task_accept,
            MessageType.TASK_STATUS: self._handle_task_status,
            MessageType.TASK_RESULT: self._handle_task_result,
            MessageType.PING: self._handle_ping,
        }

        # Running state
        self._running = False

    # ─────────────────────────────────────────────────────────────────────────
    # REGISTRY MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def register_agent(self, card: AgentCard) -> bool:
        """
        Register an agent in the local registry.

        Validates:
        - Ihsān score meets minimum
        - Public key is present

        Returns:
            True if registered successfully
        """
        # Validate Ihsān
        if card.ihsan_score < IHSAN_MINIMUM:
            print(
                f"❌ Rejected agent {card.agent_id}: Ihsān {card.ihsan_score} < {IHSAN_MINIMUM}"
            )
            return False

        # Validate identity
        if not card.public_key:
            print(f"⚠️ Agent {card.agent_id} has no public key")

        # Register
        self.registry[card.agent_id] = card

        # Update capability index
        for cap in card.capabilities:
            if cap.name not in self._capability_index:
                self._capability_index[cap.name] = []
            if card.agent_id not in self._capability_index[cap.name]:
                self._capability_index[cap.name].append(card.agent_id)

        print(
            f"✅ Registered agent: {card.agent_id} ({len(card.capabilities)} capabilities)"
        )
        return True

    def unregister_agent(self, agent_id: str):
        """Remove an agent from the registry."""
        if agent_id in self.registry:
            card = self.registry.pop(agent_id)
            # Clean capability index
            for cap in card.capabilities:
                if cap.name in self._capability_index:
                    self._capability_index[cap.name] = [
                        a for a in self._capability_index[cap.name] if a != agent_id
                    ]
            print(f"🔌 Unregistered agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[AgentCard]:
        """Get an agent by ID."""
        return self.registry.get(agent_id)

    def find_agents_by_capability(self, capability_name: str) -> List[AgentCard]:
        """Find all agents with a specific capability."""
        agent_ids = self._capability_index.get(capability_name, [])
        return [self.registry[aid] for aid in agent_ids if aid in self.registry]

    def find_best_agent(self, capability_name: str) -> Optional[AgentCard]:
        """
        Find the best agent for a capability.

        Selection criteria:
        1. Has the capability
        2. Highest Ihsān score
        3. Highest success rate
        """
        candidates = self.find_agents_by_capability(capability_name)
        if not candidates:
            return None

        # Score: Ihsān * 0.6 + success_rate * 0.4
        def score(c: AgentCard) -> float:
            return c.ihsan_score * 0.6 + c.success_rate * 0.4

        return max(candidates, key=score)

    # ─────────────────────────────────────────────────────────────────────────
    # MESSAGE CREATION
    # ─────────────────────────────────────────────────────────────────────────

    def create_message(
        self, message_type: MessageType, payload: Dict, recipient_id: str = ""
    ) -> A2AMessage:
        """
        Create a signed A2A message.

        Signs the message with this agent's private key.
        """
        msg = A2AMessage(
            message_type=message_type,
            sender_id=self.agent_card.agent_id,
            sender_public_key=self.agent_card.public_key,
            recipient_id=recipient_id,
            payload=payload,
            ihsan_score=self.agent_card.ihsan_score,
        )

        # Sign with PCI
        content = msg.signing_content()
        digest = domain_separated_digest(content)
        msg.signature = sign_message(digest, self.private_key)

        return msg

    def verify_message(self, msg: A2AMessage) -> bool:
        """
        Verify a message signature.

        Returns:
            True if signature is valid
        """
        if not msg.sender_public_key:
            return False

        content = msg.signing_content()
        digest = domain_separated_digest(content)

        return verify_signature(digest, msg.signature, msg.sender_public_key)

    # ─────────────────────────────────────────────────────────────────────────
    # MESSAGE HANDLING
    # ─────────────────────────────────────────────────────────────────────────

    async def handle_message(self, msg: A2AMessage) -> Optional[A2AMessage]:
        """
        Process an incoming A2A message.

        Validates signature and routes to appropriate handler.
        """
        # Verify signature
        if msg.signature and not self.verify_message(msg):
            print(f"⚠️ Invalid signature from {msg.sender_id}")
            return None

        # Verify Ihsān
        if msg.ihsan_score < IHSAN_MINIMUM:
            print(f"⚠️ Ihsān too low from {msg.sender_id}: {msg.ihsan_score}")
            return None

        # Route to handler
        handler = self._handlers.get(msg.message_type)
        if handler:
            return await handler(msg)

        # Default: pass to callback
        if self.on_message_received:
            await self.on_message_received(msg)

        return None

    async def _handle_discover(self, msg: A2AMessage) -> A2AMessage:
        """Handle discovery request - respond with our agent card."""
        return self.create_message(
            MessageType.ANNOUNCE,
            {"agent_card": self.agent_card.to_dict()},
            recipient_id=msg.sender_id,
        )

    async def _handle_announce(self, msg: A2AMessage) -> None:
        """Handle agent announcement - add to registry."""
        card_data = msg.payload.get("agent_card", {})
        if card_data:
            card = AgentCard.from_dict(card_data)
            self.register_agent(card)
        return None

    async def _handle_task_request(self, msg: A2AMessage) -> A2AMessage:
        """Handle incoming task request."""
        task_data = msg.payload.get("task", {})
        task = TaskCard.from_dict(task_data)

        # Check if we have the capability
        cap = self.agent_card.get_capability(task.capability_required)
        if not cap:
            return self.create_message(
                MessageType.TASK_REJECT,
                {
                    "task_id": task.task_id,
                    "reason": f"Capability not found: {task.capability_required}",
                },
                recipient_id=msg.sender_id,
            )

        # Accept the task
        task.mark_started(self.agent_card.agent_id)
        self.pending_tasks[task.task_id] = task

        # Send acceptance
        accept_msg = self.create_message(
            MessageType.TASK_ACCEPT,
            {"task_id": task.task_id, "assignee": self.agent_card.agent_id},
            recipient_id=msg.sender_id,
        )

        # Execute task asynchronously
        if self.on_task_received:
            asyncio.create_task(self._execute_task(task, msg.sender_id))

        return accept_msg

    async def _execute_task(self, task: TaskCard, requester_id: str):
        """Execute a task and send result via transport layer."""
        try:
            if self.on_task_received:
                result = await self.on_task_received(task)
                task.mark_completed(result)
            else:
                task.mark_failed("No task handler registered")
        except Exception as e:
            task.mark_failed(str(e))

        # Send result via transport layer
        result_msg = self.create_message(
            MessageType.TASK_RESULT,
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
            },
            recipient_id=requester_id,
        )

        if self._transport:
            try:
                await self._transport.send(result_msg, requester_id)
                print(f"📤 Task {task.task_id} result sent to {requester_id}")
            except Exception as e:
                print(f"⚠️ Failed to send result for {task.task_id}: {e}")
        else:
            print(
                f"📤 Task {task.task_id} completed: {task.status.value} (no transport)"
            )

    async def _handle_task_accept(self, msg: A2AMessage) -> None:
        """Handle task acceptance."""
        task_id = msg.payload.get("task_id")
        if task_id in self.pending_tasks:
            self.pending_tasks[task_id].status = TaskStatus.ACCEPTED
            self.pending_tasks[task_id].assignee_id = msg.payload.get("assignee", "")
        return None

    async def _handle_task_status(self, msg: A2AMessage) -> None:
        """Handle task status update."""
        task_id = msg.payload.get("task_id")
        status = msg.payload.get("status")
        if task_id in self.pending_tasks and status:
            self.pending_tasks[task_id].status = TaskStatus(status)
        return None

    async def _handle_task_result(self, msg: A2AMessage) -> None:
        """Handle task result."""
        task_id = msg.payload.get("task_id")
        if task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            task.result = msg.payload.get("result")
            task.status = TaskStatus(msg.payload.get("status", "completed"))
            task.error = msg.payload.get("error")
        return None

    async def _handle_ping(self, msg: A2AMessage) -> A2AMessage:
        """Handle ping - respond with pong."""
        return self.create_message(
            MessageType.PONG,
            {"timestamp": datetime.now(timezone.utc).isoformat()},
            recipient_id=msg.sender_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TASK DELEGATION
    # ─────────────────────────────────────────────────────────────────────────

    def create_task(
        self,
        capability: str,
        prompt: str,
        parameters: Optional[Dict] = None,
        target_agent: Optional[str] = None,
        priority: int = 5,
    ) -> TaskCard:
        """
        Create a new task for delegation.

        Args:
            capability: Required capability name
            prompt: Natural language task description
            parameters: Structured parameters
            target_agent: Specific agent to target (optional)
            priority: 1-10 (10 = highest)

        Returns:
            TaskCard ready for delegation
        """
        task = TaskCard(
            capability_required=capability,
            target_agent=target_agent or "",
            prompt=prompt,
            parameters=parameters or {},
            requester_id=self.agent_card.agent_id,
            priority=priority,
        )
        self.pending_tasks[task.task_id] = task
        return task

    def create_task_message(self, task: TaskCard, target_agent_id: str) -> A2AMessage:
        """Create a task request message."""
        return self.create_message(
            MessageType.TASK_REQUEST,
            {"task": task.to_dict()},
            recipient_id=target_agent_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def create_discover_message(self) -> A2AMessage:
        """Create a discovery broadcast message."""
        return self.create_message(
            MessageType.DISCOVER, {"seeking": "all_capabilities"}
        )

    def create_announce_message(self) -> A2AMessage:
        """Create an announcement message with our agent card."""
        return self.create_message(
            MessageType.ANNOUNCE, {"agent_card": self.agent_card.to_dict()}
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STATISTICS
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            "agent_id": self.agent_card.agent_id,
            "registered_agents": len(self.registry),
            "indexed_capabilities": len(self._capability_index),
            "pending_tasks": len(self.pending_tasks),
            "ihsan_score": self.agent_card.ihsan_score,
            "my_capabilities": len(self.agent_card.capabilities),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def create_a2a_engine(
    agent_id: str,
    name: str,
    description: str,
    capabilities: List[Dict],
    on_task_received: Optional[Callable[[TaskCard], Awaitable[Any]]] = None,
) -> A2AEngine:
    """
    Factory function to create a fully initialized A2A engine.

    Generates keypair and creates agent card automatically.
    """
    # Generate identity
    private_key, public_key = generate_keypair()

    # Create capabilities
    caps = [
        Capability(
            name=c["name"],
            type=CapabilityType(c.get("type", "custom")),
            description=c.get("description", ""),
            parameters=c.get("parameters", {}),
            ihsan_floor=c.get("ihsan_floor", 0.95),
        )
        for c in capabilities
    ]

    # Create agent card
    card = AgentCard(
        agent_id=agent_id,
        name=name,
        description=description,
        public_key=public_key,
        capabilities=caps,
    )

    return A2AEngine(
        agent_card=card, private_key=private_key, on_task_received=on_task_received
    )
