"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗                              ║
║   ██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝                              ║
║   ██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗                                ║
║   ██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝                                ║
║   ██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗                              ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝                              ║
║                                                                              ║
║                    SOVEREIGN INTEGRATION BRIDGE v1.0                         ║
║         Unified Interface to Federation, Inference, Memory, A2A              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   This bridge connects the Sovereign Engine to:                              ║
║   • Federation Layer (P2P gossip, consensus)                                 ║
║   • Inference Gateway (LLM backends)                                         ║
║   • Memory Systems (Vault, Mem0)                                             ║
║   • A2A Protocol (Agent-to-Agent messaging)                                  ║
║   • PCI Protocol (Proof-Carrying Inference)                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger("sovereign.bridge")

T = TypeVar("T")


# =============================================================================
# ENUMS & TYPES
# =============================================================================


class SubsystemStatus(Enum):
    """Status of a connected subsystem."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DEGRADED = auto()
    ERROR = auto()


class MessagePriority(Enum):
    """Priority levels for inter-system messages."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class InferenceTier(Enum):
    """Inference backend tiers."""

    EDGE = "edge"  # Always-on, low-power (0.5B-1.5B)
    LOCAL = "local"  # On-demand, high-power (7B)
    POOL = "pool"  # Federated compute (70B+)
    CLOUD = "cloud"  # External API (Claude, GPT)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SubsystemHealth:
    """Health metrics for a subsystem."""

    name: str
    status: SubsystemStatus = SubsystemStatus.DISCONNECTED
    latency_ms: float = 0.0
    error_count: int = 0
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        return self.status in (SubsystemStatus.CONNECTED, SubsystemStatus.DEGRADED)


@dataclass
class BridgeMessage:
    """Message passed between subsystems."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = ""
    destination: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 300
    requires_ack: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BridgeMessage":
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            source=data.get("source", ""),
            destination=data.get("destination", ""),
            payload=data.get("payload", {}),
            priority=MessagePriority(data.get("priority", 5)),
            ttl_seconds=data.get("ttl_seconds", 300),
        )


@dataclass
class InferenceRequest:
    """Request to the inference gateway."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    system_prompt: str = ""
    context: List[Dict[str, str]] = field(default_factory=list)

    # Model selection
    preferred_tier: InferenceTier = InferenceTier.LOCAL
    model_hint: str = ""  # e.g., "qwen2.5:7b", "claude-3"

    # Parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # Quality requirements
    min_snr: float = 0.85
    require_proof: bool = False  # PCI envelope

    # Timeout
    timeout_ms: int = 30000


@dataclass
class InferenceResponse:
    """Response from the inference gateway."""

    request_id: str = ""
    success: bool = False
    content: str = ""

    # Metadata
    model_used: str = ""
    tier_used: InferenceTier = InferenceTier.LOCAL
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0

    # Quality
    snr_score: float = 0.0
    proof_envelope: Optional[Dict[str, Any]] = None

    # Error
    error: Optional[str] = None


# =============================================================================
# ABSTRACT SUBSYSTEM INTERFACE
# =============================================================================


class SubsystemConnector(ABC):
    """Abstract interface for subsystem connectors."""

    def __init__(self, name: str):
        self.name = name
        self.health = SubsystemHealth(name=name)
        self._connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the subsystem."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the subsystem."""
        pass

    @abstractmethod
    async def health_check(self) -> SubsystemHealth:
        """Check subsystem health."""
        pass

    async def send(self, message: BridgeMessage) -> bool:
        """Send a message to the subsystem."""
        return False

    async def receive(self, timeout_ms: int = 5000) -> Optional[BridgeMessage]:
        """Receive a message from the subsystem."""
        return None


# =============================================================================
# INFERENCE GATEWAY CONNECTOR
# =============================================================================


class InferenceConnector(SubsystemConnector):
    """
    Connects to inference backends.

    Supports:
    - Ollama (localhost:11434)
    - LM Studio (192.168.56.1:1234)
    - LiteLLM proxy
    - Direct API (Claude, OpenAI)
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        lmstudio_url: str = "http://192.168.56.1:1234",
    ):
        super().__init__("inference")
        self.ollama_url = ollama_url
        self.lmstudio_url = lmstudio_url
        self._available_models: List[str] = []
        self._tier_backends: Dict[InferenceTier, str] = {}

    async def connect(self) -> bool:
        """Connect and discover available models."""
        self.health.status = SubsystemStatus.CONNECTING

        # Try Ollama
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._available_models = [
                            m["name"] for m in data.get("models", [])
                        ]
                        self._tier_backends[InferenceTier.LOCAL] = "ollama"
                        logger.info(
                            f"Ollama connected: {len(self._available_models)} models"
                        )
        except Exception as e:
            logger.debug(f"Ollama unavailable: {e}")

        # Try LM Studio
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.lmstudio_url}/v1/models",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        self._tier_backends[InferenceTier.LOCAL] = "lmstudio"
                        logger.info("LM Studio connected")
        except Exception as e:
            logger.debug(f"LM Studio unavailable: {e}")

        if self._tier_backends:
            self.health.status = SubsystemStatus.CONNECTED
            self._connected = True
            return True

        # Fall back to stub mode
        self.health.status = SubsystemStatus.DEGRADED
        self._connected = True
        logger.warning("Inference running in stub mode")
        return True

    async def disconnect(self) -> None:
        self._connected = False
        self.health.status = SubsystemStatus.DISCONNECTED

    async def health_check(self) -> SubsystemHealth:
        if not self._connected:
            self.health.status = SubsystemStatus.DISCONNECTED
        return self.health

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute inference request."""
        start_time = time.perf_counter()
        response = InferenceResponse(request_id=request.id)

        try:
            # Route to appropriate backend
            backend = self._tier_backends.get(request.preferred_tier)

            if backend == "ollama":
                response = await self._infer_ollama(request)
            elif backend == "lmstudio":
                response = await self._infer_lmstudio(request)
            else:
                # Stub response
                response.success = True
                response.content = f"[Stub] Response to: {request.prompt[:50]}..."
                response.model_used = "stub"
                response.tier_used = InferenceTier.EDGE

            response.latency_ms = (time.perf_counter() - start_time) * 1000
            self.health.last_success = datetime.now()

        except Exception as e:
            response.success = False
            response.error = str(e)
            self.health.error_count += 1
            self.health.last_error = str(e)

        return response

    async def _infer_ollama(self, request: InferenceRequest) -> InferenceResponse:
        """Inference via Ollama."""
        import aiohttp

        response = InferenceResponse(request_id=request.id)
        model = request.model_hint or (
            self._available_models[0] if self._available_models else "llama3.2:3b"
        )

        payload = {
            "model": model,
            "prompt": request.prompt,
            "system": request.system_prompt or "You are a helpful assistant.",
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
                "num_predict": request.max_tokens,
            },
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout_ms / 1000),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response.success = True
                    response.content = data.get("response", "")
                    response.model_used = model
                    response.tier_used = InferenceTier.LOCAL
                    response.tokens_in = data.get("prompt_eval_count", 0)
                    response.tokens_out = data.get("eval_count", 0)
                else:
                    response.success = False
                    response.error = f"Ollama error: {resp.status}"

        return response

    async def _infer_lmstudio(self, request: InferenceRequest) -> InferenceResponse:
        """Inference via LM Studio (OpenAI-compatible)."""
        import aiohttp

        response = InferenceResponse(request_id=request.id)

        messages = [
            {
                "role": "system",
                "content": request.system_prompt or "You are a helpful assistant.",
            },
            {"role": "user", "content": request.prompt},
        ]

        payload = {
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.lmstudio_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=request.timeout_ms / 1000),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    response.success = True
                    response.content = data["choices"][0]["message"]["content"]
                    response.model_used = data.get("model", "lmstudio")
                    response.tier_used = InferenceTier.LOCAL
                    usage = data.get("usage", {})
                    response.tokens_in = usage.get("prompt_tokens", 0)
                    response.tokens_out = usage.get("completion_tokens", 0)
                else:
                    response.success = False
                    response.error = f"LM Studio error: {resp.status}"

        return response


# =============================================================================
# FEDERATION CONNECTOR
# =============================================================================


class FederationConnector(SubsystemConnector):
    """
    Connects to the P2P federation layer.

    Provides:
    - Node discovery via gossip
    - Byzantine consensus participation
    - Pattern propagation
    """

    def __init__(self, node_id: str = ""):
        super().__init__("federation")
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self._peers: Set[str] = set()
        self._gossip_enabled = False

    async def connect(self) -> bool:
        """Initialize federation connection."""
        self.health.status = SubsystemStatus.CONNECTING

        try:
            # Try to import and connect to federation
            from core.federation import FederationNode

            # Note: Full implementation would initialize the node here
            self._gossip_enabled = True
            self.health.status = SubsystemStatus.CONNECTED
            logger.info(f"Federation connected: node={self.node_id}")
            return True
        except ImportError:
            # Run in standalone mode
            self.health.status = SubsystemStatus.DEGRADED
            logger.warning("Federation unavailable, running standalone")
            return True

    async def disconnect(self) -> None:
        self._gossip_enabled = False
        self._peers.clear()
        self.health.status = SubsystemStatus.DISCONNECTED

    async def health_check(self) -> SubsystemHealth:
        self.health.metadata["peers"] = len(self._peers)
        self.health.metadata["gossip"] = self._gossip_enabled
        return self.health

    async def broadcast(self, message: BridgeMessage) -> int:
        """Broadcast message to all peers. Returns number reached."""
        if not self._gossip_enabled:
            return 0
        # Stub: would use gossip protocol
        return len(self._peers)

    async def request_consensus(
        self, proposal: Dict[str, Any], timeout_ms: int = 10000
    ) -> Dict[str, Any]:
        """Request consensus on a proposal."""
        if not self._gossip_enabled:
            return {"approved": True, "votes": {"approve": 1}, "standalone": True}

        # Stub: would use Byzantine consensus
        return {
            "approved": True,
            "votes": {"approve": len(self._peers) + 1},
            "round": 1,
        }


# =============================================================================
# MEMORY CONNECTOR
# =============================================================================


class MemoryConnector(SubsystemConnector):
    """
    Connects to memory and persistence systems.

    Provides:
    - Vault (encrypted secrets)
    - Session memory
    - Long-term knowledge store
    """

    def __init__(self, state_dir: Path = Path("./sovereign_state")):
        super().__init__("memory")
        self.state_dir = state_dir
        self._session_memory: Dict[str, Any] = {}
        self._vault_available = False

    async def connect(self) -> bool:
        self.health.status = SubsystemStatus.CONNECTING

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Try to connect to vault
        try:
            from core.vault import Vault

            self._vault_available = True
            logger.info("Vault connected")
        except ImportError:
            logger.debug("Vault unavailable")

        self.health.status = SubsystemStatus.CONNECTED
        return True

    async def disconnect(self) -> None:
        await self.flush()
        self.health.status = SubsystemStatus.DISCONNECTED

    async def health_check(self) -> SubsystemHealth:
        self.health.metadata["session_keys"] = len(self._session_memory)
        self.health.metadata["vault"] = self._vault_available
        return self.health

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from session memory."""
        return self._session_memory.get(key, default)

    async def set(self, key: str, value: Any, persist: bool = False) -> None:
        """Set value in session memory."""
        self._session_memory[key] = value

        if persist:
            await self._persist(key, value)

    async def delete(self, key: str) -> bool:
        """Delete value from session memory."""
        if key in self._session_memory:
            del self._session_memory[key]
            return True
        return False

    async def flush(self) -> None:
        """Persist all session memory to disk."""
        state_file = self.state_dir / "session_memory.json"
        try:
            state_file.write_text(json.dumps(self._session_memory, default=str))
        except Exception as e:
            logger.error(f"Memory flush failed: {e}")

    async def load(self) -> None:
        """Load session memory from disk."""
        state_file = self.state_dir / "session_memory.json"
        if state_file.exists():
            try:
                self._session_memory = json.loads(state_file.read_text())
            except Exception as e:
                logger.error(f"Memory load failed: {e}")

    async def _persist(self, key: str, value: Any) -> None:
        """Persist single key to disk."""
        key_file = (
            self.state_dir / f"mem_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
        )
        try:
            key_file.write_text(json.dumps({"key": key, "value": value}, default=str))
        except Exception as e:
            logger.error(f"Persist failed for {key}: {e}")


# =============================================================================
# A2A CONNECTOR
# =============================================================================


class A2AConnector(SubsystemConnector):
    """
    Connects to the Agent-to-Agent protocol.

    Provides:
    - Agent discovery
    - Task delegation
    - Capability negotiation
    """

    def __init__(self, agent_id: str = ""):
        super().__init__("a2a")
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self._registered_agents: Dict[str, Dict[str, Any]] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()

    async def connect(self) -> bool:
        self.health.status = SubsystemStatus.CONNECTING

        try:
            from core.a2a import A2AEngine

            self.health.status = SubsystemStatus.CONNECTED
            logger.info(f"A2A connected: agent={self.agent_id}")
        except ImportError:
            self.health.status = SubsystemStatus.DEGRADED
            logger.warning("A2A unavailable, agent messaging disabled")

        return True

    async def disconnect(self) -> None:
        self._registered_agents.clear()
        self.health.status = SubsystemStatus.DISCONNECTED

    async def health_check(self) -> SubsystemHealth:
        self.health.metadata["agents"] = len(self._registered_agents)
        self.health.metadata["queue_size"] = self._message_queue.qsize()
        return self.health

    async def register_capability(
        self,
        capability: str,
        handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
    ) -> None:
        """Register a capability that this agent provides."""
        # Stub: would register with A2A discovery
        logger.info(f"Registered capability: {capability}")

    async def discover_agents(self, capability: str = "") -> List[Dict[str, Any]]:
        """Discover agents with a specific capability."""
        return list(self._registered_agents.values())

    async def delegate_task(
        self, target_agent: str, task: Dict[str, Any], timeout_ms: int = 30000
    ) -> Dict[str, Any]:
        """Delegate a task to another agent."""
        # Stub: would send via A2A protocol
        return {
            "delegated": True,
            "target": target_agent,
            "task_id": uuid.uuid4().hex[:8],
        }


# =============================================================================
# UNIFIED BRIDGE
# =============================================================================


class SovereignBridge:
    """
    Unified bridge connecting all BIZRA subsystems.

    Usage:
        bridge = SovereignBridge()
        await bridge.connect_all()

        # Inference
        response = await bridge.infer("What is sovereignty?")

        # Memory
        await bridge.memory.set("key", "value", persist=True)

        # Federation
        await bridge.federation.broadcast(message)
    """

    def __init__(
        self,
        node_id: str = "",
        state_dir: Path = Path("./sovereign_state"),
    ):
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.state_dir = state_dir

        # Initialize connectors
        self.inference = InferenceConnector()
        self.federation = FederationConnector(self.node_id)
        self.memory = MemoryConnector(state_dir)
        self.a2a = A2AConnector()

        self._connected = False

    async def connect_all(self) -> Dict[str, bool]:
        """Connect all subsystems."""
        results = {}

        results["inference"] = await self.inference.connect()
        results["federation"] = await self.federation.connect()
        results["memory"] = await self.memory.connect()
        results["a2a"] = await self.a2a.connect()

        # Load persisted memory
        await self.memory.load()

        self._connected = all(results.values())
        logger.info(f"Bridge connected: {results}")

        return results

    async def disconnect_all(self) -> None:
        """Disconnect all subsystems."""
        await self.inference.disconnect()
        await self.federation.disconnect()
        await self.memory.disconnect()
        await self.a2a.disconnect()
        self._connected = False

    async def health_check(self) -> Dict[str, SubsystemHealth]:
        """Get health of all subsystems."""
        return {
            "inference": await self.inference.health_check(),
            "federation": await self.federation.health_check(),
            "memory": await self.memory.health_check(),
            "a2a": await self.a2a.health_check(),
        }

    def status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "node_id": self.node_id,
            "connected": self._connected,
            "subsystems": {
                "inference": self.inference.health.status.name,
                "federation": self.federation.health.status.name,
                "memory": self.memory.health.status.name,
                "a2a": self.a2a.health.status.name,
            },
        }

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    async def infer(
        self,
        prompt: str,
        system_prompt: str = "",
        tier: InferenceTier = InferenceTier.LOCAL,
        **kwargs,
    ) -> InferenceResponse:
        """
        Shortcut for inference.

        Usage:
            response = await bridge.infer("What is 2+2?")
            print(response.content)
        """
        request = InferenceRequest(
            prompt=prompt, system_prompt=system_prompt, preferred_tier=tier, **kwargs
        )
        return await self.inference.infer(request)

    async def remember(self, key: str, value: Any, persist: bool = True) -> None:
        """Store a value in memory."""
        await self.memory.set(key, value, persist=persist)

    async def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from memory."""
        return await self.memory.get(key, default)

    async def broadcast(
        self,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> int:
        """Broadcast to federation peers."""
        message = BridgeMessage(
            source=self.node_id,
            destination="*",
            payload=payload,
            priority=priority,
        )
        return await self.federation.broadcast(message)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


async def create_bridge(
    node_id: str = "",
    state_dir: str = "./sovereign_state",
    auto_connect: bool = True,
) -> SovereignBridge:
    """
    Create and optionally connect a SovereignBridge.

    Usage:
        bridge = await create_bridge()
        response = await bridge.infer("Hello!")
    """
    bridge = SovereignBridge(
        node_id=node_id,
        state_dir=Path(state_dir),
    )

    if auto_connect:
        await bridge.connect_all()

    return bridge


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main bridge
    "SovereignBridge",
    "create_bridge",
    # Connectors
    "InferenceConnector",
    "FederationConnector",
    "MemoryConnector",
    "A2AConnector",
    # Data classes
    "SubsystemHealth",
    "BridgeMessage",
    "InferenceRequest",
    "InferenceResponse",
    # Enums
    "SubsystemStatus",
    "MessagePriority",
    "InferenceTier",
]
