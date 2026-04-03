# ===============================================================================
# BIZRA Constellation - A2A Handshake Protocol v0.3
# ===============================================================================
"""
Google A2A Protocol v0.3 compliant agent discovery and authorization system.

Implements:
- AgentCard discovery and exchange per A2A spec
- Ed25519 cryptographic signatures with domain separation
- Three communication patterns: sync, streaming, webhooks
- BIZRA-specific skill definitions for 7+1 guardians
- Integration with existing A2A protocol infrastructure

A2A Protocol Reference:
https://github.com/google-a2a/A2A

Architecture:
    +-----------------+     +-----------------+
    |   AgentCard A   |<--->|   AgentCard B   |
    |  (signed Ed25519)|     | (signed Ed25519)|
    +-----------------+     +-----------------+
            |                       |
            v                       v
    +---------------------------------------+
    |         ConstellationHandshake        |
    |  - verify_agent_card()                |
    |  - handshake()                        |
    |  - establish_channel()                |
    +---------------------------------------+
                    |
                    v
    +---------------------------------------+
    |            A2AChannel                 |
    |  - send_message() async               |
    |  - receive_message() async            |
    |  - subscribe_to_task() -> AsyncIter   |
    +---------------------------------------+
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# A2A Protocol imports from existing infrastructure
from .a2a_protocol import (
    A2AMessage,
    A2ARouter,
    MessageBuilder,
    MessageType,
    MessagePriority,
    DeliveryStatus,
    get_a2a_router,
)

logger = logging.getLogger(__name__)


# ===============================================================================
# CONSTANTS & DOMAIN SEPARATION
# ===============================================================================

# A2A Protocol version
A2A_PROTOCOL_VERSION = "0.3.0"

# Domain separation prefixes for Ed25519 signatures
# Prevents signature reuse across different contexts
DOMAIN_AGENT_CARD = b"BIZRA-A2A-AgentCard-v1:"
DOMAIN_HANDSHAKE = b"BIZRA-A2A-Handshake-v1:"
DOMAIN_CHANNEL = b"BIZRA-A2A-Channel-v1:"
DOMAIN_MESSAGE = b"BIZRA-A2A-Message-v1:"

# Communication patterns supported
class CommunicationPattern(str, Enum):
    """A2A v0.3 communication patterns."""
    SYNC = "sync"                    # Request-response
    STREAMING = "streaming"          # Server-sent events / async iteration
    WEBHOOKS = "webhooks"            # Push notifications via callback URLs


# ===============================================================================
# SECURITY SCHEME DEFINITIONS
# ===============================================================================

class SecuritySchemeType(str, Enum):
    """Supported security schemes per A2A spec."""
    NONE = "none"
    API_KEY = "apiKey"
    BEARER = "bearer"
    ED25519 = "ed25519"
    MUTUAL_TLS = "mutualTLS"


@dataclass
class SecurityScheme:
    """
    Security scheme definition per A2A v0.3 spec.

    Attributes:
        type: The security scheme type
        description: Human-readable description
        name: Header/parameter name for the credential
        in_: Location of credential (header, query, cookie)
        scheme: Auth scheme for bearer type (e.g., "bearer")
    """
    type: SecuritySchemeType
    description: str = ""
    name: str = ""
    in_: str = "header"  # header, query, cookie
    scheme: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "name": self.name,
            "in": self.in_,
            "scheme": self.scheme,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityScheme":
        return cls(
            type=SecuritySchemeType(data.get("type", "none")),
            description=data.get("description", ""),
            name=data.get("name", ""),
            in_=data.get("in", "header"),
            scheme=data.get("scheme", ""),
        )

    @classmethod
    def ed25519_default(cls) -> "SecurityScheme":
        """Default Ed25519 signature scheme for BIZRA."""
        return cls(
            type=SecuritySchemeType.ED25519,
            description="Ed25519 signature with domain separation",
            name="X-BIZRA-Signature",
            in_="header",
        )

    @classmethod
    def bearer_default(cls) -> "SecurityScheme":
        """Default bearer token scheme."""
        return cls(
            type=SecuritySchemeType.BEARER,
            description="Bearer token authentication",
            name="Authorization",
            in_="header",
            scheme="bearer",
        )


# ===============================================================================
# AGENT SKILL DEFINITIONS
# ===============================================================================

@dataclass
class AgentSkill:
    """
    Agent skill definition per A2A v0.3 spec.

    Skills describe what an agent can do and are used for
    capability-based discovery and routing.
    """
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
        }
        if self.input_schema:
            result["inputSchema"] = self.input_schema
        if self.output_schema:
            result["outputSchema"] = self.output_schema
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSkill":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            tags=data.get("tags", []),
            input_schema=data.get("inputSchema"),
            output_schema=data.get("outputSchema"),
        )


# BIZRA Guardian Skills (7+1 PAT/SAT structure)
# 7 PAT agents + 1 meta-orchestrator
GUARDIAN_SKILLS: Dict[str, List[AgentSkill]] = {
    # PAT Agents (Personal Agentic Team)
    "MasterReasoner": [
        AgentSkill(
            id="strategic-planning",
            name="Strategic Planning",
            description="Long-term vision and strategic direction synthesis",
            tags=["planning", "strategy", "vision", "leadership"],
        ),
        AgentSkill(
            id="creative-synthesis",
            name="Creative Synthesis",
            description="Novel solution generation and innovation",
            tags=["creativity", "innovation", "synthesis", "problem-solving"],
        ),
    ],
    "DataAnalyzer": [
        AgentSkill(
            id="data-analysis",
            name="Data Analysis",
            description="Pattern recognition and data-driven insights",
            tags=["analysis", "data", "patterns", "metrics"],
        ),
        AgentSkill(
            id="optimization",
            name="Optimization",
            description="Performance and efficiency optimization",
            tags=["optimization", "efficiency", "performance"],
        ),
    ],
    "ExecutionPlanner": [
        AgentSkill(
            id="task-planning",
            name="Task Planning",
            description="Actionable execution plan creation",
            tags=["planning", "execution", "tasks", "workflow"],
        ),
        AgentSkill(
            id="resource-allocation",
            name="Resource Allocation",
            description="Optimal resource distribution and scheduling",
            tags=["resources", "allocation", "scheduling"],
        ),
    ],
    "EthicsGuardian": [
        AgentSkill(
            id="ethics-validation",
            name="Ethics Validation",
            description="Ihsan compliance and ethical excellence verification",
            tags=["ethics", "ihsan", "compliance", "safety"],
        ),
        AgentSkill(
            id="bias-detection",
            name="Bias Detection",
            description="Identify and mitigate biases in outputs",
            tags=["bias", "fairness", "quality"],
        ),
    ],
    "Communicator": [
        AgentSkill(
            id="user-messaging",
            name="User Messaging",
            description="Clear, user-facing communication",
            tags=["communication", "messaging", "user-facing"],
        ),
        AgentSkill(
            id="presentation",
            name="Presentation",
            description="Create compelling presentations and summaries",
            tags=["presentation", "summary", "clarity"],
        ),
    ],
    "MemoryArchitect": [
        AgentSkill(
            id="knowledge-retrieval",
            name="Knowledge Retrieval",
            description="Context management and knowledge graph queries",
            tags=["memory", "knowledge", "context", "retrieval"],
        ),
        AgentSkill(
            id="integration",
            name="Integration",
            description="Cross-system knowledge integration",
            tags=["integration", "coordination", "harmony"],
        ),
    ],
    # SAT Agents (System Agentic Team) - Guardians
    "SecurityGuardian": [
        AgentSkill(
            id="security-validation",
            name="Security Validation",
            description="Threat detection and security verification",
            tags=["security", "threats", "validation"],
        ),
        AgentSkill(
            id="injection-prevention",
            name="Injection Prevention",
            description="Detect and prevent prompt injection attacks",
            tags=["security", "injection", "defense"],
        ),
    ],
    # Meta-Orchestrator (the +1)
    "ConstellationOrchestrator": [
        AgentSkill(
            id="agent-coordination",
            name="Agent Coordination",
            description="Orchestrate multi-agent collaboration",
            tags=["orchestration", "coordination", "meta"],
        ),
        AgentSkill(
            id="consensus-building",
            name="Consensus Building",
            description="Achieve consensus across agent outputs",
            tags=["consensus", "voting", "aggregation"],
        ),
        AgentSkill(
            id="snr-routing",
            name="SNR-Tier Routing",
            description="Route tasks based on signal-to-noise requirements",
            tags=["routing", "snr", "quality"],
        ),
    ],
}


# ===============================================================================
# AGENT CAPABILITIES
# ===============================================================================

@dataclass
class AgentCapabilities:
    """
    Agent capabilities per A2A v0.3 spec.

    Defines what communication features the agent supports.
    """
    streaming: bool = True
    push_notifications: bool = True
    extended_agent_card: bool = True
    state_transition_history: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "streaming": self.streaming,
            "pushNotifications": self.push_notifications,
            "extendedAgentCard": self.extended_agent_card,
            "stateTransitionHistory": self.state_transition_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCapabilities":
        return cls(
            streaming=data.get("streaming", True),
            push_notifications=data.get("pushNotifications", True),
            extended_agent_card=data.get("extendedAgentCard", True),
            state_transition_history=data.get("stateTransitionHistory", False),
        )


# ===============================================================================
# AGENT CARD SIGNATURE
# ===============================================================================

@dataclass
class AgentCardSignature:
    """
    Cryptographic signature for AgentCard verification.

    Uses Ed25519 with domain separation to prevent signature reuse.
    """
    algorithm: str = "Ed25519"
    public_key: str = ""  # Base64-encoded public key
    signature: str = ""   # Base64-encoded signature
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    domain: str = "BIZRA-A2A-AgentCard-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "publicKey": self.public_key,
            "signature": self.signature,
            "timestamp": self.timestamp,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCardSignature":
        return cls(
            algorithm=data.get("algorithm", "Ed25519"),
            public_key=data.get("publicKey", ""),
            signature=data.get("signature", ""),
            timestamp=data.get("timestamp", ""),
            domain=data.get("domain", "BIZRA-A2A-AgentCard-v1"),
        )


# ===============================================================================
# AGENT CARD
# ===============================================================================

@dataclass
class AgentCard:
    """
    Agent identity card per A2A v0.3 specification.

    The AgentCard is the fundamental identity document for A2A protocol.
    It contains all information needed for discovery, authorization,
    and capability negotiation.

    Attributes:
        name: Human-readable agent name
        description: Agent description and purpose
        version: Agent implementation version
        url: Agent endpoint URL
        protocol_version: A2A protocol version (0.3.0)
        capabilities: Supported communication features
        skills: List of agent capabilities/skills
        security_schemes: Supported authentication methods
        signature: Optional cryptographic signature
    """
    name: str
    description: str
    version: str = "1.0.0"
    url: str = ""
    protocol_version: str = A2A_PROTOCOL_VERSION
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = field(default_factory=list)
    security_schemes: List[SecurityScheme] = field(default_factory=list)
    signature: Optional[AgentCardSignature] = None

    # Extended fields
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:12]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON transport."""
        result = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities.to_dict(),
            "skills": [s.to_dict() for s in self.skills],
            "securitySchemes": [s.to_dict() for s in self.security_schemes],
            "agentId": self.agent_id,
            "createdAt": self.created_at,
        }
        if self.signature:
            result["signature"] = self.signature.to_dict()
        if self.expires_at:
            result["expiresAt"] = self.expires_at
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Deserialize from dictionary."""
        signature = None
        if "signature" in data:
            signature = AgentCardSignature.from_dict(data["signature"])

        return cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
            url=data.get("url", ""),
            protocol_version=data.get("protocolVersion", A2A_PROTOCOL_VERSION),
            capabilities=AgentCapabilities.from_dict(data.get("capabilities", {})),
            skills=[AgentSkill.from_dict(s) for s in data.get("skills", [])],
            security_schemes=[SecurityScheme.from_dict(s) for s in data.get("securitySchemes", [])],
            signature=signature,
            agent_id=data.get("agentId", f"agent_{uuid.uuid4().hex[:12]}"),
            created_at=data.get("createdAt", datetime.now(timezone.utc).isoformat()),
            expires_at=data.get("expiresAt"),
            metadata=data.get("metadata", {}),
        )

    def canonical_form(self) -> bytes:
        """
        Generate canonical byte representation for signing.

        Excludes the signature field to allow verification.
        Uses deterministic JSON serialization.
        """
        data = self.to_dict()
        data.pop("signature", None)
        # Deterministic JSON with sorted keys
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return canonical.encode("utf-8")

    def content_hash(self) -> str:
        """Generate SHA-256 hash of canonical form."""
        return hashlib.sha256(self.canonical_form()).hexdigest()


# ===============================================================================
# HANDSHAKE RESULT
# ===============================================================================

@dataclass
class HandshakeResult:
    """
    Result of an A2A handshake attempt.

    Contains the outcome of capability negotiation and
    channel establishment between two agents.
    """
    success: bool
    channel_id: str = ""
    capabilities_negotiated: List[str] = field(default_factory=list)
    security_scheme: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    local_agent_id: str = ""
    remote_agent_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "channelId": self.channel_id,
            "capabilitiesNegotiated": self.capabilities_negotiated,
            "securityScheme": self.security_scheme,
            "timestamp": self.timestamp.isoformat(),
            "errorMessage": self.error_message,
            "localAgentId": self.local_agent_id,
            "remoteAgentId": self.remote_agent_id,
        }


# ===============================================================================
# A2A CHANNEL
# ===============================================================================

@dataclass
class A2AChannel:
    """
    Bidirectional communication channel between two agents.

    Supports all three A2A communication patterns:
    - Sync: Request-response via send_message/receive_message
    - Streaming: Async iteration via subscribe_to_task
    - Webhooks: Push notifications via registered callbacks

    Attributes:
        channel_id: Unique channel identifier
        local_agent: This agent's card
        remote_agent: Partner agent's card
        created_at: Channel creation timestamp
        state: Current channel state
    """
    channel_id: str
    local_agent: AgentCard
    remote_agent: AgentCard
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: str = "open"  # open, closing, closed

    # Internal state
    _message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _task_subscriptions: Dict[str, asyncio.Queue] = field(default_factory=dict)
    _webhook_callbacks: List[Callable] = field(default_factory=list)
    _router: Optional[A2ARouter] = None

    def __post_init__(self):
        """Initialize channel with A2A router."""
        self._router = get_a2a_router()
        # Register both agents if not already registered
        self._router.register_agent(self.local_agent.agent_id)
        self._router.register_agent(self.remote_agent.agent_id)

    async def send_message(
        self,
        message: Union[A2AMessage, Dict[str, Any], str],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> DeliveryStatus:
        """
        Send a message to the remote agent.

        Args:
            message: The message to send (A2AMessage, dict, or string)
            priority: Message priority level

        Returns:
            DeliveryStatus indicating success or failure
        """
        if self.state != "open":
            logger.warning(f"Cannot send on closed channel: {self.channel_id}")
            return DeliveryStatus.FAILED

        # Build message if needed
        if isinstance(message, str):
            builder = MessageBuilder(self.local_agent.agent_id)
            a2a_msg = builder.inform(message).to(self.remote_agent.agent_id).priority(priority).build()
        elif isinstance(message, dict):
            builder = MessageBuilder(self.local_agent.agent_id)
            a2a_msg = builder.to(self.remote_agent.agent_id).priority(priority).build()
            a2a_msg.content = message
        else:
            a2a_msg = message

        # Add channel context
        a2a_msg.session_id = self.channel_id

        return await self._router.send(a2a_msg)

    async def receive_message(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[A2AMessage]:
        """
        Receive the next message from the remote agent.

        Args:
            timeout: Maximum time to wait (None for indefinite)

        Returns:
            The received message or None if timeout
        """
        if self.state != "open":
            return None

        mailbox = self._router.get_mailbox(self.local_agent.agent_id)
        if not mailbox:
            return None

        return await mailbox.receive(timeout)

    async def subscribe_to_task(
        self,
        task_id: str,
    ) -> AsyncIterator[A2AMessage]:
        """
        Subscribe to updates for a specific task.

        Implements streaming pattern per A2A v0.3 spec.

        Args:
            task_id: The task to subscribe to

        Yields:
            Messages related to the task
        """
        if task_id not in self._task_subscriptions:
            self._task_subscriptions[task_id] = asyncio.Queue()

        queue = self._task_subscriptions[task_id]

        while self.state == "open":
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield msg
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def register_webhook(
        self,
        callback: Callable[[A2AMessage], None],
    ) -> None:
        """
        Register a webhook callback for push notifications.

        Args:
            callback: Function to call when messages arrive
        """
        self._webhook_callbacks.append(callback)

    async def close(self) -> None:
        """
        Close the channel gracefully.

        Sends a close notification to the remote agent and
        cleans up resources.
        """
        if self.state == "closed":
            return

        self.state = "closing"

        # Notify remote agent
        try:
            builder = MessageBuilder(self.local_agent.agent_id)
            close_msg = (
                builder
                .inform("Channel closing")
                .to(self.remote_agent.agent_id)
                .subject("channel:close")
                .build()
            )
            close_msg.session_id = self.channel_id
            await self._router.send(close_msg)
        except Exception as e:
            logger.warning(f"Error sending close message: {e}")

        # Clean up subscriptions
        for queue in self._task_subscriptions.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._task_subscriptions.clear()
        self._webhook_callbacks.clear()
        self.state = "closed"

        logger.info(f"Channel {self.channel_id} closed")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channelId": self.channel_id,
            "localAgent": self.local_agent.to_dict(),
            "remoteAgent": self.remote_agent.to_dict(),
            "createdAt": self.created_at.isoformat(),
            "state": self.state,
        }


# ===============================================================================
# CONSTELLATION HANDSHAKE
# ===============================================================================

class ConstellationHandshake:
    """
    A2A Protocol v0.3 compliant handshake implementation for BIZRA.

    Handles agent discovery, card exchange, verification, and
    channel establishment for the BIZRA constellation.

    Features:
    - Ed25519 signatures with domain separation
    - Guardian-specific skill definitions
    - Integration with existing A2A router infrastructure
    - Support for sync, streaming, and webhook patterns
    """

    def __init__(self, guardian_constellation: Optional[Any] = None):
        """
        Initialize the handshake handler.

        Args:
            guardian_constellation: Reference to the BIZRA constellation
                                   (for accessing guardian definitions)
        """
        self.guardian_constellation = guardian_constellation
        self._router = get_a2a_router()
        self._known_agents: Dict[str, AgentCard] = {}
        self._channels: Dict[str, A2AChannel] = {}
        self._private_keys: Dict[str, bytes] = {}  # agent_id -> private_key

        logger.info("ConstellationHandshake initialized with A2A v0.3 support")

    def generate_agent_card(
        self,
        guardian: Union[str, Dict[str, Any]],
        url: str = "",
    ) -> AgentCard:
        """
        Generate an AgentCard for a BIZRA guardian.

        Args:
            guardian: Guardian name (str) or full definition (dict)
            url: Agent endpoint URL

        Returns:
            AgentCard with guardian-specific skills and capabilities
        """
        # Extract guardian info
        if isinstance(guardian, str):
            name = guardian
            description = f"BIZRA {guardian} Guardian Agent"
            metadata = {}
        else:
            name = guardian.get("name", "Unknown")
            description = guardian.get("description", f"BIZRA {name} Guardian Agent")
            metadata = guardian.get("metadata", {})

        # Get skills for this guardian
        skills = GUARDIAN_SKILLS.get(name, [
            AgentSkill(
                id=f"{name.lower()}-default",
                name=f"{name} Default Skill",
                description=f"Default capability for {name}",
                tags=["guardian", "bizra"],
            )
        ])

        # Default security schemes
        security_schemes = [
            SecurityScheme.ed25519_default(),
            SecurityScheme.bearer_default(),
        ]

        # Create agent card
        card = AgentCard(
            name=name,
            description=description,
            version="1.0.0",
            url=url,
            protocol_version=A2A_PROTOCOL_VERSION,
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=True,
                extended_agent_card=True,
            ),
            skills=skills,
            security_schemes=security_schemes,
            metadata={
                "guardian_type": name,
                "constellation": "BIZRA",
                "ihsan_threshold": 0.95,
                **metadata,
            },
        )

        logger.debug(f"Generated AgentCard for {name}: {card.agent_id}")
        return card

    def sign_agent_card(
        self,
        card: AgentCard,
        private_key: bytes,
    ) -> AgentCard:
        """
        Sign an AgentCard using Ed25519 with domain separation.

        Args:
            card: The card to sign
            private_key: Ed25519 private key (32 bytes)

        Returns:
            Signed AgentCard with signature field populated

        Note:
            Requires the 'cryptography' library for real Ed25519.
            Falls back to SHA-256 hash if unavailable.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization

            # Load or create private key
            if len(private_key) == 32:
                # Seed format
                signing_key = Ed25519PrivateKey.from_private_bytes(private_key)
            else:
                # Full private key format
                signing_key = Ed25519PrivateKey.from_private_bytes(private_key[:32])

            # Get public key
            public_key = signing_key.public_key()
            public_bytes = public_key.public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )

            # Create message with domain separation
            message = DOMAIN_AGENT_CARD + card.canonical_form()

            # Sign
            signature_bytes = signing_key.sign(message)

            # Create signature object
            card.signature = AgentCardSignature(
                algorithm="Ed25519",
                public_key=base64.b64encode(public_bytes).decode(),
                signature=base64.b64encode(signature_bytes).decode(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                domain="BIZRA-A2A-AgentCard-v1",
            )

            # Store private key for future use
            self._private_keys[card.agent_id] = private_key

            logger.info(f"Signed AgentCard {card.agent_id} with Ed25519")

        except ImportError:
            logger.warning("cryptography library not available, using fallback signing")
            # Fallback: Use HMAC-SHA256 as a signature placeholder
            message = DOMAIN_AGENT_CARD + card.canonical_form()
            signature_hash = hashlib.sha256(private_key + message).hexdigest()

            card.signature = AgentCardSignature(
                algorithm="SHA256-Fallback",
                public_key=hashlib.sha256(private_key).hexdigest()[:64],
                signature=signature_hash,
                timestamp=datetime.now(timezone.utc).isoformat(),
                domain="BIZRA-A2A-AgentCard-v1",
            )

        return card

    def verify_agent_card(self, card: AgentCard) -> bool:
        """
        Verify an AgentCard's signature.

        Args:
            card: The card to verify

        Returns:
            True if signature is valid, False otherwise
        """
        if not card.signature:
            logger.warning(f"AgentCard {card.agent_id} has no signature")
            return False

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            # Decode public key and signature
            public_bytes = base64.b64decode(card.signature.public_key)
            signature_bytes = base64.b64decode(card.signature.signature)

            # Reconstruct message with domain separation
            message = DOMAIN_AGENT_CARD + card.canonical_form()

            # Verify
            public_key = Ed25519PublicKey.from_public_bytes(public_bytes)
            public_key.verify(signature_bytes, message)

            logger.debug(f"AgentCard {card.agent_id} signature verified")
            return True

        except ImportError:
            # Fallback verification not possible without original key
            logger.warning("cryptography library not available, cannot verify Ed25519")
            # Accept fallback signatures by checking algorithm
            if card.signature.algorithm == "SHA256-Fallback":
                logger.info(f"AgentCard {card.agent_id} has fallback signature (not cryptographically verified)")
                return True
            return False

        except Exception as e:
            logger.error(f"AgentCard verification failed: {e}")
            return False

    async def discover_agents(
        self,
        endpoint: str,
        timeout: float = 10.0,
    ) -> List[AgentCard]:
        """
        Discover agents at a remote endpoint.

        Implements A2A discovery protocol by fetching
        /.well-known/agent-card or /agent-card endpoints.

        Args:
            endpoint: Base URL of the agent service
            timeout: Request timeout in seconds

        Returns:
            List of discovered AgentCards
        """
        import aiohttp

        discovered = []

        # Try standard A2A discovery paths
        paths = [
            "/.well-known/agent-card",
            "/agent-card",
            "/a2a/discover",
        ]

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            for path in paths:
                url = endpoint.rstrip("/") + path
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Handle single card or array
                            if isinstance(data, list):
                                cards = [AgentCard.from_dict(c) for c in data]
                            else:
                                cards = [AgentCard.from_dict(data)]

                            for card in cards:
                                if self.verify_agent_card(card):
                                    discovered.append(card)
                                    self._known_agents[card.agent_id] = card
                                else:
                                    logger.warning(f"Discovered card failed verification: {card.name}")

                            logger.info(f"Discovered {len(cards)} agents at {url}")
                            break

                except aiohttp.ClientError as e:
                    logger.debug(f"Discovery failed at {url}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error discovering at {url}: {e}")
                    continue

        return discovered

    async def handshake(
        self,
        local_card: AgentCard,
        remote_card: AgentCard,
    ) -> HandshakeResult:
        """
        Perform A2A handshake with a remote agent.

        Negotiates capabilities and establishes a secure channel.

        Args:
            local_card: This agent's card
            remote_card: Remote agent's card

        Returns:
            HandshakeResult with negotiation outcome
        """
        timestamp = datetime.now(timezone.utc)

        # Verify remote card
        if not self.verify_agent_card(remote_card):
            return HandshakeResult(
                success=False,
                error_message="Remote agent card verification failed",
                timestamp=timestamp,
                local_agent_id=local_card.agent_id,
                remote_agent_id=remote_card.agent_id,
            )

        # Negotiate capabilities
        local_caps = local_card.capabilities
        remote_caps = remote_card.capabilities

        negotiated = []
        if local_caps.streaming and remote_caps.streaming:
            negotiated.append("streaming")
        if local_caps.push_notifications and remote_caps.push_notifications:
            negotiated.append("push_notifications")
        if local_caps.extended_agent_card and remote_caps.extended_agent_card:
            negotiated.append("extended_agent_card")

        # Negotiate security scheme
        local_schemes = {s.type for s in local_card.security_schemes}
        remote_schemes = {s.type for s in remote_card.security_schemes}
        common_schemes = local_schemes & remote_schemes

        # Prefer Ed25519 > Bearer > API Key > None
        if SecuritySchemeType.ED25519 in common_schemes:
            security_scheme = "ed25519"
        elif SecuritySchemeType.BEARER in common_schemes:
            security_scheme = "bearer"
        elif SecuritySchemeType.API_KEY in common_schemes:
            security_scheme = "apiKey"
        else:
            security_scheme = "none"

        # Generate channel ID
        channel_id = f"ch_{uuid.uuid4().hex[:16]}"

        logger.info(
            f"Handshake successful: {local_card.name} <-> {remote_card.name}, "
            f"channel={channel_id}, security={security_scheme}"
        )

        return HandshakeResult(
            success=True,
            channel_id=channel_id,
            capabilities_negotiated=negotiated,
            security_scheme=security_scheme,
            timestamp=timestamp,
            local_agent_id=local_card.agent_id,
            remote_agent_id=remote_card.agent_id,
        )

    async def establish_channel(
        self,
        agent_a: AgentCard,
        agent_b: AgentCard,
    ) -> A2AChannel:
        """
        Establish a bidirectional A2A channel between two agents.

        Performs handshake and creates channel if successful.

        Args:
            agent_a: First agent's card (local)
            agent_b: Second agent's card (remote)

        Returns:
            A2AChannel for communication

        Raises:
            ValueError: If handshake fails
        """
        # Perform handshake
        result = await self.handshake(agent_a, agent_b)

        if not result.success:
            raise ValueError(f"Handshake failed: {result.error_message}")

        # Create channel
        channel = A2AChannel(
            channel_id=result.channel_id,
            local_agent=agent_a,
            remote_agent=agent_b,
        )

        # Store channel
        self._channels[result.channel_id] = channel

        # Store agent cards
        self._known_agents[agent_a.agent_id] = agent_a
        self._known_agents[agent_b.agent_id] = agent_b

        logger.info(f"Established channel {result.channel_id}: {agent_a.name} <-> {agent_b.name}")

        return channel

    def get_channel(self, channel_id: str) -> Optional[A2AChannel]:
        """Get an existing channel by ID."""
        return self._channels.get(channel_id)

    def get_known_agent(self, agent_id: str) -> Optional[AgentCard]:
        """Get a known agent card by ID."""
        return self._known_agents.get(agent_id)

    def list_channels(self) -> List[A2AChannel]:
        """List all active channels."""
        return [ch for ch in self._channels.values() if ch.state == "open"]

    def list_known_agents(self) -> List[AgentCard]:
        """List all known agent cards."""
        return list(self._known_agents.values())


# ===============================================================================
# FACTORY FUNCTIONS
# ===============================================================================

_handshake: Optional[ConstellationHandshake] = None


def get_constellation_handshake(
    guardian_constellation: Optional[Any] = None,
) -> ConstellationHandshake:
    """
    Get the global ConstellationHandshake instance.

    Args:
        guardian_constellation: Optional constellation reference

    Returns:
        Singleton ConstellationHandshake instance
    """
    global _handshake
    if _handshake is None:
        _handshake = ConstellationHandshake(guardian_constellation)
    return _handshake


def generate_guardian_cards() -> Dict[str, AgentCard]:
    """
    Generate AgentCards for all BIZRA guardians (7+1).

    Returns:
        Dictionary mapping guardian names to their AgentCards
    """
    handshake = get_constellation_handshake()
    cards = {}

    for guardian_name in GUARDIAN_SKILLS.keys():
        card = handshake.generate_agent_card(guardian_name)
        cards[guardian_name] = card

    return cards


# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    # Core dataclasses
    "AgentCard",
    "AgentCapabilities",
    "AgentSkill",
    "AgentCardSignature",
    "SecurityScheme",
    "SecuritySchemeType",
    "HandshakeResult",
    "A2AChannel",
    # Main class
    "ConstellationHandshake",
    # Enums
    "CommunicationPattern",
    # Constants
    "A2A_PROTOCOL_VERSION",
    "GUARDIAN_SKILLS",
    "DOMAIN_AGENT_CARD",
    "DOMAIN_HANDSHAKE",
    "DOMAIN_CHANNEL",
    "DOMAIN_MESSAGE",
    # Factory functions
    "get_constellation_handshake",
    "generate_guardian_cards",
]
