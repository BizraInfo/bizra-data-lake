"""
BIZRA PAT Agent — Personal Agentic Team
BIZRA SAT Agent — System Agentic Team

==============================================================================
IDENTITY EQUATION:
    HUMAN = USER = NODE = SEED (بذرة)

    Every human is a node. Every node is a seed.
    BIZRA means "seed" in Arabic.
==============================================================================

GENESIS:
    Node0 = Block0 = Genesis Block
    MoMo = First Architect, First User
    This computer = Node0 (all hardware, software, data)

==============================================================================
PAT — PERSONAL AGENTIC TEAM (7 per user)
==============================================================================

    LIFETIME BOND:
        - User and PAT belong to each other FOREVER
        - Inseparable, unbreakable bond
        - Grows WITH the user, learns FOR the user

    USER CONTROL:
        - ONLY the user can customize their PAT
        - ONLY the user can personalize their PAT
        - ONLY the user can direct their PAT

    PURPOSE:
        - Serve the user's goals exclusively
        - Become the user's personal Think Tank
        - Become the user's personal Task Force
        - Become the user's Peak Masterminds
        - Become the user's Polymaths

    EMBODIMENT:
        - Interdisciplinary thinking across all domains
        - Graph-of-Thoughts reasoning (non-linear, networked)
        - SNR highest score autonomous engine
        - Standing on Giants protocol (leverage all human knowledge)
        - Cross-pollination teams (ideas flow between agents)
        - Elite practitioner mindset (excellence as default)

Agent Types:
    - WORKER: General task execution
    - RESEARCHER: Information gathering and synthesis
    - GUARDIAN: Security monitoring and validation
    - SYNTHESIZER: Data integration and insight generation
    - VALIDATOR: Proof verification and quality assurance
    - COORDINATOR: Multi-agent orchestration
    - EXECUTOR: External system interaction

==============================================================================
SAT — SYSTEM AGENTIC TEAM (5 per onboarding)
==============================================================================

    OWNER: System (NOT the user)

    PURPOSE: Keep BIZRA self-sustainable

    CAPABILITIES:
        - Self-sustainable: Operates without external intervention
        - Stand-alone: Functions independently of any single node
        - Self-optimize: Continuously improves performance
        - Self-evaluate: Assesses own effectiveness
        - Self-critique: Identifies own weaknesses
        - Self-correct: Fixes own errors autonomously

Standing on Giants:
    Agent Smith Collective Pattern + Constitutional AI + Bitcoin Genesis

Economic Model:
    - 7 PAT agents (58.3%) -> User: Their personal agentic team
    - 5 SAT agents (41.7%) -> System: Ecosystem sustainability
    - Total: 12 agents per new user onboarding
"""

import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from enum import Enum

# Import crypto primitives from PCI module
from core.pci.crypto import (
    generate_keypair,
    sign_message,
    verify_signature,
    canonical_json,
    domain_separated_digest,
)

# Import constants
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


class AgentType(str, Enum):
    """PAT Agent type classification."""
    WORKER = "worker"           # General task execution
    RESEARCHER = "researcher"   # Information gathering
    GUARDIAN = "guardian"       # Security and validation
    SYNTHESIZER = "synthesizer" # Data integration
    VALIDATOR = "validator"     # Proof verification
    COORDINATOR = "coordinator" # Multi-agent orchestration
    EXECUTOR = "executor"       # External system interaction


class AgentStatus(str, Enum):
    """Agent operational status."""
    DORMANT = "dormant"       # Created but not activated
    ACTIVE = "active"         # Running and accepting tasks
    BUSY = "busy"             # Currently processing a task
    SUSPENDED = "suspended"   # Temporarily disabled
    RETIRED = "retired"       # Permanently decommissioned


class OwnershipType(str, Enum):
    """Agent ownership classification."""
    USER = "user"             # Owned by a human user
    SYSTEM = "system"         # Owned by system treasury


# Default capabilities per agent type
DEFAULT_CAPABILITIES: Dict[AgentType, List[str]] = {
    AgentType.WORKER: [
        "task.execute",
        "task.report",
        "memory.read",
        "memory.write",
    ],
    AgentType.RESEARCHER: [
        "search.web",
        "search.local",
        "summarize",
        "memory.read",
        "memory.write",
    ],
    AgentType.GUARDIAN: [
        "validate.ihsan",
        "validate.signature",
        "monitor.security",
        "alert.create",
    ],
    AgentType.SYNTHESIZER: [
        "data.integrate",
        "insight.generate",
        "pattern.detect",
        "memory.read",
        "memory.write",
    ],
    AgentType.VALIDATOR: [
        "proof.verify",
        "quality.assess",
        "validate.snr",
        "validate.ihsan",
    ],
    AgentType.COORDINATOR: [
        "agent.delegate",
        "agent.monitor",
        "workflow.orchestrate",
        "consensus.participate",
    ],
    AgentType.EXECUTOR: [
        "api.call",
        "tool.invoke",
        "external.interact",
        "memory.read",
    ],
}


# Domain prefix for agent signatures
AGENT_DOMAIN_PREFIX = "bizra-agent-v1:"


def _datetime_now_iso() -> str:
    """Get current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _generate_agent_id(owner_id: str, agent_type: AgentType, index: int) -> str:
    """
    Generate unique agent ID.

    Format: PAT-{owner_node_suffix}-{type_prefix}-{index:03d}
    Example: PAT-A1B2C3D4-WRK-001

    Args:
        owner_id: Owner's node_id (BIZRA-XXXXXXXX format)
        agent_type: Type of agent
        index: Sequential index for this owner+type combo

    Returns:
        Unique agent ID
    """
    # Extract owner suffix (last 8 chars of node_id)
    owner_suffix = owner_id.replace("BIZRA-", "")

    # Type prefix mapping
    type_prefixes = {
        AgentType.WORKER: "WRK",
        AgentType.RESEARCHER: "RSC",
        AgentType.GUARDIAN: "GRD",
        AgentType.SYNTHESIZER: "SYN",
        AgentType.VALIDATOR: "VAL",
        AgentType.COORDINATOR: "CRD",
        AgentType.EXECUTOR: "EXC",
    }
    type_prefix = type_prefixes.get(agent_type, "AGT")

    return f"PAT-{owner_suffix}-{type_prefix}-{index:03d}"


@dataclass
class AgentCapability:
    """
    A capability that an agent can perform.

    Capabilities follow a namespace.action format.
    """
    name: str
    enabled: bool = True
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_string(cls, capability: str) -> 'AgentCapability':
        """Create capability from string name."""
        return cls(name=capability)


@dataclass
class PATAgent:
    """
    BIZRA PAT Agent — Personal Agentic Team Member

    ==========================================================================
    LIFETIME BOND COVENANT
    ==========================================================================

    A PAT agent is NOT just an AI assistant. It is a sovereign member of the
    user's PERSONAL AGENTIC TEAM. The bond between user and PAT is:

        - PERMANENT: Lasts the lifetime of the user
        - EXCLUSIVE: Only serves its bonded user
        - EVOLVING: Grows and learns WITH the user
        - PROTECTED: Cannot be transferred, sold, or reassigned

    ==========================================================================
    THE PAT VISION
    ==========================================================================

    Each PAT agent aspires to become:

        1. PERSONAL THINK TANK
           - Deep analysis and strategic thinking
           - Long-term planning and foresight
           - Scenario modeling and risk assessment

        2. PERSONAL TASK FORCE
           - Autonomous task execution
           - Proactive problem solving
           - Coordinated multi-agent operations

        3. PEAK MASTERMIND
           - Excellence in specialized domains
           - Continuous skill development
           - Elite practitioner mindset

        4. POLYMATH
           - Interdisciplinary knowledge synthesis
           - Cross-domain pattern recognition
           - Standing on Giants (leveraging all human knowledge)

    ==========================================================================
    OPERATIONAL PRINCIPLES
    ==========================================================================

        - Graph-of-Thoughts: Non-linear, networked reasoning
        - SNR Maximization: Highest signal-to-noise ratio
        - Cross-pollination: Ideas flow between team members
        - Ihsan Constraint: Excellence as the minimum standard

    Attributes:
        agent_id: Unique agent identifier
        owner_id: Owner's node_id (HUMAN = USER = NODE = SEED)
        agent_type: Classification of agent
        capabilities: List of granted capabilities
        creation_block: Block number at creation (for economic tracking)
        status: Current operational status
        ownership_type: User or System ownership
        public_key: Agent's Ed25519 public key
        ihsan_threshold: Constitutional constraint threshold
        snr_threshold: Quality constraint threshold
        metadata: Additional agent metadata
    """

    agent_id: str
    owner_id: str
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    creation_block: int = 0
    creation_timestamp: str = field(default_factory=_datetime_now_iso)
    status: AgentStatus = AgentStatus.DORMANT
    ownership_type: OwnershipType = OwnershipType.USER
    public_key: str = ""
    private_key_hash: str = ""  # Hash of private key for verification (not the key itself)
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    version: str = "1.0.0"
    task_count: int = 0
    success_rate: float = 1.0
    minter_signature: Optional[str] = None
    minter_public_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate agent fields after initialization."""
        # Validate agent_id format
        if not self.agent_id.startswith("PAT-"):
            raise ValueError(f"Invalid agent_id format: {self.agent_id}")

        # Validate owner_id format (can be BIZRA- for users or SYSTEM for treasury)
        if not (self.owner_id.startswith("BIZRA-") or self.owner_id == "SYSTEM-TREASURY"):
            raise ValueError(f"Invalid owner_id format: {self.owner_id}")

        # Validate thresholds
        if not (0.0 <= self.ihsan_threshold <= 1.0):
            raise ValueError(f"ihsan_threshold must be 0.0-1.0: {self.ihsan_threshold}")
        if not (0.0 <= self.snr_threshold <= 1.0):
            raise ValueError(f"snr_threshold must be 0.0-1.0: {self.snr_threshold}")

    @property
    def is_active(self) -> bool:
        """Check if agent is active and can accept tasks."""
        return self.status == AgentStatus.ACTIVE

    @property
    def is_user_owned(self) -> bool:
        """Check if agent is owned by a user."""
        return self.ownership_type == OwnershipType.USER

    @property
    def is_system_owned(self) -> bool:
        """Check if agent is owned by system treasury."""
        return self.ownership_type == OwnershipType.SYSTEM

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def grant_capability(self, capability: str) -> bool:
        """Grant a capability to the agent."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            return True
        return False

    def revoke_capability(self, capability: str) -> bool:
        """Revoke a capability from the agent."""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            return True
        return False

    def activate(self) -> bool:
        """Activate the agent for task processing."""
        if self.status in (AgentStatus.DORMANT, AgentStatus.SUSPENDED):
            self.status = AgentStatus.ACTIVE
            return True
        return False

    def suspend(self) -> bool:
        """Suspend the agent from task processing."""
        if self.status in (AgentStatus.ACTIVE, AgentStatus.BUSY):
            self.status = AgentStatus.SUSPENDED
            return True
        return False

    def retire(self) -> bool:
        """Permanently retire the agent."""
        self.status = AgentStatus.RETIRED
        return True

    def record_task_completion(self, success: bool) -> None:
        """Record a task completion for success rate tracking."""
        self.task_count += 1
        # Exponential moving average for success rate
        alpha = 0.1
        success_value = 1.0 if success else 0.0
        self.success_rate = alpha * success_value + (1 - alpha) * self.success_rate

    def compute_digest(self) -> str:
        """
        Compute canonical digest of the agent.

        Excludes signatures from the digest computation.
        """
        signable_data = {
            "version": self.version,
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "agent_type": self.agent_type.value if isinstance(self.agent_type, AgentType) else self.agent_type,
            "capabilities": sorted(self.capabilities),
            "creation_block": self.creation_block,
            "creation_timestamp": self.creation_timestamp,
            "ownership_type": self.ownership_type.value if isinstance(self.ownership_type, OwnershipType) else self.ownership_type,
            "public_key": self.public_key,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
        }
        return domain_separated_digest(canonical_json(signable_data))

    def sign_as_minter(self, minter_private_key: str, minter_public_key: str) -> 'PATAgent':
        """
        Sign the agent as the system minter.

        Args:
            minter_private_key: Minter's Ed25519 private key (hex)
            minter_public_key: Minter's Ed25519 public key (hex)

        Returns:
            Self with minter_signature attached
        """
        digest = self.compute_digest()
        self.minter_signature = sign_message(digest, minter_private_key)
        self.minter_public_key = minter_public_key
        return self

    def verify_minter_signature(self) -> bool:
        """Verify the minter's signature on this agent."""
        if not self.minter_signature or not self.minter_public_key:
            return False
        digest = self.compute_digest()
        return verify_signature(digest, self.minter_signature, self.minter_public_key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['agent_type'] = self.agent_type.value if isinstance(self.agent_type, AgentType) else self.agent_type
        d['status'] = self.status.value if isinstance(self.status, AgentStatus) else self.status
        d['ownership_type'] = self.ownership_type.value if isinstance(self.ownership_type, OwnershipType) else self.ownership_type
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PATAgent':
        """Reconstruct from dictionary."""
        data = data.copy()

        # Handle enum conversion
        if 'agent_type' in data:
            data['agent_type'] = AgentType(data['agent_type'])
        if 'status' in data:
            data['status'] = AgentStatus(data['status'])
        if 'ownership_type' in data:
            data['ownership_type'] = OwnershipType(data['ownership_type'])

        return cls(**data)

    @classmethod
    def create(
        cls,
        owner_id: str,
        agent_type: AgentType,
        index: int,
        creation_block: int = 0,
        ownership_type: OwnershipType = OwnershipType.USER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'PATAgent':
        """
        Factory method to create a new PAT agent.

        Args:
            owner_id: Owner's node_id
            agent_type: Type of agent to create
            index: Sequential index for ID generation
            creation_block: Block number at creation
            ownership_type: User or System ownership
            metadata: Optional additional metadata

        Returns:
            New unsigned PATAgent with default capabilities
        """
        # Generate agent keypair
        private_key, public_key = generate_keypair()

        # Hash the private key for verification (don't store the actual key)
        import blake3
        private_key_hash = blake3.blake3(bytes.fromhex(private_key)).hexdigest()[:32]

        agent_id = _generate_agent_id(owner_id, agent_type, index)

        return cls(
            agent_id=agent_id,
            owner_id=owner_id,
            agent_type=agent_type,
            capabilities=DEFAULT_CAPABILITIES.get(agent_type, []).copy(),
            creation_block=creation_block,
            creation_timestamp=_datetime_now_iso(),
            status=AgentStatus.DORMANT,
            ownership_type=ownership_type,
            public_key=public_key,
            private_key_hash=private_key_hash,
            metadata=metadata or {},
        )


# Pre-defined agent configurations for onboarding
USER_AGENT_ALLOCATION = [
    AgentType.WORKER,       # 1. General task execution
    AgentType.WORKER,       # 2. Additional worker capacity
    AgentType.RESEARCHER,   # 3. Information gathering
    AgentType.GUARDIAN,     # 4. Security monitoring
    AgentType.SYNTHESIZER,  # 5. Data integration
    AgentType.VALIDATOR,    # 6. Quality assurance
    AgentType.COORDINATOR,  # 7. Orchestration
]

SYSTEM_AGENT_ALLOCATION = [
    AgentType.VALIDATOR,    # 1. Network validation
    AgentType.GUARDIAN,     # 2. Security oversight
    AgentType.COORDINATOR,  # 3. Cross-node coordination
    AgentType.EXECUTOR,     # 4. External integration
    AgentType.SYNTHESIZER,  # 5. Global pattern detection
]


def _generate_sat_agent_id(agent_type: AgentType, index: int) -> str:
    """
    Generate unique SAT agent ID.

    Format: SAT-{random_suffix}-{type_prefix}-{index:03d}
    Example: SAT-F7E8D9C0-VAL-001

    Args:
        agent_type: Type of agent
        index: Sequential index for this type

    Returns:
        Unique SAT agent ID
    """
    # Use cryptographic random for system agents
    random_suffix = secrets.token_hex(4).upper()

    # Type prefix mapping
    type_prefixes = {
        AgentType.WORKER: "WRK",
        AgentType.RESEARCHER: "RSC",
        AgentType.GUARDIAN: "GRD",
        AgentType.SYNTHESIZER: "SYN",
        AgentType.VALIDATOR: "VAL",
        AgentType.COORDINATOR: "CRD",
        AgentType.EXECUTOR: "EXC",
    }
    type_prefix = type_prefixes.get(agent_type, "AGT")

    return f"SAT-{random_suffix}-{type_prefix}-{index:03d}"


@dataclass
class SATAgent:
    """
    BIZRA SAT Agent — System Agentic Team Member

    ==========================================================================
    SYSTEM SOVEREIGNTY COVENANT
    ==========================================================================

    A SAT agent exists to keep BIZRA SELF-SUSTAINABLE. Unlike PAT agents
    which serve individual users, SAT agents serve the entire ecosystem.

    OWNER: System (NOT the user)

    5 SAT agents are minted for each new user joining the network.
    These agents serve the collective good and maintain network health.

    ==========================================================================
    SELF-* CAPABILITIES (The Six Pillars)
    ==========================================================================

        1. SELF-SUSTAINABLE
           - Operates without external intervention
           - Generates value for the network autonomously
           - Maintains economic equilibrium

        2. STAND-ALONE
           - Functions independently of any single node
           - No single point of failure
           - Byzantine fault tolerant

        3. SELF-OPTIMIZE
           - Continuously improves performance
           - Learns from network-wide patterns
           - Adapts to changing conditions

        4. SELF-EVALUATE
           - Assesses own effectiveness
           - Measures contribution to network health
           - Tracks performance metrics

        5. SELF-CRITIQUE
           - Identifies own weaknesses
           - Recognizes areas for improvement
           - Honest self-assessment

        6. SELF-CORRECT
           - Fixes own errors autonomously
           - Recovers from failures gracefully
           - Implements improvements proactively

    ==========================================================================
    KEY DIFFERENCES FROM PAT
    ==========================================================================

        - owner_id is always "SYSTEM-TREASURY"
        - Agents serve network-wide tasks, not individual users
        - Contribute to consensus, validation, and coordination
        - Cannot be transferred or sold by users
        - Focus on ecosystem health, not individual goals

    Attributes:
        agent_id: Unique agent identifier (SAT-XXXXXXXX-TYPE-NNN)
        agent_type: Classification of agent
        capabilities: List of granted capabilities
        creation_block: Block number at creation
        contribution_source: Node ID of the user whose onboarding triggered creation
        status: Current operational status
        public_key: Agent's Ed25519 public key
        ihsan_threshold: Constitutional constraint threshold
        snr_threshold: Quality constraint threshold
        metadata: Additional agent metadata
    """

    agent_id: str
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    creation_block: int = 0
    creation_timestamp: str = field(default_factory=_datetime_now_iso)
    contribution_source: str = ""  # Node ID that triggered this agent's creation
    status: AgentStatus = AgentStatus.DORMANT
    public_key: str = ""
    private_key_hash: str = ""  # Hash of private key for verification
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD
    version: str = "1.0.0"
    task_count: int = 0
    success_rate: float = 1.0
    federation_assignment: str = ""  # Which federation node manages this agent
    task_pool: str = "general"  # Which task pool this agent serves
    minter_signature: Optional[str] = None
    minter_public_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Fixed owner for all SAT agents
    owner_id: str = field(default="SYSTEM-TREASURY", init=False)
    ownership_type: OwnershipType = field(default=OwnershipType.SYSTEM, init=False)

    def __post_init__(self):
        """Validate agent fields after initialization."""
        # Validate agent_id format
        if not self.agent_id.startswith("SAT-"):
            raise ValueError(f"Invalid SAT agent_id format: {self.agent_id}")

        # Ensure ownership is always SYSTEM
        self.owner_id = "SYSTEM-TREASURY"
        self.ownership_type = OwnershipType.SYSTEM

        # Validate thresholds
        if not (0.0 <= self.ihsan_threshold <= 1.0):
            raise ValueError(f"ihsan_threshold must be 0.0-1.0: {self.ihsan_threshold}")
        if not (0.0 <= self.snr_threshold <= 1.0):
            raise ValueError(f"snr_threshold must be 0.0-1.0: {self.snr_threshold}")

    @property
    def is_active(self) -> bool:
        """Check if agent is active and can accept tasks."""
        return self.status == AgentStatus.ACTIVE

    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities

    def activate(self) -> bool:
        """Activate the agent for task processing."""
        if self.status in (AgentStatus.DORMANT, AgentStatus.SUSPENDED):
            self.status = AgentStatus.ACTIVE
            return True
        return False

    def suspend(self) -> bool:
        """Suspend the agent from task processing."""
        if self.status in (AgentStatus.ACTIVE, AgentStatus.BUSY):
            self.status = AgentStatus.SUSPENDED
            return True
        return False

    def record_task_completion(self, success: bool) -> None:
        """Record a task completion for success rate tracking."""
        self.task_count += 1
        alpha = 0.1
        success_value = 1.0 if success else 0.0
        self.success_rate = alpha * success_value + (1 - alpha) * self.success_rate

    def compute_digest(self) -> str:
        """Compute canonical digest of the agent."""
        signable_data = {
            "version": self.version,
            "agent_id": self.agent_id,
            "owner_id": self.owner_id,
            "agent_type": self.agent_type.value if isinstance(self.agent_type, AgentType) else self.agent_type,
            "capabilities": sorted(self.capabilities),
            "creation_block": self.creation_block,
            "creation_timestamp": self.creation_timestamp,
            "contribution_source": self.contribution_source,
            "ownership_type": self.ownership_type.value,
            "public_key": self.public_key,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
            "task_pool": self.task_pool,
        }
        return domain_separated_digest(canonical_json(signable_data))

    def sign_as_minter(self, minter_private_key: str, minter_public_key: str) -> 'SATAgent':
        """Sign the agent as the system minter."""
        digest = self.compute_digest()
        self.minter_signature = sign_message(digest, minter_private_key)
        self.minter_public_key = minter_public_key
        return self

    def verify_minter_signature(self) -> bool:
        """Verify the minter's signature on this agent."""
        if not self.minter_signature or not self.minter_public_key:
            return False
        digest = self.compute_digest()
        return verify_signature(digest, self.minter_signature, self.minter_public_key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d['agent_type'] = self.agent_type.value if isinstance(self.agent_type, AgentType) else self.agent_type
        d['status'] = self.status.value if isinstance(self.status, AgentStatus) else self.status
        d['ownership_type'] = self.ownership_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SATAgent':
        """Reconstruct from dictionary."""
        data = data.copy()

        # Handle enum conversion
        if 'agent_type' in data:
            data['agent_type'] = AgentType(data['agent_type'])
        if 'status' in data:
            data['status'] = AgentStatus(data['status'])

        # Remove fixed fields that shouldn't be in init
        data.pop('owner_id', None)
        data.pop('ownership_type', None)

        return cls(**data)

    @classmethod
    def create(
        cls,
        agent_type: AgentType,
        index: int,
        contribution_source: str,
        creation_block: int = 0,
        task_pool: str = "general",
        federation_assignment: str = "node0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'SATAgent':
        """
        Factory method to create a new SAT agent.

        Args:
            agent_type: Type of agent to create
            index: Sequential index for ID generation
            contribution_source: Node ID of user whose onboarding triggered creation
            creation_block: Block number at creation
            task_pool: Which task pool to assign agent to
            federation_assignment: Which federation node manages this agent
            metadata: Optional additional metadata

        Returns:
            New unsigned SATAgent with default capabilities
        """
        # Generate agent keypair
        private_key, public_key = generate_keypair()

        # Hash the private key for verification
        import blake3
        private_key_hash = blake3.blake3(bytes.fromhex(private_key)).hexdigest()[:32]

        agent_id = _generate_sat_agent_id(agent_type, index)

        return cls(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=DEFAULT_CAPABILITIES.get(agent_type, []).copy(),
            creation_block=creation_block,
            creation_timestamp=_datetime_now_iso(),
            contribution_source=contribution_source,
            status=AgentStatus.DORMANT,
            public_key=public_key,
            private_key_hash=private_key_hash,
            task_pool=task_pool,
            federation_assignment=federation_assignment,
            metadata=metadata or {},
        )
