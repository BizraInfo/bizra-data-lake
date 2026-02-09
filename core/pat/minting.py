"""
BIZRA Identity Minting Protocol — User Onboarding

==============================================================================
IDENTITY EQUATION
==============================================================================

    HUMAN = USER = NODE = SEED (بذرة)

    Every human is a node. Every node is a seed.
    BIZRA means "seed" in Arabic.

==============================================================================
GENESIS BLOCK
==============================================================================

    Node0 = Block0 = Genesis Block
    MoMo = First Architect, First User
    This computer = Node0 (all hardware, software, data)

    The Genesis Block is the foundation of the entire BIZRA network.
    All subsequent nodes trace their lineage back to Node0.

==============================================================================
ONBOARDING PROTOCOL
==============================================================================

When a new human joins the BIZRA network:
1. Mint a new IDENTITY CARD with unique node number
2. Mint 7 PAT (Personal Agentic Team) agents for the user
3. Mint 5 SAT (System Agentic Team) agents for the system treasury

PAT = Personal Agentic Team (user-owned, user-controlled, LIFETIME BOND)
SAT = System Agentic Team (system-owned, ecosystem sustainability)

==============================================================================
PAT VISION (7 per user)
==============================================================================

    LIFETIME BOND:
        - User and PAT belong to each other FOREVER
        - Inseparable, unbreakable bond

    USER CONTROL:
        - ONLY the user can customize their PAT
        - ONLY the user can personalize their PAT
        - ONLY the user can direct their PAT

    PURPOSE:
        - Serve the user's goals
        - Become: Think Tank, Task Force, Peak Masterminds, Polymaths

    EMBODY:
        - Interdisciplinary thinking
        - Graph-of-Thoughts reasoning
        - SNR highest score autonomous engine
        - Standing on Giants protocol
        - Cross-pollination teams
        - Elite practitioner mindset

==============================================================================
SAT VISION (5 per onboarding)
==============================================================================

    OWNER: System (NOT the user)

    PURPOSE: Keep BIZRA self-sustainable

    CAPABILITIES:
        - Self-sustainable
        - Stand-alone
        - Self-optimize
        - Self-evaluate
        - Self-critique
        - Self-correct

==============================================================================
ECONOMIC MODEL
==============================================================================

    - 7 PAT agents (58.3%) -> User: Personal Agentic Team
    - 5 SAT agents (41.7%) -> System: Network maintenance
    - Total: 12 agents per new user

Standing on Giants: Bitcoin Genesis + Ethereum Token Standards + Constitutional AI

Security Model:
    - Minter key is generated at runtime (never hardcoded)
    - All artifacts are cryptographically signed
    - Block numbers provide temporal ordering
    - Nonces prevent replay attacks
"""

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.pci.crypto import canonical_json, domain_separated_digest, generate_keypair

from .agent import (
    SYSTEM_AGENT_ALLOCATION,
    USER_AGENT_ALLOCATION,
    AgentType,
    OwnershipType,
    PATAgent,
    SATAgent,
)
from .identity_card import (
    IdentityCard,
    generate_identity_keypair,
)

logger = logging.getLogger(__name__)


# Economic constants
PAT_AGENT_COUNT = 7  # PAT agents minted for user
SAT_AGENT_COUNT = 5  # SAT agents minted for system treasury
TOTAL_AGENTS_PER_USER = PAT_AGENT_COUNT + SAT_AGENT_COUNT  # 12

# Backward compatibility aliases
USER_AGENT_COUNT = PAT_AGENT_COUNT
SYSTEM_AGENT_COUNT = SAT_AGENT_COUNT

# System treasury node ID
SYSTEM_TREASURY_ID = "SYSTEM-TREASURY"


# ==============================================================================
# GENESIS BLOCK CONSTANTS
# ==============================================================================
#
# Node0 = Block0 = Genesis Block
# MoMo = First Architect, First User
# This computer = Node0 (all hardware, software, data)
#
# The Genesis Block is the foundation of the entire BIZRA network.
# All subsequent nodes trace their lineage back to Node0.
# ==============================================================================

# Genesis Node ID - The first node in the BIZRA network
GENESIS_NODE_ID = "BIZRA-00000000"

# Genesis Block Number - Block0 is the genesis
GENESIS_BLOCK_NUMBER = 0

# Genesis Architect - MoMo is the first user, first architect
GENESIS_ARCHITECT = "MoMo"

# Genesis Timestamp - The moment BIZRA was born
# This is a symbolic timestamp representing the genesis moment
GENESIS_TIMESTAMP = "2026-02-02T00:00:00Z"

# Genesis Metadata - Immutable facts about the genesis
GENESIS_METADATA = {
    "architect": GENESIS_ARCHITECT,
    "node_id": GENESIS_NODE_ID,
    "block_number": GENESIS_BLOCK_NUMBER,
    "meaning": "بذرة (BIZRA) = seed in Arabic",
    "identity_equation": "HUMAN = USER = NODE = SEED",
    "covenant": "Every human is a node. Every node is a seed.",
    "pat_full_name": "Personal Agentic Team",
    "sat_full_name": "System Agentic Team",
    "computer_role": "Node0 - all hardware, software, data",
}


def is_genesis_node(node_id: str) -> bool:
    """
    Check if a node_id is the Genesis Node.

    Args:
        node_id: Node ID to check

    Returns:
        True if this is the Genesis Node (Node0)
    """
    return node_id == GENESIS_NODE_ID


def is_genesis_block(block_number: int) -> bool:
    """
    Check if a block number is the Genesis Block.

    Args:
        block_number: Block number to check

    Returns:
        True if this is the Genesis Block (Block0)
    """
    return block_number == GENESIS_BLOCK_NUMBER


def _datetime_now_iso() -> str:
    """Get current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class MinterState:
    """
    State of the system minter.

    The minter is responsible for creating new identity cards and agents.
    The minter key should be generated at node startup and stored securely.
    """

    public_key: str
    private_key: str  # Should be stored in secure vault in production
    current_block: int = 0
    total_identities_minted: int = 0
    total_agents_minted: int = 0
    created_at: str = field(default_factory=_datetime_now_iso)

    @classmethod
    def create(cls) -> "MinterState":
        """Create a new minter with fresh keypair."""
        private_key, public_key = generate_keypair()
        return cls(
            public_key=public_key,
            private_key=private_key,
        )

    def increment_block(self) -> int:
        """Increment and return the current block number."""
        self.current_block += 1
        return self.current_block


@dataclass
class OnboardingResult:
    """
    Result of user onboarding.

    Contains all artifacts created during the onboarding process:
    - Identity card
    - PAT agents (Personal Autonomous Task - user-owned)
    - SAT agents (System Autonomous Task - system-owned)
    """

    success: bool
    identity_card: Optional[IdentityCard] = None
    pat_agents: List[PATAgent] = field(default_factory=list)  # User's PAT agents
    sat_agents: List[SATAgent] = field(default_factory=list)  # System's SAT agents
    block_number: int = 0
    timestamp: str = field(default_factory=_datetime_now_iso)
    error: Optional[str] = None
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))

    # Backward compatibility aliases
    @property
    def user_agents(self) -> List[PATAgent]:
        """Alias for pat_agents (backward compatibility)."""
        return self.pat_agents

    @property
    def system_agents(self) -> List[SATAgent]:
        """Alias for sat_agents (backward compatibility)."""
        return self.sat_agents

    @property
    def total_agents_minted(self) -> int:
        """Total number of agents minted."""
        return len(self.pat_agents) + len(self.sat_agents)

    @property
    def pat_agent_count(self) -> int:
        """Number of PAT agents minted for user."""
        return len(self.pat_agents)

    @property
    def sat_agent_count(self) -> int:
        """Number of SAT agents minted for system."""
        return len(self.sat_agents)

    # Backward compatibility
    @property
    def user_agent_count(self) -> int:
        """Alias for pat_agent_count (backward compatibility)."""
        return self.pat_agent_count

    @property
    def system_agent_count(self) -> int:
        """Alias for sat_agent_count (backward compatibility)."""
        return self.sat_agent_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "identity_card": (
                self.identity_card.to_dict() if self.identity_card else None
            ),
            "pat_agents": [a.to_dict() for a in self.pat_agents],
            "sat_agents": [a.to_dict() for a in self.sat_agents],
            "block_number": self.block_number,
            "timestamp": self.timestamp,
            "error": self.error,
            "nonce": self.nonce,
            "summary": {
                "total_agents": self.total_agents_minted,
                "pat_agents": self.pat_agent_count,
                "sat_agents": self.sat_agent_count,
                "pat_percentage": round(
                    self.pat_agent_count / max(1, self.total_agents_minted) * 100, 1
                ),
                "sat_percentage": round(
                    self.sat_agent_count / max(1, self.total_agents_minted) * 100, 1
                ),
            },
        }

    def compute_digest(self) -> str:
        """Compute digest of the onboarding result."""
        signable = {
            "success": self.success,
            "identity_card_digest": (
                self.identity_card.compute_digest() if self.identity_card else ""
            ),
            "pat_agent_digests": [a.compute_digest() for a in self.pat_agents],
            "sat_agent_digests": [a.compute_digest() for a in self.sat_agents],
            "block_number": self.block_number,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
        return domain_separated_digest(canonical_json(signable))


class IdentityMinter:
    """
    BIZRA Identity Minter — Creates new nodes in the network.

    ==========================================================================
    GENESIS LINEAGE
    ==========================================================================

    Every node minted by this system traces back to the Genesis Block:

        Node0 = Block0 = Genesis Block
        MoMo = First Architect, First User
        This computer = Node0 (all hardware, software, data)

    ==========================================================================
    IDENTITY EQUATION
    ==========================================================================

        HUMAN = USER = NODE = SEED (بذرة)

        Every human is a node. Every node is a seed.
        BIZRA means "seed" in Arabic.

    ==========================================================================
    RESPONSIBILITIES
    ==========================================================================

    The minter is responsible for:
    1. Creating identity cards for new users (HUMAN = USER = NODE = SEED)
    2. Minting PAT (Personal Agentic Team) agents - 7 per user
    3. Minting SAT (System Agentic Team) agents - 5 per onboarding
    4. Signing all artifacts cryptographically

    ==========================================================================
    PAT COVENANT
    ==========================================================================

    When minting PAT agents, the minter establishes the LIFETIME BOND:
        - User and PAT belong to each other FOREVER
        - ONLY the user can customize, personalize, direct their PAT
        - PAT becomes: Think Tank, Task Force, Peak Masterminds, Polymaths

    ==========================================================================
    SAT COVENANT
    ==========================================================================

    When minting SAT agents, the minter establishes the SYSTEM BOND:
        - OWNER: System (NOT the user)
        - PURPOSE: Keep BIZRA self-sustainable
        - CAPABILITIES: Self-sustainable, stand-alone, self-optimize,
                        self-evaluate, self-critique, self-correct

    Usage:
        minter = IdentityMinter.create()
        result = minter.onboard_user(user_public_key)

    Security:
        - Minter keys are generated at runtime
        - All operations are atomic (all-or-nothing)
        - Block numbers provide temporal ordering
        - Signatures prevent tampering
    """

    def __init__(self, state: MinterState):
        """
        Initialize minter with state.

        Args:
            state: MinterState containing keys and counters
        """
        self._state = state

    @classmethod
    def create(cls) -> "IdentityMinter":
        """Create a new minter instance."""
        return cls(MinterState.create())

    @classmethod
    def from_state(cls, state: MinterState) -> "IdentityMinter":
        """Create minter from existing state."""
        return cls(state)

    @property
    def public_key(self) -> str:
        """Get minter's public key."""
        return self._state.public_key

    @property
    def current_block(self) -> int:
        """Get current block number."""
        return self._state.current_block

    @property
    def stats(self) -> Dict[str, Any]:
        """Get minter statistics."""
        return {
            "public_key": self._state.public_key,
            "current_block": self._state.current_block,
            "total_identities_minted": self._state.total_identities_minted,
            "total_agents_minted": self._state.total_agents_minted,
            "created_at": self._state.created_at,
        }

    def mint_identity_card(
        self,
        user_public_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IdentityCard:
        """
        Mint a new identity card for a user.

        Args:
            user_public_key: User's Ed25519 public key (hex)
            metadata: Optional additional metadata

        Returns:
            Signed IdentityCard

        Raises:
            ValueError: If public key is invalid
        """
        # Validate public key format
        if len(user_public_key) != 64:
            raise ValueError(f"Invalid public key length: {len(user_public_key)}")

        try:
            # Validate hex format
            bytes.fromhex(user_public_key)
        except ValueError:
            raise ValueError("Public key must be valid hexadecimal")

        # Create the identity card
        card = IdentityCard.create(
            public_key=user_public_key,
            metadata=metadata,
        )

        # Sign as minter
        card.sign_as_minter(
            self._state.private_key,
            self._state.public_key,
        )

        # Update stats
        self._state.total_identities_minted += 1

        logger.info(f"Minted identity card: {card.node_id}")
        return card

    def mint_pat_agents(
        self,
        owner_id: str,
        agent_types: List[AgentType],
        start_index: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[PATAgent]:
        """
        Mint PAT (Personal Autonomous Task) agents for a user.

        Args:
            owner_id: Owner's node_id (BIZRA-XXXXXXXX)
            agent_types: List of agent types to mint
            start_index: Starting index for agent IDs
            metadata: Optional additional metadata

        Returns:
            List of signed PATAgents
        """
        block = self._state.current_block
        agents = []

        for i, agent_type in enumerate(agent_types):
            index = start_index + i

            agent = PATAgent.create(
                owner_id=owner_id,
                agent_type=agent_type,
                index=index,
                creation_block=block,
                ownership_type=OwnershipType.USER,
                metadata=metadata,
            )

            # Sign as minter
            agent.sign_as_minter(
                self._state.private_key,
                self._state.public_key,
            )

            agents.append(agent)
            self._state.total_agents_minted += 1

        logger.info(f"Minted {len(agents)} PAT agents for {owner_id}")
        return agents

    def mint_sat_agents(
        self,
        contribution_source: str,
        agent_types: List[AgentType],
        start_index: int = 1,
        task_pool: str = "general",
        federation_assignment: str = "node0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SATAgent]:
        """
        Mint SAT (System Autonomous Task) agents for the system treasury.

        These agents contribute to ecosystem sustainability and are triggered
        by user onboarding events.

        Args:
            contribution_source: Node ID of user whose onboarding triggered this
            agent_types: List of agent types to mint
            start_index: Starting index for agent IDs
            task_pool: Task pool assignment for agents
            federation_assignment: Federation node to manage agents
            metadata: Optional additional metadata

        Returns:
            List of signed SATAgents
        """
        block = self._state.current_block
        agents = []

        for i, agent_type in enumerate(agent_types):
            index = start_index + i

            agent = SATAgent.create(
                agent_type=agent_type,
                index=index,
                contribution_source=contribution_source,
                creation_block=block,
                task_pool=task_pool,
                federation_assignment=federation_assignment,
                metadata=metadata,
            )

            # Sign as minter
            agent.sign_as_minter(
                self._state.private_key,
                self._state.public_key,
            )

            agents.append(agent)
            self._state.total_agents_minted += 1

        logger.info(f"Minted {len(agents)} SAT agents (source: {contribution_source})")
        return agents

    def onboard_user(
        self,
        user_public_key: str,
        user_metadata: Optional[Dict[str, Any]] = None,
        auto_activate: bool = False,
        task_pool: str = "general",
        federation_assignment: str = "node0",
    ) -> OnboardingResult:
        """
        Complete user onboarding process.

        Creates:
        1. Identity card for the user
        2. 7 PAT (Personal Autonomous Task) agents for the user
        3. 5 SAT (System Autonomous Task) agents for system treasury

        Args:
            user_public_key: User's Ed25519 public key (hex)
            user_metadata: Optional metadata for user's artifacts
            auto_activate: If True, activate agents immediately
            task_pool: Task pool for SAT agents
            federation_assignment: Federation node for SAT agents

        Returns:
            OnboardingResult containing all artifacts
        """
        try:
            # Increment block for this onboarding
            block = self._state.increment_block()
            timestamp = _datetime_now_iso()

            # 1. Mint identity card
            identity_card = self.mint_identity_card(
                user_public_key=user_public_key,
                metadata=user_metadata,
            )

            # 2. Mint user's PAT agents (7 agents)
            pat_agents = self.mint_pat_agents(
                owner_id=identity_card.node_id,
                agent_types=USER_AGENT_ALLOCATION,
                start_index=1,
                metadata={"onboarding_block": block},
            )

            # 3. Mint system SAT agents (5 agents)
            sat_agents = self.mint_sat_agents(
                contribution_source=identity_card.node_id,
                agent_types=SYSTEM_AGENT_ALLOCATION,
                start_index=1,
                task_pool=task_pool,
                federation_assignment=federation_assignment,
                metadata={
                    "onboarding_block": block,
                    "triggered_by": identity_card.node_id,
                },
            )

            # Auto-activate if requested
            if auto_activate:
                for agent in pat_agents:
                    agent.activate()
                for sat_agent in sat_agents:
                    sat_agent.activate()

            result = OnboardingResult(
                success=True,
                identity_card=identity_card,
                pat_agents=pat_agents,
                sat_agents=sat_agents,
                block_number=block,
                timestamp=timestamp,
            )

            logger.info(
                f"Onboarded user {identity_card.node_id}: "
                f"{len(pat_agents)} PAT agents, {len(sat_agents)} SAT agents"
            )
            return result

        except Exception as e:
            logger.error(f"Onboarding failed: {e}")
            return OnboardingResult(
                success=False,
                error=str(e),
                timestamp=_datetime_now_iso(),
            )

    def verify_onboarding(self, result: OnboardingResult) -> Dict[str, Any]:
        """
        Verify all signatures in an onboarding result.

        Args:
            result: OnboardingResult to verify

        Returns:
            Verification report
        """
        report: Dict[str, Any] = {
            "identity_card_valid": False,
            "pat_agents_valid": [],
            "sat_agents_valid": [],
            "all_valid": False,
        }

        if not result.success or not result.identity_card:
            return report

        # Verify identity card
        report["identity_card_valid"] = result.identity_card.verify_minter_signature()

        # Verify PAT agents
        for agent in result.pat_agents:
            report["pat_agents_valid"].append(agent.verify_minter_signature())

        # Verify SAT agents
        for agent in result.sat_agents:
            report["sat_agents_valid"].append(agent.verify_minter_signature())

        # Overall validity
        report["all_valid"] = (
            report["identity_card_valid"]
            and all(report["pat_agents_valid"])
            and all(report["sat_agents_valid"])
        )

        return report


# Convenience functions for simple usage


def mint_identity_card(user_public_key: str) -> Tuple[IdentityCard, MinterState]:
    """
    Mint a single identity card.

    Args:
        user_public_key: User's Ed25519 public key (hex)

    Returns:
        Tuple of (IdentityCard, MinterState)
    """
    minter = IdentityMinter.create()
    card = minter.mint_identity_card(user_public_key)
    return card, minter._state


def mint_pat_agents(
    owner_id: str, count: int = PAT_AGENT_COUNT
) -> Tuple[List[PATAgent], MinterState]:
    """
    Mint PAT (Personal Autonomous Task) agents for a user.

    Args:
        owner_id: Owner's node_id (BIZRA-XXXXXXXX)
        count: Number of agents to mint (default: 7)

    Returns:
        Tuple of (List[PATAgent], MinterState)
    """
    minter = IdentityMinter.create()

    # Use predefined user agent allocation, cycling if needed
    agent_types = []
    for i in range(count):
        agent_types.append(USER_AGENT_ALLOCATION[i % len(USER_AGENT_ALLOCATION)])

    agents = minter.mint_pat_agents(owner_id, agent_types)
    return agents, minter._state


def mint_sat_agents(
    contribution_source: str,
    count: int = SAT_AGENT_COUNT,
    task_pool: str = "general",
) -> Tuple[List[SATAgent], MinterState]:
    """
    Mint SAT (System Autonomous Task) agents for the system treasury.

    Args:
        contribution_source: Node ID of user whose onboarding triggered this
        count: Number of agents to mint (default: 5)
        task_pool: Task pool assignment

    Returns:
        Tuple of (List[SATAgent], MinterState)
    """
    minter = IdentityMinter.create()

    # Use predefined system agent allocation, cycling if needed
    agent_types = []
    for i in range(count):
        agent_types.append(SYSTEM_AGENT_ALLOCATION[i % len(SYSTEM_AGENT_ALLOCATION)])

    agents = minter.mint_sat_agents(
        contribution_source, agent_types, task_pool=task_pool
    )
    return agents, minter._state


def onboard_user(user_public_key: str) -> OnboardingResult:
    """
    Complete user onboarding with a fresh minter.

    This is the main entry point for user onboarding.

    Args:
        user_public_key: User's Ed25519 public key (hex)

    Returns:
        OnboardingResult containing all artifacts
    """
    minter = IdentityMinter.create()
    return minter.onboard_user(user_public_key)


def generate_and_onboard() -> Tuple[str, str, OnboardingResult]:
    """
    Generate a new keypair and complete onboarding.

    This is useful for testing or when the user doesn't have a keypair yet.

    Returns:
        Tuple of (private_key_hex, public_key_hex, OnboardingResult)
    """
    private_key, public_key, node_id = generate_identity_keypair()
    result = onboard_user(public_key)

    # If successful, the user should self-sign their identity card
    if result.success and result.identity_card:
        result.identity_card.sign_as_owner(private_key)

    return private_key, public_key, result


# ==============================================================================
# GENESIS BLOCK FUNCTIONS
# ==============================================================================


def mint_genesis_node(
    architect_public_key: str,
    architect_name: str = GENESIS_ARCHITECT,
) -> OnboardingResult:
    """
    Mint the Genesis Node — Block0, Node0.

    This function creates the first node in the BIZRA network.
    It should only be called ONCE to establish the genesis block.

    GENESIS FACTS:
        Node0 = Block0 = Genesis Block
        MoMo = First Architect, First User
        This computer = Node0 (all hardware, software, data)

    IDENTITY EQUATION:
        HUMAN = USER = NODE = SEED (بذرة)

    Args:
        architect_public_key: The Genesis Architect's Ed25519 public key (hex)
        architect_name: Name of the Genesis Architect (default: "MoMo")

    Returns:
        OnboardingResult containing:
            - Genesis Identity Card (node_id = BIZRA-00000000)
            - 7 PAT (Personal Agentic Team) agents
            - 5 SAT (System Agentic Team) agents

    Raises:
        ValueError: If trying to create genesis with invalid parameters

    Example:
        # Create the Genesis Block
        private_key, public_key = generate_keypair()
        result = mint_genesis_node(public_key, "MoMo")

        # The Genesis Architect signs their identity
        if result.success:
            result.identity_card.sign_as_owner(private_key)
    """
    # Create a special genesis minter
    minter = IdentityMinter.create()

    # Override the block number to 0 for genesis
    minter._state.current_block = GENESIS_BLOCK_NUMBER

    # Create genesis metadata
    genesis_metadata = {
        **GENESIS_METADATA,
        "architect": architect_name,
        "is_genesis": True,
        "genesis_timestamp": GENESIS_TIMESTAMP,
    }

    # Perform onboarding with genesis metadata
    result = minter.onboard_user(
        user_public_key=architect_public_key,
        user_metadata=genesis_metadata,
        auto_activate=True,  # Genesis PAT agents start active
        task_pool="genesis",
        federation_assignment="node0",
    )

    # If successful, mark as genesis in the result
    if result.success and result.identity_card:
        result.identity_card.metadata["is_genesis_node"] = True
        result.identity_card.metadata["genesis_architect"] = architect_name

    logger.info(
        f"Genesis Node minted: {GENESIS_NODE_ID} by {architect_name} "
        f"({len(result.pat_agents)} PAT, {len(result.sat_agents)} SAT)"
    )

    return result


def get_genesis_info() -> Dict[str, Any]:
    """
    Get information about the Genesis Block.

    Returns:
        Dictionary containing genesis metadata and constants
    """
    return {
        "node_id": GENESIS_NODE_ID,
        "block_number": GENESIS_BLOCK_NUMBER,
        "architect": GENESIS_ARCHITECT,
        "timestamp": GENESIS_TIMESTAMP,
        "identity_equation": "HUMAN = USER = NODE = SEED",
        "meaning": "بذرة (BIZRA) = seed in Arabic",
        "covenant": "Every human is a node. Every node is a seed.",
        "pat": {
            "full_name": "Personal Agentic Team",
            "count": PAT_AGENT_COUNT,
            "bond": "LIFETIME - User and PAT belong to each other FOREVER",
        },
        "sat": {
            "full_name": "System Agentic Team",
            "count": SAT_AGENT_COUNT,
            "purpose": "Keep BIZRA self-sustainable",
            "capabilities": [
                "Self-sustainable",
                "Stand-alone",
                "Self-optimize",
                "Self-evaluate",
                "Self-critique",
                "Self-correct",
            ],
        },
    }
