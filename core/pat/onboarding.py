"""
BIZRA Node Onboarding Wizard — The Front Door

Implements the 60-second onboarding flow from Article IX § 9.6:
    Install → verify identity → agents activate → immediate value

This module wires together:
    - Identity minting (core.pat.minting)
    - Agent activation (core.pat.agent)
    - Credential persistence (local filesystem)
    - First-query experience

Standing on Giants: General Magic (1990) + Bitcoin Genesis + Constitutional AI
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .agent import AgentStatus, AgentType, PATAgent, SATAgent
from .identity_card import IdentityCard, SovereigntyTier, generate_identity_keypair
from .minting import (
    IdentityMinter,
    OnboardingResult,
    generate_and_onboard,
)

logger = logging.getLogger(__name__)

# Default node data directory
DEFAULT_NODE_DIR = Path.home() / ".bizra-node"

# Sensitive file permissions (owner-only read/write)
_CREDENTIAL_FILE_MODE = 0o600
_DIR_MODE = 0o700


@dataclass
class NodeCredentials:
    """Persisted node credentials — the user's sovereign identity."""

    node_id: str
    public_key: str
    private_key: str  # Stored encrypted in production; plaintext for MVP
    sovereignty_tier: str
    sovereignty_score: float
    created_at: str
    pat_agent_ids: List[str] = field(default_factory=list)
    sat_agent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeCredentials":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OnboardingState:
    """Tracks the onboarding wizard state for resumability."""

    step: str = "start"  # start, keypair, minting, activation, complete
    node_dir: Path = field(default_factory=lambda: DEFAULT_NODE_DIR)
    credentials: Optional[NodeCredentials] = None
    onboarding_result: Optional[OnboardingResult] = None
    error: Optional[str] = None


class OnboardingWizard:
    """
    The Front Door — 60-second onboarding from zero to sovereign.

    Flow:
        1. Generate Ed25519 keypair (local, never leaves device)
        2. Mint sovereign identity (BIZRA-XXXXXXXX)
        3. Activate 7 PAT + 5 SAT agents
        4. Persist credentials to ~/.bizra-node/
        5. Return ready-to-use identity

    Usage:
        wizard = OnboardingWizard()
        result = wizard.onboard()
        print(result.node_id)  # BIZRA-A1B2C3D4

        # Or interactive:
        wizard = OnboardingWizard()
        wizard.run_interactive()
    """

    def __init__(self, node_dir: Optional[Path] = None):
        self._node_dir = node_dir or DEFAULT_NODE_DIR
        self._state = OnboardingState(node_dir=self._node_dir)

    @property
    def node_dir(self) -> Path:
        return self._node_dir

    @property
    def credentials_file(self) -> Path:
        return self._node_dir / "credentials.json"

    @property
    def identity_file(self) -> Path:
        return self._node_dir / "identity.json"

    @property
    def agents_file(self) -> Path:
        return self._node_dir / "agents.json"

    def is_already_onboarded(self) -> bool:
        """Check if this node has already been onboarded."""
        return self.credentials_file.exists()

    def load_existing_credentials(self) -> Optional[NodeCredentials]:
        """Load credentials from disk if they exist."""
        if not self.credentials_file.exists():
            return None
        try:
            data = json.loads(self.credentials_file.read_text())
            return NodeCredentials.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load credentials: {e}")
            return None

    def onboard(self, name: Optional[str] = None) -> NodeCredentials:
        """
        Execute the full onboarding flow (non-interactive).

        Returns:
            NodeCredentials with the new sovereign identity

        Raises:
            RuntimeError: If onboarding fails at any step
            FileExistsError: If node is already onboarded
        """
        # Check for existing identity
        existing = self.load_existing_credentials()
        if existing is not None:
            raise FileExistsError(
                f"Node already onboarded as {existing.node_id}. "
                f"Credentials at: {self.credentials_file}"
            )

        # Step 1: Generate keypair
        private_key, public_key, node_id = generate_identity_keypair()

        # Step 2: Mint identity + agents
        minter = IdentityMinter.create()
        metadata = {}
        if name:
            metadata["display_name"] = name
        result = minter.onboard_user(
            user_public_key=public_key,
            user_metadata=metadata,
            auto_activate=True,
        )

        if not result.success:
            raise RuntimeError(f"Minting failed: {result.error}")

        # Step 3: Self-sign the identity card
        result.identity_card.sign_as_owner(private_key)

        # Step 4: Build credentials
        credentials = NodeCredentials(
            node_id=result.identity_card.node_id,
            public_key=public_key,
            private_key=private_key,
            sovereignty_tier=result.identity_card.sovereignty_tier.value,
            sovereignty_score=result.identity_card.sovereignty_score,
            created_at=result.identity_card.creation_timestamp,
            pat_agent_ids=[a.agent_id for a in result.pat_agents],
            sat_agent_ids=[a.agent_id for a in result.sat_agents],
        )

        # Step 5: Persist to disk
        self._persist(credentials, result)

        # Step 6: Initialize impact tracker (sovereignty growth engine)
        self._init_impact_tracker(credentials.node_id)

        self._state.credentials = credentials
        self._state.onboarding_result = result
        self._state.step = "complete"

        logger.info(f"Onboarding complete: {credentials.node_id}")
        return credentials

    def _init_impact_tracker(self, node_id: str) -> None:
        """Initialize the impact tracker with baseline state for a new node."""
        try:
            from .impact_tracker import ImpactTracker

            tracker = ImpactTracker(
                node_id=node_id,
                state_dir=self._node_dir,
            )
            # Save baseline state (creates impact_tracker.json)
            tracker._save_state()
            logger.info(f"Impact tracker initialized for {node_id}")
        except Exception as e:
            # Impact tracker init should not block onboarding
            logger.warning(f"Impact tracker init skipped: {e}")

    def _persist(self, credentials: NodeCredentials, result: OnboardingResult) -> None:
        """Write identity, agents, and credentials to disk."""
        self._node_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._node_dir, _DIR_MODE)

        # Credentials (private key — atomic restricted permissions)
        # Use os.open() with mode to avoid TOCTOU race between write and chmod
        cred_path = self.credentials_file
        cred_data = json.dumps(credentials.to_dict(), indent=2)
        fd = os.open(
            str(cred_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, _CREDENTIAL_FILE_MODE
        )
        try:
            os.write(fd, cred_data.encode("utf-8"))
        finally:
            os.close(fd)

        # Identity card (public — shareable)
        identity_path = self.identity_file
        identity_path.write_text(json.dumps(result.identity_card.to_dict(), indent=2))

        # Agent manifest
        agents_data = {
            "pat_agents": [a.to_dict() for a in result.pat_agents],
            "sat_agents": [a.to_dict() for a in result.sat_agents],
            "total": result.total_agents_minted,
        }
        agents_path = self.agents_file
        agents_path.write_text(json.dumps(agents_data, indent=2))

        logger.info(f"Credentials persisted to {self._node_dir}")

    def run_interactive(self) -> Optional[NodeCredentials]:
        """
        Run the interactive onboarding wizard (CLI).

        Returns:
            NodeCredentials if successful, None if user aborts
        """
        # Check existing identity
        existing = self.load_existing_credentials()
        if existing is not None:
            print(f"\n  Node already onboarded as: {existing.node_id}")
            print(f"  Tier: {existing.sovereignty_tier.upper()}")
            print(f"  PAT Agents: {len(existing.pat_agent_ids)}")
            print(f"  SAT Agents: {len(existing.sat_agent_ids)}")
            print(f"  Credentials: {self.credentials_file}")
            return existing

        print()
        print("=" * 60)
        print("  BIZRA NODE ONBOARDING")
        print("  Every human is a node. Every node is a seed.")
        print("=" * 60)
        print()

        # Step 1: Name (optional)
        print("  Step 1/4: Identity")
        print("  Your sovereign identity will be generated from a")
        print("  cryptographic keypair. The private key never leaves")
        print("  this device.")
        print()
        name = input("  Display name (optional, press Enter to skip): ").strip()
        if not name:
            name = None

        # Step 2: Generate and mint
        print()
        print("  Step 2/4: Generating Ed25519 keypair...")
        try:
            credentials = self.onboard(name=name)
        except RuntimeError as e:
            print(f"\n  ERROR: {e}")
            return None

        # Step 3: Show result
        print(f"  Identity minted: {credentials.node_id}")
        print()
        print("  Step 3/4: Activating your agentic team...")
        print(f"  PAT (Personal Agentic Team): {len(credentials.pat_agent_ids)} agents")
        print(f"  SAT (System Agentic Team):   {len(credentials.sat_agent_ids)} agents")
        print()

        # Step 4: Summary
        print("  Step 4/4: Credentials saved")
        print(f"  Location: {self._node_dir}")
        print()
        print("=" * 60)
        print(f"  Welcome to BIZRA, {credentials.node_id}")
        print(f"  Sovereignty Tier: {credentials.sovereignty_tier.upper()} (SEED)")
        print()
        print("  Your 12 agents are active and ready to serve.")
        print()
        print("  Next steps:")
        print('    bizra query "What can you do for me?"')
        print("    bizra status")
        print("    bizra dashboard")
        print("=" * 60)
        print()

        return credentials


def get_node_credentials(node_dir: Optional[Path] = None) -> Optional[NodeCredentials]:
    """
    Load existing node credentials.

    Args:
        node_dir: Override default node directory

    Returns:
        NodeCredentials if onboarded, None otherwise
    """
    wizard = OnboardingWizard(node_dir=node_dir)
    return wizard.load_existing_credentials()


def is_onboarded(node_dir: Optional[Path] = None) -> bool:
    """Check if this node has been onboarded."""
    wizard = OnboardingWizard(node_dir=node_dir)
    return wizard.is_already_onboarded()


def quick_onboard(
    name: Optional[str] = None,
    node_dir: Optional[Path] = None,
) -> NodeCredentials:
    """
    One-call onboarding for programmatic use.

    Args:
        name: Optional display name
        node_dir: Override default node directory

    Returns:
        NodeCredentials with the new identity

    Raises:
        RuntimeError: If onboarding fails
        FileExistsError: If already onboarded
    """
    wizard = OnboardingWizard(node_dir=node_dir)
    return wizard.onboard(name=name)
