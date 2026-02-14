"""
Genesis Identity Loader — Node0 Knows Who It Is
=================================================
Loads the persistent node identity, PAT team, and SAT team from the
genesis ceremony output. Without this, the runtime generates a random
node_id on every startup and forgets everything.

The genesis ceremony is executed once (by bizra-omega/bizra-resourcepool)
and writes:
  - sovereign_state/node0_genesis.json  (full identity + agents + hashes)
  - sovereign_state/pat_roster.txt      (7 PAT agents)
  - sovereign_state/sat_roster.txt      (5 SAT agents)
  - sovereign_state/genesis_hash.txt    (root Merkle hash)

This module reads those files and makes them available to the runtime.

Standing on Giants:
- Al-Ghazali (1095): Self-knowledge precedes all knowledge
- Lamport (1982): Persistent identity in distributed systems
- Nakamoto (2008): Genesis block as immutable origin
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

GENESIS_FILE = "node0_genesis.json"
GENESIS_HASH_FILE = "genesis_hash.txt"


@dataclass
class AgentIdentity:
    """A minted PAT or SAT agent with cryptographic identity."""

    agent_id: str
    role: str
    public_key: str
    capabilities: list[str]
    giants: list[str]
    created_at: int
    agent_hash: bytes

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentIdentity":
        return cls(
            agent_id=data["agent_id"],
            role=data["role"],
            public_key=data["public_key"],
            capabilities=data.get("capabilities", []),
            giants=data.get("giants", []),
            created_at=data.get("created_at", 0),
            agent_hash=bytes(data.get("agent_hash", [])),
        )


@dataclass
class NodeIdentity:
    """Persistent Node0 identity from the genesis ceremony."""

    node_id: str
    public_key: str
    name: str
    location: str
    created_at: int
    identity_hash: bytes

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeIdentity":
        return cls(
            node_id=data["node_id"],
            public_key=data["public_key"],
            name=data.get("name", ""),
            location=data.get("location", ""),
            created_at=data.get("created_at", 0),
            identity_hash=bytes(data.get("identity_hash", [])),
        )


@dataclass
class GenesisState:
    """Complete genesis state — the system's birth certificate."""

    # Core identity
    identity: NodeIdentity

    # Agent teams
    pat_team: list[AgentIdentity] = field(default_factory=list)
    sat_team: list[AgentIdentity] = field(default_factory=list)

    # Cryptographic roots
    pat_team_hash: bytes = b""
    sat_team_hash: bytes = b""
    partnership_hash: bytes = b""
    genesis_hash: bytes = b""

    # Governance
    quorum: float = 0.67
    voting_period_hours: int = 72
    upgrade_threshold: float = 0.8

    # Hardware attestation
    hardware: dict[str, Any] = field(default_factory=dict)

    # Knowledge attestation
    knowledge: dict[str, Any] = field(default_factory=dict)

    # Genesis timestamp
    timestamp: int = 0

    @property
    def node_id(self) -> str:
        return self.identity.node_id

    @property
    def node_name(self) -> str:
        return self.identity.name

    @property
    def pat_agent_ids(self) -> list[str]:
        return [a.agent_id for a in self.pat_team]

    @property
    def sat_agent_ids(self) -> list[str]:
        return [a.agent_id for a in self.sat_team]

    def get_agent(self, agent_id: str) -> Optional[AgentIdentity]:
        """Look up an agent by ID."""
        for agent in self.pat_team + self.sat_team:
            if agent.agent_id == agent_id:
                return agent
        return None

    def get_agents_by_role(self, role: str) -> list[AgentIdentity]:
        """Get agents by role name."""
        return [a for a in self.pat_team + self.sat_team if a.role == role]

    def summary(self) -> dict[str, Any]:
        """Human-readable summary of genesis state."""
        return {
            "node_id": self.identity.node_id,
            "name": self.identity.name,
            "location": self.identity.location,
            "pat_agents": len(self.pat_team),
            "sat_agents": len(self.sat_team),
            "genesis_hash": self.genesis_hash.hex() if self.genesis_hash else "none",
            "timestamp": self.timestamp,
        }


def load_genesis(state_dir: Path) -> Optional[GenesisState]:
    """
    Load genesis state from sovereign_state directory.

    Returns None if no genesis file exists (first-time setup or non-genesis node).
    Raises ValueError if genesis file exists but is corrupted.
    """
    genesis_path = state_dir / GENESIS_FILE
    if not genesis_path.exists():
        return None

    try:
        with open(genesis_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Genesis file corrupted: {e}") from e

    # Parse identity
    identity_data = data.get("identity", {})
    if not identity_data.get("node_id"):
        raise ValueError("Genesis file missing node_id")
    identity = NodeIdentity.from_dict(identity_data)

    # Parse PAT team
    pat_data = data.get("pat_team", {})
    pat_agents = [AgentIdentity.from_dict(a) for a in pat_data.get("agents", [])]
    pat_team_hash = bytes(pat_data.get("team_hash", []))

    # Parse SAT team
    sat_data = data.get("sat_team", {})
    sat_agents = [AgentIdentity.from_dict(a) for a in sat_data.get("agents", [])]
    sat_team_hash = bytes(sat_data.get("team_hash", []))

    # Governance
    governance = sat_data.get("governance", {})

    # Hashes
    partnership_hash = bytes(data.get("partnership_hash", []))
    genesis_hash = bytes(data.get("genesis_hash", []))

    state = GenesisState(
        identity=identity,
        pat_team=pat_agents,
        sat_team=sat_agents,
        pat_team_hash=pat_team_hash,
        sat_team_hash=sat_team_hash,
        partnership_hash=partnership_hash,
        genesis_hash=genesis_hash,
        quorum=governance.get("quorum", 0.67),
        voting_period_hours=governance.get("voting_period_hours", 72),
        upgrade_threshold=governance.get("upgrade_threshold", 0.8),
        hardware=data.get("hardware", {}),
        knowledge=data.get("knowledge", {}),
        timestamp=data.get("timestamp", 0),
    )

    return state


def validate_genesis_hash(state: GenesisState, state_dir: Path) -> bool:
    """
    Validate the genesis hash against the stored hash file.

    Returns True if valid, False if hash mismatch or no hash file.
    """
    hash_path = state_dir / GENESIS_HASH_FILE
    if not hash_path.exists():
        logger.warning("No genesis_hash.txt found — cannot validate")
        return False

    stored_hash = hash_path.read_text().strip()
    computed_hash = state.genesis_hash.hex()

    if stored_hash == computed_hash:
        logger.info(f"Genesis hash validated: {stored_hash[:16]}...")
        return True
    else:
        logger.error(
            f"GENESIS HASH MISMATCH: stored={stored_hash[:16]}... "
            f"computed={computed_hash[:16]}..."
        )
        return False


def load_and_validate_genesis(state_dir: Path) -> Optional[GenesisState]:
    """
    Load genesis state and validate its integrity.

    This is the main entry point — call this at runtime startup.

    Returns:
        GenesisState if valid genesis found, None if no genesis exists.
    Raises:
        ValueError if genesis exists but is corrupted or invalid.
    """
    state = load_genesis(state_dir)
    if state is None:
        logger.info("No genesis file found — running as ephemeral node")
        return None

    # Validate hash
    valid = validate_genesis_hash(state, state_dir)
    if not valid:
        logger.warning("Genesis hash validation failed — proceeding with loaded state")

    logger.info(
        f"Genesis loaded: {state.identity.node_id} "
        f"({state.identity.name}) — "
        f"{len(state.pat_team)} PAT + {len(state.sat_team)} SAT agents"
    )

    return state


__all__ = [
    "AgentIdentity",
    "NodeIdentity",
    "GenesisState",
    "load_genesis",
    "validate_genesis_hash",
    "load_and_validate_genesis",
]
