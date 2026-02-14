"""
Genesis Ceremony — Root of Trust Bootstrap
============================================
Creates the immutable genesis state for a BIZRA node. This is the
Python-side counterpart to bizra-omega/bizra-resourcepool/node0_genesis.rs.

The ceremony produces:
  - sovereign_state/node0_genesis.json  (full identity + agents + hashes)
  - sovereign_state/pat_roster.txt      (7 PAT agents)
  - sovereign_state/sat_roster.txt      (5 SAT agents)
  - sovereign_state/genesis_hash.txt    (root Merkle hash)

Determinism Invariants:
  - Given the same seed, the ceremony produces byte-identical output
  - All hashes use BLAKE3 (SEC-001: Python-Rust interop parity)
  - Canonical JSON encoding: sorted keys, compact separators, UTF-8
  - No datetime.now() in hash paths — timestamp is passed explicitly

Standing on Giants:
- Nakamoto (2008): Genesis block as immutable origin
- Bernstein (2011): Ed25519 for identity keypairs
- Merkle (1979): Hash chains for integrity
- Lamport (1982): Persistent identity in distributed systems
- Al-Ghazali (1095): Self-knowledge precedes all knowledge
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.proof_engine.canonical import blake3_digest, canonical_bytes, hex_digest

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# PAT + SAT Agent Definitions (canonical roles)
# ═══════════════════════════════════════════════════════════════════════════

PAT_ROLES = [
    {
        "role": "Strategist",
        "capabilities": ["planning", "goal-decomposition", "resource-allocation"],
        "giants": ["Sun Tzu", "Mintzberg", "Porter"],
    },
    {
        "role": "Researcher",
        "capabilities": [
            "literature-review",
            "hypothesis-generation",
            "evidence-gathering",
        ],
        "giants": ["Shannon", "Popper", "Kuhn"],
    },
    {
        "role": "Developer",
        "capabilities": ["code-generation", "architecture", "testing"],
        "giants": ["Knuth", "Dijkstra", "Lamport"],
    },
    {
        "role": "Analyst",
        "capabilities": [
            "data-analysis",
            "pattern-recognition",
            "statistical-inference",
        ],
        "giants": ["Tukey", "Fisher", "Bayes"],
    },
    {
        "role": "Reviewer",
        "capabilities": ["code-review", "quality-assurance", "security-audit"],
        "giants": ["Fagan", "Parnas", "Hoare"],
    },
    {
        "role": "Executor",
        "capabilities": ["task-execution", "deployment", "automation"],
        "giants": ["Deming", "Toyota", "Ohno"],
    },
    {
        "role": "Guardian",
        "capabilities": ["safety-check", "constraint-enforcement", "anomaly-detection"],
        "giants": ["Lamport", "Schneier", "Amodei"],
    },
]

SAT_ROLES = [
    {
        "role": "Validator",
        "capabilities": ["proof-verification", "consensus-voting", "attestation"],
        "giants": ["Nakamoto", "Lamport", "Pease"],
    },
    {
        "role": "Oracle",
        "capabilities": ["external-data", "ground-truth", "fact-checking"],
        "giants": ["Shannon", "Delphi", "Szabo"],
    },
    {
        "role": "Mediator",
        "capabilities": ["conflict-resolution", "consensus-building", "governance"],
        "giants": ["Ostrom", "Arrow", "Nash"],
    },
    {
        "role": "Archivist",
        "capabilities": ["knowledge-preservation", "indexing", "retrieval"],
        "giants": ["Berners-Lee", "Nelson", "Bush"],
    },
    {
        "role": "Sentinel",
        "capabilities": ["monitoring", "alerting", "threat-detection"],
        "giants": ["Schneier", "Denning", "Anderson"],
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CeremonyConfig:
    """Configuration for the genesis ceremony."""

    node_name: str = "Node0"
    node_location: str = "Genesis"
    timestamp_ms: int = 0  # 0 = auto (current time in ms)

    # Governance defaults
    quorum: float = 0.67
    voting_period_hours: int = 72
    upgrade_threshold: float = 0.80

    # Hardware attestation (optional)
    hardware: dict[str, Any] = field(default_factory=dict)

    # Knowledge attestation (optional)
    knowledge: dict[str, Any] = field(default_factory=dict)


@dataclass
class CeremonyResult:
    """Output of the genesis ceremony."""

    genesis_json: dict[str, Any]
    genesis_hash: str
    pat_roster: str
    sat_roster: str
    output_dir: Optional[Path] = None


# ═══════════════════════════════════════════════════════════════════════════
# Core Ceremony Logic
# ═══════════════════════════════════════════════════════════════════════════


def _derive_keypair(seed: bytes) -> tuple[bytes, bytes]:
    """Derive an Ed25519 keypair from a seed.

    Uses the seed as the Ed25519 private key seed (32 bytes).
    Returns (private_key_bytes, public_key_hex_string_as_bytes).
    """
    try:
        from nacl.signing import SigningKey

        if len(seed) < 32:
            # Pad with BLAKE3 hash of seed
            seed = blake3_digest(seed)[:32]
        else:
            seed = seed[:32]

        sk = SigningKey(seed)
        pk = sk.verify_key
        return bytes(sk), pk.encode().hex().encode("ascii")
    except ImportError:
        # Fallback: use BLAKE3 hash as deterministic "public key"
        pk_hash = hex_digest(seed)
        return seed[:32], pk_hash.encode("ascii")


def _build_agent_identity(
    role_def: dict[str, Any],
    prefix: str,
    node_seed: bytes,
    timestamp_ms: int,
) -> dict[str, Any]:
    """Build a single agent identity deterministically.

    The agent ID is derived from BLAKE3(node_seed + role), making it
    reproducible given the same node seed.
    """
    role = role_def["role"]
    agent_seed = blake3_digest(node_seed + role.lower().encode("utf-8"))
    agent_id_suffix = hex_digest(agent_seed)[:16]
    agent_id = f"{prefix}_{role.lower()}_{agent_id_suffix[:8]}"

    _, pub_key_bytes = _derive_keypair(agent_seed)
    pub_key = pub_key_bytes.decode("ascii")

    agent_data = {
        "agent_id": agent_id,
        "role": role,
        "public_key": pub_key,
        "capabilities": role_def["capabilities"],
        "giants": role_def["giants"],
        "created_at": timestamp_ms,
    }

    # Compute agent hash
    agent_hash = blake3_digest(canonical_bytes(agent_data))
    agent_data["agent_hash"] = list(agent_hash)

    return agent_data


def _compute_team_hash(agents: list[dict[str, Any]]) -> bytes:
    """Compute Merkle-style hash of an agent team.

    Deterministic: sorted by agent_id, canonical encoding, BLAKE3.
    """
    sorted_agents = sorted(agents, key=lambda a: a["agent_id"])
    canonical = canonical_bytes(
        [
            {"agent_id": a["agent_id"], "agent_hash": a["agent_hash"]}
            for a in sorted_agents
        ]
    )
    return blake3_digest(canonical)


def run_ceremony(
    node_seed: bytes,
    config: Optional[CeremonyConfig] = None,
) -> CeremonyResult:
    """Execute the genesis ceremony.

    This is the main entry point. Given a node seed (32+ bytes), it
    deterministically produces the complete genesis state.

    Args:
        node_seed: Cryptographic seed for the node (32+ bytes).
                   MUST be kept secret — it derives all keypairs.
        config: Optional ceremony configuration.

    Returns:
        CeremonyResult with the complete genesis state.
    """
    if config is None:
        config = CeremonyConfig()

    timestamp_ms = config.timestamp_ms or int(time.time() * 1000)

    # ─── Node Identity ────────────────────────────────────────
    _, node_pub_bytes = _derive_keypair(node_seed)
    node_pub = node_pub_bytes.decode("ascii")
    node_id_suffix = hex_digest(node_seed)[:16]
    node_id = f"node0_{node_id_suffix}"

    identity = {
        "node_id": node_id,
        "public_key": node_pub,
        "name": config.node_name,
        "location": config.node_location,
        "created_at": timestamp_ms,
    }
    identity_hash = blake3_digest(canonical_bytes(identity))
    identity["identity_hash"] = list(identity_hash)

    # ─── PAT Team (Personal Agentic Team) ─────────────────────
    pat_agents = [
        _build_agent_identity(role, "pat", node_seed, timestamp_ms)
        for role in PAT_ROLES
    ]
    pat_team_hash = _compute_team_hash(pat_agents)

    # ─── SAT Team (Sovereign Advisory Team) ───────────────────
    sat_agents = [
        _build_agent_identity(role, "sat", node_seed, timestamp_ms)
        for role in SAT_ROLES
    ]
    sat_team_hash = _compute_team_hash(sat_agents)

    # ─── Partnership Hash ─────────────────────────────────────
    partnership_hash = blake3_digest(
        canonical_bytes(
            {
                "pat_team_hash": list(pat_team_hash),
                "sat_team_hash": list(sat_team_hash),
            }
        )
    )

    # ─── Genesis Hash (root of trust) ─────────────────────────
    genesis_preimage = canonical_bytes(
        {
            "identity_hash": list(identity_hash),
            "pat_team_hash": list(pat_team_hash),
            "sat_team_hash": list(sat_team_hash),
            "partnership_hash": list(partnership_hash),
            "timestamp": timestamp_ms,
        }
    )
    genesis_hash = blake3_digest(genesis_preimage)

    # ─── Assemble Full Genesis JSON ───────────────────────────
    genesis_json: dict[str, Any] = {
        "timestamp": timestamp_ms,
        "identity": identity,
        "hardware": config.hardware,
        "knowledge": config.knowledge,
        "pat_team": {
            "agents": pat_agents,
            "team_hash": list(pat_team_hash),
        },
        "sat_team": {
            "agents": sat_agents,
            "governance": {
                "quorum": config.quorum,
                "voting_period_hours": config.voting_period_hours,
                "upgrade_threshold": config.upgrade_threshold,
            },
            "team_hash": list(sat_team_hash),
        },
        "partnership_hash": list(partnership_hash),
        "genesis_hash": list(genesis_hash),
    }

    # ─── Build Rosters ────────────────────────────────────────
    pat_roster_lines = []
    for a in pat_agents:
        aid = a["agent_id"]
        role = a["role"]
        short = aid.split("_", 2)[-1] if "_" in aid else aid
        pat_roster_lines.append(f"{role}: {aid} ({short})")
    pat_roster = "\n".join(pat_roster_lines) + "\n"

    sat_roster_lines = []
    for a in sat_agents:
        aid = a["agent_id"]
        role = a["role"]
        short = aid.split("_", 2)[-1] if "_" in aid else aid
        sat_roster_lines.append(f"{role}: {aid} ({short})")
    sat_roster = "\n".join(sat_roster_lines) + "\n"

    return CeremonyResult(
        genesis_json=genesis_json,
        genesis_hash=genesis_hash.hex(),
        pat_roster=pat_roster,
        sat_roster=sat_roster,
    )


def write_ceremony(result: CeremonyResult, output_dir: Path) -> Path:
    """Write ceremony output to disk.

    Creates the sovereign_state directory structure:
      - node0_genesis.json
      - genesis_hash.txt
      - pat_roster.txt
      - sat_roster.txt

    Returns the output directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    genesis_path = output_dir / "node0_genesis.json"
    genesis_path.write_text(
        json.dumps(result.genesis_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    hash_path = output_dir / "genesis_hash.txt"
    hash_path.write_text(result.genesis_hash + "\n", encoding="utf-8")

    pat_path = output_dir / "pat_roster.txt"
    pat_path.write_text(result.pat_roster, encoding="utf-8")

    sat_path = output_dir / "sat_roster.txt"
    sat_path.write_text(result.sat_roster, encoding="utf-8")

    result.output_dir = output_dir
    logger.info(
        f"Genesis ceremony complete: {genesis_path} "
        f"(hash: {result.genesis_hash[:16]}...)"
    )

    return output_dir


def verify_ceremony(genesis_path: Path) -> tuple[bool, list[str]]:
    """Verify an existing genesis file's internal integrity.

    Checks:
    1. JSON parses correctly
    2. Identity hash matches recomputed hash
    3. PAT team hash matches recomputed hash
    4. SAT team hash matches recomputed hash
    5. Partnership hash matches recomputed hash
    6. Genesis hash matches recomputed hash

    Returns:
        (is_valid, list_of_reason_codes)
        Reason codes are empty on success, or contain failure descriptions.
    """
    reasons: list[str] = []

    try:
        with open(genesis_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"FILE_CORRUPT: {e}"]

    # 1. Identity hash
    identity = data.get("identity", {})
    stored_identity_hash = bytes(identity.get("identity_hash", []))
    identity_check = {k: v for k, v in identity.items() if k != "identity_hash"}
    computed_identity_hash = blake3_digest(canonical_bytes(identity_check))
    if stored_identity_hash != computed_identity_hash:
        reasons.append(
            f"IDENTITY_HASH_MISMATCH: "
            f"stored={stored_identity_hash.hex()[:16]} "
            f"computed={computed_identity_hash.hex()[:16]}"
        )

    # 2. PAT team hash
    pat_data = data.get("pat_team", {})
    pat_agents = pat_data.get("agents", [])
    stored_pat_hash = bytes(pat_data.get("team_hash", []))
    computed_pat_hash = _compute_team_hash(pat_agents)
    if stored_pat_hash != computed_pat_hash:
        reasons.append(
            f"PAT_TEAM_HASH_MISMATCH: "
            f"stored={stored_pat_hash.hex()[:16]} "
            f"computed={computed_pat_hash.hex()[:16]}"
        )

    # 3. SAT team hash
    sat_data = data.get("sat_team", {})
    sat_agents = sat_data.get("agents", [])
    stored_sat_hash = bytes(sat_data.get("team_hash", []))
    computed_sat_hash = _compute_team_hash(sat_agents)
    if stored_sat_hash != computed_sat_hash:
        reasons.append(
            f"SAT_TEAM_HASH_MISMATCH: "
            f"stored={stored_sat_hash.hex()[:16]} "
            f"computed={computed_sat_hash.hex()[:16]}"
        )

    # 4. Partnership hash
    stored_partnership = bytes(data.get("partnership_hash", []))
    computed_partnership = blake3_digest(
        canonical_bytes(
            {
                "pat_team_hash": list(computed_pat_hash),
                "sat_team_hash": list(computed_sat_hash),
            }
        )
    )
    if stored_partnership != computed_partnership:
        reasons.append(
            f"PARTNERSHIP_HASH_MISMATCH: "
            f"stored={stored_partnership.hex()[:16]} "
            f"computed={computed_partnership.hex()[:16]}"
        )

    # 5. Genesis hash
    stored_genesis = bytes(data.get("genesis_hash", []))
    genesis_preimage = canonical_bytes(
        {
            "identity_hash": list(computed_identity_hash),
            "pat_team_hash": list(computed_pat_hash),
            "sat_team_hash": list(computed_sat_hash),
            "partnership_hash": list(computed_partnership),
            "timestamp": data.get("timestamp", 0),
        }
    )
    computed_genesis = blake3_digest(genesis_preimage)
    if stored_genesis != computed_genesis:
        reasons.append(
            f"GENESIS_HASH_MISMATCH: "
            f"stored={stored_genesis.hex()[:16]} "
            f"computed={computed_genesis.hex()[:16]}"
        )

    is_valid = len(reasons) == 0
    if is_valid:
        logger.info(f"Genesis verification passed: {genesis_path}")
    else:
        logger.warning(f"Genesis verification failed: {reasons}")

    return is_valid, reasons


__all__ = [
    "CeremonyConfig",
    "CeremonyResult",
    "run_ceremony",
    "write_ceremony",
    "verify_ceremony",
]
