"""
Origin Guard — Canonical Node0/Block0 identity enforcement.

Single source of truth for origin projection and fail-closed Node0 startup:
  - sovereign_state/node0_genesis.json
  - sovereign_state/genesis_hash.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .genesis_identity import load_genesis, validate_genesis_hash

NODE_ROLE_ENV = "BIZRA_NODE_ROLE"
ROLE_NODE0 = "node0"
ROLE_NODE = "node"


def normalize_node_role(role: str | None) -> str:
    """Normalize a requested node role to a supported value."""
    if not role:
        return ROLE_NODE
    normalized = role.strip().lower()
    if normalized == ROLE_NODE0:
        return ROLE_NODE0
    return ROLE_NODE


def validate_genesis_chain(state_dir: Path) -> tuple[bool, str]:
    """
    Validate Node0 genesis chain from canonical authority files.

    Returns:
        (is_valid, reason)
    """
    genesis_path = state_dir / "node0_genesis.json"
    hash_path = state_dir / "genesis_hash.txt"

    if not genesis_path.exists():
        return False, f"missing {genesis_path}"
    if not hash_path.exists():
        return False, f"missing {hash_path}"

    try:
        state = load_genesis(state_dir)
    except ValueError as exc:
        return False, f"corrupted genesis identity: {exc}"

    if state is None:
        return False, "genesis identity not loadable"

    if not validate_genesis_hash(state, state_dir):
        return False, "genesis hash validation failed"

    return True, "ok"


def resolve_origin_snapshot(state_dir: Path, role: str) -> dict[str, Any]:
    """
    Resolve canonical origin snapshot for status and receipt projection.

    Node mode is always projected as non-genesis. Node0 mode projects genesis
    identity only when the hash chain is valid.
    """
    normalized_role = normalize_node_role(role)
    snapshot: dict[str, Any] = {
        "designation": "ephemeral_node",
        "genesis_node": False,
        "genesis_block": False,
        "home_base_device": False,
        "authority_source": "genesis_files",
        "hash_validated": False,
    }

    if normalized_role != ROLE_NODE0:
        return snapshot

    is_valid, _reason = validate_genesis_chain(state_dir)
    if not is_valid:
        return snapshot

    state = load_genesis(state_dir)
    if state is None:
        return snapshot

    snapshot.update(
        {
            "designation": ROLE_NODE0,
            "genesis_node": True,
            "genesis_block": True,
            "block_id": "block0",
            "home_base_device": True,
            "node_id": state.node_id,
            "node_name": state.node_name,
            "authority_source": "genesis_files",
            "hash_validated": True,
        }
    )
    return snapshot


def enforce_node0_fail_closed(state_dir: Path, role: str) -> None:
    """Fail closed when Node0 role is requested but genesis chain is invalid."""
    normalized_role = normalize_node_role(role)
    if normalized_role != ROLE_NODE0:
        return

    is_valid, reason = validate_genesis_chain(state_dir)
    if not is_valid:
        raise RuntimeError(
            "Node0 genesis enforcement failed: "
            f"{reason}. Required authority files: "
            f"{state_dir / 'node0_genesis.json'} and {state_dir / 'genesis_hash.txt'}"
        )


__all__ = [
    "NODE_ROLE_ENV",
    "ROLE_NODE0",
    "ROLE_NODE",
    "normalize_node_role",
    "validate_genesis_chain",
    "resolve_origin_snapshot",
    "enforce_node0_fail_closed",
]
