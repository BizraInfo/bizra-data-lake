"""
Node0 Authority Resolution & Migration
========================================
Implements the canonical authority chain for MVSA:

  1. canonical  sovereign_state/node0_genesis.json + genesis_hash.txt
  2. legacy     sovereign_state/genesis.json       (migratable)
  3. legacy     bizra-storage/genesis.json          (migratable)
  4. reference  04_GOLD/genesis.json                (NOT sufficient)

Rules (§4 — fail-closed):
- If canonical exists and validates → use it, stop.
- If canonical missing, search legacy in order.
- Legacy is migratable only if it has identity + pat_team + sat_team + genesis_hash.
- 04_GOLD/genesis.json is reference-only — NOT auto-promoted.
- Conflicting migratable sources → LEGACY_GENESIS_CONFLICT (blocked).
- Only reference-only available → LEGACY_GENESIS_INSUFFICIENT (blocked).

Standing on Giants:
- Nakamoto (2008): Genesis block as immutable origin
- Lamport (1978): Persistent identity in distributed systems
- Al-Ghazali (1095): Self-knowledge precedes all knowledge
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .atomic_io import atomic_write_json, read_json
from .genesis_identity import GenesisState, load_genesis, validate_genesis_hash

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Result codes
# ═══════════════════════════════════════════════════════════════════════════════
RESULT_CANONICAL = "canonical_valid"
RESULT_MIGRATED = "migrated"
RESULT_SKIPPED = "skipped"
RESULT_BLOCKED = "blocked"

REASON_CANONICAL_VALID = "CANONICAL_AUTHORITY_VALID"
REASON_LEGACY_MIGRATED = "LEGACY_GENESIS_MIGRATED"
REASON_LEGACY_CONFLICT = "LEGACY_GENESIS_CONFLICT"
REASON_LEGACY_INSUFFICIENT = "LEGACY_GENESIS_INSUFFICIENT"
REASON_NO_AUTHORITY = "NO_AUTHORITY_FOUND"

SOURCE_CANONICAL = "canonical"
SOURCE_LEGACY_CEREMONY = "legacy_ceremony"
SOURCE_LEGACY_REFERENCE = "legacy_reference"


@dataclass
class AuthorityResult:
    """Outcome of authority resolution."""

    genesis: Optional[GenesisState]
    result: str  # canonical_valid | migrated | skipped | blocked
    reason_code: str
    source_path: Optional[str] = None
    source_kind: Optional[str] = None
    genesis_hash_hex: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.result in (RESULT_CANONICAL, RESULT_MIGRATED)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_ceremony_compatible(data: dict[str, Any]) -> bool:
    """Check if a legacy genesis JSON has the full ceremony structure."""
    identity = data.get("identity")
    if not isinstance(identity, dict) or not identity.get("node_id"):
        return False
    pat = data.get("pat_team")
    if not isinstance(pat, dict) or not pat.get("agents"):
        return False
    sat = data.get("sat_team")
    if not isinstance(sat, dict) or not sat.get("agents"):
        return False
    if not data.get("genesis_hash"):
        return False
    return True


def _genesis_hash_hex(data: dict[str, Any]) -> str:
    """Extract genesis hash as hex string regardless of storage format."""
    raw = data.get("genesis_hash", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return bytes(raw).hex()
    if isinstance(raw, bytes):
        return raw.hex()
    return ""


def _write_migration_receipt(
    state_dir: Path,
    source_path: str,
    source_kind: str,
    result: str,
    reason_code: str,
    genesis_hash: str,
) -> None:
    """Persist migration audit trail."""
    receipt = {
        "schema_version": "1.0.0",
        "migrated_at": _utc_now(),
        "source_path": source_path,
        "source_kind": source_kind,
        "result": result,
        "reason_code": reason_code,
        "genesis_hash": genesis_hash,
    }
    atomic_write_json(state_dir / "node0_authority_migration.json", receipt)


def _migrate_legacy_to_canonical(
    source_path: Path, state_dir: Path, data: dict[str, Any]
) -> None:
    """Copy a ceremony-compatible legacy genesis into canonical location."""
    canonical_genesis = state_dir / "node0_genesis.json"
    atomic_write_json(canonical_genesis, data)

    genesis_hash = _genesis_hash_hex(data)
    (state_dir / "genesis_hash.txt").write_text(genesis_hash + "\n", encoding="utf-8")

    # Extract rosters
    pat_agents = data.get("pat_team", {}).get("agents", [])
    sat_agents = data.get("sat_team", {}).get("agents", [])
    pat_lines = [a.get("agent_id", f"PAT-{i}") for i, a in enumerate(pat_agents)]
    sat_lines = [a.get("agent_id", f"SAT-{i}") for i, a in enumerate(sat_agents)]
    (state_dir / "pat_roster.txt").write_text("\n".join(pat_lines) + "\n", encoding="utf-8")
    (state_dir / "sat_roster.txt").write_text("\n".join(sat_lines) + "\n", encoding="utf-8")

    logger.info("Migrated legacy genesis from %s → canonical %s", source_path, canonical_genesis)


def resolve_authority(
    state_dir: Path,
    project_root: Path,
) -> AuthorityResult:
    """
    Resolve Node0 authority following the strict precedence chain.

    This is the single entry point for MVSA authority resolution.
    """
    # ── Step 1: Check canonical authority ──────────────────────────
    canonical_path = state_dir / "node0_genesis.json"
    hash_path = state_dir / "genesis_hash.txt"

    if canonical_path.exists() and hash_path.exists():
        try:
            genesis = load_genesis(state_dir)
            if genesis is not None:
                hash_valid = validate_genesis_hash(genesis, state_dir)
                if hash_valid:
                    gh = genesis.genesis_hash.hex() if genesis.genesis_hash else ""
                    logger.info("Canonical authority valid: %s", genesis.node_id)
                    return AuthorityResult(
                        genesis=genesis,
                        result=RESULT_CANONICAL,
                        reason_code=REASON_CANONICAL_VALID,
                        source_path=str(canonical_path),
                        source_kind=SOURCE_CANONICAL,
                        genesis_hash_hex=gh,
                    )
                # Hash mismatch — fall through to legacy search
                logger.warning("Canonical genesis hash mismatch — searching legacy sources")
        except ValueError as exc:
            logger.warning("Canonical genesis corrupted (%s) — searching legacy sources", exc)

    # ── Step 2: Search legacy sources ─────────────────────────────
    legacy_candidates: list[tuple[Path, str]] = [
        (state_dir / "genesis.json", SOURCE_LEGACY_CEREMONY),
        (project_root / "bizra-storage" / "genesis.json", SOURCE_LEGACY_CEREMONY),
    ]
    reference_path = project_root / "04_GOLD" / "genesis.json"

    migratable: list[tuple[Path, str, dict[str, Any], str]] = []

    for path, kind in legacy_candidates:
        if not path.exists():
            continue
        data = read_json(path)
        if data is None:
            continue
        if _is_ceremony_compatible(data):
            gh = _genesis_hash_hex(data)
            migratable.append((path, kind, data, gh))
            logger.info("Found migratable legacy: %s (hash=%s…)", path, gh[:16])

    has_reference = False
    if reference_path.exists():
        ref_data = read_json(reference_path)
        if ref_data is not None:
            if _is_ceremony_compatible(ref_data):
                gh = _genesis_hash_hex(ref_data)
                migratable.append((reference_path, SOURCE_LEGACY_REFERENCE, ref_data, gh))
            else:
                has_reference = True

    # ── Step 3: Migration decision ────────────────────────────────
    if not migratable:
        reason = REASON_LEGACY_INSUFFICIENT if has_reference else REASON_NO_AUTHORITY
        _write_migration_receipt(state_dir, "", "", RESULT_BLOCKED, reason, "")
        return AuthorityResult(
            genesis=None,
            result=RESULT_BLOCKED,
            reason_code=reason,
        )

    # Filter out reference-only sources for migration candidates
    ceremony_sources = [(p, k, d, h) for p, k, d, h in migratable if k != SOURCE_LEGACY_REFERENCE]

    if not ceremony_sources:
        # Only reference-only sources exist
        _write_migration_receipt(
            state_dir, str(reference_path), SOURCE_LEGACY_REFERENCE,
            RESULT_BLOCKED, REASON_LEGACY_INSUFFICIENT, "",
        )
        return AuthorityResult(
            genesis=None,
            result=RESULT_BLOCKED,
            reason_code=REASON_LEGACY_INSUFFICIENT,
            source_path=str(reference_path),
            source_kind=SOURCE_LEGACY_REFERENCE,
        )

    # Check for conflicts among ceremony sources
    unique_hashes = {h for _, _, _, h in ceremony_sources}
    if len(unique_hashes) > 1:
        _write_migration_receipt(
            state_dir, str(ceremony_sources[0][0]), SOURCE_LEGACY_CEREMONY,
            RESULT_BLOCKED, REASON_LEGACY_CONFLICT,
            "|".join(sorted(unique_hashes)),
        )
        return AuthorityResult(
            genesis=None,
            result=RESULT_BLOCKED,
            reason_code=REASON_LEGACY_CONFLICT,
        )

    # Single hash — migrate highest-precedence source
    source_path, source_kind, source_data, source_hash = ceremony_sources[0]
    _migrate_legacy_to_canonical(source_path, state_dir, source_data)
    _write_migration_receipt(
        state_dir, str(source_path), source_kind,
        "migrated", REASON_LEGACY_MIGRATED, source_hash,
    )

    # Re-load from canonical location to get proper GenesisState
    genesis = load_genesis(state_dir)
    return AuthorityResult(
        genesis=genesis,
        result=RESULT_MIGRATED,
        reason_code=REASON_LEGACY_MIGRATED,
        source_path=str(source_path),
        source_kind=source_kind,
        genesis_hash_hex=source_hash,
    )


def require_authority(state_dir: Path, project_root: Path) -> GenesisState:
    """
    Resolve authority, raising RuntimeError if blocked.

    This is the fail-closed entry point for activate() and prove-mvsa.
    """
    result = resolve_authority(state_dir, project_root)
    if not result.is_valid or result.genesis is None:
        raise RuntimeError(
            f"Node0 authority resolution failed: {result.reason_code}. "
            f"Required: sovereign_state/node0_genesis.json + genesis_hash.txt"
        )
    return result.genesis


__all__ = [
    "AuthorityResult",
    "resolve_authority",
    "require_authority",
    "RESULT_CANONICAL",
    "RESULT_MIGRATED",
    "RESULT_BLOCKED",
    "REASON_CANONICAL_VALID",
    "REASON_LEGACY_MIGRATED",
    "REASON_LEGACY_CONFLICT",
    "REASON_LEGACY_INSUFFICIENT",
    "REASON_NO_AUTHORITY",
]
