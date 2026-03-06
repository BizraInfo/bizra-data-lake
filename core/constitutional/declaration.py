"""
Declaration of Digital Sovereignty — Genesis Block Handler
══════════════════════════════════════════════════════════

The Declaration is not a document — it is the genesis block of the BIZRA
event log. Every ActionReceipt chains back to this hash. If a node's local
Declaration hash doesn't match, its receipts are invalid.

This creates Covenant Locking: the constitutional invariants (I-1 through I-7)
are cryptographically anchored to the genesis state.

Standing on Giants:
- Nakamoto (2008): Genesis block as immutable origin
- Merkle (1979): Hash chains for integrity
- Jefferson (1776): Declaration of Independence as founding document
- Al-Ghazali (1095): The covenant precedes all action

Phase 67.03 — Sovereign Instantiation
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from core.constitutional.fixed_point import fp
from core.constitutional.types import Event
from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    ZAKAT_RATE,
)

# ═══════════════════════════════════════════════════════════════════
# Canonical Declaration Hash
# ═══════════════════════════════════════════════════════════════════
# Computed from 00_CONSTITUTION/DECLARATION.md (UTF-8, Unix LF).
# This is a CONSTANT, never recomputed at runtime.
DECLARATION_BLAKE2B_256: str = (
    "17d672371bb8eff01676fdec010fef17f20f624bf1b3557d39057a0a16b65fe8"
)

# Path to the canonical Declaration file (relative to repo root)
DECLARATION_PATH: Path = Path("00_CONSTITUTION/DECLARATION.md")


# ═══════════════════════════════════════════════════════════════════
# Seven Constitutional Invariants
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConstitutionalInvariant:
    """An immutable constitutional guarantee enforced by algorithm."""

    code: str  # "I-1" through "I-7"
    guarantee: str  # Human-readable guarantee
    algorithm: str  # Which algorithm enforces it
    threshold: int | None  # Fixed-point threshold (if applicable)


INVARIANTS: tuple[ConstitutionalInvariant, ...] = (
    ConstitutionalInvariant(
        code="I-1",
        guarantee="No interest shall exist at any layer",
        algorithm="A2_SEED_MINTER",
        threshold=None,
    ),
    ConstitutionalInvariant(
        code="I-2",
        guarantee="No transaction shall contain hidden uncertainty",
        algorithm="A14_EVENT_SOURCER",
        threshold=None,
    ),
    ConstitutionalInvariant(
        code="I-3",
        guarantee="Wealth concentration shall not exceed Gini 0.35",
        algorithm="A4_GINI_ENFORCER",
        threshold=fp(ADL_GINI_THRESHOLD),
    ),
    ConstitutionalInvariant(
        code="I-4",
        guarantee="Only work of verified excellence produces value",
        algorithm="A1_IHSAN_SCORER",
        threshold=fp(UNIFIED_IHSAN_THRESHOLD),
    ),
    ConstitutionalInvariant(
        code="I-5",
        guarantee="Governance belongs to those who participate",
        algorithm="A8_SHURA_GOVERNANCE",
        threshold=None,
    ),
    ConstitutionalInvariant(
        code="I-6",
        guarantee="Every node shall be sovereign over its data",
        algorithm="SOVEREIGNTY_KERNEL",
        threshold=None,
    ),
    ConstitutionalInvariant(
        code="I-7",
        guarantee="Wealth above threshold shall be purified annually",
        algorithm="A5_ZAKAT_ENGINE",
        threshold=fp(ZAKAT_RATE),
    ),
)


# ═══════════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════════


class ConstitutionalViolation(Exception):
    """Raised when a constitutional invariant is violated.

    This exception represents a hard stop. The node must halt and
    report the violation.
    """


def load_declaration(path: Path | None = None) -> str:
    """Load the Declaration text from the canonical file.

    Normalizes line endings to Unix LF for consistent hashing.
    """
    p = path or DECLARATION_PATH
    if not p.exists():
        raise FileNotFoundError(f"Declaration not found at {p}")
    text = p.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n")


def compute_declaration_hash(text: str) -> str:
    """Compute the BLAKE2b-256 hash of the Declaration text."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()


def verify_declaration_hash(text: str) -> bool:
    """Verify the Declaration hash matches the canonical value.

    Any modification to the Declaration — even a single byte — will
    cause this check to fail, preventing covenant-broken nodes from
    participating in the network.
    """
    computed = compute_declaration_hash(text)
    return computed == DECLARATION_BLAKE2B_256


def create_genesis_event(declaration_text: str) -> Event:
    """Create the genesis event (Event #0) from the Declaration.

    This is the root of the Merkle chain. All subsequent events
    chain back to this hash.
    """
    if not verify_declaration_hash(declaration_text):
        raise ConstitutionalViolation("Declaration hash mismatch — covenant broken")

    content = declaration_text.encode("utf-8")
    genesis_hash = hashlib.blake2b(content, digest_size=32).digest()

    return Event(
        event_id=0,
        event_type="genesis",
        actor=b"\x00" * 32,  # System actor (no individual)
        data={
            "declaration_hash_blake2b": DECLARATION_BLAKE2B_256,
            "invariants": [inv.code for inv in INVARIANTS],
            "version": "1.0.0",
            "proclaimed": "2026-03-05T00:00:00Z",
            "location": "Dubai",
        },
        timestamp=0,  # Genesis timestamp = epoch
        prev_hash=b"\x00" * 32,  # No previous event
        hash=genesis_hash,
    )


def verify_covenant_chain(
    event_log: list[Event],
) -> tuple[bool, list[str]]:
    """Verify that the event log chains back to the Declaration genesis.

    Returns (valid, list_of_errors).
    """
    errors: list[str] = []

    if len(event_log) == 0:
        return (False, ["Empty event log — no genesis"])

    genesis = event_log[0]
    if genesis.event_type != "genesis":
        errors.append("First event is not genesis")

    if genesis.data.get("declaration_hash_blake2b") != DECLARATION_BLAKE2B_256:
        errors.append("Genesis declaration hash does not match canonical value")

    # Verify chain integrity (A14)
    for i in range(1, len(event_log)):
        if event_log[i].prev_hash != event_log[i - 1].hash:
            errors.append(f"Chain break at event {i}")

    return (len(errors) == 0, errors)
