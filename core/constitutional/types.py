"""
Constitutional Data Types
═════════════════════════

Immutable data structures for the constitutional kernel.
All numeric fields use fixed-point integers (6 decimal places).

Standing on Giants:
- Al-Khwarizmi (780-850): Deterministic data representation
- Hoare (1969): Data invariants as types

Phase 67.02 — Sovereign Instantiation
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionReceipt:
    """Immutable record of a verified human action.

    All scores are fixed-point integers (fp(0.95) = 950000).
    """

    receipt_id: bytes  # BLAKE3 hash of content
    actor_id: bytes  # Ed25519 public key
    action_type: str  # "contribution" | "attestation" | "governance" | ...
    timestamp: int  # Unix ms
    intent_score: int  # fp — Al-Ghazali gate (must be >= INTENT_FLOOR)
    efficiency_score: int  # fp
    impact_score: int  # fp
    reproducibility_score: int  # fp
    oracle_signature: bytes  # Ed25519 signature by validator
    metadata_hash: bytes  # BLAKE3 of metadata
    co_actors: tuple[bytes, ...] = ()  # Ed25519 keys of collaborators


@dataclass
class WalletState:
    """Sovereign economic state of a node.

    Identity = reduction of all events (A14).
    """

    node_id: bytes  # Ed25519 public key
    seed_balance: int = 0  # Fixed-point SEED token balance
    bloom_balance: int = 0  # Fixed-point BLOOM (soulbound)
    last_active: int = 0  # Last action timestamp
    total_actions: int = 0  # Lifetime action count
    ihsan_history: list[int] = field(default_factory=list)
    created_at: int = 0  # Node creation timestamp
    attestations_given: set[bytes] = field(default_factory=set)
    attestations_received: set[bytes] = field(default_factory=set)
    governance_votes: int = 0  # Lifetime governance votes
    cooperative_actions: int = 0  # Cooperative action count


@dataclass
class Proposal:
    """Governance proposal for Shura (A8)."""

    proposal_id: bytes
    proposer: bytes  # Ed25519 key
    description: str
    votes_for: int = 0  # BLOOM-weighted
    votes_against: int = 0
    status: str = "active"  # active | passed | rejected | expired
    created_at: int = 0


@dataclass(frozen=True)
class Reflex:
    """Compiled intelligence pattern for O(1) lookup (A10)."""

    pattern_hash: bytes  # BLAKE3 of input pattern
    action_chain: tuple[str, ...]  # Pre-compiled action sequence
    confidence: int  # Fixed-point confidence score
    last_used: int = 0  # Timestamp
    use_count: int = 0


@dataclass(frozen=True)
class Attestation:
    """Mutual attestation record for Asabiyyah (A12/A15)."""

    attester: bytes
    attestee: bytes
    receipt_id: bytes  # Receipt being attested
    timestamp: int
    signature: bytes  # Ed25519 signature


@dataclass
class Event:
    """Append-only event for the immutable log (A14)."""

    event_id: int  # Sequential
    event_type: str  # "mint" | "transfer" | "vote" | ...
    actor: bytes
    data: dict = field(default_factory=dict)
    timestamp: int = 0
    prev_hash: bytes = b"\x00" * 32  # Chain link
    hash: bytes = b"\x00" * 32  # BLAKE3(event_id + type + data + prev_hash)
