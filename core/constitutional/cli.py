"""
Sovereignty CLI — Production Genesis Interface
═══════════════════════════════════════════════

Pure-logic module for the 4 core commands + 2 utilities.
No interactive I/O — all functions accept parameters and return results.
A thin entry point (__main__) handles terminal formatting.

Standing on Giants:
- Thompson & Ritchie (1979): Unix philosophy — one tool, one job
- Al-Khwarizmi (780-850): Every command is a deterministic procedure
- Nakamoto (2008): Local-first, offline-first sovereignty

Phase 67.04 — Sovereign Instantiation
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.constitutional.algorithms import (
    accrue_bloom,
    asabiyyah_score,
    compute_gini,
    full_ihsan_check,
    intent_gate,
    khaldunian_throttle,
    progressive_mint,
)
from core.constitutional.declaration import DECLARATION_BLAKE2B_256
from core.constitutional.fixed_point import (
    FP_ZERO,
    fp,
    fp_add,
    fp_div,
    fp_float,
)
from core.constitutional.types import ActionReceipt, WalletState

# ═══════════════════════════════════════════════════════════════════
# Result Types
# ═══════════════════════════════════════════════════════════════════


@dataclass
class InitResult:
    """Result of bizra init."""

    success: bool
    node_id: str = ""
    covenant_hash: str = ""
    error: str = ""


@dataclass
class WorkResult:
    """Result of bizra work."""

    success: bool
    intent_passed: bool = False
    ihsan_score: float = 0.0
    seed_minted: float = 0.0
    throttle: float = 0.0
    equity_factor: float = 0.0
    receipt_hash: str = ""
    error: str = ""


@dataclass
class AttestResult:
    """Result of bizra attest."""

    success: bool
    attestation_hash: str = ""
    error: str = ""


@dataclass
class StatusResult:
    """Result of bizra status."""

    success: bool
    name: str = ""
    node_id: str = ""
    seed_balance: int = 0
    bloom_balance: int = 0
    total_actions: int = 0
    avg_ihsan: float = 0.0
    asabiyyah_score: float = 0.0
    covenant_hash: str = ""
    attestations_given: int = 0
    attestations_received: int = 0
    peers_count: int = 0
    age_days: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════════════════════════
# Node State (JSON-serializable)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class NodeState:
    """Persistent node state stored as JSON."""

    name: str
    node_id: str
    public_key: str
    covenant_hash: str
    covenant_sig: str
    seed_balance: int = 0
    bloom_balance: int = 0
    total_actions: int = 0
    ihsan_history: list[int] = field(default_factory=list)
    created_at: int = 0
    last_active: int = 0
    attestations_given: int = 0
    attestations_received: int = 0
    governance_votes: int = 0
    cooperative_actions: int = 0
    peers: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# State I/O
# ═══════════════════════════════════════════════════════════════════


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)


def _blake2b_hex(data: bytes) -> str:
    """BLAKE2b-256 hex digest."""
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def save_node_state(state: NodeState, state_dir: Path) -> None:
    """Persist node state to disk."""
    state_dir.mkdir(parents=True, exist_ok=True)
    node_file = state_dir / "node.json"
    node_file.write_text(json.dumps(asdict(state), indent=2))


def load_node_state(state_dir: Path) -> NodeState | None:
    """Load node state from disk. Returns None if not initialized."""
    node_file = state_dir / "node.json"
    if not node_file.exists():
        return None
    data = json.loads(node_file.read_text())
    return NodeState(**data)


def _append_ledger(state_dir: Path, event: dict) -> None:
    """Append event to the ledger (append-only)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = state_dir / "ledger.jsonl"
    with open(ledger_file, "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


# ═══════════════════════════════════════════════════════════════════
# Scoring Heuristic (Alpha — production uses oracle validators)
# ═══════════════════════════════════════════════════════════════════


def _score_description(description: str) -> tuple[int, int, int, int]:
    """Heuristic scoring for alpha phase. Returns (intent, efficiency, impact, reproducibility).

    In production, these scores come from oracle validators.
    For alpha, we use word count and keyword presence as proxies.
    """
    words = len(description.split())
    technical_keywords = {
        "code",
        "review",
        "audit",
        "design",
        "build",
        "research",
        "teach",
        "write",
        "fix",
        "create",
        "analyze",
        "test",
        "document",
        "optimize",
        "implement",
        "deploy",
        "contribute",
        "algorithm",
        "module",
        "kernel",
        "constitutional",
        "verified",
    }
    has_technical = any(w.lower() in technical_keywords for w in description.split())

    # Intent: high for descriptive, specific work
    intent = min(
        fp(0.99), fp(0.92 + min(words, 20) * 0.003 + (0.02 if has_technical else 0))
    )
    # Efficiency: based on specificity
    efficiency = min(fp(0.99), fp(0.94 + min(words, 30) * 0.002))
    # Impact: based on actionable content
    impact = min(
        fp(0.99), fp(0.93 + min(words, 25) * 0.002 + (0.02 if has_technical else 0))
    )
    # Reproducibility: based on clarity
    reproducibility = min(fp(0.99), fp(0.95 + min(words, 15) * 0.002))

    return intent, efficiency, impact, reproducibility


# ═══════════════════════════════════════════════════════════════════
# Command: Init
# ═══════════════════════════════════════════════════════════════════


def init_node(name: str, state_dir: Path) -> InitResult:
    """Initialize a sovereign node.

    Generates Ed25519 keypair (simulated), signs the Covenant,
    creates genesis event.
    """
    # Check if already initialized
    if load_node_state(state_dir) is not None:
        return InitResult(
            success=False,
            error="Already initialized. Run status to see current state.",
        )

    ts = _now_ms()
    seed_bytes = os.urandom(32)
    node_id = _blake2b_hex(seed_bytes)
    public_key = _blake2b_hex(seed_bytes + b"public")
    covenant_sig = _blake2b_hex((node_id + DECLARATION_BLAKE2B_256).encode())

    state = NodeState(
        name=name,
        node_id=node_id,
        public_key=public_key,
        covenant_hash=DECLARATION_BLAKE2B_256,
        covenant_sig=covenant_sig,
        created_at=ts,
        last_active=ts,
    )

    save_node_state(state, state_dir)

    _append_ledger(
        state_dir,
        {
            "type": "genesis",
            "node_id": node_id,
            "name": name,
            "covenant_hash": DECLARATION_BLAKE2B_256,
            "covenant_sig": covenant_sig,
            "timestamp": ts,
        },
    )

    return InitResult(
        success=True,
        node_id=node_id,
        covenant_hash=DECLARATION_BLAKE2B_256,
    )


# ═══════════════════════════════════════════════════════════════════
# Command: Work
# ═══════════════════════════════════════════════════════════════════


def process_work(description: str, state_dir: Path) -> WorkResult:
    """Submit verified work. Score it. Mint SEED if it passes.

    Uses the production constitutional algorithms:
    1. Intent gate (Al-Ghazali)
    2. Ihsan scoring
    3. Progressive minting with Khaldunian curve
    4. BLOOM accrual
    """
    state = load_node_state(state_dir)
    if state is None:
        return WorkResult(success=False, error="Not initialized. Run init first.")

    if not description or not description.strip():
        return WorkResult(success=False, error="Work description required.")

    ts = _now_ms()

    # Score the work description (alpha heuristic)
    intent, efficiency, impact, reproducibility = _score_description(description)

    # Create a receipt
    receipt_data = f"{state.node_id}:{description}:{ts}"
    receipt_id = hashlib.blake2b(receipt_data.encode(), digest_size=32).digest()

    receipt = ActionReceipt(
        receipt_id=receipt_id,
        actor_id=bytes.fromhex(state.node_id),
        action_type="contribution",
        timestamp=ts,
        intent_score=intent,
        efficiency_score=efficiency,
        impact_score=impact,
        reproducibility_score=reproducibility,
        oracle_signature=b"\x00" * 64,  # Self-signed in alpha
        metadata_hash=hashlib.blake2b(description.encode(), digest_size=32).digest(),
    )

    # Step 1: Intent gate
    if not intent_gate(receipt):
        return WorkResult(
            success=False,
            intent_passed=False,
            error="Intent gate failed. Work description needs more clarity.",
        )

    # Step 2: Ihsan scoring
    passed, ihsan = full_ihsan_check(receipt)
    if not passed:
        return WorkResult(
            success=False,
            intent_passed=True,
            ihsan_score=fp_float(ihsan),
            error=f"Ihsan score {fp_float(ihsan):.4f} below 0.95 floor.",
        )

    # Step 3: Progressive minting
    # Build network context (single-node alpha)
    all_balances = [state.seed_balance] if state.seed_balance > 0 else [FP_ZERO]
    gini = compute_gini(all_balances)
    mean_balance = (
        fp_div(sum(all_balances), fp(len(all_balances))) if all_balances else FP_ZERO
    )

    # Build wallet for minting
    wallet = WalletState(
        node_id=bytes.fromhex(state.node_id),
        seed_balance=state.seed_balance,
        bloom_balance=state.bloom_balance,
        last_active=state.last_active,
        total_actions=state.total_actions,
        ihsan_history=list(state.ihsan_history),
        created_at=state.created_at,
    )

    minted = progressive_mint(receipt, ihsan, wallet, gini, mean_balance)

    # Step 4: BLOOM accrual
    bloom = accrue_bloom(wallet, ihsan)

    # Step 5: Compute throttle for result reporting
    throttle = khaldunian_throttle(gini)

    # Update state
    state.seed_balance = fp_add(state.seed_balance, minted)
    state.bloom_balance = bloom
    state.total_actions += 1
    state.last_active = ts
    state.ihsan_history.append(ihsan)

    save_node_state(state, state_dir)

    # Log event
    _append_ledger(
        state_dir,
        {
            "type": "action",
            "receipt_id": receipt_id.hex(),
            "node_id": state.node_id,
            "description": description[:200],
            "ihsan_score": fp_float(ihsan),
            "seed_minted": fp_float(minted),
            "bloom_accrued": fp_float(bloom),
            "timestamp": ts,
        },
    )

    return WorkResult(
        success=True,
        intent_passed=True,
        ihsan_score=fp_float(ihsan),
        seed_minted=fp_float(minted),
        throttle=fp_float(throttle),
        receipt_hash=receipt_id.hex(),
    )


# ═══════════════════════════════════════════════════════════════════
# Command: Attest
# ═══════════════════════════════════════════════════════════════════


def attest_peer(peer_id: str, state_dir: Path) -> AttestResult:
    """Attest another node's work. Builds Asabiyyah.

    Al-Ghazali filter: Only nodes with sufficient ihsan history
    can attest. Low-quality mutual attestation builds nothing.
    """
    state = load_node_state(state_dir)
    if state is None:
        return AttestResult(success=False, error="Not initialized. Run init first.")

    # Cannot self-attest
    if peer_id == state.node_id or peer_id == state.name:
        return AttestResult(
            success=False,
            error="Cannot attest yourself. Asabiyyah requires the Other.",
        )

    # Must have work history
    if not state.ihsan_history:
        return AttestResult(
            success=False,
            error="No work history. Do at least one verified action first.",
        )

    ts = _now_ms()
    att_hash = _blake2b_hex(f"{state.node_id}:{peer_id}:{ts}".encode())

    # Update state
    state.attestations_given += 1
    if peer_id not in state.peers:
        state.peers.append(peer_id)
    state.last_active = ts

    save_node_state(state, state_dir)

    # Log event
    _append_ledger(
        state_dir,
        {
            "type": "attestation",
            "attestation_id": att_hash,
            "attester": state.node_id,
            "attestee": peer_id,
            "timestamp": ts,
        },
    )

    return AttestResult(success=True, attestation_hash=att_hash)


# ═══════════════════════════════════════════════════════════════════
# Command: Status
# ═══════════════════════════════════════════════════════════════════


def get_status(state_dir: Path) -> StatusResult:
    """Compute and return the full sovereign state."""
    state = load_node_state(state_dir)
    if state is None:
        return StatusResult(success=False, error="Not initialized. Run init first.")

    ts = _now_ms()
    age_days = (
        (ts - state.created_at) / (24 * 60 * 60 * 1000) if state.created_at else 0
    )

    # Average Ihsan
    avg_ihsan = 0.0
    if state.ihsan_history:
        recent = state.ihsan_history[-30:]
        avg_ihsan = fp_float(sum(recent) // len(recent))

    # Asabiyyah — build wallet for scoring
    wallet = WalletState(
        node_id=bytes.fromhex(state.node_id),
        total_actions=state.total_actions,
        governance_votes=state.governance_votes,
        cooperative_actions=state.cooperative_actions,
        attestations_given={p.encode() for p in state.peers} if state.peers else set(),
    )
    asab = fp_float(asabiyyah_score(wallet, max(len(state.peers) + 1, 1)))

    return StatusResult(
        success=True,
        name=state.name,
        node_id=state.node_id,
        seed_balance=state.seed_balance,
        bloom_balance=state.bloom_balance,
        total_actions=state.total_actions,
        avg_ihsan=avg_ihsan,
        asabiyyah_score=asab,
        covenant_hash=state.covenant_hash,
        attestations_given=state.attestations_given,
        attestations_received=state.attestations_received,
        peers_count=len(state.peers),
        age_days=age_days,
    )
