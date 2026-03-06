"""
Constitutional Ticker — 12-Step Heartbeat
═════════════════════════════════════════

One tick of the sovereignty kernel. Runs all constitutional algorithms
in deterministic order. This is the game loop that runs 24/7 without
any LLM dependency.

Standing on Giants:
- Al-Khwarizmi (780-850): Deterministic procedure
- Nakamoto (2008): Block processing tick
- Kahneman (2002): System-1/System-2 split

Phase 67.02 — Sovereign Instantiation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from core.constitutional.algorithms import (
    FP_ZERO,
    accrue_bloom,
    append_event,
    apply_demurrage,
    compile_reflex,
    compute_gini,
    compute_zakat,
    decay_bloom,
    ihsan_score,
    intent_gate,
    network_asabiyyah,
    progressive_mint,
    shura_resolve,
)
from core.constitutional.fixed_point import fp, fp_add, fp_div, fp_float, fp_sub
from core.constitutional.types import (
    ActionReceipt,
    Event,
    Proposal,
    Reflex,
    WalletState,
)


@dataclass
class TickResult:
    """Output of a single process_tick() execution."""

    rejected: int = 0  # Receipts rejected by intent gate
    scored: int = 0  # Receipts that passed scoring
    total_minted: int = 0  # Total SEED minted this tick
    zakat_pool: int = 0  # Zakat collected this tick
    network_gini: int = 0  # Current Gini coefficient
    network_asabiyyah_score: int = 0  # Current social cohesion
    events_logged: int = 0  # Events appended to log
    proposals_resolved: int = 0  # Governance proposals resolved


def _find_wallet(wallets: list[WalletState], actor_id: bytes) -> WalletState | None:
    """Find wallet by node_id. O(n) — acceptable for tick-level processing."""
    for w in wallets:
        if w.node_id == actor_id:
            return w
    return None


def process_tick(
    wallets: list[WalletState],
    receipts: list[ActionReceipt],
    proposals: list[Proposal],
    event_log: list[Event],
    reflex_cache: dict[bytes, Reflex],
    current_time: int | None = None,
    is_zakat_cycle: bool = False,
) -> TickResult:
    """One heartbeat of the constitutional kernel.

    12 steps, deterministic, reproducible.
    All algorithms run in constitutional order.

    Args:
        wallets: All network wallets (mutable — balances updated in-place)
        receipts: New action receipts to process this tick
        proposals: Active governance proposals
        event_log: Immutable event log (append-only)
        reflex_cache: System-1 pattern cache
        current_time: Unix ms timestamp (defaults to now)
        is_zakat_cycle: Whether this tick triggers annual Zakat collection
    """
    if current_time is None:
        current_time = int(time.time() * 1000)

    result = TickResult()

    # ──────────────────────────────────────────────────────────────
    # Step 1: Al-Ghazali Intent Gate — reject low-intent receipts
    # ──────────────────────────────────────────────────────────────
    valid_receipts = [r for r in receipts if intent_gate(r)]
    result.rejected = len(receipts) - len(valid_receipts)

    # ──────────────────────────────────────────────────────────────
    # Step 2: Ihsan Scoring — compute quality for valid receipts
    # ──────────────────────────────────────────────────────────────
    scored: list[tuple[ActionReceipt, int]] = []
    for r in valid_receipts:
        score = ihsan_score(r)
        scored.append((r, score))
    result.scored = len(scored)

    # ──────────────────────────────────────────────────────────────
    # Step 3: Compute network Gini
    # ──────────────────────────────────────────────────────────────
    balances = [w.seed_balance for w in wallets]
    gini = compute_gini(balances) if wallets else FP_ZERO
    mean_balance = fp_div(sum(balances), fp(len(wallets))) if wallets else FP_ZERO

    # ──────────────────────────────────────────────────────────────
    # Step 3.5: Compute Asabiyyah BEFORE minting (Phase 69 Sprint 1)
    # Closes the Khaldunian feedback loop: cohesion modulates minting
    # ──────────────────────────────────────────────────────────────
    asabiyyah = network_asabiyyah(wallets) if wallets else FP_ZERO

    # ──────────────────────────────────────────────────────────────
    # Step 4: Progressive Minting — SEED creation with all corrections
    # ──────────────────────────────────────────────────────────────
    for receipt, ihsan in scored:
        wallet = _find_wallet(wallets, receipt.actor_id)
        if wallet is None:
            continue
        minted = progressive_mint(receipt, ihsan, wallet, gini, mean_balance, asabiyyah)
        wallet.seed_balance = fp_add(wallet.seed_balance, minted)
        wallet.total_actions += 1
        wallet.last_active = current_time
        wallet.ihsan_history.append(ihsan)
        result.total_minted += minted

    # ──────────────────────────────────────────────────────────────
    # Step 5: BLOOM Accrual — governance token growth
    # ──────────────────────────────────────────────────────────────
    for receipt, ihsan in scored:
        wallet = _find_wallet(wallets, receipt.actor_id)
        if wallet is None:
            continue
        wallet.bloom_balance = accrue_bloom(wallet, ihsan)

    # ──────────────────────────────────────────────────────────────
    # Step 6: BLOOM Decay — inactive governance weight reduction
    # ──────────────────────────────────────────────────────────────
    for wallet in wallets:
        wallet.bloom_balance = decay_bloom(wallet, current_time)

    # ──────────────────────────────────────────────────────────────
    # Step 7: Demurrage — idle balance tax
    # ──────────────────────────────────────────────────────────────
    for wallet in wallets:
        wallet.seed_balance = apply_demurrage(wallet, current_time)

    # ──────────────────────────────────────────────────────────────
    # Step 8: Zakat Collection — annual purification
    # ──────────────────────────────────────────────────────────────
    if is_zakat_cycle:
        for wallet in wallets:
            zakat_due = compute_zakat(wallet)
            wallet.seed_balance = fp_sub(wallet.seed_balance, zakat_due)
            result.zakat_pool += zakat_due

    # ──────────────────────────────────────────────────────────────
    # Step 9: Governance — resolve expired proposals
    # ──────────────────────────────────────────────────────────────
    for proposal in proposals:
        if proposal.status == "active":
            new_status = shura_resolve(proposal)
            if new_status != "expired":
                proposal.status = new_status
                result.proposals_resolved += 1

    # ──────────────────────────────────────────────────────────────
    # Step 10: Reflex Cache — compile excellent patterns
    # ──────────────────────────────────────────────────────────────
    for receipt, ihsan in scored:
        if ihsan >= fp(0.98):  # Only excellent work becomes reflex
            reflex = compile_reflex(
                receipt.action_type,
                [receipt.action_type],
                ihsan,
            )
            reflex_cache[reflex.pattern_hash] = reflex

    # ──────────────────────────────────────────────────────────────
    # Step 11: Event Logging — immutable history
    # ──────────────────────────────────────────────────────────────
    for receipt, ihsan in scored:
        append_event(
            event_log,
            "mint",
            receipt.actor_id,
            {
                "receipt_id": receipt.receipt_id.hex(),
                "ihsan": fp_float(ihsan),
            },
        )
        result.events_logged += 1

    # ──────────────────────────────────────────────────────────────
    # Step 12: Asabiyyah — network cohesion (reuse Step 3.5 value)
    # ──────────────────────────────────────────────────────────────
    result.network_asabiyyah_score = asabiyyah
    result.network_gini = gini

    return result
