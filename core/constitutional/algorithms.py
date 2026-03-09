"""
15 Native Algorithms — Three Minds v2
═════════════════════════════════════

Constitutional kernel algorithms implementing Ihsan-based economics,
progressive minting, soulbound governance, and social cohesion.

All arithmetic uses fixed-point integers. No floating-point. No drift.

Standing on Giants:
- Al-Ghazali (1058-1111): Intent as ethical pre-gate (Niyyah)
- Ibn Khaldun (1332-1406): Asabiyyah + progressive inequality response
- Al-Khwarizmi (780-850): Algorithm as deterministic procedure
- Kahneman (2002): System 1/2 cognitive architecture

Phase 67.02 — Sovereign Instantiation
"""

from __future__ import annotations

import hashlib
import json
import time

from core.constitutional.fixed_point import (
    FP_ONE,
    FP_ZERO,
    fp,
    fp_add,
    fp_clamp,
    fp_div,
    fp_mul,
    fp_sub,
)
from core.constitutional.types import (
    ActionReceipt,
    Event,
    Proposal,
    Reflex,
    WalletState,
)
from core.integration.constants import (
    ASABIYYAH_COUPLING_CEIL,
    ASABIYYAH_COUPLING_FLOOR,
    ASABIYYAH_NEUTRAL,
    EQUITY_FACTOR_MAX,
    EQUITY_FACTOR_MIN,
)
from core.integration.constants import GINI_CRISIS as GINI_CRISIS_FLOAT
from core.integration.constants import GINI_HEALTHY as GINI_HEALTHY_FLOAT
from core.integration.constants import GINI_WARNING as GINI_WARNING_FLOAT
from core.integration.constants import INTENT_FLOOR as INTENT_FLOOR_FLOAT
from core.integration.constants import NISAB_THRESHOLD as NISAB_FLOAT
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    ZAKAT_RATE,
)

# ═══════════════════════════════════════════════════════════════════
# Algorithm Constants (derived from constants.py)
# ═══════════════════════════════════════════════════════════════════

IHSAN_FLOOR: int = fp(UNIFIED_IHSAN_THRESHOLD)  # 0.95 → 950_000
INTENT_FLOOR: int = fp(INTENT_FLOOR_FLOAT)  # 0.90 → 900_000 (SSoT: constants.py)

# Ihsan scoring weights (must sum to 1.0)
W_INTENT: int = fp(0.25)
W_EFFICIENCY: int = fp(0.25)
W_IMPACT: int = fp(0.30)
W_REPRODUCIBILITY: int = fp(0.20)

# SEED minting
BASE_MINT: int = fp(1.0)

# BLOOM governance
BLOOM_ACCRUAL: int = fp(0.01)  # Per high-ihsan action
BLOOM_DECAY_RATE: int = fp(0.01)  # Per tick of inactivity

# Gini thresholds (Khaldunian Curve) — sourced from constants.py SSoT
GINI_HEALTHY: int = fp(GINI_HEALTHY_FLOAT)  # 0.30 → 300_000
GINI_WARNING: int = fp(GINI_WARNING_FLOAT)  # 0.50 → 500_000
GINI_CRISIS: int = fp(GINI_CRISIS_FLOAT)  # 0.70 → 700_000

# Zakat — sourced from constants.py SSoT
ZAKAT_FP: int = fp(ZAKAT_RATE)  # 2.5% → 25_000
NISAB_THRESHOLD: int = fp(NISAB_FLOAT)  # 85.0 → 85_000_000

# Demurrage
DEMURRAGE_FP: int = fp(0.001)  # 0.1% per tick

# Tick interval (1 hour in ms)
TICK_INTERVAL: int = 3_600_000

# Asabiyyah weights: (reciprocal_attestations, governance, cooperation)
ASABIYYAH_W_RECIPROCAL: int = fp(0.4)
ASABIYYAH_W_GOVERNANCE: int = fp(0.3)
ASABIYYAH_W_COOPERATION: int = fp(0.3)

# V1 Red Team fix: Anti-collusion — require minimum unique connections
# before reciprocal attestation ratio can contribute to Asabiyyah.
# Prevents 2-node collusion rings from gaming social cohesion.
MIN_CONNECTIONS: int = 3

# Asabiyyah-Gini coupling (Phase 69 Sprint 1)
ASAB_FLOOR: int = fp(ASABIYYAH_COUPLING_FLOOR)  # 0.80 → 800_000
ASAB_CEIL: int = fp(ASABIYYAH_COUPLING_CEIL)  # 1.20 → 1_200_000
ASAB_NEUTRAL: int = fp(ASABIYYAH_NEUTRAL)  # 0.50 → 500_000


# ═══════════════════════════════════════════════════════════════════
# A1: Ihsan Scorer (with Al-Ghazali Intent Gate)
# ═══════════════════════════════════════════════════════════════════


def intent_gate(receipt: ActionReceipt) -> bool:
    """Al-Ghazali's correction: intent MUST pass before computation.

    Pre-gate — not a weight. If intent < 0.90, the receipt is rejected
    before any resource is spent on scoring.

    >>> from core.constitutional.types import ActionReceipt
    >>> r = ActionReceipt(b'',b'','',0,fp(0.95),0,0,0,b'',b'')
    >>> intent_gate(r)
    True
    """
    return receipt.intent_score >= INTENT_FLOOR


def ihsan_score(receipt: ActionReceipt) -> int:
    """Compute Ihsan quality score. Returns fixed-point [0, 1.0].

    Precondition: intent_gate(receipt) == True

    >>> from core.constitutional.types import ActionReceipt
    >>> r = ActionReceipt(b'',b'','',0,fp(0.95),fp(0.90),fp(0.92),fp(0.88),b'',b'')
    >>> ihsan_score(r) > 0
    True
    """
    return fp_clamp(
        fp_mul(W_INTENT, receipt.intent_score)
        + fp_mul(W_EFFICIENCY, receipt.efficiency_score)
        + fp_mul(W_IMPACT, receipt.impact_score)
        + fp_mul(W_REPRODUCIBILITY, receipt.reproducibility_score),
        FP_ZERO,
        FP_ONE,
    )


def full_ihsan_check(receipt: ActionReceipt) -> tuple[bool, int]:
    """Combined gate + score. Returns (passed, score).

    Three Minds correction: Al-Ghazali gate fires BEFORE scoring.
    """
    if not intent_gate(receipt):
        return (False, FP_ZERO)

    score = ihsan_score(receipt)
    return (score >= IHSAN_FLOOR, score)


# ═══════════════════════════════════════════════════════════════════
# A2: SEED Minter (Proof of Impact)
# ═══════════════════════════════════════════════════════════════════


def mint_seed(receipt: ActionReceipt, ihsan: int) -> int:
    """Mint SEED tokens from verified work.

    I-4: Only work of verified excellence produces value.
    """
    if ihsan < IHSAN_FLOOR:
        return FP_ZERO

    # Efficiency bonus: faster work = more value
    efficiency_bonus = fp_div(receipt.efficiency_score, FP_ONE)
    return fp_mul(BASE_MINT, fp_add(FP_ONE, fp_div(efficiency_bonus, fp(2))))


# ═══════════════════════════════════════════════════════════════════
# A3: BLOOM Accumulator (Soulbound Governance)
# ═══════════════════════════════════════════════════════════════════


def accrue_bloom(wallet: WalletState, ihsan: int) -> int:
    """Accrue BLOOM from sustained excellence. Soulbound — cannot transfer.

    I-5: Governance belongs to those who participate.
    """
    if ihsan >= IHSAN_FLOOR:
        return fp_add(wallet.bloom_balance, BLOOM_ACCRUAL)
    return wallet.bloom_balance


def decay_bloom(wallet: WalletState, current_time: int) -> int:
    """Decay BLOOM for inactive nodes. Use it or lose it.

    Al-Ghazali: governance without contribution is empty authority.
    """
    if wallet.last_active == 0 or current_time <= wallet.last_active:
        return wallet.bloom_balance

    ticks_idle = (current_time - wallet.last_active) // TICK_INTERVAL
    if ticks_idle <= 0:
        return wallet.bloom_balance

    decay = fp_mul(BLOOM_DECAY_RATE, fp(ticks_idle))
    return fp_sub(wallet.bloom_balance, decay)


# ═══════════════════════════════════════════════════════════════════
# A4: Gini Enforcer (Khaldunian Curve + Ghazali Equity Factor)
# ═══════════════════════════════════════════════════════════════════


def compute_gini(balances: list[int]) -> int:
    """Compute Gini coefficient in fixed-point.

    Returns fp value in [0, 1.0]. 0 = perfect equality, 1 = total concentration.

    >>> compute_gini([fp(100)] * 10)  # Perfect equality
    0
    """
    n = len(balances)
    if n <= 1:
        return FP_ZERO

    sorted_b = sorted(balances)
    total = sum(sorted_b)
    if total == 0:
        return FP_ZERO

    weighted_sum = 0
    for i, b in enumerate(sorted_b):
        weighted_sum += (2 * (i + 1) - n - 1) * b

    # Gini = weighted_sum / (n * total)
    return fp_div(weighted_sum, fp_mul(fp(n), total))


def asabiyyah_adjustment(asabiyyah: int) -> int:
    """Asabiyyah-Gini coupling multiplier (Phase 69 Sprint 1).

    Closes the feedback loop: social cohesion modulates minting rate.
    - asabiyyah = 0.0 → multiplier = 0.80 (fragmented: throttle down)
    - asabiyyah = 0.5 → multiplier = 1.00 (neutral: no effect)
    - asabiyyah = 1.0 → multiplier = 1.20 (cohesive: boost up)

    Linear interpolation in fixed-point:
      multiplier = FLOOR + (CEIL - FLOOR) * clamp(asabiyyah, 0, 1)

    >>> fp_float(asabiyyah_adjustment(FP_ZERO))  # Fragmented
    0.8
    >>> fp_float(asabiyyah_adjustment(ASAB_NEUTRAL))  # Neutral
    1.0
    >>> fp_float(asabiyyah_adjustment(FP_ONE))  # Cohesive
    1.2
    """
    clamped = fp_clamp(asabiyyah, FP_ZERO, FP_ONE)
    span = fp_sub(ASAB_CEIL, ASAB_FLOOR)  # 400_000 (0.40)
    return fp_add(ASAB_FLOOR, fp_mul(span, clamped))


def khaldunian_throttle(gini: int, asabiyyah: int = FP_ZERO) -> int:
    """Ibn Khaldun's progressive throttle with Asabiyyah coupling.

    v1 BUG: Binary gate (gini > 0.35 -> mint 0) caused economic death.
    v2 FIX: Progressive curve maintains activity while converging to equality.
    v3 (Phase 69): Asabiyyah modulates the throttle — cohesive networks
    mint more, fragmented networks mint less.

    T8 proved: v2 earns 238 SEED vs v1's 0.00 SEED (23,844x improvement).

    >>> khaldunian_throttle(fp(0.20))  # Healthy
    1000000
    >>> khaldunian_throttle(fp(0.80)) > 0  # Never zero
    True
    """
    # Step 1: Compute base Gini throttle
    if gini <= GINI_HEALTHY:
        base = FP_ONE  # Healthy: full minting
    elif gini <= GINI_WARNING:
        # Warning zone: quadratic dropoff from 1.0 → 0.10 (meets crisis entry)
        # Wider zone (0.30-0.50) gives smoother Khaldunian transition
        range_size = fp_sub(GINI_WARNING, GINI_HEALTHY)
        excess = fp_sub(gini, GINI_HEALTHY)
        ratio = fp_div(excess, range_size)
        squared = fp_mul(ratio, ratio)
        # Interpolate from FP_ONE (1.0) down to fp(0.10) (crisis entry)
        span = fp_sub(FP_ONE, fp(0.10))  # 0.90
        base = fp_add(fp(0.10), fp_mul(span, fp_sub(FP_ONE, squared)))
    elif gini <= GINI_CRISIS:
        base = fp(0.10)  # Crisis: minimal minting
    else:
        base = fp(0.01)  # Extreme: near-zero but never zero

    # Step 2: Apply Asabiyyah coupling (Phase 69)
    # Cohesive networks get a boost, fragmented networks get throttled further
    adjustment = asabiyyah_adjustment(asabiyyah)
    return fp_mul(base, adjustment)


def ghazali_equity_factor(wallet: WalletState, mean_balance: int) -> int:
    """Newcomer advantage multiplier.

    Those below the mean earn MORE per unit of work.
    T9 proved: 3.27x for newcomers vs wealthy nodes.

    I-3: Wealth concentration shall not exceed Gini 0.35.
    """
    if mean_balance == 0:
        return FP_ONE

    if wallet.seed_balance >= mean_balance:
        return FP_ONE  # At or above mean: standard rate

    if wallet.seed_balance == 0:
        return fp(EQUITY_FACTOR_MAX)  # Maximum newcomer boost

    ratio = fp_div(mean_balance, wallet.seed_balance)
    return fp_clamp(ratio, fp(EQUITY_FACTOR_MIN), fp(EQUITY_FACTOR_MAX))


def progressive_mint(
    receipt: ActionReceipt,
    ihsan: int,
    wallet: WalletState,
    network_gini: int,
    mean_balance: int,
    network_asabiyyah: int = FP_ZERO,
) -> int:
    """Full minting pipeline with all corrections applied.

    Three Minds integrated:
    1. Al-Ghazali: Intent gate (already passed if we're here)
    2. Ibn Khaldun: Khaldunian throttle on network Gini + Asabiyyah coupling
    3. Al-Khwarizmi: All math in fixed-point
    """
    base = mint_seed(receipt, ihsan)
    if base == 0:
        return 0

    throttle = khaldunian_throttle(network_gini, network_asabiyyah)
    equity = ghazali_equity_factor(wallet, mean_balance)

    return fp_mul(fp_mul(base, throttle), equity)


# ═══════════════════════════════════════════════════════════════════
# A5: Zakat Engine (Annual Purification)
# ═══════════════════════════════════════════════════════════════════


def compute_zakat(wallet: WalletState) -> int:
    """I-7: Wealth above threshold shall be purified annually.

    Deterministic: exactly 2.5% of balance above nisab.
    """
    if wallet.seed_balance < NISAB_THRESHOLD:
        return FP_ZERO  # Below nisab: exempt

    return fp_mul(wallet.seed_balance, ZAKAT_FP)


# ═══════════════════════════════════════════════════════════════════
# A6: Backing Ratio (Reserve Health)
# ═══════════════════════════════════════════════════════════════════


def backing_ratio(total_seed: int, total_verified_work: int) -> int:
    """Every SEED must be backed by verified work.

    Ratio < 1.0 = inflation. Ratio = 1.0 = perfect. Ratio > 1.0 = deflation.
    """
    if total_seed == 0:
        return FP_ONE
    return fp_div(total_verified_work, total_seed)


# ═══════════════════════════════════════════════════════════════════
# A7: Demurrage (Idle Tax)
# ═══════════════════════════════════════════════════════════════════


def apply_demurrage(wallet: WalletState, current_time: int) -> int:
    """Tax idle wealth to incentivize circulation.

    Active nodes (recent action within TICK_INTERVAL): exempt.
    Idle nodes: lose 0.1% per tick.
    """
    if wallet.last_active == 0 or current_time <= wallet.last_active:
        return wallet.seed_balance

    ticks_idle = (current_time - wallet.last_active) // TICK_INTERVAL
    if ticks_idle <= 0:
        return wallet.seed_balance  # Active: no demurrage

    fee = fp_mul(wallet.seed_balance, fp_mul(DEMURRAGE_FP, fp(ticks_idle)))
    return fp_sub(wallet.seed_balance, fee)


# ═══════════════════════════════════════════════════════════════════
# A8: Shura Governance (BLOOM-Weighted Voting)
# ═══════════════════════════════════════════════════════════════════


def shura_vote(proposal: Proposal, voter: WalletState, approve: bool) -> Proposal:
    """BLOOM-weighted governance. Soulbound = earned, not bought.

    I-5: Governance belongs to those who participate.
    """
    weight = voter.bloom_balance
    if weight == 0:
        return proposal  # No governance stake = no vote

    if approve:
        proposal.votes_for = fp_add(proposal.votes_for, weight)
    else:
        proposal.votes_against = fp_add(proposal.votes_against, weight)

    return proposal


def shura_resolve(proposal: Proposal) -> str:
    """Resolve proposal by BLOOM-weighted supermajority (>66.7%)."""
    total = fp_add(proposal.votes_for, proposal.votes_against)
    if total == 0:
        return "expired"  # No participation

    approval_ratio = fp_div(proposal.votes_for, total)
    if approval_ratio > fp(0.667):  # Supermajority
        return "passed"
    return "rejected"


# ═══════════════════════════════════════════════════════════════════
# A9: Trust Monitor
# ═══════════════════════════════════════════════════════════════════


def trust_score(wallet: WalletState) -> int:
    """Composite trust score from historical ihsan performance."""
    if len(wallet.ihsan_history) == 0:
        return FP_ZERO

    total = sum(wallet.ihsan_history)
    avg = fp_div(total, fp(len(wallet.ihsan_history)))

    # Consistency bonus: low variance = higher trust
    variance = _compute_variance(wallet.ihsan_history, avg)
    consistency = fp_sub(FP_ONE, fp_clamp(variance, FP_ZERO, FP_ONE))

    return fp_mul(avg, fp_add(FP_ONE, fp_div(consistency, fp(2))))


def _compute_variance(values: list[int], mean: int) -> int:
    """Fixed-point variance computation."""
    if len(values) <= 1:
        return FP_ZERO
    total_sq_diff = 0
    for v in values:
        diff = v - mean if v >= mean else mean - v
        total_sq_diff += fp_mul(diff, diff)
    return fp_div(total_sq_diff, fp(len(values)))


# ═══════════════════════════════════════════════════════════════════
# A10: Reflex Compiler (System-1 Cache)
# ═══════════════════════════════════════════════════════════════════


def compile_reflex(
    pattern: str, action_chain: list[str], confidence: int
) -> Reflex | None:
    """Compile a verified pattern into O(1) cached reflex.

    Kahneman System-1: 90% of interactions hit cache.

    V5 Red Team hardening: Ihsan gate at compile time.
    Low-quality reflexes must never enter the cache — poisoned
    cache entries corrupt System-1 decisions downstream.
    """
    # V5: Reject reflexes below Ihsan floor at compile time
    if confidence < IHSAN_FLOOR:
        return None

    pattern_hash = hashlib.blake2b(pattern.encode("utf-8"), digest_size=32).digest()
    return Reflex(
        pattern_hash=pattern_hash,
        action_chain=tuple(action_chain),
        confidence=confidence,
        last_used=int(time.time() * 1000),
        use_count=0,
    )


def reflex_lookup(cache: dict[bytes, Reflex], pattern: str) -> Reflex | None:
    """O(1) hash lookup. Returns None if no cached reflex."""
    key = hashlib.blake2b(pattern.encode("utf-8"), digest_size=32).digest()
    reflex = cache.get(key)
    if reflex is not None and reflex.confidence >= IHSAN_FLOOR:
        return reflex
    return None


# ═══════════════════════════════════════════════════════════════════
# A14: Event Sourcer (Immutable History)
# ═══════════════════════════════════════════════════════════════════


def _canonical_bytes(
    event_id: int, event_type: str, data: dict, prev_hash: bytes
) -> bytes:
    """Deterministic canonical byte representation for hashing."""
    content = json.dumps(
        {"id": event_id, "type": event_type, "data": data},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return prev_hash + content


def append_event(
    log: list[Event],
    event_type: str,
    actor: bytes,
    data: dict,
) -> Event:
    """Append event to immutable log with hash chain.

    Integrity: each event includes hash of previous event.
    """
    prev_hash = log[-1].hash if log else b"\x00" * 32
    event_id = len(log)

    content = _canonical_bytes(event_id, event_type, data, prev_hash)
    event_hash = hashlib.blake2b(content, digest_size=32).digest()

    event = Event(
        event_id=event_id,
        event_type=event_type,
        actor=actor,
        data=data,
        timestamp=int(time.time() * 1000),
        prev_hash=prev_hash,
        hash=event_hash,
    )
    log.append(event)
    return event


def verify_event_chain(log: list[Event]) -> tuple[bool, list[str]]:
    """Verify hash chain integrity of the event log."""
    errors: list[str] = []
    if not log:
        return (True, errors)

    # Check genesis
    if log[0].prev_hash != b"\x00" * 32:
        errors.append("Genesis event has non-zero prev_hash")

    # Check chain links
    for i in range(1, len(log)):
        if log[i].prev_hash != log[i - 1].hash:
            errors.append(f"Chain break at event {i}")

    # Verify each hash
    for event in log:
        content = _canonical_bytes(
            event.event_id, event.event_type, event.data, event.prev_hash
        )
        expected = hashlib.blake2b(content, digest_size=32).digest()
        if event.hash != expected:
            errors.append(f"Hash mismatch at event {event.event_id}")

    return (len(errors) == 0, errors)


# ═══════════════════════════════════════════════════════════════════
# A15: Asabiyyah Index (Social Cohesion)
# ═══════════════════════════════════════════════════════════════════


def asabiyyah_score(wallet: WalletState, network_size: int) -> int:
    """Ibn Khaldun's social cohesion metric for a single node.

    Measures: how connected, how participatory, how cooperative.
    T10 proved: monotonic growth with attestation activity.

    V1 Red Team hardening: MIN_CONNECTIONS anti-collusion gate.
    A 2-node collusion ring can trivially achieve 100% reciprocal
    ratio. Requiring >= 3 unique connections forces genuine
    community participation.
    """
    if network_size <= 1:
        return FP_ZERO

    # V1: Anti-collusion — total unique connections must meet minimum
    total_connections = len(wallet.attestations_given | wallet.attestations_received)
    if total_connections < MIN_CONNECTIONS:
        a_reciprocal = FP_ZERO  # Insufficient connections — reciprocal does not count
    else:
        # Reciprocal attestations (both gave and received)
        reciprocal = len(wallet.attestations_given & wallet.attestations_received)
        max_reciprocal = network_size - 1
        a_reciprocal = fp_div(fp(reciprocal), fp(max_reciprocal))

    # Governance participation (capped at 10 votes = 1.0)
    a_governance = fp_clamp(
        fp_div(fp(wallet.governance_votes), fp(10)), FP_ZERO, FP_ONE
    )

    # Cooperative actions (capped at 20 = 1.0)
    a_cooperative = fp_clamp(
        fp_div(fp(wallet.cooperative_actions), fp(20)), FP_ZERO, FP_ONE
    )

    return (
        fp_mul(ASABIYYAH_W_RECIPROCAL, a_reciprocal)
        + fp_mul(ASABIYYAH_W_GOVERNANCE, a_governance)
        + fp_mul(ASABIYYAH_W_COOPERATION, a_cooperative)
    )


def network_asabiyyah(wallets: list[WalletState]) -> int:
    """Network-wide social cohesion score.

    Average of all individual Asabiyyah scores.
    Ibn Khaldun: "Asabiyyah is the pillar of civilization."
    """
    if len(wallets) == 0:
        return FP_ZERO

    total = sum(asabiyyah_score(w, len(wallets)) for w in wallets)
    return fp_div(total, fp(len(wallets)))
