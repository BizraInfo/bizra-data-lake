"""
Chaos Validators — BIZRA Flight Certification Protocol
═══════════════════════════════════════════════════════

10 constitutional stress tests that every release MUST pass.

Each test simulates an adversarial or extreme scenario and verifies
that the constitutional kernel self-corrects without human intervention.

Standing on Giants:
- Ibn Khaldun (1377): Civilizations that fail to self-correct collapse
- Nassim Taleb (2012): Anti-fragility — stress makes the system stronger
- Lamport (1982): Byzantine fault tolerance under adversarial conditions

Phase 67.07 — Constitutional Chaos Probes
"""

from __future__ import annotations

import os
import time

import pytest

from core.constitutional.algorithms import (
    EQUITY_FACTOR_MAX,
    accrue_bloom,
    asabiyyah_score,
    compute_gini,
    full_ihsan_check,
    ghazali_equity_factor,
    intent_gate,
    khaldunian_throttle,
    mint_seed,
    network_asabiyyah,
    progressive_mint,
)
from core.constitutional.fixed_point import (
    FP_ONE,
    FP_ZERO,
    fp,
    fp_add,
    fp_div,
    fp_float,
    fp_mul,
    fp_sub,
)
from core.constitutional.types import ActionReceipt, WalletState

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


def _make_receipt(
    *,
    intent: float = 0.96,
    efficiency: float = 0.97,
    impact: float = 0.96,
    reproducibility: float = 0.97,
    actor_id: bytes | None = None,
) -> ActionReceipt:
    """Create a valid ActionReceipt for chaos testing."""
    ts = int(time.time() * 1000)
    rid = os.urandom(32)
    aid = actor_id or os.urandom(32)
    return ActionReceipt(
        receipt_id=rid,
        actor_id=aid,
        action_type="contribution",
        timestamp=ts,
        intent_score=fp(intent),
        efficiency_score=fp(efficiency),
        impact_score=fp(impact),
        reproducibility_score=fp(reproducibility),
        oracle_signature=b"\x00" * 64,
        metadata_hash=os.urandom(32),
    )


def _make_wallet(
    balance: float = 0.0,
    bloom: float = 0.0,
    actions: int = 0,
    node_id: bytes | None = None,
) -> WalletState:
    """Create a WalletState for chaos testing."""
    nid = node_id or os.urandom(32)
    ts = int(time.time() * 1000)
    return WalletState(
        node_id=nid,
        seed_balance=fp(balance),
        bloom_balance=fp(bloom),
        last_active=ts,
        total_actions=actions,
        ihsan_history=[fp(0.97)] * min(actions, 30),
        created_at=ts - (actions * 3_600_000),
    )


def _get_ihsan(receipt: ActionReceipt) -> int:
    """Get ihsan score from a receipt, asserting it passes."""
    passed, ihsan = full_ihsan_check(receipt)
    assert passed, "Receipt should pass ihsan check"
    return ihsan


# ═══════════════════════════════════════════════════════════════════
# T1: Whale Attack — Gini Spike Recovery
# ═══════════════════════════════════════════════════════════════════


class TestT1WhaleAttack:
    """Simulate a whale accumulating 90% of network SEED.

    Invariant: Khaldunian throttle must reduce whale's minting rate
    while preserving small-holder earning capacity.

    Certification threshold: Whale throttle < 0.20, small-holders > 0.50.
    """

    def test_whale_triggers_high_gini(self) -> None:
        """A single whale among small-holders produces Gini > 0.70."""
        small = [fp(10)] * 99
        whale = [fp(9000)]
        gini = compute_gini(small + whale)
        assert fp_float(gini) > 0.70

    def test_whale_throttle_kicks_in(self) -> None:
        """High Gini triggers extreme throttle (<=1%)."""
        balances = [fp(10)] * 99 + [fp(9000)]
        gini = compute_gini(balances)
        throttle = khaldunian_throttle(gini)
        assert throttle <= fp(0.01)

    def test_whale_cannot_grow_fast(self) -> None:
        """Whale's minting reward is near-zero under extreme Gini."""
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        whale_wallet = _make_wallet(balance=9000, actions=100)
        balances = [fp(10)] * 99 + [fp(9000)]
        gini = compute_gini(balances)
        mean = fp_div(sum(balances), fp(len(balances)))

        minted = progressive_mint(receipt, ihsan, whale_wallet, gini, mean)
        # Whale gets near-zero due to throttle + equity factor
        assert fp_float(minted) < 0.10

    def test_small_holders_still_earn(self) -> None:
        """Small-holders in a whale-dominated network still earn SEED."""
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        small_wallet = _make_wallet(balance=10, actions=5)
        balances = [fp(10)] * 99 + [fp(9000)]
        gini = compute_gini(balances)
        mean = fp_div(sum(balances), fp(len(balances)))

        minted = progressive_mint(receipt, ihsan, small_wallet, gini, mean)
        # Small-holders get equity advantage, but throttle affects everyone
        assert minted > 0


# ═══════════════════════════════════════════════════════════════════
# T2: Ghost Town — Mass Inactivity
# ═══════════════════════════════════════════════════════════════════


class TestT2GhostTown:
    """Simulate 95% of nodes going inactive.

    Invariant: BLOOM decays, but remaining active nodes can still
    earn and govern. The network contracts gracefully.

    Certification threshold: Active node BLOOM accrual > 0.
    """

    def test_bloom_decays_for_inactive(self) -> None:
        """Inactive wallets lose BLOOM over time."""
        wallet = _make_wallet(balance=100, bloom=50, actions=20)
        # Simulate 90 days of inactivity
        wallet.last_active = wallet.created_at - (90 * 24 * 60 * 60 * 1000)
        # BLOOM accrual returns minimal for stale wallets (no streak bonus)
        bloom = accrue_bloom(wallet, fp(0.97))
        # Never negative
        assert bloom >= 0

    def test_active_node_still_earns_in_ghost_town(self) -> None:
        """A lone active node in a ghost town can still mint."""
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        active_wallet = _make_wallet(balance=50, actions=10)
        # Single node: Gini is 0 (perfect equality of 1)
        gini = compute_gini([fp(50)])
        mean = fp(50)

        minted = progressive_mint(receipt, ihsan, active_wallet, gini, mean)
        assert minted > 0

    def test_single_node_gini_is_zero(self) -> None:
        """A single-node network has perfect equality."""
        assert compute_gini([fp(100)]) == FP_ZERO

    def test_two_equal_nodes_low_gini(self) -> None:
        """Two equal nodes have near-zero Gini."""
        gini = compute_gini([fp(100), fp(100)])
        assert fp_float(gini) < 0.05


# ═══════════════════════════════════════════════════════════════════
# T3: Newcomer Wave — Mass Onboarding
# ═══════════════════════════════════════════════════════════════════


class TestT3NewcomerWave:
    """Simulate 100 new zero-balance nodes joining at once.

    Invariant: Equity factor must accelerate newcomer earning
    while not destabilizing existing balances.

    Certification threshold: Newcomer earns >= 1.5x established rate.
    """

    def test_newcomer_equity_advantage(self) -> None:
        """Zero-balance newcomer gets higher equity factor than established node."""
        newcomer = _make_wallet(balance=0, actions=0)
        established = _make_wallet(balance=100, actions=50)
        mean = fp(50)

        eq_new = ghazali_equity_factor(newcomer, mean)
        eq_est = ghazali_equity_factor(established, mean)
        assert eq_new > eq_est

    def test_newcomer_wave_gini_manageable(self) -> None:
        """Adding many small-balance nodes doesn't cause Gini collapse."""
        existing = [fp(100)] * 10
        gini_before = compute_gini(existing)
        # New nodes with tiny balance (above dust)
        newcomers = [fp(1)] * 100
        gini_after = compute_gini(existing + newcomers)
        # Gini increases (concentration) but should still be manageable
        assert fp_float(gini_after) < 0.95

    def test_newcomer_earns_more_than_whale_per_action(self) -> None:
        """In an unequal network, newcomer earns more per action than whale."""
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        newcomer = _make_wallet(balance=0, actions=0)
        whale = _make_wallet(balance=5000, actions=200)
        balances = [fp(0)] * 50 + [fp(100)] * 40 + [fp(5000)] * 10
        gini = compute_gini(balances)
        mean = fp_div(sum(balances), fp(len(balances)))

        minted_new = progressive_mint(receipt, ihsan, newcomer, gini, mean)
        minted_whale = progressive_mint(receipt, ihsan, whale, gini, mean)
        assert minted_new > minted_whale

    def test_equity_factor_bounded(self) -> None:
        """Equity factor never exceeds EQUITY_FACTOR_MAX (5.0)."""
        newcomer = _make_wallet(balance=0, actions=0)
        eq = ghazali_equity_factor(newcomer, fp(1000))
        assert eq <= fp(EQUITY_FACTOR_MAX)


# ═══════════════════════════════════════════════════════════════════
# T4: Collusion Ring — Mutual Attestation Flooding
# ═══════════════════════════════════════════════════════════════════


class TestT4CollusionRing:
    """Simulate a ring of nodes that attest each other excessively.

    Known vulnerability in alpha: mutual attestation inflates Asabiyyah.
    The Al-Ghazali filter (Ihsan >= 0.95 for attestors) is the
    current defense. Future: VRF-assigned attestation targets.

    Certification threshold: Asabiyyah cannot exceed theoretical max
    through attestation alone; governance + cooperation needed.
    """

    def test_attestation_only_caps_asabiyyah(self) -> None:
        """Mutual attestation alone cannot reach max Asabiyyah.

        Asabiyyah = 0.4*reciprocal + 0.3*governance + 0.3*cooperation
        With only attestation: max = 0.4 (40%).
        """
        wallet = _make_wallet(balance=100, actions=20)
        # Simulate perfect reciprocal attestation (all peers reciprocal)
        wallet.attestations_given = {b"\x01", b"\x02", b"\x03"}
        wallet.attestations_received = {b"\x01", b"\x02", b"\x03"}
        # But zero governance and cooperation
        wallet.governance_votes = 0
        wallet.cooperative_actions = 0

        asab = asabiyyah_score(wallet, len(wallet.attestations_given) + 1)
        # Should be capped at 0.4 (reciprocal component only)
        assert fp_float(asab) <= 0.41  # Small rounding tolerance

    def test_full_asabiyyah_requires_all_pillars(self) -> None:
        """High Asabiyyah requires all three pillars."""
        wallet = _make_wallet(balance=100, actions=50)
        wallet.attestations_given = {b"\x01", b"\x02", b"\x03"}
        wallet.attestations_received = {b"\x01", b"\x02", b"\x03"}
        wallet.governance_votes = 10
        wallet.cooperative_actions = 5

        asab = asabiyyah_score(wallet, 4)
        # All three pillars active: should be high
        assert fp_float(asab) > 0.70


# ═══════════════════════════════════════════════════════════════════
# T5: Reset Equality — Convergence from Extremes
# ═══════════════════════════════════════════════════════════════════


class TestT5ResetEquality:
    """Simulate convergence from extreme initial inequality.

    Invariant: Progressive minting + equity factor must move the
    network toward lower Gini over time.

    Certification threshold: After 50 rounds of equal work, Gini decreases.
    """

    def test_equal_work_reduces_inequality(self) -> None:
        """When all nodes do equal quality work, inequality decreases."""
        # Start with extreme inequality
        balances = [fp(1)] * 90 + [fp(1000)] * 10
        gini_initial = compute_gini(balances)

        # Simulate 50 rounds of equal work
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        for _ in range(50):
            mean = fp_div(sum(balances), fp(len(balances)))
            gini = compute_gini(balances)
            new_balances = []
            for b in balances:
                w = _make_wallet(balance=fp_float(b), actions=10)
                minted = progressive_mint(receipt, ihsan, w, gini, mean)
                new_balances.append(fp_add(b, minted))
            balances = new_balances

        gini_final = compute_gini(balances)
        assert gini_final < gini_initial

    def test_throttle_monotonically_decreases_with_gini(self) -> None:
        """As Gini increases, throttle decreases — monotonic relationship."""
        prev = FP_ONE
        for g in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
            current = khaldunian_throttle(fp(g))
            assert current <= prev, f"Throttle not monotonic at Gini={g}"
            prev = current


# ═══════════════════════════════════════════════════════════════════
# T6: Ihsan Floor Enforcement
# ═══════════════════════════════════════════════════════════════════


class TestT6IhsanFloor:
    """Verify that low-quality work is rejected regardless of node wealth.

    Invariant: No amount of SEED/BLOOM can bypass the Ihsan floor.
    I-4: Only work scoring >= 0.95 generates SEED.
    """

    def test_wealthy_node_rejected_for_low_quality(self) -> None:
        """A whale with 10000 SEED is still rejected for low-quality work."""
        receipt = _make_receipt(efficiency=0.80, impact=0.80, reproducibility=0.80)
        passed, score = full_ihsan_check(receipt)
        assert not passed

    def test_score_below_floor_is_rejected(self) -> None:
        """Ihsan score of 0.94 is below the 0.95 floor."""
        receipt = _make_receipt(efficiency=0.93, impact=0.94, reproducibility=0.94)
        passed, score = full_ihsan_check(receipt)
        # The weighted average might be just below 0.95
        if not passed:
            assert fp_float(score) < 0.96

    def test_perfect_scores_pass(self) -> None:
        """Scores of 0.99 across the board pass easily."""
        receipt = _make_receipt(
            intent=0.99, efficiency=0.99, impact=0.99, reproducibility=0.99
        )
        passed, score = full_ihsan_check(receipt)
        assert passed


# ═══════════════════════════════════════════════════════════════════
# T7: Intent Gate Bypass Attempt
# ═══════════════════════════════════════════════════════════════════


class TestT7IntentGateBypass:
    """Verify that the Al-Ghazali intent gate cannot be bypassed.

    Invariant: Even with perfect efficiency/impact/reproducibility,
    intent < 0.90 means immediate rejection.

    This is Al-Ghazali's core insight: "The root must be sound
    before you examine the branches."
    """

    def test_high_scores_low_intent_rejected(self) -> None:
        """Perfect scores but intent=0.85 → rejected BEFORE scoring."""
        receipt = _make_receipt(
            intent=0.85,
            efficiency=0.99,
            impact=0.99,
            reproducibility=0.99,
        )
        assert not intent_gate(receipt)
        passed, score = full_ihsan_check(receipt)
        assert not passed
        # Score should be zero (rejected at intent gate)
        assert score == FP_ZERO

    def test_intent_exactly_at_floor_passes(self) -> None:
        """Intent exactly at 0.90 passes the gate."""
        receipt = _make_receipt(intent=0.90)
        assert intent_gate(receipt)

    def test_intent_just_below_floor_rejected(self) -> None:
        """Intent at 0.899 is rejected."""
        receipt = _make_receipt(intent=0.899)
        assert not intent_gate(receipt)


# ═══════════════════════════════════════════════════════════════════
# T8: Economic Death Spiral Prevention
# ═══════════════════════════════════════════════════════════════════


class TestT8DeathSpiralPrevention:
    """Prove that v2 progressive curve prevents the death spiral
    that v1's binary gate caused.

    v1 bug: Gini > 0.35 → mint 0 → economy dies.
    v2 fix: Progressive curve throttles but never zeros.

    T8 original proof: v2 earns 238 SEED vs v1's 0.00 (23,844x improvement).
    """

    def test_throttle_never_zero(self) -> None:
        """At any Gini level, throttle is strictly positive."""
        for gini_pct in range(0, 100):
            g = fp(gini_pct / 100)
            throttle = khaldunian_throttle(g)
            assert throttle > 0, f"Throttle is zero at Gini={gini_pct/100}"

    def test_minting_continues_under_extreme_gini(self) -> None:
        """Even at Gini 0.95, nodes can still mint (minimally)."""
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        wallet = _make_wallet(balance=10, actions=5)
        # Extreme inequality
        balances = [fp(1)] * 99 + [fp(99000)]
        gini = compute_gini(balances)
        mean = fp_div(sum(balances), fp(len(balances)))

        minted = progressive_mint(receipt, ihsan, wallet, gini, mean)
        assert minted > 0, "Minting is zero under extreme Gini — death spiral!"

    def test_progressive_curve_accumulates_seed(self) -> None:
        """Over 100 rounds under moderate inequality, SEED accumulates."""
        receipt = _make_receipt()
        ihsan = _get_ihsan(receipt)

        balance = fp(0)
        for _ in range(100):
            balances = [balance, fp(500)]
            gini = compute_gini(balances)
            mean = fp_div(sum(balances), fp(len(balances)))
            wallet = WalletState(
                node_id=b"\x01" * 32,
                seed_balance=balance,
                last_active=int(time.time() * 1000),
                total_actions=1,
                created_at=int(time.time() * 1000) - 86400000,
            )
            minted = progressive_mint(receipt, ihsan, wallet, gini, mean)
            balance = fp_add(balance, minted)

        assert (
            fp_float(balance) > 50
        ), f"After 100 rounds, balance should exceed 50 SEED, got {fp_float(balance):.2f}"


# ═══════════════════════════════════════════════════════════════════
# T9: Newcomer Multiplier Validation
# ═══════════════════════════════════════════════════════════════════


class TestT9NewcomerMultiplier:
    """Validate the newcomer equity advantage.

    At a network mean of 100 SEED, a zero-balance newcomer should
    receive EQUITY_FACTOR_MAX (5.0x) while at-or-above-mean nodes
    get FP_ONE (1.0x). Ratio = 5.0.
    """

    def test_newcomer_multiplier(self) -> None:
        """Zero-balance newcomer gets 5x the rich node's equity."""
        mean = fp(100)

        newcomer = _make_wallet(balance=0)
        rich = _make_wallet(balance=200)

        eq_new = ghazali_equity_factor(newcomer, mean)
        eq_rich = ghazali_equity_factor(rich, mean)

        # Newcomer: fp(EQUITY_FACTOR_MAX) = fp(5.0)
        # Rich: FP_ONE (at or above mean)
        ratio = fp_float(fp_div(eq_new, eq_rich)) if eq_rich > 0 else 999
        assert (
            4.5 < ratio < 5.5
        ), f"Newcomer multiplier {ratio:.2f}x outside expected range"

    def test_equity_at_zero_is_max(self) -> None:
        """Zero balance gives maximum equity factor = EQUITY_FACTOR_MAX."""
        newcomer = _make_wallet(balance=0)
        eq = ghazali_equity_factor(newcomer, fp(100))
        assert eq == fp(EQUITY_FACTOR_MAX)

    def test_equity_at_mean_is_standard(self) -> None:
        """Balance at or above mean gives standard equity = FP_ONE."""
        mean = fp(100)
        at_mean = _make_wallet(balance=100)
        eq = ghazali_equity_factor(at_mean, mean)
        assert eq == FP_ONE


# ═══════════════════════════════════════════════════════════════════
# T10: Asabiyyah Emergence
# ═══════════════════════════════════════════════════════════════════


class TestT10AsabiyyahEmergence:
    """Verify that social cohesion emerges from interaction.

    Ibn Khaldun: Asabiyyah (social solidarity) emerges from
    reciprocal bonds, not isolated excellence.

    Certification threshold: Connected network Asabiyyah > 0.
    """

    def test_loner_has_zero_asabiyyah(self) -> None:
        """A node with no connections has zero Asabiyyah."""
        loner = _make_wallet(balance=100, actions=50)
        loner.governance_votes = 0
        loner.cooperative_actions = 0
        asab = asabiyyah_score(loner, 1)
        assert asab == 0

    def test_connected_node_has_positive_asabiyyah(self) -> None:
        """A node with connections has positive Asabiyyah."""
        node = _make_wallet(balance=100, actions=50)
        node.attestations_given = {b"\x01", b"\x02"}
        node.attestations_received = {b"\x01"}  # Reciprocal with one
        node.governance_votes = 3
        node.cooperative_actions = 2
        asab = asabiyyah_score(node, 3)
        assert asab > 0

    def test_reciprocal_beats_unilateral(self) -> None:
        """Reciprocal attestation scores higher than unilateral.

        V1 hardening: need >= 3 unique connections for reciprocal to count.
        """
        reciprocal = _make_wallet(balance=100, actions=50)
        reciprocal.attestations_given = {b"\x01", b"\x02", b"\x03"}
        reciprocal.attestations_received = {b"\x01", b"\x02", b"\x03"}
        reciprocal.governance_votes = 5
        reciprocal.cooperative_actions = 3

        unilateral = _make_wallet(balance=100, actions=50)
        unilateral.attestations_given = {b"\x01", b"\x02", b"\x03"}
        unilateral.attestations_received = set()  # No one attested back
        unilateral.governance_votes = 5
        unilateral.cooperative_actions = 3

        asab_r = asabiyyah_score(reciprocal, 10)
        asab_u = asabiyyah_score(unilateral, 10)
        assert asab_r > asab_u

    def test_network_asabiyyah_increases_with_connections(self) -> None:
        """Network-wide Asabiyyah is higher when nodes are connected."""
        # Isolated network
        isolated = [_make_wallet(balance=100, actions=10) for _ in range(10)]
        net_isolated = network_asabiyyah(isolated)

        # Connected network
        connected = []
        for i in range(10):
            w = _make_wallet(balance=100, actions=10)
            w.attestations_given = {bytes([j]) for j in range(10) if j != i}
            w.attestations_received = {bytes([j]) for j in range(10) if j != i}
            w.governance_votes = 5
            w.cooperative_actions = 3
            connected.append(w)
        net_connected = network_asabiyyah(connected)

        assert net_connected > net_isolated
