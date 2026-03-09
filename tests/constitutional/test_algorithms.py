"""
Tests for the 15 Native Algorithms
═══════════════════════════════════

TDD anchors from Phase 67.07 specification.
Every test here MUST pass before the algorithm is considered complete.

Standing on Giants:
- Beck (2002): Test-Driven Development by Example
- Al-Ghazali (1058-1111): Intent as ethical pre-gate
- Ibn Khaldun (1332-1406): Asabiyyah + progressive inequality response
"""

from __future__ import annotations

import pytest

from core.constitutional.algorithms import (
    ASAB_CEIL,
    ASAB_FLOOR,
    ASAB_NEUTRAL,
    BLOOM_ACCRUAL,
    IHSAN_FLOOR,
    INTENT_FLOOR,
    MIN_CONNECTIONS,
    NISAB_THRESHOLD,
    TICK_INTERVAL,
    accrue_bloom,
    append_event,
    apply_demurrage,
    asabiyyah_adjustment,
    asabiyyah_score,
    backing_ratio,
    compile_reflex,
    compute_gini,
    compute_zakat,
    decay_bloom,
    full_ihsan_check,
    ghazali_equity_factor,
    ihsan_score,
    intent_gate,
    khaldunian_throttle,
    mint_seed,
    network_asabiyyah,
    progressive_mint,
    reflex_lookup,
    shura_resolve,
    shura_vote,
    trust_score,
    verify_event_chain,
)
from core.constitutional.fixed_point import (
    FP_ONE,
    FP_ZERO,
    fp,
    fp_float,
    fp_mul,
)
from core.constitutional.types import (
    ActionReceipt,
    Proposal,
    WalletState,
)

# ═══════════════════════════════════════════════════════════════════
# A1: Intent Gate
# ═══════════════════════════════════════════════════════════════════


class TestA1IntentGate:
    """Al-Ghazali pre-gate: intent must be >= 0.90."""

    def test_passes_above_floor(self, quality_receipt: ActionReceipt) -> None:
        assert intent_gate(quality_receipt) is True

    def test_rejects_below_floor(self, low_intent_receipt: ActionReceipt) -> None:
        assert intent_gate(low_intent_receipt) is False

    def test_boundary_at_exactly_0_90(self) -> None:
        r = ActionReceipt(
            receipt_id=b"\x00" * 32,
            actor_id=b"\x00" * 32,
            action_type="test",
            timestamp=0,
            intent_score=fp(0.90),
            efficiency_score=0,
            impact_score=0,
            reproducibility_score=0,
            oracle_signature=b"\x00" * 64,
            metadata_hash=b"\x00" * 32,
        )
        assert intent_gate(r) is True  # Exactly at floor passes

    def test_just_below_0_90(self) -> None:
        r = ActionReceipt(
            receipt_id=b"\x00" * 32,
            actor_id=b"\x00" * 32,
            action_type="test",
            timestamp=0,
            intent_score=fp(0.90) - 1,  # One LSB below
            efficiency_score=0,
            impact_score=0,
            reproducibility_score=0,
            oracle_signature=b"\x00" * 64,
            metadata_hash=b"\x00" * 32,
        )
        assert intent_gate(r) is False


# ═══════════════════════════════════════════════════════════════════
# A1: Ihsan Score
# ═══════════════════════════════════════════════════════════════════


class TestA1IhsanScore:
    """Weighted Ihsan quality computation."""

    def test_quality_receipt_above_threshold(
        self, quality_receipt: ActionReceipt
    ) -> None:
        score = ihsan_score(quality_receipt)
        assert (
            score >= IHSAN_FLOOR
        )  # 0.25*0.98 + 0.25*0.96 + 0.30*0.97 + 0.20*0.95 = 0.966

    def test_score_in_range_zero_to_one(self, quality_receipt: ActionReceipt) -> None:
        score = ihsan_score(quality_receipt)
        assert FP_ZERO <= score <= FP_ONE

    def test_weighted_sum_correct(self) -> None:
        """Manual computation: 0.25*0.9 + 0.25*0.8 + 0.30*0.7 + 0.20*0.6 = 0.755."""
        r = ActionReceipt(
            receipt_id=b"\x00" * 32,
            actor_id=b"\x00" * 32,
            action_type="test",
            timestamp=0,
            intent_score=fp(0.9),
            efficiency_score=fp(0.8),
            impact_score=fp(0.7),
            reproducibility_score=fp(0.6),
            oracle_signature=b"\x00" * 64,
            metadata_hash=b"\x00" * 32,
        )
        score = ihsan_score(r)
        assert abs(fp_float(score) - 0.755) < 0.001

    def test_full_ihsan_check_rejects_low_intent(
        self, low_intent_receipt: ActionReceipt
    ) -> None:
        passed, score = full_ihsan_check(low_intent_receipt)
        assert passed is False
        assert score == FP_ZERO

    def test_full_ihsan_check_passes_quality(
        self, quality_receipt: ActionReceipt
    ) -> None:
        passed, score = full_ihsan_check(quality_receipt)
        assert passed is True
        assert score > 0


# ═══════════════════════════════════════════════════════════════════
# A2: SEED Minter
# ═══════════════════════════════════════════════════════════════════


class TestA2SeedMinter:
    """Proof-of-Impact minting."""

    def test_mints_for_quality_work(self, quality_receipt: ActionReceipt) -> None:
        _, ihsan = full_ihsan_check(quality_receipt)
        minted = mint_seed(quality_receipt, ihsan)
        assert minted > 0

    def test_zero_mint_below_threshold(self, low_intent_receipt: ActionReceipt) -> None:
        minted = mint_seed(low_intent_receipt, fp(0.50))
        assert minted == FP_ZERO

    def test_efficiency_bonus(self) -> None:
        """Higher efficiency = more SEED."""
        low_eff = ActionReceipt(
            receipt_id=b"\x00" * 32,
            actor_id=b"\x00" * 32,
            action_type="test",
            timestamp=0,
            intent_score=fp(0.95),
            efficiency_score=fp(0.50),
            impact_score=fp(0.95),
            reproducibility_score=fp(0.95),
            oracle_signature=b"\x00" * 64,
            metadata_hash=b"\x00" * 32,
        )
        high_eff = ActionReceipt(
            receipt_id=b"\x00" * 32,
            actor_id=b"\x00" * 32,
            action_type="test",
            timestamp=0,
            intent_score=fp(0.95),
            efficiency_score=fp(0.99),
            impact_score=fp(0.95),
            reproducibility_score=fp(0.95),
            oracle_signature=b"\x00" * 64,
            metadata_hash=b"\x00" * 32,
        )
        minted_low = mint_seed(low_eff, fp(0.96))
        minted_high = mint_seed(high_eff, fp(0.96))
        assert minted_high > minted_low


# ═══════════════════════════════════════════════════════════════════
# A3: BLOOM Accumulator
# ═══════════════════════════════════════════════════════════════════


class TestA3BloomAccumulator:
    """Soulbound governance token accrual and decay."""

    def test_accrues_on_high_ihsan(self, newcomer_wallet: WalletState) -> None:
        new_bloom = accrue_bloom(newcomer_wallet, IHSAN_FLOOR)
        assert new_bloom == BLOOM_ACCRUAL

    def test_no_accrual_below_threshold(self, newcomer_wallet: WalletState) -> None:
        new_bloom = accrue_bloom(newcomer_wallet, fp(0.80))
        assert new_bloom == newcomer_wallet.bloom_balance

    def test_decay_on_inactivity(self, wealthy_wallet: WalletState) -> None:
        # Simulate 10 ticks of inactivity
        future = wealthy_wallet.last_active + (10 * TICK_INTERVAL)
        new_bloom = decay_bloom(wealthy_wallet, future)
        assert new_bloom < wealthy_wallet.bloom_balance

    def test_no_decay_if_active(self, wealthy_wallet: WalletState) -> None:
        """Active node = no decay."""
        new_bloom = decay_bloom(wealthy_wallet, wealthy_wallet.last_active)
        assert new_bloom == wealthy_wallet.bloom_balance


# ═══════════════════════════════════════════════════════════════════
# A4: Gini Enforcer
# ═══════════════════════════════════════════════════════════════════


class TestA4GiniEnforcer:
    """Khaldunian Curve + Ghazali Equity Factor."""

    def test_compute_gini_equal_distribution(self) -> None:
        """Perfect equality = Gini 0."""
        balances = [fp(100)] * 10
        assert compute_gini(balances) == FP_ZERO

    def test_compute_gini_extreme_inequality(self) -> None:
        """One whale = high Gini."""
        balances = [fp(0)] * 9 + [fp(1000)]
        gini = compute_gini(balances)
        assert fp_float(gini) > 0.80

    def test_compute_gini_moderate(self) -> None:
        """Moderate spread = moderate Gini."""
        balances = [fp(10), fp(20), fp(30), fp(40), fp(100)]
        gini = compute_gini(balances)
        assert 0 < fp_float(gini) < 0.60

    def test_compute_gini_single_wallet(self) -> None:
        assert compute_gini([fp(100)]) == FP_ZERO

    def test_compute_gini_empty(self) -> None:
        assert compute_gini([]) == FP_ZERO

    def test_khaldunian_throttle_healthy(self) -> None:
        """Gini <= 0.30, neutral asabiyyah = full minting."""
        neutral = fp(0.50)  # ASAB_NEUTRAL → 1.0x multiplier
        assert khaldunian_throttle(fp(0.20), neutral) == FP_ONE
        assert khaldunian_throttle(fp(0.30), neutral) == FP_ONE

    def test_khaldunian_throttle_warning_early(self) -> None:
        """Gini just above 0.30 = slight quadratic dropoff."""
        neutral = fp(0.50)
        throttle = khaldunian_throttle(fp(0.35), neutral)
        assert FP_ZERO < throttle < FP_ONE

    def test_khaldunian_throttle_warning_deep(self) -> None:
        """Gini 0.45 = deep into warning zone (0.30-0.50)."""
        neutral = fp(0.50)
        throttle = khaldunian_throttle(fp(0.45), neutral)
        assert fp(0.05) <= throttle < fp(0.60)

    def test_khaldunian_throttle_crisis(self) -> None:
        """Gini 0.50-0.70 = 10% minting (neutral asabiyyah)."""
        neutral = fp(0.50)
        throttle = khaldunian_throttle(fp(0.60), neutral)
        assert throttle == fp(0.10)

    def test_khaldunian_throttle_extreme(self) -> None:
        """Gini > 0.70 = 1% but NEVER zero (neutral asabiyyah)."""
        neutral = fp(0.50)
        throttle = khaldunian_throttle(fp(0.90), neutral)
        assert throttle == fp(0.01)
        assert throttle > 0  # Never zero!

    def test_ghazali_equity_newcomer_advantage(
        self, newcomer_wallet: WalletState
    ) -> None:
        """Zero-balance newcomer gets maximum equity factor."""
        factor = ghazali_equity_factor(newcomer_wallet, fp(1000))
        assert factor == fp(5.0)  # EQUITY_FACTOR_MAX

    def test_ghazali_equity_wealthy_standard(self, wealthy_wallet: WalletState) -> None:
        """Wealthy node at mean gets standard rate."""
        factor = ghazali_equity_factor(wealthy_wallet, fp(1000))
        assert factor == FP_ONE

    def test_ghazali_equity_capped_at_max(self) -> None:
        """Equity factor capped at 5.0x."""
        wallet = WalletState(node_id=b"\x00" * 32, seed_balance=fp(1))
        factor = ghazali_equity_factor(wallet, fp(100000))
        assert factor == fp(5.0)

    def test_progressive_mint_full_pipeline(
        self, quality_receipt: ActionReceipt, newcomer_wallet: WalletState
    ) -> None:
        """Full minting pipeline produces nonzero for quality work."""
        _, ihsan = full_ihsan_check(quality_receipt)
        minted = progressive_mint(
            quality_receipt,
            ihsan,
            newcomer_wallet,
            fp(0.20),  # Healthy Gini
            fp(1000),  # Mean balance
        )
        assert minted > 0


# ═══════════════════════════════════════════════════════════════════
# Phase 69 Sprint 1: Asabiyyah-Gini Coupling
# ═══════════════════════════════════════════════════════════════════


class TestAsabiyyahAdjustment:
    """Asabiyyah coupling multiplier: cohesion modulates minting."""

    def test_zero_asabiyyah_returns_floor(self) -> None:
        """Fragmented network (asabiyyah=0) → 0.80x multiplier."""
        result = asabiyyah_adjustment(FP_ZERO)
        assert result == ASAB_FLOOR  # 800_000

    def test_full_asabiyyah_returns_ceil(self) -> None:
        """Fully cohesive network (asabiyyah=1.0) → 1.20x multiplier."""
        result = asabiyyah_adjustment(FP_ONE)
        assert result == ASAB_CEIL  # 1_200_000

    def test_neutral_asabiyyah_returns_one(self) -> None:
        """Neutral point (asabiyyah=0.5) → 1.00x (no effect)."""
        result = asabiyyah_adjustment(ASAB_NEUTRAL)
        assert result == FP_ONE

    def test_monotonically_increasing(self) -> None:
        """Higher asabiyyah → higher multiplier."""
        low = asabiyyah_adjustment(fp(0.20))
        mid = asabiyyah_adjustment(fp(0.50))
        high = asabiyyah_adjustment(fp(0.80))
        assert low < mid < high

    def test_clamps_above_one(self) -> None:
        """Asabiyyah > 1.0 clamped to 1.0 → returns ceiling."""
        result = asabiyyah_adjustment(fp(2.0))
        assert result == ASAB_CEIL

    def test_clamps_below_zero(self) -> None:
        """Negative asabiyyah clamped to 0.0 → returns floor."""
        result = asabiyyah_adjustment(-fp(1))
        assert result == ASAB_FLOOR


class TestKhaldunianThrottleV3:
    """V3: Khaldunian throttle with Asabiyyah coupling (Phase 69)."""

    def test_healthy_gini_cohesive_network_boosted(self) -> None:
        """Healthy Gini + high asabiyyah → above 1.0 (boosted)."""
        throttle = khaldunian_throttle(fp(0.20), fp(0.80))
        assert throttle > FP_ONE  # 1.0 * 1.12 = 1.12

    def test_healthy_gini_fragmented_network_throttled(self) -> None:
        """Healthy Gini + zero asabiyyah → below 1.0 (0.80x)."""
        throttle = khaldunian_throttle(fp(0.20), FP_ZERO)
        assert throttle < FP_ONE
        assert throttle == ASAB_FLOOR  # 1.0 * 0.80

    def test_backward_compatible_default(self) -> None:
        """Default asabiyyah=0 gives ASAB_FLOOR multiplier (not 1.0).

        This is intentional: an unknown network (asabiyyah=0) is treated
        as fragmented. To get neutral behavior, pass asabiyyah=0.5.
        """
        throttle_default = khaldunian_throttle(fp(0.20))
        throttle_explicit = khaldunian_throttle(fp(0.20), FP_ZERO)
        assert throttle_default == throttle_explicit

    def test_crisis_zone_with_asabiyyah(self) -> None:
        """Crisis Gini + high asabiyyah: still very low but asabiyyah helps."""
        crisis_alone = khaldunian_throttle(fp(0.60), FP_ZERO)
        crisis_cohesive = khaldunian_throttle(fp(0.60), FP_ONE)
        assert crisis_cohesive > crisis_alone

    def test_extreme_gini_never_zero(self) -> None:
        """Even extreme Gini + zero asabiyyah never produces zero mint."""
        throttle = khaldunian_throttle(fp(0.90), FP_ZERO)
        assert throttle > 0

    def test_progressive_mint_uses_asabiyyah(
        self, quality_receipt: ActionReceipt, newcomer_wallet: WalletState
    ) -> None:
        """progressive_mint passes asabiyyah to khaldunian_throttle."""
        _, ihsan = full_ihsan_check(quality_receipt)
        minted_fragmented = progressive_mint(
            quality_receipt, ihsan, newcomer_wallet, fp(0.20), fp(1000), FP_ZERO
        )
        # Reset wallet balance for fair comparison
        newcomer_wallet.seed_balance = 0
        minted_cohesive = progressive_mint(
            quality_receipt, ihsan, newcomer_wallet, fp(0.20), fp(1000), FP_ONE
        )
        assert minted_cohesive > minted_fragmented


# ═══════════════════════════════════════════════════════════════════
# A5: Zakat Engine
# ═══════════════════════════════════════════════════════════════════


class TestA5ZakatEngine:
    """I-7: Wealth purification."""

    def test_zakat_above_nisab(self, wealthy_wallet: WalletState) -> None:
        zakat = compute_zakat(wealthy_wallet)
        assert zakat > 0
        # 2.5% of 5000 = 125
        expected = fp_mul(wealthy_wallet.seed_balance, fp(0.025))
        assert zakat == expected

    def test_zakat_below_nisab_exempt(self, newcomer_wallet: WalletState) -> None:
        assert compute_zakat(newcomer_wallet) == FP_ZERO

    def test_zakat_rate_exactly_2_5_percent(self) -> None:
        wallet = WalletState(node_id=b"\x00" * 32, seed_balance=fp(1000))
        zakat = compute_zakat(wallet)
        assert abs(fp_float(zakat) - 25.0) < 0.01


# ═══════════════════════════════════════════════════════════════════
# A6: Backing Ratio
# ═══════════════════════════════════════════════════════════════════


class TestA6BackingRatio:
    """Reserve health check."""

    def test_perfect_backing(self) -> None:
        ratio = backing_ratio(fp(100), fp(100))
        assert ratio == FP_ONE

    def test_inflation(self) -> None:
        ratio = backing_ratio(fp(200), fp(100))
        assert fp_float(ratio) < 1.0

    def test_deflation(self) -> None:
        ratio = backing_ratio(fp(100), fp(200))
        assert fp_float(ratio) > 1.0

    def test_zero_seed(self) -> None:
        assert backing_ratio(0, fp(100)) == FP_ONE


# ═══════════════════════════════════════════════════════════════════
# A7: Demurrage
# ═══════════════════════════════════════════════════════════════════


class TestA7Demurrage:
    """Idle tax to incentivize circulation."""

    def test_active_node_exempt(self, wealthy_wallet: WalletState) -> None:
        balance = apply_demurrage(wealthy_wallet, wealthy_wallet.last_active)
        assert balance == wealthy_wallet.seed_balance

    def test_idle_node_taxed(self, wealthy_wallet: WalletState) -> None:
        future = wealthy_wallet.last_active + (5 * TICK_INTERVAL)
        balance = apply_demurrage(wealthy_wallet, future)
        assert balance < wealthy_wallet.seed_balance

    def test_demurrage_never_negative(self) -> None:
        wallet = WalletState(
            node_id=b"\x00" * 32,
            seed_balance=fp(0.01),
            last_active=1000000,
        )
        far_future = wallet.last_active + (10000 * TICK_INTERVAL)
        balance = apply_demurrage(wallet, far_future)
        assert balance >= 0


# ═══════════════════════════════════════════════════════════════════
# A8: Shura Governance
# ═══════════════════════════════════════════════════════════════════


class TestA8ShuraGovernance:
    """BLOOM-weighted voting."""

    def test_bloom_weighted_vote(self) -> None:
        proposal = Proposal(
            proposal_id=b"\x00" * 32,
            proposer=b"\x01" * 32,
            description="test",
        )
        voter = WalletState(node_id=b"\x02" * 32, bloom_balance=fp(5))
        proposal = shura_vote(proposal, voter, approve=True)
        assert proposal.votes_for == fp(5)

    def test_zero_bloom_no_vote(self) -> None:
        proposal = Proposal(
            proposal_id=b"\x00" * 32,
            proposer=b"\x01" * 32,
            description="test",
        )
        voter = WalletState(node_id=b"\x02" * 32, bloom_balance=0)
        proposal = shura_vote(proposal, voter, approve=True)
        assert proposal.votes_for == 0

    def test_supermajority_passes(self) -> None:
        proposal = Proposal(
            proposal_id=b"\x00" * 32,
            proposer=b"\x01" * 32,
            description="test",
            votes_for=fp(70),
            votes_against=fp(30),
        )
        assert shura_resolve(proposal) == "passed"

    def test_minority_rejected(self) -> None:
        proposal = Proposal(
            proposal_id=b"\x00" * 32,
            proposer=b"\x01" * 32,
            description="test",
            votes_for=fp(30),
            votes_against=fp(70),
        )
        assert shura_resolve(proposal) == "rejected"

    def test_no_votes_expired(self) -> None:
        proposal = Proposal(
            proposal_id=b"\x00" * 32,
            proposer=b"\x01" * 32,
            description="test",
        )
        assert shura_resolve(proposal) == "expired"


# ═══════════════════════════════════════════════════════════════════
# A9: Trust Monitor
# ═══════════════════════════════════════════════════════════════════


class TestA9TrustMonitor:
    """Historical ihsan performance trust."""

    def test_no_history_zero(self) -> None:
        wallet = WalletState(node_id=b"\x00" * 32)
        assert trust_score(wallet) == FP_ZERO

    def test_consistent_high_ihsan(self) -> None:
        wallet = WalletState(
            node_id=b"\x00" * 32,
            ihsan_history=[fp(0.96)] * 20,
        )
        score = trust_score(wallet)
        assert fp_float(score) > 0.95  # High trust from consistency


# ═══════════════════════════════════════════════════════════════════
# A10: Reflex Compiler
# ═══════════════════════════════════════════════════════════════════


class TestA10ReflexCompiler:
    """System-1 O(1) cache."""

    def test_compile_and_lookup(self) -> None:
        reflex = compile_reflex("hello world", ["greet", "respond"], fp(0.98))
        cache = {reflex.pattern_hash: reflex}
        found = reflex_lookup(cache, "hello world")
        assert found is not None
        assert found.action_chain == ("greet", "respond")

    def test_cache_miss_returns_none(self) -> None:
        cache: dict[bytes, object] = {}
        found = reflex_lookup(cache, "unknown pattern")
        assert found is None

    def test_low_confidence_rejected_at_lookup(self) -> None:
        """Reflex below IHSAN_FLOOR is not returned from lookup.

        V5 hardening: compile_reflex now also rejects at compile time,
        so we manually construct a low-confidence reflex to test lookup.
        """
        from core.constitutional.types import Reflex

        pattern_hash = b"\xaa" * 32
        low_reflex = Reflex(
            pattern_hash=pattern_hash,
            action_chain=("act",),
            confidence=fp(0.50),
            last_used=0,
            use_count=0,
        )
        cache = {pattern_hash: low_reflex}
        found = reflex_lookup(cache, "weak")
        assert found is None


# ═══════════════════════════════════════════════════════════════════
# A14: Event Sourcer
# ═══════════════════════════════════════════════════════════════════


class TestA14EventSourcer:
    """Immutable event log with hash chain."""

    def test_append_creates_chain(self) -> None:
        log: list = []
        e0 = append_event(log, "genesis", b"\x00" * 32, {"v": "1.0"})
        e1 = append_event(log, "mint", b"\x01" * 32, {"amount": 100})
        assert e0.event_id == 0
        assert e1.event_id == 1
        assert e1.prev_hash == e0.hash

    def test_chain_integrity(self) -> None:
        log: list = []
        append_event(log, "genesis", b"\x00" * 32, {})
        append_event(log, "mint", b"\x01" * 32, {"amount": 100})
        append_event(log, "vote", b"\x02" * 32, {"proposal": "test"})

        valid, errors = verify_event_chain(log)
        assert valid
        assert len(errors) == 0

    def test_tampered_chain_detected(self) -> None:
        log: list = []
        append_event(log, "genesis", b"\x00" * 32, {})
        append_event(log, "mint", b"\x01" * 32, {"amount": 100})

        # Tamper with event 1's prev_hash
        log[1].prev_hash = b"\xff" * 32
        valid, errors = verify_event_chain(log)
        assert not valid
        assert len(errors) > 0


# ═══════════════════════════════════════════════════════════════════
# A15: Asabiyyah Index
# ═══════════════════════════════════════════════════════════════════


class TestA15Asabiyyah:
    """Ibn Khaldun's social cohesion metric."""

    def test_score_zero_for_isolated_node(self) -> None:
        wallet = WalletState(node_id=b"\x00" * 32)
        assert asabiyyah_score(wallet, 10) == FP_ZERO

    def test_score_increases_with_attestations(self) -> None:
        w1 = WalletState(node_id=b"\x00" * 32)
        # V1 hardening: need >= 3 unique connections for reciprocal to count
        w2 = WalletState(
            node_id=b"\x01" * 32,
            attestations_given={b"\x02" * 32, b"\x03" * 32, b"\x04" * 32},
            attestations_received={b"\x02" * 32, b"\x03" * 32, b"\x04" * 32},
            governance_votes=5,
            cooperative_actions=10,
        )
        assert asabiyyah_score(w2, 10) > asabiyyah_score(w1, 10)

    def test_network_asabiyyah_increases_with_connections(self) -> None:
        """More connected network = higher network asabiyyah."""
        isolated = [WalletState(node_id=bytes([i]) * 32) for i in range(5)]
        connected = []
        for i in range(5):
            peers = {bytes([j]) * 32 for j in range(5) if j != i}
            connected.append(
                WalletState(
                    node_id=bytes([i]) * 32,
                    attestations_given=peers,
                    attestations_received=peers,
                    governance_votes=5,
                    cooperative_actions=10,
                )
            )
        assert network_asabiyyah(connected) > network_asabiyyah(isolated)

    def test_single_node_network(self) -> None:
        wallet = WalletState(node_id=b"\x00" * 32)
        assert asabiyyah_score(wallet, 1) == FP_ZERO


# ═══════════════════════════════════════════════════════════════════
# Determinism: All algorithms must be deterministic
# ═══════════════════════════════════════════════════════════════════


class TestDeterminism:
    """Same inputs -> same outputs, always."""

    def test_ihsan_deterministic(self, quality_receipt: ActionReceipt) -> None:
        results = [ihsan_score(quality_receipt) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_gini_deterministic(self) -> None:
        balances = [fp(i * 10) for i in range(1, 51)]
        results = [compute_gini(balances) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_progressive_mint_deterministic(
        self, quality_receipt: ActionReceipt, newcomer_wallet: WalletState
    ) -> None:
        _, ihsan = full_ihsan_check(quality_receipt)
        results = [
            progressive_mint(
                quality_receipt, ihsan, newcomer_wallet, fp(0.20), fp(1000)
            )
            for _ in range(100)
        ]
        assert all(r == results[0] for r in results)


# ═══════════════════════════════════════════════════════════════════
# Red Team Hardening Tests
# ═══════════════════════════════════════════════════════════════════


class TestV1AntiCollusion:
    """V1: Asabiyyah anti-collusion — MIN_CONNECTIONS gate.

    A 2-node collusion ring achieves 100% reciprocal ratio trivially.
    MIN_CONNECTIONS = 3 forces genuine community participation.
    """

    def test_two_node_collusion_ring_blocked(self) -> None:
        """Two nodes mutually attesting should NOT get reciprocal credit."""
        alice = WalletState(
            node_id=b"\xa1" * 32,
            attestations_given={b"\xb0" * 32},
            attestations_received={b"\xb0" * 32},
            governance_votes=5,
            cooperative_actions=5,
        )
        score = asabiyyah_score(alice, 10)
        # Reciprocal component should be zero (only 1 connection < MIN_CONNECTIONS)
        # Only governance + cooperation contribute
        assert score > FP_ZERO  # Still has gov + coop
        # But reciprocal ratio = 0, so score is lower than with 3+ connections
        genuine = WalletState(
            node_id=b"\xa2" * 32,
            attestations_given={b"\xb1" * 32, b"\xb2" * 32, b"\xb3" * 32},
            attestations_received={b"\xb1" * 32, b"\xb2" * 32, b"\xb3" * 32},
            governance_votes=5,
            cooperative_actions=5,
        )
        assert asabiyyah_score(genuine, 10) > score

    def test_exactly_min_connections_passes(self) -> None:
        """Exactly MIN_CONNECTIONS unique peers should enable reciprocal score."""
        peers = {bytes([i]) * 32 for i in range(MIN_CONNECTIONS)}
        wallet = WalletState(
            node_id=b"\x00" * 32,
            attestations_given=peers,
            attestations_received=peers,
        )
        score = asabiyyah_score(wallet, 20)
        # Reciprocal component contributes (3 reciprocal / 19 max = non-zero)
        assert score > FP_ZERO

    def test_below_min_connections_no_reciprocal(self) -> None:
        """Below MIN_CONNECTIONS: reciprocal ratio = 0."""
        wallet = WalletState(
            node_id=b"\x00" * 32,
            attestations_given={b"\x01" * 32, b"\x02" * 32},
            attestations_received={b"\x01" * 32, b"\x02" * 32},
        )
        # 2 connections < MIN_CONNECTIONS(3), so reciprocal = 0
        # No governance or cooperation either → score = 0
        assert asabiyyah_score(wallet, 10) == FP_ZERO

    def test_min_connections_constant_is_three(self) -> None:
        """MIN_CONNECTIONS must be at least 3 to prevent 2-node collusion."""
        assert MIN_CONNECTIONS >= 3


class TestV5ReflexCompileGate:
    """V5: compile_reflex Ihsan gate — reject low-quality reflexes at compile time.

    Poisoned cache entries corrupt System-1 decisions downstream.
    The Ihsan gate must fire at compile time, not just at lookup.
    """

    def test_high_confidence_compiles(self) -> None:
        """Above IHSAN_FLOOR: reflex compiles successfully."""
        reflex = compile_reflex("valid pattern", ["act"], fp(0.98))
        assert reflex is not None
        assert reflex.confidence == fp(0.98)

    def test_low_confidence_rejected_at_compile(self) -> None:
        """Below IHSAN_FLOOR: compile returns None."""
        reflex = compile_reflex("bad pattern", ["act"], fp(0.50))
        assert reflex is None

    def test_exactly_floor_compiles(self) -> None:
        """Exactly IHSAN_FLOOR: should compile (>= threshold)."""
        reflex = compile_reflex("boundary", ["act"], IHSAN_FLOOR)
        assert reflex is not None

    def test_just_below_floor_rejected(self) -> None:
        """One unit below IHSAN_FLOOR: rejected."""
        reflex = compile_reflex("almost", ["act"], IHSAN_FLOOR - 1)
        assert reflex is None

    def test_zero_confidence_rejected(self) -> None:
        """Zero confidence: definitely rejected."""
        reflex = compile_reflex("zero", ["act"], 0)
        assert reflex is None

    def test_compile_gate_prevents_cache_poisoning(self) -> None:
        """End-to-end: low-confidence reflex never enters cache."""
        cache: dict[bytes, object] = {}
        reflex = compile_reflex("poison", ["malicious_action"], fp(0.30))
        if reflex is not None:
            cache[reflex.pattern_hash] = reflex
        # Cache should be empty — compile returned None
        assert len(cache) == 0
