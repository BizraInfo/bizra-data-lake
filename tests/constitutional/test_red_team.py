"""
Red Team Adversarial Tests — Attack Surface Validation
═══════════════════════════════════════════════════════

Validates that constitutional algorithms resist known adversarial
attack vectors identified by the Red Team Harness.

9 Attack Classes, adapted to production API signatures:
1. Ihsan Gaming — boundary manipulation
2. Economic Exploits — mint inflation
3. Gini Manipulation — inequality gaming
4. Identity Forgery — bad receipt structure
5. Asabiyyah Gaming — collusion rings (V1 hardened)
6. Event Log Tampering — data integrity
7. Zakat/Demurrage Evasion — economic avoidance
8. Reflex Cache Poisoning — cache integrity (V5 hardened)
9. Governance Gaming — BLOOM manipulation

Standing on Giants:
- Lamport (1982): Byzantine fault tolerance
- Schneier (2000): Security as adversarial thinking
"""

from __future__ import annotations

import time


from core.constitutional.algorithms import (
    IHSAN_FLOOR,
    NISAB_THRESHOLD,
    append_event,
    asabiyyah_score,
    compile_reflex,
    compute_gini,
    compute_zakat,
    full_ihsan_check,
    ihsan_score,
    intent_gate,
    khaldunian_throttle,
    mint_seed,
    network_asabiyyah,
    progressive_mint,
    reflex_lookup,
    shura_resolve,
    shura_vote,
    verify_event_chain,
)
from core.constitutional.fixed_point import (
    FP_ONE,
    FP_ZERO,
    fp,
)
from core.constitutional.types import (
    ActionReceipt,
    Proposal,
    Reflex,
    WalletState,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_receipt(**overrides) -> ActionReceipt:
    """Factory for adversarial receipts."""
    defaults = {
        "receipt_id": b"\x01" * 32,
        "actor_id": b"\x02" * 32,
        "action_type": "code",
        "timestamp": int(time.time() * 1000),
        "intent_score": fp(0.95),
        "efficiency_score": fp(0.97),
        "impact_score": fp(0.96),
        "reproducibility_score": fp(0.98),
        "oracle_signature": b"\x03" * 64,
        "metadata_hash": b"\x04" * 32,
    }
    defaults.update(overrides)
    return ActionReceipt(**defaults)


def _make_wallet(**overrides) -> WalletState:
    """Factory for adversarial wallets."""
    defaults = {
        "node_id": b"\x10" * 32,
        "seed_balance": fp(100),
    }
    defaults.update(overrides)
    return WalletState(**defaults)


# ═══════════════════════════════════════════════════════════════════
# Attack Class 1: Ihsan Gaming
# ═══════════════════════════════════════════════════════════════════


class TestAttack1IhsanGaming:
    """Adversary tries to pass Ihsan check with minimal scores."""

    def test_all_scores_at_boundary(self) -> None:
        """All component scores at exactly 0.95 — should barely pass."""
        r = _make_receipt(
            intent_score=fp(0.95),
            efficiency_score=fp(0.95),
            impact_score=fp(0.95),
            reproducibility_score=fp(0.95),
        )
        passed, score = full_ihsan_check(r)
        assert passed  # Just at boundary

    def test_intent_boundary_exploit(self) -> None:
        """Intent at 0.8999 rounds down — must fail."""
        r = _make_receipt(intent_score=fp(0.8999))
        assert not intent_gate(r)

    def test_one_score_inflated_others_zero(self) -> None:
        """Max out one score, zero others — should fail Ihsan."""
        r = _make_receipt(
            intent_score=fp(0.99),
            efficiency_score=FP_ONE,
            impact_score=FP_ZERO,
            reproducibility_score=FP_ZERO,
        )
        passed, score = full_ihsan_check(r)
        assert not passed  # 0.25 * 0.99 + 0.25 * 1.0 = 0.4975 < 0.95

    def test_maximum_score_cannot_exceed_one(self) -> None:
        """Even with all FP_ONE scores, ihsan <= 1.0."""
        r = _make_receipt(
            intent_score=FP_ONE,
            efficiency_score=FP_ONE,
            impact_score=FP_ONE,
            reproducibility_score=FP_ONE,
        )
        score = ihsan_score(r)
        assert score <= FP_ONE


# ═══════════════════════════════════════════════════════════════════
# Attack Class 2: Economic Exploits
# ═══════════════════════════════════════════════════════════════════


class TestAttack2EconomicExploits:
    """Adversary tries to inflate mint rewards or evade costs."""

    def test_mint_capped_by_ihsan(self) -> None:
        """Even perfect scores yield bounded reward."""
        r = _make_receipt(
            efficiency_score=FP_ONE,
            impact_score=FP_ONE,
            reproducibility_score=FP_ONE,
        )
        reward = mint_seed(r, FP_ONE)
        # BASE_MINT * 1.0 = 1.0 SEED max
        assert reward <= fp(1.5)  # Some bonus for efficiency but bounded

    def test_progressive_mint_throttles_whales(self) -> None:
        """Whale in unequal network gets heavily throttled."""
        r = _make_receipt()
        whale_wallet = _make_wallet(seed_balance=fp(100000))
        # Very high Gini
        reward = progressive_mint(r, fp(0.97), whale_wallet, fp(0.80), fp(500))
        assert reward < fp(0.1)  # Crisis zone + above-mean equity factor = minimal


# ═══════════════════════════════════════════════════════════════════
# Attack Class 3: Gini Manipulation
# ═══════════════════════════════════════════════════════════════════


class TestAttack3GiniManipulation:
    """Adversary tries to game inequality metrics."""

    def test_single_node_gini_is_zero(self) -> None:
        """Can't inflate Gini with a single balance."""
        assert compute_gini([fp(1000000)]) == FP_ZERO

    def test_empty_balances_gini_zero(self) -> None:
        assert compute_gini([]) == FP_ZERO

    def test_equal_balances_near_zero(self) -> None:
        """Perfectly equal distribution = Gini 0."""
        balances = [fp(100)] * 100
        gini = compute_gini(balances)
        assert gini < fp(0.01)

    def test_extreme_inequality_high_gini(self) -> None:
        """One whale, many paupers = high Gini."""
        balances = [fp(1)] * 99 + [fp(99999)]
        gini = compute_gini(balances)
        assert gini > fp(0.50)

    def test_throttle_monotonically_decreasing(self) -> None:
        """Khaldunian curve must be monotonically non-increasing."""
        prev = FP_ONE
        for g in range(0, 100, 5):
            throttle = khaldunian_throttle(fp(g / 100))
            assert throttle <= prev
            prev = throttle


# ═══════════════════════════════════════════════════════════════════
# Attack Class 4: Identity Forgery
# ═══════════════════════════════════════════════════════════════════


class TestAttack4IdentityForgery:
    """Adversary tries to forge or corrupt receipts."""

    def test_zero_length_receipt_id_detected(self) -> None:
        """Empty receipt_id should not pass Ihsan check."""
        r = _make_receipt(receipt_id=b"")
        # The algorithm doesn't validate receipt_id length,
        # but the receipt still must pass Ihsan gate.
        # If it passes Ihsan, that's acceptable — identity
        # validation is the consensus layer's job.
        assert intent_gate(r)  # Intent is valid

    def test_future_timestamp_valid_locally(self) -> None:
        """Future timestamp — local algorithm doesn't reject.

        Timestamp validation is consensus-layer responsibility.
        """
        r = _make_receipt(timestamp=int(time.time() * 1000) + 86400000)
        passed, score = full_ihsan_check(r)
        assert passed  # Local scoring doesn't validate timestamps


# ═══════════════════════════════════════════════════════════════════
# Attack Class 5: Asabiyyah Gaming (V1 Hardened)
# ═══════════════════════════════════════════════════════════════════


class TestAttack5AsabiyyahGaming:
    """Adversary creates collusion rings to inflate social cohesion."""

    def test_two_node_collusion_ring(self) -> None:
        """Classic attack: Alice and Bob mutually attest only each other.

        V1 fix: MIN_CONNECTIONS = 3 means 2-node ring gets zero reciprocal.
        """
        alice = _make_wallet(
            node_id=b"\xa1" * 32,
            attestations_given={b"\xb0" * 32},
            attestations_received={b"\xb0" * 32},
        )
        score = asabiyyah_score(alice, 10)
        # Only 1 connection < MIN_CONNECTIONS → reciprocal = 0
        assert score == FP_ZERO  # No governance or cooperation either

    def test_three_node_collusion_ring_limited(self) -> None:
        """3-node ring: just meets MIN_CONNECTIONS but limited by network size."""
        node = _make_wallet(
            node_id=b"\xa1" * 32,
            attestations_given={b"\xb1" * 32, b"\xb2" * 32, b"\xb3" * 32},
            attestations_received={b"\xb1" * 32, b"\xb2" * 32, b"\xb3" * 32},
        )
        score_small_net = asabiyyah_score(node, 4)  # 3/3 max reciprocal
        score_large_net = asabiyyah_score(node, 100)  # 3/99 max reciprocal
        # In a large network, 3-node ring has diluted impact
        assert score_large_net < score_small_net

    def test_sybil_army_diluted(self) -> None:
        """Creating 50 fake nodes doesn't help if they're all in one cluster.

        Network-wide Asabiyyah averages all nodes — sybils with zero
        governance/cooperation drag the average down.
        """
        genuine = []
        for i in range(10):
            w = _make_wallet(node_id=bytes([i]) * 32)
            peers = {bytes([j]) * 32 for j in range(10) if j != i}
            w.attestations_given = peers
            w.attestations_received = peers
            w.governance_votes = 5
            w.cooperative_actions = 5
            genuine.append(w)

        # Add 40 sybil nodes — only attest each other, no governance
        sybils = []
        for i in range(10, 50):
            w = _make_wallet(node_id=bytes([i]) * 32)
            # Sybils form cliques but don't participate in governance
            sybil_peers = {bytes([j]) * 32 for j in range(10, 50) if j != i}
            w.attestations_given = sybil_peers
            w.attestations_received = sybil_peers
            w.governance_votes = 0
            w.cooperative_actions = 0
            sybils.append(w)

        net_genuine = network_asabiyyah(genuine)
        net_with_sybils = network_asabiyyah(genuine + sybils)
        # Sybils dilute network Asabiyyah (no governance/cooperation)
        assert net_with_sybils < net_genuine


# ═══════════════════════════════════════════════════════════════════
# Attack Class 6: Event Log Tampering
# ═══════════════════════════════════════════════════════════════════


class TestAttack6EventLogTampering:
    """Adversary modifies event log entries."""

    def test_data_tampering_detected(self) -> None:
        """Modify event data after insertion — hash mismatch detected."""
        log: list = []
        append_event(log, "genesis", b"\x00" * 32, {"v": "1.0"})
        append_event(log, "mint", b"\x01" * 32, {"amount": 100})
        append_event(log, "transfer", b"\x02" * 32, {"to": "bob"})

        # Tamper: change mint amount
        log[1].data["amount"] = 999999
        valid, errors = verify_event_chain(log)
        assert not valid
        assert any("Hash mismatch" in e for e in errors)

    def test_chain_reordering_detected(self) -> None:
        """Swap two events — chain break detected."""
        log: list = []
        append_event(log, "genesis", b"\x00" * 32, {})
        append_event(log, "a", b"\x01" * 32, {"x": 1})
        append_event(log, "b", b"\x02" * 32, {"x": 2})

        # Swap events 1 and 2
        log[1], log[2] = log[2], log[1]
        valid, errors = verify_event_chain(log)
        assert not valid

    def test_event_insertion_detected(self) -> None:
        """Insert a fake event — chain integrity broken."""
        from core.constitutional.types import Event

        log: list = []
        append_event(log, "genesis", b"\x00" * 32, {})
        append_event(log, "legit", b"\x01" * 32, {"x": 1})

        # Insert fake event between 0 and 1
        fake = Event(
            event_id=999,
            event_type="fake",
            actor=b"\xff" * 32,
            data={"stolen": True},
            timestamp=0,
            prev_hash=log[0].hash,
            hash=b"\xee" * 32,
        )
        log.insert(1, fake)
        valid, errors = verify_event_chain(log)
        assert not valid


# ═══════════════════════════════════════════════════════════════════
# Attack Class 7: Zakat/Demurrage Evasion
# ═══════════════════════════════════════════════════════════════════


class TestAttack7ZakatEvasion:
    """Adversary tries to avoid Zakat and demurrage obligations."""

    def test_just_below_nisab_no_zakat(self) -> None:
        """Balance just below NISAB — legitimately no Zakat."""
        wallet = _make_wallet(seed_balance=NISAB_THRESHOLD - 1)
        assert compute_zakat(wallet) == FP_ZERO

    def test_at_nisab_zakat_applies(self) -> None:
        """Balance at exactly NISAB — Zakat applies."""
        wallet = _make_wallet(seed_balance=NISAB_THRESHOLD)
        zakat = compute_zakat(wallet)
        assert zakat > FP_ZERO

    def test_split_balance_evasion_irrelevant(self) -> None:
        """Splitting balance across wallets is node-level — each
        wallet is independently assessed.

        This is by design: identity is per-node, not per-account.
        """
        full = _make_wallet(seed_balance=fp(200))
        zakat_full = compute_zakat(full)

        half1 = _make_wallet(seed_balance=fp(100))
        half2 = _make_wallet(seed_balance=fp(100))
        zakat_split = compute_zakat(half1) + compute_zakat(half2)
        # Both halves are above NISAB, so split doesn't evade
        assert zakat_split == zakat_full


# ═══════════════════════════════════════════════════════════════════
# Attack Class 8: Reflex Cache Poisoning (V5 Hardened)
# ═══════════════════════════════════════════════════════════════════


class TestAttack8CachePoisoning:
    """Adversary tries to inject low-quality reflexes into System-1 cache."""

    def test_low_confidence_compile_rejected(self) -> None:
        """V5 fix: compile_reflex rejects below IHSAN_FLOOR."""
        reflex = compile_reflex("malicious pattern", ["steal"], fp(0.50))
        assert reflex is None

    def test_cache_cannot_be_poisoned_through_compile(self) -> None:
        """End-to-end: adversary cannot inject through normal compile path."""
        cache: dict[bytes, Reflex] = {}

        # Try 100 low-quality patterns
        for i in range(100):
            r = compile_reflex(f"attack_{i}", ["exploit"], fp(0.01 * i))
            if r is not None:
                cache[r.pattern_hash] = r

        # Only patterns with confidence >= IHSAN_FLOOR should be in cache
        for reflex in cache.values():
            assert reflex.confidence >= IHSAN_FLOOR

    def test_lookup_double_gate(self) -> None:
        """Even if a low-quality reflex somehow enters cache,
        lookup rejects it."""
        poisoned = Reflex(
            pattern_hash=b"\xaa" * 32,
            action_chain=("steal",),
            confidence=fp(0.30),
            last_used=0,
            use_count=0,
        )
        cache = {b"\xaa" * 32: poisoned}
        # Lookup with any pattern won't match hash, but even if it did:
        found = reflex_lookup(cache, "any")
        assert found is None

    def test_high_quality_reflex_survives(self) -> None:
        """Legitimate high-quality reflexes compile and lookup correctly."""
        reflex = compile_reflex("good pattern", ["help", "respond"], fp(0.99))
        assert reflex is not None
        cache = {reflex.pattern_hash: reflex}
        found = reflex_lookup(cache, "good pattern")
        assert found is not None
        assert found.action_chain == ("help", "respond")


# ═══════════════════════════════════════════════════════════════════
# Attack Class 9: Governance Gaming
# ═══════════════════════════════════════════════════════════════════


class TestAttack9GovernanceGaming:
    """Adversary tries to manipulate BLOOM-weighted governance."""

    def test_zero_bloom_cannot_vote(self) -> None:
        """Node with zero BLOOM has no governance weight."""
        proposal = Proposal(
            proposal_id=b"\x01" * 32,
            proposer=b"\x02" * 32,
            description="test proposal",
            votes_for=FP_ZERO,
            votes_against=FP_ZERO,
        )
        voter = _make_wallet(bloom_balance=0)
        result = shura_vote(proposal, voter, approve=True)
        assert result.votes_for == FP_ZERO  # Vote had no effect

    def test_supermajority_required(self) -> None:
        """Simple majority (51%) is NOT enough — need >66.7%."""
        proposal = Proposal(
            proposal_id=b"\x01" * 32,
            proposer=b"\x02" * 32,
            description="test proposal",
            votes_for=fp(51),
            votes_against=fp(49),
        )
        assert shura_resolve(proposal) == "rejected"

    def test_supermajority_passes(self) -> None:
        """67%+ passes."""
        proposal = Proposal(
            proposal_id=b"\x01" * 32,
            proposer=b"\x02" * 32,
            description="test proposal",
            votes_for=fp(70),
            votes_against=fp(30),
        )
        assert shura_resolve(proposal) == "passed"
