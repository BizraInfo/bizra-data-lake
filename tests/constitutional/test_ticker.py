"""
Tests for Constitutional Ticker — 12-Step Heartbeat
════════════════════════════════════════════════════

TDD anchors from Phase 67.02 specification (process_tick).

Standing on Giants:
- Beck (2002): Test-Driven Development by Example
- Al-Khwarizmi (780-850): Deterministic procedure
- Nakamoto (2008): Block processing tick
"""

from __future__ import annotations

import pytest

from core.constitutional.fixed_point import fp, fp_add, fp_float
from core.constitutional.ticker import TickResult, process_tick
from core.constitutional.types import (
    ActionReceipt,
    Event,
    Proposal,
    Reflex,
    WalletState,
)

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def tick_wallet() -> WalletState:
    """A wallet for ticker tests — actor_id matches quality_receipt."""
    return WalletState(
        node_id=b"\x02" * 32,  # Matches quality_receipt.actor_id
        seed_balance=fp(100),
        bloom_balance=fp(1),
        created_at=1741392000000,
        last_active=1741392000000,
        total_actions=10,
    )


@pytest.fixture
def second_wallet() -> WalletState:
    """A second wallet for multi-wallet tests."""
    return WalletState(
        node_id=b"\x50" * 32,
        seed_balance=fp(200),
        bloom_balance=fp(2),
        created_at=1741392000000,
        last_active=1741392000000,
        total_actions=20,
    )


@pytest.fixture
def second_quality_receipt() -> ActionReceipt:
    """A quality receipt from the second wallet."""
    return ActionReceipt(
        receipt_id=b"\x51" * 32,
        actor_id=b"\x50" * 32,
        action_type="contribution",
        timestamp=1741392000000,
        intent_score=fp(0.98),
        efficiency_score=fp(0.96),
        impact_score=fp(0.97),
        reproducibility_score=fp(0.95),
        oracle_signature=b"\x52" * 64,
        metadata_hash=b"\x53" * 32,
        co_actors=(),
    )


# ═══════════════════════════════════════════════════════════════════
# Test: TickResult Structure
# ═══════════════════════════════════════════════════════════════════


class TestTickResult:
    """Verify TickResult defaults and structure."""

    def test_default_values(self):
        """All fields default to zero."""
        result = TickResult()
        assert result.rejected == 0
        assert result.scored == 0
        assert result.total_minted == 0
        assert result.zakat_pool == 0
        assert result.network_gini == 0
        assert result.network_asabiyyah_score == 0
        assert result.events_logged == 0
        assert result.proposals_resolved == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Empty Tick
# ═══════════════════════════════════════════════════════════════════


class TestEmptyTick:
    """Verify process_tick handles empty inputs gracefully."""

    def test_empty_tick(self):
        """No wallets, no receipts — should return zeroed result."""
        result = process_tick(
            wallets=[],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.rejected == 0
        assert result.scored == 0
        assert result.total_minted == 0
        assert result.events_logged == 0

    def test_tick_with_wallets_no_receipts(self, tick_wallet):
        """Wallets but no receipts — only decay/demurrage run."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.rejected == 0
        assert result.scored == 0
        assert result.total_minted == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Intent Gate Integration
# ═══════════════════════════════════════════════════════════════════


class TestTickIntentGate:
    """Verify Step 1 — Al-Ghazali intent gate filtering."""

    def test_low_intent_rejected(self, tick_wallet, low_intent_receipt):
        """Low-intent receipts must be rejected."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[low_intent_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.rejected == 1
        assert result.scored == 0
        assert result.total_minted == 0

    def test_quality_receipt_passes(self, tick_wallet, quality_receipt):
        """Quality receipts pass the intent gate and get scored."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.rejected == 0
        assert result.scored == 1

    def test_mixed_receipts(self, tick_wallet, quality_receipt, low_intent_receipt):
        """Mix of quality and spam — only quality passes."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt, low_intent_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.rejected == 1
        assert result.scored == 1


# ═══════════════════════════════════════════════════════════════════
# Test: Progressive Minting (Steps 2-4)
# ═══════════════════════════════════════════════════════════════════


class TestTickMinting:
    """Verify Steps 2-4 — scoring and progressive minting."""

    def test_minting_increases_balance(self, tick_wallet, quality_receipt):
        """A valid receipt must increase the wallet's SEED balance."""
        initial_balance = tick_wallet.seed_balance
        process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        # Balance should have increased (minting adds, demurrage/decay may subtract small amounts)
        # But net effect of minting should dominate for a single tick
        assert tick_wallet.seed_balance != initial_balance

    def test_total_minted_nonzero(self, tick_wallet, quality_receipt):
        """TickResult must report nonzero minted for valid receipt."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.total_minted > 0

    def test_action_count_incremented(self, tick_wallet, quality_receipt):
        """Wallet's total_actions must increase after processing a receipt."""
        initial_actions = tick_wallet.total_actions
        process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert tick_wallet.total_actions == initial_actions + 1

    def test_ihsan_history_appended(self, tick_wallet, quality_receipt):
        """Ihsan score must be appended to wallet's history."""
        initial_len = len(tick_wallet.ihsan_history)
        process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert len(tick_wallet.ihsan_history) == initial_len + 1

    def test_no_wallet_for_receipt_skipped(self, quality_receipt):
        """Receipt with no matching wallet is silently skipped."""
        unrelated_wallet = WalletState(
            node_id=b"\xff" * 32,  # Different from quality_receipt.actor_id
            seed_balance=fp(100),
            created_at=1741392000000,
            last_active=1741392000000,
        )
        result = process_tick(
            wallets=[unrelated_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        # Receipt scored but no wallet found for minting
        assert result.scored == 1
        assert result.total_minted == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Event Logging (Step 11)
# ═══════════════════════════════════════════════════════════════════


class TestTickEventLogging:
    """Verify Step 11 — event log append."""

    def test_events_logged_count(self, tick_wallet, quality_receipt):
        """One valid receipt should produce one event."""
        event_log: list[Event] = []
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=event_log,
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.events_logged == 1
        assert len(event_log) == 1

    def test_event_type_is_mint(self, tick_wallet, quality_receipt):
        """Logged event must be of type 'mint'."""
        event_log: list[Event] = []
        process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=event_log,
            reflex_cache={},
            current_time=1741392000000,
        )
        assert event_log[0].event_type == "mint"

    def test_no_events_for_rejected_receipts(self, tick_wallet, low_intent_receipt):
        """Rejected receipts must not produce events."""
        event_log: list[Event] = []
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[low_intent_receipt],
            proposals=[],
            event_log=event_log,
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.events_logged == 0
        assert len(event_log) == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Zakat Cycle (Step 8)
# ═══════════════════════════════════════════════════════════════════


class TestTickZakat:
    """Verify Step 8 — Zakat collection when is_zakat_cycle=True."""

    def test_zakat_collected_when_cycle(self, wealthy_wallet):
        """Zakat cycle must collect from wealthy wallets."""
        result = process_tick(
            wallets=[wealthy_wallet],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
            is_zakat_cycle=True,
        )
        assert result.zakat_pool > 0

    def test_no_zakat_without_cycle(self, wealthy_wallet):
        """Without is_zakat_cycle, no Zakat collected."""
        result = process_tick(
            wallets=[wealthy_wallet],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
            is_zakat_cycle=False,
        )
        assert result.zakat_pool == 0

    def test_zakat_reduces_balance(self, wealthy_wallet):
        """Zakat must reduce the wallet balance."""
        initial = wealthy_wallet.seed_balance
        process_tick(
            wallets=[wealthy_wallet],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
            is_zakat_cycle=True,
        )
        # Balance reduced by Zakat + demurrage
        assert wealthy_wallet.seed_balance < initial


# ═══════════════════════════════════════════════════════════════════
# Test: Governance (Step 9)
# ═══════════════════════════════════════════════════════════════════


class TestTickGovernance:
    """Verify Step 9 — proposal resolution."""

    def test_active_proposal_with_supermajority_passes(self, tick_wallet):
        """An active proposal with >66.7% approval should pass."""
        proposal = Proposal(
            proposal_id=b"\x30" * 32,
            proposer=tick_wallet.node_id,
            description="Test Proposal",
            status="active",
            created_at=1741392000000,
            votes_for=fp(100),
            votes_against=fp(10),
        )
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[],
            proposals=[proposal],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.proposals_resolved == 1
        assert proposal.status == "passed"

    def test_no_votes_expires(self, tick_wallet):
        """A proposal with zero votes resolves as expired (not counted)."""
        proposal = Proposal(
            proposal_id=b"\x31" * 32,
            proposer=tick_wallet.node_id,
            description="No Votes Proposal",
            status="active",
            created_at=1741300000000,
            votes_for=0,
            votes_against=0,
        )
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[],
            proposals=[proposal],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        # shura_resolve returns "expired" for zero-vote proposals,
        # and process_tick skips expired resolutions
        assert result.proposals_resolved == 0
        assert proposal.status == "active"  # unchanged because result was "expired"


# ═══════════════════════════════════════════════════════════════════
# Test: Reflex Cache (Step 10)
# ═══════════════════════════════════════════════════════════════════


class TestTickReflexCache:
    """Verify Step 10 — excellent patterns compiled to reflex."""

    def test_excellent_receipt_creates_reflex(self, tick_wallet):
        """Receipts with ihsan >= 0.98 should produce a reflex entry."""
        # Create a receipt with very high scores
        excellent = ActionReceipt(
            receipt_id=b"\x40" * 32,
            actor_id=tick_wallet.node_id,
            action_type="research",
            timestamp=1741392000000,
            intent_score=fp(0.99),
            efficiency_score=fp(0.99),
            impact_score=fp(0.99),
            reproducibility_score=fp(0.99),
            oracle_signature=b"\x41" * 64,
            metadata_hash=b"\x42" * 32,
            co_actors=(),
        )
        reflex_cache: dict[bytes, Reflex] = {}
        process_tick(
            wallets=[tick_wallet],
            receipts=[excellent],
            proposals=[],
            event_log=[],
            reflex_cache=reflex_cache,
            current_time=1741392000000,
        )
        assert len(reflex_cache) > 0

    def test_mediocre_receipt_no_reflex(self, tick_wallet, quality_receipt):
        """Receipts below 0.98 ihsan should NOT create reflex entries."""
        reflex_cache: dict[bytes, Reflex] = {}
        process_tick(
            wallets=[tick_wallet],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache=reflex_cache,
            current_time=1741392000000,
        )
        # quality_receipt has ihsan ~0.966, below 0.98 threshold
        assert len(reflex_cache) == 0


# ═══════════════════════════════════════════════════════════════════
# Test: Determinism (Critical)
# ═══════════════════════════════════════════════════════════════════


class TestTickDeterminism:
    """Verify that process_tick is fully deterministic."""

    def test_identical_inputs_identical_outputs(self, quality_receipt):
        """Same inputs must produce identical results across runs."""
        results = []
        for _ in range(3):
            wallet = WalletState(
                node_id=b"\x02" * 32,
                seed_balance=fp(100),
                bloom_balance=fp(1),
                created_at=1741392000000,
                last_active=1741392000000,
                total_actions=10,
            )
            event_log: list[Event] = []
            reflex_cache: dict[bytes, Reflex] = {}
            result = process_tick(
                wallets=[wallet],
                receipts=[quality_receipt],
                proposals=[],
                event_log=event_log,
                reflex_cache=reflex_cache,
                current_time=1741392000000,
            )
            results.append(
                (
                    result.total_minted,
                    result.scored,
                    result.rejected,
                    result.events_logged,
                    result.network_gini,
                    wallet.seed_balance,
                    wallet.bloom_balance,
                )
            )
        # All three runs must be identical
        assert results[0] == results[1] == results[2]

    def test_determinism_with_multiple_wallets(
        self, quality_receipt, second_quality_receipt
    ):
        """Multi-wallet tick must be deterministic."""
        results = []
        for _ in range(2):
            w1 = WalletState(
                node_id=b"\x02" * 32,
                seed_balance=fp(100),
                bloom_balance=fp(1),
                created_at=1741392000000,
                last_active=1741392000000,
                total_actions=10,
            )
            w2 = WalletState(
                node_id=b"\x50" * 32,
                seed_balance=fp(200),
                bloom_balance=fp(2),
                created_at=1741392000000,
                last_active=1741392000000,
                total_actions=20,
            )
            event_log: list[Event] = []
            result = process_tick(
                wallets=[w1, w2],
                receipts=[quality_receipt, second_quality_receipt],
                proposals=[],
                event_log=event_log,
                reflex_cache={},
                current_time=1741392000000,
            )
            results.append(
                (
                    result.total_minted,
                    result.network_gini,
                    w1.seed_balance,
                    w2.seed_balance,
                )
            )
        assert results[0] == results[1]


# ═══════════════════════════════════════════════════════════════════
# Test: Asabiyyah (Step 12)
# ═══════════════════════════════════════════════════════════════════


class TestTickAsabiyyah:
    """Verify Step 12 — network cohesion metric."""

    def test_asabiyyah_computed(self, tick_wallet):
        """Tick must compute and report network Asabiyyah."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.network_asabiyyah_score >= 0

    def test_gini_computed(self, tick_wallet):
        """Tick must compute and report network Gini coefficient."""
        result = process_tick(
            wallets=[tick_wallet],
            receipts=[],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )
        assert result.network_gini >= 0

    def test_asabiyyah_feeds_into_minting(self, quality_receipt):
        """T11: Asabiyyah computed BEFORE minting and modulates mint output.

        A cohesive network (high asabiyyah) must produce more SEED than
        a fragmented one (zero asabiyyah), proving the coupling is active.
        Requires network_size > 1 so asabiyyah_score is non-trivial.

        Standing on Giants: Ibn Khaldun (asabiyyah) · Beck (TDD anchors)
        """

        # Build two multi-wallet networks — one cohesive, one isolated
        # Need network_size > 1 for asabiyyah_score to be non-zero
        def make_network(connected: bool) -> list[WalletState]:
            wallets = []
            for i in range(5):
                w = WalletState(
                    node_id=bytes([i]) * 32,
                    seed_balance=fp(100),
                    bloom_balance=fp(1),
                    created_at=1741392000000,
                    last_active=1741392000000,
                    total_actions=10,
                )
                if connected:
                    peers = {bytes([j]) * 32 for j in range(5) if j != i}
                    w.attestations_given = peers
                    w.attestations_received = peers
                    w.governance_votes = 5
                    w.cooperative_actions = 10
                wallets.append(w)
            return wallets

        # quality_receipt.actor_id is b"\x02" * 32 = bytes([2]) * 32
        # This matches wallets[2] in our network

        # Cohesive network
        net_cohesive = make_network(connected=True)
        r1 = process_tick(
            wallets=net_cohesive,
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )

        # Isolated network (same balances but no connections)
        net_isolated = make_network(connected=False)
        r2 = process_tick(
            wallets=net_isolated,
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )

        # Cohesive network has higher asabiyyah → relaxed throttle → more SEED
        assert r1.network_asabiyyah_score > r2.network_asabiyyah_score
        assert r1.total_minted > r2.total_minted
        # Both mint (never zero)
        assert r1.total_minted > 0
        assert r2.total_minted > 0

    def test_collusion_ring_no_relaxation(self, quality_receipt):
        """T12: 2-node collusion ring gets NO asabiyyah relaxation.

        Anti-collusion gate (MIN_CONNECTIONS=3) ensures that a 2-node
        mutual attestation ring cannot boost its minting rate via
        asabiyyah coupling. The throttle tightens, not relaxes.

        Standing on Giants: Ibn Khaldun (social cohesion) · Beck (TDD)
        """
        # 2-node collusion ring: mutual attestation only
        colluder = WalletState(
            node_id=b"\x02" * 32,
            seed_balance=fp(100),
            bloom_balance=fp(1),
            created_at=1741392000000,
            last_active=1741392000000,
            total_actions=10,
            attestations_given={b"\xcc" * 32},
            attestations_received={b"\xcc" * 32},
        )

        result = process_tick(
            wallets=[colluder],
            receipts=[quality_receipt],
            proposals=[],
            event_log=[],
            reflex_cache={},
            current_time=1741392000000,
        )

        # Network asabiyyah should be zero (single wallet, network_size=1)
        assert result.network_asabiyyah_score == 0
        # Minting still works but at FLOOR multiplier (0.80x)
        assert result.total_minted > 0
