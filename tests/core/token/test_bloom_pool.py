"""Regression coverage for BLOOM token and community pool behavior."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.integration.constants import ADL_GINI_THRESHOLD
from core.token.bloom import (
    COMMUNITY_POOL_SPLIT,
    BloomBalance,
    CommunityPool,
    TokenMinter,
    WalletState,
    check_gini_invariant,
    compute_gini,
)


class TestBloomBalance:
    def test_apply_decay_skips_recent_balance(self) -> None:
        bloom = BloomBalance(node_id="node0", balance=10.0)

        assert bloom.apply_decay() == 0.0
        assert bloom.balance == 10.0

    def test_apply_decay_reduces_old_balance_and_updates_weight(self) -> None:
        bloom = BloomBalance(
            node_id="node0",
            balance=10.0,
            last_decay=(datetime.now(timezone.utc) - timedelta(days=65)).isoformat(),
        )

        decayed = bloom.apply_decay()

        assert decayed > 0
        assert bloom.balance < 10.0
        assert bloom.governance_weight == pytest.approx(bloom.balance)


class TestCommunityPool:
    def test_receive_tracks_pool_totals(self) -> None:
        pool = CommunityPool()

        pool.receive(amount=5.0, source="seed_mint:node0", evidence_hash="hash")

        assert pool.total_received == 5.0
        assert pool.current_balance == 5.0

    def test_distribute_records_successful_distribution(self) -> None:
        pool = CommunityPool(current_balance=5.0, total_received=5.0)

        assert pool.distribute(2.0, "zakat", "recipient", "evidence") is True

        assert pool.current_balance == 3.0
        assert pool.total_distributed == 2.0
        assert pool.distributions[0]["category"] == "zakat"
        assert pool.distributions[0]["recipient"] == "recipient"

    def test_distribute_rejects_insufficient_balance(self) -> None:
        pool = CommunityPool(current_balance=1.0)

        assert pool.distribute(2.0, "zakat", "recipient", "evidence") is False
        assert pool.current_balance == 1.0
        assert pool.distributions == []


class TestBloomTokenMinter:
    def test_compute_reward_scales_with_steps_and_ihsan(self) -> None:
        minter = TokenMinter(community_pool=CommunityPool())

        simple = minter.compute_reward(ihsan=0.95, steps=1)
        complex_work = minter.compute_reward(ihsan=0.95, steps=7)

        assert simple > 0
        assert complex_work > simple

    def test_mint_seed_splits_node_and_pool_share(self) -> None:
        pool = CommunityPool()
        minter = TokenMinter(community_pool=pool)
        wallet = WalletState(node_id="node0")

        result = minter.mint_seed(wallet, amount=10.0, poi_evidence="hash", ihsan=0.96)

        assert result["minted"] is True
        assert result["node_share"] == pytest.approx(10.0 * (1 - COMMUNITY_POOL_SPLIT))
        assert result["pool_share"] == pytest.approx(10.0 * COMMUNITY_POOL_SPLIT)
        assert wallet.seed_balance == pytest.approx(result["node_share"])
        assert pool.current_balance == pytest.approx(result["pool_share"])
        assert minter.total_seed_minted == 10.0
        assert minter.mint_log[-1] == result

    def test_mint_seed_rejects_below_ihsan_floor(self) -> None:
        pool = CommunityPool()
        minter = TokenMinter(community_pool=pool)
        wallet = WalletState(node_id="node0")

        result = minter.mint_seed(wallet, amount=10.0, poi_evidence="hash", ihsan=0.1)

        assert result == {"minted": False, "reason": "below_ihsan_floor"}
        assert wallet.seed_balance == 0.0
        assert pool.current_balance == 0.0
        assert minter.total_seed_minted == 0.0

    def test_mint_bloom_is_soulbound_governance_weight(self) -> None:
        minter = TokenMinter(community_pool=CommunityPool())
        wallet = WalletState(node_id="node0")

        result = minter.mint_bloom(
            wallet,
            amount=3.0,
            contribution_type="governance_vote",
            evidence_hash="hash",
        )

        assert result["minted"] is True
        assert result["type"] == "BLOOM"
        assert wallet.bloom.balance == 3.0
        assert wallet.bloom.lifetime_earned == 3.0
        assert wallet.bloom.governance_weight == 3.0
        assert minter.total_bloom_minted == 3.0

    def test_mint_branch_records_attestation_reward(self) -> None:
        minter = TokenMinter(community_pool=CommunityPool())
        wallet = WalletState(node_id="node0")

        result = minter.mint_branch(
            wallet,
            amount=2.0,
            attestation_type="peer_review",
            evidence_hash="hash",
        )

        assert result == {
            "minted": True,
            "amount": 2.0,
            "type": "BRANCH",
            "attestation": "peer_review",
        }
        assert wallet.branch_balance == 2.0
        assert minter.total_branch_minted == 2.0


class TestBloomGiniInvariant:
    def test_compute_gini_handles_empty_zero_and_equal_balances(self) -> None:
        assert compute_gini([]) == 0.0
        assert compute_gini([0.0, 0.0]) == 0.0
        assert compute_gini([10.0, 10.0, 10.0]) == 0.0

    def test_compute_gini_detects_inequality(self) -> None:
        assert compute_gini([0.0, 0.0, 100.0]) > ADL_GINI_THRESHOLD

    def test_check_gini_invariant_passes_balanced_wallets(self) -> None:
        wallets = [
            WalletState(node_id="a", seed_balance=10.0),
            WalletState(node_id="b", seed_balance=11.0),
            WalletState(node_id="c", seed_balance=12.0),
        ]

        result = check_gini_invariant(wallets)

        assert result["passed"] is True
        assert result["gini"] <= ADL_GINI_THRESHOLD
        assert result["node_count"] == 3
        assert result["total_seed"] == 33.0

    def test_check_gini_invariant_flags_unbalanced_wallets(self) -> None:
        wallets = [
            WalletState(node_id="a", seed_balance=100.0),
            WalletState(node_id="b", seed_balance=0.0),
            WalletState(node_id="c", seed_balance=0.0),
        ]

        result = check_gini_invariant(wallets)

        assert result["passed"] is False
        assert result["gini"] > ADL_GINI_THRESHOLD
