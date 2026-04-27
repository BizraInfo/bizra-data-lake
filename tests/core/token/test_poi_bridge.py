"""Regression coverage for Proof-of-Impact token bridge wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from core.proof_engine.poi_engine import AuditTrail, PoIReasonCode, ProofOfImpact
from core.token.poi_bridge import PoITokenBridge


def _audit_trail() -> AuditTrail:
    return AuditTrail(
        epoch_id="epoch-1",
        poi_scores=[
            ProofOfImpact(
                contributor_id="alice",
                contribution_score=0.9,
                reach_score=0.8,
                longevity_score=0.7,
                poi_score=0.6,
                alpha=0.5,
                beta=0.3,
                gamma=0.2,
                config_digest="config",
                computation_id="comp-alice",
                epoch_id="epoch-1",
                reason_code=PoIReasonCode.POI_OK,
            ),
            ProofOfImpact(
                contributor_id="bob",
                contribution_score=0.0,
                reach_score=0.0,
                longevity_score=0.0,
                poi_score=0.0,
                alpha=0.5,
                beta=0.3,
                gamma=0.2,
                config_digest="config",
                computation_id="comp-bob",
                epoch_id="epoch-1",
                reason_code=PoIReasonCode.POI_OK,
            ),
        ],
        gini_coefficient=0.1,
        rebalance_triggered=False,
        config_digest="config",
    )


def _receipt(success: bool, amount: float) -> SimpleNamespace:
    return SimpleNamespace(success=success, tx_entry=SimpleNamespace(amount=amount))


def test_distribute_epoch_mints_seed_and_positive_impt(monkeypatch) -> None:
    distribution = SimpleNamespace(
        epoch_id="epoch-1",
        distributions={"alice": 60.0, "bob": 40.0},
        total_minted=100.0,
        gini_coefficient=0.1,
        to_dict=lambda: {"epoch_id": "epoch-1"},
    )
    monkeypatch.setattr(
        "core.token.poi_bridge.compute_token_distribution",
        lambda audit, epoch_reward, scaling_factor: distribution,
    )
    minter = MagicMock()
    minter.distribute_from_poi.return_value = [
        _receipt(True, 60.0),
        _receipt(False, 0.0),
    ]
    minter.mint_impt.return_value = _receipt(True, 60.0)

    result = PoITokenBridge(minter).distribute_epoch(
        _audit_trail(),
        epoch_reward=100.0,
        scaling_factor=2.0,
        impt_multiplier=100.0,
    )

    minter.distribute_from_poi.assert_called_once_with(
        distributions=distribution.distributions,
        epoch_id="epoch-1",
        epoch_reward=100.0,
        poi_scores={"alice": 0.6, "bob": 0.0},
    )
    minter.mint_impt.assert_called_once_with(
        to_account="alice",
        amount=60.0,
        epoch_id="epoch-1",
        poi_score=0.6,
        memo="IMPT reputation: PoI=0.6000, epoch=epoch-1",
    )
    assert result["summary"]["seed_succeeded"] == 1
    assert result["summary"]["impt_succeeded"] == 1
    assert result["summary"]["contributors"] == 2
    assert result["distribution"] == {"epoch_id": "epoch-1"}


def test_distribute_epoch_can_skip_impt_minting(monkeypatch) -> None:
    distribution = SimpleNamespace(
        epoch_id="epoch-2",
        distributions={"alice": 10.0},
        total_minted=10.0,
        gini_coefficient=0.0,
        to_dict=lambda: {"epoch_id": "epoch-2"},
    )
    monkeypatch.setattr(
        "core.token.poi_bridge.compute_token_distribution",
        lambda audit, epoch_reward, scaling_factor: distribution,
    )
    minter = MagicMock()
    minter.distribute_from_poi.return_value = [_receipt(True, 10.0)]

    result = PoITokenBridge(minter).distribute_epoch(
        _audit_trail(),
        epoch_reward=10.0,
        mint_impt=False,
    )

    minter.mint_impt.assert_not_called()
    assert result["impt_receipts"] == []
    assert result["summary"]["impt_distributions"] == 0


def test_status_delegates_to_minter() -> None:
    minter = MagicMock()
    minter.status.return_value = {"ledger": "ok"}

    assert PoITokenBridge(minter).status() == {"ledger": "ok"}
