from __future__ import annotations

from pathlib import Path

from core.token.emission_decay import LogisticEmissionGate
from core.token.mint import TokenMinter
from core.token.rl_rewards import (
    composite_reward,
    compute_agent_reward,
    enforce_agent_gini,
    token_efficiency_reward,
    update_agent_reputation,
)


def _new_minter(tmp_path: Path, name: str) -> TokenMinter:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    return TokenMinter.create(
        db_path=root / "ledger.db",
        log_path=root / "ledger.jsonl",
    )


def test_composite_reward_bounded() -> None:
    score = composite_reward(
        {
            "snr": 1.2,
            "ihsan": 1.1,
            "efficiency": 1.5,
            "user_feedback": 1.0,
            "penalties": 0.0,
        }
    )
    assert 0.0 <= score <= 1.0


def test_composite_reward_penalty_reduces_score() -> None:
    base = composite_reward(
        {"snr": 0.95, "ihsan": 0.95, "efficiency": 0.8, "user_feedback": 0.8}
    )
    penalized = composite_reward(
        {
            "snr": 0.95,
            "ihsan": 0.95,
            "efficiency": 0.8,
            "user_feedback": 0.8,
            "penalties": 0.3,
        }
    )
    assert penalized < base


def test_composite_reward_rejects_low_ihsan() -> None:
    score = composite_reward(
        {"snr": 0.95, "ihsan": 0.94, "efficiency": 1.0, "user_feedback": 1.0}
    )
    assert score == 0.0


def test_composite_reward_quarantines_low_snr() -> None:
    score = composite_reward(
        {"snr": 0.84, "ihsan": 1.0, "efficiency": 1.0, "user_feedback": 1.0}
    )
    assert score == 0.0


def test_token_efficiency_reward_scales() -> None:
    high = token_efficiency_reward(tokens_used=200, quality=0.95)
    low = token_efficiency_reward(tokens_used=4000, quality=0.2)
    assert high > low


def test_compute_agent_reward_mints_seed(tmp_path: Path) -> None:
    minter = _new_minter(tmp_path, "seed")
    receipt = compute_agent_reward(
        agent_id="researcher",
        mission_result={
            "snr": 0.9,
            "ihsan": 0.95,
            "tokens_used": 500,
            "quality": 0.9,
            "user_feedback": 0.8,
        },
        minter=minter,
        emission_gate=None,
        epoch_id="epoch-1",
    )
    assert receipt.success is True
    assert receipt.tx_entry is not None
    assert receipt.tx_entry.to_account == "researcher"


def test_compute_agent_reward_blocks_unverified_impact(tmp_path: Path) -> None:
    minter = _new_minter(tmp_path, "unverified-seed")
    receipt = compute_agent_reward(
        agent_id="researcher",
        mission_result={
            "snr": 0.0,
            "ihsan": 0.0,
            "efficiency": 1.0,
            "user_feedback": 1.0,
        },
        minter=minter,
        emission_gate=None,
        epoch_id="epoch-rejected",
    )
    assert receipt.success is False
    assert receipt.error == "unverified_impact"
    assert receipt.tx_entry is None


def test_compute_agent_reward_emission_gate_throttles(tmp_path: Path) -> None:
    gate = LogisticEmissionGate(e_max=1000.0, g_target=0.35, steepness=20.0)

    ungated_minter = _new_minter(tmp_path, "ungated")
    gated_minter = _new_minter(tmp_path, "gated")

    # Create whale-heavy distribution first.
    ungated_minter.mint_seed("whale", 50_000, epoch_id="epoch-0", poi_score=1.0)
    gated_minter.mint_seed("whale", 50_000, epoch_id="epoch-0", poi_score=1.0)

    mission = {
        "snr": 0.95,
        "ihsan": 0.95,
        "tokens_used": 400,
        "quality": 0.95,
        "user_feedback": 0.9,
        "seed_base": 1000.0,
    }

    ungated = compute_agent_reward("agent-a", mission, ungated_minter, None, "epoch-1")
    gated = compute_agent_reward("agent-a", mission, gated_minter, gate, "epoch-1")

    assert ungated.success is True
    assert gated.success is True
    assert gated.tx_entry is not None
    assert ungated.tx_entry is not None
    assert gated.tx_entry.amount <= ungated.tx_entry.amount


def test_update_agent_reputation_mints_impt(tmp_path: Path) -> None:
    minter = _new_minter(tmp_path, "impt")
    receipt = update_agent_reputation("creator", 0.81, minter)
    assert receipt.success is True
    assert receipt.tx_entry is not None
    assert receipt.tx_entry.to_account == "creator"


def test_update_agent_reputation_blocks_zero_reward(tmp_path: Path) -> None:
    minter = _new_minter(tmp_path, "unverified-impt")
    receipt = update_agent_reputation("creator", 0.0, minter)
    assert receipt.success is False
    assert receipt.error == "unverified_impact"
    assert receipt.tx_entry is None


def test_enforce_agent_gini_reports_noncompliance(tmp_path: Path) -> None:
    minter = _new_minter(tmp_path, "gini")
    minter.mint_seed("whale", 10_000, epoch_id="epoch-1", poi_score=1.0)
    minter.mint_seed("a", 1, epoch_id="epoch-1", poi_score=0.5)
    minter.mint_seed("b", 1, epoch_id="epoch-1", poi_score=0.5)

    report = enforce_agent_gini(minter, ["whale", "a", "b"], threshold=0.35)
    assert report["gini"] > 0.35
    assert report["compliant"] is False


def test_enforce_agent_gini_reports_compliance_for_equal_distribution(
    tmp_path: Path,
) -> None:
    minter = _new_minter(tmp_path, "gini-equal")
    minter.mint_seed("a", 10, epoch_id="epoch-1", poi_score=1.0)
    minter.mint_seed("b", 10, epoch_id="epoch-1", poi_score=1.0)
    minter.mint_seed("c", 10, epoch_id="epoch-1", poi_score=1.0)

    report = enforce_agent_gini(minter, ["a", "b", "c"], threshold=0.35)
    assert report["compliant"] is True
