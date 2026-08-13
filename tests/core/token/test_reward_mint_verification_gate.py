from __future__ import annotations

from pathlib import Path

from core.token.mint import TokenMinter
from core.token.rl_rewards import compute_agent_reward


def _new_minter(tmp_path: Path) -> TokenMinter:
    root = tmp_path / "verification-gate"
    root.mkdir(parents=True, exist_ok=True)
    return TokenMinter.create(
        db_path=root / "ledger.db",
        log_path=root / "ledger.jsonl",
    )


def _mission(ihsan: float) -> dict[str, float | int]:
    return {
        "snr": 0.99,
        "ihsan": ihsan,
        "tokens_used": 100,
        "quality": 0.99,
        "user_feedback": 1.0,
        "seed_base": 100.0,
    }


def test_seed_mint_rejects_below_ihsan_floor(tmp_path: Path) -> None:
    """Economic reward must not mint for a mission below the Ihsan proof floor."""
    minter = _new_minter(tmp_path)

    receipt = compute_agent_reward(
        agent_id="researcher",
        mission_result=_mission(0.80),
        minter=minter,
        emission_gate=None,
        epoch_id="epoch-rejected",
    )

    assert receipt.success is False
    assert receipt.error == "ihsan_below_minting_floor"
    assert receipt.tx_entry is None


def test_seed_mint_accepts_at_ihsan_floor(tmp_path: Path) -> None:
    """The gate must preserve existing mint behavior at the constitutional floor."""
    minter = _new_minter(tmp_path)

    receipt = compute_agent_reward(
        agent_id="researcher",
        mission_result=_mission(0.95),
        minter=minter,
        emission_gate=None,
        epoch_id="epoch-accepted",
    )

    assert receipt.success is True
    assert receipt.tx_entry is not None
