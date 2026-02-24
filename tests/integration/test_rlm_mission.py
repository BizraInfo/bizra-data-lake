from __future__ import annotations

from pathlib import Path

import pytest

from core.inference.rlm_bridge import BizraRLMBridge
from core.token.emission_decay import LogisticEmissionGate
from core.token.mint import TokenMinter
from core.token.rl_rewards import composite_reward, compute_agent_reward


@pytest.mark.asyncio
async def test_rlm_reward_receipt_metadata_path(tmp_path: Path) -> None:
    async def model(_: str) -> str:
        return 'score = 0.92\nFINAL_ANSWER = "Top 5 VC outreach plan"'

    bridge = BizraRLMBridge(max_iterations=4, max_sub_calls=4)
    rlm = await bridge.execute_rlm(
        prompt="Need outbound plan for decentralized AI investors",
        task="Produce concise outreach strategy",
        agent_model=model,
    )

    assert rlm.success is True
    assert rlm.final_answer

    reward = composite_reward(
        {
            "snr": 0.9,
            "ihsan": 0.96,
            "tokens_used": 700,
            "quality": 0.9,
            "user_feedback": 0.85,
        }
    )
    assert 0.0 <= reward <= 1.0

    minter = TokenMinter.create(
        db_path=tmp_path / "ledger.db",
        log_path=tmp_path / "ledger.jsonl",
    )
    gate = LogisticEmissionGate()
    receipt = compute_agent_reward(
        agent_id="researcher",
        mission_result={
            "snr": 0.9,
            "ihsan": 0.96,
            "tokens_used": 700,
            "quality": 0.9,
            "user_feedback": 0.85,
            "seed_base": 150.0,
        },
        minter=minter,
        emission_gate=gate,
        epoch_id="epoch-integration",
    )

    assert receipt.success is True
    assert receipt.tx_entry is not None

    metadata = {
        "rlm_iterations": rlm.iterations,
        "rlm_sub_calls": rlm.sub_calls,
        "reward": reward,
        "token_receipt_hash": receipt.receipt_hash,
    }
    assert metadata["rlm_iterations"] >= 1
    assert metadata["token_receipt_hash"]
