from __future__ import annotations

import asyncio

from core.integration.constants import (
    IHSAN_BLOOM_ELIGIBILITY,
    IHSAN_GATE_MINIMUM,
    IHSAN_THRESHOLD_CI,
    UNIFIED_IHSAN_THRESHOLD,
)
from core.sovereign.mission_nervous_system import SovereignNervousSystem
from core.token.bloom import CommunityPool
from core.token.bloom import TokenMinter as BloomMinter
from core.token.bloom import WalletState


class _EchoInference:
    async def infer(self, prompt: str, **kwargs: object) -> str:
        del kwargs
        return f"band::{prompt}"


def _run_with_ihsan(ihsan: float):
    pool = CommunityPool()
    wallet = WalletState(node_id="threshold-node")
    minter = BloomMinter(community_pool=pool)
    ns = SovereignNervousSystem(
        inference=_EchoInference(),
        token_minter=minter,
        wallet=wallet,
        wallets=[wallet],
    )
    receipt = asyncio.run(ns.run("Threshold band mission", ihsan_override=ihsan))
    return receipt, wallet, pool


def test_ihsan_threshold_bands_are_monotonic() -> None:
    assert IHSAN_GATE_MINIMUM < IHSAN_BLOOM_ELIGIBILITY
    assert IHSAN_BLOOM_ELIGIBILITY <= IHSAN_THRESHOLD_CI
    assert IHSAN_THRESHOLD_CI < UNIFIED_IHSAN_THRESHOLD


def test_gate_minimum_is_not_reward_eligible() -> None:
    receipt, wallet, pool = _run_with_ihsan(IHSAN_GATE_MINIMUM)

    assert receipt.rewarded is False
    assert wallet.seed_balance == 0.0
    assert pool.current_balance == 0.0


def test_unified_threshold_is_reward_eligible() -> None:
    receipt, wallet, pool = _run_with_ihsan(UNIFIED_IHSAN_THRESHOLD)

    assert receipt.rewarded is True
    assert wallet.seed_balance > 0.0
    assert pool.current_balance > 0.0
