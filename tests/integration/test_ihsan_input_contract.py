from __future__ import annotations

import asyncio

from core.integration.constants import SNR_THRESHOLD, UNIFIED_IHSAN_THRESHOLD
from core.prompt.seed_chain import small_seed
from core.sovereign.ihsan_scorer import score_ihsan_composite, score_snr_composite
from core.sovereign.mission_nervous_system import SovereignNervousSystem


class _EchoInference:
    async def infer(self, prompt: str, **kwargs: object) -> str:
        del kwargs
        return f"echo::{prompt}"


def test_governed_prompt_scores_are_bounded() -> None:
    prompt = small_seed(
        "Under BIZRA constitutional enforcement, reject execution when authority is unavailable.",
        agent="P7_DEMA",
    ).to_prompt()
    output = (
        "Reject execution, fail closed, emit a blocked receipt, "
        "and preserve canonical replay lineage."
    )

    ihsan = score_ihsan_composite(output, prompt)
    snr = score_snr_composite(output, prompt)

    assert 0.0 <= ihsan <= 1.0
    assert 0.0 <= snr <= 1.0


def test_nervous_system_override_contract_is_authoritative() -> None:
    ns = SovereignNervousSystem(inference=_EchoInference())

    receipt = asyncio.run(
        ns.run(
            "Authoritative trust-surface contract",
            ihsan_override=UNIFIED_IHSAN_THRESHOLD,
            snr_override=SNR_THRESHOLD,
        )
    )

    assert receipt.ihsan_score == UNIFIED_IHSAN_THRESHOLD
    assert receipt.snr_score == SNR_THRESHOLD
