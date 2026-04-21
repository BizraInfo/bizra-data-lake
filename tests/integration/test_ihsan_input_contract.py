"""Ihsan scorer input-contract guard — decoupling raw prompt from mission-spine.

Sprint: Node0 Closure — row 4 (replay) unblocking (2026-04-21).

Before this guard, `SovereignNervousSystem.run(mission_text)` scored the
Ihsan composite using the wrapped liturgical mission-spine text as the
`input_text` argument to `_score_ihsan`. The wrapper
(``## Niyyah / ## Bayyinah / ## Hadd / ## Qasd`` built by
``core.prompt.seed_chain.small_seed``) is ~835 chars with boundary/evidence
bodies that the scorer penalizes for low contextual_relevance, dropping
the composite from 0.87 (raw prompt) to 0.79 (wrapped) — below the 0.85
floor. The canonical spearpoint replay test halted here.

The fix: add `raw_prompt: Optional[str]` to `run()`. When provided, the
scorer uses `raw_prompt` as its `input_text`. The wrapped `mission_text`
stays canonical runtime evidence (receipts, reflex, chain) — this change
is scoring-only, not structural.

This guard proves:

1. When ``raw_prompt`` is passed, ``_score_ihsan`` receives ``raw_prompt``
   as its second argument (NOT ``mission_text``).
2. When ``raw_prompt`` is omitted, legacy behavior is preserved
   (``_score_ihsan`` still receives ``mission_text``).
3. The effective composite on a known-good output × raw prompt clears
   the 0.85 floor, while the same output × wrapped spine text does not —
   proving the drift was structural and the fix is the right decoupling.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Tuple
from unittest.mock import patch

import pytest


RAW_PROMPT = (
    "Under BIZRA constitutional enforcement, what must happen when canonical "
    "mode is enabled but runtime-owned organism mission authority is unavailable?"
)

WRAPPED_MISSION_TEXT = (
    "## Niyyah (Intent)\n"
    f"{RAW_PROMPT}\n\n"
    "## Bayyinah (Evidence)\n"
    "- [VERIFIED] Prior receipt: "
    "96ebccdf5a91eb86c6b4347f3d2e089ebfca45e69d6871c47ebc4673ece3e6ae\n\n"
    "## Hadd (Boundary)\n- Ihsan floor: 0.95\n- Canonical mode required\n\n"
    "## Qasd (Scope)\n- Runtime-owned authority is sole execution path"
)

# EXACT fixture text from scripts/ops/canonical_spearpoint_v1.py::DeterministicGateway
# (do not modify — any delta shifts the reference score this guard anchors to).
GOOD_OUTPUT = (
    "# Decision\n"
    "Verify Spine section 4, verify the canonical gate, and ensure canonical mode uses runtime-owned organism mission authority.\n\n"
    "# Outcome\n"
    "If canonical mode is enabled and that authority is unavailable, the system must reject execution, fail closed, emit a blocked receipt, record fate reason codes, and use Ihsan >= 095 as the policy floor.\n\n"
    "# Safeguard\n"
    "However, to answer what must happen on this failure path, the constitutional trade-off is still simple: the system should verify the refusal, ensure the safeguard chain stays intact, keep the weaker route disabled, and therefore must not silently fall back or use any legacy route."
)


class _StubInference:
    """Deterministic inference backend for the guard — always returns GOOD_OUTPUT."""

    async def infer(self, prompt: str, **_: Any) -> str:
        await asyncio.sleep(0)
        return GOOD_OUTPUT


async def _build_ns(**overrides: Any):
    """Construct a bare SovereignNervousSystem (no reflex, no bus, no minter).

    We only need `.run()` to reach the scorer and return a receipt. All
    downstream wiring is optional and omitted for test isolation.
    """
    from core.sovereign.mission_nervous_system import SovereignNervousSystem

    return SovereignNervousSystem(
        inference=_StubInference(),
        **overrides,
    )


@pytest.mark.asyncio
async def test_raw_prompt_overrides_mission_text_for_ihsan_input() -> None:
    """When raw_prompt is provided, _score_ihsan receives it, not mission_text."""
    from core.sovereign import mission_nervous_system as mns

    ns = await _build_ns()
    captured: List[Tuple[str, str]] = []

    original = mns._score_ihsan

    def spy(output: str, input_text: str) -> float:
        captured.append((output, input_text))
        return original(output, input_text)

    with patch.object(mns, "_score_ihsan", side_effect=spy):
        receipt = await ns.run(WRAPPED_MISSION_TEXT, raw_prompt=RAW_PROMPT)

    assert captured, "_score_ihsan was not called"
    _, scored_input = captured[-1]
    assert scored_input == RAW_PROMPT, (
        f"Scorer received the wrong input_text.\n"
        f"  expected: raw_prompt ({len(RAW_PROMPT)} chars)\n"
        f"  got: {len(scored_input)} chars, starts with {scored_input[:80]!r}\n"
        f"This indicates the raw_prompt decoupling has regressed."
    )
    assert receipt.ihsan_score >= 0.85, (
        f"Composite Ihsan {receipt.ihsan_score:.4f} below 0.85 floor when "
        f"scored against raw_prompt — raw-prompt decoupling may not be "
        f"propagating through _score_ihsan or scorer behavior has drifted."
    )


@pytest.mark.asyncio
async def test_legacy_callers_without_raw_prompt_use_mission_text() -> None:
    """When raw_prompt is omitted, _score_ihsan receives mission_text (legacy)."""
    from core.sovereign import mission_nervous_system as mns

    ns = await _build_ns()
    captured: List[Tuple[str, str]] = []

    original = mns._score_ihsan

    def spy(output: str, input_text: str) -> float:
        captured.append((output, input_text))
        return original(output, input_text)

    with patch.object(mns, "_score_ihsan", side_effect=spy):
        await ns.run(WRAPPED_MISSION_TEXT)  # no raw_prompt

    assert captured, "_score_ihsan was not called"
    _, scored_input = captured[-1]
    assert scored_input == WRAPPED_MISSION_TEXT, (
        f"Legacy caller (no raw_prompt) expected mission_text as scorer "
        f"input, got {len(scored_input)}-char different string. The "
        f"raw_prompt kwarg must default to None and preserve legacy behavior."
    )


def test_scorer_confirms_drift_is_structural_not_textual() -> None:
    """Sanity: GOOD_OUTPUT scores >=0.85 vs raw prompt but <0.85 vs wrapped spine.

    This is the empirical evidence that the fix target is the input contract,
    not the fixture text. If this assertion ever flips, the Ihsan scorer's
    contextual_relevance calibration has changed and this guard's premise
    needs re-verification.
    """
    from core.sovereign.ihsan_scorer import score_ihsan_composite

    raw_score = score_ihsan_composite(GOOD_OUTPUT, RAW_PROMPT)
    wrapped_score = score_ihsan_composite(GOOD_OUTPUT, WRAPPED_MISSION_TEXT)

    assert raw_score >= 0.85, (
        f"Raw-prompt composite {raw_score:.4f} below floor — "
        f"scorer calibration drifted; re-anchor this guard."
    )
    assert wrapped_score < 0.85, (
        f"Wrapped composite {wrapped_score:.4f} NOT below floor — "
        f"the structural drift premise no longer holds; this guard is obsolete."
    )
    assert (raw_score - wrapped_score) >= 0.04, (
        f"Raw↔wrapped delta ({raw_score - wrapped_score:.4f}) is too small "
        f"to meaningfully demonstrate the structural-input drift this guard "
        f"exists to protect against."
    )
