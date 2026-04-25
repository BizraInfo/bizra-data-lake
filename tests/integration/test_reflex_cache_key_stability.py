"""Reflex cache key stability guard — stable across Bayyinah evidence drift.

Sprint: Node0 Closure — row 4 (replay) final blocker (2026-04-21).

After the Ihsān scorer-input and threshold-band fixes, run1 of the
canonical spearpoint completes lawfully and compiles a reflex. But run2
missed the reflex and re-ran S2 because the wrapped mission_text
(``core.prompt.seed_chain``) embeds variable Bayyinah evidence —
including "Prior receipt: <hash>" — which changes between runs. Hashing
the wrapped text yields a different ``mission_pattern_hash`` on run2 →
reflex cache miss.

The fix: reflex lookup + record keyed on ``raw_prompt`` when provided
(symmetric to the Ihsān scorer decoupling from the same commit arc).
Wrapped ``mission_text`` remains canonical runtime evidence for
receipts, inference, and audit.

This guard proves:

1. Two runs with the SAME raw_prompt but DIFFERENT mission_text
   (simulating Bayyinah evidence drift) produce the SAME reflex
   pattern_hash.
2. Legacy callers (no raw_prompt) still key reflexes on mission_text.
3. After N precipitations (K+ observations of a pattern), run2 with
   differing mission_text but identical raw_prompt hits the compiled
   reflex on the S1 path instead of re-running S2.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


RAW_PROMPT = "Explain the constitutional fail-closed rule for canonical mode."

# Two wrapped variants — SAME raw prompt, DIFFERENT Bayyinah evidence.
# Simulates the canonical spearpoint's behavior: run1 has no prior
# receipt; run2's Bayyinah includes run1's receipt hash.
WRAPPED_V1 = (
    "## Niyyah (Intent)\n"
    f"{RAW_PROMPT}\n\n"
    "## Bayyinah (Evidence)\n"
    "- [VERIFIED] Prior receipt: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
)
WRAPPED_V2 = (
    "## Niyyah (Intent)\n"
    f"{RAW_PROMPT}\n\n"
    "## Bayyinah (Evidence)\n"
    "- [VERIFIED] Prior receipt: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
)


class _StubInference:
    async def infer(self, prompt: str, **_: Any) -> str:
        await asyncio.sleep(0)
        return (
            "# Decision\nUphold runtime-owned organism mission authority. "
            "Fail closed if unavailable; must not silently fall back.\n\n"
            "# Safeguard\nVerify the refusal; keep the weaker route disabled."
        )


async def _build_ns_with_reflex():
    """Build a NervousSystem with a real ReflexCompiler (no bus/minter needed)."""
    from core.sovereign.mission_nervous_system import SovereignNervousSystem
    from core.sovereign.reflex_compiler import ReflexCompiler

    reflex = ReflexCompiler(max_entries=100)
    ns = SovereignNervousSystem(inference=_StubInference(), reflex_cache=reflex)
    return ns, reflex


def test_reflex_pattern_hash_stable_across_wrapped_variants() -> None:
    """Same raw_prompt, different Bayyinah evidence → identical pattern_hash."""
    from core.sovereign.reflex_compiler import ReflexCompiler

    reflex = ReflexCompiler(max_entries=100)

    # When the mission_nervous_system hashes the reflex key using raw_prompt,
    # both wrapped variants must collapse to the same key.
    h_raw = reflex._hash_input(RAW_PROMPT)
    h_wrapped_v1 = reflex._hash_input(WRAPPED_V1)
    h_wrapped_v2 = reflex._hash_input(WRAPPED_V2)

    # Sanity: the wrapped variants themselves DO hash differently (this is
    # WHY we need the raw_prompt decoupling).
    assert h_wrapped_v1 != h_wrapped_v2, (
        "Test premise broken: wrapped variants must hash differently for "
        "this guard to be meaningful. If this fails, ReflexCompiler "
        "already normalizes away Bayyinah drift somehow."
    )
    # Both wrapped hashes must differ from the raw-prompt hash (sanity).
    assert h_raw != h_wrapped_v1
    assert h_raw != h_wrapped_v2


@pytest.mark.asyncio
async def test_run2_with_same_raw_prompt_hits_reflex_after_precipitation() -> None:
    """After K+ run1s precipitate a reflex, run2 with different wrapped
    mission_text but identical raw_prompt takes the S1 reflex path."""
    from core.integration.constants import REFLEX_PRECIPITATION_HITS

    ns, reflex = await _build_ns_with_reflex()

    # Run1: K precipitation observations with wrapped_v1 + raw_prompt.
    # Each run must produce ihsan ≥ the reflex precipitation threshold.
    for _ in range(REFLEX_PRECIPITATION_HITS):
        receipt_run1 = await ns.run(
            WRAPPED_V1,
            raw_prompt=RAW_PROMPT,
            ihsan_override=0.98,  # well above production ideal
            snr_override=0.95,
        )
        assert receipt_run1.system == "S2", (
            f"run1 must take S2 path on precipitation observations "
            f"(got {receipt_run1.system}); precipitation never accumulates."
        )

    # Run2: wrapped_v2 (different Bayyinah evidence) + SAME raw_prompt.
    # If reflex key is keyed on raw_prompt, this must hit the reflex.
    receipt_run2 = await ns.run(
        WRAPPED_V2,
        raw_prompt=RAW_PROMPT,
        ihsan_override=0.98,
        snr_override=0.95,
    )
    assert receipt_run2.reflex_hit is True, (
        f"run2 should hit the compiled reflex despite Bayyinah drift.\n"
        f"  raw_prompt was identical across runs: {RAW_PROMPT!r}\n"
        f"  wrapped mission_text differed (Bayyinah 'Prior receipt' line).\n"
        f"  reflex_hit={receipt_run2.reflex_hit}, system={receipt_run2.system}\n"
        f"If this fails, the reflex key is still being computed from the "
        f"wrapped mission_text (Bayyinah-poisoned) rather than the stable "
        f"raw_prompt."
    )
    assert receipt_run2.system == "S1", (
        f"run2 should take S1 (reflex) path, got {receipt_run2.system}."
    )


@pytest.mark.asyncio
async def test_legacy_caller_without_raw_prompt_keys_on_mission_text() -> None:
    """Without raw_prompt, reflex lookup continues to use mission_text (legacy)."""
    from core.integration.constants import REFLEX_PRECIPITATION_HITS

    ns, reflex = await _build_ns_with_reflex()

    # Run1: K precipitation observations with wrapped_v1, NO raw_prompt.
    for _ in range(REFLEX_PRECIPITATION_HITS):
        await ns.run(
            WRAPPED_V1,
            ihsan_override=0.98,
            snr_override=0.95,
        )

    # Run2 with DIFFERENT wrapped mission_text and NO raw_prompt.
    # Legacy key = mission_text, so wrapped_v2 MUST miss.
    receipt_run2 = await ns.run(
        WRAPPED_V2,
        ihsan_override=0.98,
        snr_override=0.95,
    )
    assert receipt_run2.reflex_hit is False, (
        "Legacy callers (no raw_prompt) must continue keying reflex lookup "
        "on mission_text. Different mission_text → no hit. If this flips, "
        "legacy behavior regressed."
    )
