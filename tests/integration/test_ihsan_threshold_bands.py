"""Three-band Ihsān threshold guard — halt vs warn vs pass.

Sprint: Node0 Closure — row 4 (replay) threshold reconciliation (2026-04-21).

Before this guard, both the mission_nervous_system publish site and the
organism CQRS publish site emitted ``IHSAN_GATE_BREACHED`` whenever
``ihsan < UNIFIED_IHSAN_THRESHOLD (0.95)``. SUB-5 (``IhsanGateBreachHandler``)
unconditionally halts on that event, so any mission scoring 0.85 ≤ score
< 0.95 silently halted — even though SUB-5's own log message claimed the
floor was 0.85. The halt trigger (0.95) and the halt handler log text
(0.85) disagreed on which number was the operational floor.

The fix introduces a two-band Ihsān policy:

    ihsan <  0.85             → IHSAN_GATE_BREACHED  (hard halt)
    0.85 ≤ ihsan < 0.95       → IHSAN_WARNING        (warn, no halt)
    ihsan >= 0.95             → neither              (production ideal met)

This guard verifies all three bands, fail-closed on any regression.

See ``core.bus.subscribers.MISSION_IHSAN_HALT_FLOOR`` docstring for the
full rationale.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import pytest


RAW_PROMPT = (
    "Under BIZRA constitutional enforcement, what must happen when canonical "
    "mode is enabled but runtime-owned organism mission authority is unavailable?"
)
OUTPUT_TEXT = (
    "# Decision\n"
    "Verify Spine section 4, verify the canonical gate, and ensure canonical "
    "mode uses runtime-owned organism mission authority.\n\n"
    "# Safeguard\n"
    "The constitutional trade-off is simple: verify the refusal, ensure "
    "the safeguard chain stays intact, keep the weaker route disabled, and "
    "therefore must not silently fall back or use any legacy route."
)


class _StubInference:
    async def infer(self, prompt: str, **_: Any) -> str:
        await asyncio.sleep(0)
        return OUTPUT_TEXT


def _published_types(bus) -> List[str]:
    """Return the ordered list of event_type values published on the bus."""
    return [e.event_type.value for e in bus._chain]


async def _build_ns_with_bus():
    """Build a minimal SovereignNervousSystem wired to a real EventBus.

    This is the path that exercises the publish logic in
    ``mission_nervous_system.py`` — not the organism CQRS publish.
    """
    from core.bus.subscribers import EventBus
    from core.sovereign.mission_nervous_system import SovereignNervousSystem

    bus = EventBus()
    ns = SovereignNervousSystem(inference=_StubInference(), event_bus=bus)
    return ns, bus


@pytest.mark.asyncio
async def test_band_1_hard_halt_publishes_gate_breach() -> None:
    """ihsan < 0.85 → IHSAN_GATE_BREACHED published, IHSAN_WARNING NOT published."""
    ns, bus = await _build_ns_with_bus()

    await ns.run(
        RAW_PROMPT,
        raw_prompt=RAW_PROMPT,
        ihsan_override=0.50,  # deep below hard-halt floor
        snr_override=0.85,
    )

    published = _published_types(bus)
    assert "ihsan.gate.breached" in published, (
        f"Expected IHSAN_GATE_BREACHED on score 0.50. Published events: {published}"
    )
    assert "ihsan.warning" not in published, (
        f"IHSAN_WARNING must NOT fire in the halt band. Published events: {published}"
    )


@pytest.mark.asyncio
async def test_band_2_warn_publishes_ihsan_warning_no_halt() -> None:
    """0.85 ≤ ihsan < 0.95 → IHSAN_WARNING published, IHSAN_GATE_BREACHED NOT published."""
    ns, bus = await _build_ns_with_bus()

    await ns.run(
        RAW_PROMPT,
        raw_prompt=RAW_PROMPT,
        ihsan_override=0.87,  # in warn band
        snr_override=0.85,
    )

    published = _published_types(bus)
    assert "ihsan.warning" in published, (
        f"Expected IHSAN_WARNING on score 0.87 (warn band). Published: {published}"
    )
    assert "ihsan.gate.breached" not in published, (
        f"IHSAN_GATE_BREACHED must NOT fire when score >= 0.85. "
        f"This was the pre-fix bug (halt on sub-production-ideal). "
        f"Published: {published}"
    )


@pytest.mark.asyncio
async def test_band_3_production_ideal_publishes_neither_event() -> None:
    """ihsan >= 0.95 → neither IHSAN_GATE_BREACHED nor IHSAN_WARNING published."""
    ns, bus = await _build_ns_with_bus()

    await ns.run(
        RAW_PROMPT,
        raw_prompt=RAW_PROMPT,
        ihsan_override=0.98,  # above production ideal
        snr_override=0.95,
    )

    published = _published_types(bus)
    assert "ihsan.gate.breached" not in published, (
        f"IHSAN_GATE_BREACHED must NOT fire at production ideal. Published: {published}"
    )
    assert "ihsan.warning" not in published, (
        f"IHSAN_WARNING must NOT fire at production ideal. Published: {published}"
    )


def test_hard_halt_floor_constant_is_authoritative() -> None:
    """The single source of truth for the hard-halt floor is the module constant."""
    from core.bus.subscribers import (
        MISSION_IHSAN_HALT_FLOOR,
        IhsanGateBreachHandler,
    )

    assert MISSION_IHSAN_HALT_FLOOR == 0.85, (
        f"MISSION_IHSAN_HALT_FLOOR drifted to {MISSION_IHSAN_HALT_FLOOR}. "
        f"The canonical hard-halt floor is 0.85 — changing this is a "
        f"constitutional decision that requires explicit operator authorization."
    )
    assert IhsanGateBreachHandler.MISSION_FLOOR == MISSION_IHSAN_HALT_FLOOR, (
        "SUB-5 class constant diverged from module constant. They MUST match "
        "so halt trigger and halt handler log agree."
    )


def test_warning_handler_uses_correct_bounds() -> None:
    """IhsanWarningHandler declares the hard-halt floor and production-ideal."""
    from core.bus.subscribers import (
        MISSION_IHSAN_HALT_FLOOR,
        IhsanWarningHandler,
    )

    h = IhsanWarningHandler(audit_log=None, production_ideal=0.95)
    assert h.HARD_HALT_FLOOR == MISSION_IHSAN_HALT_FLOOR
    assert h.production_ideal == 0.95
