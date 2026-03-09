"""
Verify that safety-critical code paths log failures instead of
silently swallowing exceptions. Phase 66.02 enforcement tests.

Standing on Giants:
- Shannon (1948): A lost signal cannot be recovered
- Al-Ghazali (1095): Accountability is prerequisite to excellence
"""

import logging
import pathlib
import re

import pytest

# ── Test 1: Mission emit logs on failure ──────────────────────────


async def test_mission_emit_logs_warning_on_bus_failure(caplog):
    """
    GIVEN: MissionOrchestrator with a broken event bus
    WHEN:  _emit() is called
    THEN:  Warning is logged (not silently swallowed)
    AND:   No exception propagates to caller
    """
    from unittest.mock import AsyncMock

    from core.sovereign.mission import MissionOrchestrator

    orch = MissionOrchestrator.__new__(MissionOrchestrator)
    orch._event_bus = AsyncMock()
    orch._event_bus.emit.side_effect = RuntimeError("bus down")

    with caplog.at_level(logging.WARNING):
        # Should not raise
        await orch._emit("test.topic", {"data": 1})

    assert "Event emit failed" in caplog.text or "bus down" in caplog.text


# ── Test 2: No bare except-pass in mission.py ─────────────────────


def test_no_bare_except_pass_in_mission():
    """
    GIVEN: core/sovereign/mission.py source code
    WHEN:  We scan for 'except' followed by 'pass' with no logging
    THEN:  Zero matches found
    """
    source = pathlib.Path("core/sovereign/mission.py").read_text()

    # Pattern: except ... :\n<whitespace>pass
    matches = re.findall(r"except[^:]*:\s*\n\s+pass\b", source)
    assert len(matches) == 0, f"Found {len(matches)} silent except-pass blocks"


# ── Test 3: No bare except-pass in rollback.py ────────────────────


def test_no_bare_except_pass_in_rollback():
    """
    GIVEN: core/rollout/rollback.py source code
    WHEN:  We scan for 'except' followed by 'pass' with no logging
    THEN:  Zero matches found
    """
    rollback_path = pathlib.Path("core/rollout/rollback.py")
    if not rollback_path.exists():
        pytest.skip("rollback.py not present")

    source = rollback_path.read_text()
    matches = re.findall(r"except[^:]*:\s*\n\s+pass\b", source)
    assert len(matches) == 0, f"Found {len(matches)} silent except-pass blocks"
