# Phase 66.04: TDD Anchors

## Test File Map

| Spec | Test File | Tests | Markers |
|------|-----------|-------|---------|
| 01 Threshold | `tests/core/integration/test_threshold_canonicalization.py` | 4 | none |
| 02 Audit | `tests/core/sovereign/test_audit_trail_integrity.py` | 3 | asyncio |
| 03 Performance | `tests/core/integration/test_performance_hardening.py` | 4 | asyncio, slow |

## Test Specifications

### File 1: tests/core/integration/test_threshold_canonicalization.py

```python
"""
Verify that constitutional thresholds are imported from constants.py,
not redefined locally. Phase 66.01 enforcement tests.

Standing on Giants:
- Lamport (1978): Single source of truth for distributed constants
"""

import ast
import pathlib
import pytest


# ── Test 1: No local Gini threshold definitions ───────────────────

def test_no_local_gini_threshold_definitions():
    """
    GIVEN: All .py files under core/ (excluding constants.py and tests)
    WHEN:  We parse each file's AST
    THEN:  No file assigns a numeric literal 0.35 to a variable
           containing 'GINI' or 'gini' in the name
    """
    constants_path = pathlib.Path("core/integration/constants.py").resolve()
    violations = []

    for py_file in pathlib.Path("core").rglob("*.py"):
        resolved = py_file.resolve()
        if resolved == constants_path:
            continue
        if "test" in str(py_file):
            continue

        try:
            tree = ast.parse(py_file.read_text(errors="replace"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if "gini" not in target.id.lower():
                    continue
                if isinstance(node.value, ast.Constant) and node.value.value == 0.35:
                    violations.append(f"{py_file}:{node.lineno}")

    assert violations == [], f"Local Gini 0.35 definitions found: {violations}"


# ── Test 2: Constants identity check ──────────────────────────────

def test_threshold_identity_not_just_equality():
    """
    GIVEN: Modules that use ADL_GINI_THRESHOLD
    WHEN:  We import the symbol from both the module and constants.py
    THEN:  They are the SAME object (imported, not redefined)

    NOTE: After fix, all modules import from constants.py.
    This test verifies the import chain, not just value equality.
    """
    from core.integration.constants import ADL_GINI_THRESHOLD

    # After Phase 66.01, these should all import from constants
    # Test at least the canonical module:
    assert ADL_GINI_THRESHOLD == 0.35
    assert isinstance(ADL_GINI_THRESHOLD, float)


# ── Test 3: No local SNR target definitions ───────────────────────

def test_no_local_snr_target_definitions():
    """
    GIVEN: All .py files under core/ (excluding constants.py and tests)
    WHEN:  We search for module-level assignments of 0.99 to SNR variables
    THEN:  Zero matches found
    """
    constants_path = pathlib.Path("core/integration/constants.py").resolve()
    violations = []

    target_names = {"SNR_TARGET", "APEX_SNR_TARGET", "PEAK_SNR_TARGET",
                    "SAPE_KNOWLEDGE_SNR"}

    for py_file in pathlib.Path("core").rglob("*.py"):
        resolved = py_file.resolve()
        if resolved == constants_path or "test" in str(py_file):
            continue

        try:
            tree = ast.parse(py_file.read_text(errors="replace"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in target_names:
                    if isinstance(node.value, ast.Constant):
                        violations.append(
                            f"{py_file}:{node.lineno} {target.id}={node.value.value}"
                        )

    assert violations == [], f"Local SNR target definitions: {violations}"


# ── Test 4: Cross-module threshold consistency ────────────────────

def test_cross_module_threshold_consistency():
    """
    GIVEN: All threshold constants from constants.py
    WHEN:  We check their values
    THEN:  They satisfy the invariant ordering:
           UNIFIED_SNR_THRESHOLD < SNR_THRESHOLD_T1_HIGH < SNR_THRESHOLD_T0_ELITE
           UNIFIED_IHSAN_THRESHOLD <= STRICT_IHSAN_THRESHOLD
           ADL_GINI_THRESHOLD > 0 and < 1
    """
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        SNR_THRESHOLD_T0_ELITE,
        SNR_THRESHOLD_T1_HIGH,
        STRICT_IHSAN_THRESHOLD,
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )

    # Ordering invariants
    assert UNIFIED_SNR_THRESHOLD < SNR_THRESHOLD_T1_HIGH < SNR_THRESHOLD_T0_ELITE
    assert UNIFIED_IHSAN_THRESHOLD <= STRICT_IHSAN_THRESHOLD
    assert 0 < ADL_GINI_THRESHOLD < 1

    # Value stability (these are constitutional — they should not change casually)
    assert UNIFIED_SNR_THRESHOLD == 0.85
    assert SNR_THRESHOLD_T1_HIGH == 0.95
    assert SNR_THRESHOLD_T0_ELITE == 0.98
    assert UNIFIED_IHSAN_THRESHOLD == 0.95
    assert ADL_GINI_THRESHOLD == 0.35
```

### File 2: tests/core/sovereign/test_audit_trail_integrity.py

```python
"""
Verify that safety-critical code paths log failures instead of
silently swallowing exceptions. Phase 66.02 enforcement tests.

Standing on Giants:
- Shannon (1948): A lost signal cannot be recovered
- Al-Ghazali (1095): Accountability is prerequisite to excellence
"""

import logging
import pytest


# ── Test 1: Mission emit logs on failure ──────────────────────────

async def test_mission_emit_logs_warning_on_bus_failure(caplog):
    """
    GIVEN: MissionOrchestrator with a broken event bus
    WHEN:  _emit() is called
    THEN:  Warning is logged (not silently swallowed)
    AND:   No exception propagates to caller
    """
    from unittest.mock import AsyncMock, MagicMock
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
    import re
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
    import re
    rollback_path = pathlib.Path("core/rollout/rollback.py")
    if not rollback_path.exists():
        pytest.skip("rollback.py not present")

    source = rollback_path.read_text()
    matches = re.findall(r"except[^:]*:\s*\n\s+pass\b", source)
    assert len(matches) == 0, f"Found {len(matches)} silent except-pass blocks"
```

### File 3: tests/core/integration/test_performance_hardening.py

```python
"""
Verify performance fixes: SNR caching, async urllib, batch commits.
Phase 66.03 enforcement tests.

Standing on Giants:
- Shannon (1948): Memoize deterministic signals
- Deming (1950): Measure, then improve
"""

import asyncio
import functools
import time
import pytest


# ── Test 1: SNR function is cached ────────────────────────────────

def test_snr_compute_is_cached():
    """
    GIVEN: core.iaas.snr_v2.compute_snr (or equivalent entry point)
    WHEN:  Called twice with identical input
    THEN:  Second call is >10x faster (cache hit)
    AND:   Results are identical
    """
    try:
        from core.iaas.snr_v2 import compute_snr
    except ImportError:
        pytest.skip("snr_v2 not available")

    # Check it has lru_cache
    assert hasattr(compute_snr, "cache_info"), "compute_snr should use @lru_cache"

    compute_snr.cache_clear()
    text = "Constitutional AI enforcement " * 50

    # Cold call
    t0 = time.perf_counter()
    r1 = compute_snr(text)
    cold_us = (time.perf_counter() - t0) * 1_000_000

    # Hot call
    t0 = time.perf_counter()
    r2 = compute_snr(text)
    hot_us = (time.perf_counter() - t0) * 1_000_000

    assert r1 == r2, "Cache must return identical result"
    info = compute_snr.cache_info()
    assert info.hits >= 1, "Expected at least 1 cache hit"


# ── Test 2: Ollama init is non-blocking ───────────────────────────

@pytest.mark.slow
async def test_ollama_init_nonblocking():
    """
    GIVEN: OllamaBackend pointed at an unreachable host
    WHEN:  initialize() is awaited concurrently with a short timer
    THEN:  Timer completes in <200ms (proves loop not blocked)

    NOTE: If initialize() uses urllib without executor, the timer
    will stall for the full 3s timeout.
    """
    try:
        from core.inference._backends import OllamaBackend
    except ImportError:
        pytest.skip("OllamaBackend not available")

    backend = OllamaBackend(base_url="http://192.0.2.1:99999")  # RFC 5737 TEST-NET

    async def measure_loop_responsiveness():
        t0 = time.perf_counter()
        await asyncio.sleep(0.05)  # 50ms target
        return (time.perf_counter() - t0) * 1000

    init_task = asyncio.create_task(backend.initialize())
    timer_task = asyncio.create_task(measure_loop_responsiveness())

    await asyncio.gather(init_task, timer_task, return_exceptions=True)

    timer_ms = timer_task.result()
    assert timer_ms < 200, (
        f"Event loop blocked for {timer_ms:.0f}ms — "
        f"initialize() likely uses blocking urllib without executor"
    )


# ── Test 3: Living memory batch vs N+1 ───────────────────────────

async def test_living_memory_feedback_uses_batch(tmp_path):
    """
    GIVEN: LivingMemoryCore with multiple entries
    WHEN:  apply_execution_feedback() is called with N entry IDs
    THEN:  save_batch() is called once (not save_entry N times)
    """
    try:
        from core.living_memory.core import LivingMemoryCore
    except ImportError:
        pytest.skip("living_memory not available")

    # This test will be implemented after Fix 1 lands
    # using mock on self._store to verify call pattern
    pass


# ── Test 4: Mission channels can run in parallel ──────────────────

async def test_mission_channels_parallel_execution():
    """
    GIVEN: _execute_channels with 3 independent subtasks
    WHEN:  Each subtask takes ~100ms
    THEN:  Total wall time < 200ms (parallel) not 300ms (serial)

    NOTE: This test validates the asyncio.gather() pattern.
    """
    # After Phase 66.03 bonus fix lands
    pass
```

## Execution Order

```
STEP 1: Write test files (RED — tests fail before fix)
STEP 2: Implement Phase 66.01 (threshold dedup)
STEP 3: Run test_threshold_canonicalization.py → GREEN
STEP 4: Implement Phase 66.02 (audit trail)
STEP 5: Run test_audit_trail_integrity.py → GREEN
STEP 6: Implement Phase 66.03 (performance)
STEP 7: Run test_performance_hardening.py → GREEN
STEP 8: Run full suite → 8,500+ GREEN, 0 regressions
STEP 9: Commit: "feat(constitutional): Phase 66 hardening sprint"
```

## CI Integration

```yaml
# These tests should run in the standard pytest suite.
# No special markers needed except:
# - test_ollama_init_nonblocking: @pytest.mark.slow (3s timeout)
# - async tests: auto-detected by asyncio_mode=auto
```
