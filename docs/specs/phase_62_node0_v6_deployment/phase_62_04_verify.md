# Phase 62 D4: Regression Verification

## Scope

Run 3 test suites to confirm zero regressions:

1. `bizra-constitution/` — all 332+ v6 tests
2. `tests/core/integration/test_constants.py` — 60 constants tests
3. `tests/core/` — full core regression suite

## Verification Matrix

| Suite | Location | Expected | Gate |
|-------|----------|----------|------|
| Constitution v6 | `bizra-constitution/` | 332 passed, ≤4 skipped | HARD |
| Constants v3.0.0 | `tests/core/integration/test_constants.py` | 60 passed | HARD |
| Bridge imports | `python -c "from core.bridges..."` | 11/11 components | HARD |
| Core regression | `tests/core/` | 2756+ passed, 1 pre-existing | SOFT |

## Pseudocode

```
PROCEDURE verify_regression:
    # Suite 1: Constitution v6 (HARD gate)
    result_1 := RUN pytest bizra-constitution/ -q --tb=short
    ASSERT result_1.passed >= 328
    ASSERT result_1.failed == 0
    IF result_1.failed > 0:
        ABORT "Constitution tests broken — do NOT proceed"

    # Suite 2: Constants (HARD gate)
    result_2 := RUN pytest tests/core/integration/test_constants.py -q
    ASSERT result_2.passed == 60
    ASSERT result_2.failed == 0
    IF result_2.failed > 0:
        ABORT "Constants regression — SOT integrity broken"

    # Suite 3: Bridge verification (HARD gate)
    report := IMPORT availability_report FROM core.bridges.constitutional_engine
    ASSERT report["genesis_engine_available"] == True
    ASSERT report["node0_production_available"] == True
    component_count := sum(1 for v in report["components"].values() if v)
    ASSERT component_count == 11

    # Suite 4: Core regression (SOFT gate)
    result_4 := RUN pytest tests/core/ -q --tb=short -x --timeout=60
    known_failures := ["test_signature_verification_returns_bool"]
    actual_failures := result_4.failures - known_failures
    IF len(actual_failures) > 0:
        WARN "New core regressions detected"
        REPORT actual_failures
    ELSE:
        PASS "Zero new regressions"

    RETURN {
        "constitution": result_1.summary,
        "constants": result_2.summary,
        "bridge": f"{component_count}/11 components",
        "core": result_4.summary,
    }
```

## Parallel Execution Strategy

Suites 1-3 are independent — run in parallel:

```
┌─ Stream A: pytest bizra-constitution/ ────────────── ~5s
├─ Stream B: pytest test_constants.py ──────────────── ~8s
├─ Stream C: python -c "from core.bridges..." ──────── ~2s
└─ Stream D: pytest tests/core/ ────────────────────── ~6min (background)
```

Wait for A+B+C (HARD gates). D runs in background as SOFT gate.

## TDD Anchors

```python
def test_constitution_suite_passes():
    """All 332+ constitution tests pass."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "bizra-constitution/", "-q"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "passed" in result.stdout
    assert "failed" not in result.stdout or "0 failed" in result.stdout

def test_bridge_all_components_available():
    """All 11 bridge components load."""
    from core.bridges.constitutional_engine import availability_report
    r = availability_report()
    for name, available in r["components"].items():
        assert available, f"Component {name} not available"
```

## Acceptance

- [ ] 332+ constitution tests pass
- [ ] 60 constants tests pass
- [ ] 11/11 bridge components available
- [ ] Zero NEW core regressions (pre-existing excluded)
