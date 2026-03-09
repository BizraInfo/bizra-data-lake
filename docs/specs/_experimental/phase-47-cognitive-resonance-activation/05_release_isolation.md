# 05: Release Isolation and Semantic Integrity

## Standing on Giants
Lamport (distributed reliability, 1978) · Humble/Farley (Continuous Delivery, 2010)

## Overview

Phase 47.1 ships on an isolated release branch containing only approved Phase 46 files plus rollout instrumentation. This prevents contamination from the dirty main worktree (50+ modified files from other phases) while maintaining semantic integrity.

## Release Branch Strategy

### Pseudocode: Branch Creation

```
# Step 1: Create clean release branch from origin/main
git fetch origin
git checkout -b release/phase-47.1-canary origin/main

# Step 2: Cherry-pick Phase 46 commits (in order)
git cherry-pick 366774c   # Phase 46 core
git cherry-pick 477e9d9   # Phase 46.1 MCP integration

# Step 3: Verify clean state
git diff origin/main --stat  # Should show ONLY Phase 46 files
```

### Expected File Delta (Phase 46 only)

```
core/__init__.py                                    (modified)
core/integration/constants.py                       (modified)
core/living_memory/proactive.py                     (modified)
core/memory/types.py                                (new or modified)
core/prediction/__init__.py                         (new)
core/prediction/hmm_engine.py                       (new)
core/reasoning/__init__.py                          (modified)
core/reasoning/got_bridge.py                        (new)
core/resonance.py                                   (new)
core/search/__init__.py                             (new)
core/search/vector_search.py                        (new)
core/sovereign/apex_engine.py                       (modified)
tests/core/mcp/test_sovereign_phase46_tools.py      (new)
tests/core/prediction/test_hmm_engine.py            (new)
tests/core/reasoning/test_got_bridge.py             (new)
tests/core/search/test_vector_search.py             (new)
tests/core/sovereign/test_apex_got_bridge_integration.py (new)
tests/core/test_resonance.py                        (new)
tools/mcp/sovereign_mcp_server.py                   (modified)
```

## Release Manifest

### Pseudocode: Manifest Generator

```
MODULE scripts/generate_release_manifest.py

"""Generate phase46_release_manifest.txt with file paths and SHA-256 checksums."""

IMPORT hashlib
FROM pathlib IMPORT Path


FUNCTION generate_manifest(release_files: List[str], output: str):
    """Create manifest with path:sha256 pairs."""
    manifest_lines = []

    FOR file_path IN sorted(release_files):
        p = Path(file_path)
        IF NOT p.exists():
            manifest_lines.append(f"{file_path}:MISSING")
            CONTINUE
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest_lines.append(f"{file_path}:{sha}")

    Path(output).write_text("\n".join(manifest_lines) + "\n")
    PRINT(f"Manifest written: {output} ({len(manifest_lines)} files)")


# Usage:
# python scripts/generate_release_manifest.py > artifacts/phase46_release_manifest.txt
```

### Sample Manifest Output

```
core/__init__.py:a1b2c3d4e5f6...
core/integration/constants.py:f6e5d4c3b2a1...
core/prediction/__init__.py:1234567890ab...
...
```

## Semantic Integrity Checks

Run before CI pipeline, immediately after branch creation.

### Check 1: Import Smoke

```
FUNCTION check_imports():
    """Verify all Phase 46 modules import cleanly."""
    modules = [
        "core.search",
        "core.search.vector_search",
        "core.prediction",
        "core.prediction.hmm_engine",
        "core.reasoning.got_bridge",
        "core.resonance",
    ]
    FOR module IN modules:
        TRY:
            importlib.import_module(module)
            PRINT(f"OK: {module}")
        EXCEPT ImportError AS e:
            PRINT(f"FAIL: {module} — {e}")
            RAISE SystemExit(1)
```

### Check 2: API Surface Snapshot

```
FUNCTION check_api_surface():
    """Verify exported symbols haven't changed unexpectedly."""
    expected_exports = {
        "core.search": ["VectorSearchEngine"],
        "core.prediction": ["HMMEngine", "HMMState", "PredictionResult"],
        "core.reasoning": ["GoTBridge", "GoTBridgeResult"],  # appended to existing
    }
    FOR module, symbols IN expected_exports.items():
        mod = importlib.import_module(module)
        FOR symbol IN symbols:
            IF NOT hasattr(mod, symbol):
                PRINT(f"FAIL: {module}.{symbol} missing from API surface")
                RAISE SystemExit(1)
        PRINT(f"OK: {module} exports {symbols}")
```

### Check 3: Phase 46 Test Suite

```bash
# Must pass all 210 tests on the release branch
pytest tests/core/search/ \
       tests/core/reasoning/test_got_bridge.py \
       tests/core/prediction/ \
       tests/core/test_resonance.py \
       tests/core/sovereign/test_apex_got_bridge_integration.py \
       tests/core/mcp/test_sovereign_phase46_tools.py \
       -v --tb=short

# Expected: 210 passed
```

## Pre-existing Failure Baseline

### `artifacts/known_failures_phase47_baseline.json`

```json
{
    "generated_at": "2026-02-19T...",
    "baseline_commit": "477e9d9",
    "total_known_failures": 28,
    "origin": "Phase 19 (commit 3afe972)",
    "test_node_ids": [
        "tests/integration/test_gate_chain_integration.py::TestGateChain::test_...",
        "tests/integration/test_omega_integration.py::TestOmega::test_...",
        "..."
    ],
    "error_signatures": [
        {"test": "test_gate_chain_integration::...", "error": "ImportError: ..."},
        "..."
    ]
}
```

### Comparator Logic

```
FUNCTION compare_failures(current_failures: List[str], baseline: Dict) -> bool:
    """Return True if only known failures present. False if new failures detected."""
    known_set = set(baseline["test_node_ids"])
    current_set = set(current_failures)
    new_failures = current_set - known_set
    IF new_failures:
        PRINT(f"NEW FAILURES DETECTED ({len(new_failures)}):")
        FOR f IN sorted(new_failures):
            PRINT(f"  - {f}")
        RETURN False
    PRINT(f"All {len(current_set)} failures are known baseline. OK.")
    RETURN True
```

## TDD Anchors

```python
class TestReleaseManifest:

    def test_manifest_contains_all_files(self):
        """Manifest includes every Phase 46 file."""
        # Generate manifest, check line count >= 18

    def test_manifest_checksums_match(self):
        """Checksums in manifest match actual file hashes."""

    def test_missing_file_marked(self):
        """Missing files are marked as MISSING, not silently skipped."""


class TestSemanticChecks:

    def test_import_smoke_all_modules(self):
        """All Phase 46 modules import without error."""

    def test_api_surface_exports(self):
        """Expected symbols are exported from each package."""


class TestFailureComparator:

    def test_only_known_failures_passes(self):
        """Baseline-only failures pass the comparator."""
        baseline = {"test_node_ids": ["test_a", "test_b"]}
        assert compare_failures(["test_a", "test_b"], baseline) is True

    def test_new_failure_detected(self):
        """New failure triggers comparator failure."""
        baseline = {"test_node_ids": ["test_a"]}
        assert compare_failures(["test_a", "test_new"], baseline) is False

    def test_fewer_failures_passes(self):
        """Fewer failures than baseline is acceptable."""
        baseline = {"test_node_ids": ["test_a", "test_b"]}
        assert compare_failures(["test_a"], baseline) is True
```

## Gate: Stop Conditions

**Do NOT proceed to canary ramp if any of these fail:**

1. Cherry-pick conflicts on release branch
2. Any import smoke failure
3. API surface mismatch
4. Phase 46 tests < 210 passing
5. New test failures beyond baseline
