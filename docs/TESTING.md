# BIZRA Testing Guide

Last updated: 2026-02-14

This document defines the practical test strategy for local development and CI alignment.

## 1. Testing Objectives

- Prevent regressions in constitutional gates, runtime behavior, and API contracts.
- Maintain Python and Rust compatibility across the hybrid stack.
- Keep quality gates executable on developer machines and in CI.

## 2. Baseline Commands

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate

# Python full suite (default path = tests/)
pytest tests/

# Rust workspace suite
cd bizra-omega && cargo test --workspace
```

## 3. Fast Local Gate (Recommended Before Commit)

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
pytest -q tests/core/sovereign/test_runtime_types.py --capture=no
pytest -q tests/core/proof_engine/test_receipt.py --capture=no
pytest -q tests/core/sovereign/test_api_metrics.py --capture=no
```

## 4. Marker-Based Selection

Markers are defined in `pyproject.toml` (`tool.pytest.ini_options.markers`).

```bash
# Exclude heavy/external suites
pytest tests/ -m "not slow and not requires_ollama and not requires_gpu and not requires_network and not requires_llama_cpp"

# Integration only
pytest tests/ -m integration
```

## 5. Test Suites by Module

| Suite | Path | Test Count | Description |
|-------|------|------------|-------------|
| Proof Engine | `tests/core/proof_engine/` | ~553 | Receipts, Ed25519, BLAKE3, evidence ledger, POI |
| Sovereign Runtime | `tests/core/sovereign/` | ~270 | Runtime core, types, API, metrics, state |
| Spearpoint | `tests/core/spearpoint/` | ~235 | RDVE, patterns, orchestrator, evaluator |
| Bridges | `tests/core/bridges/` | ~137 | Desktop Bridge, Sci-Reasoning, Rust bridge |
| Token System | `tests/core/token/` | ~87 | Ledger, minting, Ed25519 transactions |
| Skills | `tests/core/skills/` | ~83 | Skill router, Smart File Management |
| Integration | `tests/integration/` | ~322 | Cross-module, E2E, seven-layer stack |

### Running Specific Suites

```bash
# Proof engine
pytest tests/core/proof_engine/ -v

# Desktop Bridge
pytest tests/core/bridges/test_desktop_bridge.py -v

# Smart File Management
pytest tests/core/skills/test_smart_file_manager.py -v

# Token system
pytest tests/core/token/ -v

# Spearpoint + RDVE
pytest tests/core/spearpoint/ -v

# Integration tests
pytest tests/integration/ -v
```

## 6. Coverage

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
pytest tests/ --cov=core --cov-report=term-missing
```

Coverage policy is configured in `pyproject.toml`:

- Branch coverage enabled
- Current fail-under threshold: `60`
- Target: ratcheting toward 95%

## 7. Common Failure Patterns

### 7.1 Metrics Endpoint Regressions

- Symptom: `/v1/metrics` returns 500
- Guard tests:
  - `tests/core/sovereign/test_api_metrics.py`
  - `tests/core/sovereign/test_runtime_types.py`

### 7.2 Receipt Compatibility Regressions

- Symptom: historical receipts fail verification (`Signer mismatch`)
- Guard test:
  - `tests/core/proof_engine/test_receipt.py`

### 7.3 Runtime Dataclass Contract Drift

- Symptom: `AttributeError` in runtime pipeline (`model_used`, `processing_time_ms`, etc.)
- Guard test:
  - `tests/core/sovereign/test_runtime_types.py`

### 7.4 Desktop Bridge Auth Failures

- Symptom: All bridge commands return `-32001` or `-32002`
- Guard test:
  - `tests/core/bridges/test_desktop_bridge.py`
- Check: `BIZRA_BRIDGE_TOKEN` environment variable is set

### 7.5 Smart File Path Traversal

- Symptom: `SmartFileHandler` rejects valid paths
- Guard test:
  - `tests/core/skills/test_smart_file_manager.py`
- Check: `BIZRA_DATA_LAKE_ROOT` environment variable resolves correctly

## 8. CI Alignment Checklist

Before opening a PR:

1. Run the fast local gate.
2. Run marker-filtered suite (if no backend/GPU available).
3. Run full Python suite if change is cross-cutting.
4. Run relevant Rust workspace tests for `bizra-omega/*` changes.
5. Update docs when behavior/contracts are changed.

## 9. Documentation Gate

Documentation quality is enforced by:

- Workflow: `.github/workflows/docs-quality.yml`
- Policy: `scripts/ci_docs_quality.py`

Local run:

```bash
python scripts/ci_docs_quality.py
```

## 10. Test Design Principles

- Test externally visible contracts, not internal implementation details.
- Add regression tests for every bug fix that touched runtime/API/security contracts.
- Prefer deterministic fixtures and explicit assertions over snapshot noise.
- Use `tmp_path` for filesystem tests (Smart Files, receipt storage).
- Keep individual test timeout under 60 seconds (configured in `pyproject.toml`).
