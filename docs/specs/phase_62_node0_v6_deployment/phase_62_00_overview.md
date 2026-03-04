# Phase 62: NODE0 v6 Deployment — Overview

## Context

Phase 61 deployed the Genesis Engine v5 (constitution parser + gates + router +
pipeline + cache + evidence) into `bizra-constitution/` with 196 tests passing.
The bridge module (`core/bridges/constitutional_engine.py`) imports all 7
components with graceful fallback. Constants v3.0.0 absorbed 30+ constitutional
values into the SOT (`core/integration/constants.py`).

The SPARC analysis of `bizra-node0-v6.zip` reveals 6 NEW production modules
that complete the NODE0 stack:

```
v5 (deployed):  Constitution → Gates → Cache → Router → Pipeline → Evidence
v6 (this phase):                                          ↓ NEW ↓
                 Identity (Ed25519) → Ollama (circuit breaker) → Production Pipeline
                 → FastAPI Server → Wire Adapter → Event Bus Bridge
```

## Objective

Deploy v6 modules into `bizra-constitution/`, update the bridge, wire into
the existing Node0 MissionOrchestrator, and validate with 332+ tests.

## Deliverables

| ID | Deliverable | Spec File | Est. Time |
|----|-------------|-----------|-----------|
| D1 | Copy v6 modules + tests into workspace | `phase_62_01_deploy.md` | 10 min |
| D2 | Update bridge to import v6 components | `phase_62_02_bridge.md` | 15 min |
| D3 | Fix 3 issues from SPARC analysis | `phase_62_03_fixes.md` | 20 min |
| D4 | Regression verification: 332+ constitution tests | `phase_62_04_verify.md` | 10 min |
| D5 | Wire adapter integration test with live Ollama | `phase_62_05_wire_live.md` | 15 min |

## Constraints

- Zero modification to existing v5 modules (backward compatible)
- All 15 unchanged files remain identical
- `evidence_receipt.py` gets +1 method backport only
- Bridge fallback behavior preserved (core/ works without bizra-constitution/)
- No hardcoded secrets, URLs, or tokens
- PyNaCl optional (HMAC fallback functional)

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FastAPI dep not in venv | Low | Medium | Already installed (checked) |
| PyNaCl not installed | Medium | Low | HMAC fallback tested (4 tests skip gracefully) |
| Port collision (7770) | Low | Low | Server not auto-started; manual activation only |
| `on_event` deprecation | Low | Low | Cosmetic warning; fix in D3 |
| CORS `allow_origins=*` | Medium | Medium | Fix in D3 before any network exposure |

## Success Criteria

1. `python -m pytest bizra-constitution/ -q` → 332+ passed, 0 failed
2. `GENESIS_ENGINE_AVAILABLE = True` with all v6 components
3. Wire adapter executes mission through full constitutional pipeline
4. Evidence receipts are cryptographically signed with Integrator agent key
5. Zero regressions in `tests/core/` suite

## Dependencies

- Phase 61 completed (constants v3.0.0, bridge, 196 tests)
- Ollama running with phi3:mini for D5
- `.venv-linux` with fastapi, pydantic, uvicorn installed
