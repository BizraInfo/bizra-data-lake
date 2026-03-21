---
> **DOCUMENT CLASSIFICATION: Architectural Exploration — AI-Assisted**
>
> This document was produced through AI-assisted collaborative analysis.
> It represents architectural thinking and design exploration, NOT verified
> test output or empirical measurement. Claims within should be validated
> against the canonical codebase (`cargo test`, `pytest`, STATUS.md).
>
> For verified evidence, see: `artifacts/CANONICAL_SPEARPOINT_V1/`
---

# Phase 49 Spec — Part 5: Roadmap

> Standing on Giants: Deming (PDCA — consolidate before expanding) · Brooks (no silver bullet — eliminate waste)

## Phase 49 Theme: Refinement & Consolidation

Phase 49 is NOT about building new features. It's about:
1. Closing test coverage gaps
2. Removing dead code
3. Activating dormant infrastructure
4. Hardening CI

## Sprint Plan

### Sprint 49.1: Test Coverage (IN PROGRESS)

| Task | LOC Covered | Tests | Status |
|------|-------------|-------|--------|
| `core/graph/semantic_layer.py` | 908 | ~25 | Agent writing |
| `core/rdve/stability.py` | 422 | ~15 | Agent writing |
| `core/rdve/interdisciplinary.py` | 754 | ~15 | Agent writing |
| `core/rdve/orchestrator.py` types | 935 | ~15 | Agent writing |

**Success criteria:** All 4 test files pass, coverage gap closes from 3,076 LOC → 0 LOC.

### Sprint 49.2: Native Cleanup

```pseudocode
1. verify_source_identical("native/", "bizra-omega/")  # diff -rq
2. grep_no_python_references("native/")                 # zero hits
3. delete("native/")                                     # remove 8,472 LOC
4. delete_or_redirect(".github/workflows/native-ci.yml") # retire CI
5. update(".gitignore")                                  # remove native/target
6. cargo_test("bizra-omega/")                            # 610 tests pass
```

**Success criteria:** `native/` directory deleted, CI green, zero references.

### Sprint 49.3: Canary Ramp Stage 1

```pseudocode
1. set_env("BIZRA_PHASE46_SEARCH_PERCENT", "10")
2. monitor_for(hours=4)
3. verify_search_hit_rate >= 0.5
4. verify_search_latency_p95 < 200ms
5. verify_zero_rollbacks
```

**Success criteria:** 10% of MCP queries use FAISS search with acceptable metrics.

### Sprint 49.4: CI Hardening

| Task | File | Action |
|------|------|--------|
| Retire native-ci.yml | .github/workflows/native-ci.yml | Delete or redirect |
| Add graph/rdve test coverage | .github/workflows/ci.yml | Include in blanket run |
| Verify Rust workspace test count | CI | Assert >= 610 tests |
| Coverage floor ratchet | pyproject.toml | Bump from 60% → 65% |

### Sprint 49.5: CanaryRouter Performance (DONE)

Fixed per-call `CanaryRouter()` construction in `apex_engine.py`. Now cached on `self._canary_router`. Completed in Phase 48.1.

---

## What Phase 49 Does NOT Include

| Item | Reason | When |
|------|--------|------|
| New Rust crates (agent, node, protocol) | YAGNI — PyO3 bridge covers needs | Phase 50+ if edge deployment needed |
| Python core/ expansion | 171K LOC is enough | Never — optimize, don't expand |
| Tauri desktop UI | No stable agent runtime to wrap | Phase 51+ |
| Canary ramp beyond Stage 1 | Need Stage 1 data first | Phase 49.4+ |
| Production deployment | Need canary validation first | Phase 50 |

---

## Success Metrics (Phase 49 Complete)

| Metric | Before | Target |
|--------|--------|--------|
| Python test coverage (core/graph + core/rdve) | 0 tests | ~70 tests |
| native/ directory | 8,472 LOC duplicate | Deleted |
| Canary stage | 0 (dormant) | 1 (search 10%) |
| CI rollout regression step | Added | Verified green |
| CanaryRouter caching | Per-call construction | Cached on self |
| Total Rust tests | 610 | 610 (no regressions) |
| Total Python tests | ~7,300 | ~7,370 |
