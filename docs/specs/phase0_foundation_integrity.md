# Phase 0: Foundation Integrity — Specification

> Standing on Giants: Al-Ghazali (Ihsan, 1095) · Shannon (SNR, 1948) · Lamport (single source of truth, 1978)

## Status: PARTIALLY COMPLETE

Prior sessions (Mastermind Sprint, Activation Sprint, Deep Audit) completed the **threshold consolidation** portion of Phase 0. This spec covers the remaining work.

## Ground Truth (verified 2026-02-24)

| Metric | Value | Source |
|--------|-------|--------|
| Files referencing Ihsan | 240 | `grep -rl 'ihsan\|IHSAN' core/` |
| Files with Ihsan gates | 127 (52.9%) | Regex: `if.*ihsan.*(>=\|<=\|threshold\|gate)` |
| Files without gates | 109 (45.4%) | Reference only — constants, types, docstrings, re-exports |
| Hardcoded threshold drift | 0 | Integration audit: 151+ files, 100% compliance |
| Orphan modules (zero imports) | 4 | core/auth, core/pek, core/command, core/embedding |
| Test suite | 8,103 tests, 0 collection errors | pytest --co -q |

## Phase 0A: COMPLETE (Prior Sessions)

- [x] Consolidate 20+ hardcoded thresholds to `core/integration/constants.py`
- [x] All modules import from single source of truth
- [x] Rust `bizra-core/src/lib.rs` aligned with Python constants
- [x] `validate_cross_repo_consistency()` available for audits
- [x] CI hard-gates security scans (Trivy, pip-audit, bandit, cargo-audit)

## Phase 0B: Remaining Work

### 0B.1 — Classify the 109 Non-Gating Ihsan References

Not all 109 files need gates. Many are legitimate:
- **Re-exports / __init__.py** (~20 files): Import and expose constants
- **Type definitions** (~15 files): Define IhsanScore types
- **Documentation / docstrings** (~25 files): Reference concept without needing runtime check
- **Test files** (0 — tests are in tests/, not core/)

**Actual enforcement gap** (estimated): ~50 files that perform operations where an Ihsan gate would be meaningful but is absent.

### Pseudocode: Classify Non-Gating Files

```
FOR each file in 109_non_gating_files:
    category = classify(file):
        IF file is __init__.py → SKIP (re-export)
        IF file only has IHSAN in docstring/comment → SKIP (documentation)
        IF file defines type/dataclass with ihsan field → SKIP (type definition)
        IF file imports IHSAN_THRESHOLD but uses it in config → SKIP (configuration)
        IF file performs inference/query/mutation without ihsan check → FLAG

    IF category == FLAG:
        add to enforcement_targets[]

ASSERT len(enforcement_targets) < 60  # Expected ~50
```

### 0B.2 — Tag Orphan Modules with Archive Metadata

4 confirmed orphan modules with zero external imports:

| Module | Files | Superseded By | Archive Reason |
|--------|-------|--------------|----------------|
| `core/auth/` | 4 | `core/sovereign/genesis_identity.py` | Sovereign identity replaced centralized auth |
| `core/pek/` | 2 | `core/orchestration/proactive_*.py` | Proactive kernel absorbed into sovereign |
| `core/command/` | 2 | `core/sovereign/runtime_core.py` CLI | CLI stub replaced by runtime CLI |
| `core/embedding/` | 3 | `core/inference/` | Static embeddings superseded by dynamic inference |

**NOT an orphan**: `core/personaplex/` — has external integration via `core/voice/personaplex_bridge.py`.

### Pseudocode: Archive Orphan Modules

```
FOR each orphan in [auth, pek, command, embedding]:
    # Add archive docstring to __init__.py
    prepend_archive_header(
        module=orphan,
        superseded_by=SUPERSESSION_MAP[orphan],
        reason=REASON_MAP[orphan],
        delete_when="All test assertions covered by successor module"
    )

    # Exclude from lazy registry if registered
    IF orphan in core/__init__.py lazy imports:
        remove_from_lazy_registry(orphan)
```

### 0B.3 — Universal Ihsan Gate (Hook System)

The hook-based universal gate targets the ~50 files identified in 0B.1. Rather than adding `if ihsan < threshold` to 50 files, a single hook entry covers all action paths.

**Prerequisite**: `core/elite/hook_actions.py` hook registry (already exists).

### Pseudocode: Universal Gate

```yaml
# .bizra-kernel/hooks.yaml — universal enforcement
ihsan:
  universal_gate:
    phase: PRE_VALIDATE
    priority: HIGH
    action: ihsan_gate_check
    condition: always
    on_fail: block_with_ihsan_report
    threshold_source: core.integration.constants.UNIFIED_IHSAN_THRESHOLD
```

```python
# core/elite/hook_actions.py (extend existing)

@hook_action("ihsan_gate_check")
async def _action_ihsan_gate(ctx: HookContext) -> HookResult:
    """Universal Ihsan enforcement.

    Standing on Giants: Al-Ghazali (1095) — excellence is not optional.
    Threshold source: core/integration/constants.py (single source of truth)
    """
    from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

    threshold = ctx.params.get("threshold", UNIFIED_IHSAN_THRESHOLD)
    score = ctx.runtime.current_ihsan_score()

    if score >= threshold:
        return HookResult(status=PASS, ihsan=score)

    return HookResult(
        status=FAIL,
        ihsan=score,
        reason=f"Ihsan {score:.3f} < {threshold}",
        remediation="quality_improvement_cycle",
    )
```

## Acceptance Criteria

| Criterion | Metric | Gate |
|-----------|--------|------|
| Threshold drift | 0 files | Hard (already passing) |
| Orphan modules tagged | 4/4 with archive headers | Hard |
| Non-gating files classified | 109 categorized | Informational |
| Universal gate wired | Hook fires on PRE_VALIDATE | Hard |
| Tests pass | 8,103+ with 0 regressions | Hard |

## Dependencies

- **Blocks**: Phase 1 (HookConfigLoader needs constants consolidated — DONE)
- **Blocked by**: Nothing (can start immediately)
- **Estimated scope**: ~15 file edits, 0 new modules

## File Map

| File | Action |
|------|--------|
| `core/auth/__init__.py` | Add archive header |
| `core/pek/__init__.py` | Add archive header |
| `core/command/__init__.py` | Add archive header |
| `core/embedding/__init__.py` | Add archive header |
| `core/elite/hook_actions.py` | Add `ihsan_gate_check` action |
| `.bizra-kernel/hooks.yaml` | Add universal gate entry |
| `docs/specs/phase0_foundation_integrity.md` | This file |
