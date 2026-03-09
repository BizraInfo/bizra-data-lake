# Phase 79: Module Map & Implementation Order

---

## File Structure

```
core/sat/
├── __init__.py              # Public API + __all__
├── gate_result.py           # GateResult, CheckResult, CheckStatus
├── sentinel_gate.py         # Layer 1: Structural Integrity (12 checks)
├── oracle_s_gate.py         # Layer 2: Constitutional Compliance (14 checks)
├── ledger_gate.py           # Layer 3: Economic Soundness (10 checks)
├── conductor_gate.py        # Layer 4: Operational Readiness (13 checks)
├── ambassador_gate.py       # Layer 5: Human Verification (19 checks)
└── ceremony.py              # GenesisReceipt + genesis_100_ceremony()

tests/core/sat/
├── __init__.py
├── test_gate_result.py      # GateResult model tests
├── test_sentinel_gate.py    # Sentinel gate tests (mocked subprocess)
├── test_oracle_s_gate.py    # Oracle-S gate tests
├── test_ledger_gate.py      # Ledger gate tests (temp DBs)
├── test_conductor_gate.py   # Conductor gate tests
├── test_ambassador_gate.py  # Ambassador gate tests
└── test_ceremony.py         # Full ceremony integration test

docs/canon/
└── genesis_100_definition_of_done.md   # Canonical DoD document (LOCKED)
```

---

## Implementation Order

### Sprint 1: Foundation (Day 1)
1. `gate_result.py` — data model (no deps)
2. `tests/core/sat/test_gate_result.py` — TDD first
3. `__init__.py` — module registration

### Sprint 2: Pre-requisite Fixes (Day 1)
4. **FIX-1:** Harberger rate 7% → 5% (pending Mumo decision)
5. **FIX-2:** BLOOM soulbound transfer block in ledger.py
6. **FIX-3:** Community pool 50% enforcement (pending Mumo decision)

### Sprint 3: Gate Functions (Day 2)
7. `sentinel_gate.py` + tests
8. `oracle_s_gate.py` + tests
9. `ledger_gate.py` + tests
10. `conductor_gate.py` + tests
11. `ambassador_gate.py` + tests

### Sprint 4: Ceremony + CLI (Day 2)
12. `ceremony.py` + tests
13. CLI integration in `__main__.py`
14. `docs/canon/genesis_100_definition_of_done.md`

### Sprint 5: Verification (Day 3)
15. Run `bizra gate scorecard` — verify all 68 checks report
16. Run `bizra gate ceremony --skip-manual --skip-slow` — verify end-to-end
17. Fix any integration issues

---

## Dependency Graph

```
gate_result.py (0 deps)
    ↑
sentinel_gate.py (subprocess, constants)
oracle_s_gate.py (subprocess, constants)
ledger_gate.py (token.mint, token.ledger)
conductor_gate.py (subprocess, filesystem checks)
ambassador_gate.py (subprocess, filesystem checks)
    ↑
ceremony.py (all 5 gates, proof_engine, genesis_ceremony)
    ↑
__main__.py CLI wiring
```

---

## Constants Required (from constants.py SSoT)

| Constant | Value | Used By |
|----------|-------|---------|
| `UNIFIED_IHSAN_THRESHOLD` | 0.95 | Oracle-S 3.1, Ledger 4.1 |
| `UNIFIED_SNR_THRESHOLD` | 0.85 | Oracle-S 3.2 |
| `ADL_GINI_THRESHOLD` | 0.35 | Oracle-S 3.3, Ledger 4.6 |
| `ZAKAT_RATE` | 0.025 | Oracle-S 3.5, Ledger 4.5 |
| `ADL_HARBERGER_TAX_RATE` | 0.05* | Oracle-S 3.6 |
| `BLOOM_REDISTRIBUTION_RATE` | 0.50 | Oracle-S 3.4 |
| `SEED_YEARLY_CAP` | 1,000,000 | Ledger 4.8 |
| `COVERAGE_FLOOR` | 38 | Sentinel 1.5 |

*After FIX-1

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| FIX-1 (Harberger) changes economic behavior | HIGH | Mumo decides correct rate before implementation |
| FIX-3 (50% pool) is a major economic change | HIGH | Confirm this applies to SEED minting, not just BLOOM |
| Full test suite takes > 30 min | MEDIUM | `--skip-slow` flag for development iteration |
| k6 not installed → Conductor gates NOT-IMPL | LOW | Acceptable for Phase 79 — k6 is separate phase |
| Manual checks block ceremony in CI | LOW | `--skip-manual` flag + CI always uses it |
