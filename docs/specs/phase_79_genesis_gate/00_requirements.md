# Phase 79: Genesis-100 Gate Module — Requirements

> **Phase:** 79
> **Date:** 2026-03-08
> **Status:** SPEC
> **Depends on:** constants.py (SSoT), proof_engine, token, constitutional, sovereign/api
> **Produces:** `core/sat/` module, `genesis_100_ceremony()`, audit scorecard CLI

---

## 1. Problem Statement

The Genesis-100 Definition of Done specifies 68 checks across 5 layers, enforced by 5 SAT agents. The audit (2026-03-08) found:

- **No `core/sat/` module** — SAT logic is scattered in sat_controller.py
- **No `genesis_100_ceremony()` function** — genesis_ceremony.py is identity bootstrap only
- **No canonical DoD document** in docs/canon/
- **3 constitutional mismatches** blocking Layer 2-3 gates

The system cannot run its own release ceremony. This phase creates the executable gate infrastructure.

---

## 2. Functional Requirements

### FR-1: GateResult Data Model
- Immutable dataclass: agent, layer, passed, checks, failed, verdict, timestamp
- `to_dict()` for JSON serialization
- `to_receipt()` for evidence chain integration

### FR-2: Five SAT Gate Functions
Each returns `GateResult`. Each is independently runnable.

| Function | Layer | Automated Checks | Manual Checks |
|----------|-------|-----------------|---------------|
| `sentinel_verify()` | STRUCTURAL_INTEGRITY | 12 | 0 |
| `oracle_s_verify()` | CONSTITUTIONAL_COMPLIANCE | 10 | 4 |
| `ledger_verify()` | ECONOMIC_SOUNDNESS | 10 | 0 |
| `conductor_verify()` | OPERATIONAL_READINESS | 13 | 0 |
| `ambassador_verify()` | HUMAN_VERIFICATION | 4 | 15 |

### FR-3: Genesis-100 Ceremony
- Runs all 5 gates sequentially
- Generates signed GenesisReceipt
- Stores receipt in evidence chain
- Returns bool (all passed / blocked)
- Prints formatted result to stdout

### FR-4: CLI Integration
- `bizra gate sentinel` — run single gate
- `bizra gate all` — run all 5 gates
- `bizra gate ceremony` — full Genesis-100 ceremony
- `bizra gate status` — show last gate results
- `--skip-manual` flag to skip human attestation checks
- `--json` flag for machine-readable output

### FR-5: Audit Scorecard
- `bizra gate scorecard` — formatted table of all 68 checks
- Shows PASS/FAIL/PARTIAL/NOT-IMPL/SKIPPED per check
- Calculates per-layer and overall pass rates

---

## 3. Critical Fixes (Pre-requisite)

These must land BEFORE the gate module, or Layer 2-3 will auto-fail:

### FIX-1: Harberger Tax Rate (7% -> 5%)
- **File:** `core/integration/constants.py` L248
- **Change:** `ADL_HARBERGER_TAX_RATE: Final[float] = 0.07` → `0.05`
- **Cascade:** Update all 5 modules that replicate this value
- **Decision needed:** Is 5% correct, or should the DoD say 7%? Ask Mumo.

### FIX-2: BLOOM Soulbound Transfer Block
- **File:** `core/token/ledger.py` in `_validate_transaction()`
- **Add:** Reject BLOOM transfers (same pattern as IMPT block at L373-375)

### FIX-3: Community Pool 50% Enforcement
- **File:** `core/token/mint.py` in `mint_seed()`
- **Add:** Route 50% of minted SEED to community pool
- **Note:** Currently only zakat (2.5%) goes to pool. This is a significant economic change.
- **Decision needed:** Confirm with Mumo that 50% applies to SEED minting, not just BLOOM redistribution.

---

## 4. Non-Functional Requirements

- NFR-1: All thresholds imported from `core/integration/constants.py` — zero hardcoded values
- NFR-2: Gate functions must complete in < 30 minutes total (automated checks only)
- NFR-3: Manual checks use `prompt_human()` — returns True/False from stdin or defaults to False in CI
- NFR-4: Gate results stored as evidence blocks with Ed25519 signatures
- NFR-5: Module must work without optional deps (k6, cosign, syft) — gracefully skip with NOT-IMPL status

---

## 5. Out of Scope (Phase 79)

- k6 load test infrastructure (separate phase)
- i18n/RTL framework (separate phase)
- UX test protocol documents (Ambassador manual gates)
- `bizra uninstall` command (separate phase)
- Database backup/restore scripts (separate phase)

---

## 6. Test Plan

- Unit tests for GateResult model (serialization, to_dict, to_receipt)
- Unit tests for each gate function with mocked subprocess calls
- Integration test: full ceremony with `--skip-manual`
- Property test: GateResult.passed == all(check[1] for check in checks)
- CLI test: `bizra gate scorecard --json` produces valid JSON
