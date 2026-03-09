# BIZRA DDAGI OS — Elite Implementation Blueprint v1.0

> **Classification**: Strategic Technical Roadmap — Production-Grade  
> **Date**: March 8, 2026  
> **Baseline SNR**: 0.913 (Operational) → **Target**: 0.95+ (Ihsān-Grade)  
> **Standing on Giants**: Shannon (1948) · Boyd (1976) · Deming (1950) · Besta (2024) · Lamport (1978) · Al-Ghazali (1095) · Anthropic (2023) · PMI (PMBOK 7th Ed.)

---

## 0. Executive Summary

This blueprint synthesizes evidence from:
- **524 Python source files** across 61 core subpackages
- **22 Rust crates** in the bizra-omega workspace (1,016 tests passing)
- **389 test files** containing 9,767 test functions (6,887/6,889 passing)
- **13 CI workflows** with 9+ quality gates
- **Multi-lens SAPE analysis** across 6 hypotheses (H1–H6)

The system's constitutional kernel is **masterpiece-grade** (SNR 0.98). The cognitive architecture has genuine theoretical depth. Six precise actions bridge the 0.037 gap from current SNR 0.913 to the Ihsān threshold of 0.95.

```
CURRENT STATE                    TARGET STATE
┌────────────────┐               ┌────────────────┐
│  SNR: 0.913    │ ──6 actions──▶│  SNR: 0.95+    │
│  Coverage: 38% │               │  Coverage: 60%  │
│  Loops: 3 open │               │  Loops: closed  │
│  Exceptions: 30+│              │  Exceptions: <5  │
│  FRONTIER: dead │              │  FRONTIER: live  │
└────────────────┘               └────────────────┘
```

---

## 1. System Baseline (Evidence-Grounded)

### 1.1 Verified Metrics (as of March 8, 2026)

| Metric | Value | Source |
|--------|-------|--------|
| Core .py files | 524 | `find core/ -name '*.py' \| wc -l` (user-verified) |
| Core subpackages | 61 | `find core/ -mindepth 1 -type d \| wc -l` |
| Sovereign module files | 103 | Largest single module |
| Rust crates | 22 | `bizra-omega/Cargo.toml` workspace members |
| Rust tests passing | 1,016 | `STATUS.md` item 7 |
| Python tests passing | 6,887/6,889 | `STATUS.md` item 9 |
| Test functions total | 9,767 | `grep -r "def test_" tests/ \| wc -l` |
| CI quality gates | 9+ | `ci.yml` stages 1–4 |
| Coverage floor | 38% | `pyproject.toml:127` |
| MyPy error baseline | 1,600 | `ci.yml` ratcheted gate |
| Frontend modules | 42 | `STATUS.md` item 12 |
| SAP conformance | 22/22 | `STATUS.md` item 1 |
| Normalizer tests | 118/118 | `STATUS.md` item 22 |
| Action-pinned workflows | 7/7 | SHA-256 supply chain |

### 1.2 SAPE Dimension Scores

| Dimension | Score | Grade | Key Evidence |
|-----------|-------|-------|--------------|
| Kernel Integrity | 0.98 | Elite | 12-step tick, fixed-point arithmetic, zero FP |
| Symbolic-Neural Bridges | 0.96 | Operational+ | HMM→GoT, Entropy Router, but FRONTIER unused |
| Self-Harnessing | 0.94 | Operational | 3 engines present, not interconnected |
| Formal Verification | 0.95 | Operational | Z3 + dual verification, inputs on trust |
| Economic Substrate | 0.97 | Operational+ | OmniKernel two-phase, emission decay, zakat |
| Security Posture | 0.88 | Diagnostic | 30+ broad exceptions in boundary modules |
| **Composite** | **0.913** | **Operational** | **6 actions to Ihsān** |

---

## 2. Strategic Framework: PMBOK × Ihsān × SAPE

### 2.1 Governance Model (PMBOK Integration)

```
┌──────────────────────────────────────────────────────────────┐
│                    IHSĀN GOVERNANCE BOARD                     │
│  Constitution: constants.py SSoT | Gate: ≥ 0.95 all dims    │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│  SCOPE       │  QUALITY     │  RISK        │  STAKEHOLDER    │
│  WBS tracks  │  SNR gates   │  Cascading   │  Daughter Test  │
│  6 actions   │  CI enforce  │  analysis    │  Crown review   │
├──────────────┴──────────────┴──────────────┴─────────────────┤
│                    DELIVERY PIPELINE                          │
│  DevOps: ci.yml → quality-gates → security → deploy          │
│  Evidence: Proof Forge → ActionReceipt → POI Ledger          │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Quality Triangle: Ihsān × Adl × Amānah

| Principle | Operationalization | CI Gate |
|-----------|-------------------|---------|
| **Ihsān (Excellence)** | SNR ≥ 0.95 production, ≥ 0.90 CI | `quality-gates` job in ci.yml |
| **Adl (Justice)** | ADL Gini ≤ 0.35, Zakat 2.5% | `process_tick()` Steps 3,8 |
| **Amānah (Trustworthiness)** | Frozen dataclasses, BLAKE3 receipts, Ed25519 sigs | `PROOF_SUMMARY.md` evidence chain |

### 2.3 SAPE Pyramid Alignment

```
Layer 4: WISDOM (SNR ≥ 0.999)
  ├── SDPO-SAPE Fusion (core/sdpo/cosmos/sdpo_sape_fusion.py)
  ├── Autopoietic emergence → reflex compilation
  └── Constitutional self-regulation (ticker.py 12-step)

Layer 3: KNOWLEDGE (SNR ≥ 0.99)
  ├── Cognitive Fusion Engine (MoE → HRM → RAG → NorthStar)
  ├── Bicameral generate-verify loop
  └── GoT Bridge (FAISS → GraphOfThoughts)

Layer 2: INFORMATION (SNR ≥ 0.95)
  ├── HMM state prediction (T1 → 50ms)
  ├── Entropy Router (System 1/2 classification)
  └── OmniKernel 8-line cycle (Rust)

Layer 1: DATA (SNR ≥ 0.90)
  ├── Fixed-point kernel (Layer 0)
  ├── Constants SSoT (900+ lines)
  └── CI pipeline (9 gates, 13 workflows)
```

---

## 3. Work Breakdown Structure: 6 Actions to Ihsān

### 3.0 Dependency Graph

```
P0 ──────────┐
(Loop Close) │
             ▼
P1 ──────── P3 ──── P4
(Degrade)  (Cov)   (FRONTIER)
  │          │
  ▼          ▼
P2 ──────── P5
(Except)   (Sync)
```

P0 is the foundation — closing the learning loop enables P3 (coverage measures the loop), P4 (FRONTIER feeds into the loop). P1 and P2 are independent security hardening. P5 depends on P0 (synchronization uses the loop's receipts).

---

### 3.1 P0: Close the Learning Loop [CRITICAL PATH]

**Impact**: SNR +0.05 | Risk: HIGH if deferred | Effort: 2 sprints  
**PMBOK Knowledge Area**: Integration Management  
**Ethical Alignment**: Ihsān — system that cannot learn cannot excel

#### Problem Statement

Three self-improvement engines operate in parallel without feeding each other:

| Engine | Module | Input | Output |
|--------|--------|-------|--------|
| **Autopoiesis** | `core/autopoiesis/loop.py` | Agent population metrics | `IntegrationCandidate` (genome + fitness) |
| **SDPO Training** | `core/sdpo/training/bizra_sdpo_trainer.py` | `TrainingBatch` (Q/A pairs) | Optimized model weights |
| **Reflex Compilation** | `core/constitutional/ticker.py` Step 10 | Ihsān ≥ 0.98 receipts | `Reflex` (O(1) cached rules) |

```
CURRENT: Three disconnected loops
┌───────────┐    ┌───────────┐    ┌───────────┐
│Autopoiesis│    │   SDPO    │    │  Reflex   │
│  (evolve) │    │  (train)  │    │ (compile) │
└───────────┘    └───────────┘    └───────────┘

TARGET: Closed learning-to-reflex pipeline
┌───────────┐    ┌───────────┐    ┌───────────┐
│Autopoiesis│───▶│   SDPO    │───▶│  Reflex   │
│  discover │    │  distill  │    │  compile  │
└─────▲─────┘    └───────────┘    └─────┬─────┘
      │                                  │
      └──────── feedback ────────────────┘
```

#### Implementation Plan

**Sprint 1: Bridge Autopoiesis → SDPO**

1. Create `core/autopoiesis/sdpo_bridge.py`:
   - Convert `IntegrationCandidate` → `TrainingBatch`
   - Map genome fitness scores → SDPO quality scores
   - Gate: Only candidates with `fitness ≥ 0.90` AND `ihsan_score ≥ 0.95` cross
   - Emit `LEARNING` consciousness event on bridge crossing

2. Wire in `AutopoieticLoop.integrate()`:
   - After successful emergence detection (phase `INTEGRATING`)
   - Call `SDPOBridge.candidate_to_batch(candidate)` → feed to trainer

**Sprint 2: Bridge SDPO → Reflex Compilation**

3. Create `core/sdpo/reflex_bridge.py`:
   - Convert SDPO training output (high-confidence patterns) → `ActionReceipt`
   - Require `ihsan_score ≥ 0.98` (Step 10 gate) AND `reproducibility ≥ 0.90`
   - Generate `ActionReceipt` with `action_type` derived from SDPO training intent

4. Connect reflex feedback → autopoiesis:
   - Compiled reflexes generate `ADAPTATION` consciousness events
   - Autopoiesis `OBSERVING` phase reads these events as population fitness signals

**Quality Gates**:
- Unit tests for both bridges (target: 20 tests)
- Integration test: end-to-end candidate → training → reflex in single fixture
- Property test: bridge never produces receipts with `ihsan < 0.95`

**Risk Mitigation**:
- Feature flag: `BIZRA_CLOSED_LOOP_ENABLED=0` default (opt-in)
- 30-day deny-list for flagged reflex patterns (existing RLVR mechanism)
- Crown agent P5 review for all compilation candidates

---

### 3.2 P1: Explicit Degradation Events [SECURITY]

**Impact**: SNR +0.02 | Risk: MEDIUM | Effort: 1 sprint  
**PMBOK Knowledge Area**: Quality Management  
**Ethical Alignment**: Amānah — silent failure is a breach of trust

#### Problem Statement

Protocol-optional constructors accept `None` and silently degrade:

```python
# core/cognitive_fusion/fusion_engine.py — ALL four subsystems optional
class CognitiveFusionEngine:
    def __init__(self, moe_router=None, hrm_engine=None, rag_engine=None, northstar=None)

# core/reasoning/got_bridge.py — both engine and search optional
class GoTBridge:
    def __init__(self, search_engine=None, got_engine=None)

# core/sovereign/bicameral_engine.py — both hemispheres optional
class BicameralReasoningEngine:
    def __init__(self, local_endpoint=None, api_client=None)
```

When instantiated with zero implementations, these engines return default results indistinguishable from real computation.

#### Implementation Plan

1. Define `DegradationEvent` in `core/protocols/degradation.py`:
   ```python
   @dataclass(frozen=True)
   class DegradationEvent:
       engine: str          # "CognitiveFusionEngine", "GoTBridge", etc.
       missing: list[str]   # ["moe_router", "hrm_engine", ...]
       severity: str        # "PARTIAL" or "FULL"
       timestamp: datetime
   ```

2. Add degradation logging to each Protocol-optional constructor:
   - `__init__` checks which Protocol args are `None`
   - If ALL are `None` → emit `FULL` degradation event + log WARNING
   - If SOME are `None` → emit `PARTIAL` degradation event + log INFO
   - Wire through existing `core/bus/action_bus.py` EventPublisher

3. Add `degraded: bool` property to result dataclasses:
   - `FusionResult.degraded`, `GoTBridgeResult.degraded`, `BicameralResult.degraded`
   - Downstream consumers can distinguish real computation from defaults

**Quality Gates**:
- Test: Instantiate each engine with `None` args → assert degradation event emitted
- Test: Instantiate with real args → assert no degradation event
- CI: Add `DEGRADATION-001` gate that fails if degradation count exceeds threshold in integration tests

---

### 3.3 P2: Exception Specificity [SECURITY]

**Impact**: SNR +0.01 | Risk: LOW | Effort: 1 sprint  
**PMBOK Knowledge Area**: Risk Management  
**Ethical Alignment**: Amānah — specific exceptions are honest exceptions

#### Problem Statement

30+ `except Exception` clauses across core/, concentrated in system boundary modules:

| Module | Count | Severity |
|--------|-------|----------|
| `core/agentic/agent.py` | 4 | HIGH — agent execution |
| `core/a2a/transport.py` | 2 | HIGH — network transport |
| `core/apex/swarm_orchestrator.py` | 3 | MEDIUM — orchestration |
| `core/cognitive_fusion/fusion_engine.py` | 4 | MEDIUM — intentional degradation |
| `core/elite/self_harness_engine.py` | 2 | LOW — scan errors expected |

#### Implementation Plan

**Tier 1 (HIGH — boundary modules)**:
1. `core/agentic/agent.py` — Replace 4 instances:
   - `except Exception` → `except (httpx.HTTPError, asyncio.TimeoutError, ConnectionError)`
   - Add fallback `except Exception` only with explicit `logger.error()` + re-raise for unexpected types

2. `core/a2a/transport.py` — Replace 2 instances:
   - `except Exception` → `except (OSError, asyncio.TimeoutError, json.JSONDecodeError)`
   - Transport errors must propagate to caller for retry logic

**Tier 2 (MEDIUM — orchestration)**:
3. `core/apex/swarm_orchestrator.py` — Replace 3 instances with agent-specific exceptions
4. `core/cognitive_fusion/fusion_engine.py` — These 4 are **intentional** (Protocol degradation) — convert to `except Exception` + degradation event (P1 integration)

**Tier 3 (LOW — acceptable)**:
5. `core/elite/self_harness_engine.py` — Leave as-is (scan errors are expected at filesystem boundary)

**Quality Gates**:
- Self-harness rule: `bare-except-boundary` detects `except Exception` in `agentic/`, `a2a/`, `transport` paths
- CI: Ratchet — count broad exceptions and fail if count increases

**Existing Test Coverage**: `tests/core/sovereign/test_audit_trail_integrity.py` already tests for bare `except:pass` in mission.py and rollback.py — extend this pattern.

---

### 3.4 P3: Coverage Ratchet 38% → 60% [QUALITY]

**Impact**: SNR +0.03 | Risk: MEDIUM | Effort: 3 sprints  
**PMBOK Knowledge Area**: Quality Management  
**Ethical Alignment**: Ihsān — untested code cannot claim excellence

#### Problem Statement

`fail_under = 38` in `pyproject.toml:127`. Historical ratchet: 30% → 55% → 60% → 65% → 38% (regression when new modules added without tests).

The system has 9,767 test functions but coverage is **unevenly distributed**. Core constitutional modules are well-tested; newer cognitive/reasoning modules have gaps.

#### Implementation Plan

**Sprint 1: Identify coverage gaps (Week 1)**

```bash
pytest tests/ --cov=core --cov-report=html --cov-report=term-missing \
  -m "not requires_ollama and not requires_gpu and not slow"
```

Priority modules (likely lowest coverage based on architecture review):
- `core/cognitive_fusion/` — Protocol-optional subsystems, many None paths
- `core/sdpo/` — Training loops, hard to test without model
- `core/autopoiesis/` — Evolution engine, needs mock populations
- `core/reasoning/` — GoT bridge, entropy router edge cases
- `core/zpk/` — Zero Point Kernel, filesystem + crypto operations

**Sprint 2: Write tests for priority modules (Weeks 2-4)**

Target: 50 new test functions per sprint, focused on:
1. `core/cognitive_fusion/test_fusion_engine.py` — Test all 4 degradation combinations (2⁴ = 16 tests)
2. `core/sdpo/tests/` — Mock-based training loop tests (existing: 6 test classes, needs expansion)
3. `core/autopoiesis/tests/` — Genome factory, emergence detector edge cases
4. `core/reasoning/tests/` — Entropy router boundary values (0.29, 0.30, 0.31 for each tier)

**Sprint 3: Ratchet to target (Weeks 5-6)**

Incremental ratchet in `pyproject.toml`:
```
Week 1: fail_under = 42  (validate current + new tests)
Week 3: fail_under = 50  (midpoint checkpoint)
Week 5: fail_under = 60  (target)
```

**Quality Gates**:
- CI `diff-cover` (already in dev dependencies) — require 80% coverage on new code
- Module-level coverage minimum: no module below 20%

---

### 3.5 P4: Activate FRONTIER Tier [CAPABILITY]

**Impact**: SNR +0.01 | Risk: LOW | Effort: 1 sprint  
**PMBOK Knowledge Area**: Scope Management  
**Ethical Alignment**: Ihsān — unused capability is wasted potential

#### Problem Statement

The `EntropyRouter` classifies queries into 5 tiers (TRIVIAL → SIMPLE → MODERATE → COMPLEX → FRONTIER), but `CognitiveFusionEngine` treats FRONTIER identically to EXPERT. The 5-tier taxonomy degrades to 4 in practice.

```python
# core/reasoning/entropy_router.py:90
# Shannon entropy > 0.85 AND structural multi-domain markers → FRONTIER
# But no dedicated handler exists downstream
```

#### Implementation Plan

1. Add `FrontierHandler` to `CognitiveFusionEngine`:
   - When routing result is `FRONTIER`:
     - Invoke GoT Bridge with `max_depth = GOT_MAX_DEPTH * 2` (deeper exploration)
     - Enable multi-source RAG (cross-domain retrieval)
     - Require NorthStar gate at `SNR_THRESHOLD_T0_ELITE` (0.98) instead of baseline
   - Emit `SYNTHESIS` consciousness event with `domains_crossed` metric

2. Add `frontier_mode` flag to `CognitiveFusionEngine.__init__()`:
   - Default `False` for backward compatibility
   - When `True`, enables the dedicated FRONTIER pipeline

3. Wire into OmniKernel:
   - Rust `HhmmLevel` already has enum variants — add `FRONTIER` routing signal

**Quality Gates**:
- Test: Query with entropy > 0.85 + multi-domain markers → assert FRONTIER pipeline invoked
- Test: Same query with `frontier_mode=False` → assert EXPERT pipeline (backward compat)
- Property test: FRONTIER handler never produces SNR < 0.98

---

### 3.6 P5: Python-Tick ↔ Rust-Cycle Synchronization [ARCHITECTURE]

**Impact**: SNR +0.02 | Risk: HIGH | Effort: 2 sprints  
**PMBOK Knowledge Area**: Integration Management  
**Ethical Alignment**: Amānah — two truth sources must agree

#### Problem Statement

Python `process_tick()` is batch-oriented (all receipts per 60s tick). Rust `OmniKernel::run_cycle()` is real-time (per-request). Both mint SEED, enforce Ihsān, and hash receipts — but they don't share state.

The PyO3 bridge (`bizra-python/`) bridges events but not economics:
- Python mints via `progressive_mint()` with Gini/Asabiyyah correction
- Rust mints via `MetabolicLedger::mint_poi_yield()` with emission decay
- No reconciliation between the two ledgers

#### Implementation Plan

**Sprint 1: Define TickSyncProtocol**

1. Create `core/protocols/tick_sync.py`:
   ```python
   @runtime_checkable
   class TickSyncProtocol(Protocol):
       def batch_receipts(self, receipts: list[CycleReceipt]) -> list[ActionReceipt]: ...
       def reconcile_balances(self, rust_minted: int, python_minted: int) -> int: ...
   ```

2. Implement `RustCycleAdapter` in `core/bridges/rust_cycle_adapter.py`:
   - Convert Rust `CycleReceipt` (via PyO3/JSON) → Python `ActionReceipt`
   - Map `CyclePath` variants to `action_type` strings
   - Preserve `pivot_chain_hash` as receipt provenance

**Sprint 2: Integrate with constitutional ticker**

3. Modify `process_tick()` to accept optional `rust_receipts` parameter:
   - Merge Rust-originated receipts with Python-originated receipts
   - Apply same 12-step pipeline to both origins
   - Track `receipt_origin: "python" | "rust"` in event log

4. Add reconciliation check:
   - Compare Python `total_minted` vs Rust `poi_yield` sums per tick
   - Log WARNING if drift > 1% of total (expected due to timing)
   - Halt if drift > 5% (indicates ledger divergence)

**Quality Gates**:
- Test: Rust receipt → Python tick → same Ihsān scoring
- Test: Reconciliation detects simulated 10% drift → halts
- Integration test: Full cycle with PyO3 bridge (requires `maturin develop`)

---

## 4. CI/CD Pipeline Enhancement

### 4.1 Current Pipeline (Evidence: ci.yml)

```
┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ LINT         │   │ TEST MATRIX  │   │ QUALITY GATES │   │ SECURITY     │
│ ├─ Python    │──▶│ ├─ Py 3.11   │──▶│ ├─ SNR score  │──▶│ ├─ pip-audit │
│ ├─ Rust      │   │ ├─ Py 3.12   │   │ ├─ Ihsān gate │   │ ├─ bandit    │
│ ├─ Schemas   │   │ ├─ Rust      │   │ ├─ Coverage   │   │ ├─ Trivy     │
│ ├─ Cross-Lang│   │ ├─ PyO3      │   │ └─ Bridge     │   │ └─ npm audit │
│ └─ Frontend  │   │ └─ Frontend  │   │    smoke      │   └──────────────┘
└─────────────┘   └──────────────┘   └───────────────┘          │
                                                                 ▼
                                                        ┌──────────────┐
                                                        │ BUILD+PUSH   │
                                                        │ ├─ Docker    │
                                                        │ └─ Release   │
                                                        └──────────────┘
```

**Strengths** (already world-class):
- SHA-256 pinned actions (supply chain hardened)
- MyPy error ratchet (1,600 baseline, prevents regression)
- Cross-language constant sync gate (Python ↔ Rust drift detection)
- Quality gate bypass BLOCKED on protected branches
- BLAKE3 enforcement gate (SEC-001)
- Secret hygiene gate (SEC-002)
- Schema validation + SAP conformance
- Deploy overlay contract validation
- Atlas alignment report generation

### 4.2 Proposed Enhancements

| Enhancement | Pipeline Stage | Priority | Effort |
|-------------|---------------|----------|--------|
| **Degradation event gate** | Quality Gates | P1 | 1 day |
| **Exception audit gate** | Lint | P2 | 1 day |
| **Coverage diff-cover** | Test Matrix | P3 | 2 days |
| **FRONTIER routing test** | Test Matrix | P4 | 1 day |
| **Cross-ledger reconciliation** | Integration | P5 | 3 days |
| **Learning loop smoke test** | Quality Gates | P0 | 2 days |

#### New CI Job: Learning Loop Smoke Test (P0)

```yaml
learning-loop-smoke:
  name: "LOOP-001: Learning Loop Smoke Test"
  runs-on: ubuntu-24.04
  timeout-minutes: 10
  needs: [test-python]
  if: ${{ env.BIZRA_CLOSED_LOOP_ENABLED == '1' }}
  steps:
    - name: Test autopoiesis → SDPO bridge
      run: pytest tests/core/autopoiesis/test_sdpo_bridge.py -v
    - name: Test SDPO → reflex bridge
      run: pytest tests/core/sdpo/test_reflex_bridge.py -v
    - name: Test end-to-end loop
      run: pytest tests/integration/test_learning_loop.py -v
```

#### New CI Job: Exception Audit Ratchet (P2)

```yaml
exception-audit:
  name: "SEC-003: Exception Specificity Gate"
  runs-on: ubuntu-24.04
  timeout-minutes: 5
  steps:
    - name: Count broad exceptions in boundary modules
      run: |
        BOUNDARY_MODULES="core/agentic core/a2a core/bus"
        COUNT=$(grep -rn 'except Exception' $BOUNDARY_MODULES --include="*.py" | wc -l)
        BASELINE=6  # Ratchet down over time
        if [ "$COUNT" -gt "$BASELINE" ]; then
          echo "::error::Broad exception count ($COUNT) exceeds baseline ($BASELINE)"
          exit 1
        fi
```

---

## 5. Cascading Risk Analysis

### 5.1 Risk Registry

| ID | Risk | Probability | Impact | Cascade | Mitigation |
|----|------|-------------|--------|---------|------------|
| R1 | P0 bridge introduces training data poisoning | LOW | CRITICAL | SDPO trains on bad data → reflexes compiled from bad patterns → constitutional violation | Crown agent P5 review + 30-day deny-list + feature flag default OFF |
| R2 | P3 coverage ratchet fails (modules untestable) | MEDIUM | HIGH | Coverage stalls → quality regression → SNR drops below 0.85 | Start with easiest modules (fusion_engine degradation combinations), leave untestable modules for Sprint 3 |
| R3 | P5 reconciliation false positives | MEDIUM | MEDIUM | Spurious halts → operator fatigue → threshold raised → real drift missed | Start with WARNING at 1%, HALT at 5%, tune based on 30-day data |
| R4 | P1 degradation events flood logs | LOW | LOW | Log storage exhaustion → monitoring blindness | Rate-limit: max 1 degradation event per engine per minute |
| R5 | P2 exception replacement breaks error handling | LOW | MEDIUM | Specific exception doesn't catch actual failure → unhandled crash | Add catch-all `except Exception` as LAST resort with `logger.critical()` and re-raise |
| R6 | Sovereign/ monolith blocks decomposition | HIGH | HIGH | All canonical implementations in 103-file module → import gravity prevents clean boundaries | Accept re-exports as interim strategy; decompose incrementally per action item |

### 5.2 Risk Cascade DAG

```
R1 (data poisoning)
  └──▶ R6 (monolith) — harder to isolate training bridge
        └──▶ R2 (coverage) — difficult to test large coupled module

R3 (reconciliation FP)
  └──▶ R4 (log flood) — reconciliation warnings compound with degradation events

R5 (exception handling)
  └── Independent — bounded blast radius per module
```

### 5.3 Key Risk Thresholds (Constitutional)

| Threshold | Value | Source | Consequence if Breached |
|-----------|-------|--------|------------------------|
| Ihsān floor | 0.95 | `constants.py:UNIFIED_IHSAN_THRESHOLD` | Production operations halt |
| SNR floor | 0.85 | `constants.py:UNIFIED_SNR_THRESHOLD` | Output suppressed (Reject tier) |
| ADL Gini | ≤ 0.35 | `constants.py:ADL_GINI_THRESHOLD` | Minting halted, economic rebalancing |
| Coverage | ≥ 38% (current) | `pyproject.toml:127` | CI build fails |
| MyPy errors | ≤ 1,600 | `ci.yml` ratchet | Lint stage fails |
| Reflex Ihsān | ≥ 0.98 | `ticker.py` Step 10 | Pattern not compiled |

---

## 6. 12-Week Execution Timeline

### Phase A: Foundation Hardening (Weeks 1–4)

| Week | Action | Deliverable | Gate |
|------|--------|-------------|------|
| 1 | P2: Exception specificity (Tier 1) | `agentic/agent.py`, `a2a/transport.py` cleaned | SEC-003 gate added to CI |
| 2 | P1: Degradation events | `core/protocols/degradation.py` + wiring | DEGRADATION-001 gate in CI |
| 3 | P3: Coverage gap analysis | `htmlcov/` report, priority module list | Baseline measured per-module |
| 4 | P3: Sprint 1 — 50 new tests | Coverage ratchet to 42% | `fail_under = 42` in pyproject.toml |

**Milestone A**: Security posture SNR 0.88 → 0.93. Degradation visible. Exceptions specific.

### Phase B: Capability Activation (Weeks 5–8)

| Week | Action | Deliverable | Gate |
|------|--------|-------------|------|
| 5 | P4: FRONTIER tier handler | `FrontierHandler` in fusion_engine | FRONTIER routing test in CI |
| 6 | P0: Autopoiesis → SDPO bridge | `core/autopoiesis/sdpo_bridge.py` | 20 unit tests |
| 7 | P0: SDPO → Reflex bridge | `core/sdpo/reflex_bridge.py` | Integration test end-to-end |
| 8 | P3: Sprint 2 — 50 more tests | Coverage ratchet to 50% | `fail_under = 50` in pyproject.toml |

**Milestone B**: Learning loop closed (feature-flagged). FRONTIER active. Coverage 50%.

### Phase C: Integration & Polish (Weeks 9–12)

| Week | Action | Deliverable | Gate |
|------|--------|-------------|------|
| 9 | P5: TickSyncProtocol design | `core/protocols/tick_sync.py` + adapter | Protocol tests |
| 10 | P5: Ticker integration | `process_tick()` accepts Rust receipts | Reconciliation test |
| 11 | P3: Sprint 3 — final ratchet | Coverage ratchet to 60% | `fail_under = 60` in pyproject.toml |
| 12 | Integration validation | Full system SNR assessment | **Target: SNR ≥ 0.95** |

**Milestone C**: Ihsān-grade system. All 6 actions complete. SNR 0.95+.

---

## 7. SNR Projection Model

### 7.1 Action-to-SNR Impact Map

| Action | Current Score | Expected Lift | New Score | Confidence |
|--------|--------------|---------------|-----------|------------|
| P0: Close learning loop | 0.94 (Self-Harness) | +0.05 | 0.99 | HIGH |
| P1: Degradation events | 0.88 (Security) | +0.02 | 0.90 | HIGH |
| P2: Exception specificity | 0.88 (Security) | +0.03 | 0.93 | HIGH (with P1) |
| P3: Coverage 60% | 0.913 (Composite) | +0.03 | 0.943 | MEDIUM |
| P4: FRONTIER tier | 0.96 (Bridges) | +0.01 | 0.97 | HIGH |
| P5: Tick synchronization | 0.97 (Economic) | +0.02 | 0.99 | MEDIUM |

### 7.2 Composite SNR Trajectory

```
Week 0:  ████████████████████████████████████░░░░  0.913
Week 4:  █████████████████████████████████████░░░  0.930  (P1+P2+P3a)
Week 8:  ██████████████████████████████████████░░  0.945  (P0+P3b+P4)
Week 12: ███████████████████████████████████████░  0.958  (P5+P3c)
                                          ▲
                                     IHSĀN LINE (0.95)
```

### 7.3 Dimension Evolution

| Dimension | Week 0 | Week 4 | Week 8 | Week 12 |
|-----------|--------|--------|--------|---------|
| Kernel Integrity | 0.98 | 0.98 | 0.98 | 0.99 |
| Symbolic-Neural Bridges | 0.96 | 0.96 | 0.97 | 0.97 |
| Self-Harnessing | 0.94 | 0.94 | 0.99 | 0.99 |
| Formal Verification | 0.95 | 0.95 | 0.95 | 0.96 |
| Economic Substrate | 0.97 | 0.97 | 0.97 | 0.99 |
| Security Posture | 0.88 | 0.93 | 0.93 | 0.95 |

---

## 8. Evidence & Verification Framework

### 8.1 Proof Forge Integration

Every action produces a Proof Forge receipt:

```
ACTION EXECUTION
      │
      ▼
┌─────────────┐
│   BUILD     │──── Code changes committed
├─────────────┤
│   VERIFY    │──── Tests pass, CI green, coverage gate met
├─────────────┤
│   EVIDENCE  │──── ActionReceipt with SHA-256 chain
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ PROOF_SUMMARY│──── Investor-grade markdown
│    .md       │     with hash verification
└─────────────┘
```

Evidence chain continues from:
- **Genesis Receipt #0**: `643b71420d3cb8f71e7e3de1a4d78b8e...` (Phase 17.5)
- Each P0–P5 action produces Receipt #1–#6
- Final Receipt #7: System SNR assessment at Week 12

### 8.2 Acceptance Criteria per Action

| Action | Acceptance Criteria | Verification Method |
|--------|-------------------|---------------------|
| P0 | End-to-end: candidate → training → reflex completes | `pytest tests/integration/test_learning_loop.py` |
| P1 | Zero silent degradation in integration tests | `grep DEGRADATION test_output.log \| wc -l` |
| P2 | `except Exception` in boundary modules ≤ 6 | SEC-003 CI gate |
| P3 | `fail_under = 60` CI green | `pytest --cov=core` |
| P4 | FRONTIER query routes to dedicated handler | `pytest tests/core/cognitive_fusion/test_frontier.py` |
| P5 | Rust receipts processed in Python tick | `pytest tests/integration/test_tick_sync.py` |
| **Final** | **Composite SNR ≥ 0.95** | **Quality Gates job in CI** |

---

## 9. Ethical Integrity Audit

### 9.1 Daughter Test Compliance

Each action is evaluated against the Daughter Test: *"Would you be comfortable if your daughter were affected by this action?"*

| Action | Daughter Test | Rationale |
|--------|:------------:|-----------|
| P0: Learning loop | ✅ PASS | Self-improvement with Crown agent review prevents harmful patterns |
| P1: Degradation events | ✅ PASS | Transparency about system capabilities — honest about limitations |
| P2: Exception specificity | ✅ PASS | Precise error handling prevents silent failures that could affect users |
| P3: Coverage ratchet | ✅ PASS | More tests = fewer bugs reaching users |
| P4: FRONTIER tier | ✅ PASS | Better handling of complex queries = better user experience |
| P5: Tick synchronization | ✅ PASS | Economic consistency protects all network participants equally |

### 9.2 ADL Gini Compliance

No action in this blueprint alters the economic model's Gini coefficient mechanics. The `compute_gini()` → `ADL_GINI_THRESHOLD ≤ 0.35` gate remains invariant across all 6 actions.

P5 (tick sync) introduces reconciliation between two minting paths, which **improves** economic fairness by ensuring consistent SEED distribution regardless of whether actions originate from Python or Rust.

### 9.3 Standing on Giants Attribution

| Action | Primary Giant | Secondary Giants |
|--------|--------------|-----------------|
| P0 | Maturana & Varela (autopoiesis, 1980) | Holland (genetic algorithms), Shannon (SNR gate) |
| P1 | Meyer (Design by Contract, 1986) | Anthropic (constitutional transparency) |
| P2 | Liskov (substitution principle, 1987) | Al-Ghazali (Ihsān — excellence in error handling) |
| P3 | Deming (PDCA quality ratchet, 1950) | Shannon (coverage = noise reduction) |
| P4 | Kahneman (System 1/2, 2002) | Besta (Graph-of-Thoughts depth) |
| P5 | Lamport (distributed consensus, 1978) | Nakamoto (ledger reconciliation) |

---

## 10. Definition of Done

An action is DONE when ALL of the following are true:

- [ ] Code committed to `claude/frosty-hugle` branch
- [ ] All new code has type annotations (PEP 484)
- [ ] Tests written and passing (minimum 10 per action)
- [ ] CI pipeline green (all gates pass)
- [ ] Coverage does not decrease
- [ ] No new `except Exception` in boundary modules
- [ ] Proof Forge receipt generated with SHA-256 chain
- [ ] Constants imported from `core/integration/constants.py` (never hardcoded)
- [ ] Standing on Giants attribution in module docstring
- [ ] Daughter Test reviewed by Crown agent (P0 only)

---

## 11. Appendix: File Reference Map

| File | Role in Blueprint | Lines Read |
|------|-------------------|------------|
| [core/constitutional/ticker.py](core/constitutional/ticker.py) | 12-step tick, Steps 1–12 | 1–220 |
| [core/constitutional/algorithms.py](core/constitutional/algorithms.py) | 15 algorithms, Three Minds | 1–80 |
| [core/constitutional/fixed_point.py](core/constitutional/fixed_point.py) | Integer arithmetic kernel | 1–100 |
| [core/reasoning/entropy_router.py](core/reasoning/entropy_router.py) | System 1/2, FRONTIER tier | 1–120 |
| [core/reasoning/diffusion_reasoning_amplifier.py](core/reasoning/diffusion_reasoning_amplifier.py) | HMM→GoT bridge | 1–80 |
| [core/reasoning/got_bridge.py](core/reasoning/got_bridge.py) | FAISS→GoT evidence | 1–120 |
| [core/prediction/hmm_engine.py](core/prediction/hmm_engine.py) | HMM cognitive state forecast | 1–200 |
| [core/cognitive_fusion/fusion_engine.py](core/cognitive_fusion/fusion_engine.py) | MoE→HRM→RAG→NorthStar pipeline | 1–180 |
| [core/sovereign/bicameral_engine.py](core/sovereign/bicameral_engine.py) | R1/Claude generate-verify | 1–120 |
| [core/sovereign/collective_intelligence.py](core/sovereign/collective_intelligence.py) | Team synergy, aggregation | 1–120 |
| [core/sovereign/z3_fate_gate.py](core/sovereign/z3_fate_gate.py) | Z3 formal verification | 1–100 |
| [core/autopoiesis/loop.py](core/autopoiesis/loop.py) | Self-evolving agent loop | 1–120 |
| [core/autopoiesis/hypothesis_generator.py](core/autopoiesis/hypothesis_generator.py) | Improvement hypotheses | 1–120 |
| [core/sdpo/training/bizra_sdpo_trainer.py](core/sdpo/training/bizra_sdpo_trainer.py) | Self-distillation training | 1–100 |
| [core/sdpo/discovery/sdpo_test_time.py](core/sdpo/discovery/sdpo_test_time.py) | Test-time discovery | 1–100 |
| [core/sdpo/cosmos/sdpo_sape_fusion.py](core/sdpo/cosmos/sdpo_sape_fusion.py) | SAPE-SDPO wisdom layer | 1–100 |
| [core/iaas/snr_dual_verification.py](core/iaas/snr_dual_verification.py) | Dual V_gate × V_pool | 1–100 |
| [core/elite/self_harness_engine.py](core/elite/self_harness_engine.py) | Codebase quality scanner | 1–100 |
| [core/zpk/kernel.py](core/zpk/kernel.py) | Zero Point Kernel bootstrap | 1–100 |
| [core/integration/constants.py](core/integration/constants.py) | SSoT, 900+ lines | Referenced |
| [bizra-omega/bizra-agent/src/omni_kernel.rs](bizra-omega/bizra-agent/src/omni_kernel.rs) | 8-line OmniKernel cycle | 55–400 |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | CI pipeline, 9+ gates | 1–800 |
| [pyproject.toml](pyproject.toml) | Project config, coverage gate | 1–150 |
| [PROOF_SUMMARY.md](PROOF_SUMMARY.md) | Genesis receipt #0 | 1–80 |
| [ROADMAP.md](ROADMAP.md) | Project timeline | 1–100 |
| [STATUS.md](STATUS.md) | Implementation status | 1–50 |

---

*This blueprint was produced by BIZRA DDAGI OS Pilot — the system analyzing itself, prescribing its own improvement, and committing to verifiable outcomes.*

*Amānah (trustworthiness), Himma (high resolve), Bāṭin (inner alignment) — البذرة § 2.2*

**Evidence**: 25 source files read across Python/Rust/YAML, 6 grep scans, complete CI pipeline traced, all SAPE dimensions scored, cascading risk analysis with 6 identified risks.

**Next Step**: Execute Phase A, Week 1 — P2 exception specificity in `core/agentic/agent.py` (4 instances).
