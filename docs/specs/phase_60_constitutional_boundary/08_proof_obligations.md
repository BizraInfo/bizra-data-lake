# Step 8: Proof Obligations Matrix (Truth Layer)

## Standing on Giants: Shannon (SNR contracts) | Lamport (safety invariants) | Dijkstra (fail-closed) | Castro-Liskov (Byzantine bounds) | Al-Ghazali (Ihsan floor)

**Date:** 2026-03-03  
**Intent:** Convert theorem-style claims into evidence-backed obligations.  
**Rule:** `Signal = claim linked to code + test + runtime evidence`.  
**Rule:** `Noise = elegant statement without verifiable implementation path`.

---

## 1) Graph-of-Thoughts (Implementation Reality)

```
Observe
  -> Classify macro-state (gateway HHMM)
  -> Check reflex cache (Redis/SQLite)
  -> If miss: run mission bridge (core sovereign pipeline)
  -> Score and package plan (SNR/PoI fields)
  -> Return + cache learned steps
  -> Health/URP services remain live via shared checks
```

This is the current executable thought flow for Node0 runtime.

---

## 2) Theorem-to-Evidence Matrix

| Claim | Status | Evidence in Code | Closure Obligation |
|---|---|---|---|
| **T2.1 SNR Monotonicity** `E[SNR(t+1)] >= SNR(t)` | **Partial** | Canonical normalization exists in `core/snr_protocol.py` and gateway `app/node/snr.py`; no online monotonic learner proof path | Add longitudinal telemetry + hypothesis/stat tests for rolling SNR non-decrease under fixed task mix |
| **T2.2 Reflex Convergence** `rho(t) -> rho*` | **Partial** | Reflex cache exists and miss->hit loop works (`app/node/reflex_cache.py`, `app/routers.py`) | Implement explicit precipitation policy (`>=3` successful, high-Ihsan runs) instead of immediate first-hit cache promotion |
| **T2.3 Constitutional Safety** `PoI<tau => no actuation` | **Partial** | Gate logic exists in core gate stack; gateway `/v1/plan` is planning endpoint, not hard actuation barrier | Bind PoI/Ihsan threshold to actual actuation adapter and add negative conformance test proving zero actuation below threshold |
| **T2.4 Byzantine Resilience** `f < N/3` | **Not Implemented** | Consensus service currently stores submissions (`services/urp_consensus/app/routers.py`) with admin auth, not quorum protocol | Implement signature verification, peer identity, quorum (`2f+1`), and conflict resolution tests |
| **T2.5 Forest Super-linear Scaling** | **Conjecture** | No production multi-node benchmark harness proving scaling law | Add controlled N-node benchmark harness and publish measured scaling curve + confidence intervals |
| **L3.1 HHMM Convergence** | **Not Implemented in gateway path** | Gateway HHMM is deterministic rule classifier (`services/node_gateway/app/node/hhmm.py`) | Replace/augment with learned HHMM + persisted parameters + convergence tests |
| **L3.2 Crown Entropy Stability** | **Partial** | Verification service stores crown signatures; no end-to-end entropy gate in gateway actuation path | Implement entropy baseline check in execution path and adversarial UI spoof tests |
| **L3.3 Gossip Propagation O(log N)** | **Not Implemented** | No active gossip protocol in prod artifact services | Add gossip transport + propagation measurement harness |

---

## 3) What Is Proven Today (High-SNR Truth)

These are validated as of 2026-03-03 against live stack + tests:

1. **Runtime completeness (single-node URP stack):** all services boot and expose healthy endpoints.  
2. **Authenticated gateway path:** `/v1/plan` and reflex mutation require API key.  
3. **Reflex lifecycle exists:** miss path returns mission-derived steps; hit path returns cached steps.  
4. **Mission bridge is active:** gateway miss can execute sovereign mission pipeline and emit evidence-linked step output.  
5. **SNR normalization is canonical in both planes:** core and gateway use `snr/(1+snr)` bounded mapping.  
6. **Ihsan dimensional extension is real in core gate path:** `auditability`, `robustness` are wired in canonical weights.

---

## 4) Evidence Index (Primary References)

- `core/snr_protocol.py`
- `core/proof_engine/ihsan_gate.py`
- `core/integration/constants.py`
- `.tmp_prod_artifacts_v2/services/node_gateway/app/routers.py`
- `.tmp_prod_artifacts_v2/services/node_gateway/app/node/hhmm.py`
- `.tmp_prod_artifacts_v2/services/node_gateway/app/node/reflex_cache.py`
- `.tmp_prod_artifacts_v2/services/node_gateway/app/node/mission_bridge.py`
- `.tmp_prod_artifacts_v2/services/urp_consensus/app/routers.py`
- `.tmp_prod_artifacts_v2/contracts/proto/poi.proto`
- `tests/core/test_node_gateway_plan_bridge.py`
- `tests/integration/test_sovereignty_pipeline.py`
- `tests/core/proof_engine/test_snr_v1_ihsan_gate.py`

---

## 5) Elite Next Step (Implementation, Not Narrative)

### Sprint A: Convert Partial Proofs into Hard Guarantees

1. **Reflex precipitation contract**
   - Add per-macro-state success counters.
   - Promote to cache only after `n >= 3` successful high-quality outcomes.
   - Tests: first two successes = miss path, third = hit path.

2. **Actuation constitutional bind**
   - Enforce `PoI/Ihsan` thresholds on execution boundary (not planning only).
   - Tests: below-threshold plans cannot execute; above-threshold plans execute.

3. **Byzantine-ready PoI consensus**
   - Verify signatures + node identities.
   - Introduce quorum vote outcome model and rejection on insufficient honest quorum.
   - Tests: malicious submissions cannot force acceptance.

### Sprint A Acceptance Criteria

- `0` unauthorized actuation events below threshold.
- `100%` rejection of invalid-signature attestations in consensus API tests.
- Reflex promotion behavior deterministic and test-covered.
- Existing sovereign integration tests remain green.

---

## 6) Boundary Rule for Future Proof Text

A theorem may be marked **Proven** only when all three are present:

1. **Executable implementation** in production path.  
2. **Automated test evidence** covering nominal + adversarial cases.  
3. **Runtime trace evidence** reproducible from Node0 runbook.

Otherwise mark as **Partial** or **Conjecture**.

This keeps the architecture honest and keeps SNR high.

