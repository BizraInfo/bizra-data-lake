# Symbolic-Neural Bridge Audit — BIZRA v0.1

**Scope:** Trace the bridge `intent → plan → symbolic constraints → gate → action → receipt → review / canon / future learning`. Document where neural output meets a deterministic proof boundary.

---

## 1. Pipeline map

```
┌────────┐     ┌────────┐     ┌────────────────────┐     ┌────────┐
│ intent │ ──▶ │  plan  │ ──▶ │ symbolic constraint│ ──▶ │  gate  │
└────────┘     └────────┘     └────────────────────┘     └───┬────┘
                                                              │
                                ┌─────────────────────────────┴──┐
                                │                                │
                                ▼                                ▼
                        ┌────────────┐                    ┌──────────┐
                        │   action   │ ─── emits ────────▶│ receipt  │
                        └────────────┘                    └────┬─────┘
                                                                │
                                                                ▼
                                                        ┌────────────┐
                                                        │   review   │
                                                        └─────┬──────┘
                                                               │
                                                               ▼
                                                        ┌────────────┐
                                                        │ canon pack │──▶ (ingestion gate) ──▶ runtime canon
                                                        └────────────┘
```

## 2. Stage-by-stage evidence

### 2.1 Intent → plan

- **Intent capture:** `POST /v1/mission` on the node gateway (`services/node_gateway/app/routers.py`).
- **Plan generation:** `bizra-omega/bizra-agent` OmniKernel cognitive cycle.
- **Neural layer:** local LLMs via tiered inference (LM Studio → Ollama → cloud).
- **Bridge discipline:** plan is a *proposal*, not an action. It does not mutate state directly.

### 2.2 Plan → symbolic constraints

- **Mission state machine** (`bizra-mission`) is the symbolic layer. 14 states, `advance!` macro enforces legal transitions, illegal → `Err(TransitionError)`.
- **Constitutional thresholds** (`core/integration/constants.py`) — Ihsan, SNR, ADL Gini — are symbolic constraints the plan must satisfy.

### 2.3 Constraints → gate

- **FATE gates** — Z3-backed formal verification.
- **Conservative fallback:** `_conservative_fallback_check()` is **stricter than Z3** — fail-closed when Z3 is unavailable.
- **Ihsan gate** — 0.95 production threshold applied before an effect is allowed to ship.

### 2.4 Gate → action

- Only if gate returns allow does the action execute.
- Action emits a **canonical receipt** (BLAKE3-chained, Ed25519-signed, full-body).
- Receipt includes `previous_receipt_hash` → chain continuity.

### 2.5 Receipt → review

- Receipts are the unit of review. Each is signed and chained.
- Review surfaces (cockpit P2, Dema trust surface) read the chain head through `/v1/chain`.

### 2.6 Review → canon pack

- Selected receipts (or selected chat content as in this session's Cognitive Foundry cycle) flow into review workbooks.
- Human review → `promote.py` → canon pack.
- **Canon pack is *candidate for* canon**, never canon itself. Explicit `non_promotion_tool: true` in the manifest.

### 2.7 Canon pack → (gate) → runtime canon

- **Canon Store Ingestion Gate** is the required boundary.
- Gate does not yet exist. No auto-ingestion. No drift.
- MEMORY.md / constants.py / topology_canon.rs cannot be written without explicit human sign-off.

## 3. Where neural meets symbolic — the "proof boundary"

| Neural output | Symbolic check | Deterministic proof |
|---|---|---|
| LLM plan | Mission state-machine legality | `advance!` macro, `TransitionError` |
| LLM content | Ihsan score >= threshold | `core/integration/constants.py` |
| LLM reasoning | FATE gate | Z3 or `_conservative_fallback_check` |
| Agent action | Receipt emission | `canonical_receipt.rs` + Ed25519 sign + BLAKE3 chain |
| Promotion candidate | Human review + `promote.py` | content_hash (deterministic) + issuance_hash (per-event) |
| Canon ingestion | Canon Store Ingestion Gate | (to be defined — required boundary) |

## 4. Integrity properties

1. **Neural cannot bypass symbolic.** Every action goes through a gate. Gate is fail-closed.
2. **Receipts are the unit of truth.** Chain integrity is BLAKE3; signatures are Ed25519.
3. **Canon is sealed behind the ingestion gate.** No auto-ingestion means no automated canon drift.
4. **Content-hash determinism** (v0.2.0 split-hash in `promote.py`) — the same reviewed content always hashes the same; promotion events are separately identified.

## 5. Risks on the bridge

- **Panic surface on hot-paths** (`.unwrap()` ×806) — if a panic fires between gate-pass and receipt-emit, invariant breaks silently.
- **Z3 unavailability** is handled (`_conservative_fallback_check` is strict). ✅
- **Missing Canon Store Ingestion Gate** means the last hop of the bridge is a *policy* gate, not yet a code gate. Enforcement is currently human discipline.

## 6. Recommendations

| # | Action | Effort |
|---|---|---|
| SN1 | Hot-path `.unwrap()` audit in gate / receipt emission | M |
| SN2 | Canon Store Ingestion Gate spec (separate lane, typed-auth) | M-L |
| SN3 | Architecture diagram of the symbolic-neural bridge | S |
| SN4 | Unit tests that explicitly verify "panic on hot path ⇒ no receipt emitted" (fault-injection) | M |

---

**Bridge verdict:** ✅ Architecturally sound, with one enforcement gap (ingestion gate is policy, not code) and one tech-debt risk (panic surface). The structure is rare in consumer AI — it is a golden gem worth protecting.
