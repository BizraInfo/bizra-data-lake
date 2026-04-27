# Error Handling Audit — BIZRA v0.1

**Scope:** fallbacks, circuit breakers, fail-open / fail-closed, restart persistence, receipt emission on failure, health / readiness side effects, rare failure paths.

---

## 1. Pattern-level findings

| Rule | Count | Severity |
|---|---:|---|
| `RS_UNWRAP` (Rust `.unwrap()`) | 806 | LOW (pattern) / MEDIUM (hot-path) |
| `PY_BROAD_EXCEPT` (Python `except Exception`) | 126 | LOW |
| `RS_EXPECT` (Rust `.expect("...")`) | 41 | LOW |
| `PY_EVAL_EXEC` (Python `eval` / `exec`) | 12 | MEDIUM |
| `PY_URL_FETCH` (Python URL fetch sites) | 7 | LOW / SSRF-review |
| `PY_TODO` (Python TODO/FIXME) | 4 | LOW |
| `RS_PANIC` (Rust `panic!(...)`) | 2 | MEDIUM |
| `PY_SHELL_TRUE` | 1 | HIGH |
| `RS_TODO` (Rust TODO/FIXME) | 1 | LOW |

## 2. Fallbacks (observable)

- **Inference tier fallback** — LM Studio → Ollama → cloud (declared in `bizra_config.py`). Fails forward; no circuit breaker visible in this audit.
- **Mission state machine** — `Degraded`, `Failed`, `TimedOut`, `AwaitingReconciliation` are explicit terminal / holding states in `bizra-mission`. Illegal transitions return `Err(TransitionError)` — **fail-closed by design.** ✅
- **Receipt emission on failure** — Mission state `Failed` still emits a signed receipt. `advance!` macro in `bizra-mission` enforces this.

## 3. Circuit breakers

Not observed in this audit at the LLM-inference layer. Recommend adding:
- Fail-fast when LM Studio health endpoint is down.
- Back-off to Ollama only if LM Studio has been down for N seconds (configurable).
- Cloud fallback gated by explicit user opt-in per request.

## 4. Restart persistence

- **Reflex persistence across restart** — `bizra-omega/bizra-agent/src/persistence.rs` content-addressed file store with BLAKE3 integrity manifest. Restore on boot, save on compilation, snapshot on shutdown. ✅
- **Receipt chain survives restart** — chain head read from storage; no in-memory-only state.

## 5. Receipt emission on failure

✅ **Explicit invariant.** The `advance!` macro's `fail-closed` behavior means failing states still emit receipts with the failure reason. This is a structural property, not a best-effort convention.

**Risk:** if a panic fires *before* the state-machine captures a state, receipt emission is bypassed. The 806 `.unwrap()` sites are the most visible locus of this risk; the current generated artifact also reports 2 `panic!` sites and 41 `.expect()` sites. Hot-path audit required.

## 6. Health / readiness side effects

Not observable from pattern scan. Recommend:
- Health endpoint (`GET /health`) must be a **pure read** — no file writes, no state mutation.
- Readiness endpoint should include receipt-chain reachability check but not mutate anything.

**Known audit-log contamination pattern** (memory `project_cargo_test_audit_contamination.md`): `cargo test --workspace` appends to `action_receipts.jsonl`. Must be snapshotted + restored around CI test runs. **This is effectively a health-endpoint side-effect class** — it demonstrates that the discipline has slipped before and must be defended.

## 7. Rare-failure paths (surfaced)

- **URP offline reconciliation** — explicit state in the mission machine. Rare path but first-class. ✅
- **Panic during receipt write** — 806 unwrap sites plus 2 explicit `panic!` sites; any panicking hot path can bypass the receipt invariant.
- **Z3 solver unavailability** (build-time dep) — FATE gates have a `_conservative_fallback_check` (per CLAUDE.md). Stricter than Z3 — fail-closed. ✅
- **Gateway unreachable during trust-surface read** — Dema returns honest 503 (per memory `project_node0_closure_row6_trust_surface.md`). **No shadow state.** ✅
- **Cargo test side effects** — known; workaround documented.

## 8. Fail-open vs fail-closed

| Surface | Default |
|---|---|
| FATE gate Z3 path | fail-closed (`_conservative_fallback_check`) ✅ |
| Mission state transitions | fail-closed (`TransitionError`) ✅ |
| Receipt signing | fail-closed (no receipt → no state change) ✅ |
| LLM inference | fail-forward (tier fallback) — acceptable |
| Trust surface on gateway down | fail-honest (503) ✅ |

**Consistency check:** all safety-critical gates are fail-closed. LLM inference is the only fail-forward path, and that's the right choice (a missing answer is not a safety event).

## 9. Debts (ranked)

| # | Debt | Severity | Action |
|---|---|---|---|
| ED1 | Hot-path `.unwrap()` audit in receipt-emitting crates | HIGH | Replace panics on hot paths with `Result` + explicit receipt-on-error |
| ED2 | Circuit breaker around LM Studio fallback | MEDIUM | Document SLO + implement back-off |
| ED3 | Health-endpoint side-effect guard | MEDIUM | Unit-test that `/health` is a pure read |
| ED4 | `cargo test` audit-log contamination guard as CI gate | MEDIUM | Snapshot sha256 + diff in CI |
| ED5 | 2 `panic!` sites audit | LOW | Replace with structured errors where applicable |
| ED6 | 126 Python broad-except sweep | LOW | Tighten exception types |

---

**Bottom line:** the fail-closed discipline is architecturally correct at the gate / state-machine layer. The main visible risk is *panic surface* — 806 unwrap sites, 41 expect sites, and 2 panic sites — which is tech debt that touches claim discipline when publishing hero-level receipt claims.
