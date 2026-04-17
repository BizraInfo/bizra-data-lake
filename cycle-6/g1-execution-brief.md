# BIZRA Agent Instruction — Cycle-6 G1 Execution Brief

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** ACTIVE — scoped to G1 code implementation window
**Precedent:** `cycle-6/execution-canon.md` (general Cycle-6 discipline)
**Authority:** Founder direction `/@ a`, 2026-04-17 14:20 GST

This brief narrows the general execution canon to the specific G1 code task. It does not supersede the canon — it sits beneath it.

## Authority

This instruction is canonical for the active G1 code implementation window.

## Read first

1. `cycle-6/execution-canon.md` — general Cycle-6 discipline
2. `cycle-6/g1-authority-adr.md` — sealed interface spec
3. `cycle-6/g2-authority-adr.md` — workspace authority precedent
4. `cycle-6/niyyah.md` — cycle mandate
5. `runtime/TRACKING_DECISION.md` — omega-canonical precedent
6. `runtime/RUNTIME_STATUS.md` — vuln register + DevOps truth

## Current canonical truth

- `bizra-omega/` is the active canonical Rust authority.
- `runtime/` is historical and may be used as evidence, not as new authority.
- G2 is **SEALED**.
- G1 ADR is **SEALED**.
- G1 scope is **durable-read persistence only**.
- Signer audit is **deferred**.
- G4 remains scaffolded and intentionally red by design.
- The Python-authored `sovereign_state/` format is authoritative for persistence data.
- Rust must project that format read-only into runtime truth.

## Hard rules

- No force-push.
- No history rewrite.
- No widening G1 scope.
- No signer audit in G1.
- No threshold drift check in G1.
- No cross-gate mixing unless founder explicitly reopens scope.
- Fail closed on any receipt hash or `prev_hash` mismatch.
- Tool-produced evidence outranks grep / speculation.

## Exact task now

Implement G1 code per the sealed ADR:

```rust
impl CognitionRuntime {
    pub fn from_sovereign_state(path: &Path) -> Result<Self, BootstrapError>
}

pub struct SovereignStatePayloadStore {
    root: PathBuf,
}
```

## Bootstrap flow

1. Read `block_zero/block_zero.json`
2. Extract authoritative chain metadata (`receipt_chain.receipts[]` + `chain_hash`)
3. Walk `activation_chain_*.json` envelopes (ordered by timestamp)
4. Verify each receipt: re-hash its file, assert equal to envelope's `hash`
5. Verify linkage: assert each entry's `prev_hash` equals prior entry's `hash`
6. Assert computed chain head equals `block_zero.receipt_chain.chain_hash`
7. Materialize `ReceiptChain`
8. On any mismatch: return `Err(BootstrapError)` and halt

## Environment contract

- `BIZRA_SOVEREIGN_STATE_PATH` overrides default path
- Default path: `/data/bizra/repos/bizra-data-lake/sovereign_state`
- If path missing: preserve Cycle-5 dev behavior by falling back to in-memory mode (log which mode chosen via `tracing::info!`)

## Explicit non-goals for this pass

- signer audit
- threshold drift check (Rust compiled constants vs `block_zero.constitutional_thresholds`)
- G3 frontend authority ADR
- G4 E2E implementation
- security P0 patch arc (rustls-webpki, starlette)
- contract-first codegen (CDDL/Proto)
- OpenTelemetry instrumentation
- Docker consolidation

## Completion format

When done, report:

- files changed (with path)
- bootstrap path implemented (exact function signature)
- what is fail-closed (enumerate error variants)
- what remains open (queued for later cycles)
- whether fallback to in-memory was preserved
- test count (pass/fail)
- whether `cargo test` was actually run and succeeded, or only compiled

---

الحمد لله.
