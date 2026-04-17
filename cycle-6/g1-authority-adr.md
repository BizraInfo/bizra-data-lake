# Cycle-6 — Gate G1 (Persistence Authority) ADR

بسم الله الرحمن الرحيم

**Cycle:** 6 (Persistence + Authority Unification)
**Gate:** G1 — Persistence arc
**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** DECIDED (narrow scope per founder direction `/@ no`, 14:07 GST)
**Precedes:** G1 code work
**Supersedes:** the three-option trichotomy (Python-writes / Rust-writes / shared-format) proposed during Cycle-6 SAPE analysis — replaced by a single durable-read decision

---

## Context

`cycle-6/niyyah.md` §G1 requires the Rust gateway to bridge the gap between its ephemeral `InMemoryPayloadStore` and the Python stack's real persistent `sovereign_state/`. Verification: seal receipt X → restart gateway → `dema chain --since today` still returns X.

The founder has gated G1 scope to **durable-read persistence only** (`/@ no`, 2026-04-17 14:07 GST). Signer audit, writeback authority, and shared-format design are explicitly out of scope unless the sovereign format proves signer-inseparability during implementation.

## Evidence (tool-verified, not speculative)

Direct inspection of `sovereign_state/` and `bizra-omega/bizra-cognition/src/runtime.rs`:

### Format authority (what lives in sovereign_state/)

| Artifact | Shape | Role |
|---|---|---|
| `sovereign_state/block_zero/block_zero.json` | JSON, `receipt_chain.receipts[]` = 10 hashes + `chain_hash` aggregate + `constitutional_thresholds` | **Chain root of trust**, seal values for Ihsan/SNR/ADL/RIBA/zakat |
| `sovereign_state/genesis/*.json` | JSON per artifact: `genesis_receipt`, `hardware`, `identity`, `manifest`, `pat_manifest`, `sat_manifest`, `urp_pledge` | Genealogical anchors |
| `sovereign_state/receipts/*_<ISO-ts>.json` | Per-event JSON receipts | Individual events |
| `sovereign_state/receipts/activation_chain_<ISO-ts>.json` | JSON with `chain[]`: each entry `{file, event, hash, prev_hash}` | **Chain envelope** — orders receipts, asserts hash-chain |

**Chain structure:** Each receipt file holds an event. A chain-envelope JSON orders receipts by hash + `prev_hash` (genesis is 64 zero-hex). Hash-chain verification is straightforward: walk the envelope, re-hash each referenced file, assert match against envelope's `hash` and `prev_hash` fields.

### Current Rust constructor surface

`bizra-omega/bizra-cognition/src/runtime.rs:249` —

```rust
impl CognitionRuntime {
    pub fn new(graph: ThoughtGraph, chain: ReceiptChain, ctx: AgentCtx) -> Self { ... }
}
```

And `chain` is built via `ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()))` at every call site. This is the expansion point.

## Decision

### Authority

**Python is authoritative writer; Rust is read-only projection.**

- All new receipts are written by the Python stack (core/sovereign/atomic_io.py + related) into `sovereign_state/receipts/` and aggregated in chain envelopes.
- The Rust `bizra-cognition-gateway` reads the chain on startup and after configured TTL / inotify signal, and serves `dema chain` / gateway `/chain` endpoints from this projection.
- Rust never mutates `sovereign_state/`.

Rationale: aligns with G2 (`bizra-omega/` canonical for Rust surfaces but Python writer is institutional and pre-dates omega); preserves CLAIM_MUST_BIND (one canonical writer); eliminates NO_SHADOW_STATE of two writers; minimizes Cycle-6 scope.

### Interface specification

New associated constructor on `CognitionRuntime`:

```rust
impl CognitionRuntime {
    /// Bootstrap a CognitionRuntime from the Python-authoritative sovereign_state/ on disk.
    ///
    /// Read-only: never writes. Verifies hash-chain integrity end-to-end before returning.
    /// Fails closed (BootstrapError) on: missing block_zero, corrupted chain envelope,
    /// hash mismatch, or unreadable receipt file.
    pub fn from_sovereign_state(path: &Path) -> Result<Self, BootstrapError> {
        // 1. Parse block_zero/block_zero.json → canonical chain_hash + ordered receipts[]
        // 2. Parse genesis/*.json → genealogical anchors (materialize into ctx)
        // 3. Scan receipts/activation_chain_*.json → chain envelopes, order by timestamp
        // 4. For each chain envelope entry: read referenced receipt file, verify hash,
        //    verify prev_hash linkage against prior entry
        // 5. Assert computed chain_hash matches block_zero.receipt_chain.chain_hash
        // 6. Materialize ReceiptChain::from_verified_entries(entries, ctx)
        // 7. Construct empty ThoughtGraph (or future: hydrate from rdve_hypotheses/)
        // 8. Return Self { graph, chain, ctx }
    }
}
```

New payload store implementation:

```rust
pub struct SovereignStatePayloadStore {
    root: PathBuf,   // sovereign_state/ path
    // read-only, no write methods
}
```

`SovereignStatePayloadStore` implements a subset of the existing `PayloadStore` trait — only the read half. Write attempts return `Err(ReadOnlyProjection)`.

### Bootstrap wiring

`bizra-omega/bizra-cognition-gateway/src/main.rs` gains an env var:

```
BIZRA_SOVEREIGN_STATE_PATH   (default: /data/bizra/repos/bizra-data-lake/sovereign_state)
```

On startup:
1. If path exists and is valid: `CognitionRuntime::from_sovereign_state(path)?`
2. If path missing or env var unset: fall back to current in-memory bootstrap (preserves Cycle-5 ephemeral mode for local dev without side effects)
3. Log which mode was chosen (tracing `info!` span)

### Verification (matches niyyah §G1 line-for-line)

1. Start Python stack, seal a receipt X via `POST /api/mission` or equivalent
2. Stop gateway (Rust); `sovereign_state/receipts/` retains X's file + updated chain envelope
3. Restart gateway (Rust) pointing at `sovereign_state/`
4. `dema chain --since today` returns X

## Consequences

### Enables

- Gateway restart no longer loses chain history — first real durable loop on the Rust side
- NO_SHADOW_STATE eliminated for the persistence surface: one writer (Python), one reader (Rust), one source (`sovereign_state/`)
- G4 (E2E polyglot) gains a meaningful durable assertion — `dema chain` reads real state, not ephemeral
- Gateway CI tests can load from a test fixture `sovereign_state/` without needing a Python process

### Accepts

- Rust cannot append receipts this cycle. If a use case requires Rust-originated receipts, it must go through Python. Cycle-6 does not ship such use cases.
- Staleness window: if Python writes between two Rust reads, Rust serves the older view until re-read. Acceptable for Cycle-6 (no real-time-sync requirement in niyyah).
- Format is JSON-pinned: if Python changes receipt schema, Rust reader must be updated in lockstep. This is a known cost of format-authority-on-writer.

### Out of scope (explicitly deferred)

| Item | Deferred to |
|---|---|
| Signer audit: verify receipt signatures against `sovereign_state/key_registry.json` | Cycle-6.5b or Cycle-7 — unless implementation proves signer-inseparability, in which case founder re-gates |
| Rust-writes-back capability | Cycle-7+, post-G4 |
| Contract-first CDDL/Proto schema for receipt format | Cycle-7+ (polyglot blueprint §1) |
| Constitutional-threshold drift check (verify Rust-compiled constants match `block_zero.constitutional_thresholds` on bootstrap) | Cycle-6.5c or Cycle-7 — high value but scope-creep for G1 |
| inotify / filesystem watch for live refresh | Post-G4 |

## Constitutional filter

| Invariant | How G1 upholds it |
|---|---|
| ZANN_ZERO | Read path introduces no new economic surface |
| **CLAIM_MUST_BIND** | Hash-chain verification is mandatory on bootstrap; mismatch fails closed |
| RIBA_ZERO | No extractive pattern in read mechanism |
| **NO_SHADOW_STATE** | **Primary payoff: single writer + read-only projection = single canonical truth** |
| IHSAN_FLOOR | 0.95 enforcement remains at kernel layer; G1 does not bypass it |

## Reference points inspected (evidence chain)

- `sovereign_state/block_zero/block_zero.json` — chain root, 10 receipts, `chain_hash`, seal thresholds
- `sovereign_state/receipts/activation_chain_2026-04-13T23:55:26Z.json` — chain envelope format
- `sovereign_state/receipts/*.json` — 7 current files (per-event + `*_latest.json` aggregates)
- `sovereign_state/genesis/*.json` — 7 anchor artifacts
- `bizra-omega/bizra-cognition/src/runtime.rs:249` — `CognitionRuntime::new` surface
- `bizra-omega/bizra-cognition/src/receipts.rs` — `InMemoryPayloadStore` pattern to mirror

## References

- Cycle-6 niyyah: `cycle-6/niyyah.md` §G1
- Cycle-6 execution canon: `cycle-6/execution-canon.md`
- G2 precedent (canonical workspace): `cycle-6/g2-authority-adr.md`
- Founder direction `/@ no`: session log 2026-04-17 14:07 GST
- SAPE analysis that produced the trichotomy (superseded): in-session thread, 2026-04-17 13:45–14:00 GST

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-17 Dubai GST
Cycle chain position: 6 / G1 authority
Canon status: **SEALED** for scope; code work may proceed against this interface spec.

الحمد لله.
