# bizra-cognition-gateway

HTTP projection of the `bizra-cognition` runtime, plus `dema` — the BIZRA Principal's terminal face.

**Status:** Cycle-5 G3a complete — first principal-activation receipt sealable end-to-end.
**Category:** Verificative AI (Generative → Agentic → **Verificative**).
**Doctrine:** `docs/bizra-trust-compiler-thesis.md` + `docs/dema-cli-manifesto-v1.md`.

---

## Two binaries

```bash
# The HTTP gateway (Axum, localhost-only by default)
cargo run -p bizra-cognition-gateway

# The CLI (operator terminal face, talks to the gateway)
cargo run -p bizra-cognition-gateway --bin dema -- --help
```

Release binaries after `cargo build --release`:
- `target/release/bizra-cognition-gateway`
- `target/release/dema`

---

## 30-second quickstart

```bash
# Terminal 1 — start the gateway (stays running)
./target/release/bizra-cognition-gateway

# Terminal 2 — you are the principal
./target/release/dema                                  # status at a glance
./target/release/dema activate                         # submit the canonical intent
./target/release/dema chain                            # inspect the chain
./target/release/dema receipt <64-char-hex-receipt-id> # inspect one receipt
./target/release/dema submit "organize my Downloads"   # custom intent (Cycle-6 will make this real)
./target/release/dema submit "..." --quality 0.5       # watch IHSAN_FLOOR reject with remediation
```

Expected output for `dema activate` on a fresh gateway:

```
mission:     <64-char-hex>
admissibility:
  ✓ ZANN_ZERO          Permit    score=1.0000
  ✓ CLAIM_MUST_BIND    Permit    score=1.0000
  ✓ RIBA_ZERO          Permit    score=1.0000
  ✓ NO_SHADOW_STATE    Permit    score=1.0000
  ✓ IHSAN_FLOOR        Permit    score=0.9800
  verdict: Permit
receipt:     <64-char-hex>
stage:       Replayability
chain_head:  <same as receipt>
✓ chain head equals receipt id — sealed
```

---

## Gateway HTTP contract

Default bind: `127.0.0.1:7421` (override via `BIZRA_COGNITION_PORT` env).

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | `{status: "ok", domain: "bizra-cognition-gateway-v1"}` |
| `GET` | `/chain` | `ReceiptChainHeadDto { head, length, latestTimestamp }` |
| `GET` | `/chain/:hash` | `ReceiptDto` (header only) or 404 with structured error |
| `POST` | `/mission` | `SubmitMissionResponse` (Permit) or HTTP 422 + structured admissibility (Reject) |

### POST /mission request body

```json
{
  "intent": "activate my dual agentic system",
  "currentState": { "summary": "...", "metric": 0.0 },
  "idealState":   { "summary": "...", "metric": 1.0 },
  "qualityScore": 0.98,
  "originator":   "Operator"
}
```

### POST /mission success (HTTP 200)

```json
{
  "missionId":  "<hex>",
  "admissibility": {
    "verdict": "Permit",
    "gateVerdicts": [
      {"scorerId": "ZANN_ZERO", "invariant": "ZANN_ZERO", "verdict": "Permit", "reason": "...", "score": 1.0},
      ...
    ]
  },
  "receiptId":  "<hex>",
  "finalStage": "Replayability",
  "chainHead":  "<hex>"
}
```

### POST /mission reject (HTTP 422)

```json
{
  "error": {
    "code": "ADMISSIBILITY_REJECTED",
    "message": "mission rejected by admissibility chain",
    "domain": "bizra-cognition-gateway-v1",
    "admissibility": {
      "verdict": "Reject",
      "gateVerdicts": [...],
      "rejected": {
        "invariant": "IHSAN_FLOOR",
        "reason": "IHSAN_FLOOR violation: score 0.5000 below floor 0.9500",
        "remediationPath": "Improve claim quality score to ≥ 0.95 ...",
        "escalationAllowed": true
      }
    }
  }
}
```

**Note:** HTTP 422 is the reject semantics, NOT HTTP 500. Rejection is a lawful admissibility outcome, not a server error.

---

## CLI (`dema`) command reference

```
dema              status at a glance (health + chain head)
dema health       gateway liveness + domain tag
dema chain        chain head, length, latest timestamp
dema receipt <h>  inspect one receipt by hex id
dema activate     submit the canonical principal activation intent
dema submit "..." submit a custom intent
  --quality <f>   override quality score (default 0.98, must be ≥ 0.95 for IHSAN_FLOOR)
--json            machine-readable output on any command (global flag)
```

### Exit codes (operator discipline)

| Code | Meaning |
|---|---|
| 0 | Command succeeded |
| 1 | Gateway unreachable or protocol error (network, JSON decode, HTTP 5xx) |
| 2 | Admissibility reject — a lawful verdict, not an error |

The distinction between exit 1 and exit 2 is constitutional: **rejection is not an error**. It is the system correctly refusing to canonicalize sub-quality work. Shell scripts relying on `set -e` should test for exit 2 explicitly when expecting possible rejects.

---

## Constitutional guarantees (what this gateway WILL NOT do)

This gateway is a *verificative* surface. It enforces the five invariants from `bizra-cognition::admissibility_freeze_v1` before any chain mutation:

1. **ZANN_ZERO** — no claim without evidence binding. Every submitted mission must carry `evidence_hash` (the gateway defaults it to `mission_id` if omitted).
2. **CLAIM_MUST_BIND** — every chain-resident claim carries hash-addressed evidence.
3. **RIBA_ZERO** — no extractive economic pattern in operator-visible paths.
4. **NO_SHADOW_STATE** — rejected missions NEVER enter the chain. The chain contains only lawful completions. Rejection is recorded in derived state (the `missions` registry, queryable via `mission_by_id`) — not on the chain of source truth.
5. **IHSAN_FLOOR = 0.95** — quality score must be ≥ 0.95 for Permit. There is no override. Sub-0.95 submissions receive a structured reject with a remediation path.

**Violation of any invariant is a hard refusal** with an explanatory reason, not a soft degradation.

---

## Architecture (for the curious)

```
dema (CLI)
  ↓ HTTP (127.0.0.1:7421)
bizra-cognition-gateway (Axum)
  ↓ direct Rust call
CognitionRuntime (bizra-cognition)
  ├── ThoughtGraph (dual-rate cognition)
  ├── ReceiptChain (§10 source of truth)
  │     ↑ append_with_payload
  └── AdmissibilityChain (5-gate evaluator)
        ↑ evaluate BEFORE any chain mutation
```

- Rust runtime: `bizra-cognition` crate (11 modules, 64 tests green)
- HTTP adapter: `bizra-cognition-gateway` crate (7 tests green)
- CLI adapter: `dema` binary in same crate
- Constitutional freeze layer: `admissibility_freeze_v1.rs`, `receipt_freeze_v1.rs`, `mission_freeze_v1.rs`, `manifest_artifact.rs`

**Persistence note:** default gateway boot uses `InMemoryPayloadStore` — the chain is ephemeral across restarts unless `BIZRA_RECEIPT_STORE_PATH` is set. When set to an explicit path, the gateway bootstraps sled payloads under `<root>/payloads/` and authoritative chain metadata at `<root>/chain_snapshot.json`. When set to `default`, the path expands to the operator canonical store (see below) while still requiring an explicit env var — persistence is never enabled implicitly. `BIZRA_DEMA_CACHE_ROOT` remains a derived cache only and does not rehydrate `GET /chain`.

**Operator launch (opt-in persistence):**

```bash
# Explicit store root
export BIZRA_RECEIPT_STORE_PATH=/path/to/receipt_store
./target/release/bizra-cognition-gateway

# Operator default path (requires env var; resolves under sovereign_state / data lake / DEMA_HOME)
export BIZRA_RECEIPT_STORE_PATH=default
export BIZRA_DATA_LAKE_ROOT=/data/bizra   # optional anchor for default resolution
./target/release/bizra-cognition-gateway
```

Default path resolution order when `BIZRA_RECEIPT_STORE_PATH=default`:

1. `$BIZRA_SOVEREIGN_STATE_PATH/authoritative_receipt_store`
2. `$BIZRA_DATA_LAKE_ROOT/sovereign_state/authoritative_receipt_store`
3. `$DEMA_HOME/authoritative_receipt_store`
4. `$HOME/.dema/authoritative_receipt_store`
5. `./sovereign_state/authoritative_receipt_store`

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BIZRA_COGNITION_PORT` | `7421` | Gateway bind port |
| `BIZRA_RECEIPT_STORE_PATH` | (unset) | Authoritative receipt chain store root (sled payloads + `chain_snapshot.json`). Set to `default` for operator canonical path; unset keeps in-memory chain. |
| `BIZRA_DATA_LAKE_ROOT` | (unset) | Data lake root; used when resolving `BIZRA_RECEIPT_STORE_PATH=default` |
| `BIZRA_SOVEREIGN_STATE_PATH` | (unset) | Sovereign state root; preferred anchor when resolving `BIZRA_RECEIPT_STORE_PATH=default` |
| `DEMA_HOME` | (unset) | Operator home; used when resolving `BIZRA_RECEIPT_STORE_PATH=default` |
| `BIZRA_COGNITION_GATEWAY_URL` | `http://127.0.0.1:7421` | CLI target (for remote gateway scenarios) |
| `RUST_LOG` | (unset; defaults to `info`) | Tracing verbosity |

---

## Testing

```bash
# Gateway + CLI unit tests (7 gateway-specific)
cargo test -p bizra-cognition-gateway

# Full workspace (includes bizra-cognition 64 + gateway 7 + ~1,200 others)
cargo test --workspace
```

---

## Related documentation

- `docs/bizra-trust-compiler-thesis.md` — why this gateway exists; the verificative-AI thesis
- `docs/dema-cli-manifesto-v1.md` — operating law for the CLI surface
- `docs/ftap-function-registry-rfc-seed.md` — future architecture (NOT current scope)
- `cycle-5/retrospective.md` — formal closure of the cycle that shipped this
- Manifest v0.2 (§6, §8, §10, §16) — constitutional authority

---

*BIZRA is not building an assistant. BIZRA is building the operating law for assistants. `dema` is the principal's terminal face of that law.*

الحمد لله.
