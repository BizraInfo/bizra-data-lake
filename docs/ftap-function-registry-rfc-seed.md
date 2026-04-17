# FTAP — Functional Trust Assertion Protocol — RFC seed (Layer 2)

بسم الله الرحمن الرحيم

**Author:** Mumo (Node0 principal), distilled from Cycle-5 external-convergence analysis
**Drafted:** 2026-04-17
**Status:** RFC SEED — **not current sprint scope**. This document defines a future architectural direction; it is *not* a commitment to build it next.
**Blocked on:** Cycle-6 tool-execution arc, Cycle-7 LLM-inference arc, Cycle-8 persistence arc (in that order)
**Companion:** `docs/bizra-trust-compiler-thesis.md` (Layer 1, current executable canon)

---

## 1. Why this document exists separately

During the Cycle-5 strategic review, an external analysis proposed a "Functional Trust Assertion Protocol" (FTAP): a standard for APIs to declare machine-readable, verifiable claims about their own performance characteristics (latency, accuracy, cost, uptime), paired with a `/verify` endpoint and a decentralized auditor network.

The proposal is architecturally sound **and out of scope for current sprint.** Filing it here as a seed preserves the idea, bounds it as long-range, and prevents it from contaminating the Cycle-6 tool-execution arc.

Per the manifesto's self-governance: architecture that distracts from current-gate discipline is noise, not signal, until the current gate closes.

## 2. The core idea (preserved for later)

Every shipping API today documents its **shape** via OpenAPI / Swagger (routes, request/response schemas). No API documents its **trust profile** — how well it performs against its own claims. FTAP proposes the missing layer:

```json
{
  "function": "remove_image_background",
  "input_format": ["png", "jpg"],
  "output_format": "png",
  "p99_latency_ms": 180,
  "cost_per_call_usd": 0.004,
  "accuracy_fp_rate": 0.002,
  "availability_sla": 0.9995,
  "attestation_endpoint": "/verify/remove_image_background",
  "attestation_signature": "<Ed25519 sig over the above>"
}
```

The service's host publishes this at `.well-known/function.json`. An auditor (or the BIZRA admissibility chain) can poll the attestation endpoint and verify the claims match reality.

This transforms the API economy from *"trust the vendor's marketing"* to *"verify the vendor's own signed attestation continuously."* It is the natural extension of IHSAN_FLOOR from single-node to network-of-functions.

## 3. Why this is Cycle-8+ at the earliest

FTAP presupposes infrastructure that does not exist yet:

| FTAP prerequisite | BIZRA state today | Earliest gate |
|---|---|---|
| Tool execution with per-call receipts | not yet wired | Cycle-6 |
| LLM inference with IHSAN_FLOOR enforcement | not yet wired | Cycle-7 |
| Chain persistence across process lifecycle | InMemoryPayloadStore default | Cycle-6-parallel |
| Multi-node federation for auditor agents | principal-local only; federation is §12 | Cycle-9+ |
| Cross-node signature verification | single-principal signing only | Cycle-9+ |

Shipping FTAP before tool execution = shipping a registry for functions we cannot call, verified by auditors that cannot coordinate. Building order matters.

## 4. The localhost-first path (FTAP-lite)

When the time comes, the first build is **not** a decentralized auditor network. It is the same pattern executed locally first:

### FTAP-lite v0.1 (the single-node version)

```
A single Node0 principal ships:
  - A local Rust registry: HashMap<FunctionName, FunctionAttestation>
  - Every FTAP-compliant API, even third-party, has its attestation
    fetched, signed, and stored on the principal's own chain
  - `dema registry list` shows all currently-trusted functions
  - `dema registry verify <fn>` re-pings the attestation endpoint
  - Admissibility chain checks registry before calling any tool
```

The principal becomes their own auditor. The registry is a local extension of the chain. No cloud. No blockchain. No federation. Everything verifiable against the principal's own genesis hash.

### FTAP v1 (federation, far future)

Only after several principals run FTAP-lite successfully does the federation question become real. At that point, Manifest §12 (federation) governs — not this seed.

## 5. What we preserve from the external analysis (no blockchain, no decentralized oracle)

The external analysis proposed a Merkle-Tree-of-Trust via a decentralized auditor network. **We explicitly reject that framing** at Layer 2 seed level:

- Decentralized oracle networks are public infrastructure; BIZRA is principal-local
- Blockchain-style consensus is a governance pattern for untrusted parties; sovereign principals are **trusted by definition of their own chain**
- Cross-principal federation is a different problem with different primitives (§12)

What we DO preserve:
- The per-function attestation format (signed metadata at `.well-known/function.json`)
- The `/verify` endpoint contract as a self-describing trust claim
- The idea of IHSAN_FLOOR extending from "my node's work" to "every tool I invoke"
- The semantic API web direction — machine-readable, rateable function metadata

## 6. Where FTAP-lite eventually bolts onto the existing stack

When Cycle-8 opens FTAP-lite:

1. **New crate:** `bizra-function-registry` — owns the registry, attestation verification, cache policy
2. **Admissibility extension:** Add a new invariant `FUNCTION_TRUSTED` (optional 6th gate, principal-configurable). Any tool call where the target function does not have a fresh attestation with score ≥ principal's threshold → admissibility rejects with remediation path
3. **CLI surface:** `dema registry list | verify | trust | distrust`
4. **Gateway endpoint:** `GET /registry` → current function trust table

None of this is built. None of this is scheduled. This is a seed.

## 7. Non-goals (preventing scope creep)

1. No work on FTAP before Cycle-6 arc #1 (tool execution via MCP) is complete
2. No decentralized auditor network — ever, under this seed
3. No blockchain / consensus mechanisms
4. No cloud-hosted registry as primary mode
5. No public rating site, public leaderboards, or public oracle

## 8. When to revisit this seed

Specific green-light conditions:

1. Cycle-6 complete: `dema submit "organize Downloads"` produces verifiable per-file-move receipts
2. Cycle-7 complete: LLM completions are IHSAN_FLOOR-gated before canonicalization
3. Cycle-8 complete: chain survives process restart via sled-store
4. Only then: open Cycle-9 as "FTAP-lite — local function trust registry"

Attempting FTAP before all three gates is building a cathedral on sand.

---

## 9. The one line that makes this seed valuable

> Every tool BIZRA ever calls becomes a lawful action *by construction* — not because the tool is trusted, but because the registry certifies its trust attestation continuously and the admissibility chain refuses to invoke any function below threshold.

That is the end-state. FTAP is its name. Not now.

الحمد لله — preserve the seed, guard the gate.
