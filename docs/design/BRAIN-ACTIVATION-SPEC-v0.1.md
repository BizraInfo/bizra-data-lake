# BIZRA Brain Activation Spec · v0.1

> **Canonical reframe** — synchronised with the live `bizra-inference` HAL.
> **Status**: Draft · Unsealed (spec doc); the Rust substrate it describes is **shipped on main**.
> **Last updated**: 2026-04-20
> **Isnad**: Originates from the Cycle-8 Brain-Activation discussion. Reconciled per `docs/design/CANON-TERMS.md` v1 §02 / §07 / §10. Prior unpushed draft on branch `design/dema-brain-activation-spec-v0.1` is **superseded** by this file.

---

## 0. What changed between draft and this reframe

The earlier draft (unpushed, 327 lines) declared a new `BrainProvider` enum and an abstract HAL *as if from scratch*. That violated `NO_SHADOW_STATE`: the live `bizra-omega/bizra-inference` crate had already shipped the generation-only layer (`InferenceBackend` + `Backend` + `LMStudioBackend` + `OllamaBackend` + `LlamaCpp`), and PR #30 had shipped the provenance-binding types (`ProvenanceDescriptor`, `ProviderIdentity`, `ReasoningSessionPayload`) in `bizra-cognition`.

This reframe **points at the code**. Every term below resolves to a specific Rust path or an already-canonical doc reference. No phantom types.

---

## 1. The core principle

*Intelligence is a resource; governance is the law.*

The Brain Activation sequence defines how the BIZRA kernel **authenticates, binds, and bounds** cognitive models (LLMs today, TTS / STT / vision next) against the cryptographic receipt chain. No brain-layer output reaches a caller without a provenance-bound envelope.

---

## 2. The three-tier HAL

Two trait layers live side-by-side in `bizra-omega/bizra-inference/src/`:

| Layer | Trait | Concern | Output |
|---|---|---|---|
| **Generation-only** | [`InferenceBackend`](../../bizra-omega/bizra-inference/src/lib.rs) (legacy) | Raw text generation | `InferenceResponse` (plain) |
| **Provenance-bound** | [`CognitionBackend`](../../bizra-omega/bizra-inference/src/hal.rs) (this spec) | Receipt-chain-compatible execution with liveness bounds | [`CognitiveResponse`](../../bizra-omega/bizra-inference/src/hal.rs) — carries [`ProvenanceDescriptor`](../../bizra-omega/bizra-cognition/src/cognition_round.rs) |

Call sites that only need free-form generation continue to use `InferenceBackend`. Call sites that feed the receipt chain (every mission that seals a claim) use `CognitionBackend`. A future adapter trivially bridges the former onto the latter.

### 2.1 Provider identities

Per `bizra-cognition::cognition_round::ProviderIdentity` (PR #30), the HAL supports four provider classes negotiated at T=0 based on the operator's hardware + consent:

| Variant | Fields | Semantics |
|---|---|---|
| `CoreNone` | — | **Sovereignty-first default.** No brain layer active. Kernel still completes every `dema organize` / `dema seal` mission via deterministic primitives. |
| `LocalModel` | `weights_path: String` | Embedded inference (llama.cpp, candle). Maximum privacy. Weights on disk, no network. |
| `LocalServer` | `endpoint: String`, `vendor: String` | Local daemon (Ollama, LM Studio, Whisper, Orpheus TTS). Network-local only. |
| `RemoteApi` | `vendor: String` | Opt-in cloud (OpenAI, Anthropic, etc.). **Leaves the machine.** Caller bears the consent cost. |

Currently implemented: `LocalServer` only (`LocalServerBackend` in [`hal.rs`](../../bizra-omega/bizra-inference/src/hal.rs)). `LocalModel` and `RemoteApi` follow the same shape; implementations are dedicated arcs.

### 2.2 The `CognitiveResponse` envelope — `CLAIM_MUST_BIND` enforced

Every successful execution returns:

```rust
pub struct CognitiveResponse {
    pub response_hash: String,           // BLAKE3 hex of payload
    pub payload: Vec<u8>,
    pub duration: Duration,              // liveness evidence
    pub provenance: ProvenanceDescriptor // model_sha256 + model_signer + provider_identity
}
```

Raw text is never returned without `provenance` populated in full. The downstream `ReasoningSessionPayload` (`ReceiptKind::ReasoningSession` = `0x30`, per PR #30) consumes this struct unchanged; the hashes flow into the Merkle chain without renegotiation.

---

## 3. `IHSAN_FLOOR` enforcement at the edge

`CognitionBackend` implementations hold two contractual boundaries:

1. **LTL liveness ceiling** — every call is bounded. `LocalServerBackend` defaults to `30s`; tunable via `.with_liveness_ceiling(Duration)`. On breach the call returns `InferenceError::LivenessTimeout` and the deterministic kernel is *never* blocked by a runaway probabilistic engine. This is the ex-ante half of the liveness LTL predicate `□(Request ⇒ ◇(Response ∨ Error))` from the Whitepaper §07.
2. **Concurrency semaphore** — each backend caps in-flight requests via a Tokio `Semaphore`. When saturated, callers receive `InferenceError::CapacityExhausted`; no silent queuing. Back-pressure is explicit and propagates to the caller so the Universal Resource Pool can throttle upstream.

A third boundary — `IhsanViolation` — is the post-hoc verdict channel for when the model's self-reported or kernel-verified Ihsān score falls below `IHSAN_FLOOR` (= `0.95` Production tier per `CANON-TERMS.md` §02). Today it is declared; the kernel verdict path that raises it lands with the Governance Decision receipt arc (`ReceiptKind::GovernanceDecision` = `0x40`, see `bizra-cognition::receipts`).

---

## 4. Node-0 initialisation — Progressive Neural Activation

When a user runs `dema seal` (or any cognition-requiring mission) and the system queries the HAL:

1. **If `ProviderIdentity == CoreNone`** (default for a fresh install), the system returns the deterministic portion of the mission (listing, digest, sealed receipt) and respectfully prompts the user:
   > *"This action can continue with DEMA Core (no brain). If you want content-aware classification or drafting, activate a local brain: `dema brain activate`."*
2. **If the user opts in**, the first-run flow offers three lanes:
   - *Starter Brain* (local model, ~4 GB, opt-in download)
   - *Connect Local Server* (existing Ollama / LM Studio on the machine)
   - *Connect Remote Provider* (opt-in cloud; requires explicit consent-log entry)
3. **The chosen provider is bound** into the installer's `InstallReceipt.model_selection.provenance` (see `bizra-installer/src/install_receipt.rs`), sealing `(model_sha256, model_signer, provider_identity)` into the chain.

No brain is ever activated silently. Every activation produces a receipt.

---

## 5. What the code currently guarantees (VERIFIED)

Landed on main as of this reframe:

- ✅ `CognitionBackend` trait defined with `identity` / `execute` / `probe_vitality`
- ✅ `CognitiveRequest` / `CognitiveResponse` structs with full doc coverage
- ✅ `InferenceError` — three canon-aligned variants
- ✅ `LocalServerBackend` reference implementation with semaphore + liveness ceiling
- ✅ Schema-parity `ProvenanceDescriptor` / `ProviderIdentity` imported from `bizra-cognition`
- ✅ Test coverage: identity, execute-returns-provenance, vitality-in-unit-interval, liveness-timeout
- ✅ Clippy `-D warnings` clean, `rustfmt` clean, `#![forbid(unsafe_code)]`, `#![deny(missing_docs, clippy::unwrap_used)]`

---

## 6. What's explicitly DERIVED (not yet wired)

- `LocalServerBackend::execute` currently returns a **deterministic stub** payload. The real Hyper/Reqwest bridge to Ollama's `/api/generate` endpoint is a follow-up arc. The call-site contract (hash, provenance, liveness) does not change when the stub is replaced.
- `LocalServerBackend::probe_vitality` returns a fixed `0.99`. The real SNR+latency probe against `self.endpoint` lands with the wire arc.
- `LocalModelBackend` (embedded llama.cpp / candle) — not yet implemented.
- `RemoteApiBackend` (OpenAI / Anthropic) — not yet implemented; blocked on consent-log substrate.

---

## 7. What's PLANNED (not yet designed)

- **Model whitelist** (`docs/design/MODEL-REGISTRY.md` — does not yet exist). The `ProvenanceDescriptor.model_signer` field is already declared; the signing authority and registry are pending.
- **Signature verification routine** for downloaded model weights against the registry. Without it, `model_signer = None` is the honest state.
- **Starter Brain download CLI** (`dema brain install starter`, `dema brain connect ollama`). HAL is ready; CLI wiring is the Voice-Stack sibling arc.
- **Ihsān-score runtime integration** — the `IhsanViolation` variant is in the error surface; the gate that fires it is in the governance-decision arc.

---

## 8. Isnad — source-of-truth references

| Term | Source |
|---|---|
| Five invariants (`ZANN_ZERO`, `RIBA_ZERO`, `CLAIM_MUST_BIND`, `NO_SHADOW_STATE`, `IHSAN_FLOOR`) | `docs/design/CANON-TERMS.md` §01 |
| Ihsān tier values (0.90 / 0.95 / 0.99 / 1.0) | `docs/design/CANON-TERMS.md` §02 |
| Provider identity enum | `bizra-omega/bizra-cognition/src/cognition_round.rs::ProviderIdentity` |
| Provenance descriptor schema | `bizra-omega/bizra-cognition/src/cognition_round.rs::ProvenanceDescriptor` |
| Receipt kind `ReasoningSession` (0x30) | `bizra-omega/bizra-cognition/src/receipts.rs::ReceiptKind` |
| Install receipt model selection | `bizra-omega/bizra-installer/src/install_receipt.rs::ModelSelection` |
| 8-layer APEX stack · SAPE pipeline · 8-dim Ihsān Vector | OMNI-SYNTHESIS Whitepaper · `/home/bizra-operating-system/Downloads/DEMA.zip/OMNI-SYNTHESIS Whitepaper.html` |

---

## 9. Change protocol

Any change to this spec requires:

1. A prior or same-commit change to the Rust substrate it describes (`bizra-inference/src/hal.rs` and/or `bizra-cognition/src/cognition_round.rs`).
2. Updated tests in both crates where types change shape.
3. A brief **Why** entry added to this §9 below.

### Change log
| Date | Commit | Change |
|---|---|---|
| 2026-04-20 | *this commit* | Initial reframe — supersedes the unpushed draft by pointing at the live HAL module. |
