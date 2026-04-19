//! BIZRA Seal Primitive — Day 1 Design Artifact
//! ============================================
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-omega/bizra-cognition/src/seal.rs
//! Cycle: Cycle-8 (First Fire preparation)
//! Day: 1 of 12
//! Build step: design-only. No behavior changes, no existing files modified.
//! Truth target: DRAFT — graduates to TESTED when Day 2 lands the first
//! `impl Sealable`; PROVEN when `cargo test -p bizra-cognition` passes with
//! seal-backed organize behaving identically to pre-refactor organize.
//!
//! ─── WHY ────────────────────────────────────────────────────────────────
//!
//! Today `organize_mission.rs` couples the universal lawful-loop shape
//!     envelope → admissibility → execute → receipt → chain-append
//! to a single mission type (organize a filesystem directory). The sealing
//! act is hidden inside organize-specific code. That coupling prevents any
//! future DEMA face (`text-seal`, `artifact-seal`, `witness-seal`, …) from
//! reusing the same lawful path without copying organize's structure.
//!
//! This file extracts the universal primitive: a `Sealable` contract that
//! every future artifact type must fulfill to pass through the admissibility
//! chain and land a `ReceiptArtifact` on the receipt chain.
//!
//! The public story, in one line:
//!
//!     DEMA seals reality. Organize is the first proof.
//!
//! `Sealable` is the "seals reality" primitive. Implementations prove it.
//!
//! ─── CONSTITUTIONAL INVARIANTS PRESERVED AT THE TRAIT BOUNDARY ─────────
//!
//! This trait is the narrowest neck through which every seal act must pass.
//! All 5 Manifest §3 invariants must remain enforceable here. If any
//! cannot be enforced at this boundary, the trait shape is wrong and the
//! Day 2 refactor must halt and flag before landing.
//!
//! 1. **ZANN_ZERO** — No claim without evidence.
//!    Enforcement: `seal_envelope()` returns an `AdmissibilityClaim` whose
//!    `has_evidence` field is populated by the implementer. The
//!    admissibility chain's `ZannZeroGate` will reject the seal whenever
//!    the claim declares no evidence. The trait exposes the field; the
//!    kernel enforces it. The trait cannot bypass the gate.
//!
//! 2. **CLAIM_MUST_BIND** — Every claim must bind to verifiable evidence.
//!    Enforcement: `AdmissibilityClaim.evidence_hash` must be
//!    `Some(non-zero)` whenever `has_evidence` is `true`. The trait method
//!    `bytes_for_digest()` produces the canonical byte sequence over which
//!    the evidence hash is computed, so the binding is cryptographically
//!    derivable from the sealable artifact itself — not from external
//!    assertion. `ClaimMustBindGate` fails the seal if the hash is zero
//!    or absent when required.
//!
//! 3. **RIBA_ZERO** — No extractive economic pattern.
//!    Enforcement: `AdmissibilityClaim.economic_pattern` is populated by
//!    the implementer; the chain's `RibaZeroGate` rejects extractive
//!    patterns. The trait cannot smuggle an extractive act through — the
//!    gate runs on the claim the trait produced, not on trait surface.
//!
//! 4. **NO_SHADOW_STATE** — Visible state ≡ kernel state.
//!    Enforcement: the trait operates BEFORE any state mutation. A seal's
//!    `AdmissibilityClaim.state_mutation` declares the intended kernel
//!    transition; `NoShadowStateGate` rejects mutations that do not
//!    derive from canonical runtime. The trait can only propose; the
//!    kernel decides.
//!
//! 5. **IHSAN_FLOOR** — Quality ≥ 0.95.
//!    Enforcement: `AdmissibilityClaim.quality_score` is set by the
//!    implementer and evaluated by `IhsanFloorGate`. A `Sealable`
//!    implementation whose `quality_score < 0.95` cannot produce a
//!    `Verdict::Permit`; the seal fails honestly at the chain, not at the
//!    trait boundary.
//!
//! All 5 invariants are enforced in `AdmissibilityChain::canonical()
//! .evaluate()` (see `admissibility_freeze_v1.rs`). The trait's sole job
//! is to deliver a well-formed `AdmissibilityClaim` and the canonical
//! bytes for its evidence binding. Enforcement lives in the kernel.
//!
//! ─── DAY 2 REFACTOR SKETCH (not implemented today) ─────────────────────
//!
//! `organize_mission.rs` currently has (paraphrased):
//!
//!     fn organize(path, quality_score) -> OrganizeOutcome {
//!         // 1. build an AdmissibilityClaim from (path, quality)
//!         // 2. run AdmissibilityChain::canonical().evaluate(&claim)
//!         // 3. execute the organize-specific action (list directory)
//!         // 4. build a MissionExecuted receipt
//!         // 5. append to ReceiptChain
//!     }
//!
//! Day 2 refactor will:
//!
//!     impl Sealable for OrganizeRequest {
//!         fn seal_envelope(&self) -> AdmissibilityClaim {
//!             // step 1: path + quality_score → AdmissibilityClaim
//!         }
//!
//!         fn bytes_for_digest(&self) -> Vec<u8> {
//!             // canonical bytes of path + listing digest
//!         }
//!
//!         fn receipt_kind() -> ReceiptKind {
//!             ReceiptKind::MissionExecuted
//!         }
//!     }
//!
//! The organize-specific execution (step 3) stays in `organize_mission.rs`
//! — it is the artifact's unique semantic. The lawful-loop scaffold
//! (steps 1, 2, 4, 5) becomes a generic:
//!
//!     fn seal<S: Sealable + SealableOutcome>(
//!         req: S,
//!         runtime: &mut CognitionRuntime,
//!     ) -> Result<<S as SealableOutcome>::Ok, SealError>
//!
//! so no future face rewrites the lawful path. The Manifest §6 rule
//! "one lawful loop, no bypasses" is preserved by construction, not by
//! convention.
//!
//! ─── FUTURE IMPLEMENTATIONS (sketch; not built today) ──────────────────
//!
//! Each future DEMA face becomes a new `Sealable` impl, not a new
//! lawful-loop scaffold.
//!
//!     impl Sealable for TextSealRequest {
//!         // Seal a string of text (e.g., a published claim, a signed
//!         // statement, an attestation).
//!         // `bytes_for_digest`: UTF-8 canonical form of the text +
//!         // metadata envelope.
//!         // `receipt_kind`: ReceiptKind::MissionExecuted for now; a
//!         // dedicated `TextSealed = 0x71` variant may be proposed via
//!         // constitutional amendment if text-sealing becomes its own
//!         // mission class.
//!     }
//!
//!     impl Sealable for ArtifactSealRequest {
//!         // Seal a binary artifact (e.g., a PDF, a container image, a
//!         // code bundle, a model weights file).
//!         // `bytes_for_digest`: BLAKE3 of artifact bytes + metadata
//!         // envelope (filename, size, mime, declared origin).
//!         // `receipt_kind`: ReceiptKind::MissionExecuted; ArtifactSealed
//!         // = 0x72 is a candidate for future amendment.
//!     }
//!
//!     impl Sealable for WitnessAttestationRequest {
//!         // Seal an external witness node's chain-head observation,
//!         // closing the 4th (economic/witness) modality of the Golden
//!         // Standard (the witness-node gossip primitive).
//!         // `bytes_for_digest`: witness node-id + observed chain-head +
//!         // observation timestamp.
//!         // `receipt_kind`: a new `WitnessObserved = 0x73` byte is
//!         // expected for this class (future amendment).
//!     }
//!
//! ─── NON-GOALS TODAY ────────────────────────────────────────────────────
//!
//! - No code modifications to any existing file.
//! - No new `ReceiptKind` variants.
//! - No wiring of this module into `lib.rs` (deferred to Day 2).
//! - No tests (deferred to Day 2).
//! - No implementations (deferred to Day 2).
//! - No generic `fn seal<S: Sealable>(...)` yet (deferred to Day 2).
//! - No `SealError` enum (deferred to Day 2).
//!
//! Close it. Prove it. Reveal it.
//!
//! ─── END DOCSTRING ─────────────────────────────────────────────────────

use crate::admissibility_freeze_v1::AdmissibilityClaim;
use crate::receipts::ReceiptKind;

// ═══════════════════════════════════════════════════════════════════════
// Sealable — the universal DEMA seal primitive
// ═══════════════════════════════════════════════════════════════════════

/// The universal contract every DEMA seal must satisfy.
///
/// An implementation of `Sealable` produces:
///   1. an `AdmissibilityClaim` (input to the 5-gate admissibility chain),
///   2. canonical bytes over which the evidence hash binding is derived,
///   3. the `ReceiptKind` stamped on the resulting chain record.
///
/// The trait is deliberately minimal. It exposes only the surface the kernel
/// needs to enforce the 5 constitutional invariants. Operational richness
/// (retry policy, backpressure, observability, async behavior) belongs in
/// the implementer's own type, not in this trait.
///
/// The `Send + Sync` bound is required because sealable artifacts may be
/// moved across the gateway's async execution boundary; the kernel never
/// assumes single-threadedness.
pub trait Sealable: Send + Sync {
    /// Produce the `AdmissibilityClaim` the 5-gate chain will evaluate.
    ///
    /// The implementer is responsible for populating every field:
    ///
    /// - `claim_id`: canonical hash identifying this sealable artifact
    ///   (derived from `bytes_for_digest`).
    /// - `has_evidence`: whether binding evidence is present
    ///   (checked by ZANN_ZERO).
    /// - `evidence_hash`: the binding hash; `Some(non-zero)` when
    ///   `has_evidence` is `true` (checked by CLAIM_MUST_BIND).
    /// - `economic_pattern`: any declared economic action
    ///   (checked by RIBA_ZERO).
    /// - `state_mutation`: any declared kernel state change
    ///   (checked by NO_SHADOW_STATE).
    /// - `quality_score`: self-assessed quality in `[0.0, 1.0]`; the chain
    ///   requires `≥ 0.95` to Permit (checked by IHSAN_FLOOR).
    /// - `timestamp_ns`: monotonic timestamp for this seal attempt.
    fn seal_envelope(&self) -> AdmissibilityClaim;

    /// Produce the canonical byte sequence that uniquely identifies this
    /// sealable artifact.
    ///
    /// `CLAIM_MUST_BIND` enforces that the `evidence_hash` is the BLAKE3
    /// of these bytes (via `canonical_hasher`). Implementations MUST be
    /// deterministic: identical input artifact → identical bytes on every
    /// machine. Non-deterministic bytes break the empirical-reproducibility
    /// modality of the Four-Modality Golden Standard.
    fn bytes_for_digest(&self) -> Vec<u8>;

    /// The `ReceiptKind` stamped on the chain record sealed by this artifact.
    ///
    /// Each `Sealable` type owns exactly one `ReceiptKind`. `OrganizeRequest`
    /// will return `ReceiptKind::MissionExecuted` (0x70). Future faces
    /// either reuse `MissionExecuted` or request a new `ReceiptKind` byte
    /// via constitutional amendment (no silent additions).
    ///
    /// This is a type-associated function (not a method on `&self`) because
    /// the receipt kind is a property of the Sealable type itself, not any
    /// particular instance. The `Self: Sized` bound makes this dyn-compatible
    /// only through explicit object-safe wrappers.
    fn receipt_kind() -> ReceiptKind
    where
        Self: Sized;
}

// ═══════════════════════════════════════════════════════════════════════
// SealableOutcome — typed result boundary
// ═══════════════════════════════════════════════════════════════════════

/// The typed three-outcome boundary every `Sealable` implementation declares.
///
/// Every seal attempt has exactly three honest outcomes:
///
///   - **`Ok`**: the artifact was sealed lawfully; the chain advanced and
///     a receipt was issued.
///   - **`Refused`**: the admissibility chain denied the seal; a
///     `RejectedClaim` is available with the violated invariant and
///     remediation text.
///   - **`Unreachable`**: the chain or gateway could not be reached; no
///     verdict was obtained. The seal did NOT occur.
///
/// This mirrors the gateway client's discriminated union in
/// `dema-console/src/lib/gateway-client.ts` (`GatewayOutcome`), lifted into
/// the kernel crate so every DEMA face renders seal results from the same
/// typed contract — enforcing `NO_SHADOW_STATE` at the result boundary.
///
/// A fourth outcome (e.g., `Simulated`, `Guessed`, `Estimated`) is
/// constitutionally forbidden: the face never simulates law. If the kernel
/// cannot decide, the face says `Unreachable`; it does not invent a verdict.
pub trait SealableOutcome {
    /// The type carried when the seal succeeded lawfully.
    /// Implementers typically set this to a struct holding the sealed
    /// receipt id, chain head, and artifact-specific summary data.
    type Ok;

    /// The type carried when the admissibility chain refused the seal.
    /// Implementers typically set this to `RejectedClaim` directly, or a
    /// wrapper carrying pre-gate refusal metadata alongside it.
    type Refused;

    /// The type carried when the chain/gateway is unreachable.
    /// Implementers typically set this to a struct holding the attempted
    /// operation and a human-readable reason.
    type Unreachable;
}

// ═══════════════════════════════════════════════════════════════════════
// Day 2 plan (not implemented here)
// ═══════════════════════════════════════════════════════════════════════
//
// Day 2 will introduce in this crate:
//
//   - `enum SealError { Refused(RejectedClaim), Unreachable(String),
//         InternalInvariantViolation(&'static str) }`
//
//   - `fn seal<S: Sealable + SealableOutcome>(
//         req: S,
//         runtime: &mut CognitionRuntime,
//     ) -> Result<<S as SealableOutcome>::Ok, SealError>`
//       that wraps the full lawful loop once: envelope → admissibility →
//       execute → receipt → chain-append. Every DEMA face calls this
//       function; no face reimplements the scaffold.
//
//   - `impl Sealable for OrganizeRequest` + `impl SealableOutcome for
//         OrganizeRequest` extracted from `organize_mission.rs`, preserving
//         today's observable outcomes bit-for-bit.
//
//   - Tests asserting the refactor is behaviorally identical to the
//     pre-refactor organize path, using existing G5 fixtures.
//
//   - `pub mod seal;` added to `lib.rs` at the same commit as the first
//     impl, so the module is wired in the moment it has real work.
//
// If any of the 5 invariants above cannot be preserved cleanly at the
// trait boundary when Day 2 lands — for example, if `state_mutation`
// cannot be determined before execution for some Sealable type — HALT and
// flag before merging. The trait shape may need to change before the first
// impl can be honest.
