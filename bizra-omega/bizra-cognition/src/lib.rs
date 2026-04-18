// bizra-cognition — BIZRA cognition substrate
//
// Dual-rate thought graph, receipted myelination, replay-from-chain
// runtime, and genesis valuation. This is the kernel between the
// proof engine (bizra-proofspace) and the PAT/SAT agent runtime.
//
// Layer ordering:
//   canonical_hasher  (bridge to bizra-core)
//   receipts          (foundation: immutable receipt chain)
//   thought_graph     (dual-rate cognition + myelination)
//   configure_cognition (boot compositor: PAT-7 + SAT-5 factories)
//   runtime           (event loop + replay-from-chain rehydration)
//   eval_v1_integrated (genesis valuation via Proof-of-Impact)
//
// ── ci-hygiene waivers (2026-04-18) ──────────────────────────────────
// These suppress clippy lints that flag stable code patterns in this
// crate. Each one is documented so future contract-hardening work can
// decide whether to refactor or keep the waiver.
#![allow(
    // 10 call sites use `&hash` where `hash` already deref's correctly.
    // Refactoring all would churn receipt-hashing code; waiver retained
    // until Spearpoint A.4 (if any) touches these paths.
    clippy::needless_borrow,
    // Some test helpers + trait methods are intentionally never called
    // outside their own module or feature flag. Removing would regress
    // test fixture surface.
    dead_code,
    // One ValuationConfig nesting in eval_v1_integrated exceeds
    // clippy's complexity threshold. Type alias refactor belongs in
    // the valuation arc, not hygiene.
    clippy::type_complexity,
    // 6 hex_decode loops across cache modules use `for i in 0..32` to
    // parse hex bytes into a fixed buffer; iterator-based refactor
    // would obscure the 2-nibble stride. Deferred.
    clippy::needless_range_loop,
    // ResourceKind::from_str is a domain naming choice, not an
    // implementation of std::str::FromStr. Renaming would churn G4 API.
    clippy::should_implement_trait,
    // Manual Default + manual is_multiple_of in resource_registry_cache
    // are explicit-intent code; Derive would silence a human signal.
    clippy::derivable_impls,
    clippy::manual_is_multiple_of,
)]

pub mod admissibility_freeze_v1;
pub mod canonical_hasher;
pub mod configure_cognition;
pub mod eval_v1;
pub mod eval_v1_integrated;
pub mod manifest_artifact;
pub mod manifest_history_cache;
pub mod mission_freeze_v1;
pub mod mission_log_cache;
pub mod organize_mission;
pub mod poi_ledger;
pub mod principal_activation;
pub mod principal_cache;
pub mod receipt_freeze_v1;
pub mod receipt_history_cache;
pub mod receipts;
pub mod resource_registry;
pub mod resource_registry_cache;
pub mod runtime;
pub mod sovereign_state;
pub mod state_snapshots_cache;
pub mod thought_graph;
