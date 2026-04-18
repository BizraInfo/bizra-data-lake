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

pub mod canonical_hasher;
pub mod receipts;
pub mod receipt_freeze_v1;
pub mod admissibility_freeze_v1;
pub mod mission_freeze_v1;
pub mod manifest_artifact;
pub mod principal_activation;
pub mod principal_cache;
pub mod receipt_history_cache;
pub mod manifest_history_cache;
pub mod thought_graph;
pub mod configure_cognition;
pub mod runtime;
pub mod eval_v1_integrated;
pub mod eval_v1;
pub mod sovereign_state;
