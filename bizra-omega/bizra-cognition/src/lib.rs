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

pub mod admissibility_freeze_v1;
pub mod canonical_hasher;
pub mod configure_cognition;
pub mod eval_v1;
pub mod eval_v1_integrated;
pub mod manifest_artifact;
pub mod mission_freeze_v1;
pub mod receipt_freeze_v1;
pub mod receipts;
pub mod runtime;
pub mod sovereign_state;
pub mod thought_graph;
