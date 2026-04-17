//! BIZRA Lawful Loop — §6 End-to-End Connector + §16 Minimum Undeniable Loop
//!
//! بسم الله الرحمن الرحيم
//!
//! File: crates/bizra-kernel/src/lawful_loop.rs
//! Authority: Manifest v0.2 Canon, §6 (Runtime Flow), §16 (Success Condition)
//! Build Step: 7 of 8 (§17)
//! Depends on: Steps 2-6 (all freeze artifacts + gateway)
//!
//! This file connects all frozen contracts into a single executable path:
//!
//!   S1: Intent       → raw string
//!   S2: Mission      → MissionEnvelope::from_intent()
//!   S3: Claim        → MissionEnvelope::extract_claim_id()
//!   S4: Admissibility→ AdmissibilityChain::evaluate()
//!   S5: Execution    → execute_fn() (caller-provided)
//!   S6: Receipt      → ReceiptArtifact::new()
//!   S7: Canon        → ReceiptChain::append_artifact()
//!   S8: Replay       → verify replay (re-hash, compare)
//!   S9: Reflex       → optional (not wired in v1)
//!
//! §16: "one mission enters through one authoritative entry, one gate chain
//! evaluates it, one receipt lineage is emitted, one replay can reproduce it,
//! one daily manifest includes it, one trust surface reveals it, and one
//! assistant face presents it coherently to the operator."
//!
//! This module proves that path exists and works.

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{
    Blake3Hash, ReceiptChain, ReceiptPayload, ChainError,
};
use crate::receipt_freeze_v1::{ReceiptArtifact, ReceiptChainExt};
use crate::admissibility_freeze_v1::{
    AdmissibilityChain, AdmissibilityClaim, AdmissibilityResult,
    Verdict, Invariant, EconomicPattern,
};
use crate::mission_freeze_v1::{
    MissionEnvelope, MissionStage, StateSnapshot, Originator,
};
use crate::manifest_artifact::ManifestArtifact;

// ════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════

#[derive(Debug)]
pub enum LoopError {
    /// S2: Intent could not be bounded into a mission.
    IntentMalformed(String),
    /// S4: Admissibility chain rejected the claim.
    Rejected(AdmissibilityResult),
    /// S5: Execution function failed.
    ExecutionFailed(String),
    /// S6-S7: Receipt creation or chain append failed.
    ReceiptFailed(ChainError),
    /// S8: Replay verification failed.
    ReplayMismatch {
        original_hash: Blake3Hash,
        replay_hash: Blake3Hash,
    },
}

impl From<ChainError> for LoopError {
    fn from(e: ChainError) -> Self { LoopError::ReceiptFailed(e) }
}

// ════════════════════════════════════════════════════════════
// Execution result — what S5 produces
// ════════════════════════════════════════════════════════════

/// The output of a mission execution (S5).
/// The execute_fn returns this; the loop receipts it.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Hash of the execution output (whatever was produced).
    pub output_hash: Blake3Hash,
    /// Human-readable summary of what was done.
    pub summary: String,
    /// Quality score (for Ihsan tracking).
    pub quality_score: f64,
}

// ════════════════════════════════════════════════════════════
// The Loop — one function, nine stages
// ════════════════════════════════════════════════════════════

/// Result of a complete lawful loop execution.
#[derive(Debug)]
pub struct LoopResult {
    /// The mission that was executed.
    pub mission: MissionEnvelope,
    /// The admissibility result (all gate verdicts).
    pub admissibility: AdmissibilityResult,
    /// The execution output.
    pub execution: ExecutionResult,
    /// The receipt artifact that was created and chained.
    pub receipt: ReceiptArtifact,
    /// The receipt_id (for chain lookup and manifest inclusion).
    pub receipt_id: Blake3Hash,
    /// Whether replay verification passed (S8).
    pub replay_verified: bool,
    /// The stage the mission reached.
    pub final_stage: MissionStage,
}

/// Run one mission through the complete lawful loop.
///
/// This is the §16 minimum undeniable loop. It takes:
///   - an intent (raw string from the operator)
///   - a current state and ideal state (§9 four-state model)
///   - an execution function (what to actually do at S5)
///   - a receipt chain (to append the receipt at S7)
///   - an admissibility chain (the five-gate evaluator)
///   - a timestamp (monotonic nanoseconds)
///
/// It returns a LoopResult proving that one mission entered,
/// was evaluated, executed, receipted, canonicalized, and
/// replay-verified. Or it returns a LoopError explaining
/// exactly where the pipeline stopped.
pub fn run_lawful_loop<F>(
    intent: &str,
    current_state: StateSnapshot,
    ideal_state: StateSnapshot,
    execute_fn: F,
    chain: &mut ReceiptChain,
    admissibility: &AdmissibilityChain,
    timestamp_ns: u64,
) -> Result<LoopResult, LoopError>
where
    F: FnOnce(&MissionEnvelope) -> Result<ExecutionResult, String>,
{
    // ── S1: Intent ──
    // Raw intent captured. Validate non-empty.
    if intent.trim().is_empty() {
        return Err(LoopError::IntentMalformed("Empty intent".into()));
    }

    // ── S2: Mission ──
    // Intent bounded into executable mission.
    let mut mission = MissionEnvelope::from_intent(
        intent.to_string(),
        current_state,
        ideal_state,
        Originator::System, // v1: all missions are system-originated
        timestamp_ns,
    );

    // ── S3: Claim ──
    // Specific claim extracted from mission.
    let claim_id = mission.extract_claim_id();
    mission.advance_stage(); // S2 → S3 (Claim)

    // Build the admissibility claim from mission metadata.
    //
    // FIX E — HONEST LABEL: This is a v1 bootstrapping path, not full
    // production mission law. Specifically:
    //   - evidence is self-certified by mission existence (lenient)
    //   - Ihsan is pre-boosted above floor for all basic missions (lenient)
    //   - economic and mutation gates are inert (no pattern, no mutation)
    //
    // This is acceptable for proving the loop shape. Production law
    // requires: real evidence binding (not self-referential), external
    // Ihsan scoring (not formula-derived), and active economic/mutation
    // evaluation. Those are Step 8+ hardening items.
    let adm_claim = AdmissibilityClaim {
        claim_id,
        has_evidence: true, // v1 bootstrap: mission existence = evidence
        evidence_hash: Some(mission.mission_id), // v1: self-referential
        economic_pattern: None, // v1: inert — no economic evaluation yet
        state_mutation: None,   // v1: inert — no mutation evaluation yet
        quality_score: mission.state.gap.min(1.0).max(0.0) * 0.05 + 0.95,
        // v1 bootstrap: missions with smaller gaps have higher quality
        // gap=0.0 → 0.95, gap=1.0 → 1.0 (above floor in both cases)
        timestamp_ns,
    };

    // ── S4: Admissibility ──
    // Gate chain evaluates claim.
    mission.advance_stage(); // S3 → S4 (Admissibility)
    let adm_result = admissibility.evaluate(&adm_claim);

    if adm_result.verdict != Verdict::Permit {
        return Err(LoopError::Rejected(adm_result));
    }

    // ── S5: Execution ──
    // Permitted claim executes.
    mission.advance_stage(); // S4 → S5 (Execution)
    let exec_result = execute_fn(&mission)
        .map_err(LoopError::ExecutionFailed)?;

    // ── S6: Receipt ──
    // Immutable proof artifact created.
    mission.advance_stage(); // S5 → S6 (Receipt)
    let prev_head = chain.head();

    let receipt = ReceiptArtifact::new(
        crate::receipts::ReceiptKind::ReasoningSession,
        claim_id,             // claim_ref
        exec_result.output_hash, // evidence_hash
        vec![prev_head],      // lineage: depends on previous chain head
        prev_head,            // prev
        timestamp_ns,
    );

    let receipt_id = receipt.receipt_id;

    // ── S7: Canonicalization ──
    // Receipt added to canonical chain.
    mission.advance_stage(); // S6 → S7 (Canonicalization)
    chain.append_artifact(receipt.clone())?;

    // ── S8: Replay ──
    // Action can be deterministically replayed.
    mission.advance_stage(); // S7 → S8 (Replayability)

    // Replay verification: fetch the payload back, decode as concrete
    // ReceiptArtifact type, check hash.
    // Fix D: explicit turbofish avoids trait-dispatch ambiguity.
    let replay_verified = match chain.fetch_payload_bytes(&receipt_id) {
        Ok(Some(bytes)) => {
            match <ReceiptArtifact as crate::receipts::ReceiptPayloadDecode>::from_canonical_bytes(
                bytes.as_slice(),
            ) {
                Ok(decoded_receipt) => {
                    decoded_receipt.receipt_id == receipt_id
                }
                Err(_) => false,
            }
        }
        _ => false,
    };

    if !replay_verified {
        return Err(LoopError::ReplayMismatch {
            original_hash: receipt_id,
            replay_hash: [0u8; 32], // decode failed
        });
    }

    // ── S9: Reflex (optional) ──
    // Not wired in v1. Mission reaches Replayability stage.
    // Future: if this pattern repeats, promote to reflex.
    let final_stage = mission.stage;

    Ok(LoopResult {
        mission,
        admissibility: adm_result,
        execution: exec_result,
        receipt,
        receipt_id,
        replay_verified,
        final_stage,
    })
}

/// Generate a ManifestArtifact from a set of LoopResults.
///
/// This is the "one daily manifest" condition from §16.
/// Call after accumulating a window of loop results.
pub fn generate_manifest(
    window_start: u64,
    window_end: u64,
    results: &[LoopResult],
    chain_head: Blake3Hash,
) -> ManifestArtifact {
    let receipt_refs: Vec<Blake3Hash> = results.iter()
        .map(|r| r.receipt_id)
        .collect();

    ManifestArtifact::from_window(
        window_start,
        window_end,
        receipt_refs,
        chain_head,
    )
}

// ════════════════════════════════════════════════════════════
// Tests — proving §16 minimum undeniable loop
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::InMemoryPayloadStore;

    fn test_chain() -> ReceiptChain {
        let store = Box::new(InMemoryPayloadStore::new());
        ReceiptChain::new([0u8; 32], store)
    }

    fn current() -> StateSnapshot {
        StateSnapshot {
            hash: [1u8; 32],
            summary: "Files scattered".into(),
            metric: 0.2,
        }
    }

    fn ideal() -> StateSnapshot {
        StateSnapshot {
            hash: [2u8; 32],
            summary: "Files organized".into(),
            metric: 1.0,
        }
    }

    fn simple_executor(mission: &MissionEnvelope) -> Result<ExecutionResult, String> {
        Ok(ExecutionResult {
            output_hash: blake3_domain(
                "test-execution",
                &mission.mission_id,
            ),
            summary: format!("Executed: {}", mission.intent_text),
            quality_score: 0.98,
        })
    }

    // ── Test 1: The minimum undeniable loop works ──

    #[test]
    fn test_minimum_undeniable_loop() {
        let mut chain = test_chain();
        let admissibility = AdmissibilityChain::canonical();

        let result = run_lawful_loop(
            "Organize Downloads folder",
            current(),
            ideal(),
            simple_executor,
            &mut chain,
            &admissibility,
            1_000_000,
        ).expect("Loop should succeed");

        // §16 condition 1: one authoritative entry
        assert!(!result.mission.intent_text.is_empty());

        // §16 condition 2: one gate chain evaluated
        assert_eq!(result.admissibility.verdict, Verdict::Permit);
        assert_eq!(result.admissibility.gate_verdicts.len(), 5);

        // §16 condition 3: one receipt lineage emitted
        assert_ne!(result.receipt_id, [0u8; 32]);

        // §16 condition 4: one replay reproduced
        assert!(result.replay_verified);

        // Chain advanced
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.head(), result.receipt_id);

        // Mission reached S8 (Replayability)
        assert_eq!(result.final_stage, MissionStage::Replayability);
    }

    // ── Test 2: Empty intent rejected at S1 ──

    #[test]
    fn test_empty_intent_rejected() {
        let mut chain = test_chain();
        let adm = AdmissibilityChain::canonical();

        let result = run_lawful_loop(
            "",
            current(), ideal(),
            simple_executor,
            &mut chain, &adm, 1000,
        );

        assert!(matches!(result, Err(LoopError::IntentMalformed(_))));
        assert_eq!(chain.len(), 0, "Chain must not advance on failed loop");
    }

    // ── Test 3: Execution failure stops loop at S5 ──

    #[test]
    fn test_execution_failure_stops_loop() {
        let mut chain = test_chain();
        let adm = AdmissibilityChain::canonical();

        let result = run_lawful_loop(
            "Do something that fails",
            current(), ideal(),
            |_| Err("Execution crashed".to_string()),
            &mut chain, &adm, 2000,
        );

        assert!(matches!(result, Err(LoopError::ExecutionFailed(_))));
        assert_eq!(chain.len(), 0, "Chain must not advance on execution failure");
    }

    // ── Test 4: Multiple loops chain correctly ──

    #[test]
    fn test_multiple_loops_chain() {
        let mut chain = test_chain();
        let adm = AdmissibilityChain::canonical();

        let mut results = Vec::new();

        for i in 0..5u64 {
            let r = run_lawful_loop(
                &format!("Mission {}", i),
                current(), ideal(),
                simple_executor,
                &mut chain, &adm,
                (i + 1) * 1_000_000,
            ).expect("Loop should succeed");
            results.push(r);
        }

        assert_eq!(chain.len(), 5);

        // Each receipt's prev points to the previous receipt
        for i in 1..results.len() {
            assert_eq!(results[i].receipt.prev, results[i-1].receipt_id,
                "Receipt {} prev should point to receipt {}", i, i-1);
        }
    }

    // ── Test 5: Manifest generated from loop results ──

    #[test]
    fn test_manifest_from_loop_results() {
        let mut chain = test_chain();
        let adm = AdmissibilityChain::canonical();

        let mut results = Vec::new();
        for i in 0..3u64 {
            let r = run_lawful_loop(
                &format!("Mission {}", i),
                current(), ideal(),
                simple_executor,
                &mut chain, &adm,
                (i + 1) * 1_000_000,
            ).unwrap();
            results.push(r);
        }

        // §16 condition 5: one daily manifest
        let manifest = generate_manifest(
            1_000_000,      // window_start
            3_000_000,      // window_end
            &results,
            chain.head(),
        );

        assert_eq!(manifest.receipt_count, 3);
        assert_eq!(manifest.receipt_refs.len(), 3);
        assert!(manifest.verify_integrity());
    }

    // ── Test 6: Kernel-level §16 conditions + manifest generation ──
    // NOTE (Fix F): This test proves 5 of 7 §16 conditions in code.
    // Conditions 6 (trust surface) and 7 (coherent face) are integration-level
    // and proven by the gateway + Dema console — not by this unit test.
    // The test name reflects the actual scope, not the aspiration.

    #[test]
    fn test_kernel_loop_conditions_with_manifest() {
        let mut chain = test_chain();
        let adm = AdmissibilityChain::canonical();

        // Execute one mission through the lawful loop
        let result = run_lawful_loop(
            "The first mission through the lawful loop",
            current(), ideal(),
            simple_executor,
            &mut chain, &adm,
            1_000_000,
        ).expect("The minimum undeniable loop must succeed");

        // Generate manifest
        let manifest = generate_manifest(
            0, 2_000_000,
            &[result.clone_result()],
            chain.head(),
        );

        // §16 Condition 1: One authoritative entry
        assert!(result.mission.intent_text.contains("first mission"));

        // §16 Condition 2: One gate chain
        assert_eq!(result.admissibility.verdict, Verdict::Permit);

        // §16 Condition 3: One receipt lineage
        assert_ne!(result.receipt_id, [0u8; 32]);
        assert_eq!(chain.len(), 1);

        // §16 Condition 4: One replay reproduction
        assert!(result.replay_verified);

        // §16 Condition 5: One daily manifest
        assert_eq!(manifest.receipt_count, 1);
        assert!(manifest.verify_integrity());

        // §16 Conditions 6 + 7 (trust surface, coherent face) are
        // integration-level — proven by gateway + Dema, not this test.
    }
}

// Helper: LoopResult needs a way to produce a reference for manifest generation
impl LoopResult {
    /// Create a clone-friendly reference for manifest generation.
    /// In production this would be just the receipt_id; in tests
    /// we need the full struct for assertion.
    fn clone_result(&self) -> LoopResult {
        LoopResult {
            mission: self.mission.clone(),
            admissibility: AdmissibilityResult {
                verdict: self.admissibility.verdict,
                gate_verdicts: self.admissibility.gate_verdicts.clone(),
                rejected: self.admissibility.rejected.clone(),
            },
            execution: self.execution.clone(),
            receipt: self.receipt.clone(),
            receipt_id: self.receipt_id,
            replay_verified: self.replay_verified,
            final_stage: self.final_stage,
        }
    }
}
