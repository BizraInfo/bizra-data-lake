// bizra-mission/src/lib.rs
// ============================================================
// Sovereign Mission Control Plane
// ============================================================
//
// "The model is the rendering engine. The URP is the game server.
//  The mission contract is the quest system."
//
// This crate governs the complete lifecycle of every cognitive
// operation in BIZRA. The state machine is derived from:
//   docs/contracts/mission_lifecycle.json
//   docs/contracts/mission_envelope.json
//   docs/contracts/urp_canonical_receipt.json
//   docs/contracts/capability_negotiation.json
//   docs/contracts/degraded_experience.json
//
// Standing on Giants:
// - MMORPG industry (20 years): quest contracts are client-agnostic
// - Lamport (1974): state machine replication
// - Al-Ghazali: intent (niyyah) as structural precondition
// - Deming: variation reduction through governed process
// ============================================================

/// The canonical loop — production pipeline wiring all 4 contracts.
pub mod canonical_loop;
/// Cross-layer envelope — canonical mission contract with constitutional context.
pub mod envelope;
/// Evidence manifest — bundles receipts into reviewable, integrity-checked package.
pub mod manifest;
pub mod mission;
pub mod preflight;
pub mod receipt;
pub mod state;

#[cfg(test)]
mod tests {
    use crate::{
        mission::Mission,
        preflight::{self, Capability, PreflightResult},
        state::{DegradationReason, FailureCode, MissionState},
    };

    fn now() -> u64 {
        1773662000
    }

    // ========================================================
    // ADR-001 ACCEPTANCE TEST 1:
    // test_queued_mission_completes
    // Happy path through all states to completion.
    // ========================================================
    #[test]
    fn test_queued_mission_completes() {
        let input_hash = blake3::hash(b"What are BIZRA constitutional thresholds?").into();
        let mut m = Mission::new(input_hash, now());

        assert_eq!(m.state, MissionState::Submitted);

        // Preflight
        let pf = preflight::run_preflight(&[Capability::Chat], &["qwen2.5:3b".to_string()], None);
        assert!(pf.passed());
        m.preflight = Some(pf);

        // Walk through the full lifecycle
        m.transition(MissionState::Queued, now() + 1, "capacity available")
            .unwrap();
        m.transition(MissionState::WarmingRetrieval, now() + 2, "FAISS loading")
            .unwrap();
        m.transition(MissionState::WarmingModel, now() + 3, "model loading")
            .unwrap();
        m.transition(MissionState::Retrieving, now() + 4, "semantic search")
            .unwrap();
        m.transition(MissionState::Routing, now() + 5, "Navigator classified")
            .unwrap();
        m.chosen_model = Some("qwen2.5:3b".to_string());
        m.transition(MissionState::Running, now() + 6, "inference started")
            .unwrap();
        m.ihsan_score = Some(0.95);
        m.snr_score = Some(0.92);
        m.guardian_approved = Some(true);
        m.transition(MissionState::Scoring, now() + 7, "scored")
            .unwrap();
        m.transition(MissionState::Persisting, now() + 8, "persisting receipt")
            .unwrap();
        m.complete(now() + 9).unwrap();

        // Verify terminal state
        assert_eq!(m.state, MissionState::Complete);
        assert!(m.state.is_terminal());
        assert!(m.completed_at.is_some());

        // Verify receipt was emitted
        let receipt = m
            .receipt
            .as_ref()
            .expect("receipt must be emitted on complete");
        assert!(receipt.is_success());
        assert!(receipt.verify_hash());
        assert_eq!(receipt.degradation_tier, 0); // full quality
        assert!(receipt.failure_code.is_none());
        assert!(receipt.degradation_reasons.is_empty());

        // Verify state history is complete
        assert_eq!(m.state_history.len(), 9); // submitted->...->complete = 9 transitions
    }

    // ========================================================
    // ADR-001 ACCEPTANCE TEST 2:
    // test_missing_model_fails_at_preflight
    // Model not installed → immediate failure with receipt.
    // ========================================================
    #[test]
    fn test_missing_model_fails_at_preflight() {
        let input_hash = blake3::hash(b"Generate an image of a sunset").into();
        let mut m = Mission::new(input_hash, now());

        // Preflight: requires vision but no vision model installed
        let pf = preflight::run_preflight(
            &[Capability::Vision],
            &["qwen2.5:3b".to_string(), "mistral:latest".to_string()],
            None,
        );
        assert!(!pf.passed());
        assert!(pf.chosen_model().is_none());
        m.preflight = Some(pf);

        // Mission fails at submission — never enters queue
        m.fail(FailureCode::CapabilityNotAvailable, now() + 1)
            .unwrap();

        assert_eq!(m.state, MissionState::Failed);
        assert!(m.state.is_terminal());

        // Receipt must still be emitted on failure
        let receipt = m
            .receipt
            .as_ref()
            .expect("receipt must be emitted on failure");
        assert!(!receipt.is_success());
        assert!(receipt.verify_hash());
        assert_eq!(receipt.degradation_tier, 4); // refused
        assert_eq!(
            receipt.failure_code,
            Some(FailureCode::CapabilityNotAvailable)
        );
    }

    // ========================================================
    // ADR-001 ACCEPTANCE TEST 3:
    // test_degraded_mode_receipt_emission
    // Retrieval timeout → degraded state with receipt + reasons.
    // ========================================================
    #[test]
    fn test_degraded_mode_receipt_emission() {
        let input_hash = blake3::hash(b"What is the Adl invariant?").into();
        let mut m = Mission::new(input_hash, now());

        // Walk through lifecycle until retrieval fails
        m.transition(MissionState::Queued, now() + 1, "queued")
            .unwrap();
        m.transition(MissionState::WarmingRetrieval, now() + 2, "FAISS warmup")
            .unwrap();

        // Retrieval times out → degrade, don't fail
        m.degrade(
            vec![
                DegradationReason::RetrievalSkipped,
                DegradationReason::EmptyContext,
            ],
            now() + 12, // 10 seconds later — retrieval timed out
        )
        .unwrap();

        assert_eq!(m.state, MissionState::Degraded);
        assert!(m.state.is_terminal());

        // Receipt must be emitted with degradation reasons
        let receipt = m
            .receipt
            .as_ref()
            .expect("receipt must be emitted on degraded");
        assert!(receipt.is_degraded());
        assert!(!receipt.is_success());
        assert!(receipt.verify_hash());
        assert_eq!(receipt.degradation_tier, 2); // significant (2 reasons)
        assert_eq!(receipt.degradation_reasons.len(), 2);
        assert!(receipt
            .degradation_reasons
            .contains(&DegradationReason::RetrievalSkipped));
        assert!(receipt
            .degradation_reasons
            .contains(&DegradationReason::EmptyContext));
        assert!(receipt.failure_code.is_none()); // degraded, not failed
    }

    // ========================================================
    // ADDITIONAL: Illegal transition is rejected
    // ========================================================
    #[test]
    fn test_illegal_transition_rejected() {
        let input_hash = blake3::hash(b"test").into();
        let mut m = Mission::new(input_hash, now());
        // Cannot jump from Submitted directly to Running
        let err = m.transition(MissionState::Running, now() + 1, "skip");
        assert!(err.is_err());
        assert_eq!(m.state, MissionState::Submitted); // unchanged
    }

    // ========================================================
    // ADDITIONAL: Terminal state cannot transition further
    // ========================================================
    #[test]
    fn test_terminal_state_cannot_transition() {
        let input_hash = blake3::hash(b"test").into();
        let mut m = Mission::new(input_hash, now());
        m.fail(FailureCode::ModelNotAvailable, now() + 1).unwrap();
        let err = m.transition(MissionState::Queued, now() + 2, "retry");
        assert!(err.is_err());
    }

    // ========================================================
    // ADDITIONAL: Preflight with fallback model
    // ========================================================
    #[test]
    fn test_preflight_fallback() {
        let pf = preflight::run_preflight(
            &[Capability::Chat],
            &["qwen2.5:3b".to_string()],
            Some("gpt-4-not-installed"),
        );
        assert!(pf.passed());
        assert_eq!(pf.chosen_model(), Some("qwen2.5:3b"));
        assert!(matches!(pf, PreflightResult::FallbackUsed { .. }));
    }

    // ========================================================
    // ADDITIONAL: Receipt hash integrity
    // ========================================================
    #[test]
    fn test_receipt_hash_integrity() {
        let input_hash = blake3::hash(b"integrity test").into();
        let mut m = Mission::new(input_hash, now());
        m.transition(MissionState::Queued, now() + 1, "q").unwrap();
        m.transition(MissionState::WarmingRetrieval, now() + 2, "w")
            .unwrap();
        m.transition(MissionState::WarmingModel, now() + 3, "w")
            .unwrap();
        m.transition(MissionState::Retrieving, now() + 4, "r")
            .unwrap();
        m.transition(MissionState::Routing, now() + 5, "r").unwrap();
        m.transition(MissionState::Running, now() + 6, "r").unwrap();
        m.transition(MissionState::Scoring, now() + 7, "s").unwrap();
        m.transition(MissionState::Persisting, now() + 8, "p")
            .unwrap();
        m.complete(now() + 9).unwrap();
        let receipt = m.receipt.as_ref().unwrap();
        assert!(receipt.verify_hash());
        // Tampering detection
        let mut tampered = receipt.clone();
        tampered.mission_id[0] ^= 0xFF;
        assert!(!tampered.verify_hash());
    }

    // ========================================================
    // ADDITIONAL: Offline reconciliation flow
    // Phone executes offline → awaiting_reconciliation →
    // connectivity returns → urp_validating → complete
    // ========================================================
    #[test]
    fn test_offline_reconciliation_flow() {
        let input_hash = blake3::hash(b"offline question on the bus").into();
        let mut m = Mission::new(input_hash, now());

        // Phone executes locally while offline
        m.transition(MissionState::Queued, now() + 1, "queued")
            .unwrap();
        m.transition(MissionState::WarmingRetrieval, now() + 2, "warmup")
            .unwrap();
        m.transition(MissionState::WarmingModel, now() + 3, "model load")
            .unwrap();
        m.transition(MissionState::Retrieving, now() + 4, "search")
            .unwrap();
        m.transition(MissionState::Routing, now() + 5, "route")
            .unwrap();
        m.chosen_model = Some("qwen2.5-0.5b".to_string());
        m.transition(MissionState::Running, now() + 6, "inference")
            .unwrap();
        m.transition(MissionState::Scoring, now() + 7, "scored")
            .unwrap();
        m.transition(MissionState::Persisting, now() + 8, "persisted locally")
            .unwrap();

        // No connectivity → deferred settlement
        m.transition(
            MissionState::AwaitingReconciliation,
            now() + 9,
            "offline, URP unreachable",
        )
        .unwrap();
        assert_eq!(m.state, MissionState::AwaitingReconciliation);
        assert!(!m.state.is_terminal()); // Can still transition
        assert!(m.state.is_deferred());

        // Connectivity returns → URP validates
        m.transition(
            MissionState::UrpValidating,
            now() + 3600,
            "connectivity restored",
        )
        .unwrap();
        m.ihsan_score = Some(0.88);
        m.snr_score = Some(0.85);
        m.guardian_approved = Some(true);
        m.complete(now() + 3601).unwrap();

        assert_eq!(m.state, MissionState::Complete);
        let receipt = m.receipt.as_ref().unwrap();
        assert!(receipt.is_success());
        assert!(receipt.verify_hash());
        // State history includes the reconciliation gap
        assert!(m
            .state_history
            .iter()
            .any(|t| t.to == MissionState::AwaitingReconciliation));
    }
}
