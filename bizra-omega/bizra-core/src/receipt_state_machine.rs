//! ReceiptStateMachine v1 — Transition Law for CanonicalReceipt
//!
//! The authoritative rules governing receipt lifecycle transitions.
//! No surface (UI, API, log, federation) may render a receipt in
//! a state that this machine cannot reach through valid transitions.
//!
//! States:
//!   HYPOTHESIS → VERIFIED → EXECUTABLE → COMMITTED → REPLAYABLE → MARKETABLE
//!
//! Every transition requires evidence. No silent state changes.
//! Failed transitions are logged, not swallowed.

use serde::{Deserialize, Serialize};

use crate::canonical_receipt::ReceiptState;

/// Evidence required to justify a state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionEvidence {
    /// Gate chain evaluation completed with verdict.
    GateVerdict {
        /// Whether the gate admitted the mission.
        admitted: bool,
        /// Ihsan quality score from gate evaluation.
        ihsan: f64,
    },
    /// Constitutional gate approved execution.
    ExecutionApproved,
    /// Effects committed, receipt signed.
    ReceiptSealed {
        /// Hex-encoded receipt identifier.
        receipt_id_hex: String,
    },
    /// Replay path verified (deterministic re-execution matches).
    ReplayVerified,
    /// Published to marketplace with PoI attestation.
    MarketplacePublished {
        /// Proof-of-Impact score for marketplace listing.
        poi_score: f64,
    },
}

/// Result of a transition attempt.
#[derive(Debug, Clone)]
pub enum TransitionResult {
    /// Transition succeeded.
    Ok(ReceiptState),
    /// Transition denied — invalid from current state.
    Denied {
        /// Source state of the denied transition.
        from: ReceiptState,
        /// Target state of the denied transition.
        to: ReceiptState,
        /// Human-readable reason for denial.
        reason: &'static str,
    },
    /// Transition denied — evidence insufficient.
    InsufficientEvidence {
        /// Description of the evidence that was required.
        required: &'static str,
    },
}

impl TransitionResult {
    /// Returns true if the transition succeeded.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok(_))
    }
}

/// The receipt state machine — constitutional transition law.
pub struct ReceiptStateMachine;

impl ReceiptStateMachine {
    /// Attempt a state transition with evidence.
    pub fn transition(
        current: ReceiptState,
        target: ReceiptState,
        evidence: &TransitionEvidence,
    ) -> TransitionResult {
        match (current, target) {
            // HYPOTHESIS → VERIFIED: gate chain must have evaluated
            (ReceiptState::Hypothesis, ReceiptState::Verified) => match evidence {
                TransitionEvidence::GateVerdict { .. } => TransitionResult::Ok(target),
                _ => TransitionResult::InsufficientEvidence {
                    required: "GateVerdict (gate chain must evaluate before verification)",
                },
            },

            // VERIFIED → EXECUTABLE: gates must have admitted
            (ReceiptState::Verified, ReceiptState::Executable) => match evidence {
                TransitionEvidence::GateVerdict {
                    admitted: true,
                    ihsan,
                } if *ihsan >= 0.85 => TransitionResult::Ok(target),
                TransitionEvidence::GateVerdict {
                    admitted: false, ..
                } => TransitionResult::Denied {
                    from: current,
                    to: target,
                    reason: "Gate verdict rejected — cannot become executable",
                },
                TransitionEvidence::GateVerdict { ihsan, .. } if *ihsan < 0.85 => {
                    TransitionResult::Denied {
                        from: current,
                        to: target,
                        reason: "Ihsan below 0.85 — constitutional floor not met",
                    }
                }
                _ => TransitionResult::InsufficientEvidence {
                    required: "GateVerdict with admitted=true and ihsan>=0.85",
                },
            },

            // EXECUTABLE → COMMITTED: receipt must be sealed
            (ReceiptState::Executable, ReceiptState::Committed) => match evidence {
                TransitionEvidence::ReceiptSealed { .. } => TransitionResult::Ok(target),
                _ => TransitionResult::InsufficientEvidence {
                    required: "ReceiptSealed (Ed25519 signature + chain link)",
                },
            },

            // COMMITTED → REPLAYABLE: replay path must be verified
            (ReceiptState::Committed, ReceiptState::Replayable) => match evidence {
                TransitionEvidence::ReplayVerified => TransitionResult::Ok(target),
                _ => TransitionResult::InsufficientEvidence {
                    required: "ReplayVerified (deterministic re-execution matches)",
                },
            },

            // REPLAYABLE → MARKETABLE: PoI attestation required
            (ReceiptState::Replayable, ReceiptState::Marketable) => match evidence {
                TransitionEvidence::MarketplacePublished { poi_score } if *poi_score > 0.0 => {
                    TransitionResult::Ok(target)
                }
                _ => TransitionResult::InsufficientEvidence {
                    required: "MarketplacePublished with poi_score > 0",
                },
            },

            // All other transitions are invalid
            _ => TransitionResult::Denied {
                from: current,
                to: target,
                reason: "No valid transition path exists between these states",
            },
        }
    }

    /// Check if a transition is structurally possible (ignoring evidence).
    pub fn can_transition(from: ReceiptState, to: ReceiptState) -> bool {
        matches!(
            (from, to),
            (ReceiptState::Hypothesis, ReceiptState::Verified)
                | (ReceiptState::Verified, ReceiptState::Executable)
                | (ReceiptState::Executable, ReceiptState::Committed)
                | (ReceiptState::Committed, ReceiptState::Replayable)
                | (ReceiptState::Replayable, ReceiptState::Marketable)
        )
    }

    /// Get the next valid state in the forward path.
    pub fn next_state(current: ReceiptState) -> Option<ReceiptState> {
        match current {
            ReceiptState::Hypothesis => Some(ReceiptState::Verified),
            ReceiptState::Verified => Some(ReceiptState::Executable),
            ReceiptState::Executable => Some(ReceiptState::Committed),
            ReceiptState::Committed => Some(ReceiptState::Replayable),
            ReceiptState::Replayable => Some(ReceiptState::Marketable),
            ReceiptState::Marketable => None, // Terminal
        }
    }

    /// Check if a state is terminal.
    pub fn is_terminal(state: ReceiptState) -> bool {
        matches!(state, ReceiptState::Marketable)
    }

    /// Count transitions needed to reach target from current.
    /// Returns None if target is before current.
    pub fn distance(from: ReceiptState, to: ReceiptState) -> Option<u8> {
        let ord = |s: ReceiptState| -> u8 {
            match s {
                ReceiptState::Hypothesis => 0,
                ReceiptState::Verified => 1,
                ReceiptState::Executable => 2,
                ReceiptState::Committed => 3,
                ReceiptState::Replayable => 4,
                ReceiptState::Marketable => 5,
            }
        };
        let f = ord(from);
        let t = ord(to);
        if t >= f {
            Some(t - f)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_happy_path_full_lifecycle() {
        let mut state = ReceiptState::Hypothesis;

        let r = ReceiptStateMachine::transition(
            state,
            ReceiptState::Verified,
            &TransitionEvidence::GateVerdict {
                admitted: true,
                ihsan: 0.97,
            },
        );
        assert!(r.is_ok());
        state = ReceiptState::Verified;

        let r = ReceiptStateMachine::transition(
            state,
            ReceiptState::Executable,
            &TransitionEvidence::GateVerdict {
                admitted: true,
                ihsan: 0.97,
            },
        );
        assert!(r.is_ok());
        state = ReceiptState::Executable;

        let r = ReceiptStateMachine::transition(
            state,
            ReceiptState::Committed,
            &TransitionEvidence::ReceiptSealed {
                receipt_id_hex: "abc123".into(),
            },
        );
        assert!(r.is_ok());
        state = ReceiptState::Committed;

        let r = ReceiptStateMachine::transition(
            state,
            ReceiptState::Replayable,
            &TransitionEvidence::ReplayVerified,
        );
        assert!(r.is_ok());
        state = ReceiptState::Replayable;

        let r = ReceiptStateMachine::transition(
            state,
            ReceiptState::Marketable,
            &TransitionEvidence::MarketplacePublished { poi_score: 0.95 },
        );
        assert!(r.is_ok());
    }

    #[test]
    fn test_rejected_cannot_become_executable() {
        let r = ReceiptStateMachine::transition(
            ReceiptState::Verified,
            ReceiptState::Executable,
            &TransitionEvidence::GateVerdict {
                admitted: false,
                ihsan: 0.30,
            },
        );
        assert!(!r.is_ok());
    }

    #[test]
    fn test_low_ihsan_blocks_execution() {
        let r = ReceiptStateMachine::transition(
            ReceiptState::Verified,
            ReceiptState::Executable,
            &TransitionEvidence::GateVerdict {
                admitted: true,
                ihsan: 0.70,
            },
        );
        assert!(!r.is_ok());
    }

    #[test]
    fn test_skip_states_denied() {
        let r = ReceiptStateMachine::transition(
            ReceiptState::Hypothesis,
            ReceiptState::Committed,
            &TransitionEvidence::ReceiptSealed {
                receipt_id_hex: "x".into(),
            },
        );
        assert!(!r.is_ok());
    }

    #[test]
    fn test_wrong_evidence_type_denied() {
        let r = ReceiptStateMachine::transition(
            ReceiptState::Hypothesis,
            ReceiptState::Verified,
            &TransitionEvidence::ReplayVerified, // Wrong evidence
        );
        assert!(!r.is_ok());
    }

    #[test]
    fn test_backward_transition_denied() {
        let r = ReceiptStateMachine::transition(
            ReceiptState::Committed,
            ReceiptState::Hypothesis,
            &TransitionEvidence::GateVerdict {
                admitted: true,
                ihsan: 0.99,
            },
        );
        assert!(!r.is_ok());
    }

    #[test]
    fn test_can_transition() {
        assert!(ReceiptStateMachine::can_transition(
            ReceiptState::Hypothesis,
            ReceiptState::Verified
        ));
        assert!(!ReceiptStateMachine::can_transition(
            ReceiptState::Hypothesis,
            ReceiptState::Committed
        ));
        assert!(!ReceiptStateMachine::can_transition(
            ReceiptState::Marketable,
            ReceiptState::Hypothesis
        ));
    }

    #[test]
    fn test_next_state() {
        assert_eq!(
            ReceiptStateMachine::next_state(ReceiptState::Hypothesis),
            Some(ReceiptState::Verified)
        );
        assert_eq!(
            ReceiptStateMachine::next_state(ReceiptState::Marketable),
            None
        );
    }

    #[test]
    fn test_distance() {
        assert_eq!(
            ReceiptStateMachine::distance(ReceiptState::Hypothesis, ReceiptState::Marketable),
            Some(5)
        );
        assert_eq!(
            ReceiptStateMachine::distance(ReceiptState::Committed, ReceiptState::Committed),
            Some(0)
        );
        assert_eq!(
            ReceiptStateMachine::distance(ReceiptState::Marketable, ReceiptState::Hypothesis),
            None
        );
    }

    #[test]
    fn test_zero_poi_blocks_marketplace() {
        let r = ReceiptStateMachine::transition(
            ReceiptState::Replayable,
            ReceiptState::Marketable,
            &TransitionEvidence::MarketplacePublished { poi_score: 0.0 },
        );
        assert!(!r.is_ok());
    }
}
