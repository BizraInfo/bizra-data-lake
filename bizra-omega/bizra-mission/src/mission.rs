// bizra-mission/src/mission.rs
// ============================================================
// Mission — the governed lifecycle of a cognitive operation
// ============================================================

use serde::{Deserialize, Serialize};

use crate::{
    preflight::PreflightResult,
    receipt::MissionReceipt,
    state::{DegradationReason, FailureCode, MissionState, StateTransition},
};

/// A unique mission identifier (BLAKE3 hash, 32 bytes).
pub type MissionId = [u8; 32];

/// The Mission — every cognitive operation in BIZRA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mission {
    pub mission_id: MissionId,
    pub submitted_at: u64,
    pub completed_at: Option<u64>,
    pub state: MissionState,
    pub state_history: Vec<StateTransition>,
    pub timeout_budget_ms: u64,
    // Content (hashed — raw content stays in caller)
    pub input_content_hash: [u8; 32],
    // Cognition results (filled during lifecycle)
    pub chosen_model: Option<String>,
    pub preflight: Option<PreflightResult>,
    pub ihsan_score: Option<f32>,
    pub snr_score: Option<f32>,
    pub guardian_approved: Option<bool>,
    // Output
    pub response_hash: Option<[u8; 32]>,
    pub receipt: Option<MissionReceipt>,
    // Failure / degradation
    pub failure_code: Option<FailureCode>,
    pub degradation_reasons: Vec<DegradationReason>,
    // Chain link — hash of the previous receipt for tamper-evident ordering
    pub previous_receipt_hash: Option<[u8; 32]>,
}

impl Mission {
    /// Create a new mission from user input hash.
    pub fn new(input_content_hash: [u8; 32], now: u64) -> Self {
        let mission_id =
            blake3::hash(&[&input_content_hash[..], &now.to_le_bytes()].concat()).into();
        Self {
            mission_id,
            submitted_at: now,
            completed_at: None,
            state: MissionState::Submitted,
            state_history: vec![],
            timeout_budget_ms: 120_000,
            input_content_hash,
            chosen_model: None,
            preflight: None,
            ihsan_score: None,
            snr_score: None,
            guardian_approved: None,
            response_hash: None,
            receipt: None,
            failure_code: None,
            degradation_reasons: vec![],
            previous_receipt_hash: None,
        }
    }

    /// Chain this mission to the previous receipt (for tamper-evident ordering).
    pub fn chain_to(&mut self, previous_receipt_id: [u8; 32]) {
        self.previous_receipt_hash = Some(previous_receipt_id);
    }

    /// Transition to a new state. Returns Err if the transition is unconstitutional.
    pub fn transition(
        &mut self,
        to: MissionState,
        now: u64,
        reason: &str,
    ) -> Result<(), TransitionError> {
        if self.state.is_terminal() {
            return Err(TransitionError::AlreadyTerminal(self.state));
        }
        if !self.state.can_transition_to(to) {
            return Err(TransitionError::IllegalTransition {
                from: self.state,
                to,
            });
        }
        self.state_history.push(StateTransition {
            from: self.state,
            to,
            at: now,
            reason: reason.to_string(),
        });
        self.state = to;
        if to.is_terminal() {
            self.completed_at = Some(now);
        }
        Ok(())
    }

    /// Fail the mission immediately with a failure code. Emits receipt.
    pub fn fail(&mut self, code: FailureCode, now: u64) -> Result<(), TransitionError> {
        let reason = format!("failed: {code:?}");
        self.failure_code = Some(code);
        self.transition(MissionState::Failed, now, &reason)?;
        self.receipt = Some(MissionReceipt::from_mission(
            self,
            self.previous_receipt_hash,
        ));
        Ok(())
    }

    /// Degrade the mission with reasons. Emits receipt.
    pub fn degrade(
        &mut self,
        reasons: Vec<DegradationReason>,
        now: u64,
    ) -> Result<(), TransitionError> {
        self.degradation_reasons.extend(reasons);
        self.transition(MissionState::Degraded, now, "degraded")?;
        self.receipt = Some(MissionReceipt::from_mission(
            self,
            self.previous_receipt_hash,
        ));
        Ok(())
    }

    /// Complete the mission successfully. Emits receipt.
    pub fn complete(&mut self, now: u64) -> Result<(), TransitionError> {
        self.transition(MissionState::Complete, now, "complete")?;
        self.receipt = Some(MissionReceipt::from_mission(
            self,
            self.previous_receipt_hash,
        ));
        Ok(())
    }

    /// Total elapsed time in milliseconds.
    pub fn elapsed_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.submitted_at).saturating_mul(1000)
    }

    /// Has the mission exceeded its timeout budget?
    pub fn is_timed_out(&self, now: u64) -> bool {
        self.elapsed_ms(now) > self.timeout_budget_ms
    }

    /// The mission ID as a hex string.
    pub fn id_hex(&self) -> String {
        hex::encode(self.mission_id)
    }
}

/// Errors from illegal state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    IllegalTransition {
        from: MissionState,
        to: MissionState,
    },
    AlreadyTerminal(MissionState),
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllegalTransition { from, to } => {
                write!(f, "illegal transition: {:?} -> {:?}", from, to)
            }
            Self::AlreadyTerminal(s) => write!(f, "mission already terminal: {:?}", s),
        }
    }
}

impl std::error::Error for TransitionError {}

/// Hex encoding helper (no external dep).
mod hex {
    pub fn encode(bytes: [u8; 32]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
