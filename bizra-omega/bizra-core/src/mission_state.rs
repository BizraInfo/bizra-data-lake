//! MissionState v1 — Sovereign Mission Lifecycle
//!
//! A mission is the atomic unit of human intent in BIZRA.
//! It enters the system as PROPOSED, passes through constitutional
//! gates, executes under PAT/SAT governance, and exits as a
//! CanonicalReceipt.
//!
//! MissionState is NOT model-state. The model is a replaceable
//! brain inside the membrane. The mission is the sovereign object
//! that carries human intent through the constitutional pipeline.
//!
//! Lifecycle:
//!   PROPOSED → ADMITTED → DECOMPOSED → EXECUTING → COMPLETED → RECEIPTED
//!          ↘ REJECTED (at any gate)
//!          ↘ DEGRADED (fallback path)
//!          ↘ TIMED_OUT (deadline exceeded)
//!
//! Standing on Giants:
//!   - Boyd (1976): OODA loop — mission IS the full cycle
//!   - Garcia-Molina (1987): Sagas — mission as compensable transaction
//!   - Al-Ghazali: intent (niyyah) precedes action

use blake3::Hasher;
use serde::{Deserialize, Serialize};

/// Domain prefix for mission hashing.
pub const DOMAIN_MISSION: &str = "bizra-mission-v1";

/// Mission lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissionPhase {
    /// Human intent received, not yet evaluated.
    Proposed,
    /// Constitutional gates passed, execution authorized.
    Admitted,
    /// PAT-7 has decomposed intent into subtasks.
    Decomposed,
    /// Subtasks executing under governance.
    Executing,
    /// All subtasks completed, awaiting receipt seal.
    Completed,
    /// Receipt sealed and chained.
    Receipted,
    /// Constitutional gate rejected the mission.
    Rejected,
    /// Executing in reduced capability mode.
    Degraded,
    /// Execution deadline exceeded.
    TimedOut,
}

impl MissionPhase {
    /// Whether this phase represents a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Receipted | Self::Rejected | Self::TimedOut)
    }

    /// Whether execution is still in progress.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Admitted | Self::Decomposed | Self::Executing | Self::Degraded
        )
    }
}

/// Complexity tier determined by intent classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityTier {
    /// Simple lookup or reflex — O(1), no LLM needed.
    Reflex,
    /// Single-agent task — one PAT agent suffices.
    SingleAgent,
    /// Multi-agent coordination — PAT-7 decomposition required.
    MultiAgent,
    /// Deep reasoning — extended CoT, multiple passes.
    DeepReasoning,
}

/// The MissionState — sovereign lifecycle object.
///
/// This is the input to the constitutional pipeline.
/// The CanonicalReceipt is its output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionState {
    /// Unique mission identifier (BLAKE3 of intent + timestamp + node).
    pub mission_id: [u8; 32],
    /// Human-readable mission ID for display.
    pub mission_id_hex: String,
    /// Raw intent text from the human operator.
    pub intent: String,
    /// Current lifecycle phase.
    pub phase: MissionPhase,
    /// Classified complexity tier.
    pub complexity: ComplexityTier,
    /// Genesis hash this mission is bound to.
    pub genesis_hash: [u8; 32],
    /// Node identity that submitted this mission.
    pub node_id: String,
    /// Timestamp when mission was proposed (Unix ms).
    pub proposed_at: u64,
    /// Timestamp of last phase transition (Unix ms).
    pub last_transition_at: u64,
    /// Deadline for execution (Unix ms). 0 = no deadline.
    pub deadline_ms: u64,
    /// PAT agents assigned to this mission (by callsign).
    pub assigned_agents: Vec<String>,
    /// Subtask count (populated after decomposition).
    pub subtask_count: u32,
    /// Subtasks completed.
    pub subtasks_done: u32,
    /// BLAKE3 hash of the intent for evidence linking.
    pub intent_hash: [u8; 32],
}

impl MissionState {
    /// Create a new mission from human intent.
    pub fn propose(
        intent: impl Into<String>,
        genesis_hash: [u8; 32],
        node_id: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        let intent = intent.into();
        let node_id = node_id.into();

        let intent_hash = {
            let mut h = Hasher::new();
            h.update(b"bizra-intent-v1:");
            h.update(intent.as_bytes());
            *h.finalize().as_bytes()
        };

        let mission_id = {
            let mut h = Hasher::new();
            h.update(DOMAIN_MISSION.as_bytes());
            h.update(b":");
            h.update(&intent_hash);
            h.update(&now_ms.to_le_bytes());
            h.update(node_id.as_bytes());
            *h.finalize().as_bytes()
        };

        let hex: String = mission_id.iter().map(|b| format!("{b:02x}")).collect();

        Self {
            mission_id,
            mission_id_hex: hex[..16].to_string(),
            intent,
            phase: MissionPhase::Proposed,
            complexity: ComplexityTier::SingleAgent,
            genesis_hash,
            node_id,
            proposed_at: now_ms,
            last_transition_at: now_ms,
            deadline_ms: 0,
            assigned_agents: Vec::new(),
            subtask_count: 0,
            subtasks_done: 0,
            intent_hash,
        }
    }

    /// Transition to a new phase. Returns Err if the transition is invalid.
    pub fn transition(
        &mut self,
        to: MissionPhase,
        now_ms: u64,
    ) -> Result<(), MissionTransitionError> {
        if self.phase.is_terminal() {
            return Err(MissionTransitionError::AlreadyTerminal(self.phase));
        }

        let valid = match (self.phase, to) {
            (MissionPhase::Proposed, MissionPhase::Admitted) => true,
            (MissionPhase::Proposed, MissionPhase::Rejected) => true,
            (MissionPhase::Admitted, MissionPhase::Decomposed) => true,
            (MissionPhase::Admitted, MissionPhase::Executing) => true, // simple missions skip decompose
            (MissionPhase::Admitted, MissionPhase::Rejected) => true,
            (MissionPhase::Decomposed, MissionPhase::Executing) => true,
            (MissionPhase::Executing, MissionPhase::Completed) => true,
            (MissionPhase::Executing, MissionPhase::Degraded) => true,
            (MissionPhase::Executing, MissionPhase::TimedOut) => true,
            (MissionPhase::Degraded, MissionPhase::Completed) => true,
            (MissionPhase::Degraded, MissionPhase::TimedOut) => true,
            (MissionPhase::Completed, MissionPhase::Receipted) => true,
            _ => false,
        };

        if !valid {
            return Err(MissionTransitionError::InvalidTransition(self.phase, to));
        }

        if self.deadline_ms > 0 && now_ms > self.deadline_ms && !to.is_terminal() {
            self.phase = MissionPhase::TimedOut;
            self.last_transition_at = now_ms;
            return Err(MissionTransitionError::DeadlineExceeded);
        }

        self.phase = to;
        self.last_transition_at = now_ms;
        Ok(())
    }

    /// Assign PAT agents after decomposition.
    pub fn assign_agents(&mut self, agents: Vec<String>, subtask_count: u32) {
        self.assigned_agents = agents;
        self.subtask_count = subtask_count;
    }

    /// Mark a subtask as done.
    pub fn complete_subtask(&mut self) {
        self.subtasks_done = self.subtasks_done.saturating_add(1);
    }

    /// Progress ratio (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.subtask_count == 0 {
            return 0.0;
        }
        self.subtasks_done as f64 / self.subtask_count as f64
    }

    /// Duration since proposal (ms).
    pub fn elapsed_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.proposed_at)
    }
}

/// Errors from mission state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissionTransitionError {
    /// Mission is already in a terminal state.
    AlreadyTerminal(MissionPhase),
    /// The requested transition is not valid from the current phase.
    InvalidTransition(MissionPhase, MissionPhase),
    /// Execution deadline has been exceeded.
    DeadlineExceeded,
}

impl std::fmt::Display for MissionTransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyTerminal(p) => write!(f, "Mission already terminal at {p:?}"),
            Self::InvalidTransition(from, to) => write!(f, "Invalid transition: {from:?} → {to:?}"),
            Self::DeadlineExceeded => write!(f, "Mission deadline exceeded"),
        }
    }
}

impl std::error::Error for MissionTransitionError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = 0xAA;
        h
    }

    #[test]
    fn test_mission_propose() {
        let m = MissionState::propose("Help me plan Q3 roadmap", test_genesis(), "node0", 1000);
        assert_eq!(m.phase, MissionPhase::Proposed);
        assert!(!m.intent.is_empty());
        assert_ne!(m.mission_id, [0; 32]);
        assert_ne!(m.intent_hash, [0; 32]);
        assert_eq!(m.mission_id_hex.len(), 16);
    }

    #[test]
    fn test_mission_happy_path() {
        let mut m = MissionState::propose("Build a hash map", test_genesis(), "node0", 1000);
        assert!(m.transition(MissionPhase::Admitted, 1100).is_ok());
        assert!(m.transition(MissionPhase::Decomposed, 1200).is_ok());
        m.assign_agents(vec!["FORGE".into(), "JUDGE".into()], 3);
        assert!(m.transition(MissionPhase::Executing, 1300).is_ok());
        m.complete_subtask();
        m.complete_subtask();
        m.complete_subtask();
        assert!((m.progress() - 1.0).abs() < f64::EPSILON);
        assert!(m.transition(MissionPhase::Completed, 1500).is_ok());
        assert!(m.transition(MissionPhase::Receipted, 1600).is_ok());
        assert!(m.phase.is_terminal());
        assert_eq!(m.elapsed_ms(1600), 600);
    }

    #[test]
    fn test_mission_rejection() {
        let mut m = MissionState::propose("exploitative content", test_genesis(), "node0", 1000);
        assert!(m.transition(MissionPhase::Rejected, 1050).is_ok());
        assert!(m.phase.is_terminal());
        // Cannot transition from terminal
        assert!(m.transition(MissionPhase::Admitted, 1100).is_err());
    }

    #[test]
    fn test_mission_invalid_transition() {
        let mut m = MissionState::propose("test", test_genesis(), "node0", 1000);
        // Cannot skip directly to Executing from Proposed
        let err = m.transition(MissionPhase::Executing, 1100);
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            MissionTransitionError::InvalidTransition(..)
        ));
    }

    #[test]
    fn test_mission_deadline_enforcement() {
        let mut m = MissionState::propose("slow task", test_genesis(), "node0", 1000);
        m.deadline_ms = 2000;
        assert!(m.transition(MissionPhase::Admitted, 1100).is_ok());
        assert!(m.transition(MissionPhase::Executing, 1500).is_ok());
        // Try to complete after deadline — should fail
        let err = m.transition(MissionPhase::Completed, 3000);
        assert!(err.is_err());
        assert_eq!(m.phase, MissionPhase::TimedOut);
    }

    #[test]
    fn test_mission_degraded_path() {
        let mut m = MissionState::propose("fragile task", test_genesis(), "node0", 1000);
        assert!(m.transition(MissionPhase::Admitted, 1100).is_ok());
        assert!(m.transition(MissionPhase::Executing, 1200).is_ok());
        assert!(m.transition(MissionPhase::Degraded, 1300).is_ok());
        assert!(m.phase.is_active());
        assert!(m.transition(MissionPhase::Completed, 1400).is_ok());
        assert!(m.transition(MissionPhase::Receipted, 1500).is_ok());
    }

    #[test]
    fn test_mission_id_deterministic() {
        let m1 = MissionState::propose("same intent", test_genesis(), "node0", 1000);
        let m2 = MissionState::propose("same intent", test_genesis(), "node0", 1000);
        assert_eq!(m1.mission_id, m2.mission_id);
    }

    #[test]
    fn test_mission_id_unique_on_different_time() {
        let m1 = MissionState::propose("same intent", test_genesis(), "node0", 1000);
        let m2 = MissionState::propose("same intent", test_genesis(), "node0", 1001);
        assert_ne!(m1.mission_id, m2.mission_id);
    }
}
