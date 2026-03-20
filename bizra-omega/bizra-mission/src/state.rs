// bizra-mission/src/state.rs
// ============================================================
// Mission State Machine — derived from mission_lifecycle.json
// ============================================================
//
// Every cognitive operation transitions through these states.
// The valid_transitions map is constitutional law — any
// transition not listed is a violation.
// ============================================================

use serde::{Deserialize, Serialize};

/// The 14 states of a mission lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionState {
    Submitted,
    Queued,
    WarmingRetrieval,
    WarmingModel,
    Retrieving,
    Routing,
    Running,
    Scoring,
    Persisting,
    UrpValidating,
    UrpEnriching,
    // Terminal states
    Complete,
    Degraded,
    Failed,
    TimedOut,
    // Deferred settlement (offline node)
    AwaitingReconciliation,
}

impl MissionState {
    /// Is this a terminal state? Terminal states cannot transition further.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Complete | Self::Degraded | Self::Failed | Self::TimedOut
        )
    }

    /// Is this a deferred settlement state? (node executed offline)
    pub fn is_deferred(self) -> bool {
        matches!(self, Self::AwaitingReconciliation)
    }

    /// Is this a URP stage? (only applies to network-bound missions)
    pub fn is_urp_stage(self) -> bool {
        matches!(self, Self::UrpValidating | Self::UrpEnriching)
    }

    /// Legal transitions from this state. Constitutional law.
    pub fn valid_transitions(self) -> &'static [MissionState] {
        use MissionState::*;
        match self {
            Submitted => &[Queued, Failed],
            Queued => &[WarmingRetrieval, WarmingModel, TimedOut],
            WarmingRetrieval => &[WarmingModel, Retrieving, Degraded],
            WarmingModel => &[Retrieving, Failed],
            Retrieving => &[Routing, Degraded],
            Routing => &[Running, Failed],
            Running => &[Scoring, TimedOut, Failed],
            Scoring => &[Persisting, Degraded],
            Persisting => &[Complete, UrpValidating, Degraded, AwaitingReconciliation],
            UrpValidating => &[UrpEnriching, Complete, Degraded, Failed],
            UrpEnriching => &[Complete, Degraded],
            AwaitingReconciliation => &[UrpValidating, Complete, Degraded, Failed],
            Complete => &[],
            Degraded => &[],
            Failed => &[],
            TimedOut => &[],
        }
    }

    /// Can this state transition to the target?
    pub fn can_transition_to(self, target: MissionState) -> bool {
        self.valid_transitions().contains(&target)
    }
}

/// A recorded state transition with timestamp and reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from: MissionState,
    pub to: MissionState,
    pub at: u64,
    pub reason: String,
}

/// Failure codes — why a mission cannot complete.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    ModelNotAvailable,
    ModelLoadFailed,
    InferenceTimeout,
    InferenceError {
        detail: String,
    },
    GuardianVeto,
    IhsanBelowFloor,
    ResourceExhausted,
    QueueTimeout,
    CapabilityNotAvailable,
    /// State machine violation — an illegal transition was attempted.
    StateMachineViolation {
        from: String,
        to: String,
    },
}

/// Degradation reasons — what was degraded but the mission still produced partial output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DegradationReason {
    RetrievalSkipped,
    EmptyContext,
    UnscoredResponse,
    UnpersistedReceipt,
    FallbackModelUsed,
    PartialMemoryExtract,
    /// Guardian vetoed — response quality below constitutional standard.
    GuardianVeto,
}
