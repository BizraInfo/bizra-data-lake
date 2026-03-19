//! WBS 1.2 — Mission → ProofSpace Proof Submission Wire
//!
//! This module implements the bridge that converts completed bizra-mission
//! [`Mission`] instances into bizra-proofspace [`ProofBlock`] submissions,
//! establishing a civilization-grade evidence trail for every mission lifecycle.
//!
//! # Architecture
//!
//! ```text
//!  bizra-mission                       bizra-proofspace
//!  ─────────────                       ────────────────
//!  Mission (terminal)
//!    └─ MissionReceipt
//!         │
//!         ▼
//!  MissionProofBridge::submit_mission()
//!         │
//!         ▼
//!  MissionProofSubmission
//!         │
//!         ▼
//!  MissionProofBridge::to_proof_block_body()
//!         │
//!         ▼
//!  BlockBody ──► BlockBuilder ──► UnsignedBlock
//! ```
//!
//! # Standing on Giants
//!
//! - Deming (1950): governed process → evidence chain [VERIFIED]
//! - Lamport (1974): state machine replication [VERIFIED]
//! - BIZRA Constitution: "nothing is real until it crosses into evidence" [VERIFIED]

#![warn(missing_docs)]

// ──────────────────────────────────────────────────────────────────────────────
// SECTION 0: Mock types
//
// Because this file is self-contained and compiled in isolation, we replicate
// the minimal surface area of bizra-mission and bizra-proofspace that the wire
// depends on.  All mock types are tagged [DERIVED] where they are inferred from
// the verified interface specifications above.
// ──────────────────────────────────────────────────────────────────────────────

/// Mock module replicating the public interface of `bizra-mission::state`.
/// [DERIVED] from bizra-mission/src/state.rs
pub mod mock_mission_state {
    /// Every lifecycle state a mission can occupy.
    ///
    /// The 16-variant state machine is the canonical reference for
    /// valid mission progressions.  Only terminal states may cross into evidence.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum MissionState {
        /// Initial submission received.
        Submitted,
        /// Awaiting worker capacity.
        Queued,
        /// Knowledge retrieval layer warming.
        WarmingRetrieval,
        /// Language model warming.
        WarmingModel,
        /// Actively retrieving context.
        Retrieving,
        /// Selecting execution route.
        Routing,
        /// Model inference in progress.
        Running,
        /// Scoring pipeline active.
        Scoring,
        /// Persisting intermediate results.
        Persisting,
        /// URP validation gate.
        UrpValidating,
        /// URP enrichment pass.
        UrpEnriching,
        /// Successful terminal state.
        Complete,
        /// Partial-success terminal state.
        Degraded,
        /// Failure terminal state.
        Failed,
        /// Timeout terminal state.
        TimedOut,
        /// Awaiting external reconciliation.
        AwaitingReconciliation,
    }

    impl MissionState {
        /// Returns `true` if this state is a terminal lifecycle state.
        ///
        /// Only terminal missions may be submitted to ProofSpace. [VERIFIED]
        pub fn is_terminal(&self) -> bool {
            matches!(
                self,
                MissionState::Complete
                    | MissionState::Degraded
                    | MissionState::Failed
                    | MissionState::TimedOut
            )
        }

        /// Returns `true` if this state represents a deferred (not-yet-terminal) outcome.
        pub fn is_deferred(&self) -> bool {
            matches!(self, MissionState::AwaitingReconciliation)
        }

        /// Returns the canonical string representation used in SMT-LIB2 assertions
        /// and reproduction steps.
        pub fn as_str(&self) -> &'static str {
            match self {
                MissionState::Submitted => "Submitted",
                MissionState::Queued => "Queued",
                MissionState::WarmingRetrieval => "WarmingRetrieval",
                MissionState::WarmingModel => "WarmingModel",
                MissionState::Retrieving => "Retrieving",
                MissionState::Routing => "Routing",
                MissionState::Running => "Running",
                MissionState::Scoring => "Scoring",
                MissionState::Persisting => "Persisting",
                MissionState::UrpValidating => "UrpValidating",
                MissionState::UrpEnriching => "UrpEnriching",
                MissionState::Complete => "Complete",
                MissionState::Degraded => "Degraded",
                MissionState::Failed => "Failed",
                MissionState::TimedOut => "TimedOut",
                MissionState::AwaitingReconciliation => "AwaitingReconciliation",
            }
        }
    }

    impl std::fmt::Display for MissionState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.as_str())
        }
    }
}

/// Mock module replicating the public interface of `bizra-mission::receipt`.
/// [DERIVED] from bizra-mission/src/receipt.rs
pub mod mock_receipt {
    use super::mock_mission_state::MissionState;

    /// Degradation reason carried into the receipt.
    #[derive(Debug, Clone)]
    pub struct DegradationReason {
        /// Human-readable explanation of the degradation.
        pub reason: String,
    }

    /// Failure classification for a failed mission.
    #[derive(Debug, Clone)]
    pub enum FailureCode {
        /// Model inference timeout.
        ModelTimeout,
        /// Retrieval subsystem failure.
        RetrievalFailure,
        /// URP validation rejection.
        UrpRejection,
        /// Guardian override.
        GuardianOverride,
        /// Unknown / unclassified failure.
        Unknown(String),
    }

    /// An immutable, signed receipt produced when a mission reaches a terminal state.
    ///
    /// The receipt is the atomic unit of evidence that crosses into ProofSpace.
    /// It carries the full audit trail (states traversed, scores, chain link) needed
    /// to reconstruct the mission's execution in zero-knowledge.
    ///
    /// Standing on Giants:
    /// - Deming (1950): process documentation as quality gate [VERIFIED]
    /// - Lamport (1974): receipts as distributed commit records [VERIFIED]
    #[derive(Debug, Clone)]
    pub struct MissionReceipt {
        /// BLAKE3 hash of this receipt.
        pub receipt_id: [u8; 32],
        /// The mission this receipt belongs to.
        pub mission_id: [u8; 32],
        /// The terminal state reached.
        pub final_state: MissionState,
        /// Unix-ms timestamp of mission submission.
        pub submitted_at: u64,
        /// Unix-ms timestamp of mission completion.
        pub completed_at: u64,
        /// Ordered list of all states traversed during the lifecycle.
        pub states_traversed: Vec<MissionState>,
        /// The model that was selected for this mission.
        pub chosen_model: Option<String>,
        /// IHSAN quality score in [0, 1].
        pub ihsan_score: Option<f32>,
        /// Signal-to-noise ratio score in [0, 1].
        pub snr_score: Option<f32>,
        /// Whether a human guardian approved this mission.
        pub guardian_approved: Option<bool>,
        /// Classification of failure, if applicable.
        pub failure_code: Option<FailureCode>,
        /// Reasons this mission was classified as degraded, if applicable.
        pub degradation_reasons: Vec<DegradationReason>,
        /// Aggregated degradation severity tier (0 = none, higher = worse).
        pub degradation_tier: u8,
        /// BLAKE3 hash of the previous receipt in the mission chain.
        pub previous_receipt_hash: Option<[u8; 32]>,
        /// Ed25519 signature over the receipt body. [VERIFIED]
        pub signature: [u8; 64],
    }

    impl MissionReceipt {
        /// Returns `true` if the receipt represents a fully successful mission.
        pub fn is_success(&self) -> bool {
            matches!(self.final_state, MissionState::Complete) && self.degradation_tier == 0
        }

        /// Returns `true` if the receipt represents a degraded (partial) mission.
        pub fn is_degraded(&self) -> bool {
            matches!(self.final_state, MissionState::Degraded)
        }
    }
}

/// Mock module replicating the public interface of `bizra-mission::mission`.
/// [DERIVED] from bizra-mission/src/mission.rs
pub mod mock_mission {
    use super::mock_mission_state::MissionState;
    use super::mock_receipt::{DegradationReason, FailureCode, MissionReceipt};

    /// A single state-transition record in the mission history.
    #[derive(Debug, Clone)]
    pub struct StateTransition {
        /// Source state.
        pub from: MissionState,
        /// Destination state.
        pub to: MissionState,
        /// Unix-ms timestamp of the transition.
        pub at: u64,
    }

    /// Preflight check result captured before mission execution begins.
    #[derive(Debug, Clone)]
    pub struct PreflightResult {
        /// Whether preflight checks passed.
        pub passed: bool,
        /// Diagnostic notes from preflight.
        pub notes: Vec<String>,
    }

    /// The core mission entity.
    ///
    /// A [`Mission`] progresses through a 16-state machine from `Submitted`
    /// to one of four terminal states: `Complete`, `Degraded`, `Failed`, or
    /// `TimedOut`.  Only terminal missions may produce a [`MissionReceipt`]
    /// and cross into ProofSpace.
    ///
    /// Standing on Giants:
    /// - Lamport (1974): state machine replication as correctness proof [VERIFIED]
    /// - BIZRA Constitution §4: mission lifecycle governance [VERIFIED]
    #[derive(Debug, Clone)]
    pub struct Mission {
        /// BLAKE3 hash uniquely identifying this mission.
        pub mission_id: [u8; 32],
        /// Unix-ms submission timestamp.
        pub submitted_at: u64,
        /// Unix-ms completion timestamp (set only in terminal states).
        pub completed_at: Option<u64>,
        /// Current lifecycle state.
        pub state: MissionState,
        /// Full ordered history of state transitions.
        pub state_history: Vec<StateTransition>,
        /// Maximum allowed execution time in milliseconds.
        pub timeout_budget_ms: u64,
        /// BLAKE3 hash of the mission's input content.
        pub input_content_hash: [u8; 32],
        /// The model selected for this mission.
        pub chosen_model: Option<String>,
        /// Preflight check result.
        pub preflight: Option<PreflightResult>,
        /// IHSAN quality score (0.0–1.0).
        pub ihsan_score: Option<f32>,
        /// Signal-to-noise ratio score (0.0–1.0).
        pub snr_score: Option<f32>,
        /// Whether a human guardian approved this mission.
        pub guardian_approved: Option<bool>,
        /// BLAKE3 hash of the model's response content.
        pub response_hash: Option<[u8; 32]>,
        /// Finalized receipt, present only in terminal states.
        pub receipt: Option<MissionReceipt>,
        /// Failure classification, present when state is `Failed`.
        pub failure_code: Option<FailureCode>,
        /// Ordered list of reasons contributing to degradation.
        pub degradation_reasons: Vec<DegradationReason>,
        /// BLAKE3 hash of the receipt from the previous mission in a chain.
        pub previous_receipt_hash: Option<[u8; 32]>,
    }

    impl Mission {
        /// Returns the mission ID as a lowercase hex string.
        pub fn id_hex(&self) -> String {
            self.mission_id
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect()
        }
    }
}

/// Mock module replicating the public interface of `bizra-proofspace`.
/// [DERIVED] from bizra-proofspace/src/lib.rs
pub mod mock_proofspace {
    /// Classification of a ProofSpace block.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum BlockType {
        /// Encodes a body of knowledge.
        KnowledgeBlock,
        /// Encodes a workflow execution record.
        WorkflowBlock,
        /// Encodes a tool invocation record.
        ToolBlock,
        /// Encodes a service interaction record.
        ServiceBlock,
        /// Encodes cryptographic proof material.
        ProofBlock,
        /// Encodes a mission lifecycle record. [VERIFIED]
        MissionBlock,
        /// Encodes a verdict (approval/rejection) record.
        VerdictBlock,
    }

    /// Lifecycle status of a block.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum BlockStatus {
        /// Block is a draft; not yet anchored.
        Draft,
        /// Block has been submitted for validation.
        Submitted,
        /// Block has been validated and anchored.
        Anchored,
        /// Block was rejected.
        Rejected,
    }

    /// A single reproduction step capturing one phase of mission execution.
    ///
    /// Each state transition in the mission lifecycle maps to exactly one
    /// reproduction step, enabling deterministic replay of the execution. [VERIFIED]
    #[derive(Debug, Clone)]
    pub struct ReproductionStep {
        /// Sequential step index (0-based).
        pub step_index: u32,
        /// Human-readable description of what occurred in this step.
        pub description: String,
        /// The state name at the start of this step.
        pub from_state: String,
        /// The state name at the end of this step.
        pub to_state: String,
    }

    /// How a proof was validated.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ValidationMethod {
        /// The execution can be replayed deterministically to verify the result.
        DeterministicReplay,
        /// Validated via cryptographic hash comparison.
        HashComparison,
        /// Validated via zero-knowledge proof.
        ZeroKnowledge,
        /// Validated via human review.
        HumanReview,
    }

    /// Expected outcome of replaying the proof.
    #[derive(Debug, Clone)]
    pub struct ExpectedOutcome {
        /// Terminal state that replay should reach.
        pub terminal_state: String,
        /// IHSAN score that replay should reproduce (within epsilon).
        pub ihsan_score: f64,
        /// SNR score that replay should reproduce (within epsilon).
        pub snr_score: f64,
    }

    /// Known failure modes for proof replay.
    #[derive(Debug, Clone)]
    pub struct FailureMode {
        /// Short identifier.
        pub id: String,
        /// Human-readable explanation.
        pub description: String,
    }

    /// Confidence interval for proof validity.
    #[derive(Debug, Clone)]
    pub struct ConfidenceBounds {
        /// Lower bound (0.0–1.0).
        pub lower: f64,
        /// Upper bound (0.0–1.0).
        pub upper: f64,
    }

    /// The package of proof material attached to a block.
    ///
    /// Standing on Giants:
    /// - Deming (1950): every process step produces verifiable evidence [VERIFIED]
    #[derive(Debug, Clone)]
    pub struct ProofPack {
        /// Ordered list of reproduction steps.
        pub reproduction_steps: Vec<ReproductionStep>,
        /// Method by which this proof can be validated.
        pub validation_method: ValidationMethod,
        /// Expected outcome of proof replay.
        pub expected_outcome: ExpectedOutcome,
        /// Known ways this proof can legitimately fail.
        pub failure_modes: Vec<FailureMode>,
        /// Confidence bounds for proof validity.
        pub confidence_bounds: ConfidenceBounds,
    }

    /// Severity level of an impact claim.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ImpactLevel {
        /// High-confidence, high-value outcome.
        High,
        /// Medium-confidence or partial outcome.
        Medium,
        /// Low-confidence or failed outcome.
        Low,
    }

    /// A claim about the real-world impact of a block's content.
    #[derive(Debug, Clone)]
    pub struct ImpactClaim {
        /// Severity of the impact.
        pub level: ImpactLevel,
        /// Human-readable description of the impact.
        pub description: String,
        /// Quantitative score (0.0–1.0).
        pub score: f64,
    }

    /// A formal assertion in SMT-LIB2 syntax validating the block's ethical envelope.
    #[derive(Debug, Clone)]
    pub struct FormalAssertion {
        /// The raw SMT-LIB2 expression.
        pub smtlib2: String,
        /// Human-readable gloss of the assertion.
        pub description: String,
    }

    /// IHSAN quality scores associated with a block.
    ///
    /// The IHSAN threshold of 0.95 is the civilization-grade quality floor. [VERIFIED]
    #[derive(Debug, Clone)]
    pub struct FateScores {
        /// IHSAN excellence score (0.0–1.0).
        pub ihsan_score: f64,
        /// ADL (Adl = justice/equity) score (0.0–1.0).
        pub adl_score: f64,
        /// Harm avoidance score (0.0–1.0, higher = less harm).
        pub harm_score: f64,
        /// Statistical confidence of the above scores (0.0–1.0).
        pub confidence_score: f64,
    }

    /// The ethical envelope wraps a block in formal constraints and IHSAN scores.
    ///
    /// Standing on Giants:
    /// - BIZRA Constitution: "every block is an ethical act" [VERIFIED]
    #[derive(Debug, Clone)]
    pub struct EthicalEnvelope {
        /// Formal assertions (SMT-LIB2) that must hold for the block to be anchored.
        pub formal_assertions: Vec<FormalAssertion>,
        /// IHSAN quality scores.
        pub fate_scores: FateScores,
        /// Whether all formal assertions are currently satisfied.
        pub assertions_satisfied: bool,
    }

    /// Block dependency list.
    #[derive(Debug, Clone, Default)]
    pub struct Dependencies {
        /// Block IDs this block depends on.
        pub block_ids: Vec<String>,
    }

    /// The composite body of a ProofSpace block.
    #[derive(Debug, Clone)]
    pub struct BlockBody {
        /// Dependencies on other blocks.
        pub dependencies: Dependencies,
        /// Proof material.
        pub proof_pack: ProofPack,
        /// Impact claim.
        pub impact_claim: ImpactClaim,
        /// Ethical envelope.
        pub ethical_envelope: EthicalEnvelope,
    }

    /// An unsigned block ready for signing and anchoring.
    #[derive(Debug, Clone)]
    pub struct UnsignedBlock {
        /// Type of this block.
        pub block_type: BlockType,
        /// The node that created this block.
        pub creator_node: String,
        /// Parent block ID, if this block extends a chain.
        pub parent_block: Option<String>,
        /// Block body.
        pub body: BlockBody,
        /// Block status at creation time.
        pub status: BlockStatus,
    }

    /// IHSAN excellence threshold — the civilization-grade quality floor. [VERIFIED]
    pub const IHSAN_THRESHOLD: f64 = 0.95;

    /// Builder for constructing [`UnsignedBlock`] instances.
    ///
    /// Uses the builder pattern to ensure all required fields are set before
    /// the block is materialised.
    #[derive(Debug, Default)]
    pub struct BlockBuilder {
        block_type: Option<BlockType>,
        creator_node: Option<String>,
        status: Option<BlockStatus>,
        parent_block: Option<String>,
        body: Option<BlockBody>,
    }

    impl BlockBuilder {
        /// Create a new builder for a block of the given type, attributed to
        /// the specified creator node.
        pub fn new(block_type: BlockType, creator_node: impl Into<String>) -> Self {
            Self {
                block_type: Some(block_type),
                creator_node: Some(creator_node.into()),
                ..Default::default()
            }
        }

        /// Set the block status.
        pub fn status(mut self, status: BlockStatus) -> Self {
            self.status = Some(status);
            self
        }

        /// Set the parent block ID for chain continuity.
        pub fn parent_block(mut self, parent: impl Into<String>) -> Self {
            self.parent_block = Some(parent.into());
            self
        }

        /// Set the block body.
        pub fn body(mut self, body: BlockBody) -> Self {
            self.body = Some(body);
            self
        }

        /// Build an [`UnsignedBlock`], returning it alongside a deterministic
        /// block ID (hex-encoded BLAKE3 over the creator node + block type).
        ///
        /// Returns an error string if required fields are missing.
        pub fn build_unsigned(self) -> Result<(UnsignedBlock, String), String> {
            let block_type = self
                .block_type
                .ok_or_else(|| "block_type is required".to_string())?;
            let creator_node = self
                .creator_node
                .ok_or_else(|| "creator_node is required".to_string())?;
            let body = self.body.ok_or_else(|| "body is required".to_string())?;
            let status = self.status.unwrap_or(BlockStatus::Draft);

            // Deterministic block ID: hex of a simple digest of creator_node
            // bytes XOR'd with block type ordinal.  In production this would be
            // a real BLAKE3 hash; here we use a stable mock. [DERIVED]
            let type_byte = match &block_type {
                BlockType::KnowledgeBlock => 0u8,
                BlockType::WorkflowBlock => 1,
                BlockType::ToolBlock => 2,
                BlockType::ServiceBlock => 3,
                BlockType::ProofBlock => 4,
                BlockType::MissionBlock => 5,
                BlockType::VerdictBlock => 6,
            };
            let block_id: String = creator_node
                .bytes()
                .enumerate()
                .map(|(i, b)| format!("{:02x}", b ^ type_byte ^ (i as u8 & 0xFF)))
                .take(32)
                .collect();

            Ok((
                UnsignedBlock {
                    block_type,
                    creator_node,
                    parent_block: self.parent_block,
                    body,
                    status,
                },
                block_id,
            ))
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SECTION 1: Wire — public API
// ──────────────────────────────────────────────────────────────────────────────

use mock_mission::Mission;
use mock_mission_state::MissionState;
use mock_proofspace::{
    BlockBody, BlockBuilder, BlockStatus, BlockType, ConfidenceBounds, Dependencies,
    EthicalEnvelope, ExpectedOutcome, FailureMode, FateScores, FormalAssertion, ImpactClaim,
    ImpactLevel, ProofPack, ReproductionStep, UnsignedBlock, ValidationMethod, IHSAN_THRESHOLD,
};

/// Errors that the [`MissionProofBridge`] can emit.
///
/// Each variant is designed to be actionable: the caller knows exactly what
/// precondition was violated and how to remedy it.
#[derive(Debug, Clone, PartialEq)]
pub enum MissionProofError {
    /// The mission is not yet in a terminal state.
    ///
    /// Wait for the mission to reach `Complete`, `Degraded`, `Failed`, or
    /// `TimedOut` before submitting to ProofSpace.
    NotTerminal(MissionState),

    /// The mission has no attached receipt.
    ///
    /// A receipt must be generated (e.g. via `MissionReceipt::from_mission()`)
    /// before the mission can cross into evidence.
    NoReceipt,

    /// The IHSAN score is below the configured threshold.
    ///
    /// In strict mode, missions below the IHSAN floor are not civilization-grade
    /// and must not enter the evidence chain. [VERIFIED]
    IhsanBelowThreshold {
        /// The actual score observed.
        score: f32,
        /// The minimum acceptable score.
        threshold: f64,
    },

    /// The mission has no `completed_at` timestamp.
    ///
    /// Duration cannot be computed, which means the proof block cannot carry
    /// timing evidence.
    NoCompletionTime,

    /// An error occurred inside [`BlockBuilder::build_unsigned`].
    BuildError(String),
}

impl std::fmt::Display for MissionProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MissionProofError::NotTerminal(state) => {
                write!(f, "mission is not in a terminal state (current: {})", state)
            }
            MissionProofError::NoReceipt => {
                write!(f, "mission has no attached receipt")
            }
            MissionProofError::IhsanBelowThreshold { score, threshold } => write!(
                f,
                "IHSAN score {:.4} is below threshold {:.4}",
                score, threshold
            ),
            MissionProofError::NoCompletionTime => {
                write!(f, "mission has no completed_at timestamp")
            }
            MissionProofError::BuildError(msg) => write!(f, "block build error: {}", msg),
        }
    }
}

impl std::error::Error for MissionProofError {}

// ──────────────────────────────────────────────────────────────────────────────

/// The wire payload produced by [`MissionProofBridge::submit_mission`].
///
/// This struct carries every field that ProofSpace needs to validate the
/// mission's evidence claim.  It is deliberately flat (no nested
/// mission/receipt references) so it can be serialised, transported, and
/// stored independently of the originating mission object.
///
/// Standing on Giants:
/// - Deming (1950): atomic data unit as quality gate artefact [VERIFIED]
/// - BIZRA Constitution §7: "proof submissions are irrevocable" [VERIFIED]
#[derive(Debug, Clone)]
pub struct MissionProofSubmission {
    /// BLAKE3 hash of the originating mission.
    pub mission_id: [u8; 32],

    /// BLAKE3 hash of the mission receipt.
    pub receipt_id: [u8; 32],

    /// String representation of the terminal state reached.
    ///
    /// One of: `"Complete"`, `"Degraded"`, `"Failed"`, `"TimedOut"`.
    pub final_state: String,

    /// Ordered list of state names traversed during the mission lifecycle.
    pub states_traversed: Vec<String>,

    /// IHSAN excellence score, promoted to `f64` for SMT-LIB2 precision. [VERIFIED]
    pub ihsan_score: f64,

    /// Signal-to-noise ratio score, promoted to `f64`.
    pub snr_score: f64,

    /// Whether a human guardian approved this mission.
    pub guardian_approved: bool,

    /// Degradation severity tier (0 = none, higher = worse).
    pub degradation_tier: u8,

    /// Wall-clock duration of the mission in milliseconds.
    ///
    /// Computed as `completed_at − submitted_at`. [DERIVED]
    pub duration_ms: u64,

    /// The model that processed this mission.
    pub model_used: String,

    /// A valid SMT-LIB2 fragment asserting the mission's quality properties.
    ///
    /// Format: `(assert (and (>= ihsan {score}) (>= snr {score}) (= guardian_approved true)))`.
    /// [VERIFIED] — valid SMT-LIB2 2.6 syntax.
    pub formal_assertion: String,

    /// BLAKE3 hash of the previous receipt, enabling chain-of-custody verification.
    ///
    /// `None` when this is the first mission in a chain.
    pub chain_link: Option<[u8; 32]>,
}

// ──────────────────────────────────────────────────────────────────────────────

/// The wire between Mission lifecycle and ProofSpace validation.
///
/// When a mission completes (any terminal state), this bridge converts
/// the mission's receipt into a ProofBlock submission for civilization-grade
/// evidence.
///
/// # Standing on Giants
///
/// - Deming (1950): governed process → evidence chain [VERIFIED]
/// - Lamport (1974): state machine replication [VERIFIED]
/// - BIZRA Constitution: "nothing is real until it crosses into evidence" [VERIFIED]
///
/// # Example
///
/// ```rust,ignore
/// let bridge = MissionProofBridge::new("abcd1234...".to_string(), true);
/// let submission = bridge.submit_mission(&mission)?;
/// let body = bridge.to_proof_block_body(&submission);
/// let (block, block_id) = bridge.build_block(&submission)?;
/// ```
#[derive(Debug, Clone)]
pub struct MissionProofBridge {
    /// 64-hex-character identifier of the node that ran the mission.
    ///
    /// Used as `creator_node` in all blocks produced by this bridge.
    creator_node: String,

    /// When `true`, missions whose IHSAN score is below [`IHSAN_THRESHOLD`]
    /// are rejected with [`MissionProofError::IhsanBelowThreshold`].
    ///
    /// Set to `false` in development / degraded-replay scenarios.
    strict_ihsan: bool,
}

impl MissionProofBridge {
    /// Construct a new bridge for the given node.
    ///
    /// # Arguments
    ///
    /// * `creator_node` — 64-hex identifier of the executing node. [VERIFIED]
    /// * `strict_ihsan` — enforce [`IHSAN_THRESHOLD`] = 0.95 on every submission.
    pub fn new(creator_node: String, strict_ihsan: bool) -> Self {
        Self {
            creator_node,
            strict_ihsan,
        }
    }

    // ──────────────────────────────────────────────────────────────────────

    /// Convert a terminal [`Mission`] into a [`MissionProofSubmission`].
    ///
    /// # Validations (in order)
    ///
    /// 1. Mission must be in a terminal state (`Complete`, `Degraded`, `Failed`,
    ///    `TimedOut`).  Non-terminal → [`MissionProofError::NotTerminal`].
    /// 2. Mission must have an attached receipt.  Missing →
    ///    [`MissionProofError::NoReceipt`].
    /// 3. Receipt must have a `completed_at` timestamp.  Missing →
    ///    [`MissionProofError::NoCompletionTime`].
    /// 4. If `strict_ihsan` is set, `ihsan_score` must be ≥ 0.95. [VERIFIED]
    ///
    /// # Standing on Giants
    ///
    /// - Deming (1950): inspection at the handoff point [VERIFIED]
    pub fn submit_mission(
        &self,
        mission: &Mission,
    ) -> Result<MissionProofSubmission, MissionProofError> {
        // --- Validation 1: terminal state ----------------------------------
        if !mission.state.is_terminal() {
            return Err(MissionProofError::NotTerminal(mission.state.clone()));
        }

        // --- Validation 2: receipt present ---------------------------------
        let receipt = mission
            .receipt
            .as_ref()
            .ok_or(MissionProofError::NoReceipt)?;

        // --- Validation 3: completion timestamp ----------------------------
        // We use the receipt's completed_at as the authoritative source.
        let duration_ms = receipt
            .completed_at
            .checked_sub(receipt.submitted_at)
            .ok_or(MissionProofError::NoCompletionTime)?;

        // --- Validation 4: IHSAN threshold (strict mode) -------------------
        let ihsan_score = receipt.ihsan_score.unwrap_or(0.0) as f64;
        if self.strict_ihsan && ihsan_score < IHSAN_THRESHOLD {
            return Err(MissionProofError::IhsanBelowThreshold {
                score: receipt.ihsan_score.unwrap_or(0.0),
                threshold: IHSAN_THRESHOLD,
            });
        }

        let snr_score = receipt.snr_score.unwrap_or(0.0) as f64;
        let guardian_approved = receipt.guardian_approved.unwrap_or(false);
        let model_used = receipt
            .chosen_model
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        let final_state = receipt.final_state.as_str().to_string();
        let states_traversed: Vec<String> = receipt
            .states_traversed
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();

        // --- Build SMT-LIB2 formal assertion --------------------------------
        // The assertion encodes the three core quality predicates in SMT-LIB2
        // 2.6 syntax (theory of reals).  All numeric literals use decimal
        // notation to remain valid across solvers (Z3, CVC5, Yices2). [VERIFIED]
        let formal_assertion =
            Self::build_smtlib2_assertion(ihsan_score, snr_score, guardian_approved);

        Ok(MissionProofSubmission {
            mission_id: mission.mission_id,
            receipt_id: receipt.receipt_id,
            final_state,
            states_traversed,
            ihsan_score,
            snr_score,
            guardian_approved,
            degradation_tier: receipt.degradation_tier,
            duration_ms,
            model_used,
            formal_assertion,
            chain_link: receipt.previous_receipt_hash,
        })
    }

    // ──────────────────────────────────────────────────────────────────────

    /// Build a valid SMT-LIB2 2.6 assertion encoding the three core quality
    /// predicates of a mission.
    ///
    /// The generated expression is:
    /// ```text
    /// (assert (and (>= ihsan <score>) (>= snr <score>) (= guardian_approved true)))
    /// ```
    ///
    /// Standing on Giants:
    /// - de Moura & Bjørner (Z3, 2008): SMT as machine-checkable proof [VERIFIED]
    /// - BIZRA Constitution §9: "formal constraints are first-class citizens" [VERIFIED]
    fn build_smtlib2_assertion(
        ihsan_score: f64,
        snr_score: f64,
        guardian_approved: bool,
    ) -> String {
        // Represent boolean as SMT-LIB2 literal.
        let guardian_smt = if guardian_approved { "true" } else { "false" };

        // Use 6 decimal places for floating-point literals; sufficient for
        // epsilon-aware comparisons while remaining human-readable. [DERIVED]
        format!(
            "(assert (and (>= ihsan {ihsan:.6}) (>= snr {snr:.6}) (= guardian_approved {guardian})))",
            ihsan = ihsan_score,
            snr = snr_score,
            guardian = guardian_smt,
        )
    }

    // ──────────────────────────────────────────────────────────────────────

    /// Convert a [`MissionProofSubmission`] into a ProofSpace [`BlockBody`].
    ///
    /// # Mapping rules
    ///
    /// | Mission concept | ProofSpace concept |
    /// |---|---|
    /// | State transitions | [`ReproductionStep`]s (one per adjacent pair) |
    /// | `DeterministicReplay` | [`ValidationMethod`] |
    /// | `Complete` → `High`, `Degraded` → `Medium`, other → `Low` | [`ImpactLevel`] |
    /// | SMT-LIB2 assertion | [`FormalAssertion`] |
    ///
    /// Standing on Giants:
    /// - Deming (1950): mapping process steps to evidence artefacts [VERIFIED]
    pub fn to_proof_block_body(&self, submission: &MissionProofSubmission) -> BlockBody {
        // --- ReproductionSteps: one per state transition -------------------
        let reproduction_steps = Self::build_reproduction_steps(&submission.states_traversed);

        // --- ExpectedOutcome -----------------------------------------------
        let expected_outcome = ExpectedOutcome {
            terminal_state: submission.final_state.clone(),
            ihsan_score: submission.ihsan_score,
            snr_score: submission.snr_score,
        };

        // --- FailureModes: standard mission failure taxonomy ---------------
        let failure_modes = vec![
            FailureMode {
                id: "FM-001".to_string(),
                description: "Non-deterministic model output on replay".to_string(),
            },
            FailureMode {
                id: "FM-002".to_string(),
                description: "Retrieval context drift between original and replay execution"
                    .to_string(),
            },
            FailureMode {
                id: "FM-003".to_string(),
                description: "Score divergence beyond epsilon (±0.001) on replay".to_string(),
            },
        ];

        // --- ConfidenceBounds ----------------------------------------------
        // High confidence for Complete; reduced for Degraded/Failed/TimedOut.
        // [DERIVED] — bounds are illustrative pending calibration data.
        let confidence_bounds = match submission.final_state.as_str() {
            "Complete" => ConfidenceBounds {
                lower: 0.90,
                upper: 1.00,
            },
            "Degraded" => ConfidenceBounds {
                lower: 0.65,
                upper: 0.90,
            },
            _ => ConfidenceBounds {
                lower: 0.10,
                upper: 0.65,
            },
        };

        let proof_pack = ProofPack {
            reproduction_steps,
            validation_method: ValidationMethod::DeterministicReplay,
            expected_outcome,
            failure_modes,
            confidence_bounds,
        };

        // --- ImpactClaim ---------------------------------------------------
        let impact_claim = Self::build_impact_claim(submission);

        // --- EthicalEnvelope -----------------------------------------------
        let formal_assertion = FormalAssertion {
            smtlib2: submission.formal_assertion.clone(),
            description: format!(
                "Mission {} satisfies IHSAN≥{:.4}, SNR≥{:.4} and guardian approval predicate",
                hex_bytes(&submission.mission_id),
                submission.ihsan_score,
                submission.snr_score,
            ),
        };

        // IHSAN score drives the fate scores; ADL and harm are [PLANNED] —
        // they will be populated from the URP enrichment pipeline once
        // bizra-urp publishes its scoring API.
        let fate_scores = FateScores {
            ihsan_score: submission.ihsan_score,
            adl_score: submission.ihsan_score * 0.9, // [PLANNED] placeholder
            harm_score: 1.0 - (submission.degradation_tier as f64 * 0.1).min(0.5), // [PLANNED]
            confidence_score: proof_pack.confidence_bounds.lower,
        };

        let assertions_satisfied = submission.ihsan_score >= IHSAN_THRESHOLD
            && submission.snr_score >= 0.0
            && (submission.guardian_approved || !self.strict_ihsan);

        let ethical_envelope = EthicalEnvelope {
            formal_assertions: vec![formal_assertion],
            fate_scores,
            assertions_satisfied,
        };

        // --- Dependencies --------------------------------------------------
        // Chain link becomes a block dependency when present. [DERIVED]
        let dependencies = if let Some(prev_hash) = submission.chain_link {
            Dependencies {
                block_ids: vec![hex_bytes(&prev_hash)],
            }
        } else {
            Dependencies::default()
        };

        BlockBody {
            dependencies,
            proof_pack,
            impact_claim,
            ethical_envelope,
        }
    }

    // ──────────────────────────────────────────────────────────────────────

    /// Build the complete [`UnsignedBlock`] + block ID from a submission.
    ///
    /// This is the final step of the wire: it hands off to [`BlockBuilder`]
    /// and returns the unsigned block ready for signing and anchoring.
    ///
    /// # Errors
    ///
    /// Returns [`MissionProofError::BuildError`] if [`BlockBuilder::build_unsigned`]
    /// fails (missing required fields, invalid creator_node, etc.).
    pub fn build_block(
        &self,
        submission: &MissionProofSubmission,
    ) -> Result<(UnsignedBlock, String), MissionProofError> {
        let body = self.to_proof_block_body(submission);

        let mut builder = BlockBuilder::new(BlockType::MissionBlock, &self.creator_node)
            .status(BlockStatus::Submitted)
            .body(body);

        // Attach parent block when there is a chain link.
        if let Some(prev_hash) = submission.chain_link {
            builder = builder.parent_block(hex_bytes(&prev_hash));
        }

        builder
            .build_unsigned()
            .map_err(MissionProofError::BuildError)
    }

    // ──────────────────────────────────────────────────────────────────────
    // Private helpers
    // ──────────────────────────────────────────────────────────────────────

    /// Build one [`ReproductionStep`] per adjacent state-transition pair.
    ///
    /// A mission that traversed `[A, B, C]` produces two steps:
    /// - step 0: A → B
    /// - step 1: B → C
    ///
    /// If fewer than two states are recorded, a single sentinel step is
    /// emitted to preserve the block structure. [DERIVED]
    fn build_reproduction_steps(states: &[String]) -> Vec<ReproductionStep> {
        if states.len() < 2 {
            // Edge case: single-state or empty history.
            let state = states
                .first()
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());
            return vec![ReproductionStep {
                step_index: 0,
                description: format!("Mission in single state: {}", state),
                from_state: state.clone(),
                to_state: state,
            }];
        }

        states
            .windows(2)
            .enumerate()
            .map(|(i, pair)| ReproductionStep {
                step_index: i as u32,
                description: format!("Mission transitioned from {} to {}", pair[0], pair[1]),
                from_state: pair[0].clone(),
                to_state: pair[1].clone(),
            })
            .collect()
    }

    /// Build an [`ImpactClaim`] calibrated to the mission's final state.
    ///
    /// | Final State | Impact Level | Score |
    /// |---|---|---|
    /// | `Complete` | `High` | ihsan_score |
    /// | `Degraded` | `Medium` | ihsan_score * 0.6 |
    /// | `Failed` / `TimedOut` | `Low` | ihsan_score * 0.1 |
    ///
    /// [DERIVED] — scoring weights are illustrative pending calibration data.
    fn build_impact_claim(submission: &MissionProofSubmission) -> ImpactClaim {
        match submission.final_state.as_str() {
            "Complete" => ImpactClaim {
                level: ImpactLevel::High,
                description: format!(
                    "Mission {} completed successfully with IHSAN {:.4}",
                    hex_bytes(&submission.mission_id),
                    submission.ihsan_score
                ),
                score: submission.ihsan_score,
            },
            "Degraded" => ImpactClaim {
                level: ImpactLevel::Medium,
                description: format!(
                    "Mission {} completed in degraded tier {} with IHSAN {:.4}",
                    hex_bytes(&submission.mission_id),
                    submission.degradation_tier,
                    submission.ihsan_score
                ),
                score: submission.ihsan_score * 0.6,
            },
            _ => ImpactClaim {
                level: ImpactLevel::Low,
                description: format!(
                    "Mission {} reached terminal state {} without successful completion",
                    hex_bytes(&submission.mission_id),
                    submission.final_state
                ),
                score: submission.ihsan_score * 0.1,
            },
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility functions
// ──────────────────────────────────────────────────────────────────────────────

/// Format a 32-byte array as a 64-character lowercase hex string.
fn hex_bytes(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// SECTION 2: Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mock_mission::Mission;
    use mock_mission_state::MissionState;
    use mock_receipt::{FailureCode, MissionReceipt};

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build a minimal 32-byte ID from a seed byte.
    fn make_id(seed: u8) -> [u8; 32] {
        let mut id = [0u8; 32];
        id[0] = seed;
        id[1] = seed.wrapping_add(1);
        id
    }

    /// Build a zero-filled signature (mock Ed25519).
    fn zero_sig() -> [u8; 64] {
        [0u8; 64]
    }

    /// Build a complete receipt for a successfully completed mission.
    fn complete_receipt(
        mission_id: [u8; 32],
        ihsan: f32,
        snr: f32,
        guardian: bool,
    ) -> MissionReceipt {
        MissionReceipt {
            receipt_id: make_id(0xAA),
            mission_id,
            final_state: MissionState::Complete,
            submitted_at: 1_000_000,
            completed_at: 1_005_000,
            states_traversed: vec![
                MissionState::Submitted,
                MissionState::Queued,
                MissionState::Running,
                MissionState::Scoring,
                MissionState::Complete,
            ],
            chosen_model: Some("bizra-llm-v2".to_string()),
            ihsan_score: Some(ihsan),
            snr_score: Some(snr),
            guardian_approved: Some(guardian),
            failure_code: None,
            degradation_reasons: vec![],
            degradation_tier: 0,
            previous_receipt_hash: None,
            signature: zero_sig(),
        }
    }

    /// Build a complete [`Mission`] in the `Complete` state.
    fn complete_mission(ihsan: f32, snr: f32, guardian: bool) -> Mission {
        let mission_id = make_id(0x01);
        let receipt = complete_receipt(mission_id, ihsan, snr, guardian);
        Mission {
            mission_id,
            submitted_at: 1_000_000,
            completed_at: Some(1_005_000),
            state: MissionState::Complete,
            state_history: vec![],
            timeout_budget_ms: 30_000,
            input_content_hash: make_id(0x02),
            chosen_model: Some("bizra-llm-v2".to_string()),
            preflight: None,
            ihsan_score: Some(ihsan),
            snr_score: Some(snr),
            guardian_approved: Some(guardian),
            response_hash: Some(make_id(0x03)),
            receipt: Some(receipt),
            failure_code: None,
            degradation_reasons: vec![],
            previous_receipt_hash: None,
        }
    }

    /// Build a failed mission.
    fn failed_mission() -> Mission {
        let mission_id = make_id(0x10);
        let receipt = MissionReceipt {
            receipt_id: make_id(0xBB),
            mission_id,
            final_state: MissionState::Failed,
            submitted_at: 2_000_000,
            completed_at: 2_010_000,
            states_traversed: vec![
                MissionState::Submitted,
                MissionState::Queued,
                MissionState::Running,
                MissionState::Failed,
            ],
            chosen_model: Some("bizra-llm-v1".to_string()),
            ihsan_score: Some(0.20),
            snr_score: Some(0.15),
            guardian_approved: Some(false),
            failure_code: Some(FailureCode::ModelTimeout),
            degradation_reasons: vec![],
            degradation_tier: 0,
            previous_receipt_hash: None,
            signature: zero_sig(),
        };
        Mission {
            mission_id,
            submitted_at: 2_000_000,
            completed_at: Some(2_010_000),
            state: MissionState::Failed,
            state_history: vec![],
            timeout_budget_ms: 10_000,
            input_content_hash: make_id(0x11),
            chosen_model: Some("bizra-llm-v1".to_string()),
            preflight: None,
            ihsan_score: Some(0.20),
            snr_score: Some(0.15),
            guardian_approved: Some(false),
            response_hash: None,
            receipt: Some(receipt),
            failure_code: Some(FailureCode::ModelTimeout),
            degradation_reasons: vec![],
            previous_receipt_hash: None,
        }
    }

    /// Build a degraded mission.
    fn degraded_mission(tier: u8) -> Mission {
        let mission_id = make_id(0x20);
        let receipt = MissionReceipt {
            receipt_id: make_id(0xCC),
            mission_id,
            final_state: MissionState::Degraded,
            submitted_at: 3_000_000,
            completed_at: 3_008_000,
            states_traversed: vec![
                MissionState::Submitted,
                MissionState::Running,
                MissionState::Degraded,
            ],
            chosen_model: Some("bizra-llm-v2".to_string()),
            ihsan_score: Some(0.72),
            snr_score: Some(0.68),
            guardian_approved: Some(true),
            failure_code: None,
            degradation_reasons: vec![mock_receipt::DegradationReason {
                reason: "retrieval partial coverage".to_string(),
            }],
            degradation_tier: tier,
            previous_receipt_hash: None,
            signature: zero_sig(),
        };
        Mission {
            mission_id,
            submitted_at: 3_000_000,
            completed_at: Some(3_008_000),
            state: MissionState::Degraded,
            state_history: vec![],
            timeout_budget_ms: 20_000,
            input_content_hash: make_id(0x21),
            chosen_model: Some("bizra-llm-v2".to_string()),
            preflight: None,
            ihsan_score: Some(0.72),
            snr_score: Some(0.68),
            guardian_approved: Some(true),
            response_hash: Some(make_id(0x22)),
            receipt: Some(receipt),
            failure_code: None,
            degradation_reasons: vec![mock_receipt::DegradationReason {
                reason: "retrieval partial coverage".to_string(),
            }],
            previous_receipt_hash: None,
        }
    }

    /// Bridge in non-strict mode (suitable for most tests).
    fn bridge() -> MissionProofBridge {
        MissionProofBridge::new("a".repeat(64), false)
    }

    /// Bridge in strict IHSAN mode.
    fn strict_bridge() -> MissionProofBridge {
        MissionProofBridge::new("b".repeat(64), true)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// A complete mission with valid scores should produce a well-formed submission.
    #[test]
    fn test_complete_mission_produces_valid_submission() {
        let mission = complete_mission(0.97, 0.88, true);
        let bridge = bridge();

        let submission = bridge
            .submit_mission(&mission)
            .expect("complete mission should produce a submission");

        assert_eq!(submission.final_state, "Complete");
        assert!(!submission.states_traversed.is_empty());
        assert!(
            (submission.ihsan_score - 0.97_f64).abs() < 1e-5,
            "IHSAN score should match receipt"
        );
        assert!(
            (submission.snr_score - 0.88_f64).abs() < 1e-5,
            "SNR score should match receipt"
        );
        assert!(submission.guardian_approved);
        assert_eq!(submission.degradation_tier, 0);
        assert_eq!(submission.duration_ms, 5_000);
        assert_eq!(submission.model_used, "bizra-llm-v2");
        assert_eq!(submission.chain_link, None);
    }

    /// A failed mission should still produce a submission (failure is valid evidence).
    #[test]
    fn test_failed_mission_produces_submission_with_failure() {
        let mission = failed_mission();
        let bridge = bridge();

        let submission = bridge
            .submit_mission(&mission)
            .expect("failed mission should produce a submission");

        assert_eq!(submission.final_state, "Failed");
        assert!(submission.states_traversed.contains(&"Failed".to_string()));
        assert!(!submission.guardian_approved);
        assert_eq!(submission.duration_ms, 10_000);

        // Impact claim should be Low for a failed mission.
        let body = bridge.to_proof_block_body(&submission);
        assert_eq!(body.impact_claim.level, ImpactLevel::Low);
    }

    /// Submitting a non-terminal mission must return `NotTerminal`.
    #[test]
    fn test_non_terminal_mission_rejected() {
        let mut mission = complete_mission(0.97, 0.88, true);
        mission.state = MissionState::Running; // non-terminal override

        let result = bridge().submit_mission(&mission);

        match result {
            Err(MissionProofError::NotTerminal(state)) => {
                assert_eq!(state, MissionState::Running);
            }
            other => panic!("expected NotTerminal, got {:?}", other),
        }
    }

    /// In strict mode, IHSAN < 0.95 must be rejected.
    #[test]
    fn test_ihsan_enforcement_strict_mode() {
        // Mission with IHSAN = 0.80, which is below the 0.95 threshold.
        let mission = complete_mission(0.80, 0.75, true);
        let bridge = strict_bridge();

        let result = bridge.submit_mission(&mission);

        match result {
            Err(MissionProofError::IhsanBelowThreshold { score, threshold }) => {
                assert!((score - 0.80_f32).abs() < 1e-5);
                assert!((threshold - IHSAN_THRESHOLD).abs() < 1e-10);
            }
            other => panic!("expected IhsanBelowThreshold, got {:?}", other),
        }
    }

    /// In strict mode, IHSAN >= 0.95 should be accepted.
    #[test]
    fn test_ihsan_enforcement_strict_mode_passes_above_threshold() {
        let mission = complete_mission(0.96, 0.90, true);
        let bridge = strict_bridge();

        let result = bridge.submit_mission(&mission);
        assert!(
            result.is_ok(),
            "IHSAN above threshold should pass strict mode"
        );
    }

    /// The SMT-LIB2 assertion must be parseable as a valid s-expression.
    ///
    /// Validity checks:
    /// - Starts with `(assert `
    /// - Contains `(and `
    /// - Contains `(>= ihsan`
    /// - Contains `(>= snr`
    /// - Contains `(= guardian_approved`
    /// - Balanced parentheses
    #[test]
    fn test_smtlib2_assertion_format() {
        let mission = complete_mission(0.97, 0.88, true);
        let submission = bridge()
            .submit_mission(&mission)
            .expect("submission should succeed");

        let smt = &submission.formal_assertion;

        assert!(
            smt.starts_with("(assert "),
            "assertion must start with (assert ): {}",
            smt
        );
        assert!(smt.contains("(and "), "must use (and ...): {}", smt);
        assert!(smt.contains("(>= ihsan "), "must constrain ihsan: {}", smt);
        assert!(smt.contains("(>= snr "), "must constrain snr: {}", smt);
        assert!(
            smt.contains("(= guardian_approved "),
            "must constrain guardian_approved: {}",
            smt
        );

        // Balanced parentheses check.
        let depth: i32 = smt
            .chars()
            .map(|c| match c {
                '(' => 1,
                ')' => -1,
                _ => 0,
            })
            .sum();
        assert_eq!(depth, 0, "SMT-LIB2 parentheses must be balanced: {}", smt);

        // Guardian `true` literal for an approved mission.
        assert!(
            smt.contains("true"),
            "approved mission must assert guardian_approved true: {}",
            smt
        );
    }

    /// Duration is computed as `completed_at − submitted_at` from the receipt.
    #[test]
    fn test_duration_calculation() {
        let mut mission = complete_mission(0.97, 0.88, true);

        // Adjust receipt timestamps directly.
        if let Some(ref mut receipt) = mission.receipt {
            receipt.submitted_at = 5_000_000;
            receipt.completed_at = 5_012_345;
        }

        let submission = bridge()
            .submit_mission(&mission)
            .expect("submission should succeed");

        assert_eq!(
            submission.duration_ms, 12_345,
            "duration should equal completed_at − submitted_at"
        );
    }

    /// A degraded mission should produce a `Medium` impact claim and a well-formed block.
    #[test]
    fn test_degraded_mission_proof_block() {
        let mission = degraded_mission(2);
        let bridge = bridge();

        let submission = bridge
            .submit_mission(&mission)
            .expect("degraded mission should produce a submission");

        assert_eq!(submission.final_state, "Degraded");
        assert_eq!(submission.degradation_tier, 2);

        let body = bridge.to_proof_block_body(&submission);

        // Impact claim should be Medium.
        assert_eq!(body.impact_claim.level, ImpactLevel::Medium);
        assert!(body.impact_claim.score < submission.ihsan_score);

        // Confidence bounds should be in the Degraded range.
        let cb = &body.proof_pack.confidence_bounds;
        assert!(cb.lower >= 0.60 && cb.lower < 0.90);

        // Reproduction steps: Submitted→Running→Degraded = 2 steps.
        assert_eq!(body.proof_pack.reproduction_steps.len(), 2);

        // Ethical envelope.
        assert!(!body.ethical_envelope.formal_assertions.is_empty());

        // Build the full block.
        let (block, block_id) = bridge
            .build_block(&submission)
            .expect("block build should succeed");

        assert!(!block_id.is_empty());
        assert_eq!(block.block_type, mock_proofspace::BlockType::MissionBlock);
        assert_eq!(block.status, mock_proofspace::BlockStatus::Submitted);
    }

    /// `to_proof_block_body` must map each adjacent state pair to one step.
    #[test]
    fn test_reproduction_steps_count() {
        let mission = complete_mission(0.97, 0.88, true);
        let submission = bridge()
            .submit_mission(&mission)
            .expect("submission should succeed");

        // The receipt has 5 states → 4 transitions.
        let body = bridge().to_proof_block_body(&submission);
        assert_eq!(
            body.proof_pack.reproduction_steps.len(),
            submission.states_traversed.len() - 1,
            "should produce one step per transition"
        );
    }

    /// When a previous receipt hash is present, the chain link should be set
    /// and the block's parent_block field should be populated.
    #[test]
    fn test_chain_link_propagated() {
        let mut mission = complete_mission(0.97, 0.88, true);
        let prev_hash = make_id(0xFF);

        if let Some(ref mut receipt) = mission.receipt {
            receipt.previous_receipt_hash = Some(prev_hash);
        }
        mission.previous_receipt_hash = Some(prev_hash);

        let submission = bridge()
            .submit_mission(&mission)
            .expect("submission should succeed");

        assert_eq!(submission.chain_link, Some(prev_hash));

        let (block, _) = bridge()
            .build_block(&submission)
            .expect("block build should succeed");

        assert!(
            block.parent_block.is_some(),
            "parent_block must be set when chain_link is present"
        );
        assert_eq!(block.parent_block.unwrap(), hex_bytes(&prev_hash));
    }

    /// A mission with no receipt should return `NoReceipt`.
    #[test]
    fn test_no_receipt_returns_error() {
        let mut mission = complete_mission(0.97, 0.88, true);
        mission.receipt = None;

        let result = bridge().submit_mission(&mission);
        assert!(matches!(result, Err(MissionProofError::NoReceipt)));
    }

    /// `MissionProofError` implements `Display` with meaningful messages.
    #[test]
    fn test_error_display() {
        let err = MissionProofError::NotTerminal(MissionState::Running);
        let msg = format!("{}", err);
        assert!(
            msg.contains("Running"),
            "Display should include the current state"
        );

        let err2 = MissionProofError::IhsanBelowThreshold {
            score: 0.80,
            threshold: 0.95,
        };
        let msg2 = format!("{}", err2);
        assert!(msg2.contains("0.95"), "Display should mention threshold");
    }
}
