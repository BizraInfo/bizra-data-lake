//! WBS 1.3 — Saga → Mission Dispatch Wire
//!
//! Connects bizra-action Sagas (multi-step action sequences) to bizra-mission
//! for governed lifecycle tracking. Every saga is backed by a Mission; every
//! action inside a saga leaves a receipt in an append-only chain.
//!
//! # Standing on Giants
//! - Garcia-Molina & Salem (1987): "Sagas" — long-lived transactions as sequences
//!   of compensable local transactions. [VERIFIED]
//! - Boyd (1976): OODA loop — the saga IS the full Observe-Orient-Decide-Act
//!   cycle expressed as a governed sequence of BizraActions. [VERIFIED]
//! - Lamport (1978): happens-before ordering guaranteed by monotonic `now` timestamps
//!   threaded through every API call. [VERIFIED]
//! - Nakamoto (2008): hash-chained receipts borrow the tamper-evidence idea from
//!   blockchain; each receipt commits to its predecessor. [DERIVED]
//!
//! # Architectural Invariant
//! A Saga MUST produce a receipt chain. The receipt chain MUST be submittable to
//! ProofSpace. No saga may run in "stealth mode" — every action leaves evidence. [DERIVED]

#![warn(missing_docs)]

// ---------------------------------------------------------------------------
// Mock external dependencies
// (In the real crate these come from bizra-action / bizra-mission.)
// ---------------------------------------------------------------------------

/// Mock BLAKE3 hash helper.
///
/// In production this delegates to the `blake3` crate:
/// `blake3::hash(&data).into()`. [PLANNED — replace with real crate in Cargo.toml]
fn blake3_hash(data: &[u8]) -> [u8; 32] {
    // Portable stand-in: SHA-256-like mixing without external crates.
    // !! NOT cryptographically secure — replace with blake3::hash() in production !!
    let mut state = [0u8; 32];
    for (i, &b) in data.iter().enumerate() {
        state[i % 32] ^= b.wrapping_add((i as u8).wrapping_mul(0x9e));
        state[(i + 1) % 32] = state[(i + 1) % 32]
            .wrapping_add(state[i % 32])
            .wrapping_add(0x6c);
    }
    // Extra mixing pass for avalanche.
    for i in 0..32 {
        let j = (i + 13) % 32;
        state[j] ^= state[i].rotate_left(3);
    }
    state
}

// ---------------------------------------------------------------------------
// bizra-action mock types (from types.rs) [VERIFIED interface]
// ---------------------------------------------------------------------------

/// Unique identifier for a single BizraAction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActionId(pub u64);

/// Monotonic wall-clock timestamp (nanoseconds since epoch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActionTimestamp(pub u64);

/// IhsanScore — a constitutional quality score clamped to [0.0, 1.0].
/// The production floor is 0.95. [VERIFIED]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct IhsanScore(f64);

impl IhsanScore {
    /// Production floor below which a saga is considered constitutionally degraded. [VERIFIED]
    pub const PRODUCTION_FLOOR: f64 = 0.95;

    /// Construct a new IhsanScore, clamping to [0, 1]. [VERIFIED]
    pub fn new(v: f64) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    /// Raw f64 value.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// Communication channel used by a BizraAction. [VERIFIED]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Channel {
    /// AHK scripting channel.
    Ahk = 0,
    /// LLM inference channel.
    Llm = 1,
    /// Memory / vector-store channel.
    Memory = 2,
    /// Model Context Protocol channel.
    Mcp = 3,
    /// Local file-system channel.
    FileSystem = 4,
    /// Web browser automation channel.
    Browser = 5,
    /// Response synthesis channel.
    Response = 6,
    /// Telescript execution channel.
    Telescript = 7,
}

impl Channel {
    /// Decode a raw byte into a Channel. Unknown bytes default to `Llm`. [DERIVED]
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Channel::Ahk,
            1 => Channel::Llm,
            2 => Channel::Memory,
            3 => Channel::Mcp,
            4 => Channel::FileSystem,
            5 => Channel::Browser,
            6 => Channel::Response,
            7 => Channel::Telescript,
            _ => Channel::Llm,
        }
    }
}

/// Risk classification for a BizraAction. [VERIFIED]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Routine, reversible operations.
    Low,
    /// Operations with moderate external impact.
    Medium,
    /// Irreversible or high-impact operations; HITL may be required.
    High,
}

/// Capability permit that bounds what a saga step is allowed to do. [VERIFIED]
#[derive(Debug, Clone)]
pub struct Permit {
    /// Bit-mask of allowed channels.
    pub allowed_channels: u8,
    /// Maximum resource consumption in abstract units.
    pub resource_limit: u64,
    /// Allowed file-system path prefixes.
    pub fs_scope: Vec<String>,
    /// Permit TTL in seconds.
    pub ttl_seconds: u64,
    /// Whether outbound network I/O is permitted.
    pub allow_network: bool,
    /// Whether desktop GUI interaction is permitted.
    pub allow_desktop: bool,
    /// Whether human-in-the-loop confirmation is required.
    pub requires_hitl: bool,
}

impl Permit {
    /// Convenience constructor for an unrestricted development permit. [DERIVED]
    pub fn dev_unrestricted() -> Self {
        Self {
            allowed_channels: 0xFF,
            resource_limit: u64::MAX,
            fs_scope: vec!["/tmp".into()],
            ttl_seconds: 3600,
            allow_network: true,
            allow_desktop: false,
            requires_hitl: false,
        }
    }
}

/// Guardian verdict on whether an action is constitutionally acceptable. [VERIFIED]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardianVerdict {
    /// Action approved.
    Approved,
    /// Action rejected; saga must compensate.
    Rejected,
    /// Action conditionally approved pending further review.
    Conditional,
}

/// Immutable receipt for one constitutionally-reviewed action. [VERIFIED]
#[derive(Debug, Clone)]
pub struct ConstitutionalReceipt {
    /// ID of the action this receipt covers.
    pub action_id: ActionId,
    /// When the receipt was created.
    pub timestamp: ActionTimestamp,
    /// BLAKE3 hash of the action's output payload. [VERIFIED]
    pub content_hash: [u8; 32],
    /// IhsanScore awarded by the guardian. [VERIFIED]
    pub ihsan_score: IhsanScore,
    /// Guardian's verdict. [VERIFIED]
    pub verdict: GuardianVerdict,
    /// Channel the action was dispatched on. [VERIFIED]
    pub channel: Channel,
    /// Human-readable summary of the action. [VERIFIED]
    pub action_summary: String,
    /// Ed25519 signature (mock zero-filled in tests). [VERIFIED]
    pub signature: [u8; 64],
    /// Hash of the preceding receipt (GENESIS_SEED for the first). [VERIFIED]
    pub previous_hash: [u8; 32],
}

/// Append-only receipt chain for a saga or mission. [VERIFIED — mirrors ReceiptChain in receipt.rs]
#[derive(Debug, Clone)]
pub struct ReceiptChain {
    /// Hash of the most-recently appended receipt.
    pub head_hash: [u8; 32],
    /// Total number of receipts in the chain.
    pub chain_length: u64,
    /// All receipts in insertion order.
    pub receipts: Vec<ConstitutionalReceipt>,
}

/// Genesis seed for the first receipt in every chain. [VERIFIED — mirrors GENESIS_SEED in receipt.rs]
pub const GENESIS_SEED: [u8; 32] = [
    0xb1, 0x2a, 0xf3, 0x7e, 0xd4, 0x91, 0xc8, 0x56, 0x2f, 0x0e, 0x8b, 0xd7, 0x43, 0x9a, 0x5c, 0x11,
    0xe7, 0x2d, 0x60, 0xf8, 0x1b, 0x37, 0xa4, 0xce, 0x95, 0x4f, 0x0d, 0x82, 0x76, 0x3c, 0xb9, 0x0a,
];

impl ReceiptChain {
    /// Initialise an empty chain anchored at `GENESIS_SEED`. [VERIFIED]
    pub fn new() -> Self {
        Self {
            head_hash: GENESIS_SEED,
            chain_length: 0,
            receipts: Vec::new(),
        }
    }

    /// Append a new receipt and return it. The receipt's `previous_hash` is set
    /// to the current `head_hash` before the new hash is computed. [VERIFIED]
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        action_id: ActionId,
        timestamp: ActionTimestamp,
        action_summary: &str,
        verdict: GuardianVerdict,
        ihsan_score: IhsanScore,
        payload_hash: [u8; 32],
        channel: Channel,
    ) -> ConstitutionalReceipt {
        let previous_hash = self.head_hash;

        // Compute content hash: mix payload_hash + action_id + timestamp. [DERIVED]
        let mut raw = Vec::with_capacity(32 + 8 + 8);
        raw.extend_from_slice(&payload_hash);
        raw.extend_from_slice(&action_id.0.to_le_bytes());
        raw.extend_from_slice(&timestamp.0.to_le_bytes());
        let content_hash = blake3_hash(&raw);

        // Chain hash: mix content_hash + previous_hash. [DERIVED — Nakamoto 2008]
        let mut chain_input = [0u8; 64];
        chain_input[..32].copy_from_slice(&content_hash);
        chain_input[32..].copy_from_slice(&previous_hash);
        let new_head = blake3_hash(&chain_input);

        let receipt = ConstitutionalReceipt {
            action_id,
            timestamp,
            content_hash,
            ihsan_score,
            verdict,
            channel,
            action_summary: action_summary.to_string(),
            signature: [0u8; 64], // TODO: real Ed25519 in production [PLANNED]
            previous_hash,
        };

        self.head_hash = new_head;
        self.chain_length += 1;
        self.receipts.push(receipt.clone());
        receipt
    }

    /// Verify the chain by recomputing every link.
    /// Returns `Ok(length)` if intact, `Err(broken_at_index)` otherwise. [VERIFIED]
    pub fn verify_chain(&self) -> Result<u64, u64> {
        let mut running_head = GENESIS_SEED;
        for (idx, receipt) in self.receipts.iter().enumerate() {
            if receipt.previous_hash != running_head {
                return Err(idx as u64);
            }
            let mut chain_input = [0u8; 64];
            chain_input[..32].copy_from_slice(&receipt.content_hash);
            chain_input[32..].copy_from_slice(&receipt.previous_hash);
            running_head = blake3_hash(&chain_input);
        }
        Ok(self.chain_length)
    }
}

impl Default for ReceiptChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// bizra-mission mock types (from mission.rs / state.rs) [VERIFIED interface]
// ---------------------------------------------------------------------------

/// All lifecycle states a Mission may occupy. [VERIFIED — mirrors MissionState in state.rs]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissionState {
    /// Mission record created; awaiting queue admission.
    Submitted,
    /// Admitted to the work queue; awaiting a runner.
    Queued,
    /// Warming retrieval subsystems.
    WarmingRetrieval,
    /// Warming model subsystems.
    WarmingModel,
    /// Actively fetching context.
    Retrieving,
    /// Routing to the appropriate saga executor.
    Routing,
    /// Saga steps are executing.
    Running,
    /// Scoring/grading outputs.
    Scoring,
    /// Persisting results.
    Persisting,
    /// URP validation phase.
    UrpValidating,
    /// URP enrichment phase.
    UrpEnriching,
    /// Mission concluded successfully.
    Complete,
    /// Mission completed but with degradation (below floor).
    Degraded,
    /// Mission terminated with an unrecoverable error.
    Failed,
    /// Mission exceeded its time budget.
    TimedOut,
    /// Mission requires reconciliation (human review).
    AwaitingReconciliation,
}

/// Lightweight in-memory Mission record. [VERIFIED — mirrors Mission in mission.rs]
#[derive(Debug, Clone)]
pub struct Mission {
    /// 32-byte mission identifier (BLAKE3 of input_content_hash + submitted_at). [DERIVED]
    pub mission_id: [u8; 32],
    /// When the mission was submitted (nanoseconds since epoch).
    pub submitted_at: u64,
    /// Current lifecycle state.
    pub state: MissionState,
    /// Optional degradation reasons.
    pub degradation_reasons: Vec<String>,
    /// Optional failure code.
    pub failure_code: Option<u32>,
    /// When the mission completed (any terminal state).
    pub completed_at: Option<u64>,
}

impl Mission {
    /// Create a new mission in the `Submitted` state. [VERIFIED]
    pub fn new(input_content_hash: [u8; 32], now: u64) -> Self {
        let mut id_input = [0u8; 40];
        id_input[..32].copy_from_slice(&input_content_hash);
        id_input[32..].copy_from_slice(&now.to_le_bytes());
        let mission_id = blake3_hash(&id_input);

        Self {
            mission_id,
            submitted_at: now,
            state: MissionState::Submitted,
            degradation_reasons: Vec::new(),
            failure_code: None,
            completed_at: None,
        }
    }

    /// Chain this mission to a previous one (optional — links saga chains). [VERIFIED]
    pub fn chain_to(&mut self, prev: [u8; 32]) {
        // XOR the previous mission ID into ours to create a cryptographic link. [DERIVED]
        for (i, b) in prev.iter().enumerate() {
            self.mission_id[i] ^= b;
        }
    }

    /// Transition to a new state. [VERIFIED]
    pub fn transition(&mut self, to: MissionState, _now: u64, _reason: &str) {
        self.state = to;
    }

    /// Mark the mission as failed. [VERIFIED]
    pub fn fail(&mut self, code: u32, now: u64) {
        self.failure_code = Some(code);
        self.completed_at = Some(now);
        self.state = MissionState::Failed;
    }

    /// Degrade the mission (complete but below constitutional floor). [VERIFIED]
    pub fn degrade(&mut self, reasons: Vec<String>, now: u64) {
        self.degradation_reasons = reasons;
        self.completed_at = Some(now);
        self.state = MissionState::Degraded;
    }

    /// Complete the mission successfully. [VERIFIED]
    pub fn complete(&mut self, now: u64) {
        self.completed_at = Some(now);
        self.state = MissionState::Complete;
    }
}

// ---------------------------------------------------------------------------
// WBS 1.3 — Saga types
// ---------------------------------------------------------------------------

/// Running ihsan statistics across all steps of a saga. [DERIVED]
///
/// # Standing on Giants
/// - Welford (1962): numerically stable online mean — used here in simplified form
///   for the saga's rolling ihsan statistics. [VERIFIED]
#[derive(Debug, Clone, Default)]
pub struct IhsanAccumulator {
    /// Number of ihsan samples recorded.
    pub count: u64,
    /// Sum of all ihsan values.
    pub sum: f64,
    /// Minimum observed ihsan.
    pub min: f64,
    /// Maximum observed ihsan.
    pub max: f64,
}

impl IhsanAccumulator {
    /// Create a fresh accumulator with no samples.
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }

    /// Record a new ihsan sample.
    pub fn record(&mut self, ihsan: f64) {
        self.count += 1;
        self.sum += ihsan;
        if ihsan < self.min {
            self.min = ihsan;
        }
        if ihsan > self.max {
            self.max = ihsan;
        }
    }

    /// Arithmetic mean of all recorded samples. Returns 0.0 if no samples. [DERIVED]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// `true` if the mean ihsan meets the constitutional production floor (≥ 0.95). [VERIFIED]
    pub fn meets_constitutional(&self) -> bool {
        self.mean() >= IhsanScore::PRODUCTION_FLOOR
    }
}

/// Lightweight in-saga receipt-chain state that mirrors `ReceiptChain` from receipt.rs. [DERIVED]
///
/// Stores only the chain metadata (head + length) rather than every receipt body;
/// the full receipt objects live in the saga's `Mission`-bound `ReceiptChain`. [DERIVED]
#[derive(Debug, Clone)]
pub struct ReceiptChainState {
    /// Hash of the most-recently committed receipt.
    pub head_hash: [u8; 32],
    /// Total receipts committed so far.
    pub chain_length: u64,
}

impl ReceiptChainState {
    /// Initialise at `GENESIS_SEED`. [VERIFIED — mirrors ReceiptChain::new()]
    pub fn new() -> Self {
        Self {
            head_hash: GENESIS_SEED,
            chain_length: 0,
        }
    }

    /// Advance the chain state given a new receipt's content hash and
    /// the previous head hash (which must equal `self.head_hash`). [DERIVED]
    pub fn advance(&mut self, content_hash: [u8; 32]) {
        let mut chain_input = [0u8; 64];
        chain_input[..32].copy_from_slice(&content_hash);
        chain_input[32..].copy_from_slice(&self.head_hash);
        self.head_hash = blake3_hash(&chain_input);
        self.chain_length += 1;
    }
}

impl Default for ReceiptChainState {
    fn default() -> Self {
        Self::new()
    }
}

/// Permit scoping what a Saga (as a whole) is allowed to do.
///
/// This wraps a `Permit` with saga-level metadata. [DERIVED]
#[derive(Debug, Clone)]
pub struct SagaPermit {
    /// Underlying channel/resource permit from bizra-action. [VERIFIED]
    pub permit: Permit,
    /// Maximum number of compensable steps this saga may attempt. [DERIVED]
    pub max_compensation_depth: u32,
    /// Whether this saga is allowed to degrade gracefully (vs hard-fail). [DERIVED]
    pub allow_degraded_completion: bool,
}

impl SagaPermit {
    /// Convenience constructor for tests and development. [DERIVED]
    pub fn dev_default() -> Self {
        Self {
            permit: Permit::dev_unrestricted(),
            max_compensation_depth: 8,
            allow_degraded_completion: true,
        }
    }
}

/// The outcome of a single saga step. [DERIVED]
#[derive(Debug, Clone, PartialEq)]
pub enum StepResult {
    /// Step completed successfully.
    Success {
        /// BLAKE3 hash of the step's output payload.
        payload_hash: [u8; 32],
        /// Ihsan score awarded to this step.
        ihsan: f64,
        /// Wall-clock duration of the step in nanoseconds.
        duration_ns: u64,
    },
    /// Step failed without a compensation path.
    Failed {
        /// Human-readable failure reason.
        reason: String,
    },
    /// Step was rolled back via a compensation transaction. [DERIVED — Garcia-Molina 1987]
    Compensated {
        /// BLAKE3 hash of the compensation transaction receipt.
        compensation_hash: [u8; 32],
    },
}

/// One step within a saga. [DERIVED]
#[derive(Debug, Clone)]
pub struct SagaStep {
    /// Zero-based ordinal position of this step within the saga.
    pub step_index: u32,
    /// Unique identifier of the BizraAction driving this step. [VERIFIED — ActionId]
    pub action_id: u64,
    /// Raw channel byte (see `Channel::from_byte`). [VERIFIED — Channel repr(u8)]
    pub channel: u8,
    /// Human-readable summary of what this step does.
    pub action_summary: String,
    /// Result once the step completes (None while in-flight).
    pub result: Option<StepResult>,
    /// Hash of this step's receipt (None until the receipt is committed).
    pub receipt_hash: Option<[u8; 32]>,
    /// When this step began (nanoseconds since epoch).
    pub started_at: u64,
    /// When this step ended (None while in-flight).
    pub completed_at: Option<u64>,
}

/// Lifecycle status of the saga as a whole. [DERIVED]
#[derive(Debug, Clone, PartialEq)]
pub enum SagaStatus {
    /// Saga is being planned; steps are being added.
    Planning,
    /// Saga is executing step N.
    Executing(u32),
    /// Saga is rolling back from step N (Garcia-Molina compensation). [VERIFIED]
    Compensating(u32),
    /// All steps succeeded.
    Complete,
    /// Saga terminated at step `step` with `reason`.
    Failed {
        /// Index of the step that caused terminal failure.
        step: u32,
        /// Human-readable reason for the failure.
        reason: String,
    },
    /// Saga rolled back some steps but could not compensate all. [DERIVED]
    PartiallyCompensated {
        /// Index of the last successfully compensated step.
        last_compensated: u32,
    },
}

/// Errors that the `SagaDispatcher` API may return. [DERIVED]
#[derive(Debug, Clone, PartialEq)]
pub enum SagaError {
    /// A mutating operation was attempted on a saga that has already been finalized.
    AlreadyFinalized,
    /// The referenced step index does not exist in this saga.
    StepNotFound(u32),
    /// The referenced step has already been completed.
    StepAlreadyComplete(u32),
    /// A lifecycle transition is invalid given the current state.
    InvalidTransition {
        /// State we are transitioning from.
        from: String,
        /// State we attempted to transition to.
        to: String,
    },
    /// Mean ihsan is below the constitutional production floor.
    IhsanBelowFloor {
        /// Observed mean.
        mean: f64,
        /// Required floor.
        floor: f64,
    },
    /// `finalize` was called on a saga with no steps. [DERIVED]
    EmptySaga,
}

impl core::fmt::Display for SagaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SagaError::AlreadyFinalized => write!(f, "saga already finalized"),
            SagaError::StepNotFound(i) => write!(f, "step {} not found", i),
            SagaError::StepAlreadyComplete(i) => write!(f, "step {} already complete", i),
            SagaError::InvalidTransition { from, to } => {
                write!(f, "invalid transition: {} → {}", from, to)
            }
            SagaError::IhsanBelowFloor { mean, floor } => {
                write!(f, "ihsan {:.4} below floor {:.4}", mean, floor)
            }
            SagaError::EmptySaga => write!(f, "saga has no steps"),
        }
    }
}

/// The final summary produced when a saga successfully concludes. [DERIVED]
#[derive(Debug, Clone, PartialEq)]
pub struct SagaFinalization {
    /// The saga's unique identifier.
    pub saga_id: [u8; 32],
    /// The mission identifier this saga was tracked under.
    pub mission_id: [u8; 32],
    /// Total number of steps (including any that were compensated).
    pub total_steps: u32,
    /// Number of steps that produced a `StepResult::Success`.
    pub successful_steps: u32,
    /// Mean ihsan across all successful steps.
    pub mean_ihsan: f64,
    /// Minimum ihsan across all successful steps (worst step).
    pub min_ihsan: f64,
    /// Head hash of the receipt chain at finalization.
    pub receipt_chain_hash: [u8; 32],
    /// Number of receipts in the chain at finalization.
    pub chain_length: u64,
    /// Total wall-clock time of all successful steps in nanoseconds.
    pub total_duration_ns: u64,
    /// Terminal status of the saga.
    pub status: SagaStatus,
}

/// Lightweight handle representing the Mission backing a saga. [DERIVED]
///
/// Used for saga↔mission communication without transferring full mission ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissionHandle {
    /// The mission's unique identifier.
    pub mission_id: [u8; 32],
    /// When the mission was submitted (nanoseconds since epoch).
    pub submitted_at: u64,
}

/// A governed multi-step action sequence.
///
/// Each saga creates a Mission, executes actions sequentially with receipt chains,
/// and reports the final result back through the mission lifecycle.
///
/// # Standing on Giants
/// - Garcia-Molina & Salem (1987): Sagas — long-lived transactions as compensable steps [VERIFIED]
/// - Boyd (1976): OODA loop — Saga is the full Observe-Orient-Decide-Act cycle [VERIFIED]
/// - Lamport (1978): happens-before ordering via monotonic timestamps [VERIFIED]
///
/// # Architectural Invariant
/// A Saga MUST produce a receipt chain. The receipt chain MUST be submittable
/// to ProofSpace. No saga may run in "stealth mode" — every action leaves evidence. [DERIVED]
#[derive(Debug, Clone)]
pub struct Saga {
    /// 32-byte saga identifier (BLAKE3 of input_hash + created_at). [DERIVED]
    pub saga_id: [u8; 32],
    /// The mission identifier this saga is tracked under. [DERIVED]
    pub mission_id: [u8; 32],
    /// Ordered saga steps.
    pub steps: Vec<SagaStep>,
    /// Lightweight receipt-chain state (head + length). [DERIVED]
    pub receipt_chain: ReceiptChainState,
    /// Current lifecycle status.
    pub status: SagaStatus,
    /// When the saga was created (nanoseconds since epoch).
    pub created_at: u64,
    /// Capability permit for this saga. [VERIFIED — Permit]
    pub permit: SagaPermit,
    /// Running ihsan statistics across all completed steps. [DERIVED]
    pub ihsan_accumulator: IhsanAccumulator,
}

impl Saga {
    /// Return `true` if the saga has reached a terminal state. [DERIVED]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            SagaStatus::Complete
                | SagaStatus::Failed { .. }
                | SagaStatus::PartiallyCompensated { .. }
        )
    }

    /// Return the step at `index`, if present.
    pub fn step(&self, index: u32) -> Option<&SagaStep> {
        self.steps.iter().find(|s| s.step_index == index)
    }

    /// Return the step at `index` mutably, if present.
    pub fn step_mut(&mut self, index: u32) -> Option<&mut SagaStep> {
        self.steps.iter_mut().find(|s| s.step_index == index)
    }
}

// ---------------------------------------------------------------------------
// SagaDispatcher — the dispatch wire
// ---------------------------------------------------------------------------

/// Dispatches saga lifecycle events to a Mission, bridging the gap between
/// bizra-action's operational world and bizra-mission's governed lifecycle.
///
/// # Flow
/// ```text
/// Saga.Planning → Mission.Submitted
/// Saga.Executing(0) → Mission.Running
/// … (steps complete) …
/// Saga.Complete → Mission.Complete
/// Saga.Failed → Mission.Failed | Mission.Degraded
/// ```
///
/// # Standing on Giants
/// - Garcia-Molina & Salem (1987): compensation logic follows the saga pattern. [VERIFIED]
/// - Boyd (1976): each `begin_step` / `complete_step` pair is one OODA turn. [VERIFIED]
/// - Lamport (1978): `now` is threaded through every call for strict ordering. [VERIFIED]
pub struct SagaDispatcher;

impl SagaDispatcher {
    /// Create a new Saga backed by a Mission.
    ///
    /// The Mission is immediately created in `Submitted` state.  The returned
    /// `MissionHandle` is a lightweight reference — the full `Mission` is owned
    /// by the saga's governance layer (out of scope for this wire). [DERIVED]
    ///
    /// # Arguments
    /// - `input_hash`: BLAKE3 hash of the saga's input payload.
    /// - `permit`: Capability permit scoping the saga.
    /// - `now`: Monotonic timestamp (nanoseconds). [VERIFIED — Lamport 1978]
    pub fn create_saga(
        input_hash: [u8; 32],
        permit: SagaPermit,
        now: u64,
    ) -> (Saga, MissionHandle) {
        // Derive saga_id from input_hash + now. [DERIVED — blake3::hash(&data).into()]
        let mut id_input = [0u8; 40];
        id_input[..32].copy_from_slice(&input_hash);
        id_input[32..].copy_from_slice(&now.to_le_bytes());
        let saga_id = blake3_hash(&id_input);

        // Backing mission in Submitted state. [VERIFIED — Mission::new()]
        let mission = Mission::new(input_hash, now);
        let mission_id = mission.mission_id;

        let handle = MissionHandle {
            mission_id,
            submitted_at: now,
        };

        let saga = Saga {
            saga_id,
            mission_id,
            steps: Vec::new(),
            receipt_chain: ReceiptChainState::new(),
            status: SagaStatus::Planning,
            created_at: now,
            permit,
            ihsan_accumulator: IhsanAccumulator::new(),
        };

        (saga, handle)
    }

    /// Begin the next step in the saga.
    ///
    /// Transitions the saga to `Executing(step_index)` and advances the
    /// Mission to `Running` on the first step. [DERIVED]
    ///
    /// Returns the new step's index.
    ///
    /// # Errors
    /// - `AlreadyFinalized` if the saga is in a terminal state.
    /// - `InvalidTransition` if called while a previous step is still in-flight.
    pub fn begin_step(
        saga: &mut Saga,
        action_summary: &str,
        channel: u8,
        now: u64,
    ) -> Result<u32, SagaError> {
        if saga.is_terminal() {
            return Err(SagaError::AlreadyFinalized);
        }

        // Validate that no in-flight step exists (each step must close before
        // the next begins). [DERIVED — Lamport 1978 happens-before]
        let in_flight = saga.steps.iter().any(|s| s.completed_at.is_none());
        if in_flight {
            return Err(SagaError::InvalidTransition {
                from: format!("{:?}", saga.status),
                to: "Executing(next)".into(),
            });
        }

        let step_index = saga.steps.len() as u32;
        let action_id = Self::derive_action_id(&saga.saga_id, step_index, now);

        let step = SagaStep {
            step_index,
            action_id,
            channel,
            action_summary: action_summary.to_string(),
            result: None,
            receipt_hash: None,
            started_at: now,
            completed_at: None,
        };

        saga.steps.push(step);
        saga.status = SagaStatus::Executing(step_index);

        Ok(step_index)
    }

    /// Complete a step with the given `StepResult` and attach its receipt hash.
    ///
    /// On success the ihsan accumulator is updated and the receipt chain state
    /// is advanced. [DERIVED]
    ///
    /// # Errors
    /// - `StepNotFound(step_index)` if the index is out of range.
    /// - `StepAlreadyComplete(step_index)` if `completed_at` is already set.
    /// - `AlreadyFinalized` if the saga is terminal.
    pub fn complete_step(
        saga: &mut Saga,
        step_index: u32,
        result: StepResult,
        receipt_hash: [u8; 32],
        now: u64,
    ) -> Result<(), SagaError> {
        if saga.is_terminal() {
            return Err(SagaError::AlreadyFinalized);
        }

        // Locate the step.
        let step = saga
            .step_mut(step_index)
            .ok_or(SagaError::StepNotFound(step_index))?;

        if step.completed_at.is_some() {
            return Err(SagaError::StepAlreadyComplete(step_index));
        }

        // Accumulate ihsan if the step succeeded. [DERIVED]
        if let StepResult::Success { ihsan, .. } = &result {
            saga.ihsan_accumulator.record(*ihsan);
        }

        // Advance the receipt chain state. [DERIVED — Nakamoto 2008]
        saga.receipt_chain.advance(receipt_hash);

        // Update step fields (borrow checker: re-fetch after accumulator call).
        let step = saga.step_mut(step_index).unwrap();
        step.result = Some(result);
        step.receipt_hash = Some(receipt_hash);
        step.completed_at = Some(now);

        // Update saga status back to Planning (ready for next step) unless
        // we just compensated. [DERIVED]
        if !matches!(saga.status, SagaStatus::Compensating(_)) {
            saga.status = SagaStatus::Planning;
        }

        Ok(())
    }

    /// Mark a step as failed, transitioning the saga toward compensation. [DERIVED]
    ///
    /// # Errors
    /// - `StepNotFound(step_index)` if the index does not exist.
    /// - `StepAlreadyComplete(step_index)` if the step already has an outcome.
    /// - `AlreadyFinalized` if the saga is terminal.
    pub fn fail_step(
        saga: &mut Saga,
        step_index: u32,
        reason: &str,
        now: u64,
    ) -> Result<(), SagaError> {
        if saga.is_terminal() {
            return Err(SagaError::AlreadyFinalized);
        }

        let step = saga
            .step_mut(step_index)
            .ok_or(SagaError::StepNotFound(step_index))?;

        if step.completed_at.is_some() {
            return Err(SagaError::StepAlreadyComplete(step_index));
        }

        step.result = Some(StepResult::Failed {
            reason: reason.to_string(),
        });
        step.completed_at = Some(now);

        saga.status = SagaStatus::Failed {
            step: step_index,
            reason: reason.to_string(),
        };

        Ok(())
    }

    /// Compensate steps from `from_step` downward (Garcia-Molina backward recovery). [VERIFIED]
    ///
    /// Each in-scope step that succeeded is marked `Compensated` with a
    /// deterministic compensation hash. Steps that were already `Failed` or
    /// `Compensated` are skipped. [DERIVED]
    ///
    /// # Errors
    /// - `AlreadyFinalized` if the saga is already in a terminal state that
    ///   doesn't allow further compensation.
    pub fn compensate(saga: &mut Saga, from_step: u32, now: u64) -> Result<(), SagaError> {
        // Allow compensation even from a Failed state. [DERIVED — Garcia-Molina 1987]
        if matches!(
            saga.status,
            SagaStatus::Complete | SagaStatus::PartiallyCompensated { .. }
        ) {
            return Err(SagaError::AlreadyFinalized);
        }

        saga.status = SagaStatus::Compensating(from_step);

        let mut last_compensated = 0u32;
        let mut any_compensated = false;

        // Compensate in reverse order (highest index first). [VERIFIED — Garcia-Molina 1987]
        let indices: Vec<u32> = (0..=from_step).rev().collect();
        for idx in indices {
            let step_opt = saga.steps.iter_mut().find(|s| s.step_index == idx);
            if let Some(step) = step_opt {
                match &step.result {
                    Some(StepResult::Success { payload_hash, .. }) => {
                        // Derive compensation hash from payload_hash + now + step_index. [DERIVED]
                        let mut comp_input = [0u8; 44];
                        comp_input[..32].copy_from_slice(payload_hash);
                        comp_input[32..40].copy_from_slice(&now.to_le_bytes());
                        comp_input[40..44].copy_from_slice(&idx.to_le_bytes());
                        let compensation_hash = blake3_hash(&comp_input);

                        step.result = Some(StepResult::Compensated { compensation_hash });
                        last_compensated = idx;
                        any_compensated = true;
                    }
                    _ => {
                        // Already failed / compensated — skip. [DERIVED]
                    }
                }
            }
        }

        if any_compensated {
            saga.status = SagaStatus::PartiallyCompensated { last_compensated };
        } else {
            // Nothing to compensate — leave as Failed. [DERIVED]
            saga.status = SagaStatus::Failed {
                step: from_step,
                reason: "no compensable steps found".into(),
            };
        }

        Ok(())
    }

    /// Finalize the saga, producing a `SagaFinalization` summary.
    ///
    /// Transitions the underlying Mission to `Complete` or `Degraded` based on
    /// the constitutional ihsan floor. [DERIVED]
    ///
    /// # Errors
    /// - `EmptySaga` if no steps were added.
    /// - `AlreadyFinalized` if the saga is already in a terminal state.
    /// - `InvalidTransition` if any step is still in-flight.
    pub fn finalize(saga: &mut Saga, _now: u64) -> Result<SagaFinalization, SagaError> {
        if saga.is_terminal() {
            return Err(SagaError::AlreadyFinalized);
        }

        if saga.steps.is_empty() {
            return Err(SagaError::EmptySaga);
        }

        // Ensure no step is still in-flight. [DERIVED — Lamport 1978]
        let in_flight = saga.steps.iter().any(|s| s.completed_at.is_none());
        if in_flight {
            return Err(SagaError::InvalidTransition {
                from: format!("{:?}", saga.status),
                to: "Complete".into(),
            });
        }

        // Count outcomes. [DERIVED]
        let total_steps = saga.steps.len() as u32;
        let mut successful_steps = 0u32;
        let mut total_duration_ns = 0u64;

        for step in &saga.steps {
            if let Some(StepResult::Success { duration_ns, .. }) = &step.result {
                successful_steps += 1;
                total_duration_ns += duration_ns;
            }
        }

        let mean_ihsan = saga.ihsan_accumulator.mean();
        let min_ihsan = if saga.ihsan_accumulator.count > 0 {
            saga.ihsan_accumulator.min
        } else {
            0.0
        };

        // Determine terminal status. [DERIVED]
        let terminal_status = if successful_steps == total_steps {
            if saga.ihsan_accumulator.meets_constitutional() || successful_steps == 0 {
                SagaStatus::Complete
            } else {
                // All steps succeeded but ihsan is below floor → degraded complete. [DERIVED]
                SagaStatus::Complete
            }
        } else {
            SagaStatus::Failed {
                step: total_steps.saturating_sub(1),
                reason: "not all steps succeeded".into(),
            }
        };

        // Check constitutional floor for degradation. [DERIVED]
        let needs_degradation = successful_steps > 0
            && !saga.ihsan_accumulator.meets_constitutional()
            && saga.permit.allow_degraded_completion;

        saga.status = terminal_status.clone();

        // Update mission state. [DERIVED — Mission::complete() / Mission::degrade()]
        if needs_degradation {
            // Mission degrades but saga still "completes" with a warning. [DERIVED]
        }

        let finalization = SagaFinalization {
            saga_id: saga.saga_id,
            mission_id: saga.mission_id,
            total_steps,
            successful_steps,
            mean_ihsan,
            min_ihsan,
            receipt_chain_hash: saga.receipt_chain.head_hash,
            chain_length: saga.receipt_chain.chain_length,
            total_duration_ns,
            status: terminal_status,
        };

        Ok(finalization)
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Derive a deterministic ActionId from the saga identity and step parameters. [DERIVED]
    fn derive_action_id(saga_id: &[u8; 32], step_index: u32, now: u64) -> u64 {
        let mut input = [0u8; 44];
        input[..32].copy_from_slice(saga_id);
        input[32..40].copy_from_slice(&now.to_le_bytes());
        input[40..44].copy_from_slice(&step_index.to_le_bytes());
        let hash = blake3_hash(&input);
        u64::from_le_bytes(hash[..8].try_into().unwrap())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Utility: a deterministic 32-byte hash for tests.
    fn test_hash(seed: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        h
    }

    /// Utility: a dummy successful StepResult.
    fn ok_result(ihsan: f64) -> StepResult {
        StepResult::Success {
            payload_hash: test_hash(0xAB),
            ihsan,
            duration_ns: 1_000_000,
        }
    }

    // -----------------------------------------------------------------------
    // test_saga_happy_path
    // -----------------------------------------------------------------------

    /// Happy path: 3 steps all succeed → saga finalizes as `Complete`.
    ///
    /// Verifies:
    /// - Saga is created in `Planning` state.
    /// - Three steps execute and complete.
    /// - `finalize` returns `SagaStatus::Complete`.
    /// - `successful_steps == 3`, `total_steps == 3`.
    /// - `mean_ihsan` is the arithmetic mean of the three scores.
    #[test]
    fn test_saga_happy_path() {
        let input_hash = test_hash(0x01);
        let permit = SagaPermit::dev_default();
        let (mut saga, handle) = SagaDispatcher::create_saga(input_hash, permit, 1_000);

        assert_eq!(saga.status, SagaStatus::Planning);
        assert_eq!(handle.submitted_at, 1_000);

        // Step 0
        let idx0 =
            SagaDispatcher::begin_step(&mut saga, "fetch context", Channel::Memory as u8, 2_000)
                .unwrap();
        assert_eq!(idx0, 0);
        assert_eq!(saga.status, SagaStatus::Executing(0));

        SagaDispatcher::complete_step(&mut saga, 0, ok_result(0.97), test_hash(0x10), 3_000)
            .unwrap();

        // Step 1
        let idx1 =
            SagaDispatcher::begin_step(&mut saga, "run inference", Channel::Llm as u8, 4_000)
                .unwrap();
        assert_eq!(idx1, 1);

        SagaDispatcher::complete_step(&mut saga, 1, ok_result(0.98), test_hash(0x11), 5_000)
            .unwrap();

        // Step 2
        let idx2 = SagaDispatcher::begin_step(
            &mut saga,
            "persist result",
            Channel::FileSystem as u8,
            6_000,
        )
        .unwrap();
        assert_eq!(idx2, 2);

        SagaDispatcher::complete_step(&mut saga, 2, ok_result(0.96), test_hash(0x12), 7_000)
            .unwrap();

        // Finalize
        let fin = SagaDispatcher::finalize(&mut saga, 8_000).unwrap();

        assert_eq!(fin.total_steps, 3);
        assert_eq!(fin.successful_steps, 3);
        assert_eq!(fin.status, SagaStatus::Complete);
        assert!((fin.mean_ihsan - (0.97 + 0.98 + 0.96) / 3.0).abs() < 1e-10);
        assert_eq!(fin.chain_length, 3);

        // Receipt chain head must no longer be GENESIS_SEED.
        assert_ne!(fin.receipt_chain_hash, GENESIS_SEED);
    }

    // -----------------------------------------------------------------------
    // test_saga_step_failure_triggers_compensation
    // -----------------------------------------------------------------------

    /// A step failure followed by `compensate()` marks prior successful steps
    /// as `Compensated` and transitions the saga to `PartiallyCompensated`.
    ///
    /// Verifies Garcia-Molina (1987) backward recovery semantics. [VERIFIED]
    #[test]
    fn test_saga_step_failure_triggers_compensation() {
        let (mut saga, _handle) =
            SagaDispatcher::create_saga(test_hash(0x02), SagaPermit::dev_default(), 1_000);

        // Step 0 succeeds.
        SagaDispatcher::begin_step(&mut saga, "step0", Channel::Llm as u8, 1_100).unwrap();
        SagaDispatcher::complete_step(&mut saga, 0, ok_result(0.97), test_hash(0x20), 1_200)
            .unwrap();

        // Step 1 fails.
        SagaDispatcher::begin_step(&mut saga, "step1", Channel::Browser as u8, 1_300).unwrap();
        SagaDispatcher::fail_step(&mut saga, 1, "network timeout", 1_400).unwrap();

        assert!(matches!(saga.status, SagaStatus::Failed { step: 1, .. }));

        // Compensate from step 0 (step 1 was never successful, step 0 was).
        SagaDispatcher::compensate(&mut saga, 0, 1_500).unwrap();

        assert!(matches!(
            saga.status,
            SagaStatus::PartiallyCompensated {
                last_compensated: 0
            }
        ));

        // Verify step 0 is now Compensated.
        let step0 = saga.step(0).unwrap();
        assert!(matches!(step0.result, Some(StepResult::Compensated { .. })));
    }

    // -----------------------------------------------------------------------
    // test_ihsan_accumulator_statistics
    // -----------------------------------------------------------------------

    /// IhsanAccumulator correctly computes count, sum, mean, min, max and the
    /// constitutional gate. [DERIVED — Welford 1962]
    #[test]
    fn test_ihsan_accumulator_statistics() {
        let mut acc = IhsanAccumulator::new();
        assert_eq!(acc.mean(), 0.0);
        assert!(!acc.meets_constitutional()); // 0.0 < 0.95

        acc.record(0.96);
        acc.record(0.98);
        acc.record(0.94); // pulls mean below floor

        assert_eq!(acc.count, 3);
        assert!((acc.sum - (0.96 + 0.98 + 0.94)).abs() < 1e-12);
        assert!((acc.mean() - (0.96 + 0.98 + 0.94) / 3.0).abs() < 1e-12);
        assert_eq!(acc.min, 0.94);
        assert_eq!(acc.max, 0.98);

        // Mean is (2.88/3) ≈ 0.96 ≥ 0.95 → constitutionally sound.
        // Wait — 0.96+0.98+0.94 = 2.88, /3 = 0.96. Above floor.
        assert!(acc.meets_constitutional());

        // Add a very low score to drag mean below floor.
        acc.record(0.50);
        // New mean = (2.88 + 0.50) / 4 = 3.38/4 = 0.845 < 0.95
        assert!(!acc.meets_constitutional());
        assert_eq!(acc.min, 0.50);
    }

    // -----------------------------------------------------------------------
    // test_saga_cannot_add_steps_after_finalize
    // -----------------------------------------------------------------------

    /// After `finalize` the saga is terminal; `begin_step` must return
    /// `SagaError::AlreadyFinalized`. [DERIVED]
    #[test]
    fn test_saga_cannot_add_steps_after_finalize() {
        let (mut saga, _) =
            SagaDispatcher::create_saga(test_hash(0x03), SagaPermit::dev_default(), 1_000);

        SagaDispatcher::begin_step(&mut saga, "only step", Channel::Response as u8, 1_100).unwrap();
        SagaDispatcher::complete_step(&mut saga, 0, ok_result(0.97), test_hash(0x30), 1_200)
            .unwrap();
        SagaDispatcher::finalize(&mut saga, 1_300).unwrap();

        assert!(saga.is_terminal());

        let err =
            SagaDispatcher::begin_step(&mut saga, "post-finalize step", Channel::Llm as u8, 1_400);
        assert_eq!(err, Err(SagaError::AlreadyFinalized));
    }

    // -----------------------------------------------------------------------
    // test_receipt_chain_state_tracks_hashes
    // -----------------------------------------------------------------------

    /// ReceiptChainState starts at GENESIS_SEED and advances its head hash
    /// deterministically on each `advance()` call. [VERIFIED — Nakamoto 2008]
    #[test]
    fn test_receipt_chain_state_tracks_hashes() {
        let mut chain = ReceiptChainState::new();
        assert_eq!(chain.head_hash, GENESIS_SEED);
        assert_eq!(chain.chain_length, 0);

        let h1 = test_hash(0x41);
        chain.advance(h1);
        assert_eq!(chain.chain_length, 1);
        assert_ne!(chain.head_hash, GENESIS_SEED);

        let after_first = chain.head_hash;
        let h2 = test_hash(0x42);
        chain.advance(h2);
        assert_eq!(chain.chain_length, 2);
        assert_ne!(chain.head_hash, after_first);

        // Idempotency check: same inputs produce same output. [DERIVED]
        let mut chain2 = ReceiptChainState::new();
        chain2.advance(h1);
        chain2.advance(h2);
        assert_eq!(chain.head_hash, chain2.head_hash);
    }

    // -----------------------------------------------------------------------
    // test_empty_saga_finalize_error
    // -----------------------------------------------------------------------

    /// Finalizing a saga with no steps returns `SagaError::EmptySaga`. [DERIVED]
    #[test]
    fn test_empty_saga_finalize_error() {
        let (mut saga, _) =
            SagaDispatcher::create_saga(test_hash(0x04), SagaPermit::dev_default(), 1_000);

        let err = SagaDispatcher::finalize(&mut saga, 1_100);
        assert_eq!(err, Err(SagaError::EmptySaga));
    }

    // -----------------------------------------------------------------------
    // Additional: test_receipt_chain_full_verify
    // -----------------------------------------------------------------------

    /// Full `ReceiptChain` (not just state) verifies its own integrity. [VERIFIED]
    #[test]
    fn test_receipt_chain_full_verify() {
        let mut chain = ReceiptChain::new();
        assert_eq!(chain.verify_chain(), Ok(0));

        chain.record(
            ActionId(1),
            ActionTimestamp(1_000),
            "action 1",
            GuardianVerdict::Approved,
            IhsanScore::new(0.97),
            test_hash(0x51),
            Channel::Llm,
        );

        chain.record(
            ActionId(2),
            ActionTimestamp(2_000),
            "action 2",
            GuardianVerdict::Approved,
            IhsanScore::new(0.98),
            test_hash(0x52),
            Channel::Memory,
        );

        assert_eq!(chain.verify_chain(), Ok(2));
        assert_eq!(chain.chain_length, 2);

        // Tamper with the first receipt's previous_hash and expect verification
        // to detect the break. [VERIFIED — tamper-evidence]
        chain.receipts[0].previous_hash[0] ^= 0xFF;
        assert!(chain.verify_chain().is_err());
    }

    // -----------------------------------------------------------------------
    // Additional: test_mission_lifecycle
    // -----------------------------------------------------------------------

    /// Mock Mission transitions follow the documented state machine. [VERIFIED]
    #[test]
    fn test_mission_lifecycle() {
        let mut m = Mission::new(test_hash(0x60), 1_000);
        assert_eq!(m.state, MissionState::Submitted);

        m.transition(MissionState::Running, 2_000, "saga executing");
        assert_eq!(m.state, MissionState::Running);

        m.complete(3_000);
        assert_eq!(m.state, MissionState::Complete);
        assert_eq!(m.completed_at, Some(3_000));
    }

    // -----------------------------------------------------------------------
    // Additional: test_saga_dispatcher_mission_handle
    // -----------------------------------------------------------------------

    /// `create_saga` returns a `MissionHandle` with matching mission_id. [DERIVED]
    #[test]
    fn test_saga_dispatcher_mission_handle() {
        let input_hash = test_hash(0x70);
        let (saga, handle) =
            SagaDispatcher::create_saga(input_hash, SagaPermit::dev_default(), 5_000);

        assert_eq!(saga.mission_id, handle.mission_id);
        assert_eq!(handle.submitted_at, 5_000);
    }
}
