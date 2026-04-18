//! BIZRA Mission v1 Freeze — §7 MissionEnvelope + §9 State Migration
//!
//! بسم الله الرحمن الرحيم
//!
//! File: crates/bizra-kernel/src/mission/freeze_v1.rs
//! Authority: Manifest v0.2 Canon, §7 (Canonical Contracts), §9 (State Migration Law)
//! Build Step: 4 of 8 (§17)
//! Truth Target: PROVEN
//! Depends on: Step 2 (Receipt v1), Step 3 (Admissibility v1)
//!
//! This file delivers:
//!
//!   1. MissionEnvelope — §7 frozen contract. Bounded user/system intent
//!      with scope and constraints. The entry point for Stage S2.
//!
//!   2. FourStateModel — §9 State Migration Law. Every mission operates
//!      on: current state, ideal state, state gap, next admissible action.
//!
//!   3. MissionLifecycle — state machine for mission progression through
//!      the nine-stage canonical runtime flow (§6).
//!
//! §9: "All major UI flows and runtime behaviors should eventually reduce
//! to controlled state migration from current to ideal through admissible
//! intermediate steps."

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{Blake3Hash, ReceiptKind, ReceiptPayload};

// ════════════════════════════════════════════════════════════
// MissionEnvelope — §7 Frozen Contract
// ════════════════════════════════════════════════════════════

/// The canonical bounded-intent contract per Manifest §7 Table 7-1.
///
/// FROZEN after Step 4 completes.
///
/// §7 specifies:
///   "mission_id, intent, bounds, priority, timestamp"
///   Plane: Graph → Kernel
///   Description: Bounded user/system intent with scope and constraints
///   Lifetime: Mission lifecycle
///
/// Every consequential action in BIZRA begins as a MissionEnvelope.
/// Raw intent (Stage S1) is captured and bounded into this structure
/// at Stage S2. From here it proceeds through S3 (claim extraction),
/// S4 (admissibility), and onward.
#[derive(Debug, Clone)]
pub struct MissionEnvelope {
    // ── §7 required fields ──
    /// Unique identifier for this mission.
    /// Computed as blake3 of (intent_hash || bounds_hash || timestamp).
    pub mission_id: Blake3Hash,

    /// Hash of the raw intent that originated this mission.
    /// The raw intent is stored separately; this is the binding reference.
    pub intent_hash: Blake3Hash,

    /// Human-readable intent description (bounded length).
    pub intent_text: String,

    /// Scope bounds: what this mission may and may not do.
    pub bounds: MissionBounds,

    /// Priority level for scheduling.
    pub priority: MissionPriority,

    /// Monotonic timestamp at creation.
    pub timestamp_ns: u64,

    // ── State Migration (§9) ──
    /// The four-state model governing this mission's progression.
    pub state: FourStateModel,

    // ── Lifecycle ──
    /// Current stage in the nine-stage runtime flow (§6).
    pub stage: MissionStage,

    /// Reference to the operator or agent that originated this mission.
    pub originator: Originator,
}

/// Scope bounds constraining what a mission may do.
#[derive(Debug, Clone)]
pub struct MissionBounds {
    /// Maximum computational cost allowed (in abstract units).
    pub max_cost: u64,

    /// Maximum wall-clock duration allowed (nanoseconds).
    pub max_duration_ns: u64,

    /// Planes this mission may interact with.
    /// Most missions touch Graph + Proof. Only Kernel-plane missions
    /// can modify law (and those require constitutional amendment).
    pub allowed_planes: Vec<Plane>,

    /// Explicit list of invariants that must be checked.
    /// Default: all five. Can be narrowed for SCORE_ONLY evaluations.
    pub required_invariants: Vec<crate::admissibility_freeze_v1::Invariant>,
}

impl Default for MissionBounds {
    fn default() -> Self {
        use crate::admissibility_freeze_v1::Invariant;
        MissionBounds {
            max_cost: 10_000,
            max_duration_ns: 30_000_000_000, // 30 seconds
            allowed_planes: vec![Plane::Graph, Plane::Proof, Plane::Face],
            required_invariants: vec![
                Invariant::ZannZero,
                Invariant::ClaimMustBind,
                Invariant::RibaZero,
                Invariant::NoShadowState,
                Invariant::IhsanFloor,
            ],
        }
    }
}

/// The four planes per §4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Plane {
    Kernel = 0x01, // Defines law
    Graph = 0x02,  // Interprets
    Proof = 0x03,  // Enforces
    Face = 0x04,   // Reveals
}

impl Plane {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Kernel),
            0x02 => Some(Self::Graph),
            0x03 => Some(Self::Proof),
            0x04 => Some(Self::Face),
            _ => None,
        }
    }
}

/// Mission priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum MissionPriority {
    /// Background task. Yields to all others.
    Low = 0x01,
    /// Standard priority. Default for user intents.
    Normal = 0x02,
    /// Elevated priority. Time-sensitive user requests.
    High = 0x03,
    /// Constitutional. Invariant enforcement, governance.
    Critical = 0x04,
}

impl MissionPriority {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Low),
            0x02 => Some(Self::Normal),
            0x03 => Some(Self::High),
            0x04 => Some(Self::Critical),
            _ => None,
        }
    }
}

/// Who originated this mission.
#[derive(Debug, Clone)]
pub enum Originator {
    /// Human operator via Dema (§8).
    Operator { session_id: Blake3Hash },
    /// PAT-7 agent acting on operator's behalf.
    PatAgent { agent_id: u8 },
    /// SAT-5 system agent acting on constitutional mandate.
    SatAgent { agent_id: u8 },
    /// Internal system event (boot, consolidation, lifecycle).
    System,
}

// ════════════════════════════════════════════════════════════
// FourStateModel — §9 State Migration Law
// ════════════════════════════════════════════════════════════

/// §9: "BIZRA organizes all missions around a universal four-state
/// control model: current state, ideal state, state gap, and next
/// admissible action."
///
/// Every mission carries this model. The state gap is the distance
/// between current and ideal. Every admissible action must demonstrably
/// reduce that gap.
#[derive(Debug, Clone)]
pub struct FourStateModel {
    /// Factual present condition, receipted and verified.
    /// Source: Proof → Face (§9 Table 9-1)
    pub current_state: StateSnapshot,

    /// Target condition defined by operator intent.
    /// Source: MissionEnvelope bounds (§9 Table 9-1)
    pub ideal_state: StateSnapshot,

    /// Quantified distance between current and ideal.
    /// Source: derived calculation (§9 Table 9-1)
    pub gap: f64,

    /// Single next action that reduces gap lawfully.
    /// Source: GateVerdict (PERMIT) (§9 Table 9-1)
    pub next_admissible: Option<String>,
}

/// A snapshot of state — abstract enough to cover all mission domains.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// Hash of the state data (for receipting).
    pub hash: Blake3Hash,
    /// Human-readable summary of this state.
    pub summary: String,
    /// Numerical metric representing this state (domain-specific).
    pub metric: f64,
}

impl FourStateModel {
    /// Create with initial and target states. Gap computed automatically.
    pub fn new(current: StateSnapshot, ideal: StateSnapshot) -> Self {
        let gap = (ideal.metric - current.metric).abs();
        FourStateModel {
            current_state: current,
            ideal_state: ideal,
            gap,
            next_admissible: None,
        }
    }

    /// Update current state after an action. Recomputes gap.
    pub fn advance(&mut self, new_current: StateSnapshot) {
        self.current_state = new_current;
        self.gap = (self.ideal_state.metric - self.current_state.metric).abs();
    }

    /// Is the mission complete (gap ≤ threshold)?
    pub fn is_complete(&self, threshold: f64) -> bool {
        self.gap <= threshold
    }
}

// ════════════════════════════════════════════════════════════
// MissionStage — §6 Nine-Stage Runtime Flow
// ════════════════════════════════════════════════════════════

/// The nine stages per §6 Table 6-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MissionStage {
    Intent = 0x01,           // S1: Raw intent captured
    Mission = 0x02,          // S2: Intent bounded into MissionEnvelope
    Claim = 0x03,            // S3: Specific claim extracted
    Admissibility = 0x04,    // S4: Gate chain evaluates
    Execution = 0x05,        // S5: Permitted claim executes
    Receipt = 0x06,          // S6: Immutable proof created
    Canonicalization = 0x07, // S7: Receipt added to chain
    Replayability = 0x08,    // S8: Replay reproduces original
    Reflex = 0x09,           // S9: Optional pattern promotion
}

impl MissionStage {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Intent),
            0x02 => Some(Self::Mission),
            0x03 => Some(Self::Claim),
            0x04 => Some(Self::Admissibility),
            0x05 => Some(Self::Execution),
            0x06 => Some(Self::Receipt),
            0x07 => Some(Self::Canonicalization),
            0x08 => Some(Self::Replayability),
            0x09 => Some(Self::Reflex),
            _ => None,
        }
    }

    /// Can this stage advance to the next?
    /// S9 (Reflex) is terminal — no stage after it.
    pub fn next(&self) -> Option<MissionStage> {
        match self {
            Self::Intent => Some(Self::Mission),
            Self::Mission => Some(Self::Claim),
            Self::Claim => Some(Self::Admissibility),
            Self::Admissibility => Some(Self::Execution),
            Self::Execution => Some(Self::Receipt),
            Self::Receipt => Some(Self::Canonicalization),
            Self::Canonicalization => Some(Self::Replayability),
            Self::Replayability => Some(Self::Reflex),
            Self::Reflex => None, // terminal
        }
    }
}

// ════════════════════════════════════════════════════════════
// MissionEnvelope construction + serialization
// ════════════════════════════════════════════════════════════

impl MissionEnvelope {
    /// Create a new MissionEnvelope from raw intent.
    /// This is the Stage S1→S2 transition.
    pub fn from_intent(
        intent_text: String,
        current_state: StateSnapshot,
        ideal_state: StateSnapshot,
        originator: Originator,
        timestamp_ns: u64,
    ) -> Self {
        let intent_hash = blake3_domain("bizra-mission-intent-v1", intent_text.as_bytes());

        let bounds = MissionBounds::default();

        let bounds_hash = blake3_domain("bizra-mission-bounds-v1", &bounds.max_cost.to_le_bytes());

        // mission_id = hash(intent || bounds || timestamp)
        let mut id_buf = Vec::new();
        id_buf.extend_from_slice(&intent_hash);
        id_buf.extend_from_slice(&bounds_hash);
        id_buf.extend_from_slice(&timestamp_ns.to_le_bytes());
        let mission_id = blake3_domain("bizra-mission-id-v1", &id_buf);

        let state = FourStateModel::new(current_state, ideal_state);

        MissionEnvelope {
            mission_id,
            intent_hash,
            intent_text,
            bounds,
            priority: MissionPriority::Normal,
            timestamp_ns,
            state,
            stage: MissionStage::Mission, // created at S2
            originator,
        }
    }

    /// Advance the mission to the next stage.
    /// Returns false if already at terminal stage.
    pub fn advance_stage(&mut self) -> bool {
        match self.stage.next() {
            Some(next) => {
                self.stage = next;
                true
            }
            None => false,
        }
    }

    /// Extract the claim_id for submission to the admissibility chain.
    /// This is the S2→S3 transition.
    pub fn extract_claim_id(&self) -> Blake3Hash {
        blake3_domain("bizra-mission-claim-v1", &self.mission_id)
    }
}

impl ReceiptPayload for MissionEnvelope {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::GovernanceDecision // TODO: add MissionCreated variant
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);
        buf.extend_from_slice(&self.mission_id);
        buf.extend_from_slice(&self.intent_hash);
        // intent_text: length-prefixed
        buf.extend_from_slice(&(self.intent_text.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.intent_text.as_bytes());
        // bounds (simplified: max_cost + max_duration)
        buf.extend_from_slice(&self.bounds.max_cost.to_le_bytes());
        buf.extend_from_slice(&self.bounds.max_duration_ns.to_le_bytes());
        // priority
        buf.push(self.priority as u8);
        // timestamp
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        // stage
        buf.push(self.stage as u8);
        // state model (current + ideal hashes + gap)
        buf.extend_from_slice(&self.state.current_state.hash);
        buf.extend_from_slice(&self.state.ideal_state.hash);
        buf.extend_from_slice(&self.state.gap.to_le_bytes());
        buf
    }

    fn hash(&self) -> Blake3Hash {
        self.mission_id // mission_id IS the canonical hash
    }
}

// ════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn current_state() -> StateSnapshot {
        StateSnapshot {
            hash: [1u8; 32],
            summary: "Current: files scattered across 6 locations".into(),
            metric: 0.2,
        }
    }

    fn ideal_state() -> StateSnapshot {
        StateSnapshot {
            hash: [2u8; 32],
            summary: "Ideal: all files organized in project folders".into(),
            metric: 1.0,
        }
    }

    fn make_envelope() -> MissionEnvelope {
        MissionEnvelope::from_intent(
            "Organize my Downloads folder into project folders".into(),
            current_state(),
            ideal_state(),
            Originator::Operator {
                session_id: [99u8; 32],
            },
            1_000_000,
        )
    }

    // ── Test 1: MissionEnvelope has all §7 fields ──

    #[test]
    fn test_envelope_has_section7_fields() {
        let env = make_envelope();

        assert_ne!(env.mission_id, [0u8; 32], "mission_id must be computed");
        assert_ne!(env.intent_hash, [0u8; 32], "intent_hash must be computed");
        assert!(!env.intent_text.is_empty());
        assert!(env.bounds.max_cost > 0);
        assert_eq!(env.priority, MissionPriority::Normal);
        assert!(env.timestamp_ns > 0);
    }

    // ── Test 2: Deterministic mission_id ──

    #[test]
    fn test_mission_id_deterministic() {
        let e1 = MissionEnvelope::from_intent(
            "Same intent".into(),
            current_state(),
            ideal_state(),
            Originator::System,
            500,
        );
        let e2 = MissionEnvelope::from_intent(
            "Same intent".into(),
            current_state(),
            ideal_state(),
            Originator::System,
            500,
        );

        assert_eq!(e1.mission_id, e2.mission_id);
    }

    // ── Test 3: Different intents → different mission_ids ──

    #[test]
    fn test_different_intents_different_ids() {
        let e1 = MissionEnvelope::from_intent(
            "Intent A".into(),
            current_state(),
            ideal_state(),
            Originator::System,
            500,
        );
        let e2 = MissionEnvelope::from_intent(
            "Intent B".into(),
            current_state(),
            ideal_state(),
            Originator::System,
            500,
        );

        assert_ne!(e1.mission_id, e2.mission_id);
    }

    // ── Test 4: Stage progression follows §6 ──

    #[test]
    fn test_stage_progression() {
        let mut env = make_envelope();
        assert_eq!(env.stage, MissionStage::Mission); // starts at S2

        let stages = [
            MissionStage::Claim,
            MissionStage::Admissibility,
            MissionStage::Execution,
            MissionStage::Receipt,
            MissionStage::Canonicalization,
            MissionStage::Replayability,
            MissionStage::Reflex,
        ];

        for expected in &stages {
            assert!(env.advance_stage(), "Should advance to {:?}", expected);
            assert_eq!(env.stage, *expected);
        }

        // S9 is terminal
        assert!(!env.advance_stage(), "Should not advance past Reflex");
        assert_eq!(env.stage, MissionStage::Reflex);
    }

    // ── Test 5: FourStateModel gap computation ──

    #[test]
    fn test_four_state_gap() {
        let model = FourStateModel::new(current_state(), ideal_state());

        // current metric = 0.2, ideal = 1.0, gap = 0.8
        assert!((model.gap - 0.8).abs() < 0.001);
        assert!(!model.is_complete(0.01));
    }

    // ── Test 6: FourStateModel advance reduces gap ──

    #[test]
    fn test_four_state_advance() {
        let mut model = FourStateModel::new(current_state(), ideal_state());

        // Advance current to 0.9
        model.advance(StateSnapshot {
            hash: [3u8; 32],
            summary: "After organizing: 90% complete".into(),
            metric: 0.9,
        });

        assert!((model.gap - 0.1).abs() < 0.001);
        assert!(!model.is_complete(0.01));

        // Advance to 1.0
        model.advance(StateSnapshot {
            hash: [4u8; 32],
            summary: "After organizing: complete".into(),
            metric: 1.0,
        });

        assert!((model.gap - 0.0).abs() < 0.001);
        assert!(model.is_complete(0.01));
    }

    // ── Test 7: Extract claim_id is deterministic ──

    #[test]
    fn test_extract_claim_id_deterministic() {
        let env = make_envelope();
        let c1 = env.extract_claim_id();
        let c2 = env.extract_claim_id();
        assert_eq!(c1, c2);
        assert_ne!(c1, [0u8; 32]);
    }

    // ── Test 8: Default bounds include all five invariants ──

    #[test]
    fn test_default_bounds_include_all_invariants() {
        let bounds = MissionBounds::default();
        assert_eq!(
            bounds.required_invariants.len(),
            5,
            "Default bounds must require all five invariants"
        );
    }
}
