//! BIZRA Cognition Runtime — with replay-from-chain rehydration
//! ==============================================================
//! File: crates/bizra-kernel/src/runtime/cognition_loop.rs
//! Domain tag: bizra-runtime-v1
//!
//! R1 (Lamport): the chain is truth, the graph is derived state.
//! This file makes R1 operational: on boot, the runtime replays the chain,
//! reconstructs the compiled-reflex set from Myelination/Demyelination
//! receipts, and produces a graph consistent with the chain's decisions.
//!
//! Rehydrate contract:
//!
//! - Input: a ThoughtGraph in its fresh-boot state (no reflexes installed)
//!   and a ReceiptChain whose records have been loaded (but whose derived
//!   state the graph has not yet applied).
//! - Output: a runtime whose graph has the exact reflex set the chain
//!   records commit to, with no side effects beyond state.
//!
//! Determinism: rehydrate is pure replay. Same chain + same graph skeleton
//! → same final state, byte-for-byte. This is what makes Node1 reproducibility
//! and crash recovery work.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::admissibility_freeze_v1::{
    AdmissibilityChain, AdmissibilityClaim, AdmissibilityResult, EconomicPattern, GateVerdict,
    RejectedClaim, StateMutation, Verdict,
};
use crate::canonical_hasher::blake3_domain;
use crate::manifest_artifact::ManifestArtifact;
use crate::manifest_history_cache::{
    ManifestHistoryCache, ManifestHistoryCacheError, ManifestHistorySnapshot, ManifestSummary,
};
use crate::mission_freeze_v1::{MissionEnvelope, MissionStage, Originator, StateSnapshot};
use crate::mission_log_cache::{
    MissionLogCache, MissionLogCacheError, MissionLogEntry, MissionLogSnapshot,
};
use crate::organize_mission::{OrganizeListing, OrganizeMissionReceipt};
use crate::poi_ledger::{
    compute_impact_score, PoiEntry, PoiLedgerCache, PoiLedgerCacheError, PoiLedgerSnapshot,
};
use crate::principal_activation::{
    PrincipalActivationEnvelope, PrincipalActivationReceipt, PrincipalProfile,
};
use crate::principal_cache::{PrincipalCacheError, PrincipalProfileCache};
use crate::receipt_freeze_v1::{ReceiptArtifact, ReceiptChainExt};
use crate::receipt_history_cache::{
    ReceiptHistoryCache, ReceiptHistoryCacheError, ReceiptHistorySnapshot,
};
use crate::receipts::{
    Blake3Hash, ChainError, InMemoryPayloadStore, ReceiptChain, ReceiptKind, ReceiptPayload,
};
use crate::resource_registry::{
    RegisterOutcome, ResourceKind, ResourceRegistryError, TypedResource, UrpView,
};
use crate::resource_registry_cache::{
    ResourceRegistryCache, ResourceRegistryCacheError, ResourceRegistrySnapshot,
};
use crate::state_snapshots_cache::{
    StateSnapshotEntry, StateSnapshotView, StateSnapshotsCache, StateSnapshotsCacheError,
    StateSnapshotsSnapshot,
};
use crate::thought_graph::{
    AgentCtx, CompiledReflex, DemyelinationReason, DemyelinationReceipt, MyelinationReceipt,
    ReasoningError, ShadowMode, Thought, ThoughtGraph,
};

// ============================================================================
// Events
// ============================================================================

#[derive(Debug, Clone)]
pub enum CognitionEvent {
    ReasoningRequest {
        request_id: Blake3Hash,
    },
    ConsolidationTick,
    GovernanceDemyelination {
        edge: Blake3Hash,
        decision: Blake3Hash,
    },
    Shutdown,
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum LoopError {
    Chain(ChainError),
    Reasoning(ReasoningError),
    Clock(String),
    Shutdown,
    Rehydrate(RehydrateError),
}

impl From<ChainError> for LoopError {
    fn from(e: ChainError) -> Self {
        LoopError::Chain(e)
    }
}

#[derive(Debug)]
pub enum RehydrateError {
    ChainFetch(ChainError),
    MissingPayload(Blake3Hash),
    InconsistentState { reason: String },
}

impl From<ChainError> for RehydrateError {
    fn from(e: ChainError) -> Self {
        RehydrateError::ChainFetch(e)
    }
}

#[derive(Debug)]
pub enum MissionRuntimeError {
    Chain(ChainError),
    Clock(String),
    DuplicateMission(Blake3Hash),
    MissionNotFound(Blake3Hash),
    ClaimMismatch {
        expected: Blake3Hash,
        got: Blake3Hash,
    },
}

impl From<ChainError> for MissionRuntimeError {
    fn from(e: ChainError) -> Self {
        MissionRuntimeError::Chain(e)
    }
}

/// The full record of a mission's passage through the lawful loop.
///
/// G2-hardening (Cycle-5, 2026-04-17 per spec g2-patches-abc.md):
/// - `rejected` explicitly distinguishes denied claims from permitted ones
/// - `receipt_id` is None for rejected missions (§10: "chain reflects what
///   actually happened by ABSENCE, not by presence of a rejection receipt")
/// - `stage` is the authoritative final stage (Admissibility on reject,
///   Canonicalization if replay decode fails, Replayability on full success)
/// - `final_receipt` is Option so reject can carry no receipt without lying
#[derive(Debug, Clone)]
pub struct MissionRuntimeRecord {
    pub envelope: MissionEnvelope,
    pub claim: AdmissibilityClaim,
    pub admissibility: AdmissibilityResult,
    /// Hash of the MissionEnvelope chain record (permit path only).
    /// `None` on reject: nothing was appended to the chain.
    pub mission_payload_hash: Option<Blake3Hash>,
    /// Hashes of gate verdict chain records (permit path only). Empty on reject.
    pub gate_receipt_hashes: Vec<Blake3Hash>,
    /// Final `NodeLifecycle` ReceiptArtifact (permit path only). `None` on reject.
    pub final_receipt: Option<ReceiptArtifact>,
    /// Convenience copy of `final_receipt.as_ref().map(|r| r.receipt_id)`.
    pub receipt_id: Option<Blake3Hash>,
    /// Immutable chain head at mission conclusion. For permitted missions this is
    /// the sealed receipt head; for rejected missions it is the unchanged head
    /// that existed when admissibility completed.
    pub chain_head: Blake3Hash,
    /// `true` iff admissibility returned a non-Permit verdict. Mirrors
    /// `admissibility.verdict != Verdict::Permit`.
    pub rejected: bool,
    /// Authoritative final stage of the envelope.
    pub stage: MissionStage,
    /// Nanosecond timestamp of submission (monotonic, not wall-clock).
    pub timestamp_ns: u64,
    /// Cycle-7 G1 — ManifestArtifact binding this mission's full chain
    /// footprint (mission_payload + gate verdicts + final receipt) into
    /// one queryable object. `None` on reject (§10 Proof Law: rejected
    /// missions never produce a manifest). `Some` only when stage reaches
    /// Replayability (S8) under Patch B discipline from Cycle-5.
    pub manifest: Option<ManifestArtifact>,
}

/// Cycle-7 G2 — the full outcome of a principal-activation submission.
///
/// Wraps the underlying `MissionRuntimeRecord` (which carries the
/// admissibility result, NodeLifecycle receipt, and Manifest) with the
/// activation-specific artifacts:
///
/// - `profile` is Some only on permit — the PrincipalProfile bound to
///   the NodeLifecycle receipt. None on reject per §10 Proof Law.
/// - `activation_receipt` is the chain-sealed PrincipalActivationReceipt
///   (kind 0x61) appended immediately after Manifest. None on reject.
/// - `rejected` mirrors `mission_record.rejected` for caller ergonomics.
/// - `remediation` is Some on reject — honest, structured text describing
///   which gate failed, why, and the remediation path.
#[derive(Debug, Clone)]
pub struct PrincipalActivationRecord {
    pub envelope: PrincipalActivationEnvelope,
    pub mission_record: MissionRuntimeRecord,
    pub profile: Option<PrincipalProfile>,
    pub activation_receipt: Option<PrincipalActivationReceipt>,
    pub rejected: bool,
    pub remediation: Option<String>,
    /// Cycle-7 G2 Commit-3 — on-disk cache write outcome. `None` if
    /// no dema_cache was attached, or if write succeeded. `Some(msg)`
    /// if write failed — the chain is still sealed and profile is
    /// still in-memory; only the derived cache is stale. Caller may
    /// retry cache write or rebuild from chain.
    pub cache_warning: Option<String>,
    /// Server-reported `dema_cache/` directory where the profile was
    /// persisted. `Some(dir)` iff a dema_cache was attached at activation
    /// time (regardless of write outcome — write failures surface via
    /// `cache_warning`). `None` when no cache was attached, which is the
    /// canonically correct signal that no on-disk persistence happened.
    /// Consumers: CLI and web face echo this verbatim rather than
    /// deriving it from their own env — closes ZANN_ZERO drift under
    /// remote gateway / divergent env scenarios.
    pub effective_cache_dir: Option<std::path::PathBuf>,
}

/// Cycle-7 G5 — outcome of a `submit_organize_mission` call.
///
/// The four variants partition the semantic space:
///   - `NotAllowlisted`: constitutional pre-gate refusal. No chain
///     mutation. Operator must `dema register-resource --allowlisted`
///     before retrying.
///   - `IoError`: filesystem read failed before the lawful loop
///     could be entered. No chain mutation.
///   - `Rejected`: mission entered the lawful loop and was rejected
///     at admissibility. Chain UNCHANGED per §10 Proof Law.
///   - `Executed`: permit path — lawful loop produced NodeLifecycle
///     mission receipt + Manifest; G5 appends MissionExecuted receipt.
#[derive(Debug)]
pub enum OrganizeOutcome {
    NotAllowlisted {
        path: String,
        remediation: String,
    },
    IoError {
        path: String,
        error: String,
    },
    Rejected {
        mission_record: MissionRuntimeRecord,
        remediation: String,
    },
    Executed {
        mission_record: MissionRuntimeRecord,
        organize_receipt: OrganizeMissionReceipt,
        listing: OrganizeListing,
    },
}

impl OrganizeOutcome {
    pub fn is_executed(&self) -> bool {
        matches!(self, Self::Executed { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionReplayResult {
    Match,
    Divergent,
}

#[derive(Debug, Clone, Copy)]
pub struct MissionReplayReport {
    pub mission_id: Blake3Hash,
    pub replay_result: MissionReplayResult,
    pub matches_previous: bool,
    pub chain_head: Blake3Hash,
}

// ============================================================================
// Degraded-path receipt
// ============================================================================

#[derive(Debug, Clone)]
pub struct DegradedPathReceipt {
    pub occasion: DegradedOccasion,
    pub prev_chain: Blake3Hash,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub enum DegradedOccasion {
    MyelinationPersistFailed { edge: Blake3Hash, cause: String },
    DemyelinationPersistFailed { edge: Blake3Hash, cause: String },
    ReasoningFailed { cause: String },
    ConsolidationDivergence { edge: Blake3Hash, divergence: f64 },
    ClockFailure { cause: String },
}

impl DegradedPathReceipt {
    fn occasion_discriminant(&self) -> u8 {
        match self.occasion {
            DegradedOccasion::MyelinationPersistFailed { .. } => 0x01,
            DegradedOccasion::DemyelinationPersistFailed { .. } => 0x02,
            DegradedOccasion::ReasoningFailed { .. } => 0x03,
            DegradedOccasion::ConsolidationDivergence { .. } => 0x04,
            DegradedOccasion::ClockFailure { .. } => 0x05,
        }
    }
}

impl ReceiptPayload for DegradedPathReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::DegradedPath
    }
    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.push(self.occasion_discriminant());
        match &self.occasion {
            DegradedOccasion::MyelinationPersistFailed { edge, cause }
            | DegradedOccasion::DemyelinationPersistFailed { edge, cause } => {
                buf.extend_from_slice(edge);
                let cb = cause.as_bytes();
                buf.extend_from_slice(&(cb.len() as u32).to_le_bytes());
                buf.extend_from_slice(cb);
            }
            DegradedOccasion::ReasoningFailed { cause }
            | DegradedOccasion::ClockFailure { cause } => {
                let cb = cause.as_bytes();
                buf.extend_from_slice(&(cb.len() as u32).to_le_bytes());
                buf.extend_from_slice(cb);
            }
            DegradedOccasion::ConsolidationDivergence { edge, divergence } => {
                buf.extend_from_slice(edge);
                buf.extend_from_slice(&divergence.to_le_bytes());
            }
        }
        buf.extend_from_slice(&self.prev_chain);
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }
    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-degraded-path-v1", &self.canonical_bytes())
    }
}

#[derive(Debug, Clone)]
pub struct ReasoningSessionReceipt {
    pub request_id: Blake3Hash,
    pub thoughts_digest: Blake3Hash,
    pub thoughts_count: u32,
    pub prev_chain: Blake3Hash,
    pub timestamp_ns: u64,
}

impl ReceiptPayload for ReasoningSessionReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::ReasoningSession
    }
    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(&self.request_id);
        buf.extend_from_slice(&self.thoughts_digest);
        buf.extend_from_slice(&self.thoughts_count.to_le_bytes());
        buf.extend_from_slice(&self.prev_chain);
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }
    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-reasoning-session-v1", &self.canonical_bytes())
    }
}

// ============================================================================
// Runtime
// ============================================================================

pub struct CognitionRuntime {
    pub graph: ThoughtGraph,
    pub chain: ReceiptChain,
    pub ctx: AgentCtx,
    missions: HashMap<Blake3Hash, MissionRuntimeRecord>,
    session_counter: u64,
    // Cycle-6 G1 Phase 1 — optional read-only durable projection loaded
    // via `from_sovereign_state`. None means in-memory bootstrap (dev).
    // Some(snap) means the gateway served its last pre-restart truth
    // and the chain is anchored to Python-authoritative persistence.
    sovereign_snapshot: Option<crate::sovereign_state::SovereignStateSnapshot>,
    // Cycle-7 G2 — in-memory principal profile populated by a permitted
    // submit_principal_activation call. Derived and rebuildable from
    // chain per niyyah §"Writer authority decision (HYBRID)".
    principal_profile: Option<PrincipalProfile>,
    // Cycle-7 G2 Commit-3 — optional on-disk cache at
    // sovereign_state/dema_cache/principal.json. When set, permitted
    // activations auto-write; restart rehydration via
    // `rehydrate_principal_from_cache`. Opt-in to preserve in-memory
    // unit-test isolation.
    dema_cache: Option<PrincipalProfileCache>,
    // Cycle-7 G3 Commit-1 — optional on-disk cache at
    // sovereign_state/dema_cache/receipt_history.json. When set, every
    // public API that advances the chain auto-writes a snapshot after
    // the append. Best-effort: write failures do not invalidate the
    // sealed chain. Rehydrate via `rehydrate_receipt_history_from_cache`.
    receipt_history_cache: Option<ReceiptHistoryCache>,
    // Cycle-7 G3 Commit-2 — optional on-disk cache at
    // sovereign_state/dema_cache/manifest_history.json. Derived from
    // the `missions` registry (permit-path-only manifests). Refreshed
    // on every submit_mission permit return. Best-effort; chain stays
    // truth.
    manifest_history_cache: Option<ManifestHistoryCache>,
    // Cycle-7 G3 Commit-3 — optional on-disk cache at
    // sovereign_state/dema_cache/mission_log.json. Derived from the
    // full `missions` registry (permit + reject). Each entry carries
    // intent_text, timestamp, stage, optional receipt_id, optional
    // remediation. Refreshed at every submit_mission boundary.
    mission_log_cache: Option<MissionLogCache>,
    // Cycle-7 G3 Commit-4 — optional on-disk cache at
    // sovereign_state/dema_cache/state_snapshots.json. Derived from
    // each mission's FourStateModel (current + ideal + gap). Answers:
    // "where NODE0 is vs where it aims to be, per mission attempt."
    state_snapshots_cache: Option<StateSnapshotsCache>,
    // Cycle-7 G3 Commit-5 — optional on-disk cache at
    // sovereign_state/dema_cache/resource_registry.json. G3 seeds an
    // empty registry so G4 can assume the file exists; G4 owns the
    // mutation API (register-resource / allowlist / URP view).
    resource_registry_cache: Option<ResourceRegistryCache>,
    // Cycle-7 G6 — local-only Proof-of-Impact ledger. In-memory list
    // of PoiEntry records, one per permitted MissionExecuted /
    // PrincipalActivation receipt. Auto-appended on permit.
    // Rebuildable from chain; cache at
    // sovereign_state/dema_cache/poi_ledger.json is a read fast-path,
    // never outranks chain.
    poi_entries: Vec<PoiEntry>,
    poi_ledger_cache: Option<PoiLedgerCache>,
}

/// Errors produced by CognitionRuntime bootstrap constructors.
#[derive(Debug)]
pub enum BootstrapError {
    SovereignState(crate::sovereign_state::SovereignStateError),
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SovereignState(e) => write!(f, "sovereign_state bootstrap: {}", e),
        }
    }
}

impl std::error::Error for BootstrapError {}

impl From<crate::sovereign_state::SovereignStateError> for BootstrapError {
    fn from(e: crate::sovereign_state::SovereignStateError) -> Self {
        Self::SovereignState(e)
    }
}

impl CognitionRuntime {
    pub fn new(graph: ThoughtGraph, chain: ReceiptChain, ctx: AgentCtx) -> Self {
        Self {
            graph,
            chain,
            ctx,
            missions: HashMap::new(),
            session_counter: 0,
            sovereign_snapshot: None,
            principal_profile: None,
            dema_cache: None,
            receipt_history_cache: None,
            manifest_history_cache: None,
            mission_log_cache: None,
            state_snapshots_cache: None,
            resource_registry_cache: None,
            poi_entries: Vec::new(),
            poi_ledger_cache: None,
        }
    }

    /// Cycle-6 G1 Phase 1 — bootstrap from the Python-authoritative
    /// `sovereign_state/` directory on disk. Returns a runtime with
    /// an empty ThoughtGraph + in-memory ReceiptChain (for new
    /// activity this session) PLUS an attached `SovereignStateSnapshot`
    /// that carries the verified read-only projection of durable history.
    ///
    /// Gateway handlers query the snapshot via `sovereign_snapshot()` to
    /// serve durable-chain answers that survive restart.
    ///
    /// Read-only: this constructor never writes to `path`.
    /// Fails closed: any snapshot integrity error is returned; caller
    /// decides whether to fall back (dev) or abort (production).
    pub fn from_sovereign_state(path: &std::path::Path) -> Result<Self, BootstrapError> {
        let snapshot = crate::sovereign_state::SovereignStateSnapshot::load(path)?;

        let genesis: Blake3Hash = [0u8; 32];
        let graph = ThoughtGraph::from_parts(HashMap::new(), Vec::new(), HashMap::new(), genesis);
        let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx = AgentCtx {
            receipt_chain: genesis,
        };

        Ok(Self {
            graph,
            chain,
            ctx,
            missions: HashMap::new(),
            session_counter: 0,
            sovereign_snapshot: Some(snapshot),
            principal_profile: None,
            dema_cache: None,
            receipt_history_cache: None,
            manifest_history_cache: None,
            mission_log_cache: None,
            state_snapshots_cache: None,
            resource_registry_cache: None,
            poi_entries: Vec::new(),
            poi_ledger_cache: None,
        })
    }

    /// Access the attached sovereign-state snapshot (if this runtime
    /// was bootstrapped via `from_sovereign_state`).
    pub fn sovereign_snapshot(&self) -> Option<&crate::sovereign_state::SovereignStateSnapshot> {
        self.sovereign_snapshot.as_ref()
    }

    /// Rehydrate a runtime by replaying the chain.
    ///
    /// Contract:
    ///   - The `graph` must be a fresh ThoughtGraph from configure (no reflexes,
    ///     no hit counts). rehydrate() will populate compiled_reflexes to match
    ///     the chain's committed state.
    ///   - The `chain` must already have its records loaded (ReceiptChain itself
    ///     is stateful; if you're loading from disk, reconstruct it first).
    ///   - `ctx` is set to the chain's current head on return.
    ///
    /// Replay semantics:
    ///   - For each Myelination record: fetch payload, decode, install reflex.
    ///   - For each Demyelination record: fetch payload, decode, remove reflex.
    ///   - All other record kinds (ReasoningSession, DegradedPath, Governance,
    ///     Boot, Lifecycle): ignored for reflex state, but chain continuity is
    ///     verified.
    ///
    /// What this does NOT rebuild:
    ///   - Hit counts (they are hints; start at 0 post-rehydrate)
    ///   - Quarantine state (same — hints, not truth)
    ///   - Shadow mode (caller configures post-rehydrate)
    ///
    /// This is intentional: per Lampson, hints can be wrong without affecting
    /// correctness. Reflex installation IS truth-state and must be rebuilt.
    /// Hit counts and quarantines are observation-state and will repopulate
    /// naturally as the runtime processes new events.
    pub fn rehydrate(graph: ThoughtGraph, chain: ReceiptChain) -> Result<Self, RehydrateError> {
        let mut rt = Self {
            graph,
            chain,
            ctx: AgentCtx {
                receipt_chain: [0u8; 32],
            },
            missions: HashMap::new(),
            session_counter: 0,
            sovereign_snapshot: None,
            principal_profile: None,
            dema_cache: None,
            receipt_history_cache: None,
            manifest_history_cache: None,
            mission_log_cache: None,
            state_snapshots_cache: None,
            resource_registry_cache: None,
            poi_entries: Vec::new(),
            poi_ledger_cache: None,
        };

        // Collect replay actions in order. We iterate records oldest-to-newest.
        // For each Myelination, we must decode the payload to recover the
        // CompiledReflex's policy_version and source hash.
        let records: Vec<(ReceiptKind, Blake3Hash)> =
            rt.chain.records().map(|r| (r.kind, r.hash)).collect();

        for (kind, hash) in records {
            match kind {
                ReceiptKind::Myelination => {
                    let receipt: MyelinationReceipt = rt
                        .chain
                        .fetch_and_decode(&hash)
                        .map_err(RehydrateError::ChainFetch)?;

                    // Reconstruct the CompiledReflex. In production this calls
                    // skill_reflex_bridge to rebuild the specialized runtime; here
                    // we use the stub constructor. Source hash and policy version
                    // come from the receipt, which is the authoritative record.
                    let reflex = CompiledReflex {
                        source_s2: receipt.source_s2,
                        policy_version: receipt.policy_version,
                    };
                    rt.graph
                        .install_reflex_from_replay(receipt.source_s2, reflex);
                }
                ReceiptKind::Demyelination => {
                    let receipt: DemyelinationReceipt = rt
                        .chain
                        .fetch_and_decode(&hash)
                        .map_err(RehydrateError::ChainFetch)?;
                    rt.graph.remove_reflex_from_replay(&receipt.reflex);
                }
                // All other kinds do not affect reflex truth-state.
                // ReceiptKind::Manifest (Cycle-7 G1) is summary-only: a
                // manifest binds mission receipts into one queryable object
                // but does not itself install/remove reflexes.
                ReceiptKind::Genesis
                | ReceiptKind::CognitionBoot
                | ReceiptKind::ReasoningSession
                | ReceiptKind::GovernanceDecision
                | ReceiptKind::NodeLifecycle
                | ReceiptKind::Manifest
                | ReceiptKind::PrincipalActivation
                | ReceiptKind::MissionExecuted
                | ReceiptKind::DegradedPath => {}
            }
        }

        // Align ctx and graph chain_head to the chain's current head.
        let head = rt.chain.head();
        rt.ctx.receipt_chain = head;
        rt.graph.set_chain_head(head);

        Ok(rt)
    }

    /// Submit a mission through the lawful loop.
    ///
    /// Returns a `MissionRuntimeRecord` always (on both Permit and Reject). The
    /// caller branches on `record.rejected`:
    /// - `rejected == false`: receipt_id populated, stage=Replayability (or
    ///   Canonicalization if decode round-trip failed), chain advanced by
    ///   mission envelope + 5 gate verdicts + final NodeLifecycle receipt.
    /// - `rejected == true`: receipt_id=None, stage=Admissibility, chain
    ///   UNCHANGED. The rejection is recorded in the `missions` registry
    ///   (derived state per §10) so it remains queryable via `mission_by_id`,
    ///   but it does not enter the chain of source truth.
    ///
    /// Errors are reserved for STRUCTURAL failures (claim mismatch, duplicate,
    /// chain append error, clock failure). Admissibility rejection is NOT an
    /// error — it is a structured outcome.
    ///
    /// G2-hardening (2026-04-17 per g2-patches-abc.md):
    /// - A: eval-first ordering; rejected claims do not enter the chain
    /// - B: S8 Replayability only confirmed via decode round-trip
    /// - C: (applied separately in manifest_artifact.rs)
    pub fn submit_mission(
        &mut self,
        mut envelope: MissionEnvelope,
        claim: AdmissibilityClaim,
    ) -> Result<MissionRuntimeRecord, MissionRuntimeError> {
        let expected_claim = envelope.extract_claim_id();
        if expected_claim != claim.claim_id {
            return Err(MissionRuntimeError::ClaimMismatch {
                expected: expected_claim,
                got: claim.claim_id,
            });
        }

        if self.missions.contains_key(&envelope.mission_id) {
            return Err(MissionRuntimeError::DuplicateMission(envelope.mission_id));
        }

        // PATCH A (2026-04-17, Cycle-5 G2-hardening): evaluate admissibility
        // BEFORE any chain mutation. Rejected claims do not enter the chain at
        // all — their rejection is recorded in derived state (missions registry)
        // per §10 Proof Law. "Chain reflects what actually happened by ABSENCE,
        // not by presence of a rejection receipt."
        envelope.advance_stage(); // S2 -> S3 claim extraction
        let admissibility = AdmissibilityChain::canonical().evaluate(&claim);
        envelope.advance_stage(); // S3 -> S4 admissibility

        let timestamp_ns = self.now_ns_mission()?;

        if admissibility.verdict != Verdict::Permit {
            let mission_id = envelope.mission_id;
            let record = MissionRuntimeRecord {
                envelope,
                claim,
                admissibility,
                mission_payload_hash: None,
                gate_receipt_hashes: Vec::new(),
                final_receipt: None,
                receipt_id: None,
                chain_head: self.chain.head(),
                rejected: true,
                stage: MissionStage::Admissibility,
                timestamp_ns,
                manifest: None, // §10 Proof Law: rejected missions produce no manifest
            };
            self.missions.insert(mission_id, record.clone());
            // Cycle-7 G3 Commit-1 — best-effort cache snapshot on the
            // reject path. Chain did not advance, but the cache write
            // here keeps file mtime moving and exposes the (unchanged)
            // head to readers that poll the cache after an operator
            // attempt. Failure is silent — see trailing write on
            // permit-return for the same rationale.
            let _ = self.write_receipt_history_cache();
            let _ = self.write_manifest_history_cache();
            let _ = self.write_mission_log_cache();
            let _ = self.write_state_snapshots_cache();
            return Ok(record);
        }

        // PERMIT path: now safe to mutate the chain. Append the mission
        // envelope first (CLAIM_MUST_BIND evidence), then each gate verdict
        // as a separate receipt, then the final NodeLifecycle ReceiptArtifact.
        let mission_payload_hash = self.chain.append_with_payload(envelope.clone())?;

        let mut gate_receipt_hashes = Vec::with_capacity(admissibility.gate_verdicts.len());
        for verdict in &admissibility.gate_verdicts {
            gate_receipt_hashes.push(self.chain.append_with_payload(verdict.clone())?);
        }

        envelope.advance_stage(); // S4 -> S5 Execution (submission is the act)
        envelope.advance_stage(); // S5 -> S6 Receipt (next line mints it)

        let final_receipt = ReceiptArtifact::new(
            ReceiptKind::NodeLifecycle,
            envelope.mission_id,
            claim.evidence_hash.unwrap_or(envelope.intent_hash),
            {
                let mut lineage = Vec::with_capacity(1 + gate_receipt_hashes.len());
                lineage.push(mission_payload_hash);
                lineage.extend(gate_receipt_hashes.iter().copied());
                lineage
            },
            self.chain.head(),
            self.now_ns_mission()?,
        );
        let receipt_id = final_receipt.receipt_id;
        self.chain.append_artifact(final_receipt.clone())?;

        envelope.advance_stage(); // S6 -> S7 Canonicalization (chain append confirmed)

        // PATCH B (2026-04-17, Cycle-5 G2-hardening): advance to S8 Replayability
        // ONLY if decode round-trip verifies. If the appended payload cannot be
        // decoded back to an equivalent ReceiptArtifact, the mission stays at S7.
        // This prevents over-claiming replayability on a corrupted encode/decode.
        let replay_ok = match self.chain.fetch_payload_bytes(&receipt_id) {
            Ok(Some(bytes)) => match <ReceiptArtifact as crate::receipts::ReceiptPayloadDecode>::from_canonical_bytes(&bytes) {
                Ok(decoded) => decoded.receipt_id == receipt_id,
                Err(_) => false,
            }
            _ => false,
        };
        let mut manifest: Option<ManifestArtifact> = None;
        if replay_ok {
            envelope.advance_stage(); // S7 -> S8 Replayability

            // Cycle-7 G1 — emit ManifestArtifact binding this mission's full
            // chain footprint (mission_payload + gate verdicts + final
            // receipt) into one queryable, chain-sealed artifact.
            //
            // Permit-only: rejected missions never reach this branch (§10
            // Proof Law). Only reaches this point if S8 Replayability was
            // confirmed via decode round-trip (Patch B discipline).
            //
            // Mission-binding (caution #2 from Cycle-7 niyyah review):
            // receipt_refs includes this mission's receipt_id, so
            // manifest_for_mission lookup is unambiguous: "find the
            // manifest whose refs contain this mission's receipt_id."
            let mut manifest_refs: Vec<Blake3Hash> =
                Vec::with_capacity(2 + gate_receipt_hashes.len());
            manifest_refs.push(mission_payload_hash);
            manifest_refs.extend(gate_receipt_hashes.iter().copied());
            manifest_refs.push(receipt_id);

            let m = ManifestArtifact::from_window(
                timestamp_ns,           // window_start: mission submission time
                self.now_ns_mission()?, // window_end: post-replay-verify time
                manifest_refs,
                self.chain.head(), // chain_head: post-ReceiptArtifact append
            );
            self.chain.append_with_payload(m.clone())?;
            manifest = Some(m);
        }

        let final_stage = envelope.stage;
        let mission_id = envelope.mission_id;
        let record = MissionRuntimeRecord {
            envelope,
            claim,
            admissibility,
            mission_payload_hash: Some(mission_payload_hash),
            gate_receipt_hashes,
            final_receipt: Some(final_receipt),
            receipt_id: Some(receipt_id),
            chain_head: self.chain.head(),
            rejected: false,
            stage: final_stage,
            timestamp_ns,
            manifest,
        };
        self.missions.insert(mission_id, record.clone());
        // Best-effort receipt-history cache write (Cycle-7 G3 Commit-1).
        // Failure is silent here — the sealed chain remains truth; a
        // stale cache will be detected by the chain-vs-cache diff on
        // next rehydrate. Operator-visible warnings propagate via the
        // principal-activation path which carries `cache_warning`.
        let _ = self.write_receipt_history_cache();
        let _ = self.write_manifest_history_cache();
        let _ = self.write_mission_log_cache();
        let _ = self.write_state_snapshots_cache();
        Ok(record)
    }

    /// Cycle-7 G1 — accessor for the mission's bound `ManifestArtifact`.
    ///
    /// Returns `Some(&ManifestArtifact)` when the mission reached
    /// Replayability (S8) and a manifest was sealed into the chain;
    /// `None` for rejected missions or if the mission_id is unknown.
    pub fn manifest_for_mission(&self, mission_id: &Blake3Hash) -> Option<&ManifestArtifact> {
        self.missions
            .get(mission_id)
            .and_then(|r| r.manifest.as_ref())
    }

    pub fn mission_by_id(&self, mission_id: &Blake3Hash) -> Option<&MissionRuntimeRecord> {
        self.missions.get(mission_id)
    }

    /// Cycle-7 G2 — submit a principal activation through the lawful
    /// mission-runtime loop.
    ///
    /// Wraps `submit_mission` with an activation-specific envelope shape
    /// anchored to a node identity pubkey (caller builds the envelope via
    /// `PrincipalActivationEnvelope::from_anchor`). On Permit, appends an
    /// additional `PrincipalActivationReceipt` (kind 0x61) to the chain
    /// that binds the NodeLifecycle mission receipt to a `PrincipalProfile`,
    /// and stores the profile in-memory for Dema queries.
    ///
    /// §10 Proof Law preserved: rejected activations produce no
    /// PrincipalActivationReceipt, no profile, and `remediation` carries
    /// structured honest text.
    ///
    /// Chain footprint (permit path) is exactly +1 vs `submit_mission`:
    ///   envelope + 5 gates + NodeLifecycle + Manifest + PrincipalActivation = 9.
    pub fn submit_principal_activation(
        &mut self,
        activation_envelope: PrincipalActivationEnvelope,
        quality_score: f64,
    ) -> Result<PrincipalActivationRecord, MissionRuntimeError> {
        let session_id = activation_envelope.node_pubkey;
        let current = StateSnapshot {
            hash: blake3_domain(
                "bizra-principal-state-pre-v1",
                &activation_envelope.node_pubkey,
            ),
            summary: "Principal unactivated — no chain receipt yet".into(),
            metric: 0.0,
        };
        let ideal = StateSnapshot {
            hash: blake3_domain(
                "bizra-principal-state-ideal-v1",
                &activation_envelope.node_pubkey,
            ),
            summary: "Principal activated — receipted through lawful loop".into(),
            metric: 1.0,
        };
        let mission_env = MissionEnvelope::from_intent(
            activation_envelope.intent_text(),
            current,
            ideal,
            Originator::Operator { session_id },
            activation_envelope.created_ns,
        );

        let claim = AdmissibilityClaim {
            claim_id: mission_env.extract_claim_id(),
            has_evidence: true,
            evidence_hash: Some(activation_envelope.intent_hash),
            economic_pattern: Some(EconomicPattern::None),
            state_mutation: Some(StateMutation {
                derives_from_canonical: true,
                face_only: false,
            }),
            quality_score,
            timestamp_ns: activation_envelope.created_ns,
        };

        let mission_record = self.submit_mission(mission_env, claim)?;

        if mission_record.rejected {
            let remediation = reject_remediation_text(&mission_record);
            return Ok(PrincipalActivationRecord {
                envelope: activation_envelope,
                mission_record,
                profile: None,
                activation_receipt: None,
                rejected: true,
                remediation: Some(remediation),
                cache_warning: None,
                effective_cache_dir: None,
            });
        }

        let activation_receipt_id = mission_record
            .receipt_id
            .expect("permit path invariant: receipt_id must be Some");
        let profile = PrincipalProfile::new(
            &activation_envelope,
            activation_receipt_id,
            mission_record.timestamp_ns,
        );
        let profile_hash = profile.profile_hash();
        let now_ns = self.now_ns_mission()?;
        let prev_chain = self.chain.head();
        let pa_receipt = PrincipalActivationReceipt::new(
            activation_receipt_id,
            profile_hash,
            activation_envelope.node_pubkey,
            profile.principal_id,
            now_ns,
            prev_chain,
        );
        self.chain.append_with_payload(pa_receipt.clone())?;
        self.principal_profile = Some(profile.clone());

        // Cycle-7 G6 — record PoI entry for this activation BEFORE
        // refreshing caches, so the ledger surface reflects the new
        // entry in the same cache-refresh pass.
        self.record_poi_for_mission(
            pa_receipt.receipt_id,
            ReceiptKind::PrincipalActivation as u8,
            &mission_record,
            0, // activation has no file-listing volume
        );

        // Cycle-7 G3 Commit-1 — refresh the receipt-history cache now
        // that the PrincipalActivation receipt has advanced the chain
        // beyond what submit_mission wrote. Best-effort; failure is
        // silent because the principal_profile cache write below
        // already carries the operator-visible cache_warning channel.
        let _ = self.write_receipt_history_cache();
        let _ = self.write_manifest_history_cache();
        let _ = self.write_mission_log_cache();
        let _ = self.write_state_snapshots_cache();

        // Best-effort disk cache write. Failure does not invalidate the
        // sealed chain — the profile remains in-memory and is rebuildable
        // from the chain on the next call to rehydrate_principal_from_cache
        // (once the underlying disk issue is resolved).
        let cache_warning = if let Some(cache) = self.dema_cache.as_ref() {
            match cache.write(&profile) {
                Ok(()) => None,
                Err(e) => Some(format!(
                    "dema_cache write failed: {} — chain sealed, profile in-memory, \
                     retry or rebuild from chain",
                    e
                )),
            }
        } else {
            None
        };

        // Server-authoritative record of the dema_cache dir that was
        // attached at activation time. Populated iff a cache is attached,
        // independent of write outcome — callers combine this with
        // cache_warning to interpret persistence state.
        let effective_cache_dir = self
            .dema_cache
            .as_ref()
            .map(|c| c.cache_dir().to_path_buf());

        Ok(PrincipalActivationRecord {
            envelope: activation_envelope,
            mission_record,
            profile: Some(profile),
            activation_receipt: Some(pa_receipt),
            rejected: false,
            remediation: None,
            cache_warning,
            effective_cache_dir,
        })
    }

    /// Cycle-7 G5 — first real operator mission. Read-only organize of
    /// an allowlisted filesystem path. Flow:
    ///
    /// 1. Allowlist pre-gate — if `(FilesystemPath, path)` is not in
    ///    the registry or `allowlisted=false`, refuse immediately with
    ///    structured remediation. **No chain mutation.**
    /// 2. Filesystem read — produce a deterministic `OrganizeListing`
    ///    (top-level entries sorted by name, kind bytes). Any IO error
    ///    returns `IoError`. **No chain mutation.**
    /// 3. Lawful loop — build MissionEnvelope + AdmissibilityClaim
    ///    with listing.digest() as evidence_hash; submit via
    ///    `submit_mission`. Normal permit/reject semantics apply.
    /// 4. Permit path — append `OrganizeMissionReceipt` (kind 0x70)
    ///    binding the NodeLifecycle mission receipt to the listing.
    ///    Chain head advances to the MissionExecuted receipt.
    ///
    /// Read-only: `OrganizeListing::from_path` never mutates. Niyyah
    /// §10 Proof Law: refused intents leave no chain trace (absence);
    /// permitted intents leave a sealed, replayable trace (presence).
    pub fn submit_organize_mission(
        &mut self,
        path: &std::path::Path,
        quality_score: f64,
    ) -> Result<OrganizeOutcome, MissionRuntimeError> {
        let path_str = path.to_string_lossy().into_owned();

        // Step 1 — allowlist pre-gate. Constitutional refusal, not a
        // rejection receipt. "Chain reflects what happened by absence,
        // not by presence of a rejection receipt" (niyyah §10).
        let allowed = self
            .is_allowlisted(&ResourceKind::FilesystemPath, &path_str)
            .unwrap_or(false);
        if !allowed {
            return Ok(OrganizeOutcome::NotAllowlisted {
                path: path_str.clone(),
                remediation: format!(
                    "path '{}' is not allowlisted — run `dema register-resource \
                     --kind filesystem --id {} --allowlisted` before retrying",
                    path_str, path_str
                ),
            });
        }

        // Step 2 — filesystem read. Pure I/O; no lawful-loop entry yet.
        let listing = match OrganizeListing::from_path(path) {
            Ok(l) => l,
            Err(e) => {
                return Ok(OrganizeOutcome::IoError {
                    path: path_str,
                    error: e.to_string(),
                });
            }
        };

        // Step 3 — lawful loop. MissionEnvelope + AdmissibilityClaim.
        let now_ns = self.now_ns_mission()?;
        let listing_digest = listing.digest();
        let session_id = self
            .principal_profile
            .as_ref()
            .map(|p| p.principal_id)
            .unwrap_or([0u8; 32]);

        let current = StateSnapshot {
            hash: blake3_domain("bizra-organize-state-pre-v1", path_str.as_bytes()),
            summary: format!(
                "Path '{}' unindexed — no listing digest sealed yet",
                path_str
            ),
            metric: 0.0,
        };
        let ideal = StateSnapshot {
            hash: blake3_domain("bizra-organize-state-ideal-v1", path_str.as_bytes()),
            summary: format!(
                "Path '{}' indexed — listing digest sealed to chain",
                path_str
            ),
            metric: 1.0,
        };
        let mission_env = MissionEnvelope::from_intent(
            format!("organize {}", path_str),
            current,
            ideal,
            Originator::Operator { session_id },
            now_ns,
        );
        let claim = AdmissibilityClaim {
            claim_id: mission_env.extract_claim_id(),
            has_evidence: true,
            evidence_hash: Some(listing_digest),
            economic_pattern: Some(EconomicPattern::None),
            state_mutation: Some(StateMutation {
                derives_from_canonical: true,
                face_only: false,
            }),
            quality_score,
            timestamp_ns: now_ns,
        };

        let mission_record = self.submit_mission(mission_env, claim)?;

        if mission_record.rejected {
            let remediation = reject_remediation_text(&mission_record);
            return Ok(OrganizeOutcome::Rejected {
                mission_record,
                remediation,
            });
        }

        // Step 4 — permit path. Append MissionExecuted receipt.
        let mission_receipt_id = mission_record
            .receipt_id
            .expect("permit path invariant: receipt_id must be Some");
        let now_ns_seal = self.now_ns_mission()?;
        let prev_chain = self.chain.head();
        let organize_receipt =
            OrganizeMissionReceipt::new(mission_receipt_id, &listing, now_ns_seal, prev_chain);
        self.chain.append_with_payload(organize_receipt.clone())?;

        // Cycle-7 G6 — record PoI entry for this organize execution.
        // Entry count = number of top-level listing entries (work volume).
        self.record_poi_for_mission(
            organize_receipt.receipt_id,
            ReceiptKind::MissionExecuted as u8,
            &mission_record,
            organize_receipt.entry_count,
        );

        // Refresh G3 caches now that the chain has advanced once more.
        let _ = self.write_receipt_history_cache();
        let _ = self.write_manifest_history_cache();
        let _ = self.write_mission_log_cache();
        let _ = self.write_state_snapshots_cache();

        Ok(OrganizeOutcome::Executed {
            mission_record,
            organize_receipt,
            listing,
        })
    }

    /// Read-access to the currently activated principal profile, if any.
    /// Niyyah §"Writer authority decision (HYBRID)": derived, rebuildable.
    pub fn principal_profile(&self) -> Option<&PrincipalProfile> {
        self.principal_profile.as_ref()
    }

    /// Cycle-7 G2 Commit-3 — attach an on-disk dema_cache rooted at a
    /// sovereign_state/ directory. Subsequent permitted activations will
    /// auto-persist the PrincipalProfile to
    /// `<root>/dema_cache/principal.json`. Returns `&mut self` for
    /// builder-style composition.
    pub fn attach_dema_cache(&mut self, sovereign_root: &std::path::Path) -> &mut Self {
        self.dema_cache = Some(PrincipalProfileCache::at_sovereign_root(sovereign_root));
        self.receipt_history_cache = Some(ReceiptHistoryCache::at_sovereign_root(sovereign_root));
        self.manifest_history_cache = Some(ManifestHistoryCache::at_sovereign_root(sovereign_root));
        self.mission_log_cache = Some(MissionLogCache::at_sovereign_root(sovereign_root));
        self.state_snapshots_cache = Some(StateSnapshotsCache::at_sovereign_root(sovereign_root));
        self.resource_registry_cache =
            Some(ResourceRegistryCache::at_sovereign_root(sovereign_root));
        self.poi_ledger_cache = Some(PoiLedgerCache::at_sovereign_root(sovereign_root));
        self
    }

    /// Accessor for the attached dema cache (if any). Exposed for test
    /// harness use (inspecting the cache file path, forcing re-read).
    pub fn dema_cache(&self) -> Option<&PrincipalProfileCache> {
        self.dema_cache.as_ref()
    }

    /// Load an existing principal profile from the attached dema_cache,
    /// if one is present on disk. On success, sets `self.principal_profile`.
    /// Returns `Ok(true)` when a profile was loaded, `Ok(false)` when the
    /// cache file is absent, and `Err` when the cache is attached but the
    /// file is malformed.
    ///
    /// Niyyah §"Writer authority decision": the cache is derived and
    /// rebuildable. If this loader errors, callers should consider
    /// rebuilding the profile from chain rather than aborting.
    pub fn rehydrate_principal_from_cache(&mut self) -> Result<bool, PrincipalCacheError> {
        let cache = match &self.dema_cache {
            Some(c) => c,
            None => return Ok(false),
        };
        match cache.read()? {
            Some(profile) => {
                self.principal_profile = Some(profile);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Cycle-7 G3 Commit-1 — accessor for the attached receipt-history
    /// cache (if any). Exposed for test harness + gateway inspection.
    pub fn receipt_history_cache(&self) -> Option<&ReceiptHistoryCache> {
        self.receipt_history_cache.as_ref()
    }

    /// Cycle-7 G3 Commit-1 — build an in-memory snapshot of the current
    /// receipt chain suitable for serialization into the dema_cache. The
    /// snapshot is a thin projection: chain head, last payload timestamp,
    /// and the ordered (kind, hash, prev) tuples.
    ///
    /// Niyyah §"Writer authority decision": derived and rebuildable —
    /// the chain remains authoritative; the snapshot is a read fast-path.
    pub fn receipt_history_snapshot(&self) -> ReceiptHistorySnapshot {
        ReceiptHistorySnapshot {
            head: self.chain.head(),
            last_timestamp_ns: self.chain.latest_timestamp(),
            records: self.chain.records().copied().collect(),
        }
    }

    /// Cycle-7 G3 Commit-1 — write the current receipt history snapshot
    /// to the attached cache. Returns `Ok(None)` when no cache is
    /// attached, `Ok(Some(Ok(())))` on successful write, or
    /// `Ok(Some(Err(e)))` when a cache is attached but the write failed.
    ///
    /// Best-effort by design: callers propagate the inner Result as a
    /// warning string without aborting the already-sealed chain.
    pub fn write_receipt_history_cache(&self) -> Option<Result<(), ReceiptHistoryCacheError>> {
        self.receipt_history_cache
            .as_ref()
            .map(|c| c.write(&self.receipt_history_snapshot()))
    }

    /// Cycle-7 G3 Commit-1 — load the receipt-history snapshot from the
    /// attached cache, if present. Returns `Ok(None)` when no cache is
    /// attached OR the file is absent. The snapshot is derived state —
    /// callers must still replay the canonical chain for truth.
    pub fn rehydrate_receipt_history_from_cache(
        &self,
    ) -> Result<Option<ReceiptHistorySnapshot>, ReceiptHistoryCacheError> {
        match &self.receipt_history_cache {
            Some(c) => c.read(),
            None => Ok(None),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-2 — manifest_history cache surface
    // ═══════════════════════════════════════════════════════════════

    /// Accessor for the attached manifest-history cache (if any).
    pub fn manifest_history_cache(&self) -> Option<&ManifestHistoryCache> {
        self.manifest_history_cache.as_ref()
    }

    /// Build an in-memory snapshot of all ManifestArtifacts currently
    /// bound to permitted missions. Derived from the `missions`
    /// registry + current chain head. Authoritative source stays the
    /// chain; cache is a read fast-path.
    pub fn manifest_history_snapshot(&self) -> ManifestHistorySnapshot {
        let mut manifests: Vec<ManifestSummary> = self
            .missions
            .values()
            .filter(|r| !r.rejected)
            .filter_map(|r| r.manifest.as_ref().map(ManifestSummary::from))
            .collect();
        // Deterministic order by window_start then manifest_id.
        manifests.sort_by(|a, b| {
            a.window_start
                .cmp(&b.window_start)
                .then_with(|| a.manifest_id.cmp(&b.manifest_id))
        });
        ManifestHistorySnapshot {
            chain_head: self.chain.head(),
            manifests,
        }
    }

    /// Best-effort write of the manifest-history cache. `Ok(None)` when
    /// no cache is attached. See receipt_history counterpart for
    /// niyyah-alignment rationale.
    pub fn write_manifest_history_cache(&self) -> Option<Result<(), ManifestHistoryCacheError>> {
        self.manifest_history_cache
            .as_ref()
            .map(|c| c.write(&self.manifest_history_snapshot()))
    }

    /// Load the manifest-history snapshot from the attached cache, if
    /// present. Chain remains truth — this is a restart-fast-path.
    pub fn rehydrate_manifest_history_from_cache(
        &self,
    ) -> Result<Option<ManifestHistorySnapshot>, ManifestHistoryCacheError> {
        match &self.manifest_history_cache {
            Some(c) => c.read(),
            None => Ok(None),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-3 — mission_log cache surface
    // ═══════════════════════════════════════════════════════════════

    /// Accessor for the attached mission-log cache (if any).
    pub fn mission_log_cache(&self) -> Option<&MissionLogCache> {
        self.mission_log_cache.as_ref()
    }

    /// Build an in-memory snapshot of every mission attempted this
    /// session (permit + reject). Derived from the `missions` registry.
    /// Sorted by timestamp_ns for operator-legible chronology. Chain
    /// remains authoritative; cache is a read fast-path.
    pub fn mission_log_snapshot(&self) -> MissionLogSnapshot {
        let mut entries: Vec<MissionLogEntry> = self
            .missions
            .values()
            .map(|r| {
                let remediation = if r.rejected {
                    Some(reject_remediation_text(r))
                } else {
                    None
                };
                MissionLogEntry {
                    mission_id: r.envelope.mission_id,
                    intent_text: r.envelope.intent_text.clone(),
                    timestamp_ns: r.timestamp_ns,
                    rejected: r.rejected,
                    stage_byte: r.stage as u8,
                    receipt_id: r.receipt_id,
                    chain_head_after: r.chain_head,
                    quality_score: r.claim.quality_score,
                    remediation,
                }
            })
            .collect();
        entries.sort_by(|a, b| {
            a.timestamp_ns
                .cmp(&b.timestamp_ns)
                .then_with(|| a.mission_id.cmp(&b.mission_id))
        });
        MissionLogSnapshot {
            chain_head: self.chain.head(),
            entries,
        }
    }

    /// Best-effort write of the mission_log cache.
    pub fn write_mission_log_cache(&self) -> Option<Result<(), MissionLogCacheError>> {
        self.mission_log_cache
            .as_ref()
            .map(|c| c.write(&self.mission_log_snapshot()))
    }

    /// Load the mission_log snapshot from the attached cache, if
    /// present. Restart fast-path for operator-visible mission history.
    pub fn rehydrate_mission_log_from_cache(
        &self,
    ) -> Result<Option<MissionLogSnapshot>, MissionLogCacheError> {
        match &self.mission_log_cache {
            Some(c) => c.read(),
            None => Ok(None),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-4 — state_snapshots cache surface
    // ═══════════════════════════════════════════════════════════════

    /// Accessor for the attached state-snapshots cache (if any).
    pub fn state_snapshots_cache(&self) -> Option<&StateSnapshotsCache> {
        self.state_snapshots_cache.as_ref()
    }

    /// Build an in-memory snapshot of the FourStateModel attached to
    /// every mission attempt this session. Derived from the `missions`
    /// registry. Includes both permit and reject outcomes — operator
    /// sees where Dema thought NODE0 was, and where it aimed to be,
    /// regardless of whether the attempt was admitted.
    pub fn state_snapshots_snapshot(&self) -> StateSnapshotsSnapshot {
        let mut entries: Vec<StateSnapshotEntry> = self
            .missions
            .values()
            .map(|r| {
                let m = &r.envelope.state;
                StateSnapshotEntry {
                    mission_id: r.envelope.mission_id,
                    timestamp_ns: r.timestamp_ns,
                    rejected: r.rejected,
                    current: StateSnapshotView {
                        hash: m.current_state.hash,
                        summary: m.current_state.summary.clone(),
                        metric: m.current_state.metric,
                    },
                    ideal: StateSnapshotView {
                        hash: m.ideal_state.hash,
                        summary: m.ideal_state.summary.clone(),
                        metric: m.ideal_state.metric,
                    },
                    gap: m.gap,
                }
            })
            .collect();
        entries.sort_by(|a, b| {
            a.timestamp_ns
                .cmp(&b.timestamp_ns)
                .then_with(|| a.mission_id.cmp(&b.mission_id))
        });
        StateSnapshotsSnapshot {
            chain_head: self.chain.head(),
            entries,
        }
    }

    /// Best-effort write of the state_snapshots cache.
    pub fn write_state_snapshots_cache(&self) -> Option<Result<(), StateSnapshotsCacheError>> {
        self.state_snapshots_cache
            .as_ref()
            .map(|c| c.write(&self.state_snapshots_snapshot()))
    }

    /// Load the state_snapshots snapshot from the attached cache.
    pub fn rehydrate_state_snapshots_from_cache(
        &self,
    ) -> Result<Option<StateSnapshotsSnapshot>, StateSnapshotsCacheError> {
        match &self.state_snapshots_cache {
            Some(c) => c.read(),
            None => Ok(None),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-5 — resource_registry cache surface (seed only)
    // ═══════════════════════════════════════════════════════════════
    //
    // G3 seeds an empty registry at boot. G4 will own the mutation API
    // (register / allowlist / URP view). This module ships the file
    // shape and schema-version lock so G4 can build without reshaping.

    /// Accessor for the attached resource-registry cache (if any).
    pub fn resource_registry_cache(&self) -> Option<&ResourceRegistryCache> {
        self.resource_registry_cache.as_ref()
    }

    /// Seed an empty resource_registry.json if the file does not yet
    /// exist. Returns Ok(Some(true)) when a new empty file was created,
    /// Ok(Some(false)) when a file already existed, and Ok(None) when
    /// no cache is attached.
    pub fn seed_resource_registry_if_missing(
        &self,
    ) -> Result<Option<bool>, ResourceRegistryCacheError> {
        match &self.resource_registry_cache {
            Some(c) => c.seed_empty_if_missing().map(Some),
            None => Ok(None),
        }
    }

    /// Load the resource_registry snapshot from the attached cache.
    /// Restart fast-path; G4 will add URP projection on top of this.
    pub fn rehydrate_resource_registry_from_cache(
        &self,
    ) -> Result<Option<ResourceRegistrySnapshot>, ResourceRegistryCacheError> {
        match &self.resource_registry_cache {
            Some(c) => c.read(),
            None => Ok(None),
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Cycle-7 G6 — Proof-of-Impact ledger surface
    // ═══════════════════════════════════════════════════════════════

    /// Accessor for the attached PoI ledger cache (if any).
    pub fn poi_ledger_cache(&self) -> Option<&PoiLedgerCache> {
        self.poi_ledger_cache.as_ref()
    }

    /// In-memory PoI entries for this session. Authoritative for
    /// `poi_ledger_snapshot`; the disk cache is a derived fast-path.
    pub fn poi_entries(&self) -> &[PoiEntry] {
        &self.poi_entries
    }

    /// Build a ledger snapshot suitable for serialization.
    pub fn poi_ledger_snapshot(&self) -> PoiLedgerSnapshot {
        PoiLedgerSnapshot {
            chain_head: self.chain.head(),
            entries: self.poi_entries.clone(),
        }
    }

    /// Best-effort write of the PoI ledger cache.
    pub fn write_poi_ledger_cache(&self) -> Option<Result<(), PoiLedgerCacheError>> {
        self.poi_ledger_cache
            .as_ref()
            .map(|c| c.write(&self.poi_ledger_snapshot()))
    }

    /// Load the PoI ledger snapshot from the attached cache. Returns
    /// Ok(None) when no cache is attached or the file is absent.
    pub fn rehydrate_poi_ledger_from_cache(
        &self,
    ) -> Result<Option<PoiLedgerSnapshot>, PoiLedgerCacheError> {
        match &self.poi_ledger_cache {
            Some(c) => c.read(),
            None => Ok(None),
        }
    }

    /// Replace the in-memory PoI entries from a cache snapshot. Used on
    /// gateway boot to restore session state across restarts. Chain
    /// remains truth — callers may also invoke `rebuild_poi_ledger_from_chain`
    /// to verify+repair.
    pub fn load_poi_entries_from_cache(&mut self) -> Result<bool, PoiLedgerCacheError> {
        match self.rehydrate_poi_ledger_from_cache()? {
            Some(snap) => {
                self.poi_entries = snap.entries;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Record a PoI entry derived from a just-permitted mission.
    /// Called internally by submit_organize_mission + submit_principal_activation
    /// permit paths. Pushes to in-memory list and (best-effort) refreshes
    /// the disk cache.
    fn record_poi_for_mission(
        &mut self,
        receipt_id: Blake3Hash,
        kind_byte: u8,
        record: &MissionRuntimeRecord,
        entry_count: u32,
    ) {
        let gate_min_score = record
            .admissibility
            .gate_verdicts
            .iter()
            .filter_map(|g| g.score)
            .fold(f64::INFINITY, f64::min);
        let gate_min_score = if gate_min_score.is_finite() {
            gate_min_score
        } else {
            // All scores were None — fall back to the claim's quality
            // so the entry is not worse than the operator's evidence.
            record.claim.quality_score
        };
        let quality_score = record.claim.quality_score;
        let impact_score = compute_impact_score(quality_score, gate_min_score, entry_count);
        let principal_id = match record.envelope.originator {
            Originator::Operator { session_id } if session_id != [0u8; 32] => Some(session_id),
            _ => None,
        };

        self.poi_entries.push(PoiEntry {
            receipt_id,
            receipt_kind_byte: kind_byte,
            quality_score,
            gate_min_score,
            entry_count,
            impact_score,
            timestamp_ns: record.timestamp_ns,
            principal_id,
        });
        let _ = self.write_poi_ledger_cache();
    }

    // ═══════════════════════════════════════════════════════════════
    // Cycle-7 G4 Commit-1 — typed resource registry API
    // ═══════════════════════════════════════════════════════════════
    //
    // Read-modify-write against resource_registry_cache. Local-only,
    // non-chain per niyyah §"Writer authority HYBRID". No chain receipt
    // is emitted for a register call — G5 will read the allowlist when
    // deciding whether an `organize` mission may proceed.

    /// Register or update a local resource. Behavior:
    ///   - New (kind, id) → `RegisterOutcome::Created`
    ///   - Existing (kind, id) with differing summary or allowlist flag
    ///     → `RegisterOutcome::Updated`
    ///   - Exact match already present → `RegisterOutcome::Idempotent`
    ///     (write is elided).
    ///
    /// Equality key: (kind.as_str(), id). Two different ResourceKind
    /// variants with the same canonical string collapse to one entry.
    pub fn register_resource(
        &self,
        resource: TypedResource,
    ) -> Result<RegisterOutcome, ResourceRegistryError> {
        let cache = self
            .resource_registry_cache
            .as_ref()
            .ok_or(ResourceRegistryError::NoCacheAttached)?;

        let mut snapshot = cache.read()?.unwrap_or_default();
        let key = (resource.kind.as_str().to_string(), resource.id.clone());

        let mut outcome = RegisterOutcome::Created;
        let mut replaced = false;
        for existing in snapshot.resources.iter_mut() {
            if existing.kind == key.0 && existing.id == key.1 {
                if existing.summary == resource.summary
                    && existing.allowlisted == resource.allowlisted
                {
                    return Ok(RegisterOutcome::Idempotent);
                }
                *existing = resource.to_cache_entry();
                outcome = RegisterOutcome::Updated;
                replaced = true;
                break;
            }
        }
        if !replaced {
            snapshot.resources.push(resource.to_cache_entry());
        }
        // Deterministic ordering: by (kind, id).
        snapshot
            .resources
            .sort_by(|a, b| a.kind.cmp(&b.kind).then_with(|| a.id.cmp(&b.id)));

        cache.write(&snapshot)?;
        Ok(outcome)
    }

    /// List all registered resources as typed projections. Returns an
    /// empty Vec when no cache is attached or the file is absent.
    pub fn list_resources(&self) -> Result<Vec<TypedResource>, ResourceRegistryError> {
        let cache = match &self.resource_registry_cache {
            Some(c) => c,
            None => return Ok(Vec::new()),
        };
        let snapshot = cache.read()?.unwrap_or_default();
        Ok(snapshot.resources.iter().map(TypedResource::from).collect())
    }

    /// True when (kind, id) is registered AND allowlisted. False on any
    /// other state (absent, present-but-denied). Errors only on cache
    /// malformation.
    pub fn is_allowlisted(
        &self,
        kind: &ResourceKind,
        id: &str,
    ) -> Result<bool, ResourceRegistryError> {
        let cache = match &self.resource_registry_cache {
            Some(c) => c,
            None => return Ok(false),
        };
        let snapshot = cache.read()?.unwrap_or_default();
        let kind_str = kind.as_str();
        Ok(snapshot
            .resources
            .iter()
            .any(|r| r.kind == kind_str && r.id == id && r.allowlisted))
    }

    /// Cycle-7 G4 Commit-2 — build the Universal Resource Pattern view.
    /// Canonical projection of the registry grouped by kind with
    /// deterministic ordering. G5's `dema organize` consults the
    /// FilesystemPath bucket to locate allowlisted targets.
    pub fn urp_view(&self) -> Result<UrpView, ResourceRegistryError> {
        Ok(UrpView::from_resources(self.list_resources()?))
    }

    pub fn rehydrate_mission(
        &self,
        mission_id: &Blake3Hash,
    ) -> Result<MissionReplayReport, MissionRuntimeError> {
        let record = self
            .missions
            .get(mission_id)
            .ok_or(MissionRuntimeError::MissionNotFound(*mission_id))?;

        let pre_head = self.chain.head();
        let replay = AdmissibilityChain::canonical().evaluate(&record.claim);
        let chain_head = self.chain.head();
        let matches_previous =
            pre_head == chain_head && admissibility_result_matches(&record.admissibility, &replay);

        Ok(MissionReplayReport {
            mission_id: *mission_id,
            replay_result: if matches_previous {
                MissionReplayResult::Match
            } else {
                MissionReplayResult::Divergent
            },
            matches_previous,
            chain_head,
        })
    }

    pub fn mission_count(&self) -> usize {
        self.missions.len()
    }

    /// Single event handler.
    pub fn handle(&mut self, event: CognitionEvent) -> Result<Option<Vec<Thought>>, LoopError> {
        match event {
            CognitionEvent::ReasoningRequest { request_id } => {
                self.handle_reasoning(request_id).map(Some)
            }
            CognitionEvent::ConsolidationTick => {
                self.handle_consolidation()?;
                Ok(None)
            }
            CognitionEvent::GovernanceDemyelination { edge, decision } => {
                self.handle_governance_demyelination(edge, decision)?;
                Ok(None)
            }
            CognitionEvent::Shutdown => Err(LoopError::Shutdown),
        }
    }

    fn handle_reasoning(&mut self, request_id: Blake3Hash) -> Result<Vec<Thought>, LoopError> {
        let thoughts = match self.graph.reason(&mut self.ctx) {
            Ok(t) => t,
            Err(e) => {
                self.emit_degraded(DegradedOccasion::ReasoningFailed {
                    cause: format!("{:?}", e),
                })?;
                return Err(LoopError::Reasoning(e));
            }
        };

        let thoughts_digest =
            blake3_domain("bizra-thoughts-v1", &(thoughts.len() as u32).to_le_bytes());

        let receipt = ReasoningSessionReceipt {
            request_id,
            thoughts_digest,
            thoughts_count: thoughts.len() as u32,
            prev_chain: self.chain.head(),
            timestamp_ns: self.now_ns()?,
        };

        self.chain.append_with_payload(receipt)?;
        self.session_counter += 1;

        Ok(thoughts)
    }

    fn handle_consolidation(&mut self) -> Result<(), LoopError> {
        // Enable shadow observation during consolidation so the next reasoning
        // cycle captures divergence data for candidates in quarantine.
        // Scheduler is responsible for turning it off afterward.
        self.graph.set_shadow_mode(ShadowMode::On);

        let proposals = self.graph.propose_myelinations();

        for (edge_hash, proposal) in proposals {
            let (receipt, compiled) = match proposal {
                Ok(pair) => pair,
                Err(ReasoningError::ImmutableS2Violation(_)) => continue,
                Err(ReasoningError::QuarantineDivergence {
                    candidate,
                    divergence,
                }) => {
                    self.emit_degraded(DegradedOccasion::ConsolidationDivergence {
                        edge: candidate,
                        divergence,
                    })?;
                    continue;
                }
                Err(e) => {
                    self.emit_degraded(DegradedOccasion::ReasoningFailed {
                        cause: format!("consolidation: {:?}", e),
                    })?;
                    continue;
                }
            };

            match self.chain.append_with_payload(receipt) {
                Ok(receipt_hash) => {
                    self.graph
                        .commit_myelination(edge_hash, compiled, receipt_hash);
                }
                Err(chain_err) => {
                    self.graph.abort_myelination(edge_hash);
                    self.emit_degraded(DegradedOccasion::MyelinationPersistFailed {
                        edge: edge_hash,
                        cause: format!("{:?}", chain_err),
                    })?;
                }
            }
        }

        Ok(())
    }

    fn handle_governance_demyelination(
        &mut self,
        edge: Blake3Hash,
        decision: Blake3Hash,
    ) -> Result<(), LoopError> {
        let receipt = match self
            .graph
            .propose_demyelination(edge, DemyelinationReason::GovernanceDecision(decision))
        {
            Some(r) => r,
            None => return Ok(()),
        };

        match self.chain.append_with_payload(receipt) {
            Ok(new_head) => {
                self.graph.commit_demyelination(new_head);
                Ok(())
            }
            Err(chain_err) => {
                self.emit_degraded(DegradedOccasion::DemyelinationPersistFailed {
                    edge,
                    cause: format!("{:?}", chain_err),
                })?;
                Ok(())
            }
        }
    }

    fn emit_degraded(&mut self, occasion: DegradedOccasion) -> Result<(), LoopError> {
        let receipt = DegradedPathReceipt {
            occasion,
            prev_chain: self.chain.head(),
            timestamp_ns: self.now_ns()?,
        };
        self.chain.append_with_payload(receipt)?;
        Ok(())
    }

    fn now_ns(&mut self) -> Result<u64, LoopError> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .map_err(|e| LoopError::Clock(e.to_string()))
    }

    fn now_ns_mission(&self) -> Result<u64, MissionRuntimeError> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .map_err(|e| MissionRuntimeError::Clock(e.to_string()))
    }
}

/// Cycle-7 G2 — build honest structured remediation text from a
/// rejected mission record. Never simulates success; always names the
/// failed gate and the concrete remediation path (niyyah Frozen Law #5).
fn reject_remediation_text(record: &MissionRuntimeRecord) -> String {
    if let Some(rc) = record.admissibility.rejected.as_ref() {
        format!(
            "activation REJECTED by {} — {}. Remediation: {}. Escalation: {}.",
            rc.invariant.name(),
            rc.reject_reason,
            rc.remediation_path,
            if rc.escalation_allowed {
                "allowed (REVIEW)"
            } else {
                "denied"
            },
        )
    } else {
        format!(
            "activation REJECTED — verdict {:?}. See mission_record.admissibility.gate_verdicts.",
            record.admissibility.verdict,
        )
    }
}

fn admissibility_result_matches(left: &AdmissibilityResult, right: &AdmissibilityResult) -> bool {
    left.verdict == right.verdict
        && left.gate_verdicts.len() == right.gate_verdicts.len()
        && left
            .gate_verdicts
            .iter()
            .zip(right.gate_verdicts.iter())
            .all(|(a, b)| gate_verdict_matches(a, b))
        && rejected_claim_matches(left.rejected.as_ref(), right.rejected.as_ref())
}

fn gate_verdict_matches(left: &GateVerdict, right: &GateVerdict) -> bool {
    left.verdict == right.verdict
        && left.reason == right.reason
        && left.scorer_id == right.scorer_id
        && left.chain_ref == right.chain_ref
        && left.timestamp_ns == right.timestamp_ns
        && left.invariant == right.invariant
        && left.score == right.score
}

fn rejected_claim_matches(left: Option<&RejectedClaim>, right: Option<&RejectedClaim>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.claim_ref == b.claim_ref
                && a.invariant == b.invariant
                && a.reject_reason == b.reject_reason
                && a.remediation_path == b.remediation_path
                && a.escalation_allowed == b.escalation_allowed
        }
        _ => false,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admissibility_freeze_v1::{
        AdmissibilityClaim, EconomicPattern, StateMutation, Verdict,
    };
    use crate::mission_freeze_v1::{Originator, StateSnapshot};
    use crate::receipts::InMemoryPayloadStore;
    use crate::thought_graph::{GraphNode, MyelinationPolicy};

    struct NoopNode;
    impl GraphNode for NoopNode {
        fn traverse(&self, _ctx: &mut AgentCtx) -> Vec<Thought> {
            vec![Thought]
        }
    }

    fn minimal_graph() -> (ThoughtGraph, Blake3Hash) {
        let root_hash = [1u8; 32];
        let mut nodes: HashMap<Blake3Hash, Box<dyn GraphNode>> = HashMap::new();
        nodes.insert(root_hash, Box::new(NoopNode));
        let mut policies = HashMap::new();
        policies.insert(root_hash, MyelinationPolicy::standard());
        let genesis = [0u8; 32];
        (
            ThoughtGraph::from_parts(nodes, vec![root_hash], policies, genesis),
            genesis,
        )
    }

    fn minimal_runtime() -> CognitionRuntime {
        let (graph, genesis) = minimal_graph();
        let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx = AgentCtx {
            receipt_chain: genesis,
        };
        CognitionRuntime::new(graph, chain, ctx)
    }

    fn current_state() -> StateSnapshot {
        StateSnapshot {
            hash: [0x11; 32],
            summary: "Current: principal activation not yet canonical".into(),
            metric: 0.0,
        }
    }

    fn ideal_state() -> StateSnapshot {
        StateSnapshot {
            hash: [0x22; 32],
            summary: "Ideal: principal activation receipted and canonical".into(),
            metric: 1.0,
        }
    }

    fn test_mission(now_ns: u64) -> MissionEnvelope {
        MissionEnvelope::from_intent(
            "Activate my dual-agentic system".into(),
            current_state(),
            ideal_state(),
            Originator::Operator {
                session_id: [0x33; 32],
            },
            now_ns,
        )
    }

    fn permit_claim(env: &MissionEnvelope, now_ns: u64) -> AdmissibilityClaim {
        AdmissibilityClaim {
            claim_id: env.extract_claim_id(),
            has_evidence: true,
            evidence_hash: Some([0x44; 32]),
            economic_pattern: Some(EconomicPattern::None),
            state_mutation: Some(StateMutation {
                derives_from_canonical: true,
                face_only: false,
            }),
            quality_score: 0.98,
            timestamp_ns: now_ns,
        }
    }

    fn reject_claim(env: &MissionEnvelope, now_ns: u64) -> AdmissibilityClaim {
        AdmissibilityClaim {
            quality_score: 0.40,
            ..permit_claim(env, now_ns)
        }
    }

    #[test]
    fn reasoning_request_produces_thoughts_and_advances_chain() {
        let mut rt = minimal_runtime();
        let initial_head = rt.chain.head();
        let result = rt
            .handle(CognitionEvent::ReasoningRequest {
                request_id: [42u8; 32],
            })
            .unwrap();
        assert!(result.is_some());
        assert_ne!(rt.chain.head(), initial_head);
        assert_eq!(rt.chain.len(), 1);
    }

    #[test]
    fn consolidation_with_no_candidates_is_noop() {
        let mut rt = minimal_runtime();
        let initial_head = rt.chain.head();
        let initial_len = rt.chain.len();
        rt.handle(CognitionEvent::ConsolidationTick).unwrap();
        assert_eq!(rt.chain.head(), initial_head);
        assert_eq!(rt.chain.len(), initial_len);
    }

    #[test]
    fn shutdown_returns_loop_error() {
        let mut rt = minimal_runtime();
        assert!(matches!(
            rt.handle(CognitionEvent::Shutdown),
            Err(LoopError::Shutdown)
        ));
    }

    #[test]
    fn chain_continuity_holds_across_many_events() {
        let mut rt = minimal_runtime();
        let genesis = [0u8; 32];
        for i in 0..10u8 {
            rt.handle(CognitionEvent::ReasoningRequest {
                request_id: [i; 32],
            })
            .unwrap();
        }
        assert_eq!(rt.chain.len(), 10);
        rt.chain.verify_continuity(genesis).unwrap();
    }

    /// The key test for R1: after events, rebuild the graph from the chain
    /// and verify reflex state matches.
    ///
    /// We can't test this end-to-end without a real CompiledReflex impl
    /// (the stub version will myelinate with 0 divergence, then the rehydrate
    /// will correctly reinstall it). This test proves the round-trip.
    #[test]
    fn rehydrate_reconstructs_reflex_state_from_chain() {
        let (graph, genesis) = minimal_graph();
        let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx = AgentCtx {
            receipt_chain: genesis,
        };
        let mut rt = CognitionRuntime::new(graph, chain, ctx);

        // Directly construct a MyelinationReceipt and write it to the chain —
        // this simulates a reflex that was previously committed. We use a
        // fake edge_hash that corresponds to the node we inserted above.
        let edge = [1u8; 32];
        let receipt = MyelinationReceipt {
            source_s2: edge,
            compiled_reflex: [99u8; 32],
            quarantine_evidence: vec![[7u8; 32]; 3],
            observed_divergence: 0.0,
            policy_version: 1,
            prev_chain: rt.chain.head(),
            timestamp_ns: 1_000_000,
        };
        rt.chain.append_with_payload(receipt).unwrap();

        assert!(!rt.graph.has_reflex(&edge), "reflex not yet installed");

        // Now rehydrate: rebuild the graph from the chain.
        let (fresh_graph, _) = minimal_graph();
        // Move the chain into the rehydrate call
        let rehydrated = CognitionRuntime::rehydrate(fresh_graph, rt.chain).unwrap();

        assert!(
            rehydrated.graph.has_reflex(&edge),
            "rehydrate must reconstruct reflex from Myelination receipt"
        );
    }

    /// Prove that Myelination followed by Demyelination leaves no reflex
    /// after rehydrate — the chain's final state wins.
    #[test]
    fn rehydrate_respects_demyelination() {
        let (graph, genesis) = minimal_graph();
        let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx = AgentCtx {
            receipt_chain: genesis,
        };
        let mut rt = CognitionRuntime::new(graph, chain, ctx);

        let edge = [1u8; 32];

        // Write myelination
        let myel = MyelinationReceipt {
            source_s2: edge,
            compiled_reflex: [99u8; 32],
            quarantine_evidence: vec![],
            observed_divergence: 0.0,
            policy_version: 1,
            prev_chain: rt.chain.head(),
            timestamp_ns: 1_000_000,
        };
        rt.chain.append_with_payload(myel).unwrap();

        // Write demyelination
        let demyel = DemyelinationReceipt {
            reflex: edge,
            reason: DemyelinationReason::PolicyVersionBump,
            prev_chain: rt.chain.head(),
            timestamp_ns: 2_000_000,
        };
        rt.chain.append_with_payload(demyel).unwrap();

        // Rehydrate — should end with NO reflex for this edge
        let (fresh_graph, _) = minimal_graph();
        let rehydrated = CognitionRuntime::rehydrate(fresh_graph, rt.chain).unwrap();

        assert!(
            !rehydrated.graph.has_reflex(&edge),
            "rehydrate must respect demyelination — chain final state wins"
        );
    }

    /// The operational proof of R1: two nodes starting from the same chain
    /// end in identical reflex state. This is Node1 reproducibility.
    #[test]
    fn rehydrate_is_deterministic() {
        let (graph_a, genesis) = minimal_graph();
        let chain_a = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx_a = AgentCtx {
            receipt_chain: genesis,
        };
        let mut rt_a = CognitionRuntime::new(graph_a, chain_a, ctx_a);

        let edge = [1u8; 32];
        let myel = MyelinationReceipt {
            source_s2: edge,
            compiled_reflex: [99u8; 32],
            quarantine_evidence: vec![[3u8; 32]; 2],
            observed_divergence: 0.01,
            policy_version: 1,
            prev_chain: rt_a.chain.head(),
            timestamp_ns: 1_000_000,
        };
        rt_a.chain.append_with_payload(myel.clone()).unwrap();

        // Now: two separate rehydrates from the same chain state should produce
        // identical graphs. We can't easily share a chain across two rehydrates
        // (rehydrate takes ownership), but we can rebuild an identical chain
        // and verify the result is byte-equivalent.
        let chain_a_head = rt_a.chain.head();
        let (fresh_a, _) = minimal_graph();
        let rehydrated_a = CognitionRuntime::rehydrate(fresh_a, rt_a.chain).unwrap();

        // Build an equivalent chain from scratch
        let (graph_b, _) = minimal_graph();
        let chain_b = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx_b = AgentCtx {
            receipt_chain: genesis,
        };
        let mut rt_b = CognitionRuntime::new(graph_b, chain_b, ctx_b);
        rt_b.chain.append_with_payload(myel).unwrap();

        let (fresh_b, _) = minimal_graph();
        let rehydrated_b = CognitionRuntime::rehydrate(fresh_b, rt_b.chain).unwrap();

        assert_eq!(rehydrated_a.graph.chain_head(), chain_a_head);
        assert_eq!(
            rehydrated_a.graph.chain_head(),
            rehydrated_b.graph.chain_head()
        );
        assert_eq!(
            rehydrated_a.graph.has_reflex(&edge),
            rehydrated_b.graph.has_reflex(&edge)
        );
    }

    #[test]
    fn submit_mission_records_chain_backed_runtime_state() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(10_000);
        let claim = permit_claim(&envelope, 10_500);

        let record = rt.submit_mission(envelope, claim).unwrap();

        // G2-hardening: record is returned directly; receipt_id and stage fields
        // are authoritative; registry lookup yields the same record.
        assert!(!record.rejected);
        assert_eq!(record.envelope.stage, MissionStage::Replayability);
        assert_eq!(record.stage, MissionStage::Replayability);
        assert_eq!(record.admissibility.verdict, Verdict::Permit);
        assert_eq!(record.gate_receipt_hashes.len(), 5);
        let final_receipt = record
            .final_receipt
            .as_ref()
            .expect("permit must have final receipt");
        assert_eq!(final_receipt.kind, ReceiptKind::NodeLifecycle);
        assert_eq!(final_receipt.claim_ref, record.envelope.mission_id);
        assert_eq!(record.receipt_id, Some(final_receipt.receipt_id));
        assert_eq!(
            record.mission_payload_hash,
            Some(final_receipt.claim_ref).map(|_| record.mission_payload_hash.unwrap())
        );
        assert_eq!(rt.mission_count(), 1);
        // Cycle-7 G1: chain now carries 8 records per permitted mission —
        // 1 envelope + 5 gates + 1 final ReceiptArtifact + 1 ManifestArtifact.
        // Head advances to the manifest hash (the manifest is the LAST thing
        // appended on the permit path).
        assert_eq!(
            rt.chain.len(),
            8,
            "1 mission + 5 gates + 1 final receipt + 1 manifest"
        );
        let manifest = record
            .manifest
            .as_ref()
            .expect("permit must carry a manifest");
        assert_eq!(rt.chain.head(), manifest.manifest_id);

        // Registry lookup returns an equivalent record.
        let from_registry = rt.mission_by_id(&record.envelope.mission_id).unwrap();
        assert_eq!(from_registry.receipt_id, record.receipt_id);
        assert!(!from_registry.rejected);
    }

    #[test]
    fn submit_mission_rejects_claim_id_mismatch_before_persisting() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(20_000);
        let mut claim = permit_claim(&envelope, 20_100);
        claim.claim_id = [0x99; 32];

        let err = rt.submit_mission(envelope, claim).unwrap_err();
        assert!(matches!(err, MissionRuntimeError::ClaimMismatch { .. }));
        assert_eq!(rt.mission_count(), 0);
        assert_eq!(rt.chain.len(), 0);
    }

    #[test]
    fn submit_mission_rejects_without_canonicalizing_and_preserves_in_registry() {
        // G2-hardening (Patch A, per spec g2-patches-abc.md): a REJECT verdict
        // does NOT produce a final receipt, does NOT append the envelope to the
        // chain, and does NOT raise an error. Instead, the method returns
        // Ok(record) with rejected=true, stage=Admissibility, receipt_id=None.
        // The chain stays CLEAN of rejected missions (§10 "chain is truth =
        // only lawful completions"). The rejection is recorded in the missions
        // registry as derived state — queryable via mission_by_id().
        let mut rt = minimal_runtime();
        let envelope = test_mission(30_000);
        let mission_id = envelope.mission_id;
        let pre_chain_len = rt.chain.len();
        let claim = reject_claim(&envelope, 30_100);

        let record = rt.submit_mission(envelope, claim).unwrap();

        assert!(
            record.rejected,
            "verdict was Reject — record must be marked rejected"
        );
        assert!(
            record.receipt_id.is_none(),
            "rejected mission must have no receipt_id"
        );
        assert!(
            record.final_receipt.is_none(),
            "rejected mission must have no final receipt"
        );
        assert!(
            record.mission_payload_hash.is_none(),
            "rejected mission envelope was NOT appended to chain"
        );
        assert_eq!(
            record.stage,
            MissionStage::Admissibility,
            "rejected mission stops at S4 Admissibility"
        );
        assert_eq!(record.admissibility.verdict, Verdict::Reject);
        assert!(
            record.admissibility.rejected.is_some(),
            "reject must carry RejectedClaim with remediation path"
        );

        // Chain UNCHANGED on reject (§10 chain-is-truth).
        assert_eq!(
            rt.chain.len(),
            pre_chain_len,
            "rejected mission must NOT advance the chain at all"
        );

        // Registry PRESERVES the rejection (derived state per §10).
        assert_eq!(
            rt.mission_count(),
            1,
            "rejected mission must be queryable via mission_by_id"
        );
        let from_registry = rt.mission_by_id(&mission_id).unwrap();
        assert!(from_registry.rejected);
        assert_eq!(from_registry.stage, MissionStage::Admissibility);
    }

    #[test]
    fn submit_mission_advances_to_replayability_on_permit() {
        // G2-hardening (Patch B): permitted mission ends at S8 Replayability
        // ONLY when decode round-trip verifies. For InMemoryPayloadStore with
        // a well-formed ReceiptArtifact the round-trip must succeed.
        let mut rt = minimal_runtime();
        let envelope = test_mission(50_000);
        let claim = permit_claim(&envelope, 50_100);

        let record = rt.submit_mission(envelope, claim).unwrap();

        assert_eq!(
            record.envelope.stage,
            MissionStage::Replayability,
            "permit + decode-verified replay must reach S8"
        );
        assert_eq!(record.stage, MissionStage::Replayability);
    }

    #[test]
    fn rehydrate_mission_is_pure_and_matches_previous_verdict() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(40_000);
        let claim = permit_claim(&envelope, 40_050);
        let record = rt.submit_mission(envelope, claim).unwrap();
        let mission_id = record.envelope.mission_id;
        let pre_head = rt.chain.head();

        let replay = rt.rehydrate_mission(&mission_id).unwrap();

        assert_eq!(replay.replay_result, MissionReplayResult::Match);
        assert!(replay.matches_previous);
        assert_eq!(replay.chain_head, pre_head);
        assert_eq!(rt.chain.head(), pre_head, "rehydrate_mission must be pure");
    }

    // ========================================================================
    // Cycle-7 G1 — ManifestArtifact emission + binding tests
    // ========================================================================

    #[test]
    fn c7_submit_mission_produces_manifest_on_permit_replayability() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(50_000);
        let claim = permit_claim(&envelope, 50_050);

        let record = rt.submit_mission(envelope, claim).unwrap();

        // Permit → Replayability → manifest present.
        assert!(!record.rejected);
        assert_eq!(record.stage, MissionStage::Replayability);
        let manifest = record
            .manifest
            .as_ref()
            .expect("permitted mission at Replayability must carry a manifest");

        // Manifest fields are deterministic and mission-bound.
        assert_eq!(manifest.receipt_count as usize, manifest.receipt_refs.len());
        assert!(
            manifest.verify_integrity(),
            "integrity hash must round-trip"
        );

        // chain_head_at_generation captures the PRE-manifest-append head —
        // i.e., the final ReceiptArtifact (NodeLifecycle) hash. The manifest
        // itself is then appended, so post-append chain.head() == manifest_id,
        // NOT chain_head_at_generation.
        let receipt_id = record.receipt_id.expect("permit has receipt_id");
        assert_eq!(
            manifest.chain_head_at_generation, receipt_id,
            "chain_head_at_generation captures pre-manifest head (the ReceiptArtifact)"
        );
        assert_eq!(
            rt.chain.head(),
            manifest.manifest_id,
            "post-append chain head must be the manifest itself"
        );

        // Mission's receipt_id is bound into the manifest's refs
        // (caution #2 from Cycle-7 niyyah review — unambiguous mission-binding).
        assert!(
            manifest.receipt_refs.contains(&receipt_id),
            "manifest must bind the mission's receipt_id"
        );
    }

    #[test]
    fn c7_submit_mission_rejected_emits_no_manifest_proof_law_s10() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(51_000);
        // Set quality BELOW IHSAN_FLOOR to force reject.
        let mut claim = permit_claim(&envelope, 51_050);
        claim.quality_score = 0.50;

        let record = rt.submit_mission(envelope, claim).unwrap();

        // §10 Proof Law: rejected missions do NOT emit manifests.
        assert!(record.rejected);
        assert_eq!(record.stage, MissionStage::Admissibility);
        assert!(
            record.manifest.is_none(),
            "§10 Proof Law: rejected mission must not emit a manifest"
        );
    }

    #[test]
    fn c7_manifest_for_mission_accessor_queryable_post_commit() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(52_000);
        let claim = permit_claim(&envelope, 52_050);
        let record = rt.submit_mission(envelope, claim).unwrap();
        let mission_id = record.envelope.mission_id;

        // Accessor returns Some for the known mission_id.
        let via_accessor = rt
            .manifest_for_mission(&mission_id)
            .expect("accessor must find manifest for permitted mission");
        let in_record = record.manifest.as_ref().unwrap();
        assert_eq!(via_accessor.manifest_id, in_record.manifest_id);

        // Accessor returns None for an unknown mission_id.
        let unknown: Blake3Hash = [0xCD; 32];
        assert!(rt.manifest_for_mission(&unknown).is_none());
    }

    #[test]
    fn c7_manifest_appears_in_chain_as_receipt_payload_with_manifest_kind() {
        let mut rt = minimal_runtime();
        let envelope = test_mission(53_000);
        let claim = permit_claim(&envelope, 53_050);
        let record = rt.submit_mission(envelope, claim).unwrap();

        let manifest = record.manifest.as_ref().unwrap();

        // The manifest payload is the LAST record on chain (appended after
        // ReceiptArtifact), and it carries the dedicated ReceiptKind::Manifest.
        let last = rt.chain.records().last().expect("chain must be non-empty");
        assert_eq!(last.kind, ReceiptKind::Manifest);
        assert_eq!(last.hash, manifest.manifest_id);
    }

    #[test]
    fn c7_rehydrate_mission_is_pure_when_manifest_is_appended() {
        // Regression: Cycle-7 G1 adds a record to the chain per permitted
        // mission. rehydrate_mission must remain pure (zero mutation).
        let mut rt = minimal_runtime();
        let envelope = test_mission(54_000);
        let claim = permit_claim(&envelope, 54_050);
        let record = rt.submit_mission(envelope, claim).unwrap();
        let mission_id = record.envelope.mission_id;
        let pre_head = rt.chain.head();
        let pre_len = rt.chain.len();

        let replay = rt.rehydrate_mission(&mission_id).unwrap();

        assert_eq!(replay.replay_result, MissionReplayResult::Match);
        assert_eq!(rt.chain.head(), pre_head, "rehydrate must not mutate head");
        assert_eq!(rt.chain.len(), pre_len, "rehydrate must not mutate length");
        // Manifest is still accessible after rehydrate.
        assert!(rt.manifest_for_mission(&mission_id).is_some());
    }

    // ========================================================================
    // Cycle-7 G2 Phase 2 — submit_principal_activation integration tests
    // ========================================================================

    mod principal_activation_g2 {
        use super::*;
        use crate::principal_activation::{NodeIdentityAnchor, PrincipalActivationEnvelope};

        const TEST_PUBKEY_HEX: &str =
            "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

        fn test_anchor() -> NodeIdentityAnchor {
            NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "2026-04-13T23:54:59Z")
        }

        fn test_envelope(now_ns: u64) -> PrincipalActivationEnvelope {
            PrincipalActivationEnvelope::from_anchor(
                "Mumo".into(),
                "node0_principal".into(),
                &test_anchor(),
                now_ns,
            )
            .expect("valid anchor builds envelope")
        }

        #[test]
        fn activate_principal_happy_path_seals_receipt_and_profile() {
            let mut rt = minimal_runtime();
            let env = test_envelope(1_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();

            assert!(!record.rejected, "quality 0.98 should Permit");
            assert!(record.profile.is_some());
            assert!(record.activation_receipt.is_some());
            assert!(record.remediation.is_none());
            assert!(rt.principal_profile().is_some());

            let profile = record.profile.as_ref().unwrap();
            assert_eq!(profile.name, "Mumo");
            assert_eq!(profile.node_id, "NODE0");
            assert_eq!(profile.declared_role, "node0_principal");
            let mission_receipt_id = record.mission_record.receipt_id.unwrap();
            assert_eq!(profile.activation_receipt_id, mission_receipt_id);
        }

        #[test]
        fn activate_principal_rejects_low_quality_with_remediation() {
            let mut rt = minimal_runtime();
            let env = test_envelope(2_000);
            let record = rt.submit_principal_activation(env, 0.40).unwrap();

            assert!(
                record.rejected,
                "quality 0.40 below IHSAN_FLOOR must Reject"
            );
            assert!(
                record.profile.is_none(),
                "§10 Proof Law: no profile on reject"
            );
            assert!(
                record.activation_receipt.is_none(),
                "§10 Proof Law: no PrincipalActivationReceipt on reject"
            );
            assert!(
                rt.principal_profile().is_none(),
                "runtime principal_profile must stay None on reject"
            );
            let remediation = record.remediation.as_ref().expect("remediation set");
            assert!(
                remediation.contains("REJECTED"),
                "remediation must name the rejection honestly: {}",
                remediation
            );
        }

        #[test]
        fn activate_principal_receipt_binds_mission_and_profile_hashes() {
            let mut rt = minimal_runtime();
            let env = test_envelope(3_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();

            let receipt = record.activation_receipt.as_ref().unwrap();
            let profile = record.profile.as_ref().unwrap();
            let mission_receipt_id = record.mission_record.receipt_id.unwrap();

            assert_eq!(
                receipt.activation_receipt_ref, mission_receipt_id,
                "PrincipalActivationReceipt must reference the NodeLifecycle mission receipt_id"
            );
            assert_eq!(
                receipt.principal_profile_hash,
                profile.profile_hash(),
                "PrincipalActivationReceipt must carry the profile's canonical hash"
            );
            assert_eq!(receipt.principal_id, profile.principal_id);
        }

        #[test]
        fn activate_principal_permit_grows_chain_by_exactly_nine() {
            // 1 envelope + 5 gates + NodeLifecycle + Manifest + PrincipalActivation = 9
            let mut rt = minimal_runtime();
            let before = rt.chain.len();
            let env = test_envelope(4_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();
            let after = rt.chain.len();
            assert!(!record.rejected);
            assert_eq!(
                after - before,
                9,
                "permit activation must append exactly 9 chain records (got {})",
                after - before
            );
        }

        #[test]
        fn activate_principal_reject_grows_chain_by_zero() {
            // §10 Proof Law: rejected claims do not enter the chain.
            let mut rt = minimal_runtime();
            let before = rt.chain.len();
            let env = test_envelope(5_000);
            let record = rt.submit_principal_activation(env, 0.40).unwrap();
            let after = rt.chain.len();
            assert!(record.rejected);
            assert_eq!(
                after - before,
                0,
                "rejected activation must not touch the chain (grew by {})",
                after - before
            );
        }

        #[test]
        fn activate_principal_chain_head_is_activation_receipt_on_permit() {
            let mut rt = minimal_runtime();
            let env = test_envelope(6_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();
            let pa_receipt = record.activation_receipt.as_ref().unwrap();
            assert_eq!(
                rt.chain.head(),
                pa_receipt.receipt_id,
                "chain head must equal the PrincipalActivationReceipt id after permit"
            );
        }

        // ── Cycle-7 G2 Commit-3 — disk cache integration ──

        #[test]
        fn activate_with_cache_writes_profile_to_disk() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let env = test_envelope(10_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();

            assert!(!record.rejected);
            assert!(record.cache_warning.is_none(), "cache write should succeed");
            let cache = rt.dema_cache().expect("cache attached");
            assert!(
                cache.principal_path().exists(),
                "principal.json must exist on disk after permit activation"
            );
        }

        #[test]
        fn activate_reject_does_not_write_cache() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let env = test_envelope(11_000);
            let record = rt.submit_principal_activation(env, 0.40).unwrap();

            assert!(record.rejected);
            let cache = rt.dema_cache().unwrap();
            assert!(
                !cache.principal_path().exists(),
                "rejected activation must not touch disk cache (§10 Proof Law)"
            );
        }

        #[test]
        fn activate_without_cache_skips_disk_write() {
            let mut rt = minimal_runtime();
            // No attach_dema_cache call.
            let env = test_envelope(12_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();
            assert!(!record.rejected);
            assert!(record.cache_warning.is_none());
            assert!(rt.dema_cache().is_none());
            assert!(
                rt.principal_profile().is_some(),
                "in-memory profile still set even without cache attached"
            );
        }

        #[test]
        fn restart_simulation_reloads_profile_from_cache() {
            // G2 success criterion: "principal profile persisted" — verify
            // the profile survives a fresh runtime construction pointing
            // at the same sovereign_state root.
            let td = tempfile::TempDir::new().unwrap();
            let sovereign_root = td.path();

            // 1. First runtime: activate + write cache.
            {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(sovereign_root);
                let env = test_envelope(20_000);
                let record = rt.submit_principal_activation(env, 0.98).unwrap();
                assert!(!record.rejected);
                assert!(record.cache_warning.is_none());
            } // drop — simulate process exit

            // 2. Second runtime (empty chain, fresh state): attach same
            //    cache, rehydrate profile from disk.
            let mut rt2 = minimal_runtime();
            rt2.attach_dema_cache(sovereign_root);
            assert!(
                rt2.principal_profile().is_none(),
                "fresh runtime has no in-memory profile before rehydrate"
            );
            let loaded = rt2.rehydrate_principal_from_cache().unwrap();
            assert!(loaded, "profile must be readable from cache after restart");

            let p = rt2.principal_profile().expect("profile now in memory");
            assert_eq!(p.name, "Mumo");
            assert_eq!(p.node_id, "NODE0");
            assert_eq!(p.declared_role, "node0_principal");
        }

        #[test]
        fn rehydrate_without_cache_attached_returns_false() {
            let mut rt = minimal_runtime();
            let result = rt.rehydrate_principal_from_cache().unwrap();
            assert!(!result, "no cache attached → Ok(false)");
            assert!(rt.principal_profile().is_none());
        }

        #[test]
        fn rehydrate_from_empty_cache_returns_false() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let result = rt.rehydrate_principal_from_cache().unwrap();
            assert!(!result, "cache attached but no file → Ok(false)");
        }

        #[test]
        fn cache_roundtrip_preserves_profile_hash() {
            // Strong invariant: writing then reading through the cache
            // produces a profile whose profile_hash() matches the
            // PrincipalActivationReceipt.principal_profile_hash exactly.
            // This is what G2 binding integrity depends on.
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let env = test_envelope(30_000);
            let record = rt.submit_principal_activation(env, 0.98).unwrap();
            let original_hash = record.profile.as_ref().unwrap().profile_hash();
            let receipt_hash = record
                .activation_receipt
                .as_ref()
                .unwrap()
                .principal_profile_hash;
            assert_eq!(original_hash, receipt_hash);

            let mut rt2 = minimal_runtime();
            rt2.attach_dema_cache(td.path());
            rt2.rehydrate_principal_from_cache().unwrap();
            let reloaded_hash = rt2.principal_profile().unwrap().profile_hash();
            assert_eq!(
                reloaded_hash, original_hash,
                "profile_hash must survive disk round-trip for G2 binding integrity"
            );
        }

        #[test]
        fn activate_principal_profile_id_stable_across_reactivations() {
            let mut rt = minimal_runtime();
            // Same principal identity (Mumo, same node) → stable principal_id
            // even though activation_receipt_id differs because of different
            // timestamps / chain heads.
            let env1 = test_envelope(7_000);
            let r1 = rt.submit_principal_activation(env1, 0.98).unwrap();
            let env2 = test_envelope(8_000);
            let r2 = rt.submit_principal_activation(env2, 0.98).unwrap();

            let p1 = r1.profile.as_ref().unwrap();
            let p2 = r2.profile.as_ref().unwrap();
            assert_eq!(
                p1.principal_id, p2.principal_id,
                "principal_id is stable across re-activations of the same principal"
            );
            assert_ne!(
                p1.activation_receipt_id, p2.activation_receipt_id,
                "activation_receipt_id must differ across distinct mission submissions"
            );
        }
    }

    // ========================================================================
    // Cycle-6 G1 Phase 1 Commit C — from_sovereign_state constructor tests
    // ========================================================================

    mod sovereign_state_bootstrap {
        use super::*;
        use crate::sovereign_state::{chain_entry_hash, hex_digest, GENESIS_PREV_HEX};
        use serde_json::json;
        use std::fs;
        use tempfile::TempDir;

        fn write_minimal_valid_fixture(root: &std::path::Path) {
            let receipts = root.join("receipts");
            fs::create_dir_all(&receipts).unwrap();
            let r = json!({"event": "bootstrap_test", "n": 1});
            fs::write(
                receipts.join("bootstrap_test_2026.json"),
                serde_json::to_vec(&r).unwrap(),
            )
            .unwrap();
            let h = hex_digest(&chain_entry_hash(GENESIS_PREV_HEX, &r).unwrap());
            let env = json!({
                "chain_type": "runtime_test",
                "node_id": "RT-TEST",
                "timestamp": "2026-01-01T00:00:00Z",
                "receipts": 1,
                "chain": [{
                    "file": "bootstrap_test_2026.json",
                    "event": "bootstrap_test",
                    "hash": h,
                    "prev_hash": GENESIS_PREV_HEX
                }],
                "head_hash": h
            });
            fs::write(
                receipts.join("activation_chain_2026-01-01T00:00:00Z.json"),
                serde_json::to_vec(&env).unwrap(),
            )
            .unwrap();
        }

        #[test]
        fn from_sovereign_state_with_valid_fixture_attaches_snapshot() {
            let td = TempDir::new().unwrap();
            write_minimal_valid_fixture(td.path());

            let rt = CognitionRuntime::from_sovereign_state(td.path())
                .expect("valid fixture should bootstrap");

            let snap = rt
                .sovereign_snapshot()
                .expect("snapshot should be attached");
            assert_eq!(snap.envelopes_count(), 1);
            assert_eq!(snap.total_entries(), 1);
            assert_eq!(snap.envelopes[0].entries[0].event, "bootstrap_test");
        }

        #[test]
        fn from_sovereign_state_missing_root_returns_bootstrap_error() {
            let td = TempDir::new().unwrap();
            let missing = td.path().join("does_not_exist");
            match CognitionRuntime::from_sovereign_state(&missing) {
                Err(BootstrapError::SovereignState(_)) => {}
                Ok(_) => panic!("expected error for missing root"),
            }
        }

        #[test]
        fn new_runtime_has_no_sovereign_snapshot() {
            // Regression: the existing `::new()` constructor must continue to
            // produce a runtime with sovereign_snapshot == None (dev mode).
            let genesis: Blake3Hash = [0u8; 32];
            let graph =
                ThoughtGraph::from_parts(HashMap::new(), Vec::new(), HashMap::new(), genesis);
            let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
            let ctx = AgentCtx {
                receipt_chain: genesis,
            };
            let rt = CognitionRuntime::new(graph, chain, ctx);
            assert!(rt.sovereign_snapshot().is_none());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-1 — receipt_history cache integration
    // ════════════════════════════════════════════════════════════════
    mod receipt_history_g3 {
        use super::*;
        use crate::principal_activation::{NodeIdentityAnchor, PrincipalActivationEnvelope};
        use crate::receipt_history_cache::ReceiptHistoryCache;

        const TEST_PUBKEY_HEX: &str =
            "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

        fn test_envelope(now_ns: u64) -> PrincipalActivationEnvelope {
            let a = NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "2026-04-18T06:00:00Z");
            PrincipalActivationEnvelope::from_anchor(
                "Mumo".into(),
                "node0_principal".into(),
                &a,
                now_ns,
            )
            .unwrap()
        }

        #[test]
        fn no_cache_attached_makes_writes_a_noop() {
            let mut rt = minimal_runtime();
            assert!(rt.receipt_history_cache().is_none());
            // Advance the chain via a permitted activation.
            let _ = rt
                .submit_principal_activation(test_envelope(1_000), 0.98)
                .unwrap();
            // No cache → write helper returns None, no panic.
            assert!(rt.write_receipt_history_cache().is_none());
            assert!(rt.rehydrate_receipt_history_from_cache().unwrap().is_none());
        }

        #[test]
        fn attach_dema_cache_initializes_both_surfaces() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.dema_cache().is_some());
            assert!(rt.receipt_history_cache().is_some());
        }

        #[test]
        fn submit_principal_activation_auto_writes_receipt_history_snapshot() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let record = rt
                .submit_principal_activation(test_envelope(1_000), 0.98)
                .unwrap();
            assert!(!record.rejected);

            let loaded = rt
                .rehydrate_receipt_history_from_cache()
                .unwrap()
                .expect("cache written after activation");

            // Head must equal chain head AFTER PrincipalActivation append.
            assert_eq!(loaded.head, rt.chain.head());
            // PA path seals 9 records (boot? minimal_runtime starts empty, so
            // exact count depends on runtime init — we check monotonicity).
            assert_eq!(loaded.records.len(), rt.chain.len());
            // Terminal record must be the PrincipalActivation (kind 0x61).
            let last_kind = loaded.records.last().expect("records not empty").kind;
            assert_eq!(last_kind, ReceiptKind::PrincipalActivation);
        }

        #[test]
        fn rejected_mission_still_writes_current_chain_snapshot() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            // Reject path: submit_mission returns early without advancing
            // the chain, but the trailing write_receipt_history_cache()
            // in submit_mission still fires — producing a valid snapshot
            // of the pre-reject chain head.
            let env = test_mission(500);
            let claim = reject_claim(&env, 1_000);
            let record = rt.submit_mission(env, claim).unwrap();
            assert!(record.rejected);

            let loaded = rt.rehydrate_receipt_history_from_cache().unwrap().unwrap();
            assert_eq!(loaded.head, rt.chain.head());
        }

        #[test]
        fn restart_simulation_snapshot_survives_and_matches_chain() {
            // 1. Runtime A: attach cache, activate principal, drop.
            // 2. Runtime B: attach cache at same root, read snapshot.
            // 3. Loaded head + records must match A's final chain state.
            let td = tempfile::TempDir::new().unwrap();

            let (head_a, len_a) = {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(td.path());
                rt.submit_principal_activation(test_envelope(42_000), 0.97)
                    .unwrap();
                (rt.chain.head(), rt.chain.len())
            };

            let cache = ReceiptHistoryCache::at_sovereign_root(td.path());
            let loaded = cache.read().unwrap().expect("snapshot on disk");
            assert_eq!(loaded.head, head_a);
            assert_eq!(loaded.records.len(), len_a);
            assert_eq!(
                loaded.records.last().map(|r| r.kind),
                Some(ReceiptKind::PrincipalActivation)
            );
        }

        #[test]
        fn write_receipt_history_cache_returns_ok_when_attached() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.submit_principal_activation(test_envelope(1_000), 0.98)
                .unwrap();
            // Explicit helper call round-trips Ok(Some(Ok(()))).
            let result = rt.write_receipt_history_cache().expect("cache attached");
            result.expect("write succeeds");
        }

        #[test]
        fn successive_activations_refresh_snapshot_head() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let r1 = rt
                .submit_principal_activation(test_envelope(1_000), 0.98)
                .unwrap();
            let head_after_first = rt.chain.head();
            let snap1 = rt.rehydrate_receipt_history_from_cache().unwrap().unwrap();
            assert_eq!(snap1.head, head_after_first);
            assert!(!r1.rejected);

            // A subsequent submit_mission must refresh the cache head.
            let env = test_mission(2_000);
            let claim = permit_claim(&env, 3_000);
            let _ = rt.submit_mission(env, claim).unwrap();
            let head_after_second = rt.chain.head();
            let snap2 = rt.rehydrate_receipt_history_from_cache().unwrap().unwrap();
            assert_eq!(snap2.head, head_after_second);
            assert!(snap2.records.len() > snap1.records.len());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-2 — manifest_history cache integration
    // ════════════════════════════════════════════════════════════════
    mod manifest_history_g3 {
        use super::*;
        use crate::manifest_history_cache::ManifestHistoryCache;
        use crate::principal_activation::{NodeIdentityAnchor, PrincipalActivationEnvelope};

        const TEST_PUBKEY_HEX: &str =
            "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

        fn test_envelope(now_ns: u64) -> PrincipalActivationEnvelope {
            let a = NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "2026-04-18T06:00:00Z");
            PrincipalActivationEnvelope::from_anchor(
                "Mumo".into(),
                "node0_principal".into(),
                &a,
                now_ns,
            )
            .unwrap()
        }

        #[test]
        fn empty_runtime_snapshot_has_no_manifests() {
            let rt = minimal_runtime();
            let snap = rt.manifest_history_snapshot();
            assert!(snap.manifests.is_empty());
            assert_eq!(snap.chain_head, rt.chain.head());
        }

        #[test]
        fn attach_dema_cache_also_initializes_manifest_history() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.manifest_history_cache().is_some());
        }

        #[test]
        fn permitted_mission_appends_manifest_to_cache() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let env = test_mission(100);
            let claim = permit_claim(&env, 200);
            let record = rt.submit_mission(env, claim).unwrap();
            assert!(!record.rejected);
            assert!(record.manifest.is_some(), "permit path emits manifest");

            let loaded = rt
                .rehydrate_manifest_history_from_cache()
                .unwrap()
                .expect("cache written after submit");
            assert_eq!(loaded.chain_head, rt.chain.head());
            assert_eq!(loaded.manifests.len(), 1);
            let expected_id = record.manifest.as_ref().unwrap().manifest_id;
            assert_eq!(loaded.manifests[0].manifest_id, expected_id);
        }

        #[test]
        fn rejected_mission_does_not_add_to_manifest_history() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let env = test_mission(100);
            let claim = reject_claim(&env, 200);
            let record = rt.submit_mission(env, claim).unwrap();
            assert!(record.rejected);
            assert!(record.manifest.is_none(), "§10: no manifest on reject");

            // Cache is written on reject path but must contain zero
            // manifests — the missions registry holds the rejected
            // record with manifest=None, which the snapshot filters.
            let loaded = rt.rehydrate_manifest_history_from_cache().unwrap().unwrap();
            assert!(loaded.manifests.is_empty());
        }

        #[test]
        fn principal_activation_populates_manifest_history() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let record = rt
                .submit_principal_activation(test_envelope(1_000), 0.98)
                .unwrap();
            assert!(!record.rejected);

            let loaded = rt
                .rehydrate_manifest_history_from_cache()
                .unwrap()
                .expect("cache written after activation");
            assert_eq!(loaded.manifests.len(), 1);
            // The activation's inner mission record must have a manifest
            // that matches the one surfaced via the cache.
            let expected_id = record
                .mission_record
                .manifest
                .as_ref()
                .expect("activation permit produces manifest")
                .manifest_id;
            assert_eq!(loaded.manifests[0].manifest_id, expected_id);
        }

        #[test]
        fn multiple_missions_appear_in_deterministic_order() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            // Submit two permitted missions with distinct window_start
            // (timestamp_ns) so the sort key is exercised.
            let e1 = test_mission(1_000);
            let c1 = permit_claim(&e1, 1_100);
            let r1 = rt.submit_mission(e1, c1).unwrap();

            let e2 = test_mission(2_000);
            let c2 = permit_claim(&e2, 2_100);
            let r2 = rt.submit_mission(e2, c2).unwrap();

            let loaded = rt.rehydrate_manifest_history_from_cache().unwrap().unwrap();
            assert_eq!(loaded.manifests.len(), 2);
            // window_start-ascending order.
            assert!(
                loaded.manifests[0].window_start < loaded.manifests[1].window_start,
                "manifests must be sorted by window_start asc"
            );
            // Identity membership.
            let ids: std::collections::HashSet<_> =
                loaded.manifests.iter().map(|m| m.manifest_id).collect();
            assert!(ids.contains(&r1.manifest.unwrap().manifest_id));
            assert!(ids.contains(&r2.manifest.unwrap().manifest_id));
        }

        #[test]
        fn restart_survival_reloads_manifest_history() {
            let td = tempfile::TempDir::new().unwrap();
            let expected_len;
            let expected_id;
            {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(td.path());
                let env = test_mission(42);
                let claim = permit_claim(&env, 84);
                let rec = rt.submit_mission(env, claim).unwrap();
                expected_id = rec.manifest.unwrap().manifest_id;
                expected_len = 1;
            }
            let cache = ManifestHistoryCache::at_sovereign_root(td.path());
            let loaded = cache.read().unwrap().expect("snapshot on disk");
            assert_eq!(loaded.manifests.len(), expected_len);
            assert_eq!(loaded.manifests[0].manifest_id, expected_id);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-3 — mission_log cache integration
    // ════════════════════════════════════════════════════════════════
    mod mission_log_g3 {
        use super::*;
        use crate::mission_log_cache::MissionLogCache;
        use crate::principal_activation::{NodeIdentityAnchor, PrincipalActivationEnvelope};

        const TEST_PUBKEY_HEX: &str =
            "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

        fn test_envelope(now_ns: u64) -> PrincipalActivationEnvelope {
            let a = NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "2026-04-18T06:00:00Z");
            PrincipalActivationEnvelope::from_anchor(
                "Mumo".into(),
                "node0_principal".into(),
                &a,
                now_ns,
            )
            .unwrap()
        }

        #[test]
        fn attach_initializes_mission_log_cache() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.mission_log_cache().is_some());
        }

        #[test]
        fn empty_runtime_has_empty_mission_log_snapshot() {
            let rt = minimal_runtime();
            let snap = rt.mission_log_snapshot();
            assert!(snap.entries.is_empty());
            assert_eq!(snap.chain_head, rt.chain.head());
        }

        #[test]
        fn permitted_mission_appears_in_log_without_remediation() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let env = test_mission(100);
            let claim = permit_claim(&env, 200);
            let record = rt.submit_mission(env, claim).unwrap();

            let loaded = rt
                .rehydrate_mission_log_from_cache()
                .unwrap()
                .expect("cache written");
            assert_eq!(loaded.entries.len(), 1);
            let e = &loaded.entries[0];
            assert!(!e.rejected);
            assert_eq!(e.mission_id, record.envelope.mission_id);
            assert_eq!(e.receipt_id, record.receipt_id);
            assert!(e.remediation.is_none());
            assert!(e.quality_score >= 0.95);
        }

        #[test]
        fn rejected_mission_appears_with_structured_remediation() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let env = test_mission(100);
            let claim = reject_claim(&env, 200);
            let record = rt.submit_mission(env, claim).unwrap();
            assert!(record.rejected);

            let loaded = rt.rehydrate_mission_log_from_cache().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 1);
            let e = &loaded.entries[0];
            assert!(e.rejected);
            assert_eq!(e.receipt_id, None);
            let remediation = e.remediation.as_ref().expect("remediation present");
            assert!(
                remediation.contains("REJECTED"),
                "remediation must name rejection honestly: {}",
                remediation
            );
        }

        #[test]
        fn mixed_stream_sorted_by_timestamp_ns() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            // Submit reject first (earlier), permit second (later).
            let e1 = test_mission(1_000);
            let c1 = reject_claim(&e1, 1_100);
            rt.submit_mission(e1, c1).unwrap();

            let e2 = test_mission(2_000);
            let c2 = permit_claim(&e2, 2_100);
            rt.submit_mission(e2, c2).unwrap();

            let loaded = rt.rehydrate_mission_log_from_cache().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 2);
            assert!(
                loaded.entries[0].timestamp_ns < loaded.entries[1].timestamp_ns,
                "log must sort ascending by timestamp"
            );
            assert!(loaded.entries[0].rejected);
            assert!(!loaded.entries[1].rejected);
        }

        #[test]
        fn principal_activation_appears_in_mission_log() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            rt.submit_principal_activation(test_envelope(1_000), 0.98)
                .unwrap();

            let loaded = rt.rehydrate_mission_log_from_cache().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 1);
            let e = &loaded.entries[0];
            assert!(!e.rejected);
            // intent_text from PrincipalActivationEnvelope should be the
            // canonical activation intent string.
            assert!(
                e.intent_text.contains("activate"),
                "activation intent text: {}",
                e.intent_text
            );
            assert!(e.receipt_id.is_some());
        }

        #[test]
        fn restart_survival_reloads_mission_log() {
            let td = tempfile::TempDir::new().unwrap();
            let expected_mission_id;
            {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(td.path());
                let env = test_mission(7_000);
                let claim = permit_claim(&env, 7_100);
                let rec = rt.submit_mission(env, claim).unwrap();
                expected_mission_id = rec.envelope.mission_id;
            }
            let cache = MissionLogCache::at_sovereign_root(td.path());
            let loaded = cache.read().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 1);
            assert_eq!(loaded.entries[0].mission_id, expected_mission_id);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-4 — state_snapshots cache integration
    // ════════════════════════════════════════════════════════════════
    mod state_snapshots_g3 {
        use super::*;
        use crate::state_snapshots_cache::StateSnapshotsCache;

        #[test]
        fn attach_initializes_state_snapshots_cache() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.state_snapshots_cache().is_some());
        }

        #[test]
        fn empty_runtime_has_empty_state_snapshots() {
            let rt = minimal_runtime();
            let snap = rt.state_snapshots_snapshot();
            assert!(snap.entries.is_empty());
        }

        #[test]
        fn permitted_mission_captures_current_and_ideal_states() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let env = test_mission(100);
            let claim = permit_claim(&env, 200);
            let record = rt.submit_mission(env, claim).unwrap();

            let loaded = rt
                .rehydrate_state_snapshots_from_cache()
                .unwrap()
                .expect("cache written");
            assert_eq!(loaded.entries.len(), 1);
            let e = &loaded.entries[0];
            assert_eq!(e.mission_id, record.envelope.mission_id);
            assert!(!e.rejected);
            // minimal_runtime's test_mission uses current_state/ideal_state
            // from the fixtures, so hashes + summaries must round-trip.
            assert_eq!(e.current.hash, current_state().hash);
            assert_eq!(e.ideal.hash, ideal_state().hash);
            assert_eq!(e.current.summary, current_state().summary);
            assert_eq!(e.ideal.summary, ideal_state().summary);
        }

        #[test]
        fn rejected_mission_still_records_state_snapshot() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let env = test_mission(100);
            let claim = reject_claim(&env, 200);
            let record = rt.submit_mission(env, claim).unwrap();
            assert!(record.rejected);

            let loaded = rt.rehydrate_state_snapshots_from_cache().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 1);
            let e = &loaded.entries[0];
            assert!(e.rejected, "rejected attempts are preserved in state log");
            // State snapshot must still carry the gap Dema perceived at
            // submission time.
            assert!(e.gap >= 0.0);
        }

        #[test]
        fn multiple_attempts_sorted_by_timestamp() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            let e1 = test_mission(1_000);
            let c1 = reject_claim(&e1, 1_100);
            rt.submit_mission(e1, c1).unwrap();

            let e2 = test_mission(2_000);
            let c2 = permit_claim(&e2, 2_100);
            rt.submit_mission(e2, c2).unwrap();

            let loaded = rt.rehydrate_state_snapshots_from_cache().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 2);
            assert!(loaded.entries[0].timestamp_ns < loaded.entries[1].timestamp_ns);
        }

        #[test]
        fn restart_survival() {
            let td = tempfile::TempDir::new().unwrap();
            let expected_id;
            {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(td.path());
                let env = test_mission(42);
                let claim = permit_claim(&env, 84);
                let rec = rt.submit_mission(env, claim).unwrap();
                expected_id = rec.envelope.mission_id;
            }
            let cache = StateSnapshotsCache::at_sovereign_root(td.path());
            let loaded = cache.read().unwrap().unwrap();
            assert_eq!(loaded.entries.len(), 1);
            assert_eq!(loaded.entries[0].mission_id, expected_id);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G3 Commit-5 — resource_registry seed integration
    // ════════════════════════════════════════════════════════════════
    mod resource_registry_g3 {
        use super::*;
        use crate::resource_registry_cache::ResourceRegistryCache;

        #[test]
        fn attach_initializes_resource_registry_cache() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.resource_registry_cache().is_some());
        }

        #[test]
        fn seed_on_boot_creates_empty_registry() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let seeded = rt.seed_resource_registry_if_missing().unwrap();
            assert_eq!(seeded, Some(true), "new seed should return Some(true)");
            let loaded = rt.rehydrate_resource_registry_from_cache().unwrap();
            assert!(loaded.is_some());
            assert!(loaded.unwrap().resources.is_empty());
        }

        #[test]
        fn seed_is_idempotent_and_does_not_overwrite() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.seed_resource_registry_if_missing().unwrap();
            // Directly write a non-empty registry simulating a G4 mutation.
            let cache = rt.resource_registry_cache().unwrap().clone();
            use crate::resource_registry_cache::{ResourceEntry, ResourceRegistrySnapshot};
            let populated = ResourceRegistrySnapshot {
                resources: vec![ResourceEntry {
                    id: "/home/mumo/docs".into(),
                    kind: "filesystem".into(),
                    summary: "mumo's docs".into(),
                    allowlisted: true,
                }],
            };
            cache.write(&populated).unwrap();
            // Re-seeding must not clobber.
            let seeded_again = rt.seed_resource_registry_if_missing().unwrap();
            assert_eq!(seeded_again, Some(false));
            let loaded = rt
                .rehydrate_resource_registry_from_cache()
                .unwrap()
                .unwrap();
            assert_eq!(loaded, populated);
        }

        #[test]
        fn seed_without_attached_cache_returns_none() {
            let rt = minimal_runtime();
            let seeded = rt.seed_resource_registry_if_missing().unwrap();
            assert_eq!(seeded, None);
        }

        #[test]
        fn rehydrate_on_fresh_runtime_reads_seeded_file() {
            let td = tempfile::TempDir::new().unwrap();
            {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(td.path());
                rt.seed_resource_registry_if_missing().unwrap();
            }
            let cache = ResourceRegistryCache::at_sovereign_root(td.path());
            let loaded = cache.read().unwrap().unwrap();
            assert!(loaded.resources.is_empty());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G4 Commit-1 — typed resource registry API
    // ════════════════════════════════════════════════════════════════
    mod resource_registry_g4 {
        use super::*;
        use crate::resource_registry::{RegisterOutcome, ResourceKind, TypedResource};

        fn fs_resource(id: &str, allowlisted: bool) -> TypedResource {
            TypedResource::new(
                ResourceKind::FilesystemPath,
                id.into(),
                format!("summary for {}", id),
                allowlisted,
            )
            .unwrap()
        }

        #[test]
        fn register_on_unattached_runtime_errors() {
            let rt = minimal_runtime();
            let err = rt.register_resource(fs_resource("/a", true)).unwrap_err();
            assert!(matches!(
                err,
                crate::resource_registry::ResourceRegistryError::NoCacheAttached
            ));
        }

        #[test]
        fn register_new_returns_created_and_persists() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let out = rt.register_resource(fs_resource("/docs", true)).unwrap();
            assert_eq!(out, RegisterOutcome::Created);
            let loaded = rt.list_resources().unwrap();
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].id, "/docs");
            assert!(loaded[0].allowlisted);
        }

        #[test]
        fn register_same_twice_is_idempotent() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let first = rt.register_resource(fs_resource("/docs", true)).unwrap();
            assert_eq!(first, RegisterOutcome::Created);
            let again = rt.register_resource(fs_resource("/docs", true)).unwrap();
            assert_eq!(again, RegisterOutcome::Idempotent);
            assert_eq!(rt.list_resources().unwrap().len(), 1);
        }

        #[test]
        fn register_updates_allowlist_flag_on_existing_id() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.register_resource(fs_resource("/docs", false)).unwrap();
            let out = rt.register_resource(fs_resource("/docs", true)).unwrap();
            assert_eq!(out, RegisterOutcome::Updated);
            let loaded = rt.list_resources().unwrap();
            assert_eq!(loaded.len(), 1, "update not duplicate");
            assert!(loaded[0].allowlisted);
        }

        #[test]
        fn is_allowlisted_true_only_when_registered_and_allowed() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.register_resource(fs_resource("/allowed", true)).unwrap();
            rt.register_resource(fs_resource("/denied", false)).unwrap();

            assert!(rt
                .is_allowlisted(&ResourceKind::FilesystemPath, "/allowed")
                .unwrap());
            assert!(!rt
                .is_allowlisted(&ResourceKind::FilesystemPath, "/denied")
                .unwrap());
            assert!(!rt
                .is_allowlisted(&ResourceKind::FilesystemPath, "/never-registered")
                .unwrap());
        }

        #[test]
        fn is_allowlisted_is_kind_sensitive() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    "example-id".into(),
                    "".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();
            // Same id, different kind — not allowlisted.
            assert!(!rt
                .is_allowlisted(&ResourceKind::NetworkEndpoint, "example-id")
                .unwrap());
            assert!(rt
                .is_allowlisted(&ResourceKind::FilesystemPath, "example-id")
                .unwrap());
        }

        #[test]
        fn list_returns_deterministic_order() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.register_resource(fs_resource("/zz", true)).unwrap();
            rt.register_resource(fs_resource("/aa", true)).unwrap();
            rt.register_resource(fs_resource("/mm", false)).unwrap();
            let ids: Vec<_> = rt
                .list_resources()
                .unwrap()
                .into_iter()
                .map(|r| r.id)
                .collect();
            assert_eq!(ids, vec!["/aa".to_string(), "/mm".into(), "/zz".into()]);
        }

        #[test]
        fn custom_kind_registers_and_lists_correctly() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let r = TypedResource::new(
                ResourceKind::Custom("scroll".into()),
                "tablet-01".into(),
                "ancient tablet".into(),
                false,
            )
            .unwrap();
            rt.register_resource(r).unwrap();
            let loaded = rt.list_resources().unwrap();
            assert_eq!(loaded.len(), 1);
            assert_eq!(loaded[0].kind, ResourceKind::Custom("scroll".into()));
        }

        #[test]
        fn list_on_unattached_runtime_returns_empty() {
            let rt = minimal_runtime();
            assert!(rt.list_resources().unwrap().is_empty());
            assert!(!rt
                .is_allowlisted(&ResourceKind::FilesystemPath, "/anything")
                .unwrap());
        }

        // ─── URP view via runtime ────────────────────────────────

        #[test]
        fn urp_view_on_unattached_runtime_is_empty() {
            let rt = minimal_runtime();
            let v = rt.urp_view().unwrap();
            assert!(v.buckets.is_empty());
            assert_eq!(v.total_count, 0);
        }

        #[test]
        fn urp_view_reflects_registered_resources() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.register_resource(fs_resource("/a", true)).unwrap();
            rt.register_resource(fs_resource("/b", false)).unwrap();
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::NetworkEndpoint,
                    "host:80".into(),
                    "web".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();

            let v = rt.urp_view().unwrap();
            assert_eq!(v.total_count, 3);
            assert_eq!(v.allowlisted_count, 2);
            assert_eq!(v.buckets.len(), 2);
            let fs = v.bucket(&ResourceKind::FilesystemPath).unwrap();
            assert_eq!(fs.resources.len(), 2);
            let net = v.bucket(&ResourceKind::NetworkEndpoint).unwrap();
            assert_eq!(net.resources.len(), 1);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G5 Commit-2 — submit_organize_mission integration
    // ════════════════════════════════════════════════════════════════
    mod organize_mission_g5 {
        use super::*;
        use crate::organize_mission::OrganizeMissionReceipt;
        use crate::receipts::ReceiptPayloadDecode;
        use crate::resource_registry::{ResourceKind, TypedResource};
        use std::fs;

        fn write_fixture_dir(root: &std::path::Path) {
            fs::create_dir_all(root).unwrap();
            fs::write(root.join("alpha.txt"), b"hello").unwrap();
            fs::write(root.join("beta.txt"), b"world").unwrap();
            fs::create_dir_all(root.join("subdir")).unwrap();
        }

        /// Target directory distinct from the dema_cache root so the
        /// organize listing does not accidentally include the cache dir.
        fn target_subdir(td: &tempfile::TempDir) -> std::path::PathBuf {
            td.path().join("target")
        }

        fn allowlist(rt: &mut CognitionRuntime, path: &std::path::Path) {
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    path.to_string_lossy().into_owned(),
                    "test dir".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();
        }

        #[test]
        fn non_allowlisted_path_refused_without_chain_mutation() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            let chain_before = rt.chain.head();
            let len_before = rt.chain.len();

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            match outcome {
                OrganizeOutcome::NotAllowlisted { path, remediation } => {
                    assert_eq!(path, target_subdir(&td).to_string_lossy().into_owned());
                    assert!(remediation.contains("register-resource"));
                }
                other => panic!("expected NotAllowlisted, got {:?}", other),
            }
            assert_eq!(rt.chain.head(), chain_before, "chain must not advance");
            assert_eq!(rt.chain.len(), len_before);
        }

        #[test]
        fn registered_but_not_allowlisted_is_still_refused() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            // Register WITHOUT allowlist.
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    target_subdir(&td).to_string_lossy().into_owned(),
                    "seen".into(),
                    false,
                )
                .unwrap(),
            )
            .unwrap();

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            assert!(matches!(outcome, OrganizeOutcome::NotAllowlisted { .. }));
        }

        #[test]
        fn io_error_on_missing_path_refused_without_chain_mutation() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            // Allowlist a path that does NOT exist on disk.
            let fake = td.path().join("does-not-exist");
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    fake.to_string_lossy().into_owned(),
                    "ghost".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();
            let chain_before = rt.chain.head();

            let outcome = rt.submit_organize_mission(&fake, 0.98).unwrap();
            assert!(matches!(outcome, OrganizeOutcome::IoError { .. }));
            assert_eq!(rt.chain.head(), chain_before);
        }

        #[test]
        fn allowlisted_path_executes_and_seals_mission_executed_receipt() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            match outcome {
                OrganizeOutcome::Executed {
                    mission_record,
                    organize_receipt,
                    listing,
                } => {
                    assert!(!mission_record.rejected);
                    assert_eq!(
                        organize_receipt.mission_receipt_ref,
                        mission_record.receipt_id.unwrap()
                    );
                    assert_eq!(organize_receipt.listing_digest, listing.digest());
                    assert_eq!(organize_receipt.file_count, 2);
                    assert_eq!(organize_receipt.dir_count, 1);
                    assert_eq!(organize_receipt.entry_count, 3);
                    // Chain head must equal the MissionExecuted receipt id.
                    assert_eq!(rt.chain.head(), organize_receipt.receipt_id);
                }
                other => panic!("expected Executed, got {:?}", other),
            }
        }

        #[test]
        fn executed_receipt_replays_byte_exact_via_fetch_and_decode() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            let id = match outcome {
                OrganizeOutcome::Executed {
                    organize_receipt, ..
                } => organize_receipt.receipt_id,
                other => panic!("expected Executed, got {:?}", other),
            };
            // Fetch + decode round-trip from the payload store.
            let bytes = rt
                .chain
                .fetch_payload_bytes(&id)
                .unwrap()
                .expect("payload stored");
            let decoded = OrganizeMissionReceipt::from_canonical_bytes(&bytes).unwrap();
            assert_eq!(decoded.receipt_id, id);
        }

        #[test]
        fn rejected_admissibility_leaves_chain_unchanged() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));

            // quality below IHSAN_FLOOR -> reject
            let chain_before_len = rt.chain.len();
            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.40)
                .unwrap();
            match outcome {
                OrganizeOutcome::Rejected {
                    mission_record,
                    remediation,
                } => {
                    assert!(mission_record.rejected);
                    assert!(remediation.contains("REJECTED"));
                }
                other => panic!("expected Rejected, got {:?}", other),
            }
            // §10 Proof Law: rejected missions produce no chain artifacts.
            assert_eq!(rt.chain.len(), chain_before_len);
        }

        #[test]
        fn executed_outcome_advances_chain_by_ten_records() {
            // Starting from 0 records, a permitted organize mission
            // seals: envelope + 5 gate verdicts + NodeLifecycle +
            // Manifest + MissionExecuted = 9 records. Plus whatever
            // prior boot/init pushed. The minimal_runtime starts
            // empty so 9 is the exact count.
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));

            let before = rt.chain.len();
            rt.submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            let after = rt.chain.len();
            assert_eq!(after - before, 9, "expected +9 records for permit path");
        }

        #[test]
        fn organize_twice_same_path_produces_distinct_receipt_ids() {
            // Same listing but different timestamps → different receipts.
            // Replayability is about decode-round-trip, not about
            // idempotence at the chain level.
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));

            let id1 = match rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap()
            {
                OrganizeOutcome::Executed {
                    organize_receipt, ..
                } => organize_receipt.receipt_id,
                _ => unreachable!(),
            };
            let id2 = match rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap()
            {
                OrganizeOutcome::Executed {
                    organize_receipt, ..
                } => organize_receipt.receipt_id,
                _ => unreachable!(),
            };
            assert_ne!(id1, id2);
        }

        #[test]
        fn organize_after_principal_activation_threads_principal_id() {
            // When a principal is activated, the organize mission's
            // Originator::session_id should be the principal_id.
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));

            // Activate first.
            let env = crate::principal_activation::PrincipalActivationEnvelope::from_anchor(
                "Mumo".into(),
                "node0_principal".into(),
                &crate::principal_activation::NodeIdentityAnchor::for_test(
                    "NODE0",
                    "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a",
                    "2026-04-18T07:00:00Z",
                ),
                1_000,
            )
            .unwrap();
            rt.submit_principal_activation(env, 0.98).unwrap();
            let principal_id = rt.principal_profile().unwrap().principal_id;

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            match outcome {
                OrganizeOutcome::Executed { mission_record, .. } => {
                    match mission_record.envelope.originator {
                        Originator::Operator { session_id } => {
                            assert_eq!(session_id, principal_id);
                        }
                        other => panic!("expected Operator, got {:?}", other),
                    }
                }
                other => panic!("expected Executed, got {:?}", other),
            }
        }

        #[test]
        fn organize_empty_directory_still_executes() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            fs::create_dir_all(target_subdir(&td)).unwrap();
            allowlist(&mut rt, &target_subdir(&td));

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            match outcome {
                OrganizeOutcome::Executed {
                    organize_receipt, ..
                } => {
                    assert_eq!(organize_receipt.entry_count, 0);
                    assert_eq!(organize_receipt.file_count, 0);
                    assert_eq!(organize_receipt.dir_count, 0);
                }
                other => panic!("expected Executed for empty dir, got {:?}", other),
            }
        }

        #[test]
        fn organize_outcome_is_executed_helper() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            allowlist(&mut rt, &target_subdir(&td));
            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            assert!(outcome.is_executed());
            let refused = rt
                .submit_organize_mission(&td.path().join("not-registered"), 0.98)
                .unwrap();
            assert!(!refused.is_executed());
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Cycle-7 G6 — PoI ledger integration
    // ════════════════════════════════════════════════════════════════
    mod poi_ledger_g6 {
        use super::*;
        use crate::poi_ledger::PoiLedgerCache;
        use crate::resource_registry::{ResourceKind, TypedResource};
        use std::fs;

        const TEST_PUBKEY_HEX: &str =
            "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

        fn activation_envelope(
            now_ns: u64,
        ) -> crate::principal_activation::PrincipalActivationEnvelope {
            crate::principal_activation::PrincipalActivationEnvelope::from_anchor(
                "Mumo".into(),
                "node0_principal".into(),
                &crate::principal_activation::NodeIdentityAnchor::for_test(
                    "NODE0",
                    TEST_PUBKEY_HEX,
                    "2026-04-18T07:00:00Z",
                ),
                now_ns,
            )
            .unwrap()
        }

        fn write_fixture_dir(root: &std::path::Path) {
            fs::create_dir_all(root).unwrap();
            fs::write(root.join("alpha.txt"), b"hello").unwrap();
            fs::write(root.join("beta.txt"), b"world").unwrap();
            fs::create_dir_all(root.join("subdir")).unwrap();
        }

        fn target_subdir(td: &tempfile::TempDir) -> std::path::PathBuf {
            td.path().join("target")
        }

        #[test]
        fn attach_initializes_poi_ledger_cache() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.poi_ledger_cache().is_some());
            assert!(rt.poi_entries().is_empty());
        }

        #[test]
        fn empty_runtime_has_empty_snapshot() {
            let rt = minimal_runtime();
            let snap = rt.poi_ledger_snapshot();
            assert!(snap.entries.is_empty());
        }

        #[test]
        fn activation_appends_one_poi_entry() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let rec = rt
                .submit_principal_activation(activation_envelope(1000), 0.98)
                .unwrap();
            assert!(!rec.rejected);

            assert_eq!(rt.poi_entries().len(), 1);
            let e = &rt.poi_entries()[0];
            assert_eq!(e.receipt_kind_byte, 0x61);
            assert_eq!(e.entry_count, 0);
            assert!(e.impact_score > 0.0 && e.impact_score <= 1.0);
            assert_eq!(e.quality_score, 0.98);
        }

        #[test]
        fn organize_appends_one_poi_entry_with_entry_count() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    target_subdir(&td).to_string_lossy().into_owned(),
                    "".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();

            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            assert!(outcome.is_executed());

            assert_eq!(rt.poi_entries().len(), 1);
            let e = &rt.poi_entries()[0];
            assert_eq!(e.receipt_kind_byte, 0x70);
            assert_eq!(e.entry_count, 3, "3 top-level entries in fixture");
        }

        #[test]
        fn rejected_mission_does_not_produce_poi_entry() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    target_subdir(&td).to_string_lossy().into_owned(),
                    "".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();
            // quality below IHSAN_FLOOR
            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.40)
                .unwrap();
            assert!(!outcome.is_executed());
            assert!(
                rt.poi_entries().is_empty(),
                "§10 Proof Law: no chain, no ledger"
            );
        }

        #[test]
        fn non_allowlisted_organize_produces_no_entry() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));
            // NOT allowlisted
            let outcome = rt
                .submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();
            assert!(matches!(outcome, OrganizeOutcome::NotAllowlisted { .. }));
            assert!(rt.poi_entries().is_empty());
        }

        #[test]
        fn activation_then_organize_stacks_two_entries() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            write_fixture_dir(&target_subdir(&td));

            rt.submit_principal_activation(activation_envelope(1000), 0.98)
                .unwrap();
            rt.register_resource(
                TypedResource::new(
                    ResourceKind::FilesystemPath,
                    target_subdir(&td).to_string_lossy().into_owned(),
                    "".into(),
                    true,
                )
                .unwrap(),
            )
            .unwrap();
            rt.submit_organize_mission(&target_subdir(&td), 0.98)
                .unwrap();

            assert_eq!(rt.poi_entries().len(), 2);
            assert_eq!(rt.poi_entries()[0].receipt_kind_byte, 0x61);
            assert_eq!(rt.poi_entries()[1].receipt_kind_byte, 0x70);
            // organize entry should carry the activated principal_id
            let pid = rt.principal_profile().unwrap().principal_id;
            assert_eq!(rt.poi_entries()[1].principal_id, Some(pid));
        }

        #[test]
        fn poi_cache_is_written_after_each_permit() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            rt.submit_principal_activation(activation_envelope(1000), 0.98)
                .unwrap();

            let cache = PoiLedgerCache::at_sovereign_root(td.path());
            let loaded = cache.read().unwrap().expect("cache written after permit");
            assert_eq!(loaded.entries.len(), 1);
            assert_eq!(loaded.entries[0].receipt_kind_byte, 0x61);
        }

        #[test]
        fn load_poi_entries_from_cache_restores_session_state() {
            let td = tempfile::TempDir::new().unwrap();
            // Session A: build entries.
            {
                let mut rt = minimal_runtime();
                rt.attach_dema_cache(td.path());
                rt.submit_principal_activation(activation_envelope(1000), 0.98)
                    .unwrap();
            }
            // Session B: fresh runtime, load from cache.
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            assert!(rt.poi_entries().is_empty());
            let loaded = rt.load_poi_entries_from_cache().unwrap();
            assert!(loaded);
            assert_eq!(rt.poi_entries().len(), 1);
            assert_eq!(rt.poi_entries()[0].receipt_kind_byte, 0x61);
        }

        #[test]
        fn load_from_cache_returns_false_when_absent() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());
            let loaded = rt.load_poi_entries_from_cache().unwrap();
            assert!(!loaded);
        }

        #[test]
        fn impact_score_higher_for_larger_listing() {
            let td = tempfile::TempDir::new().unwrap();
            let mut rt = minimal_runtime();
            rt.attach_dema_cache(td.path());

            // small target: 1 file
            let small = td.path().join("small");
            fs::create_dir_all(&small).unwrap();
            fs::write(small.join("a.txt"), b"a").unwrap();

            // big target: 10 files
            let big = td.path().join("big");
            fs::create_dir_all(&big).unwrap();
            for i in 0..10 {
                fs::write(big.join(format!("f{}.txt", i)), b"x").unwrap();
            }

            for p in [&small, &big] {
                rt.register_resource(
                    TypedResource::new(
                        ResourceKind::FilesystemPath,
                        p.to_string_lossy().into_owned(),
                        "".into(),
                        true,
                    )
                    .unwrap(),
                )
                .unwrap();
            }

            rt.submit_organize_mission(&small, 0.98).unwrap();
            rt.submit_organize_mission(&big, 0.98).unwrap();

            let s_score = rt.poi_entries()[0].impact_score;
            let b_score = rt.poi_entries()[1].impact_score;
            assert!(b_score > s_score, "bigger listing -> higher impact");
        }
    }
}
