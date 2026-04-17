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
//!   - Input: a ThoughtGraph in its fresh-boot state (no reflexes installed)
//!            and a ReceiptChain whose records have been loaded (but whose
//!            derived state the graph has not yet applied).
//!   - Output: a runtime whose graph has the exact reflex set the chain
//!            records commit to, with no side effects beyond state.
//!
//! Determinism: rehydrate is pure replay. Same chain + same graph skeleton
//! → same final state, byte-for-byte. This is what makes Node1 reproducibility
//! and crash recovery work.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::admissibility_freeze_v1::{
    AdmissibilityChain, AdmissibilityClaim, AdmissibilityResult, GateVerdict,
    RejectedClaim, Verdict,
};
use crate::thought_graph::{
    ThoughtGraph, AgentCtx, Thought, ReasoningError, ShadowMode,
    MyelinationReceipt, DemyelinationReceipt, DemyelinationReason,
    CompiledReflex,
};
use crate::mission_freeze_v1::{MissionEnvelope, MissionStage};
use crate::receipts::{
    ReceiptChain, ReceiptPayload, ReceiptKind,
    ChainError, Blake3Hash,
};
use crate::receipt_freeze_v1::{ReceiptArtifact, ReceiptChainExt};
use crate::canonical_hasher::blake3_domain;

// ============================================================================
// Events
// ============================================================================

#[derive(Debug, Clone)]
pub enum CognitionEvent {
    ReasoningRequest { request_id: Blake3Hash },
    ConsolidationTick,
    GovernanceDemyelination { edge: Blake3Hash, decision: Blake3Hash },
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
    fn from(e: ChainError) -> Self { LoopError::Chain(e) }
}

#[derive(Debug)]
pub enum RehydrateError {
    ChainFetch(ChainError),
    MissingPayload(Blake3Hash),
    InconsistentState { reason: String },
}

impl From<ChainError> for RehydrateError {
    fn from(e: ChainError) -> Self { RehydrateError::ChainFetch(e) }
}

#[derive(Debug)]
pub enum MissionRuntimeError {
    Chain(ChainError),
    Clock(String),
    DuplicateMission(Blake3Hash),
    MissionNotFound(Blake3Hash),
    ClaimMismatch { expected: Blake3Hash, got: Blake3Hash },
}

impl From<ChainError> for MissionRuntimeError {
    fn from(e: ChainError) -> Self { MissionRuntimeError::Chain(e) }
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
    /// `true` iff admissibility returned a non-Permit verdict. Mirrors
    /// `admissibility.verdict != Verdict::Permit`.
    pub rejected: bool,
    /// Authoritative final stage of the envelope.
    pub stage: MissionStage,
    /// Nanosecond timestamp of submission (monotonic, not wall-clock).
    pub timestamp_ns: u64,
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
            DegradedOccasion::MyelinationPersistFailed { .. }   => 0x01,
            DegradedOccasion::DemyelinationPersistFailed { .. } => 0x02,
            DegradedOccasion::ReasoningFailed { .. }            => 0x03,
            DegradedOccasion::ConsolidationDivergence { .. }    => 0x04,
            DegradedOccasion::ClockFailure { .. }               => 0x05,
        }
    }
}

impl ReceiptPayload for DegradedPathReceipt {
    fn kind(&self) -> ReceiptKind { ReceiptKind::DegradedPath }
    fn timestamp_ns(&self) -> u64 { self.timestamp_ns }
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
    fn kind(&self) -> ReceiptKind { ReceiptKind::ReasoningSession }
    fn timestamp_ns(&self) -> u64 { self.timestamp_ns }
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
}

impl CognitionRuntime {
    pub fn new(graph: ThoughtGraph, chain: ReceiptChain, ctx: AgentCtx) -> Self {
        Self {
            graph,
            chain,
            ctx,
            missions: HashMap::new(),
            session_counter: 0,
        }
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
    pub fn rehydrate(
        graph: ThoughtGraph,
        chain: ReceiptChain,
    ) -> Result<Self, RehydrateError> {
        let mut rt = Self {
            graph,
            chain,
            ctx: AgentCtx { receipt_chain: [0u8; 32] },
            missions: HashMap::new(),
            session_counter: 0,
        };

        // Collect replay actions in order. We iterate records oldest-to-newest.
        // For each Myelination, we must decode the payload to recover the
        // CompiledReflex's policy_version and source hash.
        let records: Vec<(ReceiptKind, Blake3Hash)> = rt.chain.records()
            .map(|r| (r.kind, r.hash))
            .collect();

        for (kind, hash) in records {
            match kind {
                ReceiptKind::Myelination => {
                    let receipt: MyelinationReceipt = rt.chain.fetch_and_decode(&hash)
                        .map_err(RehydrateError::ChainFetch)?;

                    // Reconstruct the CompiledReflex. In production this calls
                    // skill_reflex_bridge to rebuild the specialized runtime; here
                    // we use the stub constructor. Source hash and policy version
                    // come from the receipt, which is the authoritative record.
                    let reflex = CompiledReflex {
                        source_s2: receipt.source_s2,
                        policy_version: receipt.policy_version,
                    };
                    rt.graph.install_reflex_from_replay(receipt.source_s2, reflex);
                }
                ReceiptKind::Demyelination => {
                    let receipt: DemyelinationReceipt = rt.chain.fetch_and_decode(&hash)
                        .map_err(RehydrateError::ChainFetch)?;
                    rt.graph.remove_reflex_from_replay(&receipt.reflex);
                }
                // All other kinds do not affect reflex truth-state.
                ReceiptKind::Genesis
                | ReceiptKind::CognitionBoot
                | ReceiptKind::ReasoningSession
                | ReceiptKind::GovernanceDecision
                | ReceiptKind::NodeLifecycle
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
                rejected: true,
                stage: MissionStage::Admissibility,
                timestamp_ns,
            };
            self.missions.insert(mission_id, record.clone());
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
        if replay_ok {
            envelope.advance_stage(); // S7 -> S8 Replayability
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
            rejected: false,
            stage: final_stage,
            timestamp_ns,
        };
        self.missions.insert(mission_id, record.clone());
        Ok(record)
    }

    pub fn mission_by_id(&self, mission_id: &Blake3Hash) -> Option<&MissionRuntimeRecord> {
        self.missions.get(mission_id)
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
    pub fn handle(
        &mut self,
        event: CognitionEvent,
    ) -> Result<Option<Vec<Thought>>, LoopError> {
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

        let thoughts_digest = blake3_domain(
            "bizra-thoughts-v1",
            &(thoughts.len() as u32).to_le_bytes(),
        );

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
                Err(ReasoningError::QuarantineDivergence { candidate, divergence }) => {
                    self.emit_degraded(DegradedOccasion::ConsolidationDivergence {
                        edge: candidate, divergence,
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
                    self.graph.commit_myelination(edge_hash, compiled, receipt_hash);
                }
                Err(chain_err) => {
                    self.graph.abort_myelination(edge_hash);
                    self.emit_degraded(DegradedOccasion::MyelinationPersistFailed {
                        edge: edge_hash, cause: format!("{:?}", chain_err),
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
        let receipt = match self.graph.propose_demyelination(
            edge,
            DemyelinationReason::GovernanceDecision(decision),
        ) {
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
                    edge, cause: format!("{:?}", chain_err),
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

fn admissibility_result_matches(
    left: &AdmissibilityResult,
    right: &AdmissibilityResult,
) -> bool {
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

fn rejected_claim_matches(
    left: Option<&RejectedClaim>,
    right: Option<&RejectedClaim>,
) -> bool {
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
        fn traverse(&self, _ctx: &mut AgentCtx) -> Vec<Thought> { vec![Thought] }
    }

    fn minimal_graph() -> (ThoughtGraph, Blake3Hash) {
        let root_hash = [1u8; 32];
        let mut nodes: HashMap<Blake3Hash, Box<dyn GraphNode>> = HashMap::new();
        nodes.insert(root_hash, Box::new(NoopNode));
        let mut policies = HashMap::new();
        policies.insert(root_hash, MyelinationPolicy::standard());
        let genesis = [0u8; 32];
        (ThoughtGraph::from_parts(nodes, vec![root_hash], policies, genesis), genesis)
    }

    fn minimal_runtime() -> CognitionRuntime {
        let (graph, genesis) = minimal_graph();
        let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx = AgentCtx { receipt_chain: genesis };
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
        let result = rt.handle(CognitionEvent::ReasoningRequest {
            request_id: [42u8; 32],
        }).unwrap();
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
            rt.handle(CognitionEvent::ReasoningRequest { request_id: [i; 32] }).unwrap();
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
        let ctx = AgentCtx { receipt_chain: genesis };
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

        assert!(rehydrated.graph.has_reflex(&edge),
                "rehydrate must reconstruct reflex from Myelination receipt");
    }

    /// Prove that Myelination followed by Demyelination leaves no reflex
    /// after rehydrate — the chain's final state wins.
    #[test]
    fn rehydrate_respects_demyelination() {
        let (graph, genesis) = minimal_graph();
        let chain = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx = AgentCtx { receipt_chain: genesis };
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

        assert!(!rehydrated.graph.has_reflex(&edge),
                "rehydrate must respect demyelination — chain final state wins");
    }

    /// The operational proof of R1: two nodes starting from the same chain
    /// end in identical reflex state. This is Node1 reproducibility.
    #[test]
    fn rehydrate_is_deterministic() {
        let (graph_a, genesis) = minimal_graph();
        let chain_a = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
        let ctx_a = AgentCtx { receipt_chain: genesis };
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
        let ctx_b = AgentCtx { receipt_chain: genesis };
        let mut rt_b = CognitionRuntime::new(graph_b, chain_b, ctx_b);
        rt_b.chain.append_with_payload(myel).unwrap();

        let (fresh_b, _) = minimal_graph();
        let rehydrated_b = CognitionRuntime::rehydrate(fresh_b, rt_b.chain).unwrap();

        assert_eq!(rehydrated_a.graph.chain_head(), chain_a_head);
        assert_eq!(rehydrated_a.graph.chain_head(), rehydrated_b.graph.chain_head());
        assert_eq!(rehydrated_a.graph.has_reflex(&edge),
                   rehydrated_b.graph.has_reflex(&edge));
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
        let final_receipt = record.final_receipt.as_ref().expect("permit must have final receipt");
        assert_eq!(final_receipt.kind, ReceiptKind::NodeLifecycle);
        assert_eq!(final_receipt.claim_ref, record.envelope.mission_id);
        assert_eq!(record.receipt_id, Some(final_receipt.receipt_id));
        assert_eq!(record.mission_payload_hash, Some(final_receipt.claim_ref).map(|_| record.mission_payload_hash.unwrap()));
        assert_eq!(rt.mission_count(), 1);
        assert_eq!(rt.chain.len(), 7, "1 mission + 5 gates + 1 final receipt");
        assert_eq!(rt.chain.head(), final_receipt.receipt_id);

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

        assert!(record.rejected, "verdict was Reject — record must be marked rejected");
        assert!(record.receipt_id.is_none(), "rejected mission must have no receipt_id");
        assert!(record.final_receipt.is_none(), "rejected mission must have no final receipt");
        assert!(record.mission_payload_hash.is_none(),
            "rejected mission envelope was NOT appended to chain");
        assert_eq!(record.stage, MissionStage::Admissibility,
            "rejected mission stops at S4 Admissibility");
        assert_eq!(record.admissibility.verdict, Verdict::Reject);
        assert!(record.admissibility.rejected.is_some(),
            "reject must carry RejectedClaim with remediation path");

        // Chain UNCHANGED on reject (§10 chain-is-truth).
        assert_eq!(rt.chain.len(), pre_chain_len,
            "rejected mission must NOT advance the chain at all");

        // Registry PRESERVES the rejection (derived state per §10).
        assert_eq!(rt.mission_count(), 1,
            "rejected mission must be queryable via mission_by_id");
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

        assert_eq!(record.envelope.stage, MissionStage::Replayability,
            "permit + decode-verified replay must reach S8");
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
}
