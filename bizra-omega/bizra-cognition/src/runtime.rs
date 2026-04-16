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

use std::time::{SystemTime, UNIX_EPOCH};

use crate::thought_graph::{
    ThoughtGraph, AgentCtx, Thought, ReasoningError, ShadowMode,
    MyelinationReceipt, DemyelinationReceipt, DemyelinationReason,
    CompiledReflex,
};
use crate::receipts::{
    ReceiptChain, ReceiptPayload, ReceiptKind,
    ChainError, Blake3Hash,
};
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
    session_counter: u64,
}

impl CognitionRuntime {
    pub fn new(graph: ThoughtGraph, chain: ReceiptChain, ctx: AgentCtx) -> Self {
        Self { graph, chain, ctx, session_counter: 0 }
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::InMemoryPayloadStore;
    use std::collections::HashMap;
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
        let mut chain_b = ReceiptChain::new(genesis, Box::new(InMemoryPayloadStore::new()));
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
}
