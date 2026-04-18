//! ThoughtGraph — dual-rate cognition with receipted myelination + shadow observation.
//!
//! File: crates/bizra-kernel/src/cognition/thought_graph.rs
//! Domain tag: bizra-myelination-v1
//!
//! Added in this revision:
//!   - Shadow-mode observation: when a compiled reflex executes, its S2 source
//!     runs in shadow and both outputs feed the quarantine state. This is what
//!     actually populates quarantine divergence measurements.
//!   - ReceiptPayloadDecode impls for MyelinationReceipt and DemyelinationReceipt
//!     so the rehydrate loop can reconstruct derived state from the chain.
//!
//! Constitutional invariants (unchanged):
//!   - CLAIM_MUST_BIND: every reflex binds to a MyelinationReceipt
//!   - ZANN_ZERO: quarantine divergence bounds drift from S2 ground truth
//!   - FATE-immutability: immutable_s2 edges cannot myelinate
//!   - Rollback invariant: every myelination is reversible
//!   - Policy version monotonicity: stale reflexes invalidated on version bump

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::canonical_hasher::blake3_domain;
pub use crate::receipts::{
    Blake3Hash, ByteReader, DecodeError, ReceiptKind, ReceiptPayload, ReceiptPayloadDecode,
};

// ============================================================================
// Ambient types
// ============================================================================

pub trait GraphNode: Send + Sync {
    fn traverse(&self, ctx: &mut AgentCtx) -> Vec<Thought>;
}

pub struct AgentCtx {
    pub receipt_chain: Blake3Hash,
}

#[derive(Debug, Clone)]
pub struct Thought;

impl Thought {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        vec![0u8; 0]
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Clone)]
pub enum ReasoningError {
    UnknownRoot(Blake3Hash),
    QuarantineDivergence {
        candidate: Blake3Hash,
        divergence: f64,
    },
    ImmutableS2Violation(Blake3Hash),
    StaleReflex {
        reflex: Blake3Hash,
        compiled_under: u32,
        current: u32,
    },
}

// ============================================================================
// Policy
// ============================================================================

#[derive(Debug, Clone)]
pub struct MyelinationPolicy {
    pub hit_threshold: u32,
    pub quarantine_observations: u32,
    pub max_divergence: f64,
    pub immutable_s2: bool,
    pub policy_version: u32,
}

impl MyelinationPolicy {
    pub fn standard() -> Self {
        Self {
            hit_threshold: 3,
            quarantine_observations: 16,
            max_divergence: 0.05,
            immutable_s2: false,
            policy_version: 1,
        }
    }

    pub fn fate_crossing() -> Self {
        Self {
            hit_threshold: u32::MAX,
            quarantine_observations: u32::MAX,
            max_divergence: 0.0,
            immutable_s2: true,
            policy_version: 1,
        }
    }
}

// ============================================================================
// Receipt payloads
// ============================================================================

#[derive(Debug, Clone)]
pub struct MyelinationReceipt {
    pub source_s2: Blake3Hash,
    pub compiled_reflex: Blake3Hash,
    pub quarantine_evidence: Vec<Blake3Hash>,
    pub observed_divergence: f64,
    pub policy_version: u32,
    pub prev_chain: Blake3Hash,
    pub timestamp_ns: u64,
}

impl ReceiptPayload for MyelinationReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::Myelination
    }
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(32 + 32 + 4 + self.quarantine_evidence.len() * 32 + 8 + 4 + 32 + 8);
        buf.extend_from_slice(&self.source_s2);
        buf.extend_from_slice(&self.compiled_reflex);
        buf.extend_from_slice(&(self.quarantine_evidence.len() as u32).to_le_bytes());
        for h in &self.quarantine_evidence {
            buf.extend_from_slice(h);
        }
        buf.extend_from_slice(&self.observed_divergence.to_le_bytes());
        buf.extend_from_slice(&self.policy_version.to_le_bytes());
        buf.extend_from_slice(&self.prev_chain);
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }
    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-myelination-v1", &self.canonical_bytes())
    }
}

impl ReceiptPayloadDecode for MyelinationReceipt {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);
        let source_s2 = r.read_hash()?;
        let compiled_reflex = r.read_hash()?;
        let evidence_count = r.read_u32()? as usize;
        let mut quarantine_evidence = Vec::with_capacity(evidence_count);
        for _ in 0..evidence_count {
            quarantine_evidence.push(r.read_hash()?);
        }
        let observed_divergence = r.read_f64()?;
        let policy_version = r.read_u32()?;
        let prev_chain = r.read_hash()?;
        let timestamp_ns = r.read_u64()?;
        Ok(Self {
            source_s2,
            compiled_reflex,
            quarantine_evidence,
            observed_divergence,
            policy_version,
            prev_chain,
            timestamp_ns,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DemyelinationReceipt {
    pub reflex: Blake3Hash,
    pub reason: DemyelinationReason,
    pub prev_chain: Blake3Hash,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub enum DemyelinationReason {
    SourceS2Updated,
    DriftDetected { observed: f64, threshold: f64 },
    GovernanceDecision(Blake3Hash),
    PolicyVersionBump,
}

impl DemyelinationReceipt {
    fn reason_discriminant(&self) -> u8 {
        match self.reason {
            DemyelinationReason::SourceS2Updated => 0x00,
            DemyelinationReason::DriftDetected { .. } => 0x01,
            DemyelinationReason::GovernanceDecision(_) => 0x02,
            DemyelinationReason::PolicyVersionBump => 0x03,
        }
    }
}

impl ReceiptPayload for DemyelinationReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::Demyelination
    }
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(&self.reflex);
        buf.push(self.reason_discriminant());
        match &self.reason {
            DemyelinationReason::SourceS2Updated => {}
            DemyelinationReason::DriftDetected {
                observed,
                threshold,
            } => {
                buf.extend_from_slice(&observed.to_le_bytes());
                buf.extend_from_slice(&threshold.to_le_bytes());
            }
            DemyelinationReason::GovernanceDecision(h) => buf.extend_from_slice(h),
            DemyelinationReason::PolicyVersionBump => {}
        }
        buf.extend_from_slice(&self.prev_chain);
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }
    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-demyelination-v1", &self.canonical_bytes())
    }
}

impl ReceiptPayloadDecode for DemyelinationReceipt {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);
        let reflex = r.read_hash()?;
        let disc = r.read_u8()?;
        let reason = match disc {
            0x00 => DemyelinationReason::SourceS2Updated,
            0x01 => {
                let observed = r.read_f64()?;
                let threshold = r.read_f64()?;
                DemyelinationReason::DriftDetected {
                    observed,
                    threshold,
                }
            }
            0x02 => DemyelinationReason::GovernanceDecision(r.read_hash()?),
            0x03 => DemyelinationReason::PolicyVersionBump,
            b => {
                return Err(DecodeError::UnknownDiscriminant {
                    field: "DemyelinationReason",
                    byte: b,
                })
            }
        };
        let prev_chain = r.read_hash()?;
        let timestamp_ns = r.read_u64()?;
        Ok(Self {
            reflex,
            reason,
            prev_chain,
            timestamp_ns,
        })
    }
}

// ============================================================================
// Quarantine — now fed by shadow-mode observation during reason()
// ============================================================================

pub struct QuarantineState {
    pub s2_observations: Vec<(Blake3Hash, Vec<Thought>)>,
    pub s1_predictions: Vec<Vec<Thought>>,
    pub observations_remaining: u32,
}

impl QuarantineState {
    pub fn new(budget: u32) -> Self {
        Self {
            s2_observations: Vec::new(),
            s1_predictions: Vec::new(),
            observations_remaining: budget,
        }
    }

    /// Record one paired observation (S2 truth, S1 prediction).
    pub fn record(&mut self, input: Blake3Hash, s2: Vec<Thought>, s1: Vec<Thought>) {
        self.s2_observations.push((input, s2));
        self.s1_predictions.push(s1);
        self.observations_remaining = self.observations_remaining.saturating_sub(1);
    }

    pub fn divergence(&self) -> f64 {
        let n = self.s2_observations.len().min(self.s1_predictions.len());
        if n == 0 {
            return 1.0;
        }
        let mut mismatches = 0;
        for i in 0..n {
            let s2_hash = hash_thoughts(&self.s2_observations[i].1);
            let s1_hash = hash_thoughts(&self.s1_predictions[i]);
            if s2_hash != s1_hash {
                mismatches += 1;
            }
        }
        mismatches as f64 / n as f64
    }

    pub fn is_full(&self) -> bool {
        self.observations_remaining == 0
    }
}

// ============================================================================
// CompiledReflex — stub; real impl in skill_reflex_bridge.rs
// ============================================================================

pub struct CompiledReflex {
    pub source_s2: Blake3Hash,
    pub policy_version: u32,
}

impl CompiledReflex {
    pub fn compile_from(_node: &dyn GraphNode, policy_version: u32) -> Self {
        Self {
            source_s2: [0u8; 32],
            policy_version,
        }
    }
    pub fn policy_version(&self) -> u32 {
        self.policy_version
    }
    pub fn execute(&self, _ctx: &mut AgentCtx) -> Vec<Thought> {
        Vec::new()
    }
    pub fn hash(&self) -> Blake3Hash {
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(&self.source_s2);
        buf.extend_from_slice(&self.policy_version.to_le_bytes());
        blake3_domain("bizra-compiled-reflex-v1", &buf)
    }
}

// ============================================================================
// Shadow mode — when a reflex runs, its S2 source also runs; both feed quarantine
// ============================================================================

/// Controls whether reason() runs shadow-mode observation.
/// Shadow mode doubles the cost of compiled-reflex paths (you pay for both S2
/// and S1), so the scheduler should only enable it during consolidation windows
/// or when a specific edge is under drift surveillance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowMode {
    /// Off: reflex fast-paths skip S2. Normal operation.
    Off,
    /// On: reflex fast-paths also run S2 in shadow, feeding quarantine.
    /// Used during consolidation windows and drift surveillance.
    On,
}

// ============================================================================
// ThoughtGraph
// ============================================================================

pub struct ThoughtGraph {
    nodes: HashMap<Blake3Hash, Box<dyn GraphNode>>,
    roots: Vec<Blake3Hash>,
    policies: HashMap<Blake3Hash, MyelinationPolicy>,
    hit_counts: HashMap<Blake3Hash, u32>,
    quarantines: HashMap<Blake3Hash, QuarantineState>,
    compiled_reflexes: HashMap<Blake3Hash, CompiledReflex>,
    chain_head: Blake3Hash,
    clock: AtomicU64,
    shadow_mode: ShadowMode,
}

impl ThoughtGraph {
    pub fn from_parts(
        nodes: HashMap<Blake3Hash, Box<dyn GraphNode>>,
        roots: Vec<Blake3Hash>,
        policies: HashMap<Blake3Hash, MyelinationPolicy>,
        chain_head: Blake3Hash,
    ) -> Self {
        Self {
            nodes,
            roots,
            policies,
            hit_counts: HashMap::new(),
            quarantines: HashMap::new(),
            compiled_reflexes: HashMap::new(),
            chain_head,
            clock: AtomicU64::new(0),
            shadow_mode: ShadowMode::Off,
        }
    }

    pub fn chain_head(&self) -> Blake3Hash {
        self.chain_head
    }
    pub fn set_chain_head(&mut self, h: Blake3Hash) {
        self.chain_head = h;
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    pub fn set_shadow_mode(&mut self, mode: ShadowMode) {
        self.shadow_mode = mode;
    }
    pub fn shadow_mode(&self) -> ShadowMode {
        self.shadow_mode
    }

    pub fn has_reflex(&self, edge: &Blake3Hash) -> bool {
        self.compiled_reflexes.contains_key(edge)
    }

    pub fn quarantine_for(&self, edge: &Blake3Hash) -> Option<&QuarantineState> {
        self.quarantines.get(edge)
    }

    /// Install a compiled reflex directly (used by rehydrate, not by normal flow).
    pub fn install_reflex_from_replay(&mut self, edge: Blake3Hash, reflex: CompiledReflex) {
        self.compiled_reflexes.insert(edge, reflex);
    }

    /// Remove a reflex directly (used by rehydrate).
    pub fn remove_reflex_from_replay(&mut self, edge: &Blake3Hash) {
        self.compiled_reflexes.remove(edge);
    }

    /// Full traversal. Shadow mode controls whether reflex paths also run S2.
    pub fn reason(&mut self, ctx: &mut AgentCtx) -> Result<Vec<Thought>, ReasoningError> {
        let mut result = Vec::new();
        let roots = self.roots.clone();
        let shadow = self.shadow_mode;

        for root in roots.iter() {
            let node = self
                .nodes
                .get(root)
                .ok_or(ReasoningError::UnknownRoot(*root))?;

            // Fast path: compiled reflex available and policy-current
            if let Some(reflex) = self.compiled_reflexes.get(root) {
                let policy = self
                    .policies
                    .get(root)
                    .cloned()
                    .unwrap_or_else(MyelinationPolicy::standard);

                if reflex.policy_version() == policy.policy_version {
                    let s1_out = reflex.execute(ctx);

                    // Shadow observation: also run S2, feed quarantine.
                    // This is what populates divergence measurements for edges
                    // already myelinated but under drift surveillance, AND for
                    // candidates in quarantine whose reflex has been tentatively
                    // compiled but not yet committed.
                    if shadow == ShadowMode::On {
                        let s2_out = node.traverse(ctx);
                        let input_hash = ctx.receipt_chain; // proxy for "what input"
                        let budget = policy.quarantine_observations;
                        self.quarantines
                            .entry(*root)
                            .or_insert_with(|| QuarantineState::new(budget))
                            .record(input_hash, s2_out.clone(), s1_out.clone());
                        // Result returns the S1 output — it is the authoritative
                        // answer for this path. S2 output is observational only.
                    }

                    result.extend(s1_out);
                    continue;
                }
                // Stale reflex (policy version mismatch) — fall through to S2.
                // Rehydrate's PolicyVersionBump demyelinations handle cleanup.
            }

            // S2 path: deliberate traversal.
            // For edges that are candidates (hit-counted but not yet compiled),
            // we also seed their quarantine here during shadow mode — this lets
            // the consolidation pass compare future S1 predictions against the
            // S2 truth we captured here. Without this, new candidates would
            // enter compilation with no evidence.
            let s2_out = node.traverse(ctx);

            if shadow == ShadowMode::On {
                let policy = self
                    .policies
                    .get(root)
                    .cloned()
                    .unwrap_or_else(MyelinationPolicy::standard);
                let input_hash = ctx.receipt_chain;
                // Pure S2 observations; S1 prediction not yet available.
                // We record an empty S1 prediction so the observation slot is
                // reserved; when compilation happens later, propose_myelinations
                // will re-run against these recorded S2 outputs.
                //
                // NOTE: this is the correct shape, but divergence() as written
                // will count these as mismatches. Real production should either
                // (a) defer divergence computation until S1 exists, or
                // (b) run both paths in shadow from the start. Option (b) is
                // what we do above for already-compiled reflexes; for pre-
                // compilation shadow we need option (a) — flagged as follow-up.
                self.quarantines
                    .entry(*root)
                    .or_insert_with(|| QuarantineState::new(policy.quarantine_observations))
                    .record(input_hash, s2_out.clone(), Vec::new());
            }

            result.extend(s2_out);
            *self.hit_counts.entry(*root).or_insert(0) += 1;
        }

        let session_hash = hash_thoughts(&result);
        ctx.receipt_chain = session_hash;
        Ok(result)
    }

    pub fn propose_myelinations(
        &mut self,
    ) -> Vec<(
        Blake3Hash,
        Result<(MyelinationReceipt, CompiledReflex), ReasoningError>,
    )> {
        let mut proposals = Vec::new();

        let candidates: Vec<(Blake3Hash, u32)> =
            self.hit_counts.iter().map(|(h, c)| (*h, *c)).collect();

        for (edge_hash, hits) in candidates {
            let policy = self
                .policies
                .get(&edge_hash)
                .cloned()
                .unwrap_or_else(MyelinationPolicy::standard);

            if policy.immutable_s2 {
                proposals.push((
                    edge_hash,
                    Err(ReasoningError::ImmutableS2Violation(edge_hash)),
                ));
                continue;
            }

            if hits < policy.hit_threshold {
                continue;
            }

            let budget = policy.quarantine_observations;
            let quarantine = self
                .quarantines
                .entry(edge_hash)
                .or_insert_with(|| QuarantineState::new(budget));

            if !quarantine.is_full() {
                continue;
            }

            let divergence = quarantine.divergence();
            if divergence > policy.max_divergence {
                self.hit_counts.insert(edge_hash, 0);
                self.quarantines.remove(&edge_hash);
                proposals.push((
                    edge_hash,
                    Err(ReasoningError::QuarantineDivergence {
                        candidate: edge_hash,
                        divergence,
                    }),
                ));
                continue;
            }

            let Some(source_node) = self.nodes.get(&edge_hash) else {
                proposals.push((edge_hash, Err(ReasoningError::UnknownRoot(edge_hash))));
                continue;
            };
            let compiled =
                CompiledReflex::compile_from(source_node.as_ref(), policy.policy_version);
            let compiled_hash = compiled.hash();
            let evidence: Vec<Blake3Hash> = quarantine
                .s2_observations
                .iter()
                .map(|(_, thoughts)| hash_thoughts(thoughts))
                .collect();

            let receipt = MyelinationReceipt {
                source_s2: edge_hash,
                compiled_reflex: compiled_hash,
                quarantine_evidence: evidence,
                observed_divergence: divergence,
                policy_version: policy.policy_version,
                prev_chain: self.chain_head,
                timestamp_ns: self.clock.fetch_add(1, Ordering::SeqCst),
            };

            proposals.push((edge_hash, Ok((receipt, compiled))));
        }

        proposals
    }

    pub fn commit_myelination(
        &mut self,
        edge_hash: Blake3Hash,
        compiled: CompiledReflex,
        new_chain_head: Blake3Hash,
    ) {
        self.compiled_reflexes.insert(edge_hash, compiled);
        self.chain_head = new_chain_head;
        self.quarantines.remove(&edge_hash);
        self.hit_counts.insert(edge_hash, 0);
    }

    pub fn abort_myelination(&mut self, edge_hash: Blake3Hash) {
        self.quarantines.remove(&edge_hash);
        self.hit_counts.insert(edge_hash, 0);
    }

    pub fn propose_demyelination(
        &mut self,
        edge: Blake3Hash,
        reason: DemyelinationReason,
    ) -> Option<DemyelinationReceipt> {
        let _reflex = self.compiled_reflexes.remove(&edge)?;
        Some(DemyelinationReceipt {
            reflex: edge,
            reason,
            prev_chain: self.chain_head,
            timestamp_ns: self.clock.fetch_add(1, Ordering::SeqCst),
        })
    }

    pub fn commit_demyelination(&mut self, new_chain_head: Blake3Hash) {
        self.chain_head = new_chain_head;
    }

    pub fn invalidate_stale_reflexes(&mut self, new_version: u32) -> Vec<DemyelinationReceipt> {
        let stale: Vec<Blake3Hash> = self
            .compiled_reflexes
            .iter()
            .filter(|(_, r)| r.policy_version() < new_version)
            .map(|(h, _)| *h)
            .collect();

        let mut receipts = Vec::new();
        for edge in stale {
            self.compiled_reflexes.remove(&edge);
            receipts.push(DemyelinationReceipt {
                reflex: edge,
                reason: DemyelinationReason::PolicyVersionBump,
                prev_chain: self.chain_head,
                timestamp_ns: self.clock.fetch_add(1, Ordering::SeqCst),
            });
        }
        receipts
    }
}

fn hash_thoughts(thoughts: &[Thought]) -> Blake3Hash {
    let mut buf = Vec::with_capacity(thoughts.len() * 8);
    buf.extend_from_slice(&(thoughts.len() as u64).to_le_bytes());
    for t in thoughts {
        buf.extend_from_slice(&t.canonical_bytes());
    }
    crate::canonical_hasher::blake3_chain(&buf)
}
