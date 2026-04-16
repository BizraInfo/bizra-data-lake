//! BIZRA Cognition Substrate — Configure Layer
//! ==============================================
//! File: crates/bizra-kernel/src/configure/cognition.rs
//! Chains into: /data/bizra/docs/contracts/unified-node-contract-v1.0.md §X.Y
//! Domain tag: bizra-configure-v1
//!
//! This is the *configuration* layer — it composes the kernel primitives
//! (ThoughtGraph, MyelinationPolicy, GatePolicy, PatternMemory) into a
//! bootable cognition substrate for a single node. Every policy decision
//! is explicit and receipted.
//!
//! Ownership model: factory-based.
//! Nodes are produced by GraphNodeFactory::build() at configure time, which
//! returns owned Box<dyn GraphNode>. No Arc<dyn GraphNode> in the live
//! graph, no unsafe ownership transitions. Configs are pure templates and
//! can be reused across node instantiations (Node0, Node1, replay, fuzz).
//!
//! Constitutional invariants asserted at boot:
//!   I1. FATE-crossing edges are immutable_s2 = true (rejected at configure time)
//!   I2. At least one UserNiyyah root must exist
//!   I3. Policy version is strictly monotonic across boots (enforced here)
//!   I4. Boot receipt chains into the prior lifecycle receipt (enforced here)
//!   I5. No duplicate edge_hash within a single config (rejected at configure time)
//!   I6. Boot digests are canonical — edges and roots sorted before hashing
//!   I7. No unwrap() in the boot path; every failure is typed
//!
//! Author: BIZRA Foundation (Mumo / Mohamed Beshr)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::thought_graph::{
    ThoughtGraph, MyelinationPolicy, Blake3Hash, GraphNode, AgentCtx,
};
use crate::canonical_hasher::blake3_domain;
use crate::receipts::{ReceiptPayload, ReceiptKind, ReceiptChain};

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum ConfigureError {
    MissingRoot { role: &'static str },
    PolicyConflict { edge: Blake3Hash, reason: &'static str },
    DuplicateEdge(Blake3Hash),
    ChainDiscontinuity { expected_prev: Blake3Hash, got: Blake3Hash },
    PolicyVersionRegression { previous: u32, proposed: u32 },
    Clock(String),
    Factory(String),
    ReceiptEmission(String),
}

// ============================================================================
// GraphNodeFactory — named trait, not anonymous closure
// ============================================================================

pub trait GraphNodeFactory: Send + Sync {
    /// Build a fresh, owned node instance.
    fn build(&self) -> Result<Box<dyn GraphNode>, ConfigureError>;
    /// Stable identifier for what this factory produces. Included in the
    /// boot digest so provenance is recorded.
    fn factory_kind(&self) -> &'static str;
}

// ============================================================================
// Edge declarations — hashed enums have explicit repr
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PatAgent {
    Atlas  = 0,
    Oracle = 1,
    Forge  = 2,
    Judge  = 3,
    Crown  = 4,
    Herald = 5,
    Nexus  = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SatAgent {
    Consensus = 0,
    Resource  = 1,
    Proof     = 2,
    Impact    = 3,
    UrpLeader = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PolicyClass {
    Standard  = 0,
    HotPath   = 1,
    Sovereign = 2,
    Immutable = 3,
}

impl PolicyClass {
    pub fn to_policy(self, version: u32) -> MyelinationPolicy {
        match self {
            PolicyClass::Standard => MyelinationPolicy {
                hit_threshold: 3,
                quarantine_observations: 16,
                max_divergence: 0.05,
                immutable_s2: false,
                policy_version: version,
            },
            PolicyClass::HotPath => MyelinationPolicy {
                hit_threshold: 2,
                quarantine_observations: 8,
                max_divergence: 0.05,
                immutable_s2: false,
                policy_version: version,
            },
            PolicyClass::Sovereign => MyelinationPolicy {
                hit_threshold: 12,
                quarantine_observations: 64,
                max_divergence: 0.01,
                immutable_s2: false,
                policy_version: version,
            },
            PolicyClass::Immutable => {
                let mut p = MyelinationPolicy::fate_crossing();
                p.policy_version = version;
                p
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeRole {
    UserNiyyah,
    PatAgent(PatAgent),
    FateCrossing,
    SatAgent(SatAgent),
    Housekeeping,
}

impl EdgeRole {
    /// I1 enforcement hook: FATE must be Immutable.
    pub fn required_policy_class(&self) -> Option<PolicyClass> {
        match self {
            EdgeRole::FateCrossing => Some(PolicyClass::Immutable),
            _ => None,
        }
    }

    fn role_bytes(&self) -> [u8; 2] {
        match self {
            EdgeRole::UserNiyyah       => [0x00, 0x00],
            EdgeRole::PatAgent(a)      => [0x10, *a as u8],
            EdgeRole::FateCrossing     => [0x20, 0x00],
            EdgeRole::SatAgent(a)      => [0x30, *a as u8],
            EdgeRole::Housekeeping     => [0x40, 0x00],
        }
    }
}

pub struct EdgeDeclaration {
    pub edge_hash: Blake3Hash,
    pub role: EdgeRole,
    pub policy_class: PolicyClass,
    pub factory: Arc<dyn GraphNodeFactory>,
}

impl std::fmt::Debug for EdgeDeclaration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdgeDeclaration")
            .field("edge_hash", &self.edge_hash)
            .field("role", &self.role)
            .field("policy_class", &self.policy_class)
            .field("factory_kind", &self.factory.factory_kind())
            .finish()
    }
}

// ============================================================================
// Boot receipt payload
// ============================================================================

#[derive(Debug, Clone)]
pub struct CognitionBootReceipt {
    pub node_id: Blake3Hash,
    pub policy_version: u32,
    pub edge_count: u32,
    pub roots: Vec<Blake3Hash>,        // canonical-sorted
    pub edges_digest: Blake3Hash,      // over canonical-sorted edges
    pub policies_digest: Blake3Hash,   // over canonical-sorted policies
    pub factories_digest: Blake3Hash,  // over canonical-sorted factory kinds
    pub prev_chain: Blake3Hash,
    pub previous_policy_version: Option<u32>,
    pub timestamp_ns: u64,
}

impl ReceiptPayload for CognitionBootReceipt {
    fn kind(&self) -> ReceiptKind { ReceiptKind::CognitionBoot }
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256 + self.roots.len() * 32);
        buf.extend_from_slice(&self.node_id);
        buf.extend_from_slice(&self.policy_version.to_le_bytes());
        buf.extend_from_slice(&self.edge_count.to_le_bytes());
        buf.extend_from_slice(&(self.roots.len() as u32).to_le_bytes());
        for r in &self.roots { buf.extend_from_slice(r); }
        buf.extend_from_slice(&self.edges_digest);
        buf.extend_from_slice(&self.policies_digest);
        buf.extend_from_slice(&self.factories_digest);
        buf.extend_from_slice(&self.prev_chain);
        match self.previous_policy_version {
            Some(v) => { buf.push(1); buf.extend_from_slice(&v.to_le_bytes()); }
            None    => { buf.push(0); }
        }
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf
    }
    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-configure-v1", &self.canonical_bytes())
    }
}

// ============================================================================
// Cognition config
// ============================================================================

pub struct CognitionConfig {
    pub node_id: Blake3Hash,
    pub policy_version: u32,
    pub edges: Vec<EdgeDeclaration>,
    pub prev_chain: Blake3Hash,
    /// Previous policy version from the prior boot, if any. None means
    /// this is the genesis cognition boot for the node.
    pub previous_policy_version: Option<u32>,
}

pub struct ConfiguredCognition {
    pub graph: ThoughtGraph,
    pub boot_receipt: CognitionBootReceipt,
    pub boot_receipt_hash: Blake3Hash,
}

impl CognitionConfig {
    /// Build the cognition substrate.
    ///
    /// Sequence:
    ///   1. validate invariants I1, I2, I3, I4, I5 (no side effects)
    ///   2. canonical-sort edges by edge_hash (I6)
    ///   3. call each factory to produce owned nodes
    ///   4. compute canonical digests
    ///   5. persist boot receipt payload via chain.append_with_payload
    ///   6. construct ThoughtGraph with the appended chain head
    pub fn build(
        self,
        ctx: &mut AgentCtx,
        chain: &mut ReceiptChain,
    ) -> Result<ConfiguredCognition, ConfigureError> {
        // --- I4: chain continuity ---
        if chain.head() != self.prev_chain {
            return Err(ConfigureError::ChainDiscontinuity {
                expected_prev: self.prev_chain,
                got: chain.head(),
            });
        }

        // --- I3: policy version strict monotonicity ---
        if let Some(prev) = self.previous_policy_version {
            if self.policy_version <= prev {
                return Err(ConfigureError::PolicyVersionRegression {
                    previous: prev,
                    proposed: self.policy_version,
                });
            }
        }

        // --- I5: duplicate edge hash rejection ---
        let mut seen: HashSet<Blake3Hash> = HashSet::with_capacity(self.edges.len());
        for edge in &self.edges {
            if !seen.insert(edge.edge_hash) {
                return Err(ConfigureError::DuplicateEdge(edge.edge_hash));
            }
        }

        // --- I1: FATE crossings must be Immutable ---
        for edge in &self.edges {
            if let Some(required) = edge.role.required_policy_class() {
                if edge.policy_class != required {
                    return Err(ConfigureError::PolicyConflict {
                        edge: edge.edge_hash,
                        reason: "FATE crossing must be Immutable",
                    });
                }
            }
        }

        // --- I6: canonical sort for deterministic digests ---
        let mut sorted: Vec<EdgeDeclaration> = self.edges;
        sorted.sort_by(|a, b| a.edge_hash.cmp(&b.edge_hash));

        // --- Call factories (owned Box, no Arc-to-Box gymnastics) ---
        let mut nodes: HashMap<Blake3Hash, Box<dyn GraphNode>> =
            HashMap::with_capacity(sorted.len());
        let mut policies: HashMap<Blake3Hash, MyelinationPolicy> =
            HashMap::with_capacity(sorted.len());
        let mut roots: Vec<Blake3Hash> = Vec::new();

        // Digest buffers built over the canonical-sorted order
        let mut edges_buf: Vec<u8> = Vec::with_capacity(sorted.len() * 36);
        let mut policies_buf: Vec<u8> = Vec::with_capacity(sorted.len() * 64);
        let mut factories_buf: Vec<u8> = Vec::with_capacity(sorted.len() * 32);

        for edge in &sorted {
            // Build owned node from factory
            let node = edge.factory.build()?;
            nodes.insert(edge.edge_hash, node);

            // Materialize policy from class at current version
            let policy = edge.policy_class.to_policy(self.policy_version);
            policies.insert(edge.edge_hash, policy.clone());

            // Root classification
            if edge.role == EdgeRole::UserNiyyah {
                roots.push(edge.edge_hash);
            }

            // Canonical edge digest input
            edges_buf.extend_from_slice(&edge.edge_hash);
            edges_buf.extend_from_slice(&edge.role.role_bytes());
            edges_buf.push(edge.policy_class as u8);

            // Canonical policy digest input
            policies_buf.extend_from_slice(&edge.edge_hash);
            policies_buf.extend_from_slice(&policy.hit_threshold.to_le_bytes());
            policies_buf.extend_from_slice(&policy.quarantine_observations.to_le_bytes());
            policies_buf.extend_from_slice(&policy.max_divergence.to_le_bytes());
            policies_buf.push(policy.immutable_s2 as u8);
            policies_buf.extend_from_slice(&policy.policy_version.to_le_bytes());

            // Factory provenance digest input
            let kind = edge.factory.factory_kind().as_bytes();
            factories_buf.extend_from_slice(&edge.edge_hash);
            factories_buf.extend_from_slice(&(kind.len() as u32).to_le_bytes());
            factories_buf.extend_from_slice(kind);
        }

        // --- I2: at least one UserNiyyah root ---
        if roots.is_empty() {
            return Err(ConfigureError::MissingRoot {
                role: "UserNiyyah root (DEMA entry point)",
            });
        }
        roots.sort(); // canonical root order

        // --- Compute digests ---
        let edges_digest    = blake3_domain("bizra-configure-v1:edges",     &edges_buf);
        let policies_digest = blake3_domain("bizra-configure-v1:policies",  &policies_buf);
        let factories_digest= blake3_domain("bizra-configure-v1:factories", &factories_buf);

        // --- Build boot receipt ---
        let receipt = CognitionBootReceipt {
            node_id: self.node_id,
            policy_version: self.policy_version,
            edge_count: sorted.len() as u32,
            roots: roots.clone(),
            edges_digest,
            policies_digest,
            factories_digest,
            prev_chain: self.prev_chain,
            previous_policy_version: self.previous_policy_version,
            timestamp_ns: current_ns()?,
        };

        // --- Persist payload, append chain record. Atomicity lives in ReceiptChain. ---
        let receipt_hash = chain.append_with_payload(receipt.clone())
            .map_err(|e| ConfigureError::ReceiptEmission(format!("{:?}", e)))?;

        // --- Construct graph with new chain head ---
        let graph = ThoughtGraph::from_parts(nodes, roots, policies, receipt_hash);

        ctx.receipt_chain = receipt_hash;

        Ok(ConfiguredCognition {
            graph,
            boot_receipt: receipt,
            boot_receipt_hash: receipt_hash,
        })
    }
}

fn current_ns() -> Result<u64, ConfigureError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .map_err(|e| ConfigureError::Clock(e.to_string()))
}

// ============================================================================
// Canonical default builder: PAT-7 / SAT-5
// ============================================================================

pub struct Pat7Sat5Factories {
    pub dema: Arc<dyn GraphNodeFactory>,
    pub atlas: Arc<dyn GraphNodeFactory>,
    pub oracle: Arc<dyn GraphNodeFactory>,
    pub forge: Arc<dyn GraphNodeFactory>,
    pub judge: Arc<dyn GraphNodeFactory>,
    pub crown: Arc<dyn GraphNodeFactory>,
    pub herald: Arc<dyn GraphNodeFactory>,
    pub nexus: Arc<dyn GraphNodeFactory>,
    pub fate: Arc<dyn GraphNodeFactory>,
    pub consensus: Arc<dyn GraphNodeFactory>,
    pub resource: Arc<dyn GraphNodeFactory>,
    pub proof: Arc<dyn GraphNodeFactory>,
    pub impact: Arc<dyn GraphNodeFactory>,
    pub urp_leader: Arc<dyn GraphNodeFactory>,
}

pub fn default_pat7_sat5_config(
    node_id: Blake3Hash,
    policy_version: u32,
    prev_chain: Blake3Hash,
    previous_policy_version: Option<u32>,
    factories: Pat7Sat5Factories,
) -> CognitionConfig {
    let mut edges = Vec::with_capacity(14);

    edges.push(EdgeDeclaration {
        edge_hash: hash_role("dema.user_niyyah"),
        role: EdgeRole::UserNiyyah,
        policy_class: PolicyClass::Standard,
        factory: factories.dema,
    });

    let pat: [(PatAgent, Arc<dyn GraphNodeFactory>, PolicyClass); 7] = [
        (PatAgent::Atlas,  factories.atlas,  PolicyClass::HotPath),
        (PatAgent::Oracle, factories.oracle, PolicyClass::HotPath),
        (PatAgent::Forge,  factories.forge,  PolicyClass::HotPath),
        (PatAgent::Judge,  factories.judge,  PolicyClass::Sovereign),
        (PatAgent::Crown,  factories.crown,  PolicyClass::Standard),
        (PatAgent::Herald, factories.herald, PolicyClass::HotPath),
        (PatAgent::Nexus,  factories.nexus,  PolicyClass::HotPath),
    ];
    for (agent, factory, class) in pat {
        edges.push(EdgeDeclaration {
            edge_hash: hash_role(&format!("pat.{:?}", agent)),
            role: EdgeRole::PatAgent(agent),
            policy_class: class,
            factory,
        });
    }

    edges.push(EdgeDeclaration {
        edge_hash: hash_role("fate.crossing"),
        role: EdgeRole::FateCrossing,
        policy_class: PolicyClass::Immutable,
        factory: factories.fate,
    });

    let sat: [(SatAgent, Arc<dyn GraphNodeFactory>); 5] = [
        (SatAgent::Consensus, factories.consensus),
        (SatAgent::Resource,  factories.resource),
        (SatAgent::Proof,     factories.proof),
        (SatAgent::Impact,    factories.impact),
        (SatAgent::UrpLeader, factories.urp_leader),
    ];
    for (agent, factory) in sat {
        edges.push(EdgeDeclaration {
            edge_hash: hash_role(&format!("sat.{:?}", agent)),
            role: EdgeRole::SatAgent(agent),
            policy_class: PolicyClass::Sovereign,
            factory,
        });
    }

    CognitionConfig {
        node_id,
        policy_version,
        edges,
        prev_chain,
        previous_policy_version,
    }
}

fn hash_role(role: &str) -> Blake3Hash {
    blake3_domain("bizra-configure-v1:role", role.as_bytes())
}

// ============================================================================
// Tests — real assertions, not placeholders
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thought_graph::Thought;
    use crate::receipts::InMemoryPayloadStore;

    // --- Test scaffolding: a minimal GraphNode + Factory ---
    struct NoopNode;
    impl GraphNode for NoopNode {
        fn traverse(&self, _ctx: &mut AgentCtx) -> Vec<Thought> { Vec::new() }
    }

    struct NoopFactory { kind: &'static str }
    impl GraphNodeFactory for NoopFactory {
        fn build(&self) -> Result<Box<dyn GraphNode>, ConfigureError> {
            Ok(Box::new(NoopNode))
        }
        fn factory_kind(&self) -> &'static str { self.kind }
    }

    fn noop(kind: &'static str) -> Arc<dyn GraphNodeFactory> {
        Arc::new(NoopFactory { kind })
    }

    fn new_chain() -> (ReceiptChain, Blake3Hash) {
        let genesis = [0u8; 32];
        let store = Box::new(InMemoryPayloadStore::new());
        (ReceiptChain::new(genesis, store), genesis)
    }

    fn all_noop_factories() -> Pat7Sat5Factories {
        Pat7Sat5Factories {
            dema: noop("dema"),
            atlas: noop("atlas"),  oracle: noop("oracle"),  forge: noop("forge"),
            judge: noop("judge"),  crown: noop("crown"),    herald: noop("herald"),
            nexus: noop("nexus"),  fate: noop("fate"),
            consensus: noop("consensus"), resource: noop("resource"),
            proof: noop("proof"),  impact: noop("impact"),  urp_leader: noop("urp_leader"),
        }
    }

    #[test]
    fn fate_crossing_must_be_immutable() {
        // Construct a config where FATE is declared as Standard. Must reject.
        let (mut chain, genesis) = new_chain();
        let mut ctx = AgentCtx { receipt_chain: genesis };

        let mut cfg = default_pat7_sat5_config(
            [1u8; 32], 1, genesis, None, all_noop_factories(),
        );
        // Tamper: find the FATE edge and demote its policy class.
        for e in &mut cfg.edges {
            if e.role == EdgeRole::FateCrossing {
                e.policy_class = PolicyClass::Standard;
            }
        }

        match cfg.build(&mut ctx, &mut chain) {
            Err(ConfigureError::PolicyConflict { reason, .. }) => {
                assert_eq!(reason, "FATE crossing must be Immutable");
            }
            other => panic!("expected PolicyConflict, got {:?}", other.is_ok()),
        }
    }

    #[test]
    fn missing_user_niyyah_root_rejects() {
        let (mut chain, genesis) = new_chain();
        let mut ctx = AgentCtx { receipt_chain: genesis };

        let cfg = CognitionConfig {
            node_id: [2u8; 32],
            policy_version: 1,
            // No UserNiyyah edges at all — just a housekeeping edge
            edges: vec![EdgeDeclaration {
                edge_hash: [9u8; 32],
                role: EdgeRole::Housekeeping,
                policy_class: PolicyClass::Standard,
                factory: noop("hk"),
            }],
            prev_chain: genesis,
            previous_policy_version: None,
        };

        assert!(matches!(
            cfg.build(&mut ctx, &mut chain),
            Err(ConfigureError::MissingRoot { .. })
        ));
    }

    #[test]
    fn duplicate_edge_hash_rejects() {
        let (mut chain, genesis) = new_chain();
        let mut ctx = AgentCtx { receipt_chain: genesis };

        let cfg = CognitionConfig {
            node_id: [3u8; 32],
            policy_version: 1,
            edges: vec![
                EdgeDeclaration {
                    edge_hash: [7u8; 32],
                    role: EdgeRole::UserNiyyah,
                    policy_class: PolicyClass::Standard,
                    factory: noop("a"),
                },
                EdgeDeclaration {
                    edge_hash: [7u8; 32], // duplicate
                    role: EdgeRole::Housekeeping,
                    policy_class: PolicyClass::Standard,
                    factory: noop("b"),
                },
            ],
            prev_chain: genesis,
            previous_policy_version: None,
        };

        assert!(matches!(
            cfg.build(&mut ctx, &mut chain),
            Err(ConfigureError::DuplicateEdge(_))
        ));
    }

    #[test]
    fn chain_discontinuity_rejects() {
        let (mut chain, _genesis) = new_chain();
        let mut ctx = AgentCtx { receipt_chain: [0u8; 32] };

        let cfg = default_pat7_sat5_config(
            [4u8; 32], 1,
            [0xAAu8; 32], // wrong prev
            None,
            all_noop_factories(),
        );

        assert!(matches!(
            cfg.build(&mut ctx, &mut chain),
            Err(ConfigureError::ChainDiscontinuity { .. })
        ));
    }

    #[test]
    fn policy_version_regression_rejects() {
        let (mut chain, genesis) = new_chain();
        let mut ctx = AgentCtx { receipt_chain: genesis };

        let cfg = default_pat7_sat5_config(
            [5u8; 32], 5, genesis,
            Some(5), // proposed == previous, not strictly greater
            all_noop_factories(),
        );

        assert!(matches!(
            cfg.build(&mut ctx, &mut chain),
            Err(ConfigureError::PolicyVersionRegression { previous: 5, proposed: 5 })
        ));
    }

    #[test]
    fn canonical_sort_produces_stable_digest() {
        // Build the same config in two different declaration orders,
        // assert the resulting edges_digest is identical.
        let (mut chain_a, genesis) = new_chain();
        let (mut chain_b, _) = new_chain();
        let mut ctx_a = AgentCtx { receipt_chain: genesis };
        let mut ctx_b = AgentCtx { receipt_chain: genesis };

        let cfg_a = default_pat7_sat5_config(
            [6u8; 32], 1, genesis, None, all_noop_factories(),
        );
        let mut cfg_b = default_pat7_sat5_config(
            [6u8; 32], 1, genesis, None, all_noop_factories(),
        );
        cfg_b.edges.reverse(); // different declaration order

        let a = cfg_a.build(&mut ctx_a, &mut chain_a).unwrap();
        let b = cfg_b.build(&mut ctx_b, &mut chain_b).unwrap();

        assert_eq!(a.boot_receipt.edges_digest, b.boot_receipt.edges_digest,
                   "edges_digest must be order-independent");
        assert_eq!(a.boot_receipt.policies_digest, b.boot_receipt.policies_digest,
                   "policies_digest must be order-independent");
        assert_eq!(a.boot_receipt.roots, b.boot_receipt.roots,
                   "roots must be sorted identically");
    }

    #[test]
    fn successful_boot_appends_to_chain() {
        let (mut chain, genesis) = new_chain();
        let mut ctx = AgentCtx { receipt_chain: genesis };

        let cfg = default_pat7_sat5_config(
            [7u8; 32], 1, genesis, None, all_noop_factories(),
        );

        let configured = cfg.build(&mut ctx, &mut chain).unwrap();

        assert_eq!(chain.len(), 1);
        assert_eq!(chain.head(), configured.boot_receipt_hash);
        assert_eq!(ctx.receipt_chain, configured.boot_receipt_hash);
        assert_eq!(configured.graph.node_count(), 14);
        assert_eq!(configured.graph.root_count(), 1);
    }
}
