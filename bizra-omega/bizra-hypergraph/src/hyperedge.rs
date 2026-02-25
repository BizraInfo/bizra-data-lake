//! Core hyperedge types and data structures.
//!
//! A [`HyperEdge`] connects N >= 2 nodes through a typed relationship.
//! Edge IDs are deterministic BLAKE3 hashes of sorted member IDs + edge type,
//! ensuring that the same set of members always produces the same edge ID
//! regardless of insertion order.
//!
//! Standing on Giants: Berge (1973, hypergraph theory)

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

// ─── HyperEdge type taxonomy ────────────────────────────────────────────────

/// Classification of N-ary relationships in the knowledge graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HyperEdgeType {
    /// N nodes share a concept (e.g., "autopoiesis" links 5 papers).
    ConceptCluster,
    /// Ordered sequence of cause→effect across N nodes.
    CausalChain,
    /// N nodes from different domains share a structural pattern.
    CrossDomainBridge,
    /// N events co-occur within a time window.
    TemporalCohort,
    /// N evidence items supporting one claim.
    EvidenceBundle,
}

// ─── NodeId ─────────────────────────────────────────────────────────────────

/// Opaque node identifier (string-based for cross-language compatibility).
pub type NodeId = String;

// ─── HyperEdge ──────────────────────────────────────────────────────────────

/// An N-ary relationship connecting two or more nodes.
///
/// # Invariants
///
/// - `members.len() >= 2` (enforced by [`HyperEdge::new`]).
/// - `id` is a deterministic BLAKE3 hash of sorted members + edge type.
/// - Members are stored in a `BTreeSet` for deterministic ordering.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperEdge {
    /// BLAKE3 hash (hex) of sorted member IDs + edge type.
    pub id: String,
    /// Node IDs participating in this edge (sorted, deduplicated).
    pub members: BTreeSet<NodeId>,
    /// Semantic category of the relationship.
    pub edge_type: HyperEdgeType,
    /// Strength of the relationship in [0.0, 1.0].
    pub weight: f64,
}

impl HyperEdge {
    /// Create a new hyperedge from an iterator of node IDs.
    ///
    /// # Panics
    ///
    /// Panics if fewer than 2 unique members are provided.
    pub fn new(members: impl IntoIterator<Item = NodeId>, edge_type: HyperEdgeType) -> Self {
        let members: BTreeSet<NodeId> = members.into_iter().collect();
        assert!(
            members.len() >= 2,
            "HyperEdge requires at least 2 members, got {}",
            members.len()
        );
        let id = Self::compute_id(&members, &edge_type);
        Self {
            id,
            members,
            edge_type,
            weight: 1.0,
        }
    }

    /// Create a new hyperedge with a custom weight.
    pub fn with_weight(
        members: impl IntoIterator<Item = NodeId>,
        edge_type: HyperEdgeType,
        weight: f64,
    ) -> Self {
        let mut edge = Self::new(members, edge_type);
        edge.weight = weight.clamp(0.0, 1.0);
        edge
    }

    /// Number of nodes in this hyperedge.
    pub fn cardinality(&self) -> usize {
        self.members.len()
    }

    /// Check if a node participates in this edge.
    pub fn contains(&self, node: &str) -> bool {
        self.members.contains(node)
    }

    /// True when the hyperedge degenerates to a standard binary edge.
    pub fn is_pairwise(&self) -> bool {
        self.cardinality() == 2
    }

    /// Nodes shared between this edge and another.
    pub fn overlap(&self, other: &HyperEdge) -> BTreeSet<NodeId> {
        self.members.intersection(&other.members).cloned().collect()
    }

    /// Deterministic edge ID from sorted members + edge type.
    fn compute_id(members: &BTreeSet<NodeId>, edge_type: &HyperEdgeType) -> String {
        let mut hasher = blake3::Hasher::new();
        for m in members {
            hasher.update(m.as_bytes());
            hasher.update(b"|");
        }
        hasher.update(format!("{edge_type:?}").as_bytes());
        hasher.finalize().to_hex()[..16].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_id() {
        let e1 = HyperEdge::new(
            vec!["A".into(), "B".into(), "C".into()],
            HyperEdgeType::ConceptCluster,
        );
        let e2 = HyperEdge::new(
            vec!["C".into(), "A".into(), "B".into()],
            HyperEdgeType::ConceptCluster,
        );
        assert_eq!(e1.id, e2.id, "Order-independent ID");
    }

    #[test]
    fn test_cardinality() {
        let e = HyperEdge::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            HyperEdgeType::CausalChain,
        );
        assert_eq!(e.cardinality(), 4);
    }

    #[test]
    fn test_is_pairwise() {
        let binary = HyperEdge::new(vec!["A".into(), "B".into()], HyperEdgeType::ConceptCluster);
        assert!(binary.is_pairwise());

        let ternary = HyperEdge::new(
            vec!["A".into(), "B".into(), "C".into()],
            HyperEdgeType::ConceptCluster,
        );
        assert!(!ternary.is_pairwise());
    }

    #[test]
    fn test_overlap() {
        let e1 = HyperEdge::new(
            vec!["A".into(), "B".into(), "C".into()],
            HyperEdgeType::ConceptCluster,
        );
        let e2 = HyperEdge::new(
            vec!["B".into(), "C".into(), "D".into()],
            HyperEdgeType::CausalChain,
        );
        let shared = e1.overlap(&e2);
        assert_eq!(shared.len(), 2);
        assert!(shared.contains("B"));
        assert!(shared.contains("C"));
    }

    #[test]
    fn test_contains() {
        let e = HyperEdge::new(vec!["X".into(), "Y".into()], HyperEdgeType::TemporalCohort);
        assert!(e.contains("X"));
        assert!(!e.contains("Z"));
    }

    #[test]
    fn test_weight_clamped() {
        let e = HyperEdge::with_weight(
            vec!["A".into(), "B".into()],
            HyperEdgeType::EvidenceBundle,
            2.5,
        );
        assert!((e.weight - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "requires at least 2 members")]
    fn test_single_node_panics() {
        HyperEdge::new(vec!["A".into()], HyperEdgeType::ConceptCluster);
    }
}
