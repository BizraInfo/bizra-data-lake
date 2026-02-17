//! Incidence-list hypergraph store with structural queries.
//!
//! The store maintains three indices:
//! - `edges`: edge_id → HyperEdge
//! - `incidence`: node_id → Set<edge_id>
//! - `type_index`: HyperEdgeType → Set<edge_id>
//!
//! All operations are O(1) amortised for single-edge mutations,
//! O(degree) for neighborhood queries.
//!
//! Standing on Giants: Berge (1973, incidence structure)

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::hyperedge::{HyperEdge, HyperEdgeType, NodeId};

/// In-memory hypergraph with incidence-list indexing.
pub struct HyperGraphStore {
    edges: HashMap<String, HyperEdge>,
    incidence: HashMap<NodeId, HashSet<String>>,
    type_index: HashMap<HyperEdgeType, HashSet<String>>,
}

impl HyperGraphStore {
    /// Create an empty hypergraph store.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            incidence: HashMap::new(),
            type_index: HashMap::new(),
        }
    }

    /// Add a hyperedge to the store. Returns the edge ID.
    pub fn add_edge(&mut self, edge: HyperEdge) -> String {
        let id = edge.id.clone();
        for member in &edge.members {
            self.incidence
                .entry(member.clone())
                .or_default()
                .insert(id.clone());
        }
        self.type_index
            .entry(edge.edge_type.clone())
            .or_default()
            .insert(id.clone());
        self.edges.insert(id.clone(), edge);
        id
    }

    /// Remove a hyperedge by ID.
    pub fn remove_edge(&mut self, edge_id: &str) -> Option<HyperEdge> {
        if let Some(edge) = self.edges.remove(edge_id) {
            for member in &edge.members {
                if let Some(set) = self.incidence.get_mut(member) {
                    set.remove(edge_id);
                    if set.is_empty() {
                        self.incidence.remove(member);
                    }
                }
            }
            if let Some(set) = self.type_index.get_mut(&edge.edge_type) {
                set.remove(edge_id);
            }
            Some(edge)
        } else {
            None
        }
    }

    /// All hyperedges containing the given node.
    pub fn edges_of(&self, node_id: &str) -> Vec<&HyperEdge> {
        self.incidence
            .get(node_id)
            .map(|ids| ids.iter().filter_map(|eid| self.edges.get(eid)).collect())
            .unwrap_or_default()
    }

    /// All nodes reachable from `node_id` via one hyperedge hop.
    /// Does not include `node_id` itself.
    pub fn neighbors(&self, node_id: &str) -> BTreeSet<NodeId> {
        let mut result = BTreeSet::new();
        for edge in self.edges_of(node_id) {
            for member in &edge.members {
                result.insert(member.clone());
            }
        }
        result.remove(node_id);
        result
    }

    /// All hyperedges of a given type.
    pub fn query_by_type(&self, edge_type: &HyperEdgeType) -> Vec<&HyperEdge> {
        self.type_index
            .get(edge_type)
            .map(|ids| ids.iter().filter_map(|eid| self.edges.get(eid)).collect())
            .unwrap_or_default()
    }

    /// Total number of unique nodes in the store.
    pub fn node_count(&self) -> usize {
        self.incidence.len()
    }

    /// Total number of hyperedges in the store.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Average cardinality of all edges. Returns 0.0 if empty.
    pub fn mean_cardinality(&self) -> f64 {
        if self.edges.is_empty() {
            return 0.0;
        }
        let total: usize = self.edges.values().map(|e| e.cardinality()).sum();
        total as f64 / self.edges.len() as f64
    }

    /// Get a reference to an edge by ID.
    pub fn get_edge(&self, edge_id: &str) -> Option<&HyperEdge> {
        self.edges.get(edge_id)
    }

    /// Check if a node exists in the store.
    pub fn has_node(&self, node_id: &str) -> bool {
        self.incidence.contains_key(node_id)
    }

    /// Get the index of a node for bitset-based operations.
    /// Returns None if the node is not in the store.
    pub fn node_index(&self, node_id: &str) -> Option<usize> {
        // Deterministic ordering via sorted keys
        let keys: Vec<&str> = {
            let mut k: Vec<&str> = self.incidence.keys().map(|s| s.as_str()).collect();
            k.sort();
            k
        };
        keys.iter().position(|&k| k == node_id)
    }

    /// Return all node IDs in the store (sorted).
    pub fn all_nodes(&self) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self.incidence.keys().cloned().collect();
        nodes.sort();
        nodes
    }
}

impl Default for HyperGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_store() -> HyperGraphStore {
        let mut store = HyperGraphStore::new();
        store.add_edge(HyperEdge::new(
            vec!["A".into(), "B".into(), "C".into()],
            HyperEdgeType::ConceptCluster,
        ));
        store.add_edge(HyperEdge::new(
            vec!["C".into(), "D".into(), "E".into()],
            HyperEdgeType::CausalChain,
        ));
        store
    }

    #[test]
    fn test_add_and_counts() {
        let store = sample_store();
        assert_eq!(store.edge_count(), 2);
        assert_eq!(store.node_count(), 5); // A, B, C, D, E
    }

    #[test]
    fn test_neighbors() {
        let store = sample_store();
        let n = store.neighbors("A");
        assert!(n.contains("B"));
        assert!(n.contains("C"));
        assert!(!n.contains("D")); // D requires 2 hops
        assert!(!n.contains("A")); // Self excluded
    }

    #[test]
    fn test_edges_of() {
        let store = sample_store();
        assert_eq!(store.edges_of("C").len(), 2); // C is in both edges
        assert_eq!(store.edges_of("A").len(), 1);
        assert_eq!(store.edges_of("Z").len(), 0); // Nonexistent
    }

    #[test]
    fn test_query_by_type() {
        let store = sample_store();
        let clusters = store.query_by_type(&HyperEdgeType::ConceptCluster);
        assert_eq!(clusters.len(), 1);
        let chains = store.query_by_type(&HyperEdgeType::CausalChain);
        assert_eq!(chains.len(), 1);
        let bridges = store.query_by_type(&HyperEdgeType::CrossDomainBridge);
        assert_eq!(bridges.len(), 0);
    }

    #[test]
    fn test_remove_edge() {
        let mut store = sample_store();
        let edge_ids: Vec<String> = store.edges.keys().cloned().collect();
        store.remove_edge(&edge_ids[0]);
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_mean_cardinality() {
        let store = sample_store();
        assert!((store.mean_cardinality() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mean_cardinality_empty() {
        let store = HyperGraphStore::new();
        assert!((store.mean_cardinality() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_has_node() {
        let store = sample_store();
        assert!(store.has_node("A"));
        assert!(!store.has_node("Z"));
    }
}
