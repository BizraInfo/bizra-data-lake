//! BFS traversal for hypergraph reachability.
//!
//! Uses a `HashSet`-based visited tracker for O(1) membership checks.
//! Future optimisation: replace with bitset for SIMD-friendly bulk operations
//! when the node count exceeds ~10,000.
//!
//! Standing on Giants: Berge (hypergraph traversal), Besta (GoT reachability)

use std::collections::HashSet;

use crate::hyperedge::NodeId;
use crate::store::HyperGraphStore;

/// Collect all nodes reachable from `seeds` within `max_hops` via hyperedge traversal.
///
/// The seed nodes themselves are always included in the result.
///
/// # Arguments
///
/// * `store` - The hypergraph to traverse.
/// * `seeds` - Starting node IDs.
/// * `max_hops` - Maximum number of hyperedge hops (0 = seeds only).
///
/// # Returns
///
/// Sorted vector of all reachable node IDs (including seeds).
pub fn bfs_reachable(store: &HyperGraphStore, seeds: &[NodeId], max_hops: usize) -> Vec<NodeId> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut frontier: Vec<NodeId> = Vec::new();

    for seed in seeds {
        if store.has_node(seed) {
            visited.insert(seed.clone());
            frontier.push(seed.clone());
        }
    }

    for _hop in 0..max_hops {
        let mut next_frontier = Vec::new();
        for node in &frontier {
            for edge in store.edges_of(node) {
                for member in &edge.members {
                    if !visited.contains(member) {
                        visited.insert(member.clone());
                        next_frontier.push(member.clone());
                    }
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    let mut result: Vec<NodeId> = visited.into_iter().collect();
    result.sort();
    result
}

/// Extract a subgraph containing all edges reachable within `max_hops` from `seeds`.
///
/// Returns a new `HyperGraphStore` containing only the visited edges.
pub fn subgraph_extract(
    store: &HyperGraphStore,
    seeds: &[NodeId],
    max_hops: usize,
) -> HyperGraphStore {
    let mut visited_nodes: HashSet<NodeId> = HashSet::new();
    let mut visited_edge_ids: HashSet<String> = HashSet::new();
    let mut frontier: Vec<NodeId> = Vec::new();

    for seed in seeds {
        if store.has_node(seed) {
            visited_nodes.insert(seed.clone());
            frontier.push(seed.clone());
        }
    }

    for _hop in 0..max_hops {
        let mut next_frontier = Vec::new();
        for node in &frontier {
            for edge in store.edges_of(node) {
                if !visited_edge_ids.contains(&edge.id) {
                    visited_edge_ids.insert(edge.id.clone());
                    for member in &edge.members {
                        if !visited_nodes.contains(member) {
                            visited_nodes.insert(member.clone());
                            next_frontier.push(member.clone());
                        }
                    }
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    let mut sub = HyperGraphStore::new();
    for eid in &visited_edge_ids {
        if let Some(edge) = store.get_edge(eid) {
            sub.add_edge(edge.clone());
        }
    }
    sub
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperedge::{HyperEdge, HyperEdgeType};

    fn linear_graph() -> HyperGraphStore {
        // A-B-C (via edge1), C-D (via edge2), D-E-F (via edge3)
        let mut store = HyperGraphStore::new();
        store.add_edge(HyperEdge::new(
            vec!["A".into(), "B".into(), "C".into()],
            HyperEdgeType::ConceptCluster,
        ));
        store.add_edge(HyperEdge::new(
            vec!["C".into(), "D".into()],
            HyperEdgeType::CausalChain,
        ));
        store.add_edge(HyperEdge::new(
            vec!["D".into(), "E".into(), "F".into()],
            HyperEdgeType::ConceptCluster,
        ));
        store
    }

    #[test]
    fn test_bfs_0_hops() {
        let store = linear_graph();
        let reachable = bfs_reachable(&store, &["A".into()], 0);
        assert_eq!(reachable, vec!["A".to_string()]);
    }

    #[test]
    fn test_bfs_1_hop() {
        let store = linear_graph();
        let reachable = bfs_reachable(&store, &["A".into()], 1);
        // A → {A, B, C} via edge1
        assert_eq!(reachable, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_bfs_2_hops() {
        let store = linear_graph();
        let reachable = bfs_reachable(&store, &["A".into()], 2);
        // A → {A, B, C} → {D} via edge2
        assert_eq!(reachable, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn test_bfs_full_traversal() {
        let store = linear_graph();
        let reachable = bfs_reachable(&store, &["A".into()], 10);
        assert_eq!(reachable, vec!["A", "B", "C", "D", "E", "F"]);
    }

    #[test]
    fn test_bfs_nonexistent_seed() {
        let store = linear_graph();
        let reachable = bfs_reachable(&store, &["Z".into()], 5);
        assert!(reachable.is_empty());
    }

    #[test]
    fn test_bfs_multiple_seeds() {
        let store = linear_graph();
        let reachable = bfs_reachable(&store, &["A".into(), "F".into()], 1);
        // A → {A, B, C}, F → {D, E, F}
        assert_eq!(reachable, vec!["A", "B", "C", "D", "E", "F"]);
    }

    #[test]
    fn test_subgraph_1hop() {
        let store = linear_graph();
        let sub = subgraph_extract(&store, &["A".into()], 1);
        assert_eq!(sub.edge_count(), 1); // Only edge {A,B,C}
        assert_eq!(sub.node_count(), 3); // A, B, C
    }

    #[test]
    fn test_subgraph_2hops() {
        let store = linear_graph();
        let sub = subgraph_extract(&store, &["A".into()], 2);
        assert_eq!(sub.edge_count(), 2); // {A,B,C} + {C,D}
        assert_eq!(sub.node_count(), 4); // A, B, C, D
    }
}
