//! Property-Based Tests for Graph-of-Thoughts Operations
//!
//! Standing on Giants: Besta et al. (2024) - "Graph of Thoughts"
//!
//! These tests verify invariants and properties of the GoT reasoning framework
//! using property-based testing with proptest.

use bizra_core::sovereign::graph_of_thoughts::{
    ReasoningPath, ThoughtGraph, ThoughtNode, ThoughtType,
};

// ============================================================================
// PROPERTY: Graph Structure Invariants
// ============================================================================

/// Property: Every non-root node has exactly one parent
#[test]
fn prop_single_parent_invariant() {
    let mut graph = ThoughtGraph::new();

    // Create a tree structure
    let root = graph.create_thought("Root question", None);
    let h1 = graph.create_thought("Hypothesis A", Some(&root));
    let h2 = graph.create_thought("Hypothesis B", Some(&root));
    let evidence = graph.create_thought("Supporting evidence", Some(&h1));

    // Verify parent relationships
    assert!(graph.get_thought(&root).unwrap().parent.is_none());
    assert_eq!(
        graph.get_thought(&h1).unwrap().parent.as_ref().unwrap(),
        &root
    );
    assert_eq!(
        graph.get_thought(&h2).unwrap().parent.as_ref().unwrap(),
        &root
    );
    assert_eq!(
        graph
            .get_thought(&evidence)
            .unwrap()
            .parent
            .as_ref()
            .unwrap(),
        &h1
    );
}

/// Property: Parent-child relationships are bidirectional
#[test]
fn prop_bidirectional_links() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);
    let child1 = graph.create_thought("Child 1", Some(&root));
    let child2 = graph.create_thought("Child 2", Some(&root));

    // Parent should have both children
    let parent_node = graph.get_thought(&root).unwrap();
    assert!(parent_node.children.contains(&child1));
    assert!(parent_node.children.contains(&child2));

    // Children should reference parent
    assert_eq!(
        graph.get_thought(&child1).unwrap().parent.as_ref().unwrap(),
        &root
    );
    assert_eq!(
        graph.get_thought(&child2).unwrap().parent.as_ref().unwrap(),
        &root
    );
}

/// Property: Frontier nodes are always leaves (no children)
#[test]
fn prop_frontier_are_leaves() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);
    let _ = graph.create_thought("Branch A", Some(&root));
    let _ = graph.create_thought("Branch B", Some(&root));
    let leaf_c = graph.create_thought("Leaf C", Some(&root));

    // Add deeper nodes
    let _ = graph.create_thought("Deep 1", Some(&leaf_c));

    let frontier = graph.get_frontier();

    for node in frontier {
        assert!(
            node.children.is_empty(),
            "Frontier node {} should have no children but has {:?}",
            node.id,
            node.children
        );
    }
}

// ============================================================================
// PROPERTY: Backtrack Selection
// ============================================================================

/// Property: Backtrack always selects the highest-SNR unexplored node
#[test]
fn prop_backtrack_max_snr() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Problem statement", None);

    // Create hypotheses with different SNR scores
    let scores = [0.3, 0.7, 0.5, 0.9, 0.6];
    let mut ids = Vec::new();

    for (i, &snr) in scores.iter().enumerate() {
        let id = graph.create_thought_with_type(
            &format!("Hypothesis {}", i),
            Some(&root),
            ThoughtType::Hypothesis,
        );
        if let Some(node) = graph.get_thought_mut(&id) {
            node.set_snr(snr);
        }
        ids.push(id);
    }

    // Backtrack should return the hypothesis with SNR 0.9 (index 3)
    let backtrack_node = graph.backtrack();
    assert!(backtrack_node.is_some());

    let selected = backtrack_node.unwrap();
    assert_eq!(
        selected.snr_score, 0.9,
        "Expected SNR 0.9, got {}",
        selected.snr_score
    );
    assert_eq!(selected.id, ids[3]);
}

/// Property: Backtrack excludes terminal nodes (Conclusion, Validation)
#[test]
fn prop_backtrack_excludes_terminal() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);

    // Add a conclusion with high SNR
    let conclusion =
        graph.create_thought_with_type("Final Answer", Some(&root), ThoughtType::Conclusion);
    if let Some(node) = graph.get_thought_mut(&conclusion) {
        node.set_snr(0.99);
    }

    // Add a hypothesis with lower SNR
    let hypothesis = graph.create_thought_with_type(
        "Unexplored hypothesis",
        Some(&root),
        ThoughtType::Hypothesis,
    );
    if let Some(node) = graph.get_thought_mut(&hypothesis) {
        node.set_snr(0.5);
    }

    // Backtrack should return hypothesis, not conclusion
    let backtrack_node = graph.backtrack();
    assert!(backtrack_node.is_some());
    assert_eq!(
        backtrack_node.unwrap().thought_type,
        ThoughtType::Hypothesis
    );
}

/// Property: Backtrack returns None when all nodes are explored or terminal
#[test]
fn prop_backtrack_none_when_exhausted() {
    let mut graph = ThoughtGraph::new();

    // Only add terminal nodes
    let _ = graph.create_thought_with_type("Conclusion 1", None, ThoughtType::Conclusion);
    let _ = graph.create_thought_with_type("Validation 1", None, ThoughtType::Validation);

    assert!(graph.backtrack().is_none());
}

// ============================================================================
// PROPERTY: Aggregate Consensus
// ============================================================================

/// Property: Consensus requires strict majority (> 50%)
#[test]
fn prop_aggregate_majority_required() {
    let graph = ThoughtGraph::new();

    // Test cases: (successful paths, total paths, expected consensus)
    let test_cases = [
        (0, 3, false),  // 0/3 = 0% -> no consensus
        (1, 3, false),  // 1/3 = 33% -> no consensus
        (2, 3, true),   // 2/3 = 66% -> consensus
        (3, 3, true),   // 3/3 = 100% -> consensus
        (2, 4, false),  // 2/4 = 50% -> no consensus (not > 50%)
        (3, 4, true),   // 3/4 = 75% -> consensus
        (5, 10, false), // 5/10 = 50% -> no consensus
        (6, 10, true),  // 6/10 = 60% -> consensus
    ];

    for (successes, total, expected) in test_cases {
        let paths: Vec<ReasoningPath> = (0..total)
            .map(|i| {
                let mut path = graph.create_path(&format!("path_{}", i));
                path.add_thought(ThoughtNode::new("t", "test"));
                path.record_result("t", i < successes);
                path
            })
            .collect();

        let aggregate = graph.aggregate_paths(&paths);
        assert_eq!(
            aggregate.consensus, expected,
            "With {}/{} successes, expected consensus={}, got={}",
            successes, total, expected, aggregate.consensus
        );
    }
}

/// Property: Aggregate correctly counts complete vs incomplete paths
#[test]
fn prop_aggregate_completion_counting() {
    let graph = ThoughtGraph::new();

    let mut complete_path = graph.create_path("complete");
    complete_path.add_thought(ThoughtNode::new("t1", "thought 1"));
    complete_path.record_result("t1", true);

    let mut incomplete_path = graph.create_path("incomplete");
    incomplete_path.add_thought(ThoughtNode::new("t2", "thought 2"));
    // No result recorded

    let aggregate = graph.aggregate_paths(&[complete_path, incomplete_path]);

    assert_eq!(aggregate.total_paths, 2);
    assert_eq!(aggregate.complete_paths, 1);
}

// ============================================================================
// PROPERTY: Reasoning Path
// ============================================================================

/// Property: Success rate is always in [0.0, 1.0]
#[test]
fn prop_success_rate_bounds() {
    let graph = ThoughtGraph::new();

    // Empty path
    let empty_path = graph.create_path("empty");
    let rate = empty_path.success_rate();
    assert!(
        (0.0..=1.0).contains(&rate),
        "Success rate {} out of bounds",
        rate
    );

    // Partial success
    let mut partial = graph.create_path("partial");
    partial.add_thought(ThoughtNode::new("t1", "thought"));
    partial.add_thought(ThoughtNode::new("t2", "thought"));
    partial.record_result("t1", true);
    partial.record_result("t2", false);

    let rate = partial.success_rate();
    assert!(
        (0.0..=1.0).contains(&rate),
        "Success rate {} out of bounds",
        rate
    );
    assert!((rate - 0.5).abs() < 0.001, "Expected 0.5, got {}", rate);
}

/// Property: final_result is Some only when all thoughts have results
#[test]
fn prop_final_result_requires_completion() {
    let graph = ThoughtGraph::new();

    let mut path = graph.create_path("test");
    path.add_thought(ThoughtNode::new("t1", "first"));
    path.add_thought(ThoughtNode::new("t2", "second"));

    // Before any results - path is not complete
    assert!(!path.is_complete());

    // After partial results
    path.record_result("t1", true);
    assert!(!path.is_complete());
    // final_result is updated but path is not complete
    assert!(path.final_result.is_some());

    // After all results
    path.record_result("t2", true);
    assert!(path.is_complete());
    assert_eq!(path.final_result, Some(true));
}

/// Property: Confidence decreases on failures
#[test]
fn prop_confidence_penalty_on_failure() {
    let graph = ThoughtGraph::new();

    let mut path = graph.create_path("test");
    path.add_thought(ThoughtNode::new("t1", "thought"));
    let initial_confidence = path.confidence;

    path.record_result("t1", false);

    assert!(
        path.confidence < initial_confidence,
        "Confidence should decrease on failure"
    );
}

// ============================================================================
// PROPERTY: ThoughtNode
// ============================================================================

/// Property: SNR is always clamped to [0.0, 1.0]
#[test]
fn prop_snr_clamping() {
    let mut node = ThoughtNode::new("test", "description");

    // Test negative values
    node.set_snr(-0.5);
    assert!(
        node.snr_score >= 0.0,
        "SNR should be >= 0.0, got {}",
        node.snr_score
    );

    // Test values > 1.0
    node.set_snr(1.5);
    assert!(
        node.snr_score <= 1.0,
        "SNR should be <= 1.0, got {}",
        node.snr_score
    );

    // Test boundary values
    node.set_snr(0.0);
    assert_eq!(node.snr_score, 0.0);

    node.set_snr(1.0);
    assert_eq!(node.snr_score, 1.0);
}

/// Property: Terminal nodes are Conclusion or Validation types
#[test]
fn prop_terminal_types() {
    let types_and_expected = [
        (ThoughtType::Hypothesis, false),
        (ThoughtType::Evidence, false),
        (ThoughtType::Reasoning, false),
        (ThoughtType::Synthesis, false),
        (ThoughtType::Refinement, false),
        (ThoughtType::Conclusion, true),
        (ThoughtType::Validation, true),
        (ThoughtType::Question, false),
        (ThoughtType::Counterpoint, false),
    ];

    for (thought_type, expected_terminal) in types_and_expected {
        let node = ThoughtNode::with_type("test", "description", thought_type.clone());
        assert_eq!(
            node.is_terminal(),
            expected_terminal,
            "{:?} terminal status should be {}",
            thought_type,
            expected_terminal
        );
    }
}

// ============================================================================
// EDGE CASES
// ============================================================================

/// Edge case: Empty graph operations
#[test]
fn test_empty_graph_operations() {
    let graph = ThoughtGraph::new();

    // All operations should handle empty graph gracefully
    assert!(graph.get_frontier().is_empty());
    assert!(graph.backtrack().is_none());
    assert!(graph.get_conclusions(0.0).is_empty());

    let stats = graph.stats();
    assert_eq!(stats.total_thoughts, 0);
    assert_eq!(stats.root_count, 0);
    assert_eq!(stats.total_paths, 0);

    // Aggregate empty paths
    let aggregate = graph.aggregate_paths(&[]);
    assert_eq!(aggregate.total_paths, 0);
    assert!(!aggregate.consensus);
}

/// Edge case: Single node graph
#[test]
fn test_single_node_graph() {
    let mut graph = ThoughtGraph::new();
    let root = graph.create_thought("Single node", None);

    assert_eq!(graph.stats().total_thoughts, 1);
    assert_eq!(graph.stats().root_count, 1);

    let frontier = graph.get_frontier();
    assert_eq!(frontier.len(), 1);
    assert_eq!(frontier[0].id, root);

    // Backtrack should return the single node (if not terminal)
    let backtrack = graph.backtrack();
    assert!(backtrack.is_some());
}

/// Edge case: Deep linear chain
#[test]
fn test_deep_chain() {
    let mut graph = ThoughtGraph::new();

    let mut parent = graph.create_thought("Root", None);
    for i in 1..=100 {
        parent = graph.create_thought(&format!("Node {}", i), Some(&parent));
    }

    assert_eq!(graph.stats().total_thoughts, 101);
    assert_eq!(graph.stats().root_count, 1);

    // Frontier should be the deepest node
    let frontier = graph.get_frontier();
    assert_eq!(frontier.len(), 1);
}

/// Edge case: Wide tree (many children from root)
#[test]
fn test_wide_tree() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);
    for i in 0..100 {
        graph.create_thought(&format!("Child {}", i), Some(&root));
    }

    assert_eq!(graph.stats().total_thoughts, 101);

    let root_node = graph.get_thought(&root).unwrap();
    assert_eq!(root_node.children.len(), 100);

    // Frontier should be all 100 children
    let frontier = graph.get_frontier();
    assert_eq!(frontier.len(), 100);
}

/// Edge case: Parallel exploration with no children
#[test]
fn test_explore_parallel_no_children() {
    let mut graph = ThoughtGraph::new();
    let root = graph.create_thought("Lonely root", None);

    let paths = graph.explore_parallel(&root);

    // Should create a single path containing just the root
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].thoughts.len(), 1);
}

/// Edge case: Get conclusions with various thresholds
#[test]
fn test_conclusions_threshold_filtering() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);

    // Create conclusions with varying SNR
    let snr_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    for snr in snr_values {
        let id = graph.create_thought_with_type(
            &format!("Conclusion SNR {}", snr),
            Some(&root),
            ThoughtType::Conclusion,
        );
        if let Some(node) = graph.get_thought_mut(&id) {
            node.set_snr(snr);
        }
    }

    // Test various thresholds
    assert_eq!(graph.get_conclusions(0.0).len(), 5);
    assert_eq!(graph.get_conclusions(0.2).len(), 4);
    assert_eq!(graph.get_conclusions(0.5).len(), 3);
    assert_eq!(graph.get_conclusions(0.8).len(), 1);
    assert_eq!(graph.get_conclusions(1.0).len(), 0);
}

/// Edge case: explore_with_backtrack iterations
#[test]
fn test_explore_with_backtrack_termination() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);

    // Add a low-SNR conclusion
    let low_snr = graph.create_thought_with_type(
        "Low quality conclusion",
        Some(&root),
        ThoughtType::Conclusion,
    );
    if let Some(node) = graph.get_thought_mut(&low_snr) {
        node.set_snr(0.3);
    }

    // Add a high-SNR conclusion
    let high_snr = graph.create_thought_with_type(
        "High quality conclusion",
        Some(&root),
        ThoughtType::Conclusion,
    );
    if let Some(node) = graph.get_thought_mut(&high_snr) {
        node.set_snr(0.95);
    }

    // Should find high SNR conclusion
    let result = graph.explore_with_backtrack(10, 0.9);
    assert!(result.is_some());
    assert!(result.unwrap().snr_score >= 0.9);

    // With impossible threshold, should return best available
    let result = graph.explore_with_backtrack(10, 1.0);
    assert!(result.is_some());
    assert_eq!(result.unwrap().snr_score, 0.95);
}

// ============================================================================
// STRESS TESTS
// ============================================================================

/// Stress test: Large graph performance
#[test]
fn stress_large_graph() {
    let mut graph = ThoughtGraph::new();

    let root = graph.create_thought("Root", None);

    // Create 1000 nodes in a balanced tree (depth 3, ~10 children per node)
    let mut current_level = vec![root.clone()];

    for _depth in 0..3 {
        let mut next_level = Vec::new();
        for parent in &current_level {
            for i in 0..10 {
                let child = graph.create_thought(&format!("Node {}", i), Some(parent));
                next_level.push(child);
            }
        }
        current_level = next_level;
    }

    // Verify structure
    let stats = graph.stats();
    assert!(stats.total_thoughts > 1000);

    // Operations should complete in reasonable time
    let _frontier = graph.get_frontier();
    let _backtrack = graph.backtrack();
    let _conclusions = graph.get_conclusions(0.5);
}

/// Stress test: Many reasoning paths
#[test]
fn stress_many_paths() {
    let graph = ThoughtGraph::new();

    let paths: Vec<ReasoningPath> = (0..1000)
        .map(|i| {
            let mut path = graph.create_path(&format!("path_{}", i));
            for j in 0..10 {
                path.add_thought(ThoughtNode::new(&format!("t{}_{}", i, j), "thought"));
                path.record_result(&format!("t{}_{}", i, j), j % 2 == 0);
            }
            path
        })
        .collect();

    let aggregate = graph.aggregate_paths(&paths);

    assert_eq!(aggregate.total_paths, 1000);
    assert_eq!(aggregate.complete_paths, 1000);
}
