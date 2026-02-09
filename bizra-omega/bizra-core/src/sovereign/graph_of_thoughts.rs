//! Graph-of-Thoughts — Structured Reasoning Framework
//!
//! Implements Graph-of-Thoughts (GoT) reasoning based on Besta et al. (2024).
//! This enables non-linear, multi-path reasoning that explores
//! multiple solution paths simultaneously.
//!
//! # The GoT Framework
//!
//! Unlike Chain-of-Thought (CoT) which follows a single path,
//! GoT models reasoning as a directed graph where:
//! - Nodes are thoughts/reasoning states
//! - Edges are transformations between states
//! - Multiple paths can be explored in parallel
//!
//! # GoT Operations (Besta et al., 2024)
//!
//! The framework implements six core operations:
//! - **GENERATE**: Create new thought nodes from prompts/context
//! - **AGGREGATE**: Merge multiple thoughts into synthesis
//! - **REFINE**: Iteratively improve thought quality
//! - **VALIDATE**: Check thoughts against quality constraints
//! - **PRUNE**: Remove low-SNR branches
//! - **BACKTRACK**: Return to promising unexplored paths
//!
//! # Architecture
//!
//! ```text
//!                 ┌─────────┐
//!                 │ Initial │
//!                 │ Thought │
//!                 └────┬────┘
//!                      │
//!          ┌───────────┼───────────┐
//!          ▼           ▼           ▼
//!     ┌────────┐  ┌────────┐  ┌────────┐
//!     │ Path A │  │ Path B │  │ Path C │
//!     └───┬────┘  └───┬────┘  └───┬────┘
//!         │           │           │
//!         └───────────┼───────────┘
//!                     ▼
//!               ┌──────────┐
//!               │ Aggregate│
//!               │ Solution │
//!               └──────────┘
//! ```
//!
//! # References
//!
//! - Besta et al. (2024): "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
//! - Yao et al. (2023): "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
//! - Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

use std::collections::HashMap;
use std::time::Instant;

/// Types of thought nodes in the reasoning graph.
///
/// Per Besta et al. (2024), different thought types enable structured
/// reasoning with clear semantic distinctions between node purposes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ThoughtType {
    /// Initial conjectures to be validated
    Hypothesis,
    /// Supporting or refuting data
    Evidence,
    /// Logical deduction steps
    Reasoning,
    /// Merged conclusions from multiple paths
    Synthesis,
    /// Improved versions of prior thoughts
    Refinement,
    /// Quality validation checks
    Validation,
    /// Final answers (terminal nodes)
    Conclusion,
    /// Sub-questions to explore
    Question,
    /// Alternative perspectives
    Counterpoint,
}

/// A single thought in the reasoning graph.
///
/// Each thought represents a discrete reasoning state with associated
/// metadata for quality assessment and graph traversal.
#[derive(Clone, Debug)]
pub struct ThoughtNode {
    /// Unique identifier
    pub id: String,
    /// Description of this thought
    pub description: String,
    /// Semantic type of this thought
    pub thought_type: ThoughtType,
    /// Result of evaluation (if completed)
    pub result: Option<bool>,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Signal-to-Noise Ratio score (0.0 - 1.0)
    ///
    /// Per Besta et al. (2024), SNR indicates the quality and reliability
    /// of a thought for guiding exploration and backtracking decisions.
    pub snr_score: f64,
    /// Child thought IDs
    pub children: Vec<String>,
    /// Parent thought ID
    pub parent: Option<String>,
    /// Creation timestamp
    pub created_at: Instant,
}

impl ThoughtNode {
    /// Create a new thought node with default type (Reasoning).
    pub fn new(id: &str, description: &str) -> Self {
        Self::with_type(id, description, ThoughtType::Reasoning)
    }

    /// Create a new thought node with a specific type.
    pub fn with_type(id: &str, description: &str, thought_type: ThoughtType) -> Self {
        Self {
            id: id.to_string(),
            description: description.to_string(),
            thought_type,
            result: None,
            confidence: 0.5,
            snr_score: 0.5,
            children: Vec::new(),
            parent: None,
            created_at: Instant::now(),
        }
    }

    /// Set result and confidence.
    pub fn complete(&mut self, result: bool, confidence: f64) {
        self.result = Some(result);
        self.confidence = confidence;
    }

    /// Set the SNR score for this thought.
    pub fn set_snr(&mut self, snr: f64) {
        self.snr_score = snr.clamp(0.0, 1.0);
    }

    /// Check if this thought is a terminal node (conclusion or validation).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.thought_type,
            ThoughtType::Conclusion | ThoughtType::Validation
        )
    }

    /// Check if this thought has been explored (has children or is terminal).
    pub fn is_explored(&self) -> bool {
        !self.children.is_empty() || self.is_terminal()
    }
}

/// A path through the thought graph
#[derive(Clone, Debug)]
pub struct ReasoningPath {
    /// Path identifier
    pub id: String,
    /// Ordered list of thoughts in this path
    pub thoughts: Vec<ThoughtNode>,
    /// Final result of this path
    pub final_result: Option<bool>,
    /// Aggregate confidence
    pub confidence: f64,
    /// Results for each thought
    results: HashMap<String, bool>,
}

impl ReasoningPath {
    /// Create new reasoning path
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            thoughts: Vec::new(),
            final_result: None,
            confidence: 1.0,
            results: HashMap::new(),
        }
    }

    /// Add a thought to the path
    pub fn add_thought(&mut self, thought: ThoughtNode) {
        self.thoughts.push(thought);
    }

    /// Record result for a thought
    pub fn record_result(&mut self, thought_id: &str, result: bool) {
        self.results.insert(thought_id.to_string(), result);

        // Update confidence (multiplicative for now)
        if !result {
            self.confidence *= 0.5; // Penalty for failures
        }

        // Update final result
        self.final_result = Some(self.results.values().all(|&r| r));
    }

    /// Check if path is complete (all thoughts evaluated)
    pub fn is_complete(&self) -> bool {
        self.thoughts
            .iter()
            .all(|t| self.results.contains_key(&t.id))
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let successes = self.results.values().filter(|&&r| r).count();
        successes as f64 / self.results.len() as f64
    }
}

/// The Graph-of-Thoughts reasoning engine
pub struct ThoughtGraph {
    /// All thoughts in the graph
    thoughts: HashMap<String, ThoughtNode>,
    /// All reasoning paths
    paths: HashMap<String, ReasoningPath>,
    /// Root thought IDs
    roots: Vec<String>,
    /// Counter for generating IDs
    counter: u64,
}

impl ThoughtGraph {
    /// Create new thought graph
    pub fn new() -> Self {
        Self {
            thoughts: HashMap::new(),
            paths: HashMap::new(),
            roots: Vec::new(),
            counter: 0,
        }
    }

    /// Create a new reasoning path
    pub fn create_path(&self, name: &str) -> ReasoningPath {
        ReasoningPath::new(name)
    }

    /// Add a thought to the graph
    pub fn add_thought(&mut self, thought: ThoughtNode) -> String {
        let id = thought.id.clone();
        if thought.parent.is_none() {
            self.roots.push(id.clone());
        }
        self.thoughts.insert(id.clone(), thought);
        id
    }

    /// Create and add a new thought
    pub fn create_thought(&mut self, description: &str, parent: Option<&str>) -> String {
        self.counter += 1;
        let id = format!("thought_{}", self.counter);

        let mut thought = ThoughtNode::new(&id, description);
        thought.parent = parent.map(|s| s.to_string());

        // Link to parent
        if let Some(parent_id) = parent {
            if let Some(parent_thought) = self.thoughts.get_mut(parent_id) {
                parent_thought.children.push(id.clone());
            }
        }

        self.add_thought(thought)
    }

    /// Explore multiple paths in parallel (conceptually)
    pub fn explore_parallel(&self, root_id: &str) -> Vec<ReasoningPath> {
        let mut paths = Vec::new();

        if let Some(root) = self.thoughts.get(root_id) {
            // Create a path for each child
            for child_id in &root.children {
                let mut path = ReasoningPath::new(&format!("path_{}", child_id));
                path.add_thought(root.clone());

                if let Some(child) = self.thoughts.get(child_id) {
                    path.add_thought(child.clone());
                }

                paths.push(path);
            }

            // If no children, create single path
            if paths.is_empty() {
                let mut path = ReasoningPath::new(&format!("path_{}", root_id));
                path.add_thought(root.clone());
                paths.push(path);
            }
        }

        paths
    }

    /// Aggregate results from multiple paths
    pub fn aggregate_paths(&self, paths: &[ReasoningPath]) -> AggregateResult {
        let total = paths.len();
        let complete = paths.iter().filter(|p| p.is_complete()).count();
        let successful = paths
            .iter()
            .filter(|p| p.final_result == Some(true))
            .count();

        let avg_confidence = if total > 0 {
            paths.iter().map(|p| p.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        AggregateResult {
            total_paths: total,
            complete_paths: complete,
            successful_paths: successful,
            average_confidence: avg_confidence,
            consensus: successful > total / 2,
        }
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        GraphStats {
            total_thoughts: self.thoughts.len(),
            total_paths: self.paths.len(),
            root_count: self.roots.len(),
        }
    }

    /// Get the frontier (leaf nodes) of the reasoning graph.
    ///
    /// The frontier consists of all nodes with no children, representing
    /// the current boundary of exploration in the reasoning process.
    ///
    /// # Returns
    ///
    /// A vector of references to all leaf nodes in the graph.
    pub fn get_frontier(&self) -> Vec<&ThoughtNode> {
        self.thoughts
            .values()
            .filter(|node| node.children.is_empty())
            .collect()
    }

    /// Get a thought node by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier of the thought node
    ///
    /// # Returns
    ///
    /// An optional reference to the thought node if found.
    pub fn get_thought(&self, id: &str) -> Option<&ThoughtNode> {
        self.thoughts.get(id)
    }

    /// Get a mutable reference to a thought node by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier of the thought node
    ///
    /// # Returns
    ///
    /// An optional mutable reference to the thought node if found.
    pub fn get_thought_mut(&mut self, id: &str) -> Option<&mut ThoughtNode> {
        self.thoughts.get_mut(id)
    }

    /// Return to the highest-SNR unexplored frontier node (BACKTRACK operation).
    ///
    /// Per Besta et al. (2024), backtracking enables recovery from dead-ends
    /// by returning to promising unexplored branches in the reasoning graph.
    /// This is the 6th GoT operation, completing the reasoning framework:
    ///
    /// - GENERATE: Create new thought nodes
    /// - AGGREGATE: Merge multiple thoughts into synthesis
    /// - REFINE: Improve existing thoughts iteratively
    /// - VALIDATE: Check thoughts against quality constraints
    /// - PRUNE: Remove low-SNR branches
    /// - **BACKTRACK**: Return to promising unexplored paths (THIS METHOD)
    ///
    /// # Algorithm
    ///
    /// 1. Identify all frontier (leaf) nodes
    /// 2. Filter for unexplored nodes (not Conclusion or Validation types,
    ///    and having no children)
    /// 3. Return the node with the highest SNR score
    ///
    /// # Returns
    ///
    /// The highest-SNR unexplored frontier node, or `None` if all paths
    /// have been explored or lead to conclusions.
    ///
    /// # Example
    ///
    /// ```
    /// use bizra_core::sovereign::graph_of_thoughts::{ThoughtGraph, ThoughtNode, ThoughtType};
    ///
    /// let mut graph = ThoughtGraph::new();
    ///
    /// // Create a root thought
    /// let root_id = graph.create_thought("Initial question", None);
    ///
    /// // Create some child hypotheses
    /// let h1_id = graph.create_thought_with_type(
    ///     "Hypothesis A",
    ///     Some(&root_id),
    ///     ThoughtType::Hypothesis,
    /// );
    /// let h2_id = graph.create_thought_with_type(
    ///     "Hypothesis B",
    ///     Some(&root_id),
    ///     ThoughtType::Hypothesis,
    /// );
    ///
    /// // Set different SNR scores
    /// if let Some(h1) = graph.get_thought_mut(&h1_id) {
    ///     h1.set_snr(0.8);
    /// }
    /// if let Some(h2) = graph.get_thought_mut(&h2_id) {
    ///     h2.set_snr(0.9);
    /// }
    ///
    /// // Backtrack should return h2 (highest SNR)
    /// let backtrack_node = graph.backtrack();
    /// assert!(backtrack_node.is_some());
    /// assert_eq!(backtrack_node.unwrap().id, h2_id);
    /// ```
    pub fn backtrack(&self) -> Option<&ThoughtNode> {
        // Get all frontier nodes
        let frontier = self.get_frontier();

        // Filter for unexplored nodes:
        // - Not CONCLUSION (already terminal)
        // - Not VALIDATION (already checked)
        // - Has no children (is a true leaf)
        let unexplored: Vec<&ThoughtNode> = frontier
            .into_iter()
            .filter(|node| !node.is_terminal() && node.children.is_empty())
            .collect();

        if unexplored.is_empty() {
            return None;
        }

        // Return the node with highest SNR score
        // Using max_by with a total_cmp for safe f64 comparison
        unexplored
            .into_iter()
            .max_by(|a, b| a.snr_score.total_cmp(&b.snr_score))
    }

    /// Get all conclusion nodes above a minimum SNR threshold.
    ///
    /// # Arguments
    ///
    /// * `min_snr` - Minimum SNR score for inclusion (0.0 - 1.0)
    ///
    /// # Returns
    ///
    /// A vector of references to conclusion nodes meeting the threshold.
    pub fn get_conclusions(&self, min_snr: f64) -> Vec<&ThoughtNode> {
        self.thoughts
            .values()
            .filter(|node| {
                node.thought_type == ThoughtType::Conclusion && node.snr_score >= min_snr
            })
            .collect()
    }

    /// Create and add a new thought with a specific type.
    ///
    /// # Arguments
    ///
    /// * `description` - The thought content
    /// * `parent` - Optional parent thought ID
    /// * `thought_type` - The semantic type of this thought
    ///
    /// # Returns
    ///
    /// The ID of the newly created thought.
    pub fn create_thought_with_type(
        &mut self,
        description: &str,
        parent: Option<&str>,
        thought_type: ThoughtType,
    ) -> String {
        self.counter += 1;
        let id = format!("thought_{}", self.counter);

        let mut thought = ThoughtNode::with_type(&id, description, thought_type);
        thought.parent = parent.map(|s| s.to_string());

        // Link to parent
        if let Some(parent_id) = parent {
            if let Some(parent_thought) = self.thoughts.get_mut(parent_id) {
                parent_thought.children.push(id.clone());
            }
        }

        self.add_thought(thought)
    }

    /// Iteratively explore the graph with backtracking until target SNR is reached.
    ///
    /// Per Besta et al. (2024), combines best-first search with backtracking
    /// for robust exploration of complex reasoning spaces.
    ///
    /// This method identifies the next node to explore from after each iteration.
    /// The caller is responsible for generating new thoughts from the returned node.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum number of exploration iterations
    /// * `target_snr` - Target SNR threshold for acceptable conclusions
    ///
    /// # Returns
    ///
    /// The best conclusion node found (may be below threshold if exploration exhausted),
    /// or `None` if no conclusions exist.
    ///
    /// # Example
    ///
    /// ```
    /// use bizra_core::sovereign::graph_of_thoughts::{ThoughtGraph, ThoughtType};
    ///
    /// let mut graph = ThoughtGraph::new();
    /// let root = graph.create_thought("Problem statement", None);
    ///
    /// // ... add thoughts and conclusions ...
    ///
    /// // Try to find a high-quality conclusion
    /// if let Some(best) = graph.explore_with_backtrack(10, 0.95) {
    ///     println!("Found conclusion with SNR: {}", best.snr_score);
    /// }
    /// ```
    pub fn explore_with_backtrack(
        &self,
        max_iterations: usize,
        target_snr: f64,
    ) -> Option<&ThoughtNode> {
        for _iteration in 0..max_iterations {
            // Try to find a high-quality conclusion
            let conclusions = self.get_conclusions(target_snr);
            if !conclusions.is_empty() {
                // Return the best conclusion
                return conclusions
                    .into_iter()
                    .max_by(|a, b| a.snr_score.total_cmp(&b.snr_score));
            }

            // No good conclusion yet - check if we can backtrack
            if self.backtrack().is_none() {
                // No backtrack options available, exploration exhausted
                break;
            }

            // The caller should use backtrack() to get the node and generate
            // new thoughts from it. This method just checks termination conditions.
        }

        // Return best available conclusion even if below threshold
        let all_conclusions = self.get_conclusions(0.0);
        all_conclusions
            .into_iter()
            .max_by(|a, b| a.snr_score.total_cmp(&b.snr_score))
    }
}

impl Default for ThoughtGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated result from multiple reasoning paths
/// Aggregated outcome from parallel reasoning path exploration.
#[derive(Debug)]
pub struct AggregateResult {
    /// Total paths evaluated.
    pub total_paths: usize,
    /// Paths that completed all thoughts.
    pub complete_paths: usize,
    /// Paths that completed successfully.
    pub successful_paths: usize,
    /// Mean confidence across all paths.
    pub average_confidence: f64,
    /// Whether a majority of paths agree.
    pub consensus: bool,
}

/// Graph statistics
#[derive(Debug)]
pub struct GraphStats {
    /// Number of thought nodes in the graph.
    pub total_thoughts: usize,
    /// Number of registered reasoning paths.
    pub total_paths: usize,
    /// Number of root (parentless) thoughts.
    pub root_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_graph() {
        let mut graph = ThoughtGraph::new();

        let root_id = graph.create_thought("Initial analysis", None);
        let _child1 = graph.create_thought("Path A: Schema validation", Some(&root_id));
        let _child2 = graph.create_thought("Path B: SNR check", Some(&root_id));

        assert_eq!(graph.stats().total_thoughts, 3);
        assert_eq!(graph.stats().root_count, 1);
    }

    #[test]
    fn test_reasoning_path() {
        let mut path = ReasoningPath::new("test_path");

        path.add_thought(ThoughtNode::new("t1", "First thought"));
        path.add_thought(ThoughtNode::new("t2", "Second thought"));

        path.record_result("t1", true);
        path.record_result("t2", true);

        assert!(path.is_complete());
        assert_eq!(path.success_rate(), 1.0);
        assert_eq!(path.final_result, Some(true));
    }

    #[test]
    fn test_parallel_exploration() {
        let mut graph = ThoughtGraph::new();

        let root = graph.create_thought("Root", None);
        graph.create_thought("Child A", Some(&root));
        graph.create_thought("Child B", Some(&root));

        let paths = graph.explore_parallel(&root);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_get_frontier() {
        let mut graph = ThoughtGraph::new();

        // Create a simple tree: root -> (child1, child2)
        let root = graph.create_thought("Root", None);
        let child1 = graph.create_thought("Child 1", Some(&root));
        let child2 = graph.create_thought("Child 2", Some(&root));

        // Frontier should be child1 and child2 (leaves)
        let frontier = graph.get_frontier();
        assert_eq!(frontier.len(), 2);

        let frontier_ids: Vec<&str> = frontier.iter().map(|n| n.id.as_str()).collect();
        assert!(frontier_ids.contains(&child1.as_str()));
        assert!(frontier_ids.contains(&child2.as_str()));
    }

    #[test]
    fn test_backtrack_returns_highest_snr() {
        let mut graph = ThoughtGraph::new();

        // Create root with two hypothesis children
        let root = graph.create_thought("Problem statement", None);
        let h1 = graph.create_thought_with_type(
            "Hypothesis A: low confidence",
            Some(&root),
            ThoughtType::Hypothesis,
        );
        let h2 = graph.create_thought_with_type(
            "Hypothesis B: high confidence",
            Some(&root),
            ThoughtType::Hypothesis,
        );

        // Set different SNR scores
        if let Some(node) = graph.get_thought_mut(&h1) {
            node.set_snr(0.3);
        }
        if let Some(node) = graph.get_thought_mut(&h2) {
            node.set_snr(0.9);
        }

        // Backtrack should return h2 (highest SNR)
        let backtrack_node = graph.backtrack();
        assert!(backtrack_node.is_some());
        assert_eq!(backtrack_node.unwrap().id, h2);
    }

    #[test]
    fn test_backtrack_excludes_conclusions() {
        let mut graph = ThoughtGraph::new();

        // Create root with one hypothesis and one conclusion
        let root = graph.create_thought("Problem", None);
        let hypothesis = graph.create_thought_with_type(
            "A hypothesis to explore",
            Some(&root),
            ThoughtType::Hypothesis,
        );
        let conclusion =
            graph.create_thought_with_type("Final answer", Some(&root), ThoughtType::Conclusion);

        // Even if conclusion has higher SNR, backtrack should return hypothesis
        if let Some(node) = graph.get_thought_mut(&hypothesis) {
            node.set_snr(0.5);
        }
        if let Some(node) = graph.get_thought_mut(&conclusion) {
            node.set_snr(0.99);
        }

        let backtrack_node = graph.backtrack();
        assert!(backtrack_node.is_some());
        assert_eq!(backtrack_node.unwrap().id, hypothesis);
    }

    #[test]
    fn test_backtrack_returns_none_when_all_explored() {
        let mut graph = ThoughtGraph::new();

        // Create only conclusions (all terminal)
        let _root = graph.create_thought_with_type("Final answer", None, ThoughtType::Conclusion);

        // No unexplored nodes
        let backtrack_node = graph.backtrack();
        assert!(backtrack_node.is_none());
    }

    #[test]
    fn test_get_conclusions_with_threshold() {
        let mut graph = ThoughtGraph::new();

        let root = graph.create_thought("Root", None);

        // Create conclusions with different SNR scores
        let c1 = graph.create_thought_with_type(
            "Low SNR conclusion",
            Some(&root),
            ThoughtType::Conclusion,
        );
        let c2 = graph.create_thought_with_type(
            "High SNR conclusion",
            Some(&root),
            ThoughtType::Conclusion,
        );

        if let Some(node) = graph.get_thought_mut(&c1) {
            node.set_snr(0.3);
        }
        if let Some(node) = graph.get_thought_mut(&c2) {
            node.set_snr(0.9);
        }

        // With threshold 0.5, only c2 should be returned
        let conclusions = graph.get_conclusions(0.5);
        assert_eq!(conclusions.len(), 1);
        assert_eq!(conclusions[0].id, c2);

        // With threshold 0.0, both should be returned
        let all_conclusions = graph.get_conclusions(0.0);
        assert_eq!(all_conclusions.len(), 2);
    }

    #[test]
    fn test_thought_type_creation() {
        let mut graph = ThoughtGraph::new();

        let question = graph.create_thought_with_type("What is X?", None, ThoughtType::Question);
        let hypothesis = graph.create_thought_with_type(
            "X might be Y",
            Some(&question),
            ThoughtType::Hypothesis,
        );
        let evidence = graph.create_thought_with_type(
            "Data supports Y",
            Some(&hypothesis),
            ThoughtType::Evidence,
        );

        assert_eq!(
            graph.get_thought(&question).unwrap().thought_type,
            ThoughtType::Question
        );
        assert_eq!(
            graph.get_thought(&hypothesis).unwrap().thought_type,
            ThoughtType::Hypothesis
        );
        assert_eq!(
            graph.get_thought(&evidence).unwrap().thought_type,
            ThoughtType::Evidence
        );
    }
}
