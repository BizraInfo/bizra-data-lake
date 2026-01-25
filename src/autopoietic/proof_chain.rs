// src/autopoietic/proof_chain.rs - Merkle Chain for Evolution Audit Trail
//
// Implements Step 8 of the 11-step cycle:
// - Merkle tree construction for proof integrity
// - Blockchain anchoring for immutable evidence
// - Evolution lineage tracking
// - Audit trail generation

use crate::autopoietic::types::GenerationPerformance;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Proof chain for maintaining evolution audit trail
pub struct ProofChain {
    /// All proof nodes indexed by generation
    nodes: HashMap<u64, ProofNode>,

    /// Current chain head (latest generation)
    head: Option<u64>,

    /// Genesis hash (root of the chain)
    genesis_hash: String,

    /// Total nodes in chain
    length: usize,

    /// Pending blockchain anchors
    pending_anchors: Vec<u64>,
}

impl ProofChain {
    /// Create a new proof chain
    pub fn new() -> Self {
        let genesis_hash = Self::compute_genesis_hash();

        info!(genesis_hash = %genesis_hash, "🔗 Proof chain initialized");

        Self {
            nodes: HashMap::new(),
            head: None,
            genesis_hash,
            length: 0,
            pending_anchors: Vec::new(),
        }
    }

    /// Compute genesis hash from system entropy
    fn compute_genesis_hash() -> String {
        let entropy = format!(
            "BIZRA-NUCLEUS-GENESIS-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or(0)
        );
        let hash = Sha256::digest(entropy.as_bytes());
        format!("genesis:{:x}", hash)
    }

    /// Append a new generation to the proof chain
    pub fn append(&mut self, generation: u64, performance: &GenerationPerformance) -> ProofNode {
        // Get previous hash
        let previous_hash = self.head
            .and_then(|gen| self.nodes.get(&gen))
            .map(|node| node.hash.clone())
            .unwrap_or_else(|| self.genesis_hash.clone());

        // Create evolution proof
        let evolution_proof = EvolutionProof::from_performance(performance);

        // Create new node
        let node = ProofNode::new(generation, previous_hash, evolution_proof);

        debug!(
            generation = generation,
            hash = %node.hash,
            "Appended proof node"
        );

        // Store and update head
        self.nodes.insert(generation, node.clone());
        self.head = Some(generation);
        self.length += 1;
        self.pending_anchors.push(generation);

        node
    }

    /// Verify the integrity of the entire chain
    pub fn verify_integrity(&self) -> ChainVerificationResult {
        let mut verified_nodes = 0;
        let mut errors = Vec::new();

        if self.nodes.is_empty() {
            return ChainVerificationResult {
                is_valid: true,
                verified_nodes: 0,
                total_nodes: 0,
                errors: vec![],
                chain_hash: self.genesis_hash.clone(),
            };
        }

        // Start from genesis and walk forward
        let mut expected_previous = self.genesis_hash.clone();

        let mut generations: Vec<u64> = self.nodes.keys().copied().collect();
        generations.sort();

        for gen in generations {
            if let Some(node) = self.nodes.get(&gen) {
                // Verify previous hash link
                if node.previous_hash != expected_previous {
                    errors.push(format!(
                        "Generation {}: previous hash mismatch (expected {}, got {})",
                        gen, expected_previous, node.previous_hash
                    ));
                }

                // Verify node's own hash
                let recomputed = node.recompute_hash();
                if recomputed != node.hash {
                    errors.push(format!(
                        "Generation {}: hash mismatch (expected {}, got {})",
                        gen, node.hash, recomputed
                    ));
                }

                expected_previous = node.hash.clone();
                verified_nodes += 1;
            }
        }

        let is_valid = errors.is_empty();
        let chain_hash = self.head
            .and_then(|gen| self.nodes.get(&gen))
            .map(|n| n.hash.clone())
            .unwrap_or_else(|| self.genesis_hash.clone());

        if is_valid {
            info!(verified_nodes = verified_nodes, "✅ Proof chain verified");
        } else {
            warn!(errors = ?errors, "❌ Proof chain verification failed");
        }

        ChainVerificationResult {
            is_valid,
            verified_nodes,
            total_nodes: self.length,
            errors,
            chain_hash,
        }
    }

    /// Get the current chain head hash
    pub fn head_hash(&self) -> String {
        self.head
            .and_then(|gen| self.nodes.get(&gen))
            .map(|n| n.hash.clone())
            .unwrap_or_else(|| self.genesis_hash.clone())
    }

    /// Get a proof node by generation
    pub fn get(&self, generation: u64) -> Option<&ProofNode> {
        self.nodes.get(&generation)
    }

    /// Get the chain length
    pub fn len(&self) -> usize {
        self.length
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Get proof of evolution from generation A to B
    pub fn get_evolution_proof(&self, from_gen: u64, to_gen: u64) -> Option<EvolutionPath> {
        if from_gen >= to_gen {
            return None;
        }

        let mut path = Vec::new();
        for gen in from_gen..=to_gen {
            if let Some(node) = self.nodes.get(&gen) {
                path.push(node.clone());
            } else {
                return None; // Gap in the chain
            }
        }

        Some(EvolutionPath {
            from_generation: from_gen,
            to_generation: to_gen,
            nodes: path,
            proof_hash: self.compute_path_hash(from_gen, to_gen),
        })
    }

    /// Compute Merkle root for a range
    fn compute_path_hash(&self, from_gen: u64, to_gen: u64) -> String {
        let hashes: Vec<String> = (from_gen..=to_gen)
            .filter_map(|gen| self.nodes.get(&gen))
            .map(|n| n.hash.clone())
            .collect();

        if hashes.is_empty() {
            return "empty".to_string();
        }

        // Simple Merkle root computation
        let combined = hashes.join("|");
        let hash = Sha256::digest(combined.as_bytes());
        format!("merkle:{:x}", hash)
    }

    /// Get pending anchors (generations not yet anchored to blockchain)
    pub fn pending_anchors(&self) -> &[u64] {
        &self.pending_anchors
    }

    /// Mark generations as anchored
    pub fn mark_anchored(&mut self, generations: &[u64], anchor: BlockchainAnchor) {
        for gen in generations {
            if let Some(node) = self.nodes.get_mut(gen) {
                node.blockchain_anchor = Some(anchor.clone());
            }
        }
        self.pending_anchors.retain(|g| !generations.contains(g));

        info!(
            anchored = generations.len(),
            remaining = self.pending_anchors.len(),
            "⛓️ Marked generations as blockchain-anchored"
        );
    }

    /// Export chain summary for external verification
    pub fn export_summary(&self) -> ChainSummary {
        let mut node_summaries: Vec<NodeSummary> = self.nodes
            .iter()
            .map(|(gen, node)| NodeSummary {
                generation: *gen,
                hash: node.hash.clone(),
                ihsan_score: node.evolution_proof.aggregate_ihsan,
                timestamp: node.timestamp,
                anchored: node.blockchain_anchor.is_some(),
            })
            .collect();

        node_summaries.sort_by_key(|n| n.generation);

        ChainSummary {
            genesis_hash: self.genesis_hash.clone(),
            head_hash: self.head_hash(),
            length: self.length,
            nodes: node_summaries,
            pending_anchors: self.pending_anchors.len(),
        }
    }
}

impl Default for ProofChain {
    fn default() -> Self {
        Self::new()
    }
}

/// A single node in the proof chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    /// Generation number
    pub generation: u64,

    /// Hash of this node
    pub hash: String,

    /// Hash of the previous node
    pub previous_hash: String,

    /// Evolution proof data
    pub evolution_proof: EvolutionProof,

    /// Timestamp when node was created
    pub timestamp: DateTime<Utc>,

    /// Blockchain anchor (if anchored)
    pub blockchain_anchor: Option<BlockchainAnchor>,
}

impl ProofNode {
    /// Create a new proof node
    pub fn new(generation: u64, previous_hash: String, evolution_proof: EvolutionProof) -> Self {
        let timestamp = Utc::now();

        // Compute hash
        let hash_content = format!(
            "{}|{}|{}|{:.6}|{}",
            generation,
            previous_hash,
            evolution_proof.improvements_hash,
            evolution_proof.aggregate_ihsan,
            timestamp.timestamp_nanos_opt().unwrap_or(0)
        );
        let hash = Sha256::digest(hash_content.as_bytes());
        let hash_str = format!("proof:{:x}", hash);

        Self {
            generation,
            hash: hash_str,
            previous_hash,
            evolution_proof,
            timestamp,
            blockchain_anchor: None,
        }
    }

    /// Recompute hash for verification
    pub fn recompute_hash(&self) -> String {
        let hash_content = format!(
            "{}|{}|{}|{:.6}|{}",
            self.generation,
            self.previous_hash,
            self.evolution_proof.improvements_hash,
            self.evolution_proof.aggregate_ihsan,
            self.timestamp.timestamp_nanos_opt().unwrap_or(0)
        );
        let hash = Sha256::digest(hash_content.as_bytes());
        format!("proof:{:x}", hash)
    }
}

/// Proof of evolution for a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionProof {
    /// Aggregate Ihsān score
    pub aggregate_ihsan: f64,

    /// SAPE average score
    pub sape_average: f64,

    /// Number of improvements applied
    pub improvements_count: usize,

    /// Hash of improvements
    pub improvements_hash: String,

    /// Number of tasks processed
    pub tasks_processed: u64,

    /// Success rate
    pub success_rate: f64,

    /// Average latency
    pub avg_latency_ms: u64,

    /// Receipt ID
    pub receipt_id: String,
}

impl EvolutionProof {
    /// Create from GenerationPerformance
    pub fn from_performance(perf: &GenerationPerformance) -> Self {
        let success_rate = if perf.tasks_processed > 0 {
            perf.successful_executions as f64 / perf.tasks_processed as f64
        } else {
            0.0
        };

        let improvements_hash = Self::hash_improvements(&perf.improvements_applied);

        Self {
            aggregate_ihsan: perf.aggregate_ihsan,
            sape_average: perf.sape_results.average_score(),
            improvements_count: perf.improvements_applied.len(),
            improvements_hash,
            tasks_processed: perf.tasks_processed,
            success_rate,
            avg_latency_ms: perf.avg_latency_ms,
            receipt_id: perf.receipt_id.clone(),
        }
    }

    fn hash_improvements(improvements: &[String]) -> String {
        if improvements.is_empty() {
            return "none".to_string();
        }
        let combined = improvements.join("|");
        let hash = Sha256::digest(combined.as_bytes());
        format!("{:x}", hash)[..16].to_string()
    }
}

/// Blockchain anchor information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainAnchor {
    /// Chain name (e.g., "bizra-native", "ethereum")
    pub chain: String,

    /// Transaction hash
    pub tx_hash: String,

    /// Block number
    pub block_number: u64,

    /// Timestamp of anchoring
    pub anchored_at: DateTime<Utc>,

    /// Generation range included in this anchor
    pub generations: Vec<u64>,
}

/// Result of chain verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainVerificationResult {
    /// Whether the chain is valid
    pub is_valid: bool,

    /// Number of nodes verified
    pub verified_nodes: usize,

    /// Total nodes in chain
    pub total_nodes: usize,

    /// Errors found during verification
    pub errors: Vec<String>,

    /// Current chain head hash
    pub chain_hash: String,
}

/// Path of evolution from one generation to another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPath {
    /// Starting generation
    pub from_generation: u64,

    /// Ending generation
    pub to_generation: u64,

    /// Nodes in the path
    pub nodes: Vec<ProofNode>,

    /// Merkle root of the path
    pub proof_hash: String,
}

/// Summary of a node for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    pub generation: u64,
    pub hash: String,
    pub ihsan_score: f64,
    pub timestamp: DateTime<Utc>,
    pub anchored: bool,
}

/// Complete chain summary for export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainSummary {
    pub genesis_hash: String,
    pub head_hash: String,
    pub length: usize,
    pub nodes: Vec<NodeSummary>,
    pub pending_anchors: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autopoietic::types::{IhsanDimensions, KEPProgress, SAPEResults};

    fn make_performance(generation: u64, ihsan: f64) -> GenerationPerformance {
        GenerationPerformance {
            generation,
            started_at: Utc::now(),
            ended_at: Utc::now(),
            duration_ms: 60000,
            aggregate_ihsan: ihsan,
            ihsan_dimensions: IhsanDimensions::default(),
            sape_results: SAPEResults::default(),
            tasks_processed: 100,
            successful_executions: 95,
            rejections: 5,
            avg_latency_ms: 150,
            p95_latency_ms: 200,
            kep_progress: KEPProgress::default(),
            improvements_applied: vec!["improvement-1".to_string()],
            proof_hash: "".to_string(),
            receipt_id: format!("GEN-{}", generation),
        }
    }

    #[test]
    fn test_proof_chain_creation() {
        let chain = ProofChain::new();
        assert!(chain.is_empty());
        assert!(chain.genesis_hash.starts_with("genesis:"));
    }

    #[test]
    fn test_append_and_verify() {
        let mut chain = ProofChain::new();

        for i in 1..=5 {
            let perf = make_performance(i, 0.95 + (i as f64 * 0.01));
            chain.append(i, &perf);
        }

        assert_eq!(chain.len(), 5);

        let result = chain.verify_integrity();
        assert!(result.is_valid);
        assert_eq!(result.verified_nodes, 5);
    }

    #[test]
    fn test_evolution_path() {
        let mut chain = ProofChain::new();

        for i in 1..=10 {
            let perf = make_performance(i, 0.95);
            chain.append(i, &perf);
        }

        let path = chain.get_evolution_proof(3, 7).unwrap();
        assert_eq!(path.from_generation, 3);
        assert_eq!(path.to_generation, 7);
        assert_eq!(path.nodes.len(), 5);
        assert!(path.proof_hash.starts_with("merkle:"));
    }

    #[test]
    fn test_chain_summary() {
        let mut chain = ProofChain::new();

        for i in 1..=3 {
            let perf = make_performance(i, 0.96);
            chain.append(i, &perf);
        }

        let summary = chain.export_summary();
        assert_eq!(summary.length, 3);
        assert_eq!(summary.nodes.len(), 3);
        assert_eq!(summary.pending_anchors, 3);
    }

    #[test]
    fn test_blockchain_anchor() {
        let mut chain = ProofChain::new();

        for i in 1..=3 {
            let perf = make_performance(i, 0.96);
            chain.append(i, &perf);
        }

        let anchor = BlockchainAnchor {
            chain: "bizra-native".to_string(),
            tx_hash: "0xabc123".to_string(),
            block_number: 12345,
            anchored_at: Utc::now(),
            generations: vec![1, 2, 3],
        };

        chain.mark_anchored(&[1, 2, 3], anchor);
        assert!(chain.pending_anchors().is_empty());

        let summary = chain.export_summary();
        assert!(summary.nodes.iter().all(|n| n.anchored));
    }
}
