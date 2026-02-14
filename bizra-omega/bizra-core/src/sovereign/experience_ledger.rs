//! Sovereign Experience Ledger (SEL) — Content-Addressed Episodic Memory
//!
//! The SEL stores episodes as content-addressed, hash-chained records.
//! Each episode captures the full reasoning experience: context, graph,
//! actions, impact, and temporal position.
//!
//! # Mathematical Foundation
//!
//! An episode is a tuple:
//!   E_k = { C_k, G_k, A_k, I_k, t_k }
//!
//! Where:
//! - C_k: Context embedding (query + environment state)
//! - G_k: Graph-of-Thoughts artifact hash (BLAKE3 of serialized ThoughtGraph)
//! - A_k: Actions taken (tool calls, inference requests, verdicts)
//! - I_k: Impact measurement (SNR score, Ihsan score, user feedback)
//! - t_k: Temporal position (UTC timestamp + monotonic sequence number)
//!
//! # Content Addressing
//!
//! Each episode is content-addressed via BLAKE3:
//!   episode_hash = BLAKE3("bizra-sel-v1:" || canonical_bytes(E_k))
//!
//! Episodes are hash-chained:
//!   chain_hash_k = BLAKE3(chain_hash_{k-1} || episode_hash_k)
//!
//! # Retrieval: Recency-Importance-Relevance (RIR)
//!
//! RIR(q, E_k) = w_r * Recency(t_k) + w_i * Importance(I_k) + w_s * Relevance(q, C_k)
//!
//! Where:
//! - Recency(t_k) = exp(-lambda * (t_now - t_k))  [exponential decay]
//! - Importance(I_k) = snr_score * ihsan_score     [quality product]
//! - Relevance(q, C_k) = cosine_sim(embed(q), C_k) [semantic similarity]
//!
//! Default weights: w_r=0.3, w_i=0.3, w_s=0.4
//!
//! # Standing on Giants
//!
//! - **Tulving** (1972): Episodic vs semantic memory distinction
//! - **Park et al.** (2023): Generative agent memory architecture
//! - **Vaswani et al.** (2017): Attention-based retrieval mechanisms
//! - **Besta et al.** (2024): Graph-of-Thoughts as first-class artifact
//! - **Shannon** (1948): Information-theoretic SNR measurement

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::genesis::blake3_domain_hash;

// ═══════════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Domain separation prefix for episode hashing
const SEL_DOMAIN: &str = "bizra-sel-v1";

/// Domain separation prefix for chain hashing
const CHAIN_DOMAIN: &str = "bizra-sel-chain-v1";

/// Default RIR weight for recency
const DEFAULT_WEIGHT_RECENCY: f64 = 0.3;

/// Default RIR weight for importance
const DEFAULT_WEIGHT_IMPORTANCE: f64 = 0.3;

/// Default RIR weight for relevance (semantic similarity)
const DEFAULT_WEIGHT_RELEVANCE: f64 = 0.4;

/// Default exponential decay rate (lambda) for recency
/// At lambda=0.001, half-life is ~693 seconds (~11.5 minutes)
const DEFAULT_DECAY_LAMBDA: f64 = 0.001;

/// Maximum episodes to store before triggering distillation
const DEFAULT_MAX_EPISODES: usize = 10_000;

/// Fixed-point precision for efficiency score (P = 10^6)
pub const EFFICIENCY_PRECISION: u64 = 1_000_000;

/// Integer floor(log2(n)) via leading zeros. No floating-point.
/// Returns 0 for n <= 1. Deterministic across all platforms.
#[inline]
pub fn integer_log2(n: u64) -> u32 {
    if n <= 1 {
        0
    } else {
        63 - n.leading_zeros()
    }
}

/// Compute Efficiency_k = (SNR * Ihsan) / max(1, floor(log2(tokens_used + 2)))
/// Uses integer log2 and fixed-point scaling for cross-platform determinism.
pub fn compute_efficiency_score(snr: f64, ihsan: f64, tokens_used: u64) -> f64 {
    let quantize = |v: f64| -> u64 { (v.clamp(0.0, 1.0) * 1_000_000.0) as u64 };
    let numerator = quantize(snr) * quantize(ihsan);
    let log_val = std::cmp::max(1, integer_log2(tokens_used + 2));
    let efficiency_fp = numerator / log_val as u64;
    efficiency_fp as f64 / (EFFICIENCY_PRECISION * EFFICIENCY_PRECISION) as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// Episode Schema
// ═══════════════════════════════════════════════════════════════════════════════

/// An action recorded within an episode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeAction {
    /// Action type identifier (e.g., "inference", "tool_call", "gate_check")
    pub action_type: String,
    /// Action description or payload summary
    pub description: String,
    /// Whether the action succeeded
    pub success: bool,
    /// Duration of the action in microseconds
    pub duration_us: u64,
}

/// Impact measurement for an episode.
///
/// Captures the quality metrics that determine how valuable this episode
/// is for future retrieval. Uses fixed-point representation internally
/// for deterministic hashing (f64 scores are quantized to u32 with 6
/// decimal places of precision).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeImpact {
    /// Signal-to-Noise Ratio score (0.0 - 1.0)
    pub snr_score: f64,
    /// Ihsan (excellence) score (0.0 - 1.0)
    pub ihsan_score: f64,
    /// Whether the SNR gate passed
    pub snr_ok: bool,
    /// Optional user feedback score (-1.0 to 1.0)
    pub user_feedback: Option<f64>,
    /// Tokens consumed by this episode (0 = not tracked)
    #[serde(default)]
    pub tokens_used: u64,
    /// Efficiency_k = (SNR * Ihsan) / max(1, log2(tokens + 2))
    #[serde(default)]
    pub efficiency_score: f64,
}

impl EpisodeImpact {
    /// Compute the importance score for RIR retrieval.
    /// I_k = SNR * Ihsan * Efficiency (when efficiency available)
    /// Falls back to SNR * Ihsan when tokens_used == 0.
    pub fn importance(&self) -> f64 {
        let base = self.snr_score * self.ihsan_score;
        if self.tokens_used > 0 && self.efficiency_score > 0.0 {
            base * self.efficiency_score
        } else {
            base
        }
    }

    /// Quantize a f64 score to a deterministic u32 (6 decimal places).
    /// This ensures hash stability across platforms.
    fn quantize(value: f64) -> u32 {
        (value.clamp(0.0, 1.0) * 1_000_000.0) as u32
    }

    /// Write deterministic bytes for hashing (fixed-point).
    fn hash_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(29);
        bytes.extend_from_slice(&Self::quantize(self.snr_score).to_le_bytes());
        bytes.extend_from_slice(&Self::quantize(self.ihsan_score).to_le_bytes());
        bytes.push(u8::from(self.snr_ok));
        match self.user_feedback {
            Some(fb) => {
                bytes.push(1u8);
                // Shift from [-1,1] to [0,2] then quantize
                let shifted = (fb.clamp(-1.0, 1.0) + 1.0) / 2.0;
                bytes.extend_from_slice(&Self::quantize(shifted).to_le_bytes());
            }
            None => {
                bytes.push(0u8);
            }
        }
        // Efficiency_k (deterministic fixed-point)
        bytes.extend_from_slice(&self.tokens_used.to_le_bytes());
        bytes.extend_from_slice(&Self::quantize(self.efficiency_score).to_le_bytes());
        bytes
    }
}

/// A single episode in the Sovereign Experience Ledger.
///
/// Episodes are immutable once committed. The `episode_hash` field
/// is computed from the canonical serialization of all other fields,
/// providing content-addressability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Episode {
    /// Monotonic sequence number within this ledger
    pub sequence: u64,
    /// UTC timestamp (seconds since UNIX epoch, deterministic)
    pub timestamp_secs: u64,
    /// Context: the query or trigger that initiated this episode
    pub context: String,
    /// BLAKE3 hash of the serialized ThoughtGraph (GoT artifact)
    pub graph_hash: String,
    /// Number of thoughts in the reasoning graph
    pub graph_node_count: usize,
    /// Actions taken during this episode
    pub actions: Vec<EpisodeAction>,
    /// Impact measurement
    pub impact: EpisodeImpact,
    /// Content-address hash of this episode (BLAKE3, hex-encoded)
    pub episode_hash: String,
    /// Hash of the previous episode in the chain (hex-encoded, empty for genesis)
    pub prev_hash: String,
    /// Chain hash: BLAKE3(prev_chain_hash || episode_hash)
    pub chain_hash: String,
    /// Optional context embedding for semantic retrieval (f32 vector)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_embedding: Option<Vec<f32>>,
    /// Optional response summary (truncated for storage efficiency)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_summary: Option<String>,
}

impl Episode {
    /// Compute the content-address hash for this episode.
    ///
    /// The hash covers: sequence, timestamp, context, graph_hash,
    /// graph_node_count, actions, and impact — but NOT the chain
    /// fields (episode_hash, prev_hash, chain_hash) or optional fields.
    fn compute_episode_hash(
        sequence: u64,
        timestamp_secs: u64,
        context: &str,
        graph_hash: &str,
        graph_node_count: usize,
        actions: &[EpisodeAction],
        impact: &EpisodeImpact,
    ) -> [u8; 32] {
        let mut hasher = Hasher::new();

        // Domain separation
        hasher.update(SEL_DOMAIN.as_bytes());
        hasher.update(b":");

        // Sequence + timestamp (fixed-width, little-endian)
        hasher.update(&sequence.to_le_bytes());
        hasher.update(&timestamp_secs.to_le_bytes());

        // Context
        hasher.update(&(context.len() as u32).to_le_bytes());
        hasher.update(context.as_bytes());

        // Graph hash
        hasher.update(&(graph_hash.len() as u32).to_le_bytes());
        hasher.update(graph_hash.as_bytes());

        // Graph node count
        hasher.update(&(graph_node_count as u32).to_le_bytes());

        // Actions (deterministic ordering: sequential)
        hasher.update(&(actions.len() as u32).to_le_bytes());
        for action in actions {
            hasher.update(&(action.action_type.len() as u32).to_le_bytes());
            hasher.update(action.action_type.as_bytes());
            hasher.update(&(action.description.len() as u32).to_le_bytes());
            hasher.update(action.description.as_bytes());
            hasher.update(&[u8::from(action.success)]);
            hasher.update(&action.duration_us.to_le_bytes());
        }

        // Impact (fixed-point)
        hasher.update(&impact.hash_bytes());

        *hasher.finalize().as_bytes()
    }

    /// Compute the chain hash: BLAKE3(prev_chain_hash || episode_hash)
    fn compute_chain_hash(prev_chain_hash: &str, episode_hash: &str) -> String {
        let combined = format!("{}:{}", prev_chain_hash, episode_hash);
        let hash = blake3_domain_hash(CHAIN_DOMAIN, combined.as_bytes());
        hex_encode(&hash)
    }

    /// Verify the content-address hash of this episode.
    pub fn verify_hash(&self) -> bool {
        let computed = Self::compute_episode_hash(
            self.sequence,
            self.timestamp_secs,
            &self.context,
            &self.graph_hash,
            self.graph_node_count,
            &self.actions,
            &self.impact,
        );
        hex_encode(&computed) == self.episode_hash
    }

    /// Verify the chain hash against a previous chain hash.
    pub fn verify_chain(&self, prev_chain_hash: &str) -> bool {
        let computed = Self::compute_chain_hash(prev_chain_hash, &self.episode_hash);
        computed == self.chain_hash
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RIR Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the Recency-Importance-Relevance retrieval algorithm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RIRConfig {
    /// Weight for recency component (default: 0.3)
    pub weight_recency: f64,
    /// Weight for importance component (default: 0.3)
    pub weight_importance: f64,
    /// Weight for relevance/similarity component (default: 0.4)
    pub weight_relevance: f64,
    /// Exponential decay rate for recency (default: 0.001)
    pub decay_lambda: f64,
}

impl Default for RIRConfig {
    fn default() -> Self {
        Self {
            weight_recency: DEFAULT_WEIGHT_RECENCY,
            weight_importance: DEFAULT_WEIGHT_IMPORTANCE,
            weight_relevance: DEFAULT_WEIGHT_RELEVANCE,
            decay_lambda: DEFAULT_DECAY_LAMBDA,
        }
    }
}

impl RIRConfig {
    /// Validate that weights sum to approximately 1.0
    pub fn validate(&self) -> bool {
        let sum = self.weight_recency + self.weight_importance + self.weight_relevance;
        (sum - 1.0).abs() < 1e-6
            && self.weight_recency >= 0.0
            && self.weight_importance >= 0.0
            && self.weight_relevance >= 0.0
            && self.decay_lambda > 0.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sovereign Experience Ledger
// ═══════════════════════════════════════════════════════════════════════════════

/// The Sovereign Experience Ledger (SEL).
///
/// An append-only, hash-chained store of episodes that captures the
/// complete reasoning experience of the sovereign runtime. Episodes
/// are content-addressed via BLAKE3 and chained for tamper-evidence.
pub struct ExperienceLedger {
    /// All episodes, ordered by sequence number
    episodes: VecDeque<Episode>,
    /// O(1) lookup: episode_hash -> index in episodes
    hash_index: HashMap<String, usize>,
    /// O(1) lookup: sequence -> index in episodes
    seq_index: HashMap<u64, usize>,
    /// Next sequence number
    next_sequence: u64,
    /// Current chain hash (last episode's chain_hash, or genesis sentinel)
    chain_head: String,
    /// Maximum episodes before distillation
    max_episodes: usize,
    /// RIR retrieval configuration
    rir_config: RIRConfig,
    /// Count of distillation cycles performed
    distillation_count: u64,
}

impl ExperienceLedger {
    /// Create a new empty experience ledger.
    pub fn new() -> Self {
        Self {
            episodes: VecDeque::new(),
            hash_index: HashMap::new(),
            seq_index: HashMap::new(),
            next_sequence: 0,
            chain_head: "genesis".to_string(),
            max_episodes: DEFAULT_MAX_EPISODES,
            rir_config: RIRConfig::default(),
            distillation_count: 0,
        }
    }

    /// Create a new ledger with custom configuration.
    pub fn with_config(max_episodes: usize, rir_config: RIRConfig) -> Self {
        Self {
            episodes: VecDeque::new(),
            hash_index: HashMap::new(),
            seq_index: HashMap::new(),
            next_sequence: 0,
            chain_head: "genesis".to_string(),
            max_episodes,
            rir_config,
            distillation_count: 0,
        }
    }

    /// Commit a new episode to the ledger.
    ///
    /// The episode is content-addressed (hash computed from fields),
    /// then hash-chained to the previous episode. Returns the episode hash.
    #[allow(clippy::too_many_arguments)]
    pub fn commit(
        &mut self,
        context: String,
        graph_hash: String,
        graph_node_count: usize,
        actions: Vec<EpisodeAction>,
        impact: EpisodeImpact,
        context_embedding: Option<Vec<f32>>,
        response_summary: Option<String>,
    ) -> String {
        let sequence = self.next_sequence;
        let timestamp_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Compute content-address hash
        let episode_hash_bytes = Episode::compute_episode_hash(
            sequence,
            timestamp_secs,
            &context,
            &graph_hash,
            graph_node_count,
            &actions,
            &impact,
        );
        let episode_hash = hex_encode(&episode_hash_bytes);

        // Compute chain hash
        let chain_hash = Episode::compute_chain_hash(&self.chain_head, &episode_hash);

        let episode = Episode {
            sequence,
            timestamp_secs,
            context,
            graph_hash,
            graph_node_count,
            actions,
            impact,
            episode_hash: episode_hash.clone(),
            prev_hash: self.chain_head.clone(),
            chain_hash: chain_hash.clone(),
            context_embedding,
            response_summary,
        };

        // Update ledger state
        self.chain_head = chain_hash;
        let idx = self.episodes.len();
        self.episodes.push_back(episode);
        self.hash_index.insert(episode_hash.clone(), idx);
        self.seq_index.insert(sequence, idx);
        self.next_sequence += 1;

        // Check if distillation needed
        if self.episodes.len() > self.max_episodes {
            self.distill();
        }

        episode_hash
    }

    /// Commit an episode with a specific timestamp (for testing/replay).
    pub fn commit_at(
        &mut self,
        timestamp_secs: u64,
        context: String,
        graph_hash: String,
        graph_node_count: usize,
        actions: Vec<EpisodeAction>,
        impact: EpisodeImpact,
    ) -> String {
        let sequence = self.next_sequence;

        let episode_hash_bytes = Episode::compute_episode_hash(
            sequence,
            timestamp_secs,
            &context,
            &graph_hash,
            graph_node_count,
            &actions,
            &impact,
        );
        let episode_hash = hex_encode(&episode_hash_bytes);
        let chain_hash = Episode::compute_chain_hash(&self.chain_head, &episode_hash);

        let episode = Episode {
            sequence,
            timestamp_secs,
            context,
            graph_hash,
            graph_node_count,
            actions,
            impact,
            episode_hash: episode_hash.clone(),
            prev_hash: self.chain_head.clone(),
            chain_hash: chain_hash.clone(),
            context_embedding: None,
            response_summary: None,
        };

        self.chain_head = chain_hash;
        let idx = self.episodes.len();
        self.episodes.push_back(episode);
        self.hash_index.insert(episode_hash.clone(), idx);
        self.seq_index.insert(sequence, idx);
        self.next_sequence += 1;

        // Check if distillation needed
        if self.episodes.len() > self.max_episodes {
            self.distill();
        }

        episode_hash
    }

    /// Retrieve the top-K episodes using the RIR algorithm.
    ///
    /// When `query_embedding` is None, relevance is set to 0.0 and
    /// retrieval is based only on recency and importance.
    pub fn retrieve(
        &self,
        query_text: &str,
        query_embedding: Option<&[f32]>,
        top_k: usize,
    ) -> Vec<&Episode> {
        if self.episodes.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut scored: Vec<(&Episode, f64)> = self
            .episodes
            .iter()
            .map(|ep| {
                let score = self.compute_rir_score(ep, now_secs, query_text, query_embedding);
                (ep, score)
            })
            .collect();

        // Sort descending by RIR score
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        scored.into_iter().take(top_k).map(|(ep, _)| ep).collect()
    }

    /// Retrieve episodes within a time window.
    pub fn retrieve_recent(&self, max_age_secs: u64) -> Vec<&Episode> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.episodes
            .iter()
            .filter(|ep| now_secs.saturating_sub(ep.timestamp_secs) <= max_age_secs)
            .collect()
    }

    /// Retrieve a specific episode by its content-address hash. O(1).
    pub fn get_by_hash(&self, hash: &str) -> Option<&Episode> {
        self.hash_index
            .get(hash)
            .and_then(|&idx| self.episodes.get(idx))
    }

    /// Retrieve a specific episode by sequence number. O(1).
    pub fn get_by_sequence(&self, sequence: u64) -> Option<&Episode> {
        self.seq_index
            .get(&sequence)
            .and_then(|&idx| self.episodes.get(idx))
    }

    /// Verify the entire chain integrity.
    ///
    /// Checks that every episode's chain_hash is correctly derived from
    /// the previous chain_hash and its own episode_hash.
    pub fn verify_chain_integrity(&self) -> bool {
        let mut prev_chain = "genesis".to_string();

        for episode in &self.episodes {
            // Verify episode content hash
            if !episode.verify_hash() {
                return false;
            }

            // Verify chain linkage
            if episode.prev_hash != prev_chain {
                return false;
            }

            if !episode.verify_chain(&prev_chain) {
                return false;
            }

            prev_chain = episode.chain_hash.clone();
        }

        // Final chain head must match
        prev_chain == self.chain_head
    }

    /// Get the current chain head hash.
    pub fn chain_head(&self) -> &str {
        &self.chain_head
    }

    /// Get the number of episodes in the ledger.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if the ledger is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get the next sequence number.
    pub fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    /// Get the number of distillation cycles performed.
    pub fn distillation_count(&self) -> u64 {
        self.distillation_count
    }

    /// Get a reference to all episodes.
    pub fn episodes(&self) -> &VecDeque<Episode> {
        &self.episodes
    }

    /// Get the RIR configuration.
    pub fn rir_config(&self) -> &RIRConfig {
        &self.rir_config
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal: RIR Scoring
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute the RIR score for a single episode.
    fn compute_rir_score(
        &self,
        episode: &Episode,
        now_secs: u64,
        query_text: &str,
        query_embedding: Option<&[f32]>,
    ) -> f64 {
        let recency = self.compute_recency(episode, now_secs);
        let importance = episode.impact.importance();
        let relevance = self.compute_relevance(episode, query_text, query_embedding);

        self.rir_config.weight_recency * recency
            + self.rir_config.weight_importance * importance
            + self.rir_config.weight_relevance * relevance
    }

    /// Recency(t_k) = exp(-lambda * (t_now - t_k))
    fn compute_recency(&self, episode: &Episode, now_secs: u64) -> f64 {
        let age_secs = now_secs.saturating_sub(episode.timestamp_secs) as f64;
        (-self.rir_config.decay_lambda * age_secs).exp()
    }

    /// Compute relevance between a query and an episode.
    ///
    /// Uses cosine similarity if embeddings are available, otherwise
    /// falls back to keyword overlap (Jaccard similarity on words).
    fn compute_relevance(
        &self,
        episode: &Episode,
        query_text: &str,
        query_embedding: Option<&[f32]>,
    ) -> f64 {
        // Try embedding-based cosine similarity first
        if let (Some(q_emb), Some(e_emb)) = (query_embedding, episode.context_embedding.as_ref()) {
            return cosine_similarity(q_emb, e_emb);
        }

        // Fallback: keyword overlap (Jaccard similarity)
        keyword_similarity(query_text, &episode.context)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal: Recursive Distillation
    // ─────────────────────────────────────────────────────────────────────────

    /// Perform recursive distillation: compress the oldest half of episodes
    /// by removing low-importance entries.
    ///
    /// Keeps episodes where importance > median importance of the batch.
    /// This is a simplified form of the theoretical distillation that
    /// preserves the highest-signal episodes.
    fn distill(&mut self) {
        let half = self.episodes.len() / 2;
        if half == 0 {
            return;
        }

        // Compute importance of the oldest half
        let mut importances: Vec<f64> = self
            .episodes
            .iter()
            .take(half)
            .map(|ep| ep.impact.importance())
            .collect();
        importances.sort_by(|a, b| a.total_cmp(b));
        let median = importances[importances.len() / 2];

        // Keep only episodes above median importance from the oldest half
        let mut new_front: VecDeque<Episode> = self
            .episodes
            .drain(..half)
            .filter(|ep| ep.impact.importance() >= median)
            .collect();

        // Prepend the kept episodes
        while let Some(ep) = new_front.pop_back() {
            self.episodes.push_front(ep);
        }

        self.distillation_count += 1;

        // Rebuild indexes after distillation
        self.rebuild_indexes();
    }

    /// Rebuild hash and sequence indexes from episodes.
    fn rebuild_indexes(&mut self) {
        self.hash_index.clear();
        self.seq_index.clear();
        for (idx, ep) in self.episodes.iter().enumerate() {
            self.hash_index.insert(ep.episode_hash.clone(), idx);
            self.seq_index.insert(ep.sequence, idx);
        }
    }
}

impl Default for ExperienceLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute cosine similarity between two f32 vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let ai = *ai as f64;
        let bi = *bi as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    (dot / denom).clamp(0.0, 1.0)
}

/// Compute keyword similarity (Jaccard) between two text strings.
fn keyword_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .collect();
    let words_b: std::collections::HashSet<&str> = b
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count() as f64;
    let union = words_a.union(&words_b).count() as f64;

    if union < 1e-12 {
        0.0
    } else {
        intersection / union
    }
}

/// Hex-encode a byte slice.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_impact(snr: f64, ihsan: f64) -> EpisodeImpact {
        EpisodeImpact {
            snr_score: snr,
            ihsan_score: ihsan,
            snr_ok: snr >= 0.85,
            user_feedback: None,
            tokens_used: 0,
            efficiency_score: 0.0,
        }
    }

    fn make_action(action_type: &str, desc: &str) -> EpisodeAction {
        EpisodeAction {
            action_type: action_type.to_string(),
            description: desc.to_string(),
            success: true,
            duration_us: 1000,
        }
    }

    #[test]
    fn test_episode_hash_determinism() {
        let mut ledger = ExperienceLedger::new();
        let hash1 = ledger.commit_at(
            1000,
            "test query".to_string(),
            "abc123".to_string(),
            5,
            vec![make_action("inference", "LLM call")],
            make_impact(0.95, 0.96),
        );

        // Create a second ledger and commit the same episode
        let mut ledger2 = ExperienceLedger::new();
        let hash2 = ledger2.commit_at(
            1000,
            "test query".to_string(),
            "abc123".to_string(),
            5,
            vec![make_action("inference", "LLM call")],
            make_impact(0.95, 0.96),
        );

        assert_eq!(hash1, hash2, "Same inputs must produce same hash");
    }

    #[test]
    fn test_episode_hash_changes_with_input() {
        let mut ledger = ExperienceLedger::new();
        let hash1 = ledger.commit_at(
            1000,
            "query A".to_string(),
            "abc123".to_string(),
            5,
            vec![make_action("inference", "LLM call")],
            make_impact(0.95, 0.96),
        );

        let hash2 = ledger.commit_at(
            1000,
            "query B".to_string(),
            "abc123".to_string(),
            5,
            vec![make_action("inference", "LLM call")],
            make_impact(0.95, 0.96),
        );

        assert_ne!(
            hash1, hash2,
            "Different inputs must produce different hashes"
        );
    }

    #[test]
    fn test_chain_integrity() {
        let mut ledger = ExperienceLedger::new();

        for i in 0..10 {
            ledger.commit_at(
                1000 + i,
                format!("query {}", i),
                format!("graph_{}", i),
                3,
                vec![make_action("inference", "call")],
                make_impact(0.90, 0.92),
            );
        }

        assert_eq!(ledger.len(), 10);
        assert!(ledger.verify_chain_integrity(), "Chain must be intact");
    }

    #[test]
    fn test_chain_tamper_detection() {
        let mut ledger = ExperienceLedger::new();

        for i in 0..5 {
            ledger.commit_at(
                1000 + i,
                format!("query {}", i),
                format!("graph_{}", i),
                3,
                vec![make_action("inference", "call")],
                make_impact(0.90, 0.92),
            );
        }

        assert!(ledger.verify_chain_integrity());

        // Tamper with an episode's context
        if let Some(ep) = ledger.episodes.get_mut(2) {
            ep.context = "TAMPERED".to_string();
        }

        assert!(!ledger.verify_chain_integrity(), "Tampered chain must fail");
    }

    #[test]
    fn test_rir_retrieval_recency() {
        let mut ledger = ExperienceLedger::new();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Commit episodes at different times (older episodes first)
        for i in 0..5u64 {
            ledger.commit_at(
                now - (4 - i) * 1000, // Oldest first, most recent last
                format!("query about topic {}", i),
                format!("graph_{}", i),
                3,
                vec![make_action("inference", "call")],
                make_impact(0.90, 0.92), // Same importance
            );
        }

        // Retrieve top 3 — most recent should rank higher
        let results = ledger.retrieve("topic", None, 3);
        assert_eq!(results.len(), 3);

        // Most recent episode (sequence 4, timestamp = now) should be first
        assert_eq!(results[0].sequence, 4);
    }

    #[test]
    fn test_rir_retrieval_importance() {
        let mut ledger = ExperienceLedger::with_config(
            10_000,
            RIRConfig {
                weight_recency: 0.0,    // Disable recency
                weight_importance: 1.0, // Only importance
                weight_relevance: 0.0,  // Disable relevance
                decay_lambda: DEFAULT_DECAY_LAMBDA,
            },
        );

        // Same timestamp, different importance
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        ledger.commit_at(
            now,
            "low quality".into(),
            "g1".into(),
            2,
            vec![make_action("inference", "call")],
            make_impact(0.50, 0.50),
        ); // importance = 0.25

        ledger.commit_at(
            now,
            "medium quality".into(),
            "g2".into(),
            3,
            vec![make_action("inference", "call")],
            make_impact(0.80, 0.85),
        ); // importance = 0.68

        ledger.commit_at(
            now,
            "high quality".into(),
            "g3".into(),
            5,
            vec![make_action("inference", "call")],
            make_impact(0.99, 0.98),
        ); // importance = 0.97

        let results = ledger.retrieve("anything", None, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].context, "high quality");
        assert_eq!(results[1].context, "medium quality");
        assert_eq!(results[2].context, "low quality");
    }

    #[test]
    fn test_rir_retrieval_relevance_keywords() {
        let mut ledger = ExperienceLedger::with_config(
            10_000,
            RIRConfig {
                weight_recency: 0.0,
                weight_importance: 0.0,
                weight_relevance: 1.0, // Only relevance
                decay_lambda: DEFAULT_DECAY_LAMBDA,
            },
        );

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        ledger.commit_at(
            now,
            "how to bake a chocolate cake".into(),
            "g1".into(),
            2,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
        );

        ledger.commit_at(
            now,
            "machine learning neural network training".into(),
            "g2".into(),
            3,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
        );

        ledger.commit_at(
            now,
            "deep learning neural network optimization".into(),
            "g3".into(),
            3,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
        );

        // Query about neural networks should rank ML episodes higher
        let results = ledger.retrieve("neural network architecture", None, 3);
        assert_eq!(results.len(), 3);
        // Both ML episodes should rank above the cake episode
        assert_ne!(
            results[2].context,
            "machine learning neural network training"
        );
        assert_ne!(
            results[2].context,
            "deep learning neural network optimization"
        );
    }

    #[test]
    fn test_rir_retrieval_cosine_similarity() {
        let mut ledger = ExperienceLedger::with_config(
            10_000,
            RIRConfig {
                weight_recency: 0.0,
                weight_importance: 0.0,
                weight_relevance: 1.0,
                decay_lambda: DEFAULT_DECAY_LAMBDA,
            },
        );

        // Episode 1: embedding pointing in direction [1, 0, 0]
        let hash1 = ledger.commit(
            "episode one".into(),
            "g1".into(),
            2,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
            Some(vec![1.0, 0.0, 0.0]),
            None,
        );

        // Episode 2: embedding pointing in direction [0, 1, 0]
        let _hash2 = ledger.commit(
            "episode two".into(),
            "g2".into(),
            2,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
            Some(vec![0.0, 1.0, 0.0]),
            None,
        );

        // Query embedding close to [1, 0, 0]
        let query_emb = vec![0.9, 0.1, 0.0];
        let results = ledger.retrieve("anything", Some(&query_emb), 2);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].episode_hash, hash1,
            "Most similar should be first"
        );
    }

    #[test]
    fn test_episode_verify_hash() {
        let mut ledger = ExperienceLedger::new();
        ledger.commit_at(
            1000,
            "test".to_string(),
            "graph".to_string(),
            3,
            vec![make_action("inference", "call")],
            make_impact(0.95, 0.96),
        );

        let ep = ledger.get_by_sequence(0).unwrap();
        assert!(ep.verify_hash());
    }

    #[test]
    fn test_get_by_hash() {
        let mut ledger = ExperienceLedger::new();
        let hash = ledger.commit_at(
            1000,
            "findme".to_string(),
            "graph".to_string(),
            3,
            vec![make_action("inference", "call")],
            make_impact(0.95, 0.96),
        );

        let ep = ledger.get_by_hash(&hash);
        assert!(ep.is_some());
        assert_eq!(ep.unwrap().context, "findme");

        assert!(ledger.get_by_hash("nonexistent").is_none());
    }

    #[test]
    fn test_retrieve_recent() {
        let mut ledger = ExperienceLedger::new();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Old episode
        ledger.commit_at(
            now - 10000,
            "old".into(),
            "g1".into(),
            2,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
        );

        // Recent episode
        ledger.commit_at(
            now,
            "recent".into(),
            "g2".into(),
            2,
            vec![make_action("inference", "call")],
            make_impact(0.90, 0.90),
        );

        let recent = ledger.retrieve_recent(5000);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].context, "recent");
    }

    #[test]
    fn test_distillation() {
        let mut ledger = ExperienceLedger::with_config(10, RIRConfig::default());

        let base_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Commit 11 episodes (exceeds max_episodes=10)
        for i in 0..11u64 {
            let snr = if i < 5 { 0.50 } else { 0.95 };
            ledger.commit_at(
                base_ts + i,
                format!("episode {}", i),
                format!("g{}", i),
                3,
                vec![make_action("inference", "call")],
                make_impact(snr, 0.90),
            );
        }

        // Distillation should have occurred
        assert!(ledger.distillation_count() > 0);
        assert!(ledger.len() <= 11); // Some low-importance episodes removed
    }

    #[test]
    fn test_impact_importance() {
        let impact = make_impact(0.95, 0.98);
        let expected = 0.95 * 0.98;
        assert!((impact.importance() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_impact_quantization_stability() {
        let impact = EpisodeImpact {
            snr_score: 0.123456789,
            ihsan_score: 0.987654321,
            snr_ok: true,
            user_feedback: Some(0.5),
            tokens_used: 5000,
            efficiency_score: 0.456789,
        };

        let bytes1 = impact.hash_bytes();
        let bytes2 = impact.hash_bytes();
        assert_eq!(bytes1, bytes2, "Quantization must be stable");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_keyword_similarity() {
        let a = "neural network training optimization";
        let b = "neural network inference deployment";
        let sim = keyword_similarity(a, b);
        assert!(sim > 0.0, "Should have keyword overlap");
        assert!(sim < 1.0, "Should not be identical");
    }

    #[test]
    fn test_keyword_similarity_no_overlap() {
        let a = "chocolate cake recipe";
        let b = "quantum physics experiment";
        let sim = keyword_similarity(a, b);
        assert!(sim < 0.01, "Should have no meaningful overlap");
    }

    #[test]
    fn test_rir_config_validation() {
        let valid = RIRConfig::default();
        assert!(valid.validate());

        let invalid = RIRConfig {
            weight_recency: 0.5,
            weight_importance: 0.5,
            weight_relevance: 0.5, // Sum = 1.5
            decay_lambda: 0.001,
        };
        assert!(!invalid.validate());

        let negative = RIRConfig {
            weight_recency: -0.1,
            weight_importance: 0.6,
            weight_relevance: 0.5,
            decay_lambda: 0.001,
        };
        assert!(!negative.validate());
    }

    #[test]
    fn test_empty_ledger() {
        let ledger = ExperienceLedger::new();
        assert!(ledger.is_empty());
        assert_eq!(ledger.len(), 0);
        assert_eq!(ledger.next_sequence(), 0);
        assert_eq!(ledger.chain_head(), "genesis");
        assert!(ledger.verify_chain_integrity());

        let results = ledger.retrieve("anything", None, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut ledger = ExperienceLedger::new();
        ledger.commit_at(
            1000,
            "test query".to_string(),
            "abc123".to_string(),
            5,
            vec![make_action("inference", "LLM call")],
            make_impact(0.95, 0.96),
        );

        let ep = ledger.get_by_sequence(0).unwrap();
        let json = serde_json::to_string(ep).expect("serialize");
        let parsed: Episode = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(parsed.sequence, ep.sequence);
        assert_eq!(parsed.context, ep.context);
        assert_eq!(parsed.episode_hash, ep.episode_hash);
        assert_eq!(parsed.chain_hash, ep.chain_hash);
        assert!(parsed.verify_hash());
    }

    // ─────────────────────────────────────────────────────────────────────
    // HashMap Index Tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_hash_index_o1_lookup() {
        let mut ledger = ExperienceLedger::new();
        let hashes: Vec<String> = (0..100)
            .map(|i| {
                ledger.commit_at(
                    1000 + i,
                    format!("query {}", i),
                    format!("g{}", i),
                    3,
                    vec![make_action("inference", "call")],
                    make_impact(0.90, 0.92),
                )
            })
            .collect();

        // All 100 hashes should be retrievable
        for (i, h) in hashes.iter().enumerate() {
            let ep = ledger.get_by_hash(h);
            assert!(ep.is_some(), "hash lookup failed for episode {}", i);
            assert_eq!(ep.unwrap().sequence, i as u64);
        }
    }

    #[test]
    fn test_seq_index_o1_lookup() {
        let mut ledger = ExperienceLedger::new();
        for i in 0..50u64 {
            ledger.commit_at(
                1000 + i,
                format!("query {}", i),
                format!("g{}", i),
                3,
                vec![make_action("inference", "call")],
                make_impact(0.90, 0.92),
            );
        }

        for seq in 0..50u64 {
            let ep = ledger.get_by_sequence(seq);
            assert!(ep.is_some(), "seq lookup failed for {}", seq);
            assert_eq!(ep.unwrap().context, format!("query {}", seq));
        }
        assert!(ledger.get_by_sequence(999).is_none());
    }

    #[test]
    fn test_index_survives_distillation() {
        let mut ledger = ExperienceLedger::with_config(10, RIRConfig::default());

        let base_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut hashes = Vec::new();
        for i in 0..11u64 {
            let snr = if i < 5 { 0.50 } else { 0.95 };
            let h = ledger.commit_at(
                base_ts + i,
                format!("ep {}", i),
                format!("g{}", i),
                3,
                vec![make_action("inference", "call")],
                make_impact(snr, 0.90),
            );
            hashes.push(h);
        }

        assert!(ledger.distillation_count() > 0);

        // Surviving episodes should still be findable via index
        for ep in ledger.episodes() {
            let by_hash = ledger.get_by_hash(&ep.episode_hash);
            assert!(by_hash.is_some(), "hash index broken after distillation");
            let by_seq = ledger.get_by_sequence(ep.sequence);
            assert!(by_seq.is_some(), "seq index broken after distillation");
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Efficiency_k Tests
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_integer_log2_values() {
        assert_eq!(integer_log2(0), 0);
        assert_eq!(integer_log2(1), 0);
        assert_eq!(integer_log2(2), 1);
        assert_eq!(integer_log2(3), 1);
        assert_eq!(integer_log2(4), 2);
        assert_eq!(integer_log2(7), 2);
        assert_eq!(integer_log2(8), 3);
        assert_eq!(integer_log2(1023), 9);
        assert_eq!(integer_log2(1024), 10);
        assert_eq!(integer_log2(1_000_000), 19);
    }

    #[test]
    fn test_efficiency_score_deterministic() {
        let e1 = compute_efficiency_score(0.95, 0.98, 1000);
        let e2 = compute_efficiency_score(0.95, 0.98, 1000);
        assert_eq!(e1, e2, "Efficiency must be deterministic");
    }

    #[test]
    fn test_efficiency_score_decreases_with_tokens() {
        let e_small = compute_efficiency_score(0.95, 0.98, 100);
        let e_large = compute_efficiency_score(0.95, 0.98, 1_000_000);
        assert!(e_small > e_large, "More tokens should lower efficiency");
    }

    #[test]
    fn test_efficiency_score_zero_tokens() {
        // tokens_used=0 means efficiency not tracked
        let impact = make_impact(0.95, 0.98);
        let base = 0.95 * 0.98;
        assert!((impact.importance() - base).abs() < 1e-10);
    }

    #[test]
    fn test_importance_with_efficiency() {
        let impact = EpisodeImpact {
            snr_score: 0.95,
            ihsan_score: 0.98,
            snr_ok: true,
            user_feedback: None,
            tokens_used: 1000,
            efficiency_score: compute_efficiency_score(0.95, 0.98, 1000),
        };
        let base = 0.95 * 0.98;
        // With efficiency, importance should be base * efficiency
        assert!(
            impact.importance() < base,
            "Efficiency should reduce importance"
        );
        assert!(
            impact.importance() > 0.0,
            "Importance should still be positive"
        );
    }
}
