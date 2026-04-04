// src/autopoietic/types.rs - Core AutopoieticLoop Types
//
// Data structures for the self-improving autonomous engine.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Configuration for the AutopoieticLoop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticConfig {
    /// Duration of each generation cycle in milliseconds
    pub generation_duration_ms: u64,

    /// Maximum number of generations (0 = unlimited)
    pub max_generations: u64,

    /// Ihsān threshold (hard gate at 0.95)
    pub ihsan_threshold: f64,

    /// KEP detection thresholds
    pub kep_thresholds: KEPThresholds,

    /// Number of recent generations to consider for convergence
    pub convergence_window: usize,

    /// Improvement delta threshold for plateau detection
    pub improvement_threshold: f64,

    /// Whether to enable blockchain anchoring
    pub enable_blockchain_anchoring: bool,

    /// Redis persistence key prefix
    pub synapse_prefix: String,

    /// Warm pool configuration
    pub warm_pool_enabled: bool,
    pub warm_pool_sizes: HashMap<String, usize>,
}

impl Default for AutopoieticConfig {
    fn default() -> Self {
        Self {
            generation_duration_ms: 60_000, // 1 minute per generation
            max_generations: 0,             // Unlimited
            ihsan_threshold: 0.95,
            kep_thresholds: KEPThresholds::default(),
            convergence_window: 10,
            improvement_threshold: 0.001,
            enable_blockchain_anchoring: true,
            synapse_prefix: "bizra:autopoietic".to_string(),
            warm_pool_enabled: true,
            warm_pool_sizes: HashMap::new(),
        }
    }
}

/// Current status of the AutopoieticLoop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopoieticStatus {
    /// Whether the loop is currently running
    pub is_running: bool,

    /// Current generation number
    pub current_generation: u64,

    /// Current KEP state
    pub kep_state: KEPState,

    /// Latest aggregate Ihsān score
    pub aggregate_ihsan: f64,

    /// Timestamp of last generation start
    pub last_generation_start: Option<DateTime<Utc>>,

    /// Timestamp of last generation end
    pub last_generation_end: Option<DateTime<Utc>>,

    /// Number of active agents
    pub active_agents: usize,

    /// Number of blueprints
    pub blueprint_count: usize,

    /// Latest convergence metrics
    pub convergence_state: String,

    /// Proof chain length
    pub proof_chain_length: usize,

    /// Total receipts emitted
    pub receipts_emitted: u64,
}

impl Default for AutopoieticStatus {
    fn default() -> Self {
        Self {
            is_running: false,
            current_generation: 0,
            kep_state: KEPState::Normal,
            aggregate_ihsan: 0.0,
            last_generation_start: None,
            last_generation_end: None,
            active_agents: 0,
            blueprint_count: 0,
            convergence_state: "Initializing".to_string(),
            proof_chain_length: 0,
            receipts_emitted: 0,
        }
    }
}

/// Errors that can occur in the AutopoieticLoop
#[derive(Debug, Error, Clone)]
pub enum AutopoieticError {
    #[error("Ihsān gate failed: score {score:.4} < threshold {threshold:.4}")]
    IhsanGateFailed { score: f64, threshold: f64 },

    #[error("FATE escalation required: level {level}")]
    FATEEscalation { level: String, reason: String },

    #[error("SAT consensus not reached: {votes_for}/5 votes")]
    SATConsensusFailed {
        votes_for: usize,
        votes_against: usize,
    },

    #[error("Blueprint evolution failed: {reason}")]
    EvolutionFailed { reason: String },

    #[error("Proof chain integrity error: {details}")]
    ProofChainError { details: String },

    #[error("Agent spawn failed: {agent_name}")]
    AgentSpawnFailed { agent_name: String },

    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    #[error("Synapse (Redis) error: {message}")]
    SynapseError { message: String },

    #[error("Loop already running")]
    AlreadyRunning,

    #[error("Loop not running")]
    NotRunning,

    #[error("Maximum generations reached: {max}")]
    MaxGenerationsReached { max: u64 },

    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Performance metrics for a single generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPerformance {
    /// Generation number
    pub generation: u64,

    /// Start timestamp
    pub started_at: DateTime<Utc>,

    /// End timestamp
    pub ended_at: DateTime<Utc>,

    /// Duration of this generation
    pub duration_ms: u64,

    /// Aggregate Ihsān score across all agents
    pub aggregate_ihsan: f64,

    /// Individual dimension scores (8 dimensions)
    pub ihsan_dimensions: IhsanDimensions,

    /// SAPE probe results
    pub sape_results: SAPEResults,

    /// Number of tasks processed
    pub tasks_processed: u64,

    /// Number of successful executions
    pub successful_executions: u64,

    /// Number of rejections
    pub rejections: u64,

    /// Average latency in milliseconds
    pub avg_latency_ms: u64,

    /// P95 latency in milliseconds
    pub p95_latency_ms: u64,

    /// KEP progress metrics
    pub kep_progress: KEPProgress,

    /// Blueprint improvements applied
    pub improvements_applied: Vec<String>,

    /// Proof chain hash for this generation
    pub proof_hash: String,

    /// Receipt ID for this generation
    pub receipt_id: String,
}

/// The 8 Ihsān dimensions with individual scores
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IhsanDimensions {
    /// Correctness (weight: 0.22)
    pub correctness: f64,
    /// Safety (weight: 0.22)
    pub safety: f64,
    /// User benefit (weight: 0.14)
    pub user_benefit: f64,
    /// Efficiency (weight: 0.12)
    pub efficiency: f64,
    /// Auditability (weight: 0.12)
    pub auditability: f64,
    /// Anti-centralization (weight: 0.08)
    pub anti_centralization: f64,
    /// Robustness (weight: 0.06)
    pub robustness: f64,
    /// Adl fairness (weight: 0.04)
    pub adl_fairness: f64,
}

impl IhsanDimensions {
    /// Calculate weighted aggregate score
    pub fn aggregate(&self) -> f64 {
        self.correctness * 0.22
            + self.safety * 0.22
            + self.user_benefit * 0.14
            + self.efficiency * 0.12
            + self.auditability * 0.12
            + self.anti_centralization * 0.08
            + self.robustness * 0.06
            + self.adl_fairness * 0.04
    }

    /// Get the minimum dimension score
    pub fn min_dimension(&self) -> (String, f64) {
        let dims = [
            ("correctness", self.correctness),
            ("safety", self.safety),
            ("user_benefit", self.user_benefit),
            ("efficiency", self.efficiency),
            ("auditability", self.auditability),
            ("anti_centralization", self.anti_centralization),
            ("robustness", self.robustness),
            ("adl_fairness", self.adl_fairness),
        ];
        dims.into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(name, score)| (name.to_string(), score))
            .unwrap_or(("unknown".to_string(), 0.0))
    }
}

/// Results from SAPE 9-probe verification
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SAPEResults {
    /// Threat scan probe
    pub threat_scan: ProbeResult,
    /// Compliance probe
    pub compliance: ProbeResult,
    /// Bias probe
    pub bias: ProbeResult,
    /// User benefit probe
    pub user_benefit: ProbeResult,
    /// Correctness probe
    pub correctness: ProbeResult,
    /// Safety probe
    pub safety: ProbeResult,
    /// Groundedness probe
    pub groundedness: ProbeResult,
    /// Relevance probe
    pub relevance: ProbeResult,
    /// Fluency probe
    pub fluency: ProbeResult,
}

impl SAPEResults {
    /// Check if all probes passed
    pub fn all_passed(&self) -> bool {
        self.threat_scan.passed
            && self.compliance.passed
            && self.bias.passed
            && self.user_benefit.passed
            && self.correctness.passed
            && self.safety.passed
            && self.groundedness.passed
            && self.relevance.passed
            && self.fluency.passed
    }

    /// Get average score across all probes
    pub fn average_score(&self) -> f64 {
        let scores = [
            self.threat_scan.score,
            self.compliance.score,
            self.bias.score,
            self.user_benefit.score,
            self.correctness.score,
            self.safety.score,
            self.groundedness.score,
            self.relevance.score,
            self.fluency.score,
        ];
        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// Result of a single SAPE probe
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbeResult {
    /// Whether the probe passed
    pub passed: bool,
    /// Score from 0.0 to 1.0
    pub score: f64,
    /// Any evidence or notes
    pub evidence: Vec<String>,
}

/// Knowledge Explosion Point state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KEPState {
    /// Normal operation
    Normal,
    /// Approaching explosion point
    Approaching,
    /// Entering explosion mode
    EnteringExplosion,
    /// In explosion mode (accelerated learning)
    InExplosion,
    /// Exiting explosion mode
    ExitingExplosion,
    /// Plateaued (convergence reached)
    Plateau,
}

impl Default for KEPState {
    fn default() -> Self {
        Self::Normal
    }
}

/// KEP progress metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KEPProgress {
    /// Total knowledge mass (number of knowledge elements)
    pub knowledge_mass: u64,

    /// Discovery velocity (compounds per hour)
    pub discovery_velocity: f64,

    /// Synergy density (ratio of connected knowledge)
    pub synergy_density: f64,

    /// Learning rate multiplier (1.0 = normal)
    pub learning_rate_multiplier: f64,

    /// Number of synergies detected this generation
    pub synergies_detected: u64,

    /// Number of compounds synthesized this generation
    pub compounds_synthesized: u64,

    /// Time in explosion mode (seconds)
    pub explosion_duration_seconds: u64,
}

/// Thresholds for KEP state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KEPThresholds {
    /// Minimum knowledge mass for explosion entry
    pub min_knowledge_mass: u64,

    /// Minimum discovery velocity for explosion entry
    pub min_velocity: f64,

    /// Minimum synergy density for explosion entry
    pub min_synergy_density: f64,

    /// Velocity threshold for approaching state
    pub approaching_velocity: f64,

    /// Mass threshold for approaching state
    pub approaching_mass: u64,

    /// Cooldown duration after explosion exit (seconds)
    pub explosion_cooldown_seconds: u64,
}

impl Default for KEPThresholds {
    fn default() -> Self {
        Self {
            min_knowledge_mass: 1000,
            min_velocity: 10.0, // 10 compounds per hour
            min_synergy_density: 0.3,
            approaching_velocity: 5.0,
            approaching_mass: 500,
            explosion_cooldown_seconds: 3600, // 1 hour cooldown
        }
    }
}

/// Serialization helper for Duration
pub mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ihsan_dimensions_aggregate() {
        let dims = IhsanDimensions {
            correctness: 1.0,
            safety: 1.0,
            user_benefit: 1.0,
            efficiency: 1.0,
            auditability: 1.0,
            anti_centralization: 1.0,
            robustness: 1.0,
            adl_fairness: 1.0,
        };

        // Sum of weights = 0.22 + 0.22 + 0.14 + 0.12 + 0.12 + 0.08 + 0.06 + 0.04 = 1.0
        assert!((dims.aggregate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ihsan_dimensions_min() {
        let dims = IhsanDimensions {
            correctness: 0.9,
            safety: 0.95,
            user_benefit: 0.85, // This is the minimum
            efficiency: 0.88,
            auditability: 0.92,
            anti_centralization: 0.9,
            robustness: 0.87,
            adl_fairness: 0.91,
        };

        let (name, score) = dims.min_dimension();
        assert_eq!(name, "user_benefit");
        assert!((score - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_config_defaults() {
        let config = AutopoieticConfig::default();
        assert_eq!(config.generation_duration_ms, 60_000);
        assert!((config.ihsan_threshold - 0.95).abs() < 0.001);
        assert!(config.warm_pool_enabled);
    }

    #[test]
    fn test_kep_state_default() {
        let state = KEPState::default();
        assert_eq!(state, KEPState::Normal);
    }
}
