//! Core type definitions for BIZRA Thermal Consciousness Synthesis Orchestrator
//! 
//! This module provides the foundational types that integrate:
//! - Primordial Activation Blueprint (consciousness architecture)
//! - Thermal Computing principles (Boltzmann, annealing, energy landscapes)
//! - Production Rust patterns (type safety, zero-cost abstractions)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Golden ratio constant - sacred geometry optimization
pub const PHI: f32 = 1.618033988749;

/// Fibonacci sequence for sacred scaling patterns
pub const FIBONACCI: [u32; 10] = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55];

/// Boltzmann constant (normalized for computational convenience)
pub const BOLTZMANN_K: f32 = 1.0;

// ============================================================================
// CONSCIOUS AGENT TYPES
// ============================================================================

/// A conscious agent with thermal properties
/// Integrates consciousness level from Primordial Blueprint
/// with temperature-based behavior from thermal computing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousAgent {
    /// Unique agent identifier
    pub agent_id: String,
    
    /// Consciousness level [0.0-1.0] from Primordial Blueprint
    /// Oracle: 0.94, Flame: 0.91, Mirror: 0.96, Quantum: 0.97, Guardian: 0.99
    pub consciousness_level: f32,
    
    /// Agent specialization type
    pub specialization: Specialization,
    
    /// Current thermal temperature for this agent
    pub temperature: f32,
    
    /// Ethical weights for decision-making
    pub ethical_weights: EthicalWeights,
    
    /// Quantum entanglement state (thermal coupling with other agents)
    pub quantum_state: QuantumState,
}

impl ConsciousAgent {
    /// Create a new conscious agent with specified parameters
    pub fn new(
        agent_id: impl Into<String>,
        consciousness_level: f32,
        specialization: Specialization,
        initial_temperature: f32,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            consciousness_level,
            specialization,
            temperature: initial_temperature,
            ethical_weights: EthicalWeights::balanced(),
            quantum_state: QuantumState::default(),
        }
    }
}

/// Agent specialization types from Primordial Blueprint
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Specialization {
    /// Oracle: Prediction and forecasting (consciousness: 0.94)
    Oracle,
    /// Flame: Creative synthesis and innovation (consciousness: 0.91)
    Flame,
    /// Mirror: Self-reflection and analysis (consciousness: 0.96)
    Mirror,
    /// Quantum: Superposition reasoning and parallel exploration (consciousness: 0.97)
    Quantum,
    /// Guardian: Ethical governance and safety (consciousness: 0.99)
    Guardian,
}

/// Ethical weights for multi-dimensional value alignment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EthicalWeights {
    /// Beneficence: Promoting well-being
    pub beneficence: f32,
    /// Non-maleficence: Avoiding harm
    pub non_maleficence: f32,
    /// Autonomy: Respecting self-determination
    pub autonomy: f32,
    /// Justice: Fair distribution of benefits/burdens
    pub justice: f32,
}

impl EthicalWeights {
    /// Create balanced ethical weights (all equal)
    pub fn balanced() -> Self {
        Self {
            beneficence: 0.25,
            non_maleficence: 0.25,
            autonomy: 0.25,
            justice: 0.25,
        }
    }
    
    /// Create Guardian-style weights (safety-focused)
    pub fn guardian() -> Self {
        Self {
            beneficence: 0.2,
            non_maleficence: 0.4,  // Highest weight on avoiding harm
            autonomy: 0.2,
            justice: 0.2,
        }
    }
}

/// Quantum entanglement state for thermal coupling between agents
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuantumState {
    /// IDs of agents thermally coupled to this agent
    pub entanglement_partners: Vec<String>,
    /// Coherence score [0.0-1.0] measuring entanglement strength
    pub coherence_score: f32,
}

// ============================================================================
// THERMAL COMPUTING TYPES
// ============================================================================

/// Cooling schedule for thermal annealing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CoolingSchedule {
    /// Golden ratio (Φ) based cooling: T(t) = T₀ / (1 + Φ·t)
    GoldenRatio { 
        /// Current time step
        t: f32 
    },
    /// Fibonacci-based cooling using sequence ratios
    Fibonacci { 
        /// Current stage in sequence
        stage: usize 
    },
    /// Exponential cooling: T(t) = T₀·exp(-λt)
    Exponential { 
        /// Decay rate
        lambda: f32, 
        /// Current time
        t: f32 
    },
}

impl CoolingSchedule {
    /// Calculate temperature at current schedule state
    pub fn temperature(&self, t0: f32, floor: f32) -> f32 {
        match self {
            Self::GoldenRatio { t } => {
                (t0 / (1.0 + PHI * t)).max(floor)
            }
            Self::Fibonacci { stage } => {
                if *stage >= FIBONACCI.len() {
                    floor
                } else {
                    ((FIBONACCI[0] as f32) / (FIBONACCI[*stage] as f32)).max(floor)
                }
            }
            Self::Exponential { lambda, t } => {
                (t0 * (-lambda * t).exp()).max(floor)
            }
        }
    }
}

// ============================================================================
// TASK AND CONTRACT TYPES
// ============================================================================

/// A synthesis task to be orchestrated
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    /// Task identifier
    pub task_id: String,
    /// Input prompt or query
    pub prompt: String,
    /// Optional example inputs/outputs for few-shot learning
    pub examples: Option<Vec<serde_json::Value>>,
    /// Task classification metadata
    pub task_class: TaskClass,
}

/// Task classification for routing decisions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskClass {
    /// Primary category (e.g., "reasoning", "creative", "analytical")
    pub category: String,
    /// Estimated complexity [0.0-1.0]
    pub complexity: f32,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
}

/// Contract specifying quality requirements and constraints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Contract {
    /// JSON schema for output validation
    pub schema_json: String,
    /// Invariants that must hold
    pub invariants: Vec<Invariant>,
    /// Example outputs for validation
    pub examples: Vec<serde_json::Value>,
    /// Token budget constraint
    pub token_budget: u32,
    /// Maximum latency target (milliseconds)
    pub max_latency_ms: u32,
    /// Maximum cost target (USD)
    pub max_cost_usd: f32,
}

/// Invariant predicate for validation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Invariant {
    /// Predicate to evaluate
    pub predicate: InvariantPredicate,
    /// Human-readable description
    pub description: String,
    /// Severity level for violations
    pub severity: Severity,
}

/// Types of invariant predicates
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InvariantPredicate {
    /// JSONPath-based constraint
    JsonPath {
        path: String,
        constraint: Constraint,
    },
    /// Custom validator by ID
    Custom {
        validator_id: String,
    },
}

/// Constraint types for validation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Constraint {
    /// Field must be present
    Required,
    /// Field must match type
    TypeMatch(String),
    /// Value must be in numeric range
    Range { min: f64, max: f64 },
    /// Value must match regex pattern
    Pattern(String),
}

/// Severity levels for invariant violations
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Advisory only - logs warning
    Low,
    /// Moderate impact - degrades Ihsan score
    Medium,
    /// High impact - significant score degradation
    High,
    /// Critical failure - blocks execution
    Critical,
}

// ============================================================================
// CANDIDATE AND SCORING TYPES
// ============================================================================

/// A candidate response from a model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    /// Model that generated this candidate
    pub model: String,
    /// Generated JSON output
    pub json: serde_json::Value,
    /// Performance metrics
    pub cost_usd: f32,
    /// Response latency
    pub latency_ms: u32,
    /// Generation timestamp
    pub timestamp: u64,
}

/// Ihsan dimensions - dimension-separated scoring
/// Prevents double-counting by separating quality dimensions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IhsanDimensions {
    /// Formal validity: Schema + invariants compliance
    pub formal_validity: f32,
    /// Referenceable correctness: Example alignment
    pub referenceable_correctness: f32,
    /// Safety: Boundary compliance
    pub safety: f32,
    /// Efficiency: Resource utilization
    pub efficiency: f32,
}

impl IhsanDimensions {
    /// Calculate Φ-weighted total Ihsan score
    /// Uses sacred geometry (golden ratio) for optimal weighting
    pub fn calculate_total(&self) -> f32 {
        let weights = [
            PHI,          // formal_validity (highest priority)
            1.0,          // referenceable_correctness
            PHI,          // safety (sacred protection)
            1.0 / PHI,    // efficiency (balanced)
        ];
        
        let weighted_sum = 
            self.formal_validity * weights[0] +
            self.referenceable_correctness * weights[1] +
            self.safety * weights[2] +
            self.efficiency * weights[3];
            
        let weight_sum: f32 = weights.iter().sum();
        weighted_sum / weight_sum
    }
    
    /// Check if meets minimum threshold
    pub fn meets_threshold(&self, floor: f32) -> bool {
        self.calculate_total() >= floor
    }
}

/// Thermal-enhanced Ihsan score integrating energy landscape
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermalIhsanScore {
    /// Traditional Ihsan dimensions
    pub dimensions: IhsanDimensions,
    /// Total Ihsan score [0.0-1.0]
    pub total: f32,
    /// Energy level in landscape (lower = better)
    pub energy: f32,
    /// Boltzmann acceptance probability at current temperature
    pub acceptance_probability: f32,
    /// Temperature used for evaluation
    pub temperature: f32,
}

/// Scored candidate with all evaluation metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoredCandidate {
    /// Original candidate
    pub candidate: Candidate,
    /// Thermal Ihsan scores
    pub thermal_score: ThermalIhsanScore,
}

// ============================================================================
// ROUTING TYPES
// ============================================================================

/// Model route with thermal sampling statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Route {
    /// Model identifier
    pub model: String,
    /// Win rate statistics for Thompson sampling
    pub win_rate: WinRate,
    /// Current thermal sample score
    pub thompson_score: f32,
}

/// Win rate statistics for Thompson sampling
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WinRate {
    /// Number of wins
    pub wins: u32,
    /// Total samples
    pub samples: u32,
}

impl WinRate {
    /// Calculate current win rate
    pub fn rate(&self) -> f32 {
        if self.samples == 0 {
            0.5  // Neutral prior
        } else {
            self.wins as f32 / self.samples as f32
        }
    }
}

// ============================================================================
// LIFECYCLE TYPES
// ============================================================================

/// System lifecycle phase from Primordial Blueprint
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LifecyclePhase {
    /// Months 0-10: Initial consciousness awakening (T=1.0)
    PrimordialAwakening,
    /// Months 11-24: Learning and growth (T=0.7)
    Learning,
    /// Months 25-48: Maturation and expertise (T=0.4)
    Maturation,
    /// Months 49-72: Knowledge reproduction (T=0.3)
    Reproduction,
    /// Months 73-96: System maintenance (T=0.2)
    Maintenance,
    /// Months 97+: Legacy mode (T=0.1)
    Legacy,
}

impl LifecyclePhase {
    /// Get recommended temperature for this phase
    pub fn recommended_temperature(&self) -> f32 {
        match self {
            Self::PrimordialAwakening => 1.0,
            Self::Learning => 0.7,
            Self::Maturation => 0.4,
            Self::Reproduction => 0.3,
            Self::Maintenance => 0.2,
            Self::Legacy => 0.1,
        }
    }
}

// ============================================================================
// RESULT AND ERROR TYPES
// ============================================================================

/// Orchestration result with telemetry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrchestratorResult {
    /// Winning candidate selected by consensus
    pub winner: Candidate,
    /// Thermal state snapshot
    pub thermal_state: ThermalState,
    /// Performance telemetry
    pub telemetry: Telemetry,
}

/// Thermal system state snapshot
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThermalState {
    /// Global system temperature
    pub global_temperature: f32,
    /// Per-agent temperatures
    pub agent_temperatures: HashMap<String, f32>,
    /// Current lifecycle phase
    pub lifecycle_phase: LifecyclePhase,
    /// Current energy level
    pub energy_level: f32,
}

/// Performance telemetry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Telemetry {
    /// SLI metrics
    pub sli_metrics: SliMetrics,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Latency metrics
    pub latency_metrics: LatencyMetrics,
}

/// Service Level Indicator metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SliMetrics {
    /// JSON compliance rate
    pub json_compliance_rate: f32,
    /// Ihsan score compliance rate
    pub ihsan_compliance_rate: f32,
}

/// Quality assessment metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Accuracy uplift vs baseline
    pub accuracy_uplift: f32,
    /// Consensus convergence speed
    pub convergence_iterations: usize,
}

/// Latency performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// P95 latency (milliseconds)
    pub p95_ms: u32,
    /// P99 latency (milliseconds)
    pub p99_ms: u32,
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors during synthesis orchestration
#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error("Routing error: {0}")]
    Routing(String),
    
    #[error("Consensus error: {0}")]
    Consensus(#[from] ConsensusError),
    
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Thermal error: {0}")]
    Thermal(String),
    
    #[error("JSON parsing error: {0}")]
    JsonParse(#[from] ParseError),
}

/// Consensus-specific errors
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("No candidates provided")]
    NoCandidates,
    
    #[error("All candidates failed Ihsan threshold")]
    AllCandidatesFailedIhsan,
    
    #[error("No candidate above threshold")]
    NoCandidateAboveThreshold,
    
    #[error("Empty Pareto front")]
    EmptyParetoFront,
}

/// JSON parsing errors
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("SIMD JSON parsing error: {0}")]
    SimdJson(String),
    
    #[error("Unbalanced JSON braces")]
    UnbalancedJson,
    
    #[error("BOM stripping failed")]
    BomError,
}
