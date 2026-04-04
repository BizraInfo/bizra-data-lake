// src/autopoietic/mod.rs - BIZRA NUCLEUS AutopoieticLoop Module
//
// Self-creating, self-improving autonomous engine implementing:
// - Interdisciplinary Thinking: Hyperon/MeTTa logic + 0G storage + FATE ethics
// - Graph of Thoughts: Non-linear reasoning across knowledge domains
// - SNR Optimization: Highest signal-to-noise autonomous decisions
// - Standing on Giants: Ralph loops + RLM recursion + MassGen collaboration
//
// The 11-step cycle:
//   1. Create agents from current blueprints (reuse AgentFactory warm pools)
//   2. Deploy agents (PATOrchestrator + A2A registration)
//   3. Monitor for generation_duration (OperationMonitor metrics)
//   4. Evaluate (operational + environment + economic + ethical - Ihsān 8-dim + SAPE 9-probe)
//   5. Improve blueprints (ImprovementGenome.evolve() with FATE gate)
//   6. Record GenerationPerformance (Extended ExecutionReceipt + KEP fields)
//   7. Update current_blueprints (apply improvements, maintain lineage)
//   8. Update proof chain (Merkle append + blockchain anchor)
//   9. Economic/ethical model updates (adjust incentives based on history)
//  10. Check convergence (KEP detection - plateau vs explosion)
//  11. Increment counter and persist (Redis via Synapse)

pub mod blueprints;
pub mod convergence;
pub mod evaluation;
pub mod loop_engine;
pub mod proof_chain;
pub mod step9_implementation;
pub mod types;

// Re-export core types for convenience
pub use types::{
    AutopoieticConfig, AutopoieticError, AutopoieticStatus, GenerationPerformance, KEPProgress,
    KEPState, KEPThresholds,
};

pub use blueprints::{
    AgentBlueprint, AgentTeam, CapabilitySlot, FitnessCriterion, ImprovementGenome, PromptMutation,
    RoutingPreferences,
};

pub use evaluation::{
    EconomicMetrics, EnvironmentMetrics, EthicalMetrics, EvaluationResult, OperationMonitor,
    OperationalMetrics,
};

pub use convergence::{ConvergenceDetector, ConvergenceMetrics, ConvergenceState};

pub use proof_chain::{BlockchainAnchor, EvolutionProof, ProofChain, ProofNode};

pub use loop_engine::{AutopoieticEvent, AutopoieticLoop, LoopControl};
