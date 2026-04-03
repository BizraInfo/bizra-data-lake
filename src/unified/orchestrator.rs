// src/unified/orchestrator.rs - The Unified Orchestrator
//
// SAPE v1.∞: The Cybernetic Organism
// ====================================
// Brain (Python) + Body (Rust) + Soul (Ihsān Protocol)
//
// This is the main autopoetic loop that:
// 1. Manages agent lifecycle with cryptographic attestation
// 2. Routes cognition through the Python Brain
// 3. Enforces Ihsān constraints at every step
// 4. Distills wisdom from successful executions

use super::{
    attestor::{AttestorError, CryptographicAttestor},
    cognitive_bridge::{
        CognitiveBridge, CognitiveErrorCode, CognitiveRequest, CognitiveResponse, ThinkingMode,
    },
    wisdom::{ActionPrimitive, Symbol, WisdomAtom, WisdomStore},
};
use crate::{entropy::global_pool, fate::FATECoordinator, ihsan};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tracing::{debug, error, info, instrument, warn};

/// Agent state in the evolutionary pool
#[derive(Debug, Clone)]
pub struct AgentState {
    pub id: String,
    pub generation: u32,
    pub fitness_score: f64,
    pub ihsan_score: f64,
    pub context_vector: Vec<f32>,
    pub last_active: u64,
    pub status: AgentStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentStatus {
    Ready,
    Processing,
    Suspended,
    Terminated,
}

/// Evolution configuration
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,
    /// Selection pressure (top % to keep)
    pub selection_pressure: f64,
    /// Mutation rate (0.0-1.0)
    pub mutation_rate: f64,
    /// Minimum fitness to survive
    pub min_fitness: f64,
    /// Maximum generations
    pub max_generations: u32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            selection_pressure: 0.3,
            mutation_rate: 0.1,
            min_fitness: 0.5,
            max_generations: 1000,
        }
    }
}

/// Unified Orchestrator - The Heart of the Cybernetic Organism
///
/// Implements:
/// - Autopoetic Loop: Self-maintaining evolution
/// - Pipelined Consciousness: Async SAT updating PAT intuition
/// - Proactive Attestation: Every action is signed
/// - Wisdom Distillation: Successful patterns become atoms
pub struct UnifiedOrchestrator {
    // === The Brain ===
    cognitive_bridge: Arc<CognitiveBridge>,

    // === The Body ===
    /// Agent population
    agents: Arc<RwLock<HashMap<String, AgentState>>>,
    /// Evolution configuration
    evolution_config: EvolutionConfig,
    /// Current generation
    generation: std::sync::atomic::AtomicU32,

    // === The Soul ===
    /// Cryptographic attestor
    attestor: Arc<CryptographicAttestor>,
    /// Wisdom repository
    wisdom_store: Arc<WisdomStore>,
    /// FATE escalation coordinator
    fate: Arc<Mutex<FATECoordinator>>,

    // === Control ===
    /// Concurrency control
    processing_semaphore: Arc<Semaphore>,
    /// Cycle counter
    cycle_counter: std::sync::atomic::AtomicU64,
    /// Running flag
    running: std::sync::atomic::AtomicBool,
}

impl UnifiedOrchestrator {
    /// Create a new unified orchestrator
    pub async fn new(
        cognitive_bridge: CognitiveBridge,
        evolution_config: EvolutionConfig,
        max_concurrent_processing: usize,
    ) -> anyhow::Result<Self> {
        let constitution = ihsan::constitution();

        info!(
            population_size = evolution_config.population_size,
            ihsan_threshold = constitution.threshold(),
            "🦾 UnifiedOrchestrator initializing"
        );

        let orchestrator = Self {
            cognitive_bridge: Arc::new(cognitive_bridge),
            agents: Arc::new(RwLock::new(HashMap::new())),
            evolution_config,
            generation: std::sync::atomic::AtomicU32::new(1),
            attestor: Arc::new(CryptographicAttestor::new(
                constitution.threshold(),
                5, // max violations before revocation
            )),
            wisdom_store: Arc::new(WisdomStore::new(10000)),
            fate: Arc::new(Mutex::new(FATECoordinator::new())),
            processing_semaphore: Arc::new(Semaphore::new(max_concurrent_processing)),
            cycle_counter: std::sync::atomic::AtomicU64::new(0),
            running: std::sync::atomic::AtomicBool::new(false),
        };

        // Initialize agent population
        orchestrator.initialize_population().await?;

        info!("✅ UnifiedOrchestrator ready");
        Ok(orchestrator)
    }

    /// Initialize the agent population
    async fn initialize_population(&self) -> anyhow::Result<()> {
        let mut agents = self.agents.write().await;

        for i in 0..self.evolution_config.population_size {
            let agent_id = format!("agent-{:04}", i);

            // Register with attestor
            self.attestor
                .register_agent(&agent_id, None, 1, 0.95)
                .await
                .map_err(|e| anyhow::anyhow!("Failed to register agent: {}", e))?;

            // Create agent state
            let state = AgentState {
                id: agent_id.clone(),
                generation: 1,
                fitness_score: 0.5,
                ihsan_score: 0.95,
                context_vector: vec![0.0; 768], // Default embedding size
                last_active: Self::now(),
                status: AgentStatus::Ready,
            };

            agents.insert(agent_id, state);
        }

        info!(population = agents.len(), "Agent population initialized");

        Ok(())
    }

    /// Run the main autopoetic loop
    #[instrument(skip(self))]
    pub async fn run_cycle(&self) -> Result<CycleResult, OrchestratorError> {
        let cycle_start = Instant::now();
        let cycle_id = self
            .cycle_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        debug!(cycle = cycle_id, "🔄 Starting autopoetic cycle");

        // 1. Select agents for processing
        let selected_agents = self.select_agents().await;
        if selected_agents.is_empty() {
            return Err(OrchestratorError::NoAgentsAvailable);
        }

        // 2. Process agents in parallel (with backpressure)
        let mut futures = Vec::new();
        for agent in &selected_agents {
            let bridge = self.cognitive_bridge.clone();
            let attestor = self.attestor.clone();
            let wisdom_store = self.wisdom_store.clone();
            let permit = self
                .processing_semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| OrchestratorError::SemaphoreClosed)?;
            let agent_id = agent.id.clone();

            futures.push(tokio::spawn(async move {
                let result =
                    Self::process_agent(&bridge, &attestor, &wisdom_store, &agent_id).await;
                drop(permit);
                (agent_id, result)
            }));
        }

        // 3. Collect results
        let mut successes = 0;
        let mut failures = 0;
        let mut total_ihsan = 0.0;
        let mut total_snr = 0.0;
        let mut results = Vec::new();

        for future in futures {
            match future.await {
                Ok((agent_id, Ok(response))) => {
                    if response.success {
                        successes += 1;
                        total_ihsan += response.ihsan_score;
                        total_snr += response.snr_score;

                        // Update agent fitness
                        self.update_agent_fitness(&agent_id, &response).await;

                        // Potentially distill wisdom
                        if response.ihsan_score >= 0.99 && response.snr_score >= 20.0 {
                            self.distill_wisdom(&agent_id, &response).await;
                        }

                        results.push((agent_id, response));
                    } else {
                        failures += 1;
                        self.penalize_agent(&agent_id, &response.error_message.unwrap_or_default())
                            .await;
                    }
                }
                Ok((agent_id, Err(e))) => {
                    failures += 1;
                    warn!(agent = %agent_id, error = %e, "Agent processing failed");
                    self.penalize_agent(&agent_id, &e.to_string()).await;
                }
                Err(e) => {
                    failures += 1;
                    error!(error = %e, "Task join error");
                }
            }
        }

        // 4. Evolution step (if enough successes)
        if successes > 0 && cycle_id.is_multiple_of(10) {
            self.evolve().await?;
        }

        let cycle_time = cycle_start.elapsed();

        let result = CycleResult {
            cycle_id,
            agents_processed: selected_agents.len(),
            successes,
            failures,
            average_ihsan: if successes > 0 {
                total_ihsan / successes as f64
            } else {
                0.0
            },
            average_snr: if successes > 0 {
                total_snr / successes as f64
            } else {
                0.0
            },
            cycle_time,
            generation: self.generation.load(std::sync::atomic::Ordering::Relaxed),
            wisdom_atoms_created: self.wisdom_store.stats().await.total_atoms,
        };

        info!(
            cycle = cycle_id,
            successes = successes,
            failures = failures,
            avg_ihsan = %format!("{:.3}", result.average_ihsan),
            avg_snr = %format!("{:.1}", result.average_snr),
            time_ms = cycle_time.as_millis(),
            "✅ Cycle complete"
        );

        Ok(result)
    }

    /// Select agents for processing based on fitness
    async fn select_agents(&self) -> Vec<AgentState> {
        let agents = self.agents.read().await;

        let mut ready_agents: Vec<_> = agents
            .values()
            .filter(|a| a.status == AgentStatus::Ready)
            .cloned()
            .collect();

        // Sort by fitness (descending)
        ready_agents.sort_by(|a, b| b.fitness_score.partial_cmp(&a.fitness_score).unwrap());

        // Select top agents based on selection pressure
        let select_count =
            (ready_agents.len() as f64 * self.evolution_config.selection_pressure) as usize;
        ready_agents.into_iter().take(select_count.max(1)).collect()
    }

    /// Process a single agent through the cognitive pipeline
    async fn process_agent(
        bridge: &CognitiveBridge,
        attestor: &CryptographicAttestor,
        _wisdom_store: &WisdomStore,
        agent_id: &str,
    ) -> Result<CognitiveResponse, OrchestratorError> {
        // Check authorization
        let authorized = attestor
            .is_authorized(agent_id)
            .await
            .map_err(OrchestratorError::AttestationError)?;

        if !authorized {
            return Err(OrchestratorError::AgentUnauthorized(agent_id.to_string()));
        }

        // Create cognitive request
        let request = CognitiveRequest {
            agent_id: agent_id.to_string(),
            task_id: format!("task-{}", global_pool().next_sequence()),
            context_vector: Vec::new(),
            mode: ThinkingMode::HybridSynergy,
            prompt: "Analyze the current system state and suggest optimizations.".to_string(),
            metadata: HashMap::new(),
            min_snr_threshold: 15.0,
            min_ihsan_score: ihsan::constitution().threshold(),
            max_thinking_depth: 5,
            timeout_ms: 30000,
        };

        // Process through cognitive bridge
        let response = bridge.process(request).await;

        // Attest behavior
        let behavior_data = serde_json::to_vec(&response)
            .map_err(|e| OrchestratorError::SerializationError(e.to_string()))?;

        attestor
            .attest_behavior(
                agent_id,
                &behavior_data,
                response.ihsan_score,
                response.success,
            )
            .await
            .map_err(OrchestratorError::AttestationError)?;

        Ok(response)
    }

    /// Update agent fitness based on response
    async fn update_agent_fitness(&self, agent_id: &str, response: &CognitiveResponse) {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(agent_id) {
            // Fitness = weighted combination of metrics
            let new_fitness = 0.3 * response.confidence
                + 0.3 * response.ihsan_score
                + 0.2 * (response.snr_score / 30.0).min(1.0)
                + 0.2 * response.utility_score;

            // Exponential moving average
            agent.fitness_score = 0.2 * new_fitness + 0.8 * agent.fitness_score;
            agent.ihsan_score = 0.2 * response.ihsan_score + 0.8 * agent.ihsan_score;
            agent.last_active = Self::now();
        }
    }

    /// Penalize an agent for failure
    async fn penalize_agent(&self, agent_id: &str, reason: &str) {
        let mut agents = self.agents.write().await;
        if let Some(agent) = agents.get_mut(agent_id) {
            agent.fitness_score *= 0.9; // 10% penalty
            debug!(agent = agent_id, reason = reason, "Agent penalized");

            // Check for termination threshold
            if agent.fitness_score < self.evolution_config.min_fitness {
                agent.status = AgentStatus::Terminated;
                warn!(
                    agent = agent_id,
                    fitness = agent.fitness_score,
                    "Agent terminated - below minimum fitness"
                );
            }
        }
    }

    /// Distill wisdom from successful execution
    async fn distill_wisdom(&self, agent_id: &str, response: &CognitiveResponse) {
        // Create WisdomAtom from successful execution
        let preconditions = vec![
            Symbol::new("TaskReceived"),
            Symbol::new("IhsanThresholdMet"),
        ];
        let action = ActionPrimitive::emit(&response.synthesis);
        let postconditions = vec![Symbol::new("TaskCompleted"), Symbol::new("WisdomDistilled")];

        let atom = WisdomAtom::new(preconditions, action, postconditions, agent_id);

        if let Err(e) = self.wisdom_store.store(atom).await {
            warn!(error = %e, "Failed to store wisdom atom");
        } else {
            debug!(
                agent = agent_id,
                "Wisdom distilled from successful execution"
            );
        }
    }

    /// Evolutionary step - selection, mutation, reproduction
    async fn evolve(&self) -> Result<(), OrchestratorError> {
        let gen = self
            .generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut agents = self.agents.write().await;

        // Remove terminated agents
        agents.retain(|_, a| a.status != AgentStatus::Terminated);

        // Sort by fitness
        let mut sorted: Vec<_> = agents.values().cloned().collect();
        sorted.sort_by(|a, b| b.fitness_score.partial_cmp(&a.fitness_score).unwrap());

        // Keep top performers
        let keep_count = (sorted.len() as f64 * self.evolution_config.selection_pressure) as usize;
        let survivors: Vec<_> = sorted.into_iter().take(keep_count.max(1)).collect();

        // Create offspring
        let offspring_needed = self.evolution_config.population_size - agents.len();
        for i in 0..offspring_needed {
            let parent = &survivors[i % survivors.len()];
            let child_id = format!("agent-g{}-{:04}", gen + 1, i);

            // Register child with attestor
            if let Ok(_) = self
                .attestor
                .register_agent(
                    &child_id,
                    Some(parent.id.clone()),
                    gen + 1,
                    parent.ihsan_score,
                )
                .await
            {
                let child = AgentState {
                    id: child_id.clone(),
                    generation: gen + 1,
                    fitness_score: parent.fitness_score * 0.9, // Slight penalty for being new
                    ihsan_score: parent.ihsan_score,
                    context_vector: Self::mutate_context(
                        &parent.context_vector,
                        self.evolution_config.mutation_rate,
                    ),
                    last_active: Self::now(),
                    status: AgentStatus::Ready,
                };

                agents.insert(child_id, child);
            }
        }

        info!(
            generation = gen + 1,
            population = agents.len(),
            survivors = keep_count,
            "Evolution complete"
        );

        Ok(())
    }

    /// Mutate context vector
    fn mutate_context(context: &[f32], rate: f64) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        context
            .iter()
            .map(|v| {
                if rng.r#gen::<f64>() < rate {
                    v + (rng.r#gen::<f32>() - 0.5) * 0.1
                } else {
                    *v
                }
            })
            .collect()
    }

    /// Get current time as Unix timestamp
    fn now() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Get orchestrator statistics
    pub async fn stats(&self) -> OrchestratorStats {
        let agents = self.agents.read().await;
        let attestor_stats = self.attestor.stats().await;
        let wisdom_stats = self.wisdom_store.stats().await;

        let total_agents = agents.len();
        let ready_agents = agents
            .values()
            .filter(|a| a.status == AgentStatus::Ready)
            .count();
        let avg_fitness = if total_agents > 0 {
            agents.values().map(|a| a.fitness_score).sum::<f64>() / total_agents as f64
        } else {
            0.0
        };
        let avg_ihsan = if total_agents > 0 {
            agents.values().map(|a| a.ihsan_score).sum::<f64>() / total_agents as f64
        } else {
            0.0
        };

        OrchestratorStats {
            total_agents,
            ready_agents,
            terminated_agents: total_agents - ready_agents,
            current_generation: self.generation.load(std::sync::atomic::Ordering::Relaxed),
            cycles_completed: self
                .cycle_counter
                .load(std::sync::atomic::Ordering::Relaxed),
            average_fitness: avg_fitness,
            average_ihsan: avg_ihsan,
            wisdom_atoms: wisdom_stats.total_atoms,
            attestations: attestor_stats.total_attestations,
            violations: attestor_stats.total_violations,
            trust_level: attestor_stats.average_trust_level,
        }
    }

    /// Check if orchestrator is healthy
    pub async fn is_healthy(&self) -> bool {
        let agents = self.agents.read().await;
        let ready = agents
            .values()
            .filter(|a| a.status == AgentStatus::Ready)
            .count();
        ready >= self.evolution_config.population_size / 2
    }
}

/// Result of a single autopoetic cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    pub cycle_id: u64,
    pub agents_processed: usize,
    pub successes: usize,
    pub failures: usize,
    pub average_ihsan: f64,
    pub average_snr: f64,
    pub cycle_time: Duration,
    pub generation: u32,
    pub wisdom_atoms_created: usize,
}

/// Orchestrator statistics
#[derive(Debug, Clone)]
pub struct OrchestratorStats {
    pub total_agents: usize,
    pub ready_agents: usize,
    pub terminated_agents: usize,
    pub current_generation: u32,
    pub cycles_completed: u64,
    pub average_fitness: f64,
    pub average_ihsan: f64,
    pub wisdom_atoms: usize,
    pub attestations: u64,
    pub violations: u32,
    pub trust_level: f64,
}

/// Orchestrator errors
#[derive(Debug)]
pub enum OrchestratorError {
    NoAgentsAvailable,
    AgentUnauthorized(String),
    AttestationError(AttestorError),
    CognitionError(CognitiveErrorCode),
    SemaphoreClosed,
    SerializationError(String),
    EvolutionError(String),
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorError::NoAgentsAvailable => write!(f, "No agents available for processing"),
            OrchestratorError::AgentUnauthorized(id) => write!(f, "Agent {} not authorized", id),
            OrchestratorError::AttestationError(e) => write!(f, "Attestation error: {}", e),
            OrchestratorError::CognitionError(c) => write!(f, "Cognition error: {:?}", c),
            OrchestratorError::SemaphoreClosed => write!(f, "Processing semaphore closed"),
            OrchestratorError::SerializationError(e) => write!(f, "Serialization error: {}", e),
            OrchestratorError::EvolutionError(e) => write!(f, "Evolution error: {}", e),
        }
    }
}

impl std::error::Error for OrchestratorError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolution_config_default() {
        let config = EvolutionConfig::default();
        assert_eq!(config.population_size, 20);
        assert_eq!(config.selection_pressure, 0.3);
    }

    #[test]
    fn test_mutate_context() {
        let context = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mutated = UnifiedOrchestrator::mutate_context(&context, 1.0); // 100% mutation
        assert_eq!(mutated.len(), context.len());
        // Values should be slightly different
        assert!(context
            .iter()
            .zip(mutated.iter())
            .any(|(a, b)| (a - b).abs() > 0.001));
    }
}
