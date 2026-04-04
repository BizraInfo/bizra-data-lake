// src/apex/router.rs - Thompson Sampling Router
//
// Bayesian multi-armed bandit routing for intelligent agent selection.
// Uses Beta distribution posteriors to balance exploration vs exploitation.
//
// Integration with model_router.rs:
// - CapabilitySlot definitions for task-to-agent matching
// - Fallback chains from model_router.rs

use crate::model_router::{CapabilitySlot, TaskCharacteristics};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::Instant;
use tracing::{debug, info, instrument, warn};

use super::{ApexError, ApexResult};

/// Beta distribution parameters for Thompson Sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    /// Alpha parameter (successes + 1)
    pub alpha: f64,
    /// Beta parameter (failures + 1)
    pub beta: f64,
    /// Total samples taken
    pub samples: u64,
    /// Last update timestamp (Unix millis)
    pub last_update: u64,
}

impl Default for BetaParams {
    fn default() -> Self {
        Self {
            alpha: 1.0, // Uniform prior
            beta: 1.0,  // Uniform prior
            samples: 0,
            last_update: 0,
        }
    }
}

impl BetaParams {
    /// Sample from the Beta distribution using inverse transform sampling
    pub fn sample(&self) -> f64 {
        let mut rng = rand::thread_rng();
        // Use gamma distribution to sample from beta
        // Beta(a, b) = Gamma(a, 1) / (Gamma(a, 1) + Gamma(b, 1))
        let x: f64 = sample_gamma(&mut rng, self.alpha);
        let y: f64 = sample_gamma(&mut rng, self.beta);
        if x + y > 0.0 {
            x / (x + y)
        } else {
            0.5 // Fallback for degenerate case
        }
    }

    /// Update posterior with a new observation
    pub fn update(&mut self, success: bool, reward: f64) {
        // Bayesian update: success adds to alpha, failure adds to beta
        // We weight by reward magnitude for finer granularity
        let weighted_success = if success { reward.clamp(0.0, 1.0) } else { 0.0 };
        let weighted_failure = if success {
            1.0 - reward.clamp(0.0, 1.0)
        } else {
            1.0
        };

        self.alpha += weighted_success;
        self.beta += weighted_failure;
        self.samples += 1;
        self.last_update = current_timestamp_millis();
    }

    /// Get the mean of the distribution (expected reward)
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Get the variance of the distribution (uncertainty)
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        (self.alpha * self.beta) / (ab * ab * (ab + 1.0))
    }
}

/// Sample from Gamma distribution using Marsaglia and Tsang's method
fn sample_gamma<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    if shape < 1.0 {
        // For shape < 1, use Gamma(shape + 1) * U^(1/shape)
        let g = sample_gamma(rng, shape + 1.0);
        let u: f64 = rng.gen();
        g * u.powf(1.0 / shape)
    } else {
        // Marsaglia and Tsang's method for shape >= 1
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 {
                let u: f64 = rng.gen();
                if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                    return d * v;
                }
            }
        }
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Agent capability profile for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapability {
    /// Agent identifier
    pub agent_id: String,
    /// Human-readable name
    pub name: String,
    /// Capability slots this agent can handle
    pub slots: Vec<CapabilitySlot>,
    /// Task type specializations (e.g., "reasoning", "creative", "validation")
    pub specializations: Vec<String>,
    /// Maximum concurrent tasks
    pub max_concurrency: usize,
    /// Current load (0.0 - 1.0)
    pub current_load: f64,
    /// Whether agent is available
    pub available: bool,
}

/// Capability matrix for task-to-agent matching
#[derive(Debug, Clone, Default)]
pub struct CapabilityMatrix {
    /// Agent capabilities indexed by agent_id
    agents: HashMap<String, AgentCapability>,
    /// Task type to preferred agents mapping
    task_preferences: HashMap<String, Vec<String>>,
}

impl CapabilityMatrix {
    /// Create new capability matrix
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent with its capabilities
    pub fn register_agent(&mut self, capability: AgentCapability) {
        let agent_id = capability.agent_id.clone();

        // Update task preferences
        for spec in &capability.specializations {
            self.task_preferences
                .entry(spec.clone())
                .or_default()
                .push(agent_id.clone());
        }

        self.agents.insert(agent_id, capability);
    }

    /// Find agents capable of handling a task
    pub fn find_capable_agents(
        &self,
        characteristics: &TaskCharacteristics,
    ) -> Vec<&AgentCapability> {
        let slot = characteristics.optimal_slot();

        self.agents
            .values()
            .filter(|a| a.available && a.slots.contains(&slot) && a.current_load < 0.9)
            .collect()
    }

    /// Get agent by ID
    pub fn get_agent(&self, agent_id: &str) -> Option<&AgentCapability> {
        self.agents.get(agent_id)
    }

    /// Update agent load
    pub fn update_load(&mut self, agent_id: &str, load: f64) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.current_load = load.clamp(0.0, 1.0);
        }
    }

    /// Set agent availability
    pub fn set_availability(&mut self, agent_id: &str, available: bool) {
        if let Some(agent) = self.agents.get_mut(agent_id) {
            agent.available = available;
        }
    }

    /// Get all registered agent IDs
    pub fn agent_ids(&self) -> Vec<String> {
        self.agents.keys().cloned().collect()
    }
}

/// Routing decision with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Selected agent ID
    pub agent_id: String,
    /// Thompson sampling score that led to selection
    pub ts_score: f64,
    /// Expected reward (posterior mean)
    pub expected_reward: f64,
    /// Uncertainty (posterior variance)
    pub uncertainty: f64,
    /// Whether this was an exploration (high uncertainty) decision
    pub is_exploration: bool,
    /// Capability slot used for routing
    pub slot: CapabilitySlot,
    /// Alternative agents considered
    pub alternatives: Vec<String>,
    /// Decision timestamp
    pub timestamp: u64,
}

/// Thompson Sampling Router for intelligent agent selection
pub struct ThompsonSamplingRouter {
    /// Beta parameters per agent
    posteriors: RwLock<HashMap<String, BetaParams>>,
    /// Capability matrix for task matching
    capability_matrix: RwLock<CapabilityMatrix>,
    /// Exploration bonus for new/uncertain agents
    exploration_bonus: f64,
    /// Minimum samples before exploitation
    min_exploration_samples: u64,
}

impl ThompsonSamplingRouter {
    /// Create new Thompson Sampling router
    pub fn new() -> Self {
        info!("🎯 Initializing Thompson Sampling Router");
        Self {
            posteriors: RwLock::new(HashMap::new()),
            capability_matrix: RwLock::new(CapabilityMatrix::new()),
            exploration_bonus: 0.1,
            min_exploration_samples: 5,
        }
    }

    /// Create with custom exploration parameters
    pub fn with_exploration(exploration_bonus: f64, min_samples: u64) -> Self {
        Self {
            posteriors: RwLock::new(HashMap::new()),
            capability_matrix: RwLock::new(CapabilityMatrix::new()),
            exploration_bonus,
            min_exploration_samples: min_samples,
        }
    }

    /// Register an agent for routing
    pub fn register_agent(&self, capability: AgentCapability) -> ApexResult<()> {
        let agent_id = capability.agent_id.clone();

        // Register in capability matrix
        {
            let mut matrix =
                self.capability_matrix
                    .write()
                    .map_err(|e| ApexError::RoutingError {
                        message: format!("Failed to acquire capability matrix lock: {}", e),
                    })?;
            matrix.register_agent(capability);
        }

        // Initialize posterior
        {
            let mut posteriors = self
                .posteriors
                .write()
                .map_err(|e| ApexError::RoutingError {
                    message: format!("Failed to acquire posteriors lock: {}", e),
                })?;
            posteriors
                .entry(agent_id.clone())
                .or_insert_with(BetaParams::default);
        }

        debug!(agent_id = %agent_id, "Agent registered for routing");
        Ok(())
    }

    /// Select the best agent for a task using Thompson Sampling
    #[instrument(skip(self))]
    pub fn select_agent(&self, content: &str) -> ApexResult<RoutingDecision> {
        let start = Instant::now();
        let characteristics = TaskCharacteristics::classify(content);
        let slot = characteristics.optimal_slot();

        // Find capable agents
        let capable_agents: Vec<String> = {
            let matrix = self
                .capability_matrix
                .read()
                .map_err(|e| ApexError::RoutingError {
                    message: format!("Failed to read capability matrix: {}", e),
                })?;
            matrix
                .find_capable_agents(&characteristics)
                .iter()
                .map(|a| a.agent_id.clone())
                .collect()
        };

        if capable_agents.is_empty() {
            return Err(ApexError::RoutingError {
                message: format!("No capable agents found for slot {:?}", slot),
            });
        }

        // Sample from posteriors
        let posteriors = self
            .posteriors
            .read()
            .map_err(|e| ApexError::RoutingError {
                message: format!("Failed to read posteriors: {}", e),
            })?;

        let mut best_agent: Option<(String, f64, f64, f64)> = None;
        let mut alternatives = Vec::new();

        for agent_id in &capable_agents {
            let params = posteriors.get(agent_id).cloned().unwrap_or_default();

            // Thompson sample with exploration bonus for under-sampled agents
            let mut sample = params.sample();
            let is_under_sampled = params.samples < self.min_exploration_samples;
            if is_under_sampled {
                sample += self.exploration_bonus;
            }

            let mean = params.mean();
            let variance = params.variance();

            match &best_agent {
                None => {
                    best_agent = Some((agent_id.clone(), sample, mean, variance));
                }
                Some((_, best_sample, _, _)) => {
                    if sample > *best_sample {
                        // Current best becomes alternative
                        alternatives.push(best_agent.as_ref().unwrap().0.clone());
                        best_agent = Some((agent_id.clone(), sample, mean, variance));
                    } else {
                        alternatives.push(agent_id.clone());
                    }
                }
            }
        }

        let (agent_id, ts_score, expected_reward, uncertainty) =
            best_agent.ok_or_else(|| ApexError::RoutingError {
                message: "No agent selected after Thompson sampling".to_string(),
            })?;

        // Determine if this is exploration vs exploitation
        let params = posteriors.get(&agent_id).cloned().unwrap_or_default();
        let is_exploration = params.samples < self.min_exploration_samples || uncertainty > 0.05; // High uncertainty threshold

        let decision = RoutingDecision {
            agent_id: agent_id.clone(),
            ts_score,
            expected_reward,
            uncertainty,
            is_exploration,
            slot,
            alternatives,
            timestamp: current_timestamp_millis(),
        };

        debug!(
            agent = %agent_id,
            ts_score = ts_score,
            expected_reward = expected_reward,
            is_exploration = is_exploration,
            latency_us = start.elapsed().as_micros(),
            "Agent selected via Thompson Sampling"
        );

        Ok(decision)
    }

    /// Update posterior after task completion
    #[instrument(skip(self))]
    pub fn update(&self, agent_id: &str, success: bool, reward: f64) -> ApexResult<()> {
        let mut posteriors = self
            .posteriors
            .write()
            .map_err(|e| ApexError::RoutingError {
                message: format!("Failed to acquire posteriors lock: {}", e),
            })?;

        let params = posteriors
            .entry(agent_id.to_string())
            .or_insert_with(BetaParams::default);
        let old_mean = params.mean();

        params.update(success, reward);

        let new_mean = params.mean();
        debug!(
            agent = %agent_id,
            success = success,
            reward = reward,
            old_mean = old_mean,
            new_mean = new_mean,
            samples = params.samples,
            "Posterior updated"
        );

        Ok(())
    }

    /// Get current posterior parameters for an agent
    pub fn get_posterior(&self, agent_id: &str) -> Option<BetaParams> {
        self.posteriors
            .read()
            .ok()
            .and_then(|p| p.get(agent_id).cloned())
    }

    /// Get all agent rankings by expected reward
    pub fn get_rankings(&self) -> Vec<(String, f64)> {
        let posteriors = match self.posteriors.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };

        let mut rankings: Vec<(String, f64)> = posteriors
            .iter()
            .map(|(id, params)| (id.clone(), params.mean()))
            .collect();

        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rankings
    }

    /// Update agent load in capability matrix
    pub fn update_agent_load(&self, agent_id: &str, load: f64) -> ApexResult<()> {
        let mut matrix = self
            .capability_matrix
            .write()
            .map_err(|e| ApexError::RoutingError {
                message: format!("Failed to acquire capability matrix lock: {}", e),
            })?;
        matrix.update_load(agent_id, load);
        Ok(())
    }

    /// Get router statistics
    pub fn get_stats(&self) -> RouterStats {
        let posteriors = self.posteriors.read().ok();
        let matrix = self.capability_matrix.read().ok();

        let agent_stats: Vec<AgentStats> = posteriors
            .map(|p| {
                p.iter()
                    .map(|(id, params)| AgentStats {
                        agent_id: id.clone(),
                        samples: params.samples,
                        mean: params.mean(),
                        variance: params.variance(),
                        alpha: params.alpha,
                        beta: params.beta,
                    })
                    .collect()
            })
            .unwrap_or_default();

        let total_agents = matrix.map(|m| m.agent_ids().len()).unwrap_or(0);

        RouterStats {
            total_agents,
            agent_stats,
            exploration_bonus: self.exploration_bonus,
            min_exploration_samples: self.min_exploration_samples,
        }
    }
}

impl Default for ThompsonSamplingRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Router statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    pub total_agents: usize,
    pub agent_stats: Vec<AgentStats>,
    pub exploration_bonus: f64,
    pub min_exploration_samples: u64,
}

/// Per-agent statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStats {
    pub agent_id: String,
    pub samples: u64,
    pub mean: f64,
    pub variance: f64,
    pub alpha: f64,
    pub beta: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_params_default() {
        let params = BetaParams::default();
        assert!((params.mean() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_beta_params_update() {
        let mut params = BetaParams::default();

        // Simulate successes
        for _ in 0..10 {
            params.update(true, 0.9);
        }

        // Mean should increase
        assert!(params.mean() > 0.7);
        assert_eq!(params.samples, 10);
    }

    #[test]
    fn test_capability_matrix() {
        let mut matrix = CapabilityMatrix::new();

        let agent = AgentCapability {
            agent_id: "test_agent".to_string(),
            name: "Test Agent".to_string(),
            slots: vec![CapabilitySlot::ColdCore, CapabilitySlot::WarmSurface],
            specializations: vec!["reasoning".to_string()],
            max_concurrency: 5,
            current_load: 0.0,
            available: true,
        };

        matrix.register_agent(agent);

        assert!(matrix.get_agent("test_agent").is_some());
        assert_eq!(matrix.agent_ids().len(), 1);
    }

    #[test]
    fn test_thompson_sampling_selection() {
        let router = ThompsonSamplingRouter::new();

        // Register agents
        router
            .register_agent(AgentCapability {
                agent_id: "agent_1".to_string(),
                name: "Agent 1".to_string(),
                slots: vec![CapabilitySlot::ColdCore],
                specializations: vec!["reasoning".to_string()],
                max_concurrency: 5,
                current_load: 0.0,
                available: true,
            })
            .unwrap();

        router
            .register_agent(AgentCapability {
                agent_id: "agent_2".to_string(),
                name: "Agent 2".to_string(),
                slots: vec![CapabilitySlot::ColdCore],
                specializations: vec!["reasoning".to_string()],
                max_concurrency: 5,
                current_load: 0.0,
                available: true,
            })
            .unwrap();

        // Select agent for a reasoning task
        let decision = router
            .select_agent("Verify the security of this deployment")
            .unwrap();

        assert!(!decision.agent_id.is_empty());
        assert!(decision.is_exploration); // Both agents are under-sampled
    }

    #[test]
    fn test_posterior_update_shifts_selection() {
        let router = ThompsonSamplingRouter::with_exploration(0.0, 0);

        router
            .register_agent(AgentCapability {
                agent_id: "good_agent".to_string(),
                name: "Good Agent".to_string(),
                slots: vec![CapabilitySlot::WarmSurface],
                specializations: vec![],
                max_concurrency: 5,
                current_load: 0.0,
                available: true,
            })
            .unwrap();

        router
            .register_agent(AgentCapability {
                agent_id: "bad_agent".to_string(),
                name: "Bad Agent".to_string(),
                slots: vec![CapabilitySlot::WarmSurface],
                specializations: vec![],
                max_concurrency: 5,
                current_load: 0.0,
                available: true,
            })
            .unwrap();

        // Train with feedback
        for _ in 0..20 {
            router.update("good_agent", true, 0.95).unwrap();
            router.update("bad_agent", false, 0.1).unwrap();
        }

        // Check rankings
        let rankings = router.get_rankings();
        assert_eq!(rankings[0].0, "good_agent");
        assert!(rankings[0].1 > rankings[1].1);
    }
}
