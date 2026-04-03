// src/autopoietic/blueprints.rs - Agent Evolution Blueprints
//
// Defines the genetic structure for agent evolution:
// - AgentBlueprint: Complete agent definition with evolution lineage
// - ImprovementGenome: Mutation and capability extension rules
// - PromptMutation: System prompt evolution operations
// - FitnessCriterion: Selection pressure for improvements

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Agent team classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentTeam {
    /// Personal Agentic Team - 7 agents for task execution
    PAT,
    /// System Agentic Team - 5 guardian agents for validation
    SAT,
}

impl std::fmt::Display for AgentTeam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentTeam::PAT => write!(f, "PAT"),
            AgentTeam::SAT => write!(f, "SAT"),
        }
    }
}

/// Capability slot for agent specialization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapabilitySlot {
    // PAT Slots
    MasterReasoner,
    MemoryArchitect,
    CreativeSynthesizer,
    DataAnalyzer,
    Communicator,
    ExecutionPlanner,
    EthicsGuardian,
    // SAT Slots
    PoiVerifier,
    ResourceAllocator,
    RiskGuardian,
    GovernanceEngine,
    EvidenceEngine,
    // Dynamic slots for spawned sub-agents
    Dynamic(String),
}

impl std::fmt::Display for CapabilitySlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilitySlot::MasterReasoner => write!(f, "MasterReasoner"),
            CapabilitySlot::MemoryArchitect => write!(f, "MemoryArchitect"),
            CapabilitySlot::CreativeSynthesizer => write!(f, "CreativeSynthesizer"),
            CapabilitySlot::DataAnalyzer => write!(f, "DataAnalyzer"),
            CapabilitySlot::Communicator => write!(f, "Communicator"),
            CapabilitySlot::ExecutionPlanner => write!(f, "ExecutionPlanner"),
            CapabilitySlot::EthicsGuardian => write!(f, "EthicsGuardian"),
            CapabilitySlot::PoiVerifier => write!(f, "PoiVerifier"),
            CapabilitySlot::ResourceAllocator => write!(f, "ResourceAllocator"),
            CapabilitySlot::RiskGuardian => write!(f, "RiskGuardian"),
            CapabilitySlot::GovernanceEngine => write!(f, "GovernanceEngine"),
            CapabilitySlot::EvidenceEngine => write!(f, "EvidenceEngine"),
            CapabilitySlot::Dynamic(name) => write!(f, "Dynamic:{}", name),
        }
    }
}

/// Complete agent blueprint with evolution lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentBlueprint {
    /// Unique blueprint ID
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Team assignment (PAT or SAT)
    pub team: AgentTeam,

    /// Capability slot this agent fills
    pub capability_slot: CapabilitySlot,

    /// Current system prompt
    pub system_prompt: String,

    /// Model to use (e.g., "deepseek-r1:7b", "qwen2.5:7b")
    pub model: String,

    /// Backend (e.g., "ollama", "lmstudio")
    pub backend: String,

    /// VRAM requirement in GB
    pub vram_gb: f64,

    /// Improvement genome for evolution
    pub improvement_genome: ImprovementGenome,

    /// Current generation number
    pub generation: u64,

    /// Parent blueprint ID (None for genesis blueprints)
    pub parent_id: Option<String>,

    /// Hash of the evolution lineage
    pub lineage_hash: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last evolution timestamp
    pub evolved_at: DateTime<Utc>,

    /// Performance history (recent generations)
    pub performance_history: Vec<BlueprintPerformance>,

    /// Tags for categorization
    pub tags: Vec<String>,

    /// Active status
    pub is_active: bool,
}

impl AgentBlueprint {
    /// Create a new genesis blueprint (generation 0)
    pub fn genesis(
        id: &str,
        name: &str,
        team: AgentTeam,
        capability_slot: CapabilitySlot,
        system_prompt: &str,
        model: &str,
        backend: &str,
        vram_gb: f64,
    ) -> Self {
        let now = Utc::now();
        let lineage_hash = Self::compute_lineage_hash(None, id, 0);

        Self {
            id: id.to_string(),
            name: name.to_string(),
            team,
            capability_slot,
            system_prompt: system_prompt.to_string(),
            model: model.to_string(),
            backend: backend.to_string(),
            vram_gb,
            improvement_genome: ImprovementGenome::default(),
            generation: 0,
            parent_id: None,
            lineage_hash,
            created_at: now,
            evolved_at: now,
            performance_history: Vec::new(),
            tags: Vec::new(),
            is_active: true,
        }
    }

    /// Evolve this blueprint into a new generation
    pub fn evolve(&self, mutations: Vec<PromptMutation>, generation: u64) -> Self {
        let new_id = format!(
            "{}-gen{}",
            self.id.split("-gen").next().unwrap_or(&self.id),
            generation
        );
        let new_prompt = self.apply_mutations(&mutations);
        let lineage_hash =
            Self::compute_lineage_hash(Some(&self.lineage_hash), &new_id, generation);

        let mut new_genome = self.improvement_genome.clone();
        new_genome.prompt_mutations.extend(mutations.clone());

        Self {
            id: new_id,
            name: self.name.clone(),
            team: self.team,
            capability_slot: self.capability_slot.clone(),
            system_prompt: new_prompt,
            model: self.model.clone(),
            backend: self.backend.clone(),
            vram_gb: self.vram_gb,
            improvement_genome: new_genome,
            generation,
            parent_id: Some(self.id.clone()),
            lineage_hash,
            created_at: self.created_at,
            evolved_at: Utc::now(),
            performance_history: Vec::new(),
            tags: self.tags.clone(),
            is_active: true,
        }
    }

    /// Apply mutations to the system prompt
    fn apply_mutations(&self, mutations: &[PromptMutation]) -> String {
        let mut prompt = self.system_prompt.clone();

        for mutation in mutations {
            prompt = mutation.apply(&prompt);
        }

        prompt
    }

    /// Compute lineage hash from parent and current info
    fn compute_lineage_hash(parent_hash: Option<&str>, id: &str, generation: u64) -> String {
        let content = format!("{}|{}|{}", parent_hash.unwrap_or("genesis"), id, generation);
        let hash = Sha256::digest(content.as_bytes());
        format!("lin:{:x}", hash)
    }

    /// Record performance for this generation
    pub fn record_performance(&mut self, performance: BlueprintPerformance) {
        self.performance_history.push(performance);
        // Keep only last 10 generations
        if self.performance_history.len() > 10 {
            self.performance_history.remove(0);
        }
    }

    /// Get average Ihsān score from recent history
    pub fn average_ihsan(&self) -> Option<f64> {
        if self.performance_history.is_empty() {
            return None;
        }
        let sum: f64 = self.performance_history.iter().map(|p| p.ihsan_score).sum();
        Some(sum / self.performance_history.len() as f64)
    }
}

/// Performance metrics for a blueprint in a generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlueprintPerformance {
    /// Generation number
    pub generation: u64,

    /// Ihsān score achieved
    pub ihsan_score: f64,

    /// Tasks completed
    pub tasks_completed: u64,

    /// Average latency (ms)
    pub avg_latency_ms: u64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,

    /// Contribution score
    pub contribution_score: f64,

    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Genome for controlling agent improvement/evolution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImprovementGenome {
    /// History of prompt mutations applied
    pub prompt_mutations: Vec<PromptMutation>,

    /// Capability extensions (MCP tools, A2A capabilities)
    pub capability_extensions: Vec<String>,

    /// Routing preferences for task distribution
    pub routing_preferences: RoutingPreferences,

    /// Fitness criteria for selection
    pub fitness_criteria: Vec<FitnessCriterion>,

    /// Mutation rate (0.0 to 1.0)
    pub mutation_rate: f64,

    /// Cross-over enabled with other blueprints
    pub crossover_enabled: bool,

    /// Elitism: top N blueprints always survive
    pub elitism_count: usize,
}

impl ImprovementGenome {
    /// Generate candidate mutations based on performance
    pub fn generate_mutations(&self, performance: &BlueprintPerformance) -> Vec<PromptMutation> {
        let mut mutations = Vec::new();

        // If Ihsān is low, add safety-focused mutations
        if performance.ihsan_score < 0.95 {
            mutations.push(PromptMutation::Append {
                text: "\n\nCRITICAL: Ensure all outputs meet 0.95 Ihsān threshold.".to_string(),
            });
        }

        // If latency is high, add efficiency mutations
        if performance.avg_latency_ms > 200 {
            mutations.push(PromptMutation::Append {
                text: "\n\nOptimize for response speed while maintaining quality.".to_string(),
            });
        }

        // If success rate is low, add reliability mutations
        if performance.success_rate < 0.9 {
            mutations.push(PromptMutation::Append {
                text: "\n\nPrioritize task completion reliability.".to_string(),
            });
        }

        mutations
    }

    /// Evaluate fitness of a blueprint
    pub fn evaluate_fitness(&self, performance: &BlueprintPerformance) -> f64 {
        if self.fitness_criteria.is_empty() {
            // Default fitness: weighted combination
            return performance.ihsan_score * 0.4
                + performance.success_rate * 0.3
                + performance.contribution_score * 0.2
                + (1.0 - (performance.avg_latency_ms as f64 / 1000.0).min(1.0)) * 0.1;
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for criterion in &self.fitness_criteria {
            let score = criterion.evaluate(performance);
            weighted_sum += score * criterion.weight;
            total_weight += criterion.weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
}

/// Types of mutations that can be applied to prompts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptMutation {
    /// Append text to the prompt
    Append { text: String },

    /// Prepend text to the prompt
    Prepend { text: String },

    /// Replace a pattern with new text
    Replace {
        pattern: String,
        replacement: String,
    },

    /// Insert text at a specific position
    Insert { position: usize, text: String },

    /// Remove text matching a pattern
    Remove { pattern: String },

    /// Emphasize a section (add markers)
    Emphasize { pattern: String },

    /// Reorder sections (by headers)
    ReorderSections { order: Vec<String> },

    /// Custom mutation with description
    Custom {
        description: String,
        transform: String,
    },
}

impl PromptMutation {
    /// Apply this mutation to a prompt
    pub fn apply(&self, prompt: &str) -> String {
        match self {
            PromptMutation::Append { text } => format!("{}{}", prompt, text),
            PromptMutation::Prepend { text } => format!("{}{}", text, prompt),
            PromptMutation::Replace {
                pattern,
                replacement,
            } => prompt.replace(pattern, replacement),
            PromptMutation::Insert { position, text } => {
                let pos = (*position).min(prompt.len());
                format!("{}{}{}", &prompt[..pos], text, &prompt[pos..])
            }
            PromptMutation::Remove { pattern } => prompt.replace(pattern, ""),
            PromptMutation::Emphasize { pattern } => {
                prompt.replace(pattern, &format!("**IMPORTANT: {}**", pattern))
            }
            PromptMutation::ReorderSections { order: _ } => {
                // Complex reordering - simplified for now
                prompt.to_string()
            }
            PromptMutation::Custom {
                description: _,
                transform: _,
            } => {
                // Custom transforms require external processing
                prompt.to_string()
            }
        }
    }

    /// Get a description of this mutation
    pub fn description(&self) -> String {
        match self {
            PromptMutation::Append { text } => {
                format!("Append: {:?}...", &text[..text.len().min(50)])
            }
            PromptMutation::Prepend { text } => {
                format!("Prepend: {:?}...", &text[..text.len().min(50)])
            }
            PromptMutation::Replace {
                pattern,
                replacement,
            } => {
                format!("Replace '{}' with '{}'", pattern, replacement)
            }
            PromptMutation::Insert { position, text } => {
                format!(
                    "Insert at {}: {:?}...",
                    position,
                    &text[..text.len().min(30)]
                )
            }
            PromptMutation::Remove { pattern } => format!("Remove: {}", pattern),
            PromptMutation::Emphasize { pattern } => format!("Emphasize: {}", pattern),
            PromptMutation::ReorderSections { order } => format!("Reorder: {:?}", order),
            PromptMutation::Custom { description, .. } => description.clone(),
        }
    }
}

/// Preferences for routing tasks to agents
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RoutingPreferences {
    /// Task types this agent prefers
    pub preferred_task_types: Vec<String>,

    /// Task types this agent should avoid
    pub avoided_task_types: Vec<String>,

    /// Priority boost (0.0 to 1.0)
    pub priority_boost: f64,

    /// Maximum concurrent tasks
    pub max_concurrent: usize,

    /// Collaboration preferences (agent IDs)
    pub preferred_collaborators: Vec<String>,
}

/// Criteria for evaluating blueprint fitness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessCriterion {
    /// Name of the criterion
    pub name: String,

    /// Weight in overall fitness (0.0 to 1.0)
    pub weight: f64,

    /// Metric to evaluate
    pub metric: FitnessMetric,

    /// Target value (for comparison)
    pub target: f64,

    /// Whether higher is better
    pub higher_is_better: bool,
}

impl FitnessCriterion {
    /// Evaluate this criterion against performance
    pub fn evaluate(&self, performance: &BlueprintPerformance) -> f64 {
        let value = match &self.metric {
            FitnessMetric::IhsanScore => performance.ihsan_score,
            FitnessMetric::SuccessRate => performance.success_rate,
            FitnessMetric::Latency => 1.0 - (performance.avg_latency_ms as f64 / 1000.0).min(1.0),
            FitnessMetric::ContributionScore => performance.contribution_score,
            FitnessMetric::TasksCompleted => (performance.tasks_completed as f64 / 100.0).min(1.0),
            FitnessMetric::Custom(name) => {
                // Custom metrics need external evaluation
                tracing::debug!("Custom metric '{}' not implemented, returning 0.5", name);
                0.5
            }
        };

        if self.higher_is_better {
            (value / self.target).min(1.0)
        } else {
            (self.target / value.max(0.001)).min(1.0)
        }
    }
}

/// Metrics that can be used for fitness evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitnessMetric {
    IhsanScore,
    SuccessRate,
    Latency,
    ContributionScore,
    TasksCompleted,
    Custom(String),
}

/// Blueprint manager for handling collections
pub struct BlueprintManager {
    /// All blueprints indexed by ID
    blueprints: HashMap<String, AgentBlueprint>,

    /// Active blueprint IDs by capability slot
    active_by_slot: HashMap<CapabilitySlot, String>,
}

impl BlueprintManager {
    pub fn new() -> Self {
        Self {
            blueprints: HashMap::new(),
            active_by_slot: HashMap::new(),
        }
    }

    /// Register a new blueprint
    pub fn register(&mut self, blueprint: AgentBlueprint) {
        let slot = blueprint.capability_slot.clone();
        let id = blueprint.id.clone();

        // Update active if this is newer generation
        if let Some(current_id) = self.active_by_slot.get(&slot) {
            if let Some(current) = self.blueprints.get(current_id) {
                if blueprint.generation > current.generation {
                    self.active_by_slot.insert(slot.clone(), id.clone());
                }
            }
        } else {
            self.active_by_slot.insert(slot, id.clone());
        }

        self.blueprints.insert(id, blueprint);
    }

    /// Get the active blueprint for a slot
    pub fn get_active(&self, slot: &CapabilitySlot) -> Option<&AgentBlueprint> {
        self.active_by_slot
            .get(slot)
            .and_then(|id| self.blueprints.get(id))
    }

    /// Get all active blueprints
    pub fn get_all_active(&self) -> Vec<&AgentBlueprint> {
        self.active_by_slot
            .values()
            .filter_map(|id| self.blueprints.get(id))
            .collect()
    }

    /// Get blueprint by ID
    pub fn get(&self, id: &str) -> Option<&AgentBlueprint> {
        self.blueprints.get(id)
    }

    /// Get mutable blueprint by ID
    pub fn get_mut(&mut self, id: &str) -> Option<&mut AgentBlueprint> {
        self.blueprints.get_mut(id)
    }

    /// Get all blueprints for a team
    pub fn get_by_team(&self, team: AgentTeam) -> Vec<&AgentBlueprint> {
        self.blueprints
            .values()
            .filter(|b| b.team == team)
            .collect()
    }

    /// Get lineage (ancestry) of a blueprint
    pub fn get_lineage(&self, id: &str) -> Vec<&AgentBlueprint> {
        let mut lineage = Vec::new();
        let mut current_id = Some(id.to_string());

        while let Some(ref cid) = current_id {
            if let Some(blueprint) = self.blueprints.get(cid) {
                lineage.push(blueprint);
                current_id = blueprint.parent_id.clone();
            } else {
                break;
            }
        }

        lineage.reverse();
        lineage
    }
}

impl Default for BlueprintManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_blueprint() {
        let blueprint = AgentBlueprint::genesis(
            "master-reasoner",
            "MasterReasoner",
            AgentTeam::PAT,
            CapabilitySlot::MasterReasoner,
            "You are a strategic thinking agent.",
            "deepseek-r1:7b",
            "ollama",
            4.5,
        );

        assert_eq!(blueprint.generation, 0);
        assert!(blueprint.parent_id.is_none());
        assert!(blueprint.lineage_hash.starts_with("lin:"));
    }

    #[test]
    fn test_blueprint_evolution() {
        let genesis = AgentBlueprint::genesis(
            "test-agent",
            "TestAgent",
            AgentTeam::PAT,
            CapabilitySlot::MasterReasoner,
            "Original prompt",
            "test-model",
            "test",
            1.0,
        );

        let mutations = vec![PromptMutation::Append {
            text: " - Enhanced".to_string(),
        }];

        let evolved = genesis.evolve(mutations, 1);

        assert_eq!(evolved.generation, 1);
        assert_eq!(evolved.parent_id, Some("test-agent".to_string()));
        assert!(evolved.system_prompt.contains("Enhanced"));
        assert_ne!(evolved.lineage_hash, genesis.lineage_hash);
    }

    #[test]
    fn test_prompt_mutations() {
        let prompt = "Hello world";

        let append = PromptMutation::Append {
            text: "!".to_string(),
        };
        assert_eq!(append.apply(prompt), "Hello world!");

        let replace = PromptMutation::Replace {
            pattern: "world".to_string(),
            replacement: "BIZRA".to_string(),
        };
        assert_eq!(replace.apply(prompt), "Hello BIZRA");

        let emphasize = PromptMutation::Emphasize {
            pattern: "world".to_string(),
        };
        assert!(emphasize.apply(prompt).contains("**IMPORTANT: world**"));
    }

    #[test]
    fn test_blueprint_manager() {
        let mut manager = BlueprintManager::new();

        let bp1 = AgentBlueprint::genesis(
            "agent-1",
            "Agent1",
            AgentTeam::PAT,
            CapabilitySlot::MasterReasoner,
            "Prompt 1",
            "model",
            "backend",
            1.0,
        );

        manager.register(bp1.clone());

        assert!(manager
            .get_active(&CapabilitySlot::MasterReasoner)
            .is_some());
        assert_eq!(manager.get_all_active().len(), 1);

        // Evolve and register
        let bp2 = bp1.evolve(vec![], 1);
        manager.register(bp2);

        // Should have updated to newer generation
        assert_eq!(
            manager
                .get_active(&CapabilitySlot::MasterReasoner)
                .unwrap()
                .generation,
            1
        );
    }
}
