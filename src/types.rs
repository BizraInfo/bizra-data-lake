// src/types.rs - Core types and data structures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Enhanced agent with full arsenal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedAgentCapabilities {
    /// MCP: Tool access
    pub mcp_tools: Vec<String>,
    
    /// A2A: Agent communication
    pub a2a_capabilities: Vec<String>,
    
    /// Reasoning methods
    pub reasoning_methods: Vec<ReasoningMethod>,
    
    /// Sub-agent generation
    pub can_spawn_sub_agents: bool,
    pub max_sub_agents: usize,
    
    /// Swarm capabilities
    pub swarm_modes: Vec<SwarmMode>,
    
    /// Memory access
    pub memory_tiers: Vec<MemoryTier>,
    
    /// Hook support
    pub hooks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReasoningMethod {
    ChainOfThought,
    TreeOfThought,
    GraphOfThought,
    ReAct,
    Reflexion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmMode {
    Independent,
    Collaborative,
    HiveMind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTier {
    Working,
    Episodic,
    Semantic,
    Procedural,
}

/// Base dual agentic request
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DualAgenticRequest {
    pub user_id: String,
    pub task: String,
    pub requirements: Vec<String>,
    pub target: String,
    #[serde(default)]
    pub priority: Priority,
    #[serde(default)]
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Priority {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

/// Enhanced request with full control
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnhancedDualAgenticRequest {
    /// Base request
    pub base: DualAgenticRequest,
    
    /// Advanced controls
    pub reasoning_preference: Option<ReasoningMethod>,
    #[serde(default)]
    pub enable_sub_agents: bool,
    pub enable_swarm: Option<SwarmMode>,
    pub mcp_tools_whitelist: Option<Vec<String>>,
    pub memory_context: Option<serde_json::Value>,
    #[serde(default)]
    pub hooks_config: HashMap<String, serde_json::Value>,
    
    /// Slash command support
    pub slash_command: Option<SlashCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SlashCommand {
    Reason { method: ReasoningMethod },
    Spawn { role: String, task: String },
    Swarm { count: usize, mode: SwarmMode },
    Memory { tier: MemoryTier, query: String },
    Hook { name: String, action: HookAction },
    Tools { filter: String },
    Delegate { agent: String, task: String },
    Synthesize,
    Reflect { depth: usize },
    Export { format: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookAction {
    Enable,
    Disable,
    Configure(serde_json::Value),
}

/// Dual agentic response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualAgenticResponse {
    pub pat_contributions: Vec<String>,
    pub sat_contributions: Vec<String>,
    pub synergy_score: f64,
    pub ihsan_score: f64,
    #[serde(with = "duration_serde")]
    pub latency: Duration,
    pub meta: serde_json::Value,
}

/// Agent execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub agent_name: String,
    pub contribution: String,
    pub confidence: f64,
    #[serde(with = "duration_serde")]
    pub execution_time: Duration,
}

// Custom Duration serialization
mod duration_serde {
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
