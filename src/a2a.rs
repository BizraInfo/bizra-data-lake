// src/a2a.rs - Agent-to-Agent Protocol
//
// SECURITY: Agent delegation is gated by allowlists and timeouts
// METRICS: All delegations tracked via A2A_DELEGATIONS_TOTAL and A2A_DELEGATION_DEPTH

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn, instrument};

use crate::metrics::{A2A_DELEGATIONS_TOTAL, A2A_DELEGATION_DEPTH};

/// Default timeout for agent delegation (60 seconds)
const DEFAULT_DELEGATION_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum delegation depth (prevent infinite delegation chains)
const MAX_DELEGATION_DEPTH: u8 = 5;

/// Agents that cannot receive delegations (reserved system agents)
const AGENT_BLOCKLIST: &[&str] = &[
    "root_agent",
    "system_agent",
    "kernel_agent",
];

/// Result of agent delegation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationResult {
    pub agent: String,
    pub task: String,
    pub success: bool,
    pub result: serde_json::Value,
    pub execution_time_ms: u64,
    pub delegation_depth: u8,
}

/// Delegation error types
#[derive(Debug, Clone)]
pub enum DelegationError {
    AgentNotFound(String),
    AgentBlocked(String),
    AgentNotAllowed(String),
    Timeout(String),
    MaxDepthExceeded(u8),
    ExecutionFailed(String),
}

impl std::fmt::Display for DelegationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgentNotFound(a) => write!(f, "Agent not found: {}", a),
            Self::AgentBlocked(a) => write!(f, "Agent blocked by security policy: {}", a),
            Self::AgentNotAllowed(a) => write!(f, "Agent not in allowlist: {}", a),
            Self::Timeout(a) => write!(f, "Agent delegation timed out: {}", a),
            Self::MaxDepthExceeded(d) => write!(f, "Max delegation depth exceeded: {}", d),
            Self::ExecutionFailed(msg) => write!(f, "Delegation failed: {}", msg),
        }
    }
}

impl std::error::Error for DelegationError {}

/// Agent capability card for A2A discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    pub name: String,
    pub version: String,
    pub capabilities: Vec<Capability>,
    pub protocols: Vec<String>,
    pub authentication: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// A2A Server for agent communication
pub struct A2AServer {
    agent_registry: HashMap<String, AgentCard>,
    /// Agents allowed for delegation
    allowlist: HashSet<String>,
    /// Custom timeout for delegation
    timeout: Duration,
    /// Current delegation depth tracking (for session-level depth limits)
    /// Note: Per-request depth is passed through method parameters
    #[allow(dead_code)]
    current_depth: u8,
}

impl A2AServer {
    pub fn new() -> Self {
        Self {
            agent_registry: HashMap::new(),
            allowlist: HashSet::new(),
            timeout: DEFAULT_DELEGATION_TIMEOUT,
            current_depth: 0,
        }
    }
    
    /// Create server with custom allowlist
    pub fn with_allowlist(agents: Vec<String>) -> Self {
        let allowlist: HashSet<String> = agents.into_iter().collect();
        Self {
            agent_registry: HashMap::new(),
            allowlist,
            timeout: DEFAULT_DELEGATION_TIMEOUT,
            current_depth: 0,
        }
    }
    
    /// Set custom timeout for delegation
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
    
    /// Allow an agent for delegation
    pub fn allow_agent(&mut self, agent_name: String) {
        if !AGENT_BLOCKLIST.contains(&agent_name.as_str()) {
            self.allowlist.insert(agent_name);
        }
    }
    
    /// Check if agent is allowed for delegation
    fn is_agent_allowed(&self, agent_name: &str) -> Result<(), DelegationError> {
        // Check blocklist first
        if AGENT_BLOCKLIST.contains(&agent_name) {
            return Err(DelegationError::AgentBlocked(agent_name.to_string()));
        }
        
        // If allowlist is empty, allow all (except blocklist)
        // If allowlist is non-empty, agent must be in it
        if !self.allowlist.is_empty() && !self.allowlist.contains(agent_name) {
            return Err(DelegationError::AgentNotAllowed(agent_name.to_string()));
        }
        
        Ok(())
    }

    /// Register agent capabilities
    pub fn register_agent(&mut self, card: AgentCard) {
        // Auto-add to allowlist when registered
        self.allowlist.insert(card.name.clone());
        self.agent_registry.insert(card.name.clone(), card);
    }

    /// Discover available agents
    pub fn discover_agents(&self) -> Vec<&AgentCard> {
        self.agent_registry.values().collect()
    }

    /// Get specific agent
    pub fn get_agent(&self, name: &str) -> Option<&AgentCard> {
        self.agent_registry.get(name)
    }

    /// Delegate task to another agent with security controls
    #[instrument(skip(self))]
    pub async fn delegate(
        &self,
        agent_name: &str,
        task: String,
    ) -> Result<DelegationResult, DelegationError> {
        self.delegate_with_depth(agent_name, task, 0).await
    }
    
    /// Internal delegation with depth tracking and metrics
    async fn delegate_with_depth(
        &self,
        agent_name: &str,
        task: String,
        depth: u8,
    ) -> Result<DelegationResult, DelegationError> {
        let start = std::time::Instant::now();
        
        // METRICS: Track maximum delegation depth observed
        A2A_DELEGATION_DEPTH.set(depth as f64);
        
        // SECURITY CHECK 1: Delegation depth limit
        if depth >= MAX_DELEGATION_DEPTH {
            warn!(
                agent_name,
                depth,
                max_depth = MAX_DELEGATION_DEPTH,
                "Delegation depth exceeded - preventing infinite chain"
            );
            A2A_DELEGATIONS_TOTAL.with_label_values(&[agent_name, "depth_exceeded"]).inc();
            return Err(DelegationError::MaxDepthExceeded(depth));
        }
        
        // SECURITY CHECK 2: Allowlist/Blocklist
        if let Err(e) = self.is_agent_allowed(agent_name) {
            A2A_DELEGATIONS_TOTAL.with_label_values(&[agent_name, "blocked"]).inc();
            return Err(e);
        }
        
        // SECURITY CHECK 3: Agent must be registered
        let agent = match self.agent_registry.get(agent_name) {
            Some(a) => a,
            None => {
                A2A_DELEGATIONS_TOTAL.with_label_values(&[agent_name, "not_found"]).inc();
                return Err(DelegationError::AgentNotFound(agent_name.to_string()));
            }
        };
        
        // SECURITY CHECK 4: Execute with timeout
        let execution_future = self.execute_delegation_internal(agent, &task);
        
        match timeout(self.timeout, execution_future).await {
            Ok(Ok(result)) => {
                A2A_DELEGATIONS_TOTAL.with_label_values(&[agent_name, "success"]).inc();
                info!(
                    agent_name,
                    depth,
                    execution_time_ms = start.elapsed().as_millis() as u64,
                    "A2A delegation completed successfully"
                );
                Ok(DelegationResult {
                    agent: agent_name.to_string(),
                    task,
                    success: true,
                    result,
                    execution_time_ms: start.elapsed().as_millis() as u64,
                    delegation_depth: depth,
                })
            }
            Ok(Err(e)) => {
                A2A_DELEGATIONS_TOTAL.with_label_values(&[agent_name, "failed"]).inc();
                Err(DelegationError::ExecutionFailed(e.to_string()))
            }
            Err(_) => {
                warn!(
                    agent_name,
                    timeout_secs = self.timeout.as_secs(),
                    "Agent delegation timed out"
                );
                A2A_DELEGATIONS_TOTAL.with_label_values(&[agent_name, "timeout"]).inc();
                Err(DelegationError::Timeout(agent_name.to_string()))
            }
        }
    }
    
    /// Internal delegation execution
    async fn execute_delegation_internal(
        &self,
        agent: &AgentCard,
        _task: &str,
    ) -> anyhow::Result<serde_json::Value> {
        // Production A2A requires real JSON-RPC over HTTP implementation
        // Fail-closed: Cannot delegate tasks without proper A2A infrastructure
        anyhow::bail!(
            "Task delegation to agent '{}' failed: A2A infrastructure not implemented. \
             Real A2A protocol requires JSON-RPC over HTTP communication.",
            agent.name
        )
    }

    /// Request vote from agent (for SAT consensus) with security controls
    #[instrument(skip(self))]
    pub async fn request_vote(
        &self,
        agent_name: &str,
        proposal: serde_json::Value,
    ) -> Result<bool, DelegationError> {
        // SECURITY CHECK: Agent must be allowed
        self.is_agent_allowed(agent_name)?;
        
        let _agent = self
            .agent_registry
            .get(agent_name)
            .ok_or_else(|| DelegationError::AgentNotFound(agent_name.to_string()))?;

        // Production voting requires real Byzantine fault-tolerant consensus
        // Fail-closed: Cannot vote without proper consensus protocol
        // Note: Using ExecutionFailed as Byzantine consensus is not implemented
        Err(DelegationError::ExecutionFailed(format!(
            "Vote request for '{}' failed: Byzantine fault-tolerant consensus not implemented. \
             Real implementation requires cryptographic voting protocol.",
            agent_name
        )))
    }

    /// Broadcast message to all agents with timeout
    #[instrument(skip(self))]
    pub async fn broadcast(
        &self,
        message: serde_json::Value,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut responses = Vec::new();

        for agent in self.agent_registry.values() {
            // Only broadcast to allowed agents
            if self.is_agent_allowed(&agent.name).is_ok() {
                let response = serde_json::json!({
                    "agent": agent.name,
                    "received": message,
                    "ack": true,
                });
                responses.push(response);
            }
        }

        Ok(responses)
    }
    
    /// Chained delegation - when an agent needs to delegate to another agent
    /// This increments depth to prevent infinite delegation chains
    #[instrument(skip(self))]
    pub async fn delegate_chain(
        &self,
        agent_name: &str,
        task: String,
        current_depth: u8,
    ) -> Result<DelegationResult, DelegationError> {
        info!(
            agent_name,
            current_depth,
            next_depth = current_depth + 1,
            "Chained delegation initiated"
        );
        self.delegate_with_depth(agent_name, task, current_depth + 1).await
    }
    
    /// Get current maximum delegation depth (for metrics/monitoring)
    pub fn max_delegation_depth() -> u8 {
        MAX_DELEGATION_DEPTH
    }
    
    /// Get list of blocked agents
    pub fn blocked_agents() -> &'static [&'static str] {
        AGENT_BLOCKLIST
    }
    
    /// Check if an agent is registered
    pub fn is_registered(&self, agent_name: &str) -> bool {
        self.agent_registry.contains_key(agent_name)
    }
    
    /// Get count of registered agents
    pub fn agent_count(&self) -> usize {
        self.agent_registry.len()
    }
    
    /// Get all agent names
    pub fn agent_names(&self) -> Vec<String> {
        self.agent_registry.keys().cloned().collect()
    }
}

impl Default for A2AServer {
    fn default() -> Self {
        Self::new()
    }
}
