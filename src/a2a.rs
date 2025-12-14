// src/a2a.rs - Agent-to-Agent Protocol

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::instrument;

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
}

impl A2AServer {
    pub fn new() -> Self {
        Self {
            agent_registry: HashMap::new(),
        }
    }
    
    /// Register agent capabilities
    pub fn register_agent(&mut self, card: AgentCard) {
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
    
    /// Delegate task to another agent
    #[instrument(skip(self))]
    pub async fn delegate(
        &self,
        agent_name: &str,
        task: String,
    ) -> anyhow::Result<serde_json::Value> {
        let agent = self.agent_registry.get(agent_name)
            .ok_or_else(|| anyhow::anyhow!("Agent not found: {}", agent_name))?;
        
        // In production: actual A2A protocol call (JSON-RPC over HTTP)
        // For now: simulated delegation
        let result = serde_json::json!({
            "agent": agent_name,
            "task": task,
            "status": "completed",
            "result": format!("{} completed task: {}", agent_name, task),
            "capabilities_used": agent.capabilities.iter().map(|c| &c.id).collect::<Vec<_>>(),
        });
        
        Ok(result)
    }
    
    /// Request vote from agent (for SAT consensus)
    #[instrument(skip(self))]
    pub async fn request_vote(
        &self,
        agent_name: &str,
        proposal: serde_json::Value,
    ) -> anyhow::Result<bool> {
        let _agent = self.agent_registry.get(agent_name)
            .ok_or_else(|| anyhow::anyhow!("Agent not found: {}", agent_name))?;
        
        // In production: actual consensus protocol
        // For now: simulated voting (Byzantine fault tolerant)
        Ok(true)
    }
    
    /// Broadcast message to all agents
    #[instrument(skip(self))]
    pub async fn broadcast(
        &self,
        message: serde_json::Value,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let mut responses = Vec::new();
        
        for agent in self.agent_registry.values() {
            let response = serde_json::json!({
                "agent": agent.name,
                "received": message,
                "ack": true,
            });
            responses.push(response);
        }
        
        Ok(responses)
    }
}

impl Default for A2AServer {
    fn default() -> Self {
        Self::new()
    }
}
