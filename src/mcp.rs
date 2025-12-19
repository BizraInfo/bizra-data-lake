// src/mcp.rs - Model Context Protocol integration
// 
// SECURITY: Tool execution is gated by allowlists and timeouts

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, warn, instrument};

/// Tool execution timeout (30 seconds default)
const DEFAULT_TOOL_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum output size from tool execution (1MB)
const MAX_OUTPUT_SIZE: usize = 1024 * 1024;

/// Tools that are NEVER allowed (security blocklist)
const TOOL_BLOCKLIST: &[&str] = &[
    "shell_exec",
    "system_command",
    "raw_eval",
    "file_delete",
    "network_raw",
];

/// Default allowed tools (can be extended per-agent)
const DEFAULT_ALLOWLIST: &[&str] = &[
    "filesystem_read",
    "web_search",
    "code_analysis",
    "database_query",
];

/// Result of a tool execution with security metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub result: serde_json::Value,
    pub execution_time_ms: u64,
    pub truncated: bool,
}

/// Tool execution error types
#[derive(Debug, Clone)]
pub enum ToolError {
    NotFound(String),
    Blocked(String),
    NotAllowed(String),
    Timeout(String),
    OutputTooLarge(String),
    ExecutionFailed(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(t) => write!(f, "Tool not found: {}", t),
            Self::Blocked(t) => write!(f, "Tool blocked by security policy: {}", t),
            Self::NotAllowed(t) => write!(f, "Tool not in allowlist: {}", t),
            Self::Timeout(t) => write!(f, "Tool execution timed out: {}", t),
            Self::OutputTooLarge(t) => write!(f, "Tool output exceeded max size: {}", t),
            Self::ExecutionFailed(msg) => write!(f, "Tool execution failed: {}", msg),
        }
    }
}

impl std::error::Error for ToolError {}

/// MCP Client for tool discovery and execution
pub struct MCPClient {
    servers: HashMap<String, MCPServer>,
    tool_registry: HashMap<String, ToolDefinition>,
    /// Tools allowed for this client instance
    allowlist: HashSet<String>,
    /// Custom timeout (overrides default)
    timeout: Duration,
}

#[derive(Debug, Clone)]
struct MCPServer {
    url: String,
    transport: MCPTransport,
}

#[derive(Debug, Clone)]
pub enum MCPTransport {
    Stdio,
    HttpSse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub server: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub description: String,
    pub required: bool,
}

impl MCPClient {
    pub fn new() -> Self {
        let allowlist: HashSet<String> = DEFAULT_ALLOWLIST
            .iter()
            .map(|s| s.to_string())
            .collect();
        
        Self {
            servers: HashMap::new(),
            tool_registry: HashMap::new(),
            allowlist,
            timeout: DEFAULT_TOOL_TIMEOUT,
        }
    }
    
    /// Create client with custom allowlist
    pub fn with_allowlist(tools: Vec<String>) -> Self {
        let allowlist: HashSet<String> = tools.into_iter().collect();
        Self {
            servers: HashMap::new(),
            tool_registry: HashMap::new(),
            allowlist,
            timeout: DEFAULT_TOOL_TIMEOUT,
        }
    }
    
    /// Set custom timeout for tool execution
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }
    
    /// Add tool to allowlist
    pub fn allow_tool(&mut self, tool_name: String) {
        if !TOOL_BLOCKLIST.contains(&tool_name.as_str()) {
            self.allowlist.insert(tool_name);
        }
    }
    
    /// Check if tool is allowed
    pub fn is_tool_allowed(&self, tool_name: &str) -> Result<(), ToolError> {
        // Check blocklist first (security-critical)
        if TOOL_BLOCKLIST.contains(&tool_name) {
            return Err(ToolError::Blocked(tool_name.to_string()));
        }
        
        // Check allowlist
        if !self.allowlist.contains(tool_name) {
            return Err(ToolError::NotAllowed(tool_name.to_string()));
        }
        
        Ok(())
    }

    /// Register MCP server
    pub async fn register_server(
        &mut self,
        name: String,
        url: String,
        transport: MCPTransport,
    ) -> anyhow::Result<()> {
        self.servers.insert(name, MCPServer { url, transport });
        self.discover_tools().await?;
        Ok(())
    }

    /// Discover all available tools from registered servers
    #[instrument(skip(self))]
    async fn discover_tools(&mut self) -> anyhow::Result<()> {
        for (server_name, server) in &self.servers {
            debug!(
                server_name,
                server_url = %server.url,
                transport = ?server.transport,
                "Discovering MCP tools (simulated)"
            );
            // Simulated tool discovery (in production: actual MCP protocol)
            let tools = vec![
                ToolDefinition {
                    name: "filesystem_read".to_string(),
                    description: "Read file from filesystem".to_string(),
                    parameters: vec![ToolParameter {
                        name: "path".to_string(),
                        type_: "string".to_string(),
                        description: "File path to read".to_string(),
                        required: true,
                    }],
                    server: server_name.clone(),
                },
                ToolDefinition {
                    name: "web_search".to_string(),
                    description: "Search the web".to_string(),
                    parameters: vec![ToolParameter {
                        name: "query".to_string(),
                        type_: "string".to_string(),
                        description: "Search query".to_string(),
                        required: true,
                    }],
                    server: server_name.clone(),
                },
                ToolDefinition {
                    name: "database_query".to_string(),
                    description: "Query database".to_string(),
                    parameters: vec![ToolParameter {
                        name: "sql".to_string(),
                        type_: "string".to_string(),
                        description: "SQL query".to_string(),
                        required: true,
                    }],
                    server: server_name.clone(),
                },
                ToolDefinition {
                    name: "code_analysis".to_string(),
                    description: "Analyze source code".to_string(),
                    parameters: vec![ToolParameter {
                        name: "code".to_string(),
                        type_: "string".to_string(),
                        description: "Code to analyze".to_string(),
                        required: true,
                    }],
                    server: server_name.clone(),
                },
            ];

            for tool in tools {
                self.tool_registry.insert(tool.name.clone(), tool);
            }
        }

        debug!(
            tools_count = self.tool_registry.len(),
            "MCP tools discovered"
        );
        Ok(())
    }

    /// Execute tool via MCP with security controls
    #[instrument(skip(self))]
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: HashMap<String, serde_json::Value>,
    ) -> Result<ToolResult, ToolError> {
        let start = std::time::Instant::now();
        
        // SECURITY CHECK 1: Allowlist/Blocklist
        self.is_tool_allowed(tool_name)?;
        
        // SECURITY CHECK 2: Tool must be registered
        let _tool = self
            .tool_registry
            .get(tool_name)
            .ok_or_else(|| ToolError::NotFound(tool_name.to_string()))?;
        
        // SECURITY CHECK 3: Execute with timeout
        let execution_future = self.execute_tool_internal(tool_name, &arguments);
        
        let result = match timeout(self.timeout, execution_future).await {
            Ok(Ok(value)) => {
                // SECURITY CHECK 4: Output size limit
                let output_str = serde_json::to_string(&value).unwrap_or_default();
                let truncated = output_str.len() > MAX_OUTPUT_SIZE;
                
                if truncated {
                    warn!(
                        tool_name,
                        output_size = output_str.len(),
                        max_size = MAX_OUTPUT_SIZE,
                        "Tool output truncated due to size limit"
                    );
                }
                
                let final_value = if truncated {
                    serde_json::json!({
                        "truncated": true,
                        "message": "Output exceeded maximum size limit",
                        "partial_size": MAX_OUTPUT_SIZE,
                    })
                } else {
                    value
                };
                
                Ok(ToolResult {
                    tool_name: tool_name.to_string(),
                    success: true,
                    result: final_value,
                    execution_time_ms: start.elapsed().as_millis() as u64,
                    truncated,
                })
            }
            Ok(Err(e)) => {
                Err(ToolError::ExecutionFailed(e.to_string()))
            }
            Err(_) => {
                warn!(
                    tool_name,
                    timeout_secs = self.timeout.as_secs(),
                    "Tool execution timed out"
                );
                Err(ToolError::Timeout(tool_name.to_string()))
            }
        };
        
        result
    }
    
    /// Internal tool execution (in production: actual MCP protocol)
    async fn execute_tool_internal(
        &self,
        tool_name: &str,
        arguments: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<serde_json::Value> {
        // In production: actual MCP protocol call to server
        // For now: simulated execution
        let result = serde_json::json!({
            "tool": tool_name,
            "arguments": arguments,
            "result": format!("Executed {} successfully", tool_name),
            "status": "success",
        });

        Ok(result)
    }

    /// List available tools
    pub fn list_tools(&self) -> Vec<&ToolDefinition> {
        self.tool_registry.values().collect()
    }

    /// Filter tools by capability
    pub fn filter_tools(&self, filter: &str) -> Vec<&ToolDefinition> {
        self.tool_registry
            .values()
            .filter(|t| t.description.contains(filter) || t.name.contains(filter))
            .collect()
    }
}

impl Default for MCPClient {
    fn default() -> Self {
        Self::new()
    }
}
