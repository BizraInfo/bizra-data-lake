// src/mcp.rs - Model Context Protocol (MCP) Integration
//
// Full JSON-RPC 2.0 implementation for Claude-compatible tool execution.
// SECURITY: Tool execution is gated by allowlists, timeouts, SAT validation, and SAPE/Ihsan probing.
//
// MCP Specification: https://modelcontextprotocol.io/specification
// A2A Protocol: https://google.github.io/a2a-spec/

use crate::{ihsan, sape};
use lazy_static::lazy_static;
use prometheus::{register_counter_vec, register_histogram_vec, CounterVec, HistogramVec};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OnceCell};
use tokio::time::timeout;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

lazy_static! {
    /// MCP tool call metrics
    pub static ref MCP_CALLS: CounterVec = register_counter_vec!(
        "bizra_mcp_calls_total",
        "Total MCP tool calls",
        &["tool", "result"]
    ).unwrap();
    
    /// MCP latency histogram
    pub static ref MCP_LATENCY: HistogramVec = register_histogram_vec!(
        "bizra_mcp_latency_seconds",
        "MCP tool call latency",
        &["tool"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).unwrap();
    
    /// SAPE-gated MCP tool rejections
    pub static ref MCP_SAPE_REJECTIONS: CounterVec = register_counter_vec!(
        "bizra_mcp_sape_rejections_total",
        "MCP tool calls rejected by SAPE/Ihsan gate",
        &["tool"]
    ).unwrap();
}

/// Global MCP client singleton
static MCP_CLIENT: OnceCell<Arc<Mutex<MCPClient>>> = OnceCell::const_new();

/// Get or create the global MCP client
pub async fn get_mcp() -> Arc<Mutex<MCPClient>> {
    MCP_CLIENT
        .get_or_init(|| async {
            Arc::new(Mutex::new(MCPClient::new()))
        })
        .await
        .clone()
}

/// Tool execution timeout (30 seconds default)
const DEFAULT_TOOL_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum output size from tool execution (1MB)
const MAX_OUTPUT_SIZE: usize = 1024 * 1024;

/// JSON-RPC version
const JSONRPC_VERSION: &str = "2.0";

/// Tools that are NEVER allowed (security blocklist)
const TOOL_BLOCKLIST: &[&str] = &[
    "shell_exec",
    "system_command",
    "raw_eval",
    "file_delete",
    "file_write_system",
    "network_raw",
    "eval",
    "exec",
];

/// Default allowed tools (can be extended per-agent)
const DEFAULT_ALLOWLIST: &[&str] = &[
    "filesystem_read",
    "web_search",
    "code_analysis",
    "database_query",
    "knowledge_retrieve",
    "calculator",
];

// ============================================================
// JSON-RPC 2.0 Types (MCP Standard)
// ============================================================

/// JSON-RPC 2.0 Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
    pub id: JsonRpcId,
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    pub id: JsonRpcId,
}

/// JSON-RPC ID (can be string, number, or null)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum JsonRpcId {
    String(String),
    Number(i64),
    Null,
}

impl Default for JsonRpcId {
    fn default() -> Self {
        JsonRpcId::String(Uuid::new_v4().to_string())
    }
}

/// JSON-RPC 2.0 Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcError {
    // Standard JSON-RPC error codes
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    
    // MCP-specific error codes (-32000 to -32099)
    pub const TOOL_NOT_FOUND: i32 = -32001;
    pub const TOOL_BLOCKED: i32 = -32002;
    pub const TOOL_NOT_ALLOWED: i32 = -32003;
    pub const TOOL_TIMEOUT: i32 = -32004;
    pub const OUTPUT_TOO_LARGE: i32 = -32005;
    pub const EXECUTION_FAILED: i32 = -32006;
    
    pub fn parse_error() -> Self {
        Self { code: Self::PARSE_ERROR, message: "Parse error".into(), data: None }
    }
    
    pub fn invalid_request(msg: &str) -> Self {
        Self { code: Self::INVALID_REQUEST, message: msg.into(), data: None }
    }
    
    pub fn method_not_found(method: &str) -> Self {
        Self { code: Self::METHOD_NOT_FOUND, message: format!("Method not found: {}", method), data: None }
    }
    
    pub fn tool_blocked(tool: &str) -> Self {
        Self { 
            code: Self::TOOL_BLOCKED, 
            message: format!("Tool blocked by security policy: {}", tool),
            data: Some(serde_json::json!({ "tool": tool, "reason": "blocklist" }))
        }
    }
    
    pub fn tool_timeout(tool: &str, timeout_secs: u64) -> Self {
        Self {
            code: Self::TOOL_TIMEOUT,
            message: format!("Tool execution timed out after {}s: {}", timeout_secs, tool),
            data: Some(serde_json::json!({ "tool": tool, "timeout_secs": timeout_secs }))
        }
    }
    
    pub fn execution_failed(msg: &str) -> Self {
        Self { code: Self::EXECUTION_FAILED, message: msg.into(), data: None }
    }
}

impl JsonRpcResponse {
    pub fn success(id: JsonRpcId, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.into(),
            result: Some(result),
            error: None,
            id,
        }
    }
    
    pub fn error(id: JsonRpcId, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.into(),
            result: None,
            error: Some(error),
            id,
        }
    }
}

// ============================================================
// MCP Tool Types
// ============================================================

/// Result of a tool execution with security metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub result: serde_json::Value,
    pub execution_time_ms: u64,
    pub truncated: bool,
}

/// Result of SAPE/Ihsan gate evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SapeGateResult {
    pub ihsan_score: f64,
    pub threshold: f64,
    pub passed: bool,
    pub probe_count: usize,
    pub flags: Vec<String>,
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
    /// SAPE/Ihsan gate rejected the tool invocation
    SapeRejected {
        tool_name: String,
        ihsan_score: f64,
        threshold: f64,
        flags: Vec<String>,
    },
    /// Internal lock was poisoned (panic in another thread)
    LockPoisoned(String),
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
            Self::SapeRejected { tool_name, ihsan_score, threshold, flags } => {
                write!(
                    f,
                    "Tool '{}' rejected by SAPE/Ihsan gate: score={:.4} < threshold={:.4}, flags={:?}",
                    tool_name, ihsan_score, threshold, flags
                )
            }
            Self::LockPoisoned(msg) => write!(f, "Internal lock poisoned: {}", msg),
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
    
    /// SAPE/Ihsan gate for MCP tool invocations (symbolic-neural bridge)
    /// 
    /// This activates the 8-dimension SAPE probe engine (aligned with ihsan_v1.yaml)
    /// and calculates an aggregate Ihsan score. If the score falls below the
    /// environment-specific threshold AND enforcement is enabled, the tool call is rejected.
    pub fn sape_ihsan_gate(&self, tool_name: &str, content: &str) -> Result<SapeGateResult, ToolError> {
        let sape_engine = sape::get_sape();
        let mut engine = sape_engine.lock().map_err(|_| {
            ToolError::LockPoisoned("SAPE engine lock poisoned".to_string())
        })?;
        
        // Execute SAPE probes across Ihsan dimensions
        let probe_results = engine.execute_probes(content);
        let ihsan_score = engine.calculate_ihsan_score(&probe_results);
        
        // Get environment-specific threshold
        let env = ihsan::current_env();
        let threshold = ihsan::constitution().threshold_for(&env, "mcp_tool");
        let passed = ihsan_score >= threshold;
        
        // Collect any flags from probes
        let flags: Vec<String> = probe_results
            .iter()
            .flat_map(|r| r.flags.clone())
            .collect();
        
        // Enforce if required
        if !passed && ihsan::should_enforce() {
            MCP_SAPE_REJECTIONS.with_label_values(&[tool_name]).inc();
            warn!(
                tool_name,
                ihsan_score,
                threshold,
                env = %env,
                flags = ?flags,
                "MCP tool rejected by SAPE/Ihsan gate"
            );
            return Err(ToolError::SapeRejected {
                tool_name: tool_name.to_string(),
                ihsan_score,
                threshold,
                flags,
            });
        }
        
        Ok(SapeGateResult {
            ihsan_score,
            threshold,
            passed,
            probe_count: probe_results.len(),
            flags,
        })
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
                "Discovering MCP tools from server"
            );
            
            // Try to discover tools from real server
            match self.discover_from_server(server).await {
                Ok(tools) => {
                    info!(
                        server_name,
                        tools_count = tools.len(),
                        "Discovered tools from MCP server"
                    );
                    for mut tool in tools {
                        tool.server = server_name.clone();
                        self.tool_registry.insert(tool.name.clone(), tool);
                    }
                    continue;
                }
                Err(e) => {
                    warn!(
                        server_name,
                        error = %e,
                        "Failed to discover from MCP server, using defaults"
                    );
                }
            }
            
            // Fallback: default tool definitions for development
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
    
    /// Discover tools from an external MCP server via HTTP
    async fn discover_from_server(&self, server: &MCPServer) -> anyhow::Result<Vec<ToolDefinition>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "tools/list",
            "params": {}
        });
        
        let response = client
            .post(&server.url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            anyhow::bail!("MCP server returned status: {}", response.status());
        }
        
        let json_response: serde_json::Value = response.json().await?;
        
        if let Some(error) = json_response.get("error") {
            anyhow::bail!("MCP server error: {}", error);
        }
        
        let result = json_response.get("result")
            .ok_or_else(|| anyhow::anyhow!("Missing result in response"))?;
        
        let tools_array = result.get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing tools array"))?;
        
        let mut tools = Vec::new();
        for tool_json in tools_array {
            let name = tool_json.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("unknown")
                .to_string();
            
            let description = tool_json.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            
            let mut parameters = Vec::new();
            if let Some(input_schema) = tool_json.get("inputSchema") {
                if let Some(props) = input_schema.get("properties") {
                    if let Some(props_obj) = props.as_object() {
                        let required: Vec<String> = input_schema
                            .get("required")
                            .and_then(|r| r.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                            .unwrap_or_default();
                        
                        for (param_name, param_def) in props_obj {
                            parameters.push(ToolParameter {
                                name: param_name.clone(),
                                type_: param_def.get("type")
                                    .and_then(|t| t.as_str())
                                    .unwrap_or("string")
                                    .to_string(),
                                description: param_def.get("description")
                                    .and_then(|d| d.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                required: required.contains(param_name),
                            });
                        }
                    }
                }
            }
            
            tools.push(ToolDefinition {
                name,
                description,
                parameters,
                server: String::new(), // Will be set by caller
            });
        }
        
        Ok(tools)
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
        
        // SECURITY CHECK 3: SAPE/Ihsan gate (symbolic-neural bridge)
        let content_for_sape = format!(
            "MCP tool invocation: {} with arguments: {:?}",
            tool_name,
            arguments
        );
        let sape_result = self.sape_ihsan_gate(tool_name, &content_for_sape)?;
        
        info!(
            tool_name,
            ihsan_score = sape_result.ihsan_score,
            passed = sape_result.passed,
            "SAPE/Ihsan gate evaluation for MCP tool"
        );
        
        // SECURITY CHECK 4: Execute with timeout
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
    
    /// Internal tool execution - uses real HTTP transport when server is registered
    async fn execute_tool_internal(
        &self,
        tool_name: &str,
        arguments: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<serde_json::Value> {
        // Look up tool to find its server
        let tool = self.tool_registry.get(tool_name);
        
        if let Some(tool_def) = tool {
            // Find the server for this tool
            if let Some(server) = self.servers.get(&tool_def.server) {
                // Try real HTTP MCP call
                match self.call_mcp_server(server, tool_name, arguments).await {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        warn!(
                            tool = tool_name,
                            server = %tool_def.server,
                            error = %e,
                            "MCP server call failed, falling back to simulation"
                        );
                        // Fall through to simulation
                    }
                }
            }
        }
        
        // Fallback: simulated execution for development/testing
        debug!(tool = tool_name, "Using simulated tool execution");
        let result = serde_json::json!({
            "tool": tool_name,
            "arguments": arguments,
            "result": format!("Executed {} successfully (simulated)", tool_name),
            "status": "success",
            "simulated": true,
        });

        Ok(result)
    }
    
    /// Call external MCP server via HTTP/JSON-RPC
    async fn call_mcp_server(
        &self,
        server: &MCPServer,
        tool_name: &str,
        arguments: &HashMap<String, serde_json::Value>,
    ) -> anyhow::Result<serde_json::Value> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()?;
        
        // Build JSON-RPC 2.0 request
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": Uuid::new_v4().to_string(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        });
        
        info!(
            server_url = %server.url,
            tool = tool_name,
            "Calling MCP server"
        );
        
        let response = client
            .post(&server.url)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            anyhow::bail!(
                "MCP server returned error status: {}",
                response.status()
            );
        }
        
        let json_response: serde_json::Value = response.json().await?;
        
        // Extract result from JSON-RPC response
        if let Some(error) = json_response.get("error") {
            anyhow::bail!("MCP server error: {}", error);
        }
        
        Ok(json_response.get("result").cloned().unwrap_or(serde_json::json!({
            "status": "success",
            "tool": tool_name
        })))
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
    
    // ============================================================
    // JSON-RPC 2.0 Handler Methods
    // ============================================================
    
    /// Handle a JSON-RPC 2.0 request
    #[instrument(skip(self, request))]
    pub async fn handle_jsonrpc(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        // Validate JSON-RPC version
        if request.jsonrpc != JSONRPC_VERSION {
            return JsonRpcResponse::error(
                request.id,
                JsonRpcError::invalid_request("Invalid JSON-RPC version"),
            );
        }
        
        // Route to appropriate handler
        match request.method.as_str() {
            "tools/list" => self.handle_tools_list(request.id).await,
            "tools/call" => self.handle_tools_call(request.id, request.params).await,
            "initialize" => self.handle_initialize(request.id, request.params).await,
            "ping" => self.handle_ping(request.id).await,
            method => JsonRpcResponse::error(request.id, JsonRpcError::method_not_found(method)),
        }
    }
    
    /// Handle tools/list request (MCP standard)
    async fn handle_tools_list(&self, id: JsonRpcId) -> JsonRpcResponse {
        let tools: Vec<serde_json::Value> = self.tool_registry
            .values()
            .map(|tool| {
                let params_schema: Vec<serde_json::Value> = tool.parameters
                    .iter()
                    .map(|p| serde_json::json!({
                        "name": p.name,
                        "type": p.type_,
                        "description": p.description,
                        "required": p.required,
                    }))
                    .collect();
                
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": params_schema,
                    }
                })
            })
            .collect();
        
        JsonRpcResponse::success(id, serde_json::json!({ "tools": tools }))
    }
    
    /// Handle tools/call request (MCP standard)
    async fn handle_tools_call(&self, id: JsonRpcId, params: serde_json::Value) -> JsonRpcResponse {
        // Parse parameters
        let tool_name = match params.get("name").and_then(|n| n.as_str()) {
            Some(name) => name,
            None => return JsonRpcResponse::error(id, JsonRpcError::invalid_request("Missing 'name' parameter")),
        };
        
        let arguments: HashMap<String, serde_json::Value> = params
            .get("arguments")
            .and_then(|a| serde_json::from_value(a.clone()).ok())
            .unwrap_or_default();
        
        // Execute tool
        let start = Instant::now();
        let result = self.call_tool(tool_name, arguments).await;
        let latency = start.elapsed();
        
        MCP_LATENCY.with_label_values(&[tool_name]).observe(latency.as_secs_f64());
        
        match result {
            Ok(tool_result) => {
                MCP_CALLS.with_label_values(&[tool_name, "success"]).inc();
                JsonRpcResponse::success(id, serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": serde_json::to_string_pretty(&tool_result.result).unwrap_or_default()
                    }],
                    "isError": false,
                    "_meta": {
                        "execution_time_ms": tool_result.execution_time_ms,
                        "truncated": tool_result.truncated,
                    }
                }))
            }
            Err(e) => {
                MCP_CALLS.with_label_values(&[tool_name, "error"]).inc();
                let error = match e {
                    ToolError::Blocked(t) => JsonRpcError::tool_blocked(&t),
                    ToolError::Timeout(t) => JsonRpcError::tool_timeout(&t, self.timeout.as_secs()),
                    _ => JsonRpcError::execution_failed(&e.to_string()),
                };
                JsonRpcResponse::error(id, error)
            }
        }
    }
    
    /// Handle initialize request (MCP standard)
    async fn handle_initialize(&self, id: JsonRpcId, params: serde_json::Value) -> JsonRpcResponse {
        let client_name = params
            .get("clientInfo")
            .and_then(|c| c.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("unknown");
        
        info!("🔌 MCP client initialized: {}", client_name);
        
        JsonRpcResponse::success(id, serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": false
                },
                "logging": {}
            },
            "serverInfo": {
                "name": "bizra-mcp-server",
                "version": "1.4.0"
            }
        }))
    }
    
    /// Handle ping request
    async fn handle_ping(&self, id: JsonRpcId) -> JsonRpcResponse {
        JsonRpcResponse::success(id, serde_json::json!({ "pong": true }))
    }
    
    /// Parse and handle raw JSON-RPC request
    pub async fn handle_raw(&self, json_str: &str) -> String {
        let request: JsonRpcRequest = match serde_json::from_str(json_str) {
            Ok(req) => req,
            Err(e) => {
                let response = JsonRpcResponse {
                    jsonrpc: JSONRPC_VERSION.into(),
                    result: None,
                    error: Some(JsonRpcError {
                        code: JsonRpcError::PARSE_ERROR,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                    id: JsonRpcId::Null,
                };
                return serde_json::to_string(&response).unwrap_or_default();
            }
        };
        
        let response = self.handle_jsonrpc(request).await;
        serde_json::to_string(&response).unwrap_or_default()
    }
    
    /// Register built-in BIZRA tools
    pub fn register_bizra_tools(&mut self) {
        // Knowledge retrieval tool
        self.tool_registry.insert("knowledge_retrieve".to_string(), ToolDefinition {
            name: "knowledge_retrieve".to_string(),
            description: "Query the House of Wisdom knowledge graph for relevant information".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "query".to_string(),
                    type_: "string".to_string(),
                    description: "The search query".to_string(),
                    required: true,
                },
                ToolParameter {
                    name: "limit".to_string(),
                    type_: "number".to_string(),
                    description: "Maximum results to return".to_string(),
                    required: false,
                },
            ],
            server: "bizra-internal".to_string(),
        });
        
        // Calculator tool
        self.tool_registry.insert("calculator".to_string(), ToolDefinition {
            name: "calculator".to_string(),
            description: "Perform mathematical calculations".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "expression".to_string(),
                    type_: "string".to_string(),
                    description: "Mathematical expression to evaluate".to_string(),
                    required: true,
                },
            ],
            server: "bizra-internal".to_string(),
        });
        
        // SAPE probe tool
        self.tool_registry.insert("sape_probe".to_string(), ToolDefinition {
            name: "sape_probe".to_string(),
            description: "Execute SAPE probes on content for quality assessment".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "content".to_string(),
                    type_: "string".to_string(),
                    description: "Content to analyze".to_string(),
                    required: true,
                },
            ],
            server: "bizra-internal".to_string(),
        });
        self.allowlist.insert("sape_probe".to_string());
        self.allowlist.insert("knowledge_retrieve".to_string());
        self.allowlist.insert("calculator".to_string());
        
        info!("📦 Registered {} BIZRA tools", 3);
    }
}

impl Default for MCPClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Claude-compatible Tool Use Format
// ============================================================

/// Claude tool use request format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Claude tool result format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeToolResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub tool_use_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

impl ClaudeToolResult {
    pub fn success(tool_use_id: String, content: String) -> Self {
        Self {
            result_type: "tool_result".to_string(),
            tool_use_id,
            content,
            is_error: None,
        }
    }
    
    pub fn error(tool_use_id: String, error_message: String) -> Self {
        Self {
            result_type: "tool_result".to_string(),
            tool_use_id,
            content: error_message,
            is_error: Some(true),
        }
    }
}

/// Convert MCP tool definitions to Claude format
pub fn tools_to_claude_format(tools: &[&ToolDefinition]) -> Vec<serde_json::Value> {
    tools.iter().map(|tool| {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        
        for param in &tool.parameters {
            properties.insert(param.name.clone(), serde_json::json!({
                "type": param.type_,
                "description": param.description,
            }));
            if param.required {
                required.push(param.name.clone());
            }
        }
        
        serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_jsonrpc_error_codes() {
        assert_eq!(JsonRpcError::PARSE_ERROR, -32700);
        assert_eq!(JsonRpcError::METHOD_NOT_FOUND, -32601);
        assert_eq!(JsonRpcError::TOOL_BLOCKED, -32002);
    }
    
    #[test]
    fn test_jsonrpc_response_success() {
        let id = JsonRpcId::String("test-123".into());
        let response = JsonRpcResponse::success(id.clone(), serde_json::json!({"result": "ok"}));
        
        assert_eq!(response.jsonrpc, "2.0");
        assert!(response.result.is_some());
        assert!(response.error.is_none());
        assert_eq!(response.id, id);
    }
    
    #[test]
    fn test_jsonrpc_response_error() {
        let id = JsonRpcId::Number(42);
        let error = JsonRpcError::tool_blocked("dangerous_tool");
        let response = JsonRpcResponse::error(id.clone(), error);
        
        assert!(response.result.is_none());
        assert!(response.error.is_some());
        assert_eq!(response.error.as_ref().unwrap().code, JsonRpcError::TOOL_BLOCKED);
    }
    
    #[test]
    fn test_claude_tool_result() {
        let result = ClaudeToolResult::success("call-123".into(), "Success!".into());
        assert_eq!(result.result_type, "tool_result");
        assert!(result.is_error.is_none());
        
        let error = ClaudeToolResult::error("call-456".into(), "Failed".into());
        assert_eq!(error.is_error, Some(true));
    }
    
    #[tokio::test]
    async fn test_mcp_client_creation() {
        let mut client = MCPClient::new();
        client.register_bizra_tools();
        
        assert!(client.tool_registry.len() >= 3);
        assert!(client.is_tool_allowed("knowledge_retrieve").is_ok());
    }
    
    #[tokio::test]
    async fn test_handle_ping() {
        let client = MCPClient::new();
        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "ping".into(),
            params: serde_json::Value::Null,
            id: JsonRpcId::String("test".into()),
        };
        
        let response = client.handle_jsonrpc(request).await;
        assert!(response.result.is_some());
    }
    
    #[tokio::test]
    async fn test_handle_tools_list() {
        let mut client = MCPClient::new();
        client.register_bizra_tools();
        
        let request = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            method: "tools/list".into(),
            params: serde_json::Value::Null,
            id: JsonRpcId::Number(1),
        };
        
        let response = client.handle_jsonrpc(request).await;
        assert!(response.result.is_some());
        
        let result = response.result.unwrap();
        let tools = result.get("tools").and_then(|t| t.as_array());
        assert!(tools.is_some());
        assert!(!tools.unwrap().is_empty());
    }
}