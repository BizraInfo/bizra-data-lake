// src/sovereignty/policy.rs - Policy Sovereignty (Pillar 4: Governance)
//
// Principle: A policy engine decides what agents can do: tools, file access,
// network, budgets, escalation. "Default deny", explicit allowlists.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Policy decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyDecision {
    /// Action allowed
    Allow,
    /// Action denied
    Deny,
    /// Action requires escalation (human review)
    Escalate,
    /// Action allowed with audit logging
    AllowWithAudit,
}

/// Resource type for access control
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// File system path
    File(String),
    /// Network endpoint
    Network(String),
    /// MCP tool
    Tool(String),
    /// System capability
    Capability(String),
    /// Token budget
    TokenBudget,
    /// Compute budget
    ComputeBudget,
}

/// Action type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionType {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Execute
    Execute,
    /// Delete
    Delete,
    /// Create
    Create,
    /// Network call
    Call,
    /// Resource consumption
    Consume,
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule ID
    pub id: String,
    /// Agent ID (or * for all)
    pub agent_pattern: String,
    /// Resource pattern
    pub resource_pattern: String,
    /// Allowed actions
    pub allowed_actions: HashSet<ActionType>,
    /// Decision for matches
    pub decision: PolicyDecision,
    /// Priority (higher = more specific)
    pub priority: i32,
    /// Conditions (optional)
    pub conditions: Option<PolicyConditions>,
}

/// Policy conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConditions {
    /// Minimum Ihsān score required
    pub min_ihsan: Option<f64>,
    /// Required SAT approvals
    pub min_sat_approvals: Option<usize>,
    /// Time-based restrictions
    pub time_window: Option<TimeWindow>,
    /// Rate limit
    pub rate_limit: Option<RateLimit>,
}

/// Time window for access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeWindow {
    /// Start hour (0-23)
    pub start_hour: u8,
    /// End hour (0-23)
    pub end_hour: u8,
    /// Days of week (0=Sunday, 6=Saturday)
    pub days: Vec<u8>,
}

/// Rate limit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Max requests
    pub max_requests: u32,
    /// Per time period (seconds)
    pub period_seconds: u32,
}

/// Agent allowlist for tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentAllowlist {
    /// Agent ID
    pub agent_id: String,
    /// Allowed tools (MCP tool names)
    pub allowed_tools: HashSet<String>,
    /// Allowed file paths (glob patterns)
    pub allowed_paths: Vec<String>,
    /// Allowed network endpoints
    pub allowed_endpoints: Vec<String>,
    /// Token budget per request
    pub token_budget: u32,
    /// Max concurrent requests
    pub max_concurrent: u32,
}

impl AgentAllowlist {
    /// Create a restrictive allowlist (default deny)
    pub fn restrictive(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            allowed_tools: HashSet::new(),
            allowed_paths: Vec::new(),
            allowed_endpoints: Vec::new(),
            token_budget: 1000,
            max_concurrent: 1,
        }
    }

    /// Allow a specific tool
    pub fn allow_tool(mut self, tool: impl Into<String>) -> Self {
        self.allowed_tools.insert(tool.into());
        self
    }

    /// Allow a path pattern
    pub fn allow_path(mut self, pattern: impl Into<String>) -> Self {
        self.allowed_paths.push(pattern.into());
        self
    }

    /// Allow an endpoint
    pub fn allow_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.allowed_endpoints.push(endpoint.into());
        self
    }

    /// Check if tool is allowed
    pub fn is_tool_allowed(&self, tool: &str) -> bool {
        self.allowed_tools.contains(tool) || self.allowed_tools.contains("*")
    }

    /// Check if path is allowed
    pub fn is_path_allowed(&self, path: &str) -> bool {
        for pattern in &self.allowed_paths {
            if pattern == "*" {
                return true;
            }
            if path.starts_with(pattern) {
                return true;
            }
        }
        false
    }

    /// Check if endpoint is allowed
    pub fn is_endpoint_allowed(&self, endpoint: &str) -> bool {
        for allowed in &self.allowed_endpoints {
            if allowed == "*" {
                return true;
            }
            if endpoint.contains(allowed) {
                return true;
            }
        }
        false
    }
}

/// Policy engine with default deny
pub struct PolicyEngine {
    /// Rules ordered by priority
    rules: Vec<PolicyRule>,
    /// Agent allowlists
    allowlists: HashMap<String, AgentAllowlist>,
    /// Global blocklist
    global_blocklist: HashSet<String>,
    /// Audit log enabled
    audit_enabled: bool,
}

impl PolicyEngine {
    /// Create with default deny
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            allowlists: HashMap::new(),
            global_blocklist: HashSet::new(),
            audit_enabled: true,
        }
    }

    /// Add a policy rule
    pub fn add_rule(&mut self, rule: PolicyRule) {
        self.rules.push(rule);
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Set agent allowlist
    pub fn set_allowlist(&mut self, allowlist: AgentAllowlist) {
        self.allowlists
            .insert(allowlist.agent_id.clone(), allowlist);
    }

    /// Add to global blocklist
    pub fn block(&mut self, resource: impl Into<String>) {
        self.global_blocklist.insert(resource.into());
    }

    /// Evaluate policy for an action
    pub fn evaluate(
        &self,
        agent_id: &str,
        resource: &ResourceType,
        action: ActionType,
    ) -> PolicyDecision {
        // Check global blocklist first
        let resource_str = self.resource_to_string(resource);
        if self
            .global_blocklist
            .iter()
            .any(|b| resource_str.contains(b))
        {
            return PolicyDecision::Deny;
        }

        // Check agent allowlist
        if let Some(allowlist) = self.allowlists.get(agent_id) {
            match resource {
                ResourceType::Tool(tool) if !allowlist.is_tool_allowed(tool) => {
                    return PolicyDecision::Deny;
                }
                ResourceType::File(path) if !allowlist.is_path_allowed(path) => {
                    return PolicyDecision::Deny;
                }
                ResourceType::Network(endpoint) if !allowlist.is_endpoint_allowed(endpoint) => {
                    return PolicyDecision::Deny;
                }
                _ => {}
            }
        }

        // Check rules
        for rule in &self.rules {
            if self.rule_matches(rule, agent_id, &resource_str, action) {
                return rule.decision;
            }
        }

        // DEFAULT DENY
        PolicyDecision::Deny
    }

    /// Check if rule matches
    fn rule_matches(
        &self,
        rule: &PolicyRule,
        agent_id: &str,
        resource: &str,
        action: ActionType,
    ) -> bool {
        // Check agent pattern
        if rule.agent_pattern != "*" && !agent_id.contains(&rule.agent_pattern) {
            return false;
        }

        // Check resource pattern
        if rule.resource_pattern != "*" && !resource.contains(&rule.resource_pattern) {
            return false;
        }

        // Check action
        if !rule.allowed_actions.contains(&action) && !rule.allowed_actions.is_empty() {
            return false;
        }

        true
    }

    /// Convert resource to string for matching
    fn resource_to_string(&self, resource: &ResourceType) -> String {
        match resource {
            ResourceType::File(p) => format!("file:{}", p),
            ResourceType::Network(e) => format!("net:{}", e),
            ResourceType::Tool(t) => format!("tool:{}", t),
            ResourceType::Capability(c) => format!("cap:{}", c),
            ResourceType::TokenBudget => "budget:token".to_string(),
            ResourceType::ComputeBudget => "budget:compute".to_string(),
        }
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOOL CATEGORIES (P0 Implementation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Tool category for grouping permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToolCategory {
    /// Read-only file access
    FileRead,
    /// Write/modify file access
    FileWrite,
    /// Terminal/command execution
    Terminal,
    /// Search and discovery
    Search,
    /// Testing and validation
    Testing,
    /// Code navigation
    CodeNav,
    /// External network access (RESTRICTED)
    Network,
    /// MCP protocol tools
    McpProtocol,
    /// Database operations
    Database,
    /// Dangerous/privileged operations
    Dangerous,
}

impl ToolCategory {
    /// Get tools in this category
    pub fn tools(&self) -> &'static [&'static str] {
        match self {
            Self::FileRead => &[
                "read_file",
                "list_dir",
                "file_search",
                "read_notebook_cell_output",
                "copilot_getNotebookSummary",
            ],
            Self::FileWrite => &[
                "create_file",
                "create_directory",
                "replace_string_in_file",
                "multi_replace_string_in_file",
                "edit_notebook_file",
            ],
            Self::Terminal => &[
                "run_in_terminal",
                "get_terminal_output",
                "terminal_last_command",
                "terminal_selection",
            ],
            Self::Search => &[
                "semantic_search",
                "grep_search",
                "file_search",
                "list_code_usages",
            ],
            Self::Testing => &[
                "runTests",
                "run_notebook_cell",
                "get_errors",
                "test_failure",
            ],
            Self::CodeNav => &["list_code_usages", "get_errors", "get_changed_files"],
            Self::Network => &["fetch_webpage", "open_simple_browser"],
            Self::McpProtocol => &["mcp_microsoft_mar_convert_to_markdown"],
            Self::Database => &[
                "dbclient-execute-query",
                "dbclient-get-databases",
                "dbclient-get-tables",
                "mssql_run_query",
                "mssql_connect",
                "mssql_list_servers",
            ],
            Self::Dangerous => &[
                "run_in_terminal",        // Can execute arbitrary commands
                "create_file",            // Can overwrite files
                "fetch_webpage",          // External network
                "dbclient-execute-query", // SQL injection risk
            ],
        }
    }

    /// Check if tool belongs to this category
    pub fn contains(&self, tool: &str) -> bool {
        self.tools().contains(&tool)
    }
}

/// Permission level for agents
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PermissionLevel {
    /// No permissions
    None = 0,
    /// Read-only access
    ReadOnly = 1,
    /// Read + search access
    ReadSearch = 2,
    /// Standard development access
    Developer = 3,
    /// Full access (requires SAT approval)
    Full = 4,
    /// Admin access (requires Ihsān ≥ 0.98)
    Admin = 5,
}

impl PermissionLevel {
    /// Get allowed categories for this level
    pub fn allowed_categories(&self) -> Vec<ToolCategory> {
        match self {
            Self::None => vec![],
            Self::ReadOnly => vec![ToolCategory::FileRead],
            Self::ReadSearch => vec![
                ToolCategory::FileRead,
                ToolCategory::Search,
                ToolCategory::CodeNav,
            ],
            Self::Developer => vec![
                ToolCategory::FileRead,
                ToolCategory::FileWrite,
                ToolCategory::Search,
                ToolCategory::Testing,
                ToolCategory::CodeNav,
            ],
            Self::Full => vec![
                ToolCategory::FileRead,
                ToolCategory::FileWrite,
                ToolCategory::Terminal,
                ToolCategory::Search,
                ToolCategory::Testing,
                ToolCategory::CodeNav,
                ToolCategory::McpProtocol,
                ToolCategory::Database,
            ],
            Self::Admin => vec![
                ToolCategory::FileRead,
                ToolCategory::FileWrite,
                ToolCategory::Terminal,
                ToolCategory::Search,
                ToolCategory::Testing,
                ToolCategory::CodeNav,
                ToolCategory::Network,
                ToolCategory::McpProtocol,
                ToolCategory::Database,
                ToolCategory::Dangerous,
            ],
        }
    }
}

/// Agent permission configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPermissions {
    /// Agent identifier
    pub agent_id: String,
    /// Agent type (PAT or SAT)
    pub agent_type: AgentType,
    /// Base permission level
    pub level: PermissionLevel,
    /// Additional allowed tools (beyond level)
    pub extra_tools: HashSet<String>,
    /// Explicitly denied tools
    pub denied_tools: HashSet<String>,
    /// Allowed file path patterns
    pub allowed_paths: Vec<String>,
    /// Denied file path patterns
    pub denied_paths: Vec<String>,
    /// Token budget per request
    pub token_budget: u32,
    /// Requires SAT approval for dangerous ops
    pub require_sat_approval: bool,
    /// Minimum Ihsān score for operations
    pub min_ihsan: f64,
}

/// Agent type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentType {
    /// Personal Agentic Team (execution)
    PAT,
    /// System Agentic Team (validation)
    SAT,
    /// External/unknown
    External,
}

impl AgentPermissions {
    /// Create with permission level
    pub fn new(agent_id: impl Into<String>, agent_type: AgentType, level: PermissionLevel) -> Self {
        Self {
            agent_id: agent_id.into(),
            agent_type,
            level,
            extra_tools: HashSet::new(),
            denied_tools: HashSet::new(),
            allowed_paths: Vec::new(),
            denied_paths: Vec::new(),
            token_budget: match level {
                PermissionLevel::None => 0,
                PermissionLevel::ReadOnly => 2000,
                PermissionLevel::ReadSearch => 5000,
                PermissionLevel::Developer => 10000,
                PermissionLevel::Full => 50000,
                PermissionLevel::Admin => 100000,
            },
            require_sat_approval: level >= PermissionLevel::Full,
            min_ihsan: match level {
                PermissionLevel::Admin => 0.98,
                PermissionLevel::Full => 0.95,
                _ => 0.80,
            },
        }
    }

    /// Add extra allowed tool
    pub fn allow_extra(mut self, tool: impl Into<String>) -> Self {
        self.extra_tools.insert(tool.into());
        self
    }

    /// Deny specific tool
    pub fn deny(mut self, tool: impl Into<String>) -> Self {
        self.denied_tools.insert(tool.into());
        self
    }

    /// Allow path pattern
    pub fn allow_path(mut self, pattern: impl Into<String>) -> Self {
        self.allowed_paths.push(pattern.into());
        self
    }

    /// Deny path pattern
    pub fn deny_path(mut self, pattern: impl Into<String>) -> Self {
        self.denied_paths.push(pattern.into());
        self
    }

    /// Check if tool is allowed
    pub fn is_tool_allowed(&self, tool: &str) -> bool {
        // Explicit deny takes precedence
        if self.denied_tools.contains(tool) {
            return false;
        }

        // Check extra allowed
        if self.extra_tools.contains(tool) {
            return true;
        }

        // Check level categories
        for category in self.level.allowed_categories() {
            if category.contains(tool) {
                return true;
            }
        }

        false
    }

    /// Check if path is allowed
    pub fn is_path_allowed(&self, path: &str) -> bool {
        // Check denied paths first
        for pattern in &self.denied_paths {
            if path.starts_with(pattern) || pattern == "*" {
                return false;
            }
        }

        // Check allowed paths
        if self.allowed_paths.is_empty() {
            // Default: allow if level permits file access
            return self.level >= PermissionLevel::ReadOnly;
        }

        for pattern in &self.allowed_paths {
            if pattern == "*" || path.starts_with(pattern) {
                return true;
            }
        }

        false
    }

    /// Check if action requires SAT approval
    pub fn requires_approval(&self, tool: &str) -> bool {
        if !self.require_sat_approval {
            return false;
        }

        // Dangerous tools always require approval
        ToolCategory::Dangerous.contains(tool)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEFAULT AGENT PERMISSIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Default PAT agent allowlists
pub fn default_pat_allowlists() -> Vec<AgentAllowlist> {
    vec![
        // Strategic Agent (PRIME) - high trust
        AgentAllowlist::restrictive("PRIME")
            .allow_tool("read_file")
            .allow_tool("semantic_search")
            .allow_tool("list_dir")
            .allow_path("docs/")
            .allow_path("src/"),
        // Implementation Agent (TEKNE) - file write access
        AgentAllowlist::restrictive("TEKNE")
            .allow_tool("read_file")
            .allow_tool("create_file")
            .allow_tool("replace_string_in_file")
            .allow_tool("run_in_terminal")
            .allow_path("src/")
            .allow_path("tests/"),
        // Quality Agent (LOGOS) - read-only
        AgentAllowlist::restrictive("LOGOS")
            .allow_tool("read_file")
            .allow_tool("get_errors")
            .allow_tool("runTests")
            .allow_path("src/")
            .allow_path("tests/"),
        // Memory Agent (GNOSTIC) - knowledge access
        AgentAllowlist::restrictive("GNOSTIC")
            .allow_tool("read_file")
            .allow_tool("semantic_search")
            .allow_path("docs/")
            .allow_path(".bizra/"),
    ]
}

/// Default PAT agent permissions (new system)
pub fn default_pat_permissions() -> Vec<AgentPermissions> {
    vec![
        // PRIME (Strategist) - ReadSearch level
        AgentPermissions::new("PRIME", AgentType::PAT, PermissionLevel::ReadSearch)
            .allow_path("docs/")
            .allow_path("src/")
            .allow_path("constitution/"),
        // GNOSTIC (Memory Custodian) - ReadSearch level
        AgentPermissions::new("GNOSTIC", AgentType::PAT, PermissionLevel::ReadSearch)
            .allow_path("docs/")
            .allow_path(".bizra/")
            .allow_path("memory-bank/"),
        // TEKNE (Implementation) - Developer level
        AgentPermissions::new("TEKNE", AgentType::PAT, PermissionLevel::Developer)
            .allow_path("src/")
            .allow_path("tests/")
            .allow_path("core/")
            .deny_path("constitution/") // Can't modify constitution
            .deny("fetch_webpage"), // No external network
        // AESTHETE (UX/Design) - Developer level
        AgentPermissions::new("AESTHETE", AgentType::PAT, PermissionLevel::Developer)
            .allow_path("src/")
            .allow_path("docs/")
            .allow_path("static/")
            .deny("run_in_terminal"), // No terminal access
        // LOGOS (Critic/Safety) - Full level (validates others)
        AgentPermissions::new("LOGOS", AgentType::PAT, PermissionLevel::Full)
            .allow_path("*") // Can read all for validation
            .deny("create_file") // Read-only for validation
            .deny("replace_string_in_file"),
        // AXON (Synthesis) - ReadSearch level
        AgentPermissions::new("AXON", AgentType::PAT, PermissionLevel::ReadSearch)
            .allow_path("docs/")
            .allow_path("src/"),
        // KAIROS (Executor) - Full level
        AgentPermissions::new("KAIROS", AgentType::PAT, PermissionLevel::Full)
            .allow_path("*")
            .deny_path("constitution/") // Protected
            .deny_path(".git/"), // Protected
    ]
}

/// Default SAT agent permissions
pub fn default_sat_permissions() -> Vec<AgentPermissions> {
    vec![
        // Security Guardian - Full level (needs to inspect everything)
        AgentPermissions::new("security_guardian", AgentType::SAT, PermissionLevel::Full)
            .allow_path("*")
            .deny("create_file")
            .deny("replace_string_in_file")
            .deny("run_in_terminal"),
        // Ethics Validator - ReadSearch level
        AgentPermissions::new(
            "ethics_validator",
            AgentType::SAT,
            PermissionLevel::ReadSearch,
        )
        .allow_path("constitution/")
        .allow_path("docs/")
        .allow_path("src/"),
        // Performance Monitor - ReadSearch level
        AgentPermissions::new(
            "performance_monitor",
            AgentType::SAT,
            PermissionLevel::ReadSearch,
        )
        .allow_extra("runTests")
        .allow_path("src/")
        .allow_path("tests/"),
        // Consistency Checker - ReadSearch level
        AgentPermissions::new(
            "consistency_checker",
            AgentType::SAT,
            PermissionLevel::ReadSearch,
        )
        .allow_path("*"),
        // Resource Optimizer - ReadOnly level
        AgentPermissions::new(
            "resource_optimizer",
            AgentType::SAT,
            PermissionLevel::ReadOnly,
        )
        .allow_path("*"),
    ]
}

/// Permission registry for all agents
pub struct PermissionRegistry {
    /// Agent permissions by ID
    permissions: HashMap<String, AgentPermissions>,
    /// Global denied tools
    global_denied: HashSet<String>,
    /// Audit log of permission checks
    audit_log: Vec<PermissionAuditEntry>,
}

/// Audit entry for permission checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionAuditEntry {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Agent ID
    pub agent_id: String,
    /// Tool requested
    pub tool: String,
    /// Path (if applicable)
    pub path: Option<String>,
    /// Decision
    pub decision: PolicyDecision,
    /// Reason
    pub reason: String,
}

impl PermissionRegistry {
    /// Create with default permissions
    pub fn with_defaults() -> Self {
        let mut permissions = HashMap::new();

        for perm in default_pat_permissions() {
            permissions.insert(perm.agent_id.clone(), perm);
        }

        for perm in default_sat_permissions() {
            permissions.insert(perm.agent_id.clone(), perm);
        }

        let mut global_denied = HashSet::new();
        // Globally deny external AI APIs
        global_denied.insert("call_openai".to_string());
        global_denied.insert("call_anthropic".to_string());
        global_denied.insert("call_external_api".to_string());

        Self {
            permissions,
            global_denied,
            audit_log: Vec::new(),
        }
    }

    /// Check tool permission
    pub fn check_tool(&mut self, agent_id: &str, tool: &str) -> PolicyDecision {
        let decision = self.evaluate_tool(agent_id, tool);

        self.audit_log.push(PermissionAuditEntry {
            timestamp: chrono::Utc::now(),
            agent_id: agent_id.to_string(),
            tool: tool.to_string(),
            path: None,
            decision,
            reason: self.decision_reason(agent_id, tool),
        });

        decision
    }

    /// Check path permission
    pub fn check_path(&mut self, agent_id: &str, path: &str) -> PolicyDecision {
        let decision = self.evaluate_path(agent_id, path);

        self.audit_log.push(PermissionAuditEntry {
            timestamp: chrono::Utc::now(),
            agent_id: agent_id.to_string(),
            tool: "path_access".to_string(),
            path: Some(path.to_string()),
            decision,
            reason: format!("Path access check for {}", path),
        });

        decision
    }

    fn evaluate_tool(&self, agent_id: &str, tool: &str) -> PolicyDecision {
        // Global deny takes precedence
        if self.global_denied.contains(tool) {
            return PolicyDecision::Deny;
        }

        // Check agent permissions
        if let Some(perm) = self.permissions.get(agent_id) {
            if perm.is_tool_allowed(tool) {
                if perm.requires_approval(tool) {
                    return PolicyDecision::Escalate;
                }
                return PolicyDecision::AllowWithAudit;
            }
        }

        // Default deny for unknown agents
        PolicyDecision::Deny
    }

    fn evaluate_path(&self, agent_id: &str, path: &str) -> PolicyDecision {
        if let Some(perm) = self.permissions.get(agent_id) {
            if perm.is_path_allowed(path) {
                return PolicyDecision::AllowWithAudit;
            }
        }

        PolicyDecision::Deny
    }

    fn decision_reason(&self, agent_id: &str, tool: &str) -> String {
        if self.global_denied.contains(tool) {
            return format!("Tool '{}' is globally denied", tool);
        }

        if let Some(perm) = self.permissions.get(agent_id) {
            if perm.denied_tools.contains(tool) {
                return format!("Tool '{}' explicitly denied for agent", tool);
            }
            if perm.is_tool_allowed(tool) {
                return format!("Tool '{}' allowed by level {:?}", tool, perm.level);
            }
        }

        format!(
            "Agent '{}' not registered or insufficient permissions",
            agent_id
        )
    }

    /// Get audit log
    pub fn audit_log(&self) -> &[PermissionAuditEntry] {
        &self.audit_log
    }

    /// Register new agent
    pub fn register(&mut self, permissions: AgentPermissions) {
        self.permissions
            .insert(permissions.agent_id.clone(), permissions);
    }

    /// Get agent count
    pub fn agent_count(&self) -> usize {
        self.permissions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_deny() {
        let engine = PolicyEngine::new();

        let decision = engine.evaluate(
            "unknown-agent",
            &ResourceType::Tool("dangerous_tool".to_string()),
            ActionType::Execute,
        );

        assert_eq!(decision, PolicyDecision::Deny);
    }

    #[test]
    fn test_allowlist() {
        let mut engine = PolicyEngine::new();

        engine.set_allowlist(AgentAllowlist::restrictive("PRIME").allow_tool("read_file"));

        // Not allowed tool
        let decision = engine.evaluate(
            "PRIME",
            &ResourceType::Tool("write_file".to_string()),
            ActionType::Execute,
        );
        assert_eq!(decision, PolicyDecision::Deny);
    }

    #[test]
    fn test_global_blocklist() {
        let mut engine = PolicyEngine::new();

        engine.block("api.openai.com");

        let decision = engine.evaluate(
            "any-agent",
            &ResourceType::Network("https://api.openai.com/v1/chat".to_string()),
            ActionType::Call,
        );

        assert_eq!(decision, PolicyDecision::Deny);
    }

    #[test]
    fn test_path_pattern() {
        let allowlist = AgentAllowlist::restrictive("test").allow_path("src/");

        assert!(allowlist.is_path_allowed("src/main.rs"));
        assert!(!allowlist.is_path_allowed("secrets/key.pem"));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // NEW PERMISSION SYSTEM TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_tool_categories() {
        assert!(ToolCategory::FileRead.contains("read_file"));
        assert!(ToolCategory::FileWrite.contains("create_file"));
        assert!(ToolCategory::Terminal.contains("run_in_terminal"));
        assert!(ToolCategory::Dangerous.contains("run_in_terminal"));

        assert!(!ToolCategory::FileRead.contains("create_file"));
    }

    #[test]
    fn test_permission_levels() {
        // ReadOnly should only allow FileRead
        let read_only = PermissionLevel::ReadOnly.allowed_categories();
        assert!(read_only.contains(&ToolCategory::FileRead));
        assert!(!read_only.contains(&ToolCategory::FileWrite));

        // Developer should allow FileRead + FileWrite + Search + Testing
        let dev = PermissionLevel::Developer.allowed_categories();
        assert!(dev.contains(&ToolCategory::FileRead));
        assert!(dev.contains(&ToolCategory::FileWrite));
        assert!(dev.contains(&ToolCategory::Testing));
        assert!(!dev.contains(&ToolCategory::Terminal)); // No terminal

        // Full should include Terminal
        let full = PermissionLevel::Full.allowed_categories();
        assert!(full.contains(&ToolCategory::Terminal));
        assert!(!full.contains(&ToolCategory::Network)); // Still no external network

        // Only Admin gets Network
        let admin = PermissionLevel::Admin.allowed_categories();
        assert!(admin.contains(&ToolCategory::Network));
        assert!(admin.contains(&ToolCategory::Dangerous));
    }

    #[test]
    fn test_agent_permissions_tool_check() {
        let perm = AgentPermissions::new("TEKNE", AgentType::PAT, PermissionLevel::Developer);

        // Developer level allows file operations
        assert!(perm.is_tool_allowed("read_file"));
        assert!(perm.is_tool_allowed("create_file"));
        assert!(perm.is_tool_allowed("runTests"));

        // Developer level does NOT allow terminal
        assert!(!perm.is_tool_allowed("run_in_terminal"));

        // Does NOT allow external network
        assert!(!perm.is_tool_allowed("fetch_webpage"));
    }

    #[test]
    fn test_agent_permissions_deny_override() {
        let perm = AgentPermissions::new("TEKNE", AgentType::PAT, PermissionLevel::Developer)
            .deny("create_file"); // Explicitly deny despite level

        assert!(perm.is_tool_allowed("read_file"));
        assert!(!perm.is_tool_allowed("create_file")); // Denied!
    }

    #[test]
    fn test_agent_permissions_extra_allow() {
        let perm = AgentPermissions::new("PRIME", AgentType::PAT, PermissionLevel::ReadSearch)
            .allow_extra("run_in_terminal"); // Grant extra permission

        // ReadSearch doesn't normally allow terminal
        let base = AgentPermissions::new("test", AgentType::PAT, PermissionLevel::ReadSearch);
        assert!(!base.is_tool_allowed("run_in_terminal"));

        // But with extra permission it's allowed
        assert!(perm.is_tool_allowed("run_in_terminal"));
    }

    #[test]
    fn test_agent_path_permissions() {
        let perm = AgentPermissions::new("TEKNE", AgentType::PAT, PermissionLevel::Developer)
            .allow_path("src/")
            .allow_path("tests/")
            .deny_path("src/secrets/");

        assert!(perm.is_path_allowed("src/main.rs"));
        assert!(perm.is_path_allowed("tests/unit_test.rs"));
        assert!(!perm.is_path_allowed("src/secrets/key.pem")); // Denied!
        assert!(!perm.is_path_allowed("constitution/ihsan.yaml")); // Not in allowed
    }

    #[test]
    fn test_default_pat_permissions() {
        let perms = default_pat_permissions();

        // Should have all 7 PAT agents
        assert_eq!(perms.len(), 7);

        // PRIME should be ReadSearch
        let prime = perms.iter().find(|p| p.agent_id == "PRIME").unwrap();
        assert_eq!(prime.level, PermissionLevel::ReadSearch);

        // TEKNE should be Developer
        let tekne = perms.iter().find(|p| p.agent_id == "TEKNE").unwrap();
        assert_eq!(tekne.level, PermissionLevel::Developer);
        assert!(!tekne.is_tool_allowed("fetch_webpage")); // Explicitly denied

        // LOGOS should be Full (validator)
        let logos = perms.iter().find(|p| p.agent_id == "LOGOS").unwrap();
        assert_eq!(logos.level, PermissionLevel::Full);
        assert!(!logos.is_tool_allowed("create_file")); // Read-only validator
    }

    #[test]
    fn test_default_sat_permissions() {
        let perms = default_sat_permissions();

        // Should have all 5 SAT agents
        assert_eq!(perms.len(), 5);

        // Security Guardian should be Full but read-only
        let sec = perms
            .iter()
            .find(|p| p.agent_id == "security_guardian")
            .unwrap();
        assert_eq!(sec.level, PermissionLevel::Full);
        assert!(sec.is_tool_allowed("read_file"));
        assert!(!sec.is_tool_allowed("create_file")); // Denied
        assert!(!sec.is_tool_allowed("run_in_terminal")); // Denied
    }

    #[test]
    fn test_permission_registry() {
        let mut registry = PermissionRegistry::with_defaults();

        // Should have PAT + SAT agents
        assert_eq!(registry.agent_count(), 12); // 7 PAT + 5 SAT

        // TEKNE can read files
        let decision = registry.check_tool("TEKNE", "read_file");
        assert_eq!(decision, PolicyDecision::AllowWithAudit);

        // TEKNE cannot fetch webpages (explicitly denied)
        let decision = registry.check_tool("TEKNE", "fetch_webpage");
        assert_eq!(decision, PolicyDecision::Deny);

        // Unknown agent is denied
        let decision = registry.check_tool("unknown_agent", "read_file");
        assert_eq!(decision, PolicyDecision::Deny);

        // Globally denied tools
        let decision = registry.check_tool("PRIME", "call_openai");
        assert_eq!(decision, PolicyDecision::Deny);
    }

    #[test]
    fn test_permission_audit_log() {
        let mut registry = PermissionRegistry::with_defaults();

        registry.check_tool("TEKNE", "read_file");
        registry.check_tool("TEKNE", "fetch_webpage");
        registry.check_path("TEKNE", "src/main.rs");

        let log = registry.audit_log();
        assert_eq!(log.len(), 3);

        assert_eq!(log[0].decision, PolicyDecision::AllowWithAudit);
        assert_eq!(log[1].decision, PolicyDecision::Deny);
    }

    #[test]
    fn test_sat_approval_required() {
        let perm = AgentPermissions::new("KAIROS", AgentType::PAT, PermissionLevel::Full);

        // Full level requires SAT approval for dangerous ops
        assert!(perm.require_sat_approval);
        assert!(perm.requires_approval("run_in_terminal"));
        assert!(perm.requires_approval("fetch_webpage"));

        // Developer level doesn't require approval
        let dev = AgentPermissions::new("TEKNE", AgentType::PAT, PermissionLevel::Developer);
        assert!(!dev.require_sat_approval);
    }
}
