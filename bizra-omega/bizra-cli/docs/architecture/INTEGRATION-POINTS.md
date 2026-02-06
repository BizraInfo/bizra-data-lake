# Integration Points

External system integrations and extension mechanisms.

## Table of Contents

1. [Overview](#overview)
2. [LLM Backends](#llm-backends)
3. [MCP Integration](#mcp-integration)
4. [Federation Protocol](#federation-protocol)
5. [Calendar Integration](#calendar-integration)
6. [Notification Systems](#notification-systems)
7. [IDE Integration](#ide-integration)
8. [API Endpoints](#api-endpoints)
9. [Plugin Architecture](#plugin-architecture)
10. [Extension Guidelines](#extension-guidelines)

---

## Overview

BIZRA integrates with multiple external systems through standardized interfaces.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTEGRATION LANDSCAPE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            ┌───────────────┐                               │
│                            │   BIZRA CLI   │                               │
│                            │    Core       │                               │
│                            └───────┬───────┘                               │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│   ┌───────────┐            ┌───────────┐            ┌───────────┐         │
│   │    LLM    │            │    MCP    │            │Federation │         │
│   │ Backends  │            │  Servers  │            │  Network  │         │
│   └───────────┘            └───────────┘            └───────────┘         │
│         │                          │                          │            │
│         │                          │                          │            │
│   ┌─────┴─────┐            ┌───────┴───────┐          ┌──────┴──────┐    │
│   │           │            │               │          │             │    │
│   ▼           ▼            ▼               ▼          ▼             ▼    │
│ LM Studio  Ollama     Filesystem      GitHub     Peer Nodes    Resource  │
│                        Memory         Search                     Pool    │
│                        Claude-Flow                                       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      ADDITIONAL INTEGRATIONS                         │  │
│   ├─────────────┬─────────────┬─────────────┬─────────────┬────────────┤  │
│   │  Calendar   │Notifications│   IDEs      │  Webhooks   │  Plugins   │  │
│   │  (Google/   │ (Desktop/   │ (VSCode/    │  (Custom    │ (Custom    │  │
│   │   Outlook)  │  Slack)     │  JetBrains) │   Events)   │  Agents)   │  │
│   └─────────────┴─────────────┴─────────────┴─────────────┴────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration Types

| Type | Protocol | Purpose |
|------|----------|---------|
| **LLM Backends** | HTTP/OpenAI API | Language model inference |
| **MCP Servers** | JSON-RPC (stdio) | Tool access |
| **Federation** | gRPC + Gossip | Peer coordination |
| **Calendar** | CalDAV/OAuth | Schedule integration |
| **Notifications** | Various | Alerts and updates |
| **IDEs** | Language Server | Editor integration |

---

## LLM Backends

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM BACKEND INTEGRATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                       BIZRA Inference Gateway                       │   │
│   │                                                                     │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │   │
│   │   │   Request    │ →  │   Backend    │ →  │   Request    │        │   │
│   │   │   Builder    │    │   Selector   │    │   Executor   │        │   │
│   │   └──────────────┘    └──────────────┘    └──────────────┘        │   │
│   │                                                                     │   │
│   └─────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│               ┌─────────────────┼─────────────────┐                        │
│               │                 │                 │                        │
│               ▼                 ▼                 ▼                        │
│       ┌───────────────┐ ┌───────────────┐ ┌───────────────┐               │
│       │   LM Studio   │ │    Ollama     │ │   Federation  │               │
│       │   (Primary)   │ │  (Fallback)   │ │    (Pool)     │               │
│       └───────────────┘ └───────────────┘ └───────────────┘               │
│                                                                             │
│   Connection:                                                              │
│   • LM Studio: http://192.168.56.1:1234 (OpenAI-compatible API)           │
│   • Ollama: http://localhost:11434 (Ollama API)                           │
│   • Federation: gRPC to peer nodes                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# ~/.bizra/config/sovereign_profile.yaml
integrations:
  llm:
    # Primary backend (LM Studio)
    primary:
      type: "lmstudio"
      url: "http://192.168.56.1:1234"
      api_version: "v1"
      model: "auto"  # Use loaded model
      timeout: 60
      max_tokens: 4096
      temperature: 0.7

    # Fallback backend (Ollama)
    fallback:
      type: "ollama"
      url: "http://localhost:11434"
      model: "llama3"
      timeout: 120

    # Federation pool (distributed)
    federation:
      enabled: true
      min_peers: 3
      timeout: 180

    # Selection strategy
    strategy:
      primary_health_check: true
      fallback_on_error: true
      load_balance_federation: true
```

### Backend Interface

```rust
/// LLM Backend trait
pub trait LLMBackend: Send + Sync {
    /// Check if backend is available
    async fn is_available(&self) -> bool;

    /// Get completion
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// Stream completion
    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream>;

    /// Get embeddings
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// List available models
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
}

/// Completion request
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub model: Option<String>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}
```

---

## MCP Integration

### Server Management

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP SERVER MANAGEMENT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                        MCP Server Manager                           │   │
│   │                                                                     │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │   │
│   │   │   Lifecycle  │    │   Registry   │    │   Router     │        │   │
│   │   │   Manager    │    │   (Tools)    │    │              │        │   │
│   │   └──────────────┘    └──────────────┘    └──────────────┘        │   │
│   │                                                                     │   │
│   └─────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                           │
│     ┌───────────────────────────┼───────────────────────────┐              │
│     │                           │                           │              │
│     ▼                           ▼                           ▼              │
│ ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│ │ Filesystem  │         │   GitHub    │         │   Memory    │          │
│ │   Server    │         │   Server    │         │   Server    │          │
│ │ (npx)       │         │ (npx)       │         │ (npx)       │          │
│ └─────────────┘         └─────────────┘         └─────────────┘          │
│                                                                             │
│ Protocol: JSON-RPC over stdio                                              │
│ Tools: Dynamic registration                                                │
│ Permissions: Capability-based                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Registration

```rust
/// MCP tool registration
pub struct MCPTool {
    pub server: String,
    pub name: String,
    pub description: String,
    pub input_schema: JsonSchema,
    pub permissions: Vec<Permission>,
}

/// Tool invocation
pub struct ToolInvocation {
    pub tool: String,
    pub arguments: Value,
    pub context: InvocationContext,
}

impl MCPManager {
    /// Register tools from server
    async fn register_server_tools(&mut self, server: &str) -> Result<()> {
        let response = self.call_server(server, "tools/list", json!({})).await?;
        let tools: Vec<MCPTool> = serde_json::from_value(response)?;

        for tool in tools {
            self.registry.register(tool)?;
        }

        Ok(())
    }

    /// Invoke a tool
    async fn invoke_tool(&self, invocation: ToolInvocation) -> Result<Value> {
        // 1. Find server for tool
        let server = self.registry.find_server(&invocation.tool)?;

        // 2. Check permissions
        self.check_permissions(&invocation)?;

        // 3. Guardian approval if needed
        if self.requires_guardian(&invocation.tool) {
            Guardian::approve_tool_call(&invocation).await?;
        }

        // 4. Call server
        let response = self.call_server(
            &server,
            "tools/call",
            json!({
                "name": invocation.tool,
                "arguments": invocation.arguments
            })
        ).await?;

        Ok(response)
    }
}
```

---

## Federation Protocol

### Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FEDERATION NETWORK                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│         ┌──────────────────────────────────────────────────────┐           │
│         │                    GOSSIP LAYER                       │           │
│         │                                                       │           │
│         │    Node A ←──────→ Node B ←──────→ Node C            │           │
│         │       ↑               ↑               ↑               │           │
│         │       │               │               │               │           │
│         │       ↓               ↓               ↓               │           │
│         │    Node D ←──────→ Node E ←──────→ Node F            │           │
│         │                                                       │           │
│         └──────────────────────────────────────────────────────┘           │
│                                 │                                          │
│                                 ▼                                          │
│         ┌──────────────────────────────────────────────────────┐           │
│         │                  CONSENSUS LAYER                      │           │
│         │                                                       │           │
│         │   ┌─────────┐    ┌─────────┐    ┌─────────┐         │           │
│         │   │ Primary │    │ Replica │    │ Replica │         │           │
│         │   │  Node   │ ←─→│  Node   │ ←─→│  Node   │         │           │
│         │   └─────────┘    └─────────┘    └─────────┘         │           │
│         │                                                       │           │
│         │   Protocol: PBFT (tolerates f = (n-1)/3 failures)    │           │
│         │                                                       │           │
│         └──────────────────────────────────────────────────────┘           │
│                                 │                                          │
│                                 ▼                                          │
│         ┌──────────────────────────────────────────────────────┐           │
│         │                  RESOURCE LAYER                       │           │
│         │                                                       │           │
│         │   Compute Pool │ Memory Pool │ Knowledge Pool        │           │
│         │                                                       │           │
│         └──────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Federation Configuration

```yaml
integrations:
  federation:
    enabled: true

    # Node identity
    node:
      id: "${NODE_ID}"
      public_key: "${NODE_PUBLIC_KEY}"

    # Network settings
    network:
      listen_address: "0.0.0.0:8545"
      bootstrap_peers:
        - "node1.bizra.network:8545"
        - "node2.bizra.network:8545"

    # Gossip protocol
    gossip:
      interval: 5
      fanout: 3
      max_peers: 50

    # Consensus
    consensus:
      protocol: "pbft"
      timeout: 30
      min_replicas: 4

    # Resource sharing
    resources:
      share_compute: true
      share_memory: false
      share_knowledge: true
```

### Federation API

```rust
/// Federation node interface
pub trait FederationNode {
    /// Join the network
    async fn join(&self, bootstrap: Vec<Peer>) -> Result<()>;

    /// Leave the network
    async fn leave(&self) -> Result<()>;

    /// Get peer list
    async fn peers(&self) -> Vec<Peer>;

    /// Request resource from pool
    async fn request_resource(&self, request: ResourceRequest) -> Result<Resource>;

    /// Contribute resource to pool
    async fn contribute_resource(&self, resource: Resource) -> Result<()>;

    /// Participate in consensus
    async fn consensus(&self, proposal: Proposal) -> Result<Decision>;
}
```

---

## Calendar Integration

### Supported Providers

| Provider | Protocol | Features |
|----------|----------|----------|
| **Google Calendar** | OAuth 2.0 + REST | Full read/write |
| **Outlook/Microsoft** | OAuth 2.0 + Graph API | Full read/write |
| **CalDAV** | CalDAV | Standard calendar |
| **ICS Files** | File | Import only |

### Configuration

```yaml
integrations:
  calendar:
    enabled: true
    provider: "google"  # google | outlook | caldav

    # Google Calendar
    google:
      credentials_path: "~/.bizra/credentials/google_calendar.json"
      calendars:
        - primary
        - "work@example.com"

    # Microsoft/Outlook
    outlook:
      tenant_id: "${AZURE_TENANT_ID}"
      client_id: "${AZURE_CLIENT_ID}"

    # CalDAV
    caldav:
      url: "https://caldav.example.com"
      username: "${CALDAV_USER}"
      password: "${CALDAV_PASS}"

    # Sync settings
    sync:
      interval: 15  # minutes
      lookahead: 7  # days
      lookback: 1   # days
```

### Calendar API

```rust
/// Calendar integration
pub trait CalendarProvider {
    /// Get events
    async fn get_events(&self, range: DateRange) -> Result<Vec<Event>>;

    /// Create event
    async fn create_event(&self, event: Event) -> Result<EventId>;

    /// Update event
    async fn update_event(&self, id: EventId, event: Event) -> Result<()>;

    /// Delete event
    async fn delete_event(&self, id: EventId) -> Result<()>;

    /// Get free/busy
    async fn get_availability(&self, range: DateRange) -> Result<Availability>;
}
```

---

## Notification Systems

### Supported Channels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NOTIFICATION CHANNELS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                     Notification Router                             │   │
│   └─────────────────────────────┬───────────────────────────────────────┘   │
│                                 │                                          │
│         ┌───────────────────────┼───────────────────────┐                 │
│         │                       │                       │                 │
│         ▼                       ▼                       ▼                 │
│   ┌───────────┐         ┌───────────┐         ┌───────────┐              │
│   │  Desktop  │         │   Email   │         │   Slack   │              │
│   │           │         │           │         │           │              │
│   │ • Toast   │         │ • SMTP    │         │ • Webhook │              │
│   │ • Badge   │         │ • SendGrid│         │ • API     │              │
│   │ • Sound   │         │           │         │           │              │
│   └───────────┘         └───────────┘         └───────────┘              │
│         │                       │                       │                 │
│         └───────────────────────┼───────────────────────┘                 │
│                                 │                                          │
│                                 ▼                                          │
│                     ┌───────────────────┐                                 │
│                     │   Webhook (Custom)│                                 │
│                     │   POST to any URL │                                 │
│                     └───────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
integrations:
  notifications:
    # Desktop notifications
    desktop:
      enabled: true
      sound: true
      position: "top-right"

    # Email notifications
    email:
      enabled: false
      provider: "smtp"
      smtp:
        host: "smtp.example.com"
        port: 587
        username: "${SMTP_USER}"
        password: "${SMTP_PASS}"
      from: "bizra@example.com"
      to: "user@example.com"

    # Slack notifications
    slack:
      enabled: false
      webhook_url: "${SLACK_WEBHOOK}"
      channel: "#bizra-alerts"

    # Custom webhook
    webhook:
      enabled: false
      url: "https://example.com/webhook"
      headers:
        Authorization: "Bearer ${WEBHOOK_TOKEN}"

    # Notification rules
    rules:
      - event: "task.completed"
        channels: [desktop]
        priority: "low"
      - event: "guardian.alert"
        channels: [desktop, slack]
        priority: "high"
      - event: "fate.failure"
        channels: [desktop, email, slack]
        priority: "critical"
```

---

## IDE Integration

### VSCode Extension

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       VSCODE INTEGRATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Features:                                                                 │
│   • Inline agent suggestions                                               │
│   • Code review overlay                                                    │
│   • Task management sidebar                                                │
│   • FATE gates status bar                                                  │
│   • Command palette integration                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  VSCode                                                             │  │
│   │  ┌────────────────────┬────────────────────────────────────────┐   │  │
│   │  │ BIZRA Sidebar      │  Editor                                │   │  │
│   │  │                    │                                        │   │  │
│   │  │ 🛡 Guardian Ready  │  fn main() {                          │   │  │
│   │  │                    │      let x = 5;                        │   │  │
│   │  │ TASKS              │      // [Suggestion: Consider...] 💡  │   │  │
│   │  │ ○ Fix auth bug     │      println!("{}", x);               │   │  │
│   │  │ ○ Add tests        │  }                                     │   │  │
│   │  │                    │                                        │   │  │
│   │  │ FATE               │                                        │   │  │
│   │  │ ████████ 0.97      │                                        │   │  │
│   │  │                    │                                        │   │  │
│   │  └────────────────────┴────────────────────────────────────────┘   │  │
│   │  ────────────────────────────────────────────────────────────────  │  │
│   │  BIZRA: Ihsān 0.97 | Adl 0.28 | Harm 0.12 | Conf 0.91 | ✓ All Pass│  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
integrations:
  ide:
    vscode:
      enabled: true
      socket_path: "/tmp/bizra-vscode.sock"
      features:
        - inline_suggestions
        - code_review
        - task_sidebar
        - fate_status_bar

    jetbrains:
      enabled: false
      port: 8765
```

---

## API Endpoints

### REST API (Optional)

```yaml
integrations:
  api:
    enabled: false
    host: "127.0.0.1"
    port: 8080
    auth:
      type: "bearer"
      token_env: "BIZRA_API_TOKEN"
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/agents` | GET | List agents |
| `/api/v1/agents/{id}` | GET | Get agent status |
| `/api/v1/tasks` | GET/POST | List/create tasks |
| `/api/v1/tasks/{id}` | GET/PUT/DELETE | Task operations |
| `/api/v1/query` | POST | Send query to agent |
| `/api/v1/fate` | GET | Get FATE gates status |
| `/api/v1/memory` | GET/POST | Memory operations |

---

## Plugin Architecture

### Plugin Types

| Type | Purpose | Example |
|------|---------|---------|
| **Agent Plugin** | Add custom agents | Specialized analyst |
| **Tool Plugin** | Add MCP tools | Custom integrations |
| **Hook Plugin** | Add event hooks | Custom automation |
| **Skill Plugin** | Add skill workflows | Domain workflows |

### Plugin Structure

```
my-plugin/
├── plugin.yaml           # Plugin manifest
├── src/
│   ├── agent.rs          # Agent implementation
│   ├── tools.rs          # Tool implementations
│   └── hooks.rs          # Hook implementations
├── config/
│   └── default.yaml      # Default configuration
└── README.md             # Documentation
```

### Plugin Manifest

```yaml
# plugin.yaml
name: "my-custom-plugin"
version: "1.0.0"
author: "Your Name"
description: "A custom BIZRA plugin"

# Components
components:
  agents:
    - name: "custom_analyst"
      entry: "src/agent.rs"
      config: "config/agent.yaml"

  tools:
    - name: "custom_tool"
      entry: "src/tools.rs"

  hooks:
    - event: "task.complete"
      entry: "src/hooks.rs"

# Dependencies
dependencies:
  bizra: ">=0.1.0"

# Permissions
permissions:
  - memory:read
  - memory:write
  - network:external
```

### Plugin Registration

```bash
# Install plugin
bizra plugin install ./my-plugin

# List plugins
bizra plugin list

# Enable/disable
bizra plugin enable my-custom-plugin
bizra plugin disable my-custom-plugin

# Uninstall
bizra plugin uninstall my-custom-plugin
```

---

## Extension Guidelines

### Best Practices

1. **Follow FATE Gates** — All extensions must pass FATE validation
2. **Guardian Integration** — Register with Guardian for approval flows
3. **Minimal Permissions** — Request only needed capabilities
4. **Graceful Degradation** — Handle failures gracefully
5. **Documentation** — Document all integration points

### Integration Checklist

```
□ Define clear API boundaries
□ Implement health checks
□ Add retry logic with backoff
□ Handle timeouts appropriately
□ Log all integration events
□ Support configuration reload
□ Provide diagnostic commands
□ Write integration tests
□ Document authentication flow
□ Register with Guardian
```

### Security Considerations

```yaml
# Integration security settings
security:
  integrations:
    # Validate all external data
    validate_input: true

    # Rate limit external calls
    rate_limiting:
      enabled: true
      default: 100/minute

    # Audit all integration calls
    audit:
      enabled: true
      log_payloads: false  # Don't log sensitive data

    # TLS requirements
    tls:
      required: true
      min_version: "1.2"
```

---

**Integrate wisely, extend infinitely.** 🔗
