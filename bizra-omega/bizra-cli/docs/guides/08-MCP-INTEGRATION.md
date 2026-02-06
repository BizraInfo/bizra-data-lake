# MCP Integration Guide

Connect BIZRA to external tools through the Model Context Protocol.

## Table of Contents

1. [What Is MCP?](#what-is-mcp)
2. [Architecture](#architecture)
3. [Built-in Servers](#built-in-servers)
4. [Configuration](#configuration)
5. [Using MCP Tools](#using-mcp-tools)
6. [Server Management](#server-management)
7. [Custom Servers](#custom-servers)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

---

## What Is MCP?

MCP (Model Context Protocol) is a standardized protocol for connecting AI systems to external tools and services.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                         BIZRA CLI                                   │   │
│   │                                                                     │   │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │   │
│   │   │  Guardian   │    │  Developer  │    │  Researcher │          │   │
│   │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │   │
│   │          │                  │                  │                  │   │
│   │          └──────────────────┼──────────────────┘                  │   │
│   │                             │                                      │   │
│   │                      ┌──────▼──────┐                              │   │
│   │                      │ MCP Client  │                              │   │
│   │                      └──────┬──────┘                              │   │
│   │                             │                                      │   │
│   └─────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                          │
│                          MCP Protocol                                      │
│                                 │                                          │
│   ┌─────────────────────────────┼──────────────────────────────────────┐   │
│   │                             │                                       │   │
│   │   ┌─────────────┐    ┌──────▼──────┐    ┌─────────────┐          │   │
│   │   │ Filesystem  │    │   GitHub    │    │   Memory    │          │   │
│   │   │   Server    │    │   Server    │    │   Server    │          │   │
│   │   └─────────────┘    └─────────────┘    └─────────────┘          │   │
│   │                                                                     │   │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │   │
│   │   │ Claude-Flow │    │   Brave     │    │  Custom     │          │   │
│   │   │   Server    │    │   Search    │    │  Servers    │          │   │
│   │   └─────────────┘    └─────────────┘    └─────────────┘          │   │
│   │                                                                     │   │
│   │                        MCP SERVERS                                  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Server** | External service providing tools |
| **Tool** | Specific capability (e.g., "read_file") |
| **Resource** | Data accessible via MCP |
| **Protocol** | JSON-RPC over stdin/stdout |

---

## Architecture

### MCP Message Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MCP MESSAGE FLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Agent Request                                                             │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────┐                                                      │
│   │ Tool Selection  │  Match capability to server                          │
│   └────────┬────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ Permission Check│  Guardian approval if needed                         │
│   └────────┬────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐       ┌─────────────────┐                           │
│   │ JSON-RPC Call   │ ──────│   MCP Server    │                           │
│   │ (stdin/stdout)  │       │                 │                           │
│   └────────┬────────┘       └────────┬────────┘                           │
│            │                         │                                     │
│            │<────────────────────────┘                                     │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ Response Parse  │  Validate and structure result                       │
│   └────────┬────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ FATE Validation │  Check output against gates                          │
│   └─────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Server Lifecycle

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   Stop   │     │  Init    │     │  Ready   │     │  Active  │
└────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │                │                │                │
     │  start()       │ initialize()   │  call()       │
     │───────────────>│───────────────>│───────────────>│
     │                │                │                │
     │                │                │ <───────────── │
     │                │                │    response    │
     │                │                │                │
     │ <──────────────────────────────────────────────  │
     │              shutdown()                          │
     │                                                  │
```

---

## Built-in Servers

### claude-flow

Swarm orchestration and coordination.

```yaml
claude-flow:
  command: "npx"
  args: ["-y", "@anthropic/claude-flow-mcp"]
  capabilities:
    - swarm_orchestration
    - task_coordination
    - memory_sharing
```

**Tools:**
| Tool | Description |
|------|-------------|
| `swarm_init` | Initialize a swarm |
| `agent_spawn` | Spawn new agents |
| `task_orchestrate` | Coordinate tasks |
| `memory_share` | Share memory across swarm |

### filesystem

File system access.

```yaml
filesystem:
  command: "npx"
  args: ["-y", "@anthropic/filesystem-mcp"]
  env:
    ALLOWED_PATHS: "/home,/tmp,~/.bizra"
  capabilities:
    - file_read
    - file_write
    - directory_list
```

**Tools:**
| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write to file |
| `list_directory` | List directory contents |
| `search_files` | Search for files |

### github

GitHub integration.

```yaml
github:
  command: "npx"
  args: ["-y", "@anthropic/github-mcp"]
  env:
    GITHUB_TOKEN: "${GITHUB_TOKEN}"
  capabilities:
    - repo_access
    - pr_management
    - issue_tracking
```

**Tools:**
| Tool | Description |
|------|-------------|
| `create_pr` | Create pull request |
| `review_pr` | Review PR |
| `create_issue` | Create issue |
| `search_code` | Search repository |

### memory

Vector memory and semantic search.

```yaml
memory:
  command: "npx"
  args: ["-y", "@anthropic/memory-mcp"]
  env:
    MEMORY_PATH: "~/.bizra/memory"
    VECTOR_MODEL: "all-MiniLM-L6-v2"
  capabilities:
    - semantic_search
    - memory_store
    - knowledge_retrieval
```

**Tools:**
| Tool | Description |
|------|-------------|
| `store` | Store memory |
| `recall` | Recall by query |
| `forget` | Remove memory |
| `search` | Semantic search |

### brave-search

Web search.

```yaml
brave-search:
  command: "npx"
  args: ["-y", "@anthropic/brave-search-mcp"]
  env:
    BRAVE_API_KEY: "${BRAVE_API_KEY}"
  capabilities:
    - web_search
```

**Tools:**
| Tool | Description |
|------|-------------|
| `search` | Web search |
| `news` | News search |

---

## Configuration

### Configuration File

`~/.bizra/config/mcp_servers.yaml`

```yaml
mcp_servers:
  # Server definitions
  servers:
    claude-flow:
      command: "npx"
      args: ["-y", "@anthropic/claude-flow-mcp"]
      env:
        CLAUDE_FLOW_MODE: "coordinator"
      capabilities:
        - swarm_orchestration
        - task_coordination
      auto_start: true
      restart_on_failure: true
      timeout: 30

    filesystem:
      command: "npx"
      args: ["-y", "@anthropic/filesystem-mcp"]
      env:
        ALLOWED_PATHS: "/home,/tmp,~/.bizra"
      capabilities:
        - file_read
        - file_write
      auto_start: true

    github:
      command: "npx"
      args: ["-y", "@anthropic/github-mcp"]
      env:
        GITHUB_TOKEN: "${GITHUB_TOKEN}"
      capabilities:
        - repo_access
        - pr_management
      auto_start: false  # Start on demand

  # Server groups
  groups:
    minimal:
      - claude-flow
      - filesystem

    development:
      - claude-flow
      - filesystem
      - github
      - memory

    research:
      - claude-flow
      - memory
      - brave-search

  # Active group
  active_group: "development"

  # Global settings
  settings:
    connection_timeout: 30
    max_retries: 3
    retry_delay: 5
    health_check_interval: 60
```

### Environment Variables

Store secrets in environment:

```bash
# ~/.bashrc or ~/.zshrc
export GITHUB_TOKEN="ghp_your_token"
export BRAVE_API_KEY="your_api_key"
```

Or in `.env` file:

```bash
# ~/.bizra/.env
GITHUB_TOKEN=ghp_your_token
BRAVE_API_KEY=your_api_key
```

---

## Using MCP Tools

### Automatic Tool Selection

BIZRA automatically routes requests to appropriate MCP tools:

```
User: "Search GitHub for PBFT implementations"

→ Detects: GitHub code search request
→ Selects: github server, search_code tool
→ Executes: Returns results
```

### Explicit Tool Use

```bash
# Via command
/mcp github search_code "PBFT implementation" --language rust

# Via agent
"Developer, search GitHub for rate limiting examples in Rust"
```

### Tool Chaining

```bash
# Research and implement
/research "Rust async patterns"
/code implement "async task queue"

# Behind the scenes:
# 1. brave-search → web search
# 2. memory → store findings
# 3. filesystem → write code
```

### Listing Available Tools

```bash
# List all MCP tools
/mcp tools

# List tools from specific server
/mcp tools github

# Show tool details
/mcp describe github.create_pr
```

---

## Server Management

### Starting Servers

```bash
# Start all servers in active group
bizra mcp start

# Start specific server
bizra mcp start github

# Start server group
bizra mcp start-group development
```

### Stopping Servers

```bash
# Stop all servers
bizra mcp stop

# Stop specific server
bizra mcp stop github
```

### Server Status

```bash
# Check all servers
bizra mcp status
```

**Output:**
```
MCP Server Status
─────────────────

RUNNING:
  ✓ claude-flow    (PID: 12345, uptime: 2h 15m)
  ✓ filesystem     (PID: 12346, uptime: 2h 15m)
  ✓ memory         (PID: 12347, uptime: 2h 15m)

STOPPED:
  ○ github         (not started)
  ○ brave-search   (not started)

Active Group: development (3/5 servers)
```

### Health Checks

```bash
# Run health check
bizra mcp health

# Check specific server
bizra mcp health github
```

### Logs

```bash
# View server logs
bizra mcp logs

# View specific server logs
bizra mcp logs github --tail 50
```

---

## Custom Servers

### Creating a Custom Server

1. **Define the server**

```javascript
// my-server.js
const { Server } = require('@modelcontextprotocol/sdk/server');

const server = new Server({
  name: 'my-custom-server',
  version: '1.0.0',
});

// Define tools
server.setRequestHandler('tools/list', async () => ({
  tools: [
    {
      name: 'my_tool',
      description: 'Does something useful',
      inputSchema: {
        type: 'object',
        properties: {
          input: { type: 'string' }
        },
        required: ['input']
      }
    }
  ]
}));

// Handle tool calls
server.setRequestHandler('tools/call', async (request) => {
  const { name, arguments: args } = request.params;

  if (name === 'my_tool') {
    const result = await doSomething(args.input);
    return { content: [{ type: 'text', text: result }] };
  }

  throw new Error(`Unknown tool: ${name}`);
});

server.connect();
```

2. **Configure in BIZRA**

```yaml
mcp_servers:
  servers:
    my-server:
      command: "node"
      args: ["/path/to/my-server.js"]
      capabilities:
        - my_capability
      auto_start: true
```

3. **Test the server**

```bash
bizra mcp test my-server
```

### Server Template

```python
# Python MCP server template
import asyncio
import json
import sys

class MCPServer:
    def __init__(self, name, version):
        self.name = name
        self.version = version
        self.tools = {}

    def add_tool(self, name, description, schema, handler):
        self.tools[name] = {
            'description': description,
            'schema': schema,
            'handler': handler
        }

    async def handle_request(self, request):
        method = request.get('method')

        if method == 'tools/list':
            return {
                'tools': [
                    {
                        'name': name,
                        'description': tool['description'],
                        'inputSchema': tool['schema']
                    }
                    for name, tool in self.tools.items()
                ]
            }

        elif method == 'tools/call':
            tool_name = request['params']['name']
            args = request['params'].get('arguments', {})

            if tool_name in self.tools:
                result = await self.tools[tool_name]['handler'](args)
                return {'content': [{'type': 'text', 'text': result}]}

        return {'error': 'Unknown method'}

    async def run(self):
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            response = await self.handle_request(request)

            response['id'] = request.get('id')
            print(json.dumps(response), flush=True)

# Usage
server = MCPServer('my-server', '1.0.0')

async def my_handler(args):
    return f"Processed: {args['input']}"

server.add_tool(
    'my_tool',
    'Does something useful',
    {'type': 'object', 'properties': {'input': {'type': 'string'}}},
    my_handler
)

asyncio.run(server.run())
```

---

## Security

### Permission Model

```yaml
mcp_servers:
  servers:
    filesystem:
      permissions:
        # Allowed paths
        allowed_paths:
          - "~/.bizra"
          - "/tmp"
          - "~/projects"

        # Denied paths
        denied_paths:
          - "~/.ssh"
          - "~/.gnupg"
          - "/etc"

        # Operations
        allowed_operations:
          - "read"
          - "list"
        denied_operations:
          - "delete"
          - "execute"
```

### Guardian Review

```yaml
mcp_servers:
  security:
    # Operations requiring Guardian approval
    guardian_required:
      - "github.create_pr"
      - "filesystem.write_file"
      - "filesystem.delete"

    # Always blocked operations
    blocked:
      - "filesystem.execute"

    # Rate limiting
    rate_limits:
      github:
        requests_per_minute: 30
      brave-search:
        requests_per_minute: 10
```

### Secrets Management

```yaml
mcp_servers:
  secrets:
    # Use environment variables
    method: "env"

    # Or use vault
    # method: "vault"
    # vault_path: "~/.bizra/vault"

    # Required secrets per server
    required:
      github: ["GITHUB_TOKEN"]
      brave-search: ["BRAVE_API_KEY"]
```

---

## Troubleshooting

### Common Issues

#### Server Won't Start

```bash
# Check if command exists
which npx

# Check logs
bizra mcp logs my-server

# Test manually
npx -y @anthropic/github-mcp
```

#### Connection Timeout

```yaml
# Increase timeout
mcp_servers:
  settings:
    connection_timeout: 60
```

#### Tool Not Found

```bash
# Refresh tool list
bizra mcp refresh

# Check server status
bizra mcp status my-server

# List available tools
bizra mcp tools my-server
```

#### Permission Denied

```bash
# Check permissions
bizra mcp permissions filesystem

# Update allowed paths
# Edit mcp_servers.yaml
```

### Diagnostic Commands

```bash
# Full diagnostic
bizra mcp diagnose

# Test specific server
bizra mcp test github

# View raw communication
bizra mcp debug github --verbose
```

---

## MCP Commands Reference

| Command | Description |
|---------|-------------|
| `/mcp status` | Show server status |
| `/mcp start [server]` | Start server(s) |
| `/mcp stop [server]` | Stop server(s) |
| `/mcp restart [server]` | Restart server(s) |
| `/mcp tools [server]` | List available tools |
| `/mcp describe <tool>` | Show tool details |
| `/mcp logs [server]` | View server logs |
| `/mcp health [server]` | Run health check |
| `/mcp refresh` | Refresh tool list |

---

## Next Steps

- [Hooks Automation](09-HOOKS-AUTOMATION.md) — Automate with hooks
- [Skills System](10-SKILLS-SYSTEM.md) — Multi-step workflows
- [Config Reference](../reference/CONFIG-REFERENCE.md) — Full configuration

---

**Extend your reach with MCP.** 🔌
