# MCP Server Configuration

> 11 Model Context Protocol servers powering the BIZRA agentic ecosystem.

## Server Inventory

| # | Server | Transport | Command | Status |
|---|--------|-----------|---------|--------|
| 1 | `filesystem` | stdio | `npx @modelcontextprotocol/server-filesystem` | Active |
| 2 | `memory` | stdio | `npx @modelcontextprotocol/server-memory` | Active |
| 3 | `github` | stdio | `npx @modelcontextprotocol/server-github` | Active |
| 4 | `fetch` | stdio | `uvx mcp-server-fetch` | Active |
| 5 | `brave-search` | stdio | `npx @modelcontextprotocol/server-brave-search` | Active |
| 6 | `sqlite` | stdio | `uvx mcp-server-sqlite` | Active |
| 7 | `claude-flow-sqlite` | stdio | `uvx mcp-server-sqlite` | Active |
| 8 | `sequential-thinking` | stdio | `npx @modelcontextprotocol/server-sequential-thinking` | Active |
| 9 | `bizra-sovereign` | stdio | `/usr/bin/python3 sovereign_mcp_server.py --stdio` | Active |
| 10 | `bizra-ecosystem` | stdio | `/usr/bin/python3 ecosystem_mcp_server.py --stdio` | Active |
| 11 | `bizra-ddagi` | stdio | `/usr/bin/python3 bizra_mcp.py` | Active |

## Configuration File

Location: `.mcp.json` (project root)

### Community Servers (npm/uvx)

```json
{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/mnt/c/BIZRA-DATA-LAKE"]
  },
  "fetch": {
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  },
  "sqlite": {
    "command": "uvx",
    "args": ["mcp-server-sqlite", "--db-path", "/mnt/c/BIZRA-DATA-LAKE/04_GOLD/bizra.db"]
  }
}
```

### BIZRA Python Servers

All three use WSL Python with Linux paths. Claude Code runs in WSL and spawns
servers directly:

```json
{
  "bizra-sovereign": {
    "command": "/usr/bin/python3",
    "args": ["/mnt/c/BIZRA-DATA-LAKE/tools/mcp/sovereign_mcp_server.py", "--stdio"]
  },
  "bizra-ecosystem": {
    "command": "/usr/bin/python3",
    "args": ["/mnt/c/BIZRA-DATA-LAKE/tools/mcp/ecosystem_mcp_server.py", "--stdio"]
  },
  "bizra-ddagi": {
    "command": "/usr/bin/python3",
    "args": ["/mnt/c/BIZRA-DATA-LAKE/tools/mcp/bizra_mcp.py"]
  }
}
```

## Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | github | GitHub Personal Access Token |
| `BRAVE_API_KEY` | brave-search | Brave Search API key |

Set in shell profile or `.env`. Referenced in `.mcp.json` via `${VAR}` syntax.

## Server Capabilities

### bizra-sovereign (Sovereign Brain)

| Tool | Description |
|------|-------------|
| `sovereign_query` | Unified search across 488 knowledge nodes |
| `sovereign_patterns` | Hub nodes, type bridges, co-occurrence |
| `sovereign_communities` | Knowledge community clusters |
| `sovereign_health` | Engine status and diagnostics |
| `sovereign_stats` | Node/edge counts and engine status |
| `sovereign_reason` | Deep Graph-of-Thoughts reasoning (1-5 depth) |
| `mcp_health` | Server performance metrics |

### bizra-ecosystem (Ecosystem Bridge)

| Tool | Description |
|------|-------------|
| `ecosystem_query` | Unified query across all 6 sub-engines |
| `ecosystem_health` | Detailed health of all sub-engines |
| `check_compliance` | Verify text against BIZRA Constitution (RIBA, ZANN) |
| `perform_daughter_test` | "Would I be proud if my daughter saw this?" |
| `mcp_health` | Server performance metrics |

### bizra-ddagi (FastMCP)

| Tool | Description |
|------|-------------|
| `query_bizra` | Cognitive query with SNR threshold and deep scan |
| `get_system_health` | Kernel invariant verification |
| `mcp_health` | Server performance metrics |

## V3 Performance Optimizations

All three custom BIZRA MCP servers include V3 optimizations (2026-02-18):

### Response Caching

- **LRU + TTL**: 256 entries, 300s TTL
- **Cacheable tools**: Read-only tools (`sovereign_query`, `sovereign_patterns`, `sovereign_communities`, `sovereign_health`, `sovereign_stats`, `ecosystem_query`, `ecosystem_health`, `check_compliance`, `perform_daughter_test`)
- **Not cached**: `sovereign_reason` (varied output), `query_bizra` (deep scan)

### Compact JSON Serialization

All STDIO transport paths use `json.dumps(result, separators=(',', ':'))` for ~30% payload reduction. HTTP/HTML paths retain `indent=2` for readability.

### Timeout Guards

Every tool handler wrapped with `asyncio.wait_for(..., timeout=30.0)`. Returns structured error `{"error":"timeout","message":"...","elapsed_ms":...}` instead of hanging.

### Health Monitoring

Every server exposes `mcp_health` tool returning:
- `uptime_seconds`, `query_count`, `error_count`
- `cache_hit_rate`, `cache_size`, `avg_response_ms`

### Parallel Initialization

`ecosystem_bridge.py` initializes Orchestrator and SovereignBridge in parallel via `asyncio.gather()` for faster startup.

### Performance Targets

| Metric | Before V3 | V3 Target | Method |
|--------|-----------|-----------|--------|
| Server startup | ~1.8s | <400ms | Lazy init, parallel gather |
| JSON payload | +30% (indent) | baseline | Compact separators |
| Cache hit rate | 0% | >90% | LRU+TTL ResponseCache |
| Response p95 | unmeasured | <100ms | Caching + timeouts |
| Tool lookup | O(n) | O(1) <5ms | TypeScript FastToolRegistry |

## TypeScript Optimization Layer

The `src/core/mcp/` module provides a high-performance TypeScript layer:

| Module | Purpose |
|--------|---------|
| `connection-pool.ts` | Managed connections with health checks |
| `fast-tool-registry.ts` | O(1) tool lookup via Map |
| `load-balancer.ts` | Least-latency / round-robin / weighted selection |
| `multi-level-cache.ts` | L1 Map + L2 LRU caching |
| `optimized-transport.ts` | Request batching + deduplication |
| `metrics.ts` | Real-time p50/p95/p99 latency tracking |

Import: `import { MCPConnectionPool, FastToolRegistry, ... } from '@mcp/index'`

## Troubleshooting

### Server won't start

1. **Check Python path**: Python MCP servers use `/usr/bin/python3` (WSL system Python).
   Ensure the `mcp` package is installed: `pip3 install mcp` or from `pyproject.toml`.

2. **Check sys.path**: All three Python servers need `tools/bridges/`, `tools/engines/`,
   and the project root on `sys.path`. This is configured in each server file.

3. **Missing database**: The `sqlite` server requires `04_GOLD/bizra.db` to exist.
   Create it with: `sqlite3 04_GOLD/bizra.db "CREATE TABLE IF NOT EXISTS metadata(key TEXT, value TEXT)"`

4. **"Server disconnected" error**: Usually means the Python executable can't be found.
   Verify `/usr/bin/python3` exists and has the MCP SDK installed.

5. **Brain shows 0 nodes (DEGRADED)**: The knowledge YAML files use relative paths resolved
   from the server's working directory. When launched from the project root, the YAML files
   resolve correctly (488 nodes expected).

### npm packages removed

The `@modelcontextprotocol/server-fetch` and `@modelcontextprotocol/server-sqlite` npm
packages were removed. Use `uvx` alternatives instead:

```json
"fetch": { "command": "uvx", "args": ["mcp-server-fetch"] },
"sqlite": { "command": "uvx", "args": ["mcp-server-sqlite", "--db-path", "..."] }
```

### Ecosystem server flags

`bizra-ecosystem` now accepts `--stdio` (default) and `--http`. The `--stdio` flag uses
proper MCP SDK transport with content-length framing.

### Verify a server

Test from WSL:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | timeout 35 /usr/bin/python3 \
    /mnt/c/BIZRA-DATA-LAKE/tools/mcp/sovereign_mcp_server.py --stdio 2>/dev/null | head -1
```

Expected: JSON response with `protocolVersion` and `serverInfo`.

## File Locations

| File | Purpose |
|------|---------|
| `.mcp.json` | MCP server configuration (Claude Code reads this) |
| `tools/mcp/sovereign_mcp_server.py` | Sovereign Brain MCP server (v1.2.0) |
| `tools/mcp/ecosystem_mcp_server.py` | Ecosystem Bridge MCP server (v3.0.0) |
| `tools/mcp/bizra_mcp.py` | DDAGI OS MCP server (FastMCP, v3.0.0) |
| `tools/bridges/ecosystem_bridge.py` | Shared bridge (imported by MCP servers) |
| `tools/engines/sovereign_brain.py` | Shared engine (imported by MCP servers) |
| `src/core/mcp/` | TypeScript optimization layer (7 modules) |
| `04_GOLD/bizra.db` | SQLite database for the sqlite MCP server |
| `.swarm/memory.db` | SQLite database for the claude-flow-sqlite server |
