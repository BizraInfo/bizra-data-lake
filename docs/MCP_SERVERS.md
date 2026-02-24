# MCP Server Configuration

> 8 Model Context Protocol servers powering the BIZRA agentic ecosystem.

## Server Inventory

| # | Server | Transport | Command | Tools | Status |
|---|--------|-----------|---------|-------|--------|
| 1 | `context7` | stdio | `npx @upstash/context7-mcp` | Library docs | Active |
| 2 | `filesystem` | stdio | `npx @modelcontextprotocol/server-filesystem` | File ops | Active |
| 3 | `memory` | stdio | `npx @modelcontextprotocol/server-memory` | KG memory | Active |
| 4 | `sequential-thinking` | stdio | `npx @modelcontextprotocol/server-sequential-thinking` | Reasoning | Active |
| 5 | `github` | stdio | `npx @modelcontextprotocol/server-github` | GitHub API | Active |
| 6 | `brave-search` | stdio | `npx @modelcontextprotocol/server-brave-search` | Web search | Active |
| 7 | `bizra-sovereign` | stdio | `python tools/mcp/sovereign_mcp_server.py --stdio` | 10 tools | Active |
| 8 | `bizra-ecosystem` | stdio | `python tools/mcp/ecosystem_mcp_server.py --stdio` | 5 tools | Active |

## Configuration

**File**: `.mcp.json` (project root, v1.1.0)

Claude Code reads this file on startup. Community servers are fetched on-demand via `npx -y`.
Custom Python servers use `python` which resolves to `/usr/bin/python` (has `mcp` 1.26.0 + `fastmcp` 2.14.5 installed).

### Environment Variables

| Variable | Used By | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | github | GitHub Personal Access Token (needs `repo` scope for push) |
| `BRAVE_API_KEY` | brave-search | Brave Search API key |

Set in shell profile or `.env`. Referenced in `.mcp.json` via `${VAR}` syntax.

## Custom Python Servers

### bizra-sovereign (Sovereign Brain)

**File**: `tools/mcp/sovereign_mcp_server.py` (V4 Phase 46, ~48 KB)

| Tool | Description |
|------|-------------|
| `sovereign_query` | Unified search across 488 knowledge nodes (Apex + Nexus engines) |
| `sovereign_patterns` | Hub nodes, type bridges, co-occurrence patterns |
| `sovereign_communities` | Knowledge community clusters |
| `sovereign_health` | Engine status and diagnostics |
| `sovereign_stats` | Node/edge counts and engine statistics |
| `sovereign_reason` | Deep Graph-of-Thoughts reasoning (1-5 depth) |
| `sovereign_search` | FAISS vector search (102K vectors, 384-dim) |
| `sovereign_resonance` | Cognitive resonance pipeline (search + predict) |
| `sovereign_predict` | HMM cognitive state prediction |
| `mcp_health` | Server performance metrics |

### bizra-ecosystem (Ecosystem Bridge)

**File**: `tools/mcp/ecosystem_mcp_server.py` (v3.0.0, ~21 KB)

| Tool | Description |
|------|-------------|
| `ecosystem_query` | Unified query across all 6 sub-engines |
| `ecosystem_health` | Detailed health of all sub-engines |
| `check_compliance` | Verify text against BIZRA Constitution (RIBA, ZANN, IHSAN) |
| `perform_daughter_test` | "Would I be proud if my daughter saw this?" |
| `mcp_health` | Server performance metrics |

### Additional Servers (Not in .mcp.json)

| Server | File | Framework | Tools |
|--------|------|-----------|-------|
| `bizra_mcp.py` | `tools/mcp/bizra_mcp.py` | FastMCP | `query_bizra`, `get_system_health`, `mcp_health` |
| `peak_mcp_server.py` | `tools/mcp/peak_mcp_server.py` | MCP SDK | `peak_query`, `peak_verify`, `peak_status`, `peak_command` |
| `mcp_gateway.py` | `tools/mcp/mcp_gateway.py` | FastAPI | HTTP REST (`/query`, `/ingest`, `/health`) |
| `mcp_lake_bridge.py` | `tools/mcp/mcp_lake_bridge.py` | MCP SDK | Data pipeline bridge (ingest, chunks, search) |

## V3 Performance Optimizations

All custom BIZRA MCP servers include V3 optimizations (2026-02-18):

### Response Caching

- **LRU + TTL**: 256 entries, 300s TTL
- **Cacheable**: Read-only tools (`sovereign_query`, `sovereign_patterns`, `sovereign_communities`, `sovereign_health`, `sovereign_stats`, `ecosystem_query`, `ecosystem_health`, `check_compliance`, `perform_daughter_test`)
- **Not cached**: `sovereign_reason` (varied output), `query_bizra` (deep scan)

### Compact JSON Serialization

STDIO transport uses `json.dumps(result, separators=(',', ':'))` for ~30% payload reduction.

### Timeout Guards

Every tool handler wrapped with `asyncio.wait_for(..., timeout=30.0)`. Returns structured error on timeout instead of hanging.

### Health Monitoring

Every server exposes `mcp_health` returning: `uptime_seconds`, `query_count`, `error_count`, `cache_hit_rate`, `cache_size`, `avg_response_ms`.

### Performance Targets

| Metric | Before V3 | V3 Target | Method |
|--------|-----------|-----------|--------|
| Server startup | ~1.8s | <400ms | Lazy init, parallel gather |
| JSON payload | +30% (indent) | baseline | Compact separators |
| Cache hit rate | 0% | >90% | LRU+TTL ResponseCache |
| Response p95 | unmeasured | <100ms | Caching + timeouts |

## Core Integration Points

| File | Purpose |
|------|---------|
| `core/skills/mcp_bridge.py` | MCPBridge class: maps skills to MCP tools + permissions |
| `core/skills/router.py` | Imports MCPBridge for routing decisions |
| `core/sovereign/mcp_disclosure.py` | Progressive disclosure (3-layer: Index/Context/Deep) |
| `core/bridges/browser_mcp_client.py` | Custom Brave Search wrapper with mock support |
| `core/nexus/sovereign_nexus.py` | Lazy MCPBridge initialization at runtime |

## Troubleshooting

### Server won't start

1. **Check Python has MCP**: `python -c "from mcp.server import Server; print('OK')"`
2. **Check FastMCP**: `python -c "from fastmcp import FastMCP; print('OK')"`
3. **Install if missing**: `pip install mcp fastmcp` (or use venv: `source .venv-linux/bin/activate`)

### "Server disconnected" error

Usually means the Python executable can't be found or MCP SDK not installed.
Verify: `which python && python --version`

### Brain shows 0 nodes (DEGRADED)

Knowledge YAML files use relative paths. Launch from project root so they resolve correctly (488 nodes expected).

### .mcp.json parse error in /doctor

If `/doctor` shows "MCP config is not a valid JSON", check that `.mcp.json` has no embedded git metadata. Regenerate from the template in this doc if needed.

### Verify a server manually

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | timeout 35 python tools/mcp/sovereign_mcp_server.py --stdio 2>/dev/null | head -1
```

Expected: JSON response with `protocolVersion` and `serverInfo`.

## File Locations

| File | Purpose |
|------|---------|
| `.mcp.json` | MCP server configuration (Claude Code reads this) |
| `.claude/settings.local.json` | Enabled servers list + tool permissions |
| `tools/mcp/sovereign_mcp_server.py` | Sovereign Brain MCP server (V4 Phase 46) |
| `tools/mcp/ecosystem_mcp_server.py` | Ecosystem Bridge MCP server (v3.0.0) |
| `tools/mcp/bizra_mcp.py` | DDAGI OS MCP server (FastMCP, v3.0.0) |
| `tools/mcp/peak_mcp_server.py` | PEAK Masterpiece Engine (v3.0.0-SINGULARITY) |
| `tools/mcp/mcp_gateway.py` | FastAPI HTTP gateway (v1.0.0) |
| `tools/mcp/mcp_lake_bridge.py` | Data lake pipeline bridge |
| `tools/bridges/ecosystem_bridge.py` | Shared bridge (imported by MCP servers) |
| `tools/engines/sovereign_brain.py` | Shared engine (imported by MCP servers) |
