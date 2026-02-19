# BIZRA MCP Server Deployment Guide

The BIZRA DDAGI OS is exposed as **Model Context Protocol (MCP)** servers.
This allows AI agents (Claude Code, Claude Desktop, Windsurf, etc.) to tool-call into the full BIZRA ecosystem.

## 1. Capabilities

### bizra-ddagi (FastMCP)

- **`query_bizra`**: Full cognitive pipeline.
  - Checks Constitution & Daughter Test.
  - Queries `UltimateEngine` (GoT/FATE).
  - Enhances via `Orchestrator` (RAG/Web/Arxiv).
  - Synthesizes with `EcosystemBridge`.
- **`get_system_health`**: Diagnostics.
  - Verifies Kernel Invariants (`RIBA_ZERO`, `ZANN_ZERO`, `IHSAN_FLOOR`).
  - Checks status of all 6 engines.
- **`mcp_health`**: Server performance metrics.

### bizra-sovereign (MCP SDK)

- 6 knowledge graph tools + `mcp_health`
- Response caching (LRU, 256 entries, 300s TTL)
- Compact JSON serialization
- 30s timeout guards on all operations

### bizra-ecosystem (MCP SDK)

- 4 ecosystem tools + `mcp_health`
- Full MCP SDK transport (content-length framing)
- Migrated from raw STDIO in V3

## 2. Configuration for Claude Code

MCP servers are configured in `.mcp.json` at the project root. All three BIZRA
servers use WSL Python:

```json
{
  "mcpServers": {
    "bizra-sovereign": {
      "type": "stdio",
      "command": "/usr/bin/python3",
      "args": ["/mnt/c/BIZRA-DATA-LAKE/tools/mcp/sovereign_mcp_server.py", "--stdio"]
    },
    "bizra-ecosystem": {
      "type": "stdio",
      "command": "/usr/bin/python3",
      "args": ["/mnt/c/BIZRA-DATA-LAKE/tools/mcp/ecosystem_mcp_server.py", "--stdio"]
    },
    "bizra-ddagi": {
      "type": "stdio",
      "command": "/usr/bin/python3",
      "args": ["/mnt/c/BIZRA-DATA-LAKE/tools/mcp/bizra_mcp.py"]
    }
  }
}
```

## 3. V3 Optimization Architecture (2026-02-18)

### Python Server Optimizations

```
                  MCP Client Request
                        |
                  [ResponseCache]  ---- cache hit ---> Return cached
                        |                              (LRU, 256 entries, 300s TTL)
                   cache miss
                        |
                  [Timeout Guard]  ---- timeout ------> Return error JSON
                  (30s asyncio.wait_for)
                        |
                  [Tool Handler]
                        |
                  [Compact JSON]   ---- separators=(',',':') ---> ~30% smaller
                        |
                  [Cache Store]    ---- if cacheable ---> Update LRU cache
                        |
                  Return Response
```

**Key changes:**
- `ecosystem_mcp_server.py`: Migrated from raw `for line in sys.stdin` to MCP SDK `stdio_server()` with content-length framing
- `ecosystem_bridge.py`: Parallel initialization via `asyncio.gather()` instead of sequential `await`
- `sovereign_mcp_server.py` / `mcp_lake_bridge.py`: `contextlib.redirect_stdout()` instead of raw `sys.stdout` reassignment
- All servers: `mcp_health` tool for runtime metrics

### TypeScript Optimization Layer

```
src/core/mcp/
├── index.ts                    # Barrel exports
├── connection-pool.ts          # Managed connections + health checks
├── fast-tool-registry.ts       # O(1) Map-based tool lookup
├── load-balancer.ts            # Least-latency / round-robin / weighted
├── multi-level-cache.ts        # L1 Map + L2 LRU with TTL
├── optimized-transport.ts      # Request batching + deduplication
├── metrics.ts                  # Real-time p50/p95/p99 tracking
└── __tests__/
    └── connection-pool.test.ts # Unit tests for all modules
```

**Integration points:**
- Connection pool uses `GracefulDegradation.executeWithFallback()` pattern from `federation/graceful-degradation.ts`
- Load balancer follows `ModelRouter.route()` pattern from `sovereign/model-router.ts`
- Metrics quality score aligned with `IHSAN_THRESHOLD` (0.95) from `sovereign/capability-card.ts`

### Performance Targets

| Metric | Before V3 | V3 Target |
|--------|-----------|-----------|
| Server startup | ~1.8s | <400ms |
| Tool lookup | O(n) linear | O(1) <5ms |
| Cache hit rate | 0% (no cache) | >90% |
| Response p95 | unmeasured | <100ms |
| JSON payload | +30% (indent) | baseline (compact) |

## 4. Manual Testing

### Verify Python servers

```bash
# Sovereign
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | timeout 35 /usr/bin/python3 /mnt/c/BIZRA-DATA-LAKE/tools/mcp/sovereign_mcp_server.py --stdio 2>/dev/null | head -1

# Ecosystem
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | timeout 35 /usr/bin/python3 /mnt/c/BIZRA-DATA-LAKE/tools/mcp/ecosystem_mcp_server.py --stdio 2>/dev/null | head -1
```

### Run Python tests

```bash
cd /mnt/c/BIZRA-DATA-LAKE
pytest tests/core/mcp/test_mcp_optimization.py -v
```

### Run TypeScript tests

```bash
cd /mnt/c/BIZRA-DATA-LAKE/src
npm run typecheck
```

## 5. Requirements

- Python 3.10+ with `mcp` and `fastmcp` packages
- Node.js 20+ for TypeScript layer
- `@modelcontextprotocol/sdk` npm package
