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
| 9 | `bizra-sovereign` | stdio | `Python313\python.exe sovereign_mcp_server.py` | Active |
| 10 | `bizra-ecosystem` | stdio | `Python313\python.exe ecosystem_mcp_server.py` | Active |
| 11 | `bizra-ddagi` | stdio | `Python313\python.exe bizra_mcp.py` | Active |

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

All three use Windows Python 3.13 with Windows-native paths. Claude Code runs from Windows
and translates `/mnt/c/` to `C:\`, so the MCP servers must use Windows Python and paths:

```json
{
  "bizra-sovereign": {
    "command": "C:\\Program Files\\Python313\\python.exe",
    "args": ["C:\\BIZRA-DATA-LAKE\\tools\\mcp\\sovereign_mcp_server.py", "--stdio"]
  },
  "bizra-ecosystem": {
    "command": "C:\\Program Files\\Python313\\python.exe",
    "args": ["C:\\BIZRA-DATA-LAKE\\tools\\mcp\\ecosystem_mcp_server.py"]
  },
  "bizra-ddagi": {
    "command": "C:\\Program Files\\Python313\\python.exe",
    "args": ["C:\\BIZRA-DATA-LAKE\\tools\\mcp\\bizra_mcp.py"]
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

### bizra-ecosystem (Ecosystem Bridge)

| Tool | Description |
|------|-------------|
| `ecosystem_query` | Unified query across all 6 sub-engines |
| `ecosystem_health` | Detailed health of all sub-engines |
| `check_compliance` | Verify text against BIZRA Constitution (RIBA, ZANN) |
| `perform_daughter_test` | "Would I be proud if my daughter saw this?" |

### bizra-ddagi (FastMCP)

| Tool | Description |
|------|-------------|
| `query_bizra` | Cognitive query with SNR threshold and deep scan |
| `get_system_health` | Kernel invariant verification |

## Troubleshooting

### Server won't start

1. **Check Python path**: Python MCP servers must use **Windows Python** (`C:\Program Files\Python313\python.exe`),
   not a WSL/Linux venv. Claude Code spawns MCP servers from the Windows side, so Linux ELF
   binaries (`.venv-linux/bin/python`) cannot be executed. Windows Python 3.13 has `fastmcp`,
   `networkx`, and all required dependencies installed via `pip`.

2. **Check sys.path**: All three Python servers need `tools/bridges/`, `tools/engines/`,
   and the project root on `sys.path`. This is configured in each server file.

3. **Missing database**: The `sqlite` server requires `04_GOLD/bizra.db` to exist.
   Create it with: `sqlite3 04_GOLD/bizra.db "CREATE TABLE IF NOT EXISTS metadata(key TEXT, value TEXT)"`

4. **"Server disconnected" error**: Usually means Claude Code tried to launch a Linux Python
   binary from Windows. Verify `.mcp.json` uses Windows paths (`C:\\Program Files\\...`).

5. **Brain shows 0 nodes (DEGRADED)**: The knowledge YAML files use relative paths resolved
   from the server's working directory. When launched from Windows, the CWD is the project
   root (`C:\BIZRA-DATA-LAKE`), and the YAML files resolve correctly (488 nodes expected).

### npm packages removed

The `@modelcontextprotocol/server-fetch` and `@modelcontextprotocol/server-sqlite` npm
packages were removed. Use `uvx` alternatives instead:

```json
"fetch": { "command": "uvx", "args": ["mcp-server-fetch"] },
"sqlite": { "command": "uvx", "args": ["mcp-server-sqlite", "--db-path", "..."] }
```

### Ecosystem server flags

`bizra-ecosystem` does NOT accept `--stdio`. It defaults to stdio mode when run without
the `--http` flag. Do not add `--stdio` to its args.

### Verify a server

Test from WSL with the Linux venv (for development/debugging):

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | timeout 35 /mnt/c/BIZRA-DATA-LAKE/.venv-linux/bin/python \
    /mnt/c/BIZRA-DATA-LAKE/tools/mcp/ecosystem_mcp_server.py 2>/dev/null | head -1
```

Test from Windows (how Claude Code actually launches servers):

```cmd
echo {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}} | ^
  "C:\Program Files\Python313\python.exe" ^
    "C:\BIZRA-DATA-LAKE\tools\mcp\sovereign_mcp_server.py" --stdio 2>NUL
```

Expected: JSON response with `protocolVersion` and `serverInfo`.

## File Locations

| File | Purpose |
|------|---------|
| `.mcp.json` | MCP server configuration (Claude Code reads this) |
| `tools/mcp/sovereign_mcp_server.py` | Sovereign Brain MCP server |
| `tools/mcp/ecosystem_mcp_server.py` | Ecosystem Bridge MCP server |
| `tools/mcp/bizra_mcp.py` | DDAGI OS MCP server (FastMCP) |
| `tools/bridges/ecosystem_bridge.py` | Shared bridge (imported by MCP servers) |
| `tools/engines/sovereign_brain.py` | Shared engine (imported by MCP servers) |
| `04_GOLD/bizra.db` | SQLite database for the sqlite MCP server |
| `.swarm/memory.db` | SQLite database for the claude-flow-sqlite server |
