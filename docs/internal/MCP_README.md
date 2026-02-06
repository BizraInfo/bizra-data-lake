# BIZRA MCP Server Deployment Guide

The BIZRA DDAGI OS is now exposed as a **Model Context Protocol (MCP)** server.
This allows AI agents (like Claude Desktop, Windsurf, or others) to tool-call into the full BIZRA ecosystem.

## 1. Capabilities

- **`query_bizra`**: Full cognitive pipeline.
  - Checks Constitution & Daughter Test.
  - Queries `UltimateEngine` (GoT/FATE).
  - Enhances via `Orchestrator` (RAG/Web/Arxiv).
  - Synthesizes with `EcosystemBridge`.
- **`get_system_health`**: Diagnostics.
  - Verifies Kernel Invariants (`RIBA_ZERO`, `ZANN_ZERO`, `IHSAN_FLOOR`).
  - Checks status of all 6 engines.

## 2. Configuration for Claude Desktop

Add this to your `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bizra-os": {
      "command": "C:\\BIZRA-DATA-LAKE\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\BIZRA-DATA-LAKE\\bizra_mcp.py"
      ]
    }
  }
}
```

## 3. Manual Testing

You can run the server manually to check for startup errors:
`START-MCP-SERVER.bat`

Note: It communicates via Standard Input/Output (JSON-RPC), so it will appear to hang if run in a terminal. This is normal.

## 4. Requirements

- Python 3.10+
- `fastmcp`
- `bizra-data-lake` environment (dependencies installed)
