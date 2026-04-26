#!/bin/bash
set -euo pipefail

echo "🚀 BIZRA MCP setup"
echo
echo "This workspace now uses checked-in MCP configs instead of force-adding ad hoc servers."
echo
echo "Claude Code:"
echo "  - reads .mcp.json from the repo root"
echo "  - should be restarted or reloaded after config changes"
echo
echo "Codex:"
echo "  - reads .codex/config.toml for trusted projects"
echo
echo "No MCP servers were added by this script."
echo "If you still see a stale claude-flow entry, remove it from your user-level MCP config or re-run your MCP reload flow."
