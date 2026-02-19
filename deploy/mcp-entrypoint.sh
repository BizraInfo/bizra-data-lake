#!/bin/bash
# BIZRA MCP Service Mesh — Entrypoint Router
# Routes to the correct MCP server based on MCP_SERVER env var
#
# NOTE: Requires bash (not sh/dash) for pipefail support.
# Base image python:3.12-slim-bookworm provides /bin/bash.
#
# Valid values: sovereign, ecosystem, bizra, peak, gateway, lake
set -euo pipefail

MCP_SERVER="${MCP_SERVER:-gateway}"
MCP_HTTP_PORT="${MCP_HTTP_PORT:-8080}"

# Graceful shutdown: forward SIGTERM/SIGINT to child process
_term() {
    echo "[MCP] Received shutdown signal, draining connections..."
    if [ -n "${CHILD_PID:-}" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 0
}
trap _term SIGTERM SIGINT

echo "[MCP] Starting server: ${MCP_SERVER} on port ${MCP_HTTP_PORT}"
echo "[MCP] BIZRA_ENV=${BIZRA_ENV:-development}"
echo "[MCP] IHSAN_THRESHOLD=${IHSAN_THRESHOLD:-0.95}"

case "$MCP_SERVER" in
    sovereign)
        echo "[MCP] Launching Sovereign Brain MCP (stdio -> HTTP adapter)"
        exec python -m tools.mcp.sovereign_mcp_server "$@"
        ;;
    ecosystem)
        echo "[MCP] Launching Ecosystem MCP (HTTP mode)"
        exec python -m tools.mcp.ecosystem_mcp_server --http --port "$MCP_HTTP_PORT" "$@"
        ;;
    bizra)
        echo "[MCP] Launching BIZRA DDAGI MCP (FastMCP)"
        exec python -m tools.mcp.bizra_mcp "$@"
        ;;
    peak)
        echo "[MCP] Launching Peak Masterpiece Engine MCP"
        exec python -m tools.mcp.peak_mcp_server --http --port "$MCP_HTTP_PORT" "$@"
        ;;
    gateway)
        echo "[MCP] Launching Unified MCP Gateway (FastAPI)"
        # Gateway uses background+wait pattern so the SIGTERM trap fires.
        # --timeout-graceful-shutdown ensures uvicorn drains in-flight requests.
        uvicorn tools.mcp.mcp_gateway:app \
            --host 0.0.0.0 \
            --port "$MCP_HTTP_PORT" \
            --workers 2 \
            --timeout-keep-alive 30 \
            --timeout-graceful-shutdown 25 \
            --access-log \
            "$@" &
        CHILD_PID=$!
        wait "$CHILD_PID"
        ;;
    lake)
        echo "[MCP] Launching Data Lake Bridge MCP"
        exec python -m tools.mcp.mcp_lake_bridge --http --port "$MCP_HTTP_PORT" "$@"
        ;;
    *)
        echo "[MCP] ERROR: Unknown server '${MCP_SERVER}'"
        echo "[MCP] Valid: sovereign, ecosystem, bizra, peak, gateway, lake"
        exit 1
        ;;
esac
