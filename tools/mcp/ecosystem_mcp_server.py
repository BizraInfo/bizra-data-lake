#!/usr/bin/env python3
"""
BIZRA ECOSYSTEM MCP SERVER

Exposes the unified BIZRA Ecosystem Bridge via the Model Context Protocol (MCP).
Allows external agents to query the entire OS, run compliance checks, and access
health metrics.

TOOLS:
  - ecosystem_query: Unified query across all 6 engines
  - ecosystem_health: System diagnostics and component status
  - check_compliance: Evaluate text against Kernel Invariants (RIBA, ZANN, IHSAN)
  - perform_daughter_test: Run the Daughter Test on any proposition
  - mcp_health: Server performance metrics

TRANSPORT:
  --stdio  MCP SDK stdio transport (default)
  --http   HTTP/JSON-RPC Server (alternative)

Migrated to MCP SDK: 2026-02-18
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

# Set up path to ensure imports work
_mcp_dir = os.path.dirname(os.path.abspath(__file__))
_tools_dir = os.path.dirname(_mcp_dir)  # tools/
_project_root = os.path.dirname(_tools_dir)  # BIZRA-DATA-LAKE/
sys.path.insert(0, _mcp_dir)
sys.path.insert(0, os.path.join(_tools_dir, "bridges"))
sys.path.insert(0, os.path.join(_tools_dir, "engines"))
sys.path.insert(0, _project_root)

# ===============================================================================
# LOGGING -- stderr only, stdout reserved for MCP protocol
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | MCP | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("EcosystemMCP")

# ===============================================================================
# CONSTANTS
# ===============================================================================

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "bizra-ecosystem-mcp"
SERVER_VERSION = "3.0.0"
TOOL_TIMEOUT_SECONDS = 30.0

# ===============================================================================
# RESPONSE CACHE (LRU with TTL)
# ===============================================================================


class ResponseCache:
    """LRU cache with TTL for read-only MCP tool responses."""

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 300.0):
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(self, tool_name: str, arguments: dict) -> str:
        raw = json.dumps({"t": tool_name, "a": arguments}, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, tool_name: str, arguments: dict) -> Optional[Any]:
        key = self._make_key(tool_name, arguments)
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            return None
        ts, value = entry
        if (time.monotonic() - ts) > self._ttl:
            del self._cache[key]
            self.misses += 1
            return None
        self._cache.move_to_end(key)
        self.hits += 1
        return value

    def put(self, tool_name: str, arguments: dict, value: Any) -> None:
        key = self._make_key(tool_name, arguments)
        self._cache[key] = (time.monotonic(), value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


# Cacheable tools (read-only, deterministic)
CACHEABLE_TOOLS = {
    "ecosystem_query",
    "ecosystem_health",
    "check_compliance",
    "perform_daughter_test",
}

cache = ResponseCache(max_entries=256, ttl_seconds=300.0)

# ===============================================================================
# SERVER METRICS
# ===============================================================================

_server_start_time = time.monotonic()
_query_count = 0
_error_count = 0
_total_response_time = 0.0

# ===============================================================================
# ECOSYSTEM INTERFACE (async, lazy-loading)
# ===============================================================================

_bridge = None
_bridge_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

# Lazy import holders
_EcosystemBridge = None
_UnifiedQuery = None
_UnifiedResponse = None
_initialize_ecosystem = None
_Constitution = None
_DaughterTest = None


def _lazy_import():
    """Import ecosystem modules on first use."""
    global _EcosystemBridge, _UnifiedQuery, _UnifiedResponse, _initialize_ecosystem
    global _Constitution, _DaughterTest

    if _EcosystemBridge is not None:
        return

    log.info("Lazy importing Ecosystem Bridge...")
    from ecosystem_bridge import (
        EcosystemBridge,
        UnifiedQuery,
        UnifiedResponse,
        initialize_ecosystem,
    )

    _EcosystemBridge = EcosystemBridge
    _UnifiedQuery = UnifiedQuery
    _UnifiedResponse = UnifiedResponse
    _initialize_ecosystem = initialize_ecosystem

    try:
        from ultimate_engine import Constitution, DaughterTest

        _Constitution = Constitution
        _DaughterTest = DaughterTest
    except ImportError:
        log.warning("UltimateEngine not available for direct compliance checks")


async def _get_bridge():
    """Get or initialize the ecosystem bridge singleton."""
    global _bridge
    if _bridge is not None:
        return _bridge

    _lazy_import()
    log.info("Initializing Ecosystem Bridge...")
    _bridge = await _initialize_ecosystem()
    log.info(f"Ecosystem Online: {_bridge.node_id}")
    return _bridge


async def _do_query(query_text: str, mode: str = "standard") -> Dict[str, Any]:
    bridge = await _get_bridge()

    require_const = True
    require_daughter = mode != "fast"

    uq = _UnifiedQuery(
        text=query_text,
        require_constitution_check=require_const,
        require_daughter_test=require_daughter,
    )

    start = time.perf_counter()
    response = await bridge.query(uq)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "synthesis": response.synthesis,
        "snr_score": response.snr_score,
        "ihsan_score": response.ihsan_score,
        "components_used": response.components_used,
        "constitution_check": response.constitution_check,
        "daughter_test": getattr(
            response,
            "daughter_test_result",
            getattr(response, "daughter_test_check", None),
        ),
        "latency_ms": round(elapsed_ms, 2),
    }


async def _do_health() -> Dict[str, Any]:
    bridge = await _get_bridge()
    health = bridge.get_health()
    return health.to_dict()


async def _do_compliance(text: str) -> Dict[str, Any]:
    _lazy_import()
    if _Constitution is None:
        return {"error": "Constitution module not available"}

    start = time.perf_counter()
    issues = _Constitution.check_for_violations(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "compliant": len(issues) == 0,
        "violation_count": len(issues),
        "violations": issues,
        "latency_ms": round(elapsed_ms, 2),
    }


async def _do_daughter_test(text: str) -> Dict[str, Any]:
    await _get_bridge()
    _lazy_import()
    if _DaughterTest is None:
        return {"error": "DaughterTest module not available"}

    start = time.perf_counter()
    result = _DaughterTest.evaluate(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "passed": result.passed,
        "score": result.score,
        "explanation": result.explanation,
        "latency_ms": round(elapsed_ms, 2),
    }


def _do_mcp_health() -> Dict[str, Any]:
    uptime = time.monotonic() - _server_start_time
    return {
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "uptime_seconds": round(uptime, 1),
        "query_count": _query_count,
        "error_count": _error_count,
        "cache_hit_rate": round(cache.hit_rate, 4),
        "cache_size": cache.size,
        "avg_response_ms": (
            round(_total_response_time / _query_count, 2) if _query_count > 0 else 0.0
        ),
    }


# ===============================================================================
# MCP SERVER via SDK (proper stdio transport with content-length framing)
# ===============================================================================

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server(SERVER_NAME, version=SERVER_VERSION)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="ecosystem_query",
            description="Query the Unified BIZRA OS. Routes through Ultimate Engine, BIZRA Orchestrator, Apex, and Peak.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question or task"},
                    "mode": {
                        "type": "string",
                        "enum": ["standard", "fast", "audit"],
                        "description": "Execution mode (default: standard)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ecosystem_health",
            description="Get detailed health status of all 6 sub-engines.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="check_compliance",
            description="Verify text against the BIZRA Constitution and Kernel Invariants (RIBA, ZANN).",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Content to verify"}
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="perform_daughter_test",
            description="Run the Daughter Test: 'Would I be proud if my daughter saw this result?'",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Content or decision to evaluate",
                    }
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="mcp_health",
            description="Get MCP server performance metrics: uptime, query count, cache hit rate, errors, avg response time.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    global _query_count, _error_count, _total_response_time

    _query_count += 1
    start = time.perf_counter()

    # Check cache for read-only tools
    if name in CACHEABLE_TOOLS:
        cached = cache.get(name, arguments)
        if cached is not None:
            elapsed = (time.perf_counter() - start) * 1000
            _total_response_time += elapsed
            return [types.TextContent(type="text", text=cached)]

    try:
        result_data: Any = None

        if name == "ecosystem_query":
            result_data = await asyncio.wait_for(
                _do_query(
                    arguments.get("query", ""), arguments.get("mode", "standard")
                ),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
        elif name == "ecosystem_health":
            result_data = await asyncio.wait_for(
                _do_health(), timeout=TOOL_TIMEOUT_SECONDS
            )
        elif name == "check_compliance":
            result_data = await asyncio.wait_for(
                _do_compliance(arguments.get("text", "")), timeout=TOOL_TIMEOUT_SECONDS
            )
        elif name == "perform_daughter_test":
            result_data = await asyncio.wait_for(
                _do_daughter_test(arguments.get("text", "")),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
        elif name == "mcp_health":
            result_data = _do_mcp_health()
        else:
            _error_count += 1
            raise ValueError(f"Unknown tool: {name}")

        # Compact JSON for STDIO transport
        text = json.dumps(result_data, default=str, separators=(",", ":"))

        # Cache the result for read-only tools
        if name in CACHEABLE_TOOLS:
            cache.put(name, arguments, text)

        elapsed = (time.perf_counter() - start) * 1000
        _total_response_time += elapsed
        return [types.TextContent(type="text", text=text)]

    except asyncio.TimeoutError:
        _error_count += 1
        elapsed = (time.perf_counter() - start) * 1000
        _total_response_time += elapsed
        error = {
            "error": "timeout",
            "message": f"Tool '{name}' exceeded {TOOL_TIMEOUT_SECONDS}s timeout",
            "elapsed_ms": round(elapsed, 2),
        }
        return [
            types.TextContent(
                type="text", text=json.dumps(error, separators=(",", ":"))
            )
        ]
    except Exception as e:
        _error_count += 1
        elapsed = (time.perf_counter() - start) * 1000
        _total_response_time += elapsed
        log.error(f"Error executing tool {name}: {e}")
        error = {"error": str(type(e).__name__), "message": str(e)}
        return [
            types.TextContent(
                type="text", text=json.dumps(error, separators=(",", ":"))
            )
        ]


# ===============================================================================
# HTTP SERVER (alternative transport, retained for compatibility)
# ===============================================================================

# Legacy MCP tools list for HTTP handler
MCP_TOOLS = [
    {
        "name": "ecosystem_query",
        "description": "Query the Unified BIZRA OS. Routes through Ultimate Engine, BIZRA Orchestrator, Apex, and Peak.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or task"},
                "mode": {
                    "type": "string",
                    "enum": ["standard", "fast", "audit"],
                    "description": "Execution mode (default: standard)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ecosystem_health",
        "description": "Get detailed health status of all 6 sub-engines.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_compliance",
        "description": "Verify text against the BIZRA Constitution and Kernel Invariants (RIBA, ZANN).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Content to verify"}
            },
            "required": ["text"],
        },
    },
    {
        "name": "perform_daughter_test",
        "description": "Run the Daughter Test: 'Would I be proud if my daughter saw this result?'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Content or decision to evaluate",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "mcp_health",
        "description": "Get MCP server performance metrics.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


def _handle_http_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a JSON-RPC request for HTTP transport."""
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }

    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS}}

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        loop = asyncio.new_event_loop()
        try:
            # Run the async tool handler synchronously for HTTP
            result_content = loop.run_until_complete(call_tool(tool_name, args))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": c.type, "text": c.text} for c in result_content
                    ]
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }
        finally:
            loop.close()

    return None


class MCPHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info(f"HTTP: {args[0]}")

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        try:
            health_data = _do_mcp_health()
            "#0f0" if health_data.get("error_count", 0) == 0 else "#fa0"

            html = f"""<!DOCTYPE html>
            <html>
            <body style="background:#080808; color:#eee; font-family:monospace; padding:2rem;">
                <h1 style="color:#d4af37; border-bottom: 1px solid #333;">BIZRA ECOSYSTEM MCP v{SERVER_VERSION}</h1>
                <div style="border:1px solid #333; padding:1rem; border-radius:4px; margin-bottom:1rem;">
                    <h3>SERVER HEALTH</h3>
                    <pre>{json.dumps(health_data, indent=2)}</pre>
                </div>
                <div style="border:1px solid #333; padding:1rem; border-radius:4px;">
                    <h3>CAPABILITIES</h3>
                    <ul>
                        {''.join(f"<li><strong>{t['name']}</strong>: {t['description']}</li>" for t in MCP_TOOLS)}
                    </ul>
                </div>
            </body>
            </html>"""
            self.wfile.write(html.encode())
        except Exception as e:
            self.wfile.write(f"Server Error: {str(e)}".encode())

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            req = json.loads(data)
            resp = _handle_http_request(req)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if resp:
                self.wfile.write(json.dumps(resp).encode())
        except Exception as e:
            self.send_error(500, str(e))


# ===============================================================================
# ENTRY POINTS
# ===============================================================================


async def main_stdio():
    """Run via MCP SDK stdio transport."""
    log.info(
        f"Ecosystem MCP Server v{SERVER_VERSION} starting (SDK stdio transport)..."
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main_http(port: int = 8888):
    """Run via HTTP transport."""
    # Bind to 0.0.0.0 in container environments, 127.0.0.1 for local dev
    bind_addr = "0.0.0.0" if os.getenv("BIZRA_ENV") == "production" else "127.0.0.1"
    httpd = HTTPServer((bind_addr, port), MCPHTTPHandler)
    log.info(f"Serving HTTP on http://{bind_addr}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode")
    parser.add_argument(
        "--stdio", action="store_true", help="Run in STDIO mode (default)"
    )
    parser.add_argument("--port", type=int, default=8888, help="Port for HTTP mode")
    args = parser.parse_args()

    if args.http:
        main_http(port=args.port)
    else:
        asyncio.run(main_stdio())
