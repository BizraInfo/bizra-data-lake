#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║   BIZRA ECOSYSTEM MCP SERVER                                                               ║
╠════════════════════════════════════════════════════════════════════════════════════════════╣
║   Exposes the unified BIZRA Ecosystem Bridge via the Model Context Protocol (MCP).         ║
║   Allows external agents to query the entire OS, run compliance checks, and access         ║
║   health metrics.                                                                          ║
║                                                                                            ║
║   TOOLS:                                                                                   ║
║     - ecosystem_query: Unified query across all 6 engines                                  ║
║     - ecosystem_health: System diagnostics and component status                            ║
║     - check_compliance: Evaluate text against Kernel Invariants (RIBA, ZANN, IHSAN)         ║
║     - perform_daughter_test: Run the Daughter Test on any proposition                      ║
║                                                                                            ║
║   USAGE:                                                                                   ║
║     python ecosystem_mcp_server.py --stdio  (Standard Input/Output)                        ║
║     python ecosystem_mcp_server.py --http   (HTTP/JSON-RPC Server)                         ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import asyncio
import logging
import argparse
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional
from dataclasses import asdict

# Set up path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Ecosystem Bridge (Lazy Loading)
EcosystemBridge = Any
UnifiedQuery = Any
UnifiedResponse = Any
initialize_ecosystem = None
get_ecosystem = None

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | MCP | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("EcosystemMCP")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "bizra-ecosystem-mcp"
SERVER_VERSION = "2.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# ECOSYSTEM INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class EcosystemInterface:
    """Synchronous wrapper for the Async Ecosystem Bridge."""
    
    def __init__(self):
        self.bridge: Optional[EcosystemBridge] = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
    def initialize(self):
        """Initialize the ecosystem if not already done."""
        global EcosystemBridge, UnifiedQuery, UnifiedResponse, initialize_ecosystem, get_ecosystem, Constitution, DaughterTest, RIBA_ZERO, ZANN_ZERO, IHSAN_FLOOR
        
        if not self.bridge:
            log.info("Lazy importing Ecosystem Bridge...")
            try:
                from ecosystem_bridge import (
                    initialize_ecosystem as init_eco, 
                    EcosystemBridge as EcoBridge, 
                    UnifiedQuery as UQuery, 
                    UnifiedResponse as UResponse, 
                    get_ecosystem as get_eco
                )
                from ultimate_engine import (
                    RIBA_ZERO as RZ, ZANN_ZERO as ZZ, IHSAN_FLOOR as IF, 
                    Constitution as Const, DaughterTest as DT
                )
                
                # Update globals
                EcosystemBridge = EcoBridge
                UnifiedQuery = UQuery
                UnifiedResponse = UResponse
                initialize_ecosystem = init_eco
                get_ecosystem = get_eco
                Constitution = Const
                DaughterTest = DT
                RIBA_ZERO = RZ
                ZANN_ZERO = ZZ
                IHSAN_FLOOR = IF
                
            except ImportError as e:
                log.error(f"Failed to import Ecosystem: {e}")
                raise

            log.info("Initializing Ecosystem Bridge...")
            self.bridge = self.loop.run_until_complete(initialize_ecosystem())
            log.info(f"✓ Ecosystem Online: {self.bridge.node_id}")
            
    def query(self, text: str, mode: str = "standard") -> Dict[str, Any]:
        """Run a unified query."""
        self.initialize()
        
        require_const = True
        require_daughter = True
        
        if mode == "fast":
            require_daughter = False
        elif mode == "audit":
            require_const = True
            
        u_query = UnifiedQuery(
            text=text,
            require_constitution_check=require_const,
            require_daughter_test=require_daughter
        )
        
        start = time.perf_counter()
        response: UnifiedResponse = self.loop.run_until_complete(
            self.bridge.query(u_query)
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "synthesis": response.synthesis,
            "snr_score": response.snr_score,
            "ihsan_score": response.ihsan_score,
            "components_used": response.components_used,
            "constitution_check": response.constitution_check,
            "daughter_test": response.daughter_test_result,
            "latency_ms": round(elapsed_ms, 2)
        }

    def health(self) -> Dict[str, Any]:
        """Get system health."""
        self.initialize()
        health = self.bridge.get_health()
        return health.to_dict()
        
    def check_compliance(self, text: str) -> Dict[str, Any]:
        """Check specific BIZRA compliance."""
        # Using UltimateEngine components directly for granular check
        from ultimate_engine import Constitution
        
        start = time.perf_counter()
        issues = Constitution.check_for_violations(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "compliant": len(issues) == 0,
            "violation_count": len(issues),
            "violations": issues,
            "latency_ms": round(elapsed_ms, 2)
        }

    def daughter_test(self, text: str) -> Dict[str, Any]:
        """Run the Daughter Test."""
        self.initialize()
        start = time.perf_counter()
        result = DaughterTest.evaluate(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return {
            "passed": result.passed,
            "score": result.score,
            "explanation": result.explanation,
            "latency_ms": round(elapsed_ms, 2)
        }

interface = EcosystemInterface()

# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "ecosystem_query",
        "description": "Query the Unified BIZRA OS. Routes through Ultimate Engine, BIZRA Orchestrator, Apex, and Peak.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or task"},
                "mode": {"type": "string", "enum": ["standard", "fast", "audit"], "description": "Execution mode (default: standard)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "ecosystem_health",
        "description": "Get detailed health status of all 6 sub-engines.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "check_compliance",
        "description": "Verify text against the BIZRA Constitution and Kernel Invariants (RIBA, ZANN).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Content to verify"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "perform_daughter_test",
        "description": "Run the Daughter Test: 'Would I be proud if my daughter saw this result?'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Content or decision to evaluate"}
            },
            "required": ["text"]
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def handle_mcp_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a JSON-RPC request."""
    method = request.get("method")
    params = request.get("params", {})
    req_id = request.get("id")
    
    if method == "initialize":
        interface.initialize()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
            }
        }
        
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS}
        }
        
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        
        try:
            result_data = {}
            if name == "ecosystem_query":
                result_data = interface.query(args.get("query"), args.get("mode", "standard"))
            elif name == "ecosystem_health":
                result_data = interface.health()
            elif name == "check_compliance":
                result_data = interface.check_compliance(args.get("text"))
            elif name == "perform_daughter_test":
                result_data = interface.daughter_test(args.get("text"))
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result_data, indent=2, default=str)}]
                }
            }
            
        except Exception as e:
            log.error(f"Error executing tool {name}: {e}")
            return {
                "jsonrpc": "2.0", 
                "id": req_id, 
                "error": {"code": -32000, "message": str(e)}
            }
            
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class MCPHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info(f"HTTP: {args[0]}")
        
    def do_GET(self):
        """Simple status page."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        try:
            health = interface.health()
            status_color = "#0f0" if health.get('overall_health', 0) > 0.9 else "#fa0"
            
            html = f"""<!DOCTYPE html>
            <html>
            <body style="background:#080808; color:#eee; font-family:monospace; padding:2rem;">
                <h1 style="color:#d4af37; border-bottom: 1px solid #333;">BIZRA ECOSYSTEM MCP</h1>
                <div style="border:1px solid #333; padding:1rem; border-radius:4px; margin-bottom:1rem;">
                    <h3>SYSTEM HEALTH</h3>
                    <p style="color:{status_color}; font-size:1.5rem;">{health.get('overall_health', 0)*100:.0f}%</p>
                    <pre>{json.dumps(health, indent=2)}</pre>
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
        """Handle JSON-RPC."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = self.rfile.read(length)
            req = json.loads(data)
            
            resp = handle_mcp_request(req)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            if resp:
                self.wfile.write(json.dumps(resp).encode())
                
        except Exception as e:
            self.send_error(500, str(e))

def run_server(mode="stdio", port=8888):
    if mode == "http":
        interface.initialize()
        server = HTTPServer(('127.0.0.1', port), MCPHTTPHandler)
        log.info(f"Serving HTTP on http://127.0.0.1:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    else:
        # STDIO loop
        interface.initialize()
        log.info("MCP Server listening on STDIO")
        sys.stdout.flush()
        
        for line in sys.stdin:
            try:
                if not line.strip(): continue
                req = json.loads(line)
                resp = handle_mcp_request(req)
                if resp:
                    print(json.dumps(resp))
                    sys.stdout.flush()
            except json.JSONDecodeError:
                log.error("Invalid JSON received")
            except Exception as e:
                log.error(f"Stream error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode")
    parser.add_argument("--port", type=int, default=8888, help="Port for HTTP mode")
    args = parser.parse_args()
    
    run_server(mode="http" if args.http else "stdio", port=args.port)
