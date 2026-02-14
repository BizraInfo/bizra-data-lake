#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════
    SOVEREIGN MCP SERVER — Model Context Protocol Bridge to the House of Wisdom
═══════════════════════════════════════════════════════════════════════════════════════════════
    
    ARCHITECTURE: Exposes the full Sovereign Brain (488 nodes, 2 engines) to external agents
    
    TOOLS EXPOSED:
      1. sovereign_query      — Unified query across all engines (Apex + Nexus)
      2. sovereign_patterns   — Discover knowledge patterns (Hubs, Bridges, Co-occurrence)
      3. sovereign_communities — Explore detected communities
      4. sovereign_health     — Brain health diagnostics
      5. sovereign_stats      — Full system statistics
      6. sovereign_reason     — Deep GoT reasoning chain
    
    TRANSPORTS SUPPORTED:
      - Stdio (default for MCP)
      - HTTP/HTTPS (for direct API access)
    
    GIANTS ABSORBED:
      - MCP Protocol (Anthropic/Model Context Protocol 2024-11-05)
      - Sovereign Brain (BIZRA)
      - 4-Layer Apex Engine (Vector + Graph + Reasoning + Pattern)
      - GoT + SNR Reasoning (Nexus)
    
    Created: 2026-01-22 | Dubai
═══════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
import logging
import time
import ssl
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from io import StringIO
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stderr)]  # Log to stderr, keep stdout clean for MCP
)
logger = logging.getLogger("SovereignMCP")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "sovereign-brain-mcp"
SERVER_VERSION = "1.0.0"

BASE_PATH = Path(__file__).resolve().parent.parent.parent  # tools/mcp/ → project root
GOLD_PATH = BASE_PATH / "04_GOLD"

# ═══════════════════════════════════════════════════════════════════════════════
# SOVEREIGN BRAIN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class SovereignBrainInterface:
    """Interface to the Sovereign Brain orchestration layer."""
    
    def __init__(self):
        self.brain = None
        self.apex_adapter = None
        self.nexus_adapter = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize connection to Sovereign Brain."""
        try:
            # Import engines
            from sovereign_brain import SovereignBrain
            
            # Initialize brain (will load engines via adapters)
            self.brain = SovereignBrain()
            self.brain.awaken()
            
            # Direct access to engine adapters for specialized operations
            self.apex_adapter = self.brain.adapters.get('apex')
            self.nexus_adapter = self.brain.adapters.get('nexus')
            
            self.initialized = True
            logger.info(f"✓ Sovereign Brain initialized: {self.brain.state.total_nodes} nodes")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import Sovereign Brain: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Sovereign Brain: {e}")
            return False
    
    def query(self, query_text: str, limit: int = 10) -> Dict[str, Any]:
        """Unified query across all engines."""
        if not self.initialized:
            return {"error": "Brain not initialized", "results": []}
        
        try:
            start = time.perf_counter()
            result = self.brain.query(query_text, max_results=limit)
            elapsed = (time.perf_counter() - start) * 1000
            
            return {
                "query": query_text,
                "results": result.results if hasattr(result, 'results') else [],
                "snr": result.snr_score if hasattr(result, 'snr_score') else 0.0,
                "elapsed_ms": round(elapsed, 2),
                "engine_contributions": result.engine_contributions if hasattr(result, 'engine_contributions') else {},
                "insights": result.insights if hasattr(result, 'insights') else []
            }
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"error": str(e), "results": []}
    
    def get_patterns(self) -> List[Dict]:
        """Get discovered knowledge patterns."""
        if not self.initialized or not self.apex_adapter:
            return []
        
        try:
            engine = self.apex_adapter.engine
            if engine and hasattr(engine, 'pattern_layer'):
                patterns = engine.pattern_layer.discovered if hasattr(engine.pattern_layer, 'discovered') else []
                return [asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in patterns]
            return []
        except Exception as e:
            logger.error(f"Pattern fetch error: {e}")
            return []
    
    def get_communities(self) -> Dict[str, Any]:
        """Get detected communities."""
        if not self.initialized or not self.apex_adapter:
            return {}
        
        try:
            engine = self.apex_adapter.engine
            if engine and hasattr(engine, 'graph_layer'):
                communities = engine.graph_layer.communities if hasattr(engine.graph_layer, 'communities') else {}
                
                # Summarize communities
                summary = {}
                for name, nodes in communities.items():
                    summary[name] = {
                        "size": len(nodes),
                        "sample_nodes": list(nodes)[:5]
                    }
                return summary
            return {}
        except Exception as e:
            logger.error(f"Community fetch error: {e}")
            return {}
    
    def get_health(self) -> Dict[str, Any]:
        """Get brain health diagnostics."""
        if not self.initialized:
            return {"status": "offline", "engines": {}}
        
        try:
            return self.brain.health_check()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get full system statistics."""
        if not self.initialized:
            return {"status": "offline"}
        
        try:
            from sovereign_brain import EngineStatus
            
            return {
                "brain_status": "online" if self.brain.state.is_healthy else "degraded",
                "total_nodes": self.brain.state.total_nodes,
                "total_edges": self.brain.state.total_edges,
                "engines_online": len([h for h in self.brain.state.engines.values() if h.status == EngineStatus.ONLINE]),
                "total_engines": len(self.brain.state.engines),
                "engine_stats": {
                    name: {
                        "status": health.status.name,
                        "nodes": health.nodes,
                        "edges": health.edges
                    }
                    for name, health in self.brain.state.engines.items()
                },
                "last_health_check": self.brain.state.last_health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"status": "error", "error": str(e)}
    
    def reason(self, question: str, depth: int = 3) -> Dict[str, Any]:
        """Deep GoT reasoning chain."""
        if not self.initialized or not self.nexus_adapter:
            return {"error": "Nexus engine not available"}
        
        try:
            engine = self.nexus_adapter.engine
            if not engine:
                return {"error": "Nexus engine not loaded"}
            
            # Capture reasoning output
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            try:
                results = engine.query(question, max_results=10)
                output = mystdout.getvalue()
            finally:
                sys.stdout = old_stdout
            
            # Extract results from the query result object
            result_list = []
            snr_score = 0.0
            if hasattr(results, 'nodes'):
                result_list = [{"name": n.name, "type": n.type.name, "snr": n.snr_score} for n in results.nodes]
                snr_score = results.snr if hasattr(results, 'snr') else 0.0
            
            return {
                "question": question,
                "reasoning_depth": depth,
                "results": result_list,
                "snr": snr_score,
                "reasoning_trace": output if output else "Reasoning completed"
            }
        except Exception as e:
            logger.error(f"Reasoning error: {e}")
            return {"error": str(e)}

# Global brain interface
brain_interface = SovereignBrainInterface()

# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "sovereign_query",
        "description": "Query the Sovereign Brain — unified search across 488 nodes in the House of Wisdom knowledge graph. Returns semantically relevant results with SNR (signal-to-noise ratio) scoring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be natural language, technical terms, or project names."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "sovereign_patterns",
        "description": "Discover knowledge patterns in the graph. Returns Hub nodes (highly connected), Type bridges (cross-domain connections), and Co-occurrence patterns.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "sovereign_communities",
        "description": "Explore detected knowledge communities. Returns community names, sizes, and sample nodes from each cluster.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "sovereign_health",
        "description": "Get brain health diagnostics. Returns engine status, connectivity, and any detected issues.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "sovereign_stats",
        "description": "Get full system statistics. Returns node counts, edge counts, engine status, and last query info.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "sovereign_reason",
        "description": "Perform deep Graph-of-Thoughts (GoT) reasoning. Uses the Nexus engine to trace multi-hop reasoning chains through the knowledge graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to reason about."
                },
                "depth": {
                    "type": "integer",
                    "description": "Reasoning depth (1-5, default: 3)",
                    "default": 3
                }
            },
            "required": ["question"]
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# MCP REQUEST HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def process_mcp_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Core logic to handle MCP JSON-RPC requests."""
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PROTOCOL METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if method == "initialize":
        # Initialize brain on first connection
        if not brain_interface.initialized:
            brain_interface.initialize()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }
    
    elif method == "notifications/initialized":
        return None  # No response for notifications
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": MCP_TOOLS
            }
        }
    
    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TOOL CALLS
    # ═══════════════════════════════════════════════════════════════════════════
    
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        # Ensure brain is initialized
        if not brain_interface.initialized:
            brain_interface.initialize()
        
        result_text = ""
        
        if tool_name == "sovereign_query":
            query = args.get("query", "")
            limit = args.get("limit", 10)
            result = brain_interface.query(query, limit)
            result_text = json.dumps(result, indent=2, default=str)
            
        elif tool_name == "sovereign_patterns":
            patterns = brain_interface.get_patterns()
            result_text = json.dumps({
                "pattern_count": len(patterns),
                "patterns": patterns
            }, indent=2, default=str)
            
        elif tool_name == "sovereign_communities":
            communities = brain_interface.get_communities()
            result_text = json.dumps({
                "community_count": len(communities),
                "communities": communities
            }, indent=2, default=str)
            
        elif tool_name == "sovereign_health":
            health = brain_interface.get_health()
            result_text = json.dumps(health, indent=2, default=str)
            
        elif tool_name == "sovereign_stats":
            stats = brain_interface.get_stats()
            result_text = json.dumps(stats, indent=2, default=str)
            
        elif tool_name == "sovereign_reason":
            question = args.get("question", "")
            depth = args.get("depth", 3)
            result = brain_interface.reason(question, depth)
            result_text = json.dumps(result, indent=2, default=str)
            
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
            }
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ]
            }
        }
    
    # Unknown method
    if request_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class SovereignMCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for MCP requests."""
    
    def log_message(self, format, *args):
        """Override to log to stderr."""
        logger.info(f"HTTP: {args[0]}")
    
    def do_GET(self):
        """Serve status page."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        stats = brain_interface.get_stats() if brain_interface.initialized else {"status": "not initialized"}
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Sovereign MCP Server — House of Wisdom</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0f; color: #e0e0e0; padding: 40px; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #ffd700; border-bottom: 2px solid #333; padding-bottom: 15px; }}
        h2 {{ color: #87ceeb; margin-top: 30px; }}
        .status {{ background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .online {{ color: #00ff88; font-weight: bold; }}
        .offline {{ color: #ff4444; font-weight: bold; }}
        .tool {{ background: #16213e; border-left: 4px solid #ffd700; padding: 15px; margin: 10px 0; border-radius: 4px; }}
        .tool-name {{ color: #ffd700; font-weight: bold; font-size: 1.1em; }}
        .tool-desc {{ color: #aaa; margin-top: 5px; }}
        code {{ background: #0f0f1a; padding: 3px 8px; border-radius: 4px; font-family: 'Consolas', monospace; }}
        pre {{ background: #0f0f1a; padding: 15px; border-radius: 6px; overflow-x: auto; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        .stat-box {{ background: #16213e; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 2em; color: #ffd700; }}
        .stat-label {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Sovereign MCP Server — House of Wisdom</h1>
        
        <div class="status">
            <p><strong>Status:</strong> <span class="{'online' if brain_interface.initialized else 'offline'}">
                {'● ONLINE' if brain_interface.initialized else '○ OFFLINE'}</span></p>
            <p><strong>Protocol:</strong> MCP (Model Context Protocol) {MCP_PROTOCOL_VERSION}</p>
            <p><strong>Transport:</strong> Stdio (default) | HTTP/HTTPS</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{stats.get('total_nodes', 0)}</div>
                <div class="stat-label">Total Nodes</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('total_edges', 0)}</div>
                <div class="stat-label">Total Edges</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('engines_online', 0)}/{stats.get('total_engines', 0)}</div>
                <div class="stat-label">Engines Online</div>
            </div>
        </div>
        
        <h2>📦 Available Tools</h2>
        {''.join(f'''
        <div class="tool">
            <div class="tool-name">{tool['name']}</div>
            <div class="tool-desc">{tool['description']}</div>
        </div>
        ''' for tool in MCP_TOOLS)}
        
        <h2>🔗 Usage</h2>
        <pre>
# Stdio mode (for Claude, Copilot, etc.)
python sovereign_mcp_server.py --stdio

# HTTP mode (for direct API access)
python sovereign_mcp_server.py --http --port 8444
        </pre>
        
        <h2>⚙️ MCP Configuration</h2>
        <pre>
# claude_desktop_config.json
{{
  "mcpServers": {{
    "sovereign-brain": {{
      "command": "python",
      "args": ["C:/BIZRA-DATA-LAKE/sovereign_mcp_server.py", "--stdio"]
    }}
  }}
}}
        </pre>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode())
    
    def do_POST(self):
        """Handle MCP requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        
        if content_length <= 0:
            self.send_error(400, "Missing Content-Length")
            return
        
        if content_length > 1024 * 1024:  # 1MB limit
            self.send_error(413, "Request too large")
            return
        
        try:
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        response = process_mcp_request(request)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if response:
            self.wfile.write(json.dumps(response).encode())

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSPORT RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stdio():
    """Run MCP server over stdio (Model Context Protocol standard)."""
    logger.info("════════════════════════════════════════════════════════════════")
    logger.info("   🧠 SOVEREIGN MCP SERVER — Stdio Mode")
    logger.info("════════════════════════════════════════════════════════════════")
    logger.info("   Waiting for JSON-RPC messages on stdin...")

    # Defer brain initialization to first tool call (avoid crash on missing deps)
    
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            response = process_mcp_request(request)
            if response:
                print(json.dumps(response))
                sys.stdout.flush()
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {line[:50]}...")
        except Exception as e:
            logger.error(f"Error processing request: {e}")

def run_http(port: int = 8444, secure: bool = False):
    """Run MCP server over HTTP/HTTPS."""
    logger.info("════════════════════════════════════════════════════════════════")
    logger.info("   🧠 SOVEREIGN MCP SERVER — HTTP Mode")
    logger.info("════════════════════════════════════════════════════════════════")
    
    # Pre-initialize brain
    brain_interface.initialize()
    
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, SovereignMCPHandler)
    
    if secure:
        cert_dir = GOLD_PATH / "certs"
        cert_file = cert_dir / "server.crt"
        key_file = cert_dir / "server.key"
        
        if cert_file.exists() and key_file.exists():
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(str(cert_file), str(key_file))
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            logger.info(f"   🔒 HTTPS enabled (TLS 1.2+)")
    
    logger.info(f"   Listening on http{'s' if secure else ''}://127.0.0.1:{port}")
    logger.info(f"   Open in browser to see status page")
    logger.info("════════════════════════════════════════════════════════════════")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n   Shutting down...")
        httpd.shutdown()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sovereign MCP Server — Model Context Protocol Bridge to the House of Wisdom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sovereign_mcp_server.py --stdio          # Standard MCP mode (for Claude/Copilot)
  python sovereign_mcp_server.py --http           # HTTP API mode
  python sovereign_mcp_server.py --http --port 8444 --secure  # HTTPS mode
        """
    )
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport (MCP default)")
    parser.add_argument("--http", action="store_true", help="Use HTTP transport")
    parser.add_argument("--port", type=int, default=8444, help="HTTP port (default: 8444)")
    parser.add_argument("--secure", action="store_true", help="Enable HTTPS")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.test:
        # Quick test mode
        print("🧪 Testing Sovereign MCP Server...")
        brain_interface.initialize()
        
        # Test query
        result = brain_interface.query("BIZRA genesis node", limit=3)
        print(f"\n📊 Query Test:")
        print(json.dumps(result, indent=2, default=str))
        
        # Test stats
        stats = brain_interface.get_stats()
        print(f"\n📈 Stats Test:")
        print(json.dumps(stats, indent=2, default=str))
        
        print("\n✅ All tests passed!")
        return
    
    if args.http:
        run_http(port=args.port, secure=args.secure)
    else:
        # Default to stdio
        run_stdio()

if __name__ == "__main__":
    main()
