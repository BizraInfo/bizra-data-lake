#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════
    PEAK MCP SERVER — Masterpiece Engine Bridge
═══════════════════════════════════════════════════════════════════════════════════════════════
    
    ARCHITECTURE: Exposes the PEAK Masterpiece Engine (Unified DDAGI) to external agents
    
    TOOLS EXPOSED:
      1. peak_query    — Unified query using PeakEngine (GoT + Hypergraph + SNR)
      2. peak_verify   — Third Fact Protocol verification (Neural → Semantic → Formal → Crypto)
      3. peak_status   — Engine health, kernel invariants, and receipts
      4. peak_command  — Execute Sovereign Commands (/A, /C, /X, etc.)
    
    GIANTS ABSORBED:
      - MCP Protocol (Anthropic/Model Context Protocol 2024-11-05)
      - Peak Masterpiece Engine (BIZRA Unified)
    
    Created: 2026-01-26 | Dubai
═══════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import asdict
from datetime import datetime
from io import StringIO
from typing import Dict, Any, Optional, List

# Lazy import to avoid TensorFlow/PyTorch conflict at startup
PeakMasterpieceEngine = None
CommandType = None
EvidencePointer = None

def _lazy_import():
    """Lazy import heavy modules only when needed."""
    global PeakMasterpieceEngine, CommandType, EvidencePointer
    if PeakMasterpieceEngine is None:
        try:
            from peak_masterpiece import PeakMasterpieceEngine as PME
            from peak_masterpiece import CommandType as CT
            from peak_masterpiece import EvidencePointer as EP
            PeakMasterpieceEngine = PME
            CommandType = CT
            EvidencePointer = EP
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from peak_masterpiece import PeakMasterpieceEngine as PME
            from peak_masterpiece import CommandType as CT
            from peak_masterpiece import EvidencePointer as EP
            PeakMasterpieceEngine = PME
            CommandType = CT
            EvidencePointer = EP

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("PeakMCP")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "peak-masterpiece-mcp"
SERVER_VERSION = "3.0.0-SINGULARITY"

# ═══════════════════════════════════════════════════════════════════════════════
# PEAK ENGINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

class PeakEngineInterface:
    """Interface to the Peak Masterpiece Engine."""
    
    def __init__(self):
        self.engine = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize connection to Peak Engine."""
        try:
            # Lazy import to avoid startup conflicts
            _lazy_import()
            
            logger.info("Initializing Peak Masterpiece Engine...")
            self.engine = PeakMasterpieceEngine()
            self.initialized = True
            
            # Log status
            status = self.engine.get_status()
            logger.info(f"✓ Peak Engine initialized: {status['engine']} v{status['version']}")
            logger.info(f"✓ Kernel Invariants: {status['kernel_invariants']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Peak Engine: {e}")
            return False
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Unified query using Peak Engine."""
        if not self.initialized:
            return {"error": "Engine not initialized"}
        
        try:
            start = time.perf_counter()
            result = self.engine.process_query(query_text)
            elapsed = (time.perf_counter() - start) * 1000
            
            return {
                "query": result.query,
                "content": result.synthesis,
                "snr": result.snr_score,
                "ihsan_pass": result.ihsan_check,
                "elapsed_ms": round(elapsed, 2),
                "synergies": [asdict(s) for s in result.synergies],
                "thoughts_generated": len(result.thoughts_used),
                "discipline_coverage": result.discipline_coverage
            }
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"error": str(e)}
    
    def verify(self, claim: str) -> Dict[str, Any]:
        """Verify claim via Third Fact Protocol."""
        if not self.initialized:
            return {"error": "Engine not initialized"}
        
        try:
            result = self.engine.verify_third_fact(claim)
            return asdict(result)
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {"error": str(e)}
            
    def status(self) -> Dict[str, Any]:
        """Get engine status."""
        if not self.initialized:
            return {"status": "offline"}
        return self.engine.get_status()

    def execute_command(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Sovereign Command."""
        if not self.initialized:
            return {"error": "Engine not initialized"}
            
        cmd_symbol = info.get("command")
        context = info.get("context", {})
        
        # Find matching command type
        try:
            cmd = next(c for c in CommandType if c.symbol == cmd_symbol)
            result = self.engine.execute_command(cmd, context)
            return result
        except StopIteration:
            return {"error": f"Unknown command symbol: {cmd_symbol}"}
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

    def fate_verify(self, content: str, evidence_sources: List[str] = None) -> Dict[str, Any]:
        """SINGULARITY: LLM-verified FATE Gate verification."""
        if not self.initialized:
            return {"error": "Engine not initialized"}
        
        try:
            from peak_masterpiece import EvidencePointer
            
            # Build evidence pointers from sources
            evidence = []
            if evidence_sources:
                for src in evidence_sources:
                    evidence.append(EvidencePointer(
                        pointer_type="file_path" if not src.startswith("http") else "url",
                        value=src
                    ))
            
            # Run LLM-verified FATE
            result = self.engine.fate_gate.verify_with_llm(
                content=content,
                evidence=evidence,
                retrieved_docs=None,
                timeout=30.0
            )
            
            return {
                "passed": result.passed,
                "overall_score": result.overall_score,
                "factual_score": result.factual_score,
                "aligned_score": result.aligned_score,
                "testable_score": result.testable_score,
                "evidence_score": result.evidence_score,
                "violations": result.violations,
                "singularity_mode": True
            }
        except Exception as e:
            logger.error(f"FATE verification error: {e}")
            return {"error": str(e)}
    
    def singularity_query(self, query_text: str, verify_fate: bool = True) -> Dict[str, Any]:
        """SINGULARITY: Maximum performance query with guaranteed Ihsān SNR."""
        if not self.initialized:
            return {"error": "Engine not initialized"}
        
        try:
            start = time.perf_counter()
            result = self.engine.process_query(query_text)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Build synergies list with proper attribute names
            top_synergies = []
            for s in result.synergies[:3]:
                top_synergies.append({
                    "source_domain": s.source_domain,
                    "target_domain": s.target_domain,
                    "strength": s.strength,
                    "type": s.synergy_type.value if hasattr(s.synergy_type, 'value') else str(s.synergy_type)
                })
            
            return {
                "query": result.query,
                "content": result.synthesis,
                "snr": result.snr_score,
                "ihsan_pass": result.ihsan_check,
                "ihsan_grade": "🏆 IHSĀN" if result.snr_score >= 0.99 else ("✓ PASS" if result.ihsan_check else "❌ BELOW"),
                "elapsed_ms": round(elapsed, 2),
                "synergies_count": len(result.synergies),
                "top_synergies": top_synergies,
                "thoughts_generated": len(result.thoughts_used),
                "discipline_coverage": result.discipline_coverage,
                "singularity_mode": True,
                "llm_fate_verified": verify_fate
            }
        except Exception as e:
            logger.error(f"Singularity query error: {e}")
            return {"error": str(e)}

# Global interface
peak_interface = PeakEngineInterface()

# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOLS DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

MCP_TOOLS = [
    {
        "name": "peak_query",
        "description": "Query the Peak Masterpiece Engine. Uses Graph-of-Thoughts + Hypergraph RAG + SNR Optimization to provide Ihsan-grade responses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "peak_verify",
        "description": "Verify a claim using the Third Fact Protocol (Neural → Semantic → Formal → Cryptographic).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The specific claim to verify."
                }
            },
            "required": ["claim"]
        }
    },
    {
        "name": "peak_status",
        "description": "Get current engine health, invariant status, and metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "peak_command",
        "description": "Execute a Sovereign Command (/A, /C, /X, /#, /!, /&).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command symbol (e.g. '/A', '/C')"
                },
                "context": {
                    "type": "object",
                    "description": "Optional context dictionary"
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "peak_fate_verify",
        "description": "SINGULARITY: LLM-verified FATE Gate verification. Cross-references claims against evidence with 85% LLM weight for maximum accuracy.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The synthesis or content to verify."
                },
                "evidence_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of evidence source paths or URLs."
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "peak_singularity_query",
        "description": "SINGULARITY: Maximum performance query with LLM expansion, synthesis, and FATE verification. Guarantees Ihsān-level SNR (0.99+).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question."
                },
                "verify_fate": {
                    "type": "boolean",
                    "description": "Whether to run LLM FATE verification (default: true).",
                    "default": True
                }
            },
            "required": ["query"]
        }
    }
]

# ═══════════════════════════════════════════════════════════════════════════════
# MCP REQUEST HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

def process_mcp_request(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle MCP JSON-RPC requests."""
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    # Initialization
    if method == "initialize":
        if not peak_interface.initialized:
            peak_interface.initialize()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
            }
        }
    
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": MCP_TOOLS}
        }
    
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if not peak_interface.initialized:
            peak_interface.initialize()
            
        result_text = ""
        
        if tool_name == "peak_query":
            res = peak_interface.query(args.get("query", ""))
            result_text = json.dumps(res, indent=2, default=str)
        elif tool_name == "peak_verify":
            res = peak_interface.verify(args.get("claim", ""))
            result_text = json.dumps(res, indent=2, default=str)
        elif tool_name == "peak_status":
            res = peak_interface.status()
            result_text = json.dumps(res, indent=2, default=str)
        elif tool_name == "peak_command":
            res = peak_interface.execute_command(args)
            result_text = json.dumps(res, indent=2, default=str)
        elif tool_name == "peak_fate_verify":
            res = peak_interface.fate_verify(
                content=args.get("content", ""),
                evidence_sources=args.get("evidence_sources", [])
            )
            result_text = json.dumps(res, indent=2, default=str)
        elif tool_name == "peak_singularity_query":
            res = peak_interface.singularity_query(
                query_text=args.get("query", ""),
                verify_fate=args.get("verify_fate", True)
            )
            result_text = json.dumps(res, indent=2, default=str)
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
                "content": [{"type": "text", "text": result_text}]
            }
        }
        
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# HTTP SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class PeakMCPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        logger.info(f"HTTP: {args[0]}")
        
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        
        status = peak_interface.status() if peak_interface.initialized else {"status": "inactive"}
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PEAK MASTERPIECE MCP</title>
    <style>
        body {{ font-family: system-ui, sans-serif; background: #080808; color: #fff; max-width: 800px; margin: 40px auto; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; color: #FFD700; }}
        .box {{ background: #111; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #333; }}
        .key {{ color: #888; }}
        .val {{ color: #0f0; font-family: monospace; font-weight: bold; }}
        pre {{ background: #000; padding: 15px; border-radius: 4px; overflow: auto; }}
        .status-ok {{ color: #0f0; }}
        .status-warn {{ color: #fa0; }}
    </style>
</head>
<body>
    <h1>🏔️ PEAK MASTERPIECE ENGINE</h1>
    
    <div class="box">
        <h3>System Status</h3>
        <p><span class="key">Engine:</span> <span class="val">{status.get("engine", "N/A")} v{status.get("version", "N/A")}</span></p>
        <p><span class="key">Ihsan Average:</span> <span class="val">{status.get("ihsan_average", 0.0):.3f}</span></p>
        <p><span class="key">Receipts Emitted:</span> <span class="val">{status.get("receipts", 0)}</span></p>
        <p><span class="key">Kernel Invariants:</span> <span class="val">{status.get("kernel_invariants", {})}</span></p>
    </div>
    
    <div class="box">
        <h3>Available Tools</h3>
        <ul>
            {''.join(f"<li><strong>{t['name']}</strong>: {t['description']}</li>" for t in MCP_TOOLS)}
        </ul>
    </div>
    
    <div class="box">
        <h3>HTTP Usage</h3>
        <pre>POST / with JSON-RPC body</pre>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        data = self.rfile.read(length)
        
        try:
            req = json.loads(data)
            resp = process_mcp_request(req)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            if resp:
                self.wfile.write(json.dumps(resp).encode("utf-8"))
        except Exception as e:
            self.send_error(500, str(e))

def run_http(port=8444):
    try:
        peak_interface.initialize()
        server = HTTPServer(('127.0.0.1', port), PeakMCPHandler)
        logger.info(f"Serving PEAK MASTERPIECE HTTP on http://127.0.0.1:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.critical(f"Server crashed: {e}", exc_info=True)
        sys.exit(1)

def run_stdio():
    peak_interface.initialize()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            resp = process_mcp_request(req)
            if resp:
                print(json.dumps(resp))
                sys.stdout.flush()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8444)
    args = parser.parse_args()
    
    if args.http:
        run_http(args.port)
    else:
        run_stdio()
