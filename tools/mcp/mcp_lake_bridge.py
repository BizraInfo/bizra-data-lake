# BIZRA Data Lake Bridge to Dual-Agentic System v1.1
# 🎯 PURPOSE: Expose the 9,961-node Data Lake Hypergraph to the PAT Team as an MCP Tool.
# ⚠️ ARCHITECTURE WARNING: This script is intended to run inside WSL (connecting to /mnt/c/BIZRA-DATA-LAKE).
#    Running it on Windows (Direct C:) will conflict with the WSL port binding (8443).

import os
import json
import argparse
import platform

from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
from io import StringIO
import ssl

# Import local graph logic
try:
    from query_graph import query_graph
    from bizra_config import GRAPH_PATH
except ImportError:
    # In stdio mode, we can't easily print errors without corrupting the stream
    # so we'll just exit or log to stderr
    sys.stderr.write("❌ Failed to import query_graph. Ensure it's in the same directory.\n")
    pass

def process_mcp_request(request):
    """Core logic to handle MCP JSON-RPC requests independently of transport."""
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "bizra-data-lake",
                    "version": "1.1.0"
                }
            }
        }
    
    elif method == "notifications/initialized":
        # Client confirming initialization
        return None  # No response needed for notifications

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "knowledge_retrieve",
                        "description": "Query the BIZRA Data Lake Hypergraph (9,961 nodes) for technical context, project history, and architectural decisions.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The technical topic or question to search for in the graph."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        }
    
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "knowledge_retrieve":
            query_text = args.get("query", "")
            
            # Intercept stdout to capture query_graph's print statements
            # This is crucial for both HTTP and Stdio modes
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            try:
                # Assuming query_graph prints results to stdout
                query_graph(query_text)
                results = mystdout.getvalue()
            except Exception as e:
                results = f"Error executing query: {str(e)}"
            finally:
                sys.stdout = old_stdout

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": results if results else "No results found."
                        }
                    ]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {tool_name}"}
            }
            
    elif method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }
    
    else:
        # For unknown methods that expect a response
        if request_id is not None:
             return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
        return None

class BIZRADataLakeMCP(BaseHTTPRequestHandler):
    def _set_headers(self, content_type='application/json'):
        self.send_response(200)

    @staticmethod
    def check_environment():
        """Prevent accidental execution on Windows Host which blocks WSL."""
        if platform.system() == "Windows":
            print("\n" + "!"*60)
            print("🛑 WRONG ENVIRONMENT DETECTED")
            print("   This MCP Bridge is designed to run in WSL (Ubuntu).")
            print("   Running it on Windows blocks port 8443 for the main agent.")
            print("   Please run this from the 'bizra-genesis' WSL instance.")
            print("!"*60 + "\n")
            # We allow it with a specific flag, but warn heavily
            return False
        return True

    def do_GET(self):
        """Handle GET requests - show status page in browser."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>BIZRA Data Lake MCP Bridge</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }
        .status { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 20px; margin: 20px 0; }
        .online { color: #3fb950; }
        code { background: #21262d; padding: 2px 6px; border-radius: 4px; font-family: 'Consolas', monospace; }
        pre { background: #21262d; padding: 15px; border-radius: 6px; overflow-x: auto; }
        .tool { background: #1f2937; border-left: 3px solid #58a6ff; padding: 10px 15px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 BIZRA Data Lake MCP Bridge</h1>
        <div class="status">
            <p><strong>Status:</strong> <span class="online">● ONLINE</span></p>
            <p><strong>Protocol:</strong> HTTPS (TLS) + Stdio Support</p>
            <p><strong>Port:</strong> 8443</p>
            <p><strong>Method:</strong> POST (JSON-RPC 2.0) or Stdio</p>
        </div>
        
        <h2>📦 Available Tools</h2>
        <div class="tool">
            <strong>knowledge_retrieve</strong><br>
            Query the BIZRA Data Lake Hypergraph for technical context.
        </div>
        
        <h2>🔗 Usage (Stdio)</h2>
        <pre>python mcp_lake_bridge.py --stdio</pre>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode())

    def do_POST(self):
        # SECURITY: Validate Content-Length to prevent DoS
        MAX_CONTENT_LENGTH = 1024 * 1024  # 1MB limit
        content_length = int(self.headers.get('Content-Length', 0))

        if content_length <= 0:
            self.send_error(400, "Missing Content-Length")
            return

        if content_length > MAX_CONTENT_LENGTH:
            self.send_error(413, "Request Entity Too Large")
            return

        post_data = self.rfile.read(content_length)
        try:
            request = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Use shared logic
        response = process_mcp_request(request)
        
        self._set_headers()
        if response:
            self.wfile.write(json.dumps(response).encode())

def run_stdio():
    """Run MCP server over standard input/output (Model Context Protocol)."""
    # Force stdin/stdout to binary or unbuffered where possible, 
    # but Python's print() usually works fine for line-delimited JSON.
    # We just need to ensure no other debug prints pollute stdout.
    
    sys.stderr.write("[MCP] Stdio mode started. Waiting for JSON-RPC messages...\n")
    
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
            sys.stderr.write(f"[MCP] Error: Invalid JSON received: {line[:50]}...\n")
        except Exception as e:
            sys.stderr.write(f"[MCP] Error: {str(e)}\n")

def generate_self_signed_cert(cert_dir: Path):
    """Generate self-signed SSL certificate for HTTPS."""
    cert_file = cert_dir / "server.crt"
    key_file = cert_dir / "server.key"
    
    if cert_file.exists() and key_file.exists():
        print(f"[BRIDGE] Using existing certificates in {cert_dir}")
        return str(cert_file), str(key_file)
    
    print("[BRIDGE] Generating self-signed SSL certificate...")
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import subprocess
        # Generate self-signed cert using OpenSSL (if available)
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", str(key_file), "-out", str(cert_file),
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost/O=BIZRA/C=AE"
        ], check=True, capture_output=True)
        print("[BRIDGE] ✅ SSL certificate generated successfully")
        return str(cert_file), str(key_file)
    except Exception:
        # Simplified fallback logic if certificates missing
        return None, None

def run_server(port=8443, secure=True, localhost_only=True):
    """Run MCP bridge server with optional HTTPS."""
    bind_address = '127.0.0.1' if localhost_only else '0.0.0.0'
    server_address = (bind_address, port)
    httpd = HTTPServer(server_address, BIZRADataLakeMCP)

    print(f"[BRIDGE] 🔒 SECURITY: Bound to localhost only (127.0.0.1)")

    if secure:
        cert_dir = Path(__file__).parent / "04_GOLD" / "certs"
        cert_file, key_file = generate_self_signed_cert(cert_dir)

        if cert_file and key_file:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(cert_file, key_file)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            print(f"[BRIDGE] 🔒 BIZRA Data Lake Bridge (HTTPS/TLS 1.2+) on port {port}")
        else:
            print(f"[BRIDGE] ⚠️ Certificates missing. Falling back to HTTP.")
    
    httpd.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIZRA Data Lake MCP Bridge")
    parser.add_argument("--port", type=int, default=8443, help="Port to listen on (default: 8443)")
    parser.add_argument("--insecure", action="store_true", help="Use HTTP instead of HTTPS")
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport (MCP Default)")
    parser.add_argument("--force-windows", action="store_true", help="Allow running on Windows despite warnings")
    args = parser.parse_args()

    if not args.stdio and not args.force_windows:
        BIZRADataLakeMCP.check_environment()
    
    if args.stdio:
        run_stdio()
    else:
        run_server(port=args.port, secure=not args.insecure)
