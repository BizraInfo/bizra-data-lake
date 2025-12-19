#!/usr/bin/env python3
"""
BIZRA Spawn Endpoint - /v1/system/spawn
=========================================
RESTful API for agent instantiation.

Endpoints:
    POST   /v1/system/spawn          - Spawn a new agent
    GET    /v1/system/agents         - List all agents
    GET    /v1/system/agents/{id}    - Get agent details
    DELETE /v1/system/agents/{id}    - Terminate agent
    GET    /v1/system/status         - Factory status
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from core.agent_factory import (
    get_factory,
    PAT_SPECIFICATIONS,
    SAT_SPECIFICATIONS,
    AgentType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("spawn.api")


class SpawnHandler(BaseHTTPRequestHandler):
    """HTTP handler for agent spawn API."""
    
    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _parse_body(self) -> Dict[str, Any]:
        """Parse JSON request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/v1/system/status":
            self._handle_status()
        elif path == "/v1/system/agents":
            self._handle_list_agents()
        elif path.startswith("/v1/system/agents/"):
            agent_id = path.split("/")[-1]
            self._handle_get_agent(agent_id)
        elif path == "/v1/system/specs":
            self._handle_specs()
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/v1/system/spawn":
            self._handle_spawn()
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_DELETE(self):
        """Handle DELETE requests."""
        if self.path.startswith("/v1/system/agents/"):
            agent_id = self.path.split("/")[-1]
            self._handle_terminate(agent_id)
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def _handle_spawn(self):
        """
        POST /v1/system/spawn
        
        Body:
            {
                "agent_name": "MasterReasoner",
                "session_id": null  // Optional: resume session
            }
        """
        try:
            body = self._parse_body()
            agent_name = body.get("agent_name")
            session_id = body.get("session_id")
            
            if not agent_name:
                self._send_json({"error": "agent_name required"}, 400)
                return
            
            factory = get_factory()
            
            # Determine agent type
            if agent_name in PAT_SPECIFICATIONS:
                agent = factory.spawn_pat(agent_name, session_id)
            elif agent_name in SAT_SPECIFICATIONS:
                agent = factory.spawn_sat(agent_name)
            else:
                self._send_json({
                    "error": f"Unknown agent: {agent_name}",
                    "available_pat": list(PAT_SPECIFICATIONS.keys()),
                    "available_sat": list(SAT_SPECIFICATIONS.keys())
                }, 400)
                return
            
            self._send_json({
                "success": True,
                "agent": agent.to_dict()
            }, 201)
            
        except Exception as e:
            logger.error(f"Spawn error: {e}")
            self._send_json({
                "error": str(e),
                "error_type": type(e).__name__
            }, 500)
    
    def _handle_list_agents(self):
        """GET /v1/system/agents"""
        factory = get_factory()
        agents = factory.list_agents()
        self._send_json({
            "count": len(agents),
            "agents": [a.to_dict() for a in agents]
        })
    
    def _handle_get_agent(self, agent_id: str):
        """GET /v1/system/agents/{id}"""
        factory = get_factory()
        agent = factory.get_agent(agent_id)
        
        if agent:
            self._send_json({"agent": agent.to_dict()})
        else:
            self._send_json({"error": "Agent not found"}, 404)
    
    def _handle_terminate(self, agent_id: str):
        """DELETE /v1/system/agents/{id}"""
        factory = get_factory()
        
        if factory.terminate(agent_id):
            self._send_json({"success": True, "message": f"Agent {agent_id} terminated"})
        else:
            self._send_json({"error": "Agent not found"}, 404)
    
    def _handle_status(self):
        """GET /v1/system/status"""
        factory = get_factory()
        self._send_json(factory.snapshot())
    
    def _handle_specs(self):
        """GET /v1/system/specs"""
        self._send_json({
            "pat_agents": {
                name: {
                    "model": spec["model"],
                    "backend": spec["backend"],
                    "vram_gb": spec["vram_gb"],
                    "role": spec["role"]
                }
                for name, spec in PAT_SPECIFICATIONS.items()
            },
            "sat_agents": {
                name: {
                    "type": spec["type"],
                    "vram_gb": spec["vram_gb"],
                    "role": spec["role"]
                }
                for name, spec in SAT_SPECIFICATIONS.items()
            }
        })
    
    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {args[0]}")


def run_server(host: str = "127.0.0.1", port: int = 8080):
    """Run the spawn API server."""
    server = HTTPServer((host, port), SpawnHandler)
    logger.info(f"Spawn API running on http://{host}:{port}")
    logger.info("Endpoints:")
    logger.info("  POST   /v1/system/spawn        - Spawn agent")
    logger.info("  GET    /v1/system/agents       - List agents")
    logger.info("  GET    /v1/system/agents/{id}  - Get agent")
    logger.info("  DELETE /v1/system/agents/{id}  - Terminate agent")
    logger.info("  GET    /v1/system/status       - Factory status")
    logger.info("  GET    /v1/system/specs        - Agent specifications")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BIZRA Spawn API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    args = parser.parse_args()
    run_server(args.host, args.port)
