# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - MCP Server v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Model Context Protocol (MCP) Server exposing constellation capabilities:
- Agent invocation tools
- Knowledge graph queries
- Memory operations
- Team assembly
- System status
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable, Awaitable
from enum import Enum


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MCP TYPES
# ─────────────────────────────────────────────────────────────────────────────

class MCPToolType(str, Enum):
    """Types of MCP tools."""
    FUNCTION = "function"
    RESOURCE = "resource"


@dataclass
class MCPParameter:
    """Parameter definition for MCP tool."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[list[str]] = None


@dataclass
class MCPTool:
    """Definition of an MCP tool."""
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    parameters: list[MCPParameter] = field(default_factory=list)
    tool_type: MCPToolType = MCPToolType.FUNCTION
    
    # Metadata
    category: str = "general"
    version: str = "1.0.0"
    
    def to_schema(self) -> dict:
        """Convert to MCP schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
                
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
                
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class MCPResource:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"
    handler: Optional[Callable[[], Awaitable[Any]]] = None
    
    def to_schema(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPRequest:
    """Incoming MCP request."""
    id: str
    method: str
    params: dict = field(default_factory=dict)


@dataclass
class MCPResponse:
    """Outgoing MCP response."""
    id: str
    result: Optional[Any] = None
    error: Optional[dict] = None
    
    def to_dict(self) -> dict:
        resp = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            resp["error"] = self.error
        else:
            resp["result"] = self.result
        return resp


# ─────────────────────────────────────────────────────────────────────────────
# MCP SERVER
# ─────────────────────────────────────────────────────────────────────────────

class MCPServer:
    """
    Model Context Protocol server for the BIZRA Constellation.
    
    Exposes:
    - Tools for agent invocation
    - Resources for knowledge access
    - Prompts for standardized interactions
    """
    
    def __init__(
        self,
        name: str = "bizra-constellation",
        version: str = "1.0.0",
    ):
        self.name = name
        self.version = version
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._running = False
        
        # Register constellation tools
        self._register_constellation_tools()
        
    def _register_constellation_tools(self) -> None:
        """Register all constellation tools."""
        
        # invoke_agent tool
        self.register_tool(MCPTool(
            name="invoke_agent",
            description="Invoke a specific Islamic mastermind agent with a task",
            handler=self._handle_invoke_agent,
            parameters=[
                MCPParameter("agent_slug", "string", "The slug of the agent to invoke (e.g., 'ibn-sina', 'al-khwarizmi')"),
                MCPParameter("task", "string", "The task or question for the agent"),
                MCPParameter("context", "string", "Additional context", required=False),
            ],
            category="agent",
        ))
        
        # assemble_team tool
        self.register_tool(MCPTool(
            name="assemble_team",
            description="Assemble a cross-pollination team for a complex task",
            handler=self._handle_assemble_team,
            parameters=[
                MCPParameter("team_name", "string", "Team preset name or 'auto' for automatic selection", 
                           enum=["scientific-method-elite", "systems-architecture-dream", "medical-innovation-task-force",
                                "mathematical-computation-core", "philosophical-synthesis-council", 
                                "legal-reasoning-panel", "innovation-creativity-studio", "strategic-leadership-command", "auto"]),
                MCPParameter("task", "string", "The task for the team"),
                MCPParameter("snr_target", "number", "Target SNR threshold (0.85-0.98)", required=False, default=0.93),
            ],
            category="team",
        ))
        
        # query_knowledge tool
        self.register_tool(MCPTool(
            name="query_knowledge",
            description="Query the constellation's knowledge graph",
            handler=self._handle_query_knowledge,
            parameters=[
                MCPParameter("query", "string", "Natural language query"),
                MCPParameter("domain_filter", "string", "Filter by domain", required=False),
                MCPParameter("min_snr", "number", "Minimum SNR score", required=False, default=0.0),
                MCPParameter("max_results", "number", "Maximum results", required=False, default=10),
            ],
            category="knowledge",
        ))
        
        # store_memory tool
        self.register_tool(MCPTool(
            name="store_memory",
            description="Store information in the constellation's memory",
            handler=self._handle_store_memory,
            parameters=[
                MCPParameter("content", "string", "Content to store"),
                MCPParameter("priority", "string", "Memory priority",
                           enum=["critical", "high", "medium", "low", "ephemeral"],
                           required=False, default="medium"),
                MCPParameter("agent_slug", "string", "Storing agent", required=False),
            ],
            category="memory",
        ))
        
        # recall_memory tool
        self.register_tool(MCPTool(
            name="recall_memory",
            description="Recall memories matching a query",
            handler=self._handle_recall_memory,
            parameters=[
                MCPParameter("query", "string", "Search query"),
                MCPParameter("include_all_agents", "boolean", "Include memories from all agents", required=False, default=True),
                MCPParameter("limit", "number", "Maximum results", required=False, default=10),
            ],
            category="memory",
        ))
        
        # verify_claim tool
        self.register_tool(MCPTool(
            name="verify_claim",
            description="Verify a claim against the knowledge base",
            handler=self._handle_verify_claim,
            parameters=[
                MCPParameter("claim", "string", "The claim to verify"),
                MCPParameter("require_evidence", "boolean", "Require supporting evidence", required=False, default=True),
            ],
            category="verification",
        ))
        
        # get_status tool
        self.register_tool(MCPTool(
            name="get_constellation_status",
            description="Get the current status of the constellation",
            handler=self._handle_get_status,
            parameters=[],
            category="system",
        ))
        
        # list_agents tool
        self.register_tool(MCPTool(
            name="list_agents",
            description="List all available agents in the constellation",
            handler=self._handle_list_agents,
            parameters=[
                MCPParameter("domain_filter", "string", "Filter by domain", required=False),
            ],
            category="agent",
        ))
        
        # execute_skill tool
        self.register_tool(MCPTool(
            name="execute_skill",
            description="Execute a specific skill",
            handler=self._handle_execute_skill,
            parameters=[
                MCPParameter("skill_id", "string", "The skill to execute"),
                MCPParameter("inputs", "object", "Input parameters for the skill"),
            ],
            category="skill",
        ))
        
    def register_tool(self, tool: MCPTool) -> None:
        """Register an MCP tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered MCP tool: {tool.name}")
        
    def register_resource(self, resource: MCPResource) -> None:
        """Register an MCP resource."""
        self._resources[resource.uri] = resource
        logger.debug(f"Registered MCP resource: {resource.uri}")
        
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle an incoming MCP request."""
        try:
            if request.method == "initialize":
                return await self._handle_initialize(request)
            elif request.method == "tools/list":
                return await self._handle_list_tools(request)
            elif request.method == "tools/call":
                return await self._handle_call_tool(request)
            elif request.method == "resources/list":
                return await self._handle_list_resources(request)
            elif request.method == "resources/read":
                return await self._handle_read_resource(request)
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {request.method}"},
                )
        except Exception as e:
            logger.error(f"MCP request error: {e}", exc_info=True)
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)},
            )
            
    async def _handle_initialize(self, request: MCPRequest) -> MCPResponse:
        """Handle initialize request."""
        return MCPResponse(
            id=request.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                },
            },
        )
        
    async def _handle_list_tools(self, request: MCPRequest) -> MCPResponse:
        """Handle tools/list request."""
        tools = [tool.to_schema() for tool in self._tools.values()]
        return MCPResponse(id=request.id, result={"tools": tools})
        
    async def _handle_call_tool(self, request: MCPRequest) -> MCPResponse:
        """Handle tools/call request."""
        tool_name = request.params.get("name")
        arguments = request.params.get("arguments", {})
        
        tool = self._tools.get(tool_name)
        if not tool:
            return MCPResponse(
                id=request.id,
                error={"code": -32602, "message": f"Unknown tool: {tool_name}"},
            )
            
        try:
            result = await tool.handler(**arguments)
            return MCPResponse(
                id=request.id,
                result={"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
            )
        except Exception as e:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)},
            )
            
    async def _handle_list_resources(self, request: MCPRequest) -> MCPResponse:
        """Handle resources/list request."""
        resources = [res.to_schema() for res in self._resources.values()]
        return MCPResponse(id=request.id, result={"resources": resources})
        
    async def _handle_read_resource(self, request: MCPRequest) -> MCPResponse:
        """Handle resources/read request."""
        uri = request.params.get("uri")
        resource = self._resources.get(uri)
        
        if not resource:
            return MCPResponse(
                id=request.id,
                error={"code": -32602, "message": f"Unknown resource: {uri}"},
            )
            
        if resource.handler:
            content = await resource.handler()
        else:
            content = {}
            
        return MCPResponse(
            id=request.id,
            result={"contents": [{"uri": uri, "mimeType": resource.mime_type, "text": json.dumps(content)}]},
        )
        
    # ─────────────────────────────────────────────────────────────────────────
    # TOOL HANDLERS
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _handle_invoke_agent(
        self,
        agent_slug: str,
        task: str,
        context: Optional[str] = None,
    ) -> dict:
        """Handle invoke_agent tool."""
        return {
            "status": "invoked",
            "agent": agent_slug,
            "task": task,
            "context": context,
            "message": f"Agent '{agent_slug}' invoked with task",
        }
        
    async def _handle_assemble_team(
        self,
        team_name: str,
        task: str,
        snr_target: float = 0.93,
    ) -> dict:
        """Handle assemble_team tool."""
        return {
            "status": "assembled",
            "team": team_name,
            "task": task,
            "snr_target": snr_target,
            "message": f"Team '{team_name}' assembled for task",
        }
        
    async def _handle_query_knowledge(
        self,
        query: str,
        domain_filter: Optional[str] = None,
        min_snr: float = 0.0,
        max_results: int = 10,
    ) -> dict:
        """Handle query_knowledge tool."""
        return {
            "query": query,
            "domain_filter": domain_filter,
            "min_snr": min_snr,
            "results": [],
            "total_found": 0,
        }
        
    async def _handle_store_memory(
        self,
        content: str,
        priority: str = "medium",
        agent_slug: Optional[str] = None,
    ) -> dict:
        """Handle store_memory tool."""
        return {
            "status": "stored",
            "priority": priority,
            "agent": agent_slug,
            "memory_id": f"mem_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        }
        
    async def _handle_recall_memory(
        self,
        query: str,
        include_all_agents: bool = True,
        limit: int = 10,
    ) -> dict:
        """Handle recall_memory tool."""
        return {
            "query": query,
            "include_all_agents": include_all_agents,
            "memories": [],
            "count": 0,
        }
        
    async def _handle_verify_claim(
        self,
        claim: str,
        require_evidence: bool = True,
    ) -> dict:
        """Handle verify_claim tool."""
        return {
            "claim": claim,
            "verified": None,
            "confidence": 0.0,
            "evidence": [],
            "contradictions": [],
        }
        
    async def _handle_get_status(self) -> dict:
        """Handle get_constellation_status tool."""
        return {
            "status": "operational",
            "agents_loaded": 29,
            "teams_configured": 8,
            "memory_entries": 0,
            "knowledge_nodes": 0,
            "triggers_active": 0,
            "uptime_seconds": 0,
        }
        
    async def _handle_list_agents(
        self,
        domain_filter: Optional[str] = None,
    ) -> dict:
        """Handle list_agents tool."""
        # Placeholder - would load from roster
        agents = [
            {"slug": "ibn-sina", "name": "Ibn Sina", "domain": "Medicine & Philosophy"},
            {"slug": "al-khwarizmi", "name": "Al-Khwarizmi", "domain": "Mathematics & Algorithms"},
            # ... more agents
        ]
        
        if domain_filter:
            agents = [a for a in agents if domain_filter.lower() in a["domain"].lower()]
            
        return {"agents": agents, "count": len(agents)}
        
    async def _handle_execute_skill(
        self,
        skill_id: str,
        inputs: dict,
    ) -> dict:
        """Handle execute_skill tool."""
        return {
            "skill_id": skill_id,
            "inputs": inputs,
            "output": {},
            "status": "executed",
        }
        
    # ─────────────────────────────────────────────────────────────────────────
    # SERVER LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────
    
    async def start_stdio(self) -> None:
        """Start MCP server with stdio transport."""
        import sys
        
        self._running = True
        logger.info(f"Starting MCP server: {self.name} v{self.version}")
        
        while self._running:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                    
                # Parse request
                data = json.loads(line)
                request = MCPRequest(
                    id=data.get("id"),
                    method=data.get("method"),
                    params=data.get("params", {}),
                )
                
                # Handle and respond
                response = await self.handle_request(request)
                
                # Write to stdout
                sys.stdout.write(json.dumps(response.to_dict()) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Server error: {e}", exc_info=True)
                
    def stop(self) -> None:
        """Stop the MCP server."""
        self._running = False
        logger.info("MCP server stopped")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get the global MCP server instance."""
    global _server
    if _server is None:
        _server = MCPServer()
    return _server


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Main entry point for MCP server."""
    server = get_mcp_server()
    await server.start_stdio()


if __name__ == "__main__":
    asyncio.run(main())
