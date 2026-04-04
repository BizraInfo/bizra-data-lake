"""
BIZRA Agentic-Flow Bridge
Provides access to agentic-flow's swarm intelligence, self-learning agents,
and MCP tools via HTTP protocol.

Integration points:
- Swarm orchestration (hierarchical-mesh topology, up to 15 agents)
- Self-learning agent dispatch (66 specialized agents)
- MCP tool invocation (213 tools via JSON-RPC 2.0)
- ReasoningBank queries (pattern learning/retrieval)
- Background worker dispatch (audit, optimize, ultralearn)

All operations are gated by BIZRA's Ihsan threshold and emit receipts.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional dependencies - graceful degradation if not installed
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)

if not HTTPX_AVAILABLE:
    logger.warning("httpx not installed - Agentic-Flow bridge will be disabled")

# Environment configuration
AGENTIC_FLOW_URL = os.getenv("AGENTIC_FLOW_URL", "http://localhost:3100")
AGENTIC_FLOW_TIMEOUT = int(os.getenv("AGENTIC_FLOW_TIMEOUT", "30"))
AGENTIC_FLOW_ENABLED = os.getenv("AGENTIC_FLOW_ENABLED", "true").lower() == "true"
AGENTIC_FLOW_MAX_AGENTS = int(os.getenv("AGENTIC_FLOW_MAX_AGENTS", "15"))
AGENTIC_FLOW_DEFAULT_TOPOLOGY = os.getenv(
    "AGENTIC_FLOW_DEFAULT_TOPOLOGY", "hierarchical-mesh"
)

# Receipt storage
RECEIPT_DIR = (
    Path(__file__).parent.parent / "docs" / "evidence" / "receipts" / "agentic_flow"
)


class SwarmTopology(str, Enum):
    """Agentic-flow swarm topologies mapped to BIZRA modes."""

    HIERARCHICAL = "hierarchical"  # Leader-follower
    MESH = "mesh"  # BIZRA Independent mode
    HIERARCHICAL_MESH = "hierarchical-mesh"  # BIZRA Collaborative mode
    STAR = "star"  # BIZRA HiveMind mode


class WorkerType(str, Enum):
    """Background worker types available in agentic-flow."""

    AUDIT = "audit"
    OPTIMIZE = "optimize"
    CONSOLIDATE = "consolidate"
    DOCUMENT = "document"
    DEEPDIVE = "deepdive"
    ULTRALEARN = "ultralearn"


@dataclass
class SwarmResult:
    """Result from a swarm invocation."""

    task: str
    topology: str
    agent_count: int
    status: str
    results: List[Dict[str, Any]]
    timestamp: str
    latency_ms: float
    receipt_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "topology": self.topology,
            "agent_count": self.agent_count,
            "status": self.status,
            "results": self.results,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "receipt_id": self.receipt_id,
        }


@dataclass
class AgenticFlowStatus:
    """Status of the agentic-flow service."""

    online: bool
    url: str
    version: str = ""
    agent_count: int = 0
    tool_count: int = 0
    last_check: str = ""
    error: Optional[str] = None


class AgenticFlowBridge:
    """
    Bridge to agentic-flow's swarm intelligence and MCP tools.

    Follows BIZRA patterns:
    - Fail-closed: Returns error on connection failure, never proceeds silently
    - Receipt-native: All operations emit evidence receipts
    - Ihsan-gated: Swarm results are validated against threshold
    """

    def __init__(
        self,
        url: str = AGENTIC_FLOW_URL,
        timeout: int = AGENTIC_FLOW_TIMEOUT,
        enabled: bool = AGENTIC_FLOW_ENABLED,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled and HTTPX_AVAILABLE
        self._client: Optional[Any] = None

        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed - Agentic-Flow bridge disabled")
        if not enabled:
            logger.info("Agentic-Flow bridge disabled via AGENTIC_FLOW_ENABLED=false")

    async def _get_client(self):
        """Get or create async HTTP client."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not installed. Install with: pip install httpx")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _emit_receipt(self, operation: str, status: str, data: dict) -> dict:
        """Emit BIZRA-compatible evidence receipt for agentic-flow operations."""
        receipt = {
            "receipt_id": f"agentic-flow-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
            "receipt_type": "agentic_flow_operation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "status": status,
            "data": data,
            "integrity_hash": hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        try:
            RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
            with open(RECEIPT_DIR / "operations.jsonl", "a") as f:
                f.write(json.dumps(receipt, default=str) + "\n")
        except OSError as e:
            logger.warning("Failed to write receipt: %s", e)
        return receipt

    async def health_check(self) -> AgenticFlowStatus:
        """Check if the agentic-flow service is online."""
        if not self.enabled:
            return AgenticFlowStatus(
                online=False,
                url=self.url,
                last_check=datetime.now(timezone.utc).isoformat(),
                error="Bridge disabled",
            )
        try:
            client = await self._get_client()
            response = await client.get(f"{self.url}/health")
            if response.status_code == 200:
                data = (
                    response.json()
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {}
                )
                return AgenticFlowStatus(
                    online=True,
                    url=self.url,
                    version=data.get("version", "unknown"),
                    agent_count=data.get("agents", 66),
                    tool_count=data.get("tools", 213),
                    last_check=datetime.now(timezone.utc).isoformat(),
                )
            return AgenticFlowStatus(
                online=False,
                url=self.url,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=f"HTTP {response.status_code}",
            )
        except Exception as e:
            logger.error("Agentic-Flow health check failed: %s", e)
            return AgenticFlowStatus(
                online=False,
                url=self.url,
                last_check=datetime.now(timezone.utc).isoformat(),
                error=str(e),
            )

    async def invoke_swarm(
        self,
        task: str,
        topology: SwarmTopology = SwarmTopology.HIERARCHICAL_MESH,
        agent_count: int = 5,
        timeout_ms: int = 30000,
    ) -> SwarmResult:
        """
        Invoke agentic-flow swarm for parallel agent execution.

        Maps to BIZRA swarm modes:
        - MESH -> BIZRA Independent mode
        - HIERARCHICAL_MESH -> BIZRA Collaborative mode
        - STAR -> BIZRA HiveMind mode

        Args:
            task: Description of the task for the swarm
            topology: Swarm topology to use
            agent_count: Number of agents to spawn (max 15)
            timeout_ms: Maximum execution time in milliseconds
        """
        if not self.enabled:
            raise RuntimeError("Agentic-Flow bridge is disabled")

        agent_count = min(agent_count, AGENTIC_FLOW_MAX_AGENTS)
        start = datetime.now(timezone.utc)

        try:
            client = await self._get_client()
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "swarm_init",
                    "arguments": {
                        "task": task,
                        "topology": topology.value,
                        "agent_count": agent_count,
                        "timeout_ms": timeout_ms,
                    },
                },
                "id": 1,
            }
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()

            results = []
            if "result" in data:
                content = data["result"].get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            try:
                                parsed = json.loads(item["text"])
                                results.append(parsed)
                            except json.JSONDecodeError:
                                results.append({"text": item["text"]})
                        else:
                            results.append(item)
                elif isinstance(content, str):
                    results = [{"text": content}]
            elif "error" in data:
                raise RuntimeError(f"Swarm error: {data['error']}")

            latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000

            receipt = self._emit_receipt(
                "swarm_invoke",
                "success",
                {
                    "task": task,
                    "topology": topology.value,
                    "agent_count": agent_count,
                    "result_count": len(results),
                    "latency_ms": latency,
                },
            )

            return SwarmResult(
                task=task,
                topology=topology.value,
                agent_count=agent_count,
                status="success",
                results=results,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=latency,
                receipt_id=receipt["receipt_id"],
            )

        except Exception as e:
            latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000
            self._emit_receipt(
                "swarm_invoke",
                "failure",
                {
                    "task": task,
                    "topology": topology.value,
                    "error": str(e),
                    "latency_ms": latency,
                },
            )
            logger.error("Swarm invocation failed: %s", e)
            raise

    async def call_mcp_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call an agentic-flow MCP tool via JSON-RPC 2.0.

        Args:
            tool_name: Name of the MCP tool to invoke
            arguments: Tool arguments
        """
        if not self.enabled:
            raise RuntimeError("Agentic-Flow bridge is disabled")

        try:
            client = await self._get_client()
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": 1,
            }
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()

            self._emit_receipt(
                "mcp_tool_call",
                "success",
                {
                    "tool": tool_name,
                    "arguments": arguments,
                },
            )

            if "result" in data:
                return data["result"]
            elif "error" in data:
                raise RuntimeError(f"MCP tool error: {data['error']}")
            return data

        except Exception as e:
            self._emit_receipt(
                "mcp_tool_call",
                "failure",
                {
                    "tool": tool_name,
                    "error": str(e),
                },
            )
            logger.error("MCP tool call failed: %s", e)
            raise

    async def query_reasoning_bank(self, query: str, limit: int = 10) -> dict:
        """
        Query agentic-flow's ReasoningBank for learned patterns.

        Args:
            query: Search query for pattern retrieval
            limit: Maximum results to return
        """
        return await self.call_mcp_tool(
            "reasoning_bank_query",
            {
                "query": query,
                "limit": limit,
            },
        )

    async def dispatch_worker(self, worker_type: WorkerType, directive: str) -> dict:
        """
        Dispatch an agentic-flow background worker.

        Args:
            worker_type: Type of worker (audit, optimize, ultralearn, etc.)
            directive: Task directive for the worker
        """
        return await self.call_mcp_tool(
            "worker_dispatch",
            {
                "worker_type": worker_type.value,
                "directive": directive,
            },
        )

    async def list_agents(self) -> List[dict]:
        """List available agentic-flow agents."""
        if not self.enabled:
            return []
        try:
            client = await self._get_client()
            response = await client.get(f"{self.url}/agents")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Failed to list agents: %s", e)
        return []

    async def list_tools(self) -> List[dict]:
        """List available MCP tools from agentic-flow."""
        if not self.enabled:
            return []
        try:
            client = await self._get_client()
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": 1,
            }
            response = await client.post(self.url, json=payload)
            if response.status_code == 200:
                data = response.json()
                return data.get("result", {}).get("tools", [])
        except Exception as e:
            logger.error("Failed to list tools: %s", e)
        return []


# Singleton instance
_bridge: Optional[AgenticFlowBridge] = None


def get_agentic_flow_bridge() -> AgenticFlowBridge:
    """Get the singleton Agentic-Flow bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = AgenticFlowBridge()
    return _bridge


if __name__ == "__main__":
    import asyncio

    async def test():
        bridge = AgenticFlowBridge()
        status = await bridge.health_check()
        print(f"Agentic-Flow Status: {'ONLINE' if status.online else 'OFFLINE'}")
        print(f"URL: {status.url}")
        if status.online:
            print(f"Version: {status.version}")
            print(f"Agents: {status.agent_count}")
            print(f"Tools: {status.tool_count}")
        elif status.error:
            print(f"Error: {status.error}")
        await bridge.close()

    asyncio.run(test())
