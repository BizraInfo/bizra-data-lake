#!/usr/bin/env python3
"""
===========================================================================================
    SOVEREIGN MCP SERVER -- Model Context Protocol Bridge to the House of Wisdom
===========================================================================================

    ARCHITECTURE: Exposes the full Sovereign Brain (488 nodes, 2 engines) to external agents

    TOOLS EXPOSED:
      1. sovereign_query      -- Unified query across all engines (Apex + Nexus)
      2. sovereign_patterns   -- Discover knowledge patterns (Hubs, Bridges, Co-occurrence)
      3. sovereign_communities -- Explore detected communities
      4. sovereign_health     -- Brain health diagnostics
      5. sovereign_stats      -- Full system statistics
      6. sovereign_reason     -- Deep GoT reasoning chain

    TRANSPORT: Stdio via MCP SDK (proper content-length framing)

    Created: 2026-01-22 | Dubai
    Fixed:   2026-02-17 | Migrated from raw stdin to MCP SDK stdio_server
===========================================================================================
"""

import os
import sys
import json
import asyncio
import logging
import time
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List
from io import StringIO
from datetime import datetime

# Set up path to ensure tools/ sibling imports work
_mcp_dir = os.path.dirname(os.path.abspath(__file__))
_tools_dir = os.path.dirname(_mcp_dir)          # tools/
_project_root = os.path.dirname(_tools_dir)      # BIZRA-DATA-LAKE/
sys.path.insert(0, os.path.join(_tools_dir, "engines"))
sys.path.insert(0, os.path.join(_tools_dir, "bridges"))
sys.path.insert(0, _project_root)

# ===============================================================================
# AUTO-LOAD .env
# ===============================================================================
_env_file = os.path.join(_project_root, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(_env_file)
except ImportError:
    if os.path.isfile(_env_file):
        with open(_env_file, "r") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    _key = _key.strip()
                    _val = _val.strip()
                    if _key and not os.environ.get(_key):
                        os.environ[_key] = _val

# ===============================================================================
# LOGGING SETUP -- stderr only, stdout reserved for MCP protocol
# ===============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("SovereignMCP")

# ===============================================================================
# SOVEREIGN BRAIN INTEGRATION (unchanged)
# ===============================================================================

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
            from sovereign_brain import SovereignBrain

            self.brain = SovereignBrain()
            self.brain.awaken()

            self.apex_adapter = self.brain.adapters.get('apex')
            self.nexus_adapter = self.brain.adapters.get('nexus')

            self.initialized = True
            logger.info(f"Sovereign Brain initialized: {self.brain.state.total_nodes} nodes")
            return True

        except ImportError as e:
            logger.warning(f"Sovereign Brain not available: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize Sovereign Brain: {e}")
            return False

    def query(self, query_text: str, limit: int = 10) -> Dict[str, Any]:
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
        if not self.initialized or not self.apex_adapter:
            return {}

        try:
            engine = self.apex_adapter.engine
            if engine and hasattr(engine, 'graph_layer'):
                communities = engine.graph_layer.communities if hasattr(engine.graph_layer, 'communities') else {}
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
        if not self.initialized:
            return {"status": "offline", "engines": {}}

        try:
            return self.brain.health_check()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
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
        if not self.initialized or not self.nexus_adapter:
            return {"error": "Nexus engine not available"}

        try:
            engine = self.nexus_adapter.engine
            if not engine:
                return {"error": "Nexus engine not loaded"}

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            try:
                results = engine.query(question, max_results=10)
                output = mystdout.getvalue()
            finally:
                sys.stdout = old_stdout

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

# ===============================================================================
# MCP SERVER via SDK (proper stdio transport with content-length framing)
# ===============================================================================

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

server = Server("sovereign-brain-mcp", version="1.1.0")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="sovereign_query",
            description="Query the Sovereign Brain -- unified search across the House of Wisdom knowledge graph. Returns semantically relevant results with SNR scoring.",
            inputSchema={
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
        ),
        types.Tool(
            name="sovereign_patterns",
            description="Discover knowledge patterns in the graph. Returns Hub nodes (highly connected), Type bridges (cross-domain connections), and Co-occurrence patterns.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="sovereign_communities",
            description="Explore detected knowledge communities. Returns community names, sizes, and sample nodes from each cluster.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="sovereign_health",
            description="Get brain health diagnostics. Returns engine status, connectivity, and any detected issues.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="sovereign_stats",
            description="Get full system statistics. Returns node counts, edge counts, engine status, and last query info.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="sovereign_reason",
            description="Perform deep Graph-of-Thoughts (GoT) reasoning. Uses the Nexus engine to trace multi-hop reasoning chains through the knowledge graph.",
            inputSchema={
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
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    # Lazy-initialize brain on first tool call
    if not brain_interface.initialized:
        brain_interface.initialize()

    if name == "sovereign_query":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        result = brain_interface.query(query, limit)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    elif name == "sovereign_patterns":
        patterns = brain_interface.get_patterns()
        result = {"pattern_count": len(patterns), "patterns": patterns}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    elif name == "sovereign_communities":
        communities = brain_interface.get_communities()
        result = {"community_count": len(communities), "communities": communities}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    elif name == "sovereign_health":
        health = brain_interface.get_health()
        return [types.TextContent(type="text", text=json.dumps(health, indent=2, default=str))]

    elif name == "sovereign_stats":
        stats = brain_interface.get_stats()
        return [types.TextContent(type="text", text=json.dumps(stats, indent=2, default=str))]

    elif name == "sovereign_reason":
        question = arguments.get("question", "")
        depth = arguments.get("depth", 3)
        result = brain_interface.reason(question, depth)
        return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    logger.info("Sovereign MCP Server starting (SDK stdio transport)...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
