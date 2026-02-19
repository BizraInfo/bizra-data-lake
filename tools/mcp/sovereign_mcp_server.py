#!/usr/bin/env python3
"""
===========================================================================================
    SOVEREIGN MCP SERVER -- Model Context Protocol Bridge to the House of Wisdom
===========================================================================================

    ARCHITECTURE: Exposes the full Sovereign Brain (488 nodes, 2 engines) + Phase 46
                  Cognitive Resonance (FAISS search, HMM prediction) to external agents

    TOOLS EXPOSED:
      1. sovereign_query      -- Unified query across all engines (Apex + Nexus)
      2. sovereign_patterns   -- Discover knowledge patterns (Hubs, Bridges, Co-occurrence)
      3. sovereign_communities -- Explore detected communities
      4. sovereign_health     -- Brain health diagnostics
      5. sovereign_stats      -- Full system statistics
      6. sovereign_reason     -- Deep GoT reasoning chain
      7. sovereign_search     -- Phase 46: FAISS vector search (102K vectors, 384-dim)
      8. sovereign_resonance  -- Phase 46: Cognitive resonance pipeline (search+predict)
      9. sovereign_predict    -- Phase 46: HMM cognitive state prediction
     10. mcp_health           -- Server performance metrics

    TRANSPORT: Stdio via MCP SDK (proper content-length framing)

    Created: 2026-01-22 | Dubai
    V3 Optimized: 2026-02-18 | Caching, compact JSON, timeouts, health monitoring
    V4 Phase 46: 2026-02-19 | Cognitive Resonance tools (FAISS, HMM, Resonance)
===========================================================================================
"""

import os
import sys
import json
import asyncio
import logging
import time
import contextlib
import hashlib
from collections import OrderedDict
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Optional
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
# CONSTANTS
# ===============================================================================

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


# Cacheable tools (read-only, deterministic) -- sovereign_reason excluded (varied output)
# sovereign_search is cacheable (same query = same FAISS results)
# sovereign_resonance and sovereign_predict are NOT cacheable (HMM state changes)
CACHEABLE_TOOLS = {
    "sovereign_query", "sovereign_patterns", "sovereign_communities",
    "sovereign_health", "sovereign_stats", "sovereign_search"
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
# SOVEREIGN BRAIN INTEGRATION
# ===============================================================================

class SovereignBrainInterface:
    """Interface to the Sovereign Brain orchestration layer."""

    def __init__(self):
        self.brain = None
        self.apex_adapter = None
        self.nexus_adapter = None
        self.initialized = False

    def initialize(self) -> bool:
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

            # Thread-safe stdout capture via contextlib
            captured = StringIO()
            with contextlib.redirect_stdout(captured):
                results = engine.query(question, max_results=10)

            output = captured.getvalue()

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
# PHASE 46 — COGNITIVE RESONANCE INTEGRATION
# Standing on Giants: Shannon (1948) · Johnson/FAISS (2021) · Rabiner (1989)
# ===============================================================================

class Phase46Interface:
    """Lazy-initialized Phase 46 Cognitive Resonance components.

    Each component initializes independently — partial init is OK.
    VectorSearchEngine lazy-loads the 152MB FAISS index on first search.
    HMMEngine starts fresh per server session.
    CognitiveResonance orchestrates search → predict with graceful degradation.
    """

    def __init__(self):
        self._search = None       # VectorSearchEngine
        self._resonance = None    # CognitiveResonance
        self._hmm = None          # HMMEngine
        self.initialized = False

        # Phase 47.1: Observability metrics
        from core.rollout.metrics import Phase46Metrics
        self._metrics = Phase46Metrics()

    def initialize(self) -> bool:
        """Initialize Phase 46 components. Each is independent."""
        try:
            from core.search import VectorSearchEngine
            self._search = VectorSearchEngine()
            logger.info("Phase 46: VectorSearchEngine ready (lazy FAISS load)")
        except Exception as e:
            logger.warning(f"Phase 46 search init failed: {e}")

        try:
            from core.prediction import HMMEngine
            self._hmm = HMMEngine()
            logger.info("Phase 46: HMMEngine ready (6 cognitive states)")
        except Exception as e:
            logger.warning(f"Phase 46 HMM init failed: {e}")

        try:
            from core.resonance import CognitiveResonance
            self._resonance = CognitiveResonance(
                search=self._search,
                reasoning=None,   # GoTBridge requires async GoT engine — wire in Phase 47
                prediction=self._hmm,
            )
            logger.info("Phase 46: CognitiveResonance pipeline ready")
        except Exception as e:
            logger.warning(f"Phase 46 resonance init failed: {e}")

        self.initialized = True
        components = sum(1 for c in [self._search, self._resonance, self._hmm] if c is not None)
        logger.info(f"Phase 46 initialized: {components}/3 components online")
        return components > 0

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """FAISS vector search with cosine similarity scoring."""
        self._metrics.inc("search_requests")

        if self._search is None:
            return {"error": "Search engine not available", "results": []}

        try:
            start = time.perf_counter()
            results = self._search.search(query, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000
            self._metrics.record_latency("search", elapsed)

            serialized = []
            for sr in results:
                rec = sr.record
                serialized.append({
                    "content": rec.content[:500] if rec.content else "",
                    "score": round(sr.score, 4),
                    "source": rec.source or "",
                    "source_id": rec.source_id or "",
                    "metadata": rec.metadata or {},
                })

            if len(serialized) > 0:
                self._metrics.inc("search_hits")

            return {
                "query": query,
                "results": serialized,
                "count": len(serialized),
                "index_size": self._search.vector_count if hasattr(self._search, 'vector_count') else 0,
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception as e:
            logger.error(f"Phase 46 search error: {e}")
            return {"error": str(e), "results": []}

    async def resonance(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Full cognitive resonance pipeline: search → predict."""
        self._metrics.inc("resonance_requests")

        if self._resonance is None:
            return {"error": "Resonance pipeline not available"}

        try:
            start = time.perf_counter()
            result = await self._resonance.process(query, context)
            elapsed = (time.perf_counter() - start) * 1000
            self._metrics.record_latency("resonance", elapsed)
            self._metrics.record_snr(result.combined_snr)

            # Serialize search results
            search_serialized = []
            for sr in result.search_results:
                search_serialized.append({
                    "content": sr.record.content[:300] if sr.record.content else "",
                    "score": round(sr.score, 4),
                    "source": sr.record.source or "",
                })

            # Serialize prediction
            prediction_data = None
            if result.prediction is not None:
                prediction_data = {
                    "most_likely_state": str(result.prediction.most_likely_state.value),
                    "predicted_next": str(result.prediction.predicted_next_state.value),
                    "confidence": round(result.prediction.prediction_confidence, 4),
                }

            return {
                "query": query,
                "search_results": search_serialized,
                "search_count": len(search_serialized),
                "prediction": prediction_data,
                "combined_snr": round(result.combined_snr, 4),
                "processing_path": result.processing_path,
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception as e:
            logger.error(f"Phase 46 resonance error: {e}")
            return {"error": str(e)}

    def predict(self, action: str) -> Dict[str, Any]:
        """HMM cognitive state observation and prediction."""
        self._metrics.inc("hmm_requests")

        if self._hmm is None:
            return {"error": "HMM engine not available"}

        try:
            result = self._hmm.observe(action)
            self._metrics.record_hmm_confidence(result.prediction_confidence)
            self._metrics.record_hmm_observation(action)

            return {
                "action": action,
                "most_likely_state": result.most_likely_state.value,
                "state_probabilities": {
                    k.value if hasattr(k, 'value') else str(k): round(v, 4)
                    for k, v in result.state_probabilities.items()
                },
                "predicted_next_state": result.predicted_next_state.value,
                "prediction_confidence": round(result.prediction_confidence, 4),
                "observation_likelihood": round(result.observation_likelihood, 4),
                "observation_count": len(self._hmm._observation_history),
            }
        except Exception as e:
            logger.error(f"Phase 46 predict error: {e}")
            return {"error": str(e)}

    @property
    def status(self) -> Dict[str, Any]:
        """Phase 46 component status for health reporting."""
        return {
            "initialized": self.initialized,
            "search_available": self._search is not None,
            "hmm_available": self._hmm is not None,
            "resonance_available": self._resonance is not None,
            "hmm_current_state": (
                self._hmm.current_state.value
                if self._hmm is not None else None
            ),
            "metrics": self._metrics.snapshot(),
        }


# Global Phase 46 interface
phase46_interface = Phase46Interface()

# ===============================================================================
# MCP SERVER via SDK (proper stdio transport with content-length framing)
# ===============================================================================

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

server = Server("sovereign-brain-mcp", version="1.3.0")


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
        # ---- Phase 46: Cognitive Resonance Tools ----
        types.Tool(
            name="sovereign_search",
            description="FAISS vector search across 102K indexed chunks (384-dim, cosine similarity). Returns semantically similar content from the full BIZRA knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10, max: 50)",
                        "default": 10
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="sovereign_resonance",
            description="Full cognitive resonance pipeline: FAISS search + HMM prediction. Returns search results, cognitive state prediction, and combined SNR score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to process through the resonance pipeline."
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="sovereign_predict",
            description="HMM cognitive state prediction. Observe an action symbol and get predicted next cognitive state. States: idle, exploring, organizing, creating, analyzing, communicating.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action symbol to observe. Valid: search, edit, navigate, organize, review, compile, test, chat, deploy, file_open, file_save, idle."
                    }
                },
                "required": ["action"]
            }
        ),
        types.Tool(
            name="mcp_health",
            description="Get MCP server performance metrics: uptime, query count, cache hit rate, errors, avg response time.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
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
        # Lazy-initialize brain on first tool call
        if not brain_interface.initialized:
            brain_interface.initialize()

        result: Any = None

        if name == "sovereign_query":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 10)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, brain_interface.query, query, limit),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        elif name == "sovereign_patterns":
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, brain_interface.get_patterns),
                timeout=TOOL_TIMEOUT_SECONDS
            )
            result = {"pattern_count": len(result), "patterns": result}

        elif name == "sovereign_communities":
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, brain_interface.get_communities),
                timeout=TOOL_TIMEOUT_SECONDS
            )
            result = {"community_count": len(result), "communities": result}

        elif name == "sovereign_health":
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, brain_interface.get_health),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        elif name == "sovereign_stats":
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, brain_interface.get_stats),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        elif name == "sovereign_reason":
            question = arguments.get("question", "")
            depth = arguments.get("depth", 3)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, brain_interface.reason, question, depth),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        # ---- Phase 46: Cognitive Resonance Handlers ----

        elif name == "sovereign_search":
            if not phase46_interface.initialized:
                phase46_interface.initialize()
            query = arguments.get("query", "")
            top_k = min(arguments.get("top_k", 10), 50)
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, phase46_interface.search, query, top_k
                ),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        elif name == "sovereign_resonance":
            if not phase46_interface.initialized:
                phase46_interface.initialize()
            query = arguments.get("query", "")
            result = await asyncio.wait_for(
                phase46_interface.resonance(query),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        elif name == "sovereign_predict":
            if not phase46_interface.initialized:
                phase46_interface.initialize()
            action = arguments.get("action", "idle")
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, phase46_interface.predict, action
                ),
                timeout=TOOL_TIMEOUT_SECONDS
            )

        elif name == "mcp_health":
            uptime = time.monotonic() - _server_start_time
            result = {
                "server": "sovereign-brain-mcp",
                "version": "1.3.0",
                "uptime_seconds": round(uptime, 1),
                "query_count": _query_count,
                "error_count": _error_count,
                "cache_hit_rate": round(cache.hit_rate, 4),
                "cache_size": cache.size,
                "avg_response_ms": round(_total_response_time / _query_count, 2) if _query_count > 0 else 0.0,
                "brain_initialized": brain_interface.initialized,
                "phase46": phase46_interface.status,
            }

        else:
            _error_count += 1
            raise ValueError(f"Unknown tool: {name}")

        # Compact JSON for STDIO transport
        text = json.dumps(result, default=str, separators=(',', ':'))

        # Cache read-only tools
        if name in CACHEABLE_TOOLS:
            cache.put(name, arguments, text)

        elapsed = (time.perf_counter() - start) * 1000
        _total_response_time += elapsed
        return [types.TextContent(type="text", text=text)]

    except asyncio.TimeoutError:
        _error_count += 1
        elapsed = (time.perf_counter() - start) * 1000
        _total_response_time += elapsed
        error = {"error": "timeout", "message": f"Tool '{name}' exceeded {TOOL_TIMEOUT_SECONDS}s timeout", "elapsed_ms": round(elapsed, 2)}
        return [types.TextContent(type="text", text=json.dumps(error, separators=(',', ':')))]
    except Exception as e:
        _error_count += 1
        elapsed = (time.perf_counter() - start) * 1000
        _total_response_time += elapsed
        logger.error(f"Error executing tool {name}: {e}")
        error = {"error": str(type(e).__name__), "message": str(e)}
        return [types.TextContent(type="text", text=json.dumps(error, separators=(',', ':')))]


async def main():
    logger.info("Sovereign MCP Server v1.3.0 starting (SDK stdio transport + Phase 46)...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
