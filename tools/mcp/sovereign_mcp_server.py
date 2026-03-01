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
               + HTTP health/metrics on MCP_HTTP_PORT for K8s probes & Prometheus

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
import signal
import time
import contextlib
import hashlib
from collections import OrderedDict
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

    # Minimum requests before rollback evaluation kicks in (avoid noisy early state)
    _ROLLBACK_MIN_REQUESTS = 10

    def __init__(self):
        self._search = None       # VectorSearchEngine
        self._resonance = None    # CognitiveResonance
        self._hmm = None          # HMMEngine
        self._hmm_gate = None     # HMMCallerGate (wraps _hmm)
        self.initialized = False

        # Phase 47.1: Observability metrics (shared singleton)
        from core.rollout.metrics import get_shared_metrics
        self._metrics = get_shared_metrics()

        # Phase 49.3: Production rollback engine — monitors metrics and auto-rolls back
        # Receipt dir under /app/logs (writable in K8s) instead of default artifacts/ (read-only root FS)
        import os
        from core.rollout.rollback import RollbackEngine
        _receipt_dir = os.path.join(os.getenv("BIZRA_LOG_DIR", "logs"), "rollback_receipts")
        self._rollback = RollbackEngine(receipt_dir=_receipt_dir, metrics=self._metrics)

        # Phase 49.6: Canary gate — rollback actually stops traffic
        from core.rollout.canary import CanaryRouter
        self._canary = CanaryRouter()

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
            # Wrap with caller isolation gate
            from core.rollout.hmm_gate import HMMCallerGate
            self._hmm_gate = HMMCallerGate(self._hmm)
            logger.info("Phase 46: HMMEngine + CallerGate ready (6 cognitive states)")
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

        # Canary gate: rollback zeroing PERCENT actually stops traffic
        if not self._canary.should_route("search", query):
            return {"error": "Search temporarily disabled by canary routing", "results": []}

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

            self._evaluate_rollback()

            return {
                "query": query,
                "results": serialized,
                "count": len(serialized),
                "index_size": self._search.vector_count if hasattr(self._search, 'vector_count') else 0,
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception as e:
            logger.error(f"Phase 46 search error: {e}")
            self._metrics.inc("search_errors")
            self._evaluate_rollback()
            return {"error": str(e), "results": []}

    async def resonance(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Full cognitive resonance pipeline: search → predict."""
        self._metrics.inc("resonance_requests")

        # Canary gate: resonance depends on search, so gate on search component
        if not self._canary.should_route("search", query):
            return {"error": "Resonance temporarily disabled by canary routing"}

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

            self._evaluate_rollback()

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
            self._metrics.inc("resonance_errors")
            self._evaluate_rollback()
            return {"error": str(e)}

    def predict(self, action: str) -> Dict[str, Any]:
        """HMM cognitive state observation and prediction."""
        self._metrics.inc("hmm_requests")

        # Canary gate: rollback zeroing HMM_PERCENT actually stops observations
        if not self._canary.should_route("hmm", action):
            return {"error": "HMM prediction temporarily disabled by canary routing"}

        if self._hmm is None:
            return {"error": "HMM engine not available"}

        try:
            # Route through HMMCallerGate for caller isolation
            if self._hmm_gate is not None:
                result = self._hmm_gate.observe(action, "mcp")
                if result is None:
                    return {"error": "HMM observation rejected by caller gate"}
            else:
                result = self._hmm.observe(action)
            self._metrics.record_hmm_confidence(result.prediction_confidence)
            self._metrics.record_hmm_observation(action)

            self._evaluate_rollback()

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
            self._evaluate_rollback()
            return {"error": str(e)}

    def _evaluate_rollback(self) -> None:
        """Evaluate rollback conditions after each Phase 46 tool call.

        Checks error rates and latency regressions against thresholds
        from core.integration.constants.  Requires _ROLLBACK_MIN_REQUESTS
        to avoid false positives during cold start.

        Standing on Giants: Nygard (Release It!, 2007) · Fowler (canary, 2010)
        """
        from core.integration.constants import (
            ROLLBACK_GOT_FALLBACK_THRESHOLD,
            ROLLBACK_HMM_CONFIDENCE_FLOOR,
            ROLLBACK_LATENCY_DELTA_THRESHOLD,
            ROLLBACK_SEARCH_ERROR_THRESHOLD,
        )

        total = self._metrics.get_counter("search_requests")
        if total < self._ROLLBACK_MIN_REQUESTS:
            return  # Not enough data yet

        # Search error rate
        search_errors = self._metrics.get_counter("search_errors")
        if total > 0:
            error_rate = search_errors / total
            self._rollback.evaluate(
                "search_error_rate", error_rate > ROLLBACK_SEARCH_ERROR_THRESHOLD
            )

        # GoT fallback rate
        got_total = self._metrics.get_counter("got_requests")
        if got_total > 0:
            fallback_rate = self._metrics.compute_rate("got_fallback", "got_requests")
            self._rollback.evaluate(
                "got_fallback_rate", fallback_rate > ROLLBACK_GOT_FALLBACK_THRESHOLD
            )

        # HMM confidence floor
        hmm_total = self._metrics.get_counter("hmm_requests")
        if hmm_total >= 5:
            snap = self._metrics.snapshot()
            hmm_p50 = snap.get("hmm", {}).get("confidence_p50", 1.0)
            self._rollback.evaluate(
                "hmm_confidence", hmm_p50 < ROLLBACK_HMM_CONFIDENCE_FLOOR
            )

        # Latency regression (search p95 vs first 10 requests baseline)
        search_latencies = self._metrics._latencies.get("search", [])
        if len(search_latencies) >= 20:
            baseline_p95 = self._metrics.percentile(search_latencies[:10], 95)
            current_p95 = self._metrics.percentile(search_latencies[-10:], 95)
            if baseline_p95 > 0:
                regression = (current_p95 - baseline_p95) / baseline_p95
                self._rollback.evaluate(
                    "latency_regression", regression > ROLLBACK_LATENCY_DELTA_THRESHOLD
                )

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
            "hmm_gate": self._hmm_gate.stats if self._hmm_gate is not None else None,
            "canary_percents": self._canary.get_active_percents(),
            "metrics": self._metrics.snapshot(),
            "rollback": self._rollback.status,
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


# ===============================================================================
# HTTP HEALTH/METRICS SERVER — K8s probe + Prometheus scrape endpoint
# Runs alongside stdio MCP transport on MCP_HTTP_PORT (default 8081).
# Standing on Giants: Lamport (liveness/readiness) · Shannon (metrics exposition)
# ===============================================================================

_HEALTH_PORT = int(os.getenv("MCP_HTTP_PORT", "8081"))


_MAX_HEADER_BYTES = 8192  # 8 KB — reject oversized headers (DoS mitigation)
_DRAIN_TIMEOUT = 5.0      # seconds to wait for writer.drain()


async def _http_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Handle HTTP requests for K8s probes and Prometheus scrape.

    Security hardening (Phase 49.7):
      - Header size capped at 8 KB to prevent header-bomb DoS.
      - writer.drain() wrapped with timeout to prevent slow-client stalls.
      - Exceptions logged instead of silently swallowed.
    """
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        # Drain remaining headers, enforce size cap
        total_header_bytes = len(request_line)
        while True:
            line = await reader.readline()
            total_header_bytes += len(line)
            if total_header_bytes > _MAX_HEADER_BYTES:
                # Header bomb — send 431 and close immediately
                writer.write(b"HTTP/1.1 431 Request Header Fields Too Large\r\nConnection: close\r\n\r\n")
                await asyncio.wait_for(writer.drain(), timeout=_DRAIN_TIMEOUT)
                return
            if line in (b"\r\n", b"\n", b""):
                break

        parts = request_line.decode(errors="replace").split()
        path = parts[1] if len(parts) >= 2 else "/"

        if path == "/health":
            body = json.dumps({
                "status": "healthy",
                "server": "sovereign-brain-mcp",
                "version": "1.3.0",
                "phase46_initialized": phase46_interface.initialized,
                "query_count": _query_count,
                "error_count": _error_count,
                "uptime_seconds": round(time.monotonic() - _server_start_time, 1),
            })
            content_type = "application/json"
            status = "200 OK"

        elif path in ("/metrics", "/metrics/prometheus"):
            snap = phase46_interface._metrics.snapshot() if phase46_interface.initialized else {}
            counters = snap.get("counters", {})
            search = snap.get("search", {})
            resonance = snap.get("resonance", {})
            hmm = snap.get("hmm", {})

            lines = [
                "# HELP sovereign_mcp_queries_total Total MCP tool calls",
                "# TYPE sovereign_mcp_queries_total counter",
                f"sovereign_mcp_queries_total {_query_count}",
                "",
                "# HELP sovereign_mcp_errors_total Total MCP errors",
                "# TYPE sovereign_mcp_errors_total counter",
                f"sovereign_mcp_errors_total {_error_count}",
                "",
                "# HELP bizra_phase46_search_requests_total Phase46 search requests",
                "# TYPE bizra_phase46_search_requests_total counter",
                f"bizra_phase46_search_requests_total {counters.get('search_requests', 0)}",
                "",
                "# HELP bizra_phase46_search_hits_total Phase46 search hits",
                "# TYPE bizra_phase46_search_hits_total counter",
                f"bizra_phase46_search_hits_total {counters.get('search_hits', 0)}",
                "",
                "# HELP bizra_phase46_search_errors_total Phase46 search errors",
                "# TYPE bizra_phase46_search_errors_total counter",
                f"bizra_phase46_search_errors_total {counters.get('search_errors', 0)}",
                "",
                "# HELP bizra_phase46_resonance_requests_total Phase46 resonance requests",
                "# TYPE bizra_phase46_resonance_requests_total counter",
                f"bizra_phase46_resonance_requests_total {counters.get('resonance_requests', 0)}",
                "",
                "# HELP bizra_phase46_resonance_errors_total Phase46 resonance errors",
                "# TYPE bizra_phase46_resonance_errors_total counter",
                f"bizra_phase46_resonance_errors_total {counters.get('resonance_errors', 0)}",
                "",
                "# HELP bizra_phase46_hmm_requests_total Phase46 HMM requests",
                "# TYPE bizra_phase46_hmm_requests_total counter",
                f"bizra_phase46_hmm_requests_total {counters.get('hmm_requests', 0)}",
                "",
                "# HELP bizra_phase46_got_requests_total Phase46 GoT bridge requests",
                "# TYPE bizra_phase46_got_requests_total counter",
                f"bizra_phase46_got_requests_total {counters.get('got_requests', 0)}",
                "",
                "# HELP bizra_phase46_got_fallback_total Phase46 GoT bridge fallbacks",
                "# TYPE bizra_phase46_got_fallback_total counter",
                f"bizra_phase46_got_fallback_total {counters.get('got_fallback', 0)}",
                "",
                "# HELP bizra_phase46_search_latency_p95_ms Phase46 search p95 latency",
                "# TYPE bizra_phase46_search_latency_p95_ms gauge",
                f"bizra_phase46_search_latency_p95_ms {search.get('latency_p95_ms', 0.0)}",
                "",
                "# HELP bizra_phase46_search_hit_rate Phase46 search hit rate",
                "# TYPE bizra_phase46_search_hit_rate gauge",
                f"bizra_phase46_search_hit_rate {search.get('hit_rate', 0.0)}",
                "",
                "# HELP bizra_phase46_resonance_snr_p50 Phase46 resonance SNR p50",
                "# TYPE bizra_phase46_resonance_snr_p50 gauge",
                f"bizra_phase46_resonance_snr_p50 {resonance.get('combined_snr_p50', 0.0)}",
                "",
                "# HELP bizra_phase46_hmm_confidence_p50 Phase46 HMM confidence p50",
                "# TYPE bizra_phase46_hmm_confidence_p50 gauge",
                f"bizra_phase46_hmm_confidence_p50 {hmm.get('confidence_p50', 0.0)}",
                "",
                "# HELP bizra_phase46_hmm_entropy Phase46 HMM observation entropy",
                "# TYPE bizra_phase46_hmm_entropy gauge",
                f"bizra_phase46_hmm_entropy {hmm.get('observation_entropy', 0.0)}",
            ]
            body = "\n".join(lines)
            content_type = "text/plain; version=0.0.4; charset=utf-8"
            status = "200 OK"

        else:
            body = '{"error": "not found"}'
            content_type = "application/json"
            status = "404 Not Found"

        response = (
            f"HTTP/1.1 {status}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body.encode())}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await asyncio.wait_for(writer.drain(), timeout=_DRAIN_TIMEOUT)
    except asyncio.TimeoutError:
        logger.debug("HTTP handler: client timed out (read or drain)")
    except ConnectionResetError:
        pass  # client disconnected — benign
    except Exception:
        logger.warning("HTTP handler unexpected error", exc_info=True)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass  # best-effort close


async def main():
    logger.info("Sovereign MCP Server v1.3.0 starting (SDK stdio transport + Phase 46)...")

    # Start HTTP health/metrics server for K8s probes and Prometheus
    # Gracefully skip if port is already in use (e.g. bizra-refinery on 8081)
    health_server = None
    try:
        health_server = await asyncio.start_server(_http_handler, "127.0.0.1", _HEALTH_PORT)
        logger.info("Health/metrics HTTP server listening on port %d", _HEALTH_PORT)
    except OSError as exc:
        logger.warning("Health server skipped (port %d in use: %s) — MCP stdio still functional", _HEALTH_PORT, exc)

    # Graceful shutdown via SIGTERM/SIGINT (K8s pod termination)
    shutdown_event = asyncio.Event()

    def _request_shutdown():
        logger.info("Shutdown signal received, draining...")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _request_shutdown)
    except NotImplementedError:
        pass  # Windows ProactorEventLoop — signal handlers not supported

    # Run MCP stdio transport as a cancellable task.  If stdin is closed
    # (K8s headless container), stdio_server exits quickly.  If SIGTERM
    # arrives while stdio is active, we cancel it and shut down cleanly.
    async def _run_stdio():
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream, write_stream, server.create_initialization_options()
                )
        except Exception as exc:
            logger.info("MCP stdio transport exited: %s", exc)

    stdio_task = asyncio.create_task(_run_stdio())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    # Race: either stdio finishes (stdin closed) or shutdown signal arrives
    done, pending = await asyncio.wait(
        [stdio_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for t in pending:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t

    # stdio transport ended naturally (not via SIGTERM) — enter headless mode.
    # Keep health server alive for K8s probes until SIGTERM arrives.
    if not shutdown_event.is_set() and health_server is not None:
        logger.info(
            "stdio transport ended; entering headless mode "
            "(health/metrics on port %d, awaiting SIGTERM)",
            _HEALTH_PORT,
        )
        await shutdown_event.wait()

    logger.info("Sovereign MCP Server shutting down...")
    if health_server is not None:
        health_server.close()
        await health_server.wait_closed()
    logger.info("Sovereign MCP Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
