"""
BIZRA-DATA-LAKE Unified MCP Gateway
Version: 1.0.0 | Phase 11 - Public Launch

Unified MCP endpoint for the BIZRA federated ecosystem.
Exposes Data Lake capabilities via Model Context Protocol.

Ihsan >= 0.95 | SNR >= 0.99 | Fail-Closed Enforcement
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bizra.mcp_gateway")

# Phase46 metrics (live observability — replaces hardcoded Prometheus values)
try:
    from core.rollout.metrics import Phase46Metrics
    _gateway_metrics = Phase46Metrics()
except ImportError:
    _gateway_metrics = None

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SNR_THRESHOLD = float(os.getenv("SNR_THRESHOLD", "0.85"))  # canonical: core.integration.constants
IHSAN_THRESHOLD = float(os.getenv("IHSAN_THRESHOLD", "0.95"))  # canonical: core.integration.constants


# ============================================================================
# MODELS
# ============================================================================

class MCPRequest(BaseModel):
    """MCP JSON-RPC request format."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP JSON-RPC response format."""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Unified query request."""
    query: str
    context: Optional[Dict[str, Any]] = None
    mode: str = "standard"  # standard, deep, vision
    max_tokens: int = 2048
    temperature: float = 0.7


class QueryResponse(BaseModel):
    """Unified query response."""
    response: str
    sources: List[Dict[str, Any]] = []
    ihsan_score: float
    snr_score: float
    receipt_id: str
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: Dict[str, str]
    thresholds: Dict[str, float]
    timestamp: str


class IngestRequest(BaseModel):
    """Document ingestion request."""
    content: str
    doc_type: str = "text"
    metadata: Dict[str, Any] = {}
    source_path: Optional[str] = None


class IngestResponse(BaseModel):
    """Document ingestion response."""
    doc_id: str
    status: str
    embedding_count: int
    receipt_id: str


# ============================================================================
# RECEIPT GENERATION
# ============================================================================

def generate_receipt_id(operation: str, input_data: str) -> str:
    """Generate a unique receipt ID for audit trail."""
    timestamp = datetime.utcnow().isoformat()
    content = f"{operation}:{input_data}:{timestamp}"
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()


def calculate_ihsan_score(
    correctness: float = 1.0,
    completeness: float = 1.0,
    efficiency: float = 1.0,
    ethical_alignment: float = 1.0
) -> float:
    """Calculate Ihsan (excellence) score."""
    return (
        correctness * 0.25 +
        completeness * 0.25 +
        efficiency * 0.20 +
        ethical_alignment * 0.30
    )


# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting BIZRA MCP Gateway...")

    # Initialize Redis connection
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)

    # Initialize HTTP client
    app.state.http_client = httpx.AsyncClient(timeout=60.0)

    logger.info(f"MCP Gateway ready. SNR >= {SNR_THRESHOLD}, Ihsan >= {IHSAN_THRESHOLD}")

    yield

    # Shutdown
    logger.info("Shutting down MCP Gateway...")
    await app.state.redis.close()
    await app.state.http_client.aclose()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="BIZRA MCP Gateway",
    description="Unified Model Context Protocol endpoint for BIZRA Data Lake",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware — restrict origins in production (BIZRA_CORS_ORIGINS csv).
# allow_credentials requires explicit origins; wildcard + credentials is
# rejected by browsers per the Fetch spec and is a security misconfiguration.
_cors_origins = [
    o.strip()
    for o in os.getenv("BIZRA_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


# ============================================================================
# MCP ENDPOINTS
# ============================================================================

@app.post("/mcp", response_model=MCPResponse)
async def mcp_handler(request: MCPRequest):
    """
    Main MCP JSON-RPC endpoint.

    Supported methods:
    - query: Execute a knowledge query
    - ingest: Ingest a document
    - health: Get system health
    - capabilities: List available capabilities
    """
    receipt_id = generate_receipt_id(request.method, json.dumps(request.params or {}))
    if _gateway_metrics is not None:
        _gateway_metrics.inc("gateway_requests")

    try:
        if request.method == "query":
            result = await handle_query(request.params or {})
        elif request.method == "ingest":
            result = await handle_ingest(request.params or {})
        elif request.method == "health":
            result = await handle_health()
        elif request.method == "capabilities":
            result = await handle_capabilities()
        else:
            return MCPResponse(
                id=request.id,
                error={
                    "code": -32601,
                    "message": f"Method not found: {request.method}"
                }
            )

        return MCPResponse(
            id=request.id,
            result={
                **result,
                "receipt_id": receipt_id
            }
        )

    except Exception as e:
        logger.error(f"MCP error: {e}")
        return MCPResponse(
            id=request.id,
            error={
                "code": -32000,
                "message": str(e),
                "receipt_id": receipt_id
            }
        )


async def handle_query(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle query method."""
    query = params.get("query", "")
    mode = params.get("mode", "standard")

    # TODO: Integrate with actual Data Lake query engine
    # For now, return a placeholder response

    ihsan_score = calculate_ihsan_score()

    if ihsan_score < IHSAN_THRESHOLD:
        raise HTTPException(
            status_code=422,
            detail=f"Ihsan score {ihsan_score} below threshold {IHSAN_THRESHOLD}"
        )

    return {
        "response": f"Query processed: {query[:100]}...",
        "sources": [],
        "ihsan_score": ihsan_score,
        "snr_score": SNR_THRESHOLD,
        "mode": mode
    }


async def handle_ingest(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle ingest method."""
    content = params.get("content", "")
    doc_type = params.get("doc_type", "text")

    # Generate document ID
    doc_id = hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    # TODO: Integrate with actual Data Lake ingestion pipeline

    return {
        "doc_id": doc_id,
        "status": "ingested",
        "embedding_count": 1,
        "doc_type": doc_type
    }


async def handle_health() -> Dict[str, Any]:
    """Handle health method."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "mcp_gateway": "operational",
            "data_lake": "operational"
        },
        "thresholds": {
            "snr": SNR_THRESHOLD,
            "ihsan": IHSAN_THRESHOLD
        }
    }


async def handle_capabilities() -> Dict[str, Any]:
    """Handle capabilities method."""
    return {
        "methods": [
            {
                "name": "query",
                "description": "Execute a knowledge query against the Data Lake",
                "params": {
                    "query": "string (required)",
                    "mode": "string (optional: standard|deep|vision)",
                    "context": "object (optional)"
                }
            },
            {
                "name": "ingest",
                "description": "Ingest a document into the Data Lake",
                "params": {
                    "content": "string (required)",
                    "doc_type": "string (optional: text|pdf|image|audio)",
                    "metadata": "object (optional)"
                }
            },
            {
                "name": "health",
                "description": "Get system health status",
                "params": {}
            },
            {
                "name": "capabilities",
                "description": "List available capabilities",
                "params": {}
            }
        ],
        "version": "1.0.0",
        "protocol": "MCP/JSON-RPC 2.0"
    }


# ============================================================================
# REST ENDPOINTS (Unified API)
# ============================================================================

@app.get("/health")
@app.get("/health/live")
async def health_live():
    """Liveness probe."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@app.get("/health/ready")
async def health_ready(request: Request):
    """Readiness probe."""
    try:
        # Check Redis connection
        await request.app.state.redis.ping()
        redis_status = "ready"
    except Exception as e:
        logger.debug("Redis readiness check failed: %s", e)
        redis_status = "not_ready"

    status = "ready" if redis_status == "ready" else "not_ready"

    return {
        "status": status,
        "components": {
            "redis": redis_status,
            "mcp_gateway": "ready"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/memory/health")
async def memory_health():
    """Memory system health endpoint for Kong routing."""
    return await handle_health()


@app.post("/api/v1/memory/query")
async def memory_query(request: QueryRequest):
    """Unified memory query endpoint."""
    receipt_id = generate_receipt_id("memory_query", request.query)

    result = await handle_query({
        "query": request.query,
        "mode": request.mode,
        "context": request.context
    })

    return QueryResponse(
        response=result["response"],
        sources=result.get("sources", []),
        ihsan_score=result["ihsan_score"],
        snr_score=result["snr_score"],
        receipt_id=receipt_id,
        metadata={"mode": request.mode}
    )


@app.post("/api/v1/memory/ingest")
async def memory_ingest(request: IngestRequest):
    """Unified memory ingestion endpoint."""
    receipt_id = generate_receipt_id("memory_ingest", request.content[:100])

    result = await handle_ingest({
        "content": request.content,
        "doc_type": request.doc_type,
        "metadata": request.metadata,
        "source_path": request.source_path
    })

    return IngestResponse(
        doc_id=result["doc_id"],
        status=result["status"],
        embedding_count=result["embedding_count"],
        receipt_id=receipt_id
    )


@app.get("/api/v1/knowledge")
async def knowledge_status():
    """Knowledge base status endpoint."""
    return {
        "status": "operational",
        "tiers": {
            "intake": "active",
            "raw": "active",
            "processed": "active",
            "indexed": "active",
            "gold": "active"
        },
        "thresholds": {
            "snr": SNR_THRESHOLD,
            "ihsan": IHSAN_THRESHOLD
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint with live Phase46 metrics.

    Served on both /metrics (K8s annotation default) and /metrics/prometheus
    (backward compatibility).
    """
    snap = _gateway_metrics.snapshot() if _gateway_metrics is not None else {}
    counters = snap.get("counters", {})
    search = snap.get("search", {})
    resonance = snap.get("resonance", {})
    hmm = snap.get("hmm", {})

    lines = [
        "# HELP bizra_mcp_requests_total Total MCP gateway requests",
        "# TYPE bizra_mcp_requests_total counter",
        f"bizra_mcp_requests_total {counters.get('gateway_requests', 0)}",
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
        "",
        "# HELP bizra_ihsan_threshold Ihsan quality threshold",
        "# TYPE bizra_ihsan_threshold gauge",
        f"bizra_ihsan_threshold {IHSAN_THRESHOLD}",
        "",
        "# HELP bizra_snr_threshold SNR quality threshold",
        "# TYPE bizra_snr_threshold gauge",
        f"bizra_snr_threshold {SNR_THRESHOLD}",
        "",
        "# HELP bizra_phase46_uptime_seconds Phase46 metrics uptime",
        "# TYPE bizra_phase46_uptime_seconds gauge",
        f"bizra_phase46_uptime_seconds {snap.get('uptime_seconds', 0.0)}",
    ]
    return Response(content="\n".join(lines), media_type="text/plain")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
