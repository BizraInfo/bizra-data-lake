# BIZRA Production API v1.0
# RESTful Interface for the Production Runtime Engine
# Provides HTTP endpoints for query execution, health monitoring, and metrics
#
# Standing on Giants: FastAPI patterns, OpenAPI spec, async excellence

import asyncio
import json
import time
import uvicorn
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Query as QueryParam
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn")

# BIZRA imports
from bizra_config import SNR_THRESHOLD, IHSAN_CONSTRAINT
from bizra_runtime import (
    BIZRARuntime, LoadBalanceStrategy, QueryResult,
    RuntimeMetrics, BackendStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | API | %(message)s'
)
logger = logging.getLogger("BIZRA-API")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query execution"""
    query: str = Field(..., description="The query text", min_length=1)
    capability: str = Field("text", description="Required capability: text, vision, reasoning, code")
    use_orchestrator: bool = Field(False, description="Use full orchestrator pipeline")
    max_tokens: int = Field(2048, description="Maximum tokens in response")
    temperature: float = Field(0.7, ge=0, le=2, description="Temperature for generation")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    image_path: Optional[str] = Field(None, description="Path to image for vision queries")


class QueryResponse(BaseModel):
    """Response model for query execution"""
    success: bool
    content: str
    backend_used: str
    model_used: str
    latency_ms: float
    snr_score: float
    tokens_used: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    backends_healthy: int
    backends_total: int
    snr_average: float


class BackendHealthResponse(BaseModel):
    """Backend health status"""
    name: str
    status: str
    url: str
    latency_ms: float
    models_count: int
    capabilities: List[str]


class MetricsResponse(BaseModel):
    """Runtime metrics response"""
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    snr_average: float
    ihsan_compliance: float


class VisionQueryRequest(BaseModel):
    """Request model for vision queries"""
    image_path: str = Field(..., description="Path to image file")
    prompt: Optional[str] = Field("Analyze this image in detail", description="Analysis prompt")
    max_tokens: int = Field(2048, description="Maximum tokens in response")


# ============================================================================
# RUNTIME INSTANCE
# ============================================================================

runtime: Optional[BIZRARuntime] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global runtime

    # Startup
    logger.info("Starting BIZRA API server...")
    runtime = BIZRARuntime(
        load_balance_strategy=LoadBalanceStrategy.LEAST_LATENCY,
        enable_auto_recovery=True
    )
    await runtime.start()
    logger.info("BIZRA API server started")

    yield

    # Shutdown
    logger.info("Shutting down BIZRA API server...")
    if runtime:
        await runtime.stop()
    logger.info("BIZRA API server stopped")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="BIZRA Production API",
        description="""
# BIZRA Production API

Elite-level REST interface for the BIZRA Data Lake Production Runtime.

## Features

- **Query Execution**: Execute text, vision, reasoning, and code queries
- **Health Monitoring**: Real-time backend health status
- **Metrics**: Performance analytics and telemetry
- **Load Balancing**: Intelligent routing across LM Studio and Ollama

## Backends

- **LM Studio**: Primary backend at http://192.168.56.1:1234
- **Ollama**: Fallback backend at http://localhost:11434

## SNR Thresholds

- **Ihsān Excellence**: SNR ≥ 0.99
- **Acceptable**: SNR ≥ 0.95
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # ========================================================================
    # HEALTH ENDPOINTS
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Quick health check endpoint.

        Returns basic system health status including uptime and backend availability.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        metrics = runtime.get_metrics()

        status = "healthy"
        if metrics.backends_healthy == 0:
            status = "unhealthy"
        elif metrics.backends_healthy < metrics.backends_total:
            status = "degraded"

        return HealthResponse(
            status=status,
            version=runtime.VERSION,
            uptime_seconds=metrics.uptime_seconds,
            backends_healthy=metrics.backends_healthy,
            backends_total=metrics.backends_total,
            snr_average=metrics.snr_average
        )

    @app.get("/health/backends", response_model=List[BackendHealthResponse], tags=["Health"])
    async def get_backend_health():
        """
        Get detailed health status of all backends.

        Returns latency, model count, and capabilities for each backend.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        all_health = runtime.health_monitor.get_all_health()

        return [
            BackendHealthResponse(
                name=name,
                status=health.status.value,
                url=health.url,
                latency_ms=health.latency_ms,
                models_count=len(health.available_models),
                capabilities=list(health.capabilities)
            )
            for name, health in all_health.items()
        ]

    @app.get("/health/live", tags=["Health"])
    async def liveness_probe():
        """
        Kubernetes-style liveness probe.

        Returns 200 if the service is alive.
        """
        return {"status": "alive"}

    @app.get("/health/ready", tags=["Health"])
    async def readiness_probe():
        """
        Kubernetes-style readiness probe.

        Returns 200 if the service is ready to accept requests.
        """
        if not runtime or not runtime._initialized:
            raise HTTPException(status_code=503, detail="Not ready")

        healthy_backends = runtime.health_monitor.get_healthy_backends()
        if not healthy_backends:
            raise HTTPException(status_code=503, detail="No healthy backends")

        return {"status": "ready", "backends": healthy_backends}

    # ========================================================================
    # QUERY ENDPOINTS
    # ========================================================================

    @app.post("/query", response_model=QueryResponse, tags=["Query"])
    async def execute_query(request: QueryRequest):
        """
        Execute a query through the BIZRA runtime.

        Supports text, vision, reasoning, and code capabilities.
        Automatically routes to the best available backend.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        result = await runtime.execute_query(
            query=request.query,
            capability=request.capability,
            use_orchestrator=request.use_orchestrator,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            image_path=request.image_path
        )

        return QueryResponse(
            success=result.success,
            content=result.content,
            backend_used=result.backend_used,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            snr_score=result.snr_score,
            tokens_used=result.tokens_used,
            error=result.error,
            metadata=result.metadata
        )

    @app.post("/query/vision", response_model=QueryResponse, tags=["Query"])
    async def execute_vision_query(request: VisionQueryRequest):
        """
        Execute a vision query on an image.

        Analyzes the image using available vision models (LLaVA, Qwen-VL, etc.)
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        # Verify image exists
        if not Path(request.image_path).exists():
            raise HTTPException(status_code=400, detail=f"Image not found: {request.image_path}")

        result = await runtime.execute_query(
            query=request.prompt,
            capability="vision",
            use_orchestrator=False,
            max_tokens=request.max_tokens,
            image_path=request.image_path
        )

        return QueryResponse(
            success=result.success,
            content=result.content,
            backend_used=result.backend_used,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            snr_score=result.snr_score,
            tokens_used=result.tokens_used,
            error=result.error,
            metadata=result.metadata
        )

    @app.get("/query/capabilities", tags=["Query"])
    async def get_capabilities():
        """
        Get available query capabilities based on current backend health.

        Returns which capabilities (text, vision, code, reasoning) are available.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        all_health = runtime.health_monitor.get_all_health()

        available_capabilities = set()
        for health in all_health.values():
            if health.is_available():
                available_capabilities.update(health.capabilities)

        return {
            "capabilities": list(available_capabilities),
            "primary_backend": runtime.load_balancer.select_backend("text"),
            "vision_backend": runtime.health_monitor.get_backend_for_capability("vision")
        }

    # ========================================================================
    # METRICS ENDPOINTS
    # ========================================================================

    @app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
    async def get_metrics():
        """
        Get runtime performance metrics.

        Includes latency percentiles, success rate, and SNR statistics.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        metrics = runtime.get_metrics()

        success_rate = (
            metrics.successful_requests / max(metrics.total_requests, 1)
        )

        return MetricsResponse(
            uptime_seconds=metrics.uptime_seconds,
            total_requests=metrics.total_requests,
            successful_requests=metrics.successful_requests,
            failed_requests=metrics.failed_requests,
            success_rate=success_rate,
            avg_latency_ms=metrics.avg_latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
            p99_latency_ms=metrics.p99_latency_ms,
            snr_average=metrics.snr_average,
            ihsan_compliance=metrics.ihsan_compliance
        )

    @app.get("/metrics/prometheus", tags=["Metrics"])
    async def get_prometheus_metrics():
        """
        Get metrics in Prometheus format.

        Compatible with Prometheus scraping for production monitoring.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        metrics = runtime.get_metrics()

        prometheus_output = f"""# HELP bizra_uptime_seconds Total uptime in seconds
# TYPE bizra_uptime_seconds gauge
bizra_uptime_seconds {metrics.uptime_seconds}

# HELP bizra_requests_total Total number of requests
# TYPE bizra_requests_total counter
bizra_requests_total {metrics.total_requests}

# HELP bizra_requests_successful Successful requests
# TYPE bizra_requests_successful counter
bizra_requests_successful {metrics.successful_requests}

# HELP bizra_requests_failed Failed requests
# TYPE bizra_requests_failed counter
bizra_requests_failed {metrics.failed_requests}

# HELP bizra_latency_avg_ms Average latency in milliseconds
# TYPE bizra_latency_avg_ms gauge
bizra_latency_avg_ms {metrics.avg_latency_ms}

# HELP bizra_latency_p95_ms P95 latency in milliseconds
# TYPE bizra_latency_p95_ms gauge
bizra_latency_p95_ms {metrics.p95_latency_ms}

# HELP bizra_latency_p99_ms P99 latency in milliseconds
# TYPE bizra_latency_p99_ms gauge
bizra_latency_p99_ms {metrics.p99_latency_ms}

# HELP bizra_snr_average Average SNR score
# TYPE bizra_snr_average gauge
bizra_snr_average {metrics.snr_average}

# HELP bizra_ihsan_compliance Ihsan compliance rate (SNR >= 0.99)
# TYPE bizra_ihsan_compliance gauge
bizra_ihsan_compliance {metrics.ihsan_compliance}

# HELP bizra_backends_healthy Number of healthy backends
# TYPE bizra_backends_healthy gauge
bizra_backends_healthy {metrics.backends_healthy}

# HELP bizra_backends_total Total number of backends
# TYPE bizra_backends_total gauge
bizra_backends_total {metrics.backends_total}
"""
        return StreamingResponse(
            iter([prometheus_output]),
            media_type="text/plain"
        )

    # ========================================================================
    # STATUS ENDPOINTS
    # ========================================================================

    @app.get("/status", tags=["Status"])
    async def get_status():
        """
        Get comprehensive system status report.

        Returns full status including runtime info, backend health, and metrics.
        """
        if not runtime:
            raise HTTPException(status_code=503, detail="Runtime not initialized")

        return runtime.get_status_report()

    @app.get("/", tags=["Root"])
    async def root():
        """
        API root endpoint.

        Returns basic API information and links.
        """
        return {
            "name": "BIZRA Production API",
            "version": "1.0.0",
            "description": "Elite-level REST interface for BIZRA Data Lake",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "metrics": "/metrics"
        }


# ============================================================================
# MAIN
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the BIZRA API server"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return

    print()
    print("=" * 60)
    print("  BIZRA PRODUCTION API SERVER")
    print("  Elite-level REST Interface")
    print("=" * 60)
    print()
    print(f"  Server: http://{host}:{port}")
    print(f"  Docs:   http://{host}:{port}/docs")
    print(f"  ReDoc:  http://{host}:{port}/redoc")
    print()
    print("=" * 60)
    print()

    uvicorn.run(
        "bizra_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys

    host = "0.0.0.0"
    port = 8000

    # Parse command line args
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]

    run_server(host, port)
