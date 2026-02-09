"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    █████╗ ██████╗ ██╗    ███████╗███████╗██████╗ ██╗   ██╗███████╗██████╗   ║
║   ██╔══██╗██╔══██╗██║    ██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗  ║
║   ███████║██████╔╝██║    ███████╗█████╗  ██████╔╝██║   ██║█████╗  ██████╔╝  ║
║   ██╔══██║██╔═══╝ ██║    ╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██╔══╝  ██╔══██╗  ║
║   ██║  ██║██║     ██║    ███████║███████╗██║  ██║ ╚████╔╝ ███████╗██║  ██║  ║
║   ╚═╝  ╚═╝╚═╝     ╚═╝    ╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝  ║
║                                                                              ║
║                    SOVEREIGN API SERVER v1.0                                 ║
║         REST + WebSocket Interface for External Integration                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Endpoints:                                                                 ║
║   ══════════                                                                 ║
║   POST   /v1/query          - Submit a query                                 ║
║   GET    /v1/status         - Get runtime status                             ║
║   GET    /v1/health         - Health check (for load balancers)              ║
║   GET    /v1/metrics        - Prometheus-compatible metrics                  ║
║   WS     /v1/stream         - WebSocket streaming interface                  ║
║                                                                              ║
║   Authentication: Bearer token via X-API-Key header                          ║
║   Rate Limiting: 100 req/min per API key                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("sovereign.api")

# =============================================================================
# SECURITY LIMITS
# =============================================================================
MAX_BODY_SIZE: int = 1_048_576  # 1 MiB — reject payloads above this
MAX_QUERY_LENGTH: int = 10_000  # characters
MAX_CONTEXT_KEYS: int = 50
MAX_DEPTH_LIMIT: int = 10
MAX_TIMEOUT_MS: int = 120_000  # 2 minutes

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


@dataclass
class QueryRequest:
    """API query request model."""

    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    # Options
    require_reasoning: bool = True
    require_validation: bool = True
    max_depth: int = 3
    timeout_ms: int = 30000
    stream: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryRequest":
        return cls(
            query=data.get("query", ""),
            context=data.get("context", {}),
            options=data.get("options", {}),
            require_reasoning=data.get("require_reasoning", True),
            require_validation=data.get("require_validation", True),
            max_depth=data.get("max_depth", 3),
            timeout_ms=data.get("timeout_ms", 30000),
            stream=data.get("stream", False),
        )


@dataclass
class QueryResponse:
    """API query response model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    success: bool = False
    answer: str = ""
    confidence: float = 0.0
    reasoning_path: List[str] = field(default_factory=list)

    # Quality
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    guardian_verdict: str = ""

    # Timing
    total_time_ms: float = 0.0

    # Error
    error: Optional[str] = None

    # Metadata
    model: str = "sovereign-v1"
    cached: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "success": self.success,
            "answer": self.answer,
            "confidence": self.confidence,
            "reasoning_path": self.reasoning_path,
            "quality": {
                "snr": self.snr_score,
                "ihsan": self.ihsan_score,
                "verdict": self.guardian_verdict,
            },
            "timing": {
                "total_ms": self.total_time_ms,
            },
            "error": self.error,
            "metadata": {
                "model": self.model,
                "cached": self.cached,
                "timestamp": self.timestamp,
            },
        }


@dataclass
class HealthResponse:
    """Health check response."""

    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float = 0.0
    checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsResponse:
    """Prometheus-compatible metrics."""

    metrics: List[str] = field(default_factory=list)

    def to_prometheus(self) -> str:
        return "\n".join(self.metrics)


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 100, burst_size: int = 10):
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.burst = burst_size
        self.buckets: Dict[str, Dict[str, float]] = {}

    def check(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()

        if key not in self.buckets:
            self.buckets[key] = {"tokens": self.burst, "last": now}
            return True

        bucket = self.buckets[key]
        elapsed = now - bucket["last"]
        bucket["tokens"] = min(self.burst, bucket["tokens"] + elapsed * self.rate)
        bucket["last"] = now

        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

    def remaining(self, key: str) -> int:
        """Get remaining tokens for a key."""
        if key not in self.buckets:
            return self.burst
        return int(self.buckets[key]["tokens"])


# =============================================================================
# API SERVER (Pure asyncio, no external dependencies)
# =============================================================================


class SovereignAPIServer:
    """
    Lightweight API server built on pure asyncio.

    For production, consider using FastAPI/Starlette with:
        from sovereign.api import create_fastapi_app
        app = create_fastapi_app(runtime)
    """

    def __init__(
        self,
        runtime: Any,  # SovereignRuntime
        host: str = "0.0.0.0",  # nosec B104 — intentional: API server must be reachable from containers/WSL
        port: int = 8080,
        api_keys: Optional[Set[str]] = None,
    ):
        self.runtime = runtime
        self.host = host
        self.port = port
        self.api_keys = api_keys or set()
        self.rate_limiter = RateLimiter()

        self._server: Optional[asyncio.Server] = None
        self._request_count = 0
        self._start_time = time.time()

        # WebSocket connections
        self._ws_connections: Set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        """Start the API server."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self.port,
        )
        logger.info(f"Sovereign API Server listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the API server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Close WebSocket connections
        for writer in self._ws_connections:
            writer.close()

        logger.info("Sovereign API Server stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming HTTP connection."""
        try:
            # Read request line
            request_line = await reader.readline()
            if not request_line:
                return

            request_line = request_line.decode().strip()
            parts = request_line.split()
            if len(parts) < 2:
                return

            method, path = parts[0], parts[1]

            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if line == b"\r\n" or not line:
                    break
                if b":" in line:
                    key, value = line.decode().strip().split(":", 1)
                    headers[key.lower().strip()] = value.strip()

            # Read body if present — enforce MAX_BODY_SIZE to prevent OOM
            body = b""
            content_length = int(headers.get("content-length", 0))
            if content_length > MAX_BODY_SIZE:
                writer.write(
                    self._json_response(
                        {"error": f"Payload too large (max {MAX_BODY_SIZE} bytes)"}, 413
                    ).encode()
                )
                await writer.drain()
                return
            if content_length > 0:
                body = await reader.read(content_length)

            # Route request
            response = await self._route(method, path, headers, body)

            # Send response
            writer.write(response.encode() if isinstance(response, str) else response)
            await writer.drain()

        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            writer.close()

    async def _route(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: bytes,
    ) -> str:
        """Route request to handler."""
        self._request_count += 1

        # Check API key if configured
        if self.api_keys:
            api_key = headers.get("x-api-key", "")
            if api_key not in self.api_keys:
                return self._json_response({"error": "Unauthorized"}, 401)

        # Rate limiting
        client_key = headers.get("x-api-key", "anonymous")
        if not self.rate_limiter.check(client_key):
            return self._json_response({"error": "Rate limit exceeded"}, 429)

        # Route
        if path == "/v1/health" and method == "GET":
            return await self._handle_health()
        elif path == "/v1/status" and method == "GET":
            return await self._handle_status()
        elif path == "/v1/metrics" and method == "GET":
            return await self._handle_metrics()
        elif path == "/v1/query" and method == "POST":
            return await self._handle_query(body)
        else:
            return self._json_response({"error": "Not found"}, 404)

    async def _handle_health(self) -> str:
        """Handle health check."""
        status = self.runtime.status()
        health = HealthResponse(
            status=status["health"]["status"],
            version=status["identity"]["version"],
            uptime_seconds=time.time() - self._start_time,
            checks={
                "runtime": status["state"]["running"],
                "autonomous": status["autonomous"].get("running", False),
            },
        )
        return self._json_response(health.to_dict())

    async def _handle_status(self) -> str:
        """Handle status request."""
        status = self.runtime.status()
        return self._json_response(status)

    async def _handle_metrics(self) -> str:
        """Handle Prometheus metrics."""
        metrics = self.runtime.metrics

        lines = [
            "# HELP sovereign_queries_total Total queries processed",
            "# TYPE sovereign_queries_total counter",
            f"sovereign_queries_total {metrics.total_queries}",
            "",
            "# HELP sovereign_query_success_rate Query success rate",
            "# TYPE sovereign_query_success_rate gauge",
            f"sovereign_query_success_rate {metrics.success_rate():.4f}",
            "",
            "# HELP sovereign_snr_score Current SNR score",
            "# TYPE sovereign_snr_score gauge",
            f"sovereign_snr_score {metrics.current_snr:.4f}",
            "",
            "# HELP sovereign_ihsan_score Current Ihsan score",
            "# TYPE sovereign_ihsan_score gauge",
            f"sovereign_ihsan_score {metrics.current_ihsan:.4f}",
            "",
            "# HELP sovereign_avg_query_time_ms Average query time",
            "# TYPE sovereign_avg_query_time_ms gauge",
            f"sovereign_avg_query_time_ms {metrics.avg_query_time_ms:.2f}",
            "",
            "# HELP sovereign_health_score System health score",
            "# TYPE sovereign_health_score gauge",
            f"sovereign_health_score {metrics.health_score:.4f}",
        ]

        return self._text_response("\n".join(lines), content_type="text/plain")

    async def _handle_query(self, body: bytes) -> str:
        """Handle query request with input validation."""
        try:
            data = json.loads(body.decode()) if body else {}
            if not isinstance(data, dict):
                return self._json_response({"error": "Request body must be a JSON object"}, 400)

            request = QueryRequest.from_dict(data)

            # ── Input validation ───────────────────────────────────────────
            if not request.query:
                return self._json_response({"error": "Query required"}, 400)
            if len(request.query) > MAX_QUERY_LENGTH:
                return self._json_response(
                    {"error": f"Query too long (max {MAX_QUERY_LENGTH} chars)"}, 400
                )
            if len(request.context) > MAX_CONTEXT_KEYS:
                return self._json_response(
                    {"error": f"Too many context keys (max {MAX_CONTEXT_KEYS})"}, 400
                )
            if not (1 <= request.max_depth <= MAX_DEPTH_LIMIT):
                return self._json_response(
                    {"error": f"max_depth must be 1-{MAX_DEPTH_LIMIT}"}, 400
                )
            if not (1000 <= request.timeout_ms <= MAX_TIMEOUT_MS):
                return self._json_response(
                    {"error": f"timeout_ms must be 1000-{MAX_TIMEOUT_MS}"}, 400
                )

            result = await self.runtime.query(
                request.query,
                context=request.context,
                require_reasoning=request.require_reasoning,
                require_validation=request.require_validation,
                max_depth=request.max_depth,
                timeout_ms=request.timeout_ms,
            )

            response = QueryResponse(
                id=result.query_id,
                success=result.success,
                answer=result.answer,
                confidence=result.confidence,
                reasoning_path=result.reasoning_path,
                snr_score=result.snr_score,
                ihsan_score=result.ihsan_score,
                guardian_verdict=result.guardian_verdict,
                total_time_ms=result.total_time_ms,
                error=result.error,
                cached=result.cached,
            )

            return self._json_response(response.to_dict())

        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, 400)
        except Exception as e:
            logger.error(f"Query error: {e}")
            return self._json_response({"error": str(e)}, 500)

    def _json_response(self, data: Dict[str, Any], status: int = 200) -> str:
        """Build JSON HTTP response."""
        body = json.dumps(data)
        status_text = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            404: "Not Found",
            429: "Too Many Requests",
            500: "Internal Server Error",
        }

        return (
            f"HTTP/1.1 {status} {status_text.get(status, 'Unknown')}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"X-Request-Id: {uuid.uuid4().hex[:8]}\r\n"
            f"\r\n"
            f"{body}"
        )

    def _text_response(
        self, text: str, status: int = 200, content_type: str = "text/plain"
    ) -> str:
        """Build text HTTP response."""
        return (
            f"HTTP/1.1 {status} OK\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(text)}\r\n"
            f"\r\n"
            f"{text}"
        )


# =============================================================================
# FASTAPI INTEGRATION (Optional, for production)
# =============================================================================


def create_fastapi_app(runtime: Any) -> Any:
    """
    Create FastAPI application for production deployment.

    Usage:
        from sovereign.api import create_fastapi_app
        from sovereign.runtime import SovereignRuntime

        runtime = SovereignRuntime()
        app = create_fastapi_app(runtime)

        # Run with: uvicorn module:app --host 0.0.0.0 --port 8080
    """
    try:
        from fastapi import FastAPI, Header, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, PlainTextResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Sovereign API",
        description="BIZRA Sovereign Engine REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequestModel(BaseModel):
        query: str
        context: Dict[str, Any] = {}
        require_reasoning: bool = True
        require_validation: bool = True
        max_depth: int = 3
        timeout_ms: int = 30000

    @app.get("/v1/health")
    async def health():
        status = runtime.status()
        return {
            "status": status["health"]["status"],
            "version": status["identity"]["version"],
        }

    @app.get("/v1/status")
    async def status():
        return runtime.status()

    @app.get("/v1/metrics")
    async def metrics():
        m = runtime.metrics
        return PlainTextResponse(
            f"sovereign_queries_total {m.total_queries}\n"
            f"sovereign_snr_score {m.current_snr:.4f}\n"
            f"sovereign_ihsan_score {m.current_ihsan:.4f}\n"
        )

    @app.post("/v1/query")
    async def query(request: QueryRequestModel):
        result = await runtime.query(
            request.query,
            context=request.context,
            require_reasoning=request.require_reasoning,
            require_validation=request.require_validation,
            max_depth=request.max_depth,
            timeout_ms=request.timeout_ms,
        )

        return {
            "id": result.query_id,
            "success": result.success,
            "answer": result.answer,
            "confidence": result.confidence,
            "quality": {
                "snr": result.snr_score,
                "ihsan": result.ihsan_score,
            },
            "timing": {
                "total_ms": result.total_time_ms,
            },
        }

    return app


# =============================================================================
# CLI SERVER
# =============================================================================


async def serve(
    host: str = "0.0.0.0",  # nosec B104 — intentional: server default for local network access
    port: int = 8080,
    api_keys: Optional[List[str]] = None,
) -> None:
    """
    Run the Sovereign API server.

    Usage:
        python -m core.sovereign.api --port 8080
    """
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(autonomous_enabled=True)

    async with SovereignRuntime.create(config) as runtime:
        server = SovereignAPIServer(
            runtime=runtime,
            host=host,
            port=port,
            api_keys=set(api_keys) if api_keys else None,
        )

        await server.start()

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              SOVEREIGN API SERVER RUNNING                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Endpoints:                                                 ║
║   ──────────                                                 ║
║   GET  http://{host}:{port}/v1/health                          ║
║   GET  http://{host}:{port}/v1/status                          ║
║   GET  http://{host}:{port}/v1/metrics                         ║
║   POST http://{host}:{port}/v1/query                           ║
║                                                              ║
║   Example:                                                   ║
║   ────────                                                   ║
║   curl -X POST http://localhost:{port}/v1/query \\            ║
║     -H "Content-Type: application/json" \\                   ║
║     -d '{{"query": "What is sovereignty?"}}'                  ║
║                                                              ║
║   Press Ctrl+C to stop                                       ║
╚══════════════════════════════════════════════════════════════╝
        """)

        await runtime.wait_for_shutdown()
        await server.stop()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SovereignAPIServer",
    "QueryRequest",
    "QueryResponse",
    "HealthResponse",
    "RateLimiter",
    "create_fastapi_app",
    "serve",
]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sovereign API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")  # nosec B104 — intentional: CLI default for API server
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--api-key", action="append", help="API keys (can repeat)")

    args = parser.parse_args()

    asyncio.run(serve(args.host, args.port, args.api_key))
