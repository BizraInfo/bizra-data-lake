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
║   SEL (Experience Ledger):                                                   ║
║   GET    /v1/sel/episodes   - list episodes (paginated)                      ║
║   GET    /v1/sel/episodes/H - Get episode by hash                            ║
║   POST   /v1/sel/retrieve   - RIR retrieval by query                         ║
║   GET    /v1/sel/verify     - Verify chain integrity                         ║
║                                                                              ║
║   SJE (Judgment Telemetry — Phase A Observation):                            ║
║   GET    /v1/judgment/stats      - Verdict distribution + entropy            ║
║   GET    /v1/judgment/stability  - Stability check (is_stable)               ║
║   POST   /v1/judgment/simulate   - Epoch distribution simulation             ║
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
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("sovereign.api")

# Module-level ReflexCompiler singleton (lazy-initialized in /v1/plan)
_reflex_compiler: Any = None


def _env_truthy(var_name: str) -> bool:
    """Return True when an environment flag is explicitly enabled."""
    return os.environ.get(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# PYDANTIC MODELS (module-level for FastAPI schema generation)
# =============================================================================
try:
    from fastapi import Request  # Module-level for PEP 563 annotation resolution
    from pydantic import BaseModel as _PydanticBaseModel
    from pydantic import Field as _PydanticField

    class QueryRequestModel(_PydanticBaseModel):
        """FastAPI request model for /v1/query."""

        query: str
        context: dict[str, Any] = _PydanticField(default_factory=dict)
        require_reasoning: bool = True
        require_validation: bool = True
        max_depth: int = 3
        timeout_ms: int = (
            300000  # 5 min — reasoning models (R1) need extended think time
        )

    class OrchestrateRequestModel(_PydanticBaseModel):
        """FastAPI request model for /v1/orchestrate."""

        task: str
        context: dict[str, Any] = _PydanticField(default_factory=dict)
        max_agents: int = 5

    class ValidateRequestModel(_PydanticBaseModel):
        """FastAPI request model for /v1/validate."""

        content: str
        task: str
        level: str = "standard"  # minimal | standard | thorough | critical

    class EnvelopeVerifyModel(_PydanticBaseModel):
        """FastAPI request model for /v1/verify/envelope."""

        envelope: dict[str, Any]

    class ReceiptVerifyModel(_PydanticBaseModel):
        """FastAPI request model for /v1/verify/receipt."""

        receipt: dict[str, Any]

    class AuditLogVerifyModel(_PydanticBaseModel):
        """FastAPI request model for /v1/verify/audit-log."""

        entries: list[dict[str, Any]]

    class PoIReceiptVerifyModel(_PydanticBaseModel):
        """FastAPI request model for /v1/verify/poi."""

        receipt: dict[str, Any]

    # Phase 21: Auth request models
    class RegisterRequestModel(_PydanticBaseModel):
        """Registration request."""

        username: str
        email: str
        password: str
        accept_covenant: bool = False

    class LoginRequestModel(_PydanticBaseModel):
        """Login request."""

        username: str
        password: str

    class RefreshTokenModel(_PydanticBaseModel):
        """Token refresh request."""

        refresh_token: str

    class SELRetrieveModel(_PydanticBaseModel):
        """Request model for /v1/sel/retrieve — RIR-based episode retrieval."""

        query: str
        top_k: int = 5

    class MemorySearchModel(_PydanticBaseModel):
        """Request model for /v1/memory/search — AgentDB hybrid search."""

        query: str
        top_k: int = 10
        min_score: float = 0.1
        source: Optional[str] = None

    # Phase 31: Cognitive Fusion request model
    class CognitiveFuseModel(_PydanticBaseModel):
        """Request model for /v1/cognitive/fuse — direct cognitive fusion pipeline."""

        query: str
        context: dict[str, Any] = _PydanticField(default_factory=dict)

    # Phase 20: Spearpoint request models
    class SpearpointReproduceModel(_PydanticBaseModel):
        """Request model for /v1/spearpoint/reproduce — evaluation-first verification."""

        claim: str
        proposed_change: str = ""
        prompt: str = ""
        response: str = ""
        metrics: dict[str, Any] = _PydanticField(default_factory=dict)

    class SpearpointImproveModel(_PydanticBaseModel):
        """Request model for /v1/spearpoint/improve — innovation through evaluator gate."""

        observation: Optional[dict[str, Any]] = None
        top_k: int = 3

    class SpearpointPatternModel(_PydanticBaseModel):
        """Request model for /v1/spearpoint/pattern — pattern-aware research via Sci-Reasoning."""

        pattern_id: str
        claim_context: str = ""
        top_k: int = 3

    # Sprint 1→2 Bridge: Mission endpoint models (typed contracts for frontend codegen)
    class MissionPlanRequest(_PydanticBaseModel):
        """Request model for POST /v1/plan — sovereign mission submission."""

        description: str = _PydanticField(
            ...,
            min_length=1,
            max_length=10000,
            description="Natural-language mission objective.",
        )
        source: str = _PydanticField(
            default="api",
            description="Caller identity: 'terminal', 'api', 'test', 'ahk'.",
        )

    class ChannelResult(_PydanticBaseModel):
        """Per-channel execution result within a mission."""

        channel: str
        success: bool
        duration_ms: float

    class WalletDeltaResponse(_PydanticBaseModel):
        """Wallet balance change from a mission (Contract §8.1)."""

        seed: float = 0.0
        bloom: float = 0.0

    class ReflexDeltaResponse(_PydanticBaseModel):
        """Reflex state change from a mission (Contract §8.1)."""

        compiled: bool = False
        near_compile: bool = False
        compile_count: int = 0
        threshold: int = 3

    class MemoryDeltaResponse(_PydanticBaseModel):
        """Memory state change from a mission (Contract §8.1)."""

        episodic: int = 0
        semantic: int = 0
        procedural: int = 0

    class MissionPlanResponse(_PydanticBaseModel):
        """Response model for POST /v1/plan — receipted mission result.

        Contract §8.1: All fields always present, even if zero.
        """

        mission_id: str
        receipt_id: str = ""
        status: str = _PydanticField(
            description="COMPLETE | PARTIAL | FAILED | BLOCKED (Contract §8.1)",
        )
        synthesis: str
        ihsan_score: float = _PydanticField(
            description="Ihsan excellence score (0.0–1.0, gate at 0.95).",
        )
        snr_score: float = _PydanticField(
            description="Signal-to-noise ratio (0.0–1.0, minimum 0.85).",
        )
        duration_ms: float
        evidence_receipt_id: Optional[str] = None
        channels_executed: list[ChannelResult] = _PydanticField(default_factory=list)
        execution_path: str = _PydanticField(
            default="system_2",
            description="SYSTEM_1_CACHE_HIT | SYSTEM_2_NOVEL | MIXED",
        )
        wallet_delta: WalletDeltaResponse = _PydanticField(
            default_factory=WalletDeltaResponse,
        )
        reflex_delta: ReflexDeltaResponse = _PydanticField(
            default_factory=ReflexDeltaResponse,
        )
        memory_delta: MemoryDeltaResponse = _PydanticField(
            default_factory=MemoryDeltaResponse,
        )
        hash_chain_ref: str = ""
        action_count: int = 0
        reflex_pattern: str = ""
        reflex_latency_ms: float = 0.0
        comparison_s2_avg_ms: float = 0.0

    class OnboardingTeachRequest(_PydanticBaseModel):
        """Request model for POST /v1/onboarding/teach — user preference teaching."""

        topic: str
        content: str
        preference_type: str = "general"

    class JudgmentSimulateRequest(_PydanticBaseModel):
        """Request model for POST /v1/judgment/simulate — epoch distribution sim."""

        scenario: str = "default"
        epochs: int = _PydanticField(default=10, ge=1, le=100)

    class EpochSimulateModel(_PydanticBaseModel):
        """Request model for /v1/judgment/simulate — proportional epoch distribution."""

        impacts: list[dict[str, Any]] = _PydanticField(default_factory=list)
        epoch_cap: int = 1000

    # Rebuild models to resolve forward refs from `from __future__ import annotations`
    QueryRequestModel.model_rebuild()
    OrchestrateRequestModel.model_rebuild()
    ValidateRequestModel.model_rebuild()
    EnvelopeVerifyModel.model_rebuild()
    ReceiptVerifyModel.model_rebuild()
    AuditLogVerifyModel.model_rebuild()
    RegisterRequestModel.model_rebuild()
    LoginRequestModel.model_rebuild()
    RefreshTokenModel.model_rebuild()
    SELRetrieveModel.model_rebuild()
    CognitiveFuseModel.model_rebuild()
    SpearpointReproduceModel.model_rebuild()
    SpearpointImproveModel.model_rebuild()
    SpearpointPatternModel.model_rebuild()
    PoIReceiptVerifyModel.model_rebuild()
    MemorySearchModel.model_rebuild()
    MissionPlanRequest.model_rebuild()
    ChannelResult.model_rebuild()
    MissionPlanResponse.model_rebuild()
    OnboardingTeachRequest.model_rebuild()
    JudgmentSimulateRequest.model_rebuild()
    EpochSimulateModel.model_rebuild()

except ImportError:
    Request = None  # type: ignore[assignment,misc]
    QueryRequestModel = None  # type: ignore[assignment,misc]
    OrchestrateRequestModel = None  # type: ignore[assignment,misc]
    ValidateRequestModel = None  # type: ignore[assignment,misc]
    EnvelopeVerifyModel = None  # type: ignore[assignment,misc]
    ReceiptVerifyModel = None  # type: ignore[assignment,misc]
    AuditLogVerifyModel = None  # type: ignore[assignment,misc]
    PoIReceiptVerifyModel = None  # type: ignore[assignment,misc]
    RegisterRequestModel = None  # type: ignore[assignment,misc]
    LoginRequestModel = None  # type: ignore[assignment,misc]
    RefreshTokenModel = None  # type: ignore[assignment,misc]
    SELRetrieveModel = None  # type: ignore[assignment,misc]
    MemorySearchModel = None  # type: ignore[assignment,misc]
    CognitiveFuseModel = None  # type: ignore[assignment,misc]
    SpearpointReproduceModel = None  # type: ignore[assignment,misc]
    SpearpointImproveModel = None  # type: ignore[assignment,misc]
    SpearpointPatternModel = None  # type: ignore[assignment,misc]
    MissionPlanRequest = None  # type: ignore[assignment,misc]
    ChannelResult = None  # type: ignore[assignment,misc]
    MissionPlanResponse = None  # type: ignore[assignment,misc]
    OnboardingTeachRequest = None  # type: ignore[assignment,misc]
    JudgmentSimulateRequest = None  # type: ignore[assignment,misc]
    EpochSimulateModel = None  # type: ignore[assignment,misc]

# =============================================================================
# SECURITY LIMITS
# =============================================================================
MAX_BODY_SIZE: int = 1_048_576  # 1 MiB — reject payloads above this
MAX_QUERY_LENGTH: int = 10_000  # characters
MAX_CONTEXT_KEYS: int = 50
MAX_DEPTH_LIMIT: int = 10
MAX_TIMEOUT_MS: int = (
    600_000  # 10 minutes — reasoning models (R1/QwQ) need extended think time
)

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


@dataclass
class QueryRequest:
    """API query request model."""

    query: str
    context: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    # Options
    require_reasoning: bool = True
    require_validation: bool = True
    max_depth: int = 3
    timeout_ms: int = 30000
    stream: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryRequest":
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
    reasoning_path: list[str] = field(default_factory=list)

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

    def to_dict(self) -> dict[str, Any]:
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
    version: str = "1.2.0"
    uptime_seconds: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsResponse:
    """Prometheus-compatible metrics."""

    metrics: list[str] = field(default_factory=list)

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
        self.buckets: dict[str, dict[str, float]] = {}
        self._max_buckets = 10_000  # Evict stale entries to prevent OOM (SAPE-011)

    def check(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()

        if key not in self.buckets:
            self._evict_buckets(now, needed_slots=1)
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

    def _evict_buckets(self, now: float, needed_slots: int) -> None:
        """Keep the limiter bounded even during sustained fresh-cardinality traffic."""
        if len(self.buckets) + needed_slots <= self._max_buckets:
            return

        cutoff = now - 600  # 10 min idle = stale
        stale = [k for k, v in self.buckets.items() if v["last"] < cutoff]
        for key in stale:
            del self.buckets[key]

        overflow = len(self.buckets) + needed_slots - self._max_buckets
        if overflow <= 0:
            return

        oldest = sorted(self.buckets.items(), key=lambda item: item[1]["last"])
        for key, _ in oldest[:overflow]:
            del self.buckets[key]


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
        host: str = "127.0.0.1",
        port: int = 8080,
        api_keys: Optional[set[str]] = None,
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
        self._ws_connections: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        """Start the API server, auto-incrementing port on conflict."""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self._server = await asyncio.start_server(
                    self._handle_connection,
                    self.host,
                    self.port,
                    reuse_address=True,
                )
                logger.info(
                    f"Sovereign API Server listening on {self.host}:{self.port}"
                )
                return
            except OSError as e:
                if e.errno == 98 and attempt < max_attempts - 1:
                    logger.warning(f"Port {self.port} in use, trying {self.port + 1}")
                    self.port += 1
                else:
                    raise

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

            request_str = request_line.decode().strip()
            parts = request_str.split()
            if len(parts) < 2:
                return

            full_path = parts[1]
            # Parse querystring from URL (SAPE-005 fix)
            if "?" in full_path:
                path, qs = full_path.split("?", 1)
                params = dict(
                    pair.split("=", 1) for pair in qs.split("&") if "=" in pair
                )
            else:
                path = full_path
                params = {}
            method = parts[0]

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
            response = await self._route(method, path, headers, body, params)

            # Send response
            resp_bytes = response.encode() if isinstance(response, str) else response
            writer.write(resp_bytes)  # type: ignore[arg-type]
            await writer.drain()

        except Exception:
            logger.exception("Connection error")
        finally:
            writer.close()

    async def _route(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        params: dict[str, str] | None = None,
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
        elif path == "/v1/sel/episodes" and method == "GET":
            return await self._handle_sel_episodes()
        elif path == "/v1/sel/retrieve" and method == "POST":
            return await self._handle_sel_retrieve(body)
        elif path == "/v1/sel/verify" and method == "GET":
            return await self._handle_sel_verify()
        elif path.startswith("/v1/sel/episodes/") and method == "GET":
            episode_hash = path[len("/v1/sel/episodes/") :]
            return await self._handle_sel_episode_by_hash(episode_hash)
        elif path == "/v1/judgment/stats" and method == "GET":
            return await self._handle_judgment_stats()
        elif path == "/v1/judgment/stability" and method == "GET":
            return await self._handle_judgment_stability()
        elif path == "/v1/judgment/simulate" and method == "POST":
            return await self._handle_judgment_simulate(body)
        # Token endpoints
        elif path == "/v1/token/balance" and method == "GET":
            return await self._handle_token_balance(params)
        elif path == "/v1/token/supply" and method == "GET":
            return await self._handle_token_supply()
        elif path == "/v1/token/history" and method == "GET":
            return await self._handle_token_history(params)
        elif path == "/v1/token/verify" and method == "GET":
            return await self._handle_token_verify()
        # Seed Engine endpoints (Phase 71)
        elif path == "/v1/seed/potential" and method == "GET":
            return await self._handle_seed_potential()
        elif path == "/v1/seed/episodes" and method == "GET":
            return await self._handle_seed_episodes(params)
        else:
            return self._json_response({"error": "Not found"}, 404)

    async def _handle_health(self) -> str:
        """Handle health check."""
        status = self.runtime.status()
        pat_sat_chain = status.get("pat_sat", {}).get("negotiation_receipt_chain", {})

        checks = {
            "runtime": status["state"]["running"],
            "autonomous": status["autonomous"].get("running", False),
            "pat_sat_receipt_chain_verified": bool(
                pat_sat_chain.get("verified_end_to_end", False)
            ),
        }

        # Phase 70: Bus infrastructure health
        bus_state = getattr(self.runtime, "_bus_wiring_state", None)
        if bus_state is not None:
            checks["bus_infrastructure"] = getattr(bus_state, "all_ok", False)

        # Phase 71: Seed Engine health
        seed_engine = getattr(self.runtime, "_seed_engine", None)
        if seed_engine is not None:
            seed_health = seed_engine.health()
            checks["seed_engine"] = seed_health.get("active", False)
            checks["seed_tier"] = seed_health.get("tier", "SEED")

        health = HealthResponse(
            status=status["health"]["status"],
            version=status["identity"]["version"],
            uptime_seconds=time.time() - self._start_time,
            checks=checks,
        )
        return self._json_response(health.to_dict())

    async def _handle_status(self) -> str:
        """Handle status request."""
        status = self.runtime.status()
        return self._json_response(status)

    async def _handle_metrics(self) -> str:
        """Handle Prometheus metrics."""
        metrics = self.runtime.metrics
        return self._text_response(
            metrics.to_prometheus(include_help=True),
            content_type="text/plain",
        )

    async def _handle_query(self, body: bytes) -> str:
        """Handle query request with input validation."""
        try:
            data = json.loads(body.decode()) if body else {}
            if not isinstance(data, dict):
                return self._json_response(
                    {"error": "Request body must be a JSON object"}, 400
                )

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
        except Exception:
            logger.exception("Query error")
            return self._json_response({"error": "Internal server error"}, 500)

    async def _handle_sel_episodes(self) -> str:
        """Handle SEL episodes listing."""
        sel = getattr(self.runtime, "_experience_ledger", None)
        if sel is None:
            return self._json_response(
                {"error": "Experience Ledger not initialized"}, 404
            )
        total = len(sel)
        episodes = []
        for i in range(min(50, total) - 1, -1, -1):
            ep = sel.get_by_sequence(i)
            if ep is not None:
                episodes.append(ep.to_dict())
        return self._json_response(
            {"total": total, "count": len(episodes), "episodes": episodes}
        )

    async def _handle_sel_episode_by_hash(self, episode_hash: str) -> str:
        """Handle SEL episode lookup by hash."""
        sel = getattr(self.runtime, "_experience_ledger", None)
        if sel is None:
            return self._json_response(
                {"error": "Experience Ledger not initialized"}, 404
            )
        ep = sel.get_by_hash(episode_hash)
        if ep is None:
            return self._json_response({"error": "Episode not found"}, 404)
        return self._json_response(ep.to_dict())

    async def _handle_sel_retrieve(self, body: bytes) -> str:
        """Handle SEL RIR retrieval."""
        sel = getattr(self.runtime, "_experience_ledger", None)
        if sel is None:
            return self._json_response(
                {"error": "Experience Ledger not initialized"}, 404
            )
        try:
            data = json.loads(body.decode()) if body else {}
            query_text = data.get("query", "")
            top_k = max(1, min(data.get("top_k", 5), 100))
            if not query_text:
                return self._json_response({"error": "Query text required"}, 400)
            results = sel.retrieve(query_text, top_k=top_k)
            return self._json_response(
                {
                    "query": query_text,
                    "top_k": top_k,
                    "count": len(results),
                    "episodes": [ep.to_dict() for ep in results],
                }
            )
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, 400)

    async def _handle_sel_verify(self) -> str:
        """Handle SEL chain verification."""
        sel = getattr(self.runtime, "_experience_ledger", None)
        if sel is None:
            return self._json_response(
                {"error": "Experience Ledger not initialized"}, 404
            )
        is_valid = sel.verify_chain_integrity()
        return self._json_response(
            {
                "valid": is_valid,
                "episodes": len(sel),
                "sequence": sel.sequence,
                "chain_head": (
                    sel.chain_head[:16] + "..."
                    if len(sel.chain_head) > 16
                    else sel.chain_head
                ),
            }
        )

    async def _handle_judgment_stats(self) -> str:
        """Handle SJE telemetry stats — verdict distribution + entropy."""
        jt = getattr(self.runtime, "_judgment_telemetry", None)
        if jt is None:
            return self._json_response(
                {"error": "Judgment Telemetry not initialized"}, 404
            )
        return self._json_response(jt.to_dict())

    async def _handle_judgment_stability(self) -> str:
        """Handle SJE stability check."""
        jt = getattr(self.runtime, "_judgment_telemetry", None)
        if jt is None:
            return self._json_response(
                {"error": "Judgment Telemetry not initialized"}, 404
            )
        return self._json_response(
            {
                "is_stable": jt.is_stable(),
                "entropy": round(jt.entropy(), 6),
                "total_observations": jt.total_observations,
                "dominant_verdict": (
                    jt.dominant_verdict().value if jt.dominant_verdict() else None
                ),
            }
        )

    async def _handle_judgment_simulate(self, body: bytes) -> str:
        """Handle epoch distribution simulation."""
        try:
            from core.sovereign.judgment_telemetry import simulate_epoch_distribution

            data = json.loads(body.decode()) if body else {}
            impacts = data.get("impacts", [])
            epoch_cap = data.get("epoch_cap", 1000)
            if not isinstance(impacts, list) or not isinstance(epoch_cap, int):
                return self._json_response(
                    {"error": "impacts must be list, epoch_cap must be int"}, 400
                )
            result = simulate_epoch_distribution(impacts, epoch_cap)
            return self._json_response(
                {
                    "impacts": impacts,
                    "epoch_cap": epoch_cap,
                    "allocations": result,
                    "dust": epoch_cap - sum(result),
                }
            )
        except json.JSONDecodeError:
            return self._json_response({"error": "Invalid JSON"}, 400)

    # =========================================================================
    # TOKEN ENDPOINTS
    # =========================================================================

    async def _handle_token_balance(self, params: dict[str, str]) -> str:
        """GET /v1/token/balance?account=BIZRA-00000000"""
        try:
            from core.token.ledger import TokenLedger
            from core.token.types import TokenType

            account_id = params.get("account", "BIZRA-00000000")
            ledger = TokenLedger()
            balances = {}
            for tt in TokenType:
                bal = ledger.get_balance(account_id, tt)
                if bal.balance > 0 or bal.staked > 0:
                    balances[tt.value] = bal.to_dict()
            return self._json_response(
                {
                    "account": account_id,
                    "balances": balances,
                }
            )
        except Exception:
            logger.exception("Token balance error")
            return self._json_response({"error": "Internal server error"}, 500)

    async def _handle_token_supply(self) -> str:
        """GET /v1/token/supply"""
        try:
            from datetime import datetime, timezone

            from core.token.ledger import TokenLedger
            from core.token.types import SEED_SUPPLY_CAP_PER_YEAR, TokenType

            ledger = TokenLedger()
            year = datetime.now(timezone.utc).year
            supply = {}
            for tt in TokenType:
                total = ledger.get_total_supply(tt)
                if total > 0:
                    supply[tt.value] = {
                        "total_supply": total,
                        "yearly_minted": ledger.get_yearly_minted(tt, year),
                    }
            supply["SEED"]["yearly_cap"] = SEED_SUPPLY_CAP_PER_YEAR
            supply["SEED"]["yearly_remaining"] = (
                SEED_SUPPLY_CAP_PER_YEAR - supply["SEED"]["yearly_minted"]
            )
            return self._json_response(
                {
                    "year": year,
                    "supply": supply,
                    "ledger_sequence": ledger.sequence,
                }
            )
        except Exception:
            logger.exception("Token supply error")
            return self._json_response({"error": "Internal server error"}, 500)

    async def _handle_token_history(self, params: dict[str, str]) -> str:
        """GET /v1/token/history?account=BIZRA-00000000&limit=20"""
        try:
            from core.token.ledger import TokenLedger
            from core.token.types import TokenType

            account_id = params.get("account")
            token_type_str = params.get("token_type")
            limit = min(int(params.get("limit", "20")), 100)

            token_type = TokenType(token_type_str) if token_type_str else None
            ledger = TokenLedger()
            txns = ledger.get_transaction_history(
                account_id=account_id,
                token_type=token_type,
                limit=limit,
            )
            return self._json_response(
                {
                    "count": len(txns),
                    "transactions": [tx.to_dict() for tx in txns],
                }
            )
        except Exception:
            logger.exception("Token history error")
            return self._json_response({"error": "Internal server error"}, 500)

    async def _handle_token_verify(self) -> str:
        """GET /v1/token/verify — Verify token ledger hash chain."""
        try:
            from core.token.ledger import TokenLedger

            ledger = TokenLedger()
            valid, count, error = ledger.verify_chain()
            return self._json_response(
                {
                    "chain_valid": valid,
                    "transactions_verified": count,
                    "error": error,
                    "ledger_sequence": ledger.sequence,
                }
            )
        except Exception:
            logger.exception("Token verify error")
            return self._json_response({"error": "Internal server error"}, 500)

    async def _handle_seed_potential(self) -> str:
        """GET /v1/seed/potential — Node's growth trajectory and unlocked capacity."""
        seed_engine = getattr(self.runtime, "_seed_engine", None)
        if seed_engine is None:
            return self._json_response({"error": "Seed engine not initialized"}, 503)

        try:
            from dataclasses import asdict

            potential = seed_engine.potential()
            return self._json_response(asdict(potential))
        except Exception:
            logger.exception("Seed potential error")
            return self._json_response({"error": "Internal server error"}, 500)

    async def _handle_seed_episodes(self, params: dict[str, str]) -> str:
        """GET /v1/seed/episodes — Recent growth episodes with receipt hashes."""
        seed_engine = getattr(self.runtime, "_seed_engine", None)
        if seed_engine is None:
            return self._json_response({"error": "Seed engine not initialized"}, 503)

        try:
            limit = min(int(params.get("limit", "10")), 100)
        except (ValueError, TypeError):
            limit = 10

        try:
            episodes = seed_engine.recent_episodes(limit=limit)
            return self._json_response({"count": len(episodes), "episodes": episodes})
        except Exception:
            logger.exception("Seed episodes error")
            return self._json_response({"error": "Internal server error"}, 500)

    def _json_response(self, data: dict[str, Any], status: int = 200) -> str:
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
            f"Content-type: application/json\r\n"
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
            f"Content-type: {content_type}\r\n"
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

    Phase 21: Now includes auth endpoints (/v1/auth/register, /v1/auth/login, /v1/auth/refresh).

    Usage:
        from sovereign.api import create_fastapi_app
        from sovereign.runtime import SovereignRuntime

        runtime = SovereignRuntime()
        app = create_fastapi_app(runtime)

        # Run with: uvicorn module:app --host 127.0.0.1 --port 8080
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, PlainTextResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

    # Phase 21: Initialize auth layer
    from pathlib import Path as _Path

    try:
        from core.auth.jwt_auth import JWTAuth
        from core.auth.middleware import AuthMiddleware, init_auth_middleware
        from core.auth.user_store import UserStore

        _state_dir = getattr(runtime, "config", None)
        _db_dir_raw = getattr(_state_dir, "state_dir", None) if _state_dir else None
        _db_dir = (
            _db_dir_raw if isinstance(_db_dir_raw, _Path) else _Path("sovereign_state")
        )
        _user_store = UserStore(db_path=_db_dir / "users.db")
        _jwt_auth = JWTAuth()
        _auth_middleware = AuthMiddleware(user_store=_user_store, jwt_auth=_jwt_auth)
        init_auth_middleware(_auth_middleware)
        _auth_available = True
        logger.info("Phase 21: Auth layer initialized (UserStore + JWT + Middleware)")
    except Exception as e:
        logger.error(
            "SECURITY: Auth layer failed to initialize: %s. "
            "Protected endpoints will deny requests until auth is restored.",
            e,
        )
        _user_store = None  # type: ignore[assignment]
        _jwt_auth = None  # type: ignore[assignment]
        _auth_middleware = None  # type: ignore[assignment]
        _auth_available = False

    # ── Constitutional Heartbeat Background Task ──────────────────
    # Phase 73: Schedule the 12-step constitutional tick as a periodic
    # background task. This activates: minting, zakat, demurrage, Gini
    # enforcement, reflex compilation, bloom accrual/decay, and event logging.
    #
    # Standing on Giants:
    # - Nakamoto (2008): Block processing tick
    # - Al-Khwarizmi (780-850): Deterministic procedure on schedule

    _tick_interval_s = int(os.environ.get("BIZRA_TICK_INTERVAL_S", "60"))
    _tick_task: list[Any] = []  # mutable container for background task ref

    async def _emit_bus_event(
        topic: str, payload: dict[str, Any], source: str = "heartbeat"
    ) -> None:
        """Emit an event to the sovereign EventBus (fire-and-forget)."""
        try:
            from core.sovereign.event_bus import get_event_bus

            bus = get_event_bus()
            await bus.emit(topic=topic, payload=payload, source=source)
        except Exception:
            logger.debug("EventBus emit failed for topic=%s", topic, exc_info=True)

    async def _emit_tick_events(result: Any, reflex_cache: dict[bytes, Any]) -> None:
        """Emit bus events for constitutional tick outcomes.

        Activates topics: economy.seed_minted, economy.bloom_accrued,
        economy.zakat, economy.asabiyyah, reflex.compiled.
        """
        from core.constitutional.fixed_point import fp_float

        if result.total_minted > 0:
            await _emit_bus_event(
                "economy.seed_minted",
                {
                    "minted": fp_float(result.total_minted),
                    "scored": result.scored,
                    "rejected": result.rejected,
                },
            )
        if result.scored > 0:
            await _emit_bus_event(
                "economy.bloom_accrued",
                {"scored": result.scored},
            )
        if result.zakat_pool > 0:
            await _emit_bus_event(
                "economy.zakat",
                {"zakat_pool": fp_float(result.zakat_pool)},
            )
        if result.network_asabiyyah_score > 0:
            await _emit_bus_event(
                "economy.asabiyyah",
                {
                    "asabiyyah": fp_float(result.network_asabiyyah_score),
                    "gini": fp_float(result.network_gini),
                },
            )
        if reflex_cache:
            await _emit_bus_event(
                "reflex.compiled",
                {"count": len(reflex_cache)},
            )

        # ── CONSTITUTIONAL TIER-0: action.receipt ────────────────
        if result.scored > 0:
            await _emit_bus_event(
                "action.receipt",
                {
                    "scored": result.scored,
                    "rejected": result.rejected,
                    "minted": fp_float(result.total_minted),
                },
                source="tick",
            )

        # ── CONSTITUTIONAL TIER-0: ihsan.breach ──────────────────
        # Fire when receipts were rejected by the intent gate —
        # this is the safety-critical circuit breaker.
        if result.rejected > 0:
            await _emit_bus_event(
                "ihsan.breach",
                {
                    "rejected_count": result.rejected,
                    "scored_count": result.scored,
                    "severity": "warning" if result.scored > 0 else "critical",
                },
                source="tick",
            )

    async def _constitutional_heartbeat() -> None:
        """Background task: run constitutional tick every interval."""
        try:
            from core.constitutional.ticker import process_tick
        except ImportError:
            logger.warning("Constitutional ticker not available — heartbeat disabled")
            return

        logger.info("Constitutional heartbeat started (interval=%ds)", _tick_interval_s)
        while True:
            try:
                await asyncio.sleep(_tick_interval_s)

                wallets = getattr(runtime, "_constitutional_wallets", [])
                receipts = getattr(runtime, "_constitutional_receipts", [])
                proposals = getattr(runtime, "_constitutional_proposals", [])
                event_log = getattr(runtime, "_constitutional_event_log", [])
                reflex_cache = getattr(runtime, "_constitutional_reflex_cache", {})

                if not receipts and not wallets:
                    continue  # No work to do — skip tick

                result = process_tick(
                    wallets=wallets,
                    receipts=list(receipts),  # snapshot
                    proposals=proposals,
                    event_log=event_log,
                    reflex_cache=reflex_cache,
                )

                # Clear consumed receipts
                if hasattr(runtime, "_constitutional_receipts"):
                    runtime._constitutional_receipts = []

                # ── Emit bus events for tick outcomes ──────────
                # Activates dead topics: economy.*, reflex.compiled
                await _emit_tick_events(result, reflex_cache)

                if result.scored > 0 or result.total_minted > 0:
                    logger.info(
                        "Constitutional tick: scored=%d minted=%d reflexes=%d",
                        result.scored,
                        result.total_minted,
                        len(reflex_cache),
                    )
            except asyncio.CancelledError:
                logger.info("Constitutional heartbeat stopped")
                return
            except Exception:
                logger.exception("Constitutional tick error (will retry next interval)")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app_instance: Any):  # type: ignore[override]
        """FastAPI lifespan: start/stop constitutional heartbeat."""
        if _tick_interval_s > 0:
            task = asyncio.create_task(_constitutional_heartbeat())
            _tick_task.append(task)
        yield
        for t in _tick_task:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        lifespan=_lifespan,
        title="BIZRA Sovereign API",
        summary="Constitutional sovereign engine for decentralized agentic intelligence.",
        description=(
            "59-route REST + WebSocket API surface for the BIZRA sovereign runtime.\n\n"
            "**Route categories:**\n"
            "- **23 Public** — health, metrics, verification, token supply\n"
            "- **3 Bootstrap** — auth registration, login, refresh\n"
            "- **33 Authenticated** — query, mission, memory, orchestration\n\n"
            "**Constitutional thresholds:**\n"
            "- Ihsan (excellence): >= 0.95\n"
            "- SNR (signal quality): >= 0.85\n"
            "- ADL Gini (justice): <= 0.35\n\n"
            "**Golden path:** `POST /v1/plan` — submit mission, receive receipted result."
        ),
        version="1.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        contact={"name": "BIZRA", "url": "https://bizra.info"},
        openapi_tags=[
            {"name": "health", "description": "Health and observability probes."},
            {
                "name": "auth",
                "description": "Authentication: register, login, refresh.",
            },
            {
                "name": "mission",
                "description": "Sovereign mission planning and execution.",
            },
            {"name": "query", "description": "Knowledge query and reasoning."},
            {
                "name": "verification",
                "description": "Cryptographic receipt and chain verification.",
            },
            {"name": "memory", "description": "Semantic memory search and stats."},
            {
                "name": "economics",
                "description": "Token supply, balance, PoI/SAT epochs.",
            },
            {
                "name": "constitutional",
                "description": "Constitutional tick and status.",
            },
            {
                "name": "spearpoint",
                "description": "Benchmark evaluation and improvement.",
            },
            {"name": "cognitive", "description": "Cognitive fusion and status."},
            {"name": "experience", "description": "SEL episodes and judgment."},
            {
                "name": "sovereignty",
                "description": "Node value, lifecycle, network effect.",
            },
            {"name": "onboarding", "description": "User onboarding and teaching."},
        ],
    )

    # CORS — Phase 23: environment-aware origin restriction
    _cors_env = os.environ.get("BIZRA_CORS_ORIGINS", "")
    if _cors_env:
        allowed_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    else:
        allowed_origins = [
            "https://bizra.ai",
            "https://www.bizra.ai",
            "https://bizra.info",
            "https://www.bizra.info",
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:8080",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-type", "X-API-Key", "X-Request-ID"],
    )

    def _authenticate_http_request(
        request: Request,
    ) -> tuple[str, Any | None, JSONResponse | None]:
        """Authenticate an HTTP request with fail-closed defaults."""
        from core.auth.middleware import _anonymous_auth_allowed

        allow_anonymous = _anonymous_auth_allowed()

        if _auth_available and _auth_middleware is not None:
            try:
                user = _auth_middleware.authenticate_request(request)
            except Exception as e:
                if not allow_anonymous:
                    return (
                        "",
                        None,
                        JSONResponse(
                            status_code=401,
                            content={"error": "Authentication required"},
                        ),
                    )
                logger.warning(
                    "Auth extraction failed; anonymous fallback enabled: %s", e
                )
                return "", None, None

            if user is None:
                if not allow_anonymous:
                    return (
                        "",
                        None,
                        JSONResponse(
                            status_code=401,
                            content={"error": "Authentication required"},
                        ),
                    )
                return "", None, None

            if not _auth_middleware.check_rate_limit(user.user_id):
                return (
                    "",
                    None,
                    JSONResponse(
                        status_code=429,
                        content={"error": "Rate limit exceeded"},
                    ),
                )
            return user.user_id, user, None

        if not allow_anonymous:
            return (
                "",
                None,
                JSONResponse(
                    status_code=503,
                    content={"error": "Authentication service unavailable"},
                ),
            )
        return "", None, None

    async def _authorize_websocket(ws: "StarletteWS") -> tuple[str, bool]:
        """Authorize a WebSocket connection before accepting the session."""
        from core.auth.middleware import _anonymous_auth_allowed

        allow_anonymous = _anonymous_auth_allowed()

        if _auth_available and _auth_middleware is not None:
            try:
                user = _auth_middleware.authenticate(
                    authorization=ws.headers.get("authorization"),
                    api_key=ws.headers.get("x-api-key"),
                )
            except Exception:
                user = None

            if user is None and not allow_anonymous:
                await ws.close(code=4401, reason="Authentication required")
                return "", False

            if user is not None and not _auth_middleware.check_rate_limit(user.user_id):
                await ws.close(code=4429, reason="Rate limit exceeded")
                return "", False

            return (user.user_id if user is not None else ""), True

        if not allow_anonymous:
            await ws.close(code=1013, reason="Authentication service unavailable")
            return "", False

        return "", True

    # ── Mission → Constitutional Tick Bridge ──────────────────────────
    #
    # Converts completed mission results into ActionReceipts and submits
    # them to the constitutional tick queue. This wires the reflex cache:
    # mission → receipt → tick Step 10 → reflex compilation for ihsan ≥ 0.98.

    def _submit_mission_to_tick(rt: Any, mission_result: Any) -> None:
        """Bridge mission results into the constitutional tick queue."""
        try:
            import hashlib
            import time as _time

            from core.constitutional.fixed_point import fp
            from core.constitutional.types import ActionReceipt

            receipts = getattr(rt, "_constitutional_receipts", None)
            if receipts is None:
                return

            mission_id = getattr(mission_result, "mission_id", "") or ""
            ihsan = getattr(mission_result, "ihsan_score", 0.0) or 0.0
            snr = getattr(mission_result, "snr_score", 0.0) or 0.0

            receipt = ActionReceipt(
                receipt_id=hashlib.blake2b(
                    mission_id.encode(), digest_size=32
                ).digest(),
                actor_id=b"\x00" * 32,  # system actor
                action_type="mission",
                timestamp=int(_time.time() * 1000),
                intent_score=fp(min(1.0, snr)),
                efficiency_score=fp(min(1.0, snr)),
                impact_score=fp(min(1.0, ihsan)),
                reproducibility_score=fp(min(1.0, snr * 0.9)),
                oracle_signature=b"\x00" * 64,
                metadata_hash=hashlib.blake2b(
                    (mission_id + str(ihsan)).encode(), digest_size=32
                ).digest(),
            )
            receipts.append(receipt)

            # ── Constitutional Topic Activation ──────────────────
            # Emit action.intent (receipt queued) — TIER-0 CONSTITUTIONAL
            asyncio.get_running_loop().call_soon(
                lambda _mid=mission_id, _ih=ihsan, _sn=snr: asyncio.ensure_future(
                    _emit_bus_event(
                        "action.intent",
                        {
                            "mission_id": _mid,
                            "action_type": "mission",
                            "ihsan_score": _ih,
                            "snr_score": _sn,
                        },
                        source="tick_bridge",
                    )
                )
            )
        except Exception:
            # Never block mission return for tick wiring failure
            logger.debug("Tick bridge emission failed", exc_info=True)

    # ── Health Endpoint Tiering (Phase 60 Step 3) ─────────────────────
    #
    # Three tiers for K8s probe compatibility:
    #   /v1/health/live  — O(1), <5ms, liveness probe
    #   /v1/health/ready — 3 critical checks, <50ms, readiness probe
    #   /v1/health/deep  — full 11-subsystem audit, <500ms, startup probe
    #   /v1/health       — alias for /v1/health/ready (backward compat)
    #
    # Standing on Giants: Burns et al. (K8s Health Checking, 2015)

    _ALL_SUBSYSTEM_CHECKS = [
        ("graph_of_thoughts", "_graph_reasoner"),
        ("snr_maximizer", "_snr_optimizer"),
        ("guardian_council", "_guardian_council"),
        ("autonomous_loop", "_autonomous_loop"),
        ("cognitive_fusion", "_cognitive_fusion"),
        ("embedding_service", "_embedding_service"),
        ("memory_coordinator", "_memory_coordinator"),
        ("evidence_ledger", "_evidence_ledger"),
        ("rdve_engine", "_rdve_engine"),
        ("fate_gate", "_ihsan_watchdog"),
        ("sat_controller", "_sat_controller"),
    ]

    # Critical subsystems that must be active for readiness
    _CRITICAL_SUBSYSTEM_CHECKS = [
        ("evidence_ledger", "_evidence_ledger"),
        ("snr_maximizer", "_snr_optimizer"),
        ("guardian_council", "_guardian_council"),
    ]

    def _check_subsystems(checks: list[tuple[str, str]]) -> dict[str, str]:
        subsystems: dict[str, str] = {}
        for name, attr in checks:
            instance = getattr(runtime, attr, None)
            if instance is None:
                subsystems[name] = "unavailable"
            elif "Stub" in type(instance).__name__:
                subsystems[name] = "stub"
            else:
                subsystems[name] = "active"
        return subsystems

    @app.get("/v1/health/live")
    async def health_live():
        """Liveness probe — O(1), <5ms. Returns 200 if process is alive."""
        return {"status": "alive", "tier": "live"}

    @app.get("/v1/health/ready")
    async def health_ready():
        """Readiness probe — 3 critical checks, <50ms."""
        subsystems = _check_subsystems(_CRITICAL_SUBSYSTEM_CHECKS)
        all_ok = all(v == "active" for v in subsystems.values())
        status_code = 200 if all_ok else 503

        result = {
            "status": "ready" if all_ok else "not_ready",
            "tier": "ready",
            "critical_subsystems": subsystems,
        }

        # Phase 71: Seed Engine health
        seed_engine = getattr(runtime, "_seed_engine", None)
        if seed_engine is not None:
            result["seed_engine"] = seed_engine.health()

        # Phase 72: Node Value health
        node_value_engine = getattr(runtime, "_node_value_engine", None)
        if node_value_engine is not None:
            result["node_value"] = node_value_engine.health()
        else:
            result["node_value"] = {"active": False}

        from starlette.responses import JSONResponse

        return JSONResponse(content=result, status_code=status_code)

    @app.get("/v1/health/deep")
    async def health_deep():
        """Deep health — full 11-subsystem audit, <500ms. For startup probes."""
        status = runtime.status()
        strict_gate = status.get("health", {}).get("strict_gate", {})
        pat_sat_chain = status.get("pat_sat", {}).get("negotiation_receipt_chain", {})

        subsystems = _check_subsystems(_ALL_SUBSYSTEM_CHECKS)

        stub_count = sum(1 for v in subsystems.values() if v == "stub")
        unavailable_count = sum(1 for v in subsystems.values() if v == "unavailable")

        total = len(subsystems)
        active_count = sum(1 for v in subsystems.values() if v == "active")
        health_score = active_count / total if total > 0 else 0.0

        if health_score >= 0.8:
            health_status = "healthy"
        elif health_score >= 0.5:
            health_status = "degraded"
        else:
            health_status = "unhealthy"

        if strict_gate.get("enabled") and not strict_gate.get("passed", True):
            health_status = "unhealthy"

        # Phase 71: Seed Engine health
        seed_health = None
        seed_engine = getattr(runtime, "_seed_engine", None)
        if seed_engine is not None:
            seed_health = seed_engine.health()

        return {
            "status": health_status,
            "tier": "deep",
            "version": status["identity"]["version"],
            "health_score": round(health_score, 4),
            "subsystems": subsystems,
            "stub_count": stub_count,
            "unavailable_count": unavailable_count,
            "seed_engine": seed_health,
            "strict_gate": strict_gate,
            "pat_sat_receipt_chain": {
                "verified_end_to_end": bool(
                    pat_sat_chain.get("verified_end_to_end", False)
                ),
                "chain_valid": pat_sat_chain.get("chain_valid"),
                "total_negotiation_receipts": pat_sat_chain.get(
                    "total_negotiation_receipts", 0
                ),
                "latest_sequence": pat_sat_chain.get("latest_sequence"),
                "latest_entry_hash": pat_sat_chain.get("latest_entry_hash"),
                "latest_receipt_id": pat_sat_chain.get("latest_receipt_id"),
            },
        }

    @app.get("/v1/health", tags=["health"])
    async def health():
        """Backward-compatible alias — delegates to readiness check."""
        return await health_ready()

    @app.get("/v1/status", tags=["health"], summary="Runtime status snapshot")
    async def status():
        base = runtime.status()
        # Enrich with ReflexCompiler telemetry when available
        if _reflex_compiler is not None:
            try:
                base["reflex_compiler"] = _reflex_compiler.get_status()
            except Exception:
                pass
        return base

    @app.get(
        "/v1/reflex/status",
        tags=["reflex"],
        summary="Reflex compiler status",
        description="Returns System-1 cache statistics: hit rate, size, precipitations, invalidations.",
    )
    async def reflex_status(request: Request):
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        if _reflex_compiler is None:
            return {"status": "not_initialized", "size": 0}
        return _reflex_compiler.get_status()

    @app.get("/v1/metrics", tags=["health"], summary="Prometheus metrics")
    async def metrics():
        m = runtime.metrics
        return PlainTextResponse(m.to_prometheus(include_help=False))

    @app.post("/v1/query", tags=["query"], summary="Submit knowledge query")
    async def query(body: QueryRequestModel, request: Request):
        """Query endpoint — auth-aware when auth layer is available.

        Security: single handler prevents route-shadowing bypass (SAPE-001).
        Default behavior is fail-closed for missing/invalid auth.
        Anonymous access requires explicit BIZRA_AUTH_ALLOW_ANONYMOUS opt-in.
        """
        user_id, user, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        if not body.query:
            return JSONResponse(status_code=400, content={"error": "Query required"})
        if len(body.query) > MAX_QUERY_LENGTH:
            return JSONResponse(
                status_code=400,
                content={"error": f"Query too long (max {MAX_QUERY_LENGTH} chars)"},
            )
        if len(body.context) > MAX_CONTEXT_KEYS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Too many context keys (max {MAX_CONTEXT_KEYS})"},
            )
        if not (1 <= body.max_depth <= MAX_DEPTH_LIMIT):
            return JSONResponse(
                status_code=400,
                content={"error": f"max_depth must be 1-{MAX_DEPTH_LIMIT}"},
            )
        if not (1000 <= body.timeout_ms <= MAX_TIMEOUT_MS):
            return JSONResponse(
                status_code=400,
                content={"error": f"timeout_ms must be 1000-{MAX_TIMEOUT_MS}"},
            )
        if user is not None and _user_store is not None:
            _user_store.increment_query_count(user_id)

        try:
            result = await runtime.query(
                body.query,
                context=body.context,
                require_reasoning=body.require_reasoning,
                require_validation=body.require_validation,
                max_depth=body.max_depth,
                timeout_ms=body.timeout_ms,
                user_id=user_id,
            )
        except Exception:
            logger.exception("Query execution failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

        response: dict[str, Any] = {
            "id": result.query_id,
            "success": result.success,
            "answer": result.response,
            "quality": {
                "snr": result.snr_score,
                "ihsan": result.ihsan_score,
            },
            "timing": {
                "total_ms": result.processing_time_ms,
            },
        }
        if user_id:
            response["user_id"] = user_id
        # Spearpoint: include content-addressed graph hash when available
        if result.graph_hash:
            response["graph_hash"] = result.graph_hash

        # Spearpoint: include evidence ledger receipt reference
        ledger = getattr(runtime, "_evidence_ledger", None)
        if ledger and hasattr(ledger, "sequence") and ledger.sequence > 0:
            response["receipt"] = {
                "sequence": ledger.sequence,
                "chain_hash": ledger.last_hash[:16] + "...",
            }
        return response

    # /v1/validate — standalone content validation via SNR + Ihsān
    @app.post("/v1/validate")
    async def validate(body: ValidateRequestModel, request: Request):
        """Validate content quality using the sovereign SNR and Ihsān engines.

        Returns quality scores without executing a full query pipeline.
        Useful for TUI/CLI post-hoc validation of generated content.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            # Use the runtime's SNR optimizer for quality scoring
            snr_optimizer = getattr(runtime, "_snr_optimizer", None)
            snr_score = 0.0
            if snr_optimizer is not None:
                try:
                    snr_result = snr_optimizer.optimize(body.content)
                    # Handle both sync and async return
                    if hasattr(snr_result, "__await__"):
                        snr_result = await snr_result
                    snr_score = (
                        getattr(snr_result, "score", 0.0)
                        if hasattr(snr_result, "score")
                        else float(snr_result or 0)
                    )
                except Exception:
                    snr_score = 0.0

            # Use the runtime's constitutional validation for Ihsān scoring
            validate_fn = getattr(runtime, "_validate_constitutionally", None)
            ihsan_score = 0.0
            if validate_fn is not None:
                try:
                    ihsan_result = await validate_fn(body.content, {"task": body.task})
                    if isinstance(ihsan_result, tuple):
                        ihsan_score = ihsan_result[0]
                    else:
                        ihsan_score = float(ihsan_result or 0)
                except Exception:
                    ihsan_score = 0.0

            # Fallback: run a lightweight query with validation if engines unavailable
            if snr_score == 0.0 and ihsan_score == 0.0:
                result = await runtime.query(
                    f"Validate this content for task '{body.task}': {body.content[:500]}",
                    context={"_validation_mode": True},
                    require_reasoning=False,
                    require_validation=True,
                    timeout_ms=60000,
                )
                snr_score = result.snr_score
                ihsan_score = result.ihsan_score

            passed = snr_score >= 0.5 and ihsan_score >= 0.5
            level_thresholds = {
                "minimal": 0.5,
                "standard": 0.7,
                "thorough": 0.85,
                "critical": 0.95,
            }
            threshold = level_thresholds.get(body.level, 0.7)
            passed = ihsan_score >= threshold

            return {
                "passed": passed,
                "quality": {
                    "snr": snr_score,
                    "ihsan": ihsan_score,
                },
                "threshold": threshold,
                "level": body.level,
            }
        except Exception:
            logger.exception("Validation failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    # ─── Verification Endpoints (True Spearpoint) ─────────────────────
    # Standing on: Lamport (1982) — distributed verification
    # Merkle (1979) — hash chain integrity
    # Bernstein (2011) — Ed25519 signatures
    #
    # These endpoints expose existing cryptographic verification logic
    # as HTTP-callable surfaces — "truth that can't be verified
    # externally is not truth, it's internal belief."

    @app.post("/v1/verify/genesis")
    async def verify_genesis():
        """Verify Node0 genesis identity hash chain.

        Returns uniform VerifierResponse with genesis artifacts.
        Standing on: Lamport (event ordering), Bernstein (Ed25519).
        """
        from core.proof_engine.evidence_ledger import VerifierResponse

        try:
            import os
            from pathlib import Path

            from core.sovereign.genesis_identity import (
                load_genesis,
            )
            from core.sovereign.origin_guard import (
                NODE_ROLE_ENV,
                normalize_node_role,
                resolve_origin_snapshot,
                validate_genesis_chain,
            )

            _runtime_cfg = getattr(runtime, "config", None)
            _sd_raw = getattr(runtime, "_state_dir", None)
            if not isinstance(_sd_raw, _Path):
                _sd_raw = getattr(_runtime_cfg, "state_dir", None)
            if not isinstance(_sd_raw, _Path):
                _sd_raw = None
            state_dir = _sd_raw or _Path("sovereign_state")
            role = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
            hash_validated, reason = validate_genesis_chain(state_dir)
            if not hash_validated:
                return VerifierResponse.rejected(
                    reason_codes=["GENESIS_CHAIN_INVALID"],
                    artifacts={
                        "detail": reason,
                        "origin": resolve_origin_snapshot(state_dir, role),
                        "hash_validated": False,
                    },
                ).to_dict()

            genesis = load_genesis(state_dir)
            if genesis is None:
                return VerifierResponse.rejected(
                    reason_codes=["EVIDENCE_MISSING"],
                    artifacts={
                        "detail": "No genesis ceremony output found",
                        "origin": resolve_origin_snapshot(state_dir, role),
                        "hash_validated": False,
                    },
                ).to_dict()

            genesis_hash = (
                genesis.genesis_hash.hex()
                if isinstance(genesis.genesis_hash, bytes)
                else str(genesis.genesis_hash)
            )

            return VerifierResponse.approved(
                receipt_id=genesis_hash[:32],
                artifacts={
                    "identity": {
                        "node_id": genesis.identity.node_id,
                        "name": genesis.identity.name,
                        "public_key": genesis.identity.public_key,
                        "created_at": genesis.identity.created_at,
                    },
                    "hashes": {
                        "genesis_hash": genesis_hash,
                        "pat_team_hash": (
                            genesis.pat_team_hash.hex()
                            if isinstance(genesis.pat_team_hash, bytes)
                            else str(genesis.pat_team_hash)
                        ),
                        "sat_team_hash": (
                            genesis.sat_team_hash.hex()
                            if isinstance(genesis.sat_team_hash, bytes)
                            else str(genesis.sat_team_hash)
                        ),
                    },
                    "governance": {
                        "quorum": getattr(genesis, "quorum", 0.67),
                        "voting_period_hours": getattr(
                            genesis, "voting_period_hours", 72
                        ),
                        "upgrade_threshold": getattr(genesis, "upgrade_threshold", 0.8),
                    },
                    "agents": {
                        "pat_count": len(genesis.pat_team),
                        "sat_count": len(genesis.sat_team),
                    },
                    "origin": resolve_origin_snapshot(state_dir, "node0"),
                    "hash_validated": True,
                },
            ).to_dict()
        except Exception:
            logger.exception("Genesis verification failed")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": "Internal server error"},
                ).to_dict(),
            )

    @app.post("/v1/verify/envelope")
    async def verify_envelope(body: EnvelopeVerifyModel):
        """Verify a PCI envelope signature, freshness, and replay protection.

        Returns uniform VerifierResponse with check details in artifacts.
        Standing on: Bernstein (Ed25519), Lamport (replay protection).
        """
        from core.proof_engine.evidence_ledger import VerifierResponse

        try:
            from core.pci.envelope import PCIEnvelope

            envelope_json = body.envelope
            if not envelope_json:
                return JSONResponse(
                    status_code=400,
                    content=VerifierResponse.rejected(
                        reason_codes=["SCHEMA_VIOLATION"],
                        artifacts={"detail": "Envelope JSON body required"},
                    ).to_dict(),
                )

            envelope = PCIEnvelope.from_dict(envelope_json)

            # Run freshness check
            is_fresh, freshness_error = envelope.validate_freshness()
            if not is_fresh:
                return VerifierResponse.rejected(
                    reason_codes=["TIMESTAMP_STALE"],
                    artifacts={
                        "detail": freshness_error,
                        "checks": {
                            "signature": "skipped",
                            "freshness": "failed",
                            "replay": "skipped",
                        },
                    },
                ).to_dict()

            # Verify signature
            digest = envelope.compute_digest()
            sig_valid = False
            if envelope.signature and envelope.sender.public_key:
                from core.pci.crypto import verify_signature

                sig_valid = verify_signature(
                    digest,
                    envelope.signature.value,
                    envelope.sender.public_key,
                )

            # Check replay
            is_replay = envelope.is_replay()

            reason_codes = []
            if not sig_valid:
                reason_codes.append("SIGNATURE_INVALID")
            if is_replay:
                reason_codes.append("REPLAY_DETECTED")

            checks = {
                "signature": "passed" if sig_valid else "failed",
                "freshness": "passed",
                "replay": "clean" if not is_replay else "detected",
            }

            if reason_codes:
                return VerifierResponse.rejected(
                    reason_codes=reason_codes,
                    artifacts={
                        "checks": checks,
                        "envelope_id": envelope.envelope_id,
                        "digest": digest,
                    },
                ).to_dict()

            return VerifierResponse.approved(
                receipt_id=envelope.envelope_id,
                artifacts={
                    "checks": checks,
                    "envelope_id": envelope.envelope_id,
                    "digest": digest,
                },
            ).to_dict()
        except Exception:
            logger.exception("Envelope verification failed")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": "Internal server error"},
                ).to_dict(),
            )

    @app.post("/v1/verify/receipt")
    async def verify_receipt(body: ReceiptVerifyModel):
        """Verify a signed execution receipt.

        Returns uniform VerifierResponse with quality metrics in artifacts.
        """
        from core.proof_engine.evidence_ledger import VerifierResponse

        try:
            from core.proof_engine.receipt import (
                Receipt,
                ReceiptStatus,
                ReceiptVerifier,
            )

            receipt_json = body.receipt
            if not receipt_json:
                return JSONResponse(
                    status_code=400,
                    content=VerifierResponse.rejected(
                        reason_codes=["SCHEMA_VIOLATION"],
                        artifacts={"detail": "Receipt JSON body required"},
                    ).to_dict(),
                )

            status_value = str(receipt_json.get("status", "pending")).lower()
            try:
                receipt_status = ReceiptStatus(status_value)
            except ValueError:
                return VerifierResponse.rejected(
                    reason_codes=["SCHEMA_VIOLATION"],
                    receipt_id=receipt_json.get("receipt_id", ""),
                    artifacts={
                        "detail": f"Unknown receipt status: {status_value}",
                    },
                ).to_dict()

            def _decode_hex_field(field_name: str) -> bytes:
                raw_value = receipt_json.get(field_name, "")
                if isinstance(raw_value, bytes):
                    return raw_value
                if not isinstance(raw_value, str) or not raw_value:
                    raise ValueError(f"{field_name} is required")
                return bytes.fromhex(raw_value)

            try:
                query_digest = _decode_hex_field("query_digest")
                policy_digest = _decode_hex_field("policy_digest")
                payload_digest = _decode_hex_field("payload_digest")
                signature = _decode_hex_field("signature")
                signer_pubkey = _decode_hex_field("signer_pubkey")
            except ValueError as parse_err:
                return VerifierResponse.rejected(
                    reason_codes=["SCHEMA_VIOLATION"],
                    receipt_id=receipt_json.get("receipt_id", ""),
                    artifacts={"detail": str(parse_err)},
                ).to_dict()

            timestamp_raw = receipt_json.get("timestamp")
            if isinstance(timestamp_raw, str) and timestamp_raw:
                try:
                    receipt_timestamp = datetime.fromisoformat(
                        timestamp_raw.replace("Z", "+00:00")
                    )
                except ValueError:
                    return VerifierResponse.rejected(
                        reason_codes=["SCHEMA_VIOLATION"],
                        receipt_id=receipt_json.get("receipt_id", ""),
                        artifacts={"detail": "Invalid timestamp format"},
                    ).to_dict()
            else:
                receipt_timestamp = datetime.now(timezone.utc)

            receipt = Receipt(
                receipt_id=receipt_json.get("receipt_id", ""),
                status=receipt_status,
                query_digest=query_digest,
                policy_digest=policy_digest,
                payload_digest=payload_digest,
                snr=float(receipt_json.get("snr", 0.0)),
                ihsan_score=float(receipt_json.get("ihsan_score", 0.0)),
                gate_passed=receipt_json.get("gate_passed", ""),
                reason=receipt_json.get("reason"),
                signature=signature,
                signer_pubkey=signer_pubkey,
                timestamp=receipt_timestamp,
            )

            is_valid = False
            error_msg: str | None = "Signature verification failed"
            runtime_signer = getattr(runtime, "_node_signer", None)

            if runtime_signer is not None and hasattr(runtime_signer, "verify"):
                verifier = ReceiptVerifier(runtime_signer)
                is_valid, error_msg = verifier.verify(receipt)

                signer_pub = (
                    runtime_signer.public_key_bytes()
                    if hasattr(runtime_signer, "public_key_bytes")
                    else b""
                )
                if not is_valid and signer_pub and receipt.signer_pubkey != signer_pub:
                    from core.pci.crypto import (
                        verify_signature as verify_ed25519_signature,
                    )
                    from core.proof_engine.canonical import (
                        hex_digest as canonical_hex_digest,
                    )

                    digest_hex = canonical_hex_digest(receipt.body_bytes())
                    is_valid = verify_ed25519_signature(
                        digest_hex,
                        receipt.signature.hex(),
                        receipt.signer_pubkey.hex(),
                    )
                    error_msg = None if is_valid else "Invalid signature"
            else:
                from core.pci.crypto import verify_signature as verify_ed25519_signature
                from core.proof_engine.canonical import (
                    hex_digest as canonical_hex_digest,
                )

                digest_hex = canonical_hex_digest(receipt.body_bytes())
                is_valid = verify_ed25519_signature(
                    digest_hex,
                    receipt.signature.hex(),
                    receipt.signer_pubkey.hex(),
                )
                error_msg = None if is_valid else "Invalid signature"

            artifacts = {
                "receipt_id": receipt.receipt_id,
                "status": receipt.status.value,
                "quality": {"snr": receipt.snr, "ihsan": receipt.ihsan_score},
                "signature_verified": is_valid,
            }

            if is_valid:
                return VerifierResponse.approved(
                    receipt_id=receipt.receipt_id,
                    artifacts=artifacts,
                ).to_dict()

            reason_codes = ["SIGNATURE_INVALID"]
            if error_msg:
                artifacts["detail"] = error_msg
            return VerifierResponse.rejected(
                reason_codes=reason_codes,
                receipt_id=receipt.receipt_id,
                artifacts=artifacts,
            ).to_dict()
        except Exception:
            logger.exception("Receipt verification failed")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": "Internal server error"},
                ).to_dict(),
            )

    @app.post("/v1/verify/audit-log")
    async def verify_audit_log(body: AuditLogVerifyModel):
        """Verify tamper-evident audit log integrity.

        Returns uniform VerifierResponse with chain analysis in artifacts.
        Standing on: Merkle (1979) — hash chain integrity.
        """
        from core.proof_engine.evidence_ledger import VerifierResponse

        try:
            from core.sovereign.tamper_evident_log import (
                TamperEvidentEntry,
                TamperEvidentLog,
                detect_tampering,
            )

            entries = body.entries
            if not entries:
                return JSONResponse(
                    status_code=400,
                    content=VerifierResponse.rejected(
                        reason_codes=["SCHEMA_VIOLATION"],
                        artifacts={"detail": "Log entries list required"},
                    ).to_dict(),
                )

            report = detect_tampering(entries)

            artifacts = {
                "verification_ratio": report.verification_ratio,
                "total_entries": len(entries),
                "affected_sequences": report.affected_sequences,
                "first_invalid": report.first_invalid_sequence,
            }

            if report.is_tampered:
                return VerifierResponse.rejected(
                    reason_codes=["EVIDENCE_TAMPERED"],
                    artifacts=artifacts,
                ).to_dict()

            return VerifierResponse.approved(
                receipt_id=f"audit-{len(entries):06d}",
                artifacts=artifacts,
            ).to_dict()
        except Exception:
            logger.exception("Audit log verification failed")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": "Internal server error"},
                ).to_dict(),
            )

    @app.post("/v1/verify/ledger")
    async def verify_evidence_ledger():
        """Verify integrity of the Evidence Ledger hash chain.

        Returns uniform VerifierResponse with chain metrics in artifacts.
        Standing on: Merkle (1979) — hash chain tamper detection.
        """
        from core.proof_engine.evidence_ledger import VerifierResponse

        try:
            ledger = getattr(runtime, "_evidence_ledger", None)
            if ledger is None:
                return JSONResponse(
                    status_code=404,
                    content=VerifierResponse.rejected(
                        reason_codes=["EVIDENCE_MISSING"],
                        artifacts={"detail": "Evidence ledger is not initialized"},
                    ).to_dict(),
                )

            is_valid, errors = ledger.verify_chain()
            artifacts = {
                "entry_count": ledger.count(),
                "last_hash": (
                    ledger.last_hash[:16] + "..." if ledger.last_hash else None
                ),
                "errors": errors,
            }

            if is_valid:
                return VerifierResponse.approved(
                    receipt_id=f"ledger-{ledger.sequence:06d}",
                    artifacts=artifacts,
                ).to_dict()

            return VerifierResponse.rejected(
                reason_codes=["EVIDENCE_TAMPERED"],
                artifacts=artifacts,
            ).to_dict()
        except Exception:
            logger.exception("Ledger verification failed")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": "Internal server error"},
                ).to_dict(),
            )

    @app.get("/v1/artifacts/graph/{query_id}")
    async def get_graph_artifact(query_id: str):
        """Retrieve a Graph-of-Thoughts artifact by query ID.

        Returns the schema-compliant reasoning graph artifact produced
        during query processing. The artifact includes nodes, edges,
        roots, graph_hash, and config — matching reasoning_graph.schema.json.

        Standing on: Besta (GoT, 2024) — graph artifacts as first-class,
        Merkle (1979) — content-addressed integrity.
        """
        artifact = runtime.get_graph_artifact(query_id)
        if artifact is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Graph artifact not found",
                    "query_id": query_id,
                },
            )
        return artifact

    @app.get("/v1/verify/signature")
    async def verify_signature_info():
        """Return the node's public key for independent verification.

        External verifiers can use this to verify any signed artifact
        (receipts, envelopes, attestations) produced by this node.
        """
        try:
            genesis = getattr(runtime, "_genesis", None)
            pub_key = ""
            node_id = ""
            if genesis and hasattr(genesis, "identity"):
                pub_key = genesis.identity.public_key
                node_id = genesis.identity.node_id

            return {
                "node_id": node_id,
                "public_key": pub_key,
                "algorithms": {
                    "signing": "Ed25519",
                    "hashing": "BLAKE3 (domain-separated: bizra-pci-v1:)",
                    "canonicalization": "RFC 8785",
                    "audit_chain": "HMAC-SHA256 (domain: bizra-audit-v1:)",
                },
                "verification_endpoints": [
                    "/v1/verify/genesis",
                    "/v1/verify/envelope",
                    "/v1/verify/receipt",
                    "/v1/verify/audit-log",
                    "/v1/verify/ledger",
                    "/v1/verify/poi",
                    "/v1/verify/genesis/header",
                    "/v1/sel/verify",
                ],
            }
        except Exception:
            logger.exception("Signature info retrieval failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.post("/v1/verify/poi")
    async def verify_poi_receipt(body: PoIReceiptVerifyModel):
        """Verify a signed Proof-of-Impact receipt.

        Validates structure, reason code, score bounds, config digest,
        and Ed25519 signature if signer key is available.

        Standing on: Nakamoto (PoW verification), Bernstein (Ed25519).
        """
        from core.proof_engine.evidence_ledger import VerifierResponse

        try:
            from core.proof_engine.poi_engine import PoIReasonCode, PoIReceipt

            r = body.receipt
            if not r:
                return JSONResponse(
                    status_code=400,
                    content=VerifierResponse.rejected(
                        reason_codes=["SCHEMA_VIOLATION"],
                        artifacts={"detail": "PoI receipt JSON body required"},
                    ).to_dict(),
                )

            # Validate required fields
            required = [
                "receipt_id",
                "epoch_id",
                "contributor_id",
                "reason",
                "poi_score",
            ]
            missing = [f for f in required if f not in r]
            if missing:
                return VerifierResponse.rejected(
                    reason_codes=["SCHEMA_VIOLATION"],
                    receipt_id=r.get("receipt_id", ""),
                    artifacts={"missing_fields": missing},
                ).to_dict()

            # Validate reason code is known
            try:
                reason = PoIReasonCode(r["reason"])
            except ValueError:
                return VerifierResponse.rejected(
                    reason_codes=["UNKNOWN_REASON_CODE"],
                    receipt_id=r["receipt_id"],
                    artifacts={
                        "reason": r["reason"],
                        "valid_codes": [c.value for c in PoIReasonCode],
                    },
                ).to_dict()

            # Validate score bounds
            poi_score = float(r.get("poi_score", 0.0))
            if not (0.0 <= poi_score <= 1.0):
                return VerifierResponse.rejected(
                    reason_codes=["SCORE_OUT_OF_BOUNDS"],
                    receipt_id=r["receipt_id"],
                    artifacts={"poi_score": poi_score, "bounds": "[0.0, 1.0]"},
                ).to_dict()

            # Reconstruct receipt for signature verification
            receipt = PoIReceipt(
                receipt_id=r["receipt_id"],
                epoch_id=r["epoch_id"],
                contributor_id=r["contributor_id"],
                reason=reason,
                poi_score=poi_score,
                contribution_score=float(r.get("contribution_score", 0.0)),
                reach_score=float(r.get("reach_score", 0.0)),
                longevity_score=float(r.get("longevity_score", 0.0)),
                config_digest=r.get("config_digest", ""),
                content_hash=r.get("content_hash", ""),
            )

            # Verify signature if available
            sig_hex = r.get("signature", "")
            pubkey_hex = r.get("signer_pubkey", "")
            signature_verified = False
            if sig_hex and pubkey_hex:
                try:
                    receipt.signature = bytes.fromhex(sig_hex)
                    receipt.signer_pubkey = bytes.fromhex(pubkey_hex)
                    # Use node signer for verification if available
                    signer = getattr(runtime, "_node_signer", None)
                    if signer and hasattr(signer, "verify"):
                        signature_verified = receipt.verify_signature(signer)
                    else:
                        # Can't verify without signer — report as unverifiable
                        signature_verified = False
                except (ValueError, TypeError):
                    signature_verified = False

            artifacts = {
                "receipt_id": receipt.receipt_id,
                "epoch_id": receipt.epoch_id,
                "contributor_id": receipt.contributor_id,
                "reason": reason.value,
                "quality": {
                    "poi_score": receipt.poi_score,
                    "contribution": receipt.contribution_score,
                    "reach": receipt.reach_score,
                    "longevity": receipt.longevity_score,
                },
                "signature_verified": signature_verified,
            }

            # CRITICAL-4 FIX: Signature verification is MANDATORY, not supplementary.
            # Proof-carrying inference requires actual proof verification.
            # Standing on: Lamport — verify, don't trust.
            if not signature_verified:
                return VerifierResponse.rejected(
                    reason_codes=["SIGNATURE_INVALID"],
                    receipt_id=receipt.receipt_id,
                    artifacts=artifacts,
                ).to_dict()

            if receipt.receipt_id and receipt.epoch_id and receipt.contributor_id:
                return VerifierResponse.approved(
                    receipt_id=receipt.receipt_id,
                    artifacts=artifacts,
                ).to_dict()

            return VerifierResponse.rejected(
                reason_codes=["INCOMPLETE_RECEIPT"],
                receipt_id=receipt.receipt_id,
                artifacts=artifacts,
            ).to_dict()

        except Exception:
            logger.exception("PoI receipt verification failed")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": "Internal server error"},
                ).to_dict(),
            )

    @app.get("/v1/verify/genesis/header")
    async def verify_genesis_header():
        """Lightweight genesis verification — hashes and signature only.

        Returns minimal verification data for bandwidth-constrained clients
        (mobile, edge nodes) without full agent lists or governance details.

        Standing on: Merkle (1979) — header-only verification.
        """
        try:
            import os
            from pathlib import Path

            from core.sovereign.genesis_identity import load_genesis
            from core.sovereign.origin_guard import (
                NODE_ROLE_ENV,
                normalize_node_role,
                resolve_origin_snapshot,
                validate_genesis_chain,
            )

            _runtime_cfg = getattr(runtime, "config", None)
            _sd_raw = getattr(runtime, "_state_dir", None)
            if not isinstance(_sd_raw, _Path):
                _sd_raw = getattr(_runtime_cfg, "state_dir", None)
            if not isinstance(_sd_raw, _Path):
                _sd_raw = None
            state_dir = _sd_raw or _Path("sovereign_state")
            role = normalize_node_role(os.getenv(NODE_ROLE_ENV, "node"))
            hash_validated, reason = validate_genesis_chain(state_dir)
            if not hash_validated:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": reason,
                        "origin": resolve_origin_snapshot(state_dir, role),
                        "hash_validated": False,
                    },
                )

            genesis = load_genesis(state_dir)
            if genesis is None:
                return JSONResponse(
                    status_code=503, content={"error": "Genesis state not loaded"}
                )

            identity = getattr(genesis, "identity", None)
            genesis_hash = getattr(genesis, "genesis_hash", b"")
            pat_hash = getattr(genesis, "pat_team_hash", b"")
            sat_hash = getattr(genesis, "sat_team_hash", b"")
            return {
                "node_id": identity.node_id if identity else "",
                "public_key": identity.public_key if identity else "",
                "genesis_hash": (
                    genesis_hash.hex()
                    if isinstance(genesis_hash, bytes)
                    else str(genesis_hash)
                ),
                "pat_team_hash": (
                    pat_hash.hex() if isinstance(pat_hash, bytes) else str(pat_hash)
                ),
                "sat_team_hash": (
                    sat_hash.hex() if isinstance(sat_hash, bytes) else str(sat_hash)
                ),
                "origin": resolve_origin_snapshot(state_dir, "node0"),
                "hash_validated": True,
                "header_only": True,
            }
        except Exception:
            logger.exception("Genesis header verification failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get("/v1/gate-chain/stats")
    async def gate_chain_stats():
        """Get GateChain evaluation statistics.

        Returns pass/fail rates, failure distribution by gate,
        and average SNR across evaluations.

        Standing on: Lamport (fail-closed), BIZRA Spearpoint (6-gate chain).
        """
        stats = runtime.get_gate_chain_stats()
        if stats is None:
            return JSONResponse(
                status_code=404,
                content={"error": "GateChain is not initialized"},
            )
        return stats

    # ─── PoI (Proof-of-Impact) Endpoints ────────────────────────────

    @app.get("/v1/poi/stats")
    async def poi_stats():
        """Get Proof-of-Impact engine statistics.

        Standing on: Nakamoto (PoW), Page & Brin (PageRank), Gini (inequality).
        """
        stats = runtime.get_poi_stats()
        if stats is None:
            return JSONResponse(
                status_code=404,
                content={"error": "PoI Engine is not initialized"},
            )
        return stats

    @app.post("/v1/poi/epoch")
    async def poi_epoch(request: Request):
        """Run a full PoI computation epoch.

        Computes composite PoI scores for all contributors,
        runs Gini analysis, and applies SAT rebalancing if needed.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        result = runtime.compute_poi_epoch()
        if result is None:
            return JSONResponse(
                status_code=503,
                content={"error": "PoI Engine is not available"},
            )
        return result

    @app.get("/v1/poi/contributor/{contributor_id}")
    async def poi_contributor(contributor_id: str):
        """Get the most recent PoI for a specific contributor."""
        poi = runtime.get_contributor_poi(contributor_id)
        if poi is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"No PoI found for '{contributor_id}'"},
            )
        return poi

    # ─── SAT Controller Endpoints ───────────────────────────────

    @app.get("/v1/sat/stats")
    async def sat_stats():
        """Get SAT Controller statistics.

        Returns Gini coefficient, rebalancing history, credit distribution.
        Standing on: Ostrom (commons governance), Gini (inequality).
        """
        stats = runtime.get_sat_stats()
        if stats is None:
            return JSONResponse(
                status_code=404,
                content={"error": "SAT Controller is not initialized"},
            )
        return stats

    @app.post("/v1/sat/epoch")
    async def sat_epoch(request: Request):
        """Finalize a PoI epoch via SAT Controller.

        Computes PoI scores, distributes tokens, checks Gini,
        and triggers rebalancing if inequality exceeds threshold.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        result = runtime.finalize_sat_epoch()
        if result is None:
            return JSONResponse(
                status_code=503,
                content={"error": "SAT Controller is not available"},
            )
        return result

    # ─── Token System Endpoints ─────────────────────────────────

    @app.get("/v1/token/balance")
    async def token_balance(request: Request, account: str = "BIZRA-00000000"):
        """Get token balances for an account."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            from core.token.ledger import TokenLedger
            from core.token.types import TokenType

            ledger = TokenLedger()
            result: dict[str, Any] = {"account": account, "balances": {}}
            for tt in TokenType:
                bal = ledger.get_balance(account, tt)
                if bal.balance > 0 or bal.staked > 0:
                    result["balances"][tt.value] = {
                        "balance": bal.balance,
                        "staked": bal.staked,
                    }
            return result
        except Exception:
            logger.exception("Token balance retrieval failed")
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable"},
            )

    @app.get("/v1/token/supply")
    async def token_supply():
        """Get total token supply across all types."""
        try:
            from datetime import datetime, timezone

            from core.token.ledger import TokenLedger
            from core.token.types import SEED_SUPPLY_CAP_PER_YEAR, TokenType

            ledger = TokenLedger()
            year = datetime.now(timezone.utc).year
            valid, count, err = ledger.verify_chain()
            supply: dict[str, Any] = {}
            for tt in TokenType:
                total = ledger.get_total_supply(tt)
                if total > 0:
                    supply[tt.value] = {
                        "total_supply": total,
                        "minted_this_year": ledger.get_yearly_minted(tt, year),
                    }
                    if tt == TokenType.SEED:
                        supply[tt.value]["yearly_cap"] = SEED_SUPPLY_CAP_PER_YEAR
            return {
                "year": year,
                "supply": supply,
                "ledger_valid": valid,
                "transaction_count": count,
            }
        except Exception:
            logger.exception("Token supply retrieval failed")
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable"},
            )

    @app.get("/v1/token/verify")
    async def token_verify():
        """Verify token ledger hash chain integrity."""
        try:
            from core.token.ledger import TokenLedger

            ledger = TokenLedger()
            valid, count, err = ledger.verify_chain()
            return {
                "valid": valid,
                "transaction_count": count,
                "error": err,
            }
        except Exception:
            logger.exception("Token verify failed")
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable"},
            )

    # ═════════════════════════════════════════════════════════════
    # /v1/seed/* — Phase 71 Seed Potential Engine
    # Standing on Giants: Deming (PDCA), Shannon (SNR), Al-Ghazali (Ihsan)
    # ═════════════════════════════════════════════════════════════

    @app.get(
        "/v1/seed/potential", tags=["sovereignty"], summary="Node growth trajectory"
    )
    async def seed_potential(request: Request):
        """Node's growth trajectory and unlocked capacity."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error
        seed_engine = getattr(runtime, "_seed_engine", None)
        if seed_engine is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Seed engine not initialized"},
            )
        try:
            from dataclasses import asdict

            potential = seed_engine.potential()
            return asdict(potential)
        except Exception:
            logger.exception("Seed potential error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get(
        "/v1/seed/episodes", tags=["sovereignty"], summary="Recent growth episodes"
    )
    async def seed_episodes(request: Request, limit: int = 10):
        """Recent growth episodes with receipt hashes."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error
        seed_engine = getattr(runtime, "_seed_engine", None)
        if seed_engine is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Seed engine not initialized"},
            )
        clamped_limit = min(max(1, limit), 100)
        try:
            episodes = seed_engine.recent_episodes(limit=clamped_limit)
            return {"count": len(episodes), "episodes": episodes}
        except Exception:
            logger.exception("Seed episodes error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    # ═════════════════════════════════════════════════════════════
    # Phase 72: Node Value KPI + Human Lifecycle + Network Effect
    # Standing on Giants: Shannon · Deming · Maslow · Metcalfe
    # ═════════════════════════════════════════════════════════════

    @app.get("/v1/node/value", tags=["sovereignty"], summary="Five-factor node KPI")
    async def get_node_value(request: Request):
        """Five-factor composite KPI for this sovereign node."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error

        try:
            from core.sovereign.node_value import NodeValueEngine
            from core.sovereign.seed_engine import SeedEngine as _SE

            seed_engine = getattr(runtime, "_seed_engine", None)
            if seed_engine is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Seed engine not initialized"},
                )

            nv_engine = getattr(runtime, "_node_value_engine", None)
            if nv_engine is None:
                # Lazy initialization if not wired at startup
                nv_engine = NodeValueEngine(seed_engine)

            from dataclasses import asdict

            snapshot = nv_engine.compute()
            return asdict(snapshot)
        except Exception:
            logger.exception("Node value computation error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get(
        "/v1/node/lifecycle", tags=["sovereignty"], summary="Human lifecycle stage"
    )
    async def get_lifecycle(request: Request):
        """Current human lifecycle stage with progress toward next."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error

        try:
            from core.sovereign.human_lifecycle import stage_progress

            seed_engine = getattr(runtime, "_seed_engine", None)
            if seed_engine is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Seed engine not initialized"},
                )

            pot = seed_engine.potential()
            return stage_progress(pot.sovereignty_score)
        except Exception:
            logger.exception("Lifecycle computation error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get(
        "/v1/network/effect",
        tags=["sovereignty"],
        summary="Metcalfe network projection",
    )
    async def get_network_effect(request: Request, nodes: int = 1000):
        """Project network-wide metrics for a given node count."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error

        if nodes < 1 or nodes > 10_000_000_000:
            return JSONResponse(
                status_code=400,
                content={"error": "nodes must be 1..10B"},
            )

        try:
            from core.sovereign.network_effect import NetworkEffectEstimator

            estimator = NetworkEffectEstimator()
            projection = estimator.project(nodes)
            return {
                "nodes": projection.nodes,
                "skills_available": projection.skills_available,
                "compute_tflops": projection.compute_tflops,
                "latency_factor": projection.latency_factor,
                "intelligence_density": projection.intelligence_density,
                "cost_per_node": projection.cost_per_node,
            }
        except Exception:
            logger.exception("Network effect projection error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get(
        "/v1/network/milestones",
        tags=["sovereignty"],
        summary="Milestone projections (1→8B)",
    )
    async def get_milestones(request: Request):
        """Standard milestone projections (1 to 8B nodes)."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error

        try:
            from core.sovereign.network_effect import NetworkEffectEstimator

            estimator = NetworkEffectEstimator()
            milestones = estimator.project_milestones()
            return {
                "milestones": [
                    {
                        "nodes": m.nodes,
                        "skills": m.skills_available,
                        "tflops": m.compute_tflops,
                        "latency_factor": m.latency_factor,
                    }
                    for m in milestones
                ]
            }
        except Exception:
            logger.exception("Network milestones error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    # ═════════════════════════════════════════════════════════════
    # /v1/constitutional/tick — 12-Step Constitutional Heartbeat
    # Standing on Giants: Al-Khwarizmi, Nakamoto, Kahneman
    # ═════════════════════════════════════════════════════════════
    @app.post("/v1/constitutional/tick")
    async def constitutional_tick(request: Request):
        """Execute one tick of the constitutional kernel.

        12 deterministic steps: intent gate → ihsan scoring → gini →
        progressive mint → bloom accrual → bloom decay → demurrage →
        zakat → governance → reflex cache → event log → asabiyyah.

        Auth-guarded. Returns TickResult summary.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            from core.constitutional.ticker import TickResult, process_tick
            from core.constitutional.types import WalletState

            # Gather state from runtime if available
            _wallets = getattr(runtime, "_constitutional_wallets", [])
            _receipts = getattr(runtime, "_constitutional_receipts", [])
            _proposals = getattr(runtime, "_constitutional_proposals", [])
            _event_log = getattr(runtime, "_constitutional_event_log", [])
            _reflex_cache = getattr(runtime, "_constitutional_reflex_cache", {})

            result = process_tick(
                wallets=_wallets,
                receipts=_receipts,
                proposals=_proposals,
                event_log=_event_log,
                reflex_cache=_reflex_cache,
            )

            # Clear processed receipts (they've been consumed by the tick)
            if hasattr(runtime, "_constitutional_receipts"):
                runtime._constitutional_receipts = []

            from core.constitutional.fixed_point import fp_float

            return {
                "status": "tick_complete",
                "rejected": result.rejected,
                "scored": result.scored,
                "total_minted": fp_float(result.total_minted),
                "zakat_pool": fp_float(result.zakat_pool),
                "network_gini": fp_float(result.network_gini),
                "network_asabiyyah": fp_float(result.network_asabiyyah_score),
                "events_logged": result.events_logged,
                "proposals_resolved": result.proposals_resolved,
                "wallets_count": len(_wallets),
            }
        except ImportError:
            logger.exception("Constitutional engine not available")
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable"},
            )
        except Exception:
            logger.exception("Constitutional tick failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get("/v1/constitutional/status")
    async def constitutional_status(request: Request):
        """Get current constitutional kernel state."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            from core.constitutional.fixed_point import fp_float

            _wallets = getattr(runtime, "_constitutional_wallets", [])
            _event_log = getattr(runtime, "_constitutional_event_log", [])
            _reflex_cache = getattr(runtime, "_constitutional_reflex_cache", {})

            # Gini + tick metadata for Dashboard (Contract §10.1)
            _last_tick = getattr(runtime, "_last_tick_result", None)
            _tick_interval = getattr(runtime, "_tick_interval_s", 60)
            _last_tick_ts = getattr(runtime, "_last_tick_timestamp", None)

            gini = 0.0
            asabiyyah = 0.0
            if _last_tick is not None:
                gini = fp_float(getattr(_last_tick, "network_gini", 0))
                asabiyyah = fp_float(getattr(_last_tick, "network_asabiyyah", 0))

            return {
                "status": "active",
                "wallets": len(_wallets),
                "events": len(_event_log),
                "reflexes": len(_reflex_cache),
                "pending_receipts": len(
                    getattr(runtime, "_constitutional_receipts", [])
                ),
                "pending_proposals": len(
                    getattr(runtime, "_constitutional_proposals", [])
                ),
                "network_gini": round(gini, 4),
                "network_asabiyyah": round(asabiyyah, 4),
                "last_tick_timestamp": _last_tick_ts,
                "tick_interval_s": _tick_interval,
            }
        except Exception:
            logger.exception("Constitutional status unavailable")
            return JSONResponse(
                status_code=503,
                content={"error": "Service temporarily unavailable"},
            )

    # /v1/orchestrate — direct orchestrator task decomposition endpoint
    @app.post("/v1/orchestrate")
    async def orchestrate(body: OrchestrateRequestModel, request: Request):
        """Decompose a complex task through the orchestrator's agent swarm."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        orch = getattr(runtime, "_orchestrator", None)
        if orch is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Orchestrator not available"},
            )

        try:
            plan = await orch.decompose(body.task)
            for task_node in plan.tasks:
                await orch.submit_task(task_node)

            tasks_out = []
            for task_node in plan.tasks:
                tr = orch.task_results.get(task_node.id, {})
                tasks_out.append(
                    {
                        "id": task_node.id,
                        "title": task_node.title,
                        "agent": tr.get("agent", "unknown"),
                        "content": tr.get("content", ""),
                        "snr_score": tr.get("snr_score", 0.0),
                        "status": task_node.status.name,
                    }
                )

            return {
                "success": True,
                "plan_id": plan.id,
                "complexity": plan.complexity.name,
                "total_tasks": len(plan.tasks),
                "tasks": tasks_out,
            }
        except Exception:
            logger.exception("Orchestration failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    # ─── Mission Endpoint (Sprint 1 Task 1.4) ─────────────────────────────
    # Standing on: Boyd (OODA), Al-Ghazali (Ihsan gate)
    #
    # POST /v1/plan — submit a mission, get receipted result
    # This is the golden path: intent → route → execute → proof → return
    # Blueprint Section 5, Sprint 1, Acceptance: curl /v1/plan returns receipted result

    @app.post(
        "/v1/plan",
        response_model=MissionPlanResponse,
        tags=["mission"],
        summary="Submit a sovereign mission",
        description=(
            "Golden path endpoint. Accepts a mission description, routes through "
            "MissionOrchestrator (OODA loop), and returns a receipted result with "
            "Ihsan/SNR scores and per-channel execution breakdown."
        ),
        responses={
            400: {"description": "Empty or missing description"},
            503: {"description": "Mission engine not available"},
        },
    )
    async def submit_plan(request: Request):
        """Submit a sovereign mission and receive a receipted result."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            import secrets as _secrets
            import time as _time

            from core.sovereign.mission import (
                DesktopContext,
                MissionOrchestrator,
                MissionRequest,
            )

            body = await request.json()
            description = body.get("description", "").strip()
            if not description:
                return JSONResponse(
                    status_code=400,
                    content={"error": "description is required"},
                )

            source = body.get("source", "api")

            # ── System-1 Fast Path: Reflex Cache Lookup ──────────
            # Kahneman S1: if this pattern is cached with high Ihsan,
            # return O(1) cached response instead of full pipeline.
            try:
                from core.sovereign.reflex_compiler import ReflexCompiler

                global _reflex_compiler  # noqa: PLW0603
                if "_reflex_compiler" not in globals() or _reflex_compiler is None:
                    _persistence = os.environ.get(
                        "REFLEX_PERSISTENCE_PATH",
                        "/tmp/bizra-mission/reflexes.json",
                    )
                    from pathlib import Path

                    _reflex_compiler = ReflexCompiler(
                        persistence_path=Path(_persistence),
                    )

                reflex_entry = _reflex_compiler.lookup(description)
                if reflex_entry and not reflex_entry.needs_validation():
                    import secrets as _secrets

                    logger.info(
                        "System-1 cache hit for pattern %s (hits=%d, ihsan=%.3f)",
                        reflex_entry.pattern_hash[:12],
                        reflex_entry.hit_count,
                        reflex_entry.ihsan_composite,
                    )
                    return {
                        "status": "COMPLETED",
                        "mission_id": _secrets.token_hex(8),
                        "receipt_id": _secrets.token_hex(8),
                        "evidence_receipt_id": "",
                        "synthesis": reflex_entry.output_template,
                        "ihsan_score": reflex_entry.ihsan_composite,
                        "snr_score": reflex_entry.ihsan_composite,
                        "duration_ms": 0.1,
                        "execution_path": "system_1_reflex",
                        "channels_executed": [],
                        "action_count": 0,
                        "wallet_delta": {"seed": 0.0, "bloom": 0.0},
                        "reflex_delta": {
                            "compiled": True,
                            "near_compile": False,
                            "compile_count": reflex_entry.precipitation_count,
                            "threshold": 3,
                        },
                        "memory_delta": {"episodic": 0, "semantic": 0, "procedural": 0},
                        "hash_chain_ref": reflex_entry.pattern_hash,
                        "reflex_pattern": reflex_entry.pattern_hash[:12],
                        "reflex_latency_ms": 0.1,
                        "comparison_s2_avg_ms": 0.0,
                    }
            except ImportError:
                logger.debug("ReflexCompiler not available, using System-2 only")
            except Exception:
                logger.debug("Reflex lookup failed, falling through to System-2", exc_info=True)

            # Build mission config from Node0 ConfigMap environment
            config = {
                "memory_path": os.environ.get(
                    "SEMANTIC_MEMORY_PATH", "/tmp/bizra-mission/memory"
                ),
                "evidence_path": os.environ.get("EVENT_LOG_PATH", "/tmp/bizra-mission")
                + "/evidence.jsonl",
                "hda_port": int(os.environ.get("HDA_PORT", "9743")),
                "workspace_root": os.environ.get("BIZRA_DATA_LAKE_ROOT", "."),
            }

            orchestrator = MissionOrchestrator(config=config)

            # Inject inference gateway from runtime if available
            gateway = getattr(runtime, "inference_gateway", None)
            if gateway is not None:
                orchestrator.gateway = gateway

            mission_req = MissionRequest(
                mission_id=_secrets.token_hex(8),
                description=description,
                context=DesktopContext(),
                timestamp=_time.time(),
                source=source,
            )

            # ── Mission Lifecycle: Created ──────────────────────
            await _emit_bus_event(
                "mission.created",
                {"mission_id": mission_req.mission_id, "source": source},
                source="mission",
            )

            result = await orchestrator.execute(mission_req)

            # ── Reflex Cache Wiring ──────────────────────────────
            # Feed mission result into constitutional tick queue so
            # Step 10 (reflex compilation) can cache excellent patterns.
            # This closes the loop: mission → receipt → tick → reflex.
            _submit_mission_to_tick(runtime, result)

            # ── System-1 Precipitation: Record Observation ─────────
            # After every System-2 completion, record the pattern.
            # After K consecutive high-Ihsan observations, precipitate.
            try:
                if "_reflex_compiler" in globals() and _reflex_compiler is not None:
                    _reflex_compiler.record_observation(
                        input_text=description,
                        output_text=result.synthesis or "",
                        ihsan_composite=result.ihsan_score or 0.0,
                    )
            except Exception:
                logger.debug("Reflex observation recording failed", exc_info=True)

            # ── Terminal State Wiring (P2a fix) ──────────────────
            # Update the session's terminal controller so
            # /v1/terminal/state reflects the actual mission lifecycle.
            try:
                from core.sovereign.terminal import TerminalState as _TS
                from core.sovereign.terminal import TerminalStateController as _TSC

                _session_id = request.headers.get("X-Session-ID", "default")
                if _session_id not in _terminal_controllers:
                    _ctrl = _TSC()
                    _ctrl.transition(_TS.READY)
                    _terminal_controllers[_session_id] = _ctrl
                _ctrl = _terminal_controllers[_session_id]
                _ctrl.start_mission(mission_req.mission_id)
                # Fast-forward through intermediate states for API-driven missions
                _ctrl.transition(_TS.PERMISSION_REVIEW)
                _ctrl.transition(_TS.EXECUTING)
                if result.status == "FAILED":
                    _ctrl.fail()
                else:
                    _ctrl.complete()
            except (ImportError, Exception):
                logger.debug("Terminal state wiring unavailable", exc_info=True)

            # ── Mission Lifecycle: Executed ──────────────────────
            mission_topic = (
                "mission.executed" if result.status != "FAILED" else "mission.failed"
            )
            await _emit_bus_event(
                mission_topic,
                {
                    "mission_id": result.mission_id,
                    "status": result.status,
                    "ihsan_score": result.ihsan_score,
                    "snr_score": result.snr_score,
                    "duration_ms": round(result.duration_ms, 1),
                },
                source="mission",
            )

            # Build enriched Terminal v1 receipt
            try:
                from core.sovereign.terminal import ChannelRecord as TChannelRecord
                from core.sovereign.terminal import (
                    ExecutionPath,
                )
                from core.sovereign.terminal import MissionReceipt as TerminalReceipt
                from core.sovereign.terminal import (
                    WalletDelta,
                )

                t_channels = [
                    TChannelRecord(
                        channel=cr.channel,
                        success=cr.success,
                        duration_ms=cr.duration_ms,
                    )
                    for cr in result.channels_executed
                ]
                terminal_receipt = TerminalReceipt(
                    mission_id=result.mission_id,
                    receipt_id=result.evidence_receipt_id or _secrets.token_hex(8),
                    status=result.status,
                    synthesis=result.synthesis,
                    ihsan_score=result.ihsan_score,
                    snr_score=result.snr_score,
                    duration_ms=result.duration_ms,
                    channels_executed=t_channels,
                    action_count=len(t_channels),
                )
                return terminal_receipt.to_dict()
            except ImportError:
                pass

            # Fallback: Contract §8.1 normalized dict if terminal module unavailable
            # All required fields MUST be present even in fallback path
            _fallback_receipt_id = result.evidence_receipt_id or _secrets.token_hex(8)
            return {
                "status": result.status,
                "mission_id": result.mission_id,
                "receipt_id": _fallback_receipt_id,
                "evidence_receipt_id": _fallback_receipt_id,
                "synthesis": result.synthesis,
                "ihsan_score": result.ihsan_score,
                "snr_score": result.snr_score,
                "duration_ms": round(result.duration_ms, 1),
                "execution_path": "system_2",
                "channels_executed": [
                    {
                        "channel": cr.channel,
                        "success": cr.success,
                        "duration_ms": round(cr.duration_ms, 1),
                    }
                    for cr in result.channels_executed
                ],
                "action_count": len(result.channels_executed),
                "wallet_delta": {"seed": 0.0, "bloom": 0.0},
                "reflex_delta": {
                    "compiled": False,
                    "near_compile": False,
                    "compile_count": 0,
                    "threshold": 3,
                },
                "memory_delta": {"episodic": 0, "semantic": 0, "procedural": 0},
                "hash_chain_ref": "",
                "reflex_pattern": "",
                "reflex_latency_ms": 0.0,
                "comparison_s2_avg_ms": 0.0,
            }

        except ImportError as exc:
            logger.exception("Mission orchestrator not available")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Mission engine not available",
                    "detail": str(exc),
                },
            )
        except Exception:
            logger.exception("Mission execution failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    # ─── Spearpoint Endpoints ────────────────────────────────────────────
    # Standing on: Boyd (OODA), Goldratt (Theory of Constraints)
    #
    # expose reproduce (evaluation-first) and improve (innovation-through-gate)
    # missions, plus orchestrator statistics.

    @app.post("/v1/spearpoint/reproduce")
    async def spearpoint_reproduce(body: SpearpointReproduceModel, request: Request):
        """Evaluate/verify a claim through the Spearpoint evaluator gate."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        orch = getattr(runtime, "_spearpoint_orchestrator", None)
        if orch is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Spearpoint Orchestrator not available"},
            )

        try:
            result = orch.reproduce(
                claim=body.claim,
                proposed_change=body.proposed_change,
                prompt=body.prompt,
                response=body.response,
                metrics=body.metrics,
            )
            return result.to_dict()
        except Exception:
            logger.exception("Spearpoint reproduce failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.post("/v1/spearpoint/improve")
    async def spearpoint_improve(body: SpearpointImproveModel, request: Request):
        """Generate and evaluate improvement hypotheses through the evaluator gate."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        orch = getattr(runtime, "_spearpoint_orchestrator", None)
        if orch is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Spearpoint Orchestrator not available"},
            )

        try:
            result = orch.improve(
                observation=body.observation,
                top_k=body.top_k,
            )
            return result.to_dict()
        except Exception:
            logger.exception("Spearpoint improve failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get("/v1/spearpoint/stats")
    async def spearpoint_stats(request: Request):
        """Get Spearpoint Orchestrator statistics and mission history."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        orch = getattr(runtime, "_spearpoint_orchestrator", None)
        if orch is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Spearpoint Orchestrator not available"},
            )

        try:
            return {
                "statistics": orch.get_statistics(),
                "recent_missions": orch.get_mission_history(limit=10),
            }
        except Exception:
            logger.exception("Spearpoint stats failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.post("/v1/spearpoint/pattern")
    async def spearpoint_pattern(body: "SpearpointPatternModel", request: Request):
        """Pattern-aware research using Sci-Reasoning thinking patterns.

        Routes to SpearpointOrchestrator.research_pattern() which uses the
        15 cognitive moves from Li et al. (2025) to seed hypothesis generation.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        orch = getattr(runtime, "_spearpoint_orchestrator", None)
        if orch is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Spearpoint Orchestrator not available"},
            )

        try:
            result = orch.research_pattern(
                pattern_id=body.pattern_id,
                claim_context=body.claim_context,
                top_k=body.top_k,
            )
            return result.to_dict()
        except Exception:
            logger.exception("Spearpoint pattern failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    # ─── WebSocket Agent-to-User Push Channel ────────────────────────────
    # Standing on: RFC 6455 (WebSocket Protocol), Agent-to-User comms pattern
    #
    # Provides real-time push from agents/proactive system to connected clients.
    # Clients connect once and receive: proactive suggestions, agent status,
    # task completions, and system events.

    _ws_clients: set[Any] = set()  # Active WebSocket connections

    try:
        from starlette.websockets import WebSocket as StarletteWS
        from starlette.websockets import WebSocketDisconnect

        _WS_AVAILABLE = True
    except ImportError:
        _WS_AVAILABLE = False

    if _WS_AVAILABLE:

        @app.websocket("/v1/stream")
        async def websocket_stream(ws: "StarletteWS"):
            """Agent-to-User WebSocket channel.

            Protocol:
            - Client connects → receives welcome message with node identity
            - Server pushes: proactive_suggestion, task_completed, agent_status
            - Client can send: subscribe/unsubscribe topic filters
            """
            ws_user_id, authorized = await _authorize_websocket(ws)
            if not authorized:
                return

            await ws.accept()
            _ws_clients.add(ws)

            # Send welcome
            identity = runtime.status().get("identity", {})
            await ws.send_json(
                {
                    "type": "connected",
                    "node_id": identity.get("node_id", "unknown"),
                    "version": identity.get("version", "1.0.0"),
                    "user_id": ws_user_id,
                }
            )

            try:
                while True:
                    # Keep connection alive, handle client messages
                    data = await ws.receive_json()
                    msg_type = data.get("type", "")

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})

                    elif msg_type == "query":
                        # Allow queries over WebSocket too
                        if (
                            ws_user_id
                            and _auth_middleware is not None
                            and not _auth_middleware.check_rate_limit(ws_user_id)
                        ):
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "error": "Rate limit exceeded",
                                }
                            )
                            continue

                        if ws_user_id and _user_store is not None:
                            _user_store.increment_query_count(ws_user_id)

                        result = await runtime.query(
                            data.get("query", ""),
                            context=data.get("context", {}),
                            user_id=ws_user_id,
                        )
                        await ws.send_json(
                            {
                                "type": "query_result",
                                "id": result.query_id,
                                "answer": result.answer,
                                "confidence": result.confidence,
                                "quality": {
                                    "snr": result.snr_score,
                                    "ihsan": result.ihsan_score,
                                },
                            }
                        )

            except (WebSocketDisconnect, Exception):
                pass
            finally:
                _ws_clients.discard(ws)

    # Broadcast helper (used by background tasks to push to all clients)
    async def broadcast_to_clients(message: dict) -> int:
        """Push a message to all connected WebSocket clients."""
        sent = 0
        disconnected = set()
        for ws in _ws_clients:  # noqa: F823 — defined at function scope (line 1790)
            try:
                await ws.send_json(message)
                sent += 1
            except Exception:
                disconnected.add(ws)
        _ws_clients -= disconnected
        return sent

    # Attach broadcaster to runtime for agent access
    runtime._ws_broadcast = broadcast_to_clients  # type: ignore[attr-defined]

    # ─── Sovereign Experience Ledger (SEL) Endpoints ─────────────────
    # Standing on: Tulving (1972) — episodic memory distinction
    # Park et al. (2023) — generative agent memory architecture
    # Shannon (1948) — information-theoretic SNR measurement
    #
    # These endpoints expose the content-addressed, hash-chained
    # episodic memory store for audit, retrieval, and verification.

    @app.get("/v1/sel/episodes")
    async def sel_episodes(request: Request, limit: int = 50, offset: int = 0):
        """list episodes from the Sovereign Experience Ledger.

        Returns episodes in reverse chronological order (newest first).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        sel = getattr(runtime, "_experience_ledger", None)
        if sel is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Experience Ledger not initialized"},
            )

        try:
            total = len(sel)
            # Clamp parameters
            limit = max(1, min(limit, 200))
            offset = max(0, offset)

            episodes = []
            for i in range(total - 1, -1, -1):
                ep = sel.get_by_sequence(i)
                if ep is None:
                    continue
                if offset > 0:
                    offset -= 1
                    continue
                episodes.append(ep.to_dict())
                if len(episodes) >= limit:
                    break

            return {
                "total": total,
                "count": len(episodes),
                "chain_head": (
                    sel.chain_head[:16] + "..."
                    if len(sel.chain_head) > 16
                    else sel.chain_head
                ),
                "distillation_count": sel.distillation_count,
                "episodes": episodes,
            }
        except Exception:
            logger.exception("SEL episodes list failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/v1/sel/episodes/{episode_hash}")
    async def sel_episode_by_hash(episode_hash: str, request: Request):
        """Retrieve a single episode by its content-address hash."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        sel = getattr(runtime, "_experience_ledger", None)
        if sel is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Experience Ledger not initialized"},
            )

        try:
            ep = sel.get_by_hash(episode_hash)
            if ep is None:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Episode not found: {episode_hash[:16]}..."},
                )
            return ep.to_dict()
        except Exception:
            logger.exception("SEL episode lookup failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.post("/v1/sel/retrieve")
    async def sel_retrieve(body: SELRetrieveModel, request: Request):
        """Retrieve episodes using RIR (Recency-Importance-Relevance) algorithm.

        Returns top_k most relevant episodes for the given query text.
        Standing on: Park et al. (2023) — generative agent retrieval.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        sel = getattr(runtime, "_experience_ledger", None)
        if sel is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Experience Ledger not initialized"},
            )

        try:
            if not body.query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Query text required"},
                )
            top_k = max(1, min(body.top_k, 100))
            results = sel.retrieve(body.query, top_k=top_k)
            return {
                "query": body.query,
                "top_k": top_k,
                "count": len(results),
                "episodes": [ep.to_dict() for ep in results],
            }
        except Exception:
            logger.exception("SEL retrieve failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/v1/sel/verify")
    async def sel_verify():
        """Verify the Experience Ledger hash-chain integrity.

        Returns chain validity status and diagnostics.
        Standing on: Merkle (1979) — hash chain tamper detection.
        """
        sel = getattr(runtime, "_experience_ledger", None)
        if sel is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Experience Ledger not initialized"},
            )

        try:
            is_valid = sel.verify_chain_integrity()
            return {
                "valid": is_valid,
                "episodes": len(sel),
                "sequence": sel.sequence,
                "chain_head": (
                    sel.chain_head[:16] + "..."
                    if len(sel.chain_head) > 16
                    else sel.chain_head
                ),
                "distillation_count": sel.distillation_count,
            }
        except Exception:
            logger.exception("SEL verify failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    # ─── AgentDB Memory Search Endpoint (V3 Unified Memory) ──────────

    @app.post("/v1/memory/search")
    async def memory_search(body: MemorySearchModel, request: Request):
        """Hybrid memory search using AgentDB (HNSW + FTS5 + score fusion).

        Returns top_k most relevant memory records for the given query.
        Standing on: Malkov & Yashunin (2016) — HNSW; Robertson (2009) — BM25.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        agent_db = getattr(runtime, "_agent_db", None)
        if agent_db is None:
            return JSONResponse(
                status_code=404,
                content={"error": "AgentDB not initialized"},
            )

        try:
            if not body.query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Query text required"},
                )
            top_k = max(1, min(body.top_k, 100))
            results = agent_db.search(
                query=body.query,
                top_k=top_k,
                min_score=body.min_score,
                source=body.source,
            )
            return {
                "query": body.query,
                "top_k": top_k,
                "count": len(results),
                "results": [
                    {
                        "id": r.record.id,
                        "content": r.record.content[:500],
                        "kind": r.record.kind.value,
                        "score": round(r.score, 4),
                        "vector_score": round(r.vector_score, 4),
                        "keyword_score": round(r.keyword_score, 4),
                        "recency_score": round(r.recency_score, 4),
                        "importance_score": round(r.importance_score, 4),
                        "source": r.record.source,
                    }
                    for r in results
                ],
            }
        except Exception:
            logger.exception("AgentDB search failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/v1/memory/stats")
    async def memory_stats(request: Request):
        """Get AgentDB statistics (record counts, vector index, paths)."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        agent_db = getattr(runtime, "_agent_db", None)
        if agent_db is None:
            return JSONResponse(
                status_code=404,
                content={"error": "AgentDB not initialized"},
            )
        return agent_db.stats()

    # ─── Cognitive Fusion Endpoints (Phase 31) ──────────────────────

    @app.post("/v1/cognitive/fuse")
    async def cognitive_fuse(body: CognitiveFuseModel, request: Request):
        """Run a query through the full Cognitive Fusion pipeline.

        4-stage pipeline: MoE Route → HRM Reason → HyperGraph RAG → NorthStar Gate.
        Returns complexity classification, HRM reasoning, RAG context, and quality scores.
        Standing on: Vaswani (MoE) + Simon (hierarchy) + Shannon (SNR) + Besta (GoT).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        fusion_engine = getattr(runtime, "_cognitive_fusion", None)
        if fusion_engine is None:
            return JSONResponse(
                status_code=404,
                content={"error": "CognitiveFusionEngine not initialized"},
            )

        try:
            if not body.query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Query text required"},
                )

            # Use zero-vector placeholder (real embedding fn injected at runtime)
            dummy_embedding = [0.0] * 768
            result = fusion_engine.process(
                query=body.query,
                query_embedding=dummy_embedding,
                context=body.context,
            )

            return {
                "query": body.query,
                "complexity": result.routing.complexity_class,
                "expert_tier": result.expert_tier,
                "target_level": result.target_level,
                "hrm_snr": round(result.compound_snr, 4),
                "retrieval_count": len(result.retrieval),
                "fusion_snr": round(result.snr_score, 4),
                "fusion_ihsan": round(result.ihsan_score, 4),
                "passes_gate": result.passes_gate,
                "is_elite": result.is_elite,
                "routing": {
                    "complexity_class": result.routing.complexity_class,
                    "expert_tier": result.routing.expert_tier,
                    "confidence": round(result.routing.confidence, 4),
                },
                "hrm": {
                    "compound_snr": round(result.hrm_result.compound_snr, 4),
                    "level_reached": result.hrm_result.level_reached,
                    "observations": result.hrm_result.observations[:10],
                },
                "northstar": {
                    "unified_snr": round(result.northstar_report.unified_snr, 4),
                    "ihsan_score": round(result.northstar_report.ihsan_score, 4),
                    "passes_all_gates": result.northstar_report.passes_all_gates,
                },
            }
        except Exception:
            logger.exception("Cognitive fusion failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/v1/cognitive/status")
    async def cognitive_status():
        """Get Cognitive Fusion subsystem availability.

        Reports which of the 4 subsystems (MoE, HRM, RAG, NorthStar) are wired.
        """
        fusion_engine = getattr(runtime, "_cognitive_fusion", None)
        hrm = getattr(runtime, "_hrm_engine", None)
        northstar = getattr(runtime, "_northstar_engine", None)
        hypergraph = getattr(runtime, "_hypergraph_store", None)

        return {
            "cognitive_fusion_available": fusion_engine is not None,
            "subsystems": {
                "moe_router": (
                    getattr(fusion_engine, "_moe_router", None) is not None
                    if fusion_engine
                    else False
                ),
                "hrm_engine": hrm is not None,
                "hypergraph_rag": (
                    getattr(fusion_engine, "_hypergraph_rag", None) is not None
                    if fusion_engine
                    else False
                ),
                "northstar_engine": northstar is not None,
            },
            "hypergraph_store": hypergraph is not None,
            "memory_synthesizer": getattr(runtime, "_memory_synthesizer", None)
            is not None,
            "pattern_codebook": getattr(runtime, "_pattern_codebook", None) is not None,
        }

    # ─── SJE Judgment Telemetry Endpoints (Phase A: Observation) ──────

    @app.get("/v1/judgment/stats")
    async def judgment_stats(request: Request):
        """Get SJE verdict distribution and entropy.

        Standing on: Shannon (1948) — entropy as uncertainty measure.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        jt = getattr(runtime, "_judgment_telemetry", None)
        if jt is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Judgment Telemetry not initialized"},
            )
        return jt.to_dict()

    @app.get("/v1/judgment/stability")
    async def judgment_stability(request: Request):
        """Check if judgment verdicts are stable (low entropy).

        Stability indicates strong consensus toward one verdict.
        Standing on: Shannon (1948), Aristotle (practical wisdom).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        jt = getattr(runtime, "_judgment_telemetry", None)
        if jt is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Judgment Telemetry not initialized"},
            )
        return {
            "is_stable": jt.is_stable(),
            "entropy": round(jt.entropy(), 6),
            "total_observations": jt.total_observations,
            "dominant_verdict": (
                jt.dominant_verdict().value if jt.dominant_verdict() else None
            ),
        }

    @app.post(
        "/v1/judgment/simulate",
        tags=["experience"],
        summary="Simulate epoch distribution",
    )
    async def judgment_simulate(body: EpochSimulateModel, request: Request):
        """Simulate proportional epoch distribution (no tokens emitted).

        Pure mathematical rehearsal for genesis economy modeling.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        from core.sovereign.judgment_telemetry import simulate_epoch_distribution

        result = simulate_epoch_distribution(body.impacts, body.epoch_cap)
        return {
            "impacts": body.impacts,
            "epoch_cap": body.epoch_cap,
            "allocations": result,
            "dust": body.epoch_cap - sum(result),
        }

    # ─── Proactive Suggestions Endpoint ───────────────────────────────
    @app.get("/v1/suggestions")
    async def proactive_suggestions(request: Request):
        """Get proactive knowledge suggestions from living memory."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        living_memory = getattr(runtime, "_living_memory", None)
        if living_memory is None:
            return {"suggestions": [], "note": "Living memory not initialized"}

        try:
            from core.living_memory.proactive import ProactiveRetriever

            # Create retriever with LLM if gateway available
            gateway = getattr(runtime, "_gateway", None)
            llm_fn = None
            if gateway is not None:
                import asyncio as _aio

                async def _llm_async(prompt: str) -> str:
                    result = await gateway.infer(prompt, max_tokens=200)
                    return getattr(result, "content", str(result))

                def _llm_sync(prompt: str) -> str:
                    loop = _aio.get_event_loop()
                    return loop.run_until_complete(_llm_async(prompt))

                llm_fn = _llm_sync

            retriever = ProactiveRetriever(
                memory=living_memory, llm_fn=llm_fn, max_suggestions=5
            )

            suggestions = await retriever.get_proactive_suggestions()
            return {
                "suggestions": [
                    {
                        "content": s.memory.content[:200],
                        "reason": s.reason,
                        "confidence": round(s.confidence, 3),
                        "urgency": round(s.urgency, 3),
                    }
                    for s in suggestions
                ],
                "context": retriever.get_context_summary(),
            }
        except Exception:
            logger.exception("Proactive suggestions failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    # ─── Phase 21: Auth Endpoints ────────────────────────────────────
    # Standing on: OWASP ASVS v4 (auth verification)
    # Timing-safe password compare, PBKDF2-SHA256 600K rounds
    # JWT HMAC-SHA256 with refresh rotation

    if _auth_available:

        @app.post("/v1/auth/register")
        async def auth_register(body: RegisterRequestModel):
            """Register a new user. Returns user profile + JWT tokens."""
            if not body.accept_covenant:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Must accept the BIZRA covenant (accept_covenant=true)"
                    },
                )
            try:
                user = _user_store.register(
                    username=body.username,
                    email=body.email,
                    password=body.password,
                    accept_covenant=body.accept_covenant,
                )
                tokens = _jwt_auth.issue_tokens(user.user_id, user.username)
                return {
                    "user": {
                        "user_id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "api_key": user.api_key,
                        "namespace": user.namespace,
                        "covenant_accepted": user.covenant_accepted,
                        "created_at": user.created_at,
                    },
                    "tokens": {
                        "access_token": tokens.access_token,
                        "refresh_token": tokens.refresh_token,
                        "token_type": tokens.token_type,
                        "expires_in": tokens.expires_in,
                    },
                }
            except ValueError as e:
                return JSONResponse(status_code=409, content={"error": str(e)})
            except Exception:
                logger.exception("Registration failed")
                return JSONResponse(
                    status_code=500, content={"error": "Registration failed"}
                )

        @app.post("/v1/auth/login")
        async def auth_login(body: LoginRequestModel):
            """Authenticate and return JWT tokens."""
            user = _user_store.verify_login(body.username, body.password)
            if user is None:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid credentials"},
                )
            tokens = _jwt_auth.issue_tokens(user.user_id, user.username)
            _user_store.increment_query_count(user.user_id)
            return {
                "user_id": user.user_id,
                "username": user.username,
                "tokens": {
                    "access_token": tokens.access_token,
                    "refresh_token": tokens.refresh_token,
                    "token_type": tokens.token_type,
                    "expires_in": tokens.expires_in,
                },
            }

        @app.post("/v1/auth/refresh")
        async def auth_refresh(body: RefreshTokenModel):
            """Refresh an access token using a valid refresh token."""
            try:
                new_pair = _jwt_auth.refresh_access_token(body.refresh_token)
                if new_pair is None:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid or expired refresh token"},
                    )
                return {
                    "access_token": new_pair.access_token,
                    "refresh_token": new_pair.refresh_token,
                    "token_type": new_pair.token_type,
                    "expires_in": new_pair.expires_in,
                }
            except ValueError as e:
                return JSONResponse(status_code=401, content={"error": str(e)})

        @app.get("/v1/auth/me")
        async def auth_me(request: Request):
            """Return current user profile. Requires JWT or API key."""
            try:
                user = _auth_middleware.authenticate_request(request)
                if user is None:
                    return JSONResponse(
                        status_code=401, content={"error": "Authentication required"}
                    )
                return {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "namespace": user.namespace,
                    "status": user.status,
                    "created_at": user.created_at,
                    "query_count": user.query_count,
                }
            except Exception:
                return JSONResponse(
                    status_code=401, content={"error": "Authentication required"}
                )

    # ─── Auth Route Wiring ────────────────────────────────────────────
    # Note: /v1/query is auth-aware via single handler (SAPE-001 fix).
    # No duplicate route registration needed.

    # ═════════════════════════════════════════════════════════════
    # /v1/onboarding/* — Phase 73 Onboarding State Machine
    # Local-first: frontend owns UX state, backend enriches with
    # sovereign proof data and persists for cross-device sync.
    # ═════════════════════════════════════════════════════════════

    ONBOARDING_STEPS = [
        "welcome",
        "desire",
        "stressor",
        "capacity",
        "commitment",
        "review",
    ]

    @app.get("/v1/onboarding/state", tags=["onboarding"], summary="Onboarding progress")
    async def get_onboarding_state(request: Request):
        """Get current onboarding progress for authenticated user."""
        _, user_id, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error

        try:
            # Check sovereign state for persisted onboarding progress
            state_dir = pathlib.Path("sovereign_state") / "onboarding"
            state_file = state_dir / f"{user_id}.json"

            if state_file.exists():
                import json as _json

                state = _json.loads(state_file.read_text())
                return state

            # Default: fresh onboarding
            return {
                "step": 0,
                "total_steps": len(ONBOARDING_STEPS),
                "completed": [],
                "current": ONBOARDING_STEPS[0],
                "profile": {},
            }
        except Exception:
            logger.exception("Onboarding state error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.post("/v1/onboarding/teach", tags=["onboarding"], summary="Submit teach step")
    async def post_onboarding_teach(request: Request):
        """Record a TEACH step completion and advance onboarding."""
        _, user_id, auth_error = _authenticate_http_request(request)
        if auth_error:
            return auth_error

        try:
            import json as _json

            body = await request.json()
            step = body.get("step", "")

            if step not in ONBOARDING_STEPS:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unknown step: {step}"},
                )

            # Load or create state
            state_dir = pathlib.Path("sovereign_state") / "onboarding"
            state_dir.mkdir(parents=True, exist_ok=True)
            state_file = state_dir / f"{user_id}.json"

            if state_file.exists():
                state = _json.loads(state_file.read_text())
            else:
                state = {
                    "step": 0,
                    "total_steps": len(ONBOARDING_STEPS),
                    "completed": [],
                    "current": ONBOARDING_STEPS[0],
                    "profile": {},
                }

            # Record completion
            if step not in state["completed"]:
                state["completed"].append(step)

            # Merge profile data (exclude 'step' key)
            profile_update = {k: v for k, v in body.items() if k != "step"}
            state["profile"].update(profile_update)

            # Advance to next step
            step_idx = ONBOARDING_STEPS.index(step)
            next_idx = step_idx + 1
            state["step"] = next_idx
            next_step = (
                ONBOARDING_STEPS[next_idx] if next_idx < len(ONBOARDING_STEPS) else None
            )
            state["current"] = next_step or "complete"

            # Persist
            state_file.write_text(_json.dumps(state, indent=2))

            return {
                "step": step,
                "accepted": True,
                "next_step": next_step,
                "profile_update": profile_update,
            }
        except Exception:
            logger.exception("Onboarding teach error")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    import pathlib

    # ─── Terminal v1 Endpoints ──────────────────────────────────────────
    # Phase 74: Terminal spine types exposed via API.
    # Standing on: Harel (1987) statecharts, Kahneman (2002) System-1/2.
    # Per-app terminal controller instance (shared across requests)
    _terminal_controllers: dict[str, Any] = {}

    @app.get(
        "/v1/terminal/state",
        tags=["terminal"],
        summary="Get terminal state machine status",
    )
    async def get_terminal_state(request: Request):
        """Return current terminal state, execution path, and active mission."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            from core.sovereign.terminal import TerminalStateController

            session_id = request.headers.get("X-Session-ID", "default")
            if session_id not in _terminal_controllers:
                ctrl = TerminalStateController()
                ctrl.transition(
                    __import__(
                        "core.sovereign.terminal", fromlist=["TerminalState"]
                    ).TerminalState.READY
                )
                _terminal_controllers[session_id] = ctrl
            return _terminal_controllers[session_id].to_dict()
        except ImportError:
            return JSONResponse(
                status_code=503,
                content={"error": "Terminal module not available"},
            )

    @app.get(
        "/v1/terminal/briefing",
        tags=["terminal"],
        summary="Get session briefing context",
    )
    async def get_terminal_briefing(request: Request):
        """Return contextual briefing for terminal session continuity."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            from core.sovereign.terminal import BriefingContext

            wallets = getattr(runtime, "_constitutional_wallets", [])
            reflex_cache = getattr(runtime, "_constitutional_reflex_cache", {})

            wallet_snap = {}
            if wallets:
                from core.constitutional.fixed_point import fp_float

                w = wallets[0]
                wallet_snap = {
                    "seed": fp_float(getattr(w, "seed_balance", 0)),
                    "bloom": fp_float(getattr(w, "bloom_balance", 0)),
                }

            near_compile = []
            for key in list(reflex_cache.keys())[:5]:
                near_compile.append(key.hex() if isinstance(key, bytes) else str(key))

            ctx = BriefingContext(
                active_project="bizra-data-lake",
                near_compile_patterns=near_compile,
                wallet_snapshot=wallet_snap,
            )
            return ctx.to_dict()
        except ImportError:
            return JSONResponse(
                status_code=503,
                content={"error": "Terminal module not available"},
            )

    static_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        from fastapi.responses import FileResponse

        @app.get("/")
        async def root():
            return FileResponse(static_dir / "console.html")

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# =============================================================================
# CLI SERVER
# =============================================================================


def _run_fastapi_server(
    runtime: Any,
    host: str,
    port: int,
) -> None:
    """Launch the FastAPI application via uvicorn (production-grade).

    Standing on: Encode/uvicorn (2018) — ASGI server for async Python.
    Provides: console UI, CORS, OpenAPI docs at /docs, proper HTTP/1.1.
    """
    import uvicorn  # type: ignore[import-untyped]

    app = create_fastapi_app(runtime)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           SOVEREIGN NODE0 ONLINE (FastAPI + Uvicorn)         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Console: http://{host}:{port}/                               ║
║   Docs:    http://{host}:{port}/docs                           ║
║   Health:  http://{host}:{port}/v1/health                      ║
║   Query:   POST http://{host}:{port}/v1/query                  ║
║   Orch:    POST http://{host}:{port}/v1/orchestrate             ║
║                                                              ║
║   Press Ctrl+C to stop                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


async def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    api_keys: Optional[list[str]] = None,
    use_fastapi: bool = True,
) -> None:
    """
    Run the Sovereign API server.

    Defaults to FastAPI+Uvicorn for full features (console, docs, CORS).
    Falls back to pure-asyncio SovereignAPIServer if uvicorn unavailable.

    Usage:
        python -m core.sovereign.api --port 8080
    """
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(autonomous_enabled=True)

    async with SovereignRuntime.create(config) as runtime:
        # Prefer FastAPI + Uvicorn (console, OpenAPI docs, CORS, WebSocket)
        if use_fastapi:
            try:
                # uvicorn.run() manages its own loop; run in daemon thread
                import threading

                import uvicorn  # type: ignore[import-untyped]  # noqa: F401

                server_thread = threading.Thread(
                    target=_run_fastapi_server,
                    args=(runtime, host, port),
                    daemon=True,
                )
                server_thread.start()
                await runtime.wait_for_shutdown()
                return
            except ImportError:
                logger.warning(
                    "uvicorn not installed, falling back to pure-asyncio server. "
                    "Install with: pip install uvicorn"
                )

        # Fallback: pure asyncio server (no console, no docs)
        server = SovereignAPIServer(
            runtime=runtime,
            host=host,
            port=port,
            api_keys=set(api_keys) if api_keys else None,
        )

        await server.start()

        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              SOVEREIGN API SERVER RUNNING (asyncio)           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   GET  http://{host}:{port}/v1/health                          ║
║   GET  http://{host}:{port}/v1/status                          ║
║   POST http://{host}:{port}/v1/query                           ║
║                                                              ║
║   Note: Install uvicorn for Console UI + Swagger docs         ║
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
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--api-key", action="append", help="API keys (can repeat)")
    parser.add_argument(
        "--no-fastapi",
        action="store_true",
        help="Use pure-asyncio server instead of FastAPI+Uvicorn",
    )

    args = parser.parse_args()

    asyncio.run(
        serve(args.host, args.port, args.api_key, use_fastapi=not args.no_fastapi)
    )
