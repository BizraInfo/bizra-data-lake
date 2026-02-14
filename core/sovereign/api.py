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
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("sovereign.api")

# =============================================================================
# PYDANTIC MODELS (module-level for FastAPI schema generation)
# =============================================================================
try:
    from pydantic import BaseModel as _PydanticBaseModel

    class QueryRequestModel(_PydanticBaseModel):
        """FastAPI request model for /v1/query."""

        query: str
        context: dict[str, Any] = {}
        require_reasoning: bool = True
        require_validation: bool = True
        max_depth: int = 3
        timeout_ms: int = (
            300000  # 5 min — reasoning models (R1) need extended think time
        )

    class OrchestrateRequestModel(_PydanticBaseModel):
        """FastAPI request model for /v1/orchestrate."""

        task: str
        context: dict[str, Any] = {}
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

    # Phase 20: Spearpoint request models
    class SpearpointReproduceModel(_PydanticBaseModel):
        """Request model for /v1/spearpoint/reproduce — evaluation-first verification."""

        claim: str
        proposed_change: str = ""
        prompt: str = ""
        response: str = ""
        metrics: dict[str, Any] = {}

    class SpearpointImproveModel(_PydanticBaseModel):
        """Request model for /v1/spearpoint/improve — innovation through evaluator gate."""

        observation: Optional[dict[str, Any]] = None
        top_k: int = 3

    class SpearpointPatternModel(_PydanticBaseModel):
        """Request model for /v1/spearpoint/pattern — pattern-aware research via Sci-Reasoning."""

        pattern_id: str
        claim_context: str = ""
        top_k: int = 3

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
    SpearpointReproduceModel.model_rebuild()
    SpearpointImproveModel.model_rebuild()
    SpearpointPatternModel.model_rebuild()

except ImportError:
    QueryRequestModel = None  # type: ignore[assignment,misc]
    OrchestrateRequestModel = None  # type: ignore[assignment,misc]
    RegisterRequestModel = None  # type: ignore[assignment,misc]
    LoginRequestModel = None  # type: ignore[assignment,misc]
    RefreshTokenModel = None  # type: ignore[assignment,misc]
    SpearpointReproduceModel = None  # type: ignore[assignment,misc]
    SpearpointImproveModel = None  # type: ignore[assignment,misc]
    SpearpointPatternModel = None  # type: ignore[assignment,misc]

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
    version: str = "1.0.0"
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
            resp_bytes = response.encode() if isinstance(response, str) else response
            writer.write(resp_bytes)  # type: ignore[arg-type]
            await writer.drain()

        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            writer.close()

    async def _route(
        self,
        method: str,
        path: str,
        headers: dict[str, str],
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
        except Exception as e:
            logger.error(f"Query error: {e}")
            return self._json_response({"error": str(e)}, 500)

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
        except Exception as e:
            return self._json_response({"error": str(e)}, 500)

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
        except Exception as e:
            return self._json_response({"error": str(e)}, 500)

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
        except Exception as e:
            return self._json_response({"error": str(e)}, 500)

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
        except Exception as e:
            return self._json_response({"error": str(e)}, 500)

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
        _db_dir = (
            getattr(_state_dir, "state_dir", _Path("sovereign_state"))
            if _state_dir
            else _Path("sovereign_state")
        )
        _user_store = UserStore(db_path=_db_dir / "users.db")
        _jwt_auth = JWTAuth()
        _auth_middleware = AuthMiddleware(user_store=_user_store, jwt_auth=_jwt_auth)
        init_auth_middleware(_auth_middleware)
        _auth_available = True
        logger.info("Phase 21: Auth layer initialized (UserStore + JWT + Middleware)")
    except Exception as e:
        logger.warning(f"Phase 21: Auth layer not available: {e}")
        _user_store = None  # type: ignore[assignment]
        _jwt_auth = None  # type: ignore[assignment]
        _auth_middleware = None  # type: ignore[assignment]
        _auth_available = False

    app = FastAPI(
        title="Sovereign API",
        description="BIZRA Sovereign Engine REST API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — Phase 23: environment-aware origin restriction
    import os

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
        return PlainTextResponse(m.to_prometheus(include_help=False))

    @app.post("/v1/query")
    async def query(body: QueryRequestModel):
        result = await runtime.query(
            body.query,
            context=body.context,
            require_reasoning=body.require_reasoning,
            require_validation=body.require_validation,
            max_depth=body.max_depth,
            timeout_ms=body.timeout_ms,
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
    async def validate(body: ValidateRequestModel):
        """Validate content quality using the sovereign SNR and Ihsān engines.

        Returns quality scores without executing a full query pipeline.
        Useful for TUI/CLI post-hoc validation of generated content.
        """
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
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
            state_dir = (
                getattr(runtime, "_state_dir", None)
                or getattr(_runtime_cfg, "state_dir", None)
                or Path("sovereign_state")
            )
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
        except Exception as e:
            logger.error(f"Genesis verification failed: {e}")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": str(e)},
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
        except Exception as e:
            logger.error(f"Envelope verification failed: {e}")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": str(e)},
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
                ReceiptVerifier,
                SimpleSigner,
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

            receipt = Receipt(
                receipt_id=receipt_json.get("receipt_id", ""),
                status=receipt_json.get("status", "pending"),
                query_digest=receipt_json.get("query_digest", ""),
                policy_digest=receipt_json.get("policy_digest", ""),
                payload_digest=receipt_json.get("payload_digest", ""),
                snr=receipt_json.get("snr", 0.0),
                ihsan_score=receipt_json.get("ihsan_score", 0.0),
                gate_passed=receipt_json.get("gate_passed", ""),
                reason=receipt_json.get("reason"),
                signature=receipt_json.get("signature", b""),
                signer_pubkey=receipt_json.get("signer_pubkey", b""),
            )

            # Attempt verification with available signer
            signer_key = getattr(runtime, "_signer_key", None)
            if signer_key:
                signer = SimpleSigner(signer_key)
                verifier = ReceiptVerifier(signer)
                is_valid, error_msg = verifier.verify(receipt)
            else:
                is_valid = bool(
                    receipt.receipt_id and receipt.query_digest and receipt.signature
                )
                error_msg = None if is_valid else "Missing required fields"

            artifacts = {
                "receipt_id": receipt.receipt_id,
                "status": receipt.status,
                "quality": {"snr": receipt.snr, "ihsan": receipt.ihsan_score},
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
        except Exception as e:
            logger.error(f"Receipt verification failed: {e}")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": str(e)},
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
        except Exception as e:
            logger.error(f"Audit log verification failed: {e}")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": str(e)},
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
        except Exception as e:
            logger.error(f"Ledger verification failed: {e}")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": str(e)},
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
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

        except Exception as e:
            logger.error(f"PoI receipt verification failed: {e}")
            return JSONResponse(
                status_code=500,
                content=VerifierResponse.rejected(
                    reason_codes=["INVARIANT_FAILED"],
                    artifacts={"detail": str(e)},
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
            state_dir = (
                getattr(runtime, "_state_dir", None)
                or getattr(_runtime_cfg, "state_dir", None)
                or Path("sovereign_state")
            )
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
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
    async def poi_epoch():
        """Run a full PoI computation epoch.

        Computes composite PoI scores for all contributors,
        runs Gini analysis, and applies SAT rebalancing if needed.
        """
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
    async def sat_epoch():
        """Finalize a PoI epoch via SAT Controller.

        Computes PoI scores, distributes tokens, checks Gini,
        and triggers rebalancing if inequality exceeds threshold.
        """
        result = runtime.finalize_sat_epoch()
        if result is None:
            return JSONResponse(
                status_code=503,
                content={"error": "SAT Controller is not available"},
            )
        return result

    # ─── Token System Endpoints ─────────────────────────────────

    @app.get("/v1/token/balance")
    async def token_balance(account: str = "BIZRA-00000000"):
        """Get token balances for an account."""
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
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"error": f"Token system unavailable: {e}"},
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
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"error": f"Token system unavailable: {e}"},
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
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"error": f"Token system unavailable: {e}"},
            )

    # /v1/orchestrate — direct orchestrator task decomposition endpoint
    @app.post("/v1/orchestrate")
    async def orchestrate(body: OrchestrateRequestModel):
        """Decompose a complex task through the orchestrator's agent swarm."""
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    # ─── Spearpoint Endpoints ────────────────────────────────────────────
    # Standing on: Boyd (OODA), Goldratt (Theory of Constraints)
    #
    # expose reproduce (evaluation-first) and improve (innovation-through-gate)
    # missions, plus orchestrator statistics.

    @app.post("/v1/spearpoint/reproduce")
    async def spearpoint_reproduce(body: SpearpointReproduceModel):
        """Evaluate/verify a claim through the Spearpoint evaluator gate."""
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    @app.post("/v1/spearpoint/improve")
    async def spearpoint_improve(body: SpearpointImproveModel):
        """Generate and evaluate improvement hypotheses through the evaluator gate."""
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    @app.get("/v1/spearpoint/stats")
    async def spearpoint_stats():
        """Get Spearpoint Orchestrator statistics and mission history."""
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
            )

    @app.post("/v1/spearpoint/pattern")
    async def spearpoint_pattern(body: "SpearpointPatternModel"):
        """Pattern-aware research using Sci-Reasoning thinking patterns.

        Routes to SpearpointOrchestrator.research_pattern() which uses the
        15 cognitive moves from Li et al. (2025) to seed hypothesis generation.
        """
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
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)},
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
            await ws.accept()
            _ws_clients.add(ws)

            # Send welcome
            identity = runtime.status().get("identity", {})
            await ws.send_json(
                {
                    "type": "connected",
                    "node_id": identity.get("node_id", "unknown"),
                    "version": identity.get("version", "1.0.0"),
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
                        result = await runtime.query(
                            data.get("query", ""),
                            context=data.get("context", {}),
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
    async def sel_episodes(limit: int = 50, offset: int = 0):
        """list episodes from the Sovereign Experience Ledger.

        Returns episodes in reverse chronological order (newest first).
        """
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
        except Exception as e:
            logger.error(f"SEL episodes list failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/v1/sel/episodes/{episode_hash}")
    async def sel_episode_by_hash(episode_hash: str):
        """Retrieve a single episode by its content-address hash."""
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
        except Exception as e:
            logger.error(f"SEL episode lookup failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/v1/sel/retrieve")
    async def sel_retrieve(body: SELRetrieveModel):
        """Retrieve episodes using RIR (Recency-Importance-Relevance) algorithm.

        Returns top_k most relevant episodes for the given query text.
        Standing on: Park et al. (2023) — generative agent retrieval.
        """
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
        except Exception as e:
            logger.error(f"SEL retrieve failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

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
        except Exception as e:
            logger.error(f"SEL verify failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    # ─── SJE Judgment Telemetry Endpoints (Phase A: Observation) ──────

    @app.get("/v1/judgment/stats")
    async def judgment_stats():
        """Get SJE verdict distribution and entropy.

        Standing on: Shannon (1948) — entropy as uncertainty measure.
        """
        jt = getattr(runtime, "_judgment_telemetry", None)
        if jt is None:
            return JSONResponse(
                status_code=404,
                content={"error": "Judgment Telemetry not initialized"},
            )
        return jt.to_dict()

    @app.get("/v1/judgment/stability")
    async def judgment_stability():
        """Check if judgment verdicts are stable (low entropy).

        Stability indicates strong consensus toward one verdict.
        Standing on: Shannon (1948), Aristotle (practical wisdom).
        """
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

    class EpochSimulateModel(_PydanticBaseModel):
        impacts: list = []
        epoch_cap: int = 1000

    @app.post("/v1/judgment/simulate")
    async def judgment_simulate(body: EpochSimulateModel):
        """Simulate proportional epoch distribution (no tokens emitted).

        Pure mathematical rehearsal for genesis economy modeling.
        """
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
    async def proactive_suggestions():
        """Get proactive knowledge suggestions from living memory."""
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
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

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
                )
                tokens = _jwt_auth.issue_tokens(user.user_id, user.username)
                return {
                    "user": {
                        "user_id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                        "api_key": user.api_key,
                        "namespace": user.namespace,
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
            except Exception as e:
                logger.error(f"Registration failed: {e}")
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

    # ─── Wire Auth into Protected Routes ──────────────────────────────
    # Override /v1/query to propagate user_id when auth is available
    if _auth_available:
        _original_query = query  # capture the existing handler

        @app.post("/v1/query", name="query_authenticated", include_in_schema=False)
        async def query_with_auth(body: QueryRequestModel, request: Request):
            """Authenticated query — propagates user_id into the pipeline."""
            user_id = ""
            try:
                user = _auth_middleware.authenticate_request(request)
                if user is not None:
                    user_id = user.user_id
                    _user_store.increment_query_count(user_id)
            except Exception as e:
                logger.debug("Auth extraction failed (anonymous access): %s", e)

            result = await runtime.query(
                body.query,
                context=body.context,
                require_reasoning=body.require_reasoning,
                require_validation=body.require_validation,
                max_depth=body.max_depth,
                timeout_ms=body.timeout_ms,
                user_id=user_id,
            )

            response: dict[str, Any] = {
                "id": result.query_id,
                "success": result.success,
                "answer": result.response,
                "user_id": result.user_id,
                "quality": {
                    "snr": result.snr_score,
                    "ihsan": result.ihsan_score,
                },
                "timing": {
                    "total_ms": result.processing_time_ms,
                },
            }
            if result.graph_hash:
                response["graph_hash"] = result.graph_hash
            ledger = getattr(runtime, "_evidence_ledger", None)
            if ledger and hasattr(ledger, "sequence") and ledger.sequence > 0:
                response["receipt"] = {
                    "sequence": ledger.sequence,
                    "chain_hash": ledger.last_hash[:16] + "...",
                }
            return response

    import pathlib

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
