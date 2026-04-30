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
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger("sovereign.api")

from core.errors import (
    AuthorityError,
    BizraError,
    Boundary,
    BridgeError,
    ConstitutionalViolation,
    GateRejection,
    InferenceError,
    ResourceError,
    http_status_for_error,
    wrap_legacy_exception,
)

# Module-level ReflexCompiler singleton (lazy-initialized in /v1/plan)
_reflex_compiler: Any = None
_reflex_compiler_lock = threading.Lock()


def _env_truthy(var_name: str) -> bool:
    """Return True when an environment flag is explicitly enabled."""
    return os.environ.get(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _production_mode_enabled() -> bool:
    """Return True when the runtime is explicitly in production mode."""
    return os.environ.get("BIZRA_ENV", "").strip().lower() == "production"


def _resolved_api_keys(explicit_api_keys: Optional[list[str] | set[str]]) -> set[str]:
    """Merge CLI-provided API keys with supported environment fallbacks."""
    keys = {key.strip() for key in (explicit_api_keys or []) if key and key.strip()}
    for var_name in (
        "BIZRA_NODE0_API_KEY",
        "BIZRA_API_KEY",
        "BIZRA_NODE0_API_KEYS",
        "BIZRA_API_KEYS",
    ):
        raw = os.environ.get(var_name, "")
        if not raw:
            continue
        for candidate in raw.split(","):
            candidate = candidate.strip()
            if candidate:
                keys.add(candidate)
    return keys


def _ensure_production_auth_prerequisites(
    *,
    use_fastapi: bool,
    api_keys: set[str],
) -> None:
    """Fail closed when production startup lacks required auth material."""
    if not _production_mode_enabled():
        return

    if not os.environ.get("BIZRA_JWT_SECRET", "").strip():
        raise RuntimeError("BIZRA_JWT_SECRET is required in production")

    if not use_fastapi and not api_keys:
        raise RuntimeError(
            "Raw asyncio API fallback requires explicit API auth in production. "
            "Set BIZRA_NODE0_API_KEY, BIZRA_API_KEY, BIZRA_NODE0_API_KEYS, or "
            "BIZRA_API_KEYS, or pass --api-key."
        )


def _log_bizra_error(exc: BizraError) -> None:
    """Log typed boundary failures at a level that matches severity."""

    if isinstance(exc, ConstitutionalViolation):
        logger.critical("Constitutional halt: %s", exc)
        return
    if isinstance(exc, (GateRejection, AuthorityError)):
        logger.warning("Boundary rejection: %s", exc)
        return
    if isinstance(exc, (BridgeError, InferenceError, ResourceError)):
        logger.error("Boundary degradation: %s", exc)
        return
    logger.error("Typed system error: %s", exc)


def _wrap_query_error(
    exc: Exception,
    *,
    route: str,
    query_length: int,
    user_id: str = "",
) -> BizraError:
    """Convert legacy query failures into typed, receiptable errors."""

    return wrap_legacy_exception(
        exc,
        Boundary.MEMBRANE,
        context={
            "route": route,
            "query_length": query_length,
            "user_id": user_id,
        },
    )


def _record_boundary_error_via_node0(
    rt: Any,
    exc: BizraError,
    *,
    route: str,
) -> None:
    """Mirror typed boundary failures into Node0's canonical audit plane."""

    node0 = getattr(rt, "_node0", None)
    recorder = getattr(node0, "record_boundary_error_receipt", None)
    if not callable(recorder):
        return
    try:
        recorder(
            {
                **exc.to_receipt(),
                "source": "api.query.boundary",
                "route": route,
            }
        )
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as node_exc:
        logger.warning("Node0 boundary error ingest failed: %s", node_exc)


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
        kinds: Optional[list[str]] = None
        tags: Optional[list[str]] = None
        context_ids: Optional[list[str]] = None
        include_archived: bool = False
        debug_scores: bool = False

    class MemoryImportModel(_PydanticBaseModel):
        """Request model for bounded, user-provided Node0 memory import."""

        title: str
        content: str
        source_type: str = "user_text"
        tags: list[str] = _PydanticField(default_factory=list)
        owner_marker: str = "local-owner"
        consent: bool = False

    class Node0ActionIntentModel(_PydanticBaseModel):
        """Request model for bounded Node0 desktop/browser action handoff."""

        action_type: str
        target: str
        label: str = ""
        consent: bool = False

    class Node0LocalActionReceiptModel(_PydanticBaseModel):
        """Request model for recording explicit browser-client action execution."""

        action_id: str
        action_type: str
        result: str = "executed"
        execution_channel: str = "browser_client"
        user_confirmed: bool = False
        target_preview: str = ""
        target_hash: str = ""
        error: str = ""

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
        permission_envelope: Optional[dict[str, Any]] = _PydanticField(
            default=None,
            description="Mission-scoped approval envelope for terminal clients.",
        )
        proof_mode: str = _PydanticField(
            default="auto",
            description="Proof lane selection: auto | verified | standard.",
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

    class ReasoningProofResponse(_PydanticBaseModel):
        """Additive proof metadata for missions routed through VRG reasoning."""

        mode: str = "verified_graph"
        vrg_root: str = ""
        verified: bool = False
        receipt_id: str = ""
        status: str = ""
        payload_digest: str = ""
        branch_count: int = 0
        surviving_branches: int = 0
        detail: str = ""

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
            default="SYSTEM_2_NOVEL",
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
        reasoning_proof: Optional[ReasoningProofResponse] = None
        execution_authority: str = ""
        authority_path: str = ""
        fate_verdict: str = ""
        fate_reason_codes: list[str] = _PydanticField(default_factory=list)
        fate_mode: str = ""
        identity_mode: str = ""
        signer_public_key_prefix: str = ""

    class CriticalAcknowledgmentRequest(_PydanticBaseModel):
        """Request model for proof-bearing acknowledgment of a critical event."""

        event_hash: str = _PydanticField(
            ...,
            min_length=32,
            max_length=32,
            description="Terminal event hash being acknowledged.",
        )
        topic: str = _PydanticField(
            ...,
            min_length=1,
            description="Canonical topic of the critical event.",
        )
        summary: str = _PydanticField(
            default="",
            description="Human-readable summary shown to the operator.",
        )
        mission_id: str = ""
        receipt_id: str = ""

    class CriticalAcknowledgmentResponse(_PydanticBaseModel):
        """Response model for a proof-bearing critical acknowledgment."""

        acknowledgement_id: str
        receipt_id: str
        status: str = "ACKNOWLEDGED"
        hash_chain_ref: str
        acknowledged_event_hash: str
        acknowledged_topic: str
        mission_id: str = ""
        timestamp: str
        synthesis: str

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
    CriticalAcknowledgmentRequest.model_rebuild()
    CriticalAcknowledgmentResponse.model_rebuild()
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
    CriticalAcknowledgmentRequest = None  # type: ignore[assignment,misc]
    CriticalAcknowledgmentResponse = None  # type: ignore[assignment,misc]
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
        self._production_mode = _production_mode_enabled()
        if self._production_mode and not self.api_keys:
            raise RuntimeError(
                "SovereignAPIServer requires explicit API keys in production"
            )
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

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Decode error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — review needed
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
        allow_anonymous_vitals = path == "/v1/metrics/vitals" and method == "POST"

        # Check API key if configured
        if self.api_keys and not allow_anonymous_vitals:
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
        elif path == "/v1/metrics/vitals" and method == "POST":
            return await self._handle_metrics_vitals(body)
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

    @staticmethod
    def _validate_metrics_vitals_payload(data: Any) -> dict[str, Any] | None:
        """Validate Web Vitals beacon payloads from the frontend."""
        if not isinstance(data, dict):
            return None

        name = data.get("name")
        rating = data.get("rating")
        try:
            value = float(data.get("value"))
        except (TypeError, ValueError):
            return None

        if not isinstance(name, str) or not name:
            return None
        if rating not in {"good", "needs-improvement", "poor"}:
            return None

        return {"name": name, "value": value, "rating": rating}

    @classmethod
    def _accept_metrics_vitals(
        cls, data: Any
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Normalize and acknowledge a Web Vitals payload."""
        payload = cls._validate_metrics_vitals_payload(data)
        if payload is None:
            return None, {"error": "Invalid vitals payload"}

        logger.debug(
            "Web vitals beacon accepted: %s=%s (%s)",
            payload["name"],
            payload["value"],
            payload["rating"],
        )
        return payload, None

    async def _handle_metrics_vitals(self, body: bytes) -> str:
        """Accept frontend Web Vitals beacons."""
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json_response({"error": "Invalid vitals payload"}, 400)

        payload, error = self._accept_metrics_vitals(data)
        if error is not None:
            return self._json_response(error, 400)

        return self._json_response({"status": "accepted", "metric": payload["name"]})

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
        except BizraError as exc:
            _log_bizra_error(exc)
            _record_boundary_error_via_node0(self.runtime, exc, route="/v1/query")
            return self._json_response(exc.to_receipt(), http_status_for_error(exc))
        except Exception as exc:  # noqa: BLE001 — API boundary
            wrapped = _wrap_query_error(
                exc,
                route="/v1/query",
                query_length=len(data.get("query", "")) if "data" in locals() else 0,
            )
            logger.exception("Query error (legacy)")
            _record_boundary_error_via_node0(self.runtime, wrapped, route="/v1/query")
            return self._json_response(
                wrapped.to_receipt(),
                http_status_for_error(wrapped),
            )

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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return self._json_response({"error": str(exc) or "Operation failed"}, 500)
        except Exception:  # noqa: BLE001 — API boundary
            logger.exception("Seed episodes error")
            return self._json_response({"error": "Internal server error"}, 500)

    def _json_response(self, data: dict[str, Any], status: int = 200) -> str:
        """Build JSON HTTP response."""
        body = json.dumps(data)
        status_text = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            403: "Forbidden",
            404: "Not Found",
            422: "Unprocessable Entity",
            429: "Too Many Requests",
            502: "Bad Gateway",
            503: "Service Unavailable",
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
    except (ValueError, KeyError, PermissionError) as exc:
        if _production_mode_enabled():
            raise RuntimeError(
                "Authentication layer failed to initialize in production"
            ) from exc
        logger.warning("Auth init error (specific): %s", exc)
        _user_store = None  # type: ignore[assignment]
        _jwt_auth = None  # type: ignore[assignment]
        _auth_middleware = None  # type: ignore[assignment]
        _auth_available = False
    except Exception as e:  # noqa: BLE001 — review needed
        if _production_mode_enabled():
            raise RuntimeError(
                "Authentication layer failed to initialize in production"
            ) from e
        logger.error(
            "SECURITY: Auth layer failed to initialize: %s. "
            "Protected endpoints will deny requests until auth is restored.",
            e,
        )
        _user_store = None  # type: ignore[assignment]
        _jwt_auth = None  # type: ignore[assignment]
        _auth_middleware = None  # type: ignore[assignment]
        _auth_available = False

    from core.inference.model_routing import DEFAULT_MODEL_ROUTING
    from core.sovereign.atomic_io import atomic_write_json, read_json
    from core.sovereign.terminal import PermissionEnvelope as TerminalPermissionEnvelope

    _topic_aliases = {
        "policy.invariant.violation": "invariant.violation",
    }
    _event_severity_by_topic = {
        "mission.created": "info",
        "mission.executed": "info",
        "mission.verified": "info",
        "mission.failed": "warning",
        "economy.seed_minted": "notice",
        "economy.zakat": "info",
        "economy.bloom_accrued": "info",
        "economy.asabiyyah": "info",
        "reflex.compiled": "notice",
        "ihsan.breach": "critical",
        "invariant.violation": "critical",
        "auth.boundary.crossed": "warning",
        "critical.acknowledged": "notice",
        "receipt.generated": "info",
        "receipt.verified": "info",
        "tick.completed": "info",
    }
    _model_routing_path = _db_dir / "model_routing.json"
    _terminal_event_history: deque[dict[str, Any]] = deque(maxlen=500)
    _terminal_event_chain_head = ""
    _ws_clients: dict[Any, set[str]] = {}

    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _json_safe(value.to_dict())
        if hasattr(value, "value"):
            return _json_safe(value.value)
        if isinstance(value, bytes):
            return value.hex()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _canonical_topic_name(topic: str) -> str:
        return _topic_aliases.get(topic, topic)

    def _normalize_receipt_status(status: Any) -> str:
        value = str(status or "FAILED").upper()
        if value == "COMPLETED":
            value = "COMPLETE"
        if value not in {"COMPLETE", "PARTIAL", "FAILED", "BLOCKED"}:
            return "FAILED"
        return value

    def _normalize_execution_path(path: Any) -> str:
        value = str(path or "").strip()
        mapping = {
            "system_1": "SYSTEM_1_CACHE_HIT",
            "system_1_reflex": "SYSTEM_1_CACHE_HIT",
            "SYSTEM_1": "SYSTEM_1_CACHE_HIT",
            "SYSTEM_1_CACHE_HIT": "SYSTEM_1_CACHE_HIT",
            "system_2": "SYSTEM_2_NOVEL",
            "SYSTEM_2": "SYSTEM_2_NOVEL",
            "SYSTEM_2_NOVEL": "SYSTEM_2_NOVEL",
            "mixed": "MIXED",
            "MIXED": "MIXED",
        }
        return mapping.get(value, "SYSTEM_2_NOVEL")

    def _runtime_canonical_mode_enabled(runtime_obj: Any) -> bool:
        if getattr(runtime_obj, "_canonical_mode", False) is True:
            return True
        try:
            status_fn = getattr(runtime_obj, "status", None)
            if callable(status_fn):
                status = status_fn()
                if isinstance(status, dict):
                    return status.get("canonical", {}).get("enabled") is True
        except Exception:  # noqa: BLE001 - status is best-effort here
            logger.debug("Failed to inspect runtime canonical mode", exc_info=True)
        return False

    def _runtime_has_canonical_mission_authority(runtime_obj: Any) -> bool:
        runtime_mission = getattr(runtime_obj, "mission", None)
        organism = getattr(runtime_obj, "_organism", None)
        return organism is not None and (
            asyncio.iscoroutinefunction(runtime_mission)
            or str(type(runtime_mission)).endswith("AsyncMock'>")
        )

    def _runtime_reflex_lineage_payload(
        runtime_receipt: Any,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, str, float, float]:
        from core.integration.constants import REFLEX_PRECIPITATION_HITS

        reflex_delta = {
            "compiled": False,
            "near_compile": False,
            "compile_count": 0,
            "threshold": REFLEX_PRECIPITATION_HITS,
        }
        compiled_event: dict[str, Any] | None = None
        reflex_pattern = ""
        reflex_latency_ms = 0.0
        comparison_s2_avg_ms = 0.0

        metadata = getattr(runtime_receipt, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return (
                reflex_delta,
                compiled_event,
                reflex_pattern,
                reflex_latency_ms,
                comparison_s2_avg_ms,
            )

        raw_delta = metadata.get("reflex_delta")
        if isinstance(raw_delta, dict):
            reflex_delta = {
                "compiled": bool(raw_delta.get("compiled", False)),
                "near_compile": bool(raw_delta.get("near_compile", False)),
                "compile_count": int(raw_delta.get("compile_count", 0) or 0),
                "threshold": int(
                    raw_delta.get("threshold", REFLEX_PRECIPITATION_HITS)
                    or REFLEX_PRECIPITATION_HITS
                ),
            }

        raw_event = metadata.get("compiled_reflex_event")
        if isinstance(raw_event, dict):
            compiled_event = {
                "name": str(raw_event.get("name", ""))[:120],
                "pattern_hash": str(raw_event.get("pattern_hash", "")),
                "avg_ihsan": round(float(raw_event.get("avg_ihsan", 0.0) or 0.0), 4),
                "execution_count": int(raw_event.get("execution_count", 0) or 0),
                "precipitation_count": int(
                    raw_event.get("precipitation_count", 0) or 0
                ),
            }

        reflex_pattern = str(metadata.get("reflex_pattern", "") or "")
        reflex_latency_ms = round(
            float(metadata.get("reflex_latency_ms", 0.0) or 0.0),
            2,
        )
        comparison_s2_avg_ms = round(
            float(metadata.get("comparison_s2_avg_ms", 0.0) or 0.0),
            2,
        )
        return (
            reflex_delta,
            compiled_event,
            reflex_pattern,
            reflex_latency_ms,
            comparison_s2_avg_ms,
        )

    def _load_model_routing() -> dict[str, str]:
        stored = read_json(_model_routing_path, default={})
        routing = dict(DEFAULT_MODEL_ROUTING)
        if isinstance(stored, dict):
            for key, value in stored.items():
                if isinstance(key, str) and isinstance(value, str) and value.strip():
                    routing[key] = value.strip()
        return routing

    def _current_model_routing() -> dict[str, str]:
        routing = getattr(runtime, "_terminal_model_routing", None)
        if not isinstance(routing, dict):
            routing = _load_model_routing()
            runtime._terminal_model_routing = routing
        return dict(routing)

    def _persist_model_routing(routing: dict[str, str]) -> dict[str, str]:
        merged = dict(DEFAULT_MODEL_ROUTING)
        for key, value in routing.items():
            if isinstance(key, str) and isinstance(value, str) and value.strip():
                merged[key] = value.strip()
        atomic_write_json(_model_routing_path, merged)
        runtime._terminal_model_routing = merged
        return merged

    def _default_permission_policy() -> dict[str, Any]:
        configured = getattr(runtime, "_terminal_permission_defaults", None)
        if isinstance(configured, dict):
            return _json_safe(configured)
        return TerminalPermissionEnvelope().to_dict()

    def _wallet_snapshot() -> dict[str, float]:
        wallets = getattr(runtime, "_constitutional_wallets", [])
        if not wallets:
            return {"seed": 0.0, "bloom": 0.0}
        try:
            from core.constitutional.fixed_point import fp_float

            wallet = wallets[0]
            return {
                "seed": fp_float(getattr(wallet, "seed_balance", 0)),
                "bloom": fp_float(getattr(wallet, "bloom_balance", 0)),
            }
        except Exception:  # noqa: BLE001 - read model fallback
            return {"seed": 0.0, "bloom": 0.0}

    def _recent_episodes(limit: int = 10) -> list[dict[str, Any]]:
        seed_engine = getattr(runtime, "_seed_engine", None)
        if seed_engine is None:
            return []
        try:
            episodes = seed_engine.recent_episodes(limit=limit)
        except Exception:  # noqa: BLE001 - read model fallback
            return []
        return episodes if isinstance(episodes, list) else []

    def _last_mission_summary() -> str:
        episodes = _recent_episodes(limit=1)
        if not episodes:
            return ""
        latest = episodes[-1]
        return (
            f"Episode {latest.get('index', '?')} "
            f"qualified={latest.get('qualified', False)} "
            f"Ihsan {float(latest.get('ihsan', 0.0)):.2f}"
        )

    def _auth_state_label() -> str:
        from core.auth.middleware import _anonymous_auth_allowed

        if _auth_available and _auth_middleware is not None:
            if _anonymous_auth_allowed() and not _production_mode_enabled():
                return "anonymous-dev"
            return "authenticated"
        if _anonymous_auth_allowed() and not _production_mode_enabled():
            return "anonymous-dev"
        return "unavailable"

    def _health_snapshot() -> dict[str, Any]:
        status = runtime.status()
        last_tick = getattr(runtime, "_last_tick_result", None)
        gini = 0.0
        asabiyyah = 0.0
        minted_ihsan = 0.0
        minted_snr = 0.0
        if last_tick is not None:
            try:
                from core.constitutional.fixed_point import fp_float

                gini = fp_float(getattr(last_tick, "network_gini", 0))
                asabiyyah = fp_float(getattr(last_tick, "network_asabiyyah", 0))
                minted_ihsan = fp_float(getattr(last_tick, "avg_ihsan", 0))
                minted_snr = fp_float(getattr(last_tick, "avg_snr", 0))
            except Exception:  # noqa: BLE001 - read model fallback
                gini = 0.0
                asabiyyah = 0.0
        live = bool(status.get("state", {}).get("running", True))
        return {
            "status": status.get("health", {}).get("status", "unknown"),
            "tier": "terminal",
            "live_status": "LIVE" if live else "OFFLINE",
            "running": live,
            "ihsan_score": round(minted_ihsan, 4),
            "snr_score": round(minted_snr, 4),
            "gini": round(gini, 4),
            "asabiyyah": round(asabiyyah, 4),
            "last_tick_timestamp": getattr(runtime, "_last_tick_timestamp", ""),
            "tick_interval_s": getattr(runtime, "_tick_interval_s", _tick_interval_s),
            "wallet_snapshot": _wallet_snapshot(),
            "last_mission_summary": _last_mission_summary(),
            "auth_state": _auth_state_label(),
            "runtime_mode": os.environ.get("BIZRA_ENV", "development") or "development",
            "model_routing": _current_model_routing(),
            "permission_defaults": _default_permission_policy(),
            "version": status.get("identity", {}).get("version", ""),
            "env": os.environ.get("BIZRA_ENV", "development") or "development",
            "critical_subsystems": status.get("health", {}).get(
                "critical_subsystems", {}
            ),
        }

    def _load_spearpoint_campaign_summary() -> dict[str, Any]:
        """Load the latest local True Spearpoint campaign summary."""
        campaign_dir = os.environ.get(
            "BIZRA_SPEARPOINT_CAMPAIGN_DIR",
            "/tmp/spearpoint-campaign",
        ).strip()
        if not campaign_dir:
            return {
                "status": "unknown",
                "artifact_status": "not_configured",
                "reason": "BIZRA_SPEARPOINT_CAMPAIGN_DIR is empty",
                "official_submission": False,
            }

        summary_path = _Path(campaign_dir) / "campaign_summary.json"
        if not summary_path.exists():
            return {
                "status": "unknown",
                "artifact_status": "missing",
                "reason": "campaign_summary.json not found",
                "official_submission": False,
            }

        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "fail",
                "artifact_status": "invalid",
                "reason": str(exc),
                "official_submission": False,
            }

        if not isinstance(payload, dict):
            return {
                "status": "fail",
                "artifact_status": "invalid",
                "reason": "campaign_summary.json must contain a JSON object",
                "official_submission": False,
            }

        targets = payload.get("targets", [])
        target_summaries = []
        all_gates_passed = True
        if isinstance(targets, list):
            for item in targets:
                if not isinstance(item, dict):
                    all_gates_passed = False
                    continue
                gates = item.get("gates", {})
                gate_passed = isinstance(gates, dict) and all(
                    isinstance(gate, dict) and gate.get("passed") is True
                    for gate in gates.values()
                )
                all_gates_passed = all_gates_passed and gate_passed
                target_summaries.append(
                    {
                        "target": str(item.get("target", "")),
                        "baseline_score": item.get("baseline_score"),
                        "final_score": item.get("final_score"),
                        "gates_passed": gate_passed,
                    }
                )
        else:
            all_gates_passed = False

        summary_status = str(payload.get("status", "unknown")).lower()
        status_label = (
            "pass"
            if summary_status == "success" and all_gates_passed and target_summaries
            else "fail"
        )
        return {
            "status": status_label,
            "artifact_status": "found",
            "run_id": str(payload.get("run_id", "")),
            "mode": str(payload.get("mode", "")),
            "timestamp_utc": str(payload.get("timestamp_utc", "")),
            "targets_completed": len(target_summaries),
            "targets": target_summaries,
            "official_submission": False,
            "classification": "internal_strict_harness",
        }

    def _node0_readiness_snapshot() -> dict[str, Any]:
        """Compose Node0 product, boot, proof, and evaluation readiness."""
        health = _health_snapshot()
        node0 = getattr(runtime, "_node0", None)
        agent_db = getattr(runtime, "_agent_db", None)
        node0_health: dict[str, Any] = {}
        boot_error = ""

        if node0 is not None:
            health_fn = getattr(node0, "health", None)
            if callable(health_fn):
                try:
                    raw_health = health_fn()
                    if isinstance(raw_health, dict):
                        node0_health = _json_safe(raw_health)
                except (
                    RuntimeError,
                    AttributeError,
                    TypeError,
                    ValueError,
                    OSError,
                ) as exc:
                    boot_error = str(exc)

        booted = bool(node0_health.get("booted", False))
        if node0 is None:
            boot_status = "unavailable"
        elif boot_error:
            boot_status = "error"
        elif booted:
            boot_status = "booted"
        else:
            boot_status = "not_booted"

        spearpoint = _load_spearpoint_campaign_summary()
        runtime_live = bool(health.get("running", False))
        product_shell_available = True
        proof_surface_available = True
        memory_import_available = agent_db is not None
        memory_import_count = 0
        if agent_db is not None:
            try:
                raw_stats = agent_db.stats()
                if isinstance(raw_stats, dict):
                    for stats_key in (
                        "total_records",
                        "total_entries",
                        "active_records",
                    ):
                        if stats_key in raw_stats:
                            memory_import_count = int(raw_stats[stats_key] or 0)
                            break
            except (RuntimeError, TypeError, ValueError, AttributeError):
                memory_import_available = False
        spearpoint_passed = spearpoint.get("status") == "pass"

        if runtime_live and booted and spearpoint_passed:
            readiness = "green"
            next_action = "submit mission"
        elif runtime_live and proof_surface_available:
            readiness = "yellow"
            if not booted:
                next_action = "start Node0 boot service"
            elif not spearpoint_passed:
                next_action = "run internal Spearpoint strict campaign"
            else:
                next_action = "submit mission"
        else:
            readiness = "red"
            next_action = "start or repair Dema service"

        return {
            "status": readiness,
            "generated_at": _utcnow_iso(),
            "product_shell": {
                "available": product_shell_available,
                "version": "0.1",
                "default_view": "node0",
            },
            "proof_surface": {
                "available": proof_surface_available,
                "source": "mission_receipt",
            },
            "runtime": {
                "live": runtime_live,
                "state": health.get("status", "unknown"),
                "ihsan_score": health.get("ihsan_score", 0.0),
                "snr_score": health.get("snr_score", 0.0),
            },
            "boot_service": {
                "status": boot_status,
                "booted": booted,
                "node_id": str(node0_health.get("node_id", "")),
                "total_breaths": int(node0_health.get("total_breaths", 0) or 0),
                "chain_hash": str(node0_health.get("chain_hash", "")),
                "error": boot_error,
            },
            "memory_import": {
                "available": memory_import_available,
                "status": "ready" if memory_import_available else "unavailable",
                "mode": "single_user_provided_record",
                "imported_records": memory_import_count,
                "requires_consent": True,
                "source": "agent_db",
                "truth_label": "[ENFORCEMENT: WIRED]",
            },
            "voice_input": {
                "available": True,
                "status": "browser_required",
                "mode": "browser_speech_recognition",
                "requires_user_gesture": True,
                "stores_audio": False,
                "auto_submit": False,
                "truth_label": "[ENFORCEMENT: WIRED]",
            },
            "desktop_browser_action": {
                "available": True,
                "status": "preview_only",
                "mode": "client_handoff_only",
                "allowed_actions": ["open_url", "copy_text"],
                "requires_user_confirmation": True,
                "server_executes": False,
                "truth_label": "[ENFORCEMENT: WIRED]",
            },
            "local_action_executor": {
                "available": True,
                "status": "browser_client_ready",
                "mode": "explicit_user_gesture",
                "allowed_actions": ["copy_text", "open_url"],
                "requires_user_confirmation": True,
                "server_executes": False,
                "records_receipts": True,
                "truth_label": "[ENFORCEMENT: WIRED]",
            },
            "spearpoint": spearpoint,
            "next_action": next_action,
        }

    def _schedule_bus_event(
        topic: str, payload: dict[str, Any], source: str = "api"
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(_emit_bus_event(topic, payload, source=source))

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
    runtime._tick_interval_s = _tick_interval_s  # type: ignore[attr-defined]

    def _topic_matches_subscription(topic: str, subscriptions: set[str]) -> bool:
        if not subscriptions:
            return True
        canonical = _canonical_topic_name(topic)
        for subscription in subscriptions:
            if subscription in {"*", "all"}:
                return True
            if subscription == canonical:
                return True
            if subscription.endswith(".*") and canonical.startswith(subscription[:-1]):
                return True
        return False

    def _history_events(
        limit: int = 100, subscriptions: set[str] | None = None
    ) -> list[dict[str, Any]]:
        subscriptions = subscriptions or set()
        items = [
            event
            for event in _terminal_event_history
            if _topic_matches_subscription(event["topic"], subscriptions)
        ]
        return items[-max(1, min(limit, 100)) :]

    def _record_terminal_event(
        topic: str, payload: dict[str, Any], source: str
    ) -> dict[str, Any]:
        nonlocal _terminal_event_chain_head

        canonical_topic = _canonical_topic_name(topic)
        normalized_payload = _json_safe(payload)
        mission_id = str(normalized_payload.get("mission_id", "") or "")
        receipt_id = str(
            normalized_payload.get("receipt_id", "")
            or normalized_payload.get("evidence_receipt_id", "")
        )
        timestamp = _utcnow_iso()
        prev_hash = _terminal_event_chain_head
        digest_material = json.dumps(
            {
                "topic": canonical_topic,
                "mission_id": mission_id,
                "receipt_id": receipt_id,
                "timestamp": timestamp,
                "payload": normalized_payload,
                "prev_hash": prev_hash,
                "source": source,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        event_hash = hashlib.blake2b(
            digest_material.encode("utf-8"), digest_size=16
        ).hexdigest()
        envelope = {
            "topic": canonical_topic,
            "severity": _event_severity_by_topic.get(canonical_topic, "info"),
            "mission_id": mission_id,
            "receipt_id": receipt_id,
            "event_hash": event_hash,
            "prev_hash": prev_hash,
            "timestamp": timestamp,
            "payload": normalized_payload,
            "source": source,
        }
        _terminal_event_history.append(envelope)
        _terminal_event_chain_head = event_hash
        return envelope

    async def _push_terminal_event(event: dict[str, Any]) -> int:
        sent = 0
        disconnected: list[Any] = []
        for ws, subscriptions in list(_ws_clients.items()):
            if not _topic_matches_subscription(event["topic"], subscriptions):
                continue
            try:
                await ws.send_json({"type": "event", "event": event})
                sent += 1
            except Exception:  # noqa: BLE001 - websocket boundary
                disconnected.append(ws)
        for ws in disconnected:
            _ws_clients.pop(ws, None)
        return sent

    async def _emit_bus_event(
        topic: str, payload: dict[str, Any], source: str = "heartbeat"
    ) -> dict[str, Any] | None:
        """Emit an event to the sovereign EventBus (fire-and-forget)."""
        event = _record_terminal_event(topic, payload, source)
        try:
            from core.sovereign.event_bus import get_event_bus

            bus = get_event_bus()
            await bus.emit(topic=topic, payload=payload, source=source)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — review needed
            logger.debug("EventBus emit failed for topic=%s", topic, exc_info=True)
        await _push_terminal_event(event)
        return event

    def _mission_proof_requested(
        proof_mode: str,
        *,
        source: str,
        permission_envelope: dict[str, Any] | None,
    ) -> bool:
        normalized_mode = proof_mode.strip().lower()
        if normalized_mode in {"verified", "required", "proof"}:
            return True
        if normalized_mode in {"off", "none", "disabled", "standard"}:
            return False
        return source == "terminal" or isinstance(permission_envelope, dict)

    def _resolve_mission_got_bridge() -> Any | None:
        runtime_state = getattr(runtime, "__dict__", {})
        candidate = runtime_state.get("_got_bridge")
        candidate_reason_verified = getattr(candidate, "reason_verified", None)
        if candidate is not None and asyncio.iscoroutinefunction(
            candidate_reason_verified
        ):
            return candidate

        graph_engine = runtime_state.get("_graph_reasoner")
        try:
            from core.reasoning.got_bridge import GoTBridge

            return GoTBridge(got_engine=graph_engine)
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
            logger.debug("Mission VRG bridge unavailable", exc_info=True)
            return None

    async def _build_reasoning_proof(
        *,
        description: str,
        source: str,
        proof_mode: str,
        permission_envelope: dict[str, Any] | None,
        mission_id: str,
        mission_receipt_id: str,
        execution_path: str,
        mission_result: Any,
    ) -> dict[str, Any] | None:
        if not _mission_proof_requested(
            proof_mode,
            source=source,
            permission_envelope=permission_envelope,
        ):
            return None

        bridge = _resolve_mission_got_bridge()
        if bridge is None or not asyncio.iscoroutinefunction(
            getattr(bridge, "reason_verified", None)
        ):
            return {
                "mode": "verified_graph",
                "vrg_root": "",
                "verified": False,
                "receipt_id": "",
                "status": "UNAVAILABLE",
                "payload_digest": "",
                "branch_count": 0,
                "surviving_branches": 0,
                "detail": "reason_verified_unavailable",
            }

        context_facts = [
            f"mission_id={mission_id}",
            f"mission_receipt_id={mission_receipt_id}",
            f"source={source}",
            f"execution_path={execution_path}",
        ]
        synthesis = str(getattr(mission_result, "synthesis", "") or "").strip()
        if synthesis:
            context_facts.append(f"synthesis={synthesis[:400]}")
        for channel in getattr(mission_result, "channels_executed", [])[:8]:
            try:
                context_facts.append(
                    "channel="
                    f"{getattr(channel, 'channel', 'unknown')}"
                    f"|success={bool(getattr(channel, 'success', False))}"
                    f"|duration_ms={round(float(getattr(channel, 'duration_ms', 0.0)), 1)}"
                )
            except (AttributeError, TypeError, ValueError, OSError):
                continue

        context = {
            "domain": "mission_execution",
            "facts": context_facts,
            "mission_id": mission_id,
            "mission_receipt_id": mission_receipt_id,
            "permission_envelope": permission_envelope or {},
            "source": source,
        }

        try:
            verified_result = await bridge.reason_verified(description, context=context)
            proof_receipt = verified_result.receipt
            proof_status = getattr(
                getattr(proof_receipt, "status", ""),
                "value",
                str(getattr(proof_receipt, "status", "")),
            )
            branch_count = len(verified_result.branch_certificates)
            surviving_branches = sum(
                1
                for certificate in verified_result.branch_certificates
                if certificate.get("included_in_root")
            )
            return {
                "mode": "verified_graph",
                "vrg_root": str(verified_result.vrg_root),
                "verified": bool(verified_result.verified),
                "receipt_id": str(getattr(proof_receipt, "receipt_id", "") or ""),
                "status": str(proof_status),
                "payload_digest": getattr(
                    getattr(proof_receipt, "payload_digest", b""),
                    "hex",
                    lambda: "",
                )(),
                "branch_count": branch_count,
                "surviving_branches": surviving_branches,
                "detail": str(getattr(proof_receipt, "reason", "") or ""),
            }
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            OSError,
        ):
            logger.debug("Mission VRG proof generation failed", exc_info=True)
            return {
                "mode": "verified_graph",
                "vrg_root": "",
                "verified": False,
                "receipt_id": "",
                "status": "UNAVAILABLE",
                "payload_digest": "",
                "branch_count": 0,
                "surviving_branches": 0,
                "detail": "reason_verified_failed",
            }

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
        if fp_float(getattr(result, "network_gini", 0)) > 0.35:
            await _emit_bus_event(
                "invariant.violation",
                {
                    "metric": "gini",
                    "gini": fp_float(getattr(result, "network_gini", 0)),
                    "threshold": 0.35,
                },
                source="tick",
            )
        await _emit_bus_event(
            "tick.completed",
            {
                "scored": result.scored,
                "rejected": result.rejected,
                "minted": fp_float(result.total_minted),
                "reflexes": len(reflex_cache),
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
                runtime._last_tick_result = result  # type: ignore[attr-defined]
                runtime._last_tick_timestamp = _utcnow_iso()  # type: ignore[attr-defined]

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
            except Exception:  # noqa: BLE001 — review needed
                logger.exception("Constitutional tick error (will retry next interval)")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app_instance: Any):  # type: ignore[override]
        """FastAPI lifespan: start/stop the API-owned background heartbeat only."""
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
            except (ValueError, KeyError, PermissionError) as exc:
                logger.warning("Auth error (specific): %s", exc)
                _schedule_bus_event(
                    "auth.boundary.crossed",
                    {
                        "path": str(request.url.path),
                        "reason": "auth_error",
                        "detail": str(exc),
                        "transport": "http",
                    },
                    source="auth",
                )
                return (
                    "",
                    None,
                    JSONResponse(
                        status_code=500,
                        content={"error": str(exc) or "Operation failed"},
                    ),
                )
            except Exception as e:  # noqa: BLE001 — review needed
                if not allow_anonymous:
                    _schedule_bus_event(
                        "auth.boundary.crossed",
                        {
                            "path": str(request.url.path),
                            "reason": "authentication_required",
                            "transport": "http",
                        },
                        source="auth",
                    )
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
                    _schedule_bus_event(
                        "auth.boundary.crossed",
                        {
                            "path": str(request.url.path),
                            "reason": "missing_credentials",
                            "transport": "http",
                        },
                        source="auth",
                    )
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
                _schedule_bus_event(
                    "auth.boundary.crossed",
                    {
                        "path": str(request.url.path),
                        "reason": "rate_limit_exceeded",
                        "transport": "http",
                        "user_id": user.user_id,
                    },
                    source="auth",
                )
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
            _schedule_bus_event(
                "auth.boundary.crossed",
                {
                    "path": str(request.url.path),
                    "reason": "auth_service_unavailable",
                    "transport": "http",
                },
                source="auth",
            )
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
            except (ValueError, KeyError, PermissionError) as exc:
                logger.warning("Auth error (specific): %s", exc)
                await _emit_bus_event(
                    "auth.boundary.crossed",
                    {
                        "path": str(getattr(ws, "url", "")),
                        "reason": "auth_error",
                        "detail": str(exc),
                        "transport": "websocket",
                    },
                    source="auth",
                )
                await ws.close(code=1011, reason="Authentication failed")
                return "", False
            except Exception:  # noqa: BLE001 — review needed
                user = None

            if user is None and not allow_anonymous:
                await _emit_bus_event(
                    "auth.boundary.crossed",
                    {
                        "path": str(getattr(ws, "url", "")),
                        "reason": "authentication_required",
                        "transport": "websocket",
                    },
                    source="auth",
                )
                await ws.close(code=4401, reason="Authentication required")
                return "", False

            if user is not None and not _auth_middleware.check_rate_limit(user.user_id):
                await _emit_bus_event(
                    "auth.boundary.crossed",
                    {
                        "path": str(getattr(ws, "url", "")),
                        "reason": "rate_limit_exceeded",
                        "transport": "websocket",
                        "user_id": user.user_id,
                    },
                    source="auth",
                )
                await ws.close(code=4429, reason="Rate limit exceeded")
                return "", False

            return (user.user_id if user is not None else ""), True

        if not allow_anonymous:
            await _emit_bus_event(
                "auth.boundary.crossed",
                {
                    "path": str(getattr(ws, "url", "")),
                    "reason": "auth_service_unavailable",
                    "transport": "websocket",
                },
                source="auth",
            )
            await ws.close(code=1013, reason="Authentication service unavailable")
            return "", False

        return "", True

    # ── Mission → Constitutional Tick Bridge ──────────────────────────
    #
    # Converts completed mission results into ActionReceipts and submits
    # them to the constitutional tick queue. This wires the reflex cache:
    # mission → receipt → tick Step 10 → reflex compilation for ihsan ≥ 0.98.

    def _submit_mission_to_tick(rt: Any, mission_result: Any) -> None:
        """Bridge mission results into the constitutional tick queue.

        .. deprecated::
            Legacy path — only used when Node0 is not booted AND canonical
            mode is disabled. In canonical mode, all missions route through
            Node0 ingest authority via _ingest_via_node0().
        """
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Verification error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — review needed
            # Never block mission return for tick wiring failure
            logger.debug("Tick bridge emission failed", exc_info=True)

    def _ingest_via_node0(rt: Any, mission_result: Any) -> None:
        """Route mission receipt through the canonical Node0 ingest authority.

        When Node0Heartbeat is available, feeds the mission into the
        evidence chain → memory → reflex path.  Falls back to the
        legacy _submit_mission_to_tick() when Node0 is not booted.

        In canonical mode, fallback to legacy tick bridge is forbidden
        (fail-closed per Nakamoto single-authority principle).

        Standing on Giants:
          Nakamoto (2008) — one chain, one authority
          Deming (1950)   — PDCA: every mission closes through one loop
        """
        canonical = _runtime_canonical_mode_enabled(rt)
        node0 = getattr(rt, "_node0", None)
        if node0 is not None:
            try:
                mission_id = getattr(mission_result, "mission_id", "") or ""
                ihsan = getattr(mission_result, "ihsan_score", 0.0) or 0.0
                snr = getattr(mission_result, "snr_score", 0.0) or 0.0
                node0.ingest_mission_receipt(
                    {
                        "mission_id": mission_id,
                        "description": str(
                            getattr(mission_result, "synthesis", "") or ""
                        )[:200],
                        "ihsan_score": ihsan,
                        "snr_score": snr,
                        "agent_id": "api",
                        "gate_passed": ihsan
                        >= 0.95,  # Constitutional standard, not degradation floor
                        "duration_ms": getattr(mission_result, "duration_ms", 0.0)
                        or 0.0,
                    }
                )
                return
            except Exception:  # noqa: BLE001 — fall through to legacy
                if canonical:
                    logger.error(
                        "Node0 ingest failed in canonical mode — "
                        "refusing fallback to legacy tick bridge"
                    )
                    return
                logger.debug(
                    "Node0 ingest failed, falling back to legacy tick bridge",
                    exc_info=True,
                )

        # Fallback: legacy tick bridge (when Node0 not available)
        if canonical:
            logger.warning(
                "Canonical mode: legacy tick bridge suppressed "
                "(Node0 unavailable — mission receipt dropped)"
            )
            return
        _submit_mission_to_tick(rt, mission_result)

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
    async def health_deep(request: Request):
        """Deep health — full 11-subsystem audit, <500ms. Auth required (topology leak fix)."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
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

    @app.get(
        "/v1/health/constitutional",
        tags=["health"],
        summary="Constitutional Membrane Network invariant status",
    )
    async def health_constitutional(request: Request):
        """CMN invariant check — validates S∧M∧Z∧R constitutional properties.

        Returns the four invariants (Sovereignty, Membrane, Zann Zero, Riba Zero),
        composite Ihsan score, and the BLAKE3-chained health receipt hash.
        Auth required (topology leak fix).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        cmn = getattr(runtime, "_cmn_runtime", None)
        if cmn is None:
            return {
                "status": "not_initialized",
                "invariants": {},
                "ihsan_score": 0.0,
                "message": "CMN runtime not booted",
            }
        return cmn.constitutional_health()

    @app.get("/v1/health", tags=["health"])
    async def health(request: Request):
        """Terminal read model for Dashboard and Settings surfaces. Auth required."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        return _health_snapshot()

    @app.get(
        "/v1/node0/readiness",
        tags=["node0"],
        summary="Node0 product, boot, proof, and evaluation readiness",
    )
    async def node0_readiness(request: Request):
        """Return the operator-facing Node0 readiness contract."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        return _node0_readiness_snapshot()

    @app.get("/v1/status", tags=["health"], summary="Runtime status snapshot")
    async def status(request: Request):
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        base = runtime.status()
        # Enrich with ReflexCompiler telemetry when available
        if _reflex_compiler is not None:
            try:
                base["reflex_compiler"] = _reflex_compiler.get_status()
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Read error (specific): %s", exc)
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc) or "Operation failed"},
                )
            except Exception:  # noqa: BLE001 — review needed
                pass
        return base

    @app.get(
        "/v1/chain",
        tags=["trust"],
        summary="Authoritative receipt chain head (proxied from cognition-gateway)",
        description=(
            "Thin proxy to the Rust cognition-gateway's GET /chain endpoint. "
            "Returns {head, length, latestTimestamp, sovereignEnvelopes?, "
            "sovereignEntries?} — the authoritative chain state for Dema's "
            "trust surface. Forwards verbatim (no reshaping, no simulation). "
            "If the gateway is unreachable, returns 503 with a structured "
            "gateway_unreachable payload so the UI reveals the truth of an "
            "offline backend rather than fabricating a healthy response. "
            "Gateway URL resolves from BIZRA_COGNITION_GATEWAY_URL env var, "
            "default http://localhost:7421."
        ),
    )
    async def get_chain_head(request: Request):
        """Proxy to Rust cognition-gateway for authoritative chain state.

        Node0 Closure Sprint — row 6 (trust_surface) — 2026-04-21.

        Public (no auth) — chain head is public truth for transparency, same
        as ``dema chain`` CLI (no auth required against the Rust gateway).
        The chain is the evidence of lawful operation; anyone holding the
        machine can inspect it. Auth gates are for write operations, not
        for reading the chain's public witness.
        """
        import httpx  # local import — httpx is already in pyproject.toml deps

        gateway_base = os.getenv("BIZRA_COGNITION_GATEWAY_URL", "http://localhost:7421")
        upstream_url = f"{gateway_base.rstrip('/')}/chain"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                upstream = await client.get(upstream_url)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            # Honest 503 — don't fabricate a healthy chain head.
            return JSONResponse(
                status_code=503,
                content={
                    "status": "gateway_unreachable",
                    "gateway_url": upstream_url,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                },
            )

        if upstream.status_code != 200:
            # Pass through upstream error code + body — don't reshape.
            return JSONResponse(
                status_code=upstream.status_code,
                content={
                    "status": "gateway_non_200",
                    "gateway_url": upstream_url,
                    "upstream_status": upstream.status_code,
                    "upstream_body": upstream.text[:500],
                },
            )

        return upstream.json()

    @app.get(
        "/v1/chain/latest",
        tags=["trust"],
        summary="Chain head + latest receipt detail in one call",
        description=(
            "Combines the cognition-gateway's GET /chain (head, length, "
            "latestTimestamp) with GET /chain/{head} (latest receipt kind, "
            "id, timestamp) into a single authoritative payload for the "
            "trust surface. Lets Dema show 'RECEIPT <kind> <timestamp>' "
            "next to 'CHAIN#<length> <head>' without requiring two "
            "frontend round-trips. Same no-shadow-state contract as "
            "/v1/chain: 503 on gateway unreachable, honest null receipt "
            "when chain is at genesis (length=0)."
        ),
    )
    async def get_chain_latest(request: Request):
        """Proxy: chain head + latest receipt detail, single payload.

        Node0 Closure Sprint — row 6 (trust_surface) enrichment — 2026-04-21.
        Public (no auth) matching /v1/chain precedent.
        """
        import httpx

        gateway_base = os.getenv(
            "BIZRA_COGNITION_GATEWAY_URL", "http://localhost:7421"
        ).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Step 1: fetch chain head + length
                chain_resp = await client.get(f"{gateway_base}/chain")
                if chain_resp.status_code != 200:
                    return JSONResponse(
                        status_code=chain_resp.status_code,
                        content={
                            "status": "gateway_non_200",
                            "gateway_url": f"{gateway_base}/chain",
                            "upstream_status": chain_resp.status_code,
                            "upstream_body": chain_resp.text[:500],
                        },
                    )
                chain_data = chain_resp.json()
                head = chain_data.get("head", "")
                length = chain_data.get("length", 0)

                result: dict[str, Any] = {
                    "head": head,
                    "length": length,
                    "latestTimestamp": chain_data.get("latestTimestamp"),
                    "sovereignEnvelopes": chain_data.get("sovereignEnvelopes"),
                    "sovereignEntries": chain_data.get("sovereignEntries"),
                    "latestReceipt": None,
                }

                # Step 2: fetch latest receipt detail (skip for empty chain)
                is_genesis = length == 0 or not head or head == ("0" * 64)
                if not is_genesis:
                    receipt_resp = await client.get(f"{gateway_base}/chain/{head}")
                    if receipt_resp.status_code == 200:
                        result["latestReceipt"] = receipt_resp.json()
                    else:
                        # Head exists but receipt detail unavailable —
                        # surface honestly, don't fabricate.
                        result["latestReceiptError"] = {
                            "upstream_status": receipt_resp.status_code,
                            "detail": receipt_resp.text[:200],
                        }

                return result
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "gateway_unreachable",
                    "gateway_url": f"{gateway_base}/chain",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                },
            )

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
    async def metrics(request: Request):
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        m = runtime.metrics
        return PlainTextResponse(m.to_prometheus(include_help=False))

    @app.post("/v1/metrics/vitals", tags=["health"], summary="Web Vitals beacon")
    async def metrics_vitals(payload: dict[str, Any]):
        normalized, error = SovereignAPIServer._accept_metrics_vitals(payload)
        if error is not None:
            return JSONResponse(status_code=400, content=error)
        return {"status": "accepted", "metric": normalized["name"]}

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
        except BizraError as exc:
            _log_bizra_error(exc)
            _record_boundary_error_via_node0(runtime, exc, route="/v1/query")
            return JSONResponse(
                status_code=http_status_for_error(exc),
                content=exc.to_receipt(),
            )
        except (RuntimeError, TimeoutError, ValueError) as exc:
            wrapped = _wrap_query_error(
                exc,
                route="/v1/query",
                query_length=len(body.query),
                user_id=user_id,
            )
            logger.warning("Query error (specific legacy): %s", exc)
            _record_boundary_error_via_node0(runtime, wrapped, route="/v1/query")
            return JSONResponse(
                status_code=http_status_for_error(wrapped),
                content=wrapped.to_receipt(),
            )
        except Exception as exc:  # noqa: BLE001 — API boundary
            wrapped = _wrap_query_error(
                exc,
                route="/v1/query",
                query_length=len(body.query),
                user_id=user_id,
            )
            logger.exception("Query execution failed")
            _record_boundary_error_via_node0(runtime, wrapped, route="/v1/query")
            return JSONResponse(
                status_code=http_status_for_error(wrapped),
                content=wrapped.to_receipt(),
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
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("Read error (specific): %s", exc)
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(exc) or "Operation failed"},
                    )
                except Exception:  # noqa: BLE001 — review needed
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
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    logger.warning("Read error (specific): %s", exc)
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(exc) or "Operation failed"},
                    )
                except Exception:  # noqa: BLE001 — review needed
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
        except (RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning("Query error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Verification error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Verification error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
                await _emit_bus_event(
                    "receipt.verified",
                    {
                        "receipt_id": receipt.receipt_id,
                        "status": receipt.status.value,
                        "signature_verified": True,
                    },
                    source="verification",
                )
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Verification error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Verification error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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

        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Verification error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
            logger.exception("Genesis header verification failed")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

    @app.get("/v1/gate-chain/stats")
    async def gate_chain_stats(request: Request):
        """Get GateChain evaluation statistics.

        Returns pass/fail rates, failure distribution by gate,
        and average SNR across evaluations.
        Auth required (topology leak fix).

        Standing on: Lamport (fail-closed), BIZRA Spearpoint (6-gate chain).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        stats = runtime.get_gate_chain_stats()
        if stats is None:
            return JSONResponse(
                status_code=404,
                content={"error": "GateChain is not initialized"},
            )
        return stats

    # ─── PoI (Proof-of-Impact) Endpoints ────────────────────────────

    @app.get("/v1/poi/stats")
    async def poi_stats(request: Request):
        """Get Proof-of-Impact engine statistics.
        Auth required (topology leak fix).

        Standing on: Nakamoto (PoW), Page & Brin (PageRank), Gini (inequality).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
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
    async def poi_contributor(contributor_id: str, request: Request):
        """Get the most recent PoI for a specific contributor. Auth required."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
        poi = runtime.get_contributor_poi(contributor_id)
        if poi is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"No PoI found for '{contributor_id}'"},
            )
        return poi

    # ─── SAT Controller Endpoints ───────────────────────────────

    @app.get("/v1/sat/stats")
    async def sat_stats(request: Request):
        """Get SAT Controller statistics.
        Auth required (topology leak fix).

        Returns Gini coefficient, rebalancing history, credit distribution.
        Standing on: Ostrom (commons governance), Gini (inequality).
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error
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
        _user_id, _user, auth_err = _authenticate_http_request(request)
        if auth_err:
            return auth_err
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return JSONResponse(
                status_code=503,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return JSONResponse(
                status_code=503,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return JSONResponse(
                status_code=503,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=503,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("Orchestration error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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

            body = await request.json()
            description = body.get("description", "").strip()
            if not description:
                return JSONResponse(
                    status_code=400,
                    content={"error": "description is required"},
                )

            source = body.get("source", "api")
            proof_mode = str(body.get("proof_mode", "auto") or "auto")
            permission_envelope = body.get("permission_envelope")
            canonical_mode_enabled = _runtime_canonical_mode_enabled(runtime)
            runtime_has_canonical_authority = _runtime_has_canonical_mission_authority(
                runtime
            )

            if canonical_mode_enabled and not runtime_has_canonical_authority:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": (
                            "Canonical mode requires runtime-owned organism mission "
                            "authority"
                        )
                    },
                )

            # ── System-1 Fast Path: Reflex Cache Lookup ──────────
            # Noncanonical runtimes may still use the API-local reflex compiler.
            # Canonical runtimes must route S1/S2 through runtime.mission().
            if not runtime_has_canonical_authority:
                try:
                    from core.integration.constants import REFLEX_PRECIPITATION_HITS
                    from core.sovereign.reflex_compiler import ReflexCompiler

                    global _reflex_compiler  # noqa: PLW0603
                    if _reflex_compiler is None:
                        with _reflex_compiler_lock:
                            if _reflex_compiler is None:  # double-check under lock
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
                        logger.info(
                            "System-1 cache hit for pattern %s (hits=%d, ihsan=%.3f)",
                            reflex_entry.pattern_hash[:12],
                            reflex_entry.hit_count,
                            reflex_entry.ihsan_composite,
                        )
                        from core.sovereign.terminal import (
                            ExecutionPath as _ExecutionPath,
                        )
                        from core.sovereign.terminal import (
                            TerminalState as _TerminalState,
                        )
                        from core.sovereign.terminal import (
                            TerminalStateController as _TerminalStateController,
                        )

                        mission_id = _secrets.token_hex(8)
                        receipt_id = _secrets.token_hex(8)
                        receipt_payload = {
                            "status": "COMPLETE",
                            "mission_id": mission_id,
                            "receipt_id": receipt_id,
                            "synthesis": reflex_entry.output_template,
                            "ihsan_score": reflex_entry.ihsan_composite,
                            "snr_score": reflex_entry.ihsan_composite,
                            "duration_ms": 0.1,
                            "execution_path": _ExecutionPath.SYSTEM_1_CACHE_HIT.value,
                            "channels_executed": [],
                            "action_count": 0,
                            "wallet_delta": {"seed": 0.0, "bloom": 0.0},
                            "reflex_delta": {
                                "compiled": True,
                                "near_compile": False,
                                "compile_count": reflex_entry.precipitation_count,
                                "threshold": REFLEX_PRECIPITATION_HITS,
                            },
                            "memory_delta": {
                                "episodic": 0,
                                "semantic": 0,
                                "procedural": 0,
                            },
                            "hash_chain_ref": reflex_entry.pattern_hash,
                            "reflex_pattern": reflex_entry.pattern_hash,
                            "reflex_latency_ms": 0.1,
                            "comparison_s2_avg_ms": 0.0,
                            "execution_authority": "",
                            "authority_path": "",
                            "fate_verdict": "",
                            "fate_reason_codes": [],
                            "fate_mode": "",
                            "identity_mode": "",
                            "signer_public_key_prefix": "",
                        }
                        session_id = request.headers.get("X-Session-ID", "default")
                        if session_id not in _terminal_controllers:
                            ctrl = _TerminalStateController()
                            ctrl.transition(_TerminalState.READY)
                            _terminal_controllers[session_id] = ctrl
                        ctrl = _terminal_controllers[session_id]
                        ctrl.start_mission(
                            mission_id,
                            execution_path=_ExecutionPath.SYSTEM_1_CACHE_HIT,
                        )
                        ctrl.transition(_TerminalState.PERMISSION_REVIEW)
                        ctrl.transition(_TerminalState.EXECUTING)
                        ctrl.complete()

                        await _emit_bus_event(
                            "mission.created",
                            {"mission_id": mission_id, "source": source},
                            source="mission",
                        )
                        await _emit_bus_event(
                            "receipt.generated",
                            receipt_payload,
                            source="mission",
                        )
                        await _emit_bus_event(
                            "mission.executed",
                            {
                                "mission_id": mission_id,
                                "receipt_id": receipt_id,
                                "status": "COMPLETE",
                                "ihsan_score": reflex_entry.ihsan_composite,
                                "snr_score": reflex_entry.ihsan_composite,
                                "duration_ms": 0.1,
                                "execution_path": _ExecutionPath.SYSTEM_1_CACHE_HIT.value,
                            },
                            source="mission",
                        )
                        return receipt_payload
                except ImportError:
                    logger.debug("ReflexCompiler not available, using System-2 only")
                except Exception:  # noqa: BLE001 — review needed
                    logger.debug(
                        "Reflex lookup failed, falling through to System-2",
                        exc_info=True,
                    )

            mission_result = None
            runtime_receipt = None
            mission_start_id = ""
            runtime_mission = getattr(runtime, "mission", None)
            if runtime_has_canonical_authority:
                from types import SimpleNamespace as _MissionNamespace

                mission_context: dict[str, Any] = {}
                if isinstance(permission_envelope, dict):
                    mission_context["permission_envelope"] = permission_envelope
                    if "time_budget_seconds" in permission_envelope:
                        mission_context["time_budget_seconds"] = float(
                            permission_envelope.get("time_budget_seconds", 900)
                        )
                runtime_receipt = await runtime_mission(
                    description,
                    source=source,
                    context=mission_context,
                    proof_mode=proof_mode,
                )
                mission_start_id = runtime_receipt.mission_id
                await _emit_bus_event(
                    "mission.created",
                    {"mission_id": runtime_receipt.mission_id, "source": source},
                    source="mission",
                )
                runtime_status = runtime.status() if hasattr(runtime, "status") else {}
                canonical_info = runtime_status.get("canonical", {})
                mission_result = _MissionNamespace(
                    mission_id=runtime_receipt.mission_id,
                    status=(
                        "BLOCKED"
                        if runtime_receipt.fate_verdict == "rejected"
                        else (
                            "FAILED"
                            if runtime_receipt.system == "ERROR"
                            else "COMPLETE"
                        )
                    ),
                    channels_executed=[],
                    synthesis=runtime_receipt.output_text,
                    evidence_receipt_id=(
                        runtime_receipt.evidence_hash or runtime_receipt.chain_hash
                    ),
                    ihsan_score=runtime_receipt.ihsan_score,
                    snr_score=runtime_receipt.snr_score,
                    duration_ms=runtime_receipt.duration_ms,
                    execution_path=(
                        "SYSTEM_1_CACHE_HIT"
                        if runtime_receipt.system == "S1"
                        else "SYSTEM_2_NOVEL"
                    ),
                    execution_authority=canonical_info.get(
                        "mission_authority", "organism"
                    ),
                    authority_path=canonical_info.get(
                        "authority_path", "runtime->organism->node0"
                    ),
                    fate_verdict=runtime_receipt.fate_verdict,
                    fate_reason_codes=list(runtime_receipt.fate_reason_codes),
                    fate_mode=runtime_receipt.fate_mode,
                    identity_mode=runtime_receipt.identity_mode,
                    signer_public_key_prefix=runtime_receipt.signer_public_key_prefix,
                )
            else:
                from core.sovereign.mission import (
                    DesktopContext,
                    MissionOrchestrator,
                    MissionRequest,
                )

                # Build mission config from Node0 ConfigMap environment
                config = {
                    "memory_path": os.environ.get(
                        "SEMANTIC_MEMORY_PATH", "/tmp/bizra-mission/memory"
                    ),
                    "evidence_path": os.environ.get(
                        "EVENT_LOG_PATH", "/tmp/bizra-mission"
                    )
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
                mission_start_id = mission_req.mission_id

                # ── Mission Lifecycle: Created ──────────────────────
                await _emit_bus_event(
                    "mission.created",
                    {"mission_id": mission_req.mission_id, "source": source},
                    source="mission",
                )

                mission_result = await orchestrator.execute(mission_req)

                # ── Canonical Ingest Authority (legacy compatibility path) ───────
                _ingest_via_node0(runtime, mission_result)

            # ── System-1 Precipitation: Record Observation ─────────
            # After every System-2 completion, record the pattern.
            # After K consecutive high-Ihsan observations, precipitate.
            try:
                if (
                    not runtime_has_canonical_authority
                    and "_reflex_compiler" in globals()
                    and _reflex_compiler is not None
                ):
                    _reflex_compiler.record_observation(
                        input_text=description,
                        output_text=mission_result.synthesis or "",
                        ihsan_composite=mission_result.ihsan_score or 0.0,
                    )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Read error (specific): %s", exc)
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc) or "Operation failed"},
                )
            except Exception:  # noqa: BLE001 — review needed
                logger.debug("Reflex observation recording failed", exc_info=True)

            normalized_status = _normalize_receipt_status(mission_result.status)
            normalized_execution_path = _normalize_execution_path(
                getattr(mission_result, "execution_path", "SYSTEM_2_NOVEL")
            )
            receipt_id = mission_result.evidence_receipt_id or _secrets.token_hex(8)
            reasoning_proof = await _build_reasoning_proof(
                description=description,
                source=source,
                proof_mode=proof_mode,
                permission_envelope=(
                    permission_envelope
                    if isinstance(permission_envelope, dict)
                    else None
                ),
                mission_id=mission_result.mission_id,
                mission_receipt_id=receipt_id,
                execution_path=normalized_execution_path,
                mission_result=mission_result,
            )
            reflex_delta_payload = {
                "compiled": False,
                "near_compile": False,
                "compile_count": 0,
                "threshold": 3,
            }
            compiled_reflex_event_payload: dict[str, Any] | None = None
            runtime_reflex_pattern = ""
            runtime_reflex_latency_ms = 0.0
            runtime_comparison_s2_avg_ms = 0.0
            if runtime_receipt is not None:
                (
                    reflex_delta_payload,
                    compiled_reflex_event_payload,
                    runtime_reflex_pattern,
                    runtime_reflex_latency_ms,
                    runtime_comparison_s2_avg_ms,
                ) = _runtime_reflex_lineage_payload(runtime_receipt)
            try:
                if (
                    runtime_receipt is None
                    and "_reflex_compiler" in globals()
                    and _reflex_compiler is not None
                ):
                    pattern_hash = _reflex_compiler._hash_input(description)
                    if pattern_hash in getattr(_reflex_compiler, "_cache", {}):
                        compiled_entry = _reflex_compiler._cache[pattern_hash]
                        reflex_delta_payload = {
                            "compiled": True,
                            "near_compile": False,
                            "compile_count": int(
                                getattr(compiled_entry, "precipitation_count", 0)
                            ),
                            "threshold": 3,
                        }
                        compiled_reflex_event_payload = {
                            "mission_id": mission_result.mission_id,
                            "name": str(
                                getattr(compiled_entry, "input_template", description)
                            )[:120],
                            "pattern_hash": str(
                                getattr(compiled_entry, "pattern_hash", pattern_hash)
                            ),
                            "avg_ihsan": round(
                                float(getattr(compiled_entry, "ihsan_composite", 0.0)),
                                4,
                            ),
                            "execution_count": int(
                                getattr(compiled_entry, "hit_count", 0)
                            ),
                            "precipitation_count": int(
                                getattr(compiled_entry, "precipitation_count", 0)
                            ),
                        }
                    else:
                        candidate = getattr(_reflex_compiler, "_candidates", {}).get(
                            pattern_hash
                        )
                        if candidate is not None:
                            consecutive = int(candidate.consecutive_high_quality())
                            reflex_delta_payload = {
                                "compiled": False,
                                "near_compile": consecutive > 0,
                                "compile_count": consecutive,
                                "threshold": 3,
                            }
            except Exception:  # noqa: BLE001 - reflex telemetry fallback
                logger.debug("Reflex delta synthesis failed", exc_info=True)

            # ── Terminal State Wiring (contract alignment) ─────────────────────
            try:
                from core.sovereign.terminal import ExecutionPath as _ExecutionPath
                from core.sovereign.terminal import TerminalState as _TS
                from core.sovereign.terminal import TerminalStateController as _TSC

                _session_id = request.headers.get("X-Session-ID", "default")
                if _session_id not in _terminal_controllers:
                    _ctrl = _TSC()
                    _ctrl.transition(_TS.READY)
                    _terminal_controllers[_session_id] = _ctrl
                _ctrl = _terminal_controllers[_session_id]
                _ctrl.start_mission(
                    mission_start_id or mission_result.mission_id,
                    execution_path=_ExecutionPath(normalized_execution_path),
                )
                _ctrl.transition(_TS.PERMISSION_REVIEW)
                _ctrl.transition(_TS.EXECUTING)
                if normalized_status == "FAILED":
                    _ctrl.fail()
                elif normalized_status == "BLOCKED":
                    _ctrl.block()
                else:
                    _ctrl.complete()
            except (ImportError, Exception):  # noqa: BLE001 — API boundary
                logger.debug("Terminal state wiring unavailable", exc_info=True)

            # Build enriched Terminal v1 receipt
            try:
                from core.sovereign.terminal import ChannelRecord as TChannelRecord
                from core.sovereign.terminal import ExecutionPath
                from core.sovereign.terminal import MemoryDelta as TMemoryDelta
                from core.sovereign.terminal import MissionReceipt as TerminalReceipt
                from core.sovereign.terminal import ReflexDelta as TReflexDelta
                from core.sovereign.terminal import WalletDelta

                t_channels = [
                    TChannelRecord(
                        channel=cr.channel,
                        success=cr.success,
                        duration_ms=cr.duration_ms,
                    )
                    for cr in mission_result.channels_executed
                ]
                terminal_receipt = TerminalReceipt(
                    mission_id=mission_result.mission_id,
                    receipt_id=receipt_id,
                    status=normalized_status,
                    synthesis=mission_result.synthesis,
                    ihsan_score=mission_result.ihsan_score,
                    snr_score=mission_result.snr_score,
                    duration_ms=mission_result.duration_ms,
                    channels_executed=t_channels,
                    execution_path=ExecutionPath(normalized_execution_path),
                    wallet_delta=WalletDelta(),
                    reflex_delta=TReflexDelta(**reflex_delta_payload),
                    memory_delta=TMemoryDelta(),
                    hash_chain_ref=receipt_id,
                    action_count=len(t_channels),
                )
                receipt_payload = terminal_receipt.to_dict()
            except ImportError:
                receipt_payload = {
                    "status": normalized_status,
                    "mission_id": mission_result.mission_id,
                    "receipt_id": receipt_id,
                    "synthesis": mission_result.synthesis,
                    "ihsan_score": mission_result.ihsan_score,
                    "snr_score": mission_result.snr_score,
                    "duration_ms": round(mission_result.duration_ms, 1),
                    "execution_path": normalized_execution_path,
                    "channels_executed": [
                        {
                            "channel": cr.channel,
                            "success": cr.success,
                            "duration_ms": round(cr.duration_ms, 1),
                        }
                        for cr in mission_result.channels_executed
                    ],
                    "action_count": len(mission_result.channels_executed),
                    "wallet_delta": {"seed": 0.0, "bloom": 0.0},
                    "reflex_delta": reflex_delta_payload,
                    "memory_delta": {"episodic": 0, "semantic": 0, "procedural": 0},
                    "hash_chain_ref": receipt_id,
                    "reflex_pattern": runtime_reflex_pattern,
                    "reflex_latency_ms": runtime_reflex_latency_ms,
                    "comparison_s2_avg_ms": runtime_comparison_s2_avg_ms,
                }
            if reasoning_proof is not None:
                receipt_payload["reasoning_proof"] = reasoning_proof
            receipt_payload["execution_authority"] = str(
                getattr(mission_result, "execution_authority", "")
            )
            receipt_payload["authority_path"] = str(
                getattr(mission_result, "authority_path", "")
            )
            receipt_payload["fate_verdict"] = str(
                getattr(mission_result, "fate_verdict", "")
            )
            receipt_payload["fate_reason_codes"] = list(
                getattr(mission_result, "fate_reason_codes", [])
            )
            receipt_payload["fate_mode"] = str(getattr(mission_result, "fate_mode", ""))
            receipt_payload["identity_mode"] = str(
                getattr(mission_result, "identity_mode", "")
            )
            receipt_payload["signer_public_key_prefix"] = str(
                getattr(mission_result, "signer_public_key_prefix", "")
            )
            if runtime_receipt is not None and getattr(
                runtime_receipt, "chain_hash", ""
            ):
                receipt_payload["hash_chain_ref"] = str(runtime_receipt.chain_hash)
            if runtime_receipt is not None:
                receipt_payload["reflex_pattern"] = runtime_reflex_pattern
                receipt_payload["reflex_latency_ms"] = runtime_reflex_latency_ms
                receipt_payload["comparison_s2_avg_ms"] = runtime_comparison_s2_avg_ms

            mission_topic = (
                "mission.failed"
                if normalized_status in {"FAILED", "BLOCKED"}
                else "mission.executed"
            )
            await _emit_bus_event(
                "receipt.generated",
                receipt_payload,
                source="mission",
            )
            if compiled_reflex_event_payload is not None:
                compiled_reflex_event_payload["receipt_id"] = receipt_id
                await _emit_bus_event(
                    "reflex.compiled",
                    compiled_reflex_event_payload,
                    source="mission",
                )
            if reasoning_proof is not None:
                await _emit_bus_event(
                    "mission.verified",
                    {
                        "mission_id": mission_result.mission_id,
                        "receipt_id": receipt_id,
                        "proof_receipt_id": reasoning_proof.get("receipt_id", ""),
                        "proof_status": reasoning_proof.get("status", ""),
                        "verified": reasoning_proof.get("verified", False),
                        "vrg_root": reasoning_proof.get("vrg_root", ""),
                        "branch_count": reasoning_proof.get("branch_count", 0),
                        "surviving_branches": reasoning_proof.get(
                            "surviving_branches", 0
                        ),
                    },
                    source="mission",
                )
            await _emit_bus_event(
                mission_topic,
                {
                    "mission_id": mission_result.mission_id,
                    "receipt_id": receipt_id,
                    "status": normalized_status,
                    "ihsan_score": mission_result.ihsan_score,
                    "snr_score": mission_result.snr_score,
                    "duration_ms": round(mission_result.duration_ms, 1),
                    "execution_path": normalized_execution_path,
                },
                source="mission",
            )
            return receipt_payload

        except ImportError as exc:
            logger.exception("Mission orchestrator not available")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Mission engine not available",
                    "detail": str(exc),
                },
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("Orchestration error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("Orchestration error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("Orchestration error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (RuntimeError, ValueError) as exc:
            logger.warning("Orchestration error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
            _ws_clients[ws] = set()

            # Send welcome. Normalize identity defensively so the stream
            # remains available even if a runtime adapter returns a mock or
            # non-dict status payload.
            try:
                status_snapshot = runtime.status()
            except Exception:  # noqa: BLE001 - websocket boundary hardening
                status_snapshot = {}
            identity = (
                status_snapshot.get("identity", {})
                if isinstance(status_snapshot, dict)
                else {}
            )
            await ws.send_json(
                {
                    "type": "connected",
                    "node_id": str(identity.get("node_id", "unknown")),
                    "version": str(identity.get("version", "1.0.0")),
                    "user_id": ws_user_id,
                    "protocol": ["subscribe", "unsubscribe", "history", "ping"],
                }
            )

            try:
                while True:
                    # Keep connection alive, handle client messages
                    data = await ws.receive_json()
                    msg_type = data.get("type", "")

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})

                    elif msg_type == "subscribe":
                        topics = data.get("topics", [])
                        if isinstance(topics, str):
                            topics = [topics]
                        if not isinstance(topics, list):
                            await ws.send_json(
                                {"type": "error", "error": "topics must be a list"}
                            )
                            continue
                        subscriptions = _ws_clients.get(ws, set())
                        subscriptions.update(
                            {
                                str(topic).strip()
                                for topic in topics
                                if str(topic).strip()
                            }
                        )
                        _ws_clients[ws] = subscriptions

                    elif msg_type == "unsubscribe":
                        topics = data.get("topics", [])
                        if isinstance(topics, str):
                            topics = [topics]
                        if not isinstance(topics, list):
                            await ws.send_json(
                                {"type": "error", "error": "topics must be a list"}
                            )
                            continue
                        subscriptions = _ws_clients.get(ws, set())
                        for topic in topics:
                            subscriptions.discard(str(topic).strip())
                        _ws_clients[ws] = subscriptions

                    elif msg_type == "history":
                        topics = data.get("topics", [])
                        if isinstance(topics, str):
                            topics = [topics]
                        if topics and not isinstance(topics, list):
                            await ws.send_json(
                                {"type": "error", "error": "topics must be a list"}
                            )
                            continue
                        requested_topics = {
                            str(topic).strip() for topic in topics if str(topic).strip()
                        }
                        if not requested_topics:
                            requested_topics = set(_ws_clients.get(ws, set()))
                        history = _history_events(
                            limit=int(data.get("limit", 100) or 100),
                            subscriptions=requested_topics,
                        )
                        await ws.send_json({"type": "history", "events": history})

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

                    else:
                        await ws.send_json(
                            {
                                "type": "error",
                                "error": f"Unsupported message type: {msg_type}",
                            }
                        )

            except (WebSocketDisconnect, Exception):  # noqa: BLE001 — WS boundary
                pass
            finally:
                _ws_clients.pop(ws, None)

    # Broadcast helper (used by background tasks to push to all clients)
    async def broadcast_to_clients(message: dict) -> int:
        """Push a message to all connected WebSocket clients."""
        sent = 0
        disconnected = set()
        for ws in list(_ws_clients):
            try:
                await ws.send_json(message)
                sent += 1
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Read error (specific): %s", exc)
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc) or "Operation failed"},
                )
            except Exception:  # noqa: BLE001 — review needed
                disconnected.add(ws)
        for ws in disconnected:
            _ws_clients.pop(ws, None)
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning("Query error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.warning("Token operation error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
            from core.memory.types import MemoryKind

            top_k = max(1, min(body.top_k, 100))
            min_score = max(0.0, min(body.min_score, 1.0))
            kinds = None
            if body.kinds:
                try:
                    kinds = [
                        MemoryKind(kind.lower()) for kind in dict.fromkeys(body.kinds)
                    ]
                except ValueError as exc:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Invalid memory kind: {exc}"},
                    )
            tags = list(dict.fromkeys(body.tags or []))[:32] or None
            context_ids = list(dict.fromkeys(body.context_ids or []))[:64] or None
            results = agent_db.search(
                query=body.query,
                top_k=top_k,
                min_score=min_score,
                source=body.source,
                kinds=kinds,
                tags=tags,
                context_ids=context_ids,
                include_archived=body.include_archived,
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
                        "graph_score": round(r.graph_score, 4),
                        "source": r.record.source,
                        "tags": r.record.tags,
                        "related_ids": r.record.related_ids,
                        **(
                            {"metadata": r.record.metadata} if body.debug_scores else {}
                        ),
                    }
                    for r in results
                ],
            }
        except (RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning("Query error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
            logger.exception("AgentDB search failed")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.post("/v1/memory/import")
    async def memory_import(body: MemoryImportModel, request: Request):
        """Import one explicit user-provided memory record into AgentDB.

        This intentionally does not scan local files or ingest arbitrary paths.
        Node0 Memory Import v0.1 accepts only a bounded payload supplied by the
        authenticated caller, with consent recorded in memory metadata.
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

        title = body.title.strip()
        content = body.content.strip()
        source_type = body.source_type.strip().lower().replace(" ", "_")
        owner_marker = body.owner_marker.strip()

        if not title:
            return JSONResponse(
                status_code=400,
                content={"error": "Title required"},
            )
        if not content:
            return JSONResponse(
                status_code=400,
                content={"error": "Content required"},
            )
        if not body.consent:
            return JSONResponse(
                status_code=400,
                content={"error": "Explicit consent required"},
            )
        if not owner_marker:
            return JSONResponse(
                status_code=400,
                content={"error": "Owner marker required"},
            )
        if len(title) > 200:
            return JSONResponse(
                status_code=400,
                content={"error": "Title exceeds 200 characters"},
            )
        if len(content) > 20000:
            return JSONResponse(
                status_code=400,
                content={"error": "Content exceeds 20000 characters"},
            )
        if source_type not in {
            "user_text",
            "preference",
            "project_context",
            "note",
            "mission_context",
        }:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid source_type"},
            )

        try:
            from core.memory.types import MemoryKind

            tags = [
                tag.strip().lower().replace(" ", "_")
                for tag in list(dict.fromkeys(body.tags))[:16]
                if tag.strip()
            ]
            if "node0_import" not in tags:
                tags.append("node0_import")

            truth_label = "[ENFORCEMENT: WIRED]"
            imported_at = _utcnow_iso()
            record = agent_db.store(
                content=f"{title}\n\n{content}",
                kind=MemoryKind.SEMANTIC,
                importance=0.7,
                source=f"node0_memory_import:{source_type}",
                source_id=str(uuid.uuid4()),
                tags=tags,
                metadata={
                    "title": title,
                    "source_type": source_type,
                    "owner_marker": owner_marker,
                    "consent": True,
                    "imported_at": imported_at,
                    "truth_label": truth_label,
                    "import_mode": "single_user_provided_record",
                },
            )
            return {
                "memory_id": record.id,
                "stored": True,
                "status": "stored",
                "truth_label": truth_label,
                "source_label": f"user_provided:{source_type}",
                "next_action": "search memory or submit mission",
            }
        except (
            ImportError,
            OSError,
            RuntimeError,
            sqlite3.Error,
            ValueError,
            TypeError,
        ) as exc:
            logger.warning("Memory import failed: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "memory_id": "",
                    "stored": False,
                    "status": "failed",
                    "error": str(exc) or "Memory import failed",
                    "truth_label": "[ENFORCEMENT: WIRED]",
                    "source_label": f"user_provided:{source_type}",
                    "next_action": "repair AgentDB memory layer",
                },
            )

    @app.post("/v1/node0/action-intent")
    async def node0_action_intent(body: Node0ActionIntentModel, request: Request):
        """Validate a bounded Node0 desktop/browser action intent.

        v0.1 prepares an explicit client-side handoff only. The server does not
        launch applications, mutate files, fetch URLs, or automate a browser.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        action_type = body.action_type.strip().lower().replace(" ", "_")
        target = body.target.strip()
        label = body.label.strip()

        if action_type not in {"open_url", "copy_text"}:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid action_type"},
            )
        if not target:
            return JSONResponse(
                status_code=400,
                content={"error": "Target required"},
            )
        if not body.consent:
            return JSONResponse(
                status_code=400,
                content={"error": "Explicit user confirmation required"},
            )
        if len(label) > 120:
            return JSONResponse(
                status_code=400,
                content={"error": "Label exceeds 120 characters"},
            )

        handoff_method = "clipboard_write"
        if action_type == "open_url":
            if len(target) > 2048:
                return JSONResponse(
                    status_code=400,
                    content={"error": "URL exceeds 2048 characters"},
                )
            parsed = urlparse(target)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Only http(s) URLs are allowed"},
                )
            handoff_method = "window_open"
        elif len(target) > 5000:
            return JSONResponse(
                status_code=400,
                content={"error": "Copy text exceeds 5000 characters"},
            )

        target_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
        target_preview = target if len(target) <= 200 else f"{target[:197]}..."
        truth_label = "[ENFORCEMENT: WIRED]"
        return {
            "action_id": str(uuid.uuid4()),
            "accepted": True,
            "status": "ready_for_user_handoff",
            "action_type": action_type,
            "label": label,
            "target": target,
            "target_preview": target_preview,
            "target_hash": target_hash,
            "execution_mode": "client_handoff_only",
            "handoff_method": handoff_method,
            "server_executed": False,
            "requires_user_confirmation": True,
            "truth_label": truth_label,
            "source_label": "user_confirmed_action_intent",
            "next_action": "confirm action in the local browser",
        }

    @app.post("/v1/node0/local-action/receipt")
    async def node0_local_action_receipt(
        body: Node0LocalActionReceiptModel,
        request: Request,
    ):
        """Record an explicit browser-client local action receipt.

        The action itself must already have happened in the local browser after
        a user gesture. This endpoint records the receipt and never executes an
        OS command, launches an application, or automates a browser.
        """
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        action_id = body.action_id.strip()
        action_type = body.action_type.strip().lower().replace(" ", "_")
        result = body.result.strip().lower().replace(" ", "_")
        execution_channel = body.execution_channel.strip().lower().replace(" ", "_")
        target_preview = body.target_preview.strip()
        target_hash = body.target_hash.strip().lower()
        error = body.error.strip()

        if not action_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Action id required"},
            )
        if len(action_id) > 80:
            return JSONResponse(
                status_code=400,
                content={"error": "Action id exceeds 80 characters"},
            )
        if action_type not in {"open_url", "copy_text"}:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid action_type"},
            )
        if result not in {"executed", "blocked", "failed"}:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid result"},
            )
        if execution_channel != "browser_client":
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid execution_channel"},
            )
        if not body.user_confirmed:
            return JSONResponse(
                status_code=400,
                content={"error": "User confirmation required"},
            )
        if len(target_preview) > 220:
            return JSONResponse(
                status_code=400,
                content={"error": "Target preview exceeds 220 characters"},
            )
        if len(target_hash) != 64 or any(
            character not in "0123456789abcdef" for character in target_hash
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "Valid target hash required"},
            )
        if len(error) > 500:
            return JSONResponse(
                status_code=400,
                content={"error": "Error detail exceeds 500 characters"},
            )

        recorded_at = _utcnow_iso()
        receipt_id = str(uuid.uuid4())
        _schedule_bus_event(
            "node0.local_action.receipted",
            {
                "receipt_id": receipt_id,
                "action_id": action_id,
                "action_type": action_type,
                "result": result,
                "execution_channel": execution_channel,
                "target_hash": target_hash,
                "server_executed": False,
                "recorded_at": recorded_at,
            },
            source="node0",
        )
        return {
            "receipt_id": receipt_id,
            "action_id": action_id,
            "recorded": True,
            "status": result,
            "action_type": action_type,
            "execution_channel": execution_channel,
            "server_executed": False,
            "target_preview": target_preview,
            "target_hash": target_hash,
            "recorded_at": recorded_at,
            "truth_label": "[ENFORCEMENT: WIRED]",
            "source_label": "browser_client_local_action",
            "next_action": "inspect receipt or submit next mission",
            **({"error": error} if error else {}),
        }

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

    @app.get("/v1/memory/profile")
    async def memory_profile(request: Request):
        """Return the terminal continuity profile for Memory view rendering."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        try:
            from core.sovereign.terminal import BriefingContext

            now = datetime.now(timezone.utc)
            seed_engine = getattr(runtime, "_seed_engine", None)
            episodes = []
            if seed_engine is not None:
                episodes = seed_engine.recent_episodes(limit=10)

            wallets = getattr(runtime, "_constitutional_wallets", [])
            wallet_snapshot = {"seed": 0.0, "bloom": 0.0}
            if wallets:
                from core.constitutional.fixed_point import fp_float

                wallet = wallets[0]
                wallet_snapshot = {
                    "seed": fp_float(getattr(wallet, "seed_balance", 0)),
                    "bloom": fp_float(getattr(wallet, "bloom_balance", 0)),
                }

            last_mission_summary = ""
            time_since_last_mission_s = 0.0
            if episodes:
                latest = episodes[-1]
                ts = latest.get("timestamp", "")
                if isinstance(ts, str) and ts:
                    try:
                        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        time_since_last_mission_s = max(
                            0.0, (now - parsed).total_seconds()
                        )
                    except ValueError:
                        time_since_last_mission_s = 0.0
                last_mission_summary = (
                    "Episode "
                    f"{latest.get('index', '?')} qualified={latest.get('qualified', False)} "
                    f"Ihsan {float(latest.get('ihsan', 0.0)):.2f}"
                )

            near_compile_patterns = []
            reflex_candidates = getattr(
                globals().get("_reflex_compiler"), "_candidates", {}
            )
            if isinstance(reflex_candidates, dict):
                for candidate in reflex_candidates.values():
                    observations = getattr(candidate, "observations", [])
                    if not observations:
                        continue
                    high_quality = 0
                    for observation in reversed(observations):
                        if float(observation.get("ihsan_composite", 0.0)) >= 0.95:
                            high_quality += 1
                        else:
                            break
                    if high_quality <= 0:
                        continue
                    avg_ihsan = sum(
                        float(observation.get("ihsan_composite", 0.0))
                        for observation in observations
                    ) / len(observations)
                    near_compile_patterns.append(
                        {
                            "name": str(
                                getattr(candidate, "input_template", "pattern")
                            )[:80],
                            "count": high_quality,
                            "threshold": 3,
                            "avg_ihsan": round(avg_ihsan, 4),
                        }
                    )

            near_compile_patterns.sort(
                key=lambda item: (item["count"], item["avg_ihsan"]),
                reverse=True,
            )
            near_compile_patterns = near_compile_patterns[:5]

            agent_db = getattr(runtime, "_agent_db", None)
            stats = {
                "episodic_count": 0,
                "semantic_count": 0,
                "procedural_count": 0,
                "total_entries": 0,
                "db_size_mb": 0.0,
            }
            if agent_db is not None:
                raw_stats = agent_db.stats()
                if isinstance(raw_stats, dict):
                    stats.update(raw_stats)

            work_streak = 0
            if seed_engine is not None and hasattr(seed_engine, "potential"):
                try:
                    work_streak = int(
                        getattr(seed_engine.potential(), "streak", 0) or 0
                    )
                except Exception:  # noqa: BLE001 - read model fallback
                    work_streak = 0

            compiled_reflex_summary = []
            reflex_cache = getattr(globals().get("_reflex_compiler"), "_cache", {})
            if isinstance(reflex_cache, dict):
                for entry in list(reflex_cache.values())[:10]:
                    created_at = float(getattr(entry, "created_at", 0.0) or 0.0)
                    last_hit_at = float(getattr(entry, "last_hit_at", 0.0) or 0.0)
                    compiled_reflex_summary.append(
                        {
                            "name": str(getattr(entry, "input_template", "pattern"))[
                                :80
                            ],
                            "avg_ihsan": round(
                                float(getattr(entry, "ihsan_composite", 0.0)), 4
                            ),
                            "execution_count": int(getattr(entry, "hit_count", 0)),
                            "avg_latency_ms": 0.0,
                            "compiled_at": (
                                datetime.fromtimestamp(
                                    created_at, tz=timezone.utc
                                ).isoformat()
                                if created_at > 0.0
                                else ""
                            ),
                            "last_hit_at": (
                                datetime.fromtimestamp(
                                    last_hit_at, tz=timezone.utc
                                ).isoformat()
                                if last_hit_at > 0.0
                                else ""
                            ),
                        }
                    )

            briefing = BriefingContext(
                time_since_last_mission_s=time_since_last_mission_s,
                active_project="bizra-data-lake",
                last_mission_summary=last_mission_summary,
                near_compile_patterns=[
                    item["name"] for item in near_compile_patterns[:3]
                ],
                quality_trend="stable" if episodes else "warming",
                next_action_suggestion=(
                    "Submit one more excellent mission to progress a reflex."
                    if near_compile_patterns
                    else "Submit a mission to start building continuity."
                ),
                wallet_snapshot=wallet_snapshot,
            )

            missions = [
                {
                    "mission_id": f"episode-{episode.get('index', index + 1)}",
                    "description": f"Growth episode {episode.get('index', index + 1)}",
                    "status": "COMPLETE" if episode.get("qualified") else "PARTIAL",
                    "ihsan_score": round(float(episode.get("ihsan", 0.0)), 4),
                    "seed_earned": round(float(episode.get("reward", 0.0)), 4),
                    "timestamp": episode.get("timestamp", ""),
                    "receipt_hash": episode.get("receipt_hash", ""),
                }
                for index, episode in enumerate(reversed(episodes))
            ]

            active_projects = []
            if missions:
                active_projects.append(
                    {
                        "name": "bizra-data-lake",
                        "last_activity": missions[0]["timestamp"],
                        "mission_count": len(missions),
                    }
                )

            return {
                "privacy_note": "All data is local",
                "briefing": briefing.to_dict(),
                "semantic_profile": {
                    "preferred_domains": [],
                    "active_hours": "",
                    "vocabulary_signature": "local-first constitutional terminal",
                    "work_window": "",
                },
                "missions": missions,
                "active_projects": active_projects,
                "work_streak": work_streak,
                "near_compile_patterns": near_compile_patterns,
                "compiled_reflex_summary": compiled_reflex_summary,
                "stats": stats,
            }
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
            logger.exception("Memory profile unavailable")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"},
            )

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
        except (RuntimeError, TimeoutError, ValueError) as exc:
            logger.warning("Query error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Read error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
            except Exception:  # noqa: BLE001 — API boundary
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
            except (ValueError, KeyError, PermissionError) as exc:
                logger.warning("Auth error (specific): %s", exc)
                return JSONResponse(
                    status_code=401,
                    content={"error": str(exc) or "Operation failed"},
                )
            except Exception:  # noqa: BLE001 — review needed
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Decode error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Decode error (specific): %s", exc)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc) or "Operation failed"},
            )
        except Exception:  # noqa: BLE001 — API boundary
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

    @app.put(
        "/v1/settings/model-routing",
        tags=["terminal"],
        summary="Persist model routing preferences",
    )
    async def update_model_routing(payload: dict[str, Any], request: Request):
        """Persist editable model routing preferences for Settings view."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        if not isinstance(payload, dict):
            return JSONResponse(
                status_code=400,
                content={"error": "JSON object required"},
            )

        routing = {
            str(key): str(value).strip()
            for key, value in payload.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }
        if not routing:
            return JSONResponse(
                status_code=400,
                content={"error": "At least one routing entry is required"},
            )

        persisted = _persist_model_routing(routing)
        return {"model_routing": persisted}

    @app.post(
        "/v1/terminal/critical-acknowledgments",
        tags=["terminal"],
        summary="Record a proof-bearing acknowledgment for a critical timeline event",
    )
    async def acknowledge_critical_event(
        body: CriticalAcknowledgmentRequest,
        request: Request,
    ):
        """Convert a critical-event acknowledgment into a receipted spine event."""
        _, _, auth_error = _authenticate_http_request(request)
        if auth_error is not None:
            return auth_error

        acknowledged_topic = _canonical_topic_name(body.topic.strip())
        if _event_severity_by_topic.get(acknowledged_topic) != "critical":
            return JSONResponse(
                status_code=400,
                content={"error": "Only critical events can be acknowledged"},
            )

        event_hash = body.event_hash.strip().lower()
        if len(event_hash) != 32 or any(
            ch not in "0123456789abcdef" for ch in event_hash
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "event_hash must be a 32-character lowercase hex string"
                },
            )

        acknowledgement_id = uuid.uuid4().hex[:12]
        receipt_id = uuid.uuid4().hex[:16]
        mission_id = body.mission_id.strip()
        session_id = request.headers.get("X-Session-ID", "default")
        synthesis = f"Critical event acknowledged | {acknowledged_topic} | operator session {session_id}"
        payload = {
            "acknowledgement_id": acknowledgement_id,
            "receipt_id": receipt_id,
            "status": "ACKNOWLEDGED",
            "acknowledged_event_hash": event_hash,
            "acknowledged_topic": acknowledged_topic,
            "acknowledged_summary": body.summary.strip(),
            "mission_id": mission_id,
            "acknowledged_receipt_id": body.receipt_id.strip(),
            "operator_session_id": session_id,
            "synthesis": synthesis,
        }
        envelope = await _emit_bus_event(
            "critical.acknowledged",
            payload,
            source="operator",
        )

        return CriticalAcknowledgmentResponse(
            acknowledgement_id=acknowledgement_id,
            receipt_id=receipt_id,
            status="ACKNOWLEDGED",
            hash_chain_ref=str((envelope or {}).get("event_hash", "")),
            acknowledged_event_hash=event_hash,
            acknowledged_topic=acknowledged_topic,
            mission_id=mission_id,
            timestamp=str((envelope or {}).get("timestamp", _utcnow_iso())),
            synthesis=synthesis,
        ).model_dump()

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
            seed_engine = getattr(runtime, "_seed_engine", None)
            episodes = (
                seed_engine.recent_episodes(limit=5) if seed_engine is not None else []
            )
            now = datetime.now(timezone.utc)

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

            last_mission_summary = ""
            time_since_last_mission_s = 0.0
            if episodes:
                latest = episodes[-1]
                timestamp = latest.get("timestamp", "")
                if isinstance(timestamp, str) and timestamp:
                    try:
                        parsed = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        time_since_last_mission_s = max(
                            0.0, (now - parsed).total_seconds()
                        )
                    except ValueError:
                        time_since_last_mission_s = 0.0
                last_mission_summary = (
                    "Last mission quality "
                    f"Ihsan {float(latest.get('ihsan', 0.0)):.2f} "
                    f"SNR {float(latest.get('snr', 0.0)):.2f}"
                )

            ctx = BriefingContext(
                time_since_last_mission_s=time_since_last_mission_s,
                active_project="bizra-data-lake",
                last_mission_summary=last_mission_summary,
                near_compile_patterns=near_compile,
                quality_trend="stable" if episodes else "warming",
                next_action_suggestion=(
                    "Review the permission envelope, then execute your next mission."
                ),
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
    *,
    enable_autopoiesis: bool = False,
    autopoiesis_cycle_seconds: float | None = None,
) -> None:
    """
    Run the Sovereign API server.

    Defaults to FastAPI+Uvicorn for full features (console, docs, CORS).
    Falls back to pure-asyncio SovereignAPIServer if uvicorn unavailable.

    Usage:
        python -m core.sovereign.api --port 8080
    """
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(
        autonomous_enabled=True,
        enable_autopoiesis=enable_autopoiesis,
    )
    if autopoiesis_cycle_seconds is not None:
        config.autopoiesis_cycle_seconds = autopoiesis_cycle_seconds
    resolved_api_keys = _resolved_api_keys(api_keys)
    _ensure_production_auth_prerequisites(
        use_fastapi=use_fastapi,
        api_keys=resolved_api_keys,
    )

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
            api_keys=resolved_api_keys or None,
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
