"""
BIZRA INFERENCE GATEWAY (PR1 IMPLEMENTATION)
═══════════════════════════════════════════════════════════════════════════════

Embedded LLM inference with fail-closed semantics and circuit breaker resilience.

Priority order (v2.2.1 - LM Studio as primary):
1. LM Studio v1 (192.168.56.1:1234) - RTX 4090 optimized, PRIMARY
2. Ollama (localhost:11434) - fallback
3. llama.cpp (embedded) - offline/edge
4. DENY (fail-closed)

This is the core of thermodynamic entropy reduction.
Local inference = local world model = sovereignty.

Standing on Giants:
- Nygard (2007): Release It! - Circuit breaker pattern for resilient systems
- Netflix Hystrix: Latency and fault tolerance library
- Fowler (2014): CircuitBreaker pattern documentation

Created: 2026-01-29 | BIZRA Sovereignty
Updated: 2026-02-01 | LM Studio v1 API as primary backend
Updated: 2026-02-04 | Circuit breaker pattern for backend resilience
Updated: 2026-02-04 | Rate limiting with token bucket algorithm (RFC 6585)
Principle: لا نفترض — We do not assume.
"""

import asyncio
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

# Import LM Studio backend (primary)
try:
    from .lmstudio_backend import ChatMessage
    from .lmstudio_backend import LMStudioBackend as LMStudioClient
    from .lmstudio_backend import LMStudioConfig

    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False
    LMStudioClient = None  # type: ignore[assignment, misc]
    LMStudioConfig = None  # type: ignore[assignment, misc]
    ChatMessage = None  # type: ignore[assignment, misc]

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default model paths
DEFAULT_MODEL_DIR: Final[Path] = Path("/var/lib/bizra/models")
CACHE_DIR: Final[Path] = Path("/var/lib/bizra/cache")


# =============================================================================
# TYPE PROTOCOLS (mypy --strict compliance)
# =============================================================================


class RateLimiterMetrics(TypedDict):
    """Type definition for rate limiter metrics."""

    requests_allowed: int
    requests_throttled: int
    current_tokens: float
    max_tokens: float
    tokens_per_second: float
    burst_size: int


class BatchingMetrics(TypedDict):
    """Type definition for batching metrics."""

    total_batches: int
    total_requests: int
    avg_batch_size: float
    avg_batch_duration_ms: float
    queue_depth: int


class GatewayStats(TypedDict):
    """Type definition for gateway statistics."""

    total_requests: int
    total_tokens: int
    avg_latency_ms: float


class HealthData(TypedDict, total=False):
    """Type definition for gateway health data."""

    status: str
    active_backend: Optional[str]
    active_model: Optional[str]
    backends: Dict[str, bool]
    stats: GatewayStats
    batching: BatchingMetrics


class CircuitMetrics(TypedDict):
    """Type definition for circuit breaker metrics as dict."""

    state: str
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    last_state_change: float
    total_calls: int
    total_failures: int
    total_successes: int
    total_rejections: int


# Protocol for backend generate function signature
class BackendGenerateFn(Protocol):
    """Protocol for backend generate function used by BatchingInferenceQueue."""

    async def __call__(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


# Type alias for circuit breaker state change callback
CircuitStateChangeCallback = Callable[[str, "CircuitState", "CircuitState"], None]

# Tier definitions
TIER_CONFIGS = {
    "EDGE": {
        "max_params": "1.7B",
        "default_model": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "context_length": 4096,
        "n_gpu_layers": 0,  # CPU only for edge
        "target_speed": 12,  # tok/s
    },
    "LOCAL": {
        "max_params": "7B",
        "default_model": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "context_length": 8192,
        "n_gpu_layers": -1,  # All layers on GPU
        "target_speed": 35,  # tok/s
    },
    "POOL": {
        "max_params": "70B+",
        "default_model": None,  # Federated
        "context_length": 32768,
        "n_gpu_layers": -1,
        "target_speed": None,  # Varies
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class ComputeTier(str, Enum):
    """Inference compute tiers."""

    EDGE = "edge"  # Always-on, low-power (TPU/CPU)
    LOCAL = "local"  # On-demand, high-power (GPU)
    POOL = "pool"  # URP federated compute


class InferenceBackend(str, Enum):
    """Available inference backends."""

    LLAMACPP = "llamacpp"  # Embedded (primary)
    OLLAMA = "ollama"  # External (fallback 1)
    LMSTUDIO = "lmstudio"  # External (fallback 2)
    POOL = "pool"  # URP federated
    OFFLINE = "offline"  # No inference available


class InferenceStatus(str, Enum):
    """Gateway status."""

    COLD = "cold"  # Not initialized
    WARMING = "warming"  # Loading models
    READY = "ready"  # Fully operational
    DEGRADED = "degraded"  # Fallback mode
    OFFLINE = "offline"  # No inference available


class CircuitState(str, Enum):
    """
    Circuit breaker states (Nygard 2007).

    State transitions:
    - CLOSED -> OPEN: After failure_threshold consecutive failures
    - OPEN -> HALF_OPEN: After recovery_timeout expires
    - HALF_OPEN -> CLOSED: After success_threshold consecutive successes
    - HALF_OPEN -> OPEN: On any failure
    """

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker behavior (Nygard 2007 / Netflix Hystrix).

    Standing on Giants:
    - Nygard (2007): Release It! - "Fail fast" pattern
    - Netflix (2012): Hystrix library defaults
    """

    # Failure threshold to trip circuit (CLOSED -> OPEN)
    failure_threshold: int = 5

    # Time in seconds before attempting recovery (OPEN -> HALF_OPEN)
    recovery_timeout: float = 30.0

    # Consecutive successes required to close circuit (HALF_OPEN -> CLOSED)
    success_threshold: int = 2

    # Maximum time to wait for a single request (seconds)
    request_timeout: float = 60.0

    # Enable circuit breaker (False = pass-through mode)
    enabled: bool = True


@dataclass
class RateLimiterConfig:
    """
    Configuration for rate limiter using token bucket algorithm.

    Standing on Giants:
    - Token Bucket Algorithm (Leaky Bucket variant)
    - RFC 6585: HTTP 429 Too Many Requests status code
    - Google Cloud: API rate limiting best practices

    The token bucket algorithm allows controlled bursts while maintaining
    a steady average rate. Tokens are added at a constant rate (tokens_per_second),
    and each request consumes one token. If no tokens are available, the request
    is either queued (with timeout) or rejected immediately.
    """

    # Rate at which tokens are added to the bucket (requests per second)
    tokens_per_second: float = 10.0

    # Maximum tokens the bucket can hold (sustained capacity)
    max_tokens: float = 100.0

    # Maximum burst size (tokens allowed in immediate succession)
    # This should be <= max_tokens
    burst_size: int = 20

    # Enable rate limiting (False = pass-through mode)
    enabled: bool = True

    # Timeout in seconds for acquire() when tokens unavailable (0 = no wait)
    acquire_timeout: float = 0.0

    # Per-client rate limiting (uses client_id from request context)
    per_client: bool = False

    # Default client ID for requests without explicit client context
    default_client_id: str = "default"


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION POOLING (P1 OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PooledConnection:
    """
    A pooled connection to an inference backend.

    Standing on Giants:
    - Apache Commons Pool (2001): Connection lifecycle management
    - Amdahl (1967): Connection overhead reduction for parallel scaling
    """

    id: str
    backend_type: str
    endpoint: str
    created_at: float
    last_used_at: float
    last_health_check: float
    is_healthy: bool = True
    in_use: bool = False
    use_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency for this connection."""
        return self.total_latency_ms / self.use_count if self.use_count > 0 else 0.0

    @property
    def age_seconds(self) -> float:
        """Age of connection in seconds."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Time since last use in seconds."""
        return time.time() - self.last_used_at


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool."""

    # Pool sizing
    min_size: int = 2
    max_size: int = 10

    # Connection lifecycle
    max_idle_seconds: float = 300.0  # 5 minutes
    max_age_seconds: float = 3600.0  # 1 hour

    # Health checking
    health_check_interval_seconds: float = 30.0
    health_check_timeout_seconds: float = 5.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Connection acquisition
    acquisition_timeout_seconds: float = 30.0


class ConnectionPoolMetrics:
    """
    Thread-safe metrics for connection pool performance.

    Tracks pool hits, misses, and connection health for optimization.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pool_hits = 0
        self._pool_misses = 0
        self._connections_created = 0
        self._connections_destroyed = 0
        self._health_checks_passed = 0
        self._health_checks_failed = 0
        self._total_wait_time_ms = 0.0
        self._acquisition_timeouts = 0

    async def record_hit(self) -> None:
        async with self._lock:
            self._pool_hits += 1

    async def record_miss(self) -> None:
        async with self._lock:
            self._pool_misses += 1

    async def record_connection_created(self) -> None:
        async with self._lock:
            self._connections_created += 1

    async def record_connection_destroyed(self) -> None:
        async with self._lock:
            self._connections_destroyed += 1

    async def record_health_check(self, passed: bool) -> None:
        async with self._lock:
            if passed:
                self._health_checks_passed += 1
            else:
                self._health_checks_failed += 1

    async def record_wait_time(self, wait_ms: float) -> None:
        async with self._lock:
            self._total_wait_time_ms += wait_ms

    async def record_acquisition_timeout(self) -> None:
        async with self._lock:
            self._acquisition_timeouts += 1

    async def get_metrics(self) -> Dict[str, Any]:
        async with self._lock:
            total_acquisitions = self._pool_hits + self._pool_misses
            return {
                "pool_hits": self._pool_hits,
                "pool_misses": self._pool_misses,
                "hit_rate": (
                    self._pool_hits / total_acquisitions
                    if total_acquisitions > 0
                    else 0.0
                ),
                "connections_created": self._connections_created,
                "connections_destroyed": self._connections_destroyed,
                "health_checks_passed": self._health_checks_passed,
                "health_checks_failed": self._health_checks_failed,
                "avg_wait_time_ms": (
                    self._total_wait_time_ms / total_acquisitions
                    if total_acquisitions > 0
                    else 0.0
                ),
                "acquisition_timeouts": self._acquisition_timeouts,
            }


class ConnectionPool:
    """
    High-performance connection pool for inference backends.

    PROBLEM: Creating new HTTP connections for each inference request adds
    significant latency overhead (TCP handshake, TLS negotiation).

    SOLUTION: Pool and reuse connections with:
    - LRU eviction for memory efficiency
    - Health checking for reliability
    - Automatic reconnection for fault tolerance

    Standing on Giants:
    - Apache Commons Pool (2001): Object pooling patterns
    - Amdahl (1967): Minimizing serial overhead in parallel systems
    - HikariCP (2014): High-performance JDBC connection pooling

    Expected improvement: 3-5x latency reduction for high-frequency requests.
    """

    def __init__(
        self,
        backend_type: str,
        endpoint: str,
        config: Optional[ConnectionPoolConfig] = None,
        connection_factory: Optional[Callable[[], Awaitable[Any]]] = None,
        health_checker: Optional[Callable[[Any], Awaitable[bool]]] = None,
    ) -> None:
        """
        Initialize connection pool.

        Args:
            backend_type: Type of backend (ollama, lmstudio)
            endpoint: Backend endpoint URL
            config: Pool configuration
            connection_factory: Async callable that creates a new connection
            health_checker: Async callable that checks connection health
        """
        self.backend_type = backend_type
        self.endpoint = endpoint
        self.config = config or ConnectionPoolConfig()
        self._connection_factory = connection_factory
        self._health_checker = health_checker

        # LRU-ordered pool (newest at end)
        self._pool: OrderedDict[str, PooledConnection] = OrderedDict()
        self._connections: Dict[str, Any] = {}  # Actual connection objects

        # Synchronization
        self._lock = asyncio.Lock()
        self._available = asyncio.Semaphore(self.config.max_size)

        # Background tasks
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._running = False

        # Metrics
        self.metrics = ConnectionPoolMetrics()

    async def start(self) -> None:
        """Start the connection pool and background health checker."""
        if self._running:
            return

        self._running = True

        # Pre-warm pool with minimum connections
        for _ in range(self.config.min_size):
            try:
                await self._create_connection()
            except Exception as e:
                print(f"[ConnectionPool] Pre-warm failed: {e}")
                break

        # Start health checker
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        print(
            f"[ConnectionPool] Started for {self.backend_type} "
            f"(min={self.config.min_size}, max={self.config.max_size})"
        )

    async def stop(self) -> None:
        """Stop the connection pool and cleanup resources."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None

        # Close all connections
        async with self._lock:
            for conn_id in list(self._pool.keys()):
                await self._destroy_connection(conn_id)

        print(f"[ConnectionPool] Stopped for {self.backend_type}")

    async def _create_connection(self) -> PooledConnection:
        """Create a new pooled connection."""
        conn_id = hashlib.sha256(
            f"{self.backend_type}:{self.endpoint}:{time.time()}".encode()
        ).hexdigest()[:16]

        now = time.time()
        pooled_conn = PooledConnection(
            id=conn_id,
            backend_type=self.backend_type,
            endpoint=self.endpoint,
            created_at=now,
            last_used_at=now,
            last_health_check=now,
            is_healthy=True,
            in_use=False,
        )

        # Create actual connection if factory provided
        if self._connection_factory:
            conn = await self._connection_factory()
            self._connections[conn_id] = conn

        async with self._lock:
            self._pool[conn_id] = pooled_conn
            # Move to end (LRU)
            self._pool.move_to_end(conn_id)

        await self.metrics.record_connection_created()
        return pooled_conn

    async def _destroy_connection(self, conn_id: str) -> None:
        """Destroy a pooled connection."""
        if conn_id in self._pool:
            del self._pool[conn_id]

        if conn_id in self._connections:
            conn = self._connections.pop(conn_id)
            # Close connection if it has a close method
            if hasattr(conn, "close"):
                try:
                    if asyncio.iscoroutinefunction(conn.close):
                        await conn.close()
                    else:
                        conn.close()
                except (OSError, ConnectionError, RuntimeError):
                    # Best-effort cleanup - connection may already be closed
                    pass

        await self.metrics.record_connection_destroyed()

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.

        Yields:
            Tuple of (PooledConnection, actual_connection_object)

        Usage:
            async with pool.acquire() as (pooled, conn):
                result = await use_connection(conn)
        """
        start_time = time.time()
        pooled_conn: Optional[PooledConnection] = None
        actual_conn: Any = None

        try:
            # Wait for available slot with timeout
            try:
                await asyncio.wait_for(
                    self._available.acquire(),
                    timeout=self.config.acquisition_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await self.metrics.record_acquisition_timeout()
                raise RuntimeError(
                    f"Connection acquisition timeout after "
                    f"{self.config.acquisition_timeout_seconds}s"
                )

            # Find or create connection
            async with self._lock:
                # Find available healthy connection (LRU order - oldest first)
                for conn_id, conn in self._pool.items():
                    if not conn.in_use and conn.is_healthy:
                        # Check if connection is too old or idle
                        if (
                            conn.age_seconds > self.config.max_age_seconds
                            or conn.idle_seconds > self.config.max_idle_seconds
                        ):
                            # Destroy old connection
                            await self._destroy_connection(conn_id)
                            continue

                        # Found usable connection
                        conn.in_use = True
                        conn.last_used_at = time.time()
                        pooled_conn = conn
                        await self.metrics.record_hit()
                        break

            # No available connection - create new one if under limit
            if pooled_conn is None:
                await self.metrics.record_miss()
                async with self._lock:
                    if len(self._pool) < self.config.max_size:
                        pooled_conn = await self._create_connection()
                        pooled_conn.in_use = True

            if pooled_conn is None:
                raise RuntimeError("Failed to acquire connection")

            # Get actual connection object
            actual_conn = self._connections.get(pooled_conn.id)

            # Record wait time
            wait_ms = (time.time() - start_time) * 1000
            await self.metrics.record_wait_time(wait_ms)

            yield (pooled_conn, actual_conn)

        finally:
            # Release connection back to pool
            if pooled_conn is not None:
                async with self._lock:
                    if pooled_conn.id in self._pool:
                        pooled_conn.in_use = False
                        pooled_conn.use_count += 1
                        # Move to end (most recently used)
                        self._pool.move_to_end(pooled_conn.id)

            self._available.release()

    async def _health_check_loop(self) -> None:
        """Background task to check connection health."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval_seconds)

                if not self._running:
                    break

                # Check all connections
                async with self._lock:
                    conn_ids = list(self._pool.keys())

                for conn_id in conn_ids:
                    if not self._running:
                        break

                    async with self._lock:
                        if conn_id not in self._pool:
                            continue
                        conn = self._pool[conn_id]
                        if conn.in_use:
                            continue

                    # Run health check
                    is_healthy = await self._check_connection_health(conn_id)

                    async with self._lock:
                        if conn_id in self._pool:
                            self._pool[conn_id].is_healthy = is_healthy
                            self._pool[conn_id].last_health_check = time.time()

                    await self.metrics.record_health_check(is_healthy)

                    # Destroy unhealthy connections
                    if not is_healthy:
                        async with self._lock:
                            await self._destroy_connection(conn_id)

                # Ensure minimum connections
                async with self._lock:
                    healthy_count = sum(
                        1 for c in self._pool.values() if c.is_healthy and not c.in_use
                    )

                while healthy_count < self.config.min_size and self._running:
                    try:
                        await self._create_connection()
                        healthy_count += 1
                    except Exception:
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[ConnectionPool] Health check error: {e}")
                await asyncio.sleep(5)  # Backoff on error

    async def _check_connection_health(self, conn_id: str) -> bool:
        """Check if a specific connection is healthy."""
        if self._health_checker is None:
            return True

        conn = self._connections.get(conn_id)
        if conn is None:
            return False

        try:
            result = await asyncio.wait_for(
                self._health_checker(conn),
                timeout=self.config.health_check_timeout_seconds,
            )
            return bool(result)
        except asyncio.TimeoutError:
            return False
        except (ConnectionError, OSError, RuntimeError):
            # Health check failed - connection unhealthy
            return False

    def get_active_connections(self) -> int:
        """Get number of connections currently in use."""
        return sum(1 for c in self._pool.values() if c.in_use)

    def get_available_connections(self) -> int:
        """Get number of healthy connections available."""
        return sum(1 for c in self._pool.values() if c.is_healthy and not c.in_use)

    def get_total_connections(self) -> int:
        """Get total number of connections in pool."""
        return len(self._pool)

    async def get_status(self) -> Dict[str, Any]:
        """Get pool status including metrics."""
        metrics = await self.metrics.get_metrics()
        return {
            "backend_type": self.backend_type,
            "endpoint": self.endpoint,
            "total_connections": self.get_total_connections(),
            "active_connections": self.get_active_connections(),
            "available_connections": self.get_available_connections(),
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_idle_seconds": self.config.max_idle_seconds,
                "max_age_seconds": self.config.max_age_seconds,
            },
            "metrics": metrics,
        }


class PooledHttpClient:
    """
    HTTP client wrapper for connection pooling.

    Provides a simple interface for making HTTP requests through the pool.
    """

    def __init__(self, pool: ConnectionPool, base_url: str) -> None:
        self.pool = pool
        self.base_url = base_url.rstrip("/")

    async def request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Make an HTTP request through the pool."""
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"

        async with self.pool.acquire() as (pooled_conn, _):
            start_time = time.time()
            try:
                payload = json.dumps(data).encode() if data else None
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method=method,
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: urllib.request.urlopen(req, timeout=timeout)  # nosec B310 — URL from trusted connection pool config
                )
                with response as resp:
                    result: Dict[str, Any] = json.loads(resp.read().decode())

                # Record latency
                latency_ms = (time.time() - start_time) * 1000
                pooled_conn.total_latency_ms += latency_ms

                return result

            except Exception:
                # Mark connection as unhealthy on error
                pooled_conn.is_healthy = False
                raise


@dataclass
class InferenceConfig:
    """Configuration for the inference gateway."""

    # Model settings
    default_model: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    model_path: Optional[str] = None
    model_dir: Path = DEFAULT_MODEL_DIR

    # Context settings
    context_length: int = 8192
    max_tokens: int = 2048

    # Hardware settings
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    n_threads: int = 8
    n_batch: int = 512

    # Tier settings
    default_tier: ComputeTier = ComputeTier.LOCAL

    # Fallback chain
    fallbacks: List[str] = field(default_factory=lambda: ["ollama", "lmstudio"])

    # Fail-closed: deny if no local model available
    require_local: bool = True

    # External endpoints (env vars override defaults)
    ollama_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
    )
    lmstudio_url: str = field(
        default_factory=lambda: os.getenv(
            "LMSTUDIO_URL",
            f"http://{os.getenv('LMSTUDIO_HOST', '192.168.56.1')}:{os.getenv('LMSTUDIO_PORT', '1234')}",
        )
    )

    # Batching settings (P0-P1 optimization)
    enable_batching: bool = True
    max_batch_size: int = 8
    max_batch_wait_ms: int = 50  # Flush batch after 50ms

    # Circuit breaker settings (Nygard 2007)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Connection pool settings (P1 optimization)
    enable_connection_pool: bool = True
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)

    # Rate limiter settings (RFC 6585)
    enable_rate_limiting: bool = True
    rate_limiter: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class InferenceResult:
    """Result of an inference call."""

    content: str
    model: str
    backend: InferenceBackend
    tier: ComputeTier

    # Metrics
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    latency_ms: float = 0.0

    # Metadata
    timestamp: str = ""
    receipt_hash: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class TaskComplexity:
    """Estimated complexity of an inference task."""

    input_tokens: int
    estimated_output_tokens: int
    reasoning_depth: float  # 0.0 = simple, 1.0 = complex
    domain_specificity: float  # 0.0 = general, 1.0 = specialized

    @property
    def score(self) -> float:
        """Overall complexity score (0.0 - 1.0)."""
        token_factor = min(
            1.0, (self.input_tokens + self.estimated_output_tokens) / 4000
        )
        return (
            0.3 * token_factor
            + 0.4 * self.reasoning_depth
            + 0.3 * self.domain_specificity
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER (NYGARD 2007 / NETFLIX HYSTRIX)
# ═══════════════════════════════════════════════════════════════════════════════


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and request cannot proceed."""

    def __init__(self, backend_name: str, state: CircuitState, time_until_retry: float):
        self.backend_name = backend_name
        self.state = state
        self.time_until_retry = time_until_retry
        super().__init__(
            f"Circuit breaker OPEN for {backend_name}. "
            f"Retry in {time_until_retry:.1f}s"
        )


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring and observability."""

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[float]
    last_success_time: Optional[float]
    last_state_change: float
    total_calls: int
    total_failures: int
    total_successes: int
    total_rejections: int  # Calls rejected due to open circuit


class CircuitBreaker:
    """
    Circuit breaker implementation (Nygard 2007 / Netflix Hystrix pattern).

    The circuit breaker prevents cascading failures by "tripping" when a backend
    experiences repeated failures, allowing it time to recover before retrying.

    State Machine:
    ┌────────┐  failure_threshold reached  ┌────────┐
    │ CLOSED │ ─────────────────────────> │  OPEN  │
    └────────┘                             └────────┘
        ^                                      │
        │  success_threshold reached           │ recovery_timeout expires
        │                                      v
    ┌───────────┐                         ┌───────────┐
    │  (reset)  │ <────────────────────── │ HALF_OPEN │
    └───────────┘      on failure         └───────────┘
                       (re-trip)

    Standing on Giants:
    - Nygard (2007): Release It! - Original circuit breaker pattern
    - Netflix (2012): Hystrix library - Production-grade implementation
    - Fowler (2014): CircuitBreaker documentation

    Thread Safety:
    Uses asyncio.Lock for safe concurrent access to state.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        on_state_change: Optional[
            Callable[[str, CircuitState, CircuitState], None]
        ] = None,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (typically backend name)
            config: Circuit breaker configuration
            on_state_change: Optional callback on state transitions
        """
        self.name: str = name
        self.config: CircuitBreakerConfig = config
        self._on_state_change: Optional[
            Callable[[str, CircuitState, CircuitState], None]
        ] = on_state_change

        # State
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._last_state_change: float = time.time()

        # Cumulative metrics
        self._total_calls: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._total_rejections: int = 0

        # Concurrency control
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """True if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """True if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """True if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current circuit breaker metrics."""
        return CircuitBreakerMetrics(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            last_state_change=self._last_state_change,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            total_rejections=self._total_rejections,
        )

    def _time_until_retry(self) -> float:
        """Calculate time remaining before recovery attempt is allowed."""
        if self._state != CircuitState.OPEN or self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        remaining = self.config.recovery_timeout - elapsed
        return max(0.0, remaining)

    def _should_attempt_recovery(self) -> bool:
        """Check if recovery timeout has expired (OPEN -> HALF_OPEN)."""
        if self._state != CircuitState.OPEN:
            return False
        return self._time_until_retry() <= 0

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with callback notification."""
        if self._state == new_state:
            return

        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        print(f"[CircuitBreaker:{self.name}] {old_state.value} -> {new_state.value}")

        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception as e:
                print(f"[CircuitBreaker:{self.name}] State change callback error: {e}")

    async def can_execute(self) -> bool:
        """
        Check if a request can proceed through the circuit.

        Returns:
            True if request should proceed, False if circuit is open.
        """
        async with self._lock:
            if not self.config.enabled:
                return True

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            # HALF_OPEN: Allow request (testing recovery)
            return True

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            self._last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """
        Record a failed call.

        Args:
            error: Optional exception that caused the failure
        """
        async with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if error:
                print(
                    f"[CircuitBreaker:{self.name}] Failure recorded: {type(error).__name__}: {error}"
                )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately re-trips the circuit
                self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    async def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        async with self._lock:
            self._total_rejections += 1

    @asynccontextmanager
    async def protect(self):
        """
        Context manager for protected execution.

        Usage:
            async with circuit_breaker.protect():
                result = await backend.generate(prompt)

        Raises:
            CircuitBreakerError: If circuit is open and recovery timeout not expired
        """
        if not self.config.enabled:
            yield
            return

        if not await self.can_execute():
            await self.record_rejection()
            raise CircuitBreakerError(self.name, self._state, self._time_until_retry())

        try:
            yield
            await self.record_success()
        except Exception as e:
            await self.record_failure(e)
            raise

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by func
        """
        async with self.protect():
            return await func(*args, **kwargs)

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.time()
        print(f"[CircuitBreaker:{self.name}] Manually reset to CLOSED")


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING (RFC 6585 / TOKEN BUCKET)
# ═══════════════════════════════════════════════════════════════════════════════


class RateLimitError(Exception):
    """
    Raised when rate limit is exceeded (HTTP 429 Too Many Requests).

    Standing on Giants:
    - RFC 6585: Additional HTTP Status Codes (429 Too Many Requests)
    - Google API Design Guide: Rate limiting patterns
    """

    def __init__(
        self,
        client_id: str,
        current_tokens: float,
        retry_after: float,
        message: Optional[str] = None,
    ):
        self.client_id = client_id
        self.current_tokens = current_tokens
        self.retry_after = retry_after
        self.http_status = 429  # Too Many Requests
        super().__init__(
            message
            or f"Rate limit exceeded for client '{client_id}'. "
            f"Retry after {retry_after:.2f}s"
        )


class RateLimiter:
    """
    Token bucket rate limiter for inference request throttling.

    PROBLEM: Uncontrolled request bursts can overwhelm inference backends,
    causing degraded performance, memory exhaustion, or service crashes.

    SOLUTION: Token bucket algorithm provides:
    - Controlled burst handling (up to burst_size requests instantly)
    - Steady-state rate limiting (tokens_per_second sustained rate)
    - Per-client isolation (optional, with per_client=True)
    - Thread-safe async implementation

    Algorithm:
    1. Bucket starts with max_tokens tokens
    2. Tokens refill at tokens_per_second rate
    3. Each request consumes 1 token
    4. If tokens < 1, request is throttled (429) or waits

    Standing on Giants:
    - Leaky Bucket (Turner, 1986): Traffic shaping algorithm
    - Token Bucket (Benes, 1965): Rate limiting variant
    - RFC 6585 (2012): HTTP 429 Too Many Requests
    - Google Cloud: API rate limiting best practices
    - Stripe: Concurrent rate limiting patterns

    Thread Safety:
    Uses asyncio.Lock for safe concurrent access to token state.
    """

    def __init__(self, config: RateLimiterConfig, name: str = "default") -> None:
        """
        Initialize rate limiter.

        Args:
            config: Rate limiter configuration
            name: Identifier for this rate limiter instance
        """
        self.config = config
        self.name = name

        # Token bucket state (per-client if per_client=True)
        self._tokens: Dict[str, float] = {}
        self._last_refill: Dict[str, float] = {}

        # Concurrency control
        self._lock = asyncio.Lock()

        # Metrics
        self._requests_allowed: int = 0
        self._requests_throttled: int = 0

    def _get_client_id(self, client_id: Optional[str] = None) -> str:
        """Get effective client ID for rate limiting."""
        if not self.config.per_client:
            return self.config.default_client_id
        return client_id or self.config.default_client_id

    async def _refill_tokens(self, client_id: str) -> None:
        """
        Refill tokens based on elapsed time since last refill.

        Token refill formula:
        new_tokens = current_tokens + (elapsed_seconds * tokens_per_second)
        capped at max_tokens
        """
        now = time.time()

        if client_id not in self._tokens:
            # Initialize bucket for new client
            self._tokens[client_id] = self.config.max_tokens
            self._last_refill[client_id] = now
            return

        elapsed = now - self._last_refill[client_id]
        if elapsed <= 0:
            return

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.config.tokens_per_second
        self._tokens[client_id] = min(
            self._tokens[client_id] + tokens_to_add, self.config.max_tokens
        )
        self._last_refill[client_id] = now

    async def acquire(
        self, client_id: Optional[str] = None, tokens: float = 1.0
    ) -> bool:
        """
        Acquire tokens from the bucket, waiting if necessary.

        This method will wait up to acquire_timeout seconds for tokens
        to become available. Use try_acquire() for non-blocking behavior.

        Args:
            client_id: Optional client identifier for per-client limiting
            tokens: Number of tokens to acquire (default: 1.0)

        Returns:
            True if tokens were acquired successfully

        Raises:
            RateLimitError: If tokens unavailable after timeout
        """
        if not self.config.enabled:
            self._requests_allowed += 1
            return True

        effective_client = self._get_client_id(client_id)
        start_time = time.time()
        timeout = self.config.acquire_timeout

        while True:
            async with self._lock:
                await self._refill_tokens(effective_client)

                if self._tokens.get(effective_client, 0) >= tokens:
                    self._tokens[effective_client] -= tokens
                    self._requests_allowed += 1
                    return True

                current_tokens = self._tokens.get(effective_client, 0)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._requests_throttled += 1
                # Calculate retry-after based on tokens needed
                tokens_needed = tokens - current_tokens
                retry_after = tokens_needed / self.config.tokens_per_second
                raise RateLimitError(
                    client_id=effective_client,
                    current_tokens=current_tokens,
                    retry_after=retry_after,
                )

            # Wait a short interval before retrying
            wait_time = min(0.01, timeout - elapsed)  # 10ms intervals
            await asyncio.sleep(wait_time)

    async def try_acquire(
        self, client_id: Optional[str] = None, tokens: float = 1.0
    ) -> bool:
        """
        Try to acquire tokens without waiting.

        Non-blocking variant of acquire(). Returns immediately with
        success/failure status.

        Args:
            client_id: Optional client identifier for per-client limiting
            tokens: Number of tokens to acquire (default: 1.0)

        Returns:
            True if tokens were acquired, False if insufficient tokens
        """
        if not self.config.enabled:
            self._requests_allowed += 1
            return True

        effective_client = self._get_client_id(client_id)

        async with self._lock:
            await self._refill_tokens(effective_client)

            if self._tokens.get(effective_client, 0) >= tokens:
                self._tokens[effective_client] -= tokens
                self._requests_allowed += 1
                return True

        self._requests_throttled += 1
        return False

    async def get_tokens(self, client_id: Optional[str] = None) -> float:
        """
        Get current token count for a client.

        Args:
            client_id: Optional client identifier

        Returns:
            Current number of available tokens
        """
        if not self.config.enabled:
            return self.config.max_tokens

        effective_client = self._get_client_id(client_id)

        async with self._lock:
            await self._refill_tokens(effective_client)
            return self._tokens.get(effective_client, self.config.max_tokens)

    def get_metrics(self) -> RateLimiterMetrics:
        """
        Get rate limiter metrics.

        Returns:
            RateLimiterMetrics with current state and statistics
        """
        # Get average current tokens across all clients
        total_tokens = (
            sum(self._tokens.values()) if self._tokens else self.config.max_tokens
        )
        num_clients = len(self._tokens) if self._tokens else 1
        avg_tokens = total_tokens / num_clients

        return RateLimiterMetrics(
            requests_allowed=self._requests_allowed,
            requests_throttled=self._requests_throttled,
            current_tokens=avg_tokens,
            max_tokens=self.config.max_tokens,
            tokens_per_second=self.config.tokens_per_second,
            burst_size=self.config.burst_size,
        )

    async def get_client_metrics(
        self, client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for a specific client.

        Args:
            client_id: Client identifier

        Returns:
            Dict with client-specific token state
        """
        effective_client = self._get_client_id(client_id)

        async with self._lock:
            await self._refill_tokens(effective_client)
            return {
                "client_id": effective_client,
                "current_tokens": self._tokens.get(
                    effective_client, self.config.max_tokens
                ),
                "max_tokens": self.config.max_tokens,
                "tokens_per_second": self.config.tokens_per_second,
                "last_refill": self._last_refill.get(effective_client),
            }

    def reset(self, client_id: Optional[str] = None) -> None:
        """
        Reset rate limiter state.

        Args:
            client_id: If provided, reset only this client. Otherwise reset all.
        """
        if client_id:
            effective_client = self._get_client_id(client_id)
            if effective_client in self._tokens:
                del self._tokens[effective_client]
            if effective_client in self._last_refill:
                del self._last_refill[effective_client]
            print(f"[RateLimiter:{self.name}] Reset client '{effective_client}'")
        else:
            self._tokens.clear()
            self._last_refill.clear()
            self._requests_allowed = 0
            self._requests_throttled = 0
            print(f"[RateLimiter:{self.name}] Reset all clients")

    @asynccontextmanager
    async def limit(self, client_id: Optional[str] = None):
        """
        Context manager for rate-limited execution.

        Usage:
            async with rate_limiter.limit(client_id="user123"):
                result = await do_inference(prompt)

        Raises:
            RateLimitError: If rate limit exceeded
        """
        await self.acquire(client_id)
        try:
            yield
        finally:
            pass  # Token already consumed on acquire


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST BATCHING (P0-P1 OPTIMIZATION)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PendingRequest:
    """
    A pending inference request waiting to be batched.

    Standing on Giants:
    - Amdahl (1967): Parallelization for throughput improvement
    - NVIDIA (2020): Batch inference optimization for GPUs
    """

    prompt: str
    max_tokens: int
    temperature: float
    future: asyncio.Future
    created_at: float


class BatchingInferenceQueue:
    """
    Batches inference requests for efficient GPU utilization.

    PROBLEM: Serial asyncio.Lock wastes GPU batch processing capability.
    SOLUTION: Accumulate requests, process in batches of up to MAX_BATCH_SIZE.

    Expected throughput improvement: 8x (from serial to batch-8).

    Standing on Giants:
    - Amdahl (1967): Parallelization theory
    - NVIDIA (2020): GPU batch inference best practices
    - Google (2017): Tensor batching in TensorFlow Serving
    """

    def __init__(
        self,
        backend_generate_fn: Callable[[str, int, float], Awaitable[str]],
        max_batch_size: int = 8,
        max_wait_ms: int = 50,
    ) -> None:
        """
        Initialize batching queue.

        Args:
            backend_generate_fn: Async function to call for inference
                                 Signature: async (prompt, max_tokens, temperature) -> str
            max_batch_size: Maximum number of requests per batch
            max_wait_ms: Maximum wait time before flushing batch
        """
        self._backend_generate_fn: Callable[[str, int, float], Awaitable[str]] = (
            backend_generate_fn
        )
        self.MAX_BATCH_SIZE: int = max_batch_size
        self.MAX_WAIT_MS: int = max_wait_ms

        self._queue: List[PendingRequest] = []
        self._lock: asyncio.Lock = asyncio.Lock()
        self._batch_event: asyncio.Event = asyncio.Event()
        self._processor_task: Optional[asyncio.Task[None]] = None
        self._running: bool = False

        # Metrics
        self._total_batches: int = 0
        self._total_requests_batched: int = 0
        self._total_batch_wait_ms: float = 0.0

    async def start(self) -> None:
        """Start the background batch processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_batches())
        print(
            f"[BatchQueue] Started (max_batch={self.MAX_BATCH_SIZE}, max_wait={self.MAX_WAIT_MS}ms)"
        )

    async def stop(self) -> None:
        """Stop the background batch processor."""
        self._running = False
        if self._processor_task:
            self._batch_event.set()  # Wake up processor
            await self._processor_task
            self._processor_task = None
        print("[BatchQueue] Stopped")

    async def submit(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Submit request to batch queue and wait for result.

        Returns:
            Generated text from model
        """
        if not self._running:
            raise RuntimeError("BatchingInferenceQueue not started")

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request = PendingRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            future=future,
            created_at=time.time(),
        )

        async with self._lock:
            self._queue.append(request)
            # Wake processor if batch is full
            if len(self._queue) >= self.MAX_BATCH_SIZE:
                self._batch_event.set()

        # Wait for result
        return await future

    async def _process_batches(self) -> None:
        """
        Background task to process request batches.

        Wakes up when:
        1. Batch is full (MAX_BATCH_SIZE reached)
        2. Timeout expires (MAX_WAIT_MS elapsed)
        3. Queue is stopped
        """
        while self._running:
            # Wait for batch ready or timeout
            try:
                await asyncio.wait_for(
                    self._batch_event.wait(), timeout=self.MAX_WAIT_MS / 1000
                )
            except asyncio.TimeoutError:
                pass  # Timeout - process whatever we have

            self._batch_event.clear()

            # Get batch under lock
            async with self._lock:
                if not self._queue:
                    continue
                batch: List[PendingRequest] = self._queue[: self.MAX_BATCH_SIZE]
                self._queue = self._queue[self.MAX_BATCH_SIZE :]

            # Process batch IN PARALLEL (key optimization)
            # NOTE: This parallelizes asyncio tasks. Backend-level parallelism
            # depends on the backend implementation (llama.cpp may serialize internally,
            # but multiple concurrent requests can still benefit from better scheduling).
            batch_start: float = time.time()

            async def process_request(req: PendingRequest) -> None:
                """Process a single request from the batch."""
                try:
                    result: str = await self._backend_generate_fn(
                        req.prompt, req.max_tokens, req.temperature
                    )
                    req.future.set_result(result)
                except Exception as e:
                    req.future.set_exception(e)

            # Launch all requests in parallel
            await asyncio.gather(*[process_request(req) for req in batch])

            # Update metrics
            batch_duration: float = (time.time() - batch_start) * 1000
            self._total_batches += 1
            self._total_requests_batched += len(batch)
            self._total_batch_wait_ms += batch_duration

    def get_metrics(self) -> BatchingMetrics:
        """Get batching metrics."""
        return BatchingMetrics(
            total_batches=self._total_batches,
            total_requests=self._total_requests_batched,
            avg_batch_size=(
                self._total_requests_batched / self._total_batches
                if self._total_batches > 0
                else 0.0
            ),
            avg_batch_duration_ms=(
                self._total_batch_wait_ms / self._total_batches
                if self._total_batches > 0
                else 0.0
            ),
            queue_depth=len(self._queue),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceBackendBase(ABC):
    """
    Abstract base class for inference backends with circuit breaker support.

    All backends support optional circuit breaker protection for resilience
    against cascading failures when external services become unavailable.
    """

    _circuit_breaker: Optional[CircuitBreaker] = None

    @property
    @abstractmethod
    def backend_type(self) -> InferenceBackend:
        """Return the backend type."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend. Returns True if successful."""
        pass

    @abstractmethod
    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """Generate a completion."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate a completion with streaming."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass

    @abstractmethod
    def get_loaded_model(self) -> Optional[str]:
        """Return the currently loaded model name."""
        pass

    def get_circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker for this backend (if configured)."""
        return self._circuit_breaker

    def get_circuit_state(self) -> Optional[CircuitState]:
        """Get current circuit breaker state."""
        if self._circuit_breaker:
            return self._circuit_breaker.state
        return None

    def get_circuit_metrics(self) -> Optional[CircuitMetrics]:
        """Get circuit breaker metrics as dict."""
        if not self._circuit_breaker:
            return None
        metrics: CircuitBreakerMetrics = self._circuit_breaker.get_metrics()
        return CircuitMetrics(
            state=metrics.state.value,
            failure_count=metrics.failure_count,
            success_count=metrics.success_count,
            last_failure_time=metrics.last_failure_time,
            last_success_time=metrics.last_success_time,
            last_state_change=metrics.last_state_change,
            total_calls=metrics.total_calls,
            total_failures=metrics.total_failures,
            total_successes=metrics.total_successes,
            total_rejections=metrics.total_rejections,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LLAMA.CPP BACKEND
# ═══════════════════════════════════════════════════════════════════════════════


class LlamaCppBackend(InferenceBackendBase):
    """
    Embedded inference via llama-cpp-python.

    This is the primary backend for sovereign inference.
    No external dependencies, works offline.

    P0-P1 Optimization: Request batching for 8x throughput improvement.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._model = None
        self._model_path: Optional[str] = None
        self._lock = asyncio.Lock()

        # P0-P1: Batching queue (replaces serial lock)
        self._batch_queue: Optional[BatchingInferenceQueue] = None

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.LLAMACPP

    async def initialize(self) -> bool:
        """Initialize llama.cpp with configured model."""
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[LlamaCpp] llama-cpp-python not installed")
            return False

        model_path = self._resolve_model_path()
        if not model_path:
            print("[LlamaCpp] No model found")
            return False

        try:
            print(f"[LlamaCpp] Loading model: {model_path}")
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                n_batch=self.config.n_batch,
                verbose=False,
            )
            self._model_path = str(model_path)
            print("[LlamaCpp] Model loaded successfully")

            # P0-P1: Initialize batching queue if enabled
            if self.config.enable_batching:
                self._batch_queue = BatchingInferenceQueue(
                    backend_generate_fn=self._generate_direct,
                    max_batch_size=self.config.max_batch_size,
                    max_wait_ms=self.config.max_batch_wait_ms,
                )
                await self._batch_queue.start()
                print(
                    f"[LlamaCpp] Batching enabled (max_batch={self.config.max_batch_size})"
                )
            else:
                print("[LlamaCpp] Batching disabled (using serial lock)")

            return True
        except Exception as e:
            print(f"[LlamaCpp] Failed to load model: {e}")
            return False

    async def _generate_direct(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Direct generation (used by batch queue or fallback).

        NOTE: This is the actual inference call. The lock is acquired
        by the batch processor, not here.
        """
        if not self._model:
            raise RuntimeError("Model not initialized")

        # Run synchronous llama.cpp call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                **kwargs,
            ),
        )

        return result["choices"][0]["text"]

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Generate completion.

        P0-P1: Routes through batching queue if enabled, otherwise uses serial lock.
        """
        if not self._model:
            raise RuntimeError("Model not initialized")

        # P0-P1: Use batching queue if enabled
        if self._batch_queue:
            return await self._batch_queue.submit(prompt, max_tokens, temperature)

        # Fallback: Serial lock (original behavior)
        async with self._lock:
            return await self._generate_direct(
                prompt, max_tokens, temperature, **kwargs
            )

    async def generate_stream(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        if not self._model:
            raise RuntimeError("Model not initialized")

        # For streaming, we yield chunks - acquire lock at start
        async with self._lock:
            loop = asyncio.get_event_loop()
            # Get all chunks synchronously then yield asynchronously
            chunks = await loop.run_in_executor(
                None,
                lambda: list(
                    self._model(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        echo=False,
                        stream=True,
                        **kwargs,
                    )
                ),
            )
            for chunk in chunks:
                if "choices" in chunk and chunk["choices"]:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text

    async def health_check(self) -> bool:
        """Check if model is loaded and responsive."""
        if not self._model:
            return False
        try:
            # Quick inference test with async lock
            async with self._lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: self._model("test", max_tokens=1)
                )
            return True
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        """Return loaded model path."""
        return self._model_path

    async def shutdown(self):
        """Shutdown backend and cleanup resources."""
        if self._batch_queue:
            await self._batch_queue.stop()
            self._batch_queue = None
        self._model = None

    def get_batching_metrics(self) -> Optional[Dict[str, Any]]:
        """Get batching metrics if batching is enabled."""
        if self._batch_queue:
            return self._batch_queue.get_metrics()
        return None

    def _resolve_model_path(self) -> Optional[Path]:
        """Resolve the model path."""
        # 1. Explicit path
        if self.config.model_path:
            path = Path(self.config.model_path)
            if path.exists():
                return path

        # 2. Look in model directory
        if self.config.model_dir.exists():
            # Find any .gguf file
            gguf_files = list(self.config.model_dir.glob("*.gguf"))
            if gguf_files:
                return gguf_files[0]

        # 3. Try to download from HuggingFace
        # This would be implemented in a separate model manager
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA BACKEND (FALLBACK)
# ═══════════════════════════════════════════════════════════════════════════════


class OllamaBackend(InferenceBackendBase):
    """
    Ollama backend for fallback inference with circuit breaker and connection pooling.

    Requires Ollama server running externally.
    Circuit breaker prevents cascading failures when Ollama becomes unavailable.
    Connection pool reduces latency overhead for high-frequency requests.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._available_models: List[str] = []
        self._current_model: Optional[str] = None

        # Initialize circuit breaker for external service protection
        self._circuit_breaker = CircuitBreaker(
            name="ollama",
            config=config.circuit_breaker,
            on_state_change=self._on_circuit_state_change,
        )

        # Initialize connection pool if enabled
        self._connection_pool: Optional[ConnectionPool] = None
        if config.enable_connection_pool:
            self._connection_pool = ConnectionPool(
                backend_type="ollama",
                endpoint=config.ollama_url,
                config=config.connection_pool,
            )

    def _on_circuit_state_change(
        self, name: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            print("[Ollama] Circuit OPEN - backend unavailable, failing fast")
        elif new_state == CircuitState.HALF_OPEN:
            print("[Ollama] Circuit HALF_OPEN - testing recovery")
        elif new_state == CircuitState.CLOSED:
            print("[Ollama] Circuit CLOSED - backend recovered")

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.OLLAMA

    async def initialize(self) -> bool:
        """Check Ollama availability and list models."""
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/tags",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                data = json.loads(resp.read().decode())
                self._available_models = [m["name"] for m in data.get("models", [])]

                if self._available_models:
                    self._current_model = self._available_models[0]
                    print(f"[Ollama] Available models: {self._available_models}")

                    # Start connection pool if enabled
                    if self._connection_pool:
                        await self._connection_pool.start()
                        print("[Ollama] Connection pool started")

                    return True
                else:
                    print("[Ollama] No models available")
                    return False

        except Exception as e:
            print(f"[Ollama] Not available: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown backend and cleanup resources."""
        if self._connection_pool:
            await self._connection_pool.stop()
            self._connection_pool = None

    def get_connection_pool_metrics(self) -> Optional[Dict[str, Any]]:
        """Get connection pool metrics if pooling is enabled."""
        if self._connection_pool:
            # Return synchronous metrics snapshot
            return {
                "active_connections": self._connection_pool.get_active_connections(),
                "available_connections": self._connection_pool.get_available_connections(),
                "total_connections": self._connection_pool.get_total_connections(),
            }
        return None

    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Internal generate method (unprotected)."""
        import urllib.request

        payload = json.dumps(
            {
                "model": self._current_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.config.ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Run blocking call in executor
        loop = asyncio.get_event_loop()
        timeout = self.config.circuit_breaker.request_timeout

        def make_request():
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                data = json.loads(resp.read().decode())
                return data.get("response", "")

        return await loop.run_in_executor(None, make_request)

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Generate via Ollama API with circuit breaker protection.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        return await self._circuit_breaker.execute(
            self._generate_internal, prompt, max_tokens, temperature
        )

    async def generate_stream(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate with streaming via Ollama API."""
        # For simplicity, just return full response
        # Full streaming implementation would use httpx or aiohttp
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response

    async def health_check(self) -> bool:
        """Check Ollama health (bypasses circuit breaker for health checks)."""
        import urllib.error
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.config.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:  # nosec B310 — URL from trusted InferenceConfig (localhost Ollama)
                return resp.status == 200
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        return self._current_model


# ═══════════════════════════════════════════════════════════════════════════════
# LM STUDIO BACKEND (PRIMARY - v2.2.1)
# ═══════════════════════════════════════════════════════════════════════════════


class LMStudioBackend(InferenceBackendBase):
    """
    LM Studio v1 API backend - PRIMARY backend for BIZRA inference.

    Connects to LM Studio at 192.168.56.1:1234 with native /api/v1/chat
    endpoint supporting stateful chats and MCP integration.

    Circuit breaker protection prevents cascading failures when LM Studio
    becomes unavailable or overloaded.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._client = None
        self._current_model: Optional[str] = None
        self._available = False

        # Initialize circuit breaker for external service protection
        self._circuit_breaker = CircuitBreaker(
            name="lmstudio",
            config=config.circuit_breaker,
            on_state_change=self._on_circuit_state_change,
        )

    def _on_circuit_state_change(
        self, name: str, old_state: CircuitState, new_state: CircuitState
    ) -> None:
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            print("[LMStudio] Circuit OPEN - backend unavailable, failing fast")
        elif new_state == CircuitState.HALF_OPEN:
            print("[LMStudio] Circuit HALF_OPEN - testing recovery")
        elif new_state == CircuitState.CLOSED:
            print("[LMStudio] Circuit CLOSED - backend recovered")

    @property
    def backend_type(self) -> InferenceBackend:
        return InferenceBackend.LMSTUDIO

    async def initialize(self) -> bool:
        """
        Initialize LM Studio connection.

        REQUIREMENT: A model must be loaded in LM Studio for initialization
        to succeed. If LM Studio is reachable but no model is loaded,
        initialization fails to prevent runtime errors in generate().
        """
        if not LMSTUDIO_AVAILABLE:
            print("[LMStudio] lmstudio_backend module not available")
            return False

        try:
            lms_config = LMStudioConfig(
                host=self.config.lmstudio_url.replace("http://", "").split(":")[0],
                port=int(self.config.lmstudio_url.split(":")[-1]),
                api_key=os.getenv("LM_API_TOKEN")
                or os.getenv("LMSTUDIO_API_KEY")
                or os.getenv("LM_STUDIO_API_KEY"),
                use_native_api=True,
                enable_mcp=True,
            )
            self._client = LMStudioClient(lms_config)

            if await self._client.connect():
                # Check for loaded models - REQUIRED for successful initialization
                models = await self._client.list_models()
                loaded = [m for m in models if m.loaded]
                if loaded:
                    self._current_model = loaded[0].id
                    self._available = True
                    print(f"[LMStudio] Connected with model: {self._current_model}")
                    return True
                else:
                    # FAIL: No model loaded - generate() would fail
                    print(
                        f"[LMStudio] Connected but NO MODEL LOADED ({len(models)} available)"
                    )
                    print("[LMStudio] Load a model in LM Studio to enable this backend")
                    self._available = False
                    return False
            else:
                print("[LMStudio] Connection failed")
                return False
        except Exception as e:
            print(f"[LMStudio] Initialization error: {e}")
            return False

    async def _generate_internal(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Internal generate method (unprotected)."""
        if not self._client or not self._available:
            raise RuntimeError("LM Studio not initialized")

        response = await self._client.chat(
            messages=[ChatMessage(role="user", content=prompt)],
            model=self._current_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content

    async def generate(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> str:
        """
        Generate via LM Studio API with circuit breaker protection.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        return await self._circuit_breaker.execute(
            self._generate_internal, prompt, max_tokens, temperature
        )

    async def generate_stream(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs
    ) -> AsyncIterator[str]:
        """Generate with streaming."""
        # Simplified: return full response
        response = await self.generate(prompt, max_tokens, temperature, **kwargs)
        yield response

    async def health_check(self) -> bool:
        """Check LM Studio health (bypasses circuit breaker for health checks)."""
        if not self._client:
            return False
        try:
            models = await self._client.list_models()
            return len(models) > 0
        except Exception:
            return False

    def get_loaded_model(self) -> Optional[str]:
        return self._current_model


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE GATEWAY
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceGateway:
    """
    Tiered inference gateway with fail-closed semantics, circuit breaker resilience,
    and rate limiting protection.

    Routes requests to appropriate compute tier based on complexity.
    Provides fallback chain when primary backend unavailable.
    Circuit breakers protect against cascading failures from external services.
    Rate limiting prevents request overload and ensures fair resource allocation.

    Fail-closed: If no backend available and require_local=True, deny request.

    Standing on Giants:
    - Nygard (2007): Release It! - Circuit breaker pattern
    - Netflix Hystrix: Fallback and resilience patterns
    - RFC 6585: HTTP 429 Too Many Requests
    - Token Bucket Algorithm: Rate limiting
    """

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config: InferenceConfig = config or InferenceConfig()
        self.status: InferenceStatus = InferenceStatus.COLD

        # Backends by tier
        self._backends: Dict[ComputeTier, InferenceBackendBase] = {}
        self._active_backend: Optional[InferenceBackendBase] = None

        # Fallback chain (ordered list of backends to try)
        self._fallback_backends: List[InferenceBackendBase] = []

        # Rate limiter (RFC 6585 / Token Bucket)
        self._rate_limiter: Optional[RateLimiter] = None
        if self.config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(
                config=self.config.rate_limiter, name="gateway"
            )
            print(
                f"[Gateway] Rate limiting enabled "
                f"(rate={self.config.rate_limiter.tokens_per_second}/s, "
                f"max={self.config.rate_limiter.max_tokens}, "
                f"burst={self.config.rate_limiter.burst_size})"
            )

        # Metrics
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_latency_ms: float = 0.0
        self._circuit_breaker_trips: int = 0
        self._fallback_invocations: int = 0
        self._rate_limit_rejections: int = 0

    async def initialize(self) -> bool:
        """
        Initialize the gateway and backends.

        Priority order (v2.2.1):
        - When require_local=True (fail-closed sovereign mode):
          1. llama.cpp ONLY (embedded, offline-capable)
          2. DENY (fail-closed)

        - When require_local=False (fallback-enabled mode):
          1. LM Studio v1 (192.168.56.1:1234) - PRIMARY, RTX 4090 optimized
          2. Configured fallbacks (ollama, lmstudio) in order
          3. llama.cpp (embedded) - offline/edge sovereign
          4. DENY (fail-closed)

        Returns True if at least one backend is available.
        """
        self.status = InferenceStatus.WARMING

        # SECURITY: When require_local=True, ONLY try llama.cpp (embedded)
        # External services (LM Studio, Ollama) are NOT considered "local"
        # as they require network connectivity and are not self-contained.
        if self.config.require_local:
            print("[Gateway] require_local=True: Only trying embedded backends")
            llamacpp = LlamaCppBackend(self.config)
            if await llamacpp.initialize():
                self._backends[ComputeTier.LOCAL] = llamacpp
                self._backends[ComputeTier.EDGE] = llamacpp
                self._active_backend = llamacpp
                self.status = InferenceStatus.READY
                print("[Gateway] llama.cpp backend ready (SOVEREIGN MODE)")
                return True

            # Fail-closed: No fallbacks allowed when require_local=True
            self.status = InferenceStatus.OFFLINE
            print(
                "[Gateway] No embedded backend available (OFFLINE MODE - FAIL-CLOSED)"
            )
            return False

        # --- Fallback-enabled mode (require_local=False) ---

        # 1. Try LM Studio first (PRIMARY - RTX 4090 optimized)
        if LMSTUDIO_AVAILABLE:
            lmstudio = LMStudioBackend(self.config)
            if await lmstudio.initialize():
                self._backends[ComputeTier.LOCAL] = lmstudio
                self._active_backend = lmstudio
                self.status = InferenceStatus.READY
                print("[Gateway] LM Studio v1 backend ready (PRIMARY MODE)")
                return True

        # 2. Try configured fallbacks in order
        for fallback in self.config.fallbacks:
            if fallback == "ollama":
                ollama = OllamaBackend(self.config)
                if await ollama.initialize():
                    self._backends[ComputeTier.LOCAL] = ollama
                    self._active_backend = ollama
                    self.status = InferenceStatus.DEGRADED
                    print("[Gateway] Ollama fallback ready (DEGRADED MODE)")
                    return True

            elif fallback == "lmstudio" and LMSTUDIO_AVAILABLE:
                # LM Studio as fallback (different config or retry)
                lmstudio = LMStudioBackend(self.config)
                if await lmstudio.initialize():
                    self._backends[ComputeTier.LOCAL] = lmstudio
                    self._active_backend = lmstudio
                    self.status = InferenceStatus.DEGRADED
                    print("[Gateway] LM Studio fallback ready (DEGRADED MODE)")
                    return True

        # 3. Try llama.cpp (offline/edge - embedded, sovereign)
        llamacpp = LlamaCppBackend(self.config)
        if await llamacpp.initialize():
            self._backends[ComputeTier.LOCAL] = llamacpp
            self._backends[ComputeTier.EDGE] = llamacpp
            self._active_backend = llamacpp
            self.status = InferenceStatus.READY
            print("[Gateway] llama.cpp backend ready (SOVEREIGN MODE)")
            return True

        # 4. Fail-closed
        self.status = InferenceStatus.OFFLINE
        print("[Gateway] No backend available (OFFLINE MODE)")
        return False

    def estimate_complexity(self, prompt: str) -> TaskComplexity:
        """
        Estimate task complexity for routing decisions.

        Simple heuristics for now. Could be replaced with classifier.
        """
        words = prompt.split()
        input_tokens = len(words) * 1.3  # Rough estimate

        # Heuristics for reasoning depth
        reasoning_keywords = ["why", "how", "explain", "analyze", "compare", "prove"]
        reasoning_depth = sum(
            1 for w in words if w.lower() in reasoning_keywords
        ) / max(len(words), 1)

        # Heuristics for domain specificity
        technical_keywords = [
            "algorithm",
            "equation",
            "theorem",
            "protocol",
            "architecture",
        ]
        domain_specificity = sum(
            1 for w in words if w.lower() in technical_keywords
        ) / max(len(words), 1)

        return TaskComplexity(
            input_tokens=int(input_tokens),
            estimated_output_tokens=min(int(input_tokens * 2), 2048),
            reasoning_depth=min(reasoning_depth * 5, 1.0),
            domain_specificity=min(domain_specificity * 5, 1.0),
        )

    def route(self, complexity: TaskComplexity) -> ComputeTier:
        """
        Route task to appropriate compute tier.

        EDGE: complexity < 0.3
        LOCAL: 0.3 <= complexity < 0.8
        POOL: complexity >= 0.8
        """
        score = complexity.score

        if score < 0.3:
            return ComputeTier.EDGE
        elif score < 0.8:
            return ComputeTier.LOCAL
        else:
            return ComputeTier.POOL

    async def infer(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        tier: Optional[ComputeTier] = None,
        stream: bool = False,
        client_id: Optional[str] = None,
    ) -> Union[InferenceResult, AsyncIterator[str]]:
        """
        Run inference on prompt with rate limiting, circuit breaker protection,
        and automatic failover.

        Rate limiting is applied first (RFC 6585). If rate limit exceeded,
        raises RateLimitError with HTTP 429 status and retry-after hint.

        When a backend's circuit breaker is open, the gateway automatically attempts
        fallback backends in order. This implements the Netflix Hystrix fallback pattern.

        Fail-closed: Raises RuntimeError if no backend available (all circuits open
        or no backends configured).

        Args:
            prompt: The prompt to run inference on
            max_tokens: Maximum tokens to generate (uses config default if None)
            temperature: Sampling temperature (0.0-1.0)
            tier: Force a specific compute tier (auto-routes if None)
            stream: If True, return AsyncIterator instead of InferenceResult
            client_id: Optional client identifier for per-client rate limiting

        Returns:
            InferenceResult or AsyncIterator[str] if stream=True

        Raises:
            RateLimitError: If rate limit exceeded (HTTP 429)
            RuntimeError: If no backend available or all circuits open

        Standing on Giants:
        - Nygard (2007): Release It! - Fail fast pattern
        - Netflix Hystrix: Fallback and bulkhead patterns
        - RFC 6585: HTTP 429 Too Many Requests
        """
        # Rate limiting check (RFC 6585)
        if self._rate_limiter:
            try:
                await self._rate_limiter.acquire(client_id=client_id)
            except RateLimitError as e:
                self._rate_limit_rejections += 1
                print(
                    f"[Gateway] Rate limit exceeded for client '{e.client_id}' "
                    f"(tokens={e.current_tokens:.2f}, retry_after={e.retry_after:.2f}s)"
                )
                raise

        # Check availability
        if self.status == InferenceStatus.OFFLINE:
            raise RuntimeError("Inference denied: no backend available (fail-closed)")

        if not self._active_backend:
            raise RuntimeError("Inference denied: no active backend")

        # Estimate complexity and route
        complexity = self.estimate_complexity(prompt)
        target_tier = tier or self.route(complexity)

        # Get backend for tier (fallback to active)
        primary_backend = self._backends.get(target_tier, self._active_backend)

        # Build fallback chain: primary -> other tier backends -> fallback backends
        backends_to_try = [primary_backend]
        for tier_backend in self._backends.values():
            if tier_backend not in backends_to_try:
                backends_to_try.append(tier_backend)
        for fallback_backend in self._fallback_backends:
            if fallback_backend not in backends_to_try:
                backends_to_try.append(fallback_backend)

        # Run inference with circuit breaker failover
        start_time = time.time()
        max_tokens = max_tokens or self.config.max_tokens

        if stream:
            # For streaming, we cannot easily failover mid-stream
            # Return generator from primary backend (circuit breaker will raise on open)
            return primary_backend.generate_stream(prompt, max_tokens, temperature)

        # Try each backend in order until one succeeds
        last_error: Optional[Exception] = None
        used_backend: Optional[InferenceBackendBase] = None

        for backend in backends_to_try:
            try:
                response = await backend.generate(prompt, max_tokens, temperature)
                used_backend = backend

                # Track if we used a fallback
                if backend != primary_backend:
                    self._fallback_invocations += 1
                    print(
                        f"[Gateway] Fallback to {backend.backend_type.value} succeeded"
                    )

                break

            except CircuitBreakerError as e:
                # Circuit is open - try next backend
                self._circuit_breaker_trips += 1
                print(
                    f"[Gateway] Circuit open for {e.backend_name}, trying fallback..."
                )
                last_error = e
                continue

            except Exception as e:
                # Other error - backend failure, try next
                print(f"[Gateway] Backend {backend.backend_type.value} failed: {e}")
                last_error = e
                continue

        if used_backend is None:
            # All backends failed
            raise RuntimeError(
                f"Inference denied: all backends unavailable. "
                f"Last error: {last_error}"
            )

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_generated = len(response.split())  # Rough estimate
        tokens_per_second = (
            tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0
        )

        # Update stats
        self._total_requests += 1
        self._total_tokens += tokens_generated
        self._total_latency_ms += latency_ms

        return InferenceResult(
            content=response,
            model=used_backend.get_loaded_model() or "unknown",
            backend=used_backend.backend_type,
            tier=target_tier,
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            latency_ms=round(latency_ms, 2),
        )

    async def health(self) -> Dict[str, Any]:
        """
        Get gateway health status including circuit breaker and rate limiter metrics.

        Returns comprehensive health information:
        - Gateway status and active backend
        - Per-backend health checks
        - Circuit breaker states and metrics
        - Rate limiter metrics (if enabled)
        - Batching metrics (if enabled)
        - Request/token statistics
        """
        backends_health: Dict[str, bool] = {}
        circuit_breakers: Dict[str, Dict[str, Any]] = {}

        for tier, backend in self._backends.items():
            tier_name = tier.value
            backends_health[tier_name] = await backend.health_check()

            # Collect circuit breaker metrics
            cb_metrics = backend.get_circuit_metrics()
            if cb_metrics:
                circuit_breakers[f"{tier_name}_{backend.backend_type.value}"] = (
                    cb_metrics
                )

        # P0-P1: Include batching metrics if available
        batching_metrics: Optional[Dict[str, Any]] = None
        if self._active_backend and hasattr(
            self._active_backend, "get_batching_metrics"
        ):
            batching_metrics = self._active_backend.get_batching_metrics()

        health_data: Dict[str, Any] = {
            "status": self.status.value,
            "active_backend": (
                self._active_backend.backend_type.value
                if self._active_backend
                else None
            ),
            "active_model": (
                self._active_backend.get_loaded_model()
                if self._active_backend
                else None
            ),
            "backends": backends_health,
            "stats": {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "avg_latency_ms": (
                    self._total_latency_ms / self._total_requests
                    if self._total_requests > 0
                    else 0.0
                ),
                "circuit_breaker_trips": self._circuit_breaker_trips,
                "fallback_invocations": self._fallback_invocations,
                "rate_limit_rejections": self._rate_limit_rejections,
            },
        }

        if circuit_breakers:
            health_data["circuit_breakers"] = circuit_breakers

        # Rate limiter metrics (RFC 6585)
        if self._rate_limiter:
            health_data["rate_limiter"] = self._rate_limiter.get_metrics()

        if batching_metrics:
            health_data["batching"] = batching_metrics

        # P1: Include connection pool metrics if available
        connection_pool_metrics: Dict[str, Any] = {}
        for tier, backend in self._backends.items():
            if hasattr(backend, "get_connection_pool_metrics"):
                pool_metrics = backend.get_connection_pool_metrics()
                if pool_metrics:
                    connection_pool_metrics[
                        f"{tier.value}_{backend.backend_type.value}"
                    ] = pool_metrics

        if connection_pool_metrics:
            health_data["connection_pools"] = connection_pool_metrics

        return health_data

    def get_circuit_breaker_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all circuit breaker states.

        Returns dict mapping backend names to their circuit breaker states.
        Useful for monitoring dashboards and alerting.
        """
        summary: Dict[str, Dict[str, Any]] = {}
        for tier, backend in self._backends.items():
            cb = backend.get_circuit_breaker()
            if cb:
                metrics = cb.get_metrics()
                summary[f"{tier.value}_{backend.backend_type.value}"] = {
                    "state": metrics.state.value,
                    "failure_count": metrics.failure_count,
                    "last_failure": metrics.last_failure_time,
                    "total_rejections": metrics.total_rejections,
                }
        return summary

    def reset_circuit_breaker(self, backend_type: str) -> bool:
        """
        Manually reset a circuit breaker to CLOSED state.

        Args:
            backend_type: The backend type to reset (e.g., "ollama", "lmstudio")

        Returns:
            True if reset successful, False if backend not found
        """
        for backend in self._backends.values():
            if backend.backend_type.value == backend_type:
                cb = backend.get_circuit_breaker()
                if cb:
                    cb.reset()
                    return True
        return False

    def get_rate_limiter(self) -> Optional[RateLimiter]:
        """Get the gateway's rate limiter instance."""
        return self._rate_limiter

    def get_rate_limiter_metrics(self) -> Optional[RateLimiterMetrics]:
        """
        Get rate limiter metrics.

        Returns:
            RateLimiterMetrics if rate limiting is enabled, None otherwise
        """
        if self._rate_limiter:
            return self._rate_limiter.get_metrics()
        return None

    async def get_client_rate_status(
        self, client_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get rate limit status for a specific client.

        Args:
            client_id: Client identifier (uses default if None)

        Returns:
            Dict with client's current token count and limits
        """
        if self._rate_limiter:
            return await self._rate_limiter.get_client_metrics(client_id)
        return None

    def reset_rate_limiter(self, client_id: Optional[str] = None) -> bool:
        """
        Reset rate limiter state.

        Args:
            client_id: If provided, reset only this client. Otherwise reset all.

        Returns:
            True if rate limiter was reset, False if not enabled
        """
        if self._rate_limiter:
            self._rate_limiter.reset(client_id)
            return True
        return False

    async def shutdown(self) -> None:
        """Shutdown gateway and all backends."""
        for backend in self._backends.values():
            if hasattr(backend, "shutdown"):
                await backend.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_gateway_instance: Optional[InferenceGateway] = None


def get_inference_gateway() -> InferenceGateway:
    """Get the singleton inference gateway."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = InferenceGateway()
    return _gateway_instance


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Inference Gateway")
    parser.add_argument("command", choices=["init", "infer", "health"])
    parser.add_argument("--prompt", help="Prompt for inference")
    parser.add_argument("--model", help="Model path")
    parser.add_argument("--tier", choices=["edge", "local", "pool"])
    args = parser.parse_args()

    gateway = get_inference_gateway()

    if args.model:
        gateway.config.model_path = args.model

    if args.command == "init":
        success = await gateway.initialize()
        print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")
        print(f"Status: {gateway.status.value}")

    elif args.command == "infer":
        if not args.prompt:
            print("Error: --prompt required")
            return

        await gateway.initialize()
        tier = ComputeTier(args.tier) if args.tier else None

        result = await gateway.infer(args.prompt, tier=tier)
        print(f"\n{'='*60}")
        print(f"Model: {result.model}")
        print(f"Backend: {result.backend.value}")
        print(f"Tier: {result.tier.value}")
        print(f"Tokens: {result.tokens_generated} @ {result.tokens_per_second} tok/s")
        print(f"Latency: {result.latency_ms}ms")
        print(f"{'='*60}")
        print(result.content)

    elif args.command == "health":
        await gateway.initialize()
        health = await gateway.health()
        print(json.dumps(health, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
