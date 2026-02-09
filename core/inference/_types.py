"""
BIZRA Inference Gateway — Type Definitions
═══════════════════════════════════════════════════════════════════════════════

TypedDicts, Enums, Protocols, constants, and config dataclasses used
throughout the inference subsystem.

Extracted from gateway.py for modularity; re-exported by gateway.py
for backward compatibility.

Created: 2026-02-09 | Refactor split from gateway.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    TypedDict,
)

# Note: ``Any`` is used for the ``InferenceConfig.connection_pool`` field to
# break a circular import with ``_connection_pool.ConnectionPoolConfig``.
# The real default is injected lazily in ``InferenceConfig.__post_init__``.


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CONSTANTS
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
# ENUMS
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


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
# INFERENCE CONFIG + RESULT
# ═══════════════════════════════════════════════════════════════════════════════


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
    connection_pool: Any = field(default_factory=lambda: None)  # ConnectionPoolConfig; patched at import time

    # Rate limiter settings (RFC 6585)
    enable_rate_limiting: bool = True
    rate_limiter: RateLimiterConfig = field(default_factory=RateLimiterConfig)

    def __post_init__(self) -> None:
        # Deferred default: replace None with real ConnectionPoolConfig
        if self.connection_pool is None:
            from ._connection_pool import ConnectionPoolConfig
            self.connection_pool = ConnectionPoolConfig()


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
