"""
BIZRA Inference Gateway — Resilience (Circuit Breaker + Rate Limiter)
═══════════════════════════════════════════════════════════════════════════════

Circuit breaker (Nygard 2007 / Netflix Hystrix) and token-bucket rate
limiter (RFC 6585) for inference backend protection.

Standing on Giants:
- Nygard (2007): Release It! — Circuit breaker pattern
- Netflix (2012): Hystrix library — Production-grade implementation
- Fowler (2014): CircuitBreaker documentation
- Token Bucket Algorithm (Benes, 1965): Rate limiting
- RFC 6585 (2012): HTTP 429 Too Many Requests

Extracted from gateway.py for modularity; re-exported by gateway.py
for backward compatibility.

Created: 2026-02-09 | Refactor split from gateway.py
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ._types import (
    CircuitBreakerConfig,
    CircuitState,
    RateLimiterConfig,
    RateLimiterMetrics,
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
