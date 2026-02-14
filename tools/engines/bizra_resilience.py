# BIZRA Resilience Patterns v1.0
# Circuit Breaker, Retry Logic, and Graceful Degradation
# Part of SAPE Implementation Blueprint

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime, timedelta

logger = logging.getLogger("BIZRA.Resilience")

# Type variable for generic decorators
T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time before half-open from open
    half_open_max_calls: int = 3        # Max calls in half-open state
    exclude_exceptions: tuple = ()       # Exceptions that don't count as failures


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0             # Initial delay in seconds
    max_delay: float = 30.0             # Maximum delay
    exponential_base: float = 2.0       # Exponential backoff multiplier
    jitter: bool = True                 # Add randomness to delays
    retryable_exceptions: tuple = (Exception,)  # Exceptions to retry


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: List[Dict] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit Breaker Pattern Implementation

    Prevents cascade failures by failing fast when a service is unhealthy.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service unhealthy, requests fail immediately
    - HALF_OPEN: Testing recovery, limited requests allowed

    Usage:
        breaker = CircuitBreaker("llm_backend")

        @breaker
        async def call_llm():
            ...
    """

    _instances: Dict[str, 'CircuitBreaker'] = {}

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._half_open_calls = 0

        # Register instance for global access
        CircuitBreaker._instances[name] = self

    @classmethod
    def get(cls, name: str) -> Optional['CircuitBreaker']:
        """Get circuit breaker by name"""
        return cls._instances.get(name)

    @classmethod
    def get_all_stats(cls) -> Dict[str, CircuitBreakerStats]:
        """Get stats for all circuit breakers"""
        return {name: cb.stats for name, cb in cls._instances.items()}

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition"""
        if self.stats.state == CircuitState.OPEN:
            if self.stats.last_failure_time:
                elapsed = time.time() - self.stats.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
        return self.stats.state

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state"""
        old_state = self.stats.state
        self.stats.state = new_state
        self._half_open_calls = 0

        self.stats.state_changes.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": datetime.now().isoformat()
        })

        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")

    async def _record_success(self):
        """Record a successful call"""
        async with self._lock:
            self.stats.successes += 1
            self.stats.total_successes += 1
            self.stats.total_calls += 1
            self.stats.last_success_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self.stats.failures = 0
                    self.stats.successes = 0

    async def _record_failure(self, exception: Exception):
        """Record a failed call"""
        async with self._lock:
            # Check if exception should be excluded
            if isinstance(exception, self.config.exclude_exceptions):
                return

            self.stats.failures += 1
            self.stats.total_failures += 1
            self.stats.total_calls += 1
            self.stats.last_failure_time = time.time()

            if self.stats.state == CircuitState.CLOSED:
                if self.stats.failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

            elif self.stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
                self.stats.failures = 0
                self.stats.successes = 0

    def _can_execute(self) -> bool:
        """Check if a call can be executed"""
        state = self.state  # This checks for timeout transition

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for wrapping functions with circuit breaker"""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self._can_execute():
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable, failing fast."
                    )

                try:
                    result = await func(*args, **kwargs)
                    await self._record_success()
                    return result
                except Exception as e:
                    await self._record_failure(e)
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self._can_execute():
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable, failing fast."
                    )

                try:
                    result = func(*args, **kwargs)
                    # For sync functions, we need to run the async record in an event loop
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._record_success())
                    else:
                        loop.run_until_complete(self._record_success())
                    return result
                except Exception as e:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self._record_failure(e))
                    else:
                        loop.run_until_complete(self._record_failure(e))
                    raise

            return sync_wrapper

    def reset(self):
        """Manually reset the circuit breaker"""
        self.stats = CircuitBreakerStats()
        self._half_open_calls = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


def retry(
    config: Optional[RetryConfig] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Retry decorator with exponential backoff

    Usage:
        @retry(max_retries=3, base_delay=1.0)
        async def flaky_operation():
            ...

        @retry(config=RetryConfig(...))
        def another_operation():
            ...
    """
    if config:
        _config = config
    else:
        _config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(_config.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except _config.retryable_exceptions as e:
                        last_exception = e

                        if attempt >= _config.max_retries:
                            logger.error(
                                f"Retry exhausted for {func.__name__} after "
                                f"{_config.max_retries + 1} attempts"
                            )
                            raise

                        # Calculate delay
                        delay = min(
                            _config.base_delay * (_config.exponential_base ** attempt),
                            _config.max_delay
                        )

                        # Add jitter
                        if _config.jitter:
                            import random
                            delay = delay * (0.5 + random.random())

                        logger.warning(
                            f"Retry {attempt + 1}/{_config.max_retries} for "
                            f"{func.__name__}: {str(e)[:50]}. "
                            f"Waiting {delay:.2f}s"
                        )

                        if on_retry:
                            on_retry(attempt + 1, e)

                        await asyncio.sleep(delay)

                raise last_exception

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(_config.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except _config.retryable_exceptions as e:
                        last_exception = e

                        if attempt >= _config.max_retries:
                            logger.error(
                                f"Retry exhausted for {func.__name__} after "
                                f"{_config.max_retries + 1} attempts"
                            )
                            raise

                        delay = min(
                            _config.base_delay * (_config.exponential_base ** attempt),
                            _config.max_delay
                        )

                        if _config.jitter:
                            import random
                            delay = delay * (0.5 + random.random())

                        logger.warning(
                            f"Retry {attempt + 1}/{_config.max_retries} for "
                            f"{func.__name__}: {str(e)[:50]}. "
                            f"Waiting {delay:.2f}s"
                        )

                        if on_retry:
                            on_retry(attempt + 1, e)

                        time.sleep(delay)

                raise last_exception

            return sync_wrapper

    return decorator


def with_fallback(
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None,
    log_error: bool = True
):
    """
    Fallback decorator for graceful degradation

    Usage:
        @with_fallback(fallback_value="default response")
        async def get_llm_response():
            ...

        @with_fallback(fallback_func=lambda: cached_response())
        async def expensive_operation():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if log_error:
                        logger.warning(
                            f"Fallback triggered for {func.__name__}: {str(e)[:50]}"
                        )

                    if fallback_func:
                        if asyncio.iscoroutinefunction(fallback_func):
                            return await fallback_func()
                        return fallback_func()

                    return fallback_value

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if log_error:
                        logger.warning(
                            f"Fallback triggered for {func.__name__}: {str(e)[:50]}"
                        )

                    if fallback_func:
                        return fallback_func()

                    return fallback_value

            return sync_wrapper

    return decorator


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiter"""
    requests_per_second: float = 10.0
    burst_size: int = 20


class RateLimiter:
    """
    Token bucket rate limiter

    Limits the rate of function calls to prevent overloading backends.
    """

    def __init__(self, name: str, config: Optional[RateLimiterConfig] = None):
        self.name = name
        self.config = config or RateLimiterConfig()
        self._tokens = float(self.config.burst_size)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token, waiting if necessary"""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(
                self.config.burst_size,
                self._tokens + elapsed * self.config.requests_per_second
            )

            if self._tokens >= 1:
                self._tokens -= 1
                return True

            # Calculate wait time
            wait_time = (1 - self._tokens) / self.config.requests_per_second
            await asyncio.sleep(wait_time)
            self._tokens = 0
            return True

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for rate limiting"""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                await self.acquire()
                return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.acquire())
                return func(*args, **kwargs)

            return sync_wrapper


# Pre-configured circuit breakers for common BIZRA services
llm_circuit_breaker = CircuitBreaker(
    "llm_backend",
    CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=30.0
    )
)

embedding_circuit_breaker = CircuitBreaker(
    "embedding_service",
    CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=60.0
    )
)

graph_circuit_breaker = CircuitBreaker(
    "graph_operations",
    CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=45.0
    )
)


# Convenience function to get resilience status
def get_resilience_status() -> Dict:
    """Get status of all resilience components"""
    return {
        "circuit_breakers": {
            name: {
                "state": cb.stats.state.value,
                "failures": cb.stats.failures,
                "successes": cb.stats.successes,
                "total_calls": cb.stats.total_calls,
                "total_failures": cb.stats.total_failures
            }
            for name, cb in CircuitBreaker._instances.items()
        },
        "timestamp": datetime.now().isoformat()
    }


# Main execution for testing
if __name__ == "__main__":
    import random

    async def test_resilience():
        print("=" * 60)
        print("BIZRA Resilience Patterns Test")
        print("=" * 60)

        # Test circuit breaker
        breaker = CircuitBreaker("test_service", CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=5.0
        ))

        @breaker
        @retry(max_retries=2, base_delay=0.5)
        async def flaky_service():
            if random.random() < 0.7:
                raise ConnectionError("Service unavailable")
            return "Success!"

        print("\n--- Testing Circuit Breaker + Retry ---")
        for i in range(10):
            try:
                result = await flaky_service()
                print(f"Call {i+1}: {result}")
            except CircuitOpenError as e:
                print(f"Call {i+1}: Circuit OPEN - {e}")
            except Exception as e:
                print(f"Call {i+1}: Failed - {type(e).__name__}")

            await asyncio.sleep(1)

        # Test fallback
        @with_fallback(fallback_value="Fallback response")
        async def failing_service():
            raise RuntimeError("Always fails")

        print("\n--- Testing Fallback ---")
        result = await failing_service()
        print(f"Fallback result: {result}")

        # Print status
        print("\n--- Resilience Status ---")
        status = get_resilience_status()
        for name, stats in status["circuit_breakers"].items():
            print(f"  {name}: {stats['state']} "
                  f"(failures={stats['failures']}, "
                  f"total_calls={stats['total_calls']})")

    asyncio.run(test_resilience())
    print("\n✅ Resilience patterns test complete")
