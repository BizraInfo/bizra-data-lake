# BIZRA Resilience Patterns Tests
# Unit tests for circuit breaker and retry logic

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCircuitBreaker:
    """Test suite for circuit breaker pattern"""

    @pytest.fixture
    def circuit_breaker(self):
        """Create fresh circuit breaker"""
        from bizra_resilience import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0
        )
        return CircuitBreaker(f"test_breaker_{id(self)}", config)

    def test_circuit_starts_closed(self, circuit_breaker):
        """Test circuit breaker starts in CLOSED state"""
        from bizra_resilience import CircuitState
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_failures(self, circuit_breaker):
        """Test circuit opens after threshold failures"""
        from bizra_resilience import CircuitState

        # Simulate failures
        for _ in range(3):
            await circuit_breaker._record_failure(Exception("test"))

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to HALF_OPEN after timeout"""
        from bizra_resilience import CircuitState

        # Open the circuit
        for _ in range(3):
            await circuit_breaker._record_failure(Exception("test"))

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.5)

        # State check should trigger transition
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_successes(self, circuit_breaker):
        """Test circuit closes after success threshold"""
        from bizra_resilience import CircuitState

        # Open the circuit
        for _ in range(3):
            await circuit_breaker._record_failure(Exception("test"))

        # Wait for half-open
        await asyncio.sleep(1.5)
        _ = circuit_breaker.state  # Trigger transition

        # Record successes
        for _ in range(2):
            await circuit_breaker._record_success()

        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_reopens_on_half_open_failure(self, circuit_breaker):
        """Test circuit reopens if failure occurs in HALF_OPEN"""
        from bizra_resilience import CircuitState

        # Open the circuit
        for _ in range(3):
            await circuit_breaker._record_failure(Exception("test"))

        # Wait for half-open
        await asyncio.sleep(1.5)
        _ = circuit_breaker.state

        # Failure in half-open
        await circuit_breaker._record_failure(Exception("test"))

        assert circuit_breaker.state == CircuitState.OPEN

    def test_circuit_breaker_stats(self, circuit_breaker):
        """Test statistics tracking"""
        assert circuit_breaker.stats.total_calls == 0
        assert circuit_breaker.stats.total_failures == 0
        assert circuit_breaker.stats.total_successes == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator"""
        from bizra_resilience import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError

        breaker = CircuitBreaker(
            "decorator_test",
            CircuitBreakerConfig(failure_threshold=2, timeout_seconds=1.0)
        )

        call_count = 0

        @breaker
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        # First two calls should raise ValueError
        for _ in range(2):
            with pytest.raises(ValueError):
                await flaky_func()

        # Third call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await flaky_func()

        assert call_count == 2  # Only 2 actual calls made


class TestRetry:
    """Test suite for retry decorator"""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_first_try(self):
        """Test no retry needed when function succeeds"""
        from bizra_resilience import retry

        call_count = 0

        @retry(max_retries=3)
        async def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await always_succeeds()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        """Test retry succeeds after initial failures"""
        from bizra_resilience import retry

        call_count = 0

        @retry(max_retries=3, base_delay=0.1)
        async def fails_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        result = await fails_then_succeeds()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhausts and raises exception"""
        from bizra_resilience import retry

        call_count = 0

        @retry(max_retries=2, base_delay=0.1)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError):
            await always_fails()

        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_respects_exception_filter(self):
        """Test retry only retries specified exceptions"""
        from bizra_resilience import retry

        call_count = 0

        @retry(max_retries=3, base_delay=0.1, retryable_exceptions=(ConnectionError,))
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raises_value_error()

        assert call_count == 1  # No retries for ValueError


class TestFallback:
    """Test suite for fallback decorator"""

    @pytest.mark.asyncio
    async def test_fallback_returns_value_on_error(self):
        """Test fallback value returned on error"""
        from bizra_resilience import with_fallback

        @with_fallback(fallback_value="fallback")
        async def always_fails():
            raise RuntimeError("Error")

        result = await always_fails()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_function_called(self):
        """Test fallback function called on error"""
        from bizra_resilience import with_fallback

        fallback_called = False

        def fallback_func():
            nonlocal fallback_called
            fallback_called = True
            return "from_function"

        @with_fallback(fallback_func=fallback_func)
        async def always_fails():
            raise RuntimeError("Error")

        result = await always_fails()
        assert result == "from_function"
        assert fallback_called

    @pytest.mark.asyncio
    async def test_no_fallback_on_success(self):
        """Test no fallback when function succeeds"""
        from bizra_resilience import with_fallback

        @with_fallback(fallback_value="fallback")
        async def always_succeeds():
            return "success"

        result = await always_succeeds()
        assert result == "success"


class TestRateLimiter:
    """Test suite for rate limiter"""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst(self):
        """Test rate limiter allows burst up to capacity"""
        from bizra_resilience import RateLimiter, RateLimiterConfig

        config = RateLimiterConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter("test_limiter", config)

        call_count = 0

        @limiter
        async def limited_func():
            nonlocal call_count
            call_count += 1
            return call_count

        # Should allow 5 calls immediately (burst)
        start = asyncio.get_event_loop().time()
        for _ in range(5):
            await limited_func()
        elapsed = asyncio.get_event_loop().time() - start

        assert call_count == 5
        assert elapsed < 1.0  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles(self):
        """Test rate limiter throttles after burst"""
        from bizra_resilience import RateLimiter, RateLimiterConfig

        config = RateLimiterConfig(requests_per_second=10.0, burst_size=2)
        limiter = RateLimiter("throttle_test", config)

        @limiter
        async def limited_func():
            return True

        # Exhaust burst
        await limited_func()
        await limited_func()

        # Next call should be delayed
        start = asyncio.get_event_loop().time()
        await limited_func()
        elapsed = asyncio.get_event_loop().time() - start

        # Should have waited ~0.1 seconds (1/10 RPS)
        assert elapsed >= 0.05  # Allow some tolerance


class TestResilienceIntegration:
    """Integration tests for combined resilience patterns"""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry and circuit breaker work together"""
        from bizra_resilience import CircuitBreaker, CircuitBreakerConfig, retry, CircuitOpenError

        breaker = CircuitBreaker(
            "integration_test",
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=10.0)
        )

        call_count = 0

        @breaker
        @retry(max_retries=2, base_delay=0.1)
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary")
            return "success"

        result = await flaky_service()
        assert result == "success"
        assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
