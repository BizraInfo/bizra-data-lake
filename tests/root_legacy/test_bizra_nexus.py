#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
    BIZRA NEXUS — Comprehensive Test Suite
    
    Test-Driven Development demonstration with:
    • Unit tests for all core components
    • Integration tests for engine orchestration
    • Property-based tests for edge cases
    • Performance benchmarks
    • Stress tests for resilience patterns
═══════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import pytest
import asyncio
import threading
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bizra_nexus import (
    NexusConfig, StructuredLogger, Event, EventType, QueryResult,
    HealthStatus, HealthCheck, CircuitBreaker, CircuitState,
    TTLCache, EventBus, ResourcePool, EngineAdapter, QueryRouter,
    ResultAggregator, BizraNexus, with_retry, with_timeout
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    """Test configuration."""
    return NexusConfig(
        max_workers=4,
        cache_size=100,
        batch_size=32,
        retry_attempts=2,
        retry_delay=0.1
    )


@pytest.fixture
def cache():
    """Fresh TTL cache for testing."""
    return TTLCache[str, str](max_size=10, default_ttl=60.0)


@pytest.fixture
def circuit():
    """Fresh circuit breaker for testing."""
    return CircuitBreaker(name="test", threshold=3, timeout=1.0)


@pytest.fixture
def event_bus():
    """Fresh event bus for testing."""
    return EventBus()


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — NexusConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestNexusConfig:
    """Tests for immutable configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = NexusConfig()
        assert config.max_workers == 8
        assert config.cache_size == 10000
        assert config.batch_size == 256
    
    def test_immutability(self):
        """Test that config is immutable."""
        config = NexusConfig()
        with pytest.raises(AttributeError):
            config.max_workers = 10
    
    def test_validation_max_workers(self):
        """Test validation of max_workers."""
        with pytest.raises(ValueError) as exc_info:
            NexusConfig(max_workers=0)
        assert "max_workers must be >= 1" in str(exc_info.value)
    
    def test_validation_cache_size(self):
        """Test validation of cache_size."""
        with pytest.raises(ValueError) as exc_info:
            NexusConfig(cache_size=-1)
        assert "cache_size must be >= 0" in str(exc_info.value)
    
    def test_derived_paths(self, config):
        """Test derived path properties."""
        assert config.gold_path == config.data_lake_path / "04_GOLD"
        assert config.indexed_path == config.data_lake_path / "03_INDEXED"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — TTLCache
# ═══════════════════════════════════════════════════════════════════════════════

class TestTTLCache:
    """Tests for TTL cache implementation."""
    
    def test_basic_set_get(self, cache):
        """Test basic set and get operations."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
    
    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        assert cache.get("nonexistent") is None
    
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = TTLCache[str, str](max_size=10, default_ttl=0.1)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = TTLCache[str, str](max_size=3, default_ttl=60.0)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        cache.set("d", "4")  # Should evict 'a'
        assert cache.get("a") is None
        assert cache.get("b") == "2"
    
    def test_access_updates_order(self):
        """Test that accessing a key updates its position."""
        cache = TTLCache[str, str](max_size=3, default_ttl=60.0)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        _ = cache.get("a")  # Access 'a' to update order
        cache.set("d", "4")  # Should evict 'b', not 'a'
        assert cache.get("a") == "1"
        assert cache.get("b") is None
    
    def test_invalidate(self, cache):
        """Test cache invalidation."""
        cache.set("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None
    
    def test_clear(self, cache):
        """Test cache clear."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None
    
    def test_stats(self, cache):
        """Test cache statistics."""
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.stats
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_thread_safety(self, cache):
        """Test thread-safe operations."""
        def writer(key: str):
            for i in range(100):
                cache.set(f"{key}_{i}", f"value_{i}")
        
        def reader(key: str):
            for i in range(100):
                cache.get(f"{key}_{i}")
        
        threads = [
            threading.Thread(target=writer, args=(f"w{i}",))
            for i in range(5)
        ] + [
            threading.Thread(target=reader, args=(f"r{i}",))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should not raise any exceptions


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — CircuitBreaker
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""
    
    def test_initial_state_closed(self, circuit):
        """Test initial state is CLOSED."""
        assert circuit.state == CircuitState.CLOSED
        assert circuit.allow_request() is True
    
    def test_opens_after_threshold(self, circuit):
        """Test circuit opens after threshold failures."""
        for _ in range(3):
            circuit.record_failure()
        assert circuit.state == CircuitState.OPEN
        assert circuit.allow_request() is False
    
    def test_success_resets_failure_count(self, circuit):
        """Test success resets failure count."""
        circuit.record_failure()
        circuit.record_failure()
        circuit.record_success()
        circuit.record_failure()
        circuit.record_failure()
        # Should not be open (2 failures, not 3)
        assert circuit.state == CircuitState.CLOSED
    
    def test_half_open_after_timeout(self):
        """Test transition to HALF_OPEN after timeout."""
        circuit = CircuitBreaker(name="test", threshold=2, timeout=0.1)
        circuit.record_failure()
        circuit.record_failure()
        assert circuit.state == CircuitState.OPEN
        
        time.sleep(0.15)
        assert circuit.state == CircuitState.HALF_OPEN
        assert circuit.allow_request() is True
    
    def test_closes_on_success_from_half_open(self):
        """Test circuit closes on success from HALF_OPEN."""
        circuit = CircuitBreaker(name="test", threshold=2, timeout=0.1)
        circuit.record_failure()
        circuit.record_failure()
        time.sleep(0.15)
        
        assert circuit.state == CircuitState.HALF_OPEN
        circuit.record_success()
        assert circuit.state == CircuitState.CLOSED


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — EventBus
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventBus:
    """Tests for event bus implementation."""
    
    def test_subscribe_and_publish(self, event_bus):
        """Test basic subscribe and publish."""
        received = []
        
        def handler(event: Event):
            received.append(event)
        
        event_bus.subscribe(EventType.QUERY_START, handler)
        event_bus.publish(Event(type=EventType.QUERY_START, source="test"))
        
        assert len(received) == 1
        assert received[0].type == EventType.QUERY_START
    
    def test_unsubscribe(self, event_bus):
        """Test unsubscribe removes handler."""
        received = []
        
        def handler(event: Event):
            received.append(event)
        
        unsubscribe = event_bus.subscribe(EventType.QUERY_START, handler)
        event_bus.publish(Event(type=EventType.QUERY_START, source="test"))
        assert len(received) == 1
        
        unsubscribe()
        event_bus.publish(Event(type=EventType.QUERY_START, source="test2"))
        assert len(received) == 1  # Still 1, not 2
    
    def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers receive events."""
        received1 = []
        received2 = []
        
        event_bus.subscribe(EventType.QUERY_START, lambda e: received1.append(e))
        event_bus.subscribe(EventType.QUERY_START, lambda e: received2.append(e))
        event_bus.publish(Event(type=EventType.QUERY_START, source="test"))
        
        assert len(received1) == 1
        assert len(received2) == 1
    
    def test_event_type_filtering(self, event_bus):
        """Test handlers only receive subscribed event types."""
        received = []
        
        event_bus.subscribe(EventType.QUERY_START, lambda e: received.append(e))
        event_bus.publish(Event(type=EventType.QUERY_COMPLETE, source="test"))
        
        assert len(received) == 0
    
    def test_event_history(self, event_bus):
        """Test event history tracking."""
        event_bus.publish(Event(type=EventType.QUERY_START, source="test1"))
        event_bus.publish(Event(type=EventType.QUERY_COMPLETE, source="test2"))
        
        history = event_bus.get_history()
        assert len(history) == 2
        
        filtered = event_bus.get_history(event_type=EventType.QUERY_START)
        assert len(filtered) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — ResourcePool
# ═══════════════════════════════════════════════════════════════════════════════

class TestResourcePool:
    """Tests for resource pool pattern."""
    
    def test_acquire_creates_resource(self):
        """Test resource is created on acquire."""
        factory_calls = [0]
        
        def factory():
            factory_calls[0] += 1
            return {"id": factory_calls[0]}
        
        pool = ResourcePool(factory=factory, max_size=3)
        
        with pool.acquire() as resource:
            assert resource == {"id": 1}
        
        assert factory_calls[0] == 1
    
    def test_resource_reuse(self):
        """Test resource is reused from pool."""
        factory_calls = [0]
        
        def factory():
            factory_calls[0] += 1
            return {"id": factory_calls[0]}
        
        pool = ResourcePool(factory=factory, max_size=3)
        
        with pool.acquire() as r1:
            first_id = r1["id"]
        
        with pool.acquire() as r2:
            # Should reuse the same resource
            assert r2["id"] == first_id
        
        assert factory_calls[0] == 1  # Only created once
    
    def test_validation(self):
        """Test validation of resources."""
        def factory():
            return {"valid": True}
        
        def validate(resource):
            return resource.get("valid", False)
        
        pool = ResourcePool(factory=factory, max_size=3, validate=validate)
        
        with pool.acquire() as resource:
            resource["valid"] = False  # Invalidate
        
        with pool.acquire() as resource:
            assert resource["valid"] is True  # New resource created


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — QueryResult
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryResult:
    """Tests for query result value object."""
    
    def test_immutability(self):
        """Test QueryResult is immutable."""
        result = QueryResult(
            query="test",
            results=tuple(),
            total_count=0,
            elapsed_ms=1.0,
            snr_score=0.5,
            source_engine="test"
        )
        with pytest.raises(AttributeError):
            result.query = "modified"
    
    def test_to_dict(self):
        """Test conversion to dict."""
        result = QueryResult(
            query="test query",
            results=({"id": 1}, {"id": 2}),
            total_count=2,
            elapsed_ms=10.5,
            snr_score=0.8,
            source_engine="TestEngine"
        )
        
        d = result.to_dict()
        assert d["query"] == "test query"
        assert len(d["results"]) == 2
        assert d["total_count"] == 2
        assert d["snr_score"] == 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — Retry Decorator
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetryDecorator:
    """Tests for retry decorator."""
    
    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = [0]
        
        @with_retry(attempts=3, delay=0.01)
        def success_func():
            call_count[0] += 1
            return "success"
        
        result = success_func()
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retries_on_failure(self):
        """Test retries on failure."""
        call_count = [0]
        
        @with_retry(attempts=3, delay=0.01)
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"
        
        result = fail_twice()
        assert result == "success"
        assert call_count[0] == 3
    
    def test_max_retries_exceeded(self):
        """Test exception raised after max retries."""
        call_count = [0]
        
        @with_retry(attempts=3, delay=0.01)
        def always_fail():
            call_count[0] += 1
            raise ValueError("always fail")
        
        with pytest.raises(ValueError):
            always_fail()
        
        assert call_count[0] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — QueryRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryRouter:
    """Tests for intelligent query routing."""
    
    def test_sacred_wisdom_routing(self):
        """Test queries route to SacredWisdom for religious terms."""
        router = QueryRouter()
        
        # Mock engines
        engines = {
            "SacredWisdom": Mock(),
            "FileIndex": Mock()
        }
        
        routes = router.route("quran verses about mercy", engines)
        
        # SacredWisdom should score highest
        assert routes[0][0] == "SacredWisdom"
        assert routes[0][1] > 0.5
    
    def test_file_index_routing(self):
        """Test queries route to FileIndex for file terms."""
        router = QueryRouter()
        
        engines = {
            "SacredWisdom": Mock(),
            "FileIndex": Mock()
        }
        
        routes = router.route("find python files", engines)
        
        # FileIndex should score highest
        assert routes[0][0] == "FileIndex"
        assert routes[0][1] > 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — ResultAggregator
# ═══════════════════════════════════════════════════════════════════════════════

class TestResultAggregator:
    """Tests for result aggregation."""
    
    def test_empty_results(self):
        """Test aggregation of empty results."""
        result = ResultAggregator.aggregate([], "test query")
        assert result.total_count == 0
        assert result.snr_score == 0
    
    def test_single_result(self):
        """Test aggregation of single result."""
        single = QueryResult(
            query="test",
            results=({"id": 1},),
            total_count=1,
            elapsed_ms=10.0,
            snr_score=0.8,
            source_engine="Engine1"
        )
        
        result = ResultAggregator.aggregate([single], "test")
        assert result.total_count == 1
        assert result.snr_score == 0.8
    
    def test_multiple_results_sorted_by_snr(self):
        """Test results are sorted by SNR score."""
        result1 = QueryResult(
            query="test",
            results=({"id": 1},),
            total_count=1,
            elapsed_ms=10.0,
            snr_score=0.5,
            source_engine="Engine1"
        )
        result2 = QueryResult(
            query="test",
            results=({"id": 2},),
            total_count=1,
            elapsed_ms=10.0,
            snr_score=0.9,
            source_engine="Engine2"
        )
        
        aggregated = ResultAggregator.aggregate([result1, result2], "test")
        
        # Higher SNR should come first
        assert aggregated.results[0]["id"] == 2
        assert aggregated.results[1]["id"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — BizraNexus
# ═══════════════════════════════════════════════════════════════════════════════

class TestBizraNexusIntegration:
    """Integration tests for the full orchestrator."""
    
    def test_context_manager(self, config):
        """Test context manager lifecycle."""
        with BizraNexus(config) as nexus:
            assert nexus._initialized
        # Should be shut down after exit
    
    def test_health_check(self, config):
        """Test health check returns status for all engines."""
        with BizraNexus(config) as nexus:
            health = nexus.health()
            assert "SacredWisdom" in health or "FileIndex" in health
    
    def test_stats(self, config):
        """Test stats return comprehensive information."""
        with BizraNexus(config) as nexus:
            stats = nexus.stats()
            assert "engines" in stats
            assert "initialized" in stats
            assert "health" in stats
    
    def test_event_subscription(self, config):
        """Test event subscription works."""
        received = []
        
        with BizraNexus(config) as nexus:
            nexus.subscribe(EventType.QUERY_START, lambda e: received.append(e))
            # Trigger a query (may fail, but should still emit event)
            try:
                nexus.query("test query", engines=["FileIndex"])
            except Exception:
                pass
        
        # Should have received query start event
        assert len(received) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance benchmark tests."""
    
    def test_cache_performance(self):
        """Benchmark cache operations."""
        cache = TTLCache[str, str](max_size=10000, default_ttl=60.0)
        
        # Write benchmark
        start = time.perf_counter()
        for i in range(10000):
            cache.set(f"key_{i}", f"value_{i}")
        write_time = time.perf_counter() - start
        
        # Read benchmark
        start = time.perf_counter()
        for i in range(10000):
            cache.get(f"key_{i}")
        read_time = time.perf_counter() - start
        
        print(f"\n📊 Cache Performance:")
        print(f"   10K writes: {write_time*1000:.2f}ms ({10000/write_time:.0f} ops/sec)")
        print(f"   10K reads:  {read_time*1000:.2f}ms ({10000/read_time:.0f} ops/sec)")
        
        # Should complete in reasonable time
        assert write_time < 1.0
        assert read_time < 0.5
    
    def test_concurrent_cache_access(self):
        """Benchmark concurrent cache access."""
        cache = TTLCache[str, str](max_size=10000, default_ttl=60.0)
        
        def worker(worker_id: int):
            for i in range(1000):
                cache.set(f"w{worker_id}_key_{i}", f"value_{i}")
                cache.get(f"w{worker_id}_key_{i}")
        
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            for f in futures:
                f.result()
        elapsed = time.perf_counter() - start
        
        print(f"\n📊 Concurrent Cache (8 workers × 2K ops):")
        print(f"   Total time: {elapsed*1000:.2f}ms")
        print(f"   Throughput: {16000/elapsed:.0f} ops/sec")
        
        assert elapsed < 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStress:
    """Stress tests for resilience patterns."""
    
    def test_circuit_breaker_under_load(self):
        """Test circuit breaker behavior under load."""
        circuit = CircuitBreaker(name="stress_test", threshold=5, timeout=0.5)
        
        success_count = 0
        failure_count = 0
        rejected_count = 0
        
        for i in range(100):
            if not circuit.allow_request():
                rejected_count += 1
                continue
            
            # Simulate 50% failure rate
            if i % 2 == 0:
                circuit.record_failure()
                failure_count += 1
            else:
                circuit.record_success()
                success_count += 1
        
        print(f"\n📊 Circuit Breaker Stress Test:")
        print(f"   Successes: {success_count}")
        print(f"   Failures:  {failure_count}")
        print(f"   Rejected:  {rejected_count}")
        
        # Circuit should have rejected some requests
        assert rejected_count > 0
    
    def test_event_bus_high_throughput(self):
        """Test event bus under high throughput."""
        bus = EventBus()
        received_count = [0]
        
        def handler(event: Event):
            received_count[0] += 1
        
        bus.subscribe(EventType.QUERY_START, handler)
        
        start = time.perf_counter()
        for i in range(10000):
            bus.publish(Event(type=EventType.QUERY_START, source=f"test_{i}"))
        elapsed = time.perf_counter() - start
        
        print(f"\n📊 Event Bus Throughput:")
        print(f"   10K events: {elapsed*1000:.2f}ms")
        print(f"   Throughput: {10000/elapsed:.0f} events/sec")
        
        assert received_count[0] == 10000
        assert elapsed < 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_quick_tests():
    """Run quick validation tests."""
    print("=" * 70)
    print("   🧪 BIZRA NEXUS — Quick Test Suite")
    print("=" * 70)
    
    # Configuration tests
    print("\n📋 Testing NexusConfig...")
    config = NexusConfig()
    assert config.max_workers == 8
    assert config.cache_size == 10000
    print("   ✓ Default values correct")
    
    try:
        NexusConfig(max_workers=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("   ✓ Validation working")
    
    # Cache tests
    print("\n📋 Testing TTLCache...")
    cache = TTLCache[str, str](max_size=5, default_ttl=60.0)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    print("   ✓ Basic operations")
    
    cache.set("a", "1")
    cache.set("b", "2")
    cache.set("c", "3")
    cache.set("d", "4")
    cache.set("e", "5")
    cache.set("f", "6")  # Should evict
    assert cache.get("key1") is None or cache.stats["size"] <= 5
    print("   ✓ LRU eviction")
    
    # Circuit breaker tests
    print("\n📋 Testing CircuitBreaker...")
    circuit = CircuitBreaker(name="test", threshold=3, timeout=0.1)
    assert circuit.state == CircuitState.CLOSED
    circuit.record_failure()
    circuit.record_failure()
    circuit.record_failure()
    assert circuit.state == CircuitState.OPEN
    print("   ✓ Opens after threshold")
    
    time.sleep(0.15)
    assert circuit.state == CircuitState.HALF_OPEN
    print("   ✓ Half-opens after timeout")
    
    # Event bus tests
    print("\n📋 Testing EventBus...")
    bus = EventBus()
    received = []
    # Need to keep handler reference alive (weak ref in bus)
    handler = lambda e: received.append(e)
    bus.subscribe(EventType.QUERY_START, handler)
    bus.publish(Event(type=EventType.QUERY_START, source="test"))
    assert len(received) == 1
    print("   ✓ Pub/sub working")
    
    # Query router tests
    print("\n📋 Testing QueryRouter...")
    router = QueryRouter()
    routes = router.route("quran mercy", {"SacredWisdom": Mock(), "FileIndex": Mock()})
    assert routes[0][0] == "SacredWisdom"
    print("   ✓ Intelligent routing")
    
    # Result aggregator tests
    print("\n📋 Testing ResultAggregator...")
    r1 = QueryResult(
        query="test", results=({"id": 1},), total_count=1,
        elapsed_ms=10.0, snr_score=0.5, source_engine="E1"
    )
    r2 = QueryResult(
        query="test", results=({"id": 2},), total_count=1,
        elapsed_ms=10.0, snr_score=0.9, source_engine="E2"
    )
    agg = ResultAggregator.aggregate([r1, r2], "test")
    assert agg.results[0]["id"] == 2  # Higher SNR first
    print("   ✓ SNR-based sorting")
    
    print("\n" + "=" * 70)
    print("   ✅ ALL QUICK TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        run_quick_tests()
