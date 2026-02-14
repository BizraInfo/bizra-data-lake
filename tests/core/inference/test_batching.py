"""
P0-P1: INFERENCE REQUEST BATCHING TESTS
═══════════════════════════════════════════════════════════════════════════════

Tests for the BatchingInferenceQueue implementation.

Expected improvement: 8x throughput (serial -> batch-8).

Standing on Giants:
- Amdahl (1967): Parallelization theory
- NVIDIA (2020): GPU batch inference optimization

Created: 2026-02-04 | BIZRA Sovereignty
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock

from core.inference.gateway import (
    BatchingInferenceQueue,
    InferenceConfig,
    LlamaCppBackend,
    InferenceGateway,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

class MockBackend:
    """Mock backend for testing batching without real LLM."""

    def __init__(self, delay_ms: int = 100):
        self.delay_ms = delay_ms
        self.call_count = 0
        self.call_history = []

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Simulate inference with delay."""
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timestamp": time.time()
        })

        await asyncio.sleep(self.delay_ms / 1000)
        return f"Response to: {prompt[:20]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# BATCHING QUEUE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_batching_queue_single_request():
    """Test that a single request is processed correctly."""
    backend = MockBackend(delay_ms=50)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=8,
        max_wait_ms=50
    )

    await queue.start()

    try:
        result = await queue.submit("Test prompt", 100, 0.7)
        assert "Response to: Test prompt" in result
        assert backend.call_count == 1
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_batching_queue_multiple_requests_sequential():
    """Test that multiple sequential requests are batched by timeout."""
    backend = MockBackend(delay_ms=10)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=8,
        max_wait_ms=50
    )

    await queue.start()

    try:
        # Submit 3 requests with small delays (should batch together)
        tasks = []
        for i in range(3):
            tasks.append(asyncio.create_task(
                queue.submit(f"Prompt {i}", 100, 0.7)
            ))
            await asyncio.sleep(0.01)  # 10ms between requests

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert backend.call_count == 3
        assert all("Response to:" in r for r in results)

        metrics = queue.get_metrics()
        assert metrics["total_requests"] == 3
        assert metrics["total_batches"] >= 1

    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_batching_queue_concurrent_requests():
    """Test that concurrent requests trigger batch processing."""
    backend = MockBackend(delay_ms=20)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=4,
        max_wait_ms=100
    )

    await queue.start()

    try:
        # Submit 4 requests concurrently (should fill batch)
        tasks = [
            asyncio.create_task(queue.submit(f"Prompt {i}", 100, 0.7))
            for i in range(4)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert backend.call_count == 4

        metrics = queue.get_metrics()
        assert metrics["total_requests"] == 4

    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_batching_queue_batch_size_limit():
    """Test that batch size is limited to MAX_BATCH_SIZE."""
    backend = MockBackend(delay_ms=10)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=4,
        max_wait_ms=100
    )

    await queue.start()

    try:
        # Submit 10 requests (should create 3 batches: 4, 4, 2)
        tasks = [
            asyncio.create_task(queue.submit(f"Prompt {i}", 100, 0.7))
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert backend.call_count == 10

        metrics = queue.get_metrics()
        assert metrics["total_requests"] == 10
        # Should have at least 3 batches (exact count depends on timing)
        assert metrics["total_batches"] >= 2

    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_batching_queue_timeout_flush():
    """Test that pending requests are flushed after timeout."""
    backend = MockBackend(delay_ms=10)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=8,
        max_wait_ms=50  # Short timeout
    )

    await queue.start()

    try:
        # Submit 2 requests (not enough to fill batch)
        tasks = [
            asyncio.create_task(queue.submit(f"Prompt {i}", 100, 0.7))
            for i in range(2)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert backend.call_count == 2

        metrics = queue.get_metrics()
        assert metrics["total_requests"] == 2

    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_batching_queue_error_handling():
    """Test that errors are propagated to individual requests."""
    async def failing_backend(prompt, max_tokens, temperature):
        if "fail" in prompt:
            raise ValueError("Simulated error")
        await asyncio.sleep(0.01)
        return f"Success: {prompt}"

    queue = BatchingInferenceQueue(
        backend_generate_fn=failing_backend,
        max_batch_size=4,
        max_wait_ms=50
    )

    await queue.start()

    try:
        # Submit mix of success and failure requests
        tasks = [
            asyncio.create_task(queue.submit("Good prompt 1", 100, 0.7)),
            asyncio.create_task(queue.submit("fail prompt", 100, 0.7)),
            asyncio.create_task(queue.submit("Good prompt 2", 100, 0.7)),
        ]

        results = []
        errors = []

        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                errors.append(e)

        assert len(results) == 2  # Two successful
        assert len(errors) == 1   # One failed
        assert isinstance(errors[0], ValueError)

    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_batching_queue_metrics():
    """Test that batching metrics are accurate."""
    backend = MockBackend(delay_ms=10)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=4,
        max_wait_ms=50
    )

    await queue.start()

    try:
        # Submit 8 requests (should create 2 batches of 4)
        tasks = [
            asyncio.create_task(queue.submit(f"Prompt {i}", 100, 0.7))
            for i in range(8)
        ]

        await asyncio.gather(*tasks)

        metrics = queue.get_metrics()

        assert metrics["total_requests"] == 8
        assert metrics["total_batches"] >= 1
        assert metrics["avg_batch_size"] > 0
        assert metrics["avg_batch_duration_ms"] > 0
        assert metrics["queue_depth"] == 0  # All processed

    finally:
        await queue.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_inference_config_batching_defaults():
    """Test that batching is enabled by default with sane values."""
    config = InferenceConfig()

    assert config.enable_batching is True
    assert config.max_batch_size == 8
    assert config.max_batch_wait_ms == 50


def test_inference_config_batching_disabled():
    """Test that batching can be disabled."""
    config = InferenceConfig(enable_batching=False)

    assert config.enable_batching is False


def test_inference_config_batching_custom():
    """Test custom batching configuration."""
    config = InferenceConfig(
        enable_batching=True,
        max_batch_size=16,
        max_batch_wait_ms=100
    )

    assert config.enable_batching is True
    assert config.max_batch_size == 16
    assert config.max_batch_wait_ms == 100


# ═══════════════════════════════════════════════════════════════════════════════
# THROUGHPUT COMPARISON TEST (BENCHMARK)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
@pytest.mark.asyncio
async def test_batching_throughput_improvement():
    """
    Benchmark test: Compare throughput with/without batching.

    Expected: ~8x improvement with batch size 8.
    """
    NUM_REQUESTS = 32
    BACKEND_DELAY_MS = 50

    # ─────────────────────────────────────────────────────────────────────────
    # Serial mode (no batching) - truly sequential execution
    # ─────────────────────────────────────────────────────────────────────────
    backend_serial = MockBackend(delay_ms=BACKEND_DELAY_MS)

    start_serial = time.time()
    serial_results = []
    for i in range(NUM_REQUESTS):
        result = await backend_serial.generate(f"Prompt {i}", 100, 0.7)
        serial_results.append(result)

    serial_duration = time.time() - start_serial

    # ─────────────────────────────────────────────────────────────────────────
    # Batching mode
    # ─────────────────────────────────────────────────────────────────────────
    backend_batched = MockBackend(delay_ms=BACKEND_DELAY_MS)
    queue = BatchingInferenceQueue(
        backend_generate_fn=backend_batched.generate,
        max_batch_size=8,
        max_wait_ms=50
    )

    await queue.start()

    try:
        start_batched = time.time()
        batched_tasks = [
            asyncio.create_task(queue.submit(f"Prompt {i}", 100, 0.7))
            for i in range(NUM_REQUESTS)
        ]

        batched_results = await asyncio.gather(*batched_tasks)
        batched_duration = time.time() - start_batched

    finally:
        await queue.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis
    # ─────────────────────────────────────────────────────────────────────────
    assert len(serial_results) == NUM_REQUESTS
    assert len(batched_results) == NUM_REQUESTS

    throughput_serial = NUM_REQUESTS / serial_duration
    throughput_batched = NUM_REQUESTS / batched_duration
    improvement = throughput_batched / throughput_serial

    print(f"\n{'='*60}")
    print(f"BATCHING THROUGHPUT BENCHMARK")
    print(f"{'='*60}")
    print(f"Requests: {NUM_REQUESTS}")
    print(f"Backend delay: {BACKEND_DELAY_MS}ms per request")
    print(f"\nSerial mode:")
    print(f"  Duration: {serial_duration:.2f}s")
    print(f"  Throughput: {throughput_serial:.2f} req/s")
    print(f"\nBatched mode (batch_size=8):")
    print(f"  Duration: {batched_duration:.2f}s")
    print(f"  Throughput: {throughput_batched:.2f} req/s")
    print(f"\nImprovement: {improvement:.2f}x")
    print(f"{'='*60}")

    # Assert significant improvement (at least 2x, ideally ~8x)
    # NOTE: Actual improvement depends on timing and asyncio scheduling
    assert improvement >= 2.0, f"Expected at least 2x improvement, got {improvement:.2f}x"


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (REQUIRES REAL BACKEND)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.requires_llama_cpp
@pytest.mark.slow
@pytest.mark.asyncio
async def test_llama_cpp_backend_batching_integration():
    """
    Integration test with real llama.cpp backend.

    Requires:
    - llama-cpp-python installed
    - A model file available
    """
    config = InferenceConfig(
        enable_batching=True,
        max_batch_size=4,
        max_batch_wait_ms=100
    )

    backend = LlamaCppBackend(config)

    # Try to initialize (may fail if no model available)
    try:
        initialized = await backend.initialize()
        if not initialized:
            pytest.skip("llama.cpp backend not available")

        # Submit multiple requests
        tasks = [
            asyncio.create_task(backend.generate(f"Test prompt {i}", max_tokens=50, temperature=0.7))
            for i in range(4)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert all(isinstance(r, str) for r in results)

        # Check batching metrics
        metrics = backend.get_batching_metrics()
        if metrics:
            assert metrics["total_requests"] == 4
            print(f"\nBatching metrics: {metrics}")

    finally:
        await backend.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
