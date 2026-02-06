#!/usr/bin/env python3
"""
P0-P1: BATCHING THROUGHPUT BENCHMARK
═══════════════════════════════════════════════════════════════════════════════

Benchmarks inference throughput with and without batching.

Usage:
    python tools/benchmark_batching.py
    python tools/benchmark_batching.py --requests 64 --batch-size 16

Expected improvement: 8x throughput (serial -> batch-8).

Standing on Giants:
- Amdahl (1967): Parallelization theory
- NVIDIA (2020): GPU batch inference optimization

Created: 2026-02-04 | BIZRA Sovereignty
"""

import argparse
import asyncio
import time
from typing import List


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK BACKEND (FOR TESTING WITHOUT REAL LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class MockInferenceBackend:
    """Simulates inference with configurable delay."""

    def __init__(self, delay_ms: int = 50):
        self.delay_ms = delay_ms
        self.call_count = 0

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Simulate inference with delay."""
        self.call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)
        return f"Response to: {prompt[:30]}..."


# ═══════════════════════════════════════════════════════════════════════════════
# SERIAL MODE (NO BATCHING)
# ═══════════════════════════════════════════════════════════════════════════════

async def run_serial_benchmark(
    num_requests: int,
    backend_delay_ms: int = 50
) -> float:
    """
    Run benchmark with serial processing (no batching).

    Returns:
        Duration in seconds
    """
    backend = MockInferenceBackend(delay_ms=backend_delay_ms)
    lock = asyncio.Lock()

    async def serial_request(prompt: str) -> str:
        async with lock:
            return await backend.generate(prompt, max_tokens=100, temperature=0.7)

    print(f"\n{'─'*60}")
    print("SERIAL MODE (No Batching)")
    print(f"{'─'*60}")

    start_time = time.time()

    # Create all tasks
    tasks = [
        asyncio.create_task(serial_request(f"Prompt {i}"))
        for i in range(num_requests)
    ]

    # Wait for completion
    results = await asyncio.gather(*tasks)

    duration = time.time() - start_time
    throughput = num_requests / duration

    print(f"Requests: {num_requests}")
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} req/s")
    print(f"Avg latency: {duration / num_requests * 1000:.2f}ms")

    return duration


# ═══════════════════════════════════════════════════════════════════════════════
# BATCHING MODE
# ═══════════════════════════════════════════════════════════════════════════════

async def run_batching_benchmark(
    num_requests: int,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 50,
    backend_delay_ms: int = 50
) -> float:
    """
    Run benchmark with batching enabled.

    Returns:
        Duration in seconds
    """
    from core.inference.gateway import BatchingInferenceQueue

    backend = MockInferenceBackend(delay_ms=backend_delay_ms)

    queue = BatchingInferenceQueue(
        backend_generate_fn=backend.generate,
        max_batch_size=max_batch_size,
        max_wait_ms=max_batch_wait_ms
    )

    await queue.start()

    print(f"\n{'─'*60}")
    print(f"BATCHING MODE (max_batch={max_batch_size}, max_wait={max_batch_wait_ms}ms)")
    print(f"{'─'*60}")

    try:
        start_time = time.time()

        # Create all tasks
        tasks = [
            asyncio.create_task(queue.submit(f"Prompt {i}", 100, 0.7))
            for i in range(num_requests)
        ]

        # Wait for completion
        results = await asyncio.gather(*tasks)

        duration = time.time() - start_time
        throughput = num_requests / duration

        print(f"Requests: {num_requests}")
        print(f"Duration: {duration:.2f}s")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Avg latency: {duration / num_requests * 1000:.2f}ms")

        # Print batching metrics
        metrics = queue.get_metrics()
        print(f"\nBatching Metrics:")
        print(f"  Total batches: {metrics['total_batches']}")
        print(f"  Avg batch size: {metrics['avg_batch_size']:.2f}")
        print(f"  Avg batch duration: {metrics['avg_batch_duration_ms']:.2f}ms")

        return duration

    finally:
        await queue.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

async def compare_throughput(
    num_requests: int = 32,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 50,
    backend_delay_ms: int = 50
):
    """
    Compare serial vs batching throughput.
    """
    print(f"\n{'═'*60}")
    print("BIZRA INFERENCE BATCHING BENCHMARK")
    print(f"{'═'*60}")
    print(f"Configuration:")
    print(f"  Total requests: {num_requests}")
    print(f"  Backend delay: {backend_delay_ms}ms per request")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Max batch wait: {max_batch_wait_ms}ms")

    # Run serial benchmark
    serial_duration = await run_serial_benchmark(num_requests, backend_delay_ms)

    # Run batching benchmark
    batching_duration = await run_batching_benchmark(
        num_requests, max_batch_size, max_batch_wait_ms, backend_delay_ms
    )

    # Compare results
    print(f"\n{'═'*60}")
    print("COMPARISON")
    print(f"{'═'*60}")

    throughput_serial = num_requests / serial_duration
    throughput_batching = num_requests / batching_duration
    improvement = throughput_batching / throughput_serial
    speedup = serial_duration / batching_duration

    print(f"Serial throughput:   {throughput_serial:.2f} req/s")
    print(f"Batching throughput: {throughput_batching:.2f} req/s")
    print(f"\nImprovement:         {improvement:.2f}x")
    print(f"Speedup:             {speedup:.2f}x")

    # Expected improvement analysis
    theoretical_max = max_batch_size
    efficiency = improvement / theoretical_max * 100

    print(f"\nTheoretical max:     {theoretical_max}x")
    print(f"Batching efficiency: {efficiency:.1f}%")

    if improvement >= theoretical_max * 0.8:
        status = "✓ EXCELLENT"
    elif improvement >= theoretical_max * 0.5:
        status = "✓ GOOD"
    elif improvement >= 2.0:
        status = "✓ ACCEPTABLE"
    else:
        status = "⚠ POOR"

    print(f"Status:              {status}")
    print(f"{'═'*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference batching throughput"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=32,
        help="Number of requests to benchmark (default: 32)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum batch size (default: 8)"
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=50,
        help="Maximum batch wait time in ms (default: 50)"
    )
    parser.add_argument(
        "--backend-delay-ms",
        type=int,
        default=50,
        help="Simulated backend delay per request in ms (default: 50)"
    )

    args = parser.parse_args()

    asyncio.run(compare_throughput(
        num_requests=args.requests,
        max_batch_size=args.batch_size,
        max_batch_wait_ms=args.batch_wait_ms,
        backend_delay_ms=args.backend_delay_ms
    ))


if __name__ == "__main__":
    main()
