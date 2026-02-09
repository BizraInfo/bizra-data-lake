"""
BIZRA Inference Gateway — Request Batching
═══════════════════════════════════════════════════════════════════════════════

Batches inference requests for efficient GPU utilisation (P0-P1 optimization).

Standing on Giants:
- Amdahl (1967): Parallelization theory
- NVIDIA (2020): GPU batch inference best practices
- Google (2017): Tensor batching in TensorFlow Serving

Extracted from gateway.py for modularity; re-exported by gateway.py
for backward compatibility.

Created: 2026-02-09 | Refactor split from gateway.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from ._types import BatchingMetrics


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
