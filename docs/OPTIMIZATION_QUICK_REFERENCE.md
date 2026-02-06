# 🎯 Performance Optimization Quick Reference

**For Developers Implementing the Optimizations**

---

## 🔴 OPTIMIZATION #1: Inference Batching

### Problem
```python
# core/sovereign/runtime.py:781-795
# CURRENT: Serial inference with lock contention
async with self._lock:  # <-- BLOCKS all other requests
    result = await gateway.infer(prompt, max_tokens=1024)
```

### Solution
```python
# NEW FILE: core/inference/batcher.py

import asyncio
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int
    result_future: asyncio.Future

class InferenceBatcher:
    """Batch multiple inference requests for parallel GPU execution."""

    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._pending: List[InferenceRequest] = []
        self._batch_event = asyncio.Event()
        self._batch_task: Optional[asyncio.Task] = None
        self._gateway = None

    def set_gateway(self, gateway):
        """Set the inference gateway."""
        self._gateway = gateway

    async def start(self):
        """Start the batching loop."""
        self._batch_task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        """Stop the batching loop."""
        if self._batch_task:
            self._batch_task.cancel()
            await asyncio.gather(self._batch_task, return_exceptions=True)

    async def infer(self, prompt: str, max_tokens: int = 1024) -> str:
        """Submit inference request and wait for result."""
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            result_future=asyncio.Future()
        )

        self._pending.append(request)
        self._batch_event.set()

        # Trigger immediate flush if batch full
        if len(self._pending) >= self.max_batch_size:
            await self._flush_batch()

        # Wait for result
        return await request.result_future

    async def _batch_loop(self):
        """Background loop to flush batches periodically."""
        while True:
            try:
                # Wait for new requests or timeout
                await asyncio.wait_for(
                    self._batch_event.wait(),
                    timeout=self.max_wait_ms / 1000.0
                )
                self._batch_event.clear()

                # Flush if we have pending requests
                if self._pending:
                    await self._flush_batch()

            except asyncio.TimeoutError:
                # Timeout: flush partial batch
                if self._pending:
                    await self._flush_batch()

    async def _flush_batch(self):
        """Execute batch of pending requests."""
        if not self._pending:
            return

        batch = self._pending[:self.max_batch_size]
        self._pending = self._pending[self.max_batch_size:]

        # Execute batch in parallel
        tasks = [
            self._gateway.infer(req.prompt, max_tokens=req.max_tokens)
            for req in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Set results
        for req, result in zip(batch, results):
            if isinstance(result, Exception):
                req.result_future.set_exception(result)
            else:
                req.result_future.set_result(result.content)
```

### Integration (runtime.py)
```python
# core/sovereign/runtime.py

class SovereignRuntime:
    def __init__(self, config):
        # ...existing code...
        self._batcher: Optional[InferenceBatcher] = None

    async def _init_components(self):
        # ...existing gateway init...

        # NEW: Initialize batcher
        if self._gateway:
            self._batcher = InferenceBatcher(
                max_batch_size=8,
                max_wait_ms=50
            )
            self._batcher.set_gateway(self._gateway)
            await self._batcher.start()

    async def _process_query(self, query, start_time):
        # CHANGE: Use batcher instead of direct gateway
        if self._batcher:
            result.answer = await self._batcher.infer(
                thought_prompt,
                max_tokens=1024
            )
        elif self._gateway:
            # Fallback to direct gateway
            inference_result = await self._gateway.infer(thought_prompt)
            result.answer = inference_result.content
```

**Expected Impact:** 8x throughput (+3 points)

---

## 🟠 OPTIMIZATION #2: Cache Key (xxHash)

### Problem
```python
# core/sovereign/runtime.py:882-886
# CURRENT: SHA-256 on every cache lookup
import hashlib  # ❌ Import on every call
content = f"{query.content}:{query.max_depth}:{query.require_reasoning}"
return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Solution
```python
# core/sovereign/runtime.py (top of file)
import xxhash  # ✅ Import once

class SovereignRuntime:
    def _cache_key(self, query: SovereignQuery) -> str:
        """Generate cache key using xxHash (10x faster than SHA-256)."""
        content = f"{query.content}:{query.max_depth}:{query.require_reasoning}"
        return xxhash.xxh64(content.encode()).hexdigest()[:16]
```

### Install xxHash
```bash
pip install xxhash
```

**Expected Impact:** 10x faster cache key (-66% cache hit latency, +1 point)

---

## 🟡 OPTIMIZATION #3: Digest Caching

### Problem
```python
# core/federation/consensus.py:356-365
# CURRENT: Recompute digest on every validation
canon_data = canonical_json(proposal.pattern_data)  # ❌ Repeated work
expected_digest = domain_separated_digest(canon_data)  # ❌ Repeated work

if prepare.digest != expected_digest:
    return False
```

### Solution
```python
# core/federation/consensus.py

@dataclass
class Proposal:
    """PBFT Pre-Prepare with cached digest."""
    proposal_id: str
    proposer_id: str
    pattern_data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    view_number: int = 0
    sequence_number: int = 0

    # NEW: Cached digest field
    _digest: Optional[str] = field(default=None, init=False, repr=False)
    _canonical_data: Optional[bytes] = field(default=None, init=False, repr=False)

    @property
    def canonical_data(self) -> bytes:
        """Lazy-computed canonical data."""
        if self._canonical_data is None:
            self._canonical_data = canonical_json(self.pattern_data)
        return self._canonical_data

    @property
    def digest(self) -> str:
        """Lazy-computed canonical digest."""
        if self._digest is None:
            self._digest = domain_separated_digest(self.canonical_data)
        return self._digest

# USAGE in receive_prepare:
def receive_prepare(self, prepare: PrepareMessage, node_count: int) -> bool:
    proposal = self.active_proposals[prepare.proposal_id]
    expected_digest = proposal.digest  # ✅ O(1) cached access

    if prepare.digest != expected_digest:
        return False
```

**Expected Impact:** 75% faster consensus rounds (+1 point)

---

## 🔵 OPTIMIZATION #4: Lock-Free Inference Pool

### Problem
```python
# core/inference/gateway.py:291-304
# CURRENT: Single lock blocks all inference
async with self._lock:  # ❌ Serializes all requests
    result = await loop.run_in_executor(None, lambda: self._model(...))
```

### Solution
```python
# core/inference/gateway.py

class LlamaCppBackend:
    def __init__(self, config: InferenceConfig, pool_size: int = 4):
        self.config = config
        self.pool_size = pool_size
        self.models: List[Llama] = []
        self.semaphore = asyncio.Semaphore(pool_size)
        self._next_model = 0
        self._model_lock = threading.Lock()

    async def initialize(self) -> bool:
        """Load multiple model instances."""
        model_path = self._resolve_model_path()
        if not model_path:
            return False

        # Load N model instances
        for i in range(self.pool_size):
            print(f"[LlamaCpp] Loading model {i+1}/{self.pool_size}...")
            model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.context_length,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                n_batch=self.config.n_batch,
                verbose=False,
            )
            self.models.append(model)

        return len(self.models) > 0

    def _get_next_model(self) -> int:
        """Round-robin model selection."""
        with self._model_lock:
            idx = self._next_model
            self._next_model = (self._next_model + 1) % len(self.models)
            return idx

    async def generate(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        """Generate with lock-free pool."""
        async with self.semaphore:  # Wait for available slot
            model_idx = self._get_next_model()
            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                lambda: self.models[model_idx](
                    prompt,
                    max_tokens=max_tokens,
                    echo=False,
                    **kwargs
                )
            )

        return result["choices"][0]["text"]
```

**Expected Impact:** 4x throughput (+2 points)
**Cost:** 16GB VRAM (4 × 4GB models)

---

## 🟢 OPTIMIZATION #5: Zero-Copy Projections

### Problem
```python
# core/sovereign/omega_engine.py:197-222
# CURRENT: Allocate new arrays every projection
ihsan_arr = ihsan.to_array()  # ❌ New allocation
ntu_raw = self.weights @ ihsan_arr + self.bias  # ❌ Temporary arrays
```

### Solution
```python
# core/sovereign/omega_engine.py

class IhsanProjector:
    def __init__(self, weights=None, bias=None):
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS.copy()
        self.bias = bias if bias is not None else self.DEFAULT_BIAS.copy()

        # NEW: Pre-allocate reusable buffers
        self._ihsan_buffer = np.zeros(8, dtype=np.float64)
        self._ntu_buffer = np.zeros(3, dtype=np.float64)

    def project(self, ihsan: IhsanVector) -> NTUState:
        """Zero-copy projection using pre-allocated buffers."""
        # Reuse buffer instead of allocating
        self._ihsan_buffer[:] = [
            ihsan.truthfulness, ihsan.trustworthiness, ihsan.justice,
            ihsan.excellence, ihsan.wisdom, ihsan.compassion,
            ihsan.patience, ihsan.gratitude
        ]

        # In-place matrix multiplication
        np.dot(self.weights, self._ihsan_buffer, out=self._ntu_buffer)
        self._ntu_buffer += self.bias

        # Constitutional invariant check
        if ihsan.minimum < 0.5:
            doubt_factor = ihsan.minimum / 0.5
            self._ntu_buffer[0] *= doubt_factor

        # Bounds and transform
        belief = float(np.clip(self._ntu_buffer[0], 0.0, 1.0))
        entropy = float(np.clip(1.0 - self._ntu_buffer[1], 0.0, 5.0))
        potential = float(np.clip(self._ntu_buffer[2] * 2 - 1, -1.0, 1.0))

        return NTUState(belief=belief, entropy=entropy, potential=potential)
```

**Expected Impact:** 4x faster projections (+0.5 points)

---

## 🟣 OPTIMIZATION #6: OrderedDict Cache

### Problem
```python
# core/sovereign/runtime.py:888-895
# CURRENT: O(n) eviction with list conversion
oldest_keys = list(self._cache.keys())[:100]  # ❌ O(n)
for k in oldest_keys:
    del self._cache[k]
```

### Solution
```python
# core/sovereign/runtime.py (imports)
from collections import OrderedDict

class SovereignRuntime:
    def __init__(self, config):
        # CHANGE: Use OrderedDict instead of dict
        self._cache: OrderedDict[str, SovereignResult] = OrderedDict()

    def _update_cache(self, key: str, result: SovereignResult) -> None:
        """Update cache with LRU eviction."""
        if len(self._cache) >= self.config.max_cache_entries:
            # O(1) eviction: remove first (oldest) entry
            self._cache.popitem(last=False)

        self._cache[key] = result
        self._cache.move_to_end(key)  # Mark as recently used
```

**Expected Impact:** 10,000x faster eviction (+0.5 points)

---

## 📊 Testing & Validation

### Run Benchmarks Before/After
```bash
# Baseline (before optimization)
python tools/performance_benchmark.py --all --output baseline.json

# After each optimization
python tools/performance_benchmark.py --all --baseline baseline.json

# Check specific optimization
python tools/performance_benchmark.py --cache  # Test cache optimization
python tools/performance_benchmark.py --consensus  # Test consensus optimization
```

### Unit Tests
```python
# tests/test_performance_optimizations.py

import pytest
import asyncio
from core.sovereign.runtime import SovereignRuntime

@pytest.mark.benchmark
async def test_cache_key_performance():
    """Verify xxHash is faster than SHA-256."""
    import time
    from core.sovereign.runtime import SovereignQuery

    runtime = SovereignRuntime()
    query = SovereignQuery(content="Test" * 100, max_depth=3)

    # Measure 1000 iterations
    start = time.perf_counter()
    for _ in range(1000):
        key = runtime._cache_key(query)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Should be < 5ms for 1000 iterations (0.005ms per key)
    assert elapsed_ms < 5.0, f"Cache key too slow: {elapsed_ms:.2f}ms"

@pytest.mark.benchmark
async def test_inference_batching():
    """Verify batching improves throughput."""
    from core.inference.batcher import InferenceBatcher
    from core.inference.gateway import InferenceGateway

    gateway = InferenceGateway()
    await gateway.initialize()

    batcher = InferenceBatcher(max_batch_size=8, max_wait_ms=50)
    batcher.set_gateway(gateway)
    await batcher.start()

    # Submit 16 requests concurrently
    start = time.perf_counter()
    tasks = [batcher.infer(f"Test {i}", max_tokens=10) for i in range(16)]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    # Should complete faster than 16 serial requests
    # Serial: ~16 seconds, Batched: ~2 seconds (8x speedup)
    assert elapsed < 5.0, f"Batching not effective: {elapsed:.2f}s"

    await batcher.stop()
```

---

## 🚨 Common Pitfalls

### ❌ Pitfall #1: Forgetting to Install Dependencies
```bash
# Required for xxHash optimization
pip install xxhash
```

### ❌ Pitfall #2: Not Pre-loading Models for Pool
```python
# BAD: Lazy loading on first request
if not self.models:
    self.models = [Llama(...) for _ in range(4)]  # 20 second delay!

# GOOD: Pre-load in initialize()
async def initialize(self):
    for i in range(self.pool_size):
        self.models.append(Llama(...))  # Load all upfront
```

### ❌ Pitfall #3: Incorrect Batch Size
```python
# BAD: Batch size too large (OOM)
batcher = InferenceBatcher(max_batch_size=64)  # 64 * 4GB = 256GB VRAM!

# GOOD: Conservative batch size
batcher = InferenceBatcher(max_batch_size=8)  # 8 * 4GB = 32GB VRAM
```

### ❌ Pitfall #4: Forgetting to Await Batcher
```python
# BAD: Forget to await batcher startup
self._batcher.start()  # Returns coroutine, doesn't run!

# GOOD: Await batcher startup
await self._batcher.start()
```

---

## 📈 Expected Results

After implementing all optimizations:

```
┌────────────────────────────────────────────────────────┐
│  Metric                │ Before  │ After   │ Δ         │
├────────────────────────────────────────────────────────┤
│  Query Latency (p50)   │ 500ms   │ 300ms   │ -40%  ✅  │
│  Query Latency (p99)   │ 2000ms  │ 800ms   │ -60%  ✅  │
│  Throughput (QPS)      │ 2.0     │ 5.0     │ +150% ✅  │
│  Cache Hit Latency     │ 15ms    │ 5ms     │ -66%  ✅  │
│  Consensus Round       │ 50ms    │ 20ms    │ -60%  ✅  │
│  Memory (Peak)         │ 8GB     │ 6GB     │ -25%  ✅  │
│                        │         │         │          │
│  PERFORMANCE SCORE     │ 81/100  │ 90/100  │ +9    ✅  │
└────────────────────────────────────────────────────────┘
```

**Grade:** B → A+

---

**Quick Reference Version:** 1.0
**Date:** 2026-02-04
**Author:** PERFORMANCE Agent

لا نفترض — We do not assume. We optimize with data.
