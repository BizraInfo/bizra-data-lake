# Phase 66.03: Performance Hardening

## Problem Statement

Three HIGH-impact performance defects found by the integration review:

1. **N+1 SQLite commits** in living_memory feedback loop (100x overhead)
2. **Blocking urllib in async** Ollama backend (3s event loop stall)
3. **Zero caching** on pure SNR scoring functions (redundant entropy calc)

> Axiom: "Economics = thermodynamics." Wasted CPU cycles are wasted energy.
> A sovereignty system that wastes compute cannot reach economic equilibrium.

## Pseudocode

### Fix 1: N+1 → Batch SQLite (core/living_memory/core.py)

```
# BEFORE (lines 675-702):
# async def apply_execution_feedback(self, entry_ids, feedback):
#     for entry_id in entry_ids:
#         entry = self._memories.get(entry_id)
#         if entry:
#             entry.update(feedback)
#             await self._save_entry(entry)    # ← commits per entry

# AFTER:
async def apply_execution_feedback(self, entry_ids, feedback):
    """Apply feedback to entries using batch commit."""
    entries_to_save = []

    FOR entry_id IN entry_ids:
        entry = self._memories.get(entry_id)
        IF entry IS NOT None:
            entry.update(feedback)
            entries_to_save.append(entry)

    IF entries_to_save:
        # Single transaction for all entries
        # save_batch() already exists at persistence.py:179
        await self._store.save_batch(entries_to_save)

# VERIFY: persistence.py:179 save_batch() uses single transaction
# def save_batch(self, entries):
#     with self._conn:  # ← single transaction context
#         for entry in entries:
#             self._conn.execute("INSERT OR REPLACE ...", ...)
#     self._conn.commit()  # ← ONE commit for N entries
```

**Key insight**: `save_batch()` already exists in `persistence.py` line 179
and wraps multiple inserts in a single transaction. The N+1 pattern exists
only because `apply_execution_feedback` was written before `save_batch`.

### Fix 2: Async urllib → run_in_executor (core/inference/_backends.py)

```
# The pattern already exists in the same file at line 499-506:
# async def _generate_internal(self, ...):
#     loop = asyncio.get_running_loop()
#     raw = await loop.run_in_executor(None, lambda: urllib.request.urlopen(...))

# APPLY SAME PATTERN to initialize() and health_check():

# BEFORE (lines 391-398):
# async def initialize(self):
#     req = urllib.request.Request(url, ...)
#     resp = urllib.request.urlopen(req, timeout=3)  # ← BLOCKS event loop

# AFTER:
async def initialize(self) -> bool:
    """Initialize Ollama backend — non-blocking."""
    TRY:
        loop = asyncio.get_running_loop()
        url = f"{self._base_url}/api/tags"
        req = urllib.request.Request(url, headers=self._headers)

        resp_bytes = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=3).read()
        )
        data = json.loads(resp_bytes)
        self._available_models = [m["name"] for m in data.get("models", [])]
        RETURN True
    EXCEPT Exception AS exc:
        logger.warning("Ollama init failed: %s", exc)
        RETURN False


# SAME for health_check() (lines 537-543):
async def health_check(self) -> dict:
    """Non-blocking Ollama health check."""
    TRY:
        loop = asyncio.get_running_loop()
        url = f"{self._base_url}/api/tags"
        req = urllib.request.Request(url, headers=self._headers)

        resp_bytes = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=3).read()
        )
        data = json.loads(resp_bytes)
        RETURN {"healthy": True, "models": len(data.get("models", []))}
    EXCEPT Exception AS exc:
        RETURN {"healthy": False, "error": str(exc)}


# ALSO apply to duplicate at core/inference/backends/ollama.py:39-46,110-113
# Same pattern — wrap urllib calls in run_in_executor
```

### Fix 3: SNR LRU Cache (core/iaas/snr_v2.py)

```
# SNR scoring functions are PURE: same input → same output
# They compute Shannon entropy, Renyi entropy, vocabulary richness
# These are deterministic mathematical operations

import functools

# Identify the pure scoring functions:
# - compute_snr(text: str) → float
# - _shannon_entropy(text: str) → float
# - _vocabulary_richness(tokens: list[str]) → float
# - _information_density(text: str) → float

# Add LRU cache to the top-level entry point:

@functools.lru_cache(maxsize=512)
def compute_snr(text: str) -> float:
    """Compute Signal-to-Noise Ratio for text content.

    Cached: identical inputs return memoized result.
    Cache holds 512 most recent unique inputs.
    A 512-entry cache at ~100 bytes/key uses ~50KB — negligible.
    """
    signal = _compute_signal_score(text)
    noise = _compute_noise_score(text)
    IF noise == 0:
        RETURN 1.0
    RETURN signal / (signal + noise)


# NOTE: Do NOT cache internal helpers individually —
# caching the top-level function is sufficient and simpler.
# Internal helpers are only called from compute_snr.

# For the adapter layer (snr_v2_adapter.py):
# The adapter wraps compute_snr with additional context.
# Cache at the adapter level too if context is hashable:

@functools.lru_cache(maxsize=256)
def score_content(content: str, content_type: str = "text") -> float:
    """Cached SNR scoring with content-type awareness."""
    base_snr = compute_snr(content)
    type_multiplier = TYPE_MULTIPLIERS.get(content_type, 1.0)
    RETURN min(1.0, base_snr * type_multiplier)
```

**Cache invalidation**: Not needed. SNR is a pure function of text content.
The same text always produces the same score. LRU eviction handles memory.

**Thread safety**: `@lru_cache` is thread-safe in CPython (GIL protects
the cache dict). For async contexts, the cache is shared across coroutines
which is correct — same text = same score regardless of caller.

## Parallel Execution Opportunity (Bonus)

```
# core/sovereign/mission.py:489-528
# _execute_channels() runs subtasks serially but they are independent

# BEFORE:
# for subtask in plan.subtasks:
#     result = await self._execute_single(subtask)
#     results.append(result)

# AFTER:
async def _execute_channels(self, plan):
    """Execute independent subtasks in parallel."""
    tasks = [self._execute_single(st) for st in plan.subtasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes from failures
    final = []
    FOR i, result IN enumerate(results):
        IF isinstance(result, Exception):
            logger.warning("Subtask %s failed: %s", plan.subtasks[i].id, result)
            final.append(SubTaskResult(success=False, error=str(result)))
        ELSE:
            final.append(result)
    RETURN final
```

## Invariants

```
ASSERT: living_memory apply_execution_feedback calls save_batch, not save_entry
ASSERT: OllamaBackend.initialize() uses run_in_executor
ASSERT: OllamaBackend.health_check() uses run_in_executor
ASSERT: compute_snr has __wrapped__ attribute (proves @lru_cache applied)
```

## TDD Anchors

```python
# test_performance_hardening.py

import asyncio
import time


def test_snr_cache_hit():
    """Second call to compute_snr with same input is near-instant."""
    from core.iaas.snr_v2 import compute_snr

    text = "The quick brown fox jumps over the lazy dog." * 10
    compute_snr.cache_clear()  # Reset

    t0 = time.perf_counter()
    result1 = compute_snr(text)
    cold_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    result2 = compute_snr(text)
    hot_ms = (time.perf_counter() - t0) * 1000

    assert result1 == result2
    assert hot_ms < cold_ms * 0.1  # Cache hit should be >10x faster
    assert compute_snr.cache_info().hits >= 1


async def test_ollama_init_does_not_block_loop():
    """OllamaBackend.initialize() does not block the event loop."""
    from core.inference._backends import OllamaBackend

    backend = OllamaBackend(base_url="http://127.0.0.1:99999")  # unreachable

    # Run init with a concurrent timer
    async def timer():
        t0 = time.perf_counter()
        await asyncio.sleep(0.01)  # 10ms sleep
        return (time.perf_counter() - t0) * 1000

    init_task = asyncio.create_task(backend.initialize())
    timer_task = asyncio.create_task(timer())

    await asyncio.gather(init_task, timer_task)

    timer_ms = timer_task.result()
    # If init blocks the loop, timer will take 3000ms+ (urllib timeout)
    # If init uses executor, timer completes in ~10-20ms
    assert timer_ms < 100, f"Event loop blocked for {timer_ms:.0f}ms"


async def test_living_memory_batch_commit(tmp_path):
    """apply_execution_feedback uses batch commit, not per-entry."""
    # Verify save_batch is called instead of N × save_entry
    # Implementation: mock the store and assert call pattern
    pass  # Placeholder — see phase_66_04_tdd_anchors.md for full mock
```

## Estimated Impact

- **Fix 1 (N+1)**: 100 entries × 1 commit/entry → 1 commit total = 100x fewer fsyncs
- **Fix 2 (async)**: 3s event loop stall → 0ms stall (executor thread absorbs wait)
- **Fix 3 (cache)**: Repeated SNR calls from 2-5ms → <0.01ms per cache hit
- **Bonus (parallel)**: 4 sequential subtasks × 500ms → 500ms total (4x speedup)
