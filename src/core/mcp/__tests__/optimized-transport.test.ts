/**
 * Tests for OptimizedTransport — request batching, deduplication,
 * retry with exponential backoff, serialization, and flush.
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  OptimizedTransport,
  BatchedRequest,
  BatchedResponse,
} from '../optimized-transport';

// -- Helpers ---------------------------------------------------------------

/** Create a mock executeFn that returns success for all requests */
function successExecutor(
  delayMs: number = 0
): (requests: BatchedRequest[]) => Promise<BatchedResponse[]> {
  return async (requests) => {
    if (delayMs > 0) {
      await new Promise((r) => setTimeout(r, delayMs));
    }
    return requests.map((req) => ({
      requestId: req.id,
      result: `result-for-${req.toolName}`,
      error: null,
      durationMs: delayMs,
    }));
  };
}

/** Create an executor that fails the first N times, then succeeds */
function flakyExecutor(
  failCount: number
): (requests: BatchedRequest[]) => Promise<BatchedResponse[]> {
  let calls = 0;
  return async (requests) => {
    calls++;
    if (calls <= failCount) {
      throw new Error('Batch timeout');
    }
    return requests.map((req) => ({
      requestId: req.id,
      result: `ok-${calls}`,
      error: null,
      durationMs: 1,
    }));
  };
}

/** Create an executor that always fails with a non-retryable error */
function failExecutor(): (requests: BatchedRequest[]) => Promise<BatchedResponse[]> {
  return async () => {
    throw new Error('Connection refused');
  };
}

// -- Constructor -----------------------------------------------------------

describe('OptimizedTransport - Constructor', () => {
  it('applies default config', () => {
    const transport = new OptimizedTransport();
    const stats = transport.getStats();
    assert.equal(stats.totalRequests, 0);
    assert.equal(stats.totalBatched, 0);
    assert.equal(stats.totalDeduplicated, 0);
    assert.equal(stats.bytesSaved, 0);
    assert.equal(stats.totalRetries, 0);
  });

  it('merges partial config', () => {
    const transport = new OptimizedTransport({ maxBatchSize: 5 });
    // Should not throw — config merged
    const stats = transport.getStats();
    assert.equal(stats.totalRequests, 0);
  });
});

// -- Serialization ---------------------------------------------------------

describe('OptimizedTransport - Serialization', () => {
  it('compact serialization saves bytes', () => {
    const transport = new OptimizedTransport({ compactSerialization: true });
    const data = { key: 'value', nested: { a: 1, b: 2 } };
    const compact = transport.serialize(data);
    // Compact should have no whitespace
    assert.ok(!compact.includes('\n'));
    const stats = transport.getStats();
    assert.ok(stats.bytesSaved > 0);
  });

  it('non-compact serialization saves 0 bytes', () => {
    const transport = new OptimizedTransport({ compactSerialization: false });
    transport.serialize({ key: 'value' });
    assert.equal(transport.getStats().bytesSaved, 0);
  });
});

// -- Send + Batch ----------------------------------------------------------

describe('OptimizedTransport - Send', () => {
  it('returns result from successful request', async () => {
    const transport = new OptimizedTransport({ maxBatchSize: 1 });
    const result = await transport.send('test-tool', { x: 1 }, successExecutor());
    assert.equal(result, 'result-for-test-tool');
    assert.equal(transport.getStats().totalRequests, 1);
  });

  it('batches multiple requests up to maxBatchSize', async () => {
    const transport = new OptimizedTransport({ maxBatchSize: 3, maxBatchWaitMs: 10 });
    const exec = successExecutor();

    // Send 3 requests — should flush immediately at maxBatchSize
    const results = await Promise.all([
      transport.send('a', {}, exec),
      transport.send('b', {}, exec),
      transport.send('c', {}, exec),
    ]);

    assert.equal(results.length, 3);
    assert.equal(transport.getStats().totalRequests, 3);
    assert.ok(transport.getStats().totalBatched >= 1);
  });
});

// -- Deduplication ---------------------------------------------------------

describe('OptimizedTransport - Deduplication', () => {
  it('deduplicates identical in-flight requests', async () => {
    const transport = new OptimizedTransport({
      maxBatchSize: 1,
      deduplicateRequests: true,
    });
    const exec = successExecutor(50);

    // Send two identical requests simultaneously
    const [r1, r2] = await Promise.all([
      transport.send('same-tool', { key: 'same' }, exec),
      transport.send('same-tool', { key: 'same' }, exec),
    ]);

    assert.equal(r1, r2);
    assert.equal(transport.getStats().totalDeduplicated, 1);
  });

  it('does not deduplicate when disabled', async () => {
    const transport = new OptimizedTransport({
      maxBatchSize: 1,
      deduplicateRequests: false,
    });
    const exec = successExecutor();

    await Promise.all([
      transport.send('same-tool', { key: 'same' }, exec),
      transport.send('same-tool', { key: 'same' }, exec),
    ]);

    assert.equal(transport.getStats().totalDeduplicated, 0);
    assert.equal(transport.getStats().totalRequests, 2);
  });

  it('does not deduplicate different args', async () => {
    const transport = new OptimizedTransport({
      maxBatchSize: 1,
      deduplicateRequests: true,
    });
    const exec = successExecutor();

    await Promise.all([
      transport.send('tool', { a: 1 }, exec),
      transport.send('tool', { a: 2 }, exec),
    ]);

    assert.equal(transport.getStats().totalDeduplicated, 0);
  });
});

// -- Retry -----------------------------------------------------------------

describe('OptimizedTransport - Retry', () => {
  it('retries on timeout and succeeds', async () => {
    const transport = new OptimizedTransport({
      maxBatchSize: 1,
      maxRetries: 2,
      retryBaseDelayMs: 10,
    });

    // Fails first time (timeout), succeeds second
    const exec = flakyExecutor(1);
    const result = await transport.send('retry-tool', {}, exec);

    assert.equal(result, 'ok-2');
    assert.equal(transport.getStats().totalRetries, 1);
  });

  it('does not retry non-timeout errors', async () => {
    const transport = new OptimizedTransport({
      maxBatchSize: 1,
      maxRetries: 2,
      retryBaseDelayMs: 10,
    });

    await assert.rejects(
      () => transport.send('fail-tool', {}, failExecutor()),
      (err: Error) => {
        assert.ok(err.message.includes('Connection refused'));
        return true;
      }
    );

    assert.equal(transport.getStats().totalRetries, 0);
  });
});

// -- Flush -----------------------------------------------------------------

describe('OptimizedTransport - Flush', () => {
  it('flush() processes pending batch immediately', async () => {
    const transport = new OptimizedTransport({
      maxBatchSize: 10,
      maxBatchWaitMs: 5000,
    });
    const exec = successExecutor();

    // Send but don't wait for batch timer
    const resultPromise = transport.send('flush-tool', {}, exec);

    // Manually flush
    await transport.flush(exec);

    const result = await resultPromise;
    assert.equal(result, 'result-for-flush-tool');
  });

  it('flush() is a no-op when no pending requests', async () => {
    const transport = new OptimizedTransport();
    await transport.flush(successExecutor()); // Should not throw
  });
});

// -- Stats -----------------------------------------------------------------

describe('OptimizedTransport - Stats', () => {
  it('returns correct shape', () => {
    const transport = new OptimizedTransport();
    const stats = transport.getStats();
    assert.equal(typeof stats.totalRequests, 'number');
    assert.equal(typeof stats.totalBatched, 'number');
    assert.equal(typeof stats.totalDeduplicated, 'number');
    assert.equal(typeof stats.avgBatchSize, 'number');
    assert.equal(typeof stats.bytesSaved, 'number');
    assert.equal(typeof stats.totalRetries, 'number');
  });

  it('avgBatchSize is 0 when no batches processed', () => {
    const transport = new OptimizedTransport();
    assert.equal(transport.getStats().avgBatchSize, 0);
  });
});
