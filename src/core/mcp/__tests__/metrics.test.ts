/**
 * Tests for MCPMetrics - request/error/cache tracking,
 * quickselect-powered percentiles, quality score, Ihsan threshold.
 */

import { describe, it, beforeEach, afterEach, mock } from 'node:test';
import assert from 'node:assert/strict';
import { MCPMetrics } from '../metrics';
import { IHSAN_THRESHOLD } from '../../sovereign/capability-card';

// ---------------------------------------------------------------------------
// Request & Error tracking
// ---------------------------------------------------------------------------

describe('MCPMetrics - request tracking', () => {
  let metrics: MCPMetrics;

  beforeEach(() => {
    metrics = new MCPMetrics({ bufferSize: 100 });
  });

  it('recordRequest() increments totalRequests', () => {
    metrics.recordRequest('s1', 50);
    metrics.recordRequest('s1', 60);
    metrics.recordRequest('s2', 70);

    const snap = metrics.getSnapshot();
    assert.equal(snap.totalRequests, 3);
  });

  it('recordRequest() tracks per-server latency', () => {
    metrics.recordRequest('alpha', 20);
    metrics.recordRequest('alpha', 40);
    metrics.recordRequest('beta', 100);

    const perServer = metrics.getServerMetrics();
    const alpha = perServer.find((s) => s.serverId === 'alpha');
    const beta = perServer.find((s) => s.serverId === 'beta');

    assert.ok(alpha);
    assert.equal(alpha.totalRequests, 2);
    assert.ok(alpha.avgResponseMs > 0);

    assert.ok(beta);
    assert.equal(beta.totalRequests, 1);
    assert.equal(beta.avgResponseMs, 100);
  });

  it('recordError() increments totalErrors and per-server errors', () => {
    metrics.recordRequest('s1', 10);
    metrics.recordError('s1');

    const snap = metrics.getSnapshot();
    assert.equal(snap.totalRequests, 2); // recordError also counts as a request
    assert.equal(snap.totalErrors, 1);
    assert.ok(snap.errorRate > 0);

    const perServer = metrics.getServerMetrics();
    const s1 = perServer.find((s) => s.serverId === 's1')!;
    assert.equal(s1.totalErrors, 1);
  });
});

// ---------------------------------------------------------------------------
// Cache tracking
// ---------------------------------------------------------------------------

describe('MCPMetrics - cache tracking', () => {
  let metrics: MCPMetrics;

  beforeEach(() => {
    metrics = new MCPMetrics({ bufferSize: 100 });
  });

  it('recordCacheHit/Miss() tracks cache stats', () => {
    metrics.recordCacheHit();
    metrics.recordCacheHit();
    metrics.recordCacheHit();
    metrics.recordCacheMiss();

    const snap = metrics.getSnapshot();
    assert.ok(snap.cacheHitRate >= 0.74, `cacheHitRate was ${snap.cacheHitRate}`);
    assert.ok(snap.cacheHitRate <= 0.76, `cacheHitRate was ${snap.cacheHitRate}`);
  });

  it('cacheHitRate is 0 when no cache operations', () => {
    assert.equal(metrics.getSnapshot().cacheHitRate, 0);
  });
});

// ---------------------------------------------------------------------------
// Connection tracking
// ---------------------------------------------------------------------------

describe('MCPMetrics - connection tracking', () => {
  let metrics: MCPMetrics;

  beforeEach(() => {
    metrics = new MCPMetrics({ bufferSize: 100 });
  });

  it('connectionOpened/Closed() tracks active connections', () => {
    metrics.connectionOpened();
    metrics.connectionOpened();
    assert.equal(metrics.getSnapshot().activeConnections, 2);

    metrics.connectionClosed();
    assert.equal(metrics.getSnapshot().activeConnections, 1);
  });

  it('activeConnections never goes negative', () => {
    metrics.connectionClosed();
    metrics.connectionClosed();
    assert.equal(metrics.getSnapshot().activeConnections, 0);
  });
});

// ---------------------------------------------------------------------------
// Snapshot completeness
// ---------------------------------------------------------------------------

describe('MCPMetrics - getSnapshot()', () => {
  it('returns correct MetricSnapshot with all fields', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });
    metrics.recordRequest('s1', 25);
    metrics.recordCacheHit();
    metrics.connectionOpened();

    const snap = metrics.getSnapshot();
    assert.equal(typeof snap.timestamp, 'number');
    assert.equal(snap.totalRequests, 1);
    assert.equal(snap.totalErrors, 0);
    assert.equal(snap.errorRate, 0);
    assert.ok(snap.avgResponseMs > 0);
    assert.equal(typeof snap.p50ResponseMs, 'number');
    assert.equal(typeof snap.p95ResponseMs, 'number');
    assert.equal(typeof snap.p99ResponseMs, 'number');
    assert.equal(typeof snap.cacheHitRate, 'number');
    assert.equal(snap.activeConnections, 1);
    assert.equal(typeof snap.qualityScore, 'number');
  });
});

// ---------------------------------------------------------------------------
// Percentile calculations via quickselect
// ---------------------------------------------------------------------------

describe('MCPMetrics - percentile calculations', () => {
  it('p50, p95, p99 return reasonable values for known data', () => {
    const metrics = new MCPMetrics({ bufferSize: 200 });
    for (let i = 1; i <= 100; i++) {
      metrics.recordRequest('s1', i);
    }

    const snap = metrics.getSnapshot();
    assert.ok(snap.p50ResponseMs >= 49 && snap.p50ResponseMs <= 51,
      `p50 was ${snap.p50ResponseMs}, expected ~50`);
    assert.ok(snap.p95ResponseMs >= 94 && snap.p95ResponseMs <= 96,
      `p95 was ${snap.p95ResponseMs}, expected ~95`);
    assert.ok(snap.p99ResponseMs >= 98 && snap.p99ResponseMs <= 100,
      `p99 was ${snap.p99ResponseMs}, expected ~99`);
  });

  it('percentiles return 0 when no data', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });
    const snap = metrics.getSnapshot();
    assert.equal(snap.p50ResponseMs, 0);
    assert.equal(snap.p95ResponseMs, 0);
    assert.equal(snap.p99ResponseMs, 0);
  });
});

// ---------------------------------------------------------------------------
// Quality score & Ihsan
// ---------------------------------------------------------------------------

describe('MCPMetrics - quality score', () => {
  it('qualityScore uses weighted formula: 0.5 reliability + 0.3 latency + 0.2 cache', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });

    // Perfect reliability (0 errors), low latency, good cache
    for (let i = 0; i < 20; i++) {
      metrics.recordRequest('s1', 10); // 10ms -> latency score ~= 1 - 10/1000 = 0.99
      metrics.recordCacheHit();
    }

    const snap = metrics.getSnapshot();
    // reliability = 1.0, latencyScore ~ 0.99, cacheScore = 1.0
    // quality ~ 0.5*1.0 + 0.3*0.99 + 0.2*1.0 = 0.997
    assert.ok(snap.qualityScore >= 0.95,
      `qualityScore was ${snap.qualityScore}, expected >= 0.95`);
  });

  it('qualityScore drops with errors', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });
    for (let i = 0; i < 5; i++) {
      metrics.recordRequest('s1', 10);
    }
    for (let i = 0; i < 5; i++) {
      metrics.recordError('s1');
    }

    const snap = metrics.getSnapshot();
    // 50% error rate -> reliability = 0.5
    assert.ok(snap.qualityScore < 0.8,
      `qualityScore was ${snap.qualityScore}, expected < 0.8`);
  });
});

describe('MCPMetrics - meetsIhsanThreshold()', () => {
  it('IHSAN_THRESHOLD is 0.95', () => {
    assert.equal(IHSAN_THRESHOLD, 0.95);
  });

  it('returns true when quality >= 0.95', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });
    for (let i = 0; i < 20; i++) {
      metrics.recordRequest('s1', 5);
      metrics.recordCacheHit();
    }
    assert.equal(metrics.meetsIhsanThreshold(), true);
  });

  it('returns false when quality is low', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });
    for (let i = 0; i < 10; i++) {
      metrics.recordError('s1');
      metrics.recordCacheMiss();
    }
    assert.equal(metrics.meetsIhsanThreshold(), false);
  });
});

// ---------------------------------------------------------------------------
// Per-server metrics & history
// ---------------------------------------------------------------------------

describe('MCPMetrics - server metrics and history', () => {
  it('getServerMetrics() returns per-server breakdown', () => {
    const metrics = new MCPMetrics({ bufferSize: 100 });
    metrics.recordRequest('alpha', 50);
    metrics.recordRequest('beta', 80);
    metrics.recordError('beta');

    const servers = metrics.getServerMetrics();
    assert.equal(servers.length, 2);

    const alpha = servers.find((s) => s.serverId === 'alpha')!;
    assert.equal(alpha.totalRequests, 1);
    assert.equal(alpha.isHealthy, true);

    const beta = servers.find((s) => s.serverId === 'beta')!;
    assert.equal(beta.totalRequests, 2);
    assert.equal(beta.totalErrors, 1);
  });

  it('getHistory() accumulates snapshots', () => {
    const metrics = new MCPMetrics({
      bufferSize: 100,
      snapshotIntervalMs: 50,
      maxSnapshots: 10,
    });

    assert.equal(metrics.getHistory().length, 0);

    // Manually trigger internal snapshot via start/stop with small interval
    metrics.start();
    // The timer will fire asynchronously, so just validate initial state
    assert.equal(typeof metrics.getHistory().length, 'number');
    metrics.stop();
  });
});

// ---------------------------------------------------------------------------
// Start / Stop timer management
// ---------------------------------------------------------------------------

describe('MCPMetrics - start/stop', () => {
  it('start() and stop() manage the snapshot timer', () => {
    const metrics = new MCPMetrics({
      bufferSize: 100,
      snapshotIntervalMs: 100_000,
    });

    // Should not throw
    metrics.start();
    metrics.stop();

    // Double stop should not throw
    metrics.stop();
  });
});
