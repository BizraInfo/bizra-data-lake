/**
 * Tests for MCP Connection Pool, Cache, Registry, and Load Balancer
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  MCPConnectionPool,
  ConnectionState,
} from '../connection-pool';
import { FastToolRegistry } from '../fast-tool-registry';
import { MultiLevelCache } from '../multi-level-cache';
import { MCPLoadBalancer, BalancingStrategy } from '../load-balancer';
import { MCPMetrics } from '../metrics';

// ============================================================================
// Connection Pool Tests
// ============================================================================

describe('MCPConnectionPool', () => {
  let pool: MCPConnectionPool;

  beforeEach(() => {
    pool = new MCPConnectionPool({
      maxConnectionsPerServer: 3,
      minIdleConnections: 1,
      connectionTimeoutMs: 1000,
      healthCheckIntervalMs: 60000,
      maxConsecutiveFailures: 2,
    });
  });

  it('should register servers and create min idle connections', () => {
    pool.registerServer('sovereign');
    const stats = pool.getStats();
    assert.equal(stats.totalConnections, 1);
    assert.equal(stats.idleConnections, 1);
    assert.equal(stats.perServer['sovereign']?.total, 1);
  });

  it('should acquire and release connections', () => {
    pool.registerServer('sovereign');
    const conn = pool.acquire('sovereign');
    assert.ok(conn);
    assert.equal(conn.state, ConnectionState.ACTIVE);

    const stats = pool.getStats();
    assert.equal(stats.activeConnections, 1);

    pool.release(conn);
    assert.equal(conn.state, ConnectionState.IDLE);
  });

  it('should create new connections up to max', () => {
    pool.registerServer('ecosystem');
    const c1 = pool.acquire('ecosystem');
    const c2 = pool.acquire('ecosystem');
    const c3 = pool.acquire('ecosystem');
    assert.ok(c1);
    assert.ok(c2);
    assert.ok(c3);

    // Should return null when pool is full
    const c4 = pool.acquire('ecosystem');
    assert.equal(c4, null);
  });

  it('should mark connections unhealthy after consecutive failures', () => {
    pool.registerServer('ddagi');
    const conn = pool.acquire('ddagi');
    assert.ok(conn);

    pool.markFailed(conn);
    assert.equal(conn.state, ConnectionState.IDLE); // 1st failure: back to idle

    const conn2 = pool.acquire('ddagi');
    assert.ok(conn2);
    pool.markFailed(conn2);
    // 2nd consecutive failure: should be unhealthy
    assert.equal(conn2.state, ConnectionState.UNHEALTHY);
  });

  it('should track healthy servers', () => {
    pool.registerServer('s1');
    pool.registerServer('s2');
    assert.ok(pool.isServerHealthy('s1'));
    assert.ok(pool.isServerHealthy('s2'));
    assert.deepEqual(pool.getHealthyServers().sort(), ['s1', 's2']);
  });

  it('should return null for unregistered server', () => {
    assert.equal(pool.acquire('nonexistent'), null);
  });

  it('should stop and clean up', () => {
    pool.registerServer('test');
    pool.start();
    pool.stop();
    const stats = pool.getStats();
    assert.equal(stats.totalConnections, 0);
  });
});

// ============================================================================
// Fast Tool Registry Tests
// ============================================================================

describe('FastToolRegistry', () => {
  let registry: FastToolRegistry;

  beforeEach(() => {
    registry = new FastToolRegistry();
  });

  it('should register and lookup tools in O(1)', () => {
    registry.register({
      name: 'sovereign_query',
      serverId: 'bizra-sovereign',
      description: 'Query the sovereign brain',
      inputSchema: { type: 'object' },
      cacheable: true,
    });

    const start = performance.now();
    const entry = registry.lookup('sovereign_query');
    const elapsed = performance.now() - start;

    assert.ok(entry);
    assert.equal(entry.name, 'sovereign_query');
    assert.equal(entry.serverId, 'bizra-sovereign');
    assert.ok(elapsed < 5, `Lookup took ${elapsed}ms, expected <5ms`);
  });

  it('should register multiple tools from server', () => {
    registry.registerFromServer(
      'sovereign',
      [
        { name: 'sovereign_query', description: 'Query', inputSchema: {} },
        { name: 'sovereign_health', description: 'Health', inputSchema: {} },
      ],
      new Set(['sovereign_query', 'sovereign_health'])
    );

    assert.equal(registry.size, 2);
    assert.ok(registry.has('sovereign_query'));
    assert.ok(registry.has('sovereign_health'));
  });

  it('should track invocations', () => {
    registry.register({
      name: 'test_tool',
      serverId: 'test',
      description: 'Test',
      inputSchema: {},
      cacheable: false,
    });

    registry.recordInvocation('test_tool', 50);
    registry.recordInvocation('test_tool', 100);

    const entry = registry.lookup('test_tool');
    assert.ok(entry);
    assert.equal(entry.invocationCount, 2);
    assert.ok(entry.avgResponseMs > 0);
  });

  it('should get tools by server', () => {
    registry.register({ name: 't1', serverId: 's1', description: '', inputSchema: {}, cacheable: false });
    registry.register({ name: 't2', serverId: 's1', description: '', inputSchema: {}, cacheable: false });
    registry.register({ name: 't3', serverId: 's2', description: '', inputSchema: {}, cacheable: false });

    const s1Tools = registry.getServerTools('s1');
    assert.equal(s1Tools.length, 2);
  });

  it('should remove server and its tools', () => {
    registry.register({ name: 't1', serverId: 's1', description: '', inputSchema: {}, cacheable: false });
    registry.removeServer('s1');
    assert.equal(registry.size, 0);
    assert.ok(!registry.has('t1'));
  });
});

// ============================================================================
// Multi-Level Cache Tests
// ============================================================================

describe('MultiLevelCache', () => {
  let cache: MultiLevelCache<string>;

  beforeEach(() => {
    cache = new MultiLevelCache<string>({
      l1MaxEntries: 4,
      l2MaxEntries: 8,
      l1TtlMs: 60000,
      l2TtlMs: 300000,
    });
  });

  it('should return undefined on miss', () => {
    assert.equal(cache.get('nonexistent'), undefined);
    assert.equal(cache.getStats().misses, 1);
  });

  it('should store and retrieve values', () => {
    cache.set('key1', 'value1');
    assert.equal(cache.get('key1'), 'value1');
    assert.equal(cache.getStats().l1Hits, 1);
  });

  it('should track L1 and L2 hits separately', () => {
    cache.set('key1', 'value1');

    // L1 hit
    cache.get('key1');
    const stats1 = cache.getStats();
    assert.equal(stats1.l1Hits, 1);
    assert.equal(stats1.l2Hits, 0);
  });

  it('should evict L1 entries when full', () => {
    // Fill L1 (max 4)
    for (let i = 0; i < 6; i++) {
      cache.set(`key${i}`, `value${i}`);
    }

    const stats = cache.getStats();
    assert.ok(stats.l1Size <= 4);
    assert.ok(stats.evictions > 0);
  });

  it('should generate consistent cache keys', () => {
    const key1 = MultiLevelCache.makeKey('tool', { a: 1, b: 2 });
    const key2 = MultiLevelCache.makeKey('tool', { a: 1, b: 2 });
    assert.equal(key1, key2);

    const key3 = MultiLevelCache.makeKey('tool', { a: 1, b: 3 });
    assert.notEqual(key1, key3);
  });

  it('should calculate hit rate', () => {
    cache.set('a', 'v');
    cache.get('a'); // hit
    cache.get('a'); // hit
    cache.get('b'); // miss

    const stats = cache.getStats();
    assert.ok(stats.hitRate > 0.6);
  });

  it('should clear all entries', () => {
    cache.set('a', '1');
    cache.set('b', '2');
    cache.clear();
    assert.equal(cache.get('a'), undefined);
    assert.equal(cache.getStats().l1Size, 0);
  });
});

// ============================================================================
// Load Balancer Tests
// ============================================================================

describe('MCPLoadBalancer', () => {
  let lb: MCPLoadBalancer;

  beforeEach(() => {
    lb = new MCPLoadBalancer({
      strategy: BalancingStrategy.LEAST_LATENCY,
      excludeUnhealthy: true,
    });
    lb.addServer('s1', 1.0);
    lb.addServer('s2', 1.0);
    lb.addServer('s3', 1.0);
  });

  it('should select a server', () => {
    const result = lb.select();
    assert.ok(result);
    assert.ok(['s1', 's2', 's3'].includes(result.serverId));
  });

  it('should prefer lower latency servers', () => {
    lb.recordSuccess('s1', 100);
    lb.recordSuccess('s2', 10);
    lb.recordSuccess('s3', 50);

    const result = lb.select();
    assert.ok(result);
    assert.equal(result.serverId, 's2');
  });

  it('should exclude unhealthy servers', () => {
    lb.setHealth('s1', false);
    lb.setHealth('s2', false);

    const result = lb.select();
    assert.ok(result);
    assert.equal(result.serverId, 's3');
  });

  it('should return null when all servers excluded', () => {
    const result = lb.select(new Set(['s1', 's2', 's3']));
    assert.equal(result, null);
  });

  it('should decay weight on failure', () => {
    const before = lb.getServerStates().find((s) => s.serverId === 's1')!;
    const initialWeight = before.weight;

    lb.recordFailure('s1');

    const after = lb.getServerStates().find((s) => s.serverId === 's1')!;
    assert.ok(after.weight < initialWeight);
  });

  it('should provide alternate server', () => {
    lb.recordSuccess('s1', 10);
    lb.recordSuccess('s2', 20);
    lb.recordSuccess('s3', 30);

    const result = lb.select();
    assert.ok(result);
    assert.ok(result.alternateServerId);
    assert.notEqual(result.serverId, result.alternateServerId);
  });
});

// ============================================================================
// Metrics Tests
// ============================================================================

describe('MCPMetrics', () => {
  let metrics: MCPMetrics;

  beforeEach(() => {
    metrics = new MCPMetrics({ bufferSize: 100 });
  });

  it('should track request latencies', () => {
    metrics.recordRequest('s1', 50);
    metrics.recordRequest('s1', 100);
    metrics.recordRequest('s2', 75);

    const snapshot = metrics.getSnapshot();
    assert.equal(snapshot.totalRequests, 3);
    assert.ok(snapshot.avgResponseMs > 0);
    assert.ok(snapshot.p95ResponseMs > 0);
  });

  it('should track errors', () => {
    metrics.recordRequest('s1', 50);
    metrics.recordError('s1');

    const snapshot = metrics.getSnapshot();
    assert.equal(snapshot.totalRequests, 2);
    assert.equal(snapshot.totalErrors, 1);
    assert.ok(snapshot.errorRate > 0);
  });

  it('should track cache hits and misses', () => {
    metrics.recordCacheHit();
    metrics.recordCacheHit();
    metrics.recordCacheMiss();

    const snapshot = metrics.getSnapshot();
    assert.ok(snapshot.cacheHitRate > 0.6);
  });

  it('should provide per-server metrics', () => {
    metrics.recordRequest('s1', 50);
    metrics.recordRequest('s2', 100);

    const perServer = metrics.getServerMetrics();
    assert.equal(perServer.length, 2);
    assert.ok(perServer.some((s) => s.serverId === 's1'));
  });

  it('should calculate quality score', () => {
    // Good performance
    for (let i = 0; i < 10; i++) {
      metrics.recordRequest('s1', 20);
      metrics.recordCacheHit();
    }

    const snapshot = metrics.getSnapshot();
    assert.ok(snapshot.qualityScore > 0.5);
  });
});
