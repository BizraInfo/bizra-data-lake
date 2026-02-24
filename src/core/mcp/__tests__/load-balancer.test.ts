/**
 * Tests for MCPLoadBalancer - server selection strategies,
 * health tracking, weight decay/recovery, and failure handling.
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MCPLoadBalancer, BalancingStrategy } from '../load-balancer';

// ---------------------------------------------------------------------------
// Server management
// ---------------------------------------------------------------------------

describe('MCPLoadBalancer - server management', () => {
  it('addServer() registers with default weight 1.0', () => {
    const lb = new MCPLoadBalancer();
    lb.addServer('s1');
    const s = lb.getServerStates();
    assert.equal(s.length, 1);
    assert.equal(s[0]!.weight, 1.0);
    assert.equal(s[0]!.healthy, true);
  });

  it('removeServer() removes server', () => {
    const lb = new MCPLoadBalancer();
    lb.addServer('s1');
    lb.addServer('s2');
    lb.removeServer('s1');
    assert.equal(lb.getServerStates().length, 1);
    assert.equal(lb.getServerStates()[0]!.serverId, 's2');
  });

  it('getServerStates() returns all states', () => {
    const lb = new MCPLoadBalancer();
    lb.addServer('a');
    lb.addServer('b');
    lb.addServer('c');
    assert.equal(lb.getServerStates().length, 3);
  });
});

// ---------------------------------------------------------------------------
// Selection strategies
// ---------------------------------------------------------------------------

describe('MCPLoadBalancer - ROUND_ROBIN', () => {
  it('cycles through servers in order', () => {
    const lb = new MCPLoadBalancer({ strategy: BalancingStrategy.ROUND_ROBIN });
    lb.addServer('s1');
    lb.addServer('s2');
    lb.addServer('s3');

    const ids: string[] = [];
    for (let i = 0; i < 6; i++) {
      const r = lb.select();
      assert.ok(r);
      ids.push(r.serverId);
      lb.recordSuccess(r.serverId, 10);
    }
    assert.equal(ids[0], ids[3]);
    assert.equal(ids[1], ids[4]);
    assert.equal(ids[2], ids[5]);
    assert.equal(new Set(ids).size, 3);
  });
});

describe('MCPLoadBalancer - LEAST_LATENCY', () => {
  it('picks lowest latency server', () => {
    const lb = new MCPLoadBalancer({ strategy: BalancingStrategy.LEAST_LATENCY });
    lb.addServer('slow');
    lb.addServer('fast');
    lb.addServer('medium');
    lb.recordSuccess('slow', 200);
    lb.recordSuccess('fast', 5);
    lb.recordSuccess('medium', 50);

    const r = lb.select();
    assert.ok(r);
    assert.equal(r.serverId, 'fast');
  });
});

describe('MCPLoadBalancer - LEAST_LOADED', () => {
  it('picks server with fewest active requests', () => {
    const lb = new MCPLoadBalancer({ strategy: BalancingStrategy.LEAST_LOADED });
    lb.addServer('a');
    lb.addServer('b');

    const r1 = lb.select(); // one server gets +1 active
    assert.ok(r1);
    const r2 = lb.select(); // should pick the other (0 active)
    assert.ok(r2);
    assert.notEqual(r1.serverId, r2.serverId);
  });
});

describe('MCPLoadBalancer - WEIGHTED_RANDOM', () => {
  it('respects weights (statistical, 1000 selections)', () => {
    // Disable weight recovery so weights stay constant during the test
    const lb = new MCPLoadBalancer({
      strategy: BalancingStrategy.WEIGHTED_RANDOM,
      excludeUnhealthy: false,
      successWeightRecovery: 1.0,
    });
    lb.addServer('heavy', 0.9);
    lb.addServer('light', 0.1);

    const counts: Record<string, number> = { heavy: 0, light: 0 };
    for (let i = 0; i < 1000; i++) {
      const r = lb.select();
      assert.ok(r);
      counts[r.serverId]!++;
      lb.recordSuccess(r.serverId, 10);
    }
    assert.ok(counts.heavy! / 1000 > 0.75, `heavy ratio too low: ${counts.heavy! / 1000}`);
    assert.ok(counts.light! > 0, 'light should get some selections');
  });
});

// ---------------------------------------------------------------------------
// Exclusion filters
// ---------------------------------------------------------------------------

describe('MCPLoadBalancer - exclusion', () => {
  let lb: MCPLoadBalancer;
  beforeEach(() => {
    lb = new MCPLoadBalancer({ strategy: BalancingStrategy.LEAST_LATENCY, excludeUnhealthy: true });
    lb.addServer('s1');
    lb.addServer('s2');
    lb.addServer('s3');
  });

  it('excludes servers in excludeServers set', () => {
    const r = lb.select(new Set(['s1', 's3']));
    assert.ok(r);
    assert.equal(r.serverId, 's2');
  });

  it('excludes unhealthy servers', () => {
    lb.setHealth('s1', false);
    lb.setHealth('s2', false);
    const r = lb.select();
    assert.ok(r);
    assert.equal(r.serverId, 's3');
  });

  it('returns null when no candidates', () => {
    assert.equal(lb.select(new Set(['s1', 's2', 's3'])), null);
  });

  it('returns null when all unhealthy', () => {
    lb.setHealth('s1', false);
    lb.setHealth('s2', false);
    lb.setHealth('s3', false);
    assert.equal(lb.select(), null);
  });
});

// ---------------------------------------------------------------------------
// Selection side-effects
// ---------------------------------------------------------------------------

describe('MCPLoadBalancer - selection side-effects', () => {
  it('select() increments activeRequests and totalRequests', () => {
    const lb = new MCPLoadBalancer({ strategy: BalancingStrategy.ROUND_ROBIN });
    lb.addServer('s1');
    lb.select();
    const st = lb.getServerStates()[0]!;
    assert.equal(st.activeRequests, 1);
    assert.equal(st.totalRequests, 1);
  });

  it('select() provides alternateServerId with multiple servers', () => {
    const lb = new MCPLoadBalancer({ strategy: BalancingStrategy.LEAST_LATENCY });
    lb.addServer('s1');
    lb.addServer('s2');
    const r = lb.select();
    assert.ok(r);
    assert.ok(r.alternateServerId);
    assert.notEqual(r.serverId, r.alternateServerId);
  });

  it('select() omits alternateServerId with single server', () => {
    const lb = new MCPLoadBalancer({ strategy: BalancingStrategy.ROUND_ROBIN });
    lb.addServer('solo');
    const r = lb.select();
    assert.ok(r);
    assert.equal(r.alternateServerId, undefined);
  });
});

// ---------------------------------------------------------------------------
// Success & failure recording
// ---------------------------------------------------------------------------

describe('MCPLoadBalancer - success & failure', () => {
  let lb: MCPLoadBalancer;
  beforeEach(() => {
    lb = new MCPLoadBalancer({ strategy: BalancingStrategy.LEAST_LATENCY, errorWeightDecay: 0.8, successWeightRecovery: 1.05 });
    lb.addServer('s1', 1.0);
  });

  it('recordSuccess() decrements activeRequests and updates latency EMA', () => {
    lb.select();
    lb.recordSuccess('s1', 100);
    assert.equal(lb.getServerStates()[0]!.activeRequests, 0);
    assert.equal(lb.getServerStates()[0]!.avgLatencyMs, 100);

    lb.select();
    lb.recordSuccess('s1', 200);
    // EMA: 0.3 * 200 + 0.7 * 100 = 130
    assert.ok(Math.abs(lb.getServerStates()[0]!.avgLatencyMs - 130) < 1);
  });

  it('recordSuccess() recovers weight after failure', () => {
    lb.select();
    lb.recordFailure('s1');
    const decayed = lb.getServerStates()[0]!.weight;
    lb.select();
    lb.recordSuccess('s1', 10);
    assert.ok(lb.getServerStates()[0]!.weight > decayed);
  });

  it('recordFailure() decrements activeRequests and decays weight', () => {
    const before = lb.getServerStates()[0]!.weight;
    lb.select();
    lb.recordFailure('s1');
    const after = lb.getServerStates()[0]!;
    assert.equal(after.activeRequests, 0);
    assert.equal(after.totalErrors, 1);
    assert.ok(after.weight < before);
  });

  it('recordFailure() marks unhealthy when error rate > 50% after 5+ requests', () => {
    for (let i = 0; i < 6; i++) {
      lb.select();
      lb.recordFailure('s1');
    }
    assert.equal(lb.getServerStates()[0]!.healthy, false);
  });

  it('setHealth() updates server health status', () => {
    lb.setHealth('s1', false);
    assert.equal(lb.getServerStates()[0]!.healthy, false);
    lb.setHealth('s1', true);
    assert.equal(lb.getServerStates()[0]!.healthy, true);
  });
});
