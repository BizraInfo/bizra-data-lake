/**
 * Tests for FastToolRegistry - O(1) tool lookup, categorization,
 * server mapping, invocation tracking, and statistics.
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { FastToolRegistry } from '../fast-tool-registry';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTool(name: string, serverId: string, cacheable = false) {
  return { name, serverId, description: `${name} desc`, inputSchema: { type: 'object' }, cacheable };
}

// ---------------------------------------------------------------------------
// Core CRUD
// ---------------------------------------------------------------------------

describe('FastToolRegistry', () => {
  let registry: FastToolRegistry;

  beforeEach(() => {
    registry = new FastToolRegistry();
  });

  it('register() adds a tool and lookup() finds it in O(1)', () => {
    registry.register(makeTool('sovereign_query', 'srv-1', true));

    const start = performance.now();
    const entry = registry.lookup('sovereign_query');
    const elapsed = performance.now() - start;

    assert.ok(entry);
    assert.equal(entry.name, 'sovereign_query');
    assert.equal(entry.serverId, 'srv-1');
    assert.equal(entry.cacheable, true);
    assert.ok(elapsed < 5, `Lookup took ${elapsed}ms, expected <5ms`);
  });

  it('lookup() returns undefined for unregistered tool', () => {
    assert.equal(registry.lookup('ghost'), undefined);
  });

  it('registerFromServer() registers multiple tools from one server', () => {
    registry.registerFromServer(
      'eco',
      [
        { name: 'eco_health', description: 'Health', inputSchema: {} },
        { name: 'eco_query', description: 'Query', inputSchema: {} },
        { name: 'eco_status', description: 'Status', inputSchema: {} },
      ],
      new Set(['eco_query'])
    );

    assert.equal(registry.size, 3);
    assert.ok(registry.has('eco_health'));
    assert.ok(registry.has('eco_query'));
    assert.ok(registry.has('eco_status'));

    // Only eco_query was marked cacheable
    const q = registry.lookup('eco_query');
    assert.ok(q);
    assert.equal(q.cacheable, true);

    const h = registry.lookup('eco_health');
    assert.ok(h);
    assert.equal(h.cacheable, false);
  });

  it('has() returns true for registered, false for missing', () => {
    registry.register(makeTool('present', 'srv'));
    assert.equal(registry.has('present'), true);
    assert.equal(registry.has('absent'), false);
  });

  it('getServerTools() returns only tools for that server', () => {
    registry.register(makeTool('a', 's1'));
    registry.register(makeTool('b', 's1'));
    registry.register(makeTool('c', 's2'));

    const s1Tools = registry.getServerTools('s1');
    assert.equal(s1Tools.length, 2);
    assert.deepEqual(s1Tools.map((t) => t.name).sort(), ['a', 'b']);

    const s2Tools = registry.getServerTools('s2');
    assert.equal(s2Tools.length, 1);
    assert.equal(s2Tools[0]!.name, 'c');

    assert.deepEqual(registry.getServerTools('no-server'), []);
  });

  it('getServerForTool() returns correct server ID', () => {
    registry.register(makeTool('t1', 'alpha'));
    registry.register(makeTool('t2', 'beta'));

    assert.equal(registry.getServerForTool('t1'), 'alpha');
    assert.equal(registry.getServerForTool('t2'), 'beta');
    assert.equal(registry.getServerForTool('t3'), undefined);
  });
});

// ---------------------------------------------------------------------------
// Invocation tracking
// ---------------------------------------------------------------------------

describe('FastToolRegistry - invocation tracking', () => {
  let registry: FastToolRegistry;

  beforeEach(() => {
    registry = new FastToolRegistry();
    registry.register(makeTool('tool_a', 'srv'));
  });

  it('recordInvocation() sets avgResponseMs on first call', () => {
    registry.recordInvocation('tool_a', 80);
    const entry = registry.lookup('tool_a')!;
    assert.equal(entry.invocationCount, 1);
    assert.equal(entry.avgResponseMs, 80);
  });

  it('recordInvocation() applies EMA on subsequent calls', () => {
    registry.recordInvocation('tool_a', 100);
    registry.recordInvocation('tool_a', 200);

    const entry = registry.lookup('tool_a')!;
    assert.equal(entry.invocationCount, 2);
    // EMA: 0.3 * 200 + 0.7 * 100 = 130
    assert.ok(Math.abs(entry.avgResponseMs - 130) < 0.01,
      `EMA was ${entry.avgResponseMs}, expected 130`);
  });

  it('recordInvocation() updates lastInvokedAt', () => {
    const before = Date.now();
    registry.recordInvocation('tool_a', 10);
    const entry = registry.lookup('tool_a')!;
    assert.ok(entry.lastInvokedAt !== null);
    assert.ok(entry.lastInvokedAt >= before);
  });

  it('recordInvocation() ignores unknown tools', () => {
    // Should not throw
    registry.recordInvocation('nonexistent', 50);
    assert.equal(registry.size, 1);
  });
});

// ---------------------------------------------------------------------------
// Categorization
// ---------------------------------------------------------------------------

describe('FastToolRegistry - categorization', () => {
  let registry: FastToolRegistry;

  beforeEach(() => {
    registry = new FastToolRegistry();
    registry.register(makeTool('read_memory', 'mem'));
    registry.register(makeTool('write_memory', 'mem'));
    registry.register(makeTool('query_graph', 'graph'));
  });

  it('categorize() + getByCategory() work correctly', () => {
    registry.categorize('read_memory', 'memory');
    registry.categorize('write_memory', 'memory');
    registry.categorize('query_graph', 'reasoning');

    const memTools = registry.getByCategory('memory');
    assert.equal(memTools.length, 2);
    assert.deepEqual(memTools.map((t) => t.name).sort(), ['read_memory', 'write_memory']);

    const reasonTools = registry.getByCategory('reasoning');
    assert.equal(reasonTools.length, 1);
    assert.equal(reasonTools[0]!.name, 'query_graph');
  });

  it('getByCategory() returns empty array for unknown category', () => {
    assert.deepEqual(registry.getByCategory('void'), []);
  });

  it('categorize() ignores unregistered tool name', () => {
    registry.categorize('ghost', 'phantom');
    assert.deepEqual(registry.getByCategory('phantom'), []);
  });
});

// ---------------------------------------------------------------------------
// Removal, listing, stats
// ---------------------------------------------------------------------------

describe('FastToolRegistry - removal, listing, stats', () => {
  let registry: FastToolRegistry;

  beforeEach(() => {
    registry = new FastToolRegistry();
    registry.register(makeTool('x1', 'sA', true));
    registry.register(makeTool('x2', 'sA'));
    registry.register(makeTool('y1', 'sB', true));
  });

  it('removeServer() cleans up tools and categories', () => {
    registry.categorize('x1', 'cat1');
    registry.categorize('x2', 'cat1');

    registry.removeServer('sA');

    assert.equal(registry.size, 1);
    assert.ok(!registry.has('x1'));
    assert.ok(!registry.has('x2'));
    assert.ok(registry.has('y1'));

    // Category entries for removed tools should be gone
    const catTools = registry.getByCategory('cat1');
    assert.equal(catTools.length, 0);
  });

  it('getCacheableTools() filters correctly', () => {
    const cacheable = registry.getCacheableTools();
    assert.equal(cacheable.length, 2);
    assert.deepEqual(cacheable.map((t) => t.name).sort(), ['x1', 'y1']);
  });

  it('listAll() returns all registered tools', () => {
    const all = registry.listAll();
    assert.equal(all.length, 3);
  });

  it('getStats() returns correct counts and mostUsed sorted', () => {
    registry.recordInvocation('x2', 10);
    registry.recordInvocation('x2', 20);
    registry.recordInvocation('y1', 5);

    const stats = registry.getStats();
    assert.equal(stats.totalTools, 3);
    assert.equal(stats.totalServers, 2);
    assert.equal(stats.toolsPerServer['sA'], 2);
    assert.equal(stats.toolsPerServer['sB'], 1);
    assert.equal(stats.lookupTimeComplexity, 'O(1)');

    // mostUsed: x2 (2) > y1 (1) > x1 (0)
    assert.equal(stats.mostUsed[0]!.name, 'x2');
    assert.equal(stats.mostUsed[0]!.count, 2);
  });

  it('size getter returns tool count', () => {
    assert.equal(registry.size, 3);
    registry.register(makeTool('z1', 'sC'));
    assert.equal(registry.size, 4);
  });
});
