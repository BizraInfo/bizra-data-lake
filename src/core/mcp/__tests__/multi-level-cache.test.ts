/**
 * Tests for MultiLevelCache - L1/L2 storage, TTL expiration,
 * LRU eviction, promotion, size limits, and statistics.
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { MultiLevelCache } from '../multi-level-cache';

const LONG_TTL = { l1TtlMs: 60_000, l2TtlMs: 300_000 };

// ---------------------------------------------------------------------------
// Basic get / set / miss
// ---------------------------------------------------------------------------

describe('MultiLevelCache - basic operations', () => {
  let cache: MultiLevelCache<string>;

  beforeEach(() => {
    cache = new MultiLevelCache<string>({ l1MaxEntries: 4, l2MaxEntries: 8, ...LONG_TTL });
  });

  it('set() stores in both L1 and L2', () => {
    cache.set('k1', 'v1');
    const s = cache.getStats();
    assert.equal(s.l1Size, 1);
    assert.equal(s.l2Size, 1);
  });

  it('get() returns value from L1 (L1 hit)', () => {
    cache.set('k1', 'v1');
    assert.equal(cache.get('k1'), 'v1');
    assert.equal(cache.getStats().l1Hits, 1);
    assert.equal(cache.getStats().l2Hits, 0);
  });

  it('get() returns undefined for missing key (miss)', () => {
    assert.equal(cache.get('nonexistent'), undefined);
    assert.equal(cache.getStats().misses, 1);
  });
});

// ---------------------------------------------------------------------------
// L2 hit and promotion
// ---------------------------------------------------------------------------

describe('MultiLevelCache - L2 hit and promotion', () => {
  it('get() returns value from L2 when L1 misses', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 2, l2MaxEntries: 8, ...LONG_TTL });
    cache.set('a', '1');
    cache.set('b', '2');
    cache.set('c', '3'); // evicts 'a' from L1
    assert.ok(cache.getStats().l1Size <= 2);

    assert.equal(cache.get('a'), '1');
    assert.ok(cache.getStats().l2Hits >= 1, 'Should record L2 hit');
  });

  it('L2 hit promotes entry to L1', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 2, l2MaxEntries: 8, ...LONG_TTL });
    cache.set('a', '1');
    cache.set('b', '2');
    cache.set('c', '3'); // evicts 'a' from L1
    cache.get('a');       // L2 hit -> promotes to L1
    cache.get('a');       // should now be L1 hit
    assert.ok(cache.getStats().l1Hits >= 1, `l1Hits was ${cache.getStats().l1Hits}`);
  });
});

// ---------------------------------------------------------------------------
// TTL expiration
// ---------------------------------------------------------------------------

function busyWait(ms: number): void {
  const start = Date.now();
  while (Date.now() - start < ms) { /* spin */ }
}

describe('MultiLevelCache - TTL expiration', () => {
  it('L1 TTL: entry expires, falls through to L2', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 10, l1TtlMs: 1, l2TtlMs: 300_000 });
    cache.set('x', 'val');
    busyWait(5);
    assert.equal(cache.get('x'), 'val');
    assert.ok(cache.getStats().l2Hits >= 1);
  });

  it('L2 TTL: entry expires, returns undefined', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 10, l1TtlMs: 1, l2TtlMs: 1 });
    cache.set('x', 'gone');
    busyWait(5);
    assert.equal(cache.get('x'), undefined);
    assert.ok(cache.getStats().misses >= 1);
  });
});

// ---------------------------------------------------------------------------
// LRU eviction
// ---------------------------------------------------------------------------

describe('MultiLevelCache - LRU eviction', () => {
  it('L1 eviction when exceeding l1MaxEntries', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 3, l2MaxEntries: 20, ...LONG_TTL });
    for (let i = 0; i < 5; i++) cache.set(`k${i}`, `v${i}`);
    assert.ok(cache.getStats().l1Size <= 3);
    assert.ok(cache.getStats().evictions >= 2);
  });

  it('L2 eviction when exceeding l2MaxEntries', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 100, l2MaxEntries: 3, ...LONG_TTL });
    for (let i = 0; i < 5; i++) cache.set(`k${i}`, `v${i}`);
    assert.ok(cache.getStats().l2Size <= 3);
    assert.ok(cache.getStats().evictions >= 2);
  });

  it('LRU ordering: recently accessed entries survive eviction', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 3, l2MaxEntries: 20, ...LONG_TTL });
    cache.set('a', '1');
    cache.set('b', '2');
    cache.set('c', '3');
    cache.get('a'); // refresh 'a'
    cache.set('d', '4'); // evicts 'b', not 'a'
    assert.equal(cache.getStats().l1Size, 3);
    assert.equal(cache.get('a'), '1');
    assert.equal(cache.get('d'), '4');
  });
});

// ---------------------------------------------------------------------------
// has / delete / clear
// ---------------------------------------------------------------------------

describe('MultiLevelCache - has, delete, clear', () => {
  let cache: MultiLevelCache<string>;
  beforeEach(() => {
    cache = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 20, ...LONG_TTL });
  });

  it('has() returns true for valid, false for missing', () => {
    cache.set('exists', 'yes');
    assert.equal(cache.has('exists'), true);
    assert.equal(cache.has('ghost'), false);
  });

  it('has() returns false for expired entry', () => {
    const sc = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 10, l1TtlMs: 1, l2TtlMs: 1 });
    sc.set('tmp', 'v');
    busyWait(5);
    assert.equal(sc.has('tmp'), false);
  });

  it('delete() removes from both L1 and L2', () => {
    cache.set('d', 'v');
    assert.equal(cache.delete('d'), true);
    assert.equal(cache.get('d'), undefined);
    assert.equal(cache.getStats().l1Size, 0);
    assert.equal(cache.getStats().l2Size, 0);
  });

  it('delete() returns false for non-existent key', () => {
    assert.equal(cache.delete('nothing'), false);
  });

  it('clear() empties both caches', () => {
    cache.set('a', '1');
    cache.set('b', '2');
    cache.clear();
    assert.equal(cache.getStats().l1Size, 0);
    assert.equal(cache.getStats().l2Size, 0);
    assert.equal(cache.get('a'), undefined);
  });
});

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

describe('MultiLevelCache - getStats()', () => {
  it('returns correct hit/miss/eviction counts and hit rates', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 2, l2MaxEntries: 4, ...LONG_TTL });
    cache.set('a', '1');
    cache.set('b', '2');
    cache.get('a'); // L1 hit
    cache.get('a'); // L1 hit
    cache.get('x'); // miss

    const s = cache.getStats();
    assert.equal(s.l1Hits, 2);
    assert.equal(s.misses, 1);
    assert.ok(s.hitRate > 0.6 && s.hitRate < 0.7, `hitRate was ${s.hitRate}`);
    assert.equal(typeof s.l1HitRate, 'number');
    assert.equal(typeof s.evictions, 'number');
  });
});

// ---------------------------------------------------------------------------
// makeKey
// ---------------------------------------------------------------------------

describe('MultiLevelCache.makeKey()', () => {
  it('generates deterministic keys for same tool+args', () => {
    const k1 = MultiLevelCache.makeKey('search', { q: 'hello', limit: 10 });
    const k2 = MultiLevelCache.makeKey('search', { q: 'hello', limit: 10 });
    assert.equal(k1, k2);
  });

  it('generates different keys for different tool names', () => {
    const k1 = MultiLevelCache.makeKey('search', { q: 'hello' });
    const k2 = MultiLevelCache.makeKey('otherTool', { q: 'hello' });
    assert.notEqual(k1, k2);
    // Prefix is tool name
    assert.ok(k1.startsWith('search:'));
    assert.ok(k2.startsWith('otherTool:'));
  });
});

// ---------------------------------------------------------------------------
// maxEntrySizeBytes
// ---------------------------------------------------------------------------

describe('MultiLevelCache - maxEntrySizeBytes', () => {
  it('entries exceeding limit are not cached', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 10, ...LONG_TTL, maxEntrySizeBytes: 10 });
    cache.set('big', 'this is a very long string value that exceeds the limit');
    assert.equal(cache.get('big'), undefined);
    assert.equal(cache.getStats().l1Size, 0);
  });

  it('entries within limit are cached normally', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 10, ...LONG_TTL, maxEntrySizeBytes: 1000 });
    cache.set('small', 'ok');
    assert.equal(cache.get('small'), 'ok');
  });

  it('maxEntrySizeBytes=0 means unlimited', () => {
    const cache = new MultiLevelCache<string>({ l1MaxEntries: 10, l2MaxEntries: 10, ...LONG_TTL, maxEntrySizeBytes: 0 });
    const big = 'x'.repeat(10_000);
    cache.set('huge', big);
    assert.equal(cache.get('huge'), big);
  });
});
