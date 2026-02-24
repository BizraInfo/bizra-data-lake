/**
 * Tests for PeerDiscovery — peer management, bootstrap discovery,
 * stale peer cleanup, maxPeers eviction, and event emissions.
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  PeerDiscovery,
  PeerNode,
  PeerState,
  PeerCapabilities,
} from '../peer-discovery';

// -- Helpers ---------------------------------------------------------------

function makePeer(overrides: Partial<PeerNode> = {}): PeerNode {
  return {
    id: overrides.id ?? 'peer-' + Math.random().toString(36).slice(2, 8),
    name: overrides.name ?? 'test-peer',
    addresses: overrides.addresses ?? ['127.0.0.1:9000'],
    publicKey: overrides.publicKey ?? 'deadbeef',
    capabilities: overrides.capabilities ?? {
      tiers: ['LOCAL'],
      poolEnabled: true,
      maxConcurrent: 10,
      tasks: ['chat'],
    },
    lastSeen: overrides.lastSeen ?? new Date(),
    latencyMs: overrides.latencyMs ?? 0,
    state: overrides.state ?? PeerState.CONNECTED,
  };
}

// -- Constructor + Config --------------------------------------------------

describe('PeerDiscovery - Constructor', () => {
  it('should apply default config when none provided', () => {
    const pd = new PeerDiscovery();
    const stats = pd.getStats();
    assert.equal(stats.totalPeers, 0);
    assert.equal(stats.connectedPeers, 0);
  });

  it('should merge partial config with defaults', () => {
    const pd = new PeerDiscovery({ maxPeers: 5 });
    // Adding 6 peers should evict one
    for (let i = 0; i < 6; i++) {
      pd.addPeer(makePeer({ id: `p${i}`, lastSeen: new Date(Date.now() - i * 1000) }));
    }
    assert.ok(pd.getStats().totalPeers <= 5);
  });
});

// -- addPeer / getPeer / removePeer ----------------------------------------

describe('PeerDiscovery - Peer Management', () => {
  let pd: PeerDiscovery;

  beforeEach(() => {
    pd = new PeerDiscovery({ maxPeers: 10, enableMdns: false });
  });

  it('addPeer() stores a peer', () => {
    const peer = makePeer({ id: 'p1' });
    pd.addPeer(peer);
    assert.equal(pd.getPeer('p1')?.id, 'p1');
    assert.equal(pd.getStats().totalPeers, 1);
  });

  it('getPeer() returns undefined for unknown ID', () => {
    assert.equal(pd.getPeer('ghost'), undefined);
  });

  it('removePeer() deletes a peer', () => {
    pd.addPeer(makePeer({ id: 'p1' }));
    pd.removePeer('p1');
    assert.equal(pd.getPeer('p1'), undefined);
    assert.equal(pd.getStats().totalPeers, 0);
  });

  it('removePeer() is a no-op for unknown peer', () => {
    pd.removePeer('nonexistent'); // Should not throw
    assert.equal(pd.getStats().totalPeers, 0);
  });

  it('getPeers() returns all peers', () => {
    pd.addPeer(makePeer({ id: 'a' }));
    pd.addPeer(makePeer({ id: 'b' }));
    assert.equal(pd.getPeers().length, 2);
  });
});

// -- Filtering -------------------------------------------------------------

describe('PeerDiscovery - Filtering', () => {
  let pd: PeerDiscovery;

  beforeEach(() => {
    pd = new PeerDiscovery({ enableMdns: false });
    pd.addPeer(makePeer({ id: 'c1', state: PeerState.CONNECTED, capabilities: { tiers: ['LOCAL'], poolEnabled: true, maxConcurrent: 5, tasks: ['chat'] } }));
    pd.addPeer(makePeer({ id: 'c2', state: PeerState.CONNECTED, capabilities: { tiers: ['LOCAL'], poolEnabled: false, maxConcurrent: 5, tasks: ['chat'] } }));
    pd.addPeer(makePeer({ id: 'd1', state: PeerState.DISCONNECTED }));
  });

  it('getConnectedPeers() filters by CONNECTED state', () => {
    assert.equal(pd.getConnectedPeers().length, 2);
  });

  it('getPoolPeers() filters by poolEnabled AND CONNECTED', () => {
    const pool = pd.getPoolPeers();
    assert.equal(pool.length, 1);
    assert.equal(pool[0].id, 'c1');
  });
});

// -- maxPeers eviction -----------------------------------------------------

describe('PeerDiscovery - maxPeers Eviction', () => {
  it('evicts oldest peer when maxPeers exceeded', () => {
    const pd = new PeerDiscovery({ maxPeers: 3, enableMdns: false });

    // Add peers with staggered timestamps (p0 = oldest)
    for (let i = 0; i < 4; i++) {
      pd.addPeer(makePeer({
        id: `p${i}`,
        lastSeen: new Date(Date.now() - (3 - i) * 1000),
      }));
    }

    assert.equal(pd.getStats().totalPeers, 3);
    // p0 (oldest) should have been evicted
    assert.equal(pd.getPeer('p0'), undefined);
    assert.ok(pd.getPeer('p3'));
  });
});

// -- updatePeer ------------------------------------------------------------

describe('PeerDiscovery - updatePeer', () => {
  it('updates fields and refreshes lastSeen', () => {
    const pd = new PeerDiscovery({ enableMdns: false });
    const oldDate = new Date(Date.now() - 60_000);
    pd.addPeer(makePeer({ id: 'u1', lastSeen: oldDate, latencyMs: 100 }));

    pd.updatePeer('u1', { latencyMs: 50 });

    const peer = pd.getPeer('u1')!;
    assert.equal(peer.latencyMs, 50);
    assert.ok(peer.lastSeen.getTime() > oldDate.getTime());
  });

  it('is a no-op for unknown peer', () => {
    const pd = new PeerDiscovery({ enableMdns: false });
    pd.updatePeer('ghost', { latencyMs: 10 }); // Should not throw
  });
});

// -- Events ----------------------------------------------------------------

describe('PeerDiscovery - Events', () => {
  it('emits peer-added on addPeer()', () => {
    const pd = new PeerDiscovery({ enableMdns: false });
    let emitted = false;
    pd.on('peer-added', () => { emitted = true; });
    pd.addPeer(makePeer({ id: 'e1' }));
    assert.equal(emitted, true);
  });

  it('emits peer-removed on removePeer()', () => {
    const pd = new PeerDiscovery({ enableMdns: false });
    pd.addPeer(makePeer({ id: 'e2' }));
    let emitted = false;
    pd.on('peer-removed', () => { emitted = true; });
    pd.removePeer('e2');
    assert.equal(emitted, true);
  });

  it('emits peer-updated on updatePeer()', () => {
    const pd = new PeerDiscovery({ enableMdns: false });
    pd.addPeer(makePeer({ id: 'e3' }));
    let emitted = false;
    pd.on('peer-updated', () => { emitted = true; });
    pd.updatePeer('e3', { name: 'renamed' });
    assert.equal(emitted, true);
  });
});

// -- Lifecycle: start / stop -----------------------------------------------

describe('PeerDiscovery - Lifecycle', () => {
  let pd: PeerDiscovery;

  beforeEach(() => {
    pd = new PeerDiscovery({
      enableMdns: false,
      bootstrapNodes: ['192.168.1.100:9000'],
      discoveryIntervalMs: 60_000, // Long interval to avoid timing issues
    });
  });

  afterEach(async () => {
    await pd.stop();
  });

  it('start() emits started event', async () => {
    let emitted = false;
    pd.on('started', () => { emitted = true; });
    await pd.start();
    assert.equal(emitted, true);
  });

  it('start() discovers bootstrap peers', async () => {
    await pd.start();
    assert.ok(pd.getStats().totalPeers >= 1, 'Should discover bootstrap peer');
  });

  it('start() is idempotent', async () => {
    await pd.start();
    await pd.start(); // Should not throw or double-start
    assert.equal(pd.getStats().totalPeers >= 1, true);
  });

  it('stop() emits stopped event', async () => {
    await pd.start();
    let emitted = false;
    pd.on('stopped', () => { emitted = true; });
    await pd.stop();
    assert.equal(emitted, true);
  });
});

// -- getStats --------------------------------------------------------------

describe('PeerDiscovery - getStats', () => {
  it('returns correct shape', () => {
    const pd = new PeerDiscovery({ enableMdns: false });
    const stats = pd.getStats();
    assert.equal(typeof stats.totalPeers, 'number');
    assert.equal(typeof stats.connectedPeers, 'number');
    assert.equal(typeof stats.poolPeers, 'number');
    assert.equal(typeof stats.discoveryRuns, 'number');
    assert.ok(stats.lastDiscoveryAt instanceof Date);
  });
});
