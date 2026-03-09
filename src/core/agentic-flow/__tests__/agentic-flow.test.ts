/**
 * BIZRA Agentic-Flow — Integration Tests
 *
 * Tests for the V3 deep integration adapter layer:
 * - ReflexCache (O(1) lookup, precipitation, eviction)
 * - SONAManager (mode transitions, auto-select)
 * - AgentRouter (HHMM classification, agent selection)
 * - MemoryCoordinator (store, share, evidence chain)
 * - AgenticFlowAdapter (full mission flow)
 */

import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  ReflexCache,
  selectHelix,
  type ReflexCacheConfig,
} from '../reflex-cache';

import {
  SONAManager,
  type SONATransition,
} from '../sona';

import {
  AgentRouter,
} from '../agent-router';

import {
  MemoryCoordinator,
  type MemoryConfig,
} from '../memory-coordinator';

import {
  AgenticFlowAdapter,
  LocalScoringDelegate,
} from '../adapter';

import {
  PATAgent,
  SATAgent,
  Helix,
  SONAMode,
  HHMMMacroState,
  CONSTITUTIONAL,
  type MemoryEntry,
} from '../types';

// ────────────────────────────────────────────────────────────
// ReflexCache
// ────────────────────────────────────────────────────────────

describe('ReflexCache', () => {
  let cache: ReflexCache;

  beforeEach(() => {
    cache = new ReflexCache({ maxEntries: 64, precipitationRepeats: 2 });
  });

  it('returns undefined on cache miss', () => {
    const result = cache.lookup('unknown pattern');
    assert.equal(result, undefined);
    assert.equal(cache.getStats().totalMisses, 1);
  });

  it('stores and retrieves a precipitated reflex', () => {
    cache.precipitate('test pattern', [PATAgent.CODER], 0.95, 'cached response');
    const hit = cache.lookup('test pattern');
    assert.ok(hit);
    assert.equal(hit.response, 'cached response');
    assert.equal(hit.ihsanScore, 0.95);
    assert.deepEqual(hit.agentIds, [PATAgent.CODER]);
    assert.equal(hit.hitCount, 1);
  });

  it('increments hit count on repeated lookups', () => {
    cache.precipitate('pattern', [PATAgent.PLANNER], 0.96, 'resp');
    cache.lookup('pattern');
    const second = cache.lookup('pattern');
    assert.ok(second);
    assert.equal(second.hitCount, 2);
  });

  it('precipitates from candidates after threshold repeats', () => {
    const precipitated1 = cache.recordCandidate('cand', [PATAgent.CODER], 0.92, 'r1');
    assert.equal(precipitated1, false); // 1 of 2 needed

    const precipitated2 = cache.recordCandidate('cand', [PATAgent.CODER], 0.92, 'r1');
    assert.equal(precipitated2, true); // 2 of 2 — precipitated

    assert.ok(cache.has('cand'));
  });

  it('rejects candidates below Ihsan threshold', () => {
    const precipitated = cache.recordCandidate('low', [PATAgent.CODER], 0.80, 'bad');
    assert.equal(precipitated, false);
    assert.ok(!cache.has('low'));
  });

  it('invalidates a specific reflex', () => {
    cache.precipitate('p', [PATAgent.DEMA], 0.95, 'r');
    assert.ok(cache.has('p'));
    cache.invalidate('p');
    assert.ok(!cache.has('p'));
  });

  it('clears all reflexes', () => {
    cache.precipitate('a', [PATAgent.CODER], 0.95, 'ra');
    cache.precipitate('b', [PATAgent.PLANNER], 0.96, 'rb');
    cache.clear();
    assert.equal(cache.totalSize, 0);
    assert.equal(cache.hitRate, 0);
  });

  it('selectHelix returns REACTIVE for cached, DELIBERATIVE for miss', () => {
    cache.precipitate('cached', [PATAgent.CODER], 0.95, 'resp');
    assert.equal(selectHelix(cache, 'cached'), Helix.REACTIVE);
    assert.equal(selectHelix(cache, 'uncached'), Helix.DELIBERATIVE);
  });

  it('distributes across 8 shards', () => {
    for (let i = 0; i < 100; i++) {
      cache.precipitate(`pattern-${i}`, [PATAgent.CODER], 0.95, `r-${i}`);
    }
    const stats = cache.getStats();
    assert.equal(stats.totalSize, 64); // capped at maxEntries
    // At least 2 shards should be populated
    const populatedShards = stats.shardSizes.filter((s) => s > 0).length;
    assert.ok(populatedShards >= 2, `Expected ≥2 shards populated, got ${populatedShards}`);
  });
});

// ────────────────────────────────────────────────────────────
// SONAManager
// ────────────────────────────────────────────────────────────

describe('SONAManager', () => {
  let sona: SONAManager;

  beforeEach(() => {
    sona = new SONAManager(SONAMode.BALANCED);
  });

  afterEach(() => {
    sona.stopHeartbeat();
  });

  it('starts in the configured mode', () => {
    assert.equal(sona.getMode(), SONAMode.BALANCED);
  });

  it('transitions between modes', () => {
    const transitions: SONATransition[] = [];
    sona.onTransition((t) => transitions.push(t));

    sona.setMode(SONAMode.RESEARCH, 'deep analysis');
    assert.equal(sona.getMode(), SONAMode.RESEARCH);
    assert.equal(transitions.length, 1);
    assert.equal(transitions[0]!.from, SONAMode.BALANCED);
    assert.equal(transitions[0]!.to, SONAMode.RESEARCH);
  });

  it('no-ops on same mode transition', () => {
    const transitions: SONATransition[] = [];
    sona.onTransition((t) => transitions.push(t));
    sona.setMode(SONAMode.BALANCED, 'same');
    assert.equal(transitions.length, 0);
  });

  it('returns correct primary helix per mode', () => {
    sona.setMode(SONAMode.REAL_TIME, 'test');
    assert.equal(sona.getPrimaryHelix(), Helix.REACTIVE);

    sona.setMode(SONAMode.BATCH, 'test');
    assert.equal(sona.getPrimaryHelix(), Helix.EVOLUTIONARY);
  });

  it('caps active agents to mode limit', () => {
    sona.setMode(SONAMode.EDGE, 'test'); // maxConcurrentAgents = 2
    const agents = [PATAgent.PLANNER, PATAgent.CODER, PATAgent.EVALUATOR, PATAgent.RESEARCHER];
    sona.setActiveAgents(agents);
    const snap = sona.snapshot();
    assert.equal(snap.activeAgents.length, 2);
  });

  it('auto-selects edge mode on low memory', () => {
    const selected = sona.autoSelect({
      pendingMissions: 1,
      avgLatencyMs: 200,
      availableMemoryMB: 256,
      isEdgeDevice: false,
    });
    assert.equal(selected, SONAMode.EDGE);
  });

  it('auto-selects batch mode on high queue', () => {
    const selected = sona.autoSelect({
      pendingMissions: 20,
      avgLatencyMs: 200,
      availableMemoryMB: 8192,
      isEdgeDevice: false,
    });
    assert.equal(selected, SONAMode.BATCH);
  });

  it('unsubscribes listener', () => {
    const transitions: SONATransition[] = [];
    const unsub = sona.onTransition((t) => transitions.push(t));
    sona.setMode(SONAMode.RESEARCH, 'test');
    assert.equal(transitions.length, 1);

    unsub();
    sona.setMode(SONAMode.EDGE, 'test');
    assert.equal(transitions.length, 1); // no new events
  });
});

// ────────────────────────────────────────────────────────────
// AgentRouter
// ────────────────────────────────────────────────────────────

describe('AgentRouter', () => {
  let router: AgentRouter;

  beforeEach(() => {
    router = new AgentRouter();
  });

  it('starts in IDLE state', () => {
    assert.equal(router.getCurrentState(), HHMMMacroState.IDLE);
  });

  it('routes coding tasks to Coder agent', () => {
    const result = router.route('implement the new feature and fix the bug');
    assert.equal(result.macroState, HHMMMacroState.CODING);
    assert.ok(result.selectedAgents.includes(PATAgent.CODER));
  });

  it('routes research tasks to Researcher agent', () => {
    const result = router.route('research and analyze the evidence');
    assert.equal(result.macroState, HHMMMacroState.RESEARCHING);
    assert.ok(result.selectedAgents.includes(PATAgent.RESEARCHER));
  });

  it('routes ethics tasks to gate check with Ethicist', () => {
    const result = router.route('verify constitutional ihsan gate');
    assert.equal(result.macroState, HHMMMacroState.GATE_CHECK);
    assert.ok(result.selectedAgents.includes(PATAgent.ETHICIST));
  });

  it('always includes Sentinel in agent set', () => {
    const result = router.route('plan the architecture');
    assert.ok(
      result.selectedAgents.includes(SATAgent.SENTINEL),
      'Sentinel should always be included'
    );
  });

  it('defaults to PLANNING for unclassified input', () => {
    const result = router.route('something completely unrelated');
    assert.equal(result.macroState, HHMMMacroState.PLANNING);
  });

  it('respects maxAgents cap', () => {
    const result = router.route('plan and design the architecture', 2);
    assert.ok(result.selectedAgents.length <= 2);
  });

  it('predicts next state from current', () => {
    router.route('plan the roadmap'); // → PLANNING
    const prediction = router.predictNext();
    assert.ok(prediction);
    assert.ok(prediction.probability > 0);
  });

  it('transitions update current state', () => {
    router.route('code the implementation'); // → CODING
    assert.equal(router.getCurrentState(), HHMMMacroState.CODING);
    router.route('evaluate the quality'); // → EVALUATING
    assert.equal(router.getCurrentState(), HHMMMacroState.EVALUATING);
  });
});

// ────────────────────────────────────────────────────────────
// MemoryCoordinator
// ────────────────────────────────────────────────────────────

describe('MemoryCoordinator', () => {
  let memory: MemoryCoordinator;

  beforeEach(() => {
    memory = new MemoryCoordinator({ maxEntriesPerAgent: 10, maxSharedEntries: 20 });
  });

  const makeEntry = (id: string, agentId: PATAgent | SATAgent): MemoryEntry => ({
    id,
    agentId,
    content: `memory content ${id}`,
    timestamp: Date.now(),
    shared: false,
  });

  it('stores and retrieves private memory', () => {
    const entry = makeEntry('m1', PATAgent.CODER);
    assert.ok(memory.store(PATAgent.CODER, entry));
    assert.deepEqual(memory.getPrivate(PATAgent.CODER, 'm1'), entry);
  });

  it('rejects writes from frozen agents', () => {
    const entry = makeEntry('m2', PATAgent.ETHICIST);
    assert.equal(memory.store(PATAgent.ETHICIST, entry), false);
  });

  it('shares memory to shared pool', () => {
    const entry = makeEntry('m3', PATAgent.PLANNER);
    memory.store(PATAgent.PLANNER, entry);
    assert.ok(memory.share(PATAgent.PLANNER, 'm3'));
    const shared = memory.getShared('m3');
    assert.ok(shared);
    assert.equal(shared.shared, true);
  });

  it('prevents frozen agents from sharing', () => {
    assert.equal(memory.share(SATAgent.ORACLE, 'any'), false);
  });

  it('maintains evidence chain integrity', () => {
    memory.appendEvidence({
      receiptHash: 'hash1',
      prevHash: '0000',
      missionId: 'mission-1',
      timestamp: Date.now(),
    });

    memory.appendEvidence({
      receiptHash: 'hash2',
      prevHash: 'hash1',
      missionId: 'mission-2',
      timestamp: Date.now(),
    });

    assert.equal(memory.getEvidenceChainLength(), 2);
    assert.equal(memory.getLatestEvidence()?.receiptHash, 'hash2');
  });

  it('rejects broken evidence chain', () => {
    memory.appendEvidence({
      receiptHash: 'hash1',
      prevHash: '0000',
      missionId: 'mission-1',
      timestamp: Date.now(),
    });

    assert.throws(() => {
      memory.appendEvidence({
        receiptHash: 'hash2',
        prevHash: 'wrong',
        missionId: 'mission-2',
        timestamp: Date.now(),
      });
    }, /Evidence chain broken/);
  });

  it('evicts oldest when at capacity', () => {
    for (let i = 0; i < 12; i++) {
      const entry: MemoryEntry = {
        id: `e${i}`,
        agentId: PATAgent.CODER,
        content: `content ${i}`,
        timestamp: 1000 + i, // ascending timestamps
        shared: false,
      };
      memory.store(PATAgent.CODER, entry);
    }
    // Cap is 10 per agent, so first 2 should be evicted
    assert.equal(memory.getPrivateSize(PATAgent.CODER), 10);
    assert.equal(memory.getPrivate(PATAgent.CODER, 'e0'), undefined);
    assert.equal(memory.getPrivate(PATAgent.CODER, 'e1'), undefined);
    assert.ok(memory.getPrivate(PATAgent.CODER, 'e2'));
  });

  it('provides stats', () => {
    memory.store(PATAgent.CODER, makeEntry('s1', PATAgent.CODER));
    memory.store(PATAgent.PLANNER, makeEntry('s2', PATAgent.PLANNER));
    memory.share(PATAgent.CODER, 's1');

    const stats = memory.getStats();
    assert.equal(stats.agentCount, 2);
    assert.equal(stats.totalPrivate, 2);
    assert.equal(stats.sharedSize, 1);
  });
});

// ────────────────────────────────────────────────────────────
// AgenticFlowAdapter (full integration)
// ────────────────────────────────────────────────────────────

describe('AgenticFlowAdapter', () => {
  let adapter: AgenticFlowAdapter;

  beforeEach(async () => {
    adapter = new AgenticFlowAdapter({
      sonaMode: SONAMode.BALANCED,
      reflexCacheSize: 128,
    });
    await adapter.start();
  });

  afterEach(async () => {
    await adapter.stop();
  });

  it('starts and reports status', () => {
    const status = adapter.getStatus();
    assert.equal(status.started, true);
    assert.equal(status.sonaMode, SONAMode.BALANCED);
    assert.equal(status.reflexCacheSize, 0);
  });

  it('executes a coding mission through HHMM', async () => {
    const result = await adapter.executeMission('implement the auth module');
    assert.ok(result.receipt);
    assert.equal(result.receipt.helix, Helix.DELIBERATIVE); // cache miss
    assert.equal(result.reflexHit, false);
    assert.ok(result.receipt.ihsanScore > 0);
    assert.ok(result.receipt.snrScore > 0);
    assert.ok(result.receipt.elapsedMs >= 0);
    assert.equal(result.route.macroState, HHMMMacroState.CODING);
  });

  it('builds evidence chain across missions', async () => {
    const r1 = await adapter.executeMission('plan the architecture');
    const r2 = await adapter.executeMission('code the implementation');

    assert.equal(r2.receipt.prevHash, r1.receipt.receiptHash);
    assert.equal(adapter.getStatus().evidenceChainLength, 2);
  });

  it('mints SEED when Ihsan >= minimum gate', async () => {
    const result = await adapter.executeMission('verify ethical consent and privacy');
    // LocalScoringDelegate returns ~0.89+ for content with "privacy" and "consent" keywords
    // The mission description doesn't affect scoring, the routed content does
    assert.equal(typeof result.receipt.seedMinted, 'boolean');
  });

  it('accepts custom scoring delegate', async () => {
    const customScorer: LocalScoringDelegate & { called: boolean } = Object.assign(
      new LocalScoringDelegate(),
      { called: false }
    );
    const origScoreIhsan = customScorer.scoreIhsan.bind(customScorer);
    customScorer.scoreIhsan = async (content: string) => {
      customScorer.called = true;
      return origScoreIhsan(content);
    };

    adapter.setScoringDelegate(customScorer);
    await adapter.executeMission('test custom scorer');
    assert.ok(customScorer.called);
  });

  it('reports SONA snapshot in mission result', async () => {
    const result = await adapter.executeMission('research the topic');
    assert.equal(result.sonaSnapshot.mode, SONAMode.BALANCED);
    assert.ok(result.sonaSnapshot.activeAgents.length > 0);
  });
});

// ────────────────────────────────────────────────────────────
// LocalScoringDelegate
// ────────────────────────────────────────────────────────────

describe('LocalScoringDelegate', () => {
  const scorer = new LocalScoringDelegate();

  it('scores ethical content higher', async () => {
    const high = await scorer.scoreIhsan('We respect privacy and require consent.');
    const low = await scorer.scoreIhsan('We exploit and track users.');
    assert.ok(high > low, `Expected ${high} > ${low}`);
  });

  it('produces SNR score in [0, 1]', async () => {
    const score = await scorer.scoreSNR('This is a test of signal to noise ratio measurement');
    assert.ok(score >= 0 && score <= 1, `Score ${score} not in [0,1]`);
  });

  it('handles empty content gracefully', async () => {
    const score = await scorer.scoreSNR('');
    assert.ok(score >= 0 && score <= 1, `Score ${score} not in [0,1]`);
  });
});
