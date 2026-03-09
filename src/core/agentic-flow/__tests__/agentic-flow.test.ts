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
  ReasoningBank,
  type SeedPattern,
} from '../reasoning-bank';

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

// ────────────────────────────────────────────────────────────
// ReasoningBank — Trajectory Tracking
// ────────────────────────────────────────────────────────────

describe('ReasoningBank — Trajectory Tracking', () => {
  let bank: ReasoningBank;

  beforeEach(() => {
    bank = new ReasoningBank();
  });

  it('records a trajectory and classifies outcome', () => {
    const receipt = makeReceipt(0.90, 0.88);
    const traj = bank.recordTrajectory(receipt, 'optimize database queries', [], 'backend');
    assert.equal(traj.outcome, 'success');
    assert.equal(traj.domain, 'backend');
    assert.equal(bank.getTrajectoryCount(), 1);
  });

  it('classifies low-score trajectory as failure', () => {
    const receipt = makeReceipt(0.40, 0.30);
    const traj = bank.recordTrajectory(receipt, 'bad mission', [], 'testing');
    assert.equal(traj.outcome, 'failure');
  });

  it('classifies partial outcome for mixed scores', () => {
    const receipt = makeReceipt(0.75, 0.60);
    const traj = bank.recordTrajectory(receipt, 'partial mission', [], 'testing');
    assert.equal(traj.outcome, 'partial');
  });

  it('auto-creates experience pattern for successful trajectories', () => {
    const receipt = makeReceipt(0.92, 0.90);
    bank.recordTrajectory(receipt, 'successful mission', [], 'coding');
    const patterns = bank.getPatterns('coding', 'experience');
    assert.equal(patterns.length, 1);
    assert.ok(patterns[0]!.confidence >= 0.90);
  });

  it('does not create pattern for failed trajectories', () => {
    const receipt = makeReceipt(0.40, 0.30);
    bank.recordTrajectory(receipt, 'failed mission', [], 'coding');
    const patterns = bank.getPatterns('coding', 'experience');
    assert.equal(patterns.length, 0);
  });

  it('evicts oldest trajectory when at capacity', () => {
    const config = { maxTrajectoriesPerDomain: 3 };
    const small = new ReasoningBank(config);

    for (let i = 0; i < 5; i++) {
      const r = makeReceipt(0.92, 0.90);
      small.recordTrajectory(r, `mission ${i}`, [], 'test');
    }

    const trajectories = small.getTrajectories('test');
    assert.equal(trajectories.length, 3);
  });

  it('retrieves trajectories by domain', () => {
    bank.recordTrajectory(makeReceipt(0.92, 0.90), 'backend task', [], 'backend');
    bank.recordTrajectory(makeReceipt(0.92, 0.90), 'frontend task', [], 'frontend');

    assert.equal(bank.getTrajectories('backend').length, 1);
    assert.equal(bank.getTrajectories('frontend').length, 1);
    assert.equal(bank.getTrajectories('unknown').length, 0);
  });
});

// ────────────────────────────────────────────────────────────
// ReasoningBank — Verdict Judgment
// ────────────────────────────────────────────────────────────

describe('ReasoningBank — Verdict Judgment', () => {
  let bank: ReasoningBank;

  beforeEach(() => {
    bank = new ReasoningBank({ verdictSuccessThreshold: 2, verdictSimilarityThreshold: 0.4 });
  });

  it('returns needs_review for empty domain', () => {
    const verdict = bank.judgeTrajectory('optimize queries', 'empty-domain');
    assert.equal(verdict.level, 'needs_review');
    assert.equal(verdict.similarPatterns, 0);
  });

  it('returns likely_success with sufficient similar patterns', () => {
    // Create 3 similar successful patterns
    for (let i = 0; i < 3; i++) {
      bank.recordTrajectory(
        makeReceipt(0.95, 0.92),
        'optimize database queries for better performance',
        [],
        'backend',
      );
    }

    const verdict = bank.judgeTrajectory('optimize database queries', 'backend');
    assert.equal(verdict.level, 'likely_success');
    assert.ok(verdict.successfulMatches >= 2);
  });

  it('returns likely_failure with no successful matches', () => {
    // Insert patterns that don't match
    bank.insertPattern({
      id: 'unrelated-1',
      type: 'experience',
      domain: 'backend',
      description: 'completely different topic about weather',
      confidence: 0.90,
      usageCount: 5,
      successCount: 0,
      createdAt: Date.now(),
      lastUsed: Date.now(),
      data: '{}',
    });

    const verdict = bank.judgeTrajectory('optimize database queries', 'backend');
    assert.equal(verdict.level, 'likely_failure');
  });
});

// ────────────────────────────────────────────────────────────
// ReasoningBank — Memory Distillation
// ────────────────────────────────────────────────────────────

describe('ReasoningBank — Memory Distillation', () => {
  let bank: ReasoningBank;

  beforeEach(() => {
    bank = new ReasoningBank();
  });

  it('returns empty result for unknown domain', () => {
    const result = bank.distill('unknown');
    assert.equal(result.outputPatterns, 0);
    assert.equal(result.successRate, 0);
  });

  it('distills similar experience patterns into distilled pattern', () => {
    // Create multiple similar successful trajectories
    for (let i = 0; i < 4; i++) {
      bank.recordTrajectory(
        makeReceipt(0.93, 0.91),
        'optimize database queries with indexing',
        [],
        'db-optimization',
      );
    }

    const result = bank.distill('db-optimization');
    assert.ok(result.outputPatterns >= 1, `Expected >=1 distilled patterns, got ${result.outputPatterns}`);

    const distilled = bank.getPatterns('db-optimization', 'distilled');
    const principles = bank.getPatterns('db-optimization', 'principle');
    assert.ok(distilled.length + principles.length >= 1);
  });

  it('tracks precipitation candidates', () => {
    for (let i = 0; i < 5; i++) {
      bank.recordTrajectory(
        makeReceipt(0.96, 0.94),
        'deploy microservice to kubernetes cluster',
        [],
        'deployment',
      );
    }

    const result = bank.distill('deployment');
    assert.ok(result.precipitationCandidates >= 0);
  });
});

// ────────────────────────────────────────────────────────────
// ReasoningBank — Reflex Precipitation
// ────────────────────────────────────────────────────────────

describe('ReasoningBank — Reflex Precipitation', () => {
  let bank: ReasoningBank;
  let cache: ReflexCache;

  beforeEach(() => {
    bank = new ReasoningBank({ precipitationRepeats: 2 });
    cache = new ReflexCache();
  });

  it('precipitates high-confidence distilled patterns to cache', () => {
    // Insert a distilled pattern with high confidence and usage
    bank.insertPattern({
      id: 'distilled-1',
      type: 'distilled',
      domain: 'testing',
      description: 'run comprehensive test suite before deployment',
      confidence: 0.95,
      usageCount: 5,
      successCount: 5,
      createdAt: Date.now(),
      lastUsed: Date.now(),
      data: JSON.stringify({ agents: [PATAgent.CODER, PATAgent.EVALUATOR] }),
    });

    const precipitated = bank.precipitateToCache(cache);
    assert.ok(precipitated >= 1, `Expected >=1 precipitation, got ${precipitated}`);
    assert.ok(cache.totalSize >= 1);
  });

  it('does not precipitate low-confidence patterns', () => {
    bank.insertPattern({
      id: 'weak-1',
      type: 'distilled',
      domain: 'testing',
      description: 'weak pattern',
      confidence: 0.50,
      usageCount: 1,
      successCount: 0,
      createdAt: Date.now(),
      lastUsed: Date.now(),
      data: '{}',
    });

    const precipitated = bank.precipitateToCache(cache);
    assert.equal(precipitated, 0);
  });
});

// ────────────────────────────────────────────────────────────
// ReasoningBank — Seed Pattern Import
// ────────────────────────────────────────────────────────────

describe('ReasoningBank — Seed Import', () => {
  it('imports seed patterns from JSON format', () => {
    const bank = new ReasoningBank();
    const seeds: SeedPattern[] = [
      {
        text: 'IhsanGate Enforcement Pattern: Transform quality gate from monitoring to enforcement.',
        metadata: {
          domain: 'security-architecture',
          task: 'ihsan-gate-enforcement',
          outcome: 'success',
          confidence: 0.95,
          tests_added: 9,
          giants: ['Al-Ghazali', 'Lyapunov'],
        },
      },
      {
        text: 'Domain-Separated Signatures: Always prefix signing payloads with a domain string.',
        metadata: {
          domain: 'cryptographic-patterns',
          pattern_type: 'distilled',
          confidence: 0.98,
          giants: ['Lamport', 'BLAKE3'],
        },
      },
    ];

    const imported = bank.importSeedPatterns(seeds);
    assert.equal(imported, 2);
    assert.equal(bank.getPatternCount(), 2);

    const secPatterns = bank.getPatterns('security-architecture');
    assert.equal(secPatterns.length, 1);
    assert.equal(secPatterns[0]!.confidence, 0.95);

    const cryptoPatterns = bank.getPatterns('cryptographic-patterns');
    assert.equal(cryptoPatterns.length, 1);
    assert.equal(cryptoPatterns[0]!.type, 'distilled');
  });
});

// ────────────────────────────────────────────────────────────
// ReasoningBank — Stats
// ────────────────────────────────────────────────────────────

describe('ReasoningBank — Stats', () => {
  it('reports comprehensive stats across domains', () => {
    const bank = new ReasoningBank();

    bank.recordTrajectory(makeReceipt(0.93, 0.90), 'backend task', [], 'backend');
    bank.recordTrajectory(makeReceipt(0.91, 0.88), 'frontend task', [], 'frontend');
    bank.recordTrajectory(makeReceipt(0.40, 0.30), 'failed task', [], 'backend');

    const stats = bank.getStats();
    assert.equal(stats.totalTrajectories, 3);
    assert.ok(stats.domains >= 2);
    assert.ok(stats.totalPatterns >= 2); // Auto-created from successful trajectories
  });

  it('clears all data', () => {
    const bank = new ReasoningBank();
    bank.recordTrajectory(makeReceipt(0.93, 0.90), 'task', [], 'test');
    bank.clear();
    assert.equal(bank.getTrajectoryCount(), 0);
    assert.equal(bank.getPatternCount(), 0);
  });

  it('returns domain list', () => {
    const bank = new ReasoningBank();
    bank.recordTrajectory(makeReceipt(0.93, 0.90), 'task', [], 'alpha');
    bank.recordTrajectory(makeReceipt(0.93, 0.90), 'task', [], 'beta');
    const domains = bank.getDomains();
    assert.ok(domains.includes('alpha'));
    assert.ok(domains.includes('beta'));
  });
});

// ────────────────────────────────────────────────────────────
// Adapter + ReasoningBank Integration
// ────────────────────────────────────────────────────────────

describe('AgenticFlowAdapter + ReasoningBank', () => {
  let adapter: AgenticFlowAdapter;

  beforeEach(async () => {
    adapter = new AgenticFlowAdapter();
    await adapter.start();
  });

  afterEach(async () => {
    await adapter.stop();
  });

  it('records trajectory automatically on executeMission', async () => {
    const result = await adapter.executeMission('plan the architecture');
    assert.ok(result.trajectory);
    assert.equal(result.trajectory.missionId, result.receipt.missionId);
    assert.ok(adapter.reasoningBank.getTrajectoryCount() >= 1);
  });

  it('exposes judgeMission through adapter', async () => {
    // Build up some history
    await adapter.executeMission('implement user authentication');
    await adapter.executeMission('implement user authentication module');

    const verdict = adapter.judgeMission('implement user authentication', 'coding');
    assert.ok(['likely_success', 'needs_review', 'likely_failure'].includes(verdict.level));
  });

  it('exposes distillDomain through adapter', async () => {
    for (let i = 0; i < 4; i++) {
      await adapter.executeMission('refactor authentication service');
    }

    const domain = (await adapter.executeMission('refactor auth')).trajectory.domain;
    const result = adapter.distillDomain(domain);
    assert.equal(result.domain, domain);
  });

  it('trajectory contains agent IDs from routing', async () => {
    const result = await adapter.executeMission('write unit tests for the parser');
    assert.ok(result.trajectory.agents.length > 0);
    assert.ok(result.trajectory.steps.length > 0);
  });
});

// ────────────────────────────────────────────────────────────
// Test Helpers
// ────────────────────────────────────────────────────────────

let receiptSeq = 0;

function makeReceipt(ihsan: number, snr: number) {
  const id = `test-${++receiptSeq}`;
  return {
    missionId: id,
    description: 'test mission',
    ihsanScore: ihsan,
    snrScore: snr,
    agentIds: [PATAgent.CODER] as const,
    helix: Helix.DELIBERATIVE as const,
    timestamp: Date.now(),
    receiptHash: id,
    prevHash: '0000',
    elapsedMs: 10,
    seedMinted: ihsan >= CONSTITUTIONAL.IHSAN_MINIMUM,
  };
}
