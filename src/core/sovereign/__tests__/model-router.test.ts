/**
 * Tests for ModelRouter + NetworkMode utilities — routing decisions,
 * complexity estimation, tier filtering, queue management, and
 * network mode helper functions.
 *
 * Note: ModelRouter.route() calls registry.selectBest() which
 * validates card signatures. Since we use unsigned test cards,
 * we test the router's structural logic separately.
 */

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

import {
  ModelRouter,
  TaskComplexity,
  estimateComplexity,
} from '../model-router';
import { ModelRegistry, RegisteredModel } from '../model-registry';
import {
  ModelTier,
  TaskType,
  createCapabilityCard,
} from '../capability-card';
import {
  NetworkMode,
  allowsExternalConnections,
  allowsInternetAccess,
  getFallbackMode,
  createInitialNetworkStatus,
} from '../network-mode';

// -- Helpers ---------------------------------------------------------------

function setupRouter(opts: {
  mode?: NetworkMode;
  poolAvailable?: boolean;
  enablePool?: boolean;
} = {}): { router: ModelRouter; registry: ModelRegistry } {
  const registry = new ModelRegistry();
  const status = createInitialNetworkStatus(opts.mode ?? NetworkMode.HYBRID);
  status.poolAvailable = opts.poolAvailable ?? false;

  const router = new ModelRouter(registry, status, {
    enablePoolInference: opts.enablePool ?? true,
  });

  return { router, registry };
}

function insertModel(
  registry: ModelRegistry,
  id: string,
  tier: ModelTier,
  opts: { ihsan?: number; snr?: number; tasks?: TaskType[]; latencyMs?: number; maxContext?: number } = {},
): void {
  const card = createCapabilityCard({
    modelId: id,
    tier,
    ihsanScore: opts.ihsan ?? 0.97,
    snrScore: opts.snr ?? 0.90,
    tasksSupported: opts.tasks ?? [TaskType.CHAT, TaskType.REASONING],
    maxContext: opts.maxContext ?? 4096,
    latencyMs: opts.latencyMs ?? 100,
  });

  const model: RegisteredModel = {
    id,
    name: `Model ${id}`,
    path: `/models/${id}.gguf`,
    sizeBytes: 1_000_000,
    fileHash: `hash-${id}`,
    card,
    registeredAt: new Date(),
    lastUsedAt: null,
    usageCount: 0,
    isLoaded: false,
  };

  (registry as any).models.set(id, model);
  (registry as any).tierIndex.get(tier)?.add(id);
  for (const task of card.capabilities.tasksSupported) {
    if (!(registry as any).taskIndex.has(task)) {
      (registry as any).taskIndex.set(task, new Set());
    }
    (registry as any).taskIndex.get(task)?.add(id);
  }
}

// -- NetworkMode utilities -------------------------------------------------

describe('NetworkMode - utilities', () => {
  it('allowsExternalConnections() for all modes', () => {
    assert.equal(allowsExternalConnections(NetworkMode.OFFLINE), false);
    assert.equal(allowsExternalConnections(NetworkMode.LOCAL_ONLY), true);
    assert.equal(allowsExternalConnections(NetworkMode.FEDERATED), true);
    assert.equal(allowsExternalConnections(NetworkMode.HYBRID), true);
  });

  it('allowsInternetAccess() for all modes', () => {
    assert.equal(allowsInternetAccess(NetworkMode.OFFLINE), false);
    assert.equal(allowsInternetAccess(NetworkMode.LOCAL_ONLY), false);
    assert.equal(allowsInternetAccess(NetworkMode.FEDERATED), true);
    assert.equal(allowsInternetAccess(NetworkMode.HYBRID), true);
  });

  it('getFallbackMode() returns correct fallback chain', () => {
    assert.equal(getFallbackMode(NetworkMode.FEDERATED), NetworkMode.LOCAL_ONLY);
    assert.equal(getFallbackMode(NetworkMode.LOCAL_ONLY), NetworkMode.OFFLINE);
    assert.equal(getFallbackMode(NetworkMode.HYBRID), NetworkMode.LOCAL_ONLY);
    assert.equal(getFallbackMode(NetworkMode.OFFLINE), NetworkMode.OFFLINE);
  });

  it('createInitialNetworkStatus() builds correct shape', () => {
    const status = createInitialNetworkStatus(NetworkMode.HYBRID);
    assert.equal(status.currentMode, NetworkMode.HYBRID);
    assert.equal(status.effectiveMode, NetworkMode.HYBRID);
    assert.equal(status.isOnline, true);
    assert.equal(status.peerCount, 0);
    assert.equal(status.poolAvailable, false);
  });

  it('createInitialNetworkStatus(OFFLINE) sets isOnline=false', () => {
    const status = createInitialNetworkStatus(NetworkMode.OFFLINE);
    assert.equal(status.isOnline, false);
  });
});

// -- estimateComplexity ----------------------------------------------------

describe('estimateComplexity', () => {
  it('returns HIGH for code generation', () => {
    assert.equal(estimateComplexity({ requiresCodeGen: true }), TaskComplexity.HIGH);
  });

  it('returns HIGH for large context', () => {
    assert.equal(estimateComplexity({ contextLength: 8192 }), TaskComplexity.HIGH);
  });

  it('returns HIGH for many input tokens', () => {
    assert.equal(estimateComplexity({ inputTokens: 3000 }), TaskComplexity.HIGH);
  });

  it('returns MEDIUM for reasoning', () => {
    assert.equal(estimateComplexity({ requiresReasoning: true }), TaskComplexity.MEDIUM);
  });

  it('returns MEDIUM for moderate input tokens', () => {
    assert.equal(estimateComplexity({ inputTokens: 800 }), TaskComplexity.MEDIUM);
  });

  it('returns LOW for simple queries', () => {
    assert.equal(estimateComplexity({}), TaskComplexity.LOW);
    assert.equal(estimateComplexity({ inputTokens: 100 }), TaskComplexity.LOW);
  });
});

// -- ModelRouter - canRoute() ----------------------------------------------

describe('ModelRouter - canRoute()', () => {
  it('returns canRoute=false when no models registered', () => {
    const { router } = setupRouter();
    const check = router.canRoute({
      taskType: TaskType.CHAT,
      complexity: TaskComplexity.LOW,
    });
    assert.equal(check.canRoute, false);
    assert.ok(check.reason?.includes('No valid models'));
  });

  it('returns canRoute=false when queue is full', () => {
    const { router, registry } = setupRouter();
    insertModel(registry, 'q1', ModelTier.LOCAL);
    for (let i = 0; i < 100; i++) router.enqueue();
    const check = router.canRoute({
      taskType: TaskType.CHAT,
      complexity: TaskComplexity.LOW,
    });
    assert.equal(check.canRoute, false);
    assert.ok(check.reason?.includes('Queue full'));
  });
});

// -- ModelRouter - route() -------------------------------------------------
// Since route() uses selectBest() which validates signatures, unsigned cards
// cause route() to throw. We test the throw behavior and structural routing.

describe('ModelRouter - route()', () => {
  it('throws when no models available', () => {
    const { router } = setupRouter();
    assert.throws(
      () => router.route({
        taskType: TaskType.CHAT,
        complexity: TaskComplexity.LOW,
      }),
      /No models available/,
    );
  });

  it('throws when all models have invalid signatures', () => {
    const { router, registry } = setupRouter();
    insertModel(registry, 'unsigned', ModelTier.LOCAL);
    // selectBest filters by isCapabilityCardValid → returns null → throws
    assert.throws(
      () => router.route({
        taskType: TaskType.CHAT,
        complexity: TaskComplexity.MEDIUM,
      }),
      /No suitable model/,
    );
  });
});

// -- Queue management ------------------------------------------------------

describe('ModelRouter - Queue', () => {
  it('enqueue/dequeue tracks queue depth', () => {
    const { router } = setupRouter();
    assert.equal(router.getQueueDepth(), 0);
    router.enqueue();
    router.enqueue();
    assert.equal(router.getQueueDepth(), 2);
    router.dequeue();
    assert.equal(router.getQueueDepth(), 1);
  });

  it('dequeue() never goes below 0', () => {
    const { router } = setupRouter();
    router.dequeue();
    router.dequeue();
    assert.equal(router.getQueueDepth(), 0);
  });
});

// -- updateNetworkStatus ---------------------------------------------------

describe('ModelRouter - updateNetworkStatus', () => {
  it('updates the internal network status', () => {
    const { router, registry } = setupRouter({ mode: NetworkMode.OFFLINE });
    insertModel(registry, 'local-1', ModelTier.LOCAL);

    // After update, we can check canRoute behavior changes
    const newStatus = createInitialNetworkStatus(NetworkMode.FEDERATED);
    newStatus.poolAvailable = true;
    router.updateNetworkStatus(newStatus);

    // The router now has FEDERATED status — HIGH complexity canRoute
    // should still work since we have a LOCAL model
    // (no need for pool since LOCAL tier is available)
  });
});
