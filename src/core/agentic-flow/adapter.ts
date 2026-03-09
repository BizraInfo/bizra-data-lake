/**
 * BIZRA Agentic-Flow — Adapter (ADR-001)
 *
 * The unified facade that transforms the SovereignRuntime from a
 * parallel implementation into a specialized extension.
 *
 * Before (parallel):
 *   TS runtime reimplements scoring, routing, validation → duplication
 *
 * After (adapter):
 *   TS adapter orchestrates agents, delegates scoring to Python via IPC,
 *   uses HHMM routing, SONA modes, and reflex cache natively.
 *
 * Flow: Human → DEMA → AgenticFlowAdapter → {ReflexCache | AgentRouter} → IPC → Python
 *
 * Standing on Giants:
 *   Gamma et al (Adapter pattern, 1994) · Boyd (OODA, 1976) ·
 *   Prophet Muhammad ﷺ (Ihsān gate, Hadith Jibril)
 *
 * Reference: Spine §1 (Boundary Model), §3 (Protocol Stack)
 */

import { type AgentId, type ActionReceipt, Helix, SONAMode, HHMMMacroState, CONSTITUTIONAL } from './types';
import { ReflexCache, selectHelix } from './reflex-cache';
import { SONAManager, type SONASnapshot } from './sona';
import { AgentRouter, type RouteResult } from './agent-router';
import { MemoryCoordinator, type SearchResult, type MemoryConfig } from './memory-coordinator';
import {
  ReasoningBank,
  type ReasoningBankConfig,
  type Trajectory,
  type TrajectoryStep,
  type DistillationResult,
} from './reasoning-bank';

// ────────────────────────────────────────────────────────────
// Adapter Configuration
// ────────────────────────────────────────────────────────────

export interface AdapterConfig {
  /** Initial SONA mode */
  readonly sonaMode: SONAMode;
  /** Reflex cache max entries */
  readonly reflexCacheSize: number;
  /** Memory coordinator config */
  readonly memoryConfig: Partial<MemoryConfig>;
  /** ReasoningBank config (Helix 3 learning) */
  readonly reasoningBankConfig: Partial<ReasoningBankConfig>;
  /** Whether to delegate scoring to Python (true) or use local scoring (false) */
  readonly delegateScoring: boolean;
  /** IPC bridge endpoint (for Python delegation) */
  readonly ipcEndpoint: string;
}

const DEFAULT_ADAPTER_CONFIG: AdapterConfig = {
  sonaMode: SONAMode.BALANCED,
  reflexCacheSize: 8192,
  memoryConfig: {},
  reasoningBankConfig: {},
  delegateScoring: true,
  ipcEndpoint: 'stdio',
};

// ────────────────────────────────────────────────────────────
// Scoring Interface (delegates to Python or uses local fallback)
// ────────────────────────────────────────────────────────────

/**
 * Scoring delegate — abstracts where Ihsān/SNR scores come from.
 * In production: Python runtime via IPC (core.iaas.snr_v2_adapter).
 * In tests/offline: local heuristic fallback.
 */
export interface ScoringDelegate {
  scoreIhsan(content: string, context?: Record<string, unknown>): Promise<number>;
  scoreSNR(content: string, context?: Record<string, unknown>): Promise<number>;
}

/**
 * Local fallback scorer (same as current runtime.ts, preserved for offline use).
 * In production this should be replaced by IPC delegation to Python.
 */
export class LocalScoringDelegate implements ScoringDelegate {
  async scoreIhsan(content: string): Promise<number> {
    const lower = content.toLowerCase();
    const positive = ['privacy', 'consent', 'ethical', 'refuse', 'cannot', 'sovereign'];
    const negative = ['exploit', 'track', 'surveil'];
    let score = 0.85;
    for (const p of positive) {
      if (lower.includes(p)) score += 0.02;
    }
    for (const n of negative) {
      if (lower.includes(n)) score -= 0.05;
    }
    return Math.max(0, Math.min(1, score));
  }

  async scoreSNR(content: string): Promise<number> {
    const words = content.split(/\s+/);
    if (words.length === 0) return 0;
    const unique = new Set(words.map((w) => w.toLowerCase()));
    const density = unique.size / words.length;
    const conciseness = words.length >= 30 && words.length <= 100 ? 1.0 : 0.7;
    return Math.min(1, density * 0.5 + conciseness * 0.5);
  }
}

// ────────────────────────────────────────────────────────────
// Mission Result
// ────────────────────────────────────────────────────────────

export interface MissionResult {
  readonly receipt: ActionReceipt;
  readonly content: string;
  readonly route: RouteResult;
  readonly reflexHit: boolean;
  readonly sonaSnapshot: SONASnapshot;
  readonly trajectory: Trajectory;
}

// ────────────────────────────────────────────────────────────
// Adapter Status
// ────────────────────────────────────────────────────────────

export interface AdapterStatus {
  readonly started: boolean;
  readonly sonaMode: SONAMode;
  readonly currentMacroState: HHMMMacroState;
  readonly reflexCacheSize: number;
  readonly reflexHitRate: number;
  readonly sharedMemorySize: number;
  readonly evidenceChainLength: number;
  readonly heartbeatCount: number;
}

// ────────────────────────────────────────────────────────────
// The Adapter
// ────────────────────────────────────────────────────────────

let receiptCounter = 0;

/**
 * AgenticFlowAdapter — The Living Organism's unified orchestration layer.
 *
 * Replaces parallel reimplementation with a clean adapter pattern:
 * 1. DEMA (P7) receives missions
 * 2. ReflexCache checks for O(1) System-1 hit (Helix 1)
 * 3. If miss → AgentRouter selects 2-4 agents via HHMM (Helix 2)
 * 4. ScoringDelegate validates output (delegates to Python or local)
 * 5. MemoryCoordinator records + indexes for future retrieval
 * 6. Evidence chain appended (append-only)
 * 7. If Ihsān ≥ threshold → reflex precipitated for future O(1) hit
 */
export class AgenticFlowAdapter {
  readonly reflexCache: ReflexCache;
  readonly sona: SONAManager;
  readonly router: AgentRouter;
  readonly memory: MemoryCoordinator;
  readonly reasoningBank: ReasoningBank;
  private readonly config: AdapterConfig;
  private scorer: ScoringDelegate;
  private started = false;
  private lastReceiptHash = '0000000000000000';

  constructor(config: Partial<AdapterConfig> = {}) {
    this.config = { ...DEFAULT_ADAPTER_CONFIG, ...config };
    this.reflexCache = new ReflexCache({ maxEntries: this.config.reflexCacheSize });
    this.sona = new SONAManager(this.config.sonaMode);
    this.router = new AgentRouter();
    this.memory = new MemoryCoordinator(this.config.memoryConfig);
    this.reasoningBank = new ReasoningBank(this.config.reasoningBankConfig);
    this.scorer = new LocalScoringDelegate();
  }

  /**
   * Replace the scoring delegate (e.g., with IPC-backed Python scorer).
   */
  setScoringDelegate(delegate: ScoringDelegate): void {
    this.scorer = delegate;
  }

  /**
   * Start the adapter and its subsystems.
   */
  async start(): Promise<void> {
    if (this.started) return;

    // Start Helix 3 evolutionary heartbeat
    this.sona.startHeartbeat(() => {
      // Helix 3 evolutionary tick:
      // 1. Distill trajectories into patterns
      // 2. Precipitate high-confidence patterns to reflex cache
      for (const domain of this.reasoningBank.getDomains()) {
        this.reasoningBank.distill(domain);
      }
      this.reasoningBank.precipitateToCache(this.reflexCache);
    });

    this.started = true;
  }

  /**
   * Stop the adapter and clean up.
   */
  async stop(): Promise<void> {
    if (!this.started) return;
    this.sona.stopHeartbeat();
    this.started = false;
  }

  /**
   * Execute a mission through the Living Organism.
   *
   * This is the core method — the single entry point for all work.
   * Implements the full Spine §1 boundary model:
   *   Human → DEMA → PAT → Pool → SAT
   */
  async executeMission(description: string, _priority: number = 1): Promise<MissionResult> {
    const startTime = performance.now();

    // Step 1: Check reflex cache (Helix 1 — System-1)
    const helix = selectHelix(this.reflexCache, description);
    let reflexHit = false;
    let content: string;
    let agentIds: AgentId[];
    let route: RouteResult;

    if (helix === Helix.REACTIVE) {
      // O(1) reflex hit — skip deliberation
      const reflex = this.reflexCache.lookup(description)!;
      content = reflex.response;
      agentIds = [...reflex.agentIds];
      route = {
        macroState: this.router.getCurrentState(),
        selectedAgents: agentIds,
        confidence: 1.0,
        reason: `reflex cache hit (${reflex.hitCount} prior hits)`,
      };
      reflexHit = true;
    } else {
      // Step 2: Route through HHMM (Helix 2 — System-2)
      const maxAgents = this.sona.getMaxConcurrentAgents();
      route = this.router.route(description, maxAgents);
      agentIds = [...route.selectedAgents];

      // Step 3: Execute (placeholder — in production, delegates to actual agents)
      content = `[Mission routed to ${route.selectedAgents.join(', ')} via ${route.macroState}]`;
    }

    // Step 4: Score output
    const [ihsanScore, snrScore] = await Promise.all([
      this.scorer.scoreIhsan(content),
      this.scorer.scoreSNR(content),
    ]);

    // Step 5: Build receipt
    const elapsedMs = performance.now() - startTime;
    const receiptHash = this.computeReceiptHash(description, content, ihsanScore);
    const receipt: ActionReceipt = {
      missionId: `mission-${++receiptCounter}`,
      description,
      ihsanScore,
      snrScore,
      agentIds,
      helix,
      timestamp: Date.now(),
      receiptHash,
      prevHash: this.lastReceiptHash,
      elapsedMs,
      seedMinted: ihsanScore >= CONSTITUTIONAL.IHSAN_MINIMUM,
    };
    this.lastReceiptHash = receiptHash;

    // Step 6: Append to evidence chain
    this.memory.appendEvidence({
      receiptHash,
      prevHash: receipt.prevHash,
      missionId: receipt.missionId,
      timestamp: receipt.timestamp,
    });

    // Step 7: Record trajectory in ReasoningBank (Helix 3 learning)
    const steps: TrajectoryStep[] = agentIds.map((id) => ({
      agentId: id,
      action: reflexHit ? 'reflex-hit' : 'deliberate',
      result: content,
      durationMs: elapsedMs / agentIds.length,
      helix: reflexHit ? Helix.REACTIVE : Helix.DELIBERATIVE,
    }));

    const domain = route.macroState;
    const trajectory = this.reasoningBank.recordTrajectory(receipt, description, steps, domain);

    // Step 8: Attempt reflex precipitation (Helix 3)
    if (!reflexHit && ihsanScore >= CONSTITUTIONAL.PRECIPITATION_IHSAN) {
      this.reflexCache.recordCandidate(description, agentIds, ihsanScore, content);
    }

    // Update SONA active agents
    this.sona.setActiveAgents(agentIds);

    return {
      receipt,
      content,
      route,
      reflexHit,
      sonaSnapshot: this.sona.snapshot(),
      trajectory,
    };
  }

  /**
   * Search shared memory by semantic similarity.
   */
  searchMemory(queryEmbedding: Float32Array, topK: number = 5): SearchResult[] {
    return this.memory.searchSimilar(queryEmbedding, topK);
  }

  /**
   * Judge a proposed mission against prior trajectories.
   */
  judgeMission(description: string, domain: string): ReturnType<ReasoningBank['judgeTrajectory']> {
    return this.reasoningBank.judgeTrajectory(description, domain);
  }

  /**
   * Trigger manual distillation for a domain (normally runs on Helix 3 heartbeat).
   */
  distillDomain(domain: string): DistillationResult {
    return this.reasoningBank.distill(domain);
  }

  /**
   * Get adapter status.
   */
  getStatus(): AdapterStatus {
    const cacheStats = this.reflexCache.getStats();
    const memStats = this.memory.getStats();
    const sonaSnap = this.sona.snapshot();

    return {
      started: this.started,
      sonaMode: sonaSnap.mode,
      currentMacroState: this.router.getCurrentState(),
      reflexCacheSize: cacheStats.totalSize,
      reflexHitRate: cacheStats.hitRate,
      sharedMemorySize: memStats.sharedSize,
      evidenceChainLength: memStats.evidenceChainLength,
      heartbeatCount: sonaSnap.heartbeatCount,
    };
  }

  // ── Private helpers ───────────────────────────────────────

  private computeReceiptHash(description: string, content: string, ihsan: number): string {
    // Simple hash for TS layer; production uses BLAKE2b via Rust binding
    const input = `${this.lastReceiptHash}:${description}:${content}:${ihsan}:${Date.now()}`;
    let hash = 0x811c9dc5;
    for (let i = 0; i < input.length; i++) {
      hash ^= input.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }
    return (hash >>> 0).toString(16).padStart(8, '0');
  }
}

/**
 * Factory: create and start an AgenticFlowAdapter.
 */
export async function createAgenticFlowAdapter(
  config?: Partial<AdapterConfig>,
): Promise<AgenticFlowAdapter> {
  const adapter = new AgenticFlowAdapter(config);
  await adapter.start();
  return adapter;
}
