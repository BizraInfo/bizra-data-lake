/**
 * BIZRA Agentic-Flow — ReasoningBank (Helix 3 Learning Layer)
 *
 * Persistent adaptive learning via trajectory tracking, verdict judgment,
 * memory distillation, and reflex precipitation. Bridges the in-memory
 * MemoryCoordinator to AgentDB for durable pattern storage.
 *
 * Architecture:
 *   AgenticFlowAdapter.executeMission()
 *     → MemoryCoordinator  (in-flight, fast)
 *     → ReasoningBank       (persistence, learning, distillation)
 *       → AgentDB/SQLite    (storage backend)
 *
 * Core concepts:
 *   1. Trajectory Tracking — record agent execution paths + outcomes
 *   2. Verdict Judgment    — classify trajectory as success/failure/partial
 *   3. Memory Distillation — consolidate successful patterns into principles
 *   4. Reflex Precipitation — promote high-confidence patterns to O(1) cache
 *
 * Standing on Giants:
 *   Al-Ghazali (intent gate, 1096) · Deming (PDCA cycle, 1950) ·
 *   Sutton & Barto (RL trajectory, 2018) · Nakamoto (evidence chain, 2008)
 *
 * Reference: Spine §2 Helix 3 (Evolutionary), §7 (Evidence & Proof)
 */

import {
  type AgentId,
  type ActionReceipt,
  type Helix,
  CONSTITUTIONAL,
} from './types';
import { type ReflexCache } from './reflex-cache';

// ────────────────────────────────────────────────────────────
// Trajectory Types
// ────────────────────────────────────────────────────────────

/** A single step in an agent execution trajectory */
export interface TrajectoryStep {
  readonly agentId: AgentId;
  readonly action: string;
  readonly result: string;
  readonly durationMs: number;
  readonly helix: Helix;
}

/** Outcome classification for a trajectory */
export type TrajectoryOutcome = 'success' | 'failure' | 'partial';

/** Complete execution trajectory — the fundamental learning unit */
export interface Trajectory {
  readonly id: string;
  readonly missionId: string;
  readonly description: string;
  readonly steps: readonly TrajectoryStep[];
  readonly outcome: TrajectoryOutcome;
  readonly ihsanScore: number;
  readonly snrScore: number;
  readonly agents: readonly AgentId[];
  readonly domain: string;
  readonly timestamp: number;
  readonly receipt: ActionReceipt;
}

// ────────────────────────────────────────────────────────────
// Pattern Types
// ────────────────────────────────────────────────────────────

/** Pattern abstraction level (hierarchical memory) */
export type PatternType = 'experience' | 'trajectory' | 'distilled' | 'principle';

/** A stored pattern — distilled knowledge from trajectories */
export interface Pattern {
  readonly id: string;
  readonly type: PatternType;
  readonly domain: string;
  readonly description: string;
  readonly confidence: number;
  readonly usageCount: number;
  readonly successCount: number;
  readonly createdAt: number;
  readonly lastUsed: number;
  /** Serialized pattern data (trajectory steps, agent IDs, metrics) */
  readonly data: string;
}

// ────────────────────────────────────────────────────────────
// Verdict Types
// ────────────────────────────────────────────────────────────

/** Verdict classification levels */
export type VerdictLevel = 'likely_success' | 'needs_review' | 'likely_failure';

/** Result of judging a trajectory against prior experience */
export interface Verdict {
  readonly level: VerdictLevel;
  readonly confidence: number;
  readonly similarPatterns: number;
  readonly successfulMatches: number;
  readonly recommendation: string;
}

// ────────────────────────────────────────────────────────────
// Distillation Types
// ────────────────────────────────────────────────────────────

/** Result of distilling trajectories in a domain */
export interface DistillationResult {
  readonly domain: string;
  readonly inputTrajectories: number;
  readonly outputPatterns: number;
  readonly successRate: number;
  readonly avgConfidence: number;
  readonly precipitationCandidates: number;
}

// ────────────────────────────────────────────────────────────
// Config
// ────────────────────────────────────────────────────────────

export interface ReasoningBankConfig {
  /** Max trajectories to store per domain */
  readonly maxTrajectoriesPerDomain: number;
  /** Max patterns to store per domain */
  readonly maxPatternsPerDomain: number;
  /** Minimum similar patterns to declare likely_success */
  readonly verdictSuccessThreshold: number;
  /** Minimum similarity for a match in verdict judgment */
  readonly verdictSimilarityThreshold: number;
  /** Minimum confidence to keep during distillation */
  readonly distillMinConfidence: number;
  /** Minimum Ihsān score for precipitation to reflex cache */
  readonly precipitationIhsan: number;
  /** Minimum repetitions for precipitation */
  readonly precipitationRepeats: number;
}

const DEFAULT_CONFIG: ReasoningBankConfig = {
  maxTrajectoriesPerDomain: 500,
  maxPatternsPerDomain: 200,
  verdictSuccessThreshold: 3,
  verdictSimilarityThreshold: 0.7,
  distillMinConfidence: 0.6,
  precipitationIhsan: CONSTITUTIONAL.PRECIPITATION_IHSAN,
  precipitationRepeats: CONSTITUTIONAL.PRECIPITATION_REPEATS,
};

// ────────────────────────────────────────────────────────────
// ReasoningBank — The Learning Engine
// ────────────────────────────────────────────────────────────

let trajectoryCounter = 0;
let patternCounter = 0;

/**
 * ReasoningBank — Persistent adaptive learning for the Living Organism.
 *
 * Implements the Helix 3 evolutionary cycle:
 *   1. Record mission trajectories (after executeMission)
 *   2. Judge trajectory quality against prior experience
 *   3. Distill successful trajectories into patterns
 *   4. Precipitate high-confidence patterns into ReflexCache
 *
 * In-memory implementation with AgentDB-compatible schema.
 * Production deployment delegates to SQLite via AgentDB.
 */
export class ReasoningBank {
  private readonly config: ReasoningBankConfig;
  private readonly trajectories: Map<string, Map<string, Trajectory>> = new Map();
  private readonly patterns: Map<string, Map<string, Pattern>> = new Map();

  constructor(config: Partial<ReasoningBankConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ── 1. Trajectory Tracking ──────────────────────────────────

  /**
   * Record a mission trajectory for future learning.
   * Called after each executeMission to capture the agent execution path.
   */
  recordTrajectory(
    receipt: ActionReceipt,
    description: string,
    steps: readonly TrajectoryStep[],
    domain: string,
  ): Trajectory {
    const outcome = this.classifyOutcome(receipt.ihsanScore, receipt.snrScore);

    const trajectory: Trajectory = {
      id: `traj-${++trajectoryCounter}`,
      missionId: receipt.missionId,
      description,
      steps,
      outcome,
      ihsanScore: receipt.ihsanScore,
      snrScore: receipt.snrScore,
      agents: [...receipt.agentIds],
      domain,
      timestamp: receipt.timestamp,
      receipt,
    };

    let domainTrajectories = this.trajectories.get(domain);
    if (!domainTrajectories) {
      domainTrajectories = new Map();
      this.trajectories.set(domain, domainTrajectories);
    }

    // Evict oldest if at capacity
    if (domainTrajectories.size >= this.config.maxTrajectoriesPerDomain) {
      const oldest = this.findOldestKey(domainTrajectories);
      if (oldest) domainTrajectories.delete(oldest);
    }

    domainTrajectories.set(trajectory.id, trajectory);

    // Auto-create experience pattern for successful trajectories
    if (outcome === 'success') {
      this.insertPattern({
        id: `pat-${++patternCounter}`,
        type: 'experience',
        domain,
        description: trajectory.description,
        confidence: trajectory.ihsanScore,
        usageCount: 1,
        successCount: 1,
        createdAt: Date.now(),
        lastUsed: Date.now(),
        data: JSON.stringify({
          steps: trajectory.steps,
          agents: trajectory.agents,
          metrics: {
            ihsan: trajectory.ihsanScore,
            snr: trajectory.snrScore,
            duration: trajectory.receipt.elapsedMs,
          },
        }),
      });
    }

    return trajectory;
  }

  /**
   * Get all trajectories for a domain.
   */
  getTrajectories(domain: string): Trajectory[] {
    const domainTrajectories = this.trajectories.get(domain);
    if (!domainTrajectories) return [];
    return Array.from(domainTrajectories.values());
  }

  /**
   * Get total trajectory count across all domains.
   */
  getTrajectoryCount(): number {
    let count = 0;
    for (const domainMap of this.trajectories.values()) {
      count += domainMap.size;
    }
    return count;
  }

  // ── 2. Verdict Judgment ─────────────────────────────────────

  /**
   * Judge a trajectory's likely outcome based on similar prior patterns.
   *
   * Uses keyword overlap as a lightweight similarity proxy.
   * In production, this delegates to AgentDB's HNSW vector search (150x faster).
   */
  judgeTrajectory(description: string, domain: string): Verdict {
    const domainPatterns = this.patterns.get(domain);
    if (!domainPatterns || domainPatterns.size === 0) {
      return {
        level: 'needs_review',
        confidence: 0,
        similarPatterns: 0,
        successfulMatches: 0,
        recommendation: 'No prior patterns in domain — proceed with caution',
      };
    }

    const queryTokens = this.tokenize(description);
    let similarCount = 0;
    let successCount = 0;
    let totalConfidence = 0;

    for (const pattern of domainPatterns.values()) {
      const patternTokens = this.tokenize(pattern.description);
      const similarity = this.jaccardSimilarity(queryTokens, patternTokens);

      if (similarity >= this.config.verdictSimilarityThreshold) {
        similarCount++;
        totalConfidence += pattern.confidence;
        if (pattern.successCount > 0 && pattern.successCount / pattern.usageCount > 0.5) {
          successCount++;
        }
      }
    }

    const avgConfidence = similarCount > 0 ? totalConfidence / similarCount : 0;

    let level: VerdictLevel;
    let recommendation: string;

    if (successCount >= this.config.verdictSuccessThreshold) {
      level = 'likely_success';
      recommendation = `${successCount} similar successful patterns found (avg confidence: ${avgConfidence.toFixed(2)})`;
    } else if (similarCount > 0 && successCount > 0) {
      level = 'needs_review';
      recommendation = `${similarCount} similar patterns found but only ${successCount} successful — review approach`;
    } else {
      level = 'likely_failure';
      recommendation = similarCount > 0
        ? `${similarCount} similar patterns found, none successful — consider alternative approach`
        : 'No similar patterns found — novel territory, apply OODA observe phase';
    }

    return {
      level,
      confidence: avgConfidence,
      similarPatterns: similarCount,
      successfulMatches: successCount,
      recommendation,
    };
  }

  // ── 3. Memory Distillation ──────────────────────────────────

  /**
   * Distill trajectories in a domain into higher-level patterns.
   *
   * Consolidates similar experience-level patterns into distilled patterns,
   * and promotes very high-confidence distilled patterns to principles.
   *
   * This implements the Helix 3 evolutionary learning loop.
   */
  distill(domain: string): DistillationResult {
    const domainPatterns = this.patterns.get(domain);
    const domainTrajectories = this.trajectories.get(domain);

    const inputCount = domainTrajectories?.size ?? 0;

    if (!domainPatterns || domainPatterns.size === 0) {
      return {
        domain,
        inputTrajectories: inputCount,
        outputPatterns: 0,
        successRate: 0,
        avgConfidence: 0,
        precipitationCandidates: 0,
      };
    }

    // Group similar experience patterns by keyword overlap
    const experiences = Array.from(domainPatterns.values())
      .filter((p) => p.type === 'experience' && p.confidence >= this.config.distillMinConfidence);

    const clusters = this.clusterByDescription(experiences);
    let newDistilled = 0;
    let precipitationCandidates = 0;

    for (const cluster of clusters) {
      if (cluster.length < 2) continue;

      const totalUsage = cluster.reduce((sum, p) => sum + p.usageCount, 0);
      const totalSuccess = cluster.reduce((sum, p) => sum + p.successCount, 0);
      const avgConf = cluster.reduce((sum, p) => sum + p.confidence, 0) / cluster.length;
      const successRate = totalUsage > 0 ? totalSuccess / totalUsage : 0;

      const distilled: Pattern = {
        id: `pat-${++patternCounter}`,
        type: successRate >= 0.9 && avgConf >= CONSTITUTIONAL.IHSAN_PRODUCTION ? 'principle' : 'distilled',
        domain,
        description: `Distilled from ${cluster.length} similar experiences in ${domain}`,
        confidence: avgConf,
        usageCount: totalUsage,
        successCount: totalSuccess,
        createdAt: Date.now(),
        lastUsed: Date.now(),
        data: JSON.stringify({
          sourceCount: cluster.length,
          successRate,
          sourceIds: cluster.map((p) => p.id),
        }),
      };

      this.insertPattern(distilled);
      newDistilled++;

      // Check precipitation eligibility
      if (avgConf >= this.config.precipitationIhsan && cluster.length >= this.config.precipitationRepeats) {
        precipitationCandidates++;
      }
    }

    // Calculate stats
    const allPatterns = Array.from(domainPatterns.values());
    const totalSuccess = allPatterns.reduce((sum, p) => sum + p.successCount, 0);
    const totalUsage = allPatterns.reduce((sum, p) => sum + p.usageCount, 0);
    const avgConf = allPatterns.reduce((sum, p) => sum + p.confidence, 0) / allPatterns.length;

    return {
      domain,
      inputTrajectories: inputCount,
      outputPatterns: newDistilled,
      successRate: totalUsage > 0 ? totalSuccess / totalUsage : 0,
      avgConfidence: avgConf,
      precipitationCandidates,
    };
  }

  // ── 4. Reflex Precipitation ─────────────────────────────────

  /**
   * Precipitate high-confidence patterns into the ReflexCache.
   *
   * This is the Helix 3 → Helix 1 bridge: evolutionary learning
   * produces new reflexes for O(1) reactive retrieval.
   *
   * Returns the number of patterns precipitated.
   */
  precipitateToCache(reflexCache: ReflexCache): number {
    let precipitated = 0;

    for (const domainPatterns of this.patterns.values()) {
      for (const pattern of domainPatterns.values()) {
        if (
          (pattern.type === 'distilled' || pattern.type === 'principle') &&
          pattern.confidence >= this.config.precipitationIhsan &&
          pattern.usageCount >= this.config.precipitationRepeats
        ) {
          const successRate = pattern.usageCount > 0
            ? pattern.successCount / pattern.usageCount
            : 0;

          if (successRate >= 0.8) {
            // Parse agents from pattern data
            let agents: AgentId[] = [];
            try {
              const data = JSON.parse(pattern.data) as Record<string, unknown>;
              if (Array.isArray(data.agents)) {
                agents = data.agents as AgentId[];
              }
            } catch {
              // Data may not contain agents — use empty
            }

            reflexCache.precipitate(
              pattern.description,
              agents,
              pattern.confidence,
              `[ReasoningBank distilled: ${pattern.domain}] success_rate=${successRate.toFixed(2)}`,
            );
            precipitated++;
          }
        }
      }
    }

    return precipitated;
  }

  // ── 5. Pattern Management ───────────────────────────────────

  /**
   * Insert a pattern into the bank.
   */
  insertPattern(pattern: Pattern): void {
    let domainPatterns = this.patterns.get(pattern.domain);
    if (!domainPatterns) {
      domainPatterns = new Map();
      this.patterns.set(pattern.domain, domainPatterns);
    }

    // Evict oldest if at capacity
    if (domainPatterns.size >= this.config.maxPatternsPerDomain) {
      const oldest = this.findOldestPatternKey(domainPatterns);
      if (oldest) domainPatterns.delete(oldest);
    }

    domainPatterns.set(pattern.id, pattern);
  }

  /**
   * Retrieve patterns for a domain, optionally filtered by type.
   */
  getPatterns(domain: string, type?: PatternType): Pattern[] {
    const domainPatterns = this.patterns.get(domain);
    if (!domainPatterns) return [];

    const all = Array.from(domainPatterns.values());
    if (type) return all.filter((p) => p.type === type);
    return all;
  }

  /**
   * Get total pattern count across all domains.
   */
  getPatternCount(): number {
    let count = 0;
    for (const domainMap of this.patterns.values()) {
      count += domainMap.size;
    }
    return count;
  }

  /**
   * Get all unique domains.
   */
  getDomains(): string[] {
    return [...new Set([...this.trajectories.keys(), ...this.patterns.keys()])];
  }

  /**
   * Import seed patterns (e.g., from .agentdb/seed-patterns.json).
   */
  importSeedPatterns(seeds: readonly SeedPattern[]): number {
    let imported = 0;
    for (const seed of seeds) {
      this.insertPattern({
        id: `seed-${++patternCounter}`,
        type: seed.metadata.pattern_type === 'distilled' ? 'distilled' : 'experience',
        domain: seed.metadata.domain,
        description: seed.text,
        confidence: seed.metadata.confidence,
        usageCount: seed.metadata.tests_added ?? 1,
        successCount: seed.metadata.outcome === 'success' ? 1 : 0,
        createdAt: Date.now(),
        lastUsed: Date.now(),
        data: JSON.stringify(seed.metadata),
      });
      imported++;
    }
    return imported;
  }

  /**
   * Get comprehensive stats.
   */
  getStats(): ReasoningBankStats {
    let totalTrajectories = 0;
    let totalPatterns = 0;
    let totalSuccess = 0;
    let totalUsage = 0;
    const domainStats: DomainStats[] = [];

    for (const [domain, trajectoryMap] of this.trajectories) {
      const patternMap = this.patterns.get(domain);
      const trajectoryCount = trajectoryMap.size;
      const patternCount = patternMap?.size ?? 0;

      const trajectories = Array.from(trajectoryMap.values());
      const successfulTrajectories = trajectories.filter((t) => t.outcome === 'success').length;

      totalTrajectories += trajectoryCount;
      totalPatterns += patternCount;

      domainStats.push({
        domain,
        trajectories: trajectoryCount,
        patterns: patternCount,
        successRate: trajectoryCount > 0 ? successfulTrajectories / trajectoryCount : 0,
      });
    }

    // Also count pattern-only domains
    for (const [domain, patternMap] of this.patterns) {
      if (!this.trajectories.has(domain)) {
        totalPatterns += patternMap.size;
        for (const p of patternMap.values()) {
          totalUsage += p.usageCount;
          totalSuccess += p.successCount;
        }
        domainStats.push({
          domain,
          trajectories: 0,
          patterns: patternMap.size,
          successRate: 0,
        });
      }
    }

    return {
      totalTrajectories,
      totalPatterns,
      domains: domainStats.length,
      domainStats,
      overallSuccessRate: totalUsage > 0 ? totalSuccess / totalUsage : 0,
    };
  }

  /** Clear all data (useful for testing) */
  clear(): void {
    this.trajectories.clear();
    this.patterns.clear();
  }

  // ── Private helpers ─────────────────────────────────────────

  private classifyOutcome(ihsanScore: number, snrScore: number): TrajectoryOutcome {
    if (ihsanScore >= CONSTITUTIONAL.IHSAN_MINIMUM && snrScore >= CONSTITUTIONAL.SNR_MUSEUM) {
      return 'success';
    }
    if (ihsanScore >= 0.7 || snrScore >= 0.7) {
      return 'partial';
    }
    return 'failure';
  }

  private tokenize(text: string): Set<string> {
    return new Set(
      text.toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter((w) => w.length > 2),
    );
  }

  private jaccardSimilarity(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 && b.size === 0) return 1.0;
    let intersection = 0;
    for (const token of a) {
      if (b.has(token)) intersection++;
    }
    const union = a.size + b.size - intersection;
    return union === 0 ? 0 : intersection / union;
  }

  private clusterByDescription(patterns: Pattern[]): Pattern[][] {
    if (patterns.length === 0) return [];

    const used = new Set<string>();
    const clusters: Pattern[][] = [];

    for (let i = 0; i < patterns.length; i++) {
      const p = patterns[i]!;
      if (used.has(p.id)) continue;

      const cluster: Pattern[] = [p];
      used.add(p.id);
      const tokensI = this.tokenize(p.description);

      for (let j = i + 1; j < patterns.length; j++) {
        const q = patterns[j]!;
        if (used.has(q.id)) continue;

        const tokensJ = this.tokenize(q.description);
        if (this.jaccardSimilarity(tokensI, tokensJ) >= 0.5) {
          cluster.push(q);
          used.add(q.id);
        }
      }

      clusters.push(cluster);
    }

    return clusters;
  }

  private findOldestKey(map: Map<string, Trajectory>): string | undefined {
    let oldestId: string | undefined;
    let oldestTime = Infinity;
    for (const [id, traj] of map) {
      if (traj.timestamp < oldestTime) {
        oldestTime = traj.timestamp;
        oldestId = id;
      }
    }
    return oldestId;
  }

  private findOldestPatternKey(map: Map<string, Pattern>): string | undefined {
    let oldestId: string | undefined;
    let oldestTime = Infinity;
    for (const [id, pat] of map) {
      if (pat.lastUsed < oldestTime) {
        oldestTime = pat.lastUsed;
        oldestId = id;
      }
    }
    return oldestId;
  }
}

// ────────────────────────────────────────────────────────────
// Seed Pattern (for .agentdb/seed-patterns.json import)
// ────────────────────────────────────────────────────────────

export interface SeedPatternMetadata {
  readonly domain: string;
  readonly task?: string;
  readonly outcome?: string;
  readonly confidence: number;
  readonly pattern_type?: string;
  readonly tests_added?: number;
  readonly files?: readonly string[];
  readonly giants?: readonly string[];
  readonly source?: string;
  readonly [key: string]: unknown;
}

export interface SeedPattern {
  readonly text: string;
  readonly metadata: SeedPatternMetadata;
}

// ────────────────────────────────────────────────────────────
// Stats Types
// ────────────────────────────────────────────────────────────

export interface DomainStats {
  readonly domain: string;
  readonly trajectories: number;
  readonly patterns: number;
  readonly successRate: number;
}

export interface ReasoningBankStats {
  readonly totalTrajectories: number;
  readonly totalPatterns: number;
  readonly domains: number;
  readonly domainStats: readonly DomainStats[];
  readonly overallSuccessRate: number;
}
