/**
 * BIZRA Agentic-Flow — Reflex Cache (Helix 1)
 *
 * O(1) FNV-1a hash lookup for System-1 reactive path.
 * Target: <50ms end-to-end for cached reflexes.
 *
 * Maps to the Flash Attention concept — bypassing full deliberation
 * when a verified reflex pattern already exists.
 *
 * Standing on Giants:
 *   Kahneman (System-1 fast thinking, 2011) · Boyd (OODA observe-act, 1976) ·
 *   FNV-1a (Fowler/Noll/Vo hash, 1991)
 *
 * Reference: Spine §2 Helix 1 (Reactive), §3 L0 HHMM hash table
 */

import {
  type AgentId,
  type Helix,
  Helix as HelixEnum,
  CONSTITUTIONAL,
} from './types';

/** A verified reflex: a cached response pattern that skips deliberation */
export interface Reflex {
  readonly key: string;
  readonly pattern: string;
  readonly agentIds: readonly AgentId[];
  readonly ihsanScore: number;
  readonly response: string;
  readonly hitCount: number;
  readonly precipitatedAt: number;
  readonly lastHitAt: number;
}

export interface ReflexCacheConfig {
  /** Max reflexes to store */
  readonly maxEntries: number;
  /** Minimum Ihsān score for precipitation (§2 Helix 3) */
  readonly precipitationIhsan: number;
  /** Minimum repeat count before precipitating */
  readonly precipitationRepeats: number;
  /** TTL in ms before a reflex expires (0 = never) */
  readonly ttlMs: number;
}

const DEFAULT_CONFIG: ReflexCacheConfig = {
  maxEntries: 8192,
  precipitationIhsan: CONSTITUTIONAL.PRECIPITATION_IHSAN,
  precipitationRepeats: CONSTITUTIONAL.PRECIPITATION_REPEATS,
  ttlMs: 0,
};

/**
 * FNV-1a 32-bit hash.
 * Deterministic, fast, and well-distributed for string keys.
 */
function fnv1a32(input: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0; // Ensure unsigned 32-bit
}

/**
 * Shard index from hash (8 namespace shards via FNV-1a, matching UCF spec)
 */
function shardIndex(hash: number, shardCount: number): number {
  return hash % shardCount;
}

const SHARD_COUNT = 8;

/**
 * ReflexCache — O(1) lookup for System-1 reactive path
 *
 * Implements the HHMM hash table from Spine §2 Helix 1.
 * Reflexes are precipitated from Helix 3 (evolutionary) when a pattern
 * achieves Ihsān ≥ 0.90 over ≥ 3 repetitions.
 */
export class ReflexCache {
  private readonly shards: Map<string, Reflex>[];
  private readonly config: ReflexCacheConfig;
  private readonly candidateCounts: Map<string, number> = new Map();
  private totalHits = 0;
  private totalMisses = 0;

  constructor(config: Partial<ReflexCacheConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.shards = Array.from({ length: SHARD_COUNT }, () => new Map());
  }

  /**
   * O(1) reflex lookup.
   * Returns the cached reflex if it exists and hasn't expired, else undefined.
   */
  lookup(pattern: string): Reflex | undefined {
    const hash = fnv1a32(pattern);
    const shard = this.shards[shardIndex(hash, SHARD_COUNT)]!;
    const key = hash.toString(36);
    const reflex = shard.get(key);

    if (!reflex) {
      this.totalMisses++;
      return undefined;
    }

    // TTL check
    if (this.config.ttlMs > 0) {
      const age = Date.now() - reflex.precipitatedAt;
      if (age > this.config.ttlMs) {
        shard.delete(key);
        this.totalMisses++;
        return undefined;
      }
    }

    // Update hit stats (create new immutable object)
    const updated: Reflex = {
      ...reflex,
      hitCount: reflex.hitCount + 1,
      lastHitAt: Date.now(),
    };
    shard.set(key, updated);
    this.totalHits++;
    return updated;
  }

  /**
   * Record a candidate pattern from Helix 2 (deliberative).
   * When a pattern reaches precipitationRepeats with sufficient Ihsān,
   * it precipitates into a reflex.
   */
  recordCandidate(
    pattern: string,
    agentIds: readonly AgentId[],
    ihsanScore: number,
    response: string,
  ): boolean {
    if (ihsanScore < this.config.precipitationIhsan) {
      return false;
    }

    const count = (this.candidateCounts.get(pattern) ?? 0) + 1;
    this.candidateCounts.set(pattern, count);

    if (count >= this.config.precipitationRepeats) {
      this.precipitate(pattern, agentIds, ihsanScore, response);
      this.candidateCounts.delete(pattern);
      return true;
    }
    return false;
  }

  /**
   * Directly insert a verified reflex (e.g., from forest sync / Helix 3).
   */
  precipitate(
    pattern: string,
    agentIds: readonly AgentId[],
    ihsanScore: number,
    response: string,
  ): void {
    const hash = fnv1a32(pattern);
    const shard = this.shards[shardIndex(hash, SHARD_COUNT)]!;
    const key = hash.toString(36);

    // Evict if at capacity (LRU approximation: evict oldest in shard)
    if (this.totalSize >= this.config.maxEntries && !shard.has(key)) {
      this.evictOldest(shard);
    }

    const reflex: Reflex = {
      key,
      pattern,
      agentIds,
      ihsanScore,
      response,
      hitCount: 0,
      precipitatedAt: Date.now(),
      lastHitAt: 0,
    };
    shard.set(key, reflex);
  }

  /** Check if a pattern has a cached reflex */
  has(pattern: string): boolean {
    const hash = fnv1a32(pattern);
    const shard = this.shards[shardIndex(hash, SHARD_COUNT)]!;
    return shard.has(hash.toString(36));
  }

  /** Remove a specific reflex */
  invalidate(pattern: string): boolean {
    const hash = fnv1a32(pattern);
    const shard = this.shards[shardIndex(hash, SHARD_COUNT)]!;
    return shard.delete(hash.toString(36));
  }

  /** Clear all reflexes */
  clear(): void {
    for (const shard of this.shards) {
      shard.clear();
    }
    this.candidateCounts.clear();
    this.totalHits = 0;
    this.totalMisses = 0;
  }

  get totalSize(): number {
    let size = 0;
    for (const shard of this.shards) {
      size += shard.size;
    }
    return size;
  }

  get hitRate(): number {
    const total = this.totalHits + this.totalMisses;
    return total === 0 ? 0 : this.totalHits / total;
  }

  getStats(): {
    totalSize: number;
    hitRate: number;
    totalHits: number;
    totalMisses: number;
    shardSizes: number[];
    pendingCandidates: number;
  } {
    return {
      totalSize: this.totalSize,
      hitRate: this.hitRate,
      totalHits: this.totalHits,
      totalMisses: this.totalMisses,
      shardSizes: this.shards.map((s) => s.size),
      pendingCandidates: this.candidateCounts.size,
    };
  }

  private evictOldest(shard: Map<string, Reflex>): void {
    let oldestKey: string | undefined;
    let oldestTime = Infinity;
    for (const [key, reflex] of shard) {
      const t = reflex.lastHitAt || reflex.precipitatedAt;
      if (t < oldestTime) {
        oldestTime = t;
        oldestKey = key;
      }
    }
    if (oldestKey !== undefined) {
      shard.delete(oldestKey);
    }
  }
}

/**
 * Check whether a mission should use Helix 1 (reflex) or Helix 2 (deliberative)
 */
export function selectHelix(cache: ReflexCache, pattern: string): Helix {
  return cache.has(pattern) ? HelixEnum.REACTIVE : HelixEnum.DELIBERATIVE;
}
