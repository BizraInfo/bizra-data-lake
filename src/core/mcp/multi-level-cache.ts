/**
 * Multi-Level Cache - L1 Map + L2 LRU for MCP responses
 *
 * L1: Fast in-memory Map for hot entries (small, instant)
 * L2: LRU cache with TTL for warm entries (larger, still fast)
 *
 * Target: >90% cache hit rate for repeated MCP tool calls.
 */

export interface CacheConfig {
  /** L1 max entries (Map, O(1) lookup) */
  readonly l1MaxEntries: number;

  /** L2 max entries (LRU with TTL) */
  readonly l2MaxEntries: number;

  /** TTL in milliseconds for L2 entries */
  readonly l2TtlMs: number;

  /** L1 TTL in milliseconds (shorter than L2) */
  readonly l1TtlMs: number;

  /** Max entry size in bytes (0 = unlimited) */
  readonly maxEntrySizeBytes: number;
}

export interface CacheStats {
  /** L1 current size */
  readonly l1Size: number;

  /** L2 current size */
  readonly l2Size: number;

  /** L1 hits */
  readonly l1Hits: number;

  /** L2 hits */
  readonly l2Hits: number;

  /** Total misses */
  readonly misses: number;

  /** Overall hit rate */
  readonly hitRate: number;

  /** L1 hit rate */
  readonly l1HitRate: number;

  /** Total evictions */
  readonly evictions: number;
}

interface CacheEntry<T> {
  value: T;
  createdAt: number;
  accessedAt: number;
  sizeBytes: number;
}

const DEFAULT_CACHE_CONFIG: CacheConfig = {
  l1MaxEntries: 64,
  l2MaxEntries: 512,
  l2TtlMs: 300_000,
  l1TtlMs: 60_000,
  maxEntrySizeBytes: 0,
};

/**
 * Multi-Level Cache for MCP responses
 */
export class MultiLevelCache<T = string> {
  private readonly config: CacheConfig;
  private readonly l1: Map<string, CacheEntry<T>> = new Map();
  private readonly l2: Map<string, CacheEntry<T>> = new Map();
  private l1Hits: number = 0;
  private l2Hits: number = 0;
  private misses: number = 0;
  private evictions: number = 0;

  constructor(config: Partial<CacheConfig> = {}) {
    this.config = { ...DEFAULT_CACHE_CONFIG, ...config };
  }

  /**
   * Get a value from the cache. Checks L1 first, then L2.
   * On L2 hit, promotes to L1.
   */
  get(key: string): T | undefined {
    // L1 check
    const l1Entry = this.l1.get(key);
    if (l1Entry) {
      if (this.isExpired(l1Entry, this.config.l1TtlMs)) {
        this.l1.delete(key);
      } else {
        l1Entry.accessedAt = Date.now();
        // Move to end of Map for O(1) LRU: delete+re-insert
        this.l1.delete(key);
        this.l1.set(key, l1Entry);
        this.l1Hits++;
        return l1Entry.value;
      }
    }

    // L2 check
    const l2Entry = this.l2.get(key);
    if (l2Entry) {
      if (this.isExpired(l2Entry, this.config.l2TtlMs)) {
        this.l2.delete(key);
        this.misses++;
        return undefined;
      }
      l2Entry.accessedAt = Date.now();
      // Move to end of Map for O(1) LRU
      this.l2.delete(key);
      this.l2.set(key, l2Entry);
      this.l2Hits++;

      // Promote to L1
      this.promoteToL1(key, l2Entry);

      return l2Entry.value;
    }

    this.misses++;
    return undefined;
  }

  /**
   * Put a value into the cache. Goes to both L1 and L2.
   */
  set(key: string, value: T): void {
    const sizeBytes = this.estimateSize(value);

    // Check max entry size
    if (
      this.config.maxEntrySizeBytes > 0 &&
      sizeBytes > this.config.maxEntrySizeBytes
    ) {
      return;
    }

    const now = Date.now();
    const entry: CacheEntry<T> = {
      value,
      createdAt: now,
      accessedAt: now,
      sizeBytes,
    };

    // Insert into L1
    this.l1.set(key, entry);
    this.evictL1IfNeeded();

    // Insert into L2
    this.l2.set(key, { ...entry });
    this.evictL2IfNeeded();
  }

  /**
   * Check if key exists (without updating access time)
   */
  has(key: string): boolean {
    const l1Entry = this.l1.get(key);
    if (l1Entry && !this.isExpired(l1Entry, this.config.l1TtlMs)) {
      return true;
    }
    const l2Entry = this.l2.get(key);
    return l2Entry !== undefined && !this.isExpired(l2Entry, this.config.l2TtlMs);
  }

  /**
   * Delete a specific entry
   */
  delete(key: string): boolean {
    const l1Deleted = this.l1.delete(key);
    const l2Deleted = this.l2.delete(key);
    return l1Deleted || l2Deleted;
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.l1.clear();
    this.l2.clear();
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const totalOps = this.l1Hits + this.l2Hits + this.misses;
    const totalHits = this.l1Hits + this.l2Hits;

    return {
      l1Size: this.l1.size,
      l2Size: this.l2.size,
      l1Hits: this.l1Hits,
      l2Hits: this.l2Hits,
      misses: this.misses,
      hitRate: totalOps > 0 ? totalHits / totalOps : 0,
      l1HitRate: totalOps > 0 ? this.l1Hits / totalOps : 0,
      evictions: this.evictions,
    };
  }

  /**
   * Generate a cache key from tool name + arguments
   */
  static makeKey(toolName: string, args: Record<string, unknown>): string {
    const sorted = JSON.stringify({ t: toolName, a: args }, Object.keys(args).sort());
    // Simple hash for fast key generation
    let hash = 0;
    for (let i = 0; i < sorted.length; i++) {
      const char = sorted.charCodeAt(i);
      hash = ((hash << 5) - hash + char) | 0;
    }
    return `${toolName}:${hash.toString(36)}`;
  }

  private isExpired(entry: CacheEntry<T>, ttlMs: number): boolean {
    return Date.now() - entry.createdAt > ttlMs;
  }

  private promoteToL1(key: string, entry: CacheEntry<T>): void {
    this.l1.set(key, { ...entry, accessedAt: Date.now() });
    this.evictL1IfNeeded();
  }

  /**
   * O(1) LRU eviction using Map insertion order.
   *
   * JS Maps iterate in insertion order. On access, we delete+re-insert
   * the key so it moves to the end. The first key is always the LRU.
   */
  private evictL1IfNeeded(): void {
    while (this.l1.size > this.config.l1MaxEntries) {
      const oldest = this.l1.keys().next();
      if (oldest.done) break;
      this.l1.delete(oldest.value);
      this.evictions++;
    }
  }

  private evictL2IfNeeded(): void {
    while (this.l2.size > this.config.l2MaxEntries) {
      const oldest = this.l2.keys().next();
      if (oldest.done) break;
      this.l2.delete(oldest.value);
      this.evictions++;
    }
  }

  private estimateSize(value: T): number {
    if (typeof value === 'string') {
      return value.length * 2;
    }
    return JSON.stringify(value).length * 2;
  }
}
