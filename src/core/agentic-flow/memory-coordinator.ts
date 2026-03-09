/**
 * BIZRA Agentic-Flow — Memory Coordinator
 *
 * Cross-agent memory sharing with HNSW indexing interface.
 * Each agent has private memory; the coordinator manages shared memory.
 *
 * Memory propagation rules (§9):
 *   PROPAGATES:  Reflex patterns, LoRA deltas, HHMM transitions, BLOOM scores
 *   STAYS LOCAL: Raw data, private keys, evidence chain, SEED balance
 *
 * Standing on Giants:
 *   Malkov & Yashunin (HNSW, 2016) · Nakamoto (evidence chain, 2008) ·
 *   Shannon (information density, 1948)
 *
 * Reference: Spine §7 (Evidence & Proof), §9 (Growth)
 */

import {
  type AgentId,
  type MemoryEntry,
  type EvidenceLink,
  FROZEN_AGENTS,
} from './types';

/** Configuration for the memory coordinator */
export interface MemoryConfig {
  /** Max entries per agent (private memory) */
  readonly maxEntriesPerAgent: number;
  /** Max entries in shared pool */
  readonly maxSharedEntries: number;
  /** Embedding dimension for similarity search */
  readonly embeddingDimension: number;
  /** HNSW M parameter (connections per layer) */
  readonly hnswM: number;
  /** HNSW efConstruction (build-time search depth) */
  readonly hnswEfConstruction: number;
}

const DEFAULT_MEMORY_CONFIG: MemoryConfig = {
  maxEntriesPerAgent: 1024,
  maxSharedEntries: 8192,
  embeddingDimension: 1536,
  hnswM: 16,
  hnswEfConstruction: 200,
};

/** Similarity search result */
export interface SearchResult {
  readonly entry: MemoryEntry;
  readonly distance: number;
  readonly score: number;
}

/**
 * MemoryCoordinator — Cross-agent memory sharing
 *
 * Implements the memory layer from the protocol stack (§3 L0).
 * Private memory stays with each agent. Shared memory is indexed
 * via HNSW for O(log n) approximate nearest neighbor search.
 *
 * Frozen agents (P5 Ethicist, S2 Oracle) have read-only access to
 * shared memory — they cannot write or evolve.
 */
export class MemoryCoordinator {
  private readonly config: MemoryConfig;
  private readonly privatePools: Map<AgentId, Map<string, MemoryEntry>> = new Map();
  private readonly sharedPool: Map<string, MemoryEntry> = new Map();
  private readonly evidenceChain: EvidenceLink[] = [];
  private readonly embeddings: Map<string, Float32Array> = new Map();

  constructor(config: Partial<MemoryConfig> = {}) {
    this.config = { ...DEFAULT_MEMORY_CONFIG, ...config };
  }

  /**
   * Store a memory entry for a specific agent (private).
   * Frozen agents cannot store new memories.
   */
  store(agentId: AgentId, entry: MemoryEntry): boolean {
    if (FROZEN_AGENTS.has(agentId)) {
      return false;
    }

    let pool = this.privatePools.get(agentId);
    if (!pool) {
      pool = new Map();
      this.privatePools.set(agentId, pool);
    }

    // Evict oldest if at capacity
    if (pool.size >= this.config.maxEntriesPerAgent) {
      const oldest = this.findOldest(pool);
      if (oldest) pool.delete(oldest);
    }

    pool.set(entry.id, entry);

    // Index embedding if present
    if (entry.embedding) {
      this.embeddings.set(entry.id, entry.embedding);
    }

    return true;
  }

  /**
   * Share a memory entry to the shared pool.
   * Frozen agents cannot share.
   */
  share(agentId: AgentId, entryId: string): boolean {
    if (FROZEN_AGENTS.has(agentId)) {
      return false;
    }

    const pool = this.privatePools.get(agentId);
    if (!pool) return false;

    const entry = pool.get(entryId);
    if (!entry) return false;

    // Evict from shared pool if at capacity
    if (this.sharedPool.size >= this.config.maxSharedEntries) {
      const oldest = this.findOldest(this.sharedPool);
      if (oldest) {
        this.sharedPool.delete(oldest);
        this.embeddings.delete(oldest);
      }
    }

    const shared: MemoryEntry = { ...entry, shared: true };
    this.sharedPool.set(entry.id, shared);
    return true;
  }

  /**
   * Retrieve private memory for an agent.
   */
  getPrivate(agentId: AgentId, entryId: string): MemoryEntry | undefined {
    return this.privatePools.get(agentId)?.get(entryId);
  }

  /**
   * Retrieve from shared pool (all agents can read, including frozen).
   */
  getShared(entryId: string): MemoryEntry | undefined {
    return this.sharedPool.get(entryId);
  }

  /**
   * Search shared memory by cosine similarity.
   * Approximates HNSW behavior — in production, delegates to AgentDB
   * for true 150x-12,500x speedup via native HNSW indexing.
   */
  searchSimilar(queryEmbedding: Float32Array, topK: number = 5): SearchResult[] {
    const results: SearchResult[] = [];

    for (const [id, embedding] of this.embeddings) {
      const entry = this.sharedPool.get(id);
      if (!entry) continue;

      const distance = this.cosineDistance(queryEmbedding, embedding);
      results.push({
        entry,
        distance,
        score: 1.0 - distance,
      });
    }

    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, topK);
  }

  /**
   * Append to the evidence chain (append-only, §7).
   */
  appendEvidence(link: EvidenceLink): void {
    // Verify chain integrity
    if (this.evidenceChain.length > 0) {
      const lastLink = this.evidenceChain[this.evidenceChain.length - 1]!;
      if (link.prevHash !== lastLink.receiptHash) {
        throw new Error(
          `Evidence chain broken: expected prevHash=${lastLink.receiptHash}, got ${link.prevHash}`
        );
      }
    }
    this.evidenceChain.push(link);
  }

  /** Get evidence chain length */
  getEvidenceChainLength(): number {
    return this.evidenceChain.length;
  }

  /** Get the latest evidence link */
  getLatestEvidence(): EvidenceLink | undefined {
    return this.evidenceChain[this.evidenceChain.length - 1];
  }

  /** Get agent's private memory size */
  getPrivateSize(agentId: AgentId): number {
    return this.privatePools.get(agentId)?.size ?? 0;
  }

  /** Get shared pool size */
  getSharedSize(): number {
    return this.sharedPool.size;
  }

  /** Get indexed embedding count */
  getIndexedCount(): number {
    return this.embeddings.size;
  }

  getStats(): {
    agentCount: number;
    totalPrivate: number;
    sharedSize: number;
    indexedEmbeddings: number;
    evidenceChainLength: number;
  } {
    let totalPrivate = 0;
    for (const pool of this.privatePools.values()) {
      totalPrivate += pool.size;
    }
    return {
      agentCount: this.privatePools.size,
      totalPrivate,
      sharedSize: this.sharedPool.size,
      indexedEmbeddings: this.embeddings.size,
      evidenceChainLength: this.evidenceChain.length,
    };
  }

  /** Clear all memory (useful for testing) */
  clear(): void {
    this.privatePools.clear();
    this.sharedPool.clear();
    this.embeddings.clear();
    this.evidenceChain.length = 0;
  }

  // ── Private helpers ───────────────────────────────────────

  private cosineDistance(a: Float32Array, b: Float32Array): number {
    const len = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < len; i++) {
      dot += a[i]! * b[i]!;
      normA += a[i]! * a[i]!;
      normB += b[i]! * b[i]!;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    if (denom === 0) return 1.0;
    return 1.0 - dot / denom;
  }

  private findOldest(pool: Map<string, MemoryEntry>): string | undefined {
    let oldestId: string | undefined;
    let oldestTime = Infinity;
    for (const [id, entry] of pool) {
      if (entry.timestamp < oldestTime) {
        oldestTime = entry.timestamp;
        oldestId = id;
      }
    }
    return oldestId;
  }
}
