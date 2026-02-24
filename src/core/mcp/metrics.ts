/**
 * MCP Metrics - Real-time performance monitoring
 *
 * Tracks latency, throughput, cache hit rates, and error rates
 * across all MCP server connections. Aligned with IHSAN_THRESHOLD (0.95).
 */

import { IHSAN_THRESHOLD } from '../sovereign/capability-card';

/**
 * Quickselect — O(n) average-case k-th smallest element.
 * Replaces full O(n log n) sort for percentile calculations.
 */
function quickselect(arr: number[], k: number): number {
  let lo = 0;
  let hi = arr.length - 1;

  while (lo < hi) {
    const pivotIdx = lo + ((hi - lo) >>> 1);
    const pivot = arr[pivotIdx] ?? 0;

    // Three-way partition (Dutch national flag)
    let i = lo;
    let j = lo;
    let n = hi;

    while (j <= n) {
      const v = arr[j] ?? 0;
      if (v < pivot) {
        [arr[i], arr[j]] = [v, arr[i] ?? 0];
        i++;
        j++;
      } else if (v > pivot) {
        [arr[j], arr[n]] = [arr[n] ?? 0, v];
        n--;
      } else {
        j++;
      }
    }

    if (k < i) {
      hi = i - 1;
    } else if (k > n) {
      lo = n + 1;
    } else {
      return pivot;
    }
  }

  return arr[lo] ?? 0;
}

export interface MetricSnapshot {
  /** Timestamp of snapshot */
  readonly timestamp: number;

  /** Total requests processed */
  readonly totalRequests: number;

  /** Total errors */
  readonly totalErrors: number;

  /** Error rate (0-1) */
  readonly errorRate: number;

  /** Average response time in ms */
  readonly avgResponseMs: number;

  /** P50 response time in ms */
  readonly p50ResponseMs: number;

  /** P95 response time in ms */
  readonly p95ResponseMs: number;

  /** P99 response time in ms */
  readonly p99ResponseMs: number;

  /** Cache hit rate (0-1) */
  readonly cacheHitRate: number;

  /** Active connections */
  readonly activeConnections: number;

  /** Quality score (0-1), must exceed IHSAN_THRESHOLD */
  readonly qualityScore: number;
}

export interface PerServerMetrics {
  readonly serverId: string;
  readonly totalRequests: number;
  readonly totalErrors: number;
  readonly avgResponseMs: number;
  readonly lastResponseMs: number;
  readonly lastErrorAt: number | null;
  readonly isHealthy: boolean;
}

/**
 * Circular buffer for latency samples
 */
class LatencyBuffer {
  private readonly buffer: Float64Array;
  private index: number = 0;
  private count: number = 0;

  constructor(capacity: number) {
    this.buffer = new Float64Array(capacity);
  }

  push(value: number): void {
    this.buffer[this.index] = value;
    this.index = (this.index + 1) % this.buffer.length;
    if (this.count < this.buffer.length) {
      this.count++;
    }
  }

  /**
   * O(n) average-case percentile via quickselect (Floyd-Rivest).
   * Avoids O(n log n) full sort for each snapshot.
   */
  percentile(p: number): number {
    if (this.count === 0) return 0;

    const arr = Array.from(this.buffer.subarray(0, this.count));
    const k = Math.max(0, Math.ceil((p / 100) * arr.length) - 1);
    return quickselect(arr, k);
  }

  average(): number {
    if (this.count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.count; i++) {
      sum += this.buffer[i] ?? 0;
    }
    return sum / this.count;
  }

  get length(): number {
    return this.count;
  }
}

/**
 * Per-server tracker
 */
class ServerTracker {
  readonly serverId: string;
  totalRequests: number = 0;
  totalErrors: number = 0;
  lastResponseMs: number = 0;
  lastErrorAt: number | null = null;
  private latencies: LatencyBuffer;

  constructor(serverId: string, bufferSize: number) {
    this.serverId = serverId;
    this.latencies = new LatencyBuffer(bufferSize);
  }

  recordSuccess(latencyMs: number): void {
    this.totalRequests++;
    this.lastResponseMs = latencyMs;
    this.latencies.push(latencyMs);
  }

  recordError(): void {
    this.totalRequests++;
    this.totalErrors++;
    this.lastErrorAt = Date.now();
  }

  get avgResponseMs(): number {
    return this.latencies.average();
  }

  get isHealthy(): boolean {
    if (this.totalRequests === 0) return true;
    const errorRate = this.totalErrors / this.totalRequests;
    return errorRate < 0.1 && this.avgResponseMs < 5000;
  }

  toMetrics(): PerServerMetrics {
    return {
      serverId: this.serverId,
      totalRequests: this.totalRequests,
      totalErrors: this.totalErrors,
      avgResponseMs: Math.round(this.avgResponseMs * 100) / 100,
      lastResponseMs: Math.round(this.lastResponseMs * 100) / 100,
      lastErrorAt: this.lastErrorAt,
      isHealthy: this.isHealthy,
    };
  }
}

export interface MCPMetricsConfig {
  /** Latency buffer size per server */
  readonly bufferSize: number;

  /** Snapshot interval in ms */
  readonly snapshotIntervalMs: number;

  /** Max snapshots to retain */
  readonly maxSnapshots: number;
}

const DEFAULT_METRICS_CONFIG: MCPMetricsConfig = {
  bufferSize: 1000,
  snapshotIntervalMs: 10000,
  maxSnapshots: 360,
};

/**
 * MCP Metrics Collector
 */
export class MCPMetrics {
  private readonly config: MCPMetricsConfig;
  private readonly servers: Map<string, ServerTracker> = new Map();
  private readonly globalLatencies: LatencyBuffer;
  private readonly snapshots: MetricSnapshot[] = [];
  private totalRequests: number = 0;
  private totalErrors: number = 0;
  private cacheHits: number = 0;
  private cacheMisses: number = 0;
  private activeConnections: number = 0;
  private snapshotTimer: ReturnType<typeof setInterval> | undefined;

  constructor(config: Partial<MCPMetricsConfig> = {}) {
    this.config = { ...DEFAULT_METRICS_CONFIG, ...config };
    this.globalLatencies = new LatencyBuffer(this.config.bufferSize);
  }

  /** Start periodic snapshots */
  start(): void {
    this.snapshotTimer = setInterval(
      () => this.takeSnapshot(),
      this.config.snapshotIntervalMs
    );
  }

  /** Stop periodic snapshots */
  stop(): void {
    if (this.snapshotTimer) {
      clearInterval(this.snapshotTimer);
      this.snapshotTimer = undefined;
    }
  }

  /** Record a successful request */
  recordRequest(serverId: string, latencyMs: number): void {
    this.totalRequests++;
    this.globalLatencies.push(latencyMs);
    this.getOrCreateTracker(serverId).recordSuccess(latencyMs);
  }

  /** Record a failed request */
  recordError(serverId: string): void {
    this.totalRequests++;
    this.totalErrors++;
    this.getOrCreateTracker(serverId).recordError();
  }

  /** Record a cache hit */
  recordCacheHit(): void {
    this.cacheHits++;
  }

  /** Record a cache miss */
  recordCacheMiss(): void {
    this.cacheMisses++;
  }

  /** Track connection count */
  connectionOpened(): void {
    this.activeConnections++;
  }

  connectionClosed(): void {
    this.activeConnections = Math.max(0, this.activeConnections - 1);
  }

  /** Get current snapshot */
  getSnapshot(): MetricSnapshot {
    const totalCacheOps = this.cacheHits + this.cacheMisses;
    const cacheHitRate = totalCacheOps > 0 ? this.cacheHits / totalCacheOps : 0;
    const errorRate =
      this.totalRequests > 0 ? this.totalErrors / this.totalRequests : 0;

    // Quality score: weighted combination of error rate, latency, and cache performance
    const latencyScore = Math.max(
      0,
      1 - this.globalLatencies.percentile(95) / 1000
    );
    const cacheScore = cacheHitRate;
    const reliabilityScore = 1 - errorRate;
    const qualityScore =
      reliabilityScore * 0.5 + latencyScore * 0.3 + cacheScore * 0.2;

    return {
      timestamp: Date.now(),
      totalRequests: this.totalRequests,
      totalErrors: this.totalErrors,
      errorRate: Math.round(errorRate * 10000) / 10000,
      avgResponseMs:
        Math.round(this.globalLatencies.average() * 100) / 100,
      p50ResponseMs:
        Math.round(this.globalLatencies.percentile(50) * 100) / 100,
      p95ResponseMs:
        Math.round(this.globalLatencies.percentile(95) * 100) / 100,
      p99ResponseMs:
        Math.round(this.globalLatencies.percentile(99) * 100) / 100,
      cacheHitRate: Math.round(cacheHitRate * 10000) / 10000,
      activeConnections: this.activeConnections,
      qualityScore: Math.round(qualityScore * 10000) / 10000,
    };
  }

  /** Check if quality meets IHSAN threshold */
  meetsIhsanThreshold(): boolean {
    return this.getSnapshot().qualityScore >= IHSAN_THRESHOLD;
  }

  /** Get per-server metrics */
  getServerMetrics(): PerServerMetrics[] {
    return Array.from(this.servers.values()).map((t) => t.toMetrics());
  }

  /** Get historical snapshots */
  getHistory(): readonly MetricSnapshot[] {
    return this.snapshots;
  }

  private takeSnapshot(): void {
    const snapshot = this.getSnapshot();
    this.snapshots.push(snapshot);
    while (this.snapshots.length > this.config.maxSnapshots) {
      this.snapshots.shift();
    }
  }

  private getOrCreateTracker(serverId: string): ServerTracker {
    let tracker = this.servers.get(serverId);
    if (!tracker) {
      tracker = new ServerTracker(serverId, this.config.bufferSize);
      this.servers.set(serverId, tracker);
    }
    return tracker;
  }
}
