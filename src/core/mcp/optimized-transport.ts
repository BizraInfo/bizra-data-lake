/**
 * Optimized Transport - Request batching and deduplication
 *
 * Batches multiple MCP requests for efficient processing,
 * deduplicates identical in-flight requests, and provides
 * compact serialization.
 */

export interface TransportConfig {
  /** Maximum batch size before flush */
  readonly maxBatchSize: number;

  /** Maximum batch wait time in ms */
  readonly maxBatchWaitMs: number;

  /** Enable request deduplication */
  readonly deduplicateRequests: boolean;

  /** Compact JSON serialization (no whitespace) */
  readonly compactSerialization: boolean;

  /** Request timeout in ms */
  readonly requestTimeoutMs: number;
}

export interface BatchedRequest {
  readonly id: string;
  readonly toolName: string;
  readonly arguments: Record<string, unknown>;
  readonly timestamp: number;
}

export interface BatchedResponse {
  readonly requestId: string;
  readonly result: string | null;
  readonly error: string | null;
  readonly durationMs: number;
}

export interface TransportStats {
  /** Total requests sent */
  readonly totalRequests: number;

  /** Requests batched together */
  readonly totalBatched: number;

  /** Requests deduplicated */
  readonly totalDeduplicated: number;

  /** Average batch size */
  readonly avgBatchSize: number;

  /** Bytes saved by compact serialization */
  readonly bytesSaved: number;
}

const DEFAULT_TRANSPORT_CONFIG: TransportConfig = {
  maxBatchSize: 10,
  maxBatchWaitMs: 50,
  deduplicateRequests: true,
  compactSerialization: true,
  requestTimeoutMs: 30000,
};

type RequestResolver = {
  resolve: (value: string) => void;
  reject: (error: Error) => void;
};

/**
 * Optimized MCP Transport
 */
export class OptimizedTransport {
  private readonly config: TransportConfig;
  private pendingBatch: BatchedRequest[] = [];
  private pendingResolvers: Map<string, RequestResolver> = new Map();
  private inFlightRequests: Map<string, Promise<string>> = new Map();
  private batchTimer: ReturnType<typeof setTimeout> | undefined;
  private idCounter: number = 0;
  private totalRequests: number = 0;
  private totalBatched: number = 0;
  private totalDeduplicated: number = 0;
  private bytesSaved: number = 0;

  constructor(config: Partial<TransportConfig> = {}) {
    this.config = { ...DEFAULT_TRANSPORT_CONFIG, ...config };
  }

  /**
   * Send a request, potentially batching it with others
   */
  async send(
    toolName: string,
    args: Record<string, unknown>,
    executeFn: (requests: BatchedRequest[]) => Promise<BatchedResponse[]>
  ): Promise<string> {
    this.totalRequests++;

    // Deduplication check
    if (this.config.deduplicateRequests) {
      const dedupeKey = this.makeDedupeKey(toolName, args);
      const existing = this.inFlightRequests.get(dedupeKey);
      if (existing) {
        this.totalDeduplicated++;
        return existing;
      }

      const promise = this.enqueueAndWait(toolName, args, executeFn);
      this.inFlightRequests.set(dedupeKey, promise);

      try {
        const result = await promise;
        return result;
      } finally {
        this.inFlightRequests.delete(dedupeKey);
      }
    }

    return this.enqueueAndWait(toolName, args, executeFn);
  }

  /**
   * Serialize data with optional compaction
   */
  serialize(data: unknown): string {
    if (this.config.compactSerialization) {
      const compact = JSON.stringify(data, null, undefined);
      const pretty = JSON.stringify(data, null, 2);
      this.bytesSaved += pretty.length - compact.length;
      return compact;
    }
    return JSON.stringify(data);
  }

  /**
   * Get transport statistics
   */
  getStats(): TransportStats {
    const batches = this.totalBatched > 0
      ? this.totalRequests / this.totalBatched
      : 0;

    return {
      totalRequests: this.totalRequests,
      totalBatched: this.totalBatched,
      totalDeduplicated: this.totalDeduplicated,
      avgBatchSize: Math.round(batches * 100) / 100,
      bytesSaved: this.bytesSaved,
    };
  }

  /**
   * Flush any pending batch immediately
   */
  async flush(
    executeFn: (requests: BatchedRequest[]) => Promise<BatchedResponse[]>
  ): Promise<void> {
    if (this.batchTimer) {
      clearTimeout(this.batchTimer);
      this.batchTimer = undefined;
    }

    if (this.pendingBatch.length > 0) {
      await this.executeBatch(executeFn);
    }
  }

  private async enqueueAndWait(
    toolName: string,
    args: Record<string, unknown>,
    executeFn: (requests: BatchedRequest[]) => Promise<BatchedResponse[]>
  ): Promise<string> {
    const request: BatchedRequest = {
      id: `req_${++this.idCounter}`,
      toolName,
      arguments: args,
      timestamp: Date.now(),
    };

    return new Promise<string>((resolve, reject) => {
      this.pendingResolvers.set(request.id, { resolve, reject });
      this.pendingBatch.push(request);

      // Flush if batch is full
      if (this.pendingBatch.length >= this.config.maxBatchSize) {
        this.executeBatch(executeFn).catch(() => {
          // Errors handled per-request
        });
        return;
      }

      // Set timer for batch flush
      if (!this.batchTimer) {
        this.batchTimer = setTimeout(() => {
          this.batchTimer = undefined;
          this.executeBatch(executeFn).catch(() => {
            // Errors handled per-request
          });
        }, this.config.maxBatchWaitMs);
      }
    });
  }

  private async executeBatch(
    executeFn: (requests: BatchedRequest[]) => Promise<BatchedResponse[]>
  ): Promise<void> {
    const batch = [...this.pendingBatch];
    const resolvers = new Map(this.pendingResolvers);

    this.pendingBatch = [];
    this.pendingResolvers.clear();
    this.totalBatched++;

    try {
      const responses = await Promise.race([
        executeFn(batch),
        new Promise<never>((_, reject) =>
          setTimeout(
            () => reject(new Error('Batch timeout')),
            this.config.requestTimeoutMs
          )
        ),
      ]);

      for (const response of responses) {
        const resolver = resolvers.get(response.requestId);
        if (resolver) {
          if (response.error) {
            resolver.reject(new Error(response.error));
          } else {
            resolver.resolve(response.result ?? '');
          }
          resolvers.delete(response.requestId);
        }
      }

      // Reject any unresolved requests
      for (const [, resolver] of resolvers) {
        resolver.reject(new Error('No response received'));
      }
    } catch (error) {
      // Reject all requests in batch
      for (const [, resolver] of resolvers) {
        resolver.reject(
          error instanceof Error ? error : new Error(String(error))
        );
      }
    }
  }

  private makeDedupeKey(toolName: string, args: Record<string, unknown>): string {
    return `${toolName}:${JSON.stringify(args)}`;
  }
}
