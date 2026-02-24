/**
 * MCP Load Balancer - Intelligent server selection
 *
 * Routes tool calls to the optimal MCP server based on
 * health, latency, and load. Follows ModelRouter.route() pattern.
 */

export enum BalancingStrategy {
  /** Round-robin across healthy servers */
  ROUND_ROBIN = 'round_robin',

  /** Lowest latency server */
  LEAST_LATENCY = 'least_latency',

  /** Least loaded server (fewest active requests) */
  LEAST_LOADED = 'least_loaded',

  /** Weighted random based on health scores */
  WEIGHTED_RANDOM = 'weighted_random',
}

export interface ServerState {
  readonly serverId: string;
  healthy: boolean;
  avgLatencyMs: number;
  activeRequests: number;
  totalRequests: number;
  totalErrors: number;
  weight: number;
  lastSelectedAt: number;
}

export interface BalancerConfig {
  /** Primary balancing strategy */
  readonly strategy: BalancingStrategy;

  /** Unhealthy servers excluded from selection */
  readonly excludeUnhealthy: boolean;

  /** Minimum weight for any server */
  readonly minWeight: number;

  /** Maximum weight for any server */
  readonly maxWeight: number;

  /** Weight decay factor per error (0-1) */
  readonly errorWeightDecay: number;

  /** Weight recovery factor per success (0-1) */
  readonly successWeightRecovery: number;
}

export interface SelectionResult {
  readonly serverId: string;
  readonly strategy: BalancingStrategy;
  readonly reason: string;
  readonly alternateServerId?: string | undefined;
}

const DEFAULT_BALANCER_CONFIG: BalancerConfig = {
  strategy: BalancingStrategy.LEAST_LATENCY,
  excludeUnhealthy: true,
  minWeight: 0.1,
  maxWeight: 1.0,
  errorWeightDecay: 0.8,
  successWeightRecovery: 1.05,
};

/**
 * MCP Load Balancer
 */
export class MCPLoadBalancer {
  private readonly config: BalancerConfig;
  private readonly servers: Map<string, ServerState> = new Map();
  private roundRobinIndex: number = 0;

  constructor(config: Partial<BalancerConfig> = {}) {
    this.config = { ...DEFAULT_BALANCER_CONFIG, ...config };
  }

  /**
   * Register a server
   */
  addServer(serverId: string, weight: number = 1.0): void {
    this.servers.set(serverId, {
      serverId,
      healthy: true,
      avgLatencyMs: 0,
      activeRequests: 0,
      totalRequests: 0,
      totalErrors: 0,
      weight: Math.min(this.config.maxWeight, Math.max(this.config.minWeight, weight)),
      lastSelectedAt: 0,
    });
  }

  /**
   * Remove a server
   */
  removeServer(serverId: string): void {
    this.servers.delete(serverId);
  }

  /**
   * Select the best server for a request
   */
  select(excludeServers: Set<string> = new Set()): SelectionResult | null {
    const candidates = this.getCandidates(excludeServers);
    if (candidates.length === 0) return null;

    let selected: ServerState;
    let reason: string;

    switch (this.config.strategy) {
      case BalancingStrategy.ROUND_ROBIN:
        selected = this.selectRoundRobin(candidates);
        reason = 'Round-robin selection';
        break;

      case BalancingStrategy.LEAST_LATENCY:
        selected = this.selectLeastLatency(candidates);
        reason = `Lowest avg latency: ${selected.avgLatencyMs.toFixed(1)}ms`;
        break;

      case BalancingStrategy.LEAST_LOADED:
        selected = this.selectLeastLoaded(candidates);
        reason = `Fewest active requests: ${selected.activeRequests}`;
        break;

      case BalancingStrategy.WEIGHTED_RANDOM:
        selected = this.selectWeightedRandom(candidates);
        reason = `Weighted random, weight: ${selected.weight.toFixed(2)}`;
        break;

      default:
        selected = candidates[0]!;
        reason = 'Default selection';
    }

    selected.lastSelectedAt = Date.now();
    selected.activeRequests++;
    selected.totalRequests++;

    // Find alternate
    const remaining = candidates.filter((c) => c.serverId !== selected.serverId);
    const alternate = remaining.length > 0
      ? this.selectLeastLatency(remaining)
      : undefined;

    return {
      serverId: selected.serverId,
      strategy: this.config.strategy,
      reason,
      alternateServerId: alternate?.serverId,
    };
  }

  /**
   * Record a successful request
   */
  recordSuccess(serverId: string, latencyMs: number): void {
    const server = this.servers.get(serverId);
    if (!server) return;

    server.activeRequests = Math.max(0, server.activeRequests - 1);

    // EMA for latency
    const alpha = 0.3;
    if (server.totalRequests <= 1) {
      server.avgLatencyMs = latencyMs;
    } else {
      server.avgLatencyMs = alpha * latencyMs + (1 - alpha) * server.avgLatencyMs;
    }

    // Recover weight
    server.weight = Math.min(
      this.config.maxWeight,
      server.weight * this.config.successWeightRecovery
    );
    server.healthy = true;
  }

  /**
   * Record a failed request
   */
  recordFailure(serverId: string): void {
    const server = this.servers.get(serverId);
    if (!server) return;

    server.activeRequests = Math.max(0, server.activeRequests - 1);
    server.totalErrors++;

    // Decay weight
    server.weight = Math.max(
      this.config.minWeight,
      server.weight * this.config.errorWeightDecay
    );

    // Mark unhealthy if error rate too high
    if (server.totalRequests > 5) {
      const errorRate = server.totalErrors / server.totalRequests;
      if (errorRate > 0.5) {
        server.healthy = false;
      }
    }
  }

  /**
   * Update server health status
   */
  setHealth(serverId: string, healthy: boolean): void {
    const server = this.servers.get(serverId);
    if (server) {
      server.healthy = healthy;
    }
  }

  /**
   * Get all server states
   */
  getServerStates(): ServerState[] {
    return Array.from(this.servers.values());
  }

  private getCandidates(excludeServers: Set<string>): ServerState[] {
    const candidates: ServerState[] = [];
    for (const server of this.servers.values()) {
      if (excludeServers.has(server.serverId)) continue;
      if (this.config.excludeUnhealthy && !server.healthy) continue;
      candidates.push(server);
    }
    return candidates;
  }

  private selectRoundRobin(candidates: ServerState[]): ServerState {
    this.roundRobinIndex = this.roundRobinIndex % candidates.length;
    const selected = candidates[this.roundRobinIndex]!;
    this.roundRobinIndex++;
    return selected;
  }

  private selectLeastLatency(candidates: ServerState[]): ServerState {
    let best = candidates[0]!;
    for (let i = 1; i < candidates.length; i++) {
      const c = candidates[i]!;
      if (c.avgLatencyMs < best.avgLatencyMs) {
        best = c;
      }
    }
    return best;
  }

  private selectLeastLoaded(candidates: ServerState[]): ServerState {
    let best = candidates[0]!;
    for (let i = 1; i < candidates.length; i++) {
      const c = candidates[i]!;
      if (c.activeRequests < best.activeRequests) {
        best = c;
      }
    }
    return best;
  }

  private selectWeightedRandom(candidates: ServerState[]): ServerState {
    const totalWeight = candidates.reduce((sum, c) => sum + c.weight, 0);
    let random = Math.random() * totalWeight;

    for (const candidate of candidates) {
      random -= candidate.weight;
      if (random <= 0) {
        return candidate;
      }
    }

    return candidates[candidates.length - 1]!;
  }
}
