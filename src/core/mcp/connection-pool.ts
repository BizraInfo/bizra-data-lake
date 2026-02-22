/**
 * MCP Connection Pool - Managed connections with health checks
 *
 * Pools connections to MCP servers, provides automatic health
 * checking, and integrates with GracefulDegradation for fallback.
 *
 * Target: <400ms server startup, automatic reconnection.
 */

import { EventEmitter } from 'events';

export interface PoolConfig {
  /** Maximum connections per server */
  readonly maxConnectionsPerServer: number;

  /** Minimum idle connections to maintain */
  readonly minIdleConnections: number;

  /** Connection timeout in ms */
  readonly connectionTimeoutMs: number;

  /** Health check interval in ms */
  readonly healthCheckIntervalMs: number;

  /** Max consecutive failures before marking unhealthy */
  readonly maxConsecutiveFailures: number;

  /** Connection idle timeout before cleanup (ms) */
  readonly idleTimeoutMs: number;
}

export interface PoolStats {
  /** Total connections across all servers */
  readonly totalConnections: number;

  /** Active (in-use) connections */
  readonly activeConnections: number;

  /** Idle connections available */
  readonly idleConnections: number;

  /** Per-server connection counts */
  readonly perServer: Record<string, {
    total: number;
    active: number;
    healthy: boolean;
  }>;

  /** Total connections created since startup */
  readonly totalCreated: number;

  /** Total connections destroyed */
  readonly totalDestroyed: number;
}

export enum ConnectionState {
  IDLE = 'idle',
  ACTIVE = 'active',
  UNHEALTHY = 'unhealthy',
  CLOSED = 'closed',
}

export interface ManagedConnection {
  readonly id: string;
  readonly serverId: string;
  state: ConnectionState;
  createdAt: number;
  lastUsedAt: number;
  lastHealthCheckAt: number;
  consecutiveFailures: number;
}

const DEFAULT_POOL_CONFIG: PoolConfig = {
  maxConnectionsPerServer: 4,
  minIdleConnections: 1,
  connectionTimeoutMs: 5000,
  healthCheckIntervalMs: 30000,
  maxConsecutiveFailures: 3,
  idleTimeoutMs: 300000,
};

/**
 * MCP Connection Pool
 */
export class MCPConnectionPool extends EventEmitter {
  private readonly config: PoolConfig;
  private readonly connections: Map<string, ManagedConnection[]> = new Map();
  private readonly serverHealth: Map<string, boolean> = new Map();
  private healthCheckTimer: ReturnType<typeof setInterval> | undefined;
  private idCounter: number = 0;
  private totalCreated: number = 0;
  private totalDestroyed: number = 0;

  constructor(config: Partial<PoolConfig> = {}) {
    super();
    this.config = { ...DEFAULT_POOL_CONFIG, ...config };
  }

  /**
   * Start the connection pool and health monitoring
   */
  start(): void {
    this.healthCheckTimer = setInterval(
      () => this.runHealthChecks(),
      this.config.healthCheckIntervalMs
    );
  }

  /**
   * Stop the pool and close all connections
   */
  stop(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = undefined;
    }

    for (const conns of this.connections.values()) {
      for (const conn of conns) {
        conn.state = ConnectionState.CLOSED;
        this.totalDestroyed++;
      }
    }
    this.connections.clear();
    this.serverHealth.clear();
  }

  /**
   * Register a server in the pool
   */
  registerServer(serverId: string): void {
    if (!this.connections.has(serverId)) {
      this.connections.set(serverId, []);
      this.serverHealth.set(serverId, true);

      // Create minimum idle connections
      for (let i = 0; i < this.config.minIdleConnections; i++) {
        this.createConnection(serverId);
      }
    }
  }

  /**
   * Acquire a connection to a server
   */
  acquire(serverId: string): ManagedConnection | null {
    const conns = this.connections.get(serverId);
    if (!conns) return null;

    // Find an idle connection
    for (const conn of conns) {
      if (conn.state === ConnectionState.IDLE) {
        conn.state = ConnectionState.ACTIVE;
        conn.lastUsedAt = Date.now();
        this.emit('connection-acquired', { serverId, connectionId: conn.id });
        return conn;
      }
    }

    // Create a new connection if under limit
    if (conns.length < this.config.maxConnectionsPerServer) {
      const conn = this.createConnection(serverId);
      conn.state = ConnectionState.ACTIVE;
      conn.lastUsedAt = Date.now();
      this.emit('connection-acquired', { serverId, connectionId: conn.id });
      return conn;
    }

    // Pool exhausted
    this.emit('pool-exhausted', { serverId });
    return null;
  }

  /**
   * Release a connection back to the pool
   */
  release(connection: ManagedConnection): void {
    connection.state = ConnectionState.IDLE;
    connection.lastUsedAt = Date.now();
    connection.consecutiveFailures = 0;
    this.emit('connection-released', {
      serverId: connection.serverId,
      connectionId: connection.id,
    });
  }

  /**
   * Mark a connection as failed
   */
  markFailed(connection: ManagedConnection): void {
    connection.consecutiveFailures++;

    if (connection.consecutiveFailures >= this.config.maxConsecutiveFailures) {
      connection.state = ConnectionState.UNHEALTHY;
      this.emit('connection-unhealthy', {
        serverId: connection.serverId,
        connectionId: connection.id,
      });

      // Check if all connections are unhealthy
      this.checkServerHealth(connection.serverId);
    } else {
      connection.state = ConnectionState.IDLE;
    }
  }

  /**
   * Check if a server is healthy
   */
  isServerHealthy(serverId: string): boolean {
    return this.serverHealth.get(serverId) ?? false;
  }

  /**
   * Get list of healthy servers
   */
  getHealthyServers(): string[] {
    const healthy: string[] = [];
    for (const [serverId, isHealthy] of this.serverHealth) {
      if (isHealthy) {
        healthy.push(serverId);
      }
    }
    return healthy;
  }

  /**
   * Get pool statistics
   */
  getStats(): PoolStats {
    let totalConnections = 0;
    let activeConnections = 0;
    let idleConnections = 0;
    const perServer: PoolStats['perServer'] = {};

    for (const [serverId, conns] of this.connections) {
      let serverTotal = 0;
      let serverActive = 0;

      for (const conn of conns) {
        if (conn.state !== ConnectionState.CLOSED) {
          serverTotal++;
          totalConnections++;
        }
        if (conn.state === ConnectionState.ACTIVE) {
          serverActive++;
          activeConnections++;
        }
        if (conn.state === ConnectionState.IDLE) {
          idleConnections++;
        }
      }

      perServer[serverId] = {
        total: serverTotal,
        active: serverActive,
        healthy: this.serverHealth.get(serverId) ?? false,
      };
    }

    return {
      totalConnections,
      activeConnections,
      idleConnections,
      perServer,
      totalCreated: this.totalCreated,
      totalDestroyed: this.totalDestroyed,
    };
  }

  /**
   * Execute a function with a pooled connection (auto acquire/release)
   */
  async withConnection<T>(
    serverId: string,
    fn: (conn: ManagedConnection) => Promise<T>
  ): Promise<T> {
    const conn = this.acquire(serverId);
    if (!conn) {
      throw new Error(`No available connections for server ${serverId}`);
    }

    try {
      const result = await Promise.race([
        fn(conn),
        new Promise<never>((_, reject) =>
          setTimeout(
            () => reject(new Error('Connection timeout')),
            this.config.connectionTimeoutMs
          )
        ),
      ]);
      this.release(conn);
      return result;
    } catch (error) {
      this.markFailed(conn);
      throw error;
    }
  }

  private createConnection(serverId: string): ManagedConnection {
    const now = Date.now();
    const conn: ManagedConnection = {
      id: `conn_${++this.idCounter}`,
      serverId,
      state: ConnectionState.IDLE,
      createdAt: now,
      lastUsedAt: now,
      lastHealthCheckAt: now,
      consecutiveFailures: 0,
    };

    const conns = this.connections.get(serverId);
    if (conns) {
      conns.push(conn);
    }
    this.totalCreated++;
    this.emit('connection-created', { serverId, connectionId: conn.id });
    return conn;
  }

  private runHealthChecks(): void {
    const now = Date.now();

    for (const [serverId, conns] of this.connections) {
      // Clean up idle connections that have timed out
      const toRemove: number[] = [];

      for (let i = 0; i < conns.length; i++) {
        const conn = conns[i];
        if (!conn) continue;

        if (
          conn.state === ConnectionState.IDLE &&
          now - conn.lastUsedAt > this.config.idleTimeoutMs &&
          conns.length > this.config.minIdleConnections
        ) {
          conn.state = ConnectionState.CLOSED;
          toRemove.push(i);
          this.totalDestroyed++;
        }

        // Reset unhealthy connections
        if (conn.state === ConnectionState.UNHEALTHY) {
          conn.state = ConnectionState.IDLE;
          conn.consecutiveFailures = 0;
          conn.lastHealthCheckAt = now;
        }
      }

      // Remove closed connections (reverse order to preserve indices)
      for (let i = toRemove.length - 1; i >= 0; i--) {
        const idx = toRemove[i];
        if (idx !== undefined) {
          conns.splice(idx, 1);
        }
      }

      // Ensure minimum idle connections
      const idleCount = conns.filter(
        (c) => c.state === ConnectionState.IDLE
      ).length;
      for (let i = idleCount; i < this.config.minIdleConnections; i++) {
        this.createConnection(serverId);
      }

      this.checkServerHealth(serverId);
    }
  }

  private checkServerHealth(serverId: string): void {
    const conns = this.connections.get(serverId);
    if (!conns || conns.length === 0) {
      this.serverHealth.set(serverId, false);
      return;
    }

    const healthyCount = conns.filter(
      (c) => c.state !== ConnectionState.UNHEALTHY && c.state !== ConnectionState.CLOSED
    ).length;
    const wasHealthy = this.serverHealth.get(serverId) ?? true;
    const isHealthy = healthyCount > 0;

    this.serverHealth.set(serverId, isHealthy);

    if (wasHealthy && !isHealthy) {
      this.emit('server-unhealthy', { serverId });
    } else if (!wasHealthy && isHealthy) {
      this.emit('server-recovered', { serverId });
    }
  }
}
