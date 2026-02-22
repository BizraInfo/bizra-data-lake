/**
 * MCP Optimization Layer - Barrel exports
 *
 * Provides high-performance MCP server management with:
 * - Connection pooling with health checks
 * - O(1) tool lookup via Map-based registry
 * - Multi-level caching (L1 Map + L2 LRU)
 * - Intelligent load balancing
 * - Request batching and deduplication
 * - Real-time performance metrics
 *
 * Performance Targets:
 * - Tool lookup: <5ms (O(1) via Map)
 * - Cache hit rate: >90%
 * - Response p95: <100ms
 * - Server startup: <400ms
 */

export {
  MCPConnectionPool,
  type PoolConfig,
  type PoolStats,
  type ManagedConnection,
  ConnectionState,
} from './connection-pool';

export {
  FastToolRegistry,
  type ToolEntry,
  type RegistryStats,
} from './fast-tool-registry';

export {
  MCPLoadBalancer,
  BalancingStrategy,
  type ServerState,
  type BalancerConfig,
  type SelectionResult,
} from './load-balancer';

export {
  MCPMetrics,
  type MCPMetricsConfig,
  type MetricSnapshot,
  type PerServerMetrics,
} from './metrics';

export {
  MultiLevelCache,
  type CacheConfig,
  type CacheStats,
} from './multi-level-cache';

export {
  OptimizedTransport,
  type TransportConfig,
  type TransportStats,
  type BatchedRequest,
  type BatchedResponse,
} from './optimized-transport';
