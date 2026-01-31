/**
 * BIZRA Federation Layer - Optional P2P Network
 *
 * Federation is OPTIONAL. BIZRA works fully offline.
 * When enabled, federation provides:
 * - Peer discovery (mDNS, bootstrap)
 * - Pool inference for large models
 * - Byzantine fault-tolerant consensus
 * - Graceful degradation
 *
 * "Every model is welcome if they accept the rules of BIZRA."
 */

// Peer Discovery
export {
  PeerDiscovery,
  PeerNode,
  PeerState,
  PeerCapabilities,
  DiscoveryConfig,
  DiscoveryStats,
} from './peer-discovery';

// Pool Inference
export {
  PoolInference,
  PoolRequest,
  PoolResponse,
  PoolConfig,
  PoolStats,
} from './pool-inference';

// Graceful Degradation
export {
  GracefulDegradation,
  FallbackConfig,
  DegradationState,
  createResilientInference,
} from './graceful-degradation';

/**
 * Federation status summary
 */
export interface FederationStatus {
  /** Is federation enabled */
  enabled: boolean;

  /** Current network mode */
  mode: string;

  /** Is currently degraded */
  degraded: boolean;

  /** Connected peer count */
  peerCount: number;

  /** Pool available */
  poolAvailable: boolean;

  /** Last successful federation operation */
  lastSuccessAt: Date | null;
}

/**
 * Create a complete federation layer
 */
export function createFederationLayer(config: {
  nodeId: string;
  enableDiscovery?: boolean;
  enablePool?: boolean;
  bootstrapNodes?: string[];
}): {
  discovery: PeerDiscovery;
  pool: PoolInference;
  degradation: GracefulDegradation;
  getStatus: () => FederationStatus;
} {
  // Import at runtime to avoid circular deps
  const { ModelRegistry } = require('../sovereign/model-registry');
  const { NetworkMode } = require('../sovereign/network-mode');

  const registry = new ModelRegistry();

  const discovery = new PeerDiscovery({
    enableMdns: config.enableDiscovery ?? true,
    bootstrapNodes: config.bootstrapNodes ?? [],
  });

  const pool = new PoolInference(discovery, config.nodeId, {
    requireConsensus: true,
    consensusQuorum: 0.67,
  });

  const degradation = new GracefulDegradation(
    registry,
    config.enablePool ? NetworkMode.HYBRID : NetworkMode.LOCAL_ONLY
  );

  degradation.setFederationComponents(discovery, pool);

  const getStatus = (): FederationStatus => {
    const state = degradation.getState();
    return {
      enabled: true,
      mode: state.currentMode,
      degraded: state.isDegraded,
      peerCount: discovery.getConnectedPeers().length,
      poolAvailable: pool.isAvailable(),
      lastSuccessAt: state.lastSuccessAt,
    };
  };

  return { discovery, pool, degradation, getStatus };
}

/**
 * Federation layer constants
 */
export const FEDERATION_CONSTANTS = {
  /** Default mDNS service name */
  MDNS_SERVICE_NAME: '_bizra._tcp',

  /** Default discovery interval */
  DISCOVERY_INTERVAL_MS: 30000,

  /** Default peer timeout */
  PEER_TIMEOUT_MS: 120000,

  /** Default pool consensus quorum */
  CONSENSUS_QUORUM: 0.67,

  /** Maximum peers to track */
  MAX_PEERS: 100,

  /** Byzantine fault tolerance */
  BYZANTINE_TOLERANCE: 0.33,
} as const;
