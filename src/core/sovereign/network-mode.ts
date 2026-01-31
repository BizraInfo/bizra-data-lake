/**
 * Network Mode - Defines how BIZRA connects to the world
 *
 * BIZRA is offline-first. Federation is OPTIONAL, not required.
 * The system degrades gracefully from federated → local → offline.
 */

export enum NetworkMode {
  /** Zero network access - full sovereignty */
  OFFLINE = 'offline',

  /** LAN discovery only - no internet */
  LOCAL_ONLY = 'local',

  /** Full federation participation */
  FEDERATED = 'federated',

  /** Offline-first, federate when available (recommended) */
  HYBRID = 'hybrid',
}

export interface NetworkModeConfig {
  mode: NetworkMode;

  /** Discovery timeout in ms (only for LOCAL/FEDERATED/HYBRID) */
  discoveryTimeoutMs: number;

  /** Federation connection timeout in ms */
  federationTimeoutMs: number;

  /** Retry interval for failed connections */
  retryIntervalMs: number;

  /** Maximum retry attempts before falling back */
  maxRetries: number;

  /** Allowed peer networks (CIDR notation) */
  allowedNetworks: string[];

  /** Blocked peer networks (CIDR notation) */
  blockedNetworks: string[];
}

export const DEFAULT_NETWORK_CONFIG: NetworkModeConfig = {
  mode: NetworkMode.HYBRID,
  discoveryTimeoutMs: 5000,
  federationTimeoutMs: 10000,
  retryIntervalMs: 30000,
  maxRetries: 3,
  allowedNetworks: ['10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'],
  blockedNetworks: [],
};

/**
 * Determine if a network mode allows external connections
 */
export function allowsExternalConnections(mode: NetworkMode): boolean {
  return mode !== NetworkMode.OFFLINE;
}

/**
 * Determine if a network mode allows internet access
 */
export function allowsInternetAccess(mode: NetworkMode): boolean {
  return mode === NetworkMode.FEDERATED || mode === NetworkMode.HYBRID;
}

/**
 * Get the fallback mode when current mode fails
 */
export function getFallbackMode(mode: NetworkMode): NetworkMode {
  switch (mode) {
    case NetworkMode.FEDERATED:
      return NetworkMode.LOCAL_ONLY;
    case NetworkMode.LOCAL_ONLY:
      return NetworkMode.OFFLINE;
    case NetworkMode.HYBRID:
      return NetworkMode.LOCAL_ONLY;
    case NetworkMode.OFFLINE:
      return NetworkMode.OFFLINE; // No further fallback
  }
}

/**
 * Network status tracking
 */
export interface NetworkStatus {
  currentMode: NetworkMode;
  effectiveMode: NetworkMode;
  isOnline: boolean;
  peerCount: number;
  poolAvailable: boolean;
  lastDiscoveryAt: Date | null;
  lastFederationAt: Date | null;
  consecutiveFailures: number;
}

export function createInitialNetworkStatus(mode: NetworkMode): NetworkStatus {
  return {
    currentMode: mode,
    effectiveMode: mode,
    isOnline: mode !== NetworkMode.OFFLINE,
    peerCount: 0,
    poolAvailable: false,
    lastDiscoveryAt: null,
    lastFederationAt: null,
    consecutiveFailures: 0,
  };
}
