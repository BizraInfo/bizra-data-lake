/**
 * Peer Discovery - Find and connect to BIZRA nodes
 *
 * Supports multiple discovery mechanisms:
 * - Local network multicast (mDNS)
 * - Bootstrap nodes (for initial federation)
 * - Gossip protocol (peer exchange)
 */

import { EventEmitter } from 'events';
import { createHash } from 'crypto';

/**
 * Peer node information
 */
export interface PeerNode {
  /** Unique peer ID (derived from public key) */
  id: string;

  /** Display name */
  name: string;

  /** Network addresses (IP:port) */
  addresses: string[];

  /** Public key for verification (hex) */
  publicKey: string;

  /** Supported capabilities */
  capabilities: PeerCapabilities;

  /** Last seen timestamp */
  lastSeen: Date;

  /** Latency in ms (0 = unknown) */
  latencyMs: number;

  /** Connection state */
  state: PeerState;
}

export enum PeerState {
  DISCOVERED = 'discovered',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  BANNED = 'banned',
}

export interface PeerCapabilities {
  /** Model tiers available */
  tiers: ('EDGE' | 'LOCAL' | 'POOL')[];

  /** Can accept pool inference requests */
  poolEnabled: boolean;

  /** Maximum concurrent requests */
  maxConcurrent: number;

  /** Supported task types */
  tasks: string[];
}

/**
 * Discovery configuration
 */
export interface DiscoveryConfig {
  /** Enable mDNS local discovery */
  enableMdns: boolean;

  /** mDNS service name */
  mdnsServiceName: string;

  /** Bootstrap nodes for initial connection */
  bootstrapNodes: string[];

  /** Discovery interval in ms */
  discoveryIntervalMs: number;

  /** Peer timeout in ms (consider dead after) */
  peerTimeoutMs: number;

  /** Maximum peers to track */
  maxPeers: number;
}

const DEFAULT_DISCOVERY_CONFIG: DiscoveryConfig = {
  enableMdns: true,
  mdnsServiceName: '_bizra._tcp',
  bootstrapNodes: [],
  discoveryIntervalMs: 30000,
  peerTimeoutMs: 120000,
  maxPeers: 100,
};

/**
 * Peer Discovery Service
 */
export class PeerDiscovery extends EventEmitter {
  private config: DiscoveryConfig;
  private peers: Map<string, PeerNode> = new Map();
  private discoveryTimer?: ReturnType<typeof setInterval>;
  private running: boolean = false;

  constructor(config: Partial<DiscoveryConfig> = {}) {
    super();
    this.config = { ...DEFAULT_DISCOVERY_CONFIG, ...config };
  }

  /**
   * Start peer discovery
   */
  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;

    // Initial discovery
    await this.discover();

    // Start periodic discovery
    this.discoveryTimer = setInterval(
      () => this.discover(),
      this.config.discoveryIntervalMs
    );

    this.emit('started');
  }

  /**
   * Stop peer discovery
   */
  async stop(): Promise<void> {
    if (!this.running) return;
    this.running = false;

    if (this.discoveryTimer) {
      clearInterval(this.discoveryTimer);
      this.discoveryTimer = undefined;
    }

    this.emit('stopped');
  }

  /**
   * Run discovery
   */
  private async discover(): Promise<void> {
    // Clean up stale peers
    this.cleanupStalePeers();

    // mDNS discovery
    if (this.config.enableMdns) {
      await this.discoverMdns();
    }

    // Bootstrap nodes
    if (this.config.bootstrapNodes.length > 0) {
      await this.discoverBootstrap();
    }

    this.emit('discovery-complete', this.getPeers());
  }

  /**
   * mDNS local network discovery
   */
  private async discoverMdns(): Promise<void> {
    // In a real implementation, this would use node-mdns or similar
    // For now, we simulate the discovery process
    this.emit('mdns-discovery-started');

    // Simulated delay for discovery
    await new Promise((resolve) => setTimeout(resolve, 100));

    this.emit('mdns-discovery-complete');
  }

  /**
   * Bootstrap node discovery
   */
  private async discoverBootstrap(): Promise<void> {
    for (const address of this.config.bootstrapNodes) {
      try {
        // In a real implementation, this would connect and exchange peer info
        const peerId = this.addressToPeerId(address);

        if (!this.peers.has(peerId)) {
          const peer: PeerNode = {
            id: peerId,
            name: `bootstrap-${peerId.slice(0, 8)}`,
            addresses: [address],
            publicKey: '',
            capabilities: {
              tiers: ['LOCAL'],
              poolEnabled: true,
              maxConcurrent: 10,
              tasks: ['chat', 'reasoning'],
            },
            lastSeen: new Date(),
            latencyMs: 0,
            state: PeerState.DISCOVERED,
          };

          this.addPeer(peer);
        }
      } catch (error) {
        this.emit('bootstrap-error', { address, error });
      }
    }
  }

  /**
   * Add a discovered peer
   */
  addPeer(peer: PeerNode): void {
    if (this.peers.size >= this.config.maxPeers) {
      // Remove oldest peer to make room
      const oldest = this.getOldestPeer();
      if (oldest) {
        this.removePeer(oldest.id);
      }
    }

    this.peers.set(peer.id, peer);
    this.emit('peer-added', peer);
  }

  /**
   * Update peer information
   */
  updatePeer(peerId: string, update: Partial<PeerNode>): void {
    const peer = this.peers.get(peerId);
    if (peer) {
      Object.assign(peer, update, { lastSeen: new Date() });
      this.emit('peer-updated', peer);
    }
  }

  /**
   * Remove a peer
   */
  removePeer(peerId: string): void {
    const peer = this.peers.get(peerId);
    if (peer) {
      this.peers.delete(peerId);
      this.emit('peer-removed', peer);
    }
  }

  /**
   * Get all known peers
   */
  getPeers(): PeerNode[] {
    return Array.from(this.peers.values());
  }

  /**
   * Get connected peers only
   */
  getConnectedPeers(): PeerNode[] {
    return this.getPeers().filter((p) => p.state === PeerState.CONNECTED);
  }

  /**
   * Get peers with pool capability
   */
  getPoolPeers(): PeerNode[] {
    return this.getPeers().filter(
      (p) => p.capabilities.poolEnabled && p.state === PeerState.CONNECTED
    );
  }

  /**
   * Get peer by ID
   */
  getPeer(peerId: string): PeerNode | undefined {
    return this.peers.get(peerId);
  }

  /**
   * Clean up stale peers
   */
  private cleanupStalePeers(): void {
    const now = Date.now();
    const timeout = this.config.peerTimeoutMs;

    for (const [id, peer] of this.peers) {
      if (now - peer.lastSeen.getTime() > timeout) {
        peer.state = PeerState.DISCONNECTED;
        this.emit('peer-timeout', peer);
      }
    }
  }

  /**
   * Get the oldest peer by lastSeen
   */
  private getOldestPeer(): PeerNode | undefined {
    let oldest: PeerNode | undefined;
    for (const peer of this.peers.values()) {
      if (!oldest || peer.lastSeen < oldest.lastSeen) {
        oldest = peer;
      }
    }
    return oldest;
  }

  /**
   * Convert address to peer ID
   */
  private addressToPeerId(address: string): string {
    return createHash('sha256').update(address).digest('hex').slice(0, 16);
  }

  /**
   * Get discovery statistics
   */
  getStats(): DiscoveryStats {
    const peers = this.getPeers();
    return {
      totalPeers: peers.length,
      connectedPeers: peers.filter((p) => p.state === PeerState.CONNECTED).length,
      poolPeers: this.getPoolPeers().length,
      discoveryRuns: 0, // Would be tracked in real impl
      lastDiscoveryAt: new Date(),
    };
  }
}

export interface DiscoveryStats {
  totalPeers: number;
  connectedPeers: number;
  poolPeers: number;
  discoveryRuns: number;
  lastDiscoveryAt: Date;
}
