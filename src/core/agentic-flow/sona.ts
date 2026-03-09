/**
 * BIZRA Agentic-Flow — SONA Learning Mode Manager
 *
 * Five learning modes that map to the Triple Helix (§2).
 * Controls which helix is primary, how many agents activate,
 * and how fast adaptation occurs.
 *
 * Standing on Giants:
 *   Kahneman (dual-process, 2011) · Deming (PDCA cycle, 1950) ·
 *   Boyd (OODA loop, 1976)
 *
 * Reference: Spine §2 (Triple Helix), §9 (Growth)
 */

import {
  type AgentId,
  type SONAConfig,
  Helix,
  SONAMode,
  SONA_CONFIGS,
  CONSTITUTIONAL,
} from './types';

/** Snapshot of SONA state at a point in time */
export interface SONASnapshot {
  readonly mode: SONAMode;
  readonly config: SONAConfig;
  readonly activeAgents: readonly AgentId[];
  readonly adaptationLatencyMs: number;
  readonly heartbeatCount: number;
  readonly lastTransitionAt: number;
}

/** SONA transition event */
export interface SONATransition {
  readonly from: SONAMode;
  readonly to: SONAMode;
  readonly reason: string;
  readonly timestamp: number;
}

export type SONAListener = (transition: SONATransition) => void;

/**
 * SONAManager — Manages the 5 learning modes
 *
 * Modes map to operational context:
 * - real-time (0.05ms): Helix 1, reflexive, 2 agents max
 * - balanced (50ms):    Helix 2, general-purpose, 4 agents
 * - research (500ms):   Helix 2, deep exploration, 7 agents
 * - edge (10ms):        Helix 1, resource-constrained, 2 agents
 * - batch (1000ms):     Helix 3, high-throughput, all 12 agents
 */
export class SONAManager {
  private currentMode: SONAMode;
  private activeAgents: AgentId[] = [];
  private heartbeatCount = 0;
  private lastTransitionAt: number = Date.now();
  private readonly listeners: Set<SONAListener> = new Set();
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  constructor(initialMode: SONAMode = SONAMode.BALANCED) {
    this.currentMode = initialMode;
  }

  /** Get current configuration */
  getConfig(): SONAConfig {
    const config = SONA_CONFIGS.get(this.currentMode);
    if (!config) {
      throw new Error(`Unknown SONA mode: ${this.currentMode}`);
    }
    return config;
  }

  /** Transition to a new mode */
  setMode(mode: SONAMode, reason: string = 'manual'): void {
    if (mode === this.currentMode) return;

    const transition: SONATransition = {
      from: this.currentMode,
      to: mode,
      reason,
      timestamp: Date.now(),
    };

    this.currentMode = mode;
    this.lastTransitionAt = Date.now();

    for (const listener of this.listeners) {
      listener(transition);
    }
  }

  /** Get current mode */
  getMode(): SONAMode {
    return this.currentMode;
  }

  /** Get the primary helix for current mode */
  getPrimaryHelix(): Helix {
    return this.getConfig().primaryHelix;
  }

  /** Get max concurrent agents for current mode */
  getMaxConcurrentAgents(): number {
    return this.getConfig().maxConcurrentAgents;
  }

  /** Update active agent set (called by agent router) */
  setActiveAgents(agents: readonly AgentId[]): void {
    const max = this.getMaxConcurrentAgents();
    this.activeAgents = agents.slice(0, max) as AgentId[];
  }

  /** Get current snapshot */
  snapshot(): SONASnapshot {
    const config = this.getConfig();
    return {
      mode: this.currentMode,
      config,
      activeAgents: [...this.activeAgents],
      adaptationLatencyMs: config.adaptationMs,
      heartbeatCount: this.heartbeatCount,
      lastTransitionAt: this.lastTransitionAt,
    };
  }

  /**
   * Start the Helix 3 evolutionary heartbeat (60s constitutional cycle).
   * Each tick increments heartbeatCount and can trigger mode adaptation.
   */
  startHeartbeat(onTick?: () => void): void {
    if (this.heartbeatTimer) return;

    this.heartbeatTimer = setInterval(() => {
      this.heartbeatCount++;
      onTick?.();
    }, CONSTITUTIONAL.HEARTBEAT_MS);

    // Prevent timer from blocking Node.js exit
    if (this.heartbeatTimer && typeof this.heartbeatTimer === 'object' && 'unref' in this.heartbeatTimer) {
      this.heartbeatTimer.unref();
    }
  }

  /** Stop the evolutionary heartbeat */
  stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /** Subscribe to mode transitions */
  onTransition(listener: SONAListener): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }

  /**
   * Auto-select mode based on operational context.
   * This implements the adaptive intelligence: the organism
   * shifts its processing mode based on load and urgency.
   */
  autoSelect(context: {
    pendingMissions: number;
    avgLatencyMs: number;
    availableMemoryMB: number;
    isEdgeDevice: boolean;
  }): SONAMode {
    let selected: SONAMode;

    if (context.isEdgeDevice || context.availableMemoryMB < 512) {
      selected = SONAMode.EDGE;
    } else if (context.pendingMissions > 10) {
      selected = SONAMode.BATCH;
    } else if (context.avgLatencyMs < 100) {
      selected = SONAMode.REAL_TIME;
    } else if (context.avgLatencyMs > 1000) {
      selected = SONAMode.RESEARCH;
    } else {
      selected = SONAMode.BALANCED;
    }

    if (selected !== this.currentMode) {
      this.setMode(selected, 'auto-select');
    }
    return selected;
  }
}
