/**
 * Graceful Degradation - Offline-first resilience
 *
 * BIZRA is designed to work offline. Federation is OPTIONAL.
 * This module ensures the system degrades gracefully when:
 * - Network is unavailable
 * - Peers are unreachable
 * - Pool inference fails
 *
 * "We do not assume. We verify with formal proofs."
 */

import { EventEmitter } from 'events';
import { NetworkMode, NetworkStatus, getFallbackMode } from '../sovereign/network-mode';
import { ModelTier } from '../sovereign/capability-card';
import { ModelRegistry, RegisteredModel } from '../sovereign/model-registry';
import { PeerDiscovery } from './peer-discovery';
import { PoolInference, PoolResponse } from './pool-inference';

/**
 * Fallback strategy configuration
 */
export interface FallbackConfig {
  /** Maximum consecutive failures before fallback */
  maxConsecutiveFailures: number;

  /** Recovery check interval in ms */
  recoveryCheckIntervalMs: number;

  /** Timeout for fallback operations in ms */
  fallbackTimeoutMs: number;

  /** Enable automatic recovery attempts */
  autoRecovery: boolean;

  /** Prefer local models over pool when possible */
  preferLocalModels: boolean;
}

const DEFAULT_FALLBACK_CONFIG: FallbackConfig = {
  maxConsecutiveFailures: 3,
  recoveryCheckIntervalMs: 60000,
  fallbackTimeoutMs: 10000,
  autoRecovery: true,
  preferLocalModels: true,
};

/**
 * Degradation state
 */
export interface DegradationState {
  /** Current effective mode */
  currentMode: NetworkMode;

  /** Target/desired mode */
  targetMode: NetworkMode;

  /** Is currently degraded */
  isDegraded: boolean;

  /** Consecutive failure count */
  consecutiveFailures: number;

  /** Last successful operation timestamp */
  lastSuccessAt: Date | null;

  /** Last failure timestamp */
  lastFailureAt: Date | null;

  /** Reason for current degradation */
  degradationReason?: string;
}

/**
 * Graceful Degradation Manager
 */
export class GracefulDegradation extends EventEmitter {
  private config: FallbackConfig;
  private registry: ModelRegistry;
  private discovery?: PeerDiscovery;
  private pool?: PoolInference;
  private state: DegradationState;
  private recoveryTimer?: ReturnType<typeof setInterval>;

  constructor(
    registry: ModelRegistry,
    targetMode: NetworkMode,
    config: Partial<FallbackConfig> = {}
  ) {
    super();
    this.registry = registry;
    this.config = { ...DEFAULT_FALLBACK_CONFIG, ...config };
    this.state = {
      currentMode: targetMode,
      targetMode,
      isDegraded: false,
      consecutiveFailures: 0,
      lastSuccessAt: null,
      lastFailureAt: null,
    };
  }

  /**
   * Set federation components (optional)
   */
  setFederationComponents(discovery: PeerDiscovery, pool: PoolInference): void {
    this.discovery = discovery;
    this.pool = pool;
  }

  /**
   * Start degradation monitoring
   */
  start(): void {
    if (this.config.autoRecovery) {
      this.recoveryTimer = setInterval(
        () => this.attemptRecovery(),
        this.config.recoveryCheckIntervalMs
      );
    }
  }

  /**
   * Stop degradation monitoring
   */
  stop(): void {
    if (this.recoveryTimer) {
      clearInterval(this.recoveryTimer);
      this.recoveryTimer = undefined;
    }
  }

  /**
   * Record a successful operation
   */
  recordSuccess(): void {
    this.state.consecutiveFailures = 0;
    this.state.lastSuccessAt = new Date();

    // Attempt to upgrade if degraded
    if (this.state.isDegraded && this.config.autoRecovery) {
      this.attemptRecovery();
    }

    this.emit('success', this.state);
  }

  /**
   * Record a failed operation
   */
  recordFailure(reason: string): void {
    this.state.consecutiveFailures++;
    this.state.lastFailureAt = new Date();

    // Check if we need to degrade
    if (this.state.consecutiveFailures >= this.config.maxConsecutiveFailures) {
      this.degrade(reason);
    }

    this.emit('failure', { reason, state: this.state });
  }

  /**
   * Trigger degradation to fallback mode
   */
  degrade(reason: string): void {
    const fallbackMode = getFallbackMode(this.state.currentMode);

    if (fallbackMode !== this.state.currentMode) {
      const previousMode = this.state.currentMode;
      this.state.currentMode = fallbackMode;
      this.state.isDegraded = true;
      this.state.degradationReason = reason;

      this.emit('degraded', {
        from: previousMode,
        to: fallbackMode,
        reason,
      });
    }
  }

  /**
   * Attempt to recover to target mode
   */
  async attemptRecovery(): Promise<boolean> {
    if (!this.state.isDegraded) {
      return true;
    }

    // Try to upgrade one level at a time
    const currentMode = this.state.currentMode;
    const targetMode = this.state.targetMode;

    if (currentMode === targetMode) {
      this.state.isDegraded = false;
      this.state.degradationReason = undefined;
      return true;
    }

    // Check if we can upgrade
    const canUpgrade = await this.checkUpgradeability();

    if (canUpgrade) {
      const previousMode = this.state.currentMode;

      // Move up one level
      if (currentMode === NetworkMode.OFFLINE) {
        this.state.currentMode = NetworkMode.LOCAL_ONLY;
      } else if (currentMode === NetworkMode.LOCAL_ONLY) {
        this.state.currentMode = NetworkMode.FEDERATED;
      }

      // Check if we've reached target
      if (this.state.currentMode === targetMode) {
        this.state.isDegraded = false;
        this.state.degradationReason = undefined;
      }

      this.emit('recovered', {
        from: previousMode,
        to: this.state.currentMode,
        fullyRecovered: !this.state.isDegraded,
      });

      return !this.state.isDegraded;
    }

    return false;
  }

  /**
   * Check if we can upgrade to a higher network mode
   */
  private async checkUpgradeability(): Promise<boolean> {
    const currentMode = this.state.currentMode;

    if (currentMode === NetworkMode.OFFLINE) {
      // Check if local network is available
      // In a real implementation, this would ping local services
      return true; // Simulated
    }

    if (currentMode === NetworkMode.LOCAL_ONLY) {
      // Check if federation is available
      if (this.discovery && this.pool) {
        return this.pool.isAvailable();
      }
      return false;
    }

    return true;
  }

  /**
   * Get fallback model for a failed request
   */
  async getFallbackModel(
    originalTier: ModelTier,
    taskType?: string
  ): Promise<RegisteredModel | null> {
    // Tier fallback order
    const fallbackOrder: ModelTier[] = [
      ModelTier.POOL,
      ModelTier.LOCAL,
      ModelTier.EDGE,
    ];

    const startIndex = fallbackOrder.indexOf(originalTier);

    // Try each tier from current to EDGE
    for (let i = startIndex + 1; i < fallbackOrder.length; i++) {
      const tier = fallbackOrder[i];
      const models = this.registry.listByTier(tier);

      // Find a suitable model
      for (const model of models) {
        // Check if model supports the task
        if (taskType) {
          const supportedTasks = model.card.capabilities.tasksSupported;
          if (!supportedTasks.includes(taskType as any)) {
            continue;
          }
        }

        return model;
      }
    }

    // Return any available model as last resort
    return this.registry.getDefault();
  }

  /**
   * Execute with fallback support
   */
  async executeWithFallback<T>(
    primaryFn: () => Promise<T>,
    fallbackFn: () => Promise<T>,
    options: { timeoutMs?: number } = {}
  ): Promise<{ result: T; usedFallback: boolean }> {
    const timeout = options.timeoutMs ?? this.config.fallbackTimeoutMs;

    try {
      // Try primary with timeout
      const result = await Promise.race([
        primaryFn(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), timeout)
        ),
      ]);

      this.recordSuccess();
      return { result, usedFallback: false };
    } catch (error) {
      this.recordFailure(error instanceof Error ? error.message : 'Unknown error');

      // Try fallback
      try {
        const result = await fallbackFn();
        return { result, usedFallback: true };
      } catch (fallbackError) {
        throw new Error(
          `Primary and fallback both failed: ${error}, ${fallbackError}`
        );
      }
    }
  }

  /**
   * Get current state
   */
  getState(): DegradationState {
    return { ...this.state };
  }

  /**
   * Get effective network mode
   */
  getEffectiveMode(): NetworkMode {
    return this.state.currentMode;
  }

  /**
   * Check if currently degraded
   */
  isDegraded(): boolean {
    return this.state.isDegraded;
  }

  /**
   * Force a specific mode (manual override)
   */
  forceMode(mode: NetworkMode): void {
    this.state.currentMode = mode;
    this.state.isDegraded = mode !== this.state.targetMode;
    this.state.consecutiveFailures = 0;

    this.emit('mode-forced', { mode, isDegraded: this.state.isDegraded });
  }
}

/**
 * Create a resilient inference function with automatic fallback
 */
export function createResilientInference(
  degradation: GracefulDegradation,
  localInferenceFn: (modelId: string, prompt: string) => Promise<string>,
  poolInferenceFn?: (prompt: string) => Promise<PoolResponse>
): (
  prompt: string,
  preferredTier: ModelTier
) => Promise<{ content: string; tier: ModelTier; usedFallback: boolean }> {
  return async (prompt, preferredTier) => {
    // If pool preferred and available, try pool first
    if (preferredTier === ModelTier.POOL && poolInferenceFn) {
      try {
        const response = await poolInferenceFn(prompt);
        degradation.recordSuccess();
        return {
          content: response.content,
          tier: ModelTier.POOL,
          usedFallback: false,
        };
      } catch (error) {
        degradation.recordFailure('Pool inference failed');
      }
    }

    // Fall back to local models
    const fallbackModel = await degradation.getFallbackModel(preferredTier);
    if (!fallbackModel) {
      throw new Error('No models available for inference');
    }

    const content = await localInferenceFn(fallbackModel.id, prompt);
    return {
      content,
      tier: fallbackModel.card.tier,
      usedFallback: preferredTier !== fallbackModel.card.tier,
    };
  };
}
