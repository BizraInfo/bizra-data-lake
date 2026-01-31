/**
 * Model Router - Intelligent model selection and routing
 *
 * Routes inference requests to the most appropriate model based on:
 * - Task type and complexity
 * - Available models and their capabilities
 * - Network mode (offline/local/federated)
 * - Current load and latency requirements
 */

import {
  ModelTier,
  TaskType,
  IHSAN_THRESHOLD,
  SNR_THRESHOLD,
} from './capability-card';
import { ModelRegistry, RegisteredModel, ModelSelectionCriteria } from './model-registry';
import { NetworkMode, NetworkStatus, getFallbackMode } from './network-mode';

/**
 * Task complexity levels
 */
export enum TaskComplexity {
  /** Simple queries, classification, short responses */
  LOW = 'low',

  /** Standard reasoning, chat, summarization */
  MEDIUM = 'medium',

  /** Complex reasoning, code generation, long context */
  HIGH = 'high',
}

/**
 * Routing decision
 */
export interface RoutingDecision {
  /** Selected model ID */
  modelId: string;

  /** Model tier */
  tier: ModelTier;

  /** Whether to use federation pool */
  usePool: boolean;

  /** Fallback model if primary fails */
  fallbackModelId?: string;

  /** Routing confidence (0-1) */
  confidence: number;

  /** Reason for selection */
  reason: string;
}

/**
 * Routing request
 */
export interface RoutingRequest {
  /** Task type */
  taskType: TaskType;

  /** Task complexity */
  complexity: TaskComplexity;

  /** Estimated input tokens */
  inputTokens?: number;

  /** Maximum allowed latency in ms */
  maxLatencyMs?: number;

  /** Prefer specific tier */
  preferTier?: ModelTier;

  /** Exclude specific models */
  excludeModels?: string[];
}

/**
 * Model Router configuration
 */
export interface RouterConfig {
  /** Complexity to tier mapping */
  complexityTierMapping: Record<TaskComplexity, ModelTier[]>;

  /** Default fallback tier when preferred unavailable */
  defaultFallbackTier: ModelTier;

  /** Enable pool inference for large models */
  enablePoolInference: boolean;

  /** Maximum queue depth before rejecting */
  maxQueueDepth: number;
}

const DEFAULT_ROUTER_CONFIG: RouterConfig = {
  complexityTierMapping: {
    [TaskComplexity.LOW]: [ModelTier.EDGE, ModelTier.LOCAL],
    [TaskComplexity.MEDIUM]: [ModelTier.LOCAL, ModelTier.EDGE, ModelTier.POOL],
    [TaskComplexity.HIGH]: [ModelTier.POOL, ModelTier.LOCAL],
  },
  defaultFallbackTier: ModelTier.EDGE,
  enablePoolInference: true,
  maxQueueDepth: 100,
};

/**
 * Model Router
 */
export class ModelRouter {
  private registry: ModelRegistry;
  private networkStatus: NetworkStatus;
  private config: RouterConfig;
  private queueDepth: number = 0;

  constructor(
    registry: ModelRegistry,
    networkStatus: NetworkStatus,
    config: Partial<RouterConfig> = {}
  ) {
    this.registry = registry;
    this.networkStatus = networkStatus;
    this.config = { ...DEFAULT_ROUTER_CONFIG, ...config };
  }

  /**
   * Update network status
   */
  updateNetworkStatus(status: NetworkStatus): void {
    this.networkStatus = status;
  }

  /**
   * Route a request to the best available model
   */
  route(request: RoutingRequest): RoutingDecision {
    // Get tier preference based on complexity
    const preferredTiers = this.config.complexityTierMapping[request.complexity];
    const effectiveTiers = this.filterAvailableTiers(preferredTiers);

    if (effectiveTiers.length === 0) {
      // Emergency fallback to any available model
      const anyModel = this.registry.getDefault();
      if (anyModel) {
        return {
          modelId: anyModel.id,
          tier: anyModel.card.tier,
          usePool: false,
          confidence: 0.3,
          reason: 'Emergency fallback - no preferred tier models available',
        };
      }

      throw new Error('No models available for routing');
    }

    // Build selection criteria
    const criteria: ModelSelectionCriteria = {
      taskType: request.taskType,
      preferredTier: effectiveTiers[0],
      maxLatencyMs: request.maxLatencyMs,
      minContext: request.inputTokens ? request.inputTokens * 2 : undefined,
      excludeModels: request.excludeModels,
    };

    // Select best model
    const primary = this.registry.selectBest(criteria);

    if (!primary) {
      throw new Error(
        `No suitable model found for task ${request.taskType} at complexity ${request.complexity}`
      );
    }

    // Determine if we should use pool
    const usePool = this.shouldUsePool(primary, request);

    // Select fallback if available
    const fallback = this.selectFallback(primary, criteria);

    // Calculate confidence
    const confidence = this.calculateConfidence(primary, request, usePool);

    return {
      modelId: primary.id,
      tier: primary.card.tier,
      usePool,
      fallbackModelId: fallback?.id,
      confidence,
      reason: this.buildReason(primary, request, usePool),
    };
  }

  /**
   * Filter tiers based on network mode
   */
  private filterAvailableTiers(tiers: ModelTier[]): ModelTier[] {
    const effectiveMode = this.networkStatus.effectiveMode;

    return tiers.filter((tier) => {
      // POOL tier requires federation
      if (tier === ModelTier.POOL) {
        if (effectiveMode === NetworkMode.OFFLINE) {
          return false;
        }
        if (!this.networkStatus.poolAvailable) {
          return false;
        }
        if (!this.config.enablePoolInference) {
          return false;
        }
      }

      // Check if we have models in this tier
      const models = this.registry.listByTier(tier);
      return models.length > 0;
    });
  }

  /**
   * Determine if pool should be used
   */
  private shouldUsePool(model: RegisteredModel, request: RoutingRequest): boolean {
    // Only pool tier uses federation
    if (model.card.tier !== ModelTier.POOL) {
      return false;
    }

    // Check federation availability
    if (!this.networkStatus.poolAvailable) {
      return false;
    }

    // Check complexity warrants pool
    if (request.complexity === TaskComplexity.LOW) {
      return false;
    }

    return true;
  }

  /**
   * Select a fallback model
   */
  private selectFallback(
    primary: RegisteredModel,
    criteria: ModelSelectionCriteria
  ): RegisteredModel | null {
    // Try same tier first
    const sameTier = this.registry.selectBest({
      ...criteria,
      excludeModels: [...(criteria.excludeModels ?? []), primary.id],
    });

    if (sameTier) {
      return sameTier;
    }

    // Fall back to EDGE tier
    return this.registry.selectBest({
      ...criteria,
      preferredTier: this.config.defaultFallbackTier,
      excludeModels: [...(criteria.excludeModels ?? []), primary.id],
    });
  }

  /**
   * Calculate routing confidence
   */
  private calculateConfidence(
    model: RegisteredModel,
    request: RoutingRequest,
    usePool: boolean
  ): number {
    let confidence = 0.5;

    // Model quality boost
    const ihsanBoost = (model.card.capabilities.ihsanScore - IHSAN_THRESHOLD) * 2;
    const snrBoost = (model.card.capabilities.snrScore - SNR_THRESHOLD) * 2;
    confidence += ihsanBoost + snrBoost;

    // Task match boost
    if (model.card.capabilities.tasksSupported.includes(request.taskType)) {
      confidence += 0.15;
    }

    // Tier match boost
    const preferredTiers = this.config.complexityTierMapping[request.complexity];
    if (preferredTiers[0] === model.card.tier) {
      confidence += 0.1;
    }

    // Pool penalty (network uncertainty)
    if (usePool) {
      confidence -= 0.1;
    }

    // Queue depth penalty
    const queuePenalty = Math.min(0.2, this.queueDepth / this.config.maxQueueDepth);
    confidence -= queuePenalty;

    return Math.max(0, Math.min(1, confidence));
  }

  /**
   * Build reason string
   */
  private buildReason(
    model: RegisteredModel,
    request: RoutingRequest,
    usePool: boolean
  ): string {
    const parts: string[] = [];

    parts.push(`Selected ${model.card.tier} model ${model.id}`);
    parts.push(`for ${request.taskType} (${request.complexity} complexity)`);

    if (usePool) {
      parts.push('via federation pool');
    }

    parts.push(`- Ihsān: ${(model.card.capabilities.ihsanScore * 100).toFixed(1)}%`);
    parts.push(`SNR: ${(model.card.capabilities.snrScore * 100).toFixed(1)}%`);

    return parts.join(' ');
  }

  /**
   * Pre-flight check before routing
   */
  canRoute(request: RoutingRequest): { canRoute: boolean; reason?: string } {
    // Check queue depth
    if (this.queueDepth >= this.config.maxQueueDepth) {
      return { canRoute: false, reason: 'Queue full' };
    }

    // Check if any models available
    const stats = this.registry.getStats();
    if (stats.validModels === 0) {
      return { canRoute: false, reason: 'No valid models registered' };
    }

    // Check network for pool requests
    if (
      request.complexity === TaskComplexity.HIGH &&
      !this.networkStatus.poolAvailable &&
      this.registry.listByTier(ModelTier.LOCAL).length === 0
    ) {
      return { canRoute: false, reason: 'No local or pool models for high complexity' };
    }

    return { canRoute: true };
  }

  /**
   * Increment queue depth
   */
  enqueue(): void {
    this.queueDepth++;
  }

  /**
   * Decrement queue depth
   */
  dequeue(): void {
    this.queueDepth = Math.max(0, this.queueDepth - 1);
  }

  /**
   * Get current queue depth
   */
  getQueueDepth(): number {
    return this.queueDepth;
  }
}

/**
 * Estimate task complexity from request
 */
export function estimateComplexity(params: {
  inputTokens?: number;
  requiresReasoning?: boolean;
  requiresCodeGen?: boolean;
  contextLength?: number;
}): TaskComplexity {
  // High complexity indicators
  if (params.requiresCodeGen) {
    return TaskComplexity.HIGH;
  }

  if (params.contextLength && params.contextLength > 4096) {
    return TaskComplexity.HIGH;
  }

  if (params.inputTokens && params.inputTokens > 2000) {
    return TaskComplexity.HIGH;
  }

  // Medium complexity indicators
  if (params.requiresReasoning) {
    return TaskComplexity.MEDIUM;
  }

  if (params.inputTokens && params.inputTokens > 500) {
    return TaskComplexity.MEDIUM;
  }

  // Default to low
  return TaskComplexity.LOW;
}
