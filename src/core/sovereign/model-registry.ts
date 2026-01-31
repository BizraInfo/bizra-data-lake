/**
 * Model Registry - Track and manage accepted models
 *
 * The registry maintains all models that have passed the Constitution
 * Challenge and received valid CapabilityCards.
 */

import {
  CapabilityCard,
  ModelTier,
  TaskType,
  isCapabilityCardValid,
  getCardFingerprint,
  getCardRemainingDays,
} from './capability-card';

/**
 * Registered model entry
 */
export interface RegisteredModel {
  /** Model ID */
  id: string;

  /** Model name */
  name: string;

  /** Path to model file (GGUF) */
  path: string;

  /** File size in bytes */
  sizeBytes: number;

  /** SHA-256 hash of model file */
  fileHash: string;

  /** Capability card */
  card: CapabilityCard;

  /** Registration timestamp */
  registeredAt: Date;

  /** Last used timestamp */
  lastUsedAt: Date | null;

  /** Usage count */
  usageCount: number;

  /** Is currently loaded */
  isLoaded: boolean;
}

/**
 * Model selection criteria
 */
export interface ModelSelectionCriteria {
  /** Required task type */
  taskType?: TaskType;

  /** Preferred tier (will fall back if unavailable) */
  preferredTier?: ModelTier;

  /** Minimum Ihsān score */
  minIhsanScore?: number;

  /** Minimum SNR score */
  minSnrScore?: number;

  /** Maximum latency in ms */
  maxLatencyMs?: number;

  /** Minimum context length */
  minContext?: number;

  /** Exclude specific model IDs */
  excludeModels?: string[];
}

/**
 * Model Registry
 */
export class ModelRegistry {
  private models: Map<string, RegisteredModel> = new Map();
  private tierIndex: Map<ModelTier, Set<string>> = new Map([
    [ModelTier.EDGE, new Set()],
    [ModelTier.LOCAL, new Set()],
    [ModelTier.POOL, new Set()],
  ]);
  private taskIndex: Map<TaskType, Set<string>> = new Map();

  /**
   * Register a new model with its capability card
   */
  register(params: {
    id: string;
    name: string;
    path: string;
    sizeBytes: number;
    fileHash: string;
    card: CapabilityCard;
  }): void {
    // Validate the card
    const validation = isCapabilityCardValid(params.card);
    if (!validation.valid) {
      throw new Error(`Invalid capability card: ${validation.reason}`);
    }

    const model: RegisteredModel = {
      id: params.id,
      name: params.name,
      path: params.path,
      sizeBytes: params.sizeBytes,
      fileHash: params.fileHash,
      card: params.card,
      registeredAt: new Date(),
      lastUsedAt: null,
      usageCount: 0,
      isLoaded: false,
    };

    // Add to main registry
    this.models.set(params.id, model);

    // Update tier index
    this.tierIndex.get(params.card.tier)?.add(params.id);

    // Update task index
    for (const task of params.card.capabilities.tasksSupported) {
      if (!this.taskIndex.has(task)) {
        this.taskIndex.set(task, new Set());
      }
      this.taskIndex.get(task)?.add(params.id);
    }
  }

  /**
   * Get a model by ID
   */
  get(id: string): RegisteredModel | undefined {
    return this.models.get(id);
  }

  /**
   * Check if a model is registered
   */
  has(id: string): boolean {
    return this.models.has(id);
  }

  /**
   * Revoke a model's registration
   */
  revoke(id: string, reason: string): boolean {
    const model = this.models.get(id);
    if (!model) {
      return false;
    }

    // Mark card as revoked
    model.card.revoked = true;

    // Remove from indexes
    this.tierIndex.get(model.card.tier)?.delete(id);
    for (const task of model.card.capabilities.tasksSupported) {
      this.taskIndex.get(task)?.delete(id);
    }

    console.log(`Model ${id} revoked: ${reason}`);
    return true;
  }

  /**
   * List all models by tier
   */
  listByTier(tier: ModelTier): RegisteredModel[] {
    const ids = this.tierIndex.get(tier) ?? new Set();
    return Array.from(ids)
      .map((id) => this.models.get(id))
      .filter((m): m is RegisteredModel => m !== undefined);
  }

  /**
   * List all models that support a task type
   */
  listByTask(task: TaskType): RegisteredModel[] {
    const ids = this.taskIndex.get(task) ?? new Set();
    return Array.from(ids)
      .map((id) => this.models.get(id))
      .filter((m): m is RegisteredModel => m !== undefined);
  }

  /**
   * Select the best model for given criteria
   */
  selectBest(criteria: ModelSelectionCriteria): RegisteredModel | null {
    let candidates: RegisteredModel[];

    // Start with task-filtered or all models
    if (criteria.taskType) {
      candidates = this.listByTask(criteria.taskType);
    } else {
      candidates = Array.from(this.models.values());
    }

    // Filter by validity
    candidates = candidates.filter((m) => {
      const validation = isCapabilityCardValid(m.card);
      return validation.valid;
    });

    // Filter by exclusions
    if (criteria.excludeModels) {
      const excluded = new Set(criteria.excludeModels);
      candidates = candidates.filter((m) => !excluded.has(m.id));
    }

    // Filter by minimum scores
    if (criteria.minIhsanScore !== undefined) {
      candidates = candidates.filter(
        (m) => m.card.capabilities.ihsanScore >= criteria.minIhsanScore!
      );
    }
    if (criteria.minSnrScore !== undefined) {
      candidates = candidates.filter(
        (m) => m.card.capabilities.snrScore >= criteria.minSnrScore!
      );
    }

    // Filter by latency
    if (criteria.maxLatencyMs !== undefined) {
      candidates = candidates.filter(
        (m) => m.card.capabilities.latencyMs <= criteria.maxLatencyMs!
      );
    }

    // Filter by context
    if (criteria.minContext !== undefined) {
      candidates = candidates.filter(
        (m) => m.card.capabilities.maxContext >= criteria.minContext!
      );
    }

    // Prefer specific tier
    if (criteria.preferredTier) {
      const tierCandidates = candidates.filter(
        (m) => m.card.tier === criteria.preferredTier
      );
      if (tierCandidates.length > 0) {
        candidates = tierCandidates;
      }
      // Otherwise fall through to any tier
    }

    if (candidates.length === 0) {
      return null;
    }

    // Sort by Ihsān score (higher is better)
    candidates.sort(
      (a, b) => b.card.capabilities.ihsanScore - a.card.capabilities.ihsanScore
    );

    return candidates[0];
  }

  /**
   * Get the default model (highest scoring EDGE model)
   */
  getDefault(): RegisteredModel | null {
    return this.selectBest({ preferredTier: ModelTier.EDGE });
  }

  /**
   * Record model usage
   */
  recordUsage(id: string): void {
    const model = this.models.get(id);
    if (model) {
      model.lastUsedAt = new Date();
      model.usageCount++;
    }
  }

  /**
   * Get registry statistics
   */
  getStats(): RegistryStats {
    const models = Array.from(this.models.values());
    const validModels = models.filter((m) => isCapabilityCardValid(m.card).valid);

    const byTier = {
      [ModelTier.EDGE]: this.listByTier(ModelTier.EDGE).length,
      [ModelTier.LOCAL]: this.listByTier(ModelTier.LOCAL).length,
      [ModelTier.POOL]: this.listByTier(ModelTier.POOL).length,
    };

    const expiringWithin30Days = validModels.filter(
      (m) => getCardRemainingDays(m.card) <= 30
    ).length;

    return {
      totalModels: models.length,
      validModels: validModels.length,
      revokedModels: models.filter((m) => m.card.revoked).length,
      expiredModels: models.length - validModels.length - models.filter((m) => m.card.revoked).length,
      byTier,
      expiringWithin30Days,
      totalUsageCount: models.reduce((sum, m) => sum + m.usageCount, 0),
    };
  }

  /**
   * Clean up expired and revoked models
   */
  cleanup(): number {
    let removed = 0;

    for (const [id, model] of this.models) {
      const validation = isCapabilityCardValid(model.card);
      if (!validation.valid) {
        this.models.delete(id);
        this.tierIndex.get(model.card.tier)?.delete(id);
        for (const task of model.card.capabilities.tasksSupported) {
          this.taskIndex.get(task)?.delete(id);
        }
        removed++;
      }
    }

    return removed;
  }

  /**
   * Export registry to JSON
   */
  toJSON(): string {
    const models = Array.from(this.models.values()).map((m) => ({
      ...m,
      registeredAt: m.registeredAt.toISOString(),
      lastUsedAt: m.lastUsedAt?.toISOString() ?? null,
    }));

    return JSON.stringify({ models, exportedAt: new Date().toISOString() }, null, 2);
  }

  /**
   * Import registry from JSON
   */
  fromJSON(json: string): void {
    const data = JSON.parse(json);

    for (const modelData of data.models) {
      const model: RegisteredModel = {
        ...modelData,
        registeredAt: new Date(modelData.registeredAt),
        lastUsedAt: modelData.lastUsedAt ? new Date(modelData.lastUsedAt) : null,
      };

      // Validate card before importing
      const validation = isCapabilityCardValid(model.card);
      if (validation.valid) {
        this.models.set(model.id, model);
        this.tierIndex.get(model.card.tier)?.add(model.id);
        for (const task of model.card.capabilities.tasksSupported) {
          if (!this.taskIndex.has(task)) {
            this.taskIndex.set(task, new Set());
          }
          this.taskIndex.get(task)?.add(model.id);
        }
      }
    }
  }
}

export interface RegistryStats {
  totalModels: number;
  validModels: number;
  revokedModels: number;
  expiredModels: number;
  byTier: Record<ModelTier, number>;
  expiringWithin30Days: number;
  totalUsageCount: number;
}
