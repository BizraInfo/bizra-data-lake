// src/copilot_integration/got_engine.ts
// Graph of Thoughts (GoT) Engine - Extracted from BIZRA-copilot
//
// Standing on the Shoulders of Giants: This implementation synthesizes
// thought flow patterns from https://github.com/BizraInfo/BIZRA-copilot.git
// into the BIZRA dual-agentic architecture.

import { EventEmitter } from 'events';

// ============================================================================
// THINKING LEVEL LADDER
// ============================================================================

/**
 * ThinkLevel controls the depth of reasoning applied to a task.
 * 
 * Extracted from BIZRA-copilot's auto-reply/thinking.ts:
 * - off: No extended thinking (fastest, for simple acknowledgments)
 * - minimal: Minimal processing for confirmations
 * - low: Quick checks, simple tasks
 * - medium: Standard problem solving
 * - high: Complex multi-step reasoning
 * - xhigh: Deep reasoning, proof-level rigor (requires advanced models)
 */
export type ThinkLevel = 'off' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh';

/**
 * ReasoningLevel controls visibility of internal reasoning.
 * 
 * - off: No reasoning trace
 * - on: Reasoning present but hidden
 * - stream: Reasoning streamed in real-time
 */
export type ReasoningLevel = 'off' | 'on' | 'stream';

/**
 * VerboseLevel controls output verbosity.
 */
export type VerboseLevel = 'off' | 'on' | 'minimal' | 'medium' | 'extensive';

/**
 * ElevatedLevel controls user prompts for complex tasks.
 * 
 * - off: Never ask
 * - on: Always elevate
 * - ask: Ask user when uncertain
 * - full: Full elevation with explanation
 */
export type ElevatedLevel = 'off' | 'on' | 'ask' | 'full';

// ============================================================================
// SNR TIER CLASSIFICATION (From BIZRA SAPE)
// ============================================================================

/**
 * SNR (Signal-to-Noise Ratio) quality tiers.
 * 
 * Extracted from BIZRA's SAPE engine - maps Ihsān scores to quality tiers.
 */
export enum SnrTier {
  T0 = 0, // Rejected (Ihsān < 0.95)
  T1 = 1, // Baseline (SNR 7.0-7.4)
  T2 = 2, // Acceptable (SNR 7.4-7.8)
  T3 = 3, // Target (SNR 7.8-8.2) ★
  T4 = 4, // Strong (SNR 8.2-8.6)
  T5 = 5, // Expert (SNR 8.6-9.0)
  T6 = 6, // Elite (SNR 9.0+)
}

/** Ihsān minimum threshold (constitutional requirement) */
export const IHSAN_MINIMUM_THRESHOLD = 0.95;

/**
 * Classify Ihsān score to SNR tier.
 * 
 * @param score Ihsān composite score (0.0-1.0)
 * @returns SnrTier classification
 */
export function classifyIhsanToTier(score: number): SnrTier {
  // Constitutional threshold enforcement
  if (score < IHSAN_MINIMUM_THRESHOLD) {
    console.warn(`⚠️ Ihsān score ${score.toFixed(3)} below threshold ${IHSAN_MINIMUM_THRESHOLD} - T0 rejected`);
    return SnrTier.T0;
  }
  
  // Map 0.95-1.0 to SNR range
  const snr = 7.0 + Math.max(0, score - 0.80) * 10.0;
  
  if (snr >= 9.0) return SnrTier.T6;
  if (snr >= 8.6) return SnrTier.T5;
  if (snr >= 8.2) return SnrTier.T4;
  if (snr >= 7.8) return SnrTier.T3;
  if (snr >= 7.4) return SnrTier.T2;
  if (snr >= 7.0) return SnrTier.T1;
  return SnrTier.T0;
}

/**
 * Check if tier meets high-stakes requirements (T4+).
 */
export function meetsHighStakes(tier: SnrTier): boolean {
  return tier >= SnrTier.T4;
}

// ============================================================================
// MODEL ROUTING (Standing on Giants Protocol)
// ============================================================================

/**
 * Model routing slots - defines which models to use for which task types.
 * 
 * Extracted from model-family-genesis-v1-SEALED.yaml
 */
export interface ModelSlot {
  description: string;
  primary: string;
  fallback?: string;
  params?: {
    num_ctx?: number;
    temperature?: number;
  };
}

export const CAPABILITY_SLOTS: Record<string, ModelSlot> = {
  cold_core: {
    description: 'Deterministic reasoning + self-correction + causal trace',
    primary: 'deepseek-r1:8b',
    fallback: 'mistral:latest',
    params: { temperature: 0.6 },
  },
  warm_surface: {
    description: 'Nuance + formatting + user-facing tone control',
    primary: 'mistral:latest',
    fallback: 'qwen2.5:7b',
    params: { temperature: 0.3 },
  },
  primary_reasoning: {
    description: 'Multi-agent orchestration + strategic planning',
    primary: 'bizra-planner:latest',
    fallback: 'agentflow-planner-7b-i1',
    params: { temperature: 0.7 },
  },
  embeddings: {
    description: 'Deterministic embedding for RAG/semantic search',
    primary: 'nomic-embed-text:latest',
  },
  vision: {
    description: 'Vision-capable multimodal inference',
    primary: 'qwen/qwen3-vl-8b',
    fallback: 'qwen/qwen3-vl-4b',
  },
};

/**
 * Select model based on task requirements.
 */
export function selectModel(slot: keyof typeof CAPABILITY_SLOTS, useFallback = false): string {
  const config = CAPABILITY_SLOTS[slot];
  if (!config) {
    console.warn(`Unknown slot ${slot}, defaulting to primary_reasoning`);
    return CAPABILITY_SLOTS.primary_reasoning.primary;
  }
  return useFallback && config.fallback ? config.fallback : config.primary;
}

// ============================================================================
// GRAPH OF THOUGHTS ENGINE
// ============================================================================

/**
 * Thought node in the graph.
 */
export interface ThoughtNode {
  id: string;
  content: string;
  level: ThinkLevel;
  parent?: string;
  children: string[];
  score?: number;
  tier?: SnrTier;
  metadata?: Record<string, unknown>;
}

/**
 * Graph of Thoughts configuration.
 */
export interface GoTConfig {
  thinkLevel: ThinkLevel;
  reasoningLevel: ReasoningLevel;
  verboseLevel: VerboseLevel;
  elevatedLevel: ElevatedLevel;
  maxDepth: number;
  maxBranches: number;
  snrFloor: number; // SNR 7.0 default
  snrTarget: number; // SNR 7.8 default
}

const DEFAULT_GOT_CONFIG: GoTConfig = {
  thinkLevel: 'medium',
  reasoningLevel: 'on',
  verboseLevel: 'off',
  elevatedLevel: 'ask',
  maxDepth: 5,
  maxBranches: 3,
  snrFloor: 7.0,
  snrTarget: 7.8,
};

/**
 * Graph of Thoughts Engine - implements multi-tier thinking with SNR optimization.
 * 
 * Pattern extracted from BIZRA-copilot's thinking architecture.
 */
export class GraphOfThoughtsEngine extends EventEmitter {
  private nodes: Map<string, ThoughtNode> = new Map();
  private rootId: string | null = null;
  private config: GoTConfig;
  private nodeCounter = 0;

  constructor(config: Partial<GoTConfig> = {}) {
    super();
    this.config = { ...DEFAULT_GOT_CONFIG, ...config };
  }

  /**
   * Initialize a new thought graph with a root node.
   */
  initializeGraph(content: string): ThoughtNode {
    this.nodes.clear();
    this.nodeCounter = 0;
    
    const root = this.createNode(content, this.config.thinkLevel);
    this.rootId = root.id;
    
    this.emit('graph:initialized', { root });
    return root;
  }

  /**
   * Create a new thought node.
   */
  createNode(content: string, level: ThinkLevel, parentId?: string): ThoughtNode {
    const id = `thought_${++this.nodeCounter}`;
    const node: ThoughtNode = {
      id,
      content,
      level,
      parent: parentId,
      children: [],
    };

    this.nodes.set(id, node);

    if (parentId) {
      const parent = this.nodes.get(parentId);
      if (parent) {
        parent.children.push(id);
      }
    }

    this.emit('node:created', { node });
    return node;
  }

  /**
   * Branch a thought into multiple alternatives.
   */
  branch(parentId: string, alternatives: string[]): ThoughtNode[] {
    const limited = alternatives.slice(0, this.config.maxBranches);
    
    return limited.map((alt) => {
      const node = this.createNode(alt, this.config.thinkLevel, parentId);
      this.emit('node:branched', { parent: parentId, child: node });
      return node;
    });
  }

  /**
   * Score a node based on SNR criteria.
   */
  scoreNode(nodeId: string, ihsanScore: number): SnrTier {
    const node = this.nodes.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }

    const tier = classifyIhsanToTier(ihsanScore);
    node.score = ihsanScore;
    node.tier = tier;

    this.emit('node:scored', { nodeId, score: ihsanScore, tier });
    return tier;
  }

  /**
   * Prune low-SNR branches.
   */
  prune(): number {
    let pruned = 0;
    
    for (const [id, node] of this.nodes) {
      if (node.tier !== undefined && node.tier < SnrTier.T1) {
        // Remove low-tier nodes (below safe mode)
        this.removeNode(id);
        pruned++;
      }
    }

    this.emit('graph:pruned', { count: pruned });
    return pruned;
  }

  /**
   * Remove a node and its descendants.
   */
  private removeNode(nodeId: string): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;

    // Recursively remove children
    for (const childId of node.children) {
      this.removeNode(childId);
    }

    // Remove from parent's children
    if (node.parent) {
      const parent = this.nodes.get(node.parent);
      if (parent) {
        parent.children = parent.children.filter((id) => id !== nodeId);
      }
    }

    this.nodes.delete(nodeId);
  }

  /**
   * Get the best path through the graph (highest SNR).
   */
  getBestPath(): ThoughtNode[] {
    if (!this.rootId) return [];

    const path: ThoughtNode[] = [];
    let current: ThoughtNode | undefined = this.nodes.get(this.rootId);

    while (current) {
      path.push(current);

      if (current.children.length === 0) break;

      // Select child with highest score
      let best: ThoughtNode | undefined;
      let bestScore = -Infinity;

      for (const childId of current.children) {
        const child = this.nodes.get(childId);
        if (child && (child.score ?? 0) > bestScore) {
          best = child;
          bestScore = child.score ?? 0;
        }
      }

      current = best;
    }

    return path;
  }

  /**
   * Format thinking output based on reasoning level.
   */
  formatOutput(content: string): { think: string; final: string } {
    // Pattern: All reasoning in <think>, only final shown
    if (this.config.reasoningLevel === 'off') {
      return { think: '', final: content };
    }

    const path = this.getBestPath();
    const thinkContent = path.map((n) => `[${n.level}] ${n.content}`).join('\n');
    
    return {
      think: thinkContent,
      final: content,
    };
  }

  /**
   * Get graph statistics.
   */
  getStats(): {
    nodeCount: number;
    maxDepth: number;
    avgScore: number;
    bestTier: SnrTier;
  } {
    let maxDepth = 0;
    let totalScore = 0;
    let scoredCount = 0;
    let bestTier = SnrTier.T0;

    const calculateDepth = (nodeId: string, depth: number): void => {
      maxDepth = Math.max(maxDepth, depth);
      const node = this.nodes.get(nodeId);
      if (!node) return;
      
      if (node.score !== undefined) {
        totalScore += node.score;
        scoredCount++;
      }
      if (node.tier !== undefined && node.tier > bestTier) {
        bestTier = node.tier;
      }

      for (const childId of node.children) {
        calculateDepth(childId, depth + 1);
      }
    };

    if (this.rootId) {
      calculateDepth(this.rootId, 1);
    }

    return {
      nodeCount: this.nodes.size,
      maxDepth,
      avgScore: scoredCount > 0 ? totalScore / scoredCount : 0,
      bestTier,
    };
  }
}

// ============================================================================
// SKILL INJECTION PATTERN
// ============================================================================

/**
 * Skill entry structure.
 */
export interface SkillEntry {
  name: string;
  description: string;
  location: string;
  enabled: boolean;
}

/**
 * Build skills prompt section.
 * 
 * Extracted from BIZRA-copilot's buildSkillsSection pattern.
 */
export function buildSkillsSection(params: {
  skills: SkillEntry[];
  isMinimal: boolean;
  readToolName?: string;
}): string[] {
  if (params.isMinimal) return [];
  
  const enabledSkills = params.skills.filter((s) => s.enabled);
  if (enabledSkills.length === 0) return [];

  const readTool = params.readToolName ?? 'read_file';
  const skillsXml = enabledSkills
    .map((s) => [
      '  <skill>',
      `    <name>${s.name}</name>`,
      `    <description>${s.description}</description>`,
      `    <location>${s.location}</location>`,
      '  </skill>',
    ].join('\n'))
    .join('\n');

  return [
    '## Skills (mandatory)',
    'Before replying: scan <available_skills> <description> entries.',
    `- If exactly one skill clearly applies: read its SKILL.md at <location> with \`${readTool}\`, then follow it.`,
    '- If multiple could apply: choose the most specific one, then read/follow it.',
    '- If none clearly apply: do not read any SKILL.md.',
    'Constraints: never read more than one skill up front; only read after selecting.',
    '',
    '<available_skills>',
    skillsXml,
    '</available_skills>',
    '',
  ];
}

// ============================================================================
// REASONING FORMAT HINT
// ============================================================================

/**
 * Build reasoning tag hint for hidden chain-of-thought.
 * 
 * Pattern: All internal reasoning inside <think>, only <final> shown to user.
 */
export function buildReasoningHint(enabled: boolean): string | undefined {
  if (!enabled) return undefined;
  
  return [
    'ALL internal reasoning MUST be inside <think>...</think>.',
    'Do not output any analysis outside <think>.',
    'Format: <think>...</think> then <final>...</final>',
    'Only text inside <final> is shown to user.',
  ].join(' ');
}

// ============================================================================
// RUNTIME INFO LINE
// ============================================================================

/**
 * Build runtime info line for system prompt.
 * 
 * Pattern: agent= | host= | os= | model= | channel= | thinking=
 */
export function buildRuntimeLine(params: {
  agentId: string;
  host: string;
  os: string;
  model: string;
  channel?: string;
  thinkLevel: ThinkLevel;
  capabilities?: string[];
}): string {
  const parts = [
    `agent=${params.agentId}`,
    `host=${params.host}`,
    `os=${params.os}`,
    `model=${params.model}`,
  ];

  if (params.channel) {
    parts.push(`channel=${params.channel}`);
  }
  
  if (params.capabilities && params.capabilities.length > 0) {
    parts.push(`capabilities=${params.capabilities.join(',')}`);
  }

  parts.push(`thinking=${params.thinkLevel}`);

  return parts.join(' | ');
}

// ============================================================================
// EXPORTS
// ============================================================================

export default GraphOfThoughtsEngine;
