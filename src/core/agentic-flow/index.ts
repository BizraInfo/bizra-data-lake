/**
 * BIZRA Agentic-Flow — Barrel Exports
 *
 * The unified orchestration layer for the 12-agent Living Organism.
 * Implements ADR-001: deep integration as specialized extension,
 * not parallel implementation.
 *
 * Architecture:
 *   AgenticFlowAdapter (facade)
 *   ├── ReflexCache       (Helix 1 — System-1 O(1) lookup)
 *   ├── AgentRouter        (HHMM macro/micro state routing)
 *   ├── SONAManager        (5 learning modes × Triple Helix)
 *   └── MemoryCoordinator  (AgentDB cross-agent sharing)
 */

// Core Types
export {
  PATAgent,
  SATAgent,
  type AgentId,
  type AgentDescriptor,
  AGENT_MANIFEST,
  FROZEN_AGENTS,
  Helix,
  type HelixTiming,
  HELIX_TIMINGS,
  SONAMode,
  type SONAConfig,
  SONA_CONFIGS,
  HHMMMacroState,
  type HHMMTransition,
  type Mission,
  type ActionReceipt,
  CONSTITUTIONAL,
  type MemoryEntry,
  type EvidenceLink,
} from './types';

// Reflex Cache (Flash Attention / Helix 1)
export {
  ReflexCache,
  selectHelix,
  type Reflex,
  type ReflexCacheConfig,
} from './reflex-cache';

// SONA Learning Modes
export {
  SONAManager,
  type SONASnapshot,
  type SONATransition,
  type SONAListener,
} from './sona';

// HHMM Agent Router
export {
  AgentRouter,
  type RouteResult,
} from './agent-router';

// Memory Coordinator (AgentDB bridge)
export {
  MemoryCoordinator,
  type MemoryConfig,
  type SearchResult,
} from './memory-coordinator';

// Adapter (unified facade)
export {
  AgenticFlowAdapter,
  createAgenticFlowAdapter,
  type AdapterConfig,
  type AdapterStatus,
  type MissionResult,
  type ScoringDelegate,
  LocalScoringDelegate,
} from './adapter';

// ReasoningBank (Helix 3 learning)
export {
  ReasoningBank,
  type ReasoningBankConfig,
  type Trajectory,
  type TrajectoryStep,
  type TrajectoryOutcome,
  type Pattern,
  type PatternType,
  type Verdict,
  type VerdictLevel,
  type DistillationResult,
  type SeedPattern,
  type SeedPatternMetadata,
  type ReasoningBankStats,
  type DomainStats,
} from './reasoning-bank';
