/**
 * BIZRA Agentic-Flow — Core Types
 *
 * The 12-agent Living Organism type system.
 * PAT-7 (Personal Agent Team) + SAT-5 (System Agent Team) × 1B each.
 *
 * Standing on Giants:
 *   Prophet Muhammad ﷺ (Ihsān, Hadith Jibril) · Shannon (SNR, 1948) ·
 *   Kahneman (dual-process System-1/2, 2011) · Boyd (OODA, 1976) ·
 *   Besta (Graph-of-Thoughts, 2024) · Nakamoto (evidence chain, 2008)
 *
 * References: Enforceable Spine §1 (Living Organism), §2 (Triple Helix)
 */

// ────────────────────────────────────────────────────────────
// Agent Identity
// ────────────────────────────────────────────────────────────

/** PAT-7 agents — user's sovereign brain */
export enum PATAgent {
  PLANNER     = 'P1',
  RESEARCHER  = 'P2',
  CODER       = 'P3',
  EVALUATOR   = 'P4',
  ETHICIST    = 'P5',
  PUBLISHER   = 'P6',
  DEMA        = 'P7',
}

/** SAT-5 agents — forest's immune system */
export enum SATAgent {
  SENTINEL    = 'S1',
  ORACLE      = 'S2',
  LEDGER      = 'S3',
  CONDUCTOR   = 'S4',
  AMBASSADOR  = 'S5',
}

/** Union of all 12 agent IDs */
export type AgentId = PATAgent | SATAgent;

/** Frozen agents that never learn/evolve (§1 P5/S2 Exception) */
export const FROZEN_AGENTS: ReadonlySet<AgentId> = new Set([
  PATAgent.ETHICIST,
  SATAgent.ORACLE,
]);

export interface AgentDescriptor {
  readonly id: AgentId;
  readonly name: string;
  readonly sizeB: number;
  readonly frozen: boolean;
  readonly team: 'PAT' | 'SAT';
}

/** Complete 12-agent manifest */
export const AGENT_MANIFEST: readonly AgentDescriptor[] = [
  { id: PATAgent.PLANNER,    name: 'Planner',    sizeB: 1, frozen: false, team: 'PAT' },
  { id: PATAgent.RESEARCHER, name: 'Researcher', sizeB: 1, frozen: false, team: 'PAT' },
  { id: PATAgent.CODER,      name: 'Coder',      sizeB: 1, frozen: false, team: 'PAT' },
  { id: PATAgent.EVALUATOR,  name: 'Evaluator',  sizeB: 1, frozen: false, team: 'PAT' },
  { id: PATAgent.ETHICIST,   name: 'Ethicist',   sizeB: 1, frozen: true,  team: 'PAT' },
  { id: PATAgent.PUBLISHER,  name: 'Publisher',   sizeB: 1, frozen: false, team: 'PAT' },
  { id: PATAgent.DEMA,       name: 'DEMA',        sizeB: 1, frozen: false, team: 'PAT' },
  { id: SATAgent.SENTINEL,   name: 'Sentinel',   sizeB: 1, frozen: false, team: 'SAT' },
  { id: SATAgent.ORACLE,     name: 'Oracle',     sizeB: 1, frozen: true,  team: 'SAT' },
  { id: SATAgent.LEDGER,     name: 'Ledger',     sizeB: 1, frozen: false, team: 'SAT' },
  { id: SATAgent.CONDUCTOR,  name: 'Conductor',  sizeB: 1, frozen: false, team: 'SAT' },
  { id: SATAgent.AMBASSADOR, name: 'Ambassador', sizeB: 1, frozen: false, team: 'SAT' },
] as const;

// ────────────────────────────────────────────────────────────
// Triple Helix (§2)
// ────────────────────────────────────────────────────────────

/** Three concurrent processing cycles */
export enum Helix {
  /** System-1 reactive: 50ms reflex cache (Kahneman, 2011) */
  REACTIVE      = 'helix-1',
  /** System-2 deliberative: 800-2000ms PBFT + FATE gates */
  DELIBERATIVE  = 'helix-2',
  /** System-3 evolutionary: 60s heartbeat + SDPO */
  EVOLUTIONARY  = 'helix-3',
}

export interface HelixTiming {
  readonly helix: Helix;
  readonly targetMs: number;
  readonly maxMs: number;
}

export const HELIX_TIMINGS: readonly HelixTiming[] = [
  { helix: Helix.REACTIVE,     targetMs: 50,    maxMs: 100 },
  { helix: Helix.DELIBERATIVE, targetMs: 800,   maxMs: 2000 },
  { helix: Helix.EVOLUTIONARY, targetMs: 60000, maxMs: 120000 },
] as const;

// ────────────────────────────────────────────────────────────
// SONA Learning Modes
// ────────────────────────────────────────────────────────────

export enum SONAMode {
  REAL_TIME = 'real-time',
  BALANCED  = 'balanced',
  RESEARCH  = 'research',
  EDGE      = 'edge',
  BATCH     = 'batch',
}

export interface SONAConfig {
  readonly mode: SONAMode;
  /** Target adaptation latency in ms */
  readonly adaptationMs: number;
  /** Maps to which helix is primary */
  readonly primaryHelix: Helix;
  /** Max agents to activate concurrently */
  readonly maxConcurrentAgents: number;
}

export const SONA_CONFIGS: ReadonlyMap<SONAMode, SONAConfig> = new Map([
  [SONAMode.REAL_TIME, { mode: SONAMode.REAL_TIME, adaptationMs: 0.05, primaryHelix: Helix.REACTIVE,     maxConcurrentAgents: 2 }],
  [SONAMode.BALANCED,  { mode: SONAMode.BALANCED,  adaptationMs: 50,   primaryHelix: Helix.DELIBERATIVE, maxConcurrentAgents: 4 }],
  [SONAMode.RESEARCH,  { mode: SONAMode.RESEARCH,  adaptationMs: 500,  primaryHelix: Helix.DELIBERATIVE, maxConcurrentAgents: 7 }],
  [SONAMode.EDGE,      { mode: SONAMode.EDGE,      adaptationMs: 10,   primaryHelix: Helix.REACTIVE,     maxConcurrentAgents: 2 }],
  [SONAMode.BATCH,     { mode: SONAMode.BATCH,     adaptationMs: 1000, primaryHelix: Helix.EVOLUTIONARY, maxConcurrentAgents: 12 }],
]);

// ────────────────────────────────────────────────────────────
// HHMM State Model (Hierarchical Hidden Markov Model)
// ────────────────────────────────────────────────────────────

/** Macro-states in the HHMM (top-level routing) */
export enum HHMMMacroState {
  IDLE          = 'idle',
  PLANNING      = 'planning',
  RESEARCHING   = 'researching',
  CODING        = 'coding',
  EVALUATING    = 'evaluating',
  GATE_CHECK    = 'gate-check',
  PUBLISHING    = 'publishing',
  FEDERATING    = 'federating',
}

/** Transition probability entry */
export interface HHMMTransition {
  readonly from: HHMMMacroState;
  readonly to: HHMMMacroState;
  readonly probability: number;
  readonly agents: readonly AgentId[];
}

// ────────────────────────────────────────────────────────────
// Mission & Receipt
// ────────────────────────────────────────────────────────────

export interface Mission {
  readonly id: string;
  readonly description: string;
  readonly source: AgentId;
  readonly assignedAgents: readonly AgentId[];
  readonly helix: Helix;
  readonly sonaMode: SONAMode;
  readonly createdAt: number;
  readonly priority: number;
}

export interface ActionReceipt {
  readonly missionId: string;
  readonly description: string;
  readonly ihsanScore: number;
  readonly snrScore: number;
  readonly agentIds: readonly AgentId[];
  readonly helix: Helix;
  readonly timestamp: number;
  readonly receiptHash: string;
  readonly prevHash: string;
  readonly elapsedMs: number;
  readonly seedMinted: boolean;
}

// ────────────────────────────────────────────────────────────
// Constitutional Thresholds (§4 — immutable invariants)
// ────────────────────────────────────────────────────────────

export const CONSTITUTIONAL = {
  IHSAN_PRODUCTION:  0.95,
  IHSAN_MINIMUM:     0.85,
  IHSAN_STRICT:      0.99,
  SNR_MUSEUM:        0.85,
  SNR_T1_HIGH:       0.95,
  SNR_T0_ELITE:      0.98,
  GINI_CEILING:      0.35,
  ZAKAT_RATE:        0.025,
  RIBA_RATE:         0.0,
  HEARTBEAT_MS:      60_000,
  PRECIPITATION_REPEATS: 3,
  PRECIPITATION_IHSAN:   0.90,
} as const;

// ────────────────────────────────────────────────────────────
// Memory & Evidence
// ────────────────────────────────────────────────────────────

export interface MemoryEntry {
  readonly id: string;
  readonly agentId: AgentId;
  readonly content: string;
  readonly embedding?: Float32Array;
  readonly timestamp: number;
  readonly shared: boolean;
}

export interface EvidenceLink {
  readonly receiptHash: string;
  readonly prevHash: string;
  readonly missionId: string;
  readonly timestamp: number;
}
