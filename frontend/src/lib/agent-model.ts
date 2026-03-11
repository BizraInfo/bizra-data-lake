/**
 * BIZRA Agent Ontology — Canonical Contract
 *
 * Separates the PROTOCOL layer (mirrors core/pat/agent.py) from the
 * PRESENTATION layer (UI callsigns like ATLAS, ORACLE, etc.).
 *
 * Rule: All protocol logic references CanonicalAgentType.
 *       All rendering logic references the persona layer via agentPersona().
 *
 * Canon source: core/pat/agent.py
 *   PAT (Personal Agentic Team): 7 per user, lifetime-bonded
 *   SAT (System  Agentic Team): 5 per onboarding, system-owned
 *   Total: 12 agents per new user  (7 PAT 58.3% + 5 SAT 41.7%)
 */

// ═══ Canonical Agent Types (mirrors core/pat/agent.py AgentType enum) ═══

export const CANONICAL_AGENT_TYPES = [
  'WORKER',
  'RESEARCHER',
  'GUARDIAN',
  'SYNTHESIZER',
  'VALIDATOR',
  'COORDINATOR',
  'EXECUTOR',
] as const;

export type CanonicalAgentType = (typeof CANONICAL_AGENT_TYPES)[number];

/** Canonical slot ID — positional index into PAT. */
export type AgentSlot = 'P1' | 'P2' | 'P3' | 'P4' | 'P5' | 'P6' | 'P7';

export const AGENT_SLOTS: readonly AgentSlot[] = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'];

// ═══ Canonical Agent Definition (protocol layer) ═══

export interface CanonicalAgent {
  readonly slot: AgentSlot;
  readonly type: CanonicalAgentType;
  readonly domain: string;
}

/**
 * PAT canonical definitions — the 7 agent slots with their backend types.
 * Order matches core/pat/agent.py docstring.
 */
export const PAT_CANONICAL: Record<AgentSlot, CanonicalAgent> = {
  P1: { slot: 'P1', type: 'WORKER',       domain: 'General task execution' },
  P2: { slot: 'P2', type: 'RESEARCHER',   domain: 'Information gathering and synthesis' },
  P3: { slot: 'P3', type: 'GUARDIAN',      domain: 'Security monitoring and validation' },
  P4: { slot: 'P4', type: 'SYNTHESIZER',   domain: 'Data integration and insight generation' },
  P5: { slot: 'P5', type: 'VALIDATOR',     domain: 'Proof verification and quality assurance' },
  P6: { slot: 'P6', type: 'COORDINATOR',   domain: 'Multi-agent orchestration' },
  P7: { slot: 'P7', type: 'EXECUTOR',      domain: 'External system interaction' },
};

// ═══ SAT (System Agentic Team) ═══

export const SAT_COUNT_PER_USER = 5;

/** SAT canonical roles — system-owned agents for ecosystem sustainability. */
export const SAT_CANONICAL = [
  'Sentinel',
  'Oracle',
  'Ledger',
  'Conductor',
  'Ambassador',
] as const;

export type SatRole = (typeof SAT_CANONICAL)[number];

// ═══ Presentation Layer (UI Personas) ═══

export type UICallsign = 'ATLAS' | 'ORACLE' | 'FORGE' | 'JUDGE' | 'CROWN' | 'HERALD' | 'NEXUS';

export interface AgentPersona {
  readonly callsign: UICallsign;
  readonly displayName: string;
  readonly domainLabel: string;
  readonly icon: string;
  readonly colorKey: string;
}

/**
 * Mapping from canonical slot → UI persona.
 * This table is the ONLY place callsigns are defined.
 * Change it here → changes everywhere in the UI.
 */
const PERSONA_MAP: Record<AgentSlot, AgentPersona> = {
  P1: { callsign: 'ATLAS',  displayName: 'Planner',    domainLabel: 'Strategy',     icon: '\u25C8', colorKey: 'sapphire' },
  P2: { callsign: 'ORACLE', displayName: 'Researcher',  domainLabel: 'Knowledge',    icon: '\u25C9', colorKey: 'cyan' },
  P3: { callsign: 'FORGE',  displayName: 'Coder',       domainLabel: 'Build',        icon: '\u2B21', colorKey: 'emerald' },
  P4: { callsign: 'JUDGE',  displayName: 'Evaluator',   domainLabel: 'Quality',      icon: '\u25C7', colorKey: 'amber' },
  P5: { callsign: 'CROWN',  displayName: 'Ethicist',    domainLabel: 'Ethics',       icon: '\u2657', colorKey: 'ruby' },
  P6: { callsign: 'HERALD', displayName: 'Publisher',   domainLabel: 'Deliver',      icon: '\u25C6', colorKey: 'flame' },
  P7: { callsign: 'NEXUS',  displayName: 'Integrator',  domainLabel: 'Orchestrate',  icon: '\u2726', colorKey: 'amethyst' },
};

/** Look up persona for a canonical slot. */
export function agentPersona(slot: AgentSlot): AgentPersona {
  return PERSONA_MAP[slot];
}

/** Reverse lookup: callsign → slot. */
export function slotForCallsign(callsign: UICallsign): AgentSlot | undefined {
  for (const [slot, persona] of Object.entries(PERSONA_MAP) as [AgentSlot, AgentPersona][]) {
    if (persona.callsign === callsign) return slot;
  }
  return undefined;
}

/** All callsigns in slot order. */
export function allCallsigns(): readonly UICallsign[] {
  return AGENT_SLOTS.map(s => PERSONA_MAP[s].callsign);
}
