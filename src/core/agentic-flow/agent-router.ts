/**
 * BIZRA Agentic-Flow — HHMM Agent Router
 *
 * Hierarchical Hidden Markov Model for selecting which 2-4 of 12
 * agents to activate per mission.
 *
 * Macro-states: top-level task categories (planning, coding, etc.)
 * Micro-states: per-agent activation within the selected macro-state
 *
 * The boundary model is ABSOLUTE (§1):
 *   Human → DEMA (P7) → PAT-7 → Pool → SAT-5
 *
 * Standing on Giants:
 *   Fine (HHMM, 1998) · Boyd (OODA, 1976) · Kahneman (System-1/2, 2011) ·
 *   Porat/TeleScript (agent identity, 1994)
 *
 * Reference: Spine §1 (Living Organism), §2 (Triple Helix)
 */

import {
  type AgentId,
  PATAgent,
  SATAgent,
  HHMMMacroState,
  type HHMMTransition,
} from './types';

/** Route selection result */
export interface RouteResult {
  readonly macroState: HHMMMacroState;
  readonly selectedAgents: readonly AgentId[];
  readonly confidence: number;
  readonly reason: string;
}

/** Keyword → macro-state classification table */
const KEYWORD_CLASSIFICATION: ReadonlyMap<string, HHMMMacroState> = new Map([
  // Planning
  ['plan', HHMMMacroState.PLANNING],
  ['strategy', HHMMMacroState.PLANNING],
  ['decompose', HHMMMacroState.PLANNING],
  ['roadmap', HHMMMacroState.PLANNING],
  ['mission', HHMMMacroState.PLANNING],
  ['design', HHMMMacroState.PLANNING],
  ['architect', HHMMMacroState.PLANNING],
  // Research
  ['research', HHMMMacroState.RESEARCHING],
  ['find', HHMMMacroState.RESEARCHING],
  ['search', HHMMMacroState.RESEARCHING],
  ['analyze', HHMMMacroState.RESEARCHING],
  ['investigate', HHMMMacroState.RESEARCHING],
  ['cite', HHMMMacroState.RESEARCHING],
  ['evidence', HHMMMacroState.RESEARCHING],
  // Coding
  ['code', HHMMMacroState.CODING],
  ['implement', HHMMMacroState.CODING],
  ['fix', HHMMMacroState.CODING],
  ['refactor', HHMMMacroState.CODING],
  ['debug', HHMMMacroState.CODING],
  ['test', HHMMMacroState.CODING],
  ['build', HHMMMacroState.CODING],
  // Evaluation
  ['evaluate', HHMMMacroState.EVALUATING],
  ['score', HHMMMacroState.EVALUATING],
  ['quality', HHMMMacroState.EVALUATING],
  ['review', HHMMMacroState.EVALUATING],
  ['assess', HHMMMacroState.EVALUATING],
  ['benchmark', HHMMMacroState.EVALUATING],
  // Gate check
  ['verify', HHMMMacroState.GATE_CHECK],
  ['constitutional', HHMMMacroState.GATE_CHECK],
  ['ihsan', HHMMMacroState.GATE_CHECK],
  ['gate', HHMMMacroState.GATE_CHECK],
  ['fate', HHMMMacroState.GATE_CHECK],
  ['ethics', HHMMMacroState.GATE_CHECK],
  // Publishing
  ['publish', HHMMMacroState.PUBLISHING],
  ['format', HHMMMacroState.PUBLISHING],
  ['output', HHMMMacroState.PUBLISHING],
  ['present', HHMMMacroState.PUBLISHING],
  ['report', HHMMMacroState.PUBLISHING],
  // Federation
  ['federate', HHMMMacroState.FEDERATING],
  ['gossip', HHMMMacroState.FEDERATING],
  ['sync', HHMMMacroState.FEDERATING],
  ['propagate', HHMMMacroState.FEDERATING],
  ['peer', HHMMMacroState.FEDERATING],
]);

/**
 * Agent activation patterns per macro-state.
 * Each macro-state activates a primary set of PAT agents
 * plus the mandatory SAT guardians.
 */
const MACRO_STATE_AGENTS: ReadonlyMap<HHMMMacroState, readonly AgentId[]> = new Map([
  [HHMMMacroState.IDLE,        [PATAgent.DEMA]],
  [HHMMMacroState.PLANNING,    [PATAgent.PLANNER, PATAgent.RESEARCHER, PATAgent.EVALUATOR]],
  [HHMMMacroState.RESEARCHING, [PATAgent.RESEARCHER, PATAgent.PLANNER]],
  [HHMMMacroState.CODING,      [PATAgent.CODER, PATAgent.EVALUATOR, PATAgent.PLANNER]],
  [HHMMMacroState.EVALUATING,  [PATAgent.EVALUATOR, PATAgent.CODER, PATAgent.RESEARCHER]],
  [HHMMMacroState.GATE_CHECK,  [PATAgent.ETHICIST, PATAgent.EVALUATOR]],
  [HHMMMacroState.PUBLISHING,  [PATAgent.PUBLISHER, PATAgent.EVALUATOR]],
  [HHMMMacroState.FEDERATING,  [PATAgent.DEMA]],
]);

/**
 * SAT agents that always participate for a macro-state.
 * S2 Oracle guards every gate check; S1 Sentinel monitors everything.
 */
const MACRO_STATE_SAT: ReadonlyMap<HHMMMacroState, readonly AgentId[]> = new Map([
  [HHMMMacroState.IDLE,        [SATAgent.SENTINEL]],
  [HHMMMacroState.PLANNING,    [SATAgent.SENTINEL, SATAgent.CONDUCTOR]],
  [HHMMMacroState.RESEARCHING, [SATAgent.SENTINEL]],
  [HHMMMacroState.CODING,      [SATAgent.SENTINEL, SATAgent.CONDUCTOR]],
  [HHMMMacroState.EVALUATING,  [SATAgent.SENTINEL, SATAgent.ORACLE]],
  [HHMMMacroState.GATE_CHECK,  [SATAgent.SENTINEL, SATAgent.ORACLE, SATAgent.LEDGER]],
  [HHMMMacroState.PUBLISHING,  [SATAgent.SENTINEL, SATAgent.LEDGER]],
  [HHMMMacroState.FEDERATING,  [SATAgent.SENTINEL, SATAgent.AMBASSADOR, SATAgent.CONDUCTOR]],
]);

/**
 * HHMM transition matrix (macro-state → macro-state probabilities).
 * Used for predictive pre-activation of likely next agents.
 */
const TRANSITION_MATRIX: readonly HHMMTransition[] = [
  { from: HHMMMacroState.IDLE,        to: HHMMMacroState.PLANNING,    probability: 0.45, agents: [PATAgent.PLANNER] },
  { from: HHMMMacroState.IDLE,        to: HHMMMacroState.RESEARCHING, probability: 0.30, agents: [PATAgent.RESEARCHER] },
  { from: HHMMMacroState.IDLE,        to: HHMMMacroState.CODING,      probability: 0.20, agents: [PATAgent.CODER] },
  { from: HHMMMacroState.IDLE,        to: HHMMMacroState.IDLE,        probability: 0.05, agents: [PATAgent.DEMA] },
  { from: HHMMMacroState.PLANNING,    to: HHMMMacroState.CODING,      probability: 0.40, agents: [PATAgent.CODER] },
  { from: HHMMMacroState.PLANNING,    to: HHMMMacroState.RESEARCHING, probability: 0.35, agents: [PATAgent.RESEARCHER] },
  { from: HHMMMacroState.PLANNING,    to: HHMMMacroState.EVALUATING,  probability: 0.15, agents: [PATAgent.EVALUATOR] },
  { from: HHMMMacroState.PLANNING,    to: HHMMMacroState.GATE_CHECK,  probability: 0.10, agents: [PATAgent.ETHICIST] },
  { from: HHMMMacroState.RESEARCHING, to: HHMMMacroState.PLANNING,    probability: 0.30, agents: [PATAgent.PLANNER] },
  { from: HHMMMacroState.RESEARCHING, to: HHMMMacroState.CODING,      probability: 0.30, agents: [PATAgent.CODER] },
  { from: HHMMMacroState.RESEARCHING, to: HHMMMacroState.EVALUATING,  probability: 0.25, agents: [PATAgent.EVALUATOR] },
  { from: HHMMMacroState.RESEARCHING, to: HHMMMacroState.PUBLISHING,  probability: 0.15, agents: [PATAgent.PUBLISHER] },
  { from: HHMMMacroState.CODING,      to: HHMMMacroState.EVALUATING,  probability: 0.40, agents: [PATAgent.EVALUATOR] },
  { from: HHMMMacroState.CODING,      to: HHMMMacroState.GATE_CHECK,  probability: 0.25, agents: [PATAgent.ETHICIST] },
  { from: HHMMMacroState.CODING,      to: HHMMMacroState.CODING,      probability: 0.20, agents: [PATAgent.CODER] },
  { from: HHMMMacroState.CODING,      to: HHMMMacroState.PUBLISHING,  probability: 0.15, agents: [PATAgent.PUBLISHER] },
  { from: HHMMMacroState.EVALUATING,  to: HHMMMacroState.GATE_CHECK,  probability: 0.45, agents: [PATAgent.ETHICIST] },
  { from: HHMMMacroState.EVALUATING,  to: HHMMMacroState.CODING,      probability: 0.30, agents: [PATAgent.CODER] },
  { from: HHMMMacroState.EVALUATING,  to: HHMMMacroState.PUBLISHING,  probability: 0.25, agents: [PATAgent.PUBLISHER] },
  { from: HHMMMacroState.GATE_CHECK,  to: HHMMMacroState.PUBLISHING,  probability: 0.50, agents: [PATAgent.PUBLISHER] },
  { from: HHMMMacroState.GATE_CHECK,  to: HHMMMacroState.CODING,      probability: 0.30, agents: [PATAgent.CODER] },
  { from: HHMMMacroState.GATE_CHECK,  to: HHMMMacroState.PLANNING,    probability: 0.20, agents: [PATAgent.PLANNER] },
  { from: HHMMMacroState.PUBLISHING,  to: HHMMMacroState.IDLE,        probability: 0.50, agents: [PATAgent.DEMA] },
  { from: HHMMMacroState.PUBLISHING,  to: HHMMMacroState.FEDERATING,  probability: 0.30, agents: [PATAgent.DEMA] },
  { from: HHMMMacroState.PUBLISHING,  to: HHMMMacroState.PLANNING,    probability: 0.20, agents: [PATAgent.PLANNER] },
  { from: HHMMMacroState.FEDERATING,  to: HHMMMacroState.IDLE,        probability: 0.60, agents: [PATAgent.DEMA] },
  { from: HHMMMacroState.FEDERATING,  to: HHMMMacroState.PLANNING,    probability: 0.40, agents: [PATAgent.PLANNER] },
];

/**
 * AgentRouter — HHMM-based agent selection
 *
 * Classifies incoming missions into macro-states, then selects
 * the optimal 2-4 agents (PAT + SAT) for execution.
 */
export class AgentRouter {
  private currentState: HHMMMacroState = HHMMMacroState.IDLE;
  private readonly transitions: readonly HHMMTransition[];

  constructor(transitions?: readonly HHMMTransition[]) {
    this.transitions = transitions ?? TRANSITION_MATRIX;
  }

  /**
   * Route a mission to the optimal agent set.
   *
   * 1. Classify mission description → macro-state (keyword match)
   * 2. Select PAT agents for that macro-state
   * 3. Attach required SAT agents
   * 4. Respect maxAgents cap (from SONA mode)
   */
  route(description: string, maxAgents: number = 4): RouteResult {
    const macroState = this.classifyMacroState(description);
    const patAgents = MACRO_STATE_AGENTS.get(macroState) ?? [PATAgent.DEMA];
    const satAgents = MACRO_STATE_SAT.get(macroState) ?? [SATAgent.SENTINEL];

    // Deduplicate and cap at maxAgents
    const allAgents = [...new Set([...patAgents, ...satAgents])];
    const selected = allAgents.slice(0, Math.max(2, maxAgents));

    const confidence = this.computeConfidence(description, macroState);

    const previousState = this.currentState;
    this.currentState = macroState;

    return {
      macroState,
      selectedAgents: selected,
      confidence,
      reason: `${previousState} → ${macroState} (${selected.length} agents)`,
    };
  }

  /** Get current macro-state */
  getCurrentState(): HHMMMacroState {
    return this.currentState;
  }

  /**
   * Predict the most likely next macro-state.
   * Used for pre-warming agents before they're needed.
   */
  predictNext(): { state: HHMMMacroState; probability: number; agents: readonly AgentId[] } | undefined {
    const outgoing = this.transitions.filter((t) => t.from === this.currentState);
    if (outgoing.length === 0) return undefined;

    let best = outgoing[0]!;
    for (const t of outgoing) {
      if (t.probability > best.probability) {
        best = t;
      }
    }
    return { state: best.to, probability: best.probability, agents: best.agents };
  }

  /** Get all transitions from current state */
  getTransitions(): readonly HHMMTransition[] {
    return this.transitions.filter((t) => t.from === this.currentState);
  }

  /**
   * Classify a mission description into a macro-state.
   * Uses keyword frequency voting across the classification table.
   */
  private classifyMacroState(description: string): HHMMMacroState {
    const words = description.toLowerCase().split(/\s+/);
    const votes = new Map<HHMMMacroState, number>();

    for (const word of words) {
      const state = KEYWORD_CLASSIFICATION.get(word);
      if (state !== undefined) {
        votes.set(state, (votes.get(state) ?? 0) + 1);
      }
    }

    if (votes.size === 0) {
      return HHMMMacroState.PLANNING; // Default: route to Planner
    }

    let maxState = HHMMMacroState.PLANNING;
    let maxVotes = 0;
    for (const [state, count] of votes) {
      if (count > maxVotes) {
        maxVotes = count;
        maxState = state;
      }
    }
    return maxState;
  }

  private computeConfidence(description: string, classified: HHMMMacroState): number {
    const words = description.toLowerCase().split(/\s+/);
    let matches = 0;
    for (const word of words) {
      if (KEYWORD_CLASSIFICATION.get(word) === classified) {
        matches++;
      }
    }
    const wordCount = Math.max(1, words.length);
    return Math.min(1.0, 0.5 + (matches / wordCount) * 0.5);
  }
}
