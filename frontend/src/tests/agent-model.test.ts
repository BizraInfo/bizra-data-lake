/**
 * Agent Model Tests — PAT/SAT Ontology + Persona Layer
 * ======================================================
 * Verifies the canonical agent ontology mirrors core/pat/agent.py
 * and that the presentation layer correctly resolves callsigns.
 */

import { describe, it, expect } from 'vitest';
import {
  CANONICAL_AGENT_TYPES,
  AGENT_SLOTS,
  PAT_CANONICAL,
  SAT_COUNT_PER_USER,
  SAT_CANONICAL,
  agentPersona,
  slotForCallsign,
  allCallsigns,
} from '../lib/agent-model';
import type { AgentSlot, UICallsign } from '../lib/agent-model';

// ═══ CANONICAL AGENT TYPES ═══

describe('Canonical Agent Types', () => {
  it('has 7 canonical types (mirrors core/pat/agent.py)', () => {
    expect(CANONICAL_AGENT_TYPES).toHaveLength(7);
  });

  it('includes all expected types', () => {
    const expected = ['WORKER', 'RESEARCHER', 'GUARDIAN', 'SYNTHESIZER', 'VALIDATOR', 'COORDINATOR', 'EXECUTOR'];
    for (const t of expected) {
      expect(CANONICAL_AGENT_TYPES).toContain(t);
    }
  });
});

// ═══ PAT SLOTS ═══

describe('PAT Slots', () => {
  it('has 7 slots (P1-P7)', () => {
    expect(AGENT_SLOTS).toHaveLength(7);
    expect(AGENT_SLOTS[0]).toBe('P1');
    expect(AGENT_SLOTS[6]).toBe('P7');
  });

  it('every slot has a canonical definition', () => {
    for (const slot of AGENT_SLOTS) {
      const agent = PAT_CANONICAL[slot];
      expect(agent).toBeDefined();
      expect(agent.slot).toBe(slot);
      expect(CANONICAL_AGENT_TYPES).toContain(agent.type);
      expect(agent.domain.length).toBeGreaterThan(0);
    }
  });

  it('slot types are unique (no duplicate assignments)', () => {
    const types = AGENT_SLOTS.map(s => PAT_CANONICAL[s].type);
    expect(new Set(types).size).toBe(types.length);
  });
});

// ═══ SAT ═══

describe('SAT (System Agentic Team)', () => {
  it('SAT_COUNT_PER_USER is 5', () => {
    expect(SAT_COUNT_PER_USER).toBe(5);
  });

  it('SAT_CANONICAL has 5 roles', () => {
    expect(SAT_CANONICAL).toHaveLength(5);
  });

  it('total agents per user = 12 (7 PAT + 5 SAT)', () => {
    expect(AGENT_SLOTS.length + SAT_COUNT_PER_USER).toBe(12);
  });
});

// ═══ PERSONA LAYER ═══

describe('Agent Persona Layer', () => {
  it('every slot maps to a persona', () => {
    for (const slot of AGENT_SLOTS) {
      const persona = agentPersona(slot);
      expect(persona).toBeDefined();
      expect(persona.callsign.length).toBeGreaterThan(0);
      expect(persona.displayName.length).toBeGreaterThan(0);
      expect(persona.icon.length).toBeGreaterThan(0);
    }
  });

  it('P1 is ATLAS (Planner)', () => {
    const p = agentPersona('P1');
    expect(p.callsign).toBe('ATLAS');
  });

  it('P2 is ORACLE (Researcher)', () => {
    const p = agentPersona('P2');
    expect(p.callsign).toBe('ORACLE');
  });

  it('P3 is FORGE (Coder)', () => {
    const p = agentPersona('P3');
    expect(p.callsign).toBe('FORGE');
  });
});

// ═══ CALLSIGN RESOLUTION ═══

describe('Callsign Resolution', () => {
  it('allCallsigns returns 7 unique callsigns', () => {
    const cs = allCallsigns();
    expect(cs).toHaveLength(7);
    expect(new Set(cs).size).toBe(7);
  });

  it('slotForCallsign reverses agentPersona', () => {
    for (const slot of AGENT_SLOTS) {
      const persona = agentPersona(slot);
      const resolved = slotForCallsign(persona.callsign as UICallsign);
      expect(resolved).toBe(slot);
    }
  });

  it('slotForCallsign returns undefined for unknown callsign', () => {
    const result = slotForCallsign('UNKNOWN' as UICallsign);
    expect(result).toBeUndefined();
  });
});
