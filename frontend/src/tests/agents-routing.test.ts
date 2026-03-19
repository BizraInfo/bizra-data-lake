/**
 * Agent Routing Tests — Keyword-based Task Routing
 * ==================================================
 * Tests the routeToAgent function that maps natural language tasks
 * to the correct PAT agent slot.
 */

import { describe, it, expect } from 'vitest';
import {
  routeToAgent,
  PAT,
  SAT_AGENTS,
  AGENT_IDS,
  SKILLS,
  SCHEDULED_MISSIONS,
  TEACH_QUESTIONS,
  QUICK_MISSIONS,
  ROUTING_KEYWORDS,
} from '../lib/agents';

// ═══ ROUTING ═══

describe('Agent Routing: routeToAgent', () => {
  it('routes "plan" tasks to P1 (ATLAS / Planner)', () => {
    expect(routeToAgent('plan the next sprint')).toBe('P1');
  });

  it('routes "organize my schedule" to P1', () => {
    expect(routeToAgent('organize my schedule for next week')).toBe('P1');
  });

  it('routes "research" tasks to P2 (ORACLE / Researcher)', () => {
    expect(routeToAgent('research sovereign AI architectures')).toBe('P2');
  });

  it('routes "code" tasks to P3 (FORGE / Coder)', () => {
    expect(routeToAgent('code the authentication module')).toBe('P3');
  });

  it('routes "build" tasks to P3', () => {
    expect(routeToAgent('build the dashboard component')).toBe('P3');
  });

  it('routes "test" tasks to P3', () => {
    expect(routeToAgent('test the wallet hardening logic')).toBe('P3');
  });

  it('routes "evaluate" tasks to P4 (JUDGE / Evaluator)', () => {
    expect(routeToAgent('evaluate the results carefully')).toBe('P4');
  });

  it('routes "audit" tasks to P4', () => {
    expect(routeToAgent('audit the security posture')).toBe('P4');
  });

  it('routes "ethics check" to P5 (CROWN / Ethicist)', () => {
    expect(routeToAgent('check ethics compliance')).toBe('P5');
  });

  it('routes "write" tasks to P6 (HERALD / Publisher)', () => {
    expect(routeToAgent('write the quarterly report')).toBe('P6');
  });

  it('routes "draft" tasks to P6', () => {
    expect(routeToAgent('draft the progress report')).toBe('P6');
  });

  it('defaults to P2 for unmatched tasks', () => {
    expect(routeToAgent('do something completely unknown')).toBe('P2');
  });

  it('handles multi-keyword match — highest score wins', () => {
    // "code and test" has 2 P3 keywords vs 0 for others
    expect(routeToAgent('code and test the module')).toBe('P3');
  });

  it('is case-insensitive', () => {
    expect(routeToAgent('RESEARCH the topic')).toBe('P2');
    expect(routeToAgent('Plan The Sprint')).toBe('P1');
  });
});

// ═══ PAT DATA ═══

describe('PAT Agent Definitions', () => {
  it('has 7 agents (P1-P7)', () => {
    expect(AGENT_IDS).toHaveLength(7);
  });

  it('every agent has required fields', () => {
    for (const id of AGENT_IDS) {
      const agent = PAT[id];
      expect(agent.name.length).toBeGreaterThan(0);
      expect(agent.callsign.length).toBeGreaterThan(0);
      expect(agent.domain.length).toBeGreaterThan(0);
      expect(agent.bootMsg.length).toBeGreaterThan(0);
      expect(agent.icon.length).toBeGreaterThan(0);
      expect(agent.color.length).toBeGreaterThan(0);
      expect(agent.idle.length).toBeGreaterThan(0);
      expect(agent.working.length).toBeGreaterThan(0);
    }
  });
});

// ═══ SAT DATA ═══

describe('SAT Agent Data', () => {
  it('has 5 system agents', () => {
    expect(SAT_AGENTS).toHaveLength(5);
  });

  it('every SAT agent has name and color', () => {
    for (const agent of SAT_AGENTS) {
      expect(agent.name.length).toBeGreaterThan(0);
      expect(agent.color.length).toBeGreaterThan(0);
    }
  });
});

// ═══ SKILLS ═══

describe('Skills Data', () => {
  it('has 16 skills', () => {
    expect(SKILLS).toHaveLength(16);
  });

  it('every skill has required fields', () => {
    for (const skill of SKILLS) {
      expect(skill.id.length).toBeGreaterThan(0);
      expect(skill.name.length).toBeGreaterThan(0);
      expect(typeof skill.tier).toBe('number');
      expect(skill.tier).toBeGreaterThanOrEqual(0);
      expect(skill.tier).toBeLessThanOrEqual(5);
    }
  });

  it('skill IDs are unique', () => {
    const ids = SKILLS.map(s => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('tier 0 skills are unlocked by default', () => {
    const tier0 = SKILLS.filter(s => s.tier === 0);
    expect(tier0.length).toBeGreaterThan(0);
    for (const s of tier0) {
      expect(s.unlocked).toBe(true);
    }
  });
});

// ═══ SCHEDULED MISSIONS ═══

describe('Scheduled Missions', () => {
  it('has at least 4 missions', () => {
    expect(SCHEDULED_MISSIONS.length).toBeGreaterThanOrEqual(4);
  });

  it('every mission has required fields', () => {
    for (const m of SCHEDULED_MISSIONS) {
      expect(m.id.length).toBeGreaterThan(0);
      expect(m.name.length).toBeGreaterThan(0);
      expect(m.cron.length).toBeGreaterThan(0);
      expect(m.agents.length).toBeGreaterThan(0);
    }
  });
});

// ═══ TEACH QUESTIONS ═══

describe('Teach Questions', () => {
  it('has 5 onboarding questions', () => {
    expect(TEACH_QUESTIONS).toHaveLength(5);
  });

  it('every question has an ID and prompt', () => {
    for (const q of TEACH_QUESTIONS) {
      expect(q.id.length).toBeGreaterThan(0);
      expect(q.prompt.length).toBeGreaterThan(0);
    }
  });
});

// ═══ ROUTING KEYWORDS ═══

describe('Routing Keywords', () => {
  it('P7 has empty keywords (orchestrator catches all)', () => {
    expect(ROUTING_KEYWORDS.P7).toHaveLength(0);
  });

  it('no duplicate keywords across agents', () => {
    const all: string[] = [];
    for (const keywords of Object.values(ROUTING_KEYWORDS)) {
      all.push(...keywords);
    }
    expect(new Set(all).size).toBe(all.length);
  });
});

// ═══ QUICK MISSIONS ═══

describe('Quick Missions', () => {
  it('has pre-defined missions', () => {
    expect(QUICK_MISSIONS.length).toBeGreaterThan(0);
  });

  it('every mission is a non-empty string', () => {
    for (const m of QUICK_MISSIONS) {
      expect(typeof m).toBe('string');
      expect(m.length).toBeGreaterThan(0);
    }
  });
});
