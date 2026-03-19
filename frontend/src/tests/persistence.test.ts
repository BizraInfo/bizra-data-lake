/**
 * Persistence Layer Tests — localStorage Abstraction
 * ====================================================
 * Tests the session persistence layer that manages app state
 * and mission state in localStorage with graceful fallbacks.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  loadAppSession,
  saveAppSession,
  missionSessionKey,
  loadMissionSession,
  saveMissionSession,
  EMPTY_TEACH_DRAFT,
} from '../lib/persistence';
import type { AppSessionState, MissionSessionState } from '../lib/persistence';
import { INITIAL_NODE_STATE } from '../types';

// ═══ SETUP ═══

// In-memory localStorage mock
let store: Record<string, string> = {};

beforeEach(() => {
  store = {};
  vi.stubGlobal('window', {
    localStorage: {
      getItem: vi.fn((key: string) => store[key] ?? null),
      setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
      removeItem: vi.fn((key: string) => { delete store[key]; }),
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ═══ EMPTY TEACH DRAFT ═══

describe('EMPTY_TEACH_DRAFT', () => {
  it('has step 0, empty answers, empty text, empty selected', () => {
    expect(EMPTY_TEACH_DRAFT.step).toBe(0);
    expect(EMPTY_TEACH_DRAFT.answers).toEqual({});
    expect(EMPTY_TEACH_DRAFT.textValue).toBe('');
    expect(EMPTY_TEACH_DRAFT.selected).toEqual([]);
  });
});

// ═══ APP SESSION ═══

describe('App Session Persistence', () => {
  it('loadAppSession returns defaults when nothing stored', () => {
    const session = loadAppSession();
    expect(session.phase).toBe('trust');
    expect(session.userName).toBe('');
    expect(session.pendingIdentityName).toBe('');
    expect(session.config).toEqual({});
    expect(session.teachDraft).toEqual(EMPTY_TEACH_DRAFT);
  });

  it('saveAppSession + loadAppSession round-trips', () => {
    const saved: AppSessionState = {
      phase: 'dashboard' as any,
      userName: 'bizra-user',
      pendingIdentityName: 'test-identity',
      config: { theme: 'dark' } as any,
      teachDraft: { step: 2, answers: { q1: 'a1' }, textValue: 'hello', selected: ['opt1'] },
    };
    saveAppSession(saved);
    const loaded = loadAppSession();
    expect(loaded.userName).toBe('bizra-user');
    expect(loaded.phase).toBe('dashboard');
    expect(loaded.teachDraft.step).toBe(2);
  });

  it('loadAppSession handles corrupted JSON gracefully', () => {
    store['bizra.ddagi.app-session.v1'] = '{not valid json';
    const session = loadAppSession();
    // Should return defaults, not throw
    expect(session.phase).toBe('trust');
  });
});

// ═══ MISSION SESSION ═══

describe('Mission Session Persistence', () => {
  it('missionSessionKey generates correct key', () => {
    const key = missionSessionKey('alice');
    expect(key).toContain('alice');
    expect(key).toContain('bizra.ddagi.mission.v1');
  });

  it('different users get different keys', () => {
    const k1 = missionSessionKey('alice');
    const k2 = missionSessionKey('bob');
    expect(k1).not.toBe(k2);
  });

  it('loadMissionSession returns defaults for new user', () => {
    const session = loadMissionSession('newuser');
    expect(session.messages).toEqual([]);
    expect(session.nodeState).toEqual(INITIAL_NODE_STATE);
  });

  it('saveMissionSession + loadMissionSession round-trips', () => {
    const saved: MissionSessionState = {
      messages: [{ id: 'm1', role: 'agent', agentId: 'P1', text: 'Hello', ts: 1000 } as any],
      nodeState: { ...INITIAL_NODE_STATE, seed: 42, ihsan: 0.97 },
    };
    saveMissionSession('testuser', saved);
    const loaded = loadMissionSession('testuser');
    expect(loaded.messages).toHaveLength(1);
    expect(loaded.nodeState.seed).toBe(42);
  });
});

// ═══ EDGE CASES ═══

describe('Persistence Edge Cases', () => {
  it('handles undefined window gracefully', () => {
    vi.stubGlobal('window', undefined);
    // loadAppSession should not throw when window is undefined
    const session = loadAppSession();
    expect(session.phase).toBe('trust');
  });
});
