import { describe, it, expect, beforeEach } from 'vitest';
import {
  loadAppSession,
  saveAppSession,
  loadMissionSession,
  saveMissionSession,
  missionSessionKey,
  EMPTY_TEACH_DRAFT,
} from '../src/lib/persistence';
import { INITIAL_NODE_STATE } from '../src/types';

beforeEach(() => {
  localStorage.clear();
});

describe('App Session Persistence', () => {
  it('returns defaults when storage is empty', () => {
    const session = loadAppSession();
    expect(session.phase).toBe('trust');
    expect(session.userName).toBe('');
    expect(session.config).toEqual({});
    expect(session.teachDraft).toEqual(EMPTY_TEACH_DRAFT);
  });

  it('round-trips session state', () => {
    const state = {
      phase: 'dashboard' as const,
      userName: 'TestNode',
      pendingIdentityName: 'TestNode',
      config: { work_schedule: '9-5', autonomy: 'Auto low-risk, ask high-risk' },
      teachDraft: EMPTY_TEACH_DRAFT,
    };
    saveAppSession(state);
    const loaded = loadAppSession();
    expect(loaded.phase).toBe('dashboard');
    expect(loaded.userName).toBe('TestNode');
    expect(loaded.config.work_schedule).toBe('9-5');
  });

  it('recovers from corrupted storage', () => {
    localStorage.setItem('bizra.ddagi.app-session.v1', '{invalid json');
    const session = loadAppSession();
    expect(session.phase).toBe('trust');
  });
});

describe('Mission Session Persistence', () => {
  it('generates deterministic keys', () => {
    expect(missionSessionKey('TestUser')).toBe('bizra.ddagi.mission.v1.testuser');
    expect(missionSessionKey('Test User')).toBe('bizra.ddagi.mission.v1.test-user');
  });

  it('returns defaults when empty', () => {
    const session = loadMissionSession('nobody');
    expect(session.messages).toEqual([]);
    expect(session.nodeState).toEqual(INITIAL_NODE_STATE);
  });

  it('round-trips mission state', () => {
    const state = {
      messages: [{ agent: 'NEXUS', text: 'hello', type: 'greet' as const, ts: 1000 }],
      nodeState: { ...INITIAL_NODE_STATE, seed: 5.5, rac: 3 },
    };
    saveMissionSession('TestNode', state);
    const loaded = loadMissionSession('TestNode');
    expect(loaded.messages).toHaveLength(1);
    expect(loaded.messages[0].agent).toBe('NEXUS');
    expect(loaded.nodeState.seed).toBe(5.5);
    expect(loaded.nodeState.rac).toBe(3);
  });

  it('truncates message history to 80', () => {
    const messages = Array.from({ length: 100 }, (_, i) => ({
      agent: 'SYS',
      text: `msg-${i}`,
      type: 'agent' as const,
      ts: i,
    }));
    saveMissionSession('BigHistory', { messages, nodeState: INITIAL_NODE_STATE });
    const loaded = loadMissionSession('BigHistory');
    expect(loaded.messages).toHaveLength(80);
    expect(loaded.messages[0].text).toBe('msg-20');
  });
});
