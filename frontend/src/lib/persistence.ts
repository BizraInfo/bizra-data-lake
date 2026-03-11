import type { AppPhase, FeedMessage, NodeState, TeachDraftState, UserConfig } from '../types';
import { INITIAL_NODE_STATE } from '../types';

const APP_SESSION_KEY = 'bizra.ddagi.app-session.v1';
const MISSION_SESSION_PREFIX = 'bizra.ddagi.mission.v1';

export interface AppSessionState {
  phase: AppPhase;
  userName: string;
  pendingIdentityName: string;
  config: UserConfig;
  teachDraft: TeachDraftState;
}

export interface MissionSessionState {
  messages: FeedMessage[];
  nodeState: NodeState;
}

export const EMPTY_TEACH_DRAFT: TeachDraftState = {
  step: 0,
  answers: {},
  textValue: '',
  selected: [],
};

const DEFAULT_APP_SESSION: AppSessionState = {
  phase: 'trust',
  userName: '',
  pendingIdentityName: '',
  config: {},
  teachDraft: EMPTY_TEACH_DRAFT,
};

const DEFAULT_MISSION_SESSION: MissionSessionState = {
  messages: [],
  nodeState: INITIAL_NODE_STATE,
};

function canUseStorage(): boolean {
  return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';
}

function readJson<T>(key: string, fallback: T): T {
  if (!canUseStorage()) {
    return fallback;
  }

  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) {
      return fallback;
    }
    return JSON.parse(raw) as T;
  } catch {
    window.localStorage.removeItem(key);
    return fallback;
  }
}

function writeJson<T>(key: string, value: T): void {
  if (!canUseStorage()) {
    return;
  }

  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Ignore persistence failures; runtime stays local-first in memory.
  }
}

function sanitizeMissionSession(session: MissionSessionState): MissionSessionState {
  return {
    messages: Array.isArray(session.messages) ? session.messages.slice(-80) : [],
    nodeState: session.nodeState ?? INITIAL_NODE_STATE,
  };
}

export function loadAppSession(): AppSessionState {
  const stored = readJson<AppSessionState>(APP_SESSION_KEY, DEFAULT_APP_SESSION);

  return {
    phase: stored.phase ?? DEFAULT_APP_SESSION.phase,
    userName: stored.userName ?? '',
    pendingIdentityName: stored.pendingIdentityName ?? '',
    config: stored.config ?? {},
    teachDraft: {
      step: stored.teachDraft?.step ?? 0,
      answers: stored.teachDraft?.answers ?? {},
      textValue: stored.teachDraft?.textValue ?? '',
      selected: stored.teachDraft?.selected ?? [],
    },
  };
}

export function saveAppSession(session: AppSessionState): void {
  writeJson(APP_SESSION_KEY, session);
}

export function missionSessionKey(userName: string): string {
  const normalized = userName.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return normalized ? `${MISSION_SESSION_PREFIX}.${normalized}` : MISSION_SESSION_PREFIX;
}

export function loadMissionSession(userName: string): MissionSessionState {
  const stored = readJson<MissionSessionState>(missionSessionKey(userName), DEFAULT_MISSION_SESSION);
  return sanitizeMissionSession(stored);
}

export function saveMissionSession(userName: string, session: MissionSessionState): void {
  writeJson(missionSessionKey(userName), sanitizeMissionSession(session));
}
