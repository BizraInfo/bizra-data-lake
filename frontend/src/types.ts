/**
 * BIZRA DDAGI OS — Core Type Definitions
 *
 * Strict TypeScript types for all domain entities.
 * These map 1:1 to backend Pydantic models where applicable.
 */

// ═══ Agent System (PAT-7 + SAT-5) ═══

export type AgentId = 'P1' | 'P2' | 'P3' | 'P4' | 'P5' | 'P6' | 'P7';
export type AgentCallsign = 'ATLAS' | 'ORACLE' | 'FORGE' | 'JUDGE' | 'CROWN' | 'HERALD' | 'NEXUS';
export type AgentState = 'idle' | 'active' | 'routing' | 'scoring' | 'checking';

export interface AgentDef {
  readonly name: string;
  readonly callsign: AgentCallsign;
  readonly domain: string;
  readonly bootMsg: string;
  readonly icon: string;
  readonly color: string;
  readonly idle: readonly string[];
  readonly working: readonly string[];
}

export interface SatAgent {
  readonly name: string;
  readonly color: string;
}

// ═══ Mission System ═══

export type MessageType = 'user' | 'agent' | 'system' | 'work' | 'route' | 'score' | 'clear' | 'mint' | 'done' | 'pro' | 'greet';

export interface FeedMessage {
  agent: string;
  text: string;
  type: MessageType;
  ts: number;
}

/** @deprecated Use RewardReceipt from lib/reward-engine.ts — PoI-based scoring replaces rarity tiers. */
export type DropRarity = 'LEGENDARY' | 'EPIC' | 'RARE';

/** @deprecated Use RewardReceipt from lib/reward-engine.ts — retained only for MissionResponse API compat. */
export interface MissionReceipt {
  ihsan: number;
  seedEarned: number;
  bloomEarned: number;
  rarity: DropRarity;
  reflexCompiled: boolean;
  timestamp: number;
}

// ═══ Node State ═══

export interface NodeState {
  seed: number;
  bloom: number;
  rac: number;
  vac: number;
  tier: number;
  mye: number;
  s1: number;
  s2: number;
  streak: number;
  ihsan: number;
  reflexes: number;
  legendary: number;
  epic: number;
  sovereignty: number;
}

export const INITIAL_NODE_STATE: NodeState = {
  seed: 0, bloom: 0, rac: 0, vac: 0, tier: 0, mye: 0,
  s1: 0, s2: 0, streak: 0, ihsan: 0, reflexes: 0,
  legendary: 0, epic: 0, sovereignty: 0,
};

// ═══ TEACH Onboarding ═══

export type QuestionType = 'text' | 'single' | 'multi';

export interface TeachQuestion {
  id: string;
  prompt: string;
  type: QuestionType;
  default?: string;
  options?: string[];
  icon: string;
}

export interface UserConfig {
  work_schedule?: string;
  primary_tools?: string[];
  communication_pref?: string;
  priority_domains?: string[];
  autonomy?: string;
}

export interface TeachDraftState {
  step: number;
  answers: Record<string, string | string[]>;
  textValue: string;
  selected: string[];
}

// ═══ Application Phase ═══

export type AppPhase = 'trust' | 'splash' | 'genesis' | 'teach' | 'assembly' | 'dashboard';

// ═══ Skill System ═══

export interface Skill {
  id: string;
  name: string;
  tier: number;
  icon: string;
  unlocked?: boolean;
  hda?: boolean;
}

// ═══ Scheduled Mission ═══

export interface ScheduledMission {
  id: string;
  name: string;
  cron: string;
  icon: string;
  seedReward: string;
  description: string;
  auto: boolean;
  agents: AgentCallsign[];
}

// ═══ API Response Types ═══

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  checks?: Record<string, boolean>;
}

/**
 * Auth response shapes — MUST match backend `core/sovereign/api.py` auth endpoints.
 *
 * Contract anchored by pytest suite `tests/core/auth/test_phase21_auth.py` (61 passing)
 * and the cross-layer regression guard `tests/core/auth/test_auth_wire_contract.py`.
 *
 * Do NOT flatten these types — backend returns nested `tokens` + nested `user`
 * (on register only). The earlier flat `{ token, node_id, expires_at }` shape
 * was frontend-only drift and would cause runtime errors against the real
 * backend.
 */

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserProfile {
  user_id: string;
  username: string;
  email: string;
  api_key: string;
  namespace: string;
  covenant_accepted: boolean;
  created_at: string;
}

/** Response shape for `POST /v1/auth/register`. */
export interface RegisterResponse {
  user: UserProfile;
  tokens: AuthTokens;
}

/** Response shape for `POST /v1/auth/login`. */
export interface LoginResponse {
  user_id: string;
  username: string;
  tokens: AuthTokens;
}

export interface SeedPotentialResponse {
  potential: number;
  factors: {
    sovereignty: number;
    activation: number;
    quality: number;
    compounding: number;
    synergy: number;
  };
}

export interface NodeValueResponse {
  value: number;
  stage: string;
  sovereignty: number;
  tier: string;
}

export interface TokenBalanceResponse {
  seed: number;
  bloom: number;
  locked_seed: number;
}

export interface MissionResponse {
  status: string;
  mission_id: string;
  synthesis: string;
  ihsan: number;
  snr: number;
  duration_ms: number;
  receipt_id: string | null;
}

export interface SELEpisode {
  hash: string;
  query: string;
  verdict: string;
  ihsan: number;
  timestamp: string;
}

export interface JudgmentStats {
  total_verdicts: number;
  distribution: Record<string, number>;
  entropy: number;
  mean_ihsan: number;
}

// ═══ WebSocket Events ═══

export type WSEventType =
  | 'agent_status'
  | 'mission_update'
  | 'proactive_message'
  | 'health_ping'
  | 'receipt_minted';

export interface WSEvent {
  type: WSEventType;
  payload: unknown;
  timestamp: number;
}
