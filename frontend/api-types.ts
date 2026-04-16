/**
 * BIZRA Sovereign API — TypeScript Client Types (v1.3.0)
 *
 * Auto-generated from docs/openapi.json
 * DO NOT EDIT — regenerate with:
 *   python scripts/generate_frontend_types.py
 *
 * 59 routes | 21 models | 13 domains
 */

// ═══════════════════════════════════════════════════════════
// Request / Response Models
// ═══════════════════════════════════════════════════════════

/** FastAPI request model for /v1/verify/audit-log. */
export interface AuditLogVerifyModel {
  entries: Record<string, unknown>[];
}

/** Per-channel execution result within a mission. */
export interface ChannelResult {
  channel: string;
  success: boolean;
  duration_ms: number;
}

/** Request model for /v1/cognitive/fuse — direct cognitive fusion pipeline. */
export interface CognitiveFuseModel {
  query: string;
  context?: Record<string, unknown>;
}

/** FastAPI request model for /v1/verify/envelope. */
export interface EnvelopeVerifyModel {
  envelope: Record<string, unknown>;
}

/** Request model for /v1/judgment/simulate — proportional epoch distribution. */
export interface EpochSimulateModel {
  impacts?: Record<string, unknown>[];
  epoch_cap?: number;
}

/** Login request. */
export interface LoginRequestModel {
  username: string;
  password: string;
}

/** Request model for /v1/memory/search — AgentDB hybrid search. */
export interface MemorySearchModel {
  query: string;
  top_k?: number;
  min_score?: number;
  source?: string;
}

/** Response model for POST /v1/plan — receipted mission result. */
export interface MissionPlanResponse {
  mission_id: string;
  /** COMPLETE | PARTIAL | FAILED */
  status: string;
  synthesis: string;
  /** Ihsan excellence score (0.0–1.0, gate at 0.95). */
  ihsan_score: number;
  /** Signal-to-noise ratio (0.0–1.0, minimum 0.85). */
  snr_score: number;
  duration_ms: number;
  evidence_receipt_id?: string;
  channels_executed?: ChannelResult[];
}

/** FastAPI request model for /v1/orchestrate. */
export interface OrchestrateRequestModel {
  task: string;
  context?: Record<string, unknown>;
  max_agents?: number;
}

/** FastAPI request model for /v1/verify/poi. */
export interface PoIReceiptVerifyModel {
  receipt: Record<string, unknown>;
}

/** FastAPI request model for /v1/query. */
export interface QueryRequestModel {
  query: string;
  context?: Record<string, unknown>;
  require_reasoning?: boolean;
  require_validation?: boolean;
  max_depth?: number;
  timeout_ms?: number;
}

/** FastAPI request model for /v1/verify/receipt. */
export interface ReceiptVerifyModel {
  receipt: Record<string, unknown>;
}

/** Token refresh request. */
export interface RefreshTokenModel {
  refresh_token: string;
}

/** Registration request. */
export interface RegisterRequestModel {
  username: string;
  email: string;
  password: string;
  accept_covenant?: boolean;
}

/** Request model for /v1/sel/retrieve — RIR-based episode retrieval. */
export interface SELRetrieveModel {
  query: string;
  top_k?: number;
}

/** Request model for /v1/spearpoint/improve — innovation through evaluator gate. */
export interface SpearpointImproveModel {
  observation?: Record<string, unknown>;
  top_k?: number;
}

/** Request model for /v1/spearpoint/pattern — pattern-aware research via Sci-Reasoning. */
export interface SpearpointPatternModel {
  pattern_id: string;
  claim_context?: string;
  top_k?: number;
}

/** Request model for /v1/spearpoint/reproduce — evaluation-first verification. */
export interface SpearpointReproduceModel {
  claim: string;
  proposed_change?: string;
  prompt?: string;
  response?: string;
  metrics?: Record<string, unknown>;
}

/** FastAPI request model for /v1/validate. */
export interface ValidateRequestModel {
  content: string;
  task: string;
  level?: string;
}

// ═══════════════════════════════════════════════════════════
// API Endpoint Paths
// ═══════════════════════════════════════════════════════════

export const API_BASE = "/v1";

export const API_ENDPOINTS = {
  GET__: { method: "GET", path: "/" },
  GET_artifacts_graph_query_id: { method: "GET", path: "/v1/artifacts/graph/{query_id}" },
  POST_auth_login: { method: "POST", path: "/v1/auth/login" },
  GET_auth_me: { method: "GET", path: "/v1/auth/me" },
  POST_auth_refresh: { method: "POST", path: "/v1/auth/refresh" },
  POST_auth_register: { method: "POST", path: "/v1/auth/register" },
  POST_cognitive_fuse: { method: "POST", path: "/v1/cognitive/fuse" },
  GET_cognitive_status: { method: "GET", path: "/v1/cognitive/status" },
  GET_constitutional_status: { method: "GET", path: "/v1/constitutional/status" },
  POST_constitutional_tick: { method: "POST", path: "/v1/constitutional/tick" },
  GET_gate_chain_stats: { method: "GET", path: "/v1/gate-chain/stats" },
  GET_health: { method: "GET", path: "/v1/health" },
  GET_health_deep: { method: "GET", path: "/v1/health/deep" },
  GET_health_live: { method: "GET", path: "/v1/health/live" },
  GET_health_ready: { method: "GET", path: "/v1/health/ready" },
  POST_judgment_simulate: { method: "POST", path: "/v1/judgment/simulate" },
  GET_judgment_stability: { method: "GET", path: "/v1/judgment/stability" },
  GET_judgment_stats: { method: "GET", path: "/v1/judgment/stats" },
  POST_memory_search: { method: "POST", path: "/v1/memory/search" },
  GET_memory_stats: { method: "GET", path: "/v1/memory/stats" },
  GET_metrics: { method: "GET", path: "/v1/metrics" },
  GET_network_effect: { method: "GET", path: "/v1/network/effect" },
  GET_network_milestones: { method: "GET", path: "/v1/network/milestones" },
  GET_node_lifecycle: { method: "GET", path: "/v1/node/lifecycle" },
  GET_node_value: { method: "GET", path: "/v1/node/value" },
  GET_onboarding_state: { method: "GET", path: "/v1/onboarding/state" },
  POST_onboarding_teach: { method: "POST", path: "/v1/onboarding/teach" },
  POST_orchestrate: { method: "POST", path: "/v1/orchestrate" },
  POST_plan: { method: "POST", path: "/v1/plan" },
  GET_poi_contributor_contributor_id: { method: "GET", path: "/v1/poi/contributor/{contributor_id}" },
  POST_poi_epoch: { method: "POST", path: "/v1/poi/epoch" },
  GET_poi_stats: { method: "GET", path: "/v1/poi/stats" },
  POST_query: { method: "POST", path: "/v1/query" },
  POST_sat_epoch: { method: "POST", path: "/v1/sat/epoch" },
  GET_sat_stats: { method: "GET", path: "/v1/sat/stats" },
  GET_seed_episodes: { method: "GET", path: "/v1/seed/episodes" },
  GET_seed_potential: { method: "GET", path: "/v1/seed/potential" },
  GET_sel_episodes: { method: "GET", path: "/v1/sel/episodes" },
  GET_sel_episodes_episode_hash: { method: "GET", path: "/v1/sel/episodes/{episode_hash}" },
  POST_sel_retrieve: { method: "POST", path: "/v1/sel/retrieve" },
  GET_sel_verify: { method: "GET", path: "/v1/sel/verify" },
  POST_spearpoint_improve: { method: "POST", path: "/v1/spearpoint/improve" },
  POST_spearpoint_pattern: { method: "POST", path: "/v1/spearpoint/pattern" },
  POST_spearpoint_reproduce: { method: "POST", path: "/v1/spearpoint/reproduce" },
  GET_spearpoint_stats: { method: "GET", path: "/v1/spearpoint/stats" },
  GET_status: { method: "GET", path: "/v1/status" },
  GET_suggestions: { method: "GET", path: "/v1/suggestions" },
  GET_token_balance: { method: "GET", path: "/v1/token/balance" },
  GET_token_supply: { method: "GET", path: "/v1/token/supply" },
  GET_token_verify: { method: "GET", path: "/v1/token/verify" },
  POST_validate: { method: "POST", path: "/v1/validate" },
  POST_verify_audit_log: { method: "POST", path: "/v1/verify/audit-log" },
  POST_verify_envelope: { method: "POST", path: "/v1/verify/envelope" },
  POST_verify_genesis: { method: "POST", path: "/v1/verify/genesis" },
  GET_verify_genesis_header: { method: "GET", path: "/v1/verify/genesis/header" },
  POST_verify_ledger: { method: "POST", path: "/v1/verify/ledger" },
  POST_verify_poi: { method: "POST", path: "/v1/verify/poi" },
  POST_verify_receipt: { method: "POST", path: "/v1/verify/receipt" },
  GET_verify_signature: { method: "GET", path: "/v1/verify/signature" },
} as const;

// ═══════════════════════════════════════════════════════════
// Constitutional Thresholds (synced from constants.py)
// ═══════════════════════════════════════════════════════════

export const THRESHOLDS = {
  IHSAN_PRODUCTION: 0.95,
  SNR_MINIMUM: 0.85,
  SNR_T1_HIGH: 0.95,
  SNR_T0_ELITE: 0.98,
  ADL_GINI_MAX: 0.35,
  API_P99_LATENCY_MS: 200,
} as const;

export type MissionStatus = "COMPLETE" | "PARTIAL" | "FAILED";

export type RouteExposure = "public" | "bootstrap_public" | "authenticated";
