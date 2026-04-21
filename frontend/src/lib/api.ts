/**
 * BIZRA DDAGI OS — Typed API Client
 *
 * Production-grade HTTP client with:
 * - Circuit breaker (fail-open after 5 consecutive failures)
 * - Exponential backoff retry (3 attempts, 1s/2s/4s)
 * - Bearer token authentication
 * - Request timeout (30s default)
 * - Response type safety
 *
 * Maps to all 22 backend endpoints defined in core/sovereign/api.py
 */

import type {
  HealthResponse,
  JudgmentStats,
  LoginResponse,
  MissionResponse,
  NodeValueResponse,
  RegisterResponse,
  SELEpisode,
  SeedPotentialResponse,
  TokenBalanceResponse,
} from '../types';

// ═══ Circuit Breaker ═══

type CircuitState = 'closed' | 'open' | 'half-open';

class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failures = 0;
  private lastFailure = 0;
  private readonly threshold: number;
  private readonly resetMs: number;

  constructor(threshold = 5, resetMs = 30_000) {
    this.threshold = threshold;
    this.resetMs = resetMs;
  }

  canRequest(): boolean {
    if (this.state === 'closed') return true;
    if (this.state === 'open') {
      if (Date.now() - this.lastFailure > this.resetMs) {
        this.state = 'half-open';
        return true;
      }
      return false;
    }
    return true; // half-open: allow one probe
  }

  recordSuccess(): void {
    this.failures = 0;
    this.state = 'closed';
  }

  recordFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.state = 'open';
    }
  }

  getState(): CircuitState {
    return this.state;
  }
}

// ═══ Retry Logic ═══

async function withRetry<T>(
  fn: () => Promise<T>,
  attempts = 3,
  baseDelayMs = 1000,
): Promise<T> {
  let lastError: Error | undefined;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (i < attempts - 1) {
        const jitter = Math.random() * 200;
        await new Promise(r => setTimeout(r, baseDelayMs * Math.pow(2, i) + jitter));
      }
    }
  }
  throw lastError;
}

// ═══ API Client ═══

export interface ApiClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
}

const DEFAULT_CONFIG: ApiClientConfig = {
  baseUrl: import.meta.env.VITE_API_URL || '',
  timeout: 30_000,
  retries: 3,
};

interface BackendTokenBalanceEntry {
  balance?: number;
  staked?: number;
}

interface BackendTokenBalancePayload {
  account: string;
  balances?: Record<string, BackendTokenBalanceEntry>;
}

interface BackendTokenSupplyEntry {
  total_supply?: number;
  minted_this_year?: number;
  yearly_cap?: number;
}

interface BackendTokenSupplyPayload {
  year: number;
  supply?: Record<string, BackendTokenSupplyEntry>;
  ledger_valid?: boolean;
  transaction_count?: number;
}

interface BackendMissionPayload {
  status: string;
  mission_id: string;
  synthesis: string;
  ihsan_score: number;
  snr_score: number;
  duration_ms: number;
  evidence_receipt_id?: string | null;
}

function clampUnitScore(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

class ApiClient {
  private config: ApiClientConfig;
  private token: string | null = null;
  private breaker = new CircuitBreaker();

  constructor(config: Partial<ApiClientConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  setToken(token: string): void {
    this.token = token;
  }

  clearToken(): void {
    this.token = null;
  }

  getCircuitState(): CircuitState {
    return this.breaker.getState();
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    options: { noRetry?: boolean; noAuth?: boolean } = {},
  ): Promise<T> {
    if (!this.breaker.canRequest()) {
      throw new Error(`Circuit breaker OPEN — API unavailable. Retry after cooldown.`);
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.token && !options.noAuth) {
      headers['X-API-Key'] = this.token;
    }

    const doFetch = async (): Promise<T> => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

      try {
        const res = await fetch(`${this.config.baseUrl}${path}`, {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });

        if (!res.ok) {
          const text = await res.text().catch(() => '');
          throw new Error(`API ${method} ${path}: ${res.status} ${text.slice(0, 200)}`);
        }

        return res.json() as Promise<T>;
      } finally {
        clearTimeout(timeoutId);
      }
    };

    try {
      const result = options.noRetry
        ? await doFetch()
        : await withRetry(doFetch, this.config.retries);
      this.breaker.recordSuccess();
      return result;
    } catch (err) {
      this.breaker.recordFailure();
      throw err;
    }
  }

  // ─── Health ───
  health(): Promise<HealthResponse> {
    return this.request('GET', '/v1/health', undefined, { noRetry: true, noAuth: true });
  }

  healthDeep(): Promise<HealthResponse> {
    return this.request('GET', '/v1/health/deep');
  }

  // ─── Auth ───
  // Backend truth: core/sovereign/api.py auth endpoints.
  // RegisterResponse and LoginResponse have DIFFERENT shapes (see types.ts).
  // Register requires email + accept_covenant; login only username + password.
  login(credentials: { username: string; password: string }): Promise<LoginResponse> {
    return this.request('POST', '/v1/auth/login', credentials, { noAuth: true });
  }

  register(
    data: { username: string; email: string; password: string; accept_covenant: boolean },
  ): Promise<RegisterResponse> {
    return this.request('POST', '/v1/auth/register', data, { noAuth: true });
  }

  me(): Promise<{ node_id: string; name: string; tier: string }> {
    return this.request('GET', '/v1/auth/me');
  }

  // ─── Seed Engine (Phase 71) ───
  seedPotential(): Promise<SeedPotentialResponse> {
    return this.request('GET', '/v1/seed/potential');
  }

  seedEpisodes(page = 1, limit = 20): Promise<{ episodes: SELEpisode[]; total: number }> {
    return this.request('GET', `/v1/seed/episodes?page=${page}&limit=${limit}`);
  }

  // ─── Token Economy ───
  tokenBalance(): Promise<TokenBalanceResponse> {
    return this.request<BackendTokenBalancePayload>('GET', '/v1/token/balance').then((payload) => {
      const seed = payload.balances?.SEED ?? {};
      const bloom = payload.balances?.BLOOM ?? {};

      return {
        seed: seed.balance ?? 0,
        bloom: bloom.balance ?? 0,
        locked_seed: seed.staked ?? 0,
      };
    });
  }

  tokenSupply(): Promise<{ total_seed: number; total_bloom: number; circulating: number }> {
    return this.request<BackendTokenSupplyPayload>('GET', '/v1/token/supply').then((payload) => {
      const seed = payload.supply?.SEED ?? {};
      const bloom = payload.supply?.BLOOM ?? {};
      const totalSeed = seed.total_supply ?? 0;

      return {
        total_seed: totalSeed,
        total_bloom: bloom.total_supply ?? 0,
        // The backend does not expose a separate circulating value yet.
        circulating: totalSeed,
      };
    });
  }

  // ─── Mission ───
  submitMission(query: string, context: Record<string, unknown> = {}): Promise<MissionResponse> {
    return this.request<BackendMissionPayload>('POST', '/v1/plan', {
      description: query,
      context,
      source: 'ui',
    }).then((payload) => ({
      status: payload.status,
      mission_id: payload.mission_id,
      synthesis: payload.synthesis,
      ihsan: clampUnitScore(payload.ihsan_score),
      snr: clampUnitScore(payload.snr_score),
      duration_ms: payload.duration_ms,
      receipt_id: payload.evidence_receipt_id ?? null,
    }));
  }

  // ─── Node (Phase 72) ───
  nodeValue(): Promise<NodeValueResponse> {
    return this.request('GET', '/v1/node/value');
  }

  nodeLifecycle(): Promise<{ stage: string; sovereignty: number; progress: number }> {
    return this.request('GET', '/v1/node/lifecycle');
  }

  networkEffect(): Promise<{ multiplier: number; connected_nodes: number }> {
    return this.request('GET', '/v1/network/effect');
  }

  // ─── Onboarding (Phase 73) ───
  onboardingState(): Promise<{ completed: boolean; step: number; config: Record<string, unknown> }> {
    return this.request('GET', '/v1/onboarding/state');
  }

  submitTeach(answers: Record<string, unknown>): Promise<{ status: string }> {
    return this.request('POST', '/v1/onboarding/teach', answers);
  }

  // ─── Query (Sovereign Engine) ───
  query(q: string, context: Record<string, unknown> = {}): Promise<{
    response: string;
    reasoning_trace: string[];
    ihsan: number;
    sources: string[];
  }> {
    return this.request('POST', '/v1/query', {
      query: q,
      context,
      require_reasoning: true,
      require_validation: true,
    });
  }

  // ─── SEL (Experience Ledger) ───
  selEpisodes(page = 1): Promise<{ episodes: SELEpisode[]; total: number }> {
    return this.request('GET', `/v1/sel/episodes?page=${page}`);
  }

  selEpisode(hash: string): Promise<SELEpisode> {
    return this.request('GET', `/v1/sel/episodes/${hash}`);
  }

  selRetrieve(query: string, k = 5): Promise<{ results: SELEpisode[] }> {
    return this.request('POST', '/v1/sel/retrieve', { query, k });
  }

  selVerify(): Promise<{ valid: boolean; chain_length: number; head_hash: string }> {
    return this.request('GET', '/v1/sel/verify');
  }

  // ─── Judgment Telemetry (SJE Phase A) ───
  judgmentStats(): Promise<JudgmentStats> {
    return this.request('GET', '/v1/judgment/stats');
  }

  judgmentStability(): Promise<{ is_stable: boolean; volatility: number }> {
    return this.request('GET', '/v1/judgment/stability');
  }

  judgmentSimulate(epochs: number): Promise<{ results: Record<string, number>[] }> {
    return this.request('POST', '/v1/judgment/simulate', { epochs });
  }

  // ─── Status & Metrics ───
  status(): Promise<{ runtime: string; agents: number; uptime: number }> {
    return this.request('GET', '/v1/status');
  }

  metrics(): Promise<string> {
    // Prometheus text format
    return this.request('GET', '/v1/metrics');
  }
}

// Singleton export
export const api = new ApiClient();
export type { CircuitState };
