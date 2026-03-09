import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// We test the circuit breaker and retry logic by importing the module
// and intercepting fetch calls.

describe('ApiClient', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('should call fetch with correct path', async () => {
    const { api } = await import('../src/lib/api');

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ status: 'healthy', version: '0.3.0', uptime_seconds: 100 }),
    });

    const result = await api.health();
    expect(result.status).toBe('healthy');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/v1/health',
      expect.objectContaining({ method: 'GET' }),
    );
  });

  it('should throw on non-ok response', async () => {
    const { api } = await import('../src/lib/api');

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 500,
      text: () => Promise.resolve('Internal Server Error'),
    });

    await expect(api.health()).rejects.toThrow('API GET /v1/health: 500');
  });

  it('should submit missions through /v1/plan and map the backend payload', async () => {
    const { api } = await import('../src/lib/api');

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        status: 'completed',
        mission_id: 'mission-123',
        synthesis: 'Done.',
        ihsan_score: 0.98,
        snr_score: 0.91,
        duration_ms: 1200,
        evidence_receipt_id: 'receipt-123',
      }),
    });

    const result = await api.submitMission('Ship the feature');

    expect(result).toMatchObject({
      status: 'completed',
      mission_id: 'mission-123',
      synthesis: 'Done.',
      ihsan: 0.98,
      snr: 0.91,
      duration_ms: 1200,
      receipt_id: 'receipt-123',
    });
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/v1/plan',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          description: 'Ship the feature',
          context: {},
          source: 'ui',
        }),
      }),
    );
  });

  it('should flatten token balance payloads from the backend', async () => {
    const { api } = await import('../src/lib/api');

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        account: 'BIZRA-00000000',
        balances: {
          SEED: { balance: 42.5, staked: 5 },
          BLOOM: { balance: 1.23, staked: 0 },
        },
      }),
    });

    await expect(api.tokenBalance()).resolves.toEqual({
      seed: 42.5,
      bloom: 1.23,
      locked_seed: 5,
    });
  });

  it('should flatten token supply payloads from the backend', async () => {
    const { api } = await import('../src/lib/api');

    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({
        year: 2026,
        supply: {
          SEED: { total_supply: 10000, minted_this_year: 10000, yearly_cap: 3000000 },
          BLOOM: { total_supply: 500, minted_this_year: 500 },
        },
      }),
    });

    await expect(api.tokenSupply()).resolves.toEqual({
      total_seed: 10000,
      total_bloom: 500,
      circulating: 10000,
    });
  });

  it('should retry timed-out requests with a fresh controller', async () => {
    vi.useFakeTimers();
    vi.spyOn(Math, 'random').mockReturnValue(0);

    const { api } = await import('../src/lib/api');
    const client = api as unknown as {
      config: { timeout: number; retries: number };
    };
    const originalTimeout = client.config.timeout;
    const originalRetries = client.config.retries;
    client.config.timeout = 5;
    client.config.retries = 2;

    try {
      let callCount = 0;
      (globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation((_url, init) => {
        callCount++;
        if (callCount === 1) {
          return new Promise((_resolve, reject) => {
            init?.signal?.addEventListener('abort', () => reject(new Error('timed out')));
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ runtime: 'ok', agents: 7, uptime: 12 }),
        });
      });

      const pending = api.status();
      await vi.advanceTimersByTimeAsync(1200);

      await expect(pending).resolves.toEqual({ runtime: 'ok', agents: 7, uptime: 12 });
      expect(globalThis.fetch).toHaveBeenCalledTimes(2);
    } finally {
      client.config.timeout = originalTimeout;
      client.config.retries = originalRetries;
    }
  });
});

describe('Tokens', () => {
  it('should export constitutional thresholds', async () => {
    const { THRESHOLDS } = await import('../src/tokens');
    expect(THRESHOLDS.IHSAN_PRODUCTION).toBe(0.95);
    expect(THRESHOLDS.SNR_MINIMUM).toBe(0.85);
    expect(THRESHOLDS.ADL_GINI).toBe(0.35);
  });

  it('should resolve lifecycle stage', async () => {
    const { getStage } = await import('../src/tokens');
    expect(getStage(0).name).toBe('Seed');
    expect(getStage(0.5).name).toBe('Builder');
    expect(getStage(0.9).name).toBe('Catalyst');
  });
});

describe('Agents', () => {
  it('should route coding tasks to FORGE', async () => {
    const { routeToAgent } = await import('../src/lib/agents');
    expect(routeToAgent('build the test suite')).toBe('P3');
  });

  it('should route planning tasks to ATLAS', async () => {
    const { routeToAgent } = await import('../src/lib/agents');
    expect(routeToAgent('plan the roadmap for Q2')).toBe('P1');
  });

  it('should default to ORACLE for unknown tasks', async () => {
    const { routeToAgent } = await import('../src/lib/agents');
    expect(routeToAgent('something unrelated')).toBe('P2');
  });
});
