import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useWallet } from '../src/hooks/useWallet';
import { THRESHOLDS } from '../src/tokens';
import type { NodeState } from '../src/types';
import { INITIAL_NODE_STATE } from '../src/types';

// Mock the API module
vi.mock('../src/lib/api', () => ({
  api: {
    tokenBalance: vi.fn(),
    tokenSupply: vi.fn(),
    seedPotential: vi.fn(),
  },
}));

import { api } from '../src/lib/api';

const mockBalance = { seed: 42.5, bloom: 1.23, locked_seed: 5.0 };
const mockSupply = { total_seed: 10000, total_bloom: 500, circulating: 9500 };
const mockPotential = {
  potential: 0.78,
  factors: { sovereignty: 0.6, activation: 0.8, quality: 0.97, compounding: 0.4, synergy: 0.3 },
};

const ACTIVE_NODE: NodeState = {
  ...INITIAL_NODE_STATE,
  seed: 10,
  bloom: 0.5,
  rac: 15,
  vac: 20,
  ihsan: 0.96,
  streak: 7,
  sovereignty: 0.45,
  reflexes: 2,
};

function mockOffline() {
  (api.tokenBalance as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
  (api.tokenSupply as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
  (api.seedPotential as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('offline'));
}

function mockLive() {
  (api.tokenBalance as ReturnType<typeof vi.fn>).mockResolvedValue(mockBalance);
  (api.tokenSupply as ReturnType<typeof vi.fn>).mockResolvedValue(mockSupply);
  (api.seedPotential as ReturnType<typeof vi.fn>).mockResolvedValue(mockPotential);
}

describe('useWallet', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('returns offline state when API fails', async () => {
    mockOffline();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.live).toBe(false);
    expect(result.current.seed).toBe(ACTIVE_NODE.seed);
    expect(result.current.bloom).toBe(ACTIVE_NODE.bloom);
    expect(result.current.lastSync).toBeNull();
  });

  it('returns live state when API succeeds', async () => {
    mockLive();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.live).toBe(true));

    expect(result.current.seed).toBe(42.5);
    expect(result.current.bloom).toBe(1.23);
    expect(result.current.lockedSeed).toBe(5.0);
    expect(result.current.totalSeed).toBe(10000);
    expect(result.current.circulating).toBe(9500);
    expect(result.current.factors.quality).toBe(0.97);
    expect(result.current.lastSync).toBeGreaterThan(0);
  });

  it('calculates supply cap utilization correctly', async () => {
    mockLive();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.live).toBe(true));

    expect(result.current.supplyCapUtilization).toBeCloseTo(
      mockSupply.total_seed / THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR,
      6,
    );
  });

  it('computes zakat contribution from gross seed', async () => {
    mockLive();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.live).toBe(true));

    const grossSeed = mockBalance.seed / (1 - THRESHOLDS.ZAKAT_RATE);
    const expectedZakat = +(grossSeed * THRESHOLDS.ZAKAT_RATE).toFixed(4);
    expect(result.current.zakatContributed).toBeCloseTo(expectedZakat, 4);
  });

  it('derives offline factors from nodeState', async () => {
    mockOffline();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.factors.sovereignty).toBe(ACTIVE_NODE.sovereignty);
    expect(result.current.factors.quality).toBe(ACTIVE_NODE.ihsan);
    expect(result.current.factors.activation).toBeCloseTo(
      ACTIVE_NODE.rac / Math.max(ACTIVE_NODE.vac, 1),
      4,
    );
  });

  it('exposes a refresh function that transitions offline to live', async () => {
    mockOffline();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(typeof result.current.refresh).toBe('function');
    expect(result.current.live).toBe(false);

    // Switch mocks to live and refresh
    mockLive();
    await act(async () => { await result.current.refresh(); });

    expect(result.current.live).toBe(true);
    expect(result.current.seed).toBe(42.5);
  });

  it('initializes with zero-state nodeState', async () => {
    mockOffline();
    const { result } = renderHook(() => useWallet(INITIAL_NODE_STATE));
    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.seed).toBe(0);
    expect(result.current.bloom).toBe(0);
    expect(result.current.factors.sovereignty).toBe(0);
    expect(result.current.supplyCapUtilization).toBe(0);
  });
});

// ═══════════════════════════════════════════════════════════════
// Wallet Integrity Hardening Tests
// Race conditions, partial failures, visibility coherence
// ═══════════════════════════════════════════════════════════════

describe('useWallet hardening', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('stays live on partial API failure (balance ok, supply fails)', async () => {
    (api.tokenBalance as ReturnType<typeof vi.fn>).mockResolvedValue(mockBalance);
    (api.tokenSupply as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('503'));
    (api.seedPotential as ReturnType<typeof vi.fn>).mockResolvedValue(mockPotential);

    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.loading).toBe(false));

    // Partial merge: balance + potential succeeded → live with fetchStatus tracking failure
    expect(result.current.live).toBe(true);
    expect(result.current.seed).toBe(mockBalance.seed);
    expect(result.current.fetchStatus.balance).toBe('ok');
    expect(result.current.fetchStatus.supply).toBe('error');
    expect(result.current.fetchStatus.potential).toBe('ok');
  });

  it('stays live on partial API failure (potential fails)', async () => {
    (api.tokenBalance as ReturnType<typeof vi.fn>).mockResolvedValue(mockBalance);
    (api.tokenSupply as ReturnType<typeof vi.fn>).mockResolvedValue(mockSupply);
    (api.seedPotential as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('timeout'));

    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.loading).toBe(false));

    // Partial merge: balance + supply succeeded → live
    expect(result.current.live).toBe(true);
    expect(result.current.fetchStatus.potential).toBe('error');
  });

  it('preserves lastSync from live state when total failure after live', async () => {
    // First: go live
    mockLive();
    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.live).toBe(true));
    const syncTime = result.current.lastSync;
    expect(syncTime).toBeGreaterThan(0);

    // Then: total API failure, trigger refresh
    mockOffline();
    await act(async () => { await result.current.refresh(); });

    // Recent live state preserved (within 2x poll interval)
    expect(result.current.lastSync).toBe(syncTime);
  });

  it('live state is not overwritten by stale offline nodeState update', async () => {
    mockLive();
    const { result, rerender } = renderHook(
      ({ ns }) => useWallet(ns),
      { initialProps: { ns: ACTIVE_NODE } },
    );
    await waitFor(() => expect(result.current.live).toBe(true));

    // Live seed from API should be 42.5
    expect(result.current.seed).toBe(42.5);

    // Rerender with a different nodeState (simulating stale local state)
    const staleNode = { ...ACTIVE_NODE, seed: 999 };
    rerender({ ns: staleNode });

    // wallet.live is true, so the offline derivation should NOT overwrite
    expect(result.current.seed).toBe(42.5);
    expect(result.current.live).toBe(true);
  });

  it('concurrent refresh calls do not produce mixed state', async () => {
    // In-flight guard ensures only one fetch at a time — no mixed state possible
    mockLive();

    const { result } = renderHook(() => useWallet(ACTIVE_NODE));
    await waitFor(() => expect(result.current.live).toBe(true));

    // Fire two refreshes concurrently — second should be skipped by in-flight guard
    await act(async () => {
      const p1 = result.current.refresh();
      const p2 = result.current.refresh();
      await Promise.all([p1, p2].filter(Boolean));
    });

    // State is coherent — all from the same fetch (no field mixing)
    expect(result.current.live).toBe(true);
    expect(result.current.seed).toBe(42.5);
  });

  it('refresh function is stable across renders', async () => {
    mockLive();
    const { result, rerender } = renderHook(
      ({ ns }) => useWallet(ns),
      { initialProps: { ns: ACTIVE_NODE } },
    );
    await waitFor(() => expect(result.current.live).toBe(true));

    const refresh1 = result.current.refresh;
    rerender({ ns: ACTIVE_NODE });
    const refresh2 = result.current.refresh;

    // Same nodeState => same memoized callback
    expect(refresh1).toBe(refresh2);
  });
});
