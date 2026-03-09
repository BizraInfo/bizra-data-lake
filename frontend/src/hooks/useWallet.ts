/**
 * Wallet hook — bridges backend token ledger with offline NodeState.
 *
 * Data flow:
 *   1. Try backend APIs: tokenBalance, tokenSupply, seedPotential
 *   2. On failure: derive from local NodeState (offline simulation)
 *   3. Merge both into a single WalletState for the UI
 *
 * Polling: every 30s when tab is visible, pauses when hidden.
 * Circuit breaker: inherits from ApiClient (5 failures → open → 30s cooldown).
 *
 * Standing on Giants:
 *   Nakamoto (2008) — verifiable token supply
 *   Al-Ghazali (1111) — zakat as constitutional redistribution
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import { THRESHOLDS } from '../tokens';
import type { NodeState } from '../types';

export interface WalletState {
  /** SEED balance (liquid). */
  seed: number;
  /** BLOOM balance (soulbound governance). */
  bloom: number;
  /** Locked SEED (staking). */
  lockedSeed: number;
  /** Cumulative zakat contributed (2.5% of gross). */
  zakatContributed: number;
  /** Network-wide total SEED minted. */
  totalSeed: number;
  /** Network-wide total BLOOM minted. */
  totalBloom: number;
  /** Circulating supply. */
  circulating: number;
  /** Supply cap utilization (0-1). */
  supplyCapUtilization: number;
  /** Seed potential factors from backend. */
  factors: {
    sovereignty: number;
    activation: number;
    quality: number;
    compounding: number;
    synergy: number;
  };
  /** Whether data came from backend (true) or offline fallback (false). */
  live: boolean;
  /** Last successful fetch timestamp. */
  lastSync: number | null;
  /** Loading state for initial fetch. */
  loading: boolean;
}

const POLL_INTERVAL_MS = 30_000;

function deriveOffline(ns: NodeState): WalletState {
  const grossSeed = ns.seed / (1 - THRESHOLDS.ZAKAT_RATE);
  return {
    seed: ns.seed,
    bloom: ns.bloom,
    lockedSeed: 0,
    zakatContributed: +(grossSeed * THRESHOLDS.ZAKAT_RATE).toFixed(4),
    totalSeed: ns.seed,
    totalBloom: ns.bloom,
    circulating: ns.seed,
    supplyCapUtilization: ns.seed / THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR,
    factors: {
      sovereignty: ns.sovereignty,
      activation: ns.rac / Math.max(ns.vac, 1),
      quality: ns.ihsan,
      compounding: ns.streak / (ns.streak + 5),
      synergy: ns.reflexes > 0 ? 0.5 : 0,
    },
    live: false,
    lastSync: null,
    loading: false,
  };
}

export function useWallet(nodeState: NodeState) {
  const [wallet, setWallet] = useState<WalletState>(() => deriveOffline(nodeState));
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);

  const fetchWallet = useCallback(async () => {
    try {
      const [balance, supply, potential] = await Promise.all([
        api.tokenBalance(),
        api.tokenSupply(),
        api.seedPotential(),
      ]);

      if (!mountedRef.current) return;

      const grossSeed = balance.seed / (1 - THRESHOLDS.ZAKAT_RATE);

      setWallet({
        seed: balance.seed,
        bloom: balance.bloom,
        lockedSeed: balance.locked_seed,
        zakatContributed: +(grossSeed * THRESHOLDS.ZAKAT_RATE).toFixed(4),
        totalSeed: supply.total_seed,
        totalBloom: supply.total_bloom,
        circulating: supply.circulating,
        supplyCapUtilization: supply.total_seed / THRESHOLDS.SEED_SUPPLY_CAP_PER_YEAR,
        factors: potential.factors,
        live: true,
        lastSync: Date.now(),
        loading: false,
      });
    } catch {
      // Offline fallback — derive from local NodeState
      if (!mountedRef.current) return;
      setWallet(prev => ({
        ...deriveOffline(nodeState),
        lastSync: prev.lastSync, // preserve last known sync time
      }));
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [nodeState]);

  // Initial fetch
  useEffect(() => {
    fetchWallet();
  }, [fetchWallet]);

  // Polling — pauses when tab is hidden
  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;

    const startPolling = () => {
      timer = setInterval(fetchWallet, POLL_INTERVAL_MS);
    };

    const handleVisibility = () => {
      clearInterval(timer);
      if (document.visibilityState === 'visible') {
        fetchWallet(); // immediate refresh on return
        startPolling();
      }
    };

    startPolling();
    document.addEventListener('visibilitychange', handleVisibility);

    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', handleVisibility);
    };
  }, [fetchWallet]);

  // Cleanup
  useEffect(() => {
    return () => { mountedRef.current = false; };
  }, []);

  // Keep offline state fresh when nodeState changes and we're not live
  useEffect(() => {
    if (!wallet.live) {
      setWallet(prev => ({
        ...deriveOffline(nodeState),
        lastSync: prev.lastSync,
      }));
    }
  }, [nodeState, wallet.live]);

  return { ...wallet, loading, refresh: fetchWallet };
}
