/**
 * useWallet v2 — Race-condition hardened wallet hook.
 *
 * Fixes identified in post-sprint review:
 * 1. WebSocket receipt arriving during polling → stale overwrite
 * 2. Offline fallback overwriting fresh live state
 * 3. Visibility-change refresh colliding with manual refresh
 * 4. Partial backend failures producing mixed-state UI
 *
 * Solution: monotonic version counter. Every fetch increments version.
 * Only the highest-version result is accepted. Stale responses are dropped.
 *
 * Standing on Giants:
 *   Lamport (1978) — logical clocks for ordering concurrent events
 *   Nakamoto (2008) — longest chain wins (highest version wins)
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import { THRESHOLDS } from '../tokens';
import type { NodeState } from '../types';

export interface WalletState {
  seed: number;
  bloom: number;
  lockedSeed: number;
  zakatContributed: number;
  totalSeed: number;
  totalBloom: number;
  circulating: number;
  supplyCapUtilization: number;
  factors: {
    sovereignty: number;
    activation: number;
    quality: number;
    compounding: number;
    synergy: number;
  };
  live: boolean;
  lastSync: number | null;
  loading: boolean;
  /** Monotonic version — prevents stale overwrites */
  version: number;
  /** Fetch status for partial failure detection */
  fetchStatus: {
    balance: 'ok' | 'error' | 'pending';
    supply: 'ok' | 'error' | 'pending';
    potential: 'ok' | 'error' | 'pending';
  };
}

const POLL_INTERVAL_MS = 30_000;

function deriveOffline(ns: NodeState, version: number): WalletState {
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
    version,
    fetchStatus: { balance: 'error', supply: 'error', potential: 'error' },
  };
}

export function useWallet(nodeState: NodeState) {
  const [wallet, setWallet] = useState<WalletState>(() => deriveOffline(nodeState, 0));
  const [loading, setLoading] = useState(true);
  const mountedRef = useRef(true);

  /**
   * Monotonic version counter — Lamport clock for fetch ordering.
   * 
   * Every fetchWallet call increments this. When the response arrives,
   * it checks: is my version >= the current wallet version?
   * If not, a newer fetch already landed — discard this result.
   *
   * This prevents:
   * - Slow poll response overwriting fast WebSocket-triggered refresh
   * - Visibility-change refresh racing with manual refresh
   * - Offline fallback overwriting a live result that arrived between
   *   the fetch start and the fallback trigger
   */
  const versionRef = useRef(0);

  /**
   * In-flight guard — prevents concurrent fetches from stacking.
   * Only one fetch can be in-flight at a time.
   */
  const inFlightRef = useRef(false);

  const fetchWallet = useCallback(async () => {
    // Guard: if a fetch is already in-flight, skip this one.
    // This prevents visibility-change + poll + WebSocket all triggering
    // simultaneous fetches that race each other.
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const myVersion = ++versionRef.current;

    const fetchStatus: WalletState['fetchStatus'] = {
      balance: 'pending',
      supply: 'pending',
      potential: 'pending',
    };

    try {
      // Constitutional fail-closed: ALL endpoints must succeed for live state.
      // Any partial failure → offline fallback (Saltzer & Schroeder: default-deny).
      const [balance, supply, potential] = await Promise.all([
        api.tokenBalance(),
        api.tokenSupply(),
        api.seedPotential(),
      ]);

      if (!mountedRef.current) return;

      // Version check: has a newer fetch already landed?
      if (myVersion < versionRef.current) {
        return;
      }

      fetchStatus.balance = 'ok';
      fetchStatus.supply = 'ok';
      fetchStatus.potential = 'ok';

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
        version: myVersion,
        fetchStatus,
      });
    } catch {
      // Any failure → offline fallback, preserving lastSync from prior live state
      if (mountedRef.current && myVersion >= versionRef.current) {
        setWallet(prev => ({
          ...deriveOffline(nodeState, myVersion),
          lastSync: prev.lastSync,
          fetchStatus,
        }));
      }
    } finally {
      inFlightRef.current = false;
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
        fetchWallet();
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

  // Keep offline state fresh when nodeState changes and we're NOT live
  useEffect(() => {
    if (!wallet.live) {
      setWallet(prev => ({
        ...deriveOffline(nodeState, prev.version),
        lastSync: prev.lastSync,
      }));
    }
  }, [nodeState, wallet.live]);

  return { ...wallet, loading, refresh: fetchWallet };
}
