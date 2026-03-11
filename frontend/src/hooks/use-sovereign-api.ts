/**
 * Sovereign API Hooks — Data-fetching hooks for terminal components.
 *
 * Each hook returns { data, loading, error } following the SWR pattern.
 * Falls back to null when the backend is unreachable.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// ─── Generic Fetcher ────────────────────────────────────────────

interface HookResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

function useFetch<T>(path: string): HookResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`${res.status}`);
      const json = (await res.json()) as T;
      if (mountedRef.current) {
        setData(json);
        setError(null);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err : new Error(String(err)));
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    fetchData();
    return () => { mountedRef.current = false; };
  }, [fetchData]);

  return { data, loading, error };
}

// ─── Memory Hooks ───────────────────────────────────────────────

interface MemoryStats {
  total_episodes: number;
  compiled_reflexes: number;
  near_compile: number;
  last_sync: string | null;
  episodic: number;
  semantic: number;
  procedural: number;
}

export function useMemoryStats(): HookResult<MemoryStats> {
  return useFetch<MemoryStats>('/v1/memory/stats');
}

interface TerminalBriefing {
  summary: string;
  streak: number;
  quality_trend: number;
  suggestions: string[];
}

export function useTerminalBriefing(): HookResult<TerminalBriefing> {
  return useFetch<TerminalBriefing>('/v1/memory/briefing');
}

// ─── Seed & Episode Hooks ───────────────────────────────────────

interface SeedEpisodes {
  episodes: Record<string, unknown>[];
  total: number;
}

export function useSeedEpisodes(): HookResult<SeedEpisodes> {
  return useFetch<SeedEpisodes>('/v1/sel/episodes');
}

interface SeedPotential {
  potential: number;
  sovereignty_score: number;
  tier: string;
  episodes_total: number;
  factors: {
    sovereignty: number;
    activation: number;
    quality: number;
    compounding: number;
    synergy: number;
  };
}

export function useSeedPotential(): HookResult<SeedPotential> {
  return useFetch<SeedPotential>('/v1/token/potential');
}

// ─── Node & Network Hooks ───────────────────────────────────────

interface NodeValue {
  value: number;
  factors: { name: string; score: number; weight: number }[];
  composite: number;
}

export function useNodeValue(): HookResult<NodeValue> {
  return useFetch<NodeValue>('/v1/node/value');
}

interface NodeLifecycle {
  current_stage: string;
  sovereignty_score: number;
  rank: number;
  next_stage: string | null;
  progress: number;
}

export function useNodeLifecycle(): HookResult<NodeLifecycle> {
  return useFetch<NodeLifecycle>('/v1/node/lifecycle');
}

interface NetworkEffect {
  diffusion_eligible: boolean;
  reflex_library_size: number;
  shared_reflexes: number;
  network_score: number;
}

export function useNetworkEffect(): HookResult<NetworkEffect> {
  return useFetch<NetworkEffect>('/v1/network/effect');
}

interface NetworkMilestones {
  projections: {
    label: string;
    nodes: number;
    projected_latency_ms: number;
    pool_memory_tb: number;
    reflex_library_size: number;
  }[];
}

export function useNetworkMilestones(): HookResult<NetworkMilestones> {
  return useFetch<NetworkMilestones>('/v1/network/milestones');
}

// ─── Health & Balance Hooks ─────────────────────────────────────

interface SovereignHealth {
  status: string;
  version: string;
  uptime_seconds: number;
  env: string;
}

export function useSovereignHealth(): HookResult<SovereignHealth> {
  return useFetch<SovereignHealth>('/v1/health');
}

interface TokenBalance {
  balance: number;
  bloom: number;
  locked: number;
}

export function useTokenBalance(): HookResult<TokenBalance> {
  return useFetch<TokenBalance>('/v1/token/balance');
}

// ─── Constitutional Status ──────────────────────────────────────

interface ConstitutionalStatus {
  invariants_passing: number;
  total_invariants: number;
  last_check: string;
  violations: string[];
}

export function useConstitutionalStatus(): HookResult<ConstitutionalStatus> {
  return useFetch<ConstitutionalStatus>('/v1/constitutional/status');
}
