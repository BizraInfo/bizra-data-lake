import { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_URL ?? '/api';

interface FetchResult<T> {
  data: T;
}

function useFetch<T>(path: string, fallback: T, intervalMs = 30_000): FetchResult<T> {
  const [data, setData] = useState<T>(fallback);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const res = await fetch(`${API_BASE}${path}`);
        if (res.ok && mounted) {
          setData(await res.json());
        }
      } catch {
        // Graceful degradation — retain last known state
      }
    };
    load();
    const id = setInterval(load, intervalMs);
    return () => { mounted = false; clearInterval(id); };
  }, [path, intervalMs]);

  return { data };
}

export function useSovereignHealth() {
  return useFetch('/v1/health', {
    status: 'unknown', uptime_s: 0, version: '0.0.0',
    ihsan_score: 0, snr_score: 0,
  });
}

export function useSeedPotential() {
  return useFetch('/v1/sovereignty/potential', {
    potential: 0, tier: 'SEED', growth_rate: 0,
  });
}

export function useTokenBalance() {
  return useFetch('/v1/token/balance', {
    seed: 0, bloom: 0, staked: 0,
  });
}

export function useConstitutionalStatus() {
  return useFetch('/v1/constitutional/status', {
    ihsan: 0, snr: 0, gini: 0, gates_passed: 0, gates_total: 0,
  });
}

export function useNetworkEffect() {
  return useFetch('/v1/network/effect', {
    node_count: 0, edge_count: 0, density: 0,
  });
}

export function useNetworkMilestones() {
  return useFetch('/v1/network/milestones', [] as Array<{
    id: string; name: string; reached: boolean; timestamp?: string;
  }>);
}

export function useNodeLifecycle() {
  return useFetch('/v1/node/lifecycle', {
    phase: 'genesis', age_hours: 0, transitions: 0,
  });
}

export function useNodeValue() {
  return useFetch('/v1/node/value', {
    value: 0, rank: 0, percentile: 0,
  });
}

export function useSeedEpisodes() {
  return useFetch('/v1/sel/episodes', [] as Array<{
    id: string; action: string; outcome: string; timestamp: string;
  }>);
}

export function useTerminalBriefing() {
  return useFetch('/v1/briefing', {
    summary: '', alerts: [] as string[], recommendations: [] as string[],
  });
}

export function useMemoryStats() {
  return useFetch('/v1/memory/stats', {
    episodic_count: 0, semantic_count: 0, procedural_count: 0,
    total_entries: 0, db_size_mb: 0,
  });
}
