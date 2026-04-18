// ═══════════════════════════════════════════════════════════════
// DEMA — React Query Hooks
// Server-state layer with cache, refetch, and optimistic updates.
// ═══════════════════════════════════════════════════════════════

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { receiptsApi, manifestsApi, resourcesApi, memoryApi, actionsApi, trustApi } from "./client";
import type { Receipt, Manifest, Resource, MemoryEntry, ActionLog } from "@/types";

// ─── Query Key Factory ────────────────────────────────────────

export const queryKeys = {
  receipts: {
    all: ["receipts"] as const,
    filtered: (status?: string) => ["receipts", { status }] as const,
  },
  manifests: {
    all: ["manifests"] as const,
    filtered: (status?: string) => ["manifests", { status }] as const,
  },
  resources: {
    all: ["resources"] as const,
    filtered: (type?: string, status?: string) => ["resources", { type, status }] as const,
  },
  memory: {
    all: ["memory"] as const,
    filtered: (category?: string) => ["memory", { category }] as const,
  },
  actions: {
    all: ["actions"] as const,
    filtered: (mode?: string, status?: string) => ["actions", { mode, status }] as const,
  },
  trust: {
    current: ["trust", "current"] as const,
  },
} as const;

// ─── Receipts ─────────────────────────────────────────────────

export function useReceipts(status?: string) {
  return useQuery({
    queryKey: queryKeys.receipts.filtered(status),
    queryFn: () => receiptsApi.list(status ? { status } : undefined),
    staleTime: 30_000,
  });
}

export function useCreateReceipt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: receiptsApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.receipts.all });
    },
  });
}

// ─── Manifests ────────────────────────────────────────────────

export function useManifests(status?: string) {
  return useQuery({
    queryKey: queryKeys.manifests.filtered(status),
    queryFn: () => manifestsApi.list(status ? { status } : undefined),
    staleTime: 60_000,
  });
}

export function useCreateManifest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: manifestsApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.manifests.all });
    },
  });
}

// ─── Resources ────────────────────────────────────────────────

export function useResources(type?: string, status?: string) {
  return useQuery({
    queryKey: queryKeys.resources.filtered(type, status),
    queryFn: () => resourcesApi.list(
      type || status ? { type: type as "service" | undefined, status: status as "active" | undefined } : undefined,
    ),
    staleTime: 30_000,
  });
}

export function useCreateResource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: resourcesApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.resources.all });
    },
  });
}

export function useRemoveResource() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: resourcesApi.remove,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.resources.all });
    },
  });
}

// ─── Memory ───────────────────────────────────────────────────

export function useMemory(category?: string) {
  return useQuery({
    queryKey: queryKeys.memory.filtered(category),
    queryFn: () => memoryApi.list(category ? { category } : undefined),
    staleTime: 30_000,
  });
}

export function useCreateMemoryEntry() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: memoryApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.memory.all });
    },
  });
}

// ─── Actions ──────────────────────────────────────────────────

export function useActions(mode?: string, status?: string) {
  return useQuery({
    queryKey: queryKeys.actions.filtered(mode, status),
    queryFn: () => actionsApi.list(
      mode || status ? { mode, status } : undefined,
    ),
    staleTime: 15_000,
  });
}

export function useCreateAction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: actionsApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.actions.all });
    },
  });
}

// ─── Trust ────────────────────────────────────────────────────

export function useTrustState() {
  return useQuery({
    queryKey: queryKeys.trust.current,
    queryFn: trustApi.get,
    staleTime: 60_000,
  });
}

export function useUpdateTrust() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: trustApi.update,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.trust.current });
    },
  });
}
