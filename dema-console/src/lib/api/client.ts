// ═══════════════════════════════════════════════════════════════
// DEMA — API Client Layer
// Typed fetch wrapper with standardized error handling.
// ═══════════════════════════════════════════════════════════════

interface API_Response<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
}

class DEMAClientError extends Error {
  code: string;
  status: number;
  details?: unknown;

  constructor(code: string, message: string, status: number, details?: unknown) {
    super(message);
    this.name = "DEMAClientError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

/**
 * Core typed fetch. Centralizes all API communication.
 */
async function apiFetch<T>(
  endpoint: string,
  options?: {
    method?: string;
    body?: unknown;
    params?: Record<string, string>;
  },
): Promise<T> {
  const url = new URL(endpoint, window.location.origin);
  if (options?.params) {
    for (const [k, v] of Object.entries(options.params)) {
      url.searchParams.set(k, v);
    }
  }

  const res = await fetch(url.toString(), {
    method: options?.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
    },
    body: options?.body ? JSON.stringify(options.body) : undefined,
  });

  const json: API_Response<T> = await res.json();

  if (!json.success || !json.data) {
    throw new DEMAClientError(
      json.error?.code ?? "UNKNOWN",
      json.error?.message ?? "Unknown error",
      res.status,
      json.error?.details,
    );
  }

  return json.data;
}

// ─── Receipts ─────────────────────────────────────────────────

export const receiptsApi = {
  list: (filters?: { status?: string; limit?: number }) =>
    apiFetch<unknown[]>("/api/receipts", {
      params: filters ? {
        ...(filters.status ? { status: filters.status } : {}),
        ...(filters.limit ? { limit: String(filters.limit) } : {}),
      } : undefined,
    }),

  create: (data: { missionId?: string; type: string; title: string; description?: string; evidence?: string }) =>
    apiFetch<unknown>("/api/receipts", { method: "POST", body: data }),
};

// ─── Manifests ────────────────────────────────────────────────

export const manifestsApi = {
  list: (filters?: { status?: string }) =>
    apiFetch<unknown[]>("/api/manifests", {
      params: filters
        ? {
            ...(filters.status ? { status: filters.status } : {}),
          }
        : undefined,
    }),

  create: (data: { title: string; description?: string; artifacts?: unknown[] }) =>
    apiFetch<unknown>("/api/manifests", { method: "POST", body: data }),
};

// ─── Resources ────────────────────────────────────────────────

export const resourcesApi = {
  list: (filters?: { type?: string; status?: string }) =>
    apiFetch<unknown[]>("/api/resources", {
      params: filters ? {
        ...(filters.type ? { type: filters.type } : {}),
        ...(filters.status ? { status: filters.status } : {}),
      } : undefined,
    }),

  create: (data: { name: string; type: string; path?: string }) =>
    apiFetch<unknown>("/api/resources", { method: "POST", body: data }),

  remove: (id: string) =>
    apiFetch<{ deleted: boolean }>(`/api/resources?id=${id}`, { method: "DELETE" }),
};

// ─── Memory ───────────────────────────────────────────────────

export const memoryApi = {
  list: (filters?: { category?: string }) =>
    apiFetch<unknown[]>("/api/memory", {
      params: filters
        ? {
            ...(filters.category ? { category: filters.category } : {}),
          }
        : undefined,
    }),

  create: (data: { category: string; title: string; content: string; confidence?: number; relevance?: number; source?: string; tags?: string[] }) =>
    apiFetch<unknown>("/api/memory", { method: "POST", body: data }),
};

// ─── Actions ──────────────────────────────────────────────────

export const actionsApi = {
  list: (filters?: { mode?: string; status?: string }) =>
    apiFetch<unknown[]>("/api/actions", {
      params: filters ? {
        ...(filters.mode ? { mode: filters.mode } : {}),
        ...(filters.status ? { status: filters.status } : {}),
      } : undefined,
    }),

  create: (data: { mode: string; action: string; description?: string; permission?: string }) =>
    apiFetch<unknown>("/api/actions", { method: "POST", body: data }),
};

// ─── Trust ────────────────────────────────────────────────────

export const trustApi = {
  get: () => apiFetch<unknown>("/api/trust"),

  update: (data: { id: string; principalName?: string; level?: string; score?: number }) =>
    apiFetch<unknown>("/api/trust", { method: "PUT", body: data }),
};
