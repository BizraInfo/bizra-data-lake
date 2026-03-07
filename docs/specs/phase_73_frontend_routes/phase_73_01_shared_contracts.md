# Phase 73.01: Shared Contracts — Types, API Client, Design Tokens

**Target directory:** `src/shared/`

## Purpose

Extract the shared foundation that all three surfaces (website, onboarding,
dashboard) consume. No UI component in the app invents its own data shape,
color value, or API call pattern.

## 1. TypeScript API Types

```pseudocode
# src/shared/types/api.ts

# ─────────────────────────────────────────────────────────────
# Health (GET /v1/health)
# ─────────────────────────────────────────────────────────────
TYPE HealthResponse = {
    status: "healthy" | "degraded" | "unhealthy"
    version: string
    uptime_seconds: number
    seed_engine: {
        active: boolean
        episodes: number
        tier: "SEED" | "SPROUT" | "TREE" | "FOREST"
        compiled: boolean
        streak: number
    }
    node_value?: {  # Phase 72
        engine: "node_value"
        source: "seed_engine"
        genesis: string
        has_federation: boolean
    }
}

# ─────────────────────────────────────────────────────────────
# Seed Potential (GET /v1/seed/potential)
# ─────────────────────────────────────────────────────────────
TYPE SeedPotentialResponse = {
    sovereignty_score: number      # 0-1
    tier: "SEED" | "SPROUT" | "TREE" | "FOREST"
    tier_progress: number          # 0-1 within current tier
    episodes_total: number
    episodes_qualified: number
    qualification_rate: number     # 0-1
    reward_ema: number             # 0-1
    streak: number
    compiled: boolean
    converged: boolean
    chain_valid: boolean
    potential_unlocked: number     # 0-1
    potential_remaining: number    # 0-1
    weakest_dimension: string | null
    growth_velocity: number
    last_receipt_hash: string
}

# ─────────────────────────────────────────────────────────────
# Node Value (GET /v1/node/value) — Phase 72
# ─────────────────────────────────────────────────────────────
TYPE NodeValueResponse = {
    potential: number       # 0-1
    activation: number      # 0-1
    quality: number         # 0-1
    compounding: number     # 0-1
    synergy: number         # 0-1
    composite: number       # 0-1, geometric mean
    tier: string
    human_stage: string     # Seed..Catalyst
    timestamp: string       # ISO 8601
}

# ─────────────────────────────────────────────────────────────
# Lifecycle (GET /v1/node/lifecycle) — Phase 72
# ─────────────────────────────────────────────────────────────
TYPE LifecycleResponse = {
    current_stage: string
    rank: number            # 0-6
    progress: number        # 0-1
    sovereignty_score: number
    next_stage: string | null
    next_threshold: number | null
    points_to_next: number
    description: string
    unlock_condition: string
}

# ─────────────────────────────────────────────────────────────
# Onboarding State (GET/PUT /v1/onboarding/state)
# ─────────────────────────────────────────────────────────────
TYPE OnboardingState = {
    step: number            # 0-5
    track: "consumer" | "contributor"
    data: {
        install_verified: boolean
        provider: "lm_studio" | "ollama" | "cloud" | null
        model: string
        teach_data: Record<string, string>
        traits: string[]
        pat_agents_seen: boolean
        first_chat_complete: boolean
        sovereignty_score: number
    }
    last_checkpoint: string  # ISO 8601
}

# ─────────────────────────────────────────────────────────────
# Wallet Summary (GET /v1/wallet/summary)
# ─────────────────────────────────────────────────────────────
TYPE WalletSummaryResponse = {
    seed_balance: number
    bloom_balance: number
    impt_score: number
    zakat_contributed: number
    last_mint: string | null
}

# ─────────────────────────────────────────────────────────────
# Agent Roster (GET /v1/agents/roster)
# ─────────────────────────────────────────────────────────────
TYPE AgentStatus = {
    agent_id: string
    role: string
    type: "PAT" | "SAT"
    status: "idle" | "active" | "learning" | "error"
    last_active: string
}

TYPE AgentRosterResponse = {
    agents: AgentStatus[]
    pat_count: number       # Always 7
    sat_count: number       # Always 5
}

# ─────────────────────────────────────────────────────────────
# Mission (GET /v1/missions/pending, /v1/missions/history)
# ─────────────────────────────────────────────────────────────
TYPE Mission = {
    mission_id: string
    status: "pending" | "running" | "complete" | "partial" | "failed"
    description: string
    ihsan_score: number | null
    snr_score: number | null
    created_at: string
    completed_at: string | null
}
```

## 2. API Client

```pseudocode
# src/shared/api/client.ts

IMPORT types FROM ./types/api

# Base URL from environment — NEVER hardcoded
CONST API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8081"

CLASS BizraClient:
    """Typed API client. All calls go through here.

    Auth: Bearer token from localStorage.
    Error handling: throws typed errors, never swallows.
    """

    CONSTRUCTOR():
        self._base = API_BASE

    FUNCTION _headers() -> Record<string, string>:
        token = localStorage.getItem("bizra_api_token")
        headers = { "Content-Type": "application/json" }
        IF token:
            headers["Authorization"] = "Bearer " + token
        RETURN headers

    FUNCTION _fetch<T>(path: string, options?: RequestInit) -> T:
        response = await fetch(self._base + path, {
            ...options,
            headers: { ...self._headers(), ...options?.headers }
        })
        IF NOT response.ok:
            error = await response.json().catch(() => ({ error: response.statusText }))
            THROW new ApiError(response.status, error.error || "Unknown error")
        RETURN response.json()

    # ── Typed endpoints ──────────────────────────────────
    FUNCTION health() -> HealthResponse:
        RETURN self._fetch("/v1/health")

    FUNCTION seedPotential() -> SeedPotentialResponse:
        RETURN self._fetch("/v1/seed/potential")

    FUNCTION nodeValue() -> NodeValueResponse:
        RETURN self._fetch("/v1/node/value")

    FUNCTION lifecycle() -> LifecycleResponse:
        RETURN self._fetch("/v1/node/lifecycle")

    FUNCTION onboardingState() -> OnboardingState:
        RETURN self._fetch("/v1/onboarding/state")

    FUNCTION saveOnboardingState(state: OnboardingState) -> void:
        RETURN self._fetch("/v1/onboarding/state", {
            method: "PUT",
            body: JSON.stringify(state)
        })

    FUNCTION teach(payload: { kind: string, content: string, confidence: number }) -> any:
        RETURN self._fetch("/v1/onboarding/teach", {
            method: "POST",
            body: JSON.stringify(payload)
        })

    FUNCTION walletSummary() -> WalletSummaryResponse:
        RETURN self._fetch("/v1/wallet/summary")

    FUNCTION agentRoster() -> AgentRosterResponse:
        RETURN self._fetch("/v1/agents/roster")

    FUNCTION pendingMissions() -> Mission[]:
        RETURN self._fetch("/v1/missions/pending")

    FUNCTION missionHistory(limit: number = 20) -> Mission[]:
        RETURN self._fetch("/v1/missions/history?limit=" + limit)

    # ── Auth endpoints ─────────────────────────────────
    FUNCTION login(username: string, password: string) -> AuthTokens:
        RETURN self._fetch("/v1/auth/login", {
            method: "POST",
            body: JSON.stringify({ username, password })
        })

    FUNCTION register(username: string, email: string, password: string,
                      accept_covenant: boolean) -> AuthTokens:
        RETURN self._fetch("/v1/auth/register", {
            method: "POST",
            body: JSON.stringify({ username, email, password, accept_covenant })
        })

    FUNCTION refreshToken(refresh_token: string) -> AuthTokens:
        RETURN self._fetch("/v1/auth/refresh", {
            method: "POST",
            body: JSON.stringify({ refresh_token })
        })

    FUNCTION me() -> UserProfile:
        RETURN self._fetch("/v1/auth/me")

CLASS ApiError EXTENDS Error:
    status: number
    CONSTRUCTOR(status: number, message: string):
        super(message)
        self.status = status

# ── Auth Types ────────────────────────────────────────
TYPE AuthTokens = {
    access_token: string
    refresh_token: string
    token_type: "Bearer"
    expires_in: number
    user_id?: string
    username?: string
}

TYPE UserProfile = {
    user_id: string
    username: string
    email: string
    namespace: string
    status: string
    created_at: string
    query_count: number
}

# Singleton export
EXPORT const api = new BizraClient()
```

## 3. Design Tokens

```pseudocode
# src/shared/tokens.css
# Extracted from WEBSITE_PLAN.md + existing prototypes

:root {
    /* Colors — navy + gold brand */
    --color-bg-primary: #0a0e1a;        /* Deep space navy */
    --color-bg-secondary: #111827;       /* Card background */
    --color-bg-surface: #1f2937;         /* Elevated surface */
    --color-accent-gold: #d4a853;        /* Primary accent */
    --color-accent-gold-dim: #b8943f;    /* Gold hover */
    --color-text-primary: #f9fafb;       /* White text */
    --color-text-secondary: #9ca3af;     /* Gray text */
    --color-text-muted: #6b7280;         /* Muted text */

    /* State colors — restrained */
    --color-success: #10b981;            /* Green */
    --color-warning: #f59e0b;            /* Amber */
    --color-error: #ef4444;              /* Red */
    --color-info: #3b82f6;               /* Blue */

    /* Typography */
    --font-ui: 'Inter', system-ui, sans-serif;
    --font-arabic: 'Noto Sans Arabic', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

    /* Spacing scale (4px base) */
    --space-1: 0.25rem;
    --space-2: 0.5rem;
    --space-3: 0.75rem;
    --space-4: 1rem;
    --space-6: 1.5rem;
    --space-8: 2rem;
    --space-12: 3rem;
    --space-16: 4rem;

    /* Border radius */
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-full: 9999px;

    /* Shadows */
    --shadow-card: 0 1px 3px rgba(0, 0, 0, 0.3);
    --shadow-elevated: 0 4px 12px rgba(0, 0, 0, 0.4);

    /* Touch targets */
    --touch-min: 44px;                   /* WCAG 2.1 AA */

    /* Breakpoints (reference, use Tailwind classes) */
    /* sm: 640px, md: 768px, lg: 1024px, xl: 1280px */
}

/* Operator console overrides */
[data-surface="operator"] {
    --color-bg-primary: #030712;
    --font-ui: var(--font-mono);
    /* Denser spacing for control-room */
    --space-4: 0.75rem;
}
```

## 4. React Query Hooks

```pseudocode
# src/shared/hooks/useApi.ts

IMPORT { useQuery } FROM "@tanstack/react-query"
IMPORT { api } FROM "../api/client"

FUNCTION useHealth():
    RETURN useQuery({
        queryKey: ["health"],
        queryFn: () => api.health(),
        refetchInterval: 10_000,        # Poll every 10s
    })

FUNCTION useSeedPotential():
    RETURN useQuery({
        queryKey: ["seed-potential"],
        queryFn: () => api.seedPotential(),
        refetchInterval: 30_000,        # Poll every 30s
    })

FUNCTION useNodeValue():
    RETURN useQuery({
        queryKey: ["node-value"],
        queryFn: () => api.nodeValue(),
        refetchInterval: 60_000,        # Poll every 60s
    })

FUNCTION useLifecycle():
    RETURN useQuery({
        queryKey: ["lifecycle"],
        queryFn: () => api.lifecycle(),
        refetchInterval: 60_000,
    })

FUNCTION useAgentRoster():
    RETURN useQuery({
        queryKey: ["agent-roster"],
        queryFn: () => api.agentRoster(),
        refetchInterval: 15_000,
    })

FUNCTION useWallet():
    RETURN useQuery({
        queryKey: ["wallet"],
        queryFn: () => api.walletSummary(),
        refetchInterval: 60_000,
    })

FUNCTION usePendingMissions():
    RETURN useQuery({
        queryKey: ["missions-pending"],
        queryFn: () => api.pendingMissions(),
        refetchInterval: 15_000,
    })
```

## TDD Anchors

```pseudocode
TEST "API types match backend response shape":
    # Validate TypeScript types against actual /v1/health JSON schema
    response = await api.health()
    ASSERT "status" IN response
    ASSERT response.status IN ["healthy", "degraded", "unhealthy"]

TEST "API client uses environment base URL":
    ASSERT BizraClient._base == import.meta.env.VITE_API_BASE

TEST "API client adds auth header when token exists":
    localStorage.setItem("bizra_api_token", "test-token")
    headers = new BizraClient()._headers()
    ASSERT headers["Authorization"] == "Bearer test-token"
    localStorage.removeItem("bizra_api_token")

TEST "API client throws ApiError on 401":
    # Mock fetch to return 401
    EXPECT_THROWS(ApiError, api.nodeValue)

TEST "design tokens are valid CSS custom properties":
    styles = getComputedStyle(document.documentElement)
    ASSERT styles.getPropertyValue("--color-accent-gold") != ""
    ASSERT styles.getPropertyValue("--font-ui") CONTAINS "Inter"
    ASSERT styles.getPropertyValue("--touch-min") == "44px"

TEST "operator surface overrides apply":
    container = render(<div data-surface="operator" />)
    styles = getComputedStyle(container)
    ASSERT styles.getPropertyValue("--font-ui") CONTAINS "monospace"

TEST "React Query hooks return loading state initially":
    result = renderHook(() => useHealth())
    ASSERT result.current.isLoading == true

TEST "all hooks have refetch intervals":
    # Ensure no endpoint is left polling at 0 or infinity
    FOR hook IN [useHealth, useSeedPotential, useNodeValue, useAgentRoster]:
        ASSERT hook.options.refetchInterval > 0
        ASSERT hook.options.refetchInterval <= 60_000
```
