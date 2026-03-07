# Phase 75 — Live Domains Consolidation: bizra.ai + bizra.info

Status: spec-pseudocode
Date: 2026-03-06
Source: `C:\award-winner-design` (Git: BizraInfo/award-winner-design)
Deployment: Vercel (project: award-winner-design, prj_TqizUlg969LHSrngZI5EiEgLMJAn)
Domains: bizra.ai (primary), bizra.info (alias)
Depends: Phase 72 (sovereign API), Phase 73 (shared contracts), Phase 74 (prototype routing)

---

## 1. Current State Audit

### 1.1 Stack

```
Next.js 16 + React 19 + TypeScript
Tailwind CSS v4 (oklch tokens, tw-animate-css)
Zustand (persisted + encrypted via AES-GCM Web Crypto)
Three.js + R3F + drei + postprocessing (3D showcase)
Framer Motion + GSAP + ScrollTrigger (animation)
Chart.js (evidence metrics)
Mermaid (architecture diagrams)
Lucide React (icons)
Jose (JWT), pnpm, Vercel standalone output
```

### 1.2 Live Route Map

```
ROUTE                      COMPONENT                    PURPOSE
/                          LifecycleRouter              State machine entry — routes by phase
/showcase                  ShowcasePage                 3D citadel + pitch deck + demos
/showcase/maestro          MaestroViz                   Agent orchestration dashboard
/showcase/pipeline         PipelineDashboard            Sovereign pipeline scenarios
/genesis                   GenesisPortal                Flower of Life interactive
/atlas                     AtlasPage                    Knowledge graph viewer

API ROUTES:
/api/health                Health probe
/api/csrf-token            CSRF token endpoint
/api/auth/login            JWT login
/api/metrics               Metrics endpoint
/api/ethics                Ethics scoring
/api/knowledge-graph       Graph data for Atlas
/api/scaffold/health       Python bridge health
/api/scaffold/metrics      Python bridge metrics
/api/scaffold/evidence     Evidence endpoint
/api/scaffold/genesis-seal Genesis seal verification
```

### 1.3 Lifecycle State Machine (zustand)

```
Phase 0: FIRST_ENCOUNTER  → LandingPage      7-slide scroll (GSAP)
Phase 1: SEED_TEST        → SeedTest          4-question profile builder
Phase 2: PAT_INTRO        → PATOnboarding     7-agent selection + primary pick
Phase 3: FIRST_SESSION    → FirstSession      7-day plan generation
Phase 4: DAILY_LOOP       → DailyLoop         Check-in + work + memory + reflect
Phase 5: NODE_ACTIVATION  → NodeActivation    CPU/GPU/storage resource sharing
Phase 6: COMMUNITY        → CommunityLayer    Spaces + SAT layer
Phase 7: LEGACY           → DailyLoop (reuse) Impact record + milestones
```

Persistence: `bizra-lifecycle-storage` in encrypted localStorage (AES-GCM 256-bit).
Privacy: `primaryStressor` explicitly excluded from persistence.

### 1.4 Component Inventory (38 files)

```
components/
├── architecture/          3 files: layer-visualizer, system-diagram, tree-viz
├── dashboard/             5 files: community, daily-loop, maestro-viz, node-activation, pipeline
├── demo/                  1 file:  terminal-simulation
├── evidence/              1 file:  metrics-display
├── genesis/               1 file:  genesis-portal (Flower of Life)
├── infrastructure/        3 files: GlobalErrorBoundary, PerformanceObserver, SystemHealth
├── landing/               1 file:  landing-page (7-slide with GSAP)
├── lifecycle/             1 file:  lifecycle-router (state machine)
├── onboarding/            3 files: first-session, pat-onboarding, seed-test
├── pitch-deck/            1 file:  deck-container
├── settings/              1 file:  data-privacy (clear data button)
├── visualizations/        1 file:  GraphViewer
├── 14 standalone files:   hero-section, hero, bizra-logo-animated, citadel,
│                          citadel-optimized, cosmic-background, footer,
│                          genesis-dashboard, genesis-story, glass-interface,
│                          loading-screen, metrics-grid, nav-dock,
│                          sacred-geometry-interface, sacred-geometry,
│                          sovereignty-admission
```

### 1.5 Infrastructure

```
lib/
├── security/       6 files: api-auth, csrf-*, encrypted-storage, token-store
├── core/           3 files: SystemBootstrap, governance/PolicyEngine, telemetry
├── ihsan/          1 file:  scoring-system
├── webgl/          3 files: context-manager, three-optimizer, index
├── scaffold/       3 files: evidence, metrics, paths
├── 18 utility modules: a11y, animation, cache, data-fetching, error-boundary,
│                       events, experiments, feature-flags, graphql, i18n,
│                       observability, performance, pwa, quality, rate-limit,
│                       resilience, sape, state-machine, testing, validation,
│                       virtual-scroll, websocket
```

---

## 2. Domain Architecture

### 2.1 Domain Split

```pseudocode
DOMAIN_MAP:
  bizra.ai
    PURPOSE: Primary product domain
    ROUTES:  / (lifecycle), /showcase, /genesis, /atlas
    TARGET:  Consumer + investor + contributor
    SEO:     Canonical URL for all organic traffic

  bizra.info
    PURPOSE: Informational alias (educational, docs, trust)
    BEHAVIOR: 301 redirect → bizra.ai for all paths
    EXCEPTION: /api/* routes serve directly (no redirect)
    NOTE: Once docs portal is built, bizra.info may host docs subdomain

VERCEL_CONFIG:
  Primary domain:    bizra.ai
  Redirect domain:   bizra.info → bizra.ai (301)
  Environment vars:  NEXT_PUBLIC_APP_URL=https://bizra.ai
```

### 2.2 Middleware for Domain Routing

```pseudocode
MODULE middleware.ts:

  FUNCTION middleware(request):
    host = request.headers.get("host")
    path = request.nextUrl.pathname

    # API routes serve from both domains (no redirect)
    IF path.startsWith("/api/"):
      RETURN NextResponse.next()

    # Redirect bizra.info → bizra.ai
    IF host.includes("bizra.info"):
      target = new URL(path, "https://bizra.ai")
      target.search = request.nextUrl.search
      RETURN NextResponse.redirect(target, 301)

    RETURN NextResponse.next()

  CONFIG matcher = ["/((?!_next|favicon|manifest|robots|sitemap).*)"]
```

---

## 3. Backend API Bridge

### 3.1 Current Gap

The live site has **scaffold API routes** (`/api/scaffold/*`) that proxy to
`SCAFFOLD_API_URL` (Python backend at localhost:8000). This was a placeholder.

The Phase 72 sovereign API runs on port 8081 with typed endpoints.

### 3.2 Wire Plan

```pseudocode
MODULE api_bridge:
  """
  Connect Next.js API routes to the sovereign Python backend.
  Vercel serverless functions proxy to the production API.
  """

  # Option A: Direct proxy (production)
  # Vercel rewrites in vercel.json or next.config.js

  VERCEL_REWRITES:
    /api/v1/:path*  →  https://api.bizra.ai/v1/:path*

  # Option B: Serverless function proxy (staging)
  # Each API route forwards to SOVEREIGN_API_URL

  ENV_VARS (Vercel dashboard):
    SOVEREIGN_API_URL     = "https://api.bizra.ai"   # Production
    SOVEREIGN_API_URL     = "http://localhost:8081"   # Development
    SCAFFOLD_API_URL      = same as SOVEREIGN_API_URL # Migrate scaffold→sovereign

  MIGRATION_STEPS:
    1. Add SOVEREIGN_API_URL to Vercel env vars
    2. Create /api/v1/[...path]/route.ts as proxy
    3. Update scaffold routes to call sovereign endpoints
    4. Update landing page EvidenceSlide to use /api/v1/network/milestones
    5. Update DailyLoop to use /api/v1/seed/potential via useApi hooks
    6. Deprecate scaffold routes once sovereign proxy is stable
```

### 3.3 Proxy Route Handler

```pseudocode
FILE app/api/v1/[...path]/route.ts:

  IMPORT { NextRequest, NextResponse }

  CONST SOVEREIGN_URL = process.env.SOVEREIGN_API_URL || "http://localhost:8081"

  ASYNC FUNCTION handler(request: NextRequest, { params }):
    path = params.path.join("/")
    target = `${SOVEREIGN_URL}/v1/${path}`

    # Forward request with auth header
    headers = new Headers(request.headers)
    headers.delete("host")

    response = AWAIT fetch(target, {
      method: request.method,
      headers: headers,
      body: request.method !== "GET" ? request.body : undefined,
    })

    # Return with CORS headers for same-origin
    RETURN new NextResponse(response.body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("Content-Type"),
        "Cache-Control": "no-store",
      }
    })

  EXPORT { handler as GET, handler as POST }
```

---

## 4. Landing Page → Sovereign API Wiring

### 4.1 EvidenceSlide Enhancement

```pseudocode
MODULE landing/evidence-slide:
  """
  Currently: Static metrics display
  Target: Live metrics from /api/v1/health + /api/v1/network/milestones
  """

  FUNCTION EvidenceSlide():
    [health, setHealth]       = useState(null)
    [milestones, setMilestones] = useState(null)

    useEffect(() => {
      fetch("/api/v1/health").then(r => r.json()).then(setHealth)
      fetch("/api/v1/network/milestones").then(r => r.json()).then(setMilestones)
    }, [])

    RENDER:
      IF health:
        StatusBadge(health.status)        # "healthy" green dot
        UptimeDisplay(health.uptime_seconds)
        SeedEngineStatus(health.seed_engine)
      IF milestones:
        MilestoneTimeline(milestones.milestones)
      ELSE:
        StaticFallback()                  # Current hardcoded values
```

### 4.2 DailyLoop API Integration

```pseudocode
MODULE dashboard/daily-loop-api:
  """
  Currently: All local zustand state
  Target: Hybrid local + API (local for UX, API for sovereignty proof)
  """

  FUNCTION useSovereignSync():
    """
    Sync local lifecycle state with sovereign backend.
    Local-first: zustand is source of truth for UX.
    API call is fire-and-forget for evidence chain.
    """
    seedPotential = usePolling(() => fetch("/api/v1/seed/potential"), 30_000)
    nodeValue     = usePolling(() => fetch("/api/v1/node/value"), 60_000)
    lifecycle     = usePolling(() => fetch("/api/v1/node/lifecycle"), 60_000)

    RETURN { seedPotential, nodeValue, lifecycle }

  INTEGRATION_POINTS in DailyLoop:
    - Streak display:     seedPotential.data?.streak ?? localStreak
    - Sovereignty tier:   seedPotential.data?.tier ?? "SEED"
    - Node value:         nodeValue.data?.composite ?? 0
    - Lifecycle stage:    lifecycle.data?.current_stage ?? "Seed"
    - Progress bar:       lifecycle.data?.progress ?? 0

  FALLBACK: All displays work with local-only data if API unreachable.
```

---

## 5. SEO & Performance

### 5.1 Metadata

```pseudocode
MODULE app/layout.tsx metadata:

  METADATA:
    title:
      template: "%s | BIZRA"
      default: "BIZRA — Your Sovereign AI Team"
    description: "Every human is a node. Every node is a seed. Join the first constitutionally governed AI operating system."
    metadataBase: new URL("https://bizra.ai")
    alternates:
      canonical: "https://bizra.ai"
    openGraph:
      type: "website"
      siteName: "BIZRA"
      images: ["/og-image.png"]  # 1200x630 gold-on-navy
    twitter:
      card: "summary_large_image"
      creator: "@BizraInfo"
    robots:
      index: true
      follow: true
    manifest: "/manifest.json"
```

### 5.2 Performance Budget

```pseudocode
PERFORMANCE_TARGETS:
  LCP (Largest Contentful Paint): < 2.5s
  FID (First Input Delay):       < 100ms
  CLS (Cumulative Layout Shift):  < 0.1
  TTI (Time to Interactive):     < 3.5s

  BUNDLE_BUDGET:
    Initial JS:     < 200KB gzipped
    Three.js chunk: < 400KB gzipped (lazy-loaded on /showcase only)
    GSAP chunk:     < 50KB gzipped (lazy-loaded in landing page)
    Chart.js chunk: < 80KB gzipped (lazy-loaded in evidence slide)

  CURRENT_OPTIMIZATION:
    - Three.js: dynamic import, SSR disabled, Canvas error fallback
    - GSAP: lazy-loaded via loadGsap() singleton promise
    - Chart.js: lazy-loaded via loadChart() singleton promise
    - All dashboard components: dynamic import with loading skeletons
    - Lifecycle phases: AnimatePresence with crossfade transitions

  MISSING:
    - Image optimization (next/image not used consistently)
    - Font subsetting (4 fonts loaded: Inter, Playfair, Amiri, Cinzel)
    - Route-level prefetching for lifecycle transitions
    - Service worker for offline daily-loop capability
```

### 5.3 Accessibility

```pseudocode
ACCESSIBILITY_AUDIT:

  EXISTING (good):
    - lib/a11y/index.ts exists
    - Keyboard-navigable navigation
    - Color contrast adequate on navy bg
    - focus-visible ring styled in globals.css

  GAPS:
    - 3D showcase has no alt text / aria-label
    - Canvas fallback message exists but lacks role="img"
    - SeedTest form inputs lack aria-describedby
    - DailyLoop mood selector needs role="radiogroup"
    - No skip-navigation link
    - No ARIA live regions for phase transitions

  FIX_PRIORITY:
    1. Add skip-nav link in layout.tsx
    2. role="img" + aria-label on Canvas fallback
    3. aria-describedby on SeedTest inputs
    4. role="radiogroup" on mood selector
    5. aria-live="polite" on LifecycleRouter phase container
```

---

## 6. Security Audit

### 6.1 Current Security Posture

```pseudocode
SECURITY_REVIEW:

  STRONG:
    - CSP: Strict production CSP (no unsafe-eval, no unsafe-inline)
    - HSTS: 2-year max-age + includeSubDomains + preload
    - Frame: X-Frame-Options DENY + frame-ancestors 'none'
    - CSRF: Server-side token generation + validation
    - Encryption: AES-GCM 256-bit on zustand persistence
    - Privacy: primaryStressor excluded from persistence
    - Auth: JWT + refresh token flow via Jose library
    - Permissions-Policy: camera/mic/geo all disabled

  ACCEPTABLE:
    - CSP style-src 'self' (no inline styles in prod — Tailwind generates CSS files)
    - connect-src 'self' https: wss: (broad but necessary for API proxy)

  NEEDS_ATTENTION:
    - .env.local exists in repo (gitignored, but verify not committed)
    - JWT_SECRET and REFRESH_SECRET must be rotated in Vercel env
    - CSRF_SECRET must be set in Vercel env (not using example value)
    - No rate limiting on /api/auth/login (lib/rate-limit exists but not wired)
    - /api/scaffold/* routes lack auth — need sovereign API auth guard
```

### 6.2 Security Hardening Steps

```pseudocode
HARDENING_PLAN:
  1. Wire rate limiting on /api/auth/login (use lib/rate-limit/index.ts)
  2. Add auth middleware on /api/scaffold/* routes
  3. Verify .env.local not in git history: `git log --all --diff-filter=A -- .env.local`
  4. Set Vercel env vars: JWT_SECRET, REFRESH_SECRET, CSRF_SECRET (unique per env)
  5. Add /api/v1/[...path] proxy with auth header forwarding
  6. Enable Vercel DDoS protection + WAF rules
  7. Add robots.txt disallow for /api/* paths
```

---

## 7. Vercel Configuration

### 7.1 vercel.json Enhancement

```pseudocode
FILE vercel.json:

  {
    "framework": "nextjs",
    "buildCommand": "next build",
    "installCommand": "pnpm install --frozen-lockfile",
    "outputDirectory": ".next",

    "redirects": [
      {
        "source": "/:path*",
        "has": [{ "type": "host", "value": "bizra.info" }],
        "destination": "https://bizra.ai/:path*",
        "statusCode": 301
      }
    ],

    "rewrites": [
      {
        "source": "/api/v1/:path*",
        "destination": "${SOVEREIGN_API_URL}/v1/:path*"
      }
    ],

    "headers": [
      {
        "source": "/api/:path*",
        "headers": [
          { "key": "X-Robots-Tag", "value": "noindex" }
        ]
      }
    ]
  }
```

### 7.2 Environment Variables (Vercel Dashboard)

```pseudocode
VERCEL_ENV_VARS:

  # All environments
  NEXT_PUBLIC_APP_URL          = "https://bizra.ai"

  # Production only
  JWT_SECRET                   = <generated 256-bit hex>
  REFRESH_SECRET               = <generated 256-bit hex>
  CSRF_SECRET                  = <generated 256-bit hex>
  SOVEREIGN_API_URL            = "https://api.bizra.ai"
  NODE_ENV                     = "production"

  # Preview only
  SOVEREIGN_API_URL            = "https://staging-api.bizra.ai"

  # Development only
  SOVEREIGN_API_URL            = "http://localhost:8081"

  # NEVER set in Vercel (dev-only):
  # SCAFFOLD_API_URL (deprecated — use SOVEREIGN_API_URL)
```

---

## 8. Testing Strategy

### 8.1 Existing Tests

```
tests/
├── unit/           Vitest unit tests
├── e2e/            Playwright E2E tests
├── k6/             k6 load tests (smoke.js, load.js)
└── lighthouse/     Lighthouse CI config
```

### 8.2 TDD Anchors for Phase 75

```pseudocode
TEST_SUITE phase_75:

  # Domain routing
  test_bizra_info_redirects_to_bizra_ai:
    REQUEST GET bizra.info/showcase
    ASSERT 301 redirect → bizra.ai/showcase

  test_api_routes_no_redirect_on_bizra_info:
    REQUEST GET bizra.info/api/health
    ASSERT 200 (no redirect)

  # Sovereign API proxy
  test_api_v1_proxy_forwards_to_sovereign:
    MOCK SOVEREIGN_API_URL response
    REQUEST GET /api/v1/health
    ASSERT response matches sovereign API shape

  test_api_v1_proxy_forwards_auth_header:
    REQUEST GET /api/v1/seed/potential
    WITH Authorization: Bearer <token>
    ASSERT sovereign API receives same Bearer token

  test_api_v1_proxy_handles_sovereign_down:
    MOCK SOVEREIGN_API_URL → timeout
    REQUEST GET /api/v1/health
    ASSERT 502 or 503 with JSON error body

  # Lifecycle routing
  test_fresh_visitor_sees_landing_page:
    CLEAR localStorage
    NAVIGATE /
    ASSERT LandingPage renders (has "Genesis" slide)

  test_seed_test_stores_profile_encrypted:
    COMPLETE seed test flow
    READ localStorage "bizra-lifecycle-storage"
    ASSERT value is NOT plaintext JSON
    ASSERT primaryStressor is null in decrypted state

  test_daily_loop_api_fallback:
    MOCK /api/v1/seed/potential → 503
    NAVIGATE / (as DAILY_LOOP user)
    ASSERT DailyLoop renders with local streak data
    ASSERT no error shown to user

  # SEO
  test_canonical_url_is_bizra_ai:
    RENDER /
    ASSERT <link rel="canonical" href="https://bizra.ai/">

  test_og_image_exists:
    REQUEST GET /og-image.png
    ASSERT 200
    ASSERT content-type image/png

  # Security
  test_csp_header_present:
    REQUEST GET /
    ASSERT Content-Security-Policy header present
    ASSERT NOT contains "unsafe-eval"
    ASSERT NOT contains "unsafe-inline"

  test_login_rate_limited:
    FOR i IN 1..20:
      POST /api/auth/login { bad credentials }
    ASSERT response 429 after threshold

  # Performance
  test_initial_bundle_under_200kb:
    BUILD next build
    MEASURE .next/static chunks
    ASSERT total initial JS < 200KB gzipped

  test_three_js_not_in_initial_bundle:
    ANALYZE webpack chunks
    ASSERT "three" only in dynamic chunk
    ASSERT initial page load does NOT include three.js
```

---

## 9. Implementation Phases

```
Phase 75.01 — Domain + Middleware
  - Add middleware.ts for bizra.info → bizra.ai redirect
  - Update vercel.json with redirects + rewrites
  - Set Vercel env vars (SOVEREIGN_API_URL, secrets)
  - Add robots.txt with /api/* disallow
  - Est: 1 session

Phase 75.02 — Sovereign API Proxy
  - Create /api/v1/[...path]/route.ts
  - Wire auth header forwarding
  - Add error handling for sovereign-down
  - Migrate /api/scaffold/* to sovereign proxy
  - Est: 1 session

Phase 75.03 — Landing Page API Wiring
  - EvidenceSlide: live health + milestones
  - GenesisPortal: live seed engine status
  - DailyLoop: sovereignty score, node value, lifecycle stage
  - All with graceful fallback to static/local data
  - Est: 1 session

Phase 75.04 — SEO + Accessibility
  - Metadata enhancement (og:image, canonical, twitter card)
  - Skip-nav link, ARIA live regions, form accessibility
  - Sitemap generation (next-sitemap)
  - Est: 1 session

Phase 75.05 — Security Hardening
  - Wire rate limiting on /api/auth/login
  - Auth middleware on proxy routes
  - Verify no secrets in git history
  - Rotate JWT/CSRF secrets in Vercel
  - Est: 1 session

Phase 75.06 — Performance Optimization
  - Font subsetting (Inter Latin only, Amiri Arabic only)
  - next/image for all static images
  - Route prefetching for lifecycle transitions
  - Service worker stub for offline daily-loop
  - Est: 1 session
```

---

## 10. Decision Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | bizra.ai is canonical, bizra.info 301 redirects | Single canonical avoids duplicate content SEO penalty |
| D2 | API routes serve from both domains | API clients may be configured with either domain |
| D3 | Sovereign API proxy via Next.js rewrites | Avoids CORS, keeps single domain for cookies |
| D4 | Local-first with API overlay | Lifecycle state is instant (zustand), API enriches with proof |
| D5 | Keep Three.js showcase at /showcase | Heavy 3D bundle must never load on / for mobile users |
| D6 | Encrypted persistence stays | AES-GCM is already working, protects user profile data |
| D7 | pnpm stays as package manager | Vercel config already uses pnpm, lockfile is committed |
| D8 | No domain split for now (no docs.bizra.info) | Premature — docs portal not yet built |
| D9 | primaryStressor never persisted | Sensitive mental health data — constitutional privacy |
