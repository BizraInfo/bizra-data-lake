# Phase 40: filedfs Frontend Rebuild Specification

> Standing on Giants: Abramov (React hooks, 2018) · You (Vite, 2020) · Nicoll (Tauri, 2022) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

`filedfs/` is the BIZRA Node0 desktop frontend — a Tauri + React app serving as the
sovereign AI dashboard. Current state (Alpha-100 MVP):

| Dimension | Current | Target |
|-----------|---------|--------|
| Language | JavaScript (JSX) | TypeScript strict |
| Tests | 0 | 80%+ coverage |
| Build CI | Broken (npm ci + no lock file) | Green in CI |
| Components | 9 monolithic files (82KB largest) | Modular, <300 LOC each |
| State Mgmt | Scattered useState hooks | Centralized store |
| Bridge Layer | 4 overlapping bridges | Unified BizraClient |
| Offline | Basic SW + IndexedDB | Robust queue + retry |
| a11y / i18n | None | WCAG 2.1 AA stubs |

### Root Causes of CI Failure
1. `package-lock.json` exists locally (63KB) but is gitignored
2. CI used `npm ci` which strictly requires lock file
3. `actions/setup-node` cache failed resolving `filedfs/package-lock.json`

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────┐
│                   BIZRA Node0 Desktop               │
│           Tauri 2.0 (Rust shell + webview)          │
├────────────────────────────────────────────────────┤
│  React 18 + TypeScript Strict                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Pages    │ │ Features │ │ Shared   │           │
│  │ Dashboard│ │ Onboard  │ │ Hooks    │           │
│  │ Status   │ │ Chat     │ │ Store    │           │
│  │ Inventory│ │ Settings │ │ Theme    │           │
│  └──────────┘ └──────────┘ └──────────┘           │
│           ↓           ↓           ↓                │
│  ┌────────────────────────────────────────┐        │
│  │       BizraClient (unified bridge)     │        │
│  │  Tauri invoke | WebSocket | HTTP       │        │
│  └────────────────────────────────────────┘        │
│           ↓                                        │
│  ┌────────────────────────────────────────┐        │
│  │      Offline Queue (IndexedDB)         │        │
│  │  SW: cache-first static, net-first API │        │
│  └────────────────────────────────────────┘        │
└────────────────────────────────────────────────────┘
```

---

## 3. Phase Breakdown

### Phase 40a: Build Pipeline Fix (P0 — CI Green)
### Phase 40b: TypeScript Migration (P1 — Type Safety)
### Phase 40c: Component Decomposition (P1 — Modularity)
### Phase 40d: Bridge Unification (P2 — Single Client)
### Phase 40e: Test Harness (P2 — Vitest + Testing Library)
### Phase 40f: Offline Hardening (P3 — Robust PWA)

---

## 4. Phase 40a: Build Pipeline Fix

### Goal
Get filedfs building in CI without regressions. Zero code changes.

### Pseudocode

```
PROCEDURE fix_ci_build():
    # Step 1: Track package-lock.json
    IF package-lock.json EXISTS locally:
        REMOVE "package-lock.json" from .gitignore (if present)
        git add filedfs/package-lock.json
    ELSE:
        RUN npm install in filedfs/ to generate fresh lock file
        git add filedfs/package-lock.json

    # Step 2: Restore npm ci in CI
    UPDATE .github/workflows/ci.yml:
        build-frontend job:
            cache-dependency-path: filedfs/package-lock.json
            install command: npm ci          # strict, reproducible
            build command: npx vite build    # same as now

    # Step 3: Add .gitignore for filedfs
    CREATE filedfs/.gitignore:
        node_modules/
        dist/
        *.local

    # Step 4: Remove binaries from tracking
    # These should be build artifacts, not source
    git rm --cached filedfs/bizra-node-binary
    git rm --cached filedfs/libbizra_hooks.so
    ADD to filedfs/.gitignore:
        bizra-node-binary
        libbizra_hooks.so
        *.so
```

### TDD Anchors
```
TEST "npm ci succeeds with tracked lock file"
    GIVEN filedfs/package-lock.json is committed
    WHEN  npm ci runs in clean directory
    THEN  exit code = 0
    AND   node_modules/ contains react, react-dom, vite

TEST "vite build produces dist/"
    GIVEN npm ci has installed dependencies
    WHEN  npx vite build runs
    THEN  dist/index.html exists
    AND   dist/assets/ contains at least 1 .js file
    AND   exit code = 0

TEST "no binaries tracked in git"
    GIVEN filedfs/.gitignore blocks *.so and bizra-node-binary
    WHEN  git ls-files filedfs/ is checked
    THEN  no .so files listed
    AND   no bizra-node-binary listed
```

### Acceptance Criteria
- [ ] `npm ci` in CI passes
- [ ] `npx vite build` produces dist/ with index.html
- [ ] No ELF binaries or .so files tracked in git
- [ ] Build job turns GREEN in GitHub Actions

---

## 5. Phase 40b: TypeScript Migration

### Goal
Convert JSX → TSX with strict mode. Incremental — one file at a time.

### Pseudocode

```
PROCEDURE migrate_to_typescript():
    # Step 1: Add TypeScript tooling
    npm install --save-dev typescript @types/react @types/react-dom

    # Step 2: Create tsconfig.json
    CREATE filedfs/tsconfig.json:
        compilerOptions:
            strict: true
            target: "ES2022"
            module: "ESNext"
            moduleResolution: "bundler"
            jsx: "react-jsx"
            noEmit: true               # Vite handles transpilation
            skipLibCheck: true
            esModuleInterop: true
            isolatedModules: true
            paths:
                "@/*": ["./src/*"]      # alias for imports

    # Step 3: Update vite.config
    UPDATE vite.config.js → vite.config.ts:
        ADD resolve.alias: { "@": path.resolve(__dirname, "src") }

    # Step 4: Create src/ directory structure
    MOVE files into organized structure:
        filedfs/src/
        ├── app/
        │   └── App.tsx              ← App.jsx (82KB → split in 40c)
        ├── pages/
        │   ├── Dashboard.tsx        ← bizra-dashboard.jsx
        │   ├── Status.tsx           ← bizra-status.jsx
        │   ├── Inventory.tsx        ← bizra-inventory.jsx
        │   ├── Architecture.tsx     ← architecture.jsx
        │   └── Node0Dashboard.tsx   ← node0-dashboard.jsx
        ├── features/
        │   ├── onboarding/          ← onboarding/
        │   └── landing/
        │       └── LandingDemo.tsx  ← LandingDemo.jsx
        ├── hooks/
        │   ├── useNode.ts           ← useNode.js
        │   └── useBizraNode.ts      ← useBizraNode.js
        ├── lib/
        │   └── client.ts            ← bridge unification (40d)
        ├── types/
        │   └── index.ts             # shared type definitions
        └── main.tsx                  ← main.jsx

    # Step 5: Migrate files incrementally
    FOR EACH file IN priority_order:
        RENAME .jsx → .tsx (or .js → .ts)
        ADD type annotations to function signatures
        ADD interface definitions for props
        ADD type guards for runtime data
        VERIFY: npx tsc --noEmit passes
        COMMIT: "chore(filedfs): migrate {filename} to TypeScript"

    # Migration priority (by dependency order):
    #   1. types/index.ts (shared types)
    #   2. hooks/useNode.ts, useBizraNode.ts
    #   3. main.tsx
    #   4. pages/ (one at a time)
    #   5. features/onboarding/
    #   6. app/App.tsx (last — largest, most deps)
```

### Key Type Definitions

```typescript
// filedfs/src/types/index.ts

/** Constitutional quality score (Ihsan covenant) */
interface QualityGauge {
    label: string;
    score: number;        // 0.0–1.0
    threshold: number;    // minimum acceptable (0.95 prod)
    color: string;        // hex color for gauge segment
}

/** Node0 backend state from Tauri invoke */
interface NodeState {
    status: "booting" | "ready" | "degraded" | "offline";
    agents: AgentInfo[];
    ihsan_score: number;
    snr_score: number;
    sovereignty_tier: "SEED" | "SPROUT" | "TREE" | "FOREST";
    model_loaded: string | null;
    uptime_ms: number;
}

/** Agent information from Rust backend */
interface AgentInfo {
    id: string;
    role: "personal" | "system";
    name: string;
    status: "idle" | "active" | "error";
    last_active_ms: number;
}

/** Bridge transport mode */
type TransportMode = "tauri" | "websocket" | "http" | "simulated";

/** Onboarding wizard step */
type OnboardingStep =
    | "verify"
    | "provider"
    | "teach"
    | "first_chat"
    | "dashboard";
```

### TDD Anchors
```
TEST "tsconfig strict mode catches type errors"
    GIVEN tsconfig.json has strict: true
    WHEN  npx tsc --noEmit runs on src/
    THEN  exit code = 0 (no type errors)

TEST "vite build works with TypeScript"
    GIVEN all files migrated to .tsx/.ts
    WHEN  npx vite build runs
    THEN  dist/ produced successfully
    AND   no TypeScript files in dist/ (only JS)

TEST "path aliases resolve correctly"
    GIVEN import from "@/hooks/useNode"
    WHEN  vite resolves the import
    THEN  maps to filedfs/src/hooks/useNode.ts
```

### Acceptance Criteria
- [ ] `tsconfig.json` with strict: true
- [ ] All 40 files renamed and type-annotated
- [ ] `npx tsc --noEmit` exits 0
- [ ] `npx vite build` still produces working dist/
- [ ] CI type-check step added (runs tsc --noEmit)

---

## 6. Phase 40c: Component Decomposition

### Goal
Break monolithic components (82KB App.jsx) into focused modules <300 LOC each.

### Pseudocode

```
PROCEDURE decompose_app():
    # App.jsx (82KB, ~2,400 LOC) → split into:

    # 1. App.tsx — thin shell (router + layout)
    #    MAX 100 LOC
    CREATE src/app/App.tsx:
        IMPORT Router, Layout, ThemeProvider
        RENDER:
            <ThemeProvider theme={bizraTheme}>
                <Layout>
                    <Router>
                        IF !onboarded: <OnboardingFlow />
                        ELSE IF route == "/": <Dashboard />
                        ELSE IF route == "/status": <Status />
                        ELSE IF route == "/inventory": <Inventory />
                        ELSE IF route == "/architecture": <Architecture />
                        ELSE: <NotFound />
                    </Router>
                </Layout>
            </ThemeProvider>

    # 2. Extract SeedOfLife SVG component
    CREATE src/components/SeedOfLife.tsx:
        EXPORT SeedOfLife({ size, animate })
        # Sacred geometry branding — standalone SVG component
        # MAX 80 LOC

    # 3. Extract GaugeSegments
    CREATE src/components/QualityGauge.tsx:
        EXPORT QualityGauge({ segments: QualityGauge[] })
        # Ihsan/SNR score visualization
        # MAX 120 LOC

    # 4. Extract AgentGrid
    CREATE src/components/AgentGrid.tsx:
        EXPORT AgentGrid({ agents: AgentInfo[] })
        # PAT/SAT agent status cards
        # MAX 150 LOC

    # 5. Extract ConnectionStatus
    CREATE src/components/ConnectionStatus.tsx:
        EXPORT ConnectionStatus({ transport, latency, backend })
        # Shows Tauri/WS/HTTP connection health
        # MAX 80 LOC

    # Resulting structure:
    src/
    ├── app/
    │   ├── App.tsx           (100 LOC — router shell)
    │   ├── Layout.tsx        (60 LOC — sidebar + header)
    │   └── Router.tsx        (40 LOC — route definitions)
    ├── components/
    │   ├── SeedOfLife.tsx     (80 LOC)
    │   ├── QualityGauge.tsx   (120 LOC)
    │   ├── AgentGrid.tsx      (150 LOC)
    │   └── ConnectionStatus.tsx (80 LOC)
    ├── pages/
    │   ├── Dashboard.tsx      (250 LOC — assembles components)
    │   ├── Status.tsx         (200 LOC)
    │   ├── Inventory.tsx      (200 LOC)
    │   └── Architecture.tsx   (250 LOC)
    └── features/
        └── onboarding/
            ├── OnboardingFlow.tsx (150 LOC)
            └── steps/             (5 files, ~100 LOC each)
```

### TDD Anchors
```
TEST "no component exceeds 300 LOC"
    FOR EACH file IN src/**/*.tsx:
        ASSERT line_count(file) <= 300

TEST "App renders without crash"
    GIVEN mock NodeState with status="ready"
    WHEN  render(<App />)
    THEN  no errors thrown
    AND   Dashboard or Onboarding visible

TEST "SeedOfLife renders SVG"
    WHEN  render(<SeedOfLife size={200} />)
    THEN  SVG element present in DOM
    AND   contains circle elements (sacred geometry)

TEST "QualityGauge shows correct segments"
    GIVEN segments = [{ label: "Ihsan", score: 0.97, threshold: 0.95 }]
    WHEN  render(<QualityGauge segments={segments} />)
    THEN  "Ihsan" text visible
    AND   score 0.97 displayed
    AND   gauge shows green (above threshold)
```

---

## 7. Phase 40d: Bridge Unification

### Goal
Replace 4 overlapping bridge layers with a single `BizraClient`.

### Current Bridges (Redundant)
| Bridge | Transport | Purpose |
|--------|-----------|---------|
| `useNode.js` (24KB) | Tauri invoke | React hook → Rust |
| `useBizraNode.js` (7KB) | Tauri invoke | Alternative hook |
| `bizra-bridge.mjs` (13KB) | WebSocket | Node.js WS ↔ stdio |
| `llm_bridge.js` (15KB) | HTTP/WS | LLM provider routing |

### Pseudocode

```
PROCEDURE unify_bridges():
    # Step 1: Define unified client interface
    CREATE src/lib/client.ts:

        INTERFACE BizraClient:
            connect(): Promise<void>
            disconnect(): void
            invoke<T>(command: string, args: object): Promise<T>
            subscribe(topic: string, handler: Function): Unsubscribe
            getState(): NodeState
            isConnected(): boolean

        # Step 2: Transport adapters (strategy pattern)
        CLASS TauriTransport IMPLEMENTS Transport:
            invoke(cmd, args) → window.__TAURI__.invoke(cmd, args)
            # Used when running inside Tauri desktop app

        CLASS WebSocketTransport IMPLEMENTS Transport:
            invoke(cmd, args) → ws.send(JSON.stringify({ cmd, args }))
            # Used when running in browser (dev mode)

        CLASS HttpTransport IMPLEMENTS Transport:
            invoke(cmd, args) → fetch(`/api/v1/${cmd}`, { body: args })
            # Fallback when WS unavailable

        CLASS SimulatedTransport IMPLEMENTS Transport:
            invoke(cmd, args) → MOCK_RESPONSES[cmd]
            # Used in tests and Storybook

        # Step 3: Auto-detect transport
        FUNCTION detectTransport(): Transport:
            IF window.__TAURI__ EXISTS:
                RETURN TauriTransport()
            ELSE IF WebSocket available AND dev server running:
                RETURN WebSocketTransport(WS_URL)
            ELSE IF HTTP endpoint reachable:
                RETURN HttpTransport(API_URL)
            ELSE:
                RETURN SimulatedTransport()

        # Step 4: React hook wrapping client
        FUNCTION useBizra(): BizraHook:
            client = useRef(createBizraClient())
            state = useSyncExternalStore(client.subscribe, client.getState)
            RETURN { state, invoke: client.invoke, isConnected }

    # Step 5: Delete redundant bridges
    DELETE useBizraNode.js         # merged into useBizra
    REFACTOR useNode.js → thin re-export of useBizra
    REFACTOR bizra-bridge.mjs → WebSocketTransport adapter
    REFACTOR llm_bridge.js → LLM commands in BizraClient
```

### TDD Anchors
```
TEST "auto-detects Tauri transport"
    GIVEN window.__TAURI__ is defined
    WHEN  detectTransport() called
    THEN  returns TauriTransport instance

TEST "falls back to simulated in test env"
    GIVEN no Tauri, no WS, no HTTP
    WHEN  detectTransport() called
    THEN  returns SimulatedTransport instance

TEST "useBizra hook returns node state"
    GIVEN SimulatedTransport with mock state
    WHEN  const { state } = renderHook(() => useBizra())
    THEN  state.status === "ready"
    AND   state.ihsan_score >= 0.95

TEST "invoke routes through correct transport"
    GIVEN TauriTransport mock
    WHEN  client.invoke("get_agents", {})
    THEN  window.__TAURI__.invoke called with "get_agents"
```

---

## 8. Phase 40e: Test Harness

### Goal
Add Vitest + Testing Library with 80%+ coverage target.

### Pseudocode

```
PROCEDURE setup_test_harness():
    # Step 1: Install test dependencies
    npm install --save-dev \
        vitest \
        @testing-library/react \
        @testing-library/jest-dom \
        @testing-library/user-event \
        jsdom \
        @vitest/coverage-v8

    # Step 2: Configure Vitest
    UPDATE vite.config.ts:
        ADD test block:
            globals: true
            environment: "jsdom"
            setupFiles: ["./src/test/setup.ts"]
            coverage:
                provider: "v8"
                reporter: ["text", "lcov"]
                thresholds:
                    lines: 80
                    branches: 75
                    functions: 80

    # Step 3: Create test setup
    CREATE src/test/setup.ts:
        IMPORT "@testing-library/jest-dom"
        MOCK window.__TAURI__ globally
        MOCK IndexedDB for offline tests

    # Step 4: Create test utilities
    CREATE src/test/utils.tsx:
        EXPORT renderWithProviders(component, options):
            WRAP in ThemeProvider + BizraClientProvider
            RETURN render result + user event helpers

        EXPORT mockNodeState(overrides):
            RETURN { status: "ready", ihsan_score: 0.97, ... }

        EXPORT mockAgents(count):
            RETURN array of AgentInfo test fixtures

    # Step 5: Add test scripts to package.json
    UPDATE package.json scripts:
        "test": "vitest run"
        "test:watch": "vitest"
        "test:coverage": "vitest run --coverage"
        "test:ci": "vitest run --coverage --reporter=junit"

    # Step 6: Add CI step
    UPDATE .github/workflows/ci.yml build-frontend job:
        AFTER "Build production bundle":
            - name: Run frontend tests
              working-directory: filedfs
              run: npm run test:ci
```

### Test File Convention
```
src/
├── components/
│   ├── QualityGauge.tsx
│   └── QualityGauge.test.tsx     # co-located test
├── hooks/
│   ├── useBizra.ts
│   └── useBizra.test.ts
├── lib/
│   ├── client.ts
│   └── client.test.ts
└── test/
    ├── setup.ts                   # global test setup
    └── utils.tsx                  # shared test helpers
```

### TDD Anchors
```
TEST "vitest runs and passes"
    WHEN  npm run test:ci
    THEN  exit code = 0
    AND   coverage >= 80% lines

TEST "CI frontend job includes tests"
    GIVEN ci.yml build-frontend job
    WHEN  job runs after npm install + vite build
    THEN  test step executes
    AND   coverage report uploaded as artifact
```

---

## 9. Phase 40f: Offline Hardening

### Goal
Make offline queue robust with retry, conflict resolution, and sync indicators.

### Pseudocode

```
PROCEDURE harden_offline():
    # Current: basic IndexedDB queue in offline/queue.js
    # Target: typed, retry-aware, conflict-resolving queue

    CREATE src/lib/offline-queue.ts:

        INTERFACE QueuedAction:
            id: string               # uuid
            command: string           # Tauri/API command name
            args: Record<string, unknown>
            created_at: number        # timestamp
            retry_count: number       # attempts so far
            max_retries: number       # default 3
            status: "pending" | "sending" | "failed" | "completed"

        CLASS OfflineQueue:
            db: IDBDatabase           # "bizra_offline" database

            ASYNC enqueue(command, args):
                action = { id: uuid(), command, args, retry_count: 0 }
                STORE action in IndexedDB "pending_actions" store
                IF navigator.onLine:
                    SCHEDULE flush()

            ASYNC flush():
                actions = GET ALL pending actions ORDER BY created_at
                FOR EACH action IN actions:
                    TRY:
                        SET action.status = "sending"
                        result = AWAIT client.invoke(action.command, action.args)
                        DELETE action from IndexedDB
                    CATCH error:
                        action.retry_count += 1
                        IF action.retry_count >= action.max_retries:
                            SET action.status = "failed"
                            EMIT "action_failed" event
                        ELSE:
                            SET action.status = "pending"
                            WAIT exponential_backoff(action.retry_count)

            ON navigator.online:
                SCHEDULE flush() with 1s debounce

            getPendingCount(): number
            getFailedActions(): QueuedAction[]
            retryFailed(actionId: string): void
            clearFailed(): void

    # Update service worker
    UPDATE service-worker.js:
        ADD background sync registration
        ADD periodic sync for health checks (if supported)
        IMPROVE cache versioning (use build hash)
```

### TDD Anchors
```
TEST "enqueue stores action in IndexedDB"
    GIVEN offline queue initialized
    WHEN  queue.enqueue("send_message", { text: "hello" })
    THEN  IndexedDB contains 1 pending action
    AND   action.status === "pending"

TEST "flush sends and removes completed actions"
    GIVEN 3 queued actions, client online
    WHEN  queue.flush()
    THEN  client.invoke called 3 times
    AND   IndexedDB empty after flush

TEST "retry with exponential backoff on failure"
    GIVEN 1 queued action, client.invoke throws
    WHEN  queue.flush()
    THEN  action.retry_count === 1
    AND   action.status === "pending"
    AND   next retry scheduled after 2^1 * 1000ms

TEST "action marked failed after max retries"
    GIVEN action with retry_count === 2, max_retries === 3
    WHEN  flush fails again
    THEN  action.status === "failed"
    AND   "action_failed" event emitted
```

---

## 10. Dependency Graph

```
Phase 40a ──→ Phase 40b ──→ Phase 40c
(CI fix)      (TypeScript)   (Decompose)
                  │               │
                  └───→ Phase 40d ←┘
                        (Bridge)
                           │
                      Phase 40e
                      (Tests)
                           │
                      Phase 40f
                      (Offline)
```

### Estimated Effort
| Phase | Files | LOC Changed | Priority |
|-------|-------|-------------|----------|
| 40a | 4 | ~20 | P0 (blocks CI) |
| 40b | 40 | ~2,000 | P1 |
| 40c | 15 new | ~3,000 | P1 |
| 40d | 6 | ~800 | P2 |
| 40e | 12 new | ~1,200 | P2 |
| 40f | 3 | ~400 | P3 |

---

## 11. Constraints

1. **No hard-coded secrets** — all API URLs from env vars or Tauri config
2. **Tauri compatibility** — must work inside Tauri webview AND standalone browser
3. **Ihsan threshold** — UI must visually enforce 0.95 minimum (red below, green above)
4. **Offline-first** — every user action must queue when offline
5. **File size** — no single component file > 300 LOC
6. **Bundle size** — production build < 500KB gzipped (currently ~200KB)
7. **Accessibility** — semantic HTML, ARIA labels on interactive elements
8. **Dark theme** — #0A0B0F background, #D4A547 gold accent (brand standard)
