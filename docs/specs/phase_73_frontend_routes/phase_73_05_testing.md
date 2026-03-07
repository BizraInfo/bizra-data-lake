# Phase 73.05: Test Strategy, Accessibility, Performance

**Target:** Test infrastructure for all frontend surfaces

## Test Pyramid

```
                    ┌──────────┐
                    │  E2E (5) │  Playwright: critical user journeys
                    ├──────────┤
                ┌───┤ Integ(15)├───┐  Component + API mock: data flow
                ├───┴──────────┴───┤
            ┌───┤   Unit (40+)     ├───┐  Pure logic, utilities, hooks
            └──────────────────────────┘
```

| Layer | Tool | Count | What |
|---|---|---|---|
| Unit | Vitest | 40+ | Hooks, utils, API client, type guards |
| Integration | Vitest + Testing Library | 15+ | Components with mocked API |
| E2E | Playwright | 5 | Full user journeys (onboarding, dashboard) |
| A11y | axe-core | Per component | WCAG 2.1 AA |
| Visual | Percy (optional) | Per route | Snapshot regression |

## Unit Tests

```pseudocode
# src/shared/api/__tests__/client.test.ts

TEST "BizraClient constructs with env base URL":
    client = new BizraClient()
    ASSERT client._base == import.meta.env.VITE_API_BASE

TEST "BizraClient adds auth header":
    localStorage.setItem("bizra_api_token", "tok_123")
    headers = new BizraClient()._headers()
    ASSERT headers.Authorization == "Bearer tok_123"

TEST "BizraClient throws ApiError on non-200":
    mockFetch(401, { error: "unauthorized" })
    EXPECT_THROWS(ApiError, () => api.health())

TEST "BizraClient retries NOT implemented (fail-fast)":
    # Verify no retry logic — fail fast, let UI handle
    mockFetch(500, { error: "server error" })
    EXPECT_THROWS(ApiError, () => api.health())
    ASSERT fetch.callCount == 1

# src/shared/utils/__tests__/sovereignty.test.ts

TEST "tierColor returns correct colors":
    ASSERT tierColor("SEED") == "var(--color-text-muted)"
    ASSERT tierColor("FOREST") == "var(--color-accent-gold)"

TEST "formatScore handles edge cases":
    ASSERT formatScore(0) == "0.0"
    ASSERT formatScore(1) == "100.0"
    ASSERT formatScore(null) == "—"
    ASSERT formatScore(undefined) == "—"

TEST "stageLabel returns human-readable text":
    ASSERT stageLabel("Seed") == "Seed"
    ASSERT stageLabel(undefined) == "Unknown"
```

## Integration Tests

```pseudocode
# src/features/onboarding/__tests__/ConsumerOnboarding.test.tsx

TEST "full onboarding flow advances through 6 steps":
    mockApi({
        health: { status: "healthy", version: "1.0" },
        teach: { ok: true, traits: ["analytical"] },
        agentRoster: { agents: mockPATAgents, pat_count: 7 },
        seedPotential: { sovereignty_score: 0.05, tier: "SEED" },
        lifecycle: { current_stage: "Seed", progress: 0.5 },
    })

    page = render(<ConsumerOnboarding />)

    # Step 1: Verify
    fireEvent.click(page.getByText("Check Connection"))
    await waitFor(() => page.getByText("Connected"))
    fireEvent.click(page.getByText("Continue"))

    # Step 2: Provider
    fireEvent.click(page.getByText("LM Studio"))
    fireEvent.click(page.getByText("Verify & Continue"))
    await waitFor(() => page.getByText("3 / 6"))

    # Step 3: Teach (answer all 4 questions)
    FOR q IN 1..4:
        fireEvent.change(page.getByRole("textbox"), { target: { value: "answer " + q } })
        fireEvent.submit(page.getByRole("form"))
        await waitFor(() => {})

    # Step 4: PAT Intro
    fireEvent.click(page.getByText("Continue"))

    # Step 5: First Chat
    fireEvent.change(page.getByPlaceholderText("Ask"), { target: { value: "Hello" } })
    fireEvent.submit(page.getByRole("form"))
    await waitFor(() => page.getByText("Continue to Dashboard"))
    fireEvent.click(page.getByText("Continue to Dashboard"))

    # Step 6: Dashboard
    await waitFor(() => page.getByText("SEED"))
    ASSERT page.getByText("Go to Dashboard") IS_VISIBLE

# src/pages/__tests__/HomePage.test.tsx

TEST "home page renders all cards with live data":
    mockApi({
        seedPotential: mockPotential,
        nodeValue: mockNodeValue,
        lifecycle: mockLifecycle,
        agentRoster: mockRoster,
        pendingMissions: [],
        walletSummary: mockWallet,
        health: { status: "healthy" },
    })

    page = render(<HomePage />)
    await waitFor(() => page.getByTestId("sovereignty-card"))
    ASSERT page.getByTestId("node-value-card") IS_VISIBLE
    ASSERT page.getByTestId("agent-grid") IS_VISIBLE
    ASSERT page.getByTestId("wallet-snapshot") IS_VISIBLE

TEST "home page handles API failure gracefully":
    mockApi({ health: NETWORK_ERROR })
    page = render(<HomePage />)
    await waitFor(() => page.getByText("Cannot connect"))
    ASSERT page IS NOT null  # No crash
```

## E2E Tests (Playwright)

```pseudocode
# e2e/onboarding.spec.ts

TEST "consumer onboarding happy path":
    await page.goto("/onboarding")
    # Verify step
    await page.click("text=Check Connection")
    await page.waitForSelector("text=Connected")
    await page.click("text=Continue")
    # ... advance through all 6 steps
    await page.waitForSelector("text=Go to Dashboard")
    await page.click("text=Go to Dashboard")
    await expect(page).toHaveURL("/")

TEST "dashboard loads and shows sovereignty":
    await page.goto("/")
    await page.waitForSelector("[data-testid=sovereignty-card]")
    sovereignty = await page.textContent("[data-testid=sovereignty-score]")
    ASSERT sovereignty IS NOT null

TEST "join page routes to correct onboarding":
    await page.goto("/site/join")
    await page.click("text=I want to grow")
    await expect(page).toHaveURL("/onboarding")

TEST "public site loads under performance budget":
    await page.goto("/site")
    metrics = await page.evaluate(() => performance.getEntriesByType("navigation")[0])
    ASSERT metrics.loadEventEnd - metrics.startTime < 3000

TEST "all routes are keyboard navigable":
    FOR route IN ["/", "/onboarding", "/site"]:
        await page.goto(route)
        await page.keyboard.press("Tab")
        focused = await page.evaluate(() => document.activeElement?.tagName)
        ASSERT focused IS NOT null  # Something receives focus
```

## Accessibility Gates

```pseudocode
# Run axe-core on every component in CI

ACCESSIBILITY_RULES = {
    level: "AA",                    # WCAG 2.1 AA
    rules: {
        "color-contrast": "error",  # Fail build
        "label": "error",           # All inputs labeled
        "button-name": "error",     # All buttons named
        "image-alt": "error",       # All images have alt
        "link-name": "error",       # All links named
        "region": "warning",        # Landmark regions
    },
    touchTarget: 44,                # px minimum
}

# CI gate: `vitest --reporter=verbose && playwright test`
# Fail the build if ANY axe violation at "error" level
```

## Performance Budget

```pseudocode
PERFORMANCE_BUDGET = {
    # Public site
    "/site": {
        fcp: 1500,              # First Contentful Paint (ms)
        lcp: 2500,              # Largest Contentful Paint (ms)
        tti: 3000,              # Time to Interactive (ms)
        total_transfer: 500_000, # bytes
        js_transfer: 100_000,    # bytes gzipped
    },

    # App routes
    "/": {
        fcp: 1000,              # Cached state renders fast
        lcp: 2000,              # Sovereignty card is LCP
        tti: 3000,              # Full interactivity
        total_transfer: 800_000, # bytes (more data than site)
        js_transfer: 200_000,    # bytes gzipped
    },

    "/onboarding": {
        fcp: 800,               # Must feel instant
        lcp: 1500,
        tti: 2000,
        total_transfer: 400_000,
        js_transfer: 80_000,
    },
}

# Enforced via Lighthouse CI in GitHub Actions
```

## CI Integration

```pseudocode
# .github/workflows/frontend.yml (new)

JOBS:
    lint:
        - pnpm lint          # ESLint + TypeScript strict
        - pnpm typecheck     # tsc --noEmit

    test:
        - pnpm test          # Vitest (unit + integration)
        - pnpm test:a11y     # axe-core audit

    e2e:
        - pnpm build
        - pnpm preview &
        - pnpm test:e2e      # Playwright

    perf:
        - pnpm build
        - lighthouse ci      # Performance budget check
```
