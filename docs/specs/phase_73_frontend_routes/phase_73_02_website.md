# Phase 73.02: Public Trust Site

**Target:** `/site/*` routes (separate build or subdomain)
**Upstream:** `docs/WEBSITE_PLAN.md` (frozen page structure)

## Purpose

Build trust, differentiate, explain sovereignty, convert to signup.
This is the first surface a stranger sees. It must answer four questions fast:
1. What BIZRA is
2. Why it is safe
3. Why it is different
4. How value appears quickly

## Routes

```pseudocode
/site                  → LandingPage
/site/how-it-works     → HowItWorksPage
/site/safety           → SafetyPage
/site/demo             → DemoPage
/site/faq              → FAQPage
/site/join             → JoinPage
```

## Page Pseudocode

### Landing Page (`/site`)

```pseudocode
PROCEDURE LandingPage():
    RENDER:
        <main>
            # Hero: one sentence + CTA
            <Hero>
                headline="Your AI. Your data. Your growth."
                subline="BIZRA turns your work into intelligence that belongs to you."
                cta_primary={label: "Get Started", href: "/site/join"}
                cta_secondary={label: "See How", href: "/site/how-it-works"}
            </Hero>

            # Value props: 3 cards
            <ValueGrid>
                <ValueCard
                    icon="shield"
                    title="Sovereign"
                    body="Your private keys never leave your device. No cloud dependency."
                />
                <ValueCard
                    icon="trending-up"
                    title="Growing"
                    body="Every verified action compounds into skills and rewards."
                />
                <ValueCard
                    icon="users"
                    title="Shared"
                    body="Your compiled skills help others. Their skills help you."
                />
            </ValueGrid>

            # Live network stats (optional, from /v1/network/effect)
            <NetworkStats />

            # Social proof / constitutional commitment
            <ConstitutionalPledge
                invariants={5}
                tests={8237}
                crates={22}
            />

            # CTA repeat
            <CTABanner
                headline="Start earning from your work today"
                href="/site/join"
            />
        </main>
```

### How It Works (`/site/how-it-works`)

```pseudocode
PROCEDURE HowItWorksPage():
    RENDER:
        <main>
            # 7-layer visual (simplified for public)
            <LayerDiagram
                layers=[
                    { name: "You", desc: "Install Node0. Generate your identity." },
                    { name: "Agents", desc: "7 personal agents work for you." },
                    { name: "Verification", desc: "Every action produces proof." },
                    { name: "Learning", desc: "Your node grows smarter over time." },
                    { name: "Economy", desc: "Verified work earns SEED tokens." },
                    { name: "Network", desc: "Your skills help other nodes." },
                ]
            />

            # Node lifecycle visual
            <LifecycleTimeline
                stages=["Seed", "Node", "Apprentice", "Builder",
                        "Verifier", "Mentor", "Catalyst"]
            />

            # Reward loop animation
            <RewardLoop
                steps=["Earn", "Verify", "Compile", "Trade"]
            />
        </main>
```

### Safety Page (`/site/safety`)

```pseudocode
PROCEDURE SafetyPage():
    RENDER:
        <main>
            # 5 invariants table
            <InvariantTable
                invariants=[
                    { id: "I-1", name: "Excellence gate", threshold: "Ihsan >= 0.95" },
                    { id: "I-2", name: "Signal quality", threshold: "SNR >= 0.85" },
                    { id: "I-3", name: "Justice constraint", threshold: "Gini <= 0.35" },
                    { id: "I-4", name: "Sovereignty", desc: "Keys never leave device" },
                    { id: "I-5", name: "Accountability", desc: "Every action = receipt" },
                ]
            />

            # Data sovereignty explanation
            <SovereigntyExplainer />

            # Daughter Test callout
            <DaughterTest
                quote="Would I trust this if my daughter received it?"
            />
        </main>
```

### Join Page (`/site/join`)

```pseudocode
PROCEDURE JoinPage():
    STATE track = null  # "consumer" | "contributor"

    RENDER:
        <main>
            IF track IS null:
                # Track selector
                <TrackSelector>
                    <TrackCard
                        title="I want to grow"
                        desc="Use BIZRA to learn, earn, and build skills."
                        onClick={() => track = "consumer"}
                    />
                    <TrackCard
                        title="I want to contribute"
                        desc="Run a node. Contribute compute and expertise."
                        onClick={() => track = "contributor"}
                    />
                </TrackSelector>
            ELSE:
                # Redirect to appropriate onboarding flow
                <Navigate to={
                    track == "consumer"
                        ? "/onboarding"
                        : "/onboarding/contributor"
                } />
        </main>
```

## Performance Budget

| Metric | Target |
|---|---|
| First Contentful Paint | < 1.5s |
| Largest Contentful Paint | < 2.5s |
| Total Transfer Size | < 500KB |
| JavaScript | < 100KB gzipped |
| Images | WebP, lazy-loaded below fold |
| Fonts | Inter variable (self-hosted, < 40KB) |

## SEO Requirements

- Server-side rendered or pre-rendered (Vite SSG plugin)
- `<title>` and `<meta description>` per route
- Open Graph tags for sharing
- JSON-LD structured data for organization
- Canonical URLs

## TDD Anchors

```pseudocode
TEST "landing page renders hero and CTA":
    page = render(<LandingPage />)
    ASSERT page.getByRole("heading") CONTAINS "Your AI"
    ASSERT page.getByRole("link", { name: "Get Started" }).href CONTAINS "/join"

TEST "value cards render all three":
    page = render(<LandingPage />)
    ASSERT page.getAllByTestId("value-card").length == 3

TEST "join page shows track selector initially":
    page = render(<JoinPage />)
    ASSERT page.getByText("I want to grow") IS_VISIBLE
    ASSERT page.getByText("I want to contribute") IS_VISIBLE

TEST "join page navigates to consumer onboarding":
    page = render(<JoinPage />)
    fireEvent.click(page.getByText("I want to grow"))
    ASSERT navigation.location == "/onboarding"

TEST "join page navigates to contributor onboarding":
    page = render(<JoinPage />)
    fireEvent.click(page.getByText("I want to contribute"))
    ASSERT navigation.location == "/onboarding/contributor"

TEST "safety page renders 5 invariants":
    page = render(<SafetyPage />)
    ASSERT page.getAllByTestId("invariant-row").length == 5

TEST "all pages pass axe accessibility audit":
    FOR Page IN [LandingPage, HowItWorksPage, SafetyPage, JoinPage, FAQPage]:
        page = render(<Page />)
        results = await axe(page.container)
        ASSERT results.violations.length == 0

TEST "landing page total transfer < 500KB":
    # Lighthouse CI budget check
    ASSERT performanceBudget.transferSize < 500_000
```
