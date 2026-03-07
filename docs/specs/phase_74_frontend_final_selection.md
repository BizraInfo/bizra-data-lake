# Phase 74 — Frontend Final Selection: Prototype → Production Routing

Status: spec-pseudocode
Date: 2026-03-06
Input: `BIZRA Front end final choese/` (28 artifacts)
Depends: Phase 72 (backend endpoints), Phase 73.01 (shared contracts)
Target: BIZRA-OS (React 19 + Vite + TypeScript)

---

## 1. Artifact Inventory & Classification

Every file in `BIZRA Front end final choese/` is classified into one of four
production surfaces per FRONTEND_MASTER_SPEC.md:

```
SURFACE_A  = Public Website (bizra.info)
SURFACE_B  = Onboarding Wizard
SURFACE_C  = Daily Dashboard (core app shell)
SURFACE_D  = Contributor Desktop Client
SURFACE_E  = Operator / Admin Console
ASSET      = Design asset / reference doc (no direct route)
```

### 1.1 Classification Table

| File | Type | Surface | Extractions |
|------|------|---------|-------------|
| `bizra-flagship.html` (DATA-LAKE root) | HTML | A | Hero, Seed-of-Life SVG, particle system, Genesis-100 CTA |
| `BIZRA-Constitutional-Seed.html` | HTML | A | Parchment aesthetic, Arabic bismillah, covenant scroll |
| `BIZRA_Genesis_Status.html` | HTML | A | Status dashboard, component health grid |
| `bizra-caliber.html` | HTML | A | Proof/evidence chain visualization |
| `bizra-genesis-nexus.html` | HTML | A | System nexus map |
| `bizra-the-third-fact.html` | HTML | A | Narrative storytelling page |
| `BIZRA-DDAGI-OS-Atlas-v5_0_FINAL_RESILIENT.html` | HTML | E | 28 Mermaid diagrams, print/PDF export |
| `BIZRA-DDAGI-OS-Atlas-v6_Peak.html` | HTML | E | Sidebar nav, KPI rails, hero+section system |
| `sovereign_emergence.html` | HTML | E | p5.js organism viz, parameter controls, gate pips |
| `bizra-singularity.html` | HTML | E | Living proof dashboard, nerve canvas |
| `bizra-proof-chain-cortex.html` | HTML | E | Proof chain cortex visualization |
| `BIZRA_JARVIS.jsx` | JSX | C | PAT-7 agent chat, boot sequence, mission execution |
| `maestro-architecture.jsx` | JSX | C | Tier routing viz (S1→S2+), emotion detection, trust levels |
| `node0-pipeline-dashboard.jsx` | JSX | D | Pipeline scenario viz, agent deliberation flow |
| `BIZRA_Living_Memory.py` | Python | ASSET | Backend reference — not frontend |
| `bizra_dual_bus_architecture.md` | MD | ASSET | Architecture doc — dual-bus kernel blueprint |
| `brave_screenshot.png` | PNG | ASSET | Screenshot — reference only |
| `*.docx` (7 files) | DOCX | ASSET | Strategy/architecture docs — reference only |
| `BIZRA_Dual_Bus_Kernel_Blueprint_v1.pdf` | PDF | ASSET | Architecture PDF — reference only |

### 1.2 Previously Assessed Prototypes (DATA-LAKE root)

| File | Surface | Status |
|------|---------|--------|
| `BIZRA_SovereignOS.jsx` | C | Assessed Phase 73 — splash→genesis→assembly→dashboard |
| `BIZRA_Production_Frontend.tsx` | C | Assessed Phase 73 — TypeScript, Phase 72 stages |
| `bizra-flagship.html` | A | Assessed Phase 73 — complete landing page |

---

## 2. Design Token Convergence

All prototypes share a consistent color palette that already maps to
`BIZRA-OS/src/shared/tokens.css` (Phase 73.01):

```pseudocode
VERIFY token_alignment:
  FOR EACH prototype IN classified_artifacts:
    EXTRACT css_variables(prototype)
    COMPARE WITH tokens.css
    FLAG any color that deviates more than ΔE > 5 from nearest token

RESULT: All prototypes converge on:
  bg:      #030810 (±2 variants)     → --color-bg-primary
  gold:    #C9A962 (±#C4A35A)        → --color-accent-gold
  text:    #F8F6F1 / #E6EDF3         → --color-text-primary / --color-text-bright
  surface: #111827 / #161B22          → --color-bg-surface
  fonts:   JetBrains Mono + serif     → --font-ui / --font-narrative

NO new tokens needed. Existing tokens.css covers all 28 artifacts.
```

---

## 3. Production Route Map

### 3.1 Surface A — Public Website (`/`, `/how`, `/safety`, `/demo`, `/join`)

```pseudocode
MODULE website_routes:
  PAGES:
    /               → HomePage       # Hero from bizra-flagship.html
    /how            → HowItWorks     # Journey from Atlas-v6 §04
    /safety         → SafetyPage     # Covenant from Constitutional-Seed.html
    /demo           → DemoPage       # Live status from Genesis_Status.html
    /evidence       → EvidencePage   # Proof chain from bizra-caliber.html
    /join           → JoinPage       # Genesis-100 CTA from flagship

  SHARED_COMPONENTS extracted from prototypes:
    SeedOfLifeSVG         ← bizra-flagship.html (animated SVG, 7 circles)
    ParticleCanvas        ← bizra-flagship.html (p5.js → React useRef+canvas)
    ScrollParallax        ← bizra-flagship.html (IntersectionObserver)
    AgentConstellation    ← bizra-flagship.html (PAT-7 grid with glow)
    ParchmentScroll       ← Constitutional-Seed.html (serif, grain texture)
    StatusGrid            ← Genesis_Status.html (component health cards)
    ProofChainViz         ← bizra-caliber.html (evidence chain timeline)

  DATA_SOURCES:
    HomePage:      static content + api.health() for live status badge
    DemoPage:      api.health(), api.seedPotential(), api.nodeValue()
    EvidencePage:  api.networkMilestones()
    JoinPage:      static (mailto CTA, no backend needed yet)

  IMPLEMENTATION_ORDER:
    1. HomePage (highest conversion impact)
    2. JoinPage (Genesis-100 funnel)
    3. HowItWorks
    4. SafetyPage
    5. DemoPage (needs live API)
    6. EvidencePage
```

### 3.2 Surface B — Onboarding Wizard (`/onboard/*`)

```pseudocode
MODULE onboarding_routes:
  # Two tracks per FRONTEND_MASTER_SPEC §B

  CONSUMER_TRACK:
    /onboard/why         → WhyBizra         # Value prop (< 30s)
    /onboard/identity    → IdentityStep     # Name + email + accept covenant
    /onboard/personalize → PersonalizeStep  # Goals picker (finance/learn/connect)
    /onboard/first-win   → FirstWinStep     # Simulated mission → first SEED
    /onboard/activate    → ActivateStep     # Dashboard reveal

  CONTRIBUTOR_TRACK:
    /onboard/node/auth       → NodeAuth        # OTP + device identity
    /onboard/node/env        → EnvCheck        # Hardware floor check (Phase 64)
    /onboard/node/pack       → PackFlow        # Discovery + redaction
    /onboard/node/first-poi  → FirstPoI        # First Proof-of-Impact
    /onboard/node/mission    → FirstMission    # Claim assignment
    /onboard/node/live       → LiveReward      # Impact + reward view

  SHARED_COMPONENTS:
    CovenantAcceptCard    ← Constitutional-Seed.html (compact version)
    ProgressStepper       ← new (5-step / 6-step indicator)
    EnvironmentGauge      ← node0-pipeline-dashboard.jsx (hardware viz)

  DATA_SOURCES:
    IdentityStep:    api.register()
    FirstWinStep:    simulated (local state, no API)
    ActivateStep:    api.health() to verify connection
    EnvCheck:        navigator.hardwareConcurrency, navigator.deviceMemory
    FirstPoI:        api.seedPotential() after first action
    FirstMission:    api.agentRoster() to show available agents

  IMPLEMENTATION_ORDER:
    1. Consumer track (broader audience)
    2. Contributor track (power users)
```

### 3.3 Surface C — Daily Dashboard (`/app/*`)

```pseudocode
MODULE dashboard_routes:
  # Core app shell — 4-block layout per FRONTEND_MASTER_SPEC §C

  LAYOUT:
    AppShell
      ├── BottomNav: [Today, Progress, Rewards, Community]
      ├── StatusBar (top)
      └── <Outlet/>

  PAGES:
    /app             → redirect /app/today
    /app/today       → TodayPage         # Mission feed + quick actions
    /app/progress    → ProgressPage      # Node value + lifecycle + growth
    /app/rewards     → RewardsPage       # Wallet + SEED/BLOOM balance
    /app/community   → CommunityPage     # Network effect + leaderboard
    /app/chat        → AgentChat         # PAT-7 conversational interface
    /app/profile     → ProfilePage       # Identity + settings

  COMPONENT EXTRACTION from prototypes:

    FROM BIZRA_JARVIS.jsx:
      AgentChatPanel        # PAT-7 chat with boot sequence
      MissionExecutor       # Keyword routing → agent → quality gate → mint
      ProactiveQueue        # Agent idle messages (8s timer)
      BootSequence          # 15-step kernel initialization animation
      AgentStatusRing       # 7 PAT + 5 SAT status indicators

    FROM maestro-architecture.jsx:
      TierRoutingViz        # S1/S1.5/S2/S2+ complexity routing
      EmotionDetector       # 7 emotion states with agent boost
      TrustLevelGauge       # 5 trust levels (Stranger→Extension)
      PipelineFlowViz       # 8-step ingress→seal animation

    FROM BIZRA_Production_Frontend.tsx:
      SovereigntyCard       # Score + tier + progress ring
      NodeValueBreakdown    # 5-factor bar chart
      LifecycleTimeline     # 7-stage progression
      AgentGrid             # PAT-7 status cards

    FROM BIZRA_SovereignOS.jsx:
      SplashScreen          # Boot animation (one-time)
      GenesisFlow           # Identity creation + covenant
      CharacterSheet        # Skills + quests + level

    FROM Dashboard.tsx (Phase 73.01 — already API-wired):
      TodayCard             # Already uses useHealth()
      StatusBar             # Already uses health polling

  MERGE STRATEGY:
    Dashboard.tsx (Phase 73.01) is the CANONICAL base.
    Prototype components are extracted as NEW files that import
    from shared/hooks/useApi.ts and shared/api/client.ts.

    JARVIS agent chat    → /app/chat (new route, new component)
    Maestro routing viz  → /app/progress (embedded section)
    Production frontend  → /app/progress (node value breakdown)
    SovereignOS flow     → /app/today (character sheet section)

  DATA_SOURCES (all via shared hooks):
    TodayPage:       useHealth(), useAgentRoster(), useSeedPotential()
    ProgressPage:    useNodeValue(), useLifecycle(), useNetworkEffect()
    RewardsPage:     useWallet(), useSeedPotential()
    CommunityPage:   useNetworkEffect(8_000_000_000)
    AgentChat:       useAgentRoster() + local simulation state
    ProfilePage:     api.me()

  IMPLEMENTATION_ORDER:
    1. TodayPage (daily loop entry point)
    2. ProgressPage (node value visualization)
    3. AgentChat (differentiation feature)
    4. RewardsPage
    5. CommunityPage
    6. ProfilePage
```

### 3.4 Surface D — Contributor Desktop Client (`/node/*`)

```pseudocode
MODULE contributor_routes:
  # Power-user TUI-in-browser experience

  PAGES:
    /node            → NodeDashboard     # Pipeline status overview
    /node/missions   → MissionBoard      # Available + active missions
    /node/agents     → AgentConsole      # PAT management + delegation
    /node/evidence   → EvidenceChain     # Proof receipt explorer

  COMPONENT EXTRACTION:
    FROM node0-pipeline-dashboard.jsx:
      PipelineScenarioViz   # 6 scenario cards with agent deliberation
      AgentDeliberation     # Step-by-step agent processing animation
      IhsanGauge            # Quality score with threshold line
      FATEGateIndicator     # Pass/fail gate visualization

  DATA_SOURCES:
    NodeDashboard:   useHealth(), useNodeValue(), useAgentRoster()
    MissionBoard:    future /v1/missions/* endpoints
    AgentConsole:    useAgentRoster() + future agent management API
    EvidenceChain:   future /v1/evidence/* endpoints

  IMPLEMENTATION_ORDER:
    1. NodeDashboard (reuse existing components)
    2. MissionBoard (needs backend)
    3. AgentConsole
    4. EvidenceChain
```

### 3.5 Surface E — Operator Console (`/ops/*`)

```pseudocode
MODULE operator_routes:
  # Premium control-room tone, Mermaid diagrams, deep telemetry

  PAGES:
    /ops             → OpsOverview       # System health + KPI rails
    /ops/atlas       → SystemAtlas       # Architecture diagrams
    /ops/organism    → OrganismView      # Living proof visualization
    /ops/cortex      → CortexView        # Proof chain cortex

  COMPONENT EXTRACTION:
    FROM BIZRA-DDAGI-OS-Atlas-v6_Peak.html:
      SidebarNav            # Sticky sidebar with section links
      KPIRail               # Signal index stats panel
      HeroSection           # Full-bleed gradient hero with grid overlay
      SectionCard           # Numbered section with badge
      BarChart              # Gold-gradient progress bars
      ChapterRail           # 4-column pillar cards
      JourneySteps          # 6-step horizontal pipeline
      QuoteWall             # 2-column testimonial grid

    FROM BIZRA-DDAGI-OS-Atlas-v5_0_FINAL_RESILIENT.html:
      MermaidRenderer       # 28 diagrams with error fallback + loading state
      PDFExport             # html2pdf.js integration
      DiagramSection        # Header + meta + mermaid container

    FROM sovereign_emergence.html:
      OrganismCanvas        # p5.js seed-of-life + nerve system
      GatePipIndicator      # Constitutional gate status dots
      ParameterSliders      # Seed count, growth rate, mutation controls
      MetricGrid            # 2-column compact metric display
      LayerLegend           # Color-coded system layer toggle

    FROM bizra-singularity.html:
      NerveCanvas           # Dual-canvas nerve visualization
      TopBar                # Compact brand + gate + metric bar

  DATA_SOURCES:
    OpsOverview:   useHealth(), useNodeValue(), useNetworkEffect()
    SystemAtlas:   static Mermaid definitions
    OrganismView:  useHealth(), useSeedPotential() (feeds p5.js params)
    CortexView:    future /v1/evidence/* endpoints

  IMPLEMENTATION_ORDER:
    1. OpsOverview (immediate value for operators)
    2. SystemAtlas (Mermaid diagrams, mostly static)
    3. OrganismView (p5.js port)
    4. CortexView
```

---

## 4. Component Extraction Protocol

```pseudocode
FUNCTION extract_component(prototype_file, component_name, target_surface):
  """
  Extract a self-contained component from an HTML/JSX prototype
  into a typed React component in BIZRA-OS.
  """

  # Step 1: Identify the DOM subtree + associated CSS + JS
  source_html  = READ(prototype_file)
  dom_subtree  = ISOLATE(source_html, component_name)
  css_rules    = EXTRACT_USED_CSS(dom_subtree, source_html)
  js_logic     = EXTRACT_EVENT_HANDLERS(dom_subtree, source_html)

  # Step 2: Convert inline styles → token references
  FOR EACH css_property IN css_rules:
    IF css_property.value MATCHES any token in tokens.css:
      REPLACE WITH var(--token-name)
    ELSE:
      FLAG for manual review

  # Step 3: Convert to typed React component
  tsx_component = TEMPLATE("""
    import styles from './{component_name}.module.css';
    import {{ {hooks} }} from '../../shared/hooks/useApi';

    interface {component_name}Props {{
      {typed_props}
    }}

    export function {component_name}({{ {destructured_props} }}: {component_name}Props) {{
      {converted_js_logic}
      return (
        {converted_jsx}
      );
    }}
  """)

  # Step 4: Write files
  WRITE(f"src/components/{target_surface}/{component_name}.tsx", tsx_component)
  WRITE(f"src/components/{target_surface}/{component_name}.module.css", css_rules)

  # Step 5: Verify
  ASSERT component renders without errors
  ASSERT all tokens resolve
  ASSERT no inline hex colors remain
  ASSERT data-testid attributes present

  RETURN tsx_component
```

---

## 5. Shared Layout Primitives

```pseudocode
# Extracted from cross-prototype patterns (all 28 artifacts share these)

PRIMITIVES to create in src/shared/components/:

  GlassCard:
    background: rgba(var(--color-bg-surface-rgb), 0.56)
    border: 1px solid rgba(255,255,255, 0.08)
    border-radius: var(--radius-lg)
    backdrop-filter: blur(18px)

  GoldGradientBar:
    background: linear-gradient(90deg, var(--color-accent-gold), var(--color-accent-gold-dim))
    height: 10px
    border-radius: 999px

  SectionHeader:
    eyebrow: JetBrains Mono, 10px, gold, uppercase, letter-spacing 3px
    title: narrative font, 34px
    description: text-secondary, 14px
    badge: mono, 10px, gold border, pill shape

  StatusDot:
    width: 8px, height: 8px, border-radius: 50%
    states: green (healthy), amber (degraded), red (error), dim (offline)
    animation: pulse on active

  MonoLabel:
    font: var(--font-ui)
    size: 10px
    color: var(--color-text-muted)
    letter-spacing: 2px
    text-transform: uppercase
```

---

## 6. TDD Anchors

```pseudocode
TEST_SUITE phase_74_frontend:

  # Token coverage
  test_no_inline_hex_colors_in_components:
    SCAN all .tsx files in src/components/
    ASSERT no hex color literals (#xxx, #xxxxxx, #xxxxxxxx)
    ALLOW only in tokens.css and test fixtures

  test_all_tokens_referenced:
    PARSE tokens.css → token_set
    SCAN all .module.css files → used_tokens
    ASSERT (token_set - used_tokens).size < 5  # max 5 unused

  # Route coverage
  test_all_surfaces_have_routes:
    FOR surface IN [A, B, C, D, E]:
      ASSERT router has at least 2 routes for surface

  test_dashboard_api_wiring:
    MOCK api responses
    RENDER TodayPage
    ASSERT useHealth was called
    ASSERT sovereignty-card renders score

  test_agent_chat_boot_sequence:
    RENDER AgentChat
    TRIGGER boot()
    ASSERT 15 system messages appear in order
    ASSERT "All seven agents reporting" is final boot message

  test_node_value_breakdown_factors:
    MOCK nodeValue response with known values
    RENDER ProgressPage
    ASSERT 5 factor bars render
    ASSERT each bar width proportional to factor value

  test_onboarding_consumer_flow_completion:
    RENDER consumer onboarding
    STEP through all 5 steps
    ASSERT ActivateStep shows dashboard link
    ASSERT covenant was accepted

  test_operator_mermaid_fallback:
    RENDER SystemAtlas with mermaid FAILING to load
    ASSERT error boundary shows fallback message
    ASSERT no crash

  test_organism_canvas_renders:
    RENDER OrganismView
    ASSERT canvas element exists
    ASSERT requestAnimationFrame was called

  test_mobile_responsive_breakpoints:
    FOR width IN [375, 768, 1024, 1440]:
      RENDER HomePage at width
      ASSERT no horizontal overflow
      ASSERT touch targets >= 44px

  test_accessibility_basics:
    FOR EACH page IN all_routes:
      RENDER page
      ASSERT all images have alt text
      ASSERT all interactive elements are keyboard-focusable
      ASSERT color contrast ratio >= 4.5:1 on text
```

---

## 7. Implementation Phases

```
Phase 74.01 — Layout Primitives + Router Shell
  - Create shared primitives (GlassCard, GoldGradientBar, etc.)
  - Wire React Router with all 5 surface prefixes
  - Lazy-load each surface
  - Est: 1 session

Phase 74.02 — Surface C: Dashboard Core
  - TodayPage, ProgressPage (API-wired)
  - AgentChat (extracted from JARVIS.jsx)
  - RewardsPage, CommunityPage
  - Est: 2 sessions

Phase 74.03 — Surface A: Public Website
  - HomePage (Seed-of-Life hero from flagship)
  - JoinPage (Genesis-100 CTA)
  - HowItWorks, SafetyPage
  - Est: 2 sessions

Phase 74.04 — Surface B: Onboarding
  - Consumer track (5 steps)
  - Contributor track (6 steps)
  - Est: 2 sessions

Phase 74.05 — Surface D: Contributor Desktop
  - NodeDashboard (pipeline viz)
  - MissionBoard, AgentConsole
  - Est: 1 session

Phase 74.06 — Surface E: Operator Console
  - OpsOverview, SystemAtlas (Mermaid)
  - OrganismView (p5.js port)
  - Est: 2 sessions
```

---

## 8. Files NOT Routed (ASSET classification)

These files inform design decisions but produce no direct route:

- `BIZRA_Living_Memory.py` — Python backend, not frontend
- `bizra_dual_bus_architecture.md` — Architecture reference doc
- `brave_screenshot.png` — Browser screenshot
- `BIZRA_AaaS_Strategic_Architecture.docx` — Strategy document
- `BIZRA_Action_Bus_Skill_Tree_MMORPG.docx` — Game design reference
- `BIZRA_Identity_Genesis_SAT_Universal_v2.docx` — Identity spec
- `BIZRA_OMEGA_CUBED.docx` — Architecture document
- `BIZRA_PRD_Final_Draft.docx` — Product requirements
- `BIZRA_SAD_Software_Architecture.docx` — Software architecture
- `BIZRA_TSD_Technical_Specification.docx` — Technical specification
- `The-Silent-Collapse-Unified-Risalah.docx` — Narrative document
- `BIZRA_Dual_Bus_Kernel_Blueprint_v1.pdf` — Blueprint PDF
- `BIZRA-The-Complete-Story.docx` — Story document
- `BIZRA-DDAGI-OS-Atlas-v4_0.html` — Superseded by v6_Peak

---

## 9. Decision Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Dashboard.tsx (Phase 73.01) is the canonical base | Already API-wired, typed, uses shared hooks |
| D2 | JARVIS chat becomes `/app/chat`, not the default view | Chat is differentiator but not daily-loop entry |
| D3 | Atlas v6 supersedes v5 and v4 | v6 has sidebar nav, cleaner sections, same content |
| D4 | sovereign_emergence p5.js goes to operator console only | Complex viz unsuitable for daily dashboard |
| D5 | No Tailwind — CSS Modules + tokens.css | Matches existing BIZRA-OS conventions, no new deps |
| D6 | Constitutional-Seed.html aesthetic reserved for `/safety` | Parchment tone is intentional contrast for trust |
| D7 | All 8 DOCX files are reference-only | Content informs design but produces no routes |
