# Phase 73.04: Daily Dashboard Shell

**Target:** `/` route (Home)
**Upstream:** Phase 45 (daily loop), `FRONTEND_MASTER_SPEC.md` Section 4

## Existing Assets

| File | LOC | Reusable |
|---|---|---|
| `filedfs/bizra-dashboard.jsx` | 33K | Design patterns, card layouts |
| `filedfs/node0-dashboard.jsx` | 43K | Agent grid, telemetry structure |
| `filedfs/node0-mvp.jsx` | ~5K | Minimal viable layout |

**Strategy:** Do not port 76K LOC of prototype code. Extract the four blocks
(Today, Progress, Rewards, Community) and wire them to live API contracts.
Prototype files are design evidence, not implementation source.

## Four Blocks (Frozen)

```
┌────────────────────────────────────────────────────┐
│                   Top Bar                            │
│  [BIZRA]                    [Settings] [Profile]    │
├────────────┬───────────────────────┬───────────────┤
│            │                       │               │
│  TODAY     │  PROGRESS             │  COMMUNITY    │
│            │                       │               │
│  Mission   │  Sovereignty Card     │  Agent Grid   │
│  of the    │  (score, tier, stage) │  (7 PAT)      │
│  Day       │                       │               │
│            │  Growth Velocity      │  Activity     │
│  Quick     │  (weakest dim,        │  Feed         │
│  Actions   │   streak, compiled)   │               │
│            │                       │  Pending      │
│  REWARDS   │  Lifecycle Progress   │  Missions     │
│            │  (stage bar, next)    │               │
│  Wallet    │                       │               │
│  Snapshot  │  Node Value KPI       │               │
│            │  (5 factors, geo mean)│               │
├────────────┴───────────────────────┴───────────────┤
│                   Status Bar                         │
│  Connected ● | Tier: SEED | Ihsan: 0.97 | SEED: 12  │
└────────────────────────────────────────────────────┘
```

## Page Pseudocode

```pseudocode
# src/pages/HomePage.tsx

IMPORT { useSeedPotential, useNodeValue, useLifecycle,
         useAgentRoster, usePendingMissions, useWallet,
         useHealth } FROM "../shared/hooks/useApi"

PROCEDURE HomePage():
    potential = useSeedPotential()
    nodeValue = useNodeValue()
    lifecycle = useLifecycle()
    agents = useAgentRoster()
    missions = usePendingMissions()
    wallet = useWallet()
    health = useHealth()

    RENDER:
        <div className="home-page">
            <TopBar />

            <div className="home-grid">
                # ── Left Column: Today + Rewards ──
                <div className="home-left">
                    <TodayCard potential={potential.data} />
                    <QuickActions />
                    <WalletSnapshot wallet={wallet.data} />
                </div>

                # ── Center Column: Progress ──
                <div className="home-center">
                    <SovereigntyCard
                        potential={potential.data}
                        lifecycle={lifecycle.data}
                    />
                    <GrowthVelocity potential={potential.data} />
                    <LifecycleProgress lifecycle={lifecycle.data} />
                    <NodeValueCard nodeValue={nodeValue.data} />
                </div>

                # ── Right Column: Community ──
                <div className="home-right">
                    <AgentGrid agents={agents.data} />
                    <MissionQueue missions={missions.data} />
                </div>
            </div>

            <StatusBar health={health.data} wallet={wallet.data} />
        </div>
```

## Component Specs

### SovereigntyCard

```pseudocode
PROCEDURE SovereigntyCard({ potential, lifecycle }):
    """Central metric: sovereignty score + tier + human stage."""
    RENDER:
        <div className="card sovereignty-card">
            # Score gauge (0-100 arc)
            <ScoreGauge
                value={potential?.sovereignty_score * 100 || 0}
                max={100}
                color="var(--color-accent-gold)"
                size={160}
            />

            # Tier badge
            <TierBadge tier={potential?.tier || "SEED"} />

            # Human stage label
            <div className="stage-label">
                {lifecycle?.current_stage || "Seed"}
            </div>

            # Qualification stats
            <div className="stats-row">
                <Stat label="Episodes" value={potential?.episodes_total || 0} />
                <Stat label="Qualified" value={potential?.qualification_rate * 100 || 0} unit="%" />
                <Stat label="Streak" value={potential?.streak || 0} />
            </div>

            # Weakest dimension callout
            IF potential?.weakest_dimension:
                <WeaknessBadge dimension={potential.weakest_dimension} />
        </div>
```

### NodeValueCard

```pseudocode
PROCEDURE NodeValueCard({ nodeValue }):
    """5-factor KPI with geometric mean composite."""
    IF NOT nodeValue: RETURN <CardSkeleton />

    FACTORS = [
        { label: "Potential",   value: nodeValue.potential,   icon: "seed" },
        { label: "Activation",  value: nodeValue.activation,  icon: "zap" },
        { label: "Quality",     value: nodeValue.quality,     icon: "shield-check" },
        { label: "Compounding", value: nodeValue.compounding, icon: "trending-up" },
        { label: "Synergy",     value: nodeValue.synergy,     icon: "network" },
    ]

    RENDER:
        <div className="card node-value-card">
            <h3>"Node Value"</h3>

            # Composite score (large, centered)
            <div className="composite-score">
                <span className="score">{(nodeValue.composite * 100).toFixed(1)}</span>
                <span className="label">"/ 100"</span>
            </div>

            # 5 factor bars
            <div className="factor-grid">
                FOR factor IN FACTORS:
                    <FactorBar
                        label={factor.label}
                        value={factor.value}
                        icon={factor.icon}
                    />
            </div>

            # Human stage
            <div className="stage">
                Stage: {nodeValue.human_stage}
            </div>
        </div>
```

### LifecycleProgress

```pseudocode
PROCEDURE LifecycleProgress({ lifecycle }):
    """7-stage progress bar with current position."""
    IF NOT lifecycle: RETURN <CardSkeleton />

    STAGES = ["Seed", "Node", "Apprentice", "Builder", "Verifier", "Mentor", "Catalyst"]

    RENDER:
        <div className="card lifecycle-progress">
            <h3>"Growth Journey"</h3>

            # Stage bar (7 segments)
            <StageBar
                stages={STAGES}
                currentStage={lifecycle.current_stage}
                progress={lifecycle.progress}
            />

            # Next milestone
            IF lifecycle.next_stage:
                <div className="next-milestone">
                    Next: {lifecycle.next_stage}
                    ({(lifecycle.points_to_next * 100).toFixed(1)} points to go)
                </div>

            # Current stage description
            <p className="stage-desc">{lifecycle.description}</p>
            <p className="unlock-cond">{lifecycle.unlock_condition}</p>
        </div>
```

### WalletSnapshot

```pseudocode
PROCEDURE WalletSnapshot({ wallet }):
    """SEED + BLOOM + IMPT balances."""
    IF NOT wallet: RETURN <CardSkeleton />

    RENDER:
        <div className="card wallet-snapshot">
            <h3>"Rewards"</h3>
            <TokenBalance token="SEED" balance={wallet.seed_balance} icon="coin" />
            <TokenBalance token="BLOOM" balance={wallet.bloom_balance} icon="flower" />
            <TokenBalance token="IMPT" balance={wallet.impt_score} icon="star" />

            IF wallet.zakat_contributed > 0:
                <div className="zakat-line">
                    Zakat contributed: {wallet.zakat_contributed.toFixed(3)} SEED
                </div>
        </div>
```

### AgentGrid

```pseudocode
PROCEDURE AgentGrid({ agents }):
    """7 PAT agents with live status."""
    IF NOT agents: RETURN <CardSkeleton />

    patAgents = agents.agents.filter(a => a.type == "PAT")

    RENDER:
        <div className="card agent-grid">
            <h3>"Your Agents" <span className="count">{patAgents.length}/7</span></h3>
            <div className="grid grid-cols-2 gap-2">
                FOR agent IN patAgents:
                    <AgentTile
                        role={agent.role}
                        status={agent.status}
                        lastActive={agent.last_active}
                    />
            </div>
        </div>
```

### TodayCard

```pseudocode
PROCEDURE TodayCard({ potential }):
    """What should the user do next?"""
    RENDER:
        <div className="card today-card">
            <h3>"Today"</h3>

            IF potential?.episodes_total == 0:
                <EmptyState message="Start your first mission to begin growing." />
                <Button href="/earn">"Start a Mission"</Button>

            ELSE IF potential?.weakest_dimension:
                <SuggestionCard
                    title={"Focus on: " + potential.weakest_dimension}
                    body="This is your weakest dimension. Improve it to grow faster."
                />

            ELSE IF NOT potential?.compiled:
                <SuggestionCard
                    title="Keep going"
                    body={"Streak: " + potential.streak + "/3 — compile your first reflex!"}
                />

            ELSE:
                <SuggestionCard
                    title="You're growing"
                    body="Your reflexes are compiling. Keep the streak alive."
                />
        </div>
```

### StatusBar

```pseudocode
PROCEDURE StatusBar({ health, wallet }):
    RENDER:
        <div className="status-bar">
            <StatusDot connected={health?.status == "healthy"} />
            <span>Tier: {health?.seed_engine?.tier || "—"}</span>
            <span>SEED: {wallet?.seed_balance?.toFixed(1) || "0"}</span>
        </div>
```

## Responsive Behavior

```pseudocode
# Mobile (< 768px): single column, stacked blocks
# Tablet (768-1024px): 2 columns (left+center merged, right)
# Desktop (> 1024px): 3 columns as shown in layout diagram

BREAKPOINTS = {
    mobile: "grid-cols-1",
    tablet: "grid-cols-2",
    desktop: "grid-cols-[280px_1fr_280px]",
}
```

## TDD Anchors

```pseudocode
TEST "home page renders 4 blocks":
    page = render(<HomePage />)
    ASSERT page.getByTestId("today-card") IS_VISIBLE
    ASSERT page.getByTestId("sovereignty-card") IS_VISIBLE
    ASSERT page.getByTestId("wallet-snapshot") IS_VISIBLE
    ASSERT page.getByTestId("agent-grid") IS_VISIBLE

TEST "sovereignty card shows score from API":
    mockSeedPotential({ sovereignty_score: 0.42, tier: "SPROUT" })
    page = render(<SovereigntyCard />)
    ASSERT page.getByText("SPROUT") IS_VISIBLE

TEST "node value card shows 5 factors":
    mockNodeValue({ potential: 0.4, activation: 0.6, quality: 0.9, compounding: 0.3, synergy: 1.0 })
    page = render(<NodeValueCard />)
    ASSERT page.getAllByTestId("factor-bar").length == 5

TEST "lifecycle progress shows 7 stages":
    page = render(<LifecycleProgress lifecycle={mockLifecycle} />)
    ASSERT page.getAllByTestId("stage-segment").length == 7

TEST "wallet shows SEED and BLOOM":
    mockWallet({ seed_balance: 12.5, bloom_balance: 3.0, impt_score: 100 })
    page = render(<WalletSnapshot />)
    ASSERT page.getByText("12.5") IS_VISIBLE
    ASSERT page.getByText("BLOOM") IS_VISIBLE

TEST "agent grid shows 7 PAT agents":
    mockAgentRoster(7_pat_agents)
    page = render(<AgentGrid />)
    ASSERT page.getAllByTestId("agent-tile").length == 7

TEST "today card suggests first mission when episodes = 0":
    mockSeedPotential({ episodes_total: 0 })
    page = render(<TodayCard />)
    ASSERT page.getByText("Start your first mission") IS_VISIBLE

TEST "loading states show skeletons":
    page = render(<HomePage />)
    # Before API resolves
    ASSERT page.getAllByTestId("card-skeleton").length > 0

TEST "status bar shows connected when healthy":
    mockHealth({ status: "healthy" })
    page = render(<StatusBar />)
    ASSERT page.getByTestId("status-dot").className CONTAINS "connected"

TEST "mobile layout stacks to single column":
    setViewport(375, 667)
    page = render(<HomePage />)
    grid = page.getByTestId("home-grid")
    ASSERT getComputedStyle(grid).gridTemplateColumns == "1fr"

TEST "all dashboard components pass axe audit":
    page = render(<HomePage />)
    results = await axe(page.container)
    ASSERT results.violations.length == 0
```
