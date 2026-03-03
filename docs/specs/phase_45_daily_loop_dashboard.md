# Phase 45: Daily Loop Dashboard

> Standing on Giants: Tufte (information density, 1983) · Few (data visualization, 2004) · Shannon (SNR, 1948) · BIZRA Ihsan Covenant

## 1. Problem Statement

BIZRA Node0 has multiple dashboard variants (`App.jsx` 83K, `bizra-dashboard.jsx`
33K, `node0-dashboard.jsx` 43K) but no unified home screen that serves as the
daily landing surface. Users need a single "open the app, see your world" view
that shows sovereignty health, today's activity, agent status, and actionable
next steps — the Daily Loop.

| Dimension | Current | Target |
|-----------|---------|--------|
| Home screen | None — multiple competing dashboards | Single `/` route: Daily Loop |
| Activity feed | Not present | Today's messages, teachings, achievements |
| Agent status | Badge-only in some views | Live 7-agent grid with health/mode |
| Sovereignty metrics | Scattered across files | Unified card: KnowsMe, Ihsan, tier |
| Actionable CTAs | Not present | "Teach", "Chat", "Review missions" buttons |
| Data freshness | Static mock data | Live from HEALTH verb + cached state |
| Accumulator cycle | Concept only | Visual Seed→Bloom→Fruit progress ring |
| Responsive | Desktop-only (1280px) | Mobile-first, breakpoints at 640/768/1024 |

### Daily Loop Concept

The Daily Loop is the user's sovereignty heartbeat:

```
┌─ MORNING ─────────────────────────────────┐
│  Open app → see sovereignty score          │
│  Review overnight agent activity           │
│  Set today's intention (optional TEACH)    │
├─ DAY ─────────────────────────────────────┤
│  Chat with node throughout the day         │
│  Approve/reject mission proposals          │
│  Node learns from interactions             │
├─ EVENING ────────────────────────────────-┤
│  Review today's growth                     │
│  See KnowsMe delta (morning → evening)    │
│  Accumulator cycle: Seed→Bloom→Fruit       │
└───────────────────────────────────────────┘
```

---

## 2. Architecture

```
src/
├── pages/
│   └── HomePage.jsx                 # Daily Loop layout orchestrator
├── features/
│   └── dashboard/
│       ├── SovereigntyCard.jsx      # KnowsMe gauge + Ihsan + tier
│       ├── ActivityFeed.jsx         # Today's events timeline
│       ├── AgentGrid.jsx            # 7-agent status grid
│       ├── AccumulatorRing.jsx      # Seed→Bloom→Fruit cycle visual
│       ├── QuickActions.jsx         # CTA buttons
│       ├── GrowthDelta.jsx          # KnowsMe change since morning
│       └── MissionQueue.jsx         # Pending mission approvals
├── hooks/
│   ├── useDashboardData.js          # Aggregates HEALTH, KNOWS_ME, history
│   └── useActivityFeed.js           # Today's events from IndexedDB
└── utils/
    └── sovereignty.js               # Tier computation, accumulator math
```

### Layout (desktop: 3-column, mobile: stacked)

```
┌──────────────────────────────────────────────────┐
│                    Top Bar                         │
│  [BIZRA Node0]              [Settings] [Profile]  │
├──────────┬────────────────────────┬──────────────┤
│          │                        │              │
│ Sov.Card │   Activity Feed        │  Agent Grid  │
│          │   (scrollable)         │              │
│ Accum.   │                        │  Missions    │
│ Ring     │                        │  Queue       │
│          │                        │              │
│ Growth   │                        │              │
│ Delta    │                        │              │
│          │                        │              │
│ Quick    │                        │              │
│ Actions  │                        │              │
│          │                        │              │
├──────────┴────────────────────────┴──────────────┤
│                    Status Bar                      │
│  Connected ● | Ihsan: 0.97 | Tokens today: 2,340  │
└──────────────────────────────────────────────────┘
```

---

## 3. Pseudocode: Home Page

```
PROCEDURE HomePage():
    STATE data = useDashboardData()
    STATE activities = useActivityFeed()

    RENDER:
        <div className="home-page">
            # Top bar
            <TopBar />

            # Main grid (responsive)
            <div className="home-grid">
                # Left column: sovereignty metrics
                <div className="home-left">
                    <SovereigntyCard
                        knowsMe={data.knowsMe}
                        ihsan={data.ihsan}
                        tier={data.tier}
                    />
                    <AccumulatorRing
                        cycle={data.accumulatorCycle}
                        progress={data.accumulatorProgress}
                    />
                    <GrowthDelta
                        morningScore={data.morningKnowsMe}
                        currentScore={data.knowsMe}
                    />
                    <QuickActions />
                </div>

                # Center column: activity feed
                <div className="home-center">
                    <ActivityFeed events={activities} />
                </div>

                # Right column: agents + missions
                <div className="home-right">
                    <AgentGrid agents={data.agents} />
                    <MissionQueue missions={data.pendingMissions} />
                </div>
            </div>

            # Status bar
            <StatusBar
                connected={data.connected}
                ihsan={data.ihsan}
                tokensToday={data.tokensToday}
            />
        </div>
```

---

## 4. Pseudocode: Sovereignty Card

```
PROCEDURE SovereigntyCard({ knowsMe, ihsan, tier }):
    RENDER:
        <div className="sovereignty-card">
            # Sacred geometry background
            <SeedOfLife size={100} opacity={0.06} />

            # KnowsMe gauge (centered, 8-segment)
            <KnowsMeGauge score={knowsMe} size={140} animated={true} />

            # Score label
            <div className="score-label">
                <span className="score-value">{(knowsMe * 100).toFixed(1)}%</span>
                <span className="score-subtitle">KnowsMe</span>
            </div>

            # Ihsan bar (0-100 scale, gold fill)
            <IhsanBar value={ihsan * 100} label="Ihsan" />

            # Tier badge
            <TierBadge tier={tier} />

            # UERS dimension breakdown (compact)
            <UERSMini dimensions={['Utility', 'Efficiency', 'Resilience', 'Sustainability', 'Ethics']} />
        </div>
```

---

## 5. Pseudocode: Activity Feed

```
PROCEDURE ActivityFeed({ events }):
    # Events grouped by time window
    grouped = groupByTimeWindow(events):
        'morning' → events before 12:00
        'afternoon' → events 12:00-18:00
        'evening' → events after 18:00

    RENDER:
        <div className="activity-feed">
            <h3>Today's Activity</h3>

            IF events.length == 0:
                <EmptyState message="No activity yet today. Start a conversation!" />

            FOR EACH [window, windowEvents] IN grouped:
                <div className="time-window">
                    <h4>{window}</h4>
                    FOR EACH event IN windowEvents:
                        <ActivityItem event={event} />
                </div>

PROCEDURE ActivityItem({ event }):
    iconMap = {
        'message_sent':      '💬',
        'message_received':  '🤖',
        'taught':            '📝',
        'fragment_learned':  '🧩',
        'mission_proposed':  '🎯',
        'mission_completed': '✅',
        'achievement':       '🏆',
        'agent_activated':   '⚡',
        'guardian_veto':     '🛡️',
    }

    RENDER:
        <div className="activity-item" data-type={event.type}>
            <span className="activity-icon">{iconMap[event.type]}</span>
            <div className="activity-body">
                <span className="activity-text">{event.description}</span>
                <time className="activity-time">{formatRelativeTime(event.timestamp)}</time>
            </div>
            IF event.delta:
                <span className="activity-delta">+{event.delta}</span>
        </div>
```

---

## 6. Pseudocode: Agent Grid

```
PROCEDURE AgentGrid({ agents }):
    DEFAULT_AGENTS = [
        { name: 'Scribe',     role: 'Memory',    icon: '📝' },
        { name: 'Guardian',   role: 'Ethics',    icon: '🛡️' },
        { name: 'Strategist', role: 'Planning',  icon: '🎯' },
        { name: 'Analyst',    role: 'Patterns',  icon: '📊' },
        { name: 'Connector',  role: 'Relations', icon: '🔗' },
        { name: 'Operator',   role: 'Execution', icon: '⚡' },
        { name: 'Sentinel',   role: 'Security',  icon: '🔒' },
    ]

    RENDER:
        <div className="agent-grid">
            <h3>Your Agents</h3>
            <div className="grid grid-cols-2 gap-2">
                FOR EACH agent IN DEFAULT_AGENTS:
                    live = agents?.find(a => a.name == agent.name)
                    <AgentTile
                        name={agent.name}
                        role={agent.role}
                        icon={agent.icon}
                        status={live?.status || 'idle'}
                        lastActive={live?.lastActive}
                        tasksToday={live?.tasksToday || 0}
                    />
            </div>

PROCEDURE AgentTile({ name, role, icon, status, lastActive, tasksToday }):
    statusColors = { active: '--bz-success', idle: '--bz-text-muted', busy: '--bz-warning' }

    RENDER:
        <div className="agent-tile" data-status={status}>
            <div className="agent-header">
                <span className="agent-icon">{icon}</span>
                <span className="status-dot" style={{ background: `var(${statusColors[status]})` }} />
            </div>
            <span className="agent-name">{name}</span>
            <span className="agent-role">{role}</span>
            IF tasksToday > 0:
                <span className="agent-tasks">{tasksToday} tasks</span>
        </div>
```

---

## 7. Pseudocode: Accumulator Ring

```
PROCEDURE AccumulatorRing({ cycle, progress }):
    # cycle: 'seed' | 'bloom' | 'fruit'
    # progress: 0.0–1.0 within current cycle

    phases = [
        { name: 'Seed',  color: '--bz-gold-dark',  range: [0, 0.33] },
        { name: 'Bloom', color: '--bz-gold',        range: [0.33, 0.66] },
        { name: 'Fruit', color: '--bz-gold-light',  range: [0.66, 1.0] },
    ]

    currentPhaseIndex = phases.findIndex(p => p.name.toLowerCase() == cycle)

    RENDER:
        <div className="accumulator-ring">
            <svg viewBox="0 0 100 100">
                # Background track
                <circle cx="50" cy="50" r="40" fill="none"
                        stroke="var(--bz-border-subtle)" stroke-width="4" />

                # Three phase arcs
                FOR EACH [i, phase] IN phases.entries():
                    startAngle = phase.range[0] * 360
                    endAngle = phase.range[1] * 360

                    IF i < currentPhaseIndex:
                        # Completed phase: full arc
                        draw_arc(startAngle, endAngle, color=phase.color, opacity=1.0)
                    ELSE IF i == currentPhaseIndex:
                        # Active phase: partial arc based on progress
                        actualEnd = startAngle + (endAngle - startAngle) * progress
                        draw_arc(startAngle, actualEnd, color=phase.color, opacity=1.0)
                        draw_arc(actualEnd, endAngle, color=phase.color, opacity=0.15)
                    ELSE:
                        # Future phase: dim arc
                        draw_arc(startAngle, endAngle, color=phase.color, opacity=0.15)

                # Center label
                <text x="50" y="48" text-anchor="middle" fill="var(--bz-text-primary)">
                    {phases[currentPhaseIndex].name}
                </text>
                <text x="50" y="58" text-anchor="middle" fill="var(--bz-text-secondary)" font-size="8">
                    {(progress * 100).toFixed(0)}%
                </text>
            </svg>

            # Zakat indicator (if in Fruit phase)
            IF cycle == 'fruit' AND progress > 0.9:
                <span className="zakat-ready">Zakat ready (2.5%)</span>
        </div>
```

---

## 8. Pseudocode: Dashboard Data Hook

```
PROCEDURE useDashboardData():
    STATE data = {
        knowsMe: 0, ihsan: 0, tier: 'SEED',
        connected: false, agents: [], pendingMissions: [],
        accumulatorCycle: 'seed', accumulatorProgress: 0,
        morningKnowsMe: 0, tokensToday: 0
    }
    CONST bizraClient = useBizraClient()

    # Poll HEALTH every 10 seconds
    useInterval(async () => {
        TRY:
            health = await bizraClient.send('HEALTH')
            data.connected = true
            data.ihsan = parseInt(health.ihsan) / 10000  # ihsan is 0-10000 scale
            data.agents = parseAgents(health.agents_registered)

            knows = await bizraClient.send('KNOWS_ME')
            data.knowsMe = parseFloat(knows.score)
            data.tier = computeTier(data.knowsMe)

        CATCH:
            data.connected = false
    }, 10_000)

    # Load morning snapshot (saved at first poll of the day)
    ON_MOUNT:
        snapshot = await loadMorningSnapshot()
        IF snapshot AND snapshot.date == today():
            data.morningKnowsMe = snapshot.knowsMe
        ELSE:
            # First poll of day — save as morning snapshot
            saveMorningSnapshot({ date: today(), knowsMe: data.knowsMe })
            data.morningKnowsMe = data.knowsMe

    RETURN data
```

---

## 9. TDD Anchors

```
TEST_SUITE daily_loop_dashboard:

    TEST "home page renders sovereignty card":
        mock HEALTH + KNOWS_ME responses
        render <HomePage />
        ASSERT query('.sovereignty-card') EXISTS
        ASSERT query('.knows-me-gauge') EXISTS
        ASSERT query('.tier-badge') EXISTS

    TEST "activity feed shows today's events":
        seed IndexedDB with 5 events (3 today, 2 yesterday)
        render <ActivityFeed />
        items = queryAll('.activity-item')
        ASSERT items.length == 3  # only today's

    TEST "activity feed groups by time window":
        seed with morning + afternoon events
        render <ActivityFeed />
        ASSERT query('h4:contains("morning")') EXISTS
        ASSERT query('h4:contains("afternoon")') EXISTS

    TEST "agent grid shows 7 agents":
        render <AgentGrid agents={[]} />
        tiles = queryAll('.agent-tile')
        ASSERT tiles.length == 7

    TEST "agent grid shows live status":
        render <AgentGrid agents={[{ name: 'Scribe', status: 'active', tasksToday: 3 }]} />
        scribe = query('[data-agent="Scribe"]')
        ASSERT scribe.dataset.status == 'active'
        ASSERT scribe.text CONTAINS '3 tasks'

    TEST "accumulator ring renders correct phase":
        render <AccumulatorRing cycle="bloom" progress={0.5} />
        ASSERT center text == 'Bloom'
        ASSERT center percent == '50%'

    TEST "growth delta shows morning → current":
        render <GrowthDelta morningScore={0.30} currentScore={0.45} />
        ASSERT text CONTAINS '+15%'
        ASSERT color == green (positive growth)

    TEST "growth delta shows negative change":
        render <GrowthDelta morningScore={0.50} currentScore={0.48} />
        ASSERT text CONTAINS '-2%'

    TEST "HEALTH poll updates dashboard every 10s":
        mock HEALTH → { ihsan: 9700 }
        render <HomePage />
        advance timer 10s
        ASSERT ihsan display == '0.97'

    TEST "responsive: mobile layout stacks columns":
        set viewport 375px
        render <HomePage />
        ASSERT '.home-grid' is single column

    TEST "mission queue shows pending count":
        render <MissionQueue missions={[m1, m2]} />
        ASSERT badge shows '2'
        ASSERT cards for m1, m2 visible
```

---

## 10. Acceptance Criteria

| # | Criterion | Verification |
|---|-----------|-------------|
| 1 | `/` route renders Daily Loop dashboard | Navigate, visual check |
| 2 | Sovereignty card shows live KnowsMe + Ihsan | HEALTH verb polling confirmed |
| 3 | Activity feed shows only today's events | Compare IndexedDB dates |
| 4 | Agent grid shows 7 agents with status | Visual + test |
| 5 | Accumulator ring animates cycle progress | Visual (Seed→Bloom→Fruit) |
| 6 | Growth delta shows morning → current change | Check morning snapshot |
| 7 | Mobile responsive (375px, 768px, 1024px) | Viewport resize test |
| 8 | Page load < 200ms (cached HEALTH data) | Lighthouse or manual timing |
| 9 | Zero hardcoded colors — uses Phase 42 tokens | `grep -r '#D4A547'` = 0 |
| 10 | Each component < 300 LOC | `wc -l` on all files |
